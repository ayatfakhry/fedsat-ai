"""
federated_server.py
-------------------
Central ground-station aggregation server for FedSat AI.

The FederatedServer orchestrates the global training loop:
  1. Broadcast global model to selected satellite clients
  2. Wait for local updates (with simulated delay)
  3. Aggregate via FedAvg
  4. Evaluate global model on held-out test data
  5. Log and return per-round metrics

Classes
-------
FederatedServer   — main server orchestrator
RoundResult       — dataclass holding per-round metrics
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from .fedavg import (
    federated_average,
    compute_client_weights,
    select_clients,
    compute_update_norm,
)
from .training import evaluate_local
from .satellite_client import SatelliteClient


# ---------------------------------------------------------------------------
# Per-round result container
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    """Holds all metrics produced in one federated round."""
    round_num:           int
    selected_clients:    List[int]
    global_test_loss:    float
    global_test_acc:     float
    mean_client_loss:    float
    mean_client_acc:     float
    mean_update_norm:    float
    mean_comm_delay_s:   float
    max_comm_delay_s:    float
    num_samples_used:    int
    wall_time_s:         float
    per_client_metrics:  Dict[int, Dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FederatedServer
# ---------------------------------------------------------------------------

class FederatedServer:
    """
    Central ground-station that coordinates federated training.

    Parameters
    ----------
    global_model   : initial global model (nn.Module)
    test_loader    : DataLoader for held-out global test set
    device         : torch.device for server-side inference
    fraction_fit   : fraction of clients selected per round (0 < f ≤ 1)
    seed           : RNG seed for client selection
    verbose        : print round summaries to stdout
    """

    def __init__(
        self,
        global_model: nn.Module,
        test_loader: DataLoader,
        device: torch.device = torch.device("cpu"),
        fraction_fit: float = 1.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.global_model  = global_model.to(device)
        self.test_loader   = test_loader
        self.device        = device
        self.fraction_fit  = fraction_fit
        self.rng           = np.random.default_rng(seed)
        self.verbose       = verbose

        self.history: List[RoundResult] = []

    # ── Main training loop ───────────────────────────────────────────────────

    def fit(
        self,
        clients: List[SatelliteClient],
        num_rounds: int,
    ) -> List[RoundResult]:
        """
        Run *num_rounds* of federated learning.

        Parameters
        ----------
        clients    : list of SatelliteClient objects
        num_rounds : number of communication rounds

        Returns
        -------
        list of RoundResult (one per round)
        """
        if self.verbose:
            print(f"\n{'═'*60}")
            print(f"  FedSat AI — Federated Training")
            print(f"  Satellites : {len(clients)}")
            print(f"  Rounds     : {num_rounds}")
            print(f"  fraction_fit: {self.fraction_fit:.0%}")
            print(f"{'═'*60}\n")

        for rnd in range(1, num_rounds + 1):
            result = self._run_round(clients, rnd)
            self.history.append(result)

            if self.verbose:
                print(
                    f"  Round {rnd:>3}/{num_rounds} | "
                    f"test acc {result.global_test_acc:.4f} | "
                    f"test loss {result.global_test_loss:.4f} | "
                    f"clients {result.selected_clients} | "
                    f"delay {result.mean_comm_delay_s:.3f}s"
                )

        if self.verbose:
            best = max(self.history, key=lambda r: r.global_test_acc)
            print(f"\n  Best round : {best.round_num}  "
                  f"(test acc = {best.global_test_acc:.4f})\n")

        return self.history

    # ── Single round ─────────────────────────────────────────────────────────

    def _run_round(
        self,
        clients: List[SatelliteClient],
        round_num: int,
    ) -> RoundResult:
        t0 = time.perf_counter()

        # 1. Select clients
        selected_ids = select_clients(
            num_clients=len(clients),
            fraction=self.fraction_fit,
            rng=self.rng,
        )
        selected_clients = [clients[i] for i in selected_ids]

        # 2. Broadcast global model
        global_state = copy.deepcopy(self.global_model.state_dict())
        for client in selected_clients:
            client.receive_global_model(global_state)

        # 3. Local training (simulated parallel execution)
        client_states:  List[dict] = []
        sample_counts:  List[int]  = []
        delays:         List[float]= []
        client_metrics: Dict[int, Dict] = {}

        for client in selected_clients:
            metrics = client.local_update(verbose=False)
            state, n_samples, delay = client.get_model_update()
            client_states.append(state)
            sample_counts.append(n_samples)
            delays.append(delay)
            client_metrics[client.client_id] = {
                **metrics,
                "num_samples": n_samples,
                "delay_s":     delay,
            }

        # 4. FedAvg aggregation
        weights = compute_client_weights(sample_counts)
        # Temporarily load client states into dummy models for aggregation
        client_model_copies = []
        for state in client_states:
            m = copy.deepcopy(self.global_model)
            m.load_state_dict(state)
            client_model_copies.append(m)

        federated_average(
            global_model=self.global_model,
            client_models=client_model_copies,
            client_weights=weights,
        )

        # 5. Evaluate global model
        test_loss, test_acc = evaluate_local(
            self.global_model, self.test_loader, self.device
        )

        # 6. Aggregate client metrics
        client_losses = [m["final_loss"]   for m in client_metrics.values()]
        client_accs   = [m["final_acc"]    for m in client_metrics.values()]
        update_norms  = [m["update_norm"]  for m in client_metrics.values()]

        wall_time = time.perf_counter() - t0

        return RoundResult(
            round_num=round_num,
            selected_clients=selected_ids,
            global_test_loss=test_loss,
            global_test_acc=test_acc,
            mean_client_loss=float(np.mean(client_losses)),
            mean_client_acc=float(np.mean(client_accs)),
            mean_update_norm=float(np.mean(update_norms)),
            mean_comm_delay_s=float(np.mean(delays)),
            max_comm_delay_s=float(np.max(delays)),
            num_samples_used=sum(sample_counts),
            wall_time_s=wall_time,
            per_client_metrics=client_metrics,
        )

    # ── History accessors ────────────────────────────────────────────────────

    def get_test_accuracies(self) -> List[float]:
        return [r.global_test_acc for r in self.history]

    def get_test_losses(self) -> List[float]:
        return [r.global_test_loss for r in self.history]

    def get_client_losses(self) -> List[float]:
        return [r.mean_client_loss for r in self.history]

    def get_comm_delays(self) -> List[float]:
        return [r.mean_comm_delay_s for r in self.history]

    def get_update_norms(self) -> List[float]:
        return [r.mean_update_norm for r in self.history]

    def best_round(self) -> RoundResult:
        return max(self.history, key=lambda r: r.global_test_acc)

    def to_dataframe(self):
        """Convert history to a pandas DataFrame."""
        import pandas as pd
        rows = []
        for r in self.history:
            rows.append({
                "round":            r.round_num,
                "test_acc":         r.global_test_acc,
                "test_loss":        r.global_test_loss,
                "client_loss":      r.mean_client_loss,
                "client_acc":       r.mean_client_acc,
                "update_norm":      r.mean_update_norm,
                "mean_delay_s":     r.mean_comm_delay_s,
                "max_delay_s":      r.max_comm_delay_s,
                "samples_used":     r.num_samples_used,
                "wall_time_s":      r.wall_time_s,
            })
        return pd.DataFrame(rows)
