"""
satellite_client.py
-------------------
Simulates an individual satellite node in a federated learning constellation.

Each SatelliteClient:
  - holds a private local dataset (non-IID slice)
  - runs local SGD training
  - simulates orbital communication delay
  - tracks per-round metrics and resource usage

Classes
-------
SatelliteClient    — single satellite node
SatelliteOrbit     — orbital parameters (decorative metadata)
CommChannel        — communication delay simulator
"""

import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from .training import local_train, evaluate_local
from .model import copy_model_weights, model_size_kb


# ---------------------------------------------------------------------------
# Orbital metadata (decorative, for realism in logs/visualisations)
# ---------------------------------------------------------------------------

@dataclass
class SatelliteOrbit:
    """Simple LEO orbital parameter container."""
    satellite_id:    int
    altitude_km:     float = 550.0       # SpaceX Starlink-like altitude
    inclination_deg: float = 53.0
    longitude_deg:   float = 0.0         # ascending node (randomised externally)
    orbital_period_min: float = 95.5

    def __post_init__(self):
        # Randomise longitude so satellites are spread around the orbit
        if self.longitude_deg == 0.0:
            rng = np.random.default_rng(self.satellite_id * 1337)
            self.longitude_deg = float(rng.uniform(0, 360))

    @property
    def ground_station_distance_km(self) -> float:
        """Approximate slant range to nadir ground station (km)."""
        return self.altitude_km / np.cos(np.radians(5))  # 5° elevation angle


# ---------------------------------------------------------------------------
# Communication channel with delay simulation
# ---------------------------------------------------------------------------

class CommChannel:
    """
    Simulates a noisy, delay-prone satellite uplink/downlink channel.

    Delay model
    -----------
    delay = propagation_delay + processing_jitter
    propagation_delay ∝ altitude / speed_of_light
    jitter            ~ Uniform(min_ms, max_ms)

    Parameters
    ----------
    min_delay_s : minimum round-trip delay in seconds
    max_delay_s : maximum round-trip delay in seconds
    packet_loss_rate : probability that an update is dropped (0–1)
    seed        : RNG seed
    """

    SPEED_OF_LIGHT_KMS = 299_792.458  # km/s

    def __init__(
        self,
        min_delay_s: float = 0.1,
        max_delay_s: float = 2.0,
        packet_loss_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.min_delay_s = min_delay_s
        self.max_delay_s = max_delay_s
        self.packet_loss_rate = packet_loss_rate
        self.rng = np.random.default_rng(seed)
        self.delay_log: List[float] = []

    def simulate_delay(self, altitude_km: float = 550.0) -> float:
        """
        Return a simulated round-trip delay in seconds.

        Propagation component: 2 × altitude / c
        Jitter component:      Uniform(min, max)
        """
        propagation = 2.0 * altitude_km / self.SPEED_OF_LIGHT_KMS
        jitter = float(self.rng.uniform(self.min_delay_s, self.max_delay_s))
        total = propagation + jitter
        self.delay_log.append(total)
        return total

    def is_packet_lost(self) -> bool:
        """Return True if this transmission is dropped."""
        return float(self.rng.random()) < self.packet_loss_rate

    def mean_delay(self) -> float:
        if not self.delay_log:
            return 0.0
        return float(np.mean(self.delay_log))


# ---------------------------------------------------------------------------
# SatelliteClient — main class
# ---------------------------------------------------------------------------

class SatelliteClient:
    """
    Represents a single satellite performing local federated learning.

    Lifecycle per round
    -------------------
    1. receive_global_model()   — copy server weights
    2. local_update()           — run E epochs of local SGD
    3. get_model_update()       — return updated state_dict
    4. (server aggregates)
    5. metrics logged automatically

    Parameters
    ----------
    client_id    : unique satellite identifier
    model        : nn.Module — local copy of the model architecture
    data_loader  : DataLoader for local private data
    num_samples  : total local sample count
    orbit        : SatelliteOrbit metadata
    channel      : CommChannel for delay simulation
    local_epochs : number of local training epochs per round
    lr           : local SGD learning rate
    momentum     : SGD momentum
    weight_decay : L2 regularisation
    device       : torch.device
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_loader: DataLoader,
        num_samples: int,
        orbit: Optional[SatelliteOrbit] = None,
        channel: Optional[CommChannel] = None,
        local_epochs: int = 3,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.client_id    = client_id
        self.model        = copy.deepcopy(model).to(device)
        self.data_loader  = data_loader
        self.num_samples  = num_samples
        self.orbit        = orbit or SatelliteOrbit(client_id)
        self.channel      = channel or CommChannel()
        self.local_epochs = local_epochs
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.device       = device

        # Metrics history
        self.round_losses:       List[float] = []
        self.round_accs:         List[float] = []
        self.round_delays:       List[float] = []
        self.round_update_norms: List[float] = []
        self._global_state_before: Optional[Dict] = None

    # ── Model exchange ───────────────────────────────────────────────────────

    def receive_global_model(self, global_state_dict: Dict) -> None:
        """
        Download global model from the server.
        Stores a snapshot for computing update norms.
        """
        self._global_state_before = copy.deepcopy(global_state_dict)
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def get_model_update(self) -> Tuple[Dict, int, float]:
        """
        Return the locally-updated model state dict plus metadata.

        Returns
        -------
        (state_dict, num_samples, simulated_delay_s)
        """
        delay = self.channel.simulate_delay(self.orbit.altitude_km)
        self.round_delays.append(delay)
        return self.model.state_dict(), self.num_samples, delay

    # ── Training ─────────────────────────────────────────────────────────────

    def local_update(self, verbose: bool = False) -> Dict[str, float]:
        """
        Run local_epochs of SGD on the local dataset.

        Returns
        -------
        dict with keys: final_loss, final_acc, update_norm
        """
        self.model, epoch_losses, epoch_accs = local_train(
            model=self.model,
            data_loader=self.data_loader,
            epochs=self.local_epochs,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            device=self.device,
            verbose=verbose,
        )

        final_loss = epoch_losses[-1] if epoch_losses else float("nan")
        final_acc  = epoch_accs[-1]  if epoch_accs  else float("nan")

        self.round_losses.append(final_loss)
        self.round_accs.append(final_acc)

        # Compute update norm (client drift)
        norm = self._compute_update_norm()
        self.round_update_norms.append(norm)

        return {"final_loss": final_loss, "final_acc": final_acc, "update_norm": norm}

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate current local model on local data. Returns (loss, acc)."""
        return evaluate_local(self.model, self.data_loader, self.device)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_update_norm(self) -> float:
        """L2 norm of (local_weights − global_weights_before_update)."""
        if self._global_state_before is None:
            return 0.0
        norm_sq = 0.0
        local_state = self.model.state_dict()
        for k in self._global_state_before:
            diff = local_state[k].float() - self._global_state_before[k].float()
            norm_sq += (diff ** 2).sum().item()
        return norm_sq ** 0.5

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SatelliteClient(id={self.client_id}, "
            f"samples={self.num_samples}, "
            f"alt={self.orbit.altitude_km:.0f}km, "
            f"model_size={model_size_kb(self.model):.1f}KB)"
        )

    def summary(self) -> Dict:
        """Return a concise summary dict for logging."""
        return {
            "client_id":    self.client_id,
            "num_samples":  self.num_samples,
            "altitude_km":  self.orbit.altitude_km,
            "rounds_done":  len(self.round_losses),
            "latest_loss":  self.round_losses[-1] if self.round_losses else None,
            "latest_acc":   self.round_accs[-1]   if self.round_accs   else None,
            "mean_delay_s": float(np.mean(self.round_delays)) if self.round_delays else 0.0,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_satellite_clients(
    num_clients: int,
    model_factory,          # callable() → nn.Module
    data_loaders: Dict[int, DataLoader],
    sample_counts: Dict[int, int],
    local_epochs: int = 3,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    comm_delay_min: float = 0.1,
    comm_delay_max: float = 2.0,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> List[SatelliteClient]:
    """
    Instantiate a list of SatelliteClients.

    Parameters
    ----------
    num_clients    : number of satellites
    model_factory  : zero-argument callable returning an nn.Module
    data_loaders   : dict {client_id: DataLoader}
    sample_counts  : dict {client_id: int}
    local_epochs   : local training epochs per round
    lr             : learning rate
    momentum       : SGD momentum
    weight_decay   : L2 regularisation
    comm_delay_min : minimum channel delay (s)
    comm_delay_max : maximum channel delay (s)
    device         : torch.device
    seed           : base RNG seed

    Returns
    -------
    list of SatelliteClient objects
    """
    clients = []
    for cid in range(num_clients):
        orbit   = SatelliteOrbit(satellite_id=cid)
        channel = CommChannel(
            min_delay_s=comm_delay_min,
            max_delay_s=comm_delay_max,
            seed=seed + cid,
        )
        client = SatelliteClient(
            client_id=cid,
            model=model_factory(),
            data_loader=data_loaders[cid],
            num_samples=sample_counts[cid],
            orbit=orbit,
            channel=channel,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
        )
        clients.append(client)
    return clients
