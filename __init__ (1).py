"""
fedavg.py
---------
Implementation of the Federated Averaging algorithm (FedAvg).

Reference
---------
McMahan, H.B., Moore, E., Ramage, D., Hampson, S., & Arcas, B.A. (2017).
Communication-Efficient Learning of Deep Networks from Decentralized Data.
AISTATS 2017.

Functions
---------
federated_average()       — weighted average of model state dicts
federated_average_delta() — average gradient deltas (alternative form)
compute_client_weights()  — derive sample-count weights
server_broadcast()        — copy global weights to a list of models
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core FedAvg aggregation
# ---------------------------------------------------------------------------

def federated_average(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: Optional[List[float]] = None,
) -> nn.Module:
    """
    Compute the weighted average of client model parameters and write the
    result into *global_model* in-place.

    w_{t+1} = Σ_k  (n_k / n)  ·  w_k

    Parameters
    ----------
    global_model   : the server model — updated in-place
    client_models  : list of locally-trained client models
    client_weights : relative weights (e.g. sample counts). If None, uses
                     uniform 1/K weighting.

    Returns
    -------
    global_model (same object, updated in-place)
    """
    if not client_models:
        raise ValueError("client_models list is empty.")

    K = len(client_models)
    if client_weights is None:
        client_weights = [1.0 / K] * K
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    # Initialise accumulator to zeros
    global_state = global_model.state_dict()
    avg_state: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_state.items()
    }

    # Weighted sum
    for weight, client_model in zip(client_weights, client_models):
        client_state = client_model.state_dict()
        for k in avg_state:
            avg_state[k] += weight * client_state[k].float()

    global_model.load_state_dict(avg_state)
    return global_model


def federated_average_delta(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_weights: Optional[List[float]] = None,
    server_lr: float = 1.0,
) -> nn.Module:
    """
    FedAvg expressed as a server-side gradient step using weight deltas.

    Δ_k = w_k - w_global
    w_{t+1} = w_global + server_lr · Σ_k (n_k/n) · Δ_k

    This formulation makes it straightforward to add server-side momentum
    or adaptive optimisers (FedAdam, FedYogi).

    Parameters
    ----------
    global_model   : current global model
    client_models  : locally-trained client models
    client_weights : per-client weights (None → uniform)
    server_lr      : server learning rate (default 1.0 == standard FedAvg)

    Returns
    -------
    global_model (in-place)
    """
    if not client_models:
        raise ValueError("client_models list is empty.")

    K = len(client_models)
    if client_weights is None:
        client_weights = [1.0 / K] * K
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    global_state = global_model.state_dict()

    # Weighted delta
    delta: Dict[str, torch.Tensor] = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_state.items()
    }
    for weight, client_model in zip(client_weights, client_models):
        client_state = client_model.state_dict()
        for k in delta:
            delta[k] += weight * (client_state[k].float() - global_state[k].float())

    # Apply delta with server learning rate
    new_state = {k: global_state[k].float() + server_lr * delta[k]
                 for k in global_state}
    global_model.load_state_dict(new_state)
    return global_model


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def compute_client_weights(client_sample_counts: List[int]) -> List[float]:
    """
    Convert raw sample counts to normalised weights (sums to 1).

    Parameters
    ----------
    client_sample_counts : list of n_k values for each client

    Returns
    -------
    list of float weights
    """
    total = sum(client_sample_counts)
    if total == 0:
        raise ValueError("Total sample count is zero.")
    return [n / total for n in client_sample_counts]


# ---------------------------------------------------------------------------
# Server broadcast
# ---------------------------------------------------------------------------

def server_broadcast(
    global_model: nn.Module,
    client_models: List[nn.Module],
) -> None:
    """
    Copy global model parameters into each client model in-place.

    Parameters
    ----------
    global_model  : source model
    client_models : list of client models to update
    """
    global_state = global_model.state_dict()
    for cm in client_models:
        cm.load_state_dict(copy.deepcopy(global_state))


# ---------------------------------------------------------------------------
# Client selection
# ---------------------------------------------------------------------------

def select_clients(
    num_clients: int,
    fraction: float = 1.0,
    rng=None,
) -> List[int]:
    """
    Randomly select a fraction of client indices for a federated round.

    Parameters
    ----------
    num_clients : total number of available clients
    fraction    : fraction to select (0 < fraction <= 1)
    rng         : numpy random Generator (optional)

    Returns
    -------
    sorted list of selected client indices
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()
    k = max(1, int(round(fraction * num_clients)))
    selected = rng.choice(num_clients, size=k, replace=False)
    return sorted(selected.tolist())


# ---------------------------------------------------------------------------
# Update norm (diagnostic)
# ---------------------------------------------------------------------------

def compute_update_norm(
    global_model: nn.Module,
    client_model: nn.Module,
) -> float:
    """
    Compute the L2 norm of (client_model - global_model) parameters.

    Useful for diagnosing client drift in non-IID settings.
    """
    norm_sq = 0.0
    g_state = global_model.state_dict()
    c_state = client_model.state_dict()
    for k in g_state:
        diff = c_state[k].float() - g_state[k].float()
        norm_sq += (diff ** 2).sum().item()
    return norm_sq ** 0.5
