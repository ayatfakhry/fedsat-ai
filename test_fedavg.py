"""
evaluation.py
-------------
Model evaluation utilities for FedSat AI.

Functions
---------
evaluate_model()          — full metrics on a test DataLoader
compute_confusion_matrix()— confusion matrix as numpy array
classification_report_df()— per-class precision/recall/F1 as DataFrame
convergence_round()       — round at which accuracy first exceeds threshold
communication_cost()      — total bytes sent during training
compute_fairness_metric() — client accuracy variance (fairness proxy)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate *model* on *data_loader* and return a rich metrics dict.

    Returns
    -------
    dict with keys:
      loss, accuracy, top1_acc, macro_f1, weighted_f1,
      precision_macro, recall_macro
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, log_loss
    )

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_logits:  List[np.ndarray] = []
    all_preds:   List[int]        = []
    all_targets: List[int]        = []
    running_loss = 0.0
    total        = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            preds         = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y_batch.cpu().numpy().tolist())
            all_logits.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
            total += X_batch.size(0)

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs   = np.vstack(all_logits)

    mean_loss = running_loss / total
    accuracy  = float((all_preds == all_targets).mean())

    macro_f1    = float(f1_score(all_targets, all_preds, average="macro",   zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted",zero_division=0))
    prec_macro  = float(precision_score(all_targets, all_preds, average="macro",   zero_division=0))
    rec_macro   = float(recall_score(all_targets, all_preds,    average="macro",   zero_division=0))

    return {
        "loss":             mean_loss,
        "accuracy":         accuracy,
        "top1_acc":         accuracy,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "precision_macro":  prec_macro,
        "recall_macro":     rec_macro,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Return a (num_classes × num_classes) confusion matrix as numpy array.
    Rows = true class, Columns = predicted class.
    """
    model = model.to(device)
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch).argmax(dim=1).cpu().numpy()
            targets = y_batch.numpy()
            for t, p in zip(targets, preds):
                cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Per-class report
# ---------------------------------------------------------------------------

def classification_report_df(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with per-class precision, recall, F1, and support.
    """
    from sklearn.metrics import classification_report

    model = model.to(device)
    model.eval()
    all_preds:   List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y_batch.numpy().tolist())

    classes = sorted(set(all_targets))
    if class_names is None:
        class_names = [str(c) for c in classes]

    report = classification_report(
        all_targets, all_preds,
        labels=classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for name in class_names:
        if name in report:
            rows.append({
                "class":     name,
                "precision": report[name]["precision"],
                "recall":    report[name]["recall"],
                "f1-score":  report[name]["f1-score"],
                "support":   report[name]["support"],
            })
    return pd.DataFrame(rows).set_index("class")


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_round(
    accuracies: List[float],
    threshold: float = 0.80,
) -> Optional[int]:
    """
    Return the first round (1-indexed) where accuracy >= *threshold*.
    Returns None if the threshold is never reached.
    """
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i + 1
    return None


def final_metrics_summary(
    fed_history,          # list of RoundResult
    centralized_history:  Optional[Dict] = None,
    num_clients: int = 0,
    num_rounds: int = 0,
    model_size_kb: float = 0.0,
) -> Dict:
    """
    Build a comprehensive summary dict suitable for JSON export.

    Parameters
    ----------
    fed_history          : list of RoundResult from FederatedServer
    centralized_history  : dict from centralized_train() (optional)
    num_clients          : number of satellite clients
    num_rounds           : total rounds run
    model_size_kb        : size of model in KB

    Returns
    -------
    dict of scalar metrics
    """
    fed_accs   = [r.global_test_acc  for r in fed_history]
    fed_losses = [r.global_test_loss for r in fed_history]
    delays     = [r.mean_comm_delay_s for r in fed_history]

    summary = {
        "federated": {
            "num_clients":      num_clients,
            "num_rounds":       num_rounds,
            "final_test_acc":   fed_accs[-1]   if fed_accs   else None,
            "final_test_loss":  fed_losses[-1] if fed_losses else None,
            "best_test_acc":    max(fed_accs)  if fed_accs   else None,
            "best_round":       int(np.argmax(fed_accs)) + 1 if fed_accs else None,
            "convergence_80":   convergence_round(fed_accs, 0.80),
            "convergence_90":   convergence_round(fed_accs, 0.90),
            "mean_delay_s":     float(np.mean(delays)) if delays else 0.0,
            "model_size_kb":    model_size_kb,
            "total_comm_kb":    model_size_kb * 2 * num_clients * num_rounds,
        }
    }

    if centralized_history is not None:
        cen_accs = centralized_history.get("test_acc", [])
        summary["centralized"] = {
            "final_test_acc":  cen_accs[-1]  if cen_accs else None,
            "best_test_acc":   max(cen_accs) if cen_accs else None,
            "convergence_80":  convergence_round(cen_accs, 0.80),
            "convergence_90":  convergence_round(cen_accs, 0.90),
        }
        if cen_accs and fed_accs:
            summary["comparison"] = {
                "acc_gap":     cen_accs[-1] - fed_accs[-1],
                "fed_pct_of_centralized": fed_accs[-1] / cen_accs[-1] if cen_accs[-1] > 0 else None,
            }

    return summary


# ---------------------------------------------------------------------------
# Communication cost
# ---------------------------------------------------------------------------

def communication_cost_kb(
    model_size_kb: float,
    num_clients: int,
    num_rounds: int,
    fraction_fit: float = 1.0,
) -> Dict[str, float]:
    """
    Estimate total up/downlink communication cost.

    Assumptions:
    - Downlink (server→client): full model broadcast each round
    - Uplink   (client→server): full model update from each selected client

    Returns
    -------
    dict with total_kb, uplink_kb, downlink_kb
    """
    selected_per_round = max(1, int(round(fraction_fit * num_clients)))
    downlink = model_size_kb * selected_per_round * num_rounds
    uplink   = model_size_kb * selected_per_round * num_rounds
    return {
        "total_kb":    downlink + uplink,
        "downlink_kb": downlink,
        "uplink_kb":   uplink,
    }


# ---------------------------------------------------------------------------
# Fairness metric
# ---------------------------------------------------------------------------

def compute_fairness_metric(
    per_client_accs: List[float],
) -> Dict[str, float]:
    """
    Compute simple fairness metrics across client accuracies.

    Returns variance, std-dev, min-max gap, and Jain's fairness index.
    """
    arr = np.array(per_client_accs)
    n   = len(arr)
    jain = (arr.sum() ** 2) / (n * (arr ** 2).sum()) if (arr ** 2).sum() > 0 else 0.0
    return {
        "mean_acc":    float(arr.mean()),
        "std_acc":     float(arr.std()),
        "var_acc":     float(arr.var()),
        "min_acc":     float(arr.min()),
        "max_acc":     float(arr.max()),
        "gap":         float(arr.max() - arr.min()),
        "jains_index": float(jain),
    }
