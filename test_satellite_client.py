"""
training.py
-----------
Local training routines for satellite clients.

Functions
---------
local_train()        — run E epochs of SGD on a client's local data
evaluate_local()     — evaluate model on a DataLoader
centralized_train()  — full-data training for baseline comparison
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def local_train(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 3,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train *model* in-place for *epochs* on data from *data_loader*.

    Optimizer: SGD with momentum.
    Loss:      CrossEntropyLoss.

    Parameters
    ----------
    model       : nn.Module — will be modified in-place
    data_loader : DataLoader supplying (X_batch, y_batch) tuples
    epochs      : number of local passes over the data
    lr          : learning rate
    momentum    : SGD momentum
    weight_decay: L2 regularisation coefficient
    device      : torch.device
    verbose     : if True, print per-epoch loss

    Returns
    -------
    model         : trained model (same object, in-place)
    epoch_losses  : list of mean loss per epoch
    epoch_accs    : list of accuracy per epoch
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    epoch_losses: List[float] = []
    epoch_accs:   List[float] = []

    for epoch in range(epochs):
        running_loss  = 0.0
        correct       = 0
        total         = 0

        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping to avoid explosion with very non-IID data
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == y_batch).sum().item()
            total        += X_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)

        if verbose:
            print(f"    Epoch {epoch+1}/{epochs} — loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

    return model, epoch_losses, epoch_accs


# ---------------------------------------------------------------------------
# Local evaluation
# ---------------------------------------------------------------------------

def evaluate_local(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """
    Evaluate *model* on data from *data_loader*.

    Returns
    -------
    (mean_loss, accuracy) as float
    """
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits  = model(X_batch)
            loss    = criterion(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == y_batch).sum().item()
            total        += X_batch.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Centralized baseline training
# ---------------------------------------------------------------------------

def centralized_train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    epochs: int = 20,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Standard centralised training loop (upper-bound baseline).

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader for all training data
    test_loader  : DataLoader for held-out test data
    epochs       : number of training epochs
    lr           : learning rate
    momentum     : SGD momentum
    weight_decay : L2 regularisation
    device       : torch.device
    verbose      : print progress

    Returns
    -------
    history dict with keys: train_loss, train_acc, test_loss, test_acc
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
    }

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
            correct      += (logits.argmax(1) == y_b).sum().item()
            total        += X_b.size(0)
        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Evaluate ─────────────────────────────────────────────────────────
        test_loss, test_acc = evaluate_local(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if verbose:
            print(
                f"  Epoch {epoch+1:>3}/{epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
                f"test  loss {test_loss:.4f}  acc {test_acc:.3f}"
            )

    return history
