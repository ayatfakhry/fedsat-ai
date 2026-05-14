"""
data_generator.py
-----------------
Generates synthetic or MNIST datasets and partitions them across satellite
clients using a Dirichlet distribution to simulate non-IID data.

Key functions
-------------
generate_synthetic_dataset()  — create a labelled tabular dataset
load_mnist_dataset()          — download and return MNIST tensors
partition_data_dirichlet()    — non-IID split using Dir(α)
partition_data_iid()          — uniform IID split (baseline)
DatasetSplit                  — lightweight Dataset wrapper
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = 5000,
    n_features: int = 20,
    n_classes: int = 10,
    n_informative: int = 15,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset.

    Uses sklearn's make_classification with multiple informative features and
    class clusters to simulate multi-class satellite sensor data.

    Parameters
    ----------
    n_samples     : total number of samples
    n_features    : feature dimensionality
    n_classes     : number of target classes
    n_informative : number of truly informative features
    random_state  : RNG seed for reproducibility
    save_path     : if given, save CSV to this path

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features), float32
    y : np.ndarray of shape (n_samples,), int64
    """
    n_clusters = max(1, n_classes // 2)
    # Ensure informative + redundant + repeated < n_features
    n_redundant  = min(3, max(0, n_features - n_informative - 1))
    n_informative = min(n_informative, n_features - n_redundant - 1)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = y.astype(np.int64)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
        df["label"] = y
        df.to_csv(save_path, index=False)

    return X, y


# ---------------------------------------------------------------------------
# MNIST loader
# ---------------------------------------------------------------------------

def load_mnist_dataset(data_dir: str = "./data") -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """
    Load MNIST via torchvision and return numpy arrays.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    X shape: (N, 1, 28, 28), float32 in [0, 1]
    y shape: (N,), int64
    """
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        def _ds_to_numpy(ds):
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
            X, y = next(iter(loader))
            return X.numpy().astype(np.float32), y.numpy().astype(np.int64)

        return _ds_to_numpy(train_ds), _ds_to_numpy(test_ds)
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST: {e}. Ensure torchvision is installed.")


# ---------------------------------------------------------------------------
# DatasetSplit — lightweight Dataset wrapper for index subsets
# ---------------------------------------------------------------------------

class DatasetSplit(Dataset):
    """
    Wraps an array-backed dataset and exposes a subset via index list.

    Parameters
    ----------
    X       : np.ndarray of features
    y       : np.ndarray of labels
    indices : list of row indices to include
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: List[int]) -> None:
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Non-IID partitioning via Dirichlet distribution
# ---------------------------------------------------------------------------

def partition_data_dirichlet(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    random_state: int = 42,
) -> Dict[int, List[int]]:
    """
    Partition sample indices among clients using Dir(α) over class labels.

    A *lower* alpha produces more heterogeneous (non-IID) splits:
      - alpha → 0   : each client gets samples from a single class
      - alpha = 0.5 : moderately heterogeneous (default, realistic)
      - alpha → ∞   : IID-like uniform distribution

    Parameters
    ----------
    y           : label array (N,)
    num_clients : K — number of satellite clients
    alpha       : Dirichlet concentration parameter
    min_samples : discard assignments with fewer than this many samples
    random_state: RNG seed

    Returns
    -------
    dict mapping client_id → list of sample indices
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    client_indices: Dict[int, List[int]] = {k: [] for k in range(num_clients)}

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert to counts; ensure they sum to len(cls_idx)
        counts = (proportions * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - counts[:-1].sum()  # fix rounding residual
        # Distribute
        start = 0
        for k, cnt in enumerate(counts):
            end = start + cnt
            client_indices[k].extend(cls_idx[start:end].tolist())
            start = end

    # Fallback: if any client got too few samples, steal from the richest
    for k in range(num_clients):
        if len(client_indices[k]) < min_samples:
            donor = max(range(num_clients), key=lambda x: len(client_indices[x]))
            steal = client_indices[donor][:min_samples]
            client_indices[donor] = client_indices[donor][min_samples:]
            client_indices[k].extend(steal)

    return client_indices


def partition_data_iid(
    y: np.ndarray,
    num_clients: int,
    random_state: int = 42,
) -> Dict[int, List[int]]:
    """
    IID partition: shuffle all indices then split evenly.

    Parameters
    ----------
    y           : label array (N,)
    num_clients : K
    random_state: RNG seed

    Returns
    -------
    dict mapping client_id → list of sample indices
    """
    rng = np.random.default_rng(random_state)
    all_idx = np.arange(len(y))
    rng.shuffle(all_idx)
    splits = np.array_split(all_idx, num_clients)
    return {k: splits[k].tolist() for k in range(num_clients)}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def partition_summary(
    client_indices: Dict[int, List[int]],
    y: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    """
    Return a DataFrame with per-client class distribution.

    Columns: client_id, total_samples, class_0, class_1, ..., class_C-1
    """
    rows = []
    for cid, idx in client_indices.items():
        labels = y[idx]
        row = {"client_id": cid, "total_samples": len(idx)}
        for c in range(num_classes):
            row[f"class_{c}"] = int((labels == c).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("client_id")


def compute_class_distribution(
    client_indices: Dict[int, List[int]], y: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Return array of shape (num_clients, num_classes) with sample counts.
    """
    K = len(client_indices)
    dist = np.zeros((K, num_classes), dtype=int)
    for cid, idx in client_indices.items():
        labels = y[idx]
        for c in range(num_classes):
            dist[cid, c] = int((labels == c).sum())
    return dist


def build_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    client_indices: Dict[int, List[int]],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[int, DataLoader]:
    """
    Build a DataLoader for each client from the partitioned indices.

    Parameters
    ----------
    X, y           : full dataset arrays
    client_indices : output of partition_data_* functions
    batch_size     : mini-batch size
    num_workers    : DataLoader worker processes

    Returns
    -------
    dict mapping client_id → DataLoader
    """
    loaders = {}
    for cid, idx in client_indices.items():
        ds = DatasetSplit(X, y, idx)
        loaders[cid] = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=False)
    return loaders
