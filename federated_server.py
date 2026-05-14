"""
model.py
--------
Neural network architectures for FedSat AI.

Provides:
  - MLP: Multi-layer perceptron for tabular/synthetic data
  - CNN: Convolutional network for image data (MNIST)
  - get_model(): factory function
  - model_size_kb(): utility to report parameter count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# MLP — used for synthetic sensor-classification tasks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Fully-connected feed-forward network.

    Architecture
    ------------
    Input → Linear(hidden) → BN → ReLU → Dropout
          → Linear(hidden) → BN → ReLU → Dropout
          → Linear(num_classes)

    Parameters
    ----------
    input_dim   : int   — number of input features
    hidden_dim  : int   — width of hidden layers
    num_classes : int   — number of output classes
    dropout     : float — dropout probability
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B, C)
        return self.network(x)


# ---------------------------------------------------------------------------
# CNN — used for image-classification tasks (e.g. MNIST)
# ---------------------------------------------------------------------------

class CNN(nn.Module):
    """
    Small convolutional network.

    Architecture
    ------------
    Conv2d(1→32, 3×3) → ReLU → MaxPool(2)
    Conv2d(32→64, 3×3) → ReLU → MaxPool(2)
    Flatten → Linear(128) → ReLU → Dropout → Linear(num_classes)

    Parameters
    ----------
    in_channels : int   — image channels (1 for greyscale, 3 for RGB)
    num_classes : int   — number of output classes
    dropout     : float — dropout before final layer
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, num_classes)
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Factory & utilities
# ---------------------------------------------------------------------------

def get_model(
    model_type: str = "mlp",
    input_dim: int = 20,
    num_classes: int = 10,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Return an un-trained model instance.

    Parameters
    ----------
    model_type  : "mlp" | "cnn"
    input_dim   : feature dimension (MLP only)
    num_classes : number of output classes
    hidden_dim  : hidden layer width (MLP only)
    dropout     : dropout probability

    Returns
    -------
    nn.Module
    """
    if model_type == "mlp":
        return MLP(input_dim=input_dim, hidden_dim=hidden_dim,
                   num_classes=num_classes, dropout=dropout)
    elif model_type == "cnn":
        return CNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'mlp' or 'cnn'.")


def model_size_kb(model: nn.Module) -> float:
    """Return approximate parameter size in kilobytes (float32 assumed)."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / 1024  # 4 bytes per float32


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def copy_model_weights(src: nn.Module, dst: nn.Module) -> None:
    """Copy weights from *src* into *dst* in-place."""
    dst.load_state_dict(src.state_dict())


def model_l2_norm(model: nn.Module) -> float:
    """Compute L2 norm of all parameters (useful for debugging divergence)."""
    norm_sq = sum((p ** 2).sum().item() for p in model.parameters())
    return norm_sq ** 0.5
