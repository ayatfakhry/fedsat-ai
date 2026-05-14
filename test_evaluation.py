"""
visualization.py
----------------
Plotting and visualisation utilities for FedSat AI.

All functions accept pre-computed data and save figures to *output_dir*.

Functions
---------
plot_accuracy_curve()         — global test accuracy over rounds
plot_loss_curves()            — federated vs centralized loss
plot_per_satellite_loss()     — per-client local loss heatmap
plot_federated_vs_centralized() — side-by-side accuracy comparison
plot_data_distribution()      — non-IID class distribution heatmap
plot_communication_delays()   — delay distribution per satellite
plot_update_norms()           — client update norms over rounds
plot_convergence_comparison() — multiple runs or alpha values
plot_summary_dashboard()      — 2×3 summary figure
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "fed":     "#00C7FF",   # cyan-blue  (federated)
    "cen":     "#FF6B6B",   # coral-red  (centralised)
    "loss":    "#FFD166",   # amber      (loss)
    "bg":      "#0D1117",   # dark bg
    "grid":    "#21262D",   # subtle grid
    "text":    "#E6EDF3",   # light text
    "accent":  "#7EE787",   # green accent
}

CLIENTS_CMAP = matplotlib.colormaps.get_cmap("tab10")

def _apply_dark_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    if title:   ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel)
    if ylabel:  ax.set_ylabel(ylabel)


def _dark_figure(figsize=(10, 5)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["bg"])
    _apply_dark_style(ax)
    return fig, ax


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] Saved → {path}")


# ---------------------------------------------------------------------------
# 1. Accuracy curve
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    accuracies: List[float],
    output_path: str = "results/federated_accuracy.png",
    title: str = "FedSat AI — Global Test Accuracy",
) -> None:
    fig, ax = _dark_figure((10, 5))
    rounds = list(range(1, len(accuracies) + 1))
    ax.plot(rounds, accuracies, color=PALETTE["fed"], linewidth=2.5,
            marker="o", markersize=4, label="Federated (FedAvg)")
    ax.fill_between(rounds, accuracies, alpha=0.15, color=PALETTE["fed"])
    ax.axhline(max(accuracies), color=PALETTE["accent"], linewidth=1,
               linestyle="--", alpha=0.7, label=f"Best: {max(accuracies):.4f}")
    _apply_dark_style(ax, title, "Communication Round", "Test Accuracy")
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])
    ax.set_ylim(0, 1.05)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    fed_losses: List[float],
    client_losses: Optional[List[float]] = None,
    output_path: str = "results/federated_loss.png",
    title: str = "FedSat AI — Training Loss",
) -> None:
    fig, ax = _dark_figure((10, 5))
    rounds = list(range(1, len(fed_losses) + 1))
    ax.plot(rounds, fed_losses, color=PALETTE["fed"], linewidth=2.5,
            marker="o", markersize=4, label="Global Test Loss")
    if client_losses:
        ax.plot(rounds, client_losses, color=PALETTE["loss"], linewidth=2,
                linestyle="--", marker="s", markersize=3, label="Mean Client Train Loss")
    _apply_dark_style(ax, title, "Communication Round", "Loss")
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Per-satellite loss heatmap
# ---------------------------------------------------------------------------

def plot_per_satellite_loss(
    client_losses: Dict[int, List[float]],
    output_path: str = "results/per_satellite_loss.png",
    title: str = "Per-Satellite Local Training Loss",
) -> None:
    """client_losses: {client_id: [loss_round_1, loss_round_2, ...]}"""
    if not client_losses:
        return
    num_clients = len(client_losses)
    num_rounds  = max(len(v) for v in client_losses.values())

    matrix = np.full((num_clients, num_rounds), np.nan)
    for cid, losses in client_losses.items():
        matrix[cid, :len(losses)] = losses

    fig, ax = plt.subplots(figsize=(12, max(4, num_clients * 0.6)),
                           facecolor=PALETTE["bg"])
    cmap = LinearSegmentedColormap.from_list(
        "sat_loss", ["#00C7FF", "#FFD166", "#FF6B6B"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"SAT-{i}" for i in range(num_clients)],
                       color=PALETTE["text"], fontsize=9)
    ax.set_xlabel("Communication Round", color=PALETTE["text"])
    ax.set_title(title, color=PALETTE["text"], fontweight="bold", pad=8)
    ax.tick_params(colors=PALETTE["text"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])
    cbar.set_label("Loss", color=PALETTE["text"])
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Federated vs Centralised
# ---------------------------------------------------------------------------

def plot_federated_vs_centralized(
    fed_accuracies:  List[float],
    cen_accuracies:  List[float],
    output_path: str = "results/comparison.png",
    title: str = "Federated vs Centralised Learning",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])

    # Accuracy
    ax = axes[0]
    r_fed = list(range(1, len(fed_accuracies) + 1))
    r_cen = list(range(1, len(cen_accuracies) + 1))
    ax.plot(r_fed, fed_accuracies, color=PALETTE["fed"], linewidth=2.5,
            marker="o", markersize=4, label="Federated (FedAvg)")
    ax.plot(r_cen, cen_accuracies, color=PALETTE["cen"], linewidth=2.5,
            linestyle="--", marker="s", markersize=4, label="Centralised")
    _apply_dark_style(ax, "Test Accuracy Comparison", "Round / Epoch", "Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])

    # Bar chart: final accuracy
    ax2 = axes[1]
    methods = ["Federated\n(FedAvg)", "Centralised\n(Baseline)"]
    vals    = [fed_accuracies[-1], cen_accuracies[-1]]
    colors  = [PALETTE["fed"], PALETTE["cen"]]
    bars = ax2.bar(methods, vals, color=colors, width=0.45, edgecolor=PALETTE["grid"])
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 color=PALETTE["text"], fontsize=11, fontweight="bold")
    _apply_dark_style(ax2, "Final Test Accuracy", "", "Accuracy")
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(axis="x", labelsize=10, colors=PALETTE["text"])

    fig.suptitle(title, color=PALETTE["text"], fontsize=14,
                 fontweight="bold", y=1.02)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Data distribution heatmap
# ---------------------------------------------------------------------------

def plot_data_distribution(
    distribution: np.ndarray,
    output_path: str = "results/data_distribution.png",
    title: str = "Non-IID Data Distribution Across Satellites",
) -> None:
    """
    distribution: (num_clients, num_classes) array of sample counts.
    """
    num_clients, num_classes = distribution.shape

    # Normalise to proportions per client
    row_sums = distribution.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    proportions = distribution / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, num_clients * 0.55)),
                             facecolor=PALETTE["bg"])

    cmap = LinearSegmentedColormap.from_list("dist", ["#0D1117", "#00C7FF", "#7EE787"])

    # Left: proportions
    ax = axes[0]
    im = ax.imshow(proportions, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f"C{c}" for c in range(num_classes)], color=PALETTE["text"])
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"SAT-{i}" for i in range(num_clients)], color=PALETTE["text"])
    ax.set_title("Class Proportions per Satellite", color=PALETTE["text"],
                 fontweight="bold")
    ax.tick_params(colors=PALETTE["text"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Proportion", color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    # Right: absolute counts
    ax2 = axes[1]
    im2 = ax2.imshow(distribution, aspect="auto",
                     cmap=LinearSegmentedColormap.from_list("cnt", ["#0D1117", "#FFD166"]))
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels([f"C{c}" for c in range(num_classes)], color=PALETTE["text"])
    ax2.set_yticks(range(num_clients))
    ax2.set_yticklabels([f"SAT-{i}" for i in range(num_clients)], color=PALETTE["text"])
    ax2.set_title("Sample Counts per Class per Satellite", color=PALETTE["text"],
                  fontweight="bold")
    ax2.tick_params(colors=PALETTE["text"])
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label("Samples", color=PALETTE["text"])
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    fig.suptitle(title, color=PALETTE["text"], fontsize=13,
                 fontweight="bold", y=1.01)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Communication delay distribution
# ---------------------------------------------------------------------------

def plot_communication_delays(
    delays_per_round: List[float],
    all_satellite_delays: Optional[Dict[int, List[float]]] = None,
    output_path: str = "results/communication_delay.png",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])

    # Left: delay over rounds
    ax = axes[0]
    rounds = list(range(1, len(delays_per_round) + 1))
    ax.plot(rounds, delays_per_round, color=PALETTE["loss"], linewidth=2,
            marker="D", markersize=4)
    ax.fill_between(rounds, delays_per_round, alpha=0.2, color=PALETTE["loss"])
    _apply_dark_style(ax, "Mean Communication Delay per Round",
                      "Round", "Delay (s)")

    # Right: histogram of all delays
    ax2 = axes[1]
    if all_satellite_delays:
        for cid, d in all_satellite_delays.items():
            ax2.hist(d, bins=15, alpha=0.5, color=CLIENTS_CMAP(cid),
                     label=f"SAT-{cid}", edgecolor=PALETTE["bg"], linewidth=0.5)
        ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
                   labelcolor=PALETTE["text"], fontsize=7, ncol=2)
    else:
        ax2.hist(delays_per_round, bins=15, color=PALETTE["fed"],
                 edgecolor=PALETTE["bg"], linewidth=0.5)
    _apply_dark_style(ax2, "Communication Delay Distribution",
                      "Delay (s)", "Frequency")

    fig.suptitle("Satellite Communication Delays", color=PALETTE["text"],
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 7. Update norms
# ---------------------------------------------------------------------------

def plot_update_norms(
    update_norms: List[float],
    output_path: str = "results/update_norms.png",
) -> None:
    fig, ax = _dark_figure((10, 4))
    rounds = list(range(1, len(update_norms) + 1))
    ax.bar(rounds, update_norms, color=PALETTE["accent"],
           edgecolor=PALETTE["bg"], linewidth=0.5)
    _apply_dark_style(ax, "Mean Client Update Norm (Client Drift)",
                      "Round", "L2 Norm of Update")
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 8. Summary dashboard (2×3)
# ---------------------------------------------------------------------------

def plot_summary_dashboard(
    fed_accuracies:    List[float],
    fed_losses:        List[float],
    cen_accuracies:    Optional[List[float]],
    cen_losses:        Optional[List[float]],
    client_losses:     List[float],
    delays:            List[float],
    update_norms:      List[float],
    distribution:      np.ndarray,
    output_path: str = "results/summary_dashboard.png",
) -> None:
    fig = plt.figure(figsize=(18, 10), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    rounds_fed = list(range(1, len(fed_accuracies) + 1))

    # ── (0,0) Accuracy ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PALETTE["bg"])
    ax1.plot(rounds_fed, fed_accuracies, color=PALETTE["fed"],
             linewidth=2, marker="o", markersize=3, label="Federated")
    if cen_accuracies:
        r_c = list(range(1, len(cen_accuracies) + 1))
        ax1.plot(r_c, cen_accuracies, color=PALETTE["cen"],
                 linewidth=2, linestyle="--", label="Centralized")
    _apply_dark_style(ax1, "Test Accuracy", "Round", "Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8)

    # ── (0,1) Loss ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE["bg"])
    ax2.plot(rounds_fed, fed_losses, color=PALETTE["fed"],
             linewidth=2, label="Global Test Loss")
    ax2.plot(rounds_fed, client_losses, color=PALETTE["loss"],
             linewidth=2, linestyle="--", label="Client Train Loss")
    if cen_losses:
        r_c = list(range(1, len(cen_losses) + 1))
        ax2.plot(r_c, cen_losses, color=PALETTE["cen"],
                 linewidth=2, linestyle=":", label="Centralized Loss")
    _apply_dark_style(ax2, "Loss Curves", "Round", "Loss")
    ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8)

    # ── (0,2) Communication delay ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PALETTE["bg"])
    ax3.plot(rounds_fed, delays, color=PALETTE["loss"],
             linewidth=2, marker="D", markersize=3)
    ax3.fill_between(rounds_fed, delays, alpha=0.2, color=PALETTE["loss"])
    _apply_dark_style(ax3, "Comm. Delay per Round", "Round", "Delay (s)")

    # ── (1,0) Update norms ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(PALETTE["bg"])
    ax4.bar(rounds_fed, update_norms, color=PALETTE["accent"],
            edgecolor=PALETTE["bg"], linewidth=0.5)
    _apply_dark_style(ax4, "Client Drift (Update Norm)", "Round", "L2 Norm")

    # ── (1,1) Data distribution ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(PALETTE["bg"])
    num_clients, num_classes = distribution.shape
    row_sums = distribution.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    props = distribution / row_sums
    cmap  = LinearSegmentedColormap.from_list("d", ["#0D1117", "#00C7FF", "#7EE787"])
    im    = ax5.imshow(props, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax5.set_xticks(range(num_classes))
    ax5.set_xticklabels([f"C{c}" for c in range(num_classes)],
                        color=PALETTE["text"], fontsize=7)
    ax5.set_yticks(range(num_clients))
    ax5.set_yticklabels([f"S{i}" for i in range(num_clients)],
                        color=PALETTE["text"], fontsize=7)
    _apply_dark_style(ax5, "Non-IID Distribution", "Class", "Satellite")

    # ── (1,2) Bar: final accuracy comparison ─────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PALETTE["bg"])
    labels = ["Federated"]
    values = [fed_accuracies[-1]]
    cols   = [PALETTE["fed"]]
    if cen_accuracies:
        labels.append("Centralized")
        values.append(cen_accuracies[-1])
        cols.append(PALETTE["cen"])
    bars = ax6.bar(labels, values, color=cols, edgecolor=PALETTE["grid"], width=0.5)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", color=PALETTE["text"],
                 fontsize=11, fontweight="bold")
    _apply_dark_style(ax6, "Final Accuracy", "", "Accuracy")
    ax6.set_ylim(0, 1.15)
    ax6.tick_params(axis="x", colors=PALETTE["text"])

    fig.suptitle("🛰  FedSat AI — Training Summary Dashboard",
                 color=PALETTE["text"], fontsize=16, fontweight="bold", y=1.01)
    _save(fig, output_path)
