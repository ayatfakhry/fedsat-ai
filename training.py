"""
scripts/run_federated_training.py
----------------------------------
Advanced experiment runner for FedSat AI.

Features
--------
- Single federated run (default)
- Alpha ablation study (effect of non-IID degree)
- Client-count scaling study
- Multi-seed robustness evaluation
- Full results exported to CSV + JSON

Usage
-----
    # Single run
    python scripts/run_federated_training.py

    # Alpha ablation (several non-IID levels)
    python scripts/run_federated_training.py --experiment alpha_ablation

    # Scale satellites
    python scripts/run_federated_training.py --experiment scaling

    # Robustness over 3 seeds
    python scripts/run_federated_training.py --experiment robustness --seeds 0 1 2
"""

import os
import sys
import json
import copy
import argparse
import time
from typing import Dict, List

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generator   import (generate_synthetic_dataset, partition_data_dirichlet,
                                   build_data_loaders, compute_class_distribution)
from src.model            import get_model, model_size_kb
from src.satellite_client import build_satellite_clients
from src.federated_server import FederatedServer
from src.training         import centralized_train
from src.evaluation       import (final_metrics_summary, convergence_round,
                                   compute_fairness_metric)
from src.visualization    import (plot_federated_vs_centralized, plot_accuracy_curve,
                                   plot_loss_curves, plot_summary_dashboard,
                                   plot_data_distribution)

from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CFG = dict(
    num_satellites     = 6,
    rounds             = 20,
    local_epochs       = 3,
    batch_size         = 32,
    lr                 = 0.01,
    non_iid_alpha      = 0.5,
    n_samples          = 6000,
    n_features         = 20,
    n_classes          = 10,
    hidden_dim         = 128,
    fraction_fit       = 1.0,
    comm_delay_min     = 0.1,
    comm_delay_max     = 2.0,
    centralized_epochs = 20,
    seed               = 42,
    device             = "auto",
    results_dir        = "results",
)


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(cfg: Dict, tag: str = "") -> Dict:
    """
    Run one complete federated + centralised training experiment.

    Returns a summary dict with final metrics.
    """
    device_str = cfg["device"]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data
    n_train = int(cfg["n_samples"] * 0.8)
    X_all, y_all = generate_synthetic_dataset(
        n_samples=cfg["n_samples"],
        n_features=cfg["n_features"],
        n_classes=cfg["n_classes"],
        random_state=seed,
    )
    X_train, y_train = X_all[:n_train],  y_all[:n_train]
    X_test,  y_test  = X_all[n_train:],  y_all[n_train:]

    # Partition
    client_indices = partition_data_dirichlet(
        y_train, cfg["num_satellites"], alpha=cfg["non_iid_alpha"],
        random_state=seed,
    )
    sample_counts  = {cid: len(idx) for cid, idx in client_indices.items()}
    client_loaders = build_data_loaders(X_train, y_train, client_indices,
                                        batch_size=cfg["batch_size"])

    X_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_t  = torch.tensor(y_test,  dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256, shuffle=False)

    def model_factory():
        return get_model("mlp", input_dim=cfg["n_features"],
                         num_classes=cfg["n_classes"],
                         hidden_dim=cfg["hidden_dim"])

    # Federated
    global_model = model_factory()
    clients = build_satellite_clients(
        num_clients=cfg["num_satellites"],
        model_factory=model_factory,
        data_loaders=client_loaders,
        sample_counts=sample_counts,
        local_epochs=cfg["local_epochs"],
        lr=cfg["lr"],
        comm_delay_min=cfg["comm_delay_min"],
        comm_delay_max=cfg["comm_delay_max"],
        device=device,
        seed=seed,
    )
    server = FederatedServer(
        global_model=global_model,
        test_loader=test_loader,
        device=device,
        fraction_fit=cfg["fraction_fit"],
        seed=seed,
        verbose=False,
    )
    fed_history = server.fit(clients, num_rounds=cfg["rounds"])
    fed_accs    = server.get_test_accuracies()

    # Centralised baseline
    cen_model  = model_factory().to(device)
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    cen_loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                             batch_size=64, shuffle=True)
    cen_history = centralized_train(
        cen_model, cen_loader, test_loader,
        epochs=cfg["centralized_epochs"],
        lr=cfg["lr"], device=device, verbose=False,
    )
    cen_accs = cen_history["test_acc"]

    summary = {
        "tag":              tag,
        "num_satellites":   cfg["num_satellites"],
        "non_iid_alpha":    cfg["non_iid_alpha"],
        "rounds":           cfg["rounds"],
        "seed":             seed,
        "fed_final_acc":    fed_accs[-1],
        "fed_best_acc":     max(fed_accs),
        "cen_final_acc":    cen_accs[-1],
        "cen_best_acc":     max(cen_accs),
        "acc_gap":          cen_accs[-1] - fed_accs[-1],
        "fed_conv80":       convergence_round(fed_accs, 0.80),
        "fed_accs":         fed_accs,
        "cen_accs":         cen_accs,
    }
    return summary


# ---------------------------------------------------------------------------
# Alpha ablation
# ---------------------------------------------------------------------------

def run_alpha_ablation(cfg: Dict, alphas: List[float], results_dir: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  Alpha Ablation Study  (α values: {alphas})")
    print(f"{'═'*60}")

    rows = []
    all_fed_accs = {}
    all_cen_accs = {}

    for alpha in alphas:
        c = copy.deepcopy(cfg)
        c["non_iid_alpha"] = alpha
        tag = f"alpha={alpha}"
        print(f"\n  Running: {tag} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_single_experiment(c, tag=tag)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s) | fed={res['fed_final_acc']:.4f}  cen={res['cen_final_acc']:.4f}")
        rows.append({k: v for k, v in res.items() if not isinstance(v, list)})
        all_fed_accs[alpha] = res["fed_accs"]
        all_cen_accs[alpha] = res["cen_accs"]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "alpha_ablation.csv"), index=False)
    print(f"\n  Results saved → {results_dir}/alpha_ablation.csv")
    print(df[["non_iid_alpha","fed_final_acc","cen_final_acc","acc_gap","fed_conv80"]].to_string(index=False))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    cmap = matplotlib.colormaps.get_cmap("plasma")
    for i, alpha in enumerate(alphas):
        accs = all_fed_accs[alpha]
        rounds = list(range(1, len(accs) + 1))
        ax.plot(rounds, accs, color=cmap(i), linewidth=2,
                label=f"α={alpha}")
    ax.tick_params(colors="#E6EDF3")
    ax.xaxis.label.set_color("#E6EDF3")
    ax.yaxis.label.set_color("#E6EDF3")
    ax.title.set_color("#E6EDF3")
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("FedSat AI — Effect of Non-IID Degree (Dirichlet α)", fontweight="bold")
    ax.grid(True, color="#21262D", linewidth=0.5, linestyle="--")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262D")
    ax.legend(facecolor="#0D1117", edgecolor="#21262D",
              labelcolor="#E6EDF3", fontsize=9)
    ax.set_ylim(0, 1.05)
    path = os.path.join(results_dir, "alpha_ablation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Scaling study
# ---------------------------------------------------------------------------

def run_scaling_study(cfg: Dict, client_counts: List[int], results_dir: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  Scaling Study  (client counts: {client_counts})")
    print(f"{'═'*60}")

    rows = []
    for k in client_counts:
        c = copy.deepcopy(cfg)
        c["num_satellites"] = k
        tag = f"K={k}"
        print(f"\n  Running: {tag} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_single_experiment(c, tag=tag)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s) | fed={res['fed_final_acc']:.4f}")
        rows.append({k_: v for k_, v in res.items() if not isinstance(v, list)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "scaling_study.csv"), index=False)
    print(f"\n  Results saved → {results_dir}/scaling_study.csv")
    print(df[["num_satellites","fed_final_acc","cen_final_acc","acc_gap"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Robustness (multi-seed)
# ---------------------------------------------------------------------------

def run_robustness(cfg: Dict, seeds: List[int], results_dir: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  Robustness Study  (seeds: {seeds})")
    print(f"{'═'*60}")

    rows = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        tag = f"seed={seed}"
        print(f"\n  Running: {tag} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_single_experiment(c, tag=tag)
        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.1f}s) | fed={res['fed_final_acc']:.4f}")
        rows.append({k: v for k, v in res.items() if not isinstance(v, list)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "robustness.csv"), index=False)
    mean_acc = df["fed_final_acc"].mean()
    std_acc  = df["fed_final_acc"].std()
    print(f"\n  Federated final acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Results saved → {results_dir}/robustness.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="FedSat AI — Advanced Experiment Runner")
    p.add_argument("--experiment", type=str, default="single",
                   choices=["single", "alpha_ablation", "scaling", "robustness"],
                   help="Experiment type")
    p.add_argument("--num_satellites",  type=int,   default=6)
    p.add_argument("--rounds",          type=int,   default=20)
    p.add_argument("--local_epochs",    type=int,   default=3)
    p.add_argument("--non_iid_alpha",   type=float, default=0.5)
    p.add_argument("--n_samples",       type=int,   default=6000)
    p.add_argument("--n_classes",       type=int,   default=10)
    p.add_argument("--n_features",      type=int,   default=20)
    p.add_argument("--lr",              type=float, default=0.01)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--seeds",           type=int,   nargs="+", default=[42, 1337, 7])
    p.add_argument("--alphas",          type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 1.0, 5.0])
    p.add_argument("--client_counts",   type=int,   nargs="+",
                   default=[2, 4, 6, 8, 12])
    p.add_argument("--results_dir",     type=str,   default="results")
    p.add_argument("--compare_centralized", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.update({
        "num_satellites": args.num_satellites,
        "rounds":         args.rounds,
        "local_epochs":   args.local_epochs,
        "non_iid_alpha":  args.non_iid_alpha,
        "n_samples":      args.n_samples,
        "n_classes":      args.n_classes,
        "n_features":     args.n_features,
        "lr":             args.lr,
        "seed":           args.seed,
        "results_dir":    args.results_dir,
    })

    if args.experiment == "single":
        print("\n  Running single federated experiment...")
        res = run_single_experiment(cfg, tag="single")
        print(f"\n  Federated final acc  : {res['fed_final_acc']:.4f}")
        print(f"  Centralized final acc: {res['cen_final_acc']:.4f}")
        print(f"  Accuracy gap         : {res['acc_gap']:.4f}")
        plot_accuracy_curve(
            res["fed_accs"],
            output_path=os.path.join(args.results_dir, "script_fed_accuracy.png")
        )
        plot_federated_vs_centralized(
            res["fed_accs"], res["cen_accs"],
            output_path=os.path.join(args.results_dir, "script_comparison.png")
        )
        with open(os.path.join(args.results_dir, "single_result.json"), "w") as f:
            json.dump({k: v for k, v in res.items() if not isinstance(v, list)},
                      f, indent=2)

    elif args.experiment == "alpha_ablation":
        run_alpha_ablation(cfg, args.alphas, args.results_dir)

    elif args.experiment == "scaling":
        run_scaling_study(cfg, args.client_counts, args.results_dir)

    elif args.experiment == "robustness":
        run_robustness(cfg, args.seeds, args.results_dir)

    print(f"\n  All results written to: {os.path.abspath(args.results_dir)}/\n")


if __name__ == "__main__":
    main()
