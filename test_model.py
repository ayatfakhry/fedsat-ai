"""
FedSat AI — Federated Learning for Satellite Edge Networks
==========================================================
src package init.
"""
from .model             import MLP, CNN, get_model, model_size_kb
from .data_generator    import (generate_synthetic_dataset, partition_data_dirichlet,
                                 partition_data_iid, build_data_loaders,
                                 compute_class_distribution)
from .training          import local_train, evaluate_local, centralized_train
from .fedavg            import (federated_average, federated_average_delta,
                                 compute_client_weights, server_broadcast, select_clients)
from .satellite_client  import SatelliteClient, CommChannel, SatelliteOrbit, build_satellite_clients
from .federated_server  import FederatedServer, RoundResult
from .evaluation        import (evaluate_model, convergence_round, final_metrics_summary,
                                 communication_cost_kb, compute_fairness_metric)
from .visualization     import (plot_accuracy_curve, plot_loss_curves,
                                 plot_per_satellite_loss, plot_federated_vs_centralized,
                                 plot_data_distribution, plot_communication_delays,
                                 plot_update_norms, plot_summary_dashboard)

__all__ = [
    "MLP", "CNN", "get_model", "model_size_kb",
    "generate_synthetic_dataset", "partition_data_dirichlet",
    "partition_data_iid", "build_data_loaders", "compute_class_distribution",
    "local_train", "evaluate_local", "centralized_train",
    "federated_average", "federated_average_delta",
    "compute_client_weights", "server_broadcast", "select_clients",
    "SatelliteClient", "CommChannel", "SatelliteOrbit", "build_satellite_clients",
    "FederatedServer", "RoundResult",
    "evaluate_model", "convergence_round", "final_metrics_summary",
    "communication_cost_kb", "compute_fairness_metric",
    "plot_accuracy_curve", "plot_loss_curves", "plot_per_satellite_loss",
    "plot_federated_vs_centralized", "plot_data_distribution",
    "plot_communication_delays", "plot_update_norms", "plot_summary_dashboard",
]
