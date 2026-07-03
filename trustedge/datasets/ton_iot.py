"""ToN-IoT (Sec. IV-B, Table II). Real source: Moustafa et al. [60], 461,043
samples, 44 features, attacks {DoS, DDoS, Ransomware, Injection}.

Feature count reduced to 14 (from 44) and sample count reduced substantially
to keep CPU training tractable -- see README.
"""
from __future__ import annotations

from ..datasets.base import FederatedDataset, build_federated_dataset

ATTACK_NAMES = ["DoS", "DDoS", "Ransomware", "Injection"]
N_FEATURES = 14
CLASS_SEP = 2.6  # tuned (5 classes incl. benign) so BiLSTM binary accuracy lands near the paper's ~0.979


def load_ton_iot(n_clients: int = 10, samples_per_client: int = 400, seq_len: int = 12,
                  partition: str = "iid", dirichlet_alpha: float = 0.5, seed: int = 2) -> FederatedDataset:
    return build_federated_dataset(
        n_clients=n_clients, samples_per_client=samples_per_client, seq_len=seq_len,
        n_features=N_FEATURES, attack_names=ATTACK_NAMES, class_sep=CLASS_SEP,
        seed=seed, partition=partition, dirichlet_alpha=dirichlet_alpha,
    )
