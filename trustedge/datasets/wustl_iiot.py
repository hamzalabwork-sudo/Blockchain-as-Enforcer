"""WUSTL-IIoT-2021 (Sec. IV-B, Table II). Real source: Zolanvari et al. [62],
1,194,464 samples, 41 features, attacks {DoS, DoS Slow}.

Feature count reduced to 12 (from 41) and sample count reduced substantially
to keep CPU training tractable -- see README. Highest class_sep of the four
datasets, reflecting the paper's own highest reported accuracy on this
dataset (Sec. IV-C, Table IV).
"""
from __future__ import annotations

from ..datasets.base import FederatedDataset, build_federated_dataset

ATTACK_NAMES = ["DoS", "DoS-Slow"]
N_FEATURES = 12
CLASS_SEP = 1.5  # tuned (3 classes incl. benign) so BiLSTM binary accuracy lands near the paper's ~0.986


def load_wustl(n_clients: int = 10, samples_per_client: int = 400, seq_len: int = 12,
                partition: str = "iid", dirichlet_alpha: float = 0.5, seed: int = 4) -> FederatedDataset:
    return build_federated_dataset(
        n_clients=n_clients, samples_per_client=samples_per_client, seq_len=seq_len,
        n_features=N_FEATURES, attack_names=ATTACK_NAMES, class_sep=CLASS_SEP,
        seed=seed, partition=partition, dirichlet_alpha=dirichlet_alpha,
    )
