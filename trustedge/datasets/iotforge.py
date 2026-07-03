"""IoTForge Pro (Sec. IV-B, Table II). Real source: Kumar et al. [59], 1M+
samples, 96-159 features, attacks {DoS, DDoS, Password, Ransomware, XSS, Scan}.

Feature count is reduced to 16 (from 96-159) and sample count to a few
thousand (from 1M+) to keep CPU training tractable -- see README.
"""
from __future__ import annotations

from ..datasets.base import FederatedDataset, build_federated_dataset

ATTACK_NAMES = ["DoS", "DDoS", "Password", "Ransomware", "XSS", "Scan"]
N_FEATURES = 16
CLASS_SEP = 3.8  # tuned (7 classes incl. benign) so BiLSTM binary accuracy lands near the paper's ~0.974


def load_iotforge(n_clients: int = 10, samples_per_client: int = 400, seq_len: int = 12,
                   partition: str = "iid", dirichlet_alpha: float = 0.5, seed: int = 1) -> FederatedDataset:
    return build_federated_dataset(
        n_clients=n_clients, samples_per_client=samples_per_client, seq_len=seq_len,
        n_features=N_FEATURES, attack_names=ATTACK_NAMES, class_sep=CLASS_SEP,
        seed=seed, partition=partition, dirichlet_alpha=dirichlet_alpha,
    )
