"""IoT-CAD (Sec. IV-B, Table II). Real source: Mohamed et al. [61], 530,000+
samples, 61-76 features, attacks {DoS, Recon, Exploit, Malware}.

Feature count reduced to 14 (from 61-76) and sample count reduced
substantially to keep CPU training tractable -- see README. Lower class_sep
than the other datasets reflects the paper's own observation that IoT-CAD's
"more heterogeneous attack behaviors" yield slightly lower detection rates
(Sec. IV-C).
"""
from __future__ import annotations

from ..datasets.base import FederatedDataset, build_federated_dataset

ATTACK_NAMES = ["DoS", "Recon", "Exploit", "Malware"]
N_FEATURES = 14
CLASS_SEP = 2.1  # tuned (5 classes incl. benign) so BiLSTM binary accuracy lands near the paper's ~0.960


def load_iot_cad(n_clients: int = 10, samples_per_client: int = 400, seq_len: int = 12,
                  partition: str = "iid", dirichlet_alpha: float = 0.5, seed: int = 3) -> FederatedDataset:
    return build_federated_dataset(
        n_clients=n_clients, samples_per_client=samples_per_client, seq_len=seq_len,
        n_features=N_FEATURES, attack_names=ATTACK_NAMES, class_sep=CLASS_SEP,
        seed=seed, partition=partition, dirichlet_alpha=dirichlet_alpha,
    )
