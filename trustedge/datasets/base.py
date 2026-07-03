"""Federated sequence-classification data generation and partitioning.

None of the paper's four IIoT datasets (IoTForge Pro, ToN-IoT, IoT-CAD,
WUSTL-IIoT-2021) are bundled here -- they are multi-gigabyte third-party
downloads and this environment has no internet/Kaggle-style dataset access.
Each dataset module synthesizes sequence data with the same qualitative
structure the paper describes (attack-type list, roughly-scaled feature
count) at a reduced scale so CPU training stays tractable; see the README
for the accuracy-tuning rationale and how to plug in real CSVs.

Labels are generated so that part of the signal is a LONG-RANGE cumulative
drift across the sequence (only fully recoverable by a model that
integrates state across all L timesteps) and part is a simple per-timestep
static offset (recoverable by any architecture). This gives recurrent
models (LSTM/BiLSTM) a genuine, not fudged, advantage over a fixed-kernel
CNN -- matching the paper's own finding that CNN underperforms due to
"limited ability to capture temporal dependencies" (Sec. IV-C).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientData:
    client_id: int
    name: str
    X_train: np.ndarray  # (n, L, d)
    y_train: np.ndarray  # multiclass labels (0 = benign)
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class FederatedDataset:
    clients: list[ClientData]
    n_features: int
    n_classes: int  # multiclass attack-type count (including benign)
    attack_names: list[str]

    def global_test_set(self, binary: bool = False) -> tuple[np.ndarray, np.ndarray]:
        X = np.concatenate([c.X_test for c in self.clients])
        y = np.concatenate([c.y_test for c in self.clients])
        return X, (y > 0).astype(np.int64) if binary else y


def generate_sequences(
    n_samples: int,
    seq_len: int,
    n_features: int,
    n_classes: int,
    class_sep: float,
    noise_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_samples)
    X = rng.normal(0, noise_sd, size=(n_samples, seq_len, n_features)).astype(np.float32)

    class_levels = np.linspace(-class_sep, class_sep, n_classes)
    per_step_drift = class_levels[y] / seq_len
    time_idx = np.arange(seq_len)
    X[:, :, 0] += (per_step_drift[:, None] * time_idx[None, :]).astype(np.float32)  # long-range cumulative signal

    for f in range(1, min(3, n_features)):
        X[:, :, f] += (class_levels[y] * 0.35)[:, None].astype(np.float32)  # simple static signal

    return X, y.astype(np.int64)


def dirichlet_label_partition(y: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Dirichlet(alpha) label-skew partition; smaller alpha -> more non-IID."""
    n_classes = int(y.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, part in enumerate(np.split(idx_c, splits)):
            client_indices[client_id].extend(part.tolist())

    # Extreme alpha can leave a client with zero samples; borrow one from the
    # largest client so every client has at least a minimal, trainable slice.
    min_samples = 5
    for client_id, idx in enumerate(client_indices):
        while len(idx) < min_samples:
            donor = max(range(n_clients), key=lambda i: len(client_indices[i]))
            if len(client_indices[donor]) <= min_samples:
                break
            idx.append(client_indices[donor].pop())

    return [np.array(sorted(idx), dtype=int) for idx in client_indices]


def quantity_imbalance_partition(n_samples: int, n_clients: int, rng: np.random.Generator, majority_frac: float = 0.8, majority_clients_frac: float = 0.2) -> list[np.ndarray]:
    """80% of samples concentrated in 20% of nodes (Sec. IV-D scenario 5)."""
    idx = rng.permutation(n_samples)
    n_majority_clients = max(1, int(round(majority_clients_frac * n_clients)))
    n_majority_samples = int(round(majority_frac * n_samples))

    majority_idx, minority_idx = idx[:n_majority_samples], idx[n_majority_samples:]
    majority_splits = np.array_split(majority_idx, n_majority_clients)
    minority_splits = np.array_split(minority_idx, n_clients - n_majority_clients)
    return [np.sort(s) for s in (list(majority_splits) + list(minority_splits))]


def iid_partition(n_samples: int, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(n_samples)
    return [np.sort(s) for s in np.array_split(idx, n_clients)]


def train_test_split_indices(n: int, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac))
    return idx[n_test:], idx[:n_test]


def build_federated_dataset(
    n_clients: int,
    samples_per_client: int,
    seq_len: int,
    n_features: int,
    attack_names: list[str],
    class_sep: float,
    seed: int,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
    test_frac: float = 0.2,
) -> FederatedDataset:
    n_classes = len(attack_names) + 1  # + benign
    n_total = n_clients * samples_per_client
    X, y = generate_sequences(n_total, seq_len, n_features, n_classes, class_sep, noise_sd=1.0, seed=seed)

    rng = np.random.default_rng(seed)
    if partition == "iid":
        partitions = iid_partition(n_total, n_clients, rng)
    elif partition == "quantity_imbalance":
        partitions = quantity_imbalance_partition(n_total, n_clients, rng)
    else:
        partitions = dirichlet_label_partition(y, n_clients, dirichlet_alpha, rng)

    clients = []
    for cid, idx in enumerate(partitions):
        Xc, yc = X[idx], y[idx]
        train_idx, test_idx = train_test_split_indices(len(Xc), test_frac, rng)
        clients.append(ClientData(cid, f"node-{cid}", Xc[train_idx], yc[train_idx], Xc[test_idx], yc[test_idx]))

    return FederatedDataset(clients=clients, n_features=n_features, n_classes=n_classes, attack_names=attack_names)
