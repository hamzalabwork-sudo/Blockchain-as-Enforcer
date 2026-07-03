"""Detection models: CNN, LSTM, BiLSTM (Sec. III-C.1, Sec. IV-A).

Input is a sequence of feature vectors (batch, L, d), matching the paper's
detection-engine signature f_theta: R^{L x d} -> R^C (Sec. III-H). Hidden
sizes are reduced from the paper's reported 128/256 units to keep CPU
training tractable for this reproduction (see README for details) while
preserving the same architectural ordering (CNN < LSTM < BiLSTM capacity).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CNNDetector(nn.Module):
    """3 conv layers (Sec. III-C.1) -> limited receptive field over the sequence."""

    def __init__(self, n_features: int, n_classes: int, channels: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, d) -> (batch, d, L) for Conv1d
        h = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.fc(h)


class LSTMDetector(nn.Module):
    """2-layer unidirectional LSTM (Sec. III-C.1)."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class BiLSTMDetector(nn.Module):
    """Bidirectional LSTM (Sec. III-C.1) -- the paper's best-performing architecture."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # forward + backward final states
        return self.fc(h)


MODEL_REGISTRY = {"CNN": CNNDetector, "LSTM": LSTMDetector, "BiLSTM": BiLSTMDetector}


def build_model(name: str, n_features: int, n_classes: int) -> nn.Module:
    return MODEL_REGISTRY[name](n_features, n_classes)


# --- flatten / unflatten helpers for aggregation ------------------------------------------

def flatten_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])


def load_flat_params(model: nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = flat[offset:offset + n].reshape(p.shape)
            p.copy_(torch.as_tensor(chunk, dtype=p.dtype))
            offset += n


def clone_params(model: nn.Module) -> np.ndarray:
    return flatten_params(model).copy()


def train_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Runs local SGD (Adam) training and returns the flattened updated parameters."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    n = len(Xt)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optim.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            optim.step()

    return flatten_params(model)


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    logits = model(Xt)
    preds = logits.argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="macro", zero_division=0)

    if logits.shape[1] == 2:
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = float("nan")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}
