"""Adversarial behaviors matching Sec. IV-B (Attack Simulation) and Table XIV/XV.

Data/model-poisoning attacks operate on a client's local data or trained
update; Sybil/replay/collusion attacks operate at the blockchain layer and
are handled directly in the PoA2 consensus simulation (trustedge.poa2).
"""
from __future__ import annotations

import numpy as np


def label_flip(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Locally inverted labels for malicious nodes (Table III: 'Attack Labels')."""
    return (n_classes - 1) - y


def gradient_scaling(update: np.ndarray, global_params: np.ndarray, amplification: float = 4.0) -> np.ndarray:
    """Malicious gradient manipulation: amplifies the update's deviation from the global model."""
    delta = update - global_params
    return global_params + amplification * delta


def gradient_reversal(update: np.ndarray, global_params: np.ndarray, amplification: float = 4.0) -> np.ndarray:
    """Byzantine random/targeted update: reverses and amplifies the honest direction."""
    delta = update - global_params
    return global_params - amplification * delta


def random_update(global_params: np.ndarray, scale: float, rng: np.random.Generator,
                   shared_direction: np.ndarray | None = None) -> np.ndarray:
    """20% poisoning baseline used in Table IX/XVI: random gradient replacement.

    `shared_direction`, if given, is a single unit vector reused by every
    colluding malicious client in a round -- independent per-client noise
    self-cancels once averaged with 8+ honest updates (central-limit
    behavior), which would make this attack harmless by construction
    regardless of magnitude. Real Byzantine "random replacement" attacks
    coordinate on a shared perturbation direction; see simulation.py.
    """
    if shared_direction is not None:
        return global_params + shared_direction * scale
    noise = rng.normal(0, scale, size=global_params.shape)
    return global_params + noise


def apply_client_attack(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    attack_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies a data-level attack before local training. attack_type in
    {"label_flip", "none"}; model-level attacks (gradient_scaling/reversal/random)
    are applied post-training by the caller (see simulation.py)."""
    if attack_type == "label_flip":
        return X, label_flip(y, n_classes)
    return X, y


MODEL_LEVEL_ATTACKS = ("gradient_scaling", "gradient_reversal", "random_update")
DATA_LEVEL_ATTACKS = ("label_flip",)
ALL_ATTACK_TYPES = DATA_LEVEL_ATTACKS + MODEL_LEVEL_ATTACKS
