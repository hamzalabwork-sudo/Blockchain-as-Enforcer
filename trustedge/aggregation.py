"""Aggregation rules: trust-weighted (Eq. 9-10), vanilla FedAvg (Eq. 8), and
robust baselines (Krum, Trimmed Mean, Median) used for comparison (Table IX).
"""
from __future__ import annotations

import numpy as np


def fedavg_aggregate(updates: list[np.ndarray], sample_counts: list[int]) -> np.ndarray:
    """w_t = sum_k (N_k/N) w_k,t (Eq. 8)."""
    total = sum(sample_counts)
    return sum(n * u for n, u in zip(sample_counts, updates)) / total


def trust_weighted_aggregate(updates: list[np.ndarray], trust_scores: list[float], sample_counts: list[int]) -> np.ndarray:
    """alpha_k(t) = T_k(t) N_k / sum_j T_j(t) N_j; w_t = sum_k alpha_k(t) w_k,t (Eq. 9-10)."""
    weights = [t * n for t, n in zip(trust_scores, sample_counts)]
    total = sum(weights)
    if total <= 0:
        return fedavg_aggregate(updates, sample_counts)
    return sum(w * u for w, u in zip(weights, updates)) / total


def krum_select(updates: list[np.ndarray], num_byzantine: int) -> np.ndarray:
    """Krum [35]: pick the update closest (sum of squared dist. to n-f-2 nearest) to the honest majority."""
    n = len(updates)
    f = min(num_byzantine, max(0, (n - 3) // 2))
    k = max(1, n - f - 2)
    stacked = np.stack(updates)
    dists = np.linalg.norm(stacked[:, None, :] - stacked[None, :, :], axis=2) ** 2
    scores = np.array([np.sort(dists[i])[1:k + 1].sum() for i in range(n)])
    return updates[int(np.argmin(scores))]


def trimmed_mean(updates: list[np.ndarray], trim_fraction: float = 0.2) -> np.ndarray:
    """Coordinate-wise trimmed mean [36]."""
    stacked = np.stack(updates)
    n = stacked.shape[0]
    k = int(np.floor(trim_fraction * n))
    sorted_vals = np.sort(stacked, axis=0)
    trimmed = sorted_vals[k:n - k] if n - 2 * k > 0 else sorted_vals
    return trimmed.mean(axis=0)


def coordinate_median(updates: list[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median, a common robust-aggregation baseline."""
    return np.median(np.stack(updates), axis=0)
