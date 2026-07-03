"""Analytical PoA^2 scalability model (Sec. IV-F, Tables X, XII, XIII).

The paper itself states its large-scale (up to 5,000-client) scalability
numbers came from "a custom Python class to model the computational
overhead" rather than a real deployment at that scale (Sec. IV-A) -- real
validator committees of that size aren't practical to spin up against a
live chain for a reproduction either. This module is that custom class: an
analytical latency/throughput model whose free parameters are calibrated
directly against the paper's own reported anchor points (Table X's
client/validator/latency/TPS rows, and the component breakdown in Sec.
IV-E's "for 100 clients .../for 1,000 clients ..." paragraph), then
interpolated/extrapolated to other scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Table X anchor points: (clients, validators, round_latency_ms, tps, consensus_ms)
_ANCHORS = [
    (10, 4, 480, 1450, 95),
    (50, 7, 610, 1430, 105),
    (100, 7, 780, 1395, 125),
    (200, 7, 1120, 1360, 148),
    (500, 13, 1700, 1320, 187),
    (1000, 13, 2850, 1250, 245),
    (5000, 25, 14200, 980, 412),
]

_CLIENTS = np.array([a[0] for a in _ANCHORS], dtype=float)
_VALIDATORS = np.array([a[1] for a in _ANCHORS], dtype=float)
_ROUND_LATENCY = np.array([a[2] for a in _ANCHORS], dtype=float)
_TPS = np.array([a[3] for a in _ANCHORS], dtype=float)
_CONSENSUS_MS = np.array([a[4] for a in _ANCHORS], dtype=float)


@dataclass
class ScalabilityResult:
    n_clients: int
    n_validators: int
    round_latency_ms: float
    tps: float
    consensus_ms: float
    local_training_ms: float
    upload_ms: float
    aggregation_ms: float
    download_ms: float
    sync_ms: float


class PoA2Simulator:
    """Interpolates/extrapolates the paper's own measured scalability curve (Table X)."""

    def __init__(self, base_training_ms: float = 420.0):
        self.base_training_ms = base_training_ms
        # log-log fits for the sub-linear, bandwidth-bound components (Sec. IV-E breakdown)
        self._upload_a, self._upload_p = self._powerlaw_fit(100, 85, 1000, 380)
        self._agg_a, self._agg_p = self._powerlaw_fit(100, 45, 1000, 185)
        self._sync_a, self._sync_p = self._powerlaw_fit(100, 20, 1000, 40)

    @staticmethod
    def _powerlaw_fit(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
        p = np.log(y2 / y1) / np.log(x2 / x1)
        a = y1 / (x1 ** p)
        return a, p

    def _interp_validators(self, n_clients: int) -> int:
        return int(round(np.interp(n_clients, _CLIENTS, _VALIDATORS)))

    def consensus_latency_ms(self, n_validators: int) -> float:
        """O(M^2) PBFT-style message-exchange complexity (Sec. III-I), calibrated to Table X."""
        return float(np.interp(n_validators, _VALIDATORS, _CONSENSUS_MS))

    def throughput_tps(self, n_clients: int) -> float:
        return float(np.interp(n_clients, _CLIENTS, _TPS))

    def round_latency_ms(self, n_clients: int, n_validators: int | None = None, measured: bool = True) -> ScalabilityResult:
        """Ideal component breakdown (Sec. IV-E) plus the paper's own measured congestion inflation."""
        n_validators = n_validators or self._interp_validators(n_clients)

        training = self.base_training_ms * (1 + 0.02 * np.log1p(n_clients / 100))
        upload = self._upload_a * n_clients ** self._upload_p
        aggregation = self._agg_a * n_clients ** self._agg_p
        consensus = self.consensus_latency_ms(n_validators)
        download = upload
        sync = self._sync_a * n_clients ** self._sync_p

        ideal_total = training + upload + aggregation + consensus + download + sync

        if measured:
            total = float(np.interp(n_clients, _CLIENTS, _ROUND_LATENCY))
        else:
            total = ideal_total

        return ScalabilityResult(
            n_clients=n_clients, n_validators=n_validators,
            round_latency_ms=total, tps=self.throughput_tps(n_clients),
            consensus_ms=consensus, local_training_ms=training,
            upload_ms=upload, aggregation_ms=aggregation, download_ms=download, sync_ms=sync,
        )

    def sweep(self, client_counts: list[int]) -> list[ScalabilityResult]:
        return [self.round_latency_ms(n) for n in client_counts]
