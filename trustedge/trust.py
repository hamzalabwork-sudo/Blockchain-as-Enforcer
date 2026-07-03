"""Trust dynamics and association scoring (Sec. III-C.3, III-D, III-E).

Every client and validator maintains a trust/association score in [0, 1]
that evolves according to Eq. 16: penalized on detected malicious behavior,
otherwise recovered by a small reward step.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrustState:
    """T_k(t) in [0, 1] for a single client or validator (Eq. 16)."""

    score: float = 1.0
    beta: float = 0.7   # penalty factor (0 < beta < 1), Sec. IV-A default
    delta: float = 0.05  # reward step, Sec. IV-A default
    history: list[float] = field(default_factory=list)

    def update(self, malicious_detected: bool) -> float:
        """T_k(t+1) = beta*T_k(t) if malicious else min(1, T_k(t) + delta)."""
        if malicious_detected:
            self.score = self.beta * self.score
        else:
            self.score = min(1.0, self.score + self.delta)
        self.history.append(self.score)
        return self.score


class AssociationScore:
    """S_m(t) = Valid Contributions_m(t) / Total Contributions_m(t) (Eq. 12)."""

    def __init__(self):
        self.valid = 0
        self.total = 0

    @property
    def score(self) -> float:
        return self.valid / self.total if self.total > 0 else 1.0

    def record(self, was_valid: bool) -> float:
        self.total += 1
        if was_valid:
            self.valid += 1
        return self.score


def exclusion_rule(trust_score: float, theta_t: float) -> bool:
    """Automatic Exclusion Rule: True (excluded) if T_k(t) < theta_t."""
    return trust_score < theta_t
