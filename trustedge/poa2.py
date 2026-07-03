"""PoA^2 consensus: Proof of Authority and Association (Sec. III-D).

Implements the dual finalization condition (Eq. 15): a block is finalized
only if BOTH an authority quorum (Eq. 11) AND the association-threshold
(Eq. 13-14) are satisfied. This is the "active" trust-consensus coupling
that distinguishes PoA^2 from passive blockchain-FL designs (Table I, Def. 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .trust import AssociationScore


@dataclass
class Validator:
    validator_id: int
    authority: bool = True          # A_m in {0, 1}, cryptographically certified identity
    is_byzantine: bool = False
    association: AssociationScore = field(default_factory=AssociationScore)

    @property
    def association_score(self) -> float:
        return self.association.score


@dataclass
class ConsensusResult:
    finalized: bool
    authority_ok: bool
    association_ok: bool
    signer_count: int
    mean_association: float
    signing_ids: list[int]


class PoA2Consensus:
    """Sec. III-D: authority validation + association scoring + finalization (Eq. 11-15)."""

    def __init__(self, validators: list[Validator], tau_s: float = 0.6, byzantine_frac_bound: float = 1 / 3):
        self.validators = validators
        self.tau_s = tau_s
        self.byzantine_frac_bound = byzantine_frac_bound

    @property
    def m(self) -> int:
        return len(self.validators)

    def authority_quorum(self) -> int:
        """tau_A^(quorum) = ceil(2M/3) (Eq. 11)."""
        return math.ceil(2 * self.m / 3)

    def run_round(self, block_is_valid: bool, rng: np.random.Generator) -> ConsensusResult:
        """Executes one PBFT-style validation round (Algorithm 1, lines 20-31).

        Honest validators sign iff the block is genuinely valid; Byzantine
        validators sign regardless (attempting to force bad blocks through)
        with 50% probability of also correctly reporting invalid blocks
        (imperfect adversary), matching the "up to f < M/3 validators"
        threat model (Sec. III-J).
        """
        signing_ids: list[int] = []
        for v in self.validators:
            if not v.authority:
                v.association.record(was_valid=False)
                continue
            if v.is_byzantine:
                signs = bool(rng.random() < 0.9)  # Byzantine validators mostly try to force finalization
                correct = signs == block_is_valid
            else:
                signs = block_is_valid
                correct = True
            v.association.record(was_valid=correct)
            if signs:
                signing_ids.append(v.validator_id)

        authority_quorum_met = len(signing_ids) >= self.authority_quorum()
        mean_association = float(np.mean([v.association_score for v in self.validators])) if self.validators else 0.0
        association_ok = mean_association >= self.tau_s

        finalized = authority_quorum_met and association_ok
        return ConsensusResult(
            finalized=finalized,
            authority_ok=authority_quorum_met,
            association_ok=association_ok,
            signer_count=len(signing_ids),
            mean_association=mean_association,
            signing_ids=signing_ids,
        )
