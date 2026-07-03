"""Runs all Trustworthy Edge / PoA2 experiments end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import (
    run_ablation,
    run_detection_performance,
    run_hyperparameter_sensitivity,
    run_non_iid,
    run_robust_baselines,
    run_scalability,
    run_security_attacks,
)

SECTIONS = [
    ("Detection performance (Table IV/V/VI)", run_detection_performance),
    ("Non-IID robustness (Table VII)", run_non_iid),
    ("Robust aggregation baselines (Table IX)", run_robust_baselines),
    ("Ablation study (Table XVI)", run_ablation),
    ("Scalability (Table X, XII, XIII)", run_scalability),
    ("Attack surface (Table XV)", run_security_attacks),
    ("Hyperparameter sensitivity (Table XI)", run_hyperparameter_sensitivity),
]


def main() -> None:
    for title, module in SECTIONS:
        print("=" * 70)
        print(title)
        print("=" * 70)
        module.main()
        print()


if __name__ == "__main__":
    main()
