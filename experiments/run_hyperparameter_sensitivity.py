"""Reproduces Sec. IV-I (Table XI): PoA2 hyperparameter sensitivity --
sweeps the exclusion threshold theta_t (client-trust gate, reported as the
highest accuracy-impact parameter) and the trust penalty factor beta,
under 20% poisoning (WUSTL-IIoT-2021, BiLSTM, binary).

Generates:
  results/hyperparameter_sensitivity_table.csv
  results/hyperparameter_sensitivity_chart.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from experiments.plotting import BLUE_RAMP, apply_style
from trustedge.datasets.wustl_iiot import load_wustl
from trustedge.simulation import run_fl_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
THETA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
BETA_VALUES = [0.3, 0.5, 0.7, 0.9]
ROUNDS = 10
POISON_FRAC = 0.2


def pick_malicious(n_clients: int, frac: float, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    n_mal = max(1, int(round(frac * n_clients)))
    return set(rng.choice(n_clients, size=n_mal, replace=False).tolist())


def main(rounds: int = ROUNDS, seed: int = 0) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    apply_style()

    dataset = load_wustl(n_clients=10, samples_per_client=280, seed=seed)
    malicious = pick_malicious(10, POISON_FRAC, seed)
    attack_map = {cid: "random_update" for cid in malicious}

    theta_curves, theta_finals = [], []
    for theta in THETA_VALUES:
        res = run_fl_experiment(dataset, "BiLSTM", "poa2", binary=True, rounds=rounds, lr=0.01,
                                 malicious_ids=malicious, attack_assignment=attack_map, theta_t=theta, seed=seed)
        theta_curves.append(res["accuracy_curve"])
        theta_finals.append(res["final_metrics"]["accuracy"])
        print(f"theta_t={theta}  final_acc={res['final_metrics']['accuracy']:.4f}")

    beta_finals = []
    for beta_val in BETA_VALUES:
        res = run_fl_experiment(dataset, "BiLSTM", "poa2", binary=True, rounds=rounds, lr=0.01,
                                 malicious_ids=malicious, attack_assignment=attack_map, theta_t=0.3,
                                 trust_beta=beta_val, seed=seed)
        beta_finals.append(res["final_metrics"]["accuracy"])
        print(f"beta={beta_val}  final_acc={res['final_metrics']['accuracy']:.4f}")

    theta_range = (max(theta_finals) - min(theta_finals)) * 100
    beta_range = (max(beta_finals) - min(beta_finals)) * 100

    with open(RESULTS_DIR / "hyperparameter_sensitivity_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter"] + [str(v) for v in THETA_VALUES] + ["Accuracy Range (pp)"])
        w.writerow(["theta_t (exclusion threshold)"] + [f"{a*100:.2f}%" for a in theta_finals] + [f"{theta_range:.2f}"])
        w.writerow(["beta (penalty factor)"] + [f"{a*100:.2f}%" for a in beta_finals] +
                    [""] * (len(THETA_VALUES) - len(BETA_VALUES)) + [f"{beta_range:.2f}"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (theta, curve) in enumerate(zip(THETA_VALUES, theta_curves)):
        ax1.plot(range(1, rounds + 1), curve, label=f"theta_t={theta}", color=BLUE_RAMP[i % len(BLUE_RAMP)])
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Sensitivity to Exclusion Threshold theta_t")
    ax1.legend(fontsize=8)

    ax2.plot(BETA_VALUES, [b * 100 for b in beta_finals], marker="o", color=BLUE_RAMP[2])
    ax2.set_xlabel("Trust Penalty Factor (beta)")
    ax2.set_ylabel("Final Accuracy (%)")
    ax2.set_title("Sensitivity to Penalty Factor beta")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "hyperparameter_sensitivity_chart.png", dpi=150)
    plt.close(fig)

    print(f"\ntheta_t accuracy range: {theta_range:.2f}pp, beta accuracy range: {beta_range:.2f}pp")
    print(f"Saved table and chart to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
