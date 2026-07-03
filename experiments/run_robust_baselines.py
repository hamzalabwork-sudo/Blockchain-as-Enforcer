"""Reproduces Sec. IV-E (Table IX): PoA2 vs Krum / Trimmed Mean / Median /
vanilla FedAvg under 20% poisoning (mixed attacks), averaged over all four
datasets (BiLSTM, binary classification).

Generates:
  results/robust_baselines_table.csv
  results/robust_baselines_chart.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from experiments.plotting import apply_style, method_color
from trustedge.datasets.iotforge import load_iotforge
from trustedge.datasets.iot_cad import load_iot_cad
from trustedge.datasets.ton_iot import load_ton_iot
from trustedge.datasets.wustl_iiot import load_wustl
from trustedge.simulation import run_fl_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = {
    "IoTForge Pro": load_iotforge,
    "ToN-IoT": load_ton_iot,
    "IoT-CAD": load_iot_cad,
    "WUSTL-IIoT-2021": load_wustl,
}
METHODS = ["fedavg", "krum", "trimmed_mean", "median", "poa2"]
METHOD_LABEL = {"fedavg": "Vanilla FL", "krum": "Krum", "trimmed_mean": "Trimmed Mean",
                 "median": "Median", "poa2": "PoA2"}
ROUNDS = 10
POISON_FRAC = 0.2


def pick_malicious(n_clients: int, frac: float, seed: int) -> tuple[set[int], dict[int, str]]:
    """Matches the paper's Table IX methodology: "the same 20% poisoning
    attack (random gradient replacement)" applied uniformly, not a mix."""
    rng = np.random.default_rng(seed)
    n_mal = max(1, int(round(frac * n_clients)))
    ids = sorted(rng.choice(n_clients, size=n_mal, replace=False).tolist())
    return set(ids), {cid: "random_update" for cid in ids}


def main(rounds: int = ROUNDS, seed: int = 0) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    apply_style()

    clean_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    poisoned_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    example_curves: dict[str, list[float]] = {}

    for ds_name, loader in DATASETS.items():
        dataset = loader(n_clients=10, samples_per_client=280, seed=seed)
        malicious, attack_map = pick_malicious(10, POISON_FRAC, seed)

        for method in METHODS:
            clean = run_fl_experiment(dataset, "BiLSTM", method, binary=True, rounds=rounds, lr=0.01, seed=seed)
            poisoned = run_fl_experiment(dataset, "BiLSTM", method, binary=True, rounds=rounds, lr=0.01,
                                          malicious_ids=malicious, attack_assignment=attack_map, seed=seed)
            clean_acc[method].append(clean["final_metrics"]["accuracy"])
            poisoned_acc[method].append(poisoned["final_metrics"]["accuracy"])
            if ds_name == "WUSTL-IIoT-2021":
                example_curves[method] = poisoned["accuracy_curve"]
            print(f"{ds_name:20s} {METHOD_LABEL[method]:14s} clean={clean_acc[method][-1]:.4f} "
                  f"poisoned={poisoned_acc[method][-1]:.4f}")

    rows = []
    for method in METHODS:
        clean_avg = float(np.mean(clean_acc[method]))
        poison_avg = float(np.mean(poisoned_acc[method]))
        drop = (clean_avg - poison_avg) * 100
        rows.append([METHOD_LABEL[method], f"{clean_avg*100:.2f}%", f"{poison_avg*100:.2f}%", f"{drop:.2f}%"])

    with open(RESULTS_DIR / "robust_baselines_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Aggregation Method", "Clean Acc", f"Poisoned Acc ({int(POISON_FRAC*100)}%)", "Drop"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHODS:
        ax.plot(range(1, rounds + 1), example_curves[method], label=METHOD_LABEL[method],
                color=method_color(METHOD_LABEL[method]))
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"WUSTL-IIoT-2021: Robustness Under {int(POISON_FRAC*100)}% Poisoning")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "robust_baselines_chart.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved table and chart to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
