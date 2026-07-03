"""Reproduces Sec. IV-D (Table VII): non-IID robustness of the full PoA2
mechanism across five heterogeneity scenarios, averaged over all four
datasets (BiLSTM, binary classification).

Generates:
  results/non_iid_table.csv
  results/non_iid_chart.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from experiments.plotting import CATEGORICAL, apply_style
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
SCENARIOS = [
    ("IID", "iid", None),
    ("alpha=0.5", "dirichlet", 0.5),
    ("alpha=0.3", "dirichlet", 0.3),
    ("alpha=0.1", "dirichlet", 0.1),
    ("Quantity Imbalance", "quantity_imbalance", None),
]
ROUNDS = 10


def main(rounds: int = ROUNDS, seed: int = 0) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    apply_style()

    table: dict[str, dict[str, float]] = {name: {} for name, _, _ in SCENARIOS}

    for ds_name, loader in DATASETS.items():
        for scenario_name, partition, alpha in SCENARIOS:
            dataset = loader(n_clients=8, samples_per_client=250, seed=seed,
                              partition=partition, dirichlet_alpha=alpha or 0.5)
            res = run_fl_experiment(dataset, "BiLSTM", "poa2", binary=True, rounds=rounds, lr=0.01, seed=seed)
            acc = res["final_metrics"]["accuracy"]
            table[scenario_name][ds_name] = acc
            print(f"{ds_name:20s} {scenario_name:20s} acc={acc:.4f}")

    iid_avg = np.mean(list(table["IID"].values()))
    rows = []
    for scenario_name, _, _ in SCENARIOS:
        avg = np.mean(list(table[scenario_name].values()))
        drop = (iid_avg - avg) * 100
        row = [scenario_name] + [f"{table[scenario_name][d]*100:.2f}%" for d in DATASETS] + [f"{avg*100:.2f}%", f"{drop:.2f}%"]
        rows.append(row)

    with open(RESULTS_DIR / "non_iid_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Scenario"] + list(DATASETS.keys()) + ["Average", "Drop vs IID"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(SCENARIOS))
    colors = [CATEGORICAL["blue"], CATEGORICAL["aqua"], CATEGORICAL["yellow"], CATEGORICAL["orange"], CATEGORICAL["red"]]
    for i, ds_name in enumerate(DATASETS):
        vals = [table[s][ds_name] * 100 for s, _, _ in SCENARIOS]
        ax.plot(x, vals, marker="o", label=ds_name, color=colors[i % len(colors)])
    ax.set_xticks(x, [s for s, _, _ in SCENARIOS], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Non-IID Robustness (PoA2, BiLSTM)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "non_iid_chart.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved table and chart to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
