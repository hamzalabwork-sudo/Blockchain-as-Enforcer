"""Reproduces Sec. IV-H (Table XV): attack-surface success rates with and
without PoA2's defense mechanisms.

Generates:
  results/attack_surface_table.csv
  results/attack_surface_chart.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import matplotlib.pyplot as plt

from experiments.plotting import CATEGORICAL, apply_style
from trustedge.security_sim import ATTACK_SIMULATORS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    apply_style()

    rows = []
    names, no_defense, with_defense = [], [], []
    for name, (fn, mechanism) in ATTACK_SIMULATORS.items():
        nd = fn(defended=False)
        wd = fn(defended=True)
        rows.append([name, f"{nd*100:.1f}%", f"{wd*100:.1f}%", mechanism])
        names.append(name)
        no_defense.append(nd * 100)
        with_defense.append(wd * 100)
        print(f"{name:30s} no_defense={nd*100:5.1f}%  with_poa2={wd*100:5.1f}%  ({mechanism})")

    with open(RESULTS_DIR / "attack_surface_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Attack Type", "Success Rate (No Defense)", "Success Rate (With PoA2)", "Defense Mechanism"])
        w.writerows(rows)

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, no_defense, width, label="No Defense", color=CATEGORICAL["red"])
    ax.bar(x + width / 2, with_defense, width, label="With PoA2", color=CATEGORICAL["blue"])
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack-Surface Analysis and Defense Effectiveness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "attack_surface_chart.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved table and chart to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
