"""Reproduces Sec. IV-G (Table XVI): component-contribution ablation --
Centralized / Vanilla FL / Authority-only / Association-only / Full PoA2,
under 20% poisoning (random gradient replacement), averaged over all four
datasets (BiLSTM, binary classification).

Generates:
  results/ablation_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import time
import numpy as np

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
METHODS = ["centralized", "fedavg", "authority_only", "association_only", "poa2"]
METHOD_LABEL = {
    "centralized": "Centralized Learning", "fedavg": "Vanilla FL",
    "authority_only": "FL + PoA (only authority)", "association_only": "FL + Association (only)",
    "poa2": "FL + PoA2 (full)",
}
ROUNDS = 10
POISON_FRAC = 0.2


def pick_malicious(n_clients: int, frac: float, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    n_mal = max(1, int(round(frac * n_clients)))
    return set(rng.choice(n_clients, size=n_mal, replace=False).tolist())


def main(rounds: int = ROUNDS, seed: int = 0) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    clean_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    poison_acc: dict[str, list[float]] = {m: [] for m in METHODS}
    latency: dict[str, list[float]] = {m: [] for m in METHODS}

    for ds_name, loader in DATASETS.items():
        dataset = loader(n_clients=10, samples_per_client=280, seed=seed)
        malicious = pick_malicious(10, POISON_FRAC, seed)
        attack_map = {cid: "random_update" for cid in malicious}

        for method in METHODS:
            rounds_here = rounds if method != "centralized" else 8
            t0 = time.perf_counter()
            clean = run_fl_experiment(dataset, "BiLSTM", method, binary=True, rounds=rounds_here, lr=0.01, seed=seed)
            elapsed_clean = (time.perf_counter() - t0) * 1000 / max(rounds_here, 1)

            poisoned = run_fl_experiment(dataset, "BiLSTM", method, binary=True, rounds=rounds_here, lr=0.01,
                                          malicious_ids=malicious, attack_assignment=attack_map, seed=seed)

            clean_acc[method].append(clean["final_metrics"]["accuracy"])
            poison_acc[method].append(poisoned["final_metrics"]["accuracy"])
            latency[method].append(elapsed_clean)
            print(f"{ds_name:20s} {METHOD_LABEL[method]:28s} clean={clean_acc[method][-1]:.4f} "
                  f"poisoned={poison_acc[method][-1]:.4f}")

    rows = []
    for method in METHODS:
        c = float(np.mean(clean_acc[method]))
        p = float(np.mean(poison_acc[method]))
        lat = float(np.mean(latency[method]))
        rows.append([METHOD_LABEL[method], f"{c*100:.2f}%", f"{p*100:.2f}%", f"{(c-p)*100:.2f}%", f"{lat:.1f}"])

    with open(RESULTS_DIR / "ablation_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Framework", "Clean Acc", f"{int(POISON_FRAC*100)}% Poison Acc", "Drop", "Per-Round Latency (ms, illustrative)"])
        w.writerows(rows)

    print(f"\nSaved table to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
