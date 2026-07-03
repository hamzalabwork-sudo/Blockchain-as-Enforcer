"""Reproduces Sec. IV-F (Table X, XII, XIII): PoA2 scalability, analytically
modeled by PoA2Simulator (see trustedge/blockchain/poa2_simulator.py for why
this is analytic rather than a live 5,000-validator deployment).

Generates:
  results/scalability_table.csv
  results/scalability_chart.png
  results/communication_overhead_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import matplotlib.pyplot as plt

from experiments.plotting import CATEGORICAL, apply_style
from trustedge.blockchain.poa2_simulator import PoA2Simulator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CLIENT_COUNTS = [10, 50, 100, 200, 500, 1000, 5000]

BILSTM_MODEL_MB = 8.2  # Sec. IV-G (Table XIII): avg. BiLSTM model size reported in the paper


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    apply_style()
    sim = PoA2Simulator()

    rows = []
    for n in CLIENT_COUNTS:
        r = sim.round_latency_ms(n)
        rows.append([n, r.n_validators, f"{r.round_latency_ms:.0f}", f"{r.tps:.0f}", f"{r.consensus_ms:.0f}"])
        print(f"clients={n:5d} validators={r.n_validators:3d} latency={r.round_latency_ms:8.0f}ms "
              f"tps={r.tps:6.0f} consensus={r.consensus_ms:6.1f}ms")

    with open(RESULTS_DIR / "scalability_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Clients", "Validators", "Round Latency (ms)", "TPS", "Consensus Time (ms)"])
        w.writerows(rows)

    # --- communication overhead (Table XIII) --------------------------------------------
    r100 = sim.round_latency_ms(100)
    comm_rows = [
        ["Avg. Model Size (BiLSTM)", f"{BILSTM_MODEL_MB} MB"],
        ["Upload Size per Round", f"{BILSTM_MODEL_MB} MB"],
        ["Download Size per Round", f"{BILSTM_MODEL_MB} MB"],
        ["Total Communication/Round", f"{2*BILSTM_MODEL_MB} MB"],
        ["Blockchain Anchoring Delay (Table XII)", f"{r100.consensus_ms:.0f} ms"],
        ["Total Round Latency (100 clients)", f"{r100.round_latency_ms:.0f} ms"],
    ]
    with open(RESULTS_DIR / "communication_overhead_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerows(comm_rows)

    # --- chart: latency + TPS vs. client count -------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    latencies = [sim.round_latency_ms(n).round_latency_ms for n in CLIENT_COUNTS]
    tps_vals = [sim.round_latency_ms(n).tps for n in CLIENT_COUNTS]

    ax1.plot(CLIENT_COUNTS, latencies, marker="o", color=CATEGORICAL["blue"])
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Clients")
    ax1.set_ylabel("Round Latency (ms)")
    ax1.set_title("PoA2 Round Latency Scaling")

    ax2.plot(CLIENT_COUNTS, tps_vals, marker="o", color=CATEGORICAL["orange"])
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Clients")
    ax2.set_ylabel("Throughput (TPS)")
    ax2.set_title("PoA2 Throughput Scaling")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "scalability_chart.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved tables and chart to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
