"""Closes the gap between the local simulation and the on-chain deployment:
run_fl_with_onchain_poa2.py only ever anchored the "poa2" method. This
script runs ALL EIGHT aggregation methods from the local ablation/robust-
baseline experiments (Table IX + Table XVI) for real and anchors every
method's real per-round outcome on a single shared TrustLedger contract.

Deliberately NOT included (left as local-simulation-only, since they test
properties that don't depend on whether a real blockchain is involved):
model-architecture comparison (CNN/LSTM/BiLSTM -- BiLSTM is fixed here,
matching the paper's best/headline model), non-IID partition sweeps,
hyperparameter sensitivity, and the analytical scalability/attack-surface
simulations (which the paper itself only ever modeled analytically too,
see poa2_simulator.py).

Usage: python run_full_ablation_onchain.py [n_rounds]   (default 10;
       "centralized" always uses exactly 1 on-chain record regardless,
       since it's one-shot training, not federated rounds)
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chain_utils import record_with_retry
from wallet import resolve_private_key

HERE = Path(__file__).resolve().parent
FULL_ABLATION_ADDRESS_FILE = HERE / "full_ablation_address.txt"

N_CLIENTS = 10
MALICIOUS_FRACTION = 0.2  # matches Table IX/XVI methodology
ASSOCIATION_THRESHOLD = 0.6  # tau_S, matches trustedge.poa2 default

METHODS = ["centralized", "fedavg", "authority_only", "association_only", "poa2", "krum", "trimmed_mean", "median"]
METHOD_LABEL = {
    "centralized": "Centralized Learning", "fedavg": "Vanilla FL",
    "authority_only": "FL + PoA (only authority)", "association_only": "FL + Association (only)",
    "poa2": "FL + PoA2 (full)", "krum": "Krum", "trimmed_mean": "Trimmed Mean", "median": "Median",
}


async def main(n_rounds: int) -> None:
    from trustedge.blockchain.purechain_client import PureChainTrustLedger
    from trustedge.datasets.wustl_iiot import load_wustl
    from trustedge.simulation import run_fl_experiment

    try:
        private_key, source = resolve_private_key()
    except FileNotFoundError as e:
        print(e)
        return

    ledger = PureChainTrustLedger(network="testnet")
    await ledger.connect_with_key(private_key)
    print(f"Signing with {source} key -> address {ledger.pc.signer.address}")

    if not ledger.pc.web3.is_connected():
        print("Not connected -- see README.md in this folder for TLS troubleshooting.")
        return

    dataset = load_wustl(n_clients=N_CLIENTS, samples_per_client=280, seed=0)
    rng = np.random.default_rng(0)
    n_mal = max(1, int(round(MALICIOUS_FRACTION * N_CLIENTS)))
    malicious_ids = set(rng.choice(N_CLIENTS, size=n_mal, replace=False).tolist())
    attack_map = {cid: "random_update" for cid in malicious_ids}
    print(f"Malicious clients (random-gradient-replacement attack): {sorted(malicious_ids)}\n")

    print("Deploying a fresh TrustLedger instance for this comparison...")
    address = await ledger.deploy()
    FULL_ABLATION_ADDRESS_FILE.write_text(address)
    print(f"TrustLedger deployed at: {address}\n")

    authority_quorum = ceil(2 * N_CLIENTS / 3)
    summary_rows = []
    all_expected: dict[int, bool] = {}
    total_tx = 0
    fl_time_total = 0.0
    tx_time_total = 0.0

    for method_idx, method in enumerate(METHODS):
        rounds_here = 1 if method == "centralized" else n_rounds
        print(f"=== {METHOD_LABEL[method]} ({method}) ===")

        t0 = time.perf_counter()
        result = run_fl_experiment(
            dataset, "BiLSTM", method, binary=True, rounds=rounds_here, lr=0.01,
            malicious_ids=malicious_ids, attack_assignment=attack_map, seed=0,
        )
        fl_elapsed = time.perf_counter() - t0
        fl_time_total += fl_elapsed
        final_acc = result["final_metrics"]["accuracy"]
        print(f"  FL training done in {fl_elapsed:.1f}s, final accuracy {final_acc:.4f}")

        # Build one on-chain record per round (centralized: a single synthetic
        # round from its one-shot trained model, since it has no per-round logs).
        if method == "centralized":
            model_hash = hashlib.sha256(result["final_params"].tobytes()).digest()
            round_records = [(1, True, True, 1.0, model_hash)]
        else:
            round_records = []
            for round_id, log in enumerate(result["logs"], start=1):
                authority_ok = len(log.admitted_ids) >= authority_quorum
                association_ok = log.mean_trust >= ASSOCIATION_THRESHOLD
                round_records.append((round_id, authority_ok, association_ok, log.mean_trust, log.model_hash))

        t0 = time.perf_counter()
        for round_id, authority_ok, association_ok, mean_assoc, model_hash in round_records:
            chain_round_id = method_idx * 1000 + round_id
            receipt = await record_with_retry(
                ledger, round=chain_round_id, authority_ok=authority_ok, association_ok=association_ok,
                mean_association=mean_assoc, model_hash=model_hash,
            )
            all_expected[chain_round_id] = authority_ok and association_ok
            total_tx += 1
            print(f"    round {round_id}/{len(round_records)} -> chain_round={chain_round_id} "
                  f"finalized={all_expected[chain_round_id]} tx {receipt['transactionHash'].hex()[:16]}...")
        method_tx_time = time.perf_counter() - t0
        tx_time_total += method_tx_time

        summary_rows.append((METHOD_LABEL[method], final_acc, len(round_records), method_tx_time))
        print()

    print(f"Sent {total_tx} real transactions across {len(METHODS)} methods "
          f"({fl_time_total:.1f}s training + {tx_time_total:.1f}s anchoring).\n")

    print("Verifying every method's records against on-chain state...")
    mismatches = 0
    for chain_round_id, expected_finalized in all_expected.items():
        record = await ledger.get_record(chain_round_id)
        if bool(record[3]) != expected_finalized:
            mismatches += 1
            print(f"  MISMATCH chain_round {chain_round_id}: expected={expected_finalized}, on-chain={bool(record[3])}")

    if mismatches == 0:
        print(f"All {total_tx} records verified: on-chain state matches every method's real training outcome.\n")
    else:
        print(f"{mismatches}/{total_tx} records did not match -- investigate above.\n")

    print(f"{'Method':30s} {'Final Acc':>10s} {'Rounds':>8s}")
    for label, acc, n, _ in summary_rows:
        print(f"{label:30s} {acc*100:9.2f}% {n:8d}")

    print(f"\nContract: {address}")


if __name__ == "__main__":
    rounds_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(main(rounds_arg))
