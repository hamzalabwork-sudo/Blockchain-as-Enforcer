"""Runs REAL federated learning (the same trustedge.simulation engine used
for the local experiments in results/) and anchors each round's ACTUAL
outcome -- not an abstract placeholder -- on the real PureChain testnet:
the round's aggregated-model hash, whether enough clients were admitted
(authority-quorum analogue), and the admitted clients' mean trust score
(association-score analogue).

This is the FL-integrated counterpart to record_consensus_demo.py, which
only exercised the abstract PoA2Consensus/Validator classes with no real
training behind them. Here, a round is only "finalized" on-chain if it's
backed by a genuine BiLSTM training round with real trust-weighted
aggregation under a real (synthetic-data) poisoning attack.

Usage: python run_fl_with_onchain_poa2.py [n_rounds]   (default 10)

Deploys its OWN TrustLedger instance (saved to fl_deployed_address.txt) so
this run's records don't collide with / overwrite record_consensus_demo.py's
round numbering on the other contract.
"""
from __future__ import annotations

import asyncio
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
FL_DEPLOYED_ADDRESS_FILE = HERE / "fl_deployed_address.txt"

N_CLIENTS = 10
MALICIOUS_FRACTION = 0.2  # matches run_robust_baselines.py (Table IX methodology)
ASSOCIATION_THRESHOLD = 0.6  # tau_S, matches trustedge.poa2 default


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

    # --- Phase 1: REAL federated learning, entirely local (no chain calls yet) -------------
    print(f"\nRunning {n_rounds} real BiLSTM federated-learning rounds "
          f"({N_CLIENTS} clients, {int(MALICIOUS_FRACTION*100)}% malicious, WUSTL-IIoT-2021)...")

    dataset = load_wustl(n_clients=N_CLIENTS, samples_per_client=280, seed=0)
    rng = np.random.default_rng(0)
    n_mal = max(1, int(round(MALICIOUS_FRACTION * N_CLIENTS)))
    malicious_ids = set(rng.choice(N_CLIENTS, size=n_mal, replace=False).tolist())
    attack_map = {cid: "random_update" for cid in malicious_ids}
    print(f"Malicious clients (random-gradient-replacement attack): {sorted(malicious_ids)}")

    t0 = time.perf_counter()
    result = run_fl_experiment(
        dataset, "BiLSTM", "poa2", binary=True, rounds=n_rounds, lr=0.01,
        malicious_ids=malicious_ids, attack_assignment=attack_map, seed=0,
    )
    fl_elapsed = time.perf_counter() - t0
    print(f"FL training complete in {fl_elapsed:.1f}s. "
          f"Final accuracy: {result['final_metrics']['accuracy']:.4f} "
          f"(accuracy curve: {[round(a, 3) for a in result['accuracy_curve']]})")

    # --- Phase 2: deploy a fresh TrustLedger and anchor each round's REAL outcome -----------
    print("\nDeploying a fresh TrustLedger instance for this run...")
    address = await ledger.deploy()
    FL_DEPLOYED_ADDRESS_FILE.write_text(address)
    print(f"TrustLedger deployed at: {address}")

    authority_quorum = ceil(2 * N_CLIENTS / 3)
    tx_start = time.perf_counter()
    expected = {}

    for round_id, log in enumerate(result["logs"], start=1):
        authority_ok = len(log.admitted_ids) >= authority_quorum
        association_ok = log.mean_trust >= ASSOCIATION_THRESHOLD
        expected[round_id] = authority_ok and association_ok

        receipt = await record_with_retry(
            ledger,
            round=round_id,
            authority_ok=authority_ok,
            association_ok=association_ok,
            mean_association=log.mean_trust,
            model_hash=log.model_hash,
        )
        print(f"Round {round_id:3d}/{n_rounds}: fl_accuracy={log.accuracy:.4f} "
              f"admitted={len(log.admitted_ids)}/{N_CLIENTS} mean_trust={log.mean_trust:.3f} "
              f"-> finalized={expected[round_id]} tx {receipt['transactionHash'].hex()[:16]}...")

    tx_elapsed = time.perf_counter() - tx_start
    print(f"\nAnchored {n_rounds} real FL rounds on-chain in {tx_elapsed:.1f}s.")

    # --- Phase 3: verify every on-chain record matches what was actually computed ----------
    print("\nVerifying all rounds against on-chain state...")
    mismatches = 0
    for round_id in range(1, n_rounds + 1):
        record = await ledger.get_record(round_id)
        if bool(record[3]) != expected[round_id]:
            mismatches += 1
            print(f"  MISMATCH round {round_id}: expected finalized={expected[round_id]}, on-chain={bool(record[3])}")

    if mismatches == 0:
        print(f"All {n_rounds} rounds verified: on-chain consensus record matches the real FL training outcome.")
    else:
        print(f"{mismatches}/{n_rounds} rounds did not match -- investigate above.")

    print(f"\nSummary: final FL accuracy {result['final_metrics']['accuracy']:.4f} "
          f"({fl_elapsed:.1f}s training + {tx_elapsed:.1f}s anchoring), "
          f"contract {address}")


if __name__ == "__main__":
    rounds_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(main(rounds_arg))
