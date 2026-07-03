"""Runs real PoA2 consensus rounds (trustedge.poa2 -- the same logic used in
the local simulation) and anchors each round's finalization decision on the
real, deployed TrustLedger contract. Then reads every record back from-chain
to prove it was genuinely persisted.

Usage: python record_consensus_demo.py [n_rounds]   (default 5)

Each round's block is randomly valid (~85%) or an invalid block a Byzantine
coalition is attempting to push through (~15%), so a longer run produces a
realistic mix of finalized/rejected on-chain records rather than repeating
one trivial outcome.

Run setup_wallet.py (or create .wallet/owner_key.txt yourself) and
deploy_trust_ledger.py first.
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wallet import resolve_private_key

HERE = Path(__file__).resolve().parent
DEPLOYED_ADDRESS_FILE = HERE / "deployed_address.txt"

N_VALIDATORS = 7
BYZANTINE_FRACTION = 0.2  # under the f < M/3 bound (Assumption / Sec. III-J)
VALID_BLOCK_PROB = 0.85
MAX_RETRIES = 5
RETRY_DELAY_S = 3.0


async def _record_with_retry(ledger, **kwargs):
    """PureChain's on-demand block production can lag behind eth_getTransactionCount
    ("latest"), causing transient 'nonce too low' errors on rapid successive
    transactions. Retrying after a short delay lets the node's state catch up."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await ledger.record_consensus(**kwargs)
        except Exception as e:
            if "nonce too low" in str(e) and attempt < MAX_RETRIES:
                print(f"  (nonce race, retrying in {RETRY_DELAY_S}s -- attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY_S)
                continue
            raise


async def main(n_rounds: int) -> None:
    if not DEPLOYED_ADDRESS_FILE.exists():
        print("No deployed contract found -- run deploy_trust_ledger.py first.")
        return

    from trustedge.blockchain.purechain_client import PureChainTrustLedger
    from trustedge.poa2 import PoA2Consensus, Validator

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

    address = DEPLOYED_ADDRESS_FILE.read_text().strip()
    factory_source = (Path(__file__).resolve().parents[1] / "trustedge" / "blockchain" / "contracts" / "TrustLedger.sol").read_text()
    factory = await ledger.pc.contract(factory_source)
    ledger.contract = factory.attach(address)
    print(f"Attached to TrustLedger at {address}")
    print(f"Sending {n_rounds} real transactions (zero gas)...\n")

    rng = np.random.default_rng(0)
    n_byz = max(1, int(round(BYZANTINE_FRACTION * N_VALIDATORS)))
    validators = [Validator(i, authority=True, is_byzantine=(i < n_byz)) for i in range(N_VALIDATORS)]
    consensus = PoA2Consensus(validators, tau_s=0.6)

    start = time.perf_counter()
    finalized_count = 0
    tx_hashes = []
    expected_finalized: dict[int, bool] = {}

    for round_id in range(1, n_rounds + 1):
        block_is_valid = bool(rng.random() < VALID_BLOCK_PROB)
        result = consensus.run_round(block_is_valid=block_is_valid, rng=rng)
        model_hash = hashlib.sha256(f"round-{round_id}".encode()).digest()

        receipt = await _record_with_retry(
            ledger,
            round=round_id,
            authority_ok=result.authority_ok,
            association_ok=result.association_ok,
            mean_association=result.mean_association,
            model_hash=model_hash,
        )
        tx_hash = receipt["transactionHash"].hex()
        tx_hashes.append(tx_hash)
        expected_finalized[round_id] = result.finalized
        finalized_count += int(result.finalized)

        print(f"Round {round_id:4d}/{n_rounds}: block_valid={block_is_valid!s:5s} "
              f"finalized={result.finalized!s:5s} mean_association={result.mean_association:.3f} "
              f"-> tx {tx_hash[:16]}...")

    elapsed = time.perf_counter() - start
    print(f"\nSent {n_rounds} transactions in {elapsed:.1f}s "
          f"({finalized_count} finalized, {n_rounds - finalized_count} rejected).")

    print("\nReading all records back on-chain to verify persistence...")
    mismatches = 0
    for round_id in range(1, n_rounds + 1):
        record = await ledger.get_record(round_id)
        on_chain_finalized = bool(record[3])
        if on_chain_finalized != expected_finalized[round_id]:
            mismatches += 1
            print(f"  MISMATCH round {round_id}: expected finalized={expected_finalized[round_id]}, "
                  f"on-chain={on_chain_finalized}")

    if mismatches == 0:
        print(f"All {n_rounds} rounds verified: on-chain state matches what was sent, exactly.")
    else:
        print(f"{mismatches}/{n_rounds} rounds did NOT match what was sent -- investigate above.")

    sample_ids = sorted(set([1, n_rounds] + list(range(1, n_rounds + 1, max(1, n_rounds // 5)))))
    print("\nSample records:")
    for round_id in sample_ids:
        record = await ledger.get_record(round_id)
        print(f"  round {record[0]}: finalized={record[3]}, mean_association_scaled={record[4]}")


if __name__ == "__main__":
    rounds_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    asyncio.run(main(rounds_arg))
