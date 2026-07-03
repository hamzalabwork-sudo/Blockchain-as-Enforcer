"""Runs a few real PoA2 consensus rounds (trustedge.poa2 -- the same logic
used in the local simulation) and anchors each round's finalization
decision on the real, deployed TrustLedger contract. Then reads every
record back from-chain to prove it was genuinely persisted.

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
N_ROUNDS = 5
BYZANTINE_FRACTION = 0.2  # under the f < M/3 bound (Assumption / Sec. III-J)
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


async def main() -> None:
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

    rng = np.random.default_rng(0)
    n_byz = max(1, int(round(BYZANTINE_FRACTION * N_VALIDATORS)))
    validators = [Validator(i, authority=True, is_byzantine=(i < n_byz)) for i in range(N_VALIDATORS)]
    consensus = PoA2Consensus(validators, tau_s=0.6)

    for round_id in range(1, N_ROUNDS + 1):
        result = consensus.run_round(block_is_valid=True, rng=rng)
        model_hash = hashlib.sha256(f"round-{round_id}".encode()).digest()

        receipt = await _record_with_retry(
            ledger,
            round=round_id,
            authority_ok=result.authority_ok,
            association_ok=result.association_ok,
            mean_association=result.mean_association,
            model_hash=model_hash,
        )
        print(f"Round {round_id}: finalized={result.finalized} "
              f"(authority_ok={result.authority_ok}, association_ok={result.association_ok}, "
              f"mean_association={result.mean_association:.3f}) -> tx {receipt['transactionHash'].hex()}")

    print("\nReading records back on-chain:")
    for round_id in range(1, N_ROUNDS + 1):
        record = await ledger.get_record(round_id)
        print(f"  round {record[0]}: finalized={record[3]}, mean_association_scaled={record[4]}")


if __name__ == "__main__":
    asyncio.run(main())
