"""Deploys TrustLedger.sol to the real PureChain testnet, signed by whichever
key wallet.py resolves (the owner's own key if present, else the throwaway
deployer key). Zero gas fees -- no funding needed either way. Saves the
deployed contract address to deployed_address.txt so other scripts
(record_consensus_demo.py) can attach to it later.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OWNER_ADDRESS, RPC_URL
from wallet import resolve_private_key

HERE = Path(__file__).resolve().parent
DEPLOYED_ADDRESS_FILE = HERE / "deployed_address.txt"


async def main() -> None:
    try:
        private_key, source = resolve_private_key()
    except FileNotFoundError as e:
        print(e)
        return

    from trustedge.blockchain.purechain_client import PureChainTrustLedger

    ledger = PureChainTrustLedger(network="testnet")
    await ledger.connect_with_key(private_key)
    signer_address = ledger.pc.signer.address
    print(f"Signing with {source} key -> address {signer_address}")
    if source == "owner" and signer_address.lower() != OWNER_ADDRESS.lower():
        print(f"WARNING: this key's address ({signer_address}) does not match "
              f"the configured owner address ({OWNER_ADDRESS}) -- double-check owner_key.txt.")

    print(f"Connecting to {RPC_URL} ...")
    if not ledger.pc.web3.is_connected():
        print("Not connected -- see README.md in this folder for TLS troubleshooting.")
        return

    print("Connected. Deploying TrustLedger.sol ...")
    address = await ledger.deploy()
    DEPLOYED_ADDRESS_FILE.write_text(address)

    print(f"TrustLedger deployed at: {address}")
    print(f"Saved to {DEPLOYED_ADDRESS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
