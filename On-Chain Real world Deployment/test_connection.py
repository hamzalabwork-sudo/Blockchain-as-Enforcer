"""Read-only connectivity check against the real PureChain testnet.

Requires no private key -- only queries public chain state (connection
status, chain ID, latest block, and the owner address's balance).
Run with: python "test_connection.py" from this folder, or
python -m "On-Chain Real world Deployment.test_connection" from the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHAIN_ID, OWNER_ADDRESS, RPC_URL
from web3 import HTTPProvider, Web3


def main() -> None:
    w3 = Web3(HTTPProvider(RPC_URL, request_kwargs={"timeout": 15}))

    print(f"RPC: {RPC_URL}")
    connected = w3.is_connected()
    print(f"Connected: {connected}")
    if not connected:
        print("Could not connect -- see README.md in this folder for the TLS troubleshooting steps.")
        return

    print(f"Chain ID: {w3.eth.chain_id} (expected {CHAIN_ID})")
    print(f"Latest block: {w3.eth.block_number}")
    print(f"Gas price: {w3.eth.gas_price}")

    balance_wei = w3.eth.get_balance(Web3.to_checksum_address(OWNER_ADDRESS))
    print(f"Owner address {OWNER_ADDRESS} balance: {w3.from_wei(balance_wei, 'ether')} PCC")


if __name__ == "__main__":
    main()
