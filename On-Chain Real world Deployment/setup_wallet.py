"""Generates a fresh, local throwaway deployer keypair for PureChain.

No network call is needed for this step (key generation is pure local
crypto via eth_account). The private key is written ONLY to
`.wallet/deployer_key.txt` in this folder, which is gitignored -- it is
never printed, logged, or committed. Only the public address is shown.

Safe to re-run: if a key already exists, it's reused rather than
overwritten, so the deployer address stays stable across sessions.
"""
from __future__ import annotations

from pathlib import Path

from purechainlib import PureChain

WALLET_DIR = Path(__file__).resolve().parent / ".wallet"
KEY_FILE = WALLET_DIR / "deployer_key.txt"
ADDRESS_FILE = WALLET_DIR / "deployer_address.txt"


def main() -> None:
    WALLET_DIR.mkdir(exist_ok=True)

    if KEY_FILE.exists() and ADDRESS_FILE.exists():
        print(f"Deployer wallet already exists: {ADDRESS_FILE.read_text().strip()}")
        print("(delete .wallet/ if you want to generate a new one)")
        return

    pc = PureChain("testnet")
    account = pc.account()

    KEY_FILE.write_text(account["privateKey"])
    ADDRESS_FILE.write_text(account["address"])

    print(f"Generated new throwaway deployer wallet: {account['address']}")
    print(f"Private key saved to {KEY_FILE} (gitignored, never displayed again)")
    print("PureChain transactions are zero-gas, so this account needs no funding.")


if __name__ == "__main__":
    main()
