"""Resolves which signing key to use, without ever printing or logging it.

Priority: the project owner's own key (.wallet/owner_key.txt, created
manually outside any chat/logging surface) if present, otherwise the
locally-generated throwaway deployer key (.wallet/deployer_key.txt, see
setup_wallet.py).
"""
from __future__ import annotations

from pathlib import Path

WALLET_DIR = Path(__file__).resolve().parent / ".wallet"
OWNER_KEY_FILE = WALLET_DIR / "owner_key.txt"
DEPLOYER_KEY_FILE = WALLET_DIR / "deployer_key.txt"


def resolve_private_key() -> tuple[str, str]:
    """Returns (private_key, source_label). Raises if neither file exists."""
    if OWNER_KEY_FILE.exists():
        key = OWNER_KEY_FILE.read_text().strip()
        if key:
            return key, "owner"
    if DEPLOYER_KEY_FILE.exists():
        key = DEPLOYER_KEY_FILE.read_text().strip()
        if key:
            return key, "throwaway deployer"
    raise FileNotFoundError(
        "No signing key found. Either run setup_wallet.py (throwaway key) or "
        f"create {OWNER_KEY_FILE} yourself with your own key."
    )
