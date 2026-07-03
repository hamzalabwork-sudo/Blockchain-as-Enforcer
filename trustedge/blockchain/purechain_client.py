"""Optional real on-chain integration via purechainlib (pip install purechainlib).

STATUS: written against the real purechainlib 2.1.7 API (verified by reading
its installed source) but NOT exercised end-to-end in this reproduction --
connecting to the live testnet (`https://purechainnode.com:8547`) from this
environment fails TLS certificate verification:

    SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    Basic Constraints of CA cert not marked critical

That's a non-standard CA certificate on the node's side, not a bug here. The
project owner chose to run PoA2 entirely as a local simulation
(trustedge.blockchain.poa2_simulator, trustedge.poa2) rather than disable TLS
verification to work around it. This module is kept for anyone who *can*
reach the node normally (e.g. from inside the lab network, or once the
node's certificate is fixed) and wants genuine on-chain verification instead
of the simulator.

Usage (untested in this environment):
    from trustedge.blockchain.purechain_client import PureChainTrustLedger
    ledger = PureChainTrustLedger()
    await ledger.connect_fresh_account()   # generates a new throwaway keypair locally
    await ledger.deploy()
    await ledger.record_consensus(round=1, authority_ok=True, association_ok=True,
                                   mean_association=0.87, model_hash=b"...")
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

CONTRACT_PATH = Path(__file__).parent / "contracts" / "TrustLedger.sol"


class PureChainTrustLedger:
    def __init__(self, network: str = "testnet"):
        from purechainlib import PureChain  # deferred import: optional dependency

        self.pc = PureChain(network)
        self.contract = None

    async def connect_fresh_account(self) -> dict:
        """Generates a new local throwaway keypair (no funds needed -- PureChain
        testnet transactions are zero-gas) and connects it as the signer."""
        account = self.pc.account()
        self.pc.connect(account["privateKey"])
        return {"address": account["address"]}  # never return/log the private key

    async def connect_with_key(self, private_key: str) -> None:
        """Connects with a caller-supplied key. Read it from an environment
        variable or local untracked file -- never hardcode or log it."""
        self.pc.connect(private_key)

    async def deploy(self) -> str:
        source = CONTRACT_PATH.read_text()
        factory = await self.pc.contract(source)
        self.contract = await factory.deploy()
        return self.contract.address

    async def record_consensus(
        self,
        round: int,
        authority_ok: bool,
        association_ok: bool,
        mean_association: float,
        model_hash: Optional[bytes] = None,
    ) -> dict:
        if self.contract is None:
            raise RuntimeError("Call deploy() (or attach to an existing address) first")
        model_hash = model_hash or hashlib.sha256(str(round).encode()).digest()
        scaled = int(round_half_up(mean_association * 1e4))
        return await self.pc.execute(
            self.contract, "recordConsensus", round, authority_ok, association_ok, scaled, model_hash,
        )

    async def get_record(self, round: int) -> dict:
        if self.contract is None:
            raise RuntimeError("Call deploy() (or attach to an existing address) first")
        return await self.pc.call(self.contract, "getRecord", round)


def round_half_up(x: float) -> int:
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)
