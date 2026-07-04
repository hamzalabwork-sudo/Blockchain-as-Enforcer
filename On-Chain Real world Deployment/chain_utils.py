"""Shared retry helper for real PureChain transactions.

PureChain's on-demand block production can leave eth_getTransactionCount
("latest") briefly behind a just-confirmed transaction, so back-to-back
transactions can collide on nonces. This surfaces as either of two RPC
error messages depending on timing:
  - "nonce too low"                    -- the nonce was already used
  - "replacement transaction underpriced" -- the node still sees the
    previous tx as pending and treats the new one as a same-nonce
    replacement; since PureChain is always gasPrice=0, there's no fee bump
    that could satisfy a real replacement, so this is really the same
    underlying race, not a genuine fee issue.
Both resolve themselves by waiting briefly and retrying with a freshly
fetched nonce (which purechainlib's execute() does automatically each call).
"""
from __future__ import annotations

import time

MAX_RETRIES = 5
RETRY_DELAY_S = 3.0
TRANSIENT_ERRORS = ("nonce too low", "replacement transaction underpriced")


async def record_with_retry(ledger, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await ledger.record_consensus(**kwargs)
        except Exception as e:
            if any(msg in str(e) for msg in TRANSIENT_ERRORS) and attempt < MAX_RETRIES:
                print(f"    (transient RPC error, retrying in {RETRY_DELAY_S}s -- attempt {attempt}/{MAX_RETRIES}: {e})")
                time.sleep(RETRY_DELAY_S)
                continue
            raise
