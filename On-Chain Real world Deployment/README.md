# On-Chain Real-World Deployment

Real (not simulated) integration with the live PureChain testnet, using the
`purechainlib` SDK.

## Network

| | |
|---|---|
| RPC Endpoint | `https://purechainnode.com:8547` |
| Chain ID | `900520900520` |
| Currency | PCC (zero gas fees on all transactions) |
| Owner address | `0x6aDb63B9dBaa1e6df2bA2D16bA6273aE916D429A` (public only -- no private key is stored in this repo) |

## Status: working -- verified live on the real testnet

| | |
|---|---|
| Deployed TrustLedger contract | `0x5E415aa5F5942f8341BBA1518EA9548F80C34891` |
| Deployer (throwaway key, zero-gas, unfunded) | `0x16f57d43dA79DA9005DB73B1A91b23fdD225D23a` |
| Rounds recorded on-chain | 5/5, all finalized, all read back successfully |

Sample transaction hashes (round -> tx):
```
1 -> 22c9936f5b667dc699265690ee2d6db1f197197a5d7a44248c4242a6a4a0d400
2 -> 469f301a0a25e496f803acd2f203327dc435682f3131c546c2c4d7845e11419c
3 -> 3fb2b687f220c3ef74af6daa71fe2651486e294f7bd05b9c2b2e6d7be5cfb89e
4 -> 2c974099129d2e34bb796536ac2258ee8eff9aeb1f67b0da7948f76e36f09788
5 -> c01839164a51adc964027ad942c2147be52a0ef90ee78465bd516af71c76efa9
```

### The TLS interception issue that had to be resolved first

Connecting from this machine initially failed:

```
SSL: CERTIFICATE_VERIFY_FAILED: Basic Constraints of CA cert not marked critical
```

Diagnosis (via `openssl s_client -connect purechainnode.com:8547`): the
certificate actually reaching this machine was issued by **"Avast Web/Mail
Shield Root"**, not by PureChain. Avast's antivirus was intercepting and
re-signing HTTPS traffic to scan it (a normal "Web Shield" feature), and its
generated certificate had a malformed `Basic Constraints` extension that
modern strict TLS validation rejects outright -- adding that root to a
trusted CA bundle didn't help, because the certificate itself is
structurally non-conformant, not just untrusted.

**Resolution:** the project owner disabled Avast's shields (tray icon ->
"Avast shields control" -> "Disable for 10 minutes") to let the real
PureChain certificate through, rather than weakening TLS verification in
code. No certificate-validation code was changed -- `purechain_client.py`
still verifies certificates normally; it's just now seeing PureChain's
genuine certificate instead of Avast's malformed interception one.

Run `python test_connection.py` (no private key required, read-only) to
check current connectivity status.

### A transient nonce race condition

The first `record_consensus_demo.py` run hit `nonce too low: next nonce 3,
tx nonce 2` on round 3. PureChain's block production is on-demand (not
fixed-interval), and `eth_getTransactionCount(address, "latest")` can
briefly lag behind a just-confirmed transaction on this kind of node,
causing back-to-back transactions submitted within the same second to
collide on nonces. Fixed with a short retry-with-backoff around each
`record_consensus` call (`_record_with_retry` in `record_consensus_demo.py`)
rather than anything more invasive -- the underlying PoA2 logic and contract
were correct throughout; this was purely an RPC-timing issue.

## Signing key handling

**No private key is ever pasted into chat, printed, or committed.**
`wallet.py` resolves which key to sign with, in priority order:

1. **`.wallet/owner_key.txt`** -- if you want deployments/transactions
   attributed to `0x6aDb...29A` specifically, create this file *yourself*,
   outside of any chat session (open a text editor, paste your key, save it
   to `On-Chain Real world Deployment/.wallet/owner_key.txt`). This whole
   `.wallet/` directory is gitignored (see the repo's `.gitignore`:
   `**/.wallet/`), so it can never be committed by accident.
2. **`.wallet/deployer_key.txt`** -- a fresh, locally-generated throwaway
   keypair (created by `setup_wallet.py`, no chat involvement, no funding
   needed since PureChain is zero-gas). Used automatically if no owner key
   file is present.

`deploy_trust_ledger.py` and `record_consensus_demo.py` both print which
key source they're using (owner vs. throwaway) and the resulting public
address -- never the key itself -- and warn if an `owner_key.txt` doesn't
actually correspond to the configured `OWNER_ADDRESS`.

## Scripts

1. `python setup_wallet.py` -- generates the throwaway deployer key (skip if using your own key)
2. `python test_connection.py` -- read-only connectivity check, no key needed
3. `python deploy_trust_ledger.py` -- deploys TrustLedger.sol, saves the address
4. `python record_consensus_demo.py` -- runs 5 real PoA2 rounds, anchors each on-chain, reads them back
