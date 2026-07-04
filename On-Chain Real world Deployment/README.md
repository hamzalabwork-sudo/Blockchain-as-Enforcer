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

## Status: working -- verified live on the real testnet, matches the local simulation's method coverage

| | |
|---|---|
| Deployer (throwaway key, zero-gas, unfunded) | `0x16f57d43dA79DA9005DB73B1A91b23fdD225D23a` |
| Consensus-only contract (`record_consensus_demo.py`) | `0x5E415aa5F5942f8341BBA1518EA9548F80C34891` -- 100/100 real tx sent, 79 finalized / 21 correctly rejected, 0 mismatches |
| FL-integrated contract, PoA2 only (`run_fl_with_onchain_poa2.py`) | `0xa169Cef457C355c6301bFd6c1a69Ed2C26e10B4a` -- 10/10 real FL training rounds anchored, trust mechanism genuinely excluded 2 injected malicious clients by round 4, 0 mismatches |
| **Full 8-method comparison contract (`run_full_ablation_onchain.py`)** | `0x4960bdE9A141ba6c55734Cb6035C49Db7Bb3900E` -- 71/71 real tx across all Table IX + XVI methods, 0 mismatches (see below) |

Early on, only PoA2 had a real on-chain proof; vanilla FedAvg, Krum,
Trimmed-Mean, Median, Authority-only, Association-only and Centralized
existed only in the local simulation (`results/`). `run_full_ablation_onchain.py`
closes that gap: it runs real BiLSTM training for **all eight** methods and
anchors every one of their real per-round outcomes on-chain, so the
robustness comparison isn't just simulated anymore.

### Full 8-method comparison result (`run_full_ablation_onchain.py 10`)

10 clients, 2 malicious (random-gradient-replacement attack), BiLSTM,
WUSTL-IIoT-2021 (synthetic) -- 71 real transactions, 19.7s training + 158.2s
anchoring, **0 verification mismatches** across every method:

| Method | Final Accuracy | Rounds anchored |
|---|---|---|
| Centralized Learning | 95.18% | 1 |
| Vanilla FL | **85.71%** (clearly weakest -- no defense) | 10 |
| FL + PoA (only authority) | 98.04% | 10 |
| FL + Association (only) | 98.57% | 10 |
| FL + PoA2 (full) | 98.39% | 10 |
| Krum | 97.32% | 10 |
| Trimmed Mean | 98.57% | 10 |
| Median | 97.86% | 10 |

This matches the qualitative pattern from the local simulation
(`results/robust_baselines_table.csv`, `results/ablation_table.csv`)
closely: vanilla FedAvg is visibly the most vulnerable to the poisoning
attack, and every defense mechanism -- trust-based or classical robust
statistics -- recovers to within a couple of points of clean accuracy. The
difference here is that every single one of these 71 data points is a real
transaction on a real chain, individually verified against what was
actually computed, not a number from a local run.

The 100-transaction run (`python record_consensus_demo.py 100`) sent one
real transaction per round with a randomized ~85%/15% valid/invalid block
mix (see "Scripts" below), completing in 251.3s including 18 transient
nonce-race retries (see below), and every single round's on-chain state was
read back afterward and confirmed to exactly match what was sent -- 0
mismatches across all 100.

Sample transaction hashes (round -> tx):
```
1 -> 22c9936f5b667dc699265690ee2d6db1f197197a5d7a44248c4242a6a4a0d400
2 -> 469f301a0a25e496f803acd2f203327dc435682f3131c546c2c4d7845e11419c
3 -> 3fb2b687f220c3ef74af6daa71fe2651486e294f7bd05b9c2b2e6d7be5cfb89e
4 -> 2c974099129d2e34bb796536ac2258ee8eff9aeb1f67b0da7948f76e36f09788
5 -> c01839164a51adc964027ad942c2147be52a0ef90ee78465bd516af71c76efa9
...
100 -> 4436f5cd34d0d0d8...
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
collide on nonces. The 8-method comparison run later surfaced a second
symptom of the exact same race: `replacement transaction underpriced`
(the node still sees the previous tx as pending and treats the new one as
a same-nonce replacement -- since PureChain is always zero-gas, there's no
fee bump that could ever satisfy a real replacement, so this is the same
underlying timing issue wearing a different error message).

Both are handled by a shared retry-with-backoff helper,
`record_with_retry()` in `chain_utils.py` (used by all three on-chain
scripts) -- not anything more invasive; the underlying PoA2 logic and
contracts were correct throughout, this was purely an RPC-timing issue.
The retry logic holds up at scale: across the 100-transaction run and the
71-transaction 8-method comparison, it hit this race dozens of times and
recovered cleanly every time (never exceeding 1-2 retries).

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

All scripts that send transactions print which key source they're using
(owner vs. throwaway) and the resulting public address -- never the key
itself -- and warn if an `owner_key.txt` doesn't actually correspond to
the configured `OWNER_ADDRESS`.

## Scripts

1. `python setup_wallet.py` -- generates the throwaway deployer key (skip if using your own key)
2. `python test_connection.py` -- read-only connectivity check, no key needed
3. `python deploy_trust_ledger.py` -- deploys TrustLedger.sol, saves the address
4. `python record_consensus_demo.py [n_rounds]` -- runs `n_rounds` real PoA2 rounds
   using the *abstract* Validator/PoA2Consensus classes (no real ML training
   behind them -- block validity is randomized), anchors each on-chain, then
   reads every single one back and verifies it matches exactly what was sent
5. `python run_fl_with_onchain_poa2.py [n_rounds]` -- the FL-integrated
   version: runs real BiLSTM federated-learning rounds (the same
   `trustedge.simulation` engine as the local experiments in `results/`) on
   WUSTL-IIoT-2021 with 20% malicious clients under the "poa2" method only,
   then anchors each round's REAL outcome (aggregated-model hash, whether
   enough clients were admitted, admitted clients' mean trust score) on a
   freshly-deployed TrustLedger instance.
6. `python run_full_ablation_onchain.py [n_rounds]` -- the comprehensive
   version: runs **all eight** aggregation methods (Centralized, Vanilla
   FL, Authority-only, Association-only, PoA2, Krum, Trimmed Mean, Median)
   for real, matching the local Table IX/XVI comparison, and anchors every
   method's real per-round outcomes on one shared, freshly-deployed
   contract (round IDs namespaced per method: `method_index*1000 + round`).
   This is the one that actually closes the gap with the local simulation
   -- scripts 4-5 only ever demonstrated a single method.
- `chain_utils.py` -- the shared `record_with_retry()` helper (see "A
  transient nonce race condition" above) used by scripts 4-6.

### Real FL + on-chain PoA2 result (`run_fl_with_onchain_poa2.py 10`)

10 clients, 2 malicious (random-gradient-replacement attack), BiLSTM,
WUSTL-IIoT-2021 (synthetic):

| Round | Admitted | Mean trust (admitted) | Finalized |
|---|---|---|---|
| 1-3 | 10/10 | declining: 0.940 -> 0.898 -> 0.869 | True |
| 4-10 | 8/10 | 1.000 | True |

The trust-exclusion mechanism (Eq. 16's penalty decay + the Automatic
Exclusion Rule) genuinely caught and excluded both malicious clients by
round 4 -- this isn't scripted, it's the real `trustedge.trust`/`poa2`
logic reacting to the real attack. Final accuracy: **0.9821**. All 10
rounds anchored on contract `0xa169Cef457C355c6301bFd6c1a69Ed2C26e10B4a`
(3.9s training + 26.0s anchoring, including 2 nonce-race retries) and
verified with 0 mismatches against the actual training outcome.
