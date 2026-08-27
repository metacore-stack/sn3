# sn3

Two components so far: **sn3-monitor** (what the subnet expects) and
**fineweb-loader** (reproducible access to the evaluation corpus).

---

## sn3-monitor

Knows what Bittensor SN3 (Teutonic) currently expects of a challenger, notices when
that changes, and refuses to let a stale assumption reach an irreversible submission.

Read-only by construction. It touches no wallet, loads no model, needs no GPU, and
never writes to the chain. Standard library only — no dependencies.

## Why it exists

Everything else in an SN3 pipeline assumes a fixed target: a specific king
checkpoint, a specific dataset version, a specific acceptance threshold. All three
can change without warning, and `teutonic-miner ready` is irreversible — it
permanently consumes that hotkey's single submission.

This is the component whose only job is to notice.

## Install

```bash
cd ~/Documents/sn3
python3 -m pip install -e .        # provides the `sn3-monitor` command
```

Or run it in place with no install at all:

```bash
python3 -m sn3_monitor status
```

Requires Python 3.11+.

## Commands

```bash
sn3-monitor snapshot                    # pin the current contract, print its id
sn3-monitor targets                     # list pinned targets
sn3-monitor status                      # one screen: king, evaluator, economics, rivals
sn3-monitor check --against <id>        # FRESH / STALE / ABORT, exit code carries it
sn3-monitor history --since 24h         # what rivals are achieving
sn3-monitor watch --interval 300        # poll, log, alert on transitions
sn3-monitor preflight --offline-lcb 0.61
```

### Exit codes

Meant to be branched on from scripts.

| Code | Meaning |
|---:|---|
| 0 | fresh — work pinned to this target may continue |
| 1 | stale — re-evaluate before submitting |
| 2 | abort — the competition itself changed |
| 3 | fetch failed |
| 4 | usage error |

```bash
sn3-monitor check --against 20260826T213320Z-c345e657 || {
    echo "baseline moved; not submitting"; exit 1
}
```

## The workflow

Pin a target before each experiment, and reference it by id:

```bash
$ sn3-monitor snapshot
  snapshot   20260826T213320Z-c345e657
  king       teutonic-II-110B-A7B-5ek5koe5-v5   reign 7
  king loss  2.987152      delta 0.5      eval n 2000
```

An experiment is not "trained against the king" — that is not reproducible. It is
"trained against `20260826T213320Z-c345e657`".

Then, before the irreversible step:

```bash
$ sn3-monitor preflight --offline-lcb 0.61
  PASS  dashboard freshness             updated 11s ago (limit 30m)
  PASS  generation unchanged            matches pinned target
  PASS  king digest unchanged           matches pinned target
  PASS  acceptance threshold unchanged  matches pinned target
  PASS  dataset version unchanged       matches pinned target
  PASS  no evaluation in flight         evaluator idle
  PASS  queue clear                     no submissions queued
  PASS  offline LCB margin              measured 0.610000 vs required 0.520000
  PASS  weight publication healthy      last finalized 2026-08-26T21:33:18Z

  CLEAR — every gate passed.
```

It refuses by default: omit `--offline-lcb` and it blocks.

## What it reads

| Document | Purpose |
|---|---|
| `teutonic.ai/dashboard.json` | king, evaluator, queue, history, economics |
| `pub-fedac…r2.dev/dashboard.json` | mirror, used automatically on failure |
| `teutonic.ai/datasets/manifest.json` | dataset version, `delta_threshold`, `eval_n`, tokenizer |

Freshness is enforced on the dashboard only. The dataset manifest's `generated_at`
marks when the evaluation config was cut, not a heartbeat, and legitimately sits
unchanged for days.

Use `--local-dashboard` / `--local-datasets` to work from captured files offline.

## State

Two things on disk, both outside your Teutonic clone:

```
state/
├── targets/<snapshot_id>.json   immutable; never rewritten once created
└── observations.jsonl           append-only poll log
```

The observation log is what makes time-series questions answerable — weight
publication uptime, reign durations, how long evaluations really take. A single
status reading answers none of them. Override the location with `--root` or
`SN3_MONITOR_HOME`.

## Notes from the live data

Things the real payloads do that naive parsing gets wrong:

- **Reign numbers are not contiguous.** `king_chain` currently runs 0, 1, 2, 3, **5**,
  6, 7. Key on digest, never on reign number.
- **Every numeric field in `history` is nullable.** Rows that failed with an
  `error_code` carry `null` for `mu_hat`, `lcb`, `avg_king_loss` and `n_sequences`.
- **`current_eval` is `null` most of the time.** Absence is the normal case.
- **`delta` is published twice** — `king.delta` and `datasets.delta_threshold`. Both
  are recorded so a divergence is visible rather than silently resolved.
- **Weight publication cycles** through `claimed` → `submitted` → `finalized` on a
  101-block cadence. A mid-cycle reading is not a failure; it was observed sitting
  at `failed` for ~25 hours and then recovering.
- **`model_identity` is `hidden_until_promotion`** — challengers cannot be identified
  during evaluation, only after they are crowned.

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

55 tests. Fixtures are real captured payloads, so the null-heavy rows and the
non-contiguous reign numbers exercised are genuine rather than convenient.

## Scope

Deliberately does not: train, load models, use a GPU, read wallets, or write to the
chain. It fetches public JSON and compares strings. Keeping it that way is what
makes it safe to leave running unattended for weeks.


---

## fineweb-loader

Reproducible, verified access to the SN3 FineWeb-Edu shards. numpy is optional —
a pure-stdlib `.npy` reader is used when it is absent.

### The finding that shapes it

**Each evaluation draws all 2,000 of its sequences from exactly one shard.** All
42 scored evaluations in the dashboard history have `shards_used` of length 1,
and every one landed on a 6,144-sequence shard.

So the king's measured loss swinging between 2.985 and 3.077 is not sampling
noise — it is between-shard difficulty. Within a shard the paired difference is
very stable (`mu_hat ≈ lcb` to within 0.4–2%); between shards `mu_hat` itself
moves, and the bootstrap says nothing about that.

Holdouts are therefore stratified across crawls by default: many shards, few
sequences each. A holdout concentrated in one shard gives a confident estimate of
the wrong thing.

### Commands

```bash
fineweb manifest sync --expect-digest <sha>   # download + verify inventory
fineweb manifest stats [--crawls]
fineweb shard fetch <key>                     # download one shard, verify sha256
fineweb shard inspect <name> [--fetch]
fineweb shard verify                          # re-hash everything cached
fineweb holdout build --name val-a --seed 1 --shards 40 --per-shard 128
fineweb holdout list | show <name> | check    # 'check' asserts disjointness
fineweb cache status | prune
```

### The contamination guard

`FineWebLoader` refuses to serve held-out sequences to the training path:

```python
loader.sequences(refs)                        # evaluation: holdouts allowed
loader.sequences(refs, allow_holdout=False)   # raises ContaminationError
loader.training_stream(seed=1, shards=[...])  # holdout refs filtered, then asserted
```

It raises rather than warns on purpose. Training on your own validation data
inflates every measurement afterwards and stays invisible until the compute
budget is gone.

### What the real inventory looks like

```
shards            125,441        crawls      110  (CC-MAIN-2013-20 … 2025-26)
tokens            1,567,352,367,104
sequences         765,308,773
total size        5.7 TiB (6.27 TB)
sequences/shard   min 3, max 6,144
full shards       123,713 (98.6%)
short (<2000)     580 (0.5%)
npy header/shard  128 bytes, uniform
```

### Gotchas encoded here

- **`manifest_sha256` is over canonical JSON**, not the bytes you download:
  `sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))`. Hashing
  the raw file gives `ee192895…` instead of the published `130273b0…` and looks
  like tampering.
- **`size_bytes` includes the 128-byte `.npy` header**, so it is *not*
  `n_tokens * 4`. The verifier allows a small consistent overhead rather than
  asserting equality.
- **Shard sizes are not uniform.** 6,144 sequences is the common case, not the
  rule; 580 shards hold fewer than 2,000 and the smallest holds 3.
- **URL resolution overlaps.** `shard_prefix` is `finewebedu/shards/` while `key`
  is `shards/finewebedu__…`; the bucket path joins the prefix's first segment to
  the key.
- **The stdlib reader returns detached copies**, not live views into the mmap, so
  a sequence can outlive the shard it came from.
- **`sampler.py` is unverified by design.** It holds candidate reconstructions of
  `blake2b-64-block-hash-hotkey-v1` plus a harness to test them against known
  `(block_hash, hotkey) -> shard` outcomes. It must never be used to predict your
  own evaluation sample.

### Tests

```bash
python3 -m unittest discover -s tests -t .
```

110 tests across both packages. Manifest tests run against the real 125,441-entry
inventory when it is synced; shard fixtures are synthesised so the suite stays
fast and offline.

---

## evaluate-losses

Reproduces the SN3 validator's paired-bootstrap decision offline. The statistics
are never reimplemented — `paired_bootstrap_verdict` is loaded out of your cloned
Teutonic repo and called directly.

Needs numpy, so this package uses a venv: `.venv/bin/python -m evaluate_losses`.

### The contract, transcribed from the source

Every value below was read from `teutonic/evaluator/engine.py`, not inferred:

| | value | source |
|---|---|---|
| `alpha` | `0.001` | engine.py:87 |
| `n_bootstrap` | `10000` | engine.py:90 |
| `bootstrap_seed` | `0xB007` | engine.py:150 |
| `batch_size` | `1` — typed `Literal[1]` | engine.py:85 |
| `attn_implementation` | `"eager"` — typed `Literal` | engine.py:86 |
| `seq_len` | `2048`, so **2047 predictions** | engine.py:88 |
| `lm_head_chunk` | `1024` | engine.py:106 |
| engine fallback `delta` | `0.0015` — **not the live bar** | engine.py:89 |

That last row matters: `0.0015` is the Teutonic-I threshold the engine falls back
to. The live bar (`0.5`) arrives per request in `request.limits`. Never plan
against the engine default.

The per-sequence loss, mirrored exactly: `model.model(...)` then a chunked
`lm_head`, labels shifted by one, `reduction="none"` cross-entropy cast to fp32,
concatenated, **summed once**, divided by `2047`. No masking — the document
separator `151645` is scored like any other token.

### The per-shard view is the point

The validator draws all 2,000 sequences from **one** shard. So the overall LCB is
not your odds:

```
$ evaluate compare --king losses/king.json --challenger losses/ckpt-1200.json
  mu_hat  0.536675     lcb  0.530368     bar  0.500000     verdict ACCEPT

per shard (8)
  ! CC-MAIN-2023-40 …  mu_hat=0.4427
  ! CC-MAIN-2022-05 …  mu_hat=0.4686
  ! CC-MAIN-2020-10 …  mu_hat=0.4992
    CC-MAIN-2019-43 …  mu_hat=0.5193
    …
  spread 0.181879     shards clearing  62%
```

A checkpoint that "passes" has a 38% chance of losing the single draw it gets —
and the draw costs an irreversible hotkey.

### Commands

```bash
evaluate contract                            # print the targeted contract
evaluate compare --king K.json --challenger C.json [--by-shard] [--json]
evaluate show <vector.json>
evaluate parity [--stats] [--sampler] [-v]   # prove agreement, no GPU
```

Exit codes: `0` accept, `1` reject, `2` failure (incl. misalignment), `3` usage.

### Parity, in three tiers

```
$ .venv/bin/python -m evaluate_losses parity
tier: statistics   PASS x7      tier: sampler   PASS x11
  offline parity OK.
  Tier 3 (per-sequence loss vs the engine on real weights) still needs a GPU run.
```

Tier 1 checks our comparison against the validator's function and pins the
reign-7 coronation numbers (`3.509797 − 2.987152 = 0.522645`, penalty `0.005467`).
Tier 2 checks the sampler byte for byte. **Tier 3 needs a GPU and has not run** —
until it does, losses from `TorchBackend` are a hypothesis.

### Alignment is fatal, by design

`assert_aligned` refuses vectors covering different sequences or a different
order. A misalignment produces a well-formed number built from unrelated pairs,
and nothing downstream could detect it.

### The sampler is now exact

`fineweb_loader/sampler.py` previously held guesses; it now transcribes the real
chain from `teutonic/evaluation/configuration.py` and `evaluator/sources.py`:

```python
seed   = int.from_bytes(blake2b(f"block_hash={bh}|hotkey={hk}", digest_size=8), "little")
source = int.from_bytes(blake2b(f"{seed}:finewebedu",           digest_size=8), "little")
random.Random(source).shuffle(shards)          # stdlib RNG for shard order
np.random.default_rng(source).choice(n, size=limit, replace=False)   # numpy for sequences
```

Two different RNGs seeded from the same integer — using one where the other
belongs silently changes the sample. And `required_sequences(2000) == 3000`,
which is *why* one 6,144-sequence shard always suffices.

It still cannot predict your own evaluation: the block hash exists only after you
commit.

---

## validate-checkpoint

Runs every rule the SN3 validator will run — **before** `teutonic-miner ready`
spends the hotkey permanently.

**4 of 47** live submissions died on packaging rather than model quality: three
`GenesisContractMismatch`, one `ArtifactIntegrityError`. Roughly one in eleven,
each costing a hotkey that had done nothing wrong except be assembled slightly
incorrectly.

Stdlib only — a pure-Python safetensors header reader means a 220 GB checkpoint
is validated structurally in under a second, with no torch and no GPU.

### Validation happens in four layers

Which layer a rule lives in determines whether failing it is survivable:

| Layer | Where | When | Recoverable |
|---|---|---|---|
| 1 preflight | `miner/upload_model.py` | before upload | yes |
| 2 manifest | manifest build + signing | at upload | yes |
| 3 ingest | `access/storage.py` | **after `ready`** | no |
| 4 evaluator | `evaluator/engine.py` | **after `ready`** | no |

Every check is tagged with its layer, and failures in 3–4 are marked `!`.

### Usage

```bash
validate contract                          # print the enforced contract
validate king --king-digest <sha>          # king inventory, ~3 MB not 220 GB
validate check <dir> --king-digest <sha> --name teutonic-II-110B-… --thorough
```

Exit codes: `0` clean, `1` would be rejected, `2` undetermined (checks skipped),
`3` usage.

### What it enforces

Read from `chain.toml` and the validator's source, not hard-coded:

- **The six contract files**, re-hashed against `[seed.contract_files]` →
  `GenesisContractMismatch`
- **Inventory equality** with the king's declared file set →
  `ArtifactIntegrityError`
- **42 locked config keys** — 12 generic plus 30 from `[arch] extra_lock_keys` —
  and `architectures`
- **250 GB upload cap** (`MAX_MINER_UPLOAD_BYTES`); the king is 220.58 GB
- **No MTP tensors** — case-insensitive substring, as `engine.py` does
- **Only `configuration_mimo_v2.py` and `modeling_mimo_v2.py`** may ship
- **Copy detection** — every shard identical to the king is a rejection; one
  changed shard is enough, so staged unfreezing is safe
- Symlinks, the reserved `manifest.json`, index/shard consistency, truncated
  shards, NaN/Inf, and `repo_pattern`

### The failure that actually kills hotkeys

```
$ validate check ./challenger --king-digest c345e657…
  !FAIL  contract files byte-identical  [ingest]  hashes differ from chain.toml
   PASS  42 locked config keys match    [evaluator]  42 keys plus architectures
  would surface as: GenesisContractMismatch
```

Every config **value** still matches the king. Only the **bytes** changed —
`save_pretrained` reordered the keys. Byte identity and value identity are
different gates, and only one of them is checked at ingest.

**Copy all six contract files from the king after every save.**

### Reference data without the weights

`validate king --king-digest <sha>` fetches the published `manifest.json`,
`config.json` and `model.safetensors.index.json` — about 3 MB — giving all 16
file hashes, 8 shard digests and 34,234 tensor names. Enough for every
structural check without downloading 220 GB.

### Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

195 tests across four packages. `validate_checkpoint` and the two stdlib-only
packages also run under system Python; `evaluate_losses` needs the venv.

---

## mimo-adapter

Makes the locked MiMo `noaux_tc` router trainable — and **proves** the patched
version is numerically identical to the original before anything trains on it.

This was the project's main blocker. It is now solved and verified on CPU.

### The blocker

```python
if self.topk_method == "noaux_tc":
    if self.training:
        raise ValueError("MiMoV2 noaux_tc routing is only implemented for inference.")
```

`modeling_mimo_v2.py` is hash-pinned in `chain.toml`, so the file on disk can
never be edited. The gate's `forward` is replaced **in memory** inside a context
manager and restored on exit, including on exception.

### What the replacement actually changes

Two things, and nothing else:

1. **The guard is gone.**
2. **Selection runs under `no_grad`.** It produces integer indices, which carry
   no gradient in the original either, so detaching that branch is provably
   output-identical and avoids building a graph that would only be discarded.

Everything else is operation-for-operation identical. The key insight is that
`topk_weight = scores.gather(1, topk_idx)` was **already differentiable** —
`gather` backpropagates into `scores`, and thence the router's `weight`. The
guard was the only real obstacle.

`e_score_correction_bias` receives no gradient in either version, by design: it
steers selection only, and selection isn't differentiable. That is the
auxiliary-loss-free scheme — the bias is meant to be moved by a load-balancing
rule you run yourself. **Training without one risks expert collapse, and that,
not a mathematical obstacle, is the likeliest reason the guard exists.**

### Verified, on a 1.1M-parameter miniature

The routing question doesn't depend on model size — it depends on `topk_method`,
`scoring_func`, `norm_topk_prob`, `n_group` and `topk_group`, all copied from the
real king's `config.json`. Sizes are shrunk; routing is real.

```
$ mimo-adapter verify
  PASS  patch restores the original forward               restored even when the block raises
  PASS  shipped gate refuses training mode                MiMoV2 noaux_tc routing is only implemented for inference.
  PASS  patched routing is numerically identical in eval  max |Δlogit| = 0.000e+00 over 3 inputs
  PASS  patched routing is deterministic                  max |Δ| = 0.000e+00
  PASS  forward produces a finite loss in training mode   loss = 5.558463
  PASS  all gradients are finite                          no NaN or Inf
  PASS  gradient reaches router                           3/3 parameters, norm 3.13e-02
  PASS  gradient reaches routed_experts                   69/72 parameters, norm 3.21e+00
  PASS  gradient reaches shared_expert                    9/9 parameters
  PASS  gradient reaches attention                        8/11 parameters
  PASS  router bias receives no gradient (expected)
  PASS  routing statistics collected                      8/8 experts touched, imbalance 1.583
  PASS  optimizer step succeeds                           103/109 tensors changed; all finite
  PASS  trained checkpoint round-trips                    max |Δlogit| = 0.000e+00 under unpatched code
```

That last line is the one that matters for submission: a checkpoint trained
under the patch loads and scores under the **original**, hash-pinned code.

### Two traps found while building this

**1. The architecture requires `transformers==5.8.1`.** The king's `config.json`
declares it, and 5.16.1 changed `create_causal_mask`'s signature — the model
does not build at all. Pinned in `[project.optional-dependencies] adapter`.

**2. Routing parameters are allocated but never initialised.**

```python
self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))
```

`_init_weights` doesn't cover them. For the real king this never bites — the
values arrive from the checkpoint. But any model built from a config alone gets
whatever was in that memory: observed here as gate weights of **2e+38**, which
saturate the sigmoid to exactly 1.0 and yield exactly-zero gradients that look
like a broken router rather than a broken fixture. `build_miniature` initialises
them explicitly; pass `init_gates=False` to reproduce the hazard.

### Usage

```bash
mimo-adapter fetch --king-digest <sha>   # 3 files, ~42 KB, not the weights
mimo-adapter info                        # describe the miniature
mimo-adapter verify [--fast] [--json]    # every parity and gradient check
```

```python
from mimo_adapter import load_arch, trainable_routing

arch = load_arch()
with trainable_routing(arch):
    loss = model(input_ids=batch, labels=batch).loss
    loss.backward()          # the file on disk is untouched throughout
```

### What is still unproven

This is verified on a miniature with random weights. It has **not** been run
against the real 110B checkpoint. Before a full training run, repeat
`mimo-adapter verify` semantics on real weights — the eval-parity check
especially — as part of the first GPU session.

---

## train-mimo

Continued pretraining of a MiMo checkpoint. Runs inside
`mimo_adapter.trainable_routing`, draws batches only from `fineweb_loader`'s
contamination-guarded stream, and writes checkpoints whose six locked files are
restored byte-identical from the king.

### The end-to-end loop runs on a laptop

Real FineWeb shard → real vocabulary → train → checkpoint → validate:

```
$ train-mimo train --real-vocab --seq-len 2048 --max-steps 6 --batch-size 1 \
      --shard finewebedu__CC-MAIN-2013-20__part0__shard_000000.npy \
      --holdout val-a --holdout val-b

  run 'real-data'  stage=all  source=finewebedu
  trainable 40,071,332 / 40,071,332 parameters (100.0%)
  holdouts val-a, val-b (8,192 sequences excluded)

  step  0  loss 11.9568  lr 2.00e-05  |g| 1.307  experts 8/8  imbalance 1.5736
  step  5  loss 11.9179  lr 7.00e-05  |g| 1.502  experts 8/8  imbalance 1.6406
```

A fresh model starts at **11.96 ≈ ln(152576) = 11.935** — exactly uniform-random
over the real vocabulary, which is the sanity check that the data path is wired
correctly.

`--real-vocab` keeps the king's 152,576-token vocabulary (about 40M parameters
total) so real shards can be fed through. With a toy vocabulary the token ids are
out of range and nothing downstream is being tested honestly.

### The routing bias needs a rule

`e_score_correction_bias` receives no gradient — it steers selection, and
selection isn't differentiable. Something has to move it, or routing collapses
onto whichever experts start out slightly favoured. `LoadBalancer` applies the
standard auxiliary-loss-free update: compare each expert's share of routed tokens
against uniform, step its bias against the difference. Only the *sign* is used,
so the step is bounded and the rule cannot itself destabilise training.

Measured over 80 synthetic steps, 8 experts, top-2:

| | worst imbalance |
|---|---:|
| with the rule | 3.15 |
| `--no-balance` | 4.00 |

An imbalance of 4.0 across 8 experts means one expert absorbed **50%** of all
routing against a uniform 12.5%. The mechanism works and points the right way —
though 80 steps of random tokens is a weak test, and the real check is on real
data at scale.

### Checkpoints that are actually submittable

Two rules, both learned from the live failure history:

1. **The six locked files are restored after every save.** `save_pretrained`
   rewrites `config.json` — reordered keys, a bumped `transformers_version` —
   and the bytes then differ even though every value matches. That's
   `GenesisContractMismatch`, three of the four packaging deaths on record.
2. **Training state never enters the model directory.** Optimizer, scheduler,
   RNG and the routing bias go to a sibling `state-NNNNNN/`. A stray file in the
   uploaded tree is an *undeclared* object — `ArtifactIntegrityError`.

```
$ train-mimo inspect runs/pilot --chain ~/Documents/teutonic/chain.toml
  files (8): chat_template.jinja.txt, config.json, configuration_mimo_v2.py,
             generation_config.json, model.safetensors, modeling_mimo_v2.py,
             tokenizer.json, tokenizer_config.json
  locked files: all match chain.toml
```

There is also a shape guard: the king's `config.json` is only copied onto a model
of the king's shape. Restoring it onto a miniature would produce a directory
whose config describes a 110B model beside miniature weights — which cannot even
be reloaded.

### Commands

```bash
train-mimo config --output configs/pilot.json
train-mimo train --config configs/pilot.json [--resume] [--validate]
train-mimo train --synthetic --max-steps 20        # loop only, no shards
train-mimo inspect <run-dir> [--chain chain.toml]
train-mimo prune <run-dir> --keep 3                # a real checkpoint is ~220 GB
```

Stages for selective freezing: `shared` (6.8% of parameters on the miniature),
`shared+router`, `experts` (61.9%), `experts+attention`, `all`.

### Three bugs found while building it

- **`--resume` restored the optimizer but not the weights.** Run one ended at
  5.5620; the resume restarted at 5.5974 while claiming to continue. It now
  loads the checkpoint's weights before restoring state.
- **`from_pretrained` hung.** The restored `config.json` carries `auto_map`,
  which sends transformers to the Hugging Face Hub. All loads now pass
  `local_files_only=True` — a rented GPU node should never make that call
  mid-run.
- **Restoring locked files onto a miniature produced an incoherent checkpoint.**
  Hence the shape guard above.
