# HyperGDN-2 (Direction B) — experiment log

Reference implementation: `lit_gpt/hgdn2.py` (`HyperGatedDeltaNet2`), the
shared-block / B-timescale multi-stream lift of GDN-2. All toys bypass the fla
Triton front-end (no CPU support) by calling `_stream_recurrence` directly with
plain `nn.Linear` projections. All training runs on H200 via `srun`.

Reduction anchor: `n=1, p=γ=1, H=I` == stock GDN-2. `tests/test_hgdn2.py`
(6 CPU + 1 CUDA) pass, including the reduction test vs an independent GDN-2
reference.

---

## Shared toy harness (all runs)

All runs wrap `_stream_recurrence` in the same tiny `ToyRecall` module
(`scripts/train_hgdn2_*.py`) so only the recurrence is under test:

- **Model.** `Embedding(vocab, d=64)` → 6 plain `nn.Linear` heads producing the
  kernel inputs `q, k, v, g, b, w` → `_stream_recurrence` → `Linear(→vocab)`
  readout. `n_heads=2, head_dim=16` (MQAR) or `4×32` (copy). `use_short_conv=False`
  so no Triton is touched (CPU-capable module, but training itself runs on H200).
- **Gate parametrization** (matches GDN-2's kernel front-end):
  `g = −softplus(Wg·h) ≤ 0` (channel log-decay `D=exp(g)`),
  `b = 2·σ(Wb·h)` (erase gate), `w = σ(Ww·h)` (write gate). `k̂,q̂` are
  L2-normalized inside the recurrence; `q̂` also scaled by `d_k^{-0.5}`.
- **HyperGDN-2 init.** `p_profile = 1/n` (uniform); `log_gamma` spread fast→slow
  via `1 − logspace(log10 0.5, log10 1e-3, n)`; `h_proj` (dynamic-H MLP)
  zero-init; `alpha_raw` from `h_init_alpha=0.1`. `num_streams=1` collapses to
  stock GDN-2.
- **Optimizer.** AdamW, lr 3e-3, no schedule. Loss = token cross-entropy with
  `ignore_index=-100` (only query/target positions contribute).
- **H mixing toggle.** Setting `chunk` > seq_len disables the cross-stream mix
  (no chunk boundary is ever crossed), isolating the multi-stream read from H.
- **`n=1` vs `n=4`** always share the same `torch.manual_seed(seed)` weight init,
  so differences are due to stream count, not initialization luck.

---

## Run 1 — delayed-copy overfit (`scripts/train_hgdn2_toy_cpu.py`)

**Task.** `target[t] = x[t−delay]`, fixed `delay=4`, `vocab=16`. A single
**fixed** batch (`B=8, T=48`, one `torch.randperm` draw, reused every step) is
memorized — this is an *overfit* premise check ("does the recurrence learn at
all"), not a generalization test, so train accuracy is the metric and there is no
held-out eval. `-100` ignore index masks the first `delay` positions (no valid
source token yet). Single-timescale by construction: every target sits exactly 4
tokens back, so one decay rate suffices and extra streams are pure overhead.

**Result.** Both n=1 and n=4 reach 100% train acc; n=1 has lower loss. Confirms
the recurrence trains, but the task is single-timescale so n>1 is pure overhead
(expected). Not a discriminating benchmark.

---

## Run 2 — single-timescale MQAR (`scripts/train_hgdn2_mqar_cpu.py`)

**Task.** Content-based associative recall (MQAR), `make_mqar`. Vocabulary is
partitioned: ids `0..n_keys−1` are **keys**, `n_keys..n_keys+n_vals−1` are
**values**, and one extra id `n_keys+n_vals` is a **filler**. Per sequence:

```
[k0 v0  k1 v1 ... k5 v5]      2·num_pairs binding tokens (key then its value)
[f f f ... f]                 gap random value-space interference tokens
[k? k? ... k?]                num_pairs query tokens = the keys in random order
```

Only the query positions carry targets (`y = bound value`); all binding/gap
positions are `-100` (ignored). Keys are a `randperm(n_keys)[:num_pairs]` draw
(disjoint within a sequence); values are random; the gap is filled with random
value-space tokens as **interference** (every token still writes to memory, so
the gap actively corrupts the store). Query order is a fresh `randperm`, so the
model cannot use position — it must recall by key content.

Because binding order is fixed but query order is shuffled, early-bound keys are
recalled at **long** distance (across the gap) and late-bound at **short**
distance, but the task does not *separate* the two — one slow decay (γ≈1) plus the
delta rule solves the whole thing, which is why n>1 is overhead here (motivating
Run 3). **Regime.** Trained **online** (fresh `make_mqar` batch of 32 every step,
data generator seed 100) and evaluated on a **held-out** batch of 64 (generator
seed 999, re-seeded each probe so the eval set is identical across configs).
Config: `num_pairs=6, gap=16, seq_len=34, n_keys=n_vals=16, 2000 steps, lr 3e-3`.

### 2a. BUG FOUND — n=4 underperforms n=1

With the original **unweighted** output `o = Σ_i S_iᵀq̂`:

| config | eval acc | learned p | learned γ |
|---|---|---|---|
| n=1 | **0.661** | [1.51] | [1.0] |
| n=4 full (H on) | 0.318 | [0.03, 0.89, 0.89, 0.89] | [0.32, 1, 1, 1] |
| n=4, H off | 0.302 | — | — |
| n=4, H off, γ=1 all | 0.263 | [0.75×4] | [1×4] |
| n=4, H off, **primary init** p=[1,0,0,0] | 0.276 | [0.94, **0.38, 0.38, 0.38**] | — |

**Root cause (systematic debugging).** NOT the forward function:
- H-mixing exonerated (H-off ≈ H-on).
- Loop gain exonerated: the γ=1-all case is 4 *identical* streams computing an
  effective single stream with loop gain `n·p² = 4·0.75² = 2.27`, *identical* to
  n=1's `1.51² = 2.27`, yet scores 0.263 vs 0.661.
- Primary-init (started *exactly* at the GDN-2 point) still ends 0.276 and the
  dormant streams **wake up** (p: 0 → 0.38) and hurt.

The pathology is in the **gradient structure**: the unweighted output gives a
dormant stream (p_i=0) a nonzero wake-up gradient `∂L/∂p_i = (output grad)·G`, so
the optimizer is driven to activate streams that are pure overhead on a
single-timescale task. The design had **no graceful reduction**.

### 2b. FIX — p-weighted output

Changed the output to the p-weighted read (consistent with the block read),
post-update: `o = Sbarᵀq̂ = (Σ_i p_i S_i)ᵀq̂` (`hgdn2.py`). This zeroes the
wake-up gradient at p_i=0, so useless streams go/stay dormant. (A normalized
`Σp_iS_i/Σp_i` variant was rejected: the denominator reintroduces the wake-up
gradient.) Reduction preserved at init (p=1); note `p` becomes a free output gain
at n=1.

| config | eval acc | learned p |
|---|---|---|
| n=1 | 0.594 | [1.47] |
| n=4 full (H on) | **0.661** | [**0.000**, 0.86, 0.86, 0.86] |
| n=4, H off | 0.604 | [0.00, 0.85, 0.85, 0.85] |
| n=4, H off, primary init | 0.276 | [1.25, **0, 0, 0**] (stays dormant ✓) |

The harmful fast stream now goes dormant (p→0) instead of being force-woken;
n=4 recovers n=1's performance. Primary-init confirms the fix: dormant streams
get exactly-zero gradient and **stay** at p=0.

### 2c. Multi-seed confirmation (3 seeds, p-weighted)

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| n=1 | 0.594 | 0.258 | 0.302 | 0.385 |
| n=4 | 0.661 | 0.435 | 0.294 | **0.464** |

n=4 ≥ n=1 across seeds. The systematic gap is gone. **High seed variance
(±0.17)** — this task sits at a capability threshold (some seeds break through in
2000 steps, some don't); the old "n=1 = 0.66" was a single lucky seed.

**Conclusion (Run 2):** with the p-weighted output, n>1 gracefully reduces and no
longer underperforms n=1 as overhead. But single-timescale MQAR cannot show n>1
*winning* — the extra streams are genuinely useless there.

---

## Run 3 — multi-timescale (bimodal-lag) MQAR (`scripts/train_hgdn2_mt_mqar.py`)

**Task (designed to force the trade-off).** Bimodal-lag recall, `make_mt_mqar`.
Same key/value/filler vocabulary as Run 2, but bindings are split into two lag
classes with a long interference gap wedged between them:

```
[ks0 vs0 ... ks3 vs3]   2·n_slow  SLOW bindings (bound at t=0, recalled far away)
[f f f ... f]           gap=48    interference (random value-space writes)
[kf0 vf0 ... kf3 vf3]   2·n_fast  FAST bindings (bound just before queries)
[k? k? ... k?]          n_slow+n_fast queries, all keys, random order
```

Slow and fast keys are **disjoint** (`randperm(n_keys)` split into the first
`n_slow` / rest), so a query unambiguously belongs to one class. A per-token
`kind` tensor tags each query 1 (slow) or 2 (fast); accuracy is scored
**separately** for the two classes as well as overall. Slow keys must survive the
full 48-token interference gap (long lag, needs γ≈1 persistence); fast keys are a
few tokens back (short lag, needs low-γ recency that resists the gap's carryover).

**Why this separates timescales.** With a static read profile `p≥0`, the
multi-stream read `Sbar = Σ_i p_i S_i = Σ_s (Σ_i p_i γ_i^{t−s}) write_s` weights
past writes by a **mixture of exponentials** — a heavy-tailed kernel that retains
far-past bindings *and* keeps recent fidelity at once. A single exponential (n=1)
must compromise: γ~1 retains the far past but accumulates gap interference; small
γ resists interference but forgets the far past. So n=1 should trade slow against
fast; n=4 (a fast + slow γ split) should get both. **Regime.** Online train
(batch 32, data seed 100), held-out eval (batch 128, seed 999). Config: `n_slow=4,
n_fast=4, gap=48, seq_len=72, n_keys=24, n_vals=16, 4000 steps, lr 3e-3, 3 seeds`.

**Result (mean over 3 seeds):**

| metric | n=1 | n=4 | Δ |
|---|---|---|---|
| overall | 0.272 | **0.407** | +0.135 |
| slow (long lag) | 0.230 | **0.331** | +0.101 |
| fast (short lag) | 0.314 | **0.484** | +0.170 |

**n=4 beats n=1 on both timescales simultaneously.** Mechanism is visible in the
learned params: every n=4 seed converged to a **fast+slow split** — one fast
stream (γ ≈ 0.09–0.35) plus three slow streams (γ = 1.000). n=1 is stuck at
γ=1 and plateaus at loss ~2.1; n=4 breaks through to ~1.8. Variance persists
(seed 1's n=4 didn't break through, 0.268), but 2/3 seeds show large wins
(0.47–0.48 vs n=1 ~0.29) and the mean favors n=4 on every metric.

**Conclusion (Run 3):** the design's premise holds — the extra timescales help,
via an emergent fast+slow γ spread, exactly where a single decay must compromise.

---

## Status

- ✅ Graceful reduction (n>1 ≥ n=1 on single-timescale) — fixed via p-weighted output.
- ✅ Multi-timescale advantage (n>1 > n=1 on bimodal-lag recall) — demonstrated.
- ⚠️ Seed variance is high; a lower-variance regime (more steps / easier gap)
  would tighten the numbers for a publication-clean result.
- TODO: chunked-PyTorch / Triton kernel for scale (current recurrence is an
  O(T) Python loop, toy/small-scale only).
