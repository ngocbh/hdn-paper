# IDEA: Bringing mHC's manifold constraints to the *horizontal* residual of Gated DeltaNet-2

**Date:** 2026-07-08
**Status:** brainstorm / research proposal (pre-implementation)
**Question:** mHC constrains the *deep* (cross-layer) residual carry to a nice manifold
(doubly-stochastic / orthostochastic `H_res`). Can the same idea improve the *horizontal*
(cross-time) residual — the memory-state update — in Gated DeltaNet-2?

**Answer:** Yes, but not by dropping doubly-stochastic `H_res` in literally. The transferable
thing is mHC's *principle* — **constrain the residual carry operator to a manifold with
controlled spectral norm** — and it maps onto two different axes of the memory update, each of
which wants a *different* mHC manifold flavor.

---

## 1. Background: the two residuals

**mHC / hyper-connections (depth axis).** With `s` residual streams, the per-layer update is
```
x_{l+1} = H_res · x_l  +  H_post^T · F(H_pre · x_l, W_l)
```
- `H_res` (s×s): mixes streams. mHC constrains it to the **Birkhoff polytope** (doubly
  stochastic, via Sinkhorn) or the **orthostochastic** set (squared entries of an orthogonal
  matrix, via Newton–Schulz). This keeps the deep residual highway norm-controlled (no
  explode/vanish across depth).
- `H_pre`, `H_post`: **separate** non-negative read/write mixing maps.

**Gated DeltaNet-2 (time axis).** (Hatamizadeh, Choi, Kautz — NVIDIA, arXiv:2605.22791.)
The ladder of linear-attention memory recurrences (state `S ∈ R^{d_k×d_v}`, `o_t = S_t^T q_t`):

| Model | State update `S_t = …` | Carry operator on `S_{t-1}` |
|---|---|---|
| Linear attn | `S_{t-1} + k_t v_t^T` | `I` |
| Mamba-2 | `α_t S_{t-1} + k_t v_t^T` | `α_t I` (scalar decay) |
| **DeltaNet (Eq. 5)** | `S_{t-1} + β_t k_t (v_t − S_{t-1}^T k_t)^T` = `(I − β_t k_t k_t^T)S_{t-1} + β_t k_t v_t^T` | `I − β_t k_t k_t^T` (**symmetric**) |
| Gated DeltaNet | `α_t (I − β_t k_t k_t^T) S_{t-1} + β_t k_t v_t^T` | `α_t(I − β_t k_t k_t^T)` |
| KDA | `(I − β_t k_t k_t^T) D_t S_{t-1} + β_t k_t v_t^T`, `D_t=Diag(α_t)` | `(I − β_t k_t k_t^T) D_t` |
| **Gated DeltaNet-2** | `(I − k_t e_t^T) D_t S_{t-1} + k_t z_t^T`, `e_t=b_t⊙k_t`, `z_t=w_t⊙v_t` | `(I − k_t e_t^T) D_t` (**non-symmetric**) |

GDN-2's contribution: replace the single scalar `β_t` (which controlled *both* how much to erase
on the key side *and* how much to write on the value side) with a **channel-wise erase gate**
`b_t ∈ [0,1]^{d_k}` and a **channel-wise write gate** `w_t ∈ [0,1]^{d_v}`. It recovers KDA when
`b_t=w_t=β_t·1` and Gated DeltaNet when decay is also scalar.

---

## 2. Core mapping

Define the **horizontal carry** `T_t = (I − k_t e_t^T) D_t`. Then `S_t = T_t S_{t-1} + write_t`
is *structurally identical* to mHC's `x_{l+1} = H_res x_l + …`. So:

> **`T_t` is the time-axis analog of `H_res`.** The erase/write gates are the analog of
> `H_pre`/`H_post`.

Stability of the long-context recurrence depends on the spectral behavior of the cumulative
product `T_t T_{t-1} … T_1` — exactly what mHC controls for depth.

---

## Terminology — expansive vs non-expansive (`σ_max`)

**What it means.** A linear map `M` is **non-expansive** if it never grows the norm of any input:
`‖M x‖ ≤ ‖x‖` for all `x`. It is **expansive** if *some* direction gets stretched
(`‖M x‖ > ‖x‖`). The worst-case stretch over all directions is the largest singular value
`σ_max(M)` (= `‖M‖₂`), so:
```
non-expansive  ⟺  σ_max(M) ≤ 1     (a "contraction" — it can only keep or shrink size)
expansive      ⟺  σ_max(M) > 1
```
Scalar intuition: `×0.9` never grows a number (non-expansive); `×1.1` does (expansive).

**Why it matters.** The recurrence multiplies the state by `T_t` every step
(`S_t = T_t S_{t-1} + write`), so old memory passes through the product `T_L···T_1`. Non-expansive
`T_t` ⇒ old memory can only stay or shrink (= forgetting, what we want). Expansive `T_t` ⇒ it can
grow step after step, swamping new writes and degrading long-range retrieval.

**The subtlety (eigenvalues ≠ singular values).** Eigenvalues govern the *asymptotic* fate (many
steps); singular values govern *one step*. A **non-normal** matrix (eigenvectors not orthogonal)
can have all eigenvalues ≤ 1 yet still stretch *transiently* in a single step. Example — the
GDN-2 erase with `k=(1/√2,1/√2)`, `b=(1,0)`:
```
M = I − k eᵀ = [[ 0.5, 0  ],
                [ -0.5, 1  ]]
eigenvalues = {0.5, 1}     # both ≤ 1  → looks stable
σ_max       ≈ 1.14         # but one direction stretches 14%  → expansive
```
Symmetric matrices are *normal* → eigenvalues = singular values, so for GDN/KDA "eigenvalues ≤ 1"
already implies non-expansive. GDN-2's decoupling makes the erase non-symmetric, opening the gap.
"Put the carry on the non-expansive manifold" = constrain `T_t` so `σ_max ≤ 1` — no single step
can stretch the state in any direction.

---

## 3. The key insight (why there is a real gap)

**GDN & KDA carries are non-expansive; GDN-2's is not.**

- DeltaNet/GDN/KDA erase factor `I − β k k^T` (with `‖k‖=1`) is **symmetric**, eigenvalues
  `{1 (×d−1), 1−β}`. For `β∈[0,2]`: `σ_max = 1` — every step non-expansive. KDA stays safe even
  with `D_t` because it is a product of two *individually* non-expansive factors
  (`projector · decay`, `‖D‖≤1`).
- GDN-2 erase factor `I − k e^T` with `e = b⊙k` is **non-symmetric** (left `k`, right `b⊙k`).
  Its *eigenvalues* stay in `[0,1]` (asymptotically stable), but its *singular values* can
  exceed 1 — a **non-normal / transient amplification**. Submultiplicativity no longer saves it.

**Theorem (non-expansiveness boundary).** With `‖k‖=1`, let `c = k^T e = Σ b_i k_i²` and
`p = ‖e‖² = Σ b_i² k_i²`. Then
```
σ_max(I − k e^T)² = 1 + λ₊,   λ₊ = ½[ (p − 2c) + √( p·(p + 4 − 4c) ) ]
```
- Since `p ≥ c²` (Cauchy–Schwarz, weights `k_i²`), the discriminant `p(p+4−4c) ≥ p(c−2)² ≥ 0`
  (real) and `λ₊ ≥ 0`, so **`σ_max ≥ 1` always**.
- `σ_max = 1` **iff** `p = c²` **iff** `e ∥ k` **iff** `b_t` is a per-head scalar (= KDA/GDN).
- Example: `k=(1/√2,1/√2)`, `b=(1,0)` → `σ_max ≈ 1.14`.

**Consequence.** GDN-2's core innovation (channel-wise `b` ⇒ `e ∦ k`) *necessarily* sacrifices
step-wise non-expansiveness. Decay `D_t` masks it only when `α` is small — but the long-memory
regime `α→1` is exactly where GDN-2 claims its wins (long-context RULER / MK-NIAH). The paper
notes "the spectral effect concerns the state transition" (its negative-eigenvalue `b∈[0,2]`
variant) but **never constrains it**. That is the opening.

**Each paper holds *half* of mHC:**

| | Constrained carry (`H_res`-like) | Separated read/write (`H_pre≠H_post`) |
|---|---|---|
| DeltaNet(Eq.5)/KDA | ✅ non-expansive | ❌ tied to `k` |
| GDN-2 | ❌ `σ_max>1` | ✅ decoupled |
| **mHC** | ✅ | ✅ |

**Anchor = Eq. 5.** The plain delta rule's carry `I − β k k^T` is a one-parameter path
**identity (β=0) → orthogonal projector (β=1) → Householder reflection / orthogonal (β=2)**,
sitting inside the norm-controlled family mHC's orthostochastic/Newton–Schulz variant is built
around. It is also a **stable optimizer step**: the delta rule is one GD step on `½‖S^T k − v‖²`
(Hessian `k k^T`), so the carry is `I − β·Hessian`; GD is stable for `β ≤ 2/λ_max = 2` — the same
window. So Eq. 5 is the "on-manifold" object; GDN-2 stepped off it to gain expressiveness.

---

## 4. Design principle

> Reclaim mHC's stability without giving up GDN-2's decoupled expressiveness, by putting **each
> residual carry back on the manifold that is correct for its axis**.

**Which manifold on which axis:**

| Axis | Operator | Correct mHC manifold | Why |
|---|---|---|---|
| **Within a state (time)** | transition `T_t` | **non-expansive / orthogonal** (Direction A) | forgetting must stay legal → contraction allowed, *not* mass-preserving |
| **Across states (streams)** | mixing `H` | **doubly-stochastic / orthostochastic** (Direction B) | forgetting handled per-stream → mixing should preserve mass, no stream domination |

Birkhoff (doubly stochastic) is *wrong inside one state* (it forbids forgetting) but *exactly
right across states*. This is the crux and it makes A and B complementary rather than redundant.

---

## 5. Direction A — put the *single* carry on the non-expansive / orthogonal manifold

**Dichotomy.** We proved a rank-1 erase is non-expansive iff `e ∥ k`, i.e. you cannot have
*decoupled* **and** *rank-1* **and** *non-expansive* at once. Two escape routes: **damp** (A0)
or **enrich** (A1).

### A0 — non-expansive projection (drop-in, minimal)
Per token, compute the two scalars `c = k^T e`, `p = ‖e‖²` (O(d)), then `σ_max` from §3, then
```
T̃_t = M_t D_t / max(1, σ_max(M_t)),   M_t = I − k_t e_t^T
```
- Guarantees `‖T̃_t‖ ≤ 1` every step; kills non-normal blow-up; keeps decoupling.
- Cost: two dot products + one scalar divide, folded into the decay. **Nothing else about GDN-2
  changes.** This is the faithful analog of Sinkhorn-projecting `H_res`, adapted to the
  rank-1+diagonal structure and to the *non-expansive* set.
- Variants:
  - **A0-whole:** divide the entire transition (damps identity part too → adaptive extra decay).
  - **A0-corr:** damp only the correction `k e^T` (keeps the identity part; softer).
  - **A0-soft:** replace hard `max(1,σ)` with a smooth penalty `λ·relu(σ_max−1)²` as an
    auxiliary loss (no forward change; nudges gates toward stability).
  - **A0-cap-b:** rescale `b_t` toward the aligned direction only enough to hit `σ_max ≤ 1+ε`.

### A1 — orthogonal-by-construction (structural, bigger swing)
Factor transition into *reorganize* then *forget*:
```
S_t = D_t · R_t · S_{t-1} + k_t z_t^T,   R_t orthogonal
```
- `R_t = I − 2 k_t k_t^T` is exactly the delta rule at `β=2` — mHC's *single* Householder.
- **Product of `p` Householders** `R_t = ∏_{j≤p}(I − 2 u^{(j)} u^{(j)T})` — richer orthogonal
  transform. The WY / UT-transform kernel machinery *already* represents products of rank-1
  reflectors, so it stays chunkable and linear-time (`p = 2–4`).
- **Givens / block-diagonal 2×2 rotations** with data-dependent angles — cheaper orthogonal,
  O(d) per token; the bridge to **Mamba-3's** data-dependent complex rotations.
- `D_t` is the *only* contraction → memory reorganization is norm-preserving, forgetting is
  explicit and separate.

### A2 — hybrid (likely sweet spot)
A0-capped small erase (associative overwrite) **+** A1 orthogonal reorganization. Keeps the delta
rule's "remove old value at `k` before writing" while adding norm-preserving structure.

**Honest tension for A1:** replacing erase-by-subtraction with rotation weakens associative
overwrite; A2 mitigates. A0 is the safe first experiment.

---

## 6. Direction B — multi-stream hyper-memory (the literal HC transfer)

Replace the single state `S` with `n` streams `S^{(1..n)}` in a **shared** `d_k×d_v` space, and
let them **exchange information via a doubly-stochastic mixing** — "hyper-connected memory lanes."

### Update
Per step, each stream runs its own (gated-delta-2) update, then streams mix:
```
S̃_t^{(i)} = T_t^{(i)} S_{t-1}^{(i)} + k_t^{(i)} z_t^{(i)T}     # per-stream (A-constrained) carry
S_t^{(i)}  = Σ_j H[i,j] · S̃_t^{(j)}                            # cross-stream mixing, H doubly stochastic
```
`H ∈ R^{n×n}` from `sinkhorn_log` (Birkhoff) or Newton–Schulz (orthostochastic) — **reuse the
repo's exact code**.

### Efficiency — mix at chunk boundaries
Applying `H` per token to full states breaks chunkwise structure. Instead apply `H` **only at
chunk boundaries** (the inter-chunk state recurrence in the WY algorithm). Within a chunk streams
evolve independently in parallel (existing per-head kernel untouched); at each boundary mix the
`n` carried states:
```
S[c+1]^{(i)} = Σ_j H[i,j] · (chunk-update of stream j from S[c]^{(j)})
```
Boundaries are `O(L/C)`, so the `n×n` state mix is negligible overhead → linear time preserved.

### Why doubly-stochastic is correct here
Forgetting is handled *inside* each stream's `T_t`; cross-stream mixing should only route/share
information without amplifying or letting one stream dominate/collapse → mass-preserving =
doubly-stochastic. Exactly mHC's original justification, now correctly placed.
`H = Σ λ_p P_p` (Birkhoff–von Neumann: convex combo of permutations) ⇒ `‖H‖_2 ≤ 1`.

### What differs across streams (to make mixing meaningful)
- **B-timescale (primary):** shared projections, different decay ranges per stream (fast →
  slow). Mixing lets fast and slow memory exchange content each chunk → multi-timescale
  associative memory. Directly targets long-context.
- **B-heads:** different Q/K/V/gate projections per stream (needs a shared readout space or a
  learned map to mix, since standard heads live in different subspaces).
- **B-budget:** same projections, different Householder budget `p` / init per stream.

### Static vs dynamic `H`
Start **static per-layer** `H` (cheapest, most stable, like mHC). Optionally data-dependent
`H_t` re-projected to doubly-stochastic each chunk (more expressive, more cost).
Init near identity via the repo's `residual_identity_mix` idea: `H = (1−α)I + α·Sinkhorn(logits)`.

### Readout & cost
- Readout: `o_t = Σ_i (S_t^{(i)})^T q_t^{(i)}` (sum) or concat → output projection (multi-head
  style).
- Cost: `n×` state memory (`n` small, 2–4); `+ n×n` logits/layer; `+` Sinkhorn (cheap);
  `+` boundary mix `O(n² d_k d_v · L/C)` (negligible). Not free, but cheap.
- Prior art: multi-head linear attn (no state mixing), Mamba-3 MIMO (multi-rank, not DS
  cross-state mixing), hyper-connections (depth). B = HC along the memory axis of linear attn — novel.

---

## 7. A + B compose into a provably non-expansive memory

If each per-stream `T_t^{(i)}` is non-expansive (Direction A) **and** `H` is doubly-stochastic
(`‖H‖_2 ≤ 1`, Direction B), the whole multi-stream memory operator is non-expansive ⇒ **stable
long-context memory by construction**, while retaining GDN-2's decoupled expressiveness *inside*
each stream. Full stack:

```
per stream:   S̃^{(i)} = D^{(i)} · R^{(i)} · S^{(i)}_{prev} + k z^{(i)T}   # A: non-expansive/orthogonal
cross stream: S^{(i)}  = Σ_j H[i,j] S̃^{(j)}   (H doubly stochastic)        # B: mass-preserving mix
```

---

## 8. Validation plan (do this *before* kernel work)

1. **Confirm the premise.** Instrument `σ_max(T_t)` distribution and `‖S_t‖` / `cond(S_t)` growth
   vs sequence position in a small trained GDN-2 (recurrent-only, `α→1` regime). Does the state
   norm actually grow / conditioning degrade at long context? (Premise risk: decay + direction
   decorrelation across steps may absorb the per-step `σ_max≈1.1`.)
2. **Reference implementation first.** Implement A0/A1/B in a plain (non-Triton) recurrence for
   correctness + small-scale signal (matches this repo's "correctness + clarity" ethos) before
   touching kernels.
3. **A/B/ablation matrix:** baseline GDN-2 vs `+A0` vs `+A1` vs `+A2` vs `+B` vs `+A+B`, on:
   WikiText/LAMBADA ppl, **MK-NIAH** (multi-key long-context — GDN-2's headline), and state-norm
   stability. Hypotheses: A reduces state-norm blowup without hurting ppl; B helps multi-key
   retrieval (more lanes + mixing = separate competing associations).
4. **Testbed note:** the `examples/nanogpt` here is softmax-attention GPT, so real validation
   needs a GDN / linear-attention testbed. The **longhstu** GDN branch
   (`/data/repos/fbsource2`) is the natural home for scaled experiments; this repo is the home
   for the clean reference + toy validation.

---

## 9. Implementation reuse from this repo

- `hyper_connections/hyper_connections_mhc.py::sinkhorn_log` → `H` mixing for B.
- Orthostochastic Newton–Schulz path (`mhc_h_res_proj="orthostochastic"`) → orthostochastic `H`
  (B) and orthogonal `R_t` (A1).
- `mhc_residual_identity_mix` / `mhc_residual_alpha` → near-identity init for `H` (B) and for the
  A0 damping / A2 mix.
- Stats hooks (`collect_stats`, `last_stats`) → log `σ_max`, `‖S_t‖`, `H` entropy during
  validation.

---

## 10. Open questions & risks

- **Premise:** does transient amplification matter at trained scale, or is it absorbed by decay +
  step-to-step direction decorrelation? (→ §8.1 first.)
- **A1 semantics:** does rotation-instead-of-erase lose associative recall? (→ A2 hybrid.)
- **B collapse:** does static `H` learn non-trivial routing or collapse to `I`/permutation?
  (Watch `H` entropy; mHC found DS helps, but that was the depth axis.)
- **Backward pass:** A0's `max(1,σ)` divide has a kink (subgradient fine); A1 Householder-product
  backward = existing WY backward; B boundary-mix backward = small `n×n`. All tractable.
- **Negative-eigenvalue variant (`b∈[0,2]`):** A0's `σ_max` formula already handles `c` up to 2
  (discriminant `≥0` proven via `p≥c²`). A0 and the `[0,2]` state-tracking trick interact — A0
  will damp the extra spectral radius the trick introduces; measure whether that removes the
  trick's benefit.
- **Scope:** paper-style contribution vs a drop-in for the longhstu GDN branch — decide before
  investing in kernels.

---

## 11. References

- mHC (Manifold-Constrained Hyper-Connections), DeepSeek — arXiv:2512.24880
- Hyper-Connections — arXiv:2409.19606; Frac-Connections — arXiv:2503.14125
- Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention (Hatamizadeh, Choi, Kautz,
  NVIDIA) — arXiv:2605.22791; code: github.com/NVlabs/GatedDeltaNet-2
- Gated Delta Networks (GDN) — arXiv:2412.06464
- Parallelizing Linear Transformers with the Delta Rule over Sequence Length — arXiv:2406.06484
- KDA (Kimi Delta Attention); Mamba-2 (SSD); Mamba-3 (exp-trapezoidal, complex rotations, MIMO)
- Unlocking state-tracking in linear RNNs through negative eigenvalues (Grazzi et al., ICLR 2025)
