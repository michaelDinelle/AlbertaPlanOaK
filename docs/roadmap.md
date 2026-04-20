# OaK roadmap — closing the gap to full Step-11 Sutton agent

This doc tracks the architectural pieces of the Alberta Plan Step-11 Oak agent
(Sutton-Bowling-Pilarski 2022, [`docs/papers/AlbertaPlanForAIResearch_2208.11173.pdf`](papers/AlbertaPlanForAIResearch_2208.11173.pdf)),
which of them OaK currently has, and the order we'll add the rest. Each tier is
one PR, validated against the prior committed checkpoint with the same paired
oak-vs-random benchmark protocol.

The pacing principle (validated empirically since Tier A): one Plan-orthodox
structural change per PR, smoke (NF zero) → 1.5M paired bench (sanity) → deeper
characterize, decide next move from data, never skip multiple steps.

## State of the agent on Plan Steps 1-11

| Plan Step | What we have | Quality |
|---|---|---|
| 1. Representation I (per-feature α) | MetaOptimize L-approx + β-floor (commits `fccaed018`, `15a4632e5`) | Strong |
| 2. Representation II (feature finding) | Single-layer deep Fourier features + G&T reseed of worst pre-activation | Partial — single layer only, not compositional |
| 3. Prediction I (continual GVF) | K=32 ψ-GVFs + adaptive β bootstrap (`β_k = (1−1/R)·α₀/(α₀+utilOpt[k])`, Tier B) | Partial — no recurrent traces, no off-policy, K capped at SIMD width |
| 4. Control I (actor-critic) | Chord-blended REINFORCE + α[i] relevance gate | Strong |
| 5. Prediction II (avg-reward) | ρ̄ centering (Reward Centering 2405.09999) | Strong |
| 6. Control II (continuing) | Continual env wrappers, Inv-4 enforced | Strong |
| 7. Planning I (avg-reward planning) | `oak_plan` parallel async VI w/ ρ̄, dense option models (Tier B) | Partial — 1 planning sweep per env step, no Dyna iterations |
| 8. Prototype-AI I (one-step MBRL) | All four base components present, model-feature feedback closed | Partial — no search control / prioritized sweeping |
| 9. Planning II (search control + exploration) | Chord softmax over ψ selects "where to focus" | Weak — uniform option visitation, no priority queue |
| 10. Prototype-AI II (STOMP) | Subtasks → Options → Models → Planning all exist | **Degenerate** — chord recomputes every step → effectively 1-step "options". C2 attempt to add true termination regressed vs Tier B (5/10 vs 10/10 wins) and was rolled back; revisit after C1 lifts the representational ceiling. |
| 11. Oak (feedback / remove-replace) | Per-feature + per-option utility, G&T reseed | Strong — but only Design A keyboard; no Design B joint-subtask interpretation |

**The big honest gaps remaining** (post-rollback): Step 10 still has 1-step
"options" (the C2 attempt regressed; we'll come back to it after representation
lifts), Step 9 has no search control, Step 3 has no temporal traces in
features, Step 2 is single-layer, Step 7 still does 1 planning sweep per env
step.

## Empirical baseline

10 seeds × 3M HalfCheetah-v4 (no-floor, current `composer2` HEAD pre-Tier B):

- rewSum lift over random: **+212k/seed (10/10 wins)**, **+17.2%** less negative
- rewardRate plateau: **+0.071/step**, holds steady ~+22k per 300k window
- Walking peaks: 3 seeds reach `rewardRate > 0` transiently (max +0.221, seed 0
  HB1); none sustain to end of run

OaK is consistently above random on long horizons but plateaus rather than
reaches sustained walking (`rewardRate > +0.5`). The roadmap targets the
specific architectural reasons that's the case.

## Tier A — Plan Steps 1+2 (DONE)

```
fccaed018  deep fourier encoder + metaoptimize per-feature alpha
15a4632e5  metaoptimize beta-floor at -log(R*baseO)
```

Closes Step 1 fully. Closes Step 2 partially (single Fourier layer over `std_obs`).

## Tier B — Plan Steps 7 + partial 10 (DONE)

```
227542235  tier B (RR-Subtasks full): dense optTransition + parallel oak_plan
           + adaptive psi-beta + planAdv soft-bound: 0/10 diverged at 3M, chatter 0.17
```

Dense option transition `n̂^k_j(s) = Σ_i w^k_{j,i} · s[i]` (RR-Subtasks eq. 17),
parallel `oak_plan` (K threads + `simd_max` reduction in one simdgroup), adaptive
ψ-bootstrap `β_k = (1 − 1/R)·α₀/(α₀ + utilOpt[k])` (self-stabilizing
negative-feedback loop, replaces the 6/10-diverged constant-β attempt), and
`planAdv = x/(1+|x|)` soft-bound on the value-head update.

Closes the *planning-is-degenerate* and *chord-is-myopic* gaps. Result at 3M:
0/10 seeds diverged, chatter 0.17, plateau still ~+22k per 300k window — set
the stage for C2 to lift the plateau.

## Tier C — temporal extent (C1 NEXT, C2 ROLLED BACK, C3 pending)

### C2 attempt — true option termination (ROLLED BACK on data)

The C2 attempt added per-option termination weights `termW[K·R]`,
hold-or-refresh chord dynamics, closed-form geometric `Q^k = (r̂^k − ρ̄)/β^k +
V(n̂^k)` (Wan-Naik-Sutton 2021), and option-critic termW gradient
(Bacon-Harb-Precup 2017). It was numerically sound (0/10 NF) and structurally
correct, but **regressed against the Tier B baseline** on the 10×1.5M paired
bench:

| metric | Tier B 3M baseline | C2 v1 | C2 v2 (β₀ ≈ 0.5) |
|---|---:|---:|---:|
| oak wins | 10/10 | 5/10 | 5/10 |
| mean rewSum lift | +212 000 / seed | +728 | +3 465 |
| chatter (median) | 0.17 | 0.317 | 0.348 |
| seeds with `rewardRate > 0` | 3 | 1 | 3 |
| peak `rewardRate` | +0.221 (at 3M) | +0.134 | +0.332 |

C2 v2 actually hit *more* walking peaks than Tier B at half the steps, but the
median trajectory stayed flat — the per-seed aggregate was indistinguishable
from random. Per the principle of relentless pursuit of intelligence with no
regressions, **C2 was rolled back** in favor of recovering the Tier B
`+212k/seed at 3M` baseline, then attacking the next bottleneck (representation
via C1 below) from a known-good position. C2 will be revisited once the
encoder (C1) and feature depth (C3) have lifted what the policy is asked to
predict and act on.

**Kept from the C2 PR (pure perf wins, orthogonal to the algorithmic experiment):**

- `oak_meta` — K-thread parallel (one per option, `simd_max` + `simd_sum`).
  Now does only the Tier B chord update + αPi + ρ̄ EMA. **280 µs → 15 µs.**
- `oak_predict` — K-thread parallel (one per option, `simd_sum` NF aggregation).
  **144 µs → 15 µs.**
- `oak_bk_admit_commit` — single-TG with 256 threads, `threadgroup_barrier`
  between find-worsts and parallel reseed; bit-deterministic (same hash key as
  the prior single-threaded kernel). **186 µs → 17 µs.**

Net agent step time **~1.35 ms → ~1.0 ms** at the Tier B baseline algorithm
(measured ~970 sps in 10× concurrent 1.5M bench), so the post-rollback baseline
is *strictly faster* than the original Tier B baseline at no algorithmic cost.

### C1. Temporal aggregation in encoder (Plan Step 3 explicit)
*"include not just nonlinear combinations, but also incorporation of older
signals and traces"*

Add a structural decayed-obs trace `t_o = (1 − 1/baseO)·t_o + (1/baseO)·obs_o`
(one float per obs dim, baseO floats total). Encoder concatenates `[std_obs; t]`
so each Fourier pre-activation sees instantaneous + integrated obs.

- Plan-orthodox: trace-rate is `1/baseO` (already the established structural
  choice), no new tunable
- Targets POMDP coverage (gait phase, joint acceleration that obs doesn't expose)
- Blast radius: small (~30 lines kernel + host)

### C3. Multi-layer Fourier features (Plan Step 2 deep version)
*"creating and introducing new features made by combining existing features"*

Current single Fourier layer over `std_obs` only. Stack a second layer over
`state` to get higher-order combinations:
`state2[r] = cos(W2[r] · state + φ2[r])`. Downstream heads read `state2`.

Per Lewandowski-Schuurmans-Machado 2024 §3
([`docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf`](papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf))
deep Fourier features approximately embed deep linear networks → plasticity
preserved.

- Doubles encoder param count, doesn't change SIMD-bound K
- Plan-orthodox: same RFF-1 init, generate-and-test on (W2, φ2), no new tunables
- Blast radius: medium (new layer in encoder + state buffer doubles)

## Tier D — planning + horde (after Tier C lands)

### D1. Multi-step planning iterations (Plan Step 7 — Dyna)
*"Some amount of planning [each step], perhaps a series of small planning steps"*

Currently 1 planning sweep per env step. Run `oak_plan` M times per env step
using imagined rollouts through the option transition model. M structurally
derived from per-step compute budget (e.g., `M = baseO`).

- Trade env wall-clock for planning depth
- Blast radius: small (loop in `oak_plan`)

### D2. Search control / prioritized sweeping (Plan Step 9)
*"The order chosen [for state updates] greatly affects planning efficiency"*

Maintain a priority over option indices by |TD error| magnitude. `oak_plan`
updates highest-priority options first (or only). Replaces uniform K-option
sweep.

- Plan-orthodox: priority is a derivative state, no tunable
- Blast radius: small-medium (priority buffer + reordering in oak_plan)

### D3. Off-policy ψ + option-model learning (Plan Step 3 advanced)
*"Ideally this would be taken all the way to off-policy learning"*

Currently options learn from on-chord experience. With true termination (C2) we
can attribute experience to the active option's transitions specifically; pair
with importance ratios for cross-option learning.

- Blast radius: medium (importance ratios + per-option attribution)

## Tier E — horde scale + departures (last)

### E1. K > 32 / horde-scale GVFs
*Sutton et al. 2011, [`docs/papers/02-sutton-horde-AAMAS-2011.pdf`](papers/02-sutton-horde-AAMAS-2011.pdf):*
*"a scalable real-time architecture for learning knowledge from unsupervised
sensorimotor interaction"*

Lift K beyond SIMD width by using threadgroup-scale reductions in `oak_plan`
(multi-warp `simd_max` followed by threadgroup-`max`). Enables thousands of GVFs
per Plan Step 3 vision.

- Memory grows linearly with K; planning compute grows linearly
- Plan-orthodox: K becomes a true compute-budget knob
- Blast radius: medium-large (kernel restructure for multi-warp reductions)

### E2. R derived from compute budget
Currently `R = max(env.obsDim, 64)`. Replace with
`R = floor(M1 GPU memory budget / ε)`. Removes the env-input-dim coupling, lets
representational capacity grow with hardware. Tiny code change.

## Tier F — non-Plan-orthodox departures (avoid until justified)

- **F1. Learnable policy variance** — Plan-orthodoxy says structural σ=1
  (Williams 1992 REINFORCE variance is the score function); making σ learnable
  is a tunable
- **F2. Option keyboard Design B** — joint-subtask vs. chord-blend; the Plan §11
  alternative — would change the chord semantics entirely
- **F3. Discount factor reintroduction** — Plan Step 5 onward eschews
  discounting; we shouldn't backslide

## Decision graph

```mermaid
flowchart TD
    A[Tier A done] --> B[Tier B done: 0/10 diverged at 3M, +212k/seed]
    B --> C2att[C2 attempt: termW + closed-form Q^k + hold-or-refresh chord]
    C2att --> C2check{1.5M paired bench}
    C2check -->|5/10 wins, +3.5k/seed → REGRESSION| C2roll[C2 algorithm rolled back; kernel parallelizations kept]
    C2roll --> C1[Tier C1 next: temporal aggregation in encoder]
    C1 --> C3[Tier C3: multi-layer Fourier]
    C3 --> C2re[C2 retry: revisit termination once representation lifts]
    C2re --> D1[Tier D1: Dyna multi-step planning]
    D1 --> D2[Tier D2: prioritized sweeping]
    D2 --> D3[Tier D3: off-policy]
    D3 --> E1[Tier E1: horde-scale K]
    E1 --> E2[Tier E2: R from compute]
```

**Where we landed (post-rollback):** Tier B is the canonical baseline,
strictly faster than the original Tier B run (~970 sps vs ~700) thanks to the
parallelization wins kept from the C2 PR. C2's algorithmic changes (termW,
closed-form Q^k, hold-or-refresh chord, option-critic gradient) are removed
but documented above for the eventual retry — the principled order is
**representation first** (C1 then C3) so that when termination comes back, ψ
and the policy have non-trivial per-option signal to grow lifetimes from.
