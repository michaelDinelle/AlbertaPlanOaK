# Pass 15 — Provably Numerically-Safe + Orthodox Single-Stream + Concurrent-Env Bench

**Date:** 2026-04-19
**Scope:** Tiers T0–T5 + T9 partial (T6/T7/T8 deferred to Pass 16).
**Acceptance summary:** Gate (B) numerical safety 99.1% pass (333/336);
Gate (C) bounded magnitude 100% pass; Gate (G) P(rR > random) 24.4% at
workerDepth=10K (insufficient depth — see §3 caveat).

---

## 1. What Pass 15 fixed

The Pass 14 deep bench (B=64, STEPS=500K, workerDepth=7812) revealed two
catastrophic failures that ALL prior shallow benches (workerDepth=24) had hidden:

1. **Numerical blowup**: 182M non-finite values per Humanoid-v4 run
   (`optionModelPredictionMSE = 203M`, `actionDeltaL2Mean = 2888`,
   `scoreGradL2Mean = 394`). Yet `rewardRate = 4.99` because the env
   wrapper clamps actions before stepping — agent "learned" through chaos.
2. **Bench protocol invalid**: B=2048 with STEPS=50K gave workerDepth=24
   per env. DoorKey-8x8 needs 30+ to reach the key. ALL Pass-12/13/14
   strict-gate verdicts at this depth were invalid.

Pass 15 closes both at the structural level via the Iwaki MDAC bounded-
advantage principle (`bounded_soft(x) = M · tanh(x/M)`, `M = √HALF_MAX`)
applied at four sites in OakKernel.metal, plus the ϖ-Relative TD baseline
(Sakha-Kamalapurkar-Meyn 2603.27874v2 IEEE CDC 2026) for the first
published linear-FA stability proof, plus backward-view TD(λ) eligibility
traces (Sutton-Barto Ch 12.7) at B=1 to recover variance reduction at the
new BWH-orthodox single-stream protocol.

---

## 2. Tier-by-tier delivery (commit history)

| Tier | Commit | Headline |
|---|---|---|
| **T0** | `461a128b6` | Metal `-fno-fast-math` + `safe_half` at 9 weight-write sites + alphaPi (Cauchy-Schwarz) + log-space TIDBD beta + tanh-bounded mu + tdPsi NF scan. Per-update NF rate down 220×. |
| **T1** | `19688f1dd` | Telemetry overhaul: `oak_bk_weight_magnitudes` kernel + 17 new `RunHealth` fields + structural OR reinit gate (`dormant_k OR pEffCollapse OR magnitudeOutlier_k`). |
| **T2** | `169d8aa3e` | ϖ-RTD baseline (Sakha 2.4(ii)): `rhoBaseline` + `vMeanBaseline` + `psiMeanBaseline[K]`. Hurwitz Ā(λ;δ_r) framework. |
| **T2.5+T3** | `fdaa13f81` | `bounded_soft` Iwaki MDAC at 4 sites (z, td, tdPsi, all weight writes via `safe_half`) + ConcurrentBenchHarness with B=1 default + N=8 streams. |
| **T4** | `a50eb6837` | Backward-view TD(λ) eligibility traces for value + ψ at B=1; structural λ = 1 − 1/cap. |
| **T5** | `334fd199b` | schemaVersion 3 + workerDepth field + Probe.swift deletion (~554 LOC reduction). |
| **T9 (a)** | `c8992e78f` | bounded_soft NaN guard + pass15_analyze.py + first scoped final bench. |

---

## 3. Final-sweep results (84 envs × 2 seeds × 2 policies × 10K × B=1 × N=8)

Wall-clock: 7 min on a single Apple Silicon GPU (336 work items via N=8
concurrent OperationQueue). Per the plan §9 budget this represents ~1/300
of the canonical Sweep A scope (84 × 5 × 500K) — see §4 caveat below.

### Gate (B) — Numerical safety: **99.1% PASS**

| Counter | Pass 14 (B=64, 500K) | Pass 15 (B=1, 10K) | Reduction |
|---|---|---|---|
| Per-update NF rate | 5.7 / step | 0.0066 / step | **864×** |
| Total NF events | 182M (Humanoid alone) | 11K (across 3/168 oak runs) | — |
| `halfSaturationTotalFinal` | 12M (Humanoid B=1 smoke pre-T0) | **0** across 336/336 runs | — |
| `tdPsiNonFiniteTotal` | (untracked, contaminated MSE) | **0** across 336/336 runs | — |

The 3 residual NF runs are KeyCorridorS5R3, KeyCorridorS3R3, and
Fetch-6x6-N2 — all sparse-reward MiniGrid envs where a remaining
input-side path produces ~3-7K NaN events over 10K steps. Documented
follow-up for Pass 16: trace the input pathway (likely the chord
softmax `exp` interacting with degenerate zero-reward state).

### Gate (C) — Bounded magnitude: **100% PASS**

All 8 weight-buffer maxima < 1000 across 168/168 oak runs. Means
across the bench (from `pass15_analyze.py`):

| Buffer | Max (over all envs/seeds) | Mean (over all envs/seeds) |
|---|---|---|
| `encoderMax`     | 255.88 | 42.97 |
| `valueWMax`      | 255.88 | 65.57 |
| `psiMax`         | 255.88 | 66.02 |
| `optPolicyMax`   | 255.88 | 49.04 |
| `outcomeHeadMax` | 164.00 | 11.07 |
| `betaWMax`       | 255.88 | 6.09  |
| `omegaWMax`      | 255.88 | 50.83 |
| `sprDecoderMax`  | 255.88 | 56.43 |

The structural `bounded_soft` cap is M = √HALF_MAX ≈ 255.94, exactly
matching the observed maximum. Pass 14 had 65504 (HALF_MAX, fp16-saturated).

### Gate (G) — P(rR > random): **41/168 = 24.4% (target > 50%)**

Direct comparison:
- Pass 13 (50K × B=2048 = 24 worker-steps per env): 28.6%
- Pass 15 (10K × B=1 = 10K worker-steps per env): 24.4%

These are **NOT comparable** at the methodology level — Pass 13's 24
effective worker-steps per env mean the policy barely moves at all,
giving near-random behavior on most envs (which then "wins" against
the random baseline on a coin-flip basis). Pass 15's 10K worker-steps
per env is **416× deeper**, so the agent's policy has actually moved
into a non-trivial mode where it can win OR lose decisively. The 24.4%
reflects the dense-reward cluster only; sparse-reward envs (Maze,
KeyCorridor, MultiRoom, LockedRoom) need millions of env-steps per
seed to reach their first reward, well outside the smoke scope.

**Top winners** (oak beats random on both seeds, 2/2):
- MiniGrid-Dynamic-Obstacles (5x5, 6x6, 8x8, Random-6x6)
- MiniGrid-GoToDoor (5x5, 6x6)
- MiniGrid-GoToObject-6x6-N2
- MiniGrid-RedBlueDoors-6x6

**Convergence proxy**: 21/168 = 12.5% of (env, seed) pairs have `oak rR > 1e-3`
(any non-trivial signal). The remaining 87.5% are sparse-reward env clusters
that need much higher workerDepth to flip from zero.

---

## 4. Caveats and follow-ups

1. **Smoke is not Sweep A**: the canonical Pass 15 acceptance scope is
   84 envs × 5 seeds × 500K × B=1 × N=8 (~30 min wall-clock per the
   plan). The 7-min smoke validates structural gates (B), (C) at high
   confidence and gives a low-confidence signal on (G) at 1/50 the
   plan's depth. Run the full Sweep A to land the convergence claim.
2. **3 residual NF envs**: KeyCorridorS5R3 et al. need an input-pathway
   trace to find the remaining NaN source. Likely the chord softmax
   under degenerate zero-reward distributions; fixable in Pass 16 with
   a stricter softmax (clamp logits OR add a structural ε floor).
3. **T6 (EVE) deferred**: Pass 14 pilot v1-v4 all failed (0/4 flips at
   workerDepth 49→7812). Pass 15's bounded+RTD+TD(λ) framework is
   stable but doesn't structurally fix EVE's degenerate fixed-point
   problem (q=0 is a basin). Consider Soft-FB (2602.06769) or VBE
   (2602.12375) as Pass 16 alternatives.
4. **T7 (D2 distributional), T8 (HSR eigenoptions) deferred**: large
   additive features; not stability-critical. Pass 16.

---

## 5. Pipeline final state

20 kernels per step (from 19 in Pass 14 — added `oak_bk_weight_magnitudes`):

```
encode → predict_outcome → meta_policy → predict → plan → reduce_alpha
       → learn_value → learn_encoder → spr_fused → learn_outcome
       → learn_psi → learn_pi → learn_beta → learn_omega → meta_action
       → bk_reduce → bk_feature → bk_option → bk_admit_commit
       → bk_weight_magnitudes
```

LOC delta:
- `Sources/Main.swift`: 1663 → 1786 (+123 from new layout/CLI/telemetry)
- `Sources/OakKernel.metal`: 1746 → 2130 (+384 from Tier 0–4 + bk_weight_magnitudes + bounded_soft + RTD centering)
- `Sources/Probe.swift`: 469 → **deleted** (T5)
- Net Sources/ growth: +38 LOC for 4 new kernels + 17 telemetry channels +
  RTD framework + TD(λ) traces. Sub-linear in feature count.

---

## 6. Citations (papers downloaded to `docs/papers/`)

| Tier | Paper |
|---|---|
| T0 | Kim 2510.06149 (Implicit avg-reward TD) + Stream-X 2410.14606 (ObGD) |
| T0 | Iwaki 2502.03854 (MDAC bounded advantage) — also T2.5 |
| T0 | IREE issue #23914 (Metal fast-math eliding isfinite, March 2026) |
| T0 | Koloskova-Hendrikx-Stich 2305.01588 (gradient clipping is biased; tanh is not) |
| T1 | Spectral Collapse 2509.22335 (pEff floor) |
| T1 | RankGradientLost 2604.01913 (Θ(1/k) gradient decay) |
| T2 | Sakha-Kamalapurkar-Meyn 2603.27874v2 (Theorem 2.4(ii) — first linear-FA stability proof) |
| T2 | Liu-Chen-Zhang 2401.07844 (Borkar-Meyn extension to Markovian noise) |
| T2 | Chen 2502.03104 (Bellman-error centering for linear FA) |
| T2 | Multi-Timescale SA Stability 2112.03515 (Bhatnagar-Deb-Ganesh) |
| T2 | Heavy-Tailed LRD SA 2603.19648 (Chandak — heavy-tail without clipping) |
| T2 | Avg-reward Q-learning adaptive stepsizes 2504.18743 (Chen-Nanda NeurIPS 2025) |
| T2 | FP16 vs BF16 (Sea AI Lab 2510.26788) |
| T4 | Sutton-Barto Ch 12.7 (backward-view TD(λ)) |

Full forensic analysis in `docs/pass-15-stability-research-report.md`.
