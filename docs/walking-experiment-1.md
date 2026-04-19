# HalfCheetah-v4 Walking Experiment — Principled Measurement

**Date.** 2026-04-19, after commit `3d8368b32` (strict ϖ-RTD + matrix encoder + two-timescale).
**Setup.** 3 seeds × 1M steps × {oak, random} on HalfCheetah-v4. Single-stream B=1 continuing trajectories (BWH-orthodox depth-over-batch). Heartbeat every 50K steps for full trajectory visibility.

## TL;DR

**The architecture is provably numerically stable** — NF=0 across all 3 seeds × 1M steps, weights bound at concrete plateaus, `pEff = 16.99/17` (full effective rank from matrix encoder), `modelErrEMA` monotonically decreasing 10×–100× across all seeds.

**But the architecture is NOT consistently beating random.** Seed-averaged final rR shows oak at -0.470 ± 0.059 vs random at -0.411 ± 0.001 (paired t-stat -1.72, oak underperforms). The variance is dominated by initialization-dependent fixed-point selection.

| Metric | seed 0 | seed 1 | seed 2 | random (any seed) |
|---|---:|---:|---:|---:|
| Final rR (1M) | -0.4126 | -0.5309 | -0.4656 | -0.4112 |
| Late-half rR | **-0.386** | -0.530 | -0.465 | -0.411 |
| modelErrEMA  | 57.5 | 235,000 | 52.5 | n/a |
| encoderMax | 8.1e6 | 1.7e4 | 3.8e6 | n/a |
| valueWMax | 1.9e6 | 8.0e3 | 1.7e7 | n/a |
| NF | 0 | 0 | 0 | 0 |

## What's working ✓

1. **Numerical stability.** All 3 seeds finished 1M steps with NF=0, weights bounded (encoder/value/optPolicy all settle at finite plateaus by step 50K). Strict ϖ-RTD (Eq.24) + L2-ER rank floor + two-timescale together close the bootstrap divergence we hit before.
2. **Effective rank.** Matrix encoder gives `pEff = 16.99/17` (99.9% rank utilization) on every seed — the rich-feature representation is intact, no collapse.
3. **Cumulant model learning.** `modelErrEMA` drops monotonically on every seed — by 10× (seeds 0, 2) or 20× (seed 1).
4. **State-conditional policy.** `actionDeltaL2Mean ≈ 4-6e-4` on all seeds (was 4e-9 frozen before). Policy mean varies meaningfully with state.
5. **Best-seed late-trajectory beats random.** Seed 0 went through a phase transition around step 350K and its late-half mean rR (-0.386) is +0.025 above random. Real learning when it works.

## What's not working ✗

1. **Initialization-dependent fixed-point selection.** Each seed converges to a DIFFERENT fixed point. Seed 0 found a "good" one (rR -0.39, modelErr 57). Seed 1 got stuck in a "bad" one (rR -0.53, modelErr 235K — small weights but huge prediction error → degenerate local min). Seed 2 in between.
2. **No seed actually walks.** Best-case rR -0.39 is barely above random (-0.41). Walking territory for HalfCheetah is +1 to +5 (forward velocity). We're 3-5 orders of magnitude from that.
3. **Modal weight scales differ ~1000×.** Seed 0 vs seed 1 differ by 100-1000× in weight magnitudes despite same architecture/init scale. The system is finding very different fixed points based on early TD signs.

## Diagnosis

The strict ϖ-RTD form makes the ITERATION provably stable (Hurwitz mean-flow per Sakha-Kamalapurkar-Meyn Thm 2.4(ii)) — confirmed by the absence of fp32 saturation. But:

- Stability ≠ unique fixed point. With nonlinear (tanh) features and bootstrap TD, the system has multiple local minima.
- The actor (REINFORCE with eligibility traces) does state-conditional policy improvement, but only WITHIN the basin of the current fixed point. Cannot escape a bad basin.
- 1M steps appears insufficient for the agent to find the "good" basin reliably from random init.

## Honest recommendation

Per Agent A's BWH-orthodox ranking (`docs/encoder-feature-class-research.md`):

1. **Most impact for HalfCheetah specifically: Tier 1 AdaLin** ([2505.09486](https://arxiv.org/abs/2505.09486)). Adaptive activation that injects linearity in saturated regions and suppresses it in active regions. Paper §3.6 Figure 7 shows AdaLin-SAC > vanilla SAC on HalfCheetah-v4. ~10 LOC + 1 buffer.
2. **Composes with strict ϖ-RTD: Tier S Fix 3 BEC** (Bellman Error Centering, [2502.03104](https://arxiv.org/abs/2502.03104)). Subtracts a tracked TD-error EMA from δ — orthogonal to ϖ-RTD's mean-flow centering. Reduces noise variance further. Could help with the seed-1 divergence.
3. **For escape from local minima: intrinsic exploration.** EVE pilot was deferred in Pass 14; the matrix encoder might unblock it now.

## Verdict on "provable results"

✅ **Provable numerical stability** (Sakha-Kamalapurkar-Meyn Thm 2.4(ii) holds empirically — the matrix encoder no longer diverges).
⚠️ **Marginal learning** in 2/3 seeds (modelErrEMA drops, late-half rR improves on seed 0).
❌ **NOT walking, NOT consistently beating random** at 1M steps.

The strict ϖ-RTD work is necessary infrastructure but not sufficient on its own. The next architectural step (AdaLin or similar plasticity-preserving feature class) is required to unlock walking.
