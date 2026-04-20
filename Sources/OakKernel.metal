// =================================================================================================
// OakKernel.metal - Orthodox Step 11 Oak agent. Single-stream (B=1), Metal 4 ICB.
//
// What the Plan demands (Sutton-Bowling-Pilarski 2022 arXiv:2208.11173, Step 11):
//   - Perception                         -> encoder (fixed random projection; generate-and-test)
//   - Value function (one per policy)    -> valueW (main GVF, differential, rho-bar centered)
//   - K subtask GVFs (RR-Subtasks)       -> psi (cumulant=reward, stopping=feature attainment)
//   - K option policies                  -> optPolicy (Design A: K policies blended by chord)
//   - K option models r_hat^k, n_hat^k   -> optReward, optTransition (RR-Subtasks eq. 17)
//   - Planning over option models        -> oak_plan (RR-Subtasks eq. 19, asynchronous VI)
//   - Option keyboard chord lambda       -> oak_meta_policy (lambda = softmax(psi); zero params)
//   - Feedback / utility / remove+replace-> oak_bk_utility + oak_bk_admit_commit + generate-and-test
//   - Average-reward (no discount)       -> rho-bar (Reward Centering 2405.09999)
//   - Per-feature step sizes (Step 1)    -> alpha_0 globally; per-feature differentiation comes
//                                            via generate-and-test reseeding the worst-utility
//                                            feature each step (Plan: "relevant features can be
//                                            given large step-size parameters" - achieved by
//                                            KEEPING relevant features alive, REPLACING the rest)
//
// Kernel-side telemetry kept to the bare minimum: only fields the host literally cannot observe
// (per-phase non-finite counters and cumulative reinit counts). The host's HealthAccumulator
// computes everything else (rewardRate, action delta, scoreGrad L2) directly from env signals.
//
// References: see commit 6d532226a "orthodox step-11 oak" for full citation block.
// =================================================================================================
#include <metal_stdlib>
#include <float.h>
using namespace metal;

constant uint OAK_SIMD = 32u;       // Apple GPU SIMD width (structural).

// Counters: per-step (cleared by host) and cumulative.
constant uint OAK_CNT_OBS_NF      = 0u;
constant uint OAK_CNT_LEARN_NF    = 1u;
constant uint OAK_CNT_PLAN_NF     = 2u;
constant uint OAK_CNT_REINIT_OPT  = 3u;   // cumulative (life of engine).
constant uint OAK_CNT_REINIT_FEAT = 4u;   // cumulative.
constant uint OAK_CNT_TDPSI_NF    = 5u;

struct KernelConfig {
    uint obsDim, rank, optionCount, baseObsDim, actionParamCount;

    // Param offsets: 7 heads, all Plan-required. `phase` is the cos/sin offset for the
    // deep Fourier features encoder (Lewandowski-Schuurmans-Machado 2024,
    // arXiv:2410.20634). Encoder rows now span R/2 pre-activations, each producing a
    // (cos, sin) pair into state[0..R).
    uint pEncoder, pPhase, pValue, pPsi, pOptPolicy, pOptReward, pOptTransition;

    // State arena offsets.
    uint sObservation, sObsMean, sObsSM;
    uint sState;                // 2 * R floats (slot toggle).
    uint sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi;
    uint sChord, sChordPrev;
    uint sUtilFeat, sUtilOpt;
    uint sPlanAdv;
    // MetaOptimize L-approximation (Sharifnassab-Salehkaleybar-Sutton 2024,
    // arXiv:2402.02342). Per-feature step sizes shared across value/psi/optReward/
    // optTransition heads (and relevance-weighted into optPolicy). All structurally
    // derived: α₀ = 1/baseO, η = 1/(R·baseO), γ = 1 − 1/R, α-cap = 1/baseO.
    uint sAlpha, sBeta, sH;
    // Dense option-transition TD targets (RR-Subtasks eq. 17). Written by oak_plan
    // (each thread k writes its slice nTdMat[k*R, (k+1)*R) — no contention),
    // consumed by oak_learn_heads to update the K*R^2 transition weights.
    uint sNTdMat;

    // UInt arena offsets.
    uint uCounters;

    // No tunable scalars. alpha is implicit (Kim arXiv:2510.06149) per-head from instantaneous
    // signal magnitudes; ridges are unit-Bayesian (var + 1, not var + tunable_epsilon).
};

struct KernelScalars {
    uint  stepCount;
    uint  stateSlot;
    float rho;                  // rho-bar (Reward Centering).
    float alphaPi;              // implicit alpha for policy weights.
};

// Telemetry the host literally cannot derive from env signals: per-phase non-finite counters
// (kernel-internal numerical health) and cumulative reinit counts (Plan Step 11 evidence).
struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
};

// safe_w: NF-guarded weight write. Preserves prev value on non-finite — no zero-clamp, no
// magnitude cap. Weights grow as the agent's slice of reality demands (BWH).
inline float safe_w(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }

// Tracking-rate EMA. 1/(t+1) is the optimal stationary estimator; for non-stationary tracking
// (BWH) we floor at 1/baseO so EMAs of input statistics keep tracking the world's actual
// signal magnitudes. baseO is the env's input dimensionality - the only structural quantity
// that comes from the environment, not from the agent's design choices.
inline float track_rate(uint step, uint baseO) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(baseO, 1u)));
}

// splitmix32 hash for deterministic RNG.
inline float hash_uniform(uint a, uint b, uint c) {
    uint z = a ^ (b * 0x7feb352du) ^ (c * 0x846ca68bu);
    z = (z ^ (z >> 16)) * 0x9e3779b9u;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z ^= z >> 16;
    return float(z >> 8) * 5.9604645e-8f;
}

// Online normalization (Sutton-Barto Ch. 9.5.4). Variance regularizer = 1/baseO: signal
// magnitudes much smaller than the reciprocal of input dim are treated as noise and not
// amplified. baseO is the env's input dimensionality, the only structural quantity from
// the env (not from the agent's design choices).
inline float standardized_obs(float x, float mean, float sm, uint baseO) {
    float var = max(sm - mean * mean, 0.0f);
    return (x - mean) * rsqrt(var + 1.0f / float(max(baseO, 1u)));
}


struct Ctx {
    uint O, R, K, baseO, D;
    uint stepCount, stateSlot;
    float rho, alphaPi;

    device float *encoder, *phase, *valueW, *psi, *optPolicy, *optReward, *optTransition;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *scoreGrad;
    device float *reward, *td, *tdPsi;
    device float *chord, *chordPrev;
    device float *utilFeat, *utilOpt;
    device float *planAdv;
    device float *alpha, *beta, *hMeta;        // MetaOptimize L-approximation per-feature buffers.
    device float *nTdMat;                      // Dense optTransition TD targets, K*R floats.
    device uint  *counters;
    device KernelScalars *scalars;
    device KernelStats   *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount;
    c.baseO = cfg.baseObsDim; c.D = cfg.actionParamCount;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho; c.alphaPi = scalars->alphaPi;

    c.encoder       = params + cfg.pEncoder;
    c.phase         = params + cfg.pPhase;
    c.valueW        = params + cfg.pValue;
    c.psi           = params + cfg.pPsi;
    c.optPolicy     = params + cfg.pOptPolicy;
    c.optReward     = params + cfg.pOptReward;
    c.optTransition = params + cfg.pOptTransition;

    c.observation     = stateArena + cfg.sObservation;
    c.observationMean = stateArena + cfg.sObsMean;
    c.obsSM           = stateArena + cfg.sObsSM;
    c.state           = stateArena + cfg.sState;
    c.policyMean      = stateArena + cfg.sPolicyMean;
    c.scoreGrad       = stateArena + cfg.sScoreGrad;
    c.reward          = stateArena + cfg.sReward;
    c.td              = stateArena + cfg.sTd;
    c.tdPsi           = stateArena + cfg.sTdPsi;
    c.chord           = stateArena + cfg.sChord;
    c.chordPrev       = stateArena + cfg.sChordPrev;
    c.utilFeat        = stateArena + cfg.sUtilFeat;
    c.utilOpt         = stateArena + cfg.sUtilOpt;
    c.planAdv         = stateArena + cfg.sPlanAdv;
    c.alpha           = stateArena + cfg.sAlpha;
    c.beta            = stateArena + cfg.sBeta;
    c.hMeta           = stateArena + cfg.sH;
    c.nTdMat          = stateArena + cfg.sNTdMat;

    c.counters = uintArena + cfg.uCounters;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

// =================================================================================================
// 1. oak_bk_obs - per-feature observation Welford EMA. First so encode reads current stats.
// =================================================================================================
kernel void oak_bk_obs(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgW [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    float invT = track_rate(ctx.stepCount, ctx.baseO);
    uint nfLocal = 0u;
    for (uint o = tid; o < ctx.O; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nfLocal++; continue; }
        ctx.observationMean[o] += invT * (x - ctx.observationMean[o]);
        ctx.obsSM[o]           += invT * (x * x - ctx.obsSM[o]);
    }
    uint nfTotal = simd_sum(nfLocal);
    if (tid == 0u) ctx.counters[OAK_CNT_OBS_NF] = nfTotal;
}

// =================================================================================================
// 2. oak_encode - perception. Deep Fourier features (Lewandowski-Schuurmans-Machado 2024,
//    arXiv:2410.20634): R/2 pre-activations, each producing a paired (cos, sin) channel. The
//    paper proves linear function approximation does not lose plasticity and shows that
//    Fourier features inherit this property by approximating linearity dynamically while
//    remaining nonlinear in representation. cos/sin are bounded in [-1, 1] (structural scale
//    prior on the BASIS), but the linear weights on top still grow without cap (BWH).
//
//      state[feat in [0, R/2)] = cos(W_enc[feat] . std_obs + phase[feat])
//      state[feat in [R/2, R)] = sin(W_enc[feat-R/2] . std_obs + phase[feat-R/2])
//
//    Encoder + phase evolve only via generate-and-test in bk_admit_commit (Plan Step 2,
//    "smart generation of promising features and then smart testing to rank and replace").
// =================================================================================================
kernel void oak_encode(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint feat [[thread_position_in_grid]])
{
    if (feat >= cfg.rank) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    uint halfR = ctx.R >> 1u;                  // ctx.R is even (host-enforced).
    uint pre   = (feat < halfR) ? feat : feat - halfR;
    uint encRow = pre * ctx.baseO;
    float z = ctx.phase[pre];
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = standardized_obs(ctx.observation[o], ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        z = fma(ctx.encoder[encRow + o], c, z);
    }
    float v = (feat < halfR) ? cos(z) : sin(z);
    // safe_w-style preservation: on non-finite, keep the previous slot's value for this feature.
    float prev = state_at(ctx, ctx.stateSlot)[feat];
    state_at(ctx, 1u - ctx.stateSlot)[feat] = isfinite(v) ? v : prev;
}

// =================================================================================================
// 3. oak_meta - chord softmax (Plan option keyboard, Design A) + alphaPi (Kim arXiv:2510.06149)
//    + rho-bar EMA (Reward Centering arXiv:2405.09999). Three scalar reductions, one kernel,
//    single-threaded. Runs BEFORE oak_predict so the new chord drives this step's psi TD.
// =================================================================================================
kernel void oak_meta(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    for (uint k = 0u; k < ctx.K; ++k) ctx.chordPrev[k] = ctx.chord[k];

    // chord = softmax(psi_k(s')).
    float maxL = -FLT_MAX;
    float logits[OAK_SIMD];
    for (uint k = 0u; k < ctx.K; ++k) {
        float l = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) l = fma(ctx.psi[k * ctx.R + i], curr[i], l);
        logits[k] = l;
        maxL = max(maxL, l);
    }
    float Z = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) Z += exp(logits[k] - maxL);
    float invZ = 1.0f / max(Z, FLT_MIN);
    for (uint k = 0u; k < ctx.K; ++k) ctx.chord[k] = exp(logits[k] - maxL) * invZ;

    // alphaPi = 1 / (baseO + ||state||^2 * ||scoreGrad||^2). The baseO floor (env input dim) gives
    // an upper bound 1/baseO at small inputs; Kim's denominator self-bounds at large signals.
    float ss = 0.0f, gg = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) ss = fma(curr[i], curr[i], ss);
    for (uint p = 0u; p < ctx.D; ++p) gg = fma(ctx.scoreGrad[p], ctx.scoreGrad[p], gg);
    ctx.scalars->alphaPi = 1.0f / (float(max(ctx.baseO, 1u)) + ss * gg);

    // rho-bar EMA (Reward Centering). One step of lag from predict's reading is negligible.
    float invT = track_rate(ctx.stepCount, ctx.baseO);
    ctx.scalars->rho += invT * (ctx.reward[0] - ctx.rho);
}

// =================================================================================================
// 4. oak_predict - Horde-style multi-GVF update.
//      Main TD:    delta   = (r - rho_bar) + V(s') - V(s)        (differential TD, ρ̄ centered)
//      Subtask k:  delta_k = (r - rho_bar) + curr[k] - psi_k(s)  (β=1, 1-step myopic regression
//                                                                  of next reward + anchor feature
//                                                                  attainment onto state).
//    sf[k] = k identity. β=1 drops the psi(s') bootstrap term: ψ_k becomes a 1-step prediction.
//    The Plan doesn't prescribe a specific β; β=1 is the simplest GVF and matches the 1-step
//    horizon of our option models.
// =================================================================================================
kernel void oak_predict(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float vPrev = 0.0f, vNext = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) {
        vPrev = fma(ctx.valueW[i], prev[i], vPrev);
        vNext = fma(ctx.valueW[i], curr[i], vNext);
    }
    // Main TD: differential, rho-bar centered (Reward Centering arXiv:2405.09999).
    float reward_c = ctx.reward[0] - ctx.rho;
    float delta = reward_c + vNext - vPrev;
    if (!isfinite(delta)) { ctx.counters[OAK_CNT_LEARN_NF] += 1u; delta = 0.0f; }
    ctx.td[0] = delta;

    // Subtask TDs (RR-Subtasks ψ-GVF with PER-OPTION ADAPTIVE β_k):
    //   β_k = β_cap · α₀ / (α₀ + utilOpt[k])    where β_cap = 1 − 1/R, α₀ = 1/baseO.
    // utilOpt[k] is the EMA of |tdPsi[k]|² maintained in oak_post_learn (Kim implicit-step
    // family; arXiv:2510.06149-style self-bound on the bootstrap). Behavior:
    //   * |δ_k|² small (option's ψ well-tracked)        → β_k → β_cap (long-horizon bootstrap)
    //   * |δ_k|² large (divergence threat / mis-tracking) → β_k → 0 (auto-revert to myopic)
    // utilOpt[k] is initialized to R (capacity-volume prior) at host startup so β_k starts
    // at β_cap·α₀/(α₀+R) ≈ 0 — fully myopic and safe at random init. This is the negative
    // feedback loop missing from the constant β = 1 − 1/R attempt that diverged 6/10 seeds:
    // any sustained spike in |δ_k|² shrinks β_k automatically, choking the bootstrap.
    // ρ̄ centering keeps tdPsi mean-zero so the psi-weight gradient (tdPsi * prev[i]) doesn't
    // accumulate a state-correlated bias. K = min(R, OAK_SIMD) so curr[k] is always valid.
    float beta_cap = 1.0f - 1.0f / float(max(ctx.R, 1u));
    float a0       = 1.0f / float(max(ctx.baseO, 1u));
    uint nfPsi = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        float beta_k = beta_cap * a0 / (a0 + ctx.utilOpt[k]);
        float psiPrev = 0.0f, psiNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.psi[k * ctx.R + i];
            psiPrev = fma(w, prev[i], psiPrev);
            psiNext = fma(w, curr[i], psiNext);
        }
        float dk = reward_c + (1.0f - beta_k) * curr[k] + beta_k * psiNext - psiPrev;
        if (!isfinite(dk)) { dk = 0.0f; nfPsi++; }
        ctx.tdPsi[k] = dk;
    }
    if (nfPsi > 0u) ctx.counters[OAK_CNT_TDPSI_NF] += nfPsi;
}

// =================================================================================================
// 5. oak_plan - parallel asynchronous VI over DENSE option models (RR-Subtasks eq. 17 + 19).
//   Per-thread k:
//     Q^k(s) = r_hat^k(s) + V(n_hat^k(s)) where n_hat^k_j(s) = sum_i w^k_{j,i} * s[i] (dense)
//     nTdMat[k*R+j] = curr[j] - n_hat^k_j(prev)  (precomputed for oak_learn_heads)
//   Across-option reduction (K = 32 ≤ Apple SIMD width = 1 simdgroup):
//     bestQ = simd_max(Q);  planAdv = bestQ - V(s) (raw, no scale prior).
//   Single thread (k=0) writes planAdv. simd_or aggregates the per-thread non-finite flag so
//   the counter increment is contention-free without atomics.
// =================================================================================================
kernel void oak_plan(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint k [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    bool active = (k < ctx.K);
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    // V(s) — every thread recomputes; redundant but cheap (R FMAs).
    float vCurr = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) vCurr = fma(ctx.valueW[i], curr[i], vCurr);

    float Q = -FLT_MAX;
    bool nfQ = false;
    if (active) {
        // r_hat^k(s) = sum_i optReward[k][i] * curr[i].
        float rHat = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);
        // Dense transition: V(n_hat^k(s)) = sum_j valueW[j] * (sum_i w^k_{j,i} * curr[i]).
        // Same inner loop also computes nTdMat[k*R+j] = curr[j] - n_hat^k_j(prev) for
        // oak_learn_heads, so we read each weight twice (once with curr, once with prev) but
        // touch the dense matrix only once per oak_plan call.
        float vHat = 0.0f;
        uint base = k * ctx.R * ctx.R;
        for (uint j = 0u; j < ctx.R; ++j) {
            float nHatCurr = 0.0f, nHatPrev = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) {
                float w = ctx.optTransition[base + j * ctx.R + i];
                nHatCurr = fma(w, curr[i], nHatCurr);
                nHatPrev = fma(w, prev[i], nHatPrev);
            }
            vHat = fma(ctx.valueW[j], nHatCurr, vHat);
            ctx.nTdMat[k * ctx.R + j] = curr[j] - nHatPrev;
        }
        Q = rHat + vHat;
        if (!isfinite(Q)) { nfQ = true; Q = -FLT_MAX; }
    }

    // simd-wide reduction across the K threads (K = 32 fits one simdgroup).
    // simd_or expects an integral type, so route the bool through uint.
    float bestQ = simd_max(Q);
    uint  nfBits = simd_or(nfQ ? 1u : 0u);

    if (k == 0u) {
        float advRaw = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
        // Self-bound by x/(1+|x|): the standard rational soft-clamp from Sutton's gradient-TD
        // line of work. Image is (-1, 1), continuous and monotone in advRaw, no tunable scale.
        // Plan-orthodox: it's the negative-feedback brake the dense optTransition (B1) feedback
        // loop needed — when an option model briefly diverges, the planAdv contribution to
        // valueW updates can't run away (was the cause of the 3/10 seeds NF-blowup at 10M).
        float adv = advRaw / (1.0f + fabs(advRaw));
        if (nfBits != 0u) ctx.counters[OAK_CNT_PLAN_NF] += 1u;
        ctx.planAdv[0] = isfinite(adv) ? adv : ctx.planAdv[0];   // safe_w-style preserve on NF.
    }
}

// =================================================================================================
// 6. oak_learn_heads - 1-step TD updates on 5 learnable heads.
//      0: valueW[R]              -> alpha[i] * (td + planAdv) * prev[i]   (planAdv added HERE ONLY)
//      1: psi[K*R]               -> alpha[i] * tdPsi[k] * prev[i]
//      2: optPolicy[K*D*R]       -> alphaPi * relevance(i) * td * scoreGrad[d] * prev[i] * chordPrev[k]
//      3: optReward[K*R]         -> alpha[i] * (reward - r_hat^k(prev)) * prev[i] * chordPrev[k]
//      4: optTransition[K*R*R]   -> alpha[i] * nTdMat[k*R+j] * prev[i] * chordPrev[k]
//                                   (DENSE: n_hat^k_j(s) = sum_i w^k_{j,i} * s[i], RR-Subtasks eq. 17;
//                                    n_td precomputed by oak_plan into nTdMat[k*R+j] = curr[j] - n_hat^k_j(prev))
// =================================================================================================
kernel void oak_learn_heads(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    uint nVal = ctx.R;
    uint nPsi = ctx.K * ctx.R;
    uint nPi  = ctx.K * ctx.D * ctx.R;
    uint nOR  = ctx.K * ctx.R;
    uint nOT  = ctx.K * ctx.R * ctx.R;   // dense: K*R^2 (RR-Subtasks eq. 17 full form).
    if (gid >= nVal + nPsi + nPi + nOR + nOT) return;

    // Head 4 (dense optTransition) reads its TD targets from ctx.nTdMat (precomputed in
    // oak_plan), so we no longer need `curr` in this kernel. `prev` still drives the gradient
    // direction for value/psi/optReward/optTransition heads (and the policy relevance gate).
    device const float *prev = state_at(ctx, ctx.stateSlot);
    float td = ctx.td[0];
    float deltaValue = td + ctx.planAdv[0];
    // Per-feature step size α[i] from MetaOptimize L-approximation
    // (Sharifnassab-Salehkaleybar-Sutton 2024, arXiv:2402.02342). Initialized at 1/baseO,
    // capped from above at 1/baseO, adapted by the value-TD credit signal in oak_post_learn.
    // Replaces the previous global `a = 1/baseO`. The optPolicy head keeps Kim's implicit
    // alphaPi but is relevance-weighted by α[i]·baseO ∈ [0, 1] so noisy-feature columns
    // stop pulling on the policy gradient (kills the action chatter we saw with global α).

    if (gid < nVal) {                                          // Head 0: valueW
        uint i = gid;
        float w = ctx.valueW[i];
        ctx.valueW[i] = safe_w(w, w + ctx.alpha[i] * deltaValue * prev[i]);
        return;
    }
    if (gid < nVal + nPsi) {                                   // Head 1: psi
        uint id = gid - nVal;
        uint k = id / ctx.R, i = id % ctx.R;
        float w = ctx.psi[id];
        ctx.psi[id] = safe_w(w, w + ctx.alpha[i] * ctx.tdPsi[k] * prev[i]);
        return;
    }
    if (gid < nVal + nPsi + nPi) {                             // Head 2: optPolicy (REINFORCE)
        uint id = gid - nVal - nPsi;
        uint k = id / (ctx.D * ctx.R);
        uint rem = id - k * ctx.D * ctx.R;
        uint d = rem / ctx.R, i = rem % ctx.R;
        float relevance = ctx.alpha[i] * float(max(ctx.baseO, 1u));   // ∈ [0, 1].
        float grad = ctx.chordPrev[k] * ctx.scoreGrad[d] * prev[i];
        float w = ctx.optPolicy[id];
        ctx.optPolicy[id] = safe_w(w, w + ctx.alphaPi * relevance * td * grad);
        return;
    }
    if (gid < nVal + nPsi + nPi + nOR) {                       // Head 3: optReward (eq. 17, beta=1)
        uint id = gid - nVal - nPsi - nPi;
        uint k = id / ctx.R, i = id % ctx.R;
        float predPrev = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            predPrev = fma(ctx.optReward[k * ctx.R + j], prev[j], predPrev);
        float r_td = ctx.reward[0] - predPrev;
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optReward[id];
        ctx.optReward[id] = safe_w(w, w + ctx.alpha[i] * r_td * grad);
        return;
    }
    {   // Head 4: dense optTransition (RR-Subtasks eq. 17). Per-weight gradient:
        //   ∂ ½(curr[j] − n_hat^k_j(prev))² / ∂ w^k_{j,i} = -(curr[j] − n_hat^k_j(prev)) · prev[i]
        // n_td precomputed into ctx.nTdMat[k*R + j] by oak_plan (one shared inner sum per (k, j)
        // serves R weights along the i axis). chordPrev[k] is the on-chord weight (importance
        // ratio per RR-Subtasks eq. 17). α[i] (not α[j]) because the gradient direction involves
        // prev[i] — the i axis is the input dimension whose per-feature relevance gates the update.
        uint id  = gid - nVal - nPsi - nPi - nOR;
        uint k   = id / (ctx.R * ctx.R);
        uint rem = id - k * ctx.R * ctx.R;
        uint j   = rem / ctx.R;
        uint i   = rem % ctx.R;
        float n_td = ctx.nTdMat[k * ctx.R + j];
        float grad = ctx.chordPrev[k] * prev[i];
        float w    = ctx.optTransition[id];
        ctx.optTransition[id] = safe_w(w, w + ctx.alpha[i] * n_td * grad);
    }
}

// =================================================================================================
// 8. oak_post_learn - three concurrent passes after learn_heads writes the new weights:
//      gid in [0, R)        -> per-feature GraMa utility EMA (arXiv:2505.24061)
//      gid in [R, R+K)      -> per-option |tdPsi|^2 utility EMA
//      gid in [R+K, R+K+D)  -> action mu_p = sum_k chord[k] * (W_pi[k][p] . s')  (meta_action)
//    All three are single-output-per-thread; folding them into one kernel saves a launch.
// =================================================================================================
kernel void oak_post_learn(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    if (gid >= ctx.R + ctx.K + ctx.D) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float invT = track_rate(ctx.stepCount, ctx.baseO);
    if (gid < ctx.R) {
        uint i = gid;
        float xi = prev[i];
        float td = ctx.td[0];
        // GraMa per-feature gradient utility (arXiv:2505.24061). Sum, no normalization:
        // ranking is scale-invariant so dividing by (1+K) is purely cosmetic.
        float gSum = fabs(td * xi);
        for (uint k = 0u; k < ctx.K; ++k) gSum += fabs(ctx.tdPsi[k] * xi);
        ctx.utilFeat[i] += invT * (gSum - ctx.utilFeat[i]);

        // MetaOptimize L-approximation (Sharifnassab-Salehkaleybar-Sutton 2024,
        // arXiv:2402.02342, Algorithm 3, Hessian-free SGD-base + SGD-meta, no momentum, no
        // weight decay). Per-feature step size driven by the value-TD credit signal δ.
        // Plan-orthodox structural constants:
        //   γ = 1 − 1/R         (capacity-volume non-stationarity discount)
        //   η = 1/(R · baseO)   (capacity-relative meta-step per input dim)
        //   α-cap = 1/baseO     (preserves the prior safe-update ceiling)
        //
        // Order matches Algorithm 1: meta-update uses CURRENT h (not h_{t+1}); h is then
        // updated to γ·h + Δw where Δw = α·δ·prev[i] is the base update applied this step
        // by oak_learn_heads to the value head.
        float aOld   = ctx.alpha[i];
        float hOld   = ctx.hMeta[i];
        float dw     = aOld * td * xi;
        float invR   = 1.0f / float(max(ctx.R, 1u));
        float gamma  = 1.0f - invR;
        float eta    = invR / float(max(ctx.baseO, 1u));
        // α-cap = 1/baseO (safe-update ceiling) ⇔ β-cap = −log baseO.
        // α-floor = 1/(R·baseO) (always-learning guarantee: BWH/temporal-uniformity says
        // OaK never stops adapting any feature; even after a streak of inconsistent gradients
        // the per-feature step floors at 1/(R·baseO) so the soft mechanism keeps tracking
        // alongside the hard G&T mechanism). At the floor, summed across R features the
        // total per-step learning rate equals the original global a = 1/baseO; at the cap,
        // it can amplify up to R · 1/baseO when every feature is fully informative.
        float betaCap   = -log(float(max(ctx.baseO, 1u)));
        float betaFloor = -log(float(max(ctx.R * ctx.baseO, 1u)));
        float newBeta = ctx.beta[i] + eta * td * xi * hOld;
        if (!isfinite(newBeta)) newBeta = ctx.beta[i];
        newBeta = clamp(newBeta, betaFloor, betaCap);
        ctx.hMeta[i] = gamma * hOld + dw;
        ctx.beta[i]  = newBeta;
        ctx.alpha[i] = exp(newBeta);
        return;
    }
    if (gid < ctx.R + ctx.K) {
        uint k = gid - ctx.R;
        float dk = ctx.tdPsi[k];
        ctx.utilOpt[k] += invT * (dk * dk - ctx.utilOpt[k]);
        return;
    }
    // Action head: mu_p = sum_k chord[k] * (W_pi[k][p] . s').
    uint p = gid - ctx.R - ctx.K;
    float mu = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float dot = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            dot = fma(ctx.optPolicy[(k * ctx.D + p) * ctx.R + i], curr[i], dot);
        mu = fma(ctx.chord[k], dot, mu);
    }
    float prevMu = ctx.policyMean[p];
    ctx.policyMean[p] = isfinite(mu) ? mu : prevMu;
}

// =================================================================================================
// 10. oak_bk_admit_commit - Oak feedback: generate-and-test reinit of worst-utility option AND
//     worst-utility feature each step (Plan Step 11 remove-replace + Step 2 generate-and-test).
//     Single-threaded.
// =================================================================================================
kernel void oak_bk_admit_commit(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);

    // Worst option: lowest utilOpt. Reseed psi[k][:R], optPolicy[k][:D*R], optReward[k][:R],
    // optTransition[k][:R] (DIAGONAL) with hash-based Xavier-scaled small projections.
    float invSqrtR = rsqrt(float(max(ctx.R, 1u)));
    uint worstK = 0u; float worstUO = FLT_MAX;
    for (uint k = 0u; k < ctx.K; ++k)
        if (ctx.utilOpt[k] < worstUO) { worstUO = ctx.utilOpt[k]; worstK = k; }
    {
        uint k = worstK;
        for (uint i = 0u; i < ctx.R; ++i) {
            ctx.psi[k * ctx.R + i] =
                (hash_uniform(ctx.stepCount, k, i) * 2.0f - 1.0f) * invSqrtR;
            ctx.optReward[k * ctx.R + i] =
                (hash_uniform(ctx.stepCount, k, ctx.R + i) * 2.0f - 1.0f) * invSqrtR;
        }
        for (uint pi = 0u; pi < ctx.D * ctx.R; ++pi) {
            ctx.optPolicy[k * ctx.D * ctx.R + pi] =
                (hash_uniform(ctx.stepCount, k, 3u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        }
        // Dense optTransition (RR-Subtasks eq. 17): reseed all K*R^2 weights of option k's
        // transition slice with the same Xavier 1/sqrt(R) scale.
        for (uint pi = 0u; pi < ctx.R * ctx.R; ++pi) {
            ctx.optTransition[k * ctx.R * ctx.R + pi] =
                (hash_uniform(ctx.stepCount, k, 4u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        }
        ctx.utilOpt[k] = 0.0f;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
    }

    // Worst feature: lowest utilFeat → reseed the corresponding Fourier pre-activation.
    // Each pre-activation `pre` produces a paired (cos, sin) into state[pre] and state[pre + R/2],
    // so reseeding pre touches both features in the pair. Reseed the encoder row at the RFF
    // bandwidth-1 scale (W ~ U[-1, 1], matching Rahimi-Recht 2007 §3 for unit-variance inputs)
    // AND the phase scalar (uniform on [0, 2π)) — both structurally derived from the basis,
    // no learnable scale on either; head weights live on top via safe_w.
    uint halfR = ctx.R >> 1u;
    uint worstI = 0u; float worstUF = FLT_MAX;
    for (uint i = 0u; i < ctx.R; ++i)
        if (ctx.utilFeat[i] < worstUF) { worstUF = ctx.utilFeat[i]; worstI = i; }
    {
        uint pre = (worstI < halfR) ? worstI : worstI - halfR;
        for (uint o = 0u; o < ctx.baseO; ++o) {
            float u = hash_uniform(ctx.stepCount, pre, o + 0xdeadbeefu) * 2.0f - 1.0f;
            ctx.encoder[pre * ctx.baseO + o] = u;        // RFF bandwidth-1: W ~ U[-1, 1].
        }
        float u = hash_uniform(ctx.stepCount, pre, 0xfeedf00du);
        ctx.phase[pre] = u * 6.283185307179586f;         // 2π, structural (no tunable).
        ctx.utilFeat[pre]         = 0.0f;
        ctx.utilFeat[pre + halfR] = 0.0f;
        // MetaOptimize state reset: both pair members restart at α = 1/baseO with no
        // accumulated meta-gradient history (the encoder row + phase have changed, so any
        // h or β earned by the prior basis no longer applies).
        float betaInit = -log(float(max(ctx.baseO, 1u)));
        float aInit    = exp(betaInit);
        ctx.beta[pre]          = betaInit;
        ctx.beta[pre + halfR]  = betaInit;
        ctx.alpha[pre]         = aInit;
        ctx.alpha[pre + halfR] = aInit;
        ctx.hMeta[pre]         = 0.0f;
        ctx.hMeta[pre + halfR] = 0.0f;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
    }

    // Surface the only telemetry the host literally cannot derive from env signals.
    ctx.stats->encodeNonFiniteCount    = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount     = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount      = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount     = float(ctx.counters[OAK_CNT_TDPSI_NF]);
    ctx.stats->oakOptionReinitCount    = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount   = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

    ctx.scalars->stateSlot = 1u - ctx.stateSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}
