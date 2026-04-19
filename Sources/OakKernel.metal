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
//                                            from utilFeat-driven feature reinit, not from per-
//                                            feature alpha (Plan: "relevant features can be given
//                                            large step-size parameters" - achieved by KEEPING
//                                            relevant features alive and reinitting irrelevant ones)
//
// What the Plan does NOT demand (and is NOT here):
//   - Eligibility traces (Plan: "such as" - example, not required; traces + bootstrap + linear FA
//     is the canonical Baird divergence config; Sutton-Barto Ch. 6 1-step TD is provably stable)
//   - Encoder gradient learning (Plan: per-feature step sizes ARE representation learning)
//   - Learned termination weights betaW (no Bacon-Harb-Precup head; structural beta = alpha_0)
//   - Learned chord weights omegaW (chord = softmax(psi); zero params)
//   - Per-feature alpha tracking (featSM saturates uniformly under unit_soft state - bureaucracy
//     without differentiation; use global alpha_0)
//   - Bonus weight high-water-mark wBar (RR-Subtasks bonus = feature attainment directly)
//   - subtaskFeatureOf reranking (sf[k] = k identity; generate-and-test handles feature evolution)
//   - utilPlan / optFracEMA / pEff / activeMask (telemetry/admission gold-plating; not Plan)
//   - vMeanBaseline / psiMeanBaseline ϖ-RTD centering (cancels algebraically in the TD residual)
//   - Layer/batch normalization, discounting, episodic resets, magnitude caps
//
// References:
//   - RR-Subtasks (arXiv:2202.03466): subtask GVFs, expectation models eq. 17, planning eq. 19
//   - Reward Centering (arXiv:2405.09999): rho-bar tracking
//   - Implicit alpha (arXiv:2510.06149): alpha = alpha_0/(1+alpha_0*||x||^2) self-bounding
//   - GraMa (arXiv:2505.24061): per-feature gradient utility
//   - Generate-and-test (Mahmood-Sutton-Pilarski 2012, Rahman 2020, Samani 2022)
//   - Sutton-Barto Ch. 9.5.4 (online normalization), Ch. 6 (1-step TD stability)
//   - Big World Hypothesis (.cursor/rules/alberta_plan.mdc)
//
// Single-stream: one observation -> one action -> one reward per step. No batch dimension.
// Multi-env benchmarking lives in the Swift host as N independent agent instances.
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

// =================================================================================================
// Configuration mirrored from Swift KernelConfig.
// =================================================================================================
struct KernelConfig {
    uint obsDim, rank, actDim, optionCount, baseObsDim;
    uint actionParamCount;                  // dPolicy from the env wrapper (Inv-1).

    // Param offsets: 6 heads (all Plan-required).
    uint pEncoder, pValue, pPsi, pOptPolicy, pOptReward, pOptTransition;

    // State arena offsets.
    uint sObservation, sObsMean, sObsSM;
    uint sState;                             // 2 * R floats (slot toggle).
    uint sPrevPolicyMean, sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi, sVCurr;
    uint sChord, sChordPrev;
    uint sFeatMean;
    uint sUtilFeat, sUtilOpt;
    uint sPlanAdv;

    // UInt arena offsets.
    uint uCounters;

    // Structural meta-learning constants (derived from cap = R).
    float cap;          // cap = R.
    float alpha0;       // alpha0 = 1/cap.
    float ridge;        // ridge = 1/(cap+1) for safe-divide / track-rate.
};

// =================================================================================================
// Per-step scalars.
// =================================================================================================
struct KernelScalars {
    uint  stepCount;
    uint  stateSlot;
    float rho;                  // rho-bar (Reward Centering).
    float alphaPi;              // implicit alpha for policy weights.
};

// =================================================================================================
// Per-step telemetry (mirrors Swift KernelStats).
// =================================================================================================
struct KernelStats {
    float rewardRate;
    float latentStateMeanSquare;
    float scoreGradL2Mean;
    float optionModelPredictionMSE;
    float utilityFeatureMean, utilityOptionMean;
    float oakOptionReinitCount, oakFeatureReinitCount;
    float maxAbsLatentPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean;
    float rawNonFiniteCount, encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float tdPsiNonFiniteCount;
    float valueWMax, psiMax, optPolicyMax, optRewardMax, optTransitionMax;
    float alphaPiStat;
};

// =================================================================================================
// Numerical safety - parameter-free.
//   safe_w(old, new): NF-guarded weight write that PRESERVES on non-finite (no zero-clamp,
//                     no magnitude cap; weights grow as the agent's slice of reality demands).
//   unit_soft(x):     structural sigmoid in (-1, 1) for forward outputs only (state, mu, planAdv).
// =================================================================================================
inline float safe_w(float old_w, float new_w) {
    return isfinite(new_w) ? new_w : old_w;
}

inline float unit_soft(float x) {
    return isfinite(x) ? tanh(x) : 0.0f;
}

// Tracking-rate EMA (Sutton-Barto §2.6), with ridge floor for early steps.
inline float track_rate(uint step, float ridge) {
    return max(1.0f / (float(step) + 1.0f), ridge);
}

// splitmix32 hash for deterministic RNG (no per-step host alloc).
inline float hash_uniform(uint a, uint b, uint c) {
    uint z = a ^ (b * 0x7feb352du) ^ (c * 0x846ca68bu);
    z = (z ^ (z >> 16)) * 0x9e3779b9u;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z ^= z >> 16;
    return float(z >> 8) * 5.9604645e-8f;
}

// Standardized observation (Sutton-Barto Ch. 9.5.4 online normalization).
inline float standardized_obs(float x, float mean, float sm, float ridge) {
    float var = max(sm - mean * mean, 0.0f);
    return (x - mean) * rsqrt(var + ridge);
}

// =================================================================================================
// Per-step computation context. Built once per kernel invocation.
// =================================================================================================
struct Ctx {
    uint O, R, K, A, baseO, D;
    uint stepCount, stateSlot;
    float cap, alpha0, ridge, rho, alphaPi;

    device float *encoder, *valueW, *psi, *optPolicy, *optReward, *optTransition;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *prevPolicyMean, *policyMean, *scoreGrad;
    device float *reward, *td, *tdPsi, *vCurr;
    device float *chord, *chordPrev;
    device float *featMean;
    device float *utilFeat, *utilOpt;
    device float *planAdv;
    device uint  *counters;
    device KernelScalars *scalars;
    device KernelStats   *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount; c.A = cfg.actDim;
    c.baseO = cfg.baseObsDim; c.D = cfg.actionParamCount;
    c.stepCount = scalars->stepCount;
    c.stateSlot = scalars->stateSlot;
    c.cap = cfg.cap; c.alpha0 = cfg.alpha0; c.ridge = cfg.ridge;
    c.rho = scalars->rho; c.alphaPi = scalars->alphaPi;

    c.encoder       = params + cfg.pEncoder;
    c.valueW        = params + cfg.pValue;
    c.psi           = params + cfg.pPsi;
    c.optPolicy     = params + cfg.pOptPolicy;
    c.optReward     = params + cfg.pOptReward;
    c.optTransition = params + cfg.pOptTransition;

    c.observation     = stateArena + cfg.sObservation;
    c.observationMean = stateArena + cfg.sObsMean;
    c.obsSM           = stateArena + cfg.sObsSM;
    c.state           = stateArena + cfg.sState;
    c.prevPolicyMean  = stateArena + cfg.sPrevPolicyMean;
    c.policyMean      = stateArena + cfg.sPolicyMean;
    c.scoreGrad       = stateArena + cfg.sScoreGrad;
    c.reward          = stateArena + cfg.sReward;
    c.td              = stateArena + cfg.sTd;
    c.tdPsi           = stateArena + cfg.sTdPsi;
    c.vCurr           = stateArena + cfg.sVCurr;
    c.chord           = stateArena + cfg.sChord;
    c.chordPrev       = stateArena + cfg.sChordPrev;
    c.featMean        = stateArena + cfg.sFeatMean;
    c.utilFeat        = stateArena + cfg.sUtilFeat;
    c.utilOpt         = stateArena + cfg.sUtilOpt;
    c.planAdv         = stateArena + cfg.sPlanAdv;

    c.counters = uintArena + cfg.uCounters;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

// Threadgroup reduction primitives.
inline float tg_sum_128(threadgroup float *buf, uint tid, uint sgid, float v) {
    v = simd_sum(v);
    if (simd_is_first()) buf[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float s = (tid < 4u) ? buf[tid] : 0.0f;
        s = simd_sum(s);
        if (tid == 0u) buf[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// =================================================================================================
// 1. oak_bk_obs - per-feature observation Welford EMA. Updated FIRST so encode reads current stats.
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
    float invT = track_rate(ctx.stepCount, ctx.ridge);
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
// 2. oak_encode - perception: state[r] = unit_soft(W_enc[r] . standardized_obs). Encoder is a
//    fixed random projection at init; evolves only via generate-and-test in bk_admit_commit
//    (Plan Step 2). One thread per latent dim.
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
    uint encRow = feat * ctx.baseO;
    float z = 0.0f;
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = standardized_obs(ctx.observation[o], ctx.observationMean[o],
                                   ctx.obsSM[o], ctx.ridge);
        z = fma(ctx.encoder[encRow + o], c, z);
    }
    state_at(ctx, 1u - ctx.stateSlot)[feat] = unit_soft(z);
    if (feat == 0u) ctx.planAdv[0] = 0.0f;       // reset planAdv (one writer per step).
}

// =================================================================================================
// 3. oak_meta_policy - parameter-free option keyboard. chord[k] = softmax(psi_k(s')) where
//    psi_k(s) = sum_i psi[k][i] * s[i] (function of state ALONE; chord_prev not needed in psi
//    per Plan Design A: chord blends component options, model treats chord as opaque).
// =================================================================================================
kernel void oak_meta_policy(
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
}

// =================================================================================================
// 4. oak_predict - Horde-style multi-GVF update.
//      Main TD:    delta = (r - rho_bar) + V(s') - V(s)                    (differential TD)
//      Subtask k:  delta_k = (r - rho_bar) + alpha_0 * curr[k] + (1-alpha_0)*psi_k(s') - psi_k(s)
//                  (cumulant = reward, stopping-value = feature attainment curr[k] per RR-
//                   Subtasks; structural beta = alpha_0 = 1/cap so option terminates ~every cap
//                   steps on average; sf[k] = k identity, no separate subtaskFeatureOf table).
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
    ctx.vCurr[0] = vNext;
    float reward_c = ctx.reward[0] - ctx.rho;
    float delta = reward_c + vNext - vPrev;
    if (!isfinite(delta)) { ctx.counters[OAK_CNT_LEARN_NF] += 1u; delta = 0.0f; }
    ctx.td[0] = delta;

    float bp = ctx.alpha0;
    uint nfPsi = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        float psiPrev = 0.0f, psiNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            psiPrev = fma(ctx.psi[k * ctx.R + i], prev[i], psiPrev);
            psiNext = fma(ctx.psi[k * ctx.R + i], curr[i], psiNext);
        }
        // Subtask k anchored at feature k (sf[k] = k identity).
        float anchor = (k < ctx.R) ? curr[k] : 0.0f;
        float dk = reward_c + bp * anchor + (1.0f - bp) * psiNext - psiPrev;
        if (!isfinite(dk)) { dk = 0.0f; nfPsi++; }
        ctx.tdPsi[k] = dk;
    }
    if (nfPsi > 0u) ctx.counters[OAK_CNT_TDPSI_NF] += nfPsi;
}

// =================================================================================================
// 5. oak_plan - asynchronous VI over option models (RR-Subtasks eq. 19).
//   Q^k(s) = r_hat^k(s) + V(n_hat^k(s));  planAdv = max_k Q^k - V(s) (self-bounded to (-1, 1)).
// =================================================================================================
kernel void oak_plan(
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
    float vCurr = ctx.vCurr[0];

    float bestQ = -FLT_MAX;
    uint nfPlan = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        float rHat = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);
        float vHat = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j) {
            float nj = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i)
                nj = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + i], curr[i], nj);
            vHat = fma(ctx.valueW[j], nj, vHat);
        }
        float Q = rHat + vHat;
        if (!isfinite(Q)) { nfPlan++; continue; }
        if (Q > bestQ) bestQ = Q;
    }
    float adv = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
    if (!isfinite(adv)) { adv = 0.0f; nfPlan++; }
    // Self-bounding x/(1+|x|) keeps planAdv on the same scale as on-policy td.
    ctx.planAdv[0] = adv / (1.0f + fabs(adv));
    if (nfPlan > 0u) ctx.counters[OAK_CNT_PLAN_NF] += nfPlan;
}

// =================================================================================================
// 6. oak_reduce_alpha - implicit alpha for the policy head (Kim arXiv:2510.06149).
//   alphaPi = alpha_0 / (1 + alpha_0 * ||s'||^2 * ||scoreGrad||^2)
// =================================================================================================
kernel void oak_reduce_alpha(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    threadgroup float buf[4];
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float s2 = 0.0f;
    for (uint i = tid; i < ctx.R; i += tgW) s2 = fma(curr[i], curr[i], s2);
    float ss = tg_sum_128(buf, tid, sgid, s2);

    float g2 = 0.0f;
    for (uint p = tid; p < ctx.D; p += tgW) g2 = fma(ctx.scoreGrad[p], ctx.scoreGrad[p], g2);
    float gg = tg_sum_128(buf, tid, sgid, g2);

    if (tid == 0u) {
        ctx.scalars->alphaPi = ctx.alpha0 / (1.0f + ctx.alpha0 * ss * gg);
        ctx.stats->latentStateMeanSquare = ss / float(max(ctx.R, 1u));
        ctx.stats->scoreGradL2Mean       = gg / float(max(ctx.D, 1u));
    }
}

// =================================================================================================
// 7. oak_learn_heads - 1-step TD updates on 5 learnable heads. NO eligibility traces, NO
//    encoder gradient, NO per-feature alpha (all stripped as non-Plan; see file header).
//
//    Heads (subid):
//      0: valueW[R]            -> alpha_0 * (td + planAdv) * prev[i]   (planAdv added HERE ONLY)
//      1: psi[K*R]             -> alpha_0 * tdPsi[k] * prev[i]
//      2: optPolicy[K*D*R]     -> alphaPi * td * scoreGrad[d] * prev[i] * chordPrev[k]
//      3: optReward[K*R]       -> alpha_0 * (reward - r_hat^k(prev)) * prev[i] * chordPrev[k]
//      4: optTransition[K*R*R] -> alpha_0 * (curr[j] - n_hat_j^k(prev)) * prev[i] * chordPrev[k]
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
    uint nOT  = ctx.K * ctx.R * ctx.R;
    uint nTotal = nVal + nPsi + nPi + nOR + nOT;
    if (gid >= nTotal) return;

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float td = ctx.td[0];
    float deltaValue = td + ctx.planAdv[0];
    float a0 = ctx.alpha0;

    // ---- Head 0: valueW[i] ----
    if (gid < nVal) {
        uint i = gid;
        float w = ctx.valueW[i];
        ctx.valueW[i] = safe_w(w, w + a0 * deltaValue * prev[i]);
        return;
    }
    // ---- Head 1: psi[k * R + i] ----
    if (gid < nVal + nPsi) {
        uint id = gid - nVal;
        uint k = id / ctx.R, i = id % ctx.R;
        float w = ctx.psi[id];
        ctx.psi[id] = safe_w(w, w + a0 * ctx.tdPsi[k] * prev[i]);
        return;
    }
    // ---- Head 2: optPolicy[k][d][i] - REINFORCE chord-attributed (Williams 1992). ----
    if (gid < nVal + nPsi + nPi) {
        uint id = gid - nVal - nPsi;
        uint k  = id / (ctx.D * ctx.R);
        uint rem = id - k * ctx.D * ctx.R;
        uint d  = rem / ctx.R, i = rem % ctx.R;
        float grad = ctx.chordPrev[k] * ctx.scoreGrad[d] * prev[i];
        float w = ctx.optPolicy[id];
        ctx.optPolicy[id] = safe_w(w, w + ctx.alphaPi * td * grad);
        return;
    }
    // ---- Head 3: optReward[k][i] - 1-step option reward model (RR-Subtasks eq. 17, beta=1). ----
    if (gid < nVal + nPsi + nPi + nOR) {
        uint id = gid - nVal - nPsi - nPi;
        uint k = id / ctx.R, i = id % ctx.R;
        float predPrev = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            predPrev = fma(ctx.optReward[k * ctx.R + j], prev[j], predPrev);
        float r_td = ctx.reward[0] - predPrev;
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optReward[id];
        ctx.optReward[id] = safe_w(w, w + a0 * r_td * grad);
        return;
    }
    // ---- Head 4: optTransition[k][j][i] - 1-step transition model. ----
    {
        uint id = gid - nVal - nPsi - nPi - nOR;
        uint k  = id / (ctx.R * ctx.R);
        uint rem = id - k * ctx.R * ctx.R;
        uint j  = rem / ctx.R, i = rem % ctx.R;
        float predPrev = 0.0f;
        for (uint ii = 0u; ii < ctx.R; ++ii)
            predPrev = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + ii], prev[ii], predPrev);
        float n_td = curr[j] - predPrev;
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optTransition[id];
        ctx.optTransition[id] = safe_w(w, w + a0 * n_td * grad);
    }
}

// =================================================================================================
// 8. oak_meta_action - soft-chord blended linear policy. mu_p = sum_k chord[k] * (W_pi[k][p] . s).
//    Wrapper does the structural-sigma=1 Gaussian draw with PRE-CLAMP z scoreGrad.
// =================================================================================================
kernel void oak_meta_action(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint p [[thread_position_in_grid]])
{
    if (p >= cfg.actionParamCount) return;
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float mu = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float lam = ctx.chord[k];
        if (lam == 0.0f) continue;
        float dot = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            dot = fma(ctx.optPolicy[(k * ctx.D + p) * ctx.R + i], curr[i], dot);
        mu = fma(lam, dot, mu);
    }
    ctx.policyMean[p] = unit_soft(mu);
}

// =================================================================================================
// 9. oak_bk_reduce - per-feature featMean Welford EMA + rho-bar update + LPM model error EMA.
// =================================================================================================
kernel void oak_bk_reduce(
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
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    for (uint i = tid; i < ctx.R; i += tgW) {
        ctx.featMean[i] += invT * (curr[i] - ctx.featMean[i]);
    }
    if (tid == 0u) {
        ctx.scalars->rho += invT * (ctx.reward[0] - ctx.rho);   // Reward Centering.
        ctx.stats->rewardRate = ctx.scalars->rho;
        float pa = ctx.planAdv[0];
        ctx.stats->optionModelPredictionMSE = pa * pa;
        ctx.stats->alphaPiStat = ctx.scalars->alphaPi;
    }
}

// =================================================================================================
// 10. oak_bk_utility - per-feature GraMa utility, per-option |tdPsi|^2 utility, and per-buffer
//     max(|w|) for telemetry. Grid = R + K + 5 threads.
// =================================================================================================
kernel void oak_bk_utility(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    device const float *prev = state_at(ctx, ctx.stateSlot);

    if (gid < ctx.R) {
        uint i = gid;
        float xi = prev[i];
        // GraMa: utility = EMA(|delta * x_i| + sum_k |delta_k * x_i|).
        float gSum = fabs(ctx.td[0] * xi);
        for (uint k = 0u; k < ctx.K; ++k) gSum += fabs(ctx.tdPsi[k] * xi);
        float perI = gSum / float(1u + ctx.K);
        ctx.utilFeat[i] += invT * (perI - ctx.utilFeat[i]);
        return;
    }
    uint k = gid - ctx.R;
    if (k < ctx.K) {
        float dk = ctx.tdPsi[k];
        ctx.utilOpt[k] += invT * (dk * dk - ctx.utilOpt[k]);
        return;
    }
    uint b = gid - ctx.R - ctx.K;
    if (b >= 5u) return;
    device const float *base; uint count;
    switch (b) {
        case 0u: base = ctx.valueW;        count = ctx.R; break;
        case 1u: base = ctx.psi;           count = ctx.K * ctx.R; break;
        case 2u: base = ctx.optPolicy;     count = ctx.K * ctx.D * ctx.R; break;
        case 3u: base = ctx.optReward;     count = ctx.K * ctx.R; break;
        default: base = ctx.optTransition; count = ctx.K * ctx.R * ctx.R; break;
    }
    float m = 0.0f;
    for (uint i = 0u; i < count; ++i) m = max(m, fabs(base[i]));
    switch (b) {
        case 0u: ctx.stats->valueWMax        = m; break;
        case 1u: ctx.stats->psiMax           = m; break;
        case 2u: ctx.stats->optPolicyMax     = m; break;
        case 3u: ctx.stats->optRewardMax     = m; break;
        default: ctx.stats->optTransitionMax = m; break;
    }
}

// =================================================================================================
// 11. oak_bk_admit_commit - Oak feedback: generate-and-test reinit of worst-utility option AND
//     worst-utility feature (Plan Step 11 remove-replace; Step 2 generate-and-test). Single
//     reinit per step. NO threshold gates - if utility is bottom-of-pack, reseed.
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

    // Find worst option: lowest utilOpt. Reseed psi[k][:R], optPolicy[k][:D*R], optReward[k][:R],
    // optTransition[k][:R*R] with hash-based small projections.
    uint worstK = 0u; float worstUO = FLT_MAX;
    for (uint k = 0u; k < ctx.K; ++k) {
        if (ctx.utilOpt[k] < worstUO) { worstUO = ctx.utilOpt[k]; worstK = k; }
    }
    {
        uint k = worstK;
        for (uint i = 0u; i < ctx.R; ++i) {
            ctx.psi[k * ctx.R + i] =
                (hash_uniform(ctx.stepCount, k, i) * 2.0f - 1.0f) * ctx.alpha0;
            ctx.optReward[k * ctx.R + i] =
                (hash_uniform(ctx.stepCount, k, ctx.R + i) * 2.0f - 1.0f) * ctx.alpha0;
            for (uint j = 0u; j < ctx.R; ++j) {
                ctx.optTransition[(k * ctx.R + j) * ctx.R + i] =
                    (hash_uniform(ctx.stepCount, (k * ctx.R + j) ^ 0xa3c59ac3u, i) * 2.0f - 1.0f) * ctx.alpha0;
            }
        }
        for (uint pi = 0u; pi < ctx.D * ctx.R; ++pi) {
            ctx.optPolicy[k * ctx.D * ctx.R + pi] =
                (hash_uniform(ctx.stepCount, k, 2u * ctx.R + pi) * 2.0f - 1.0f) * ctx.alpha0;
        }
        ctx.utilOpt[k] = 0.0f;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
    }

    // Find worst feature: lowest utilFeat. Reseed encoder column with random projection and
    // reset associated stats. (Plan Step 2 generate-and-test.)
    uint worstI = 0u; float worstUF = FLT_MAX;
    for (uint i = 0u; i < ctx.R; ++i) {
        if (ctx.utilFeat[i] < worstUF) { worstUF = ctx.utilFeat[i]; worstI = i; }
    }
    {
        uint i = worstI;
        for (uint o = 0u; o < ctx.baseO; ++o) {
            float u = hash_uniform(ctx.stepCount, i, o + 0xdeadbeefu) * 2.0f - 1.0f;
            ctx.encoder[i * ctx.baseO + o] = u * rsqrt(float(max(ctx.baseO, 1u)));
        }
        ctx.utilFeat[i] = 0.0f;
        ctx.featMean[i] = 0.0f;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
    }

    ctx.stats->oakOptionReinitCount  = float(ctx.counters[OAK_CNT_REINIT_OPT]);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

    // Telemetry roll-ups.
    float utilFeatMean = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) utilFeatMean += ctx.utilFeat[i];
    ctx.stats->utilityFeatureMean = utilFeatMean / float(max(ctx.R, 1u));
    float utilOptMean = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) utilOptMean += ctx.utilOpt[k];
    ctx.stats->utilityOptionMean = utilOptMean / float(max(ctx.K, 1u));
    ctx.stats->rawNonFiniteCount   = float(ctx.counters[OAK_CNT_OBS_NF]
                                         + ctx.counters[OAK_CNT_LEARN_NF]
                                         + ctx.counters[OAK_CNT_PLAN_NF]
                                         + ctx.counters[OAK_CNT_TDPSI_NF]);
    ctx.stats->encodeNonFiniteCount = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount  = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount   = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount  = float(ctx.counters[OAK_CNT_TDPSI_NF]);

    float mLat = 0.0f, mPol = 0.0f;
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    for (uint i = 0u; i < ctx.R; ++i) mLat = max(mLat, fabs(curr[i]));
    for (uint p = 0u; p < ctx.D; ++p) mPol = max(mPol, fabs(ctx.policyMean[p]));
    ctx.stats->maxAbsLatentPrediction = mLat;
    ctx.stats->maxAbsPolicyOutput     = mPol;
    float dl = 0.0f;
    for (uint p = 0u; p < ctx.D; ++p) {
        float d = ctx.policyMean[p] - ctx.prevPolicyMean[p];
        dl = fma(d, d, dl);
    }
    ctx.stats->actionDeltaL2Mean = dl / float(max(ctx.D, 1u));

    ctx.scalars->stateSlot = 1u - ctx.stateSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}
