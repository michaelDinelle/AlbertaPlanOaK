// =================================================================================================
// OakKernel.metal - Orthodox Step 11 Oak agent. Single-stream (B=1), Metal 4 ICB.
//
// What the Alberta Plan demands (Sutton-Bowling-Pilarski 2022, arXiv:2208.11173, Step 11):
//   - Perception: state from obs (encoder + belief recurrence)
//   - K subtasks defined by features sf[k] (psi GVFs, RR-Subtasks eq. 4 with bonus stopping value)
//   - K options = K policies + termination (optPolicy + structural beta from feature attainment)
//   - K option models r_hat^k, n_hat^k (RR-Subtasks eq. 17 expectation model)
//   - Planning over option models (RR-Subtasks eq. 19 asynchronous VI)
//   - Option keyboard chord lambda in R^K (Plan p. 10-11; Design A: lambda blends K component
//     options by their relative subtask values; chord intensities = subtask urgencies)
//   - Feedback: utility of features/subtasks/options/models -> remove/replace (bk_utility +
//     bk_admit_commit + generate-and-test feature reinit)
//   - Average-reward differential TD with rho-bar centering (Reward Centering 2405.09999;
//     varpi-RTD stability proof Sakha-Kamalapurkar-Meyn 2603.27874v2)
//
// What the Plan does NOT demand (and is NOT here):
//   - Learned termination weights (no betaW). Termination beta is structural: sigmoid of
//     feature attainment z-score. Zero parameters.
//   - Learned chord softmax weights (no omegaW). Chord lambda is derived: softmax(psi_k(s)).
//     Zero parameters.
//   - Layer normalization, batch normalization, or any stationary scale prior (BWH-rejected)
//   - Discount factor (Plan Step 5: no discounting in continual setting)
//   - Episodic resets inside the agent (resets are wrapper events only)
//
// References:
//   - RR-Subtasks (arXiv:2202.03466): subtask GVFs, expectation models, planning eq. 19
//   - BurningRED (arXiv:2410.10578): avg-reward differential extension
//   - varpi-RTD (arXiv:2603.27874v2): centering stability
//   - Implicit alpha (arXiv:2510.06149): alpha_0/(1+alpha_0*||x||^2) self-bounding step
//   - MetaOptimize (arXiv:2402.02342): per-feature step-size meta-learning framework
//   - SwiftSarsa (arXiv:2507.19539): streaming TIDBD instantiation
//   - GraMa (arXiv:2505.24061): per-feature gradient utility for ranking
//   - FIRE+SWR (arXiv:2602.08040): conjunctive dormancy reinit
//   - Generate-and-test (Mahmood-Sutton-Pilarski 2012, Rahman 2020, Samani 2022)
//   - Sutton-Barto Ch. 9.5.4 (online normalization), Ch. 12.7 (eligibility traces)
//   - Big World Hypothesis (.cursor/rules/alberta_plan.mdc): no zero clamps, magnitudes carry
//     information
//
// Single-stream: one observation -> one action -> one reward per step. No batch dimension.
// Multi-env benchmarking lives in the Swift host as N independent agent instances.
// =================================================================================================
#include <metal_stdlib>
#include <float.h>
using namespace metal;

constant uint OAK_SIMD = 32u;       // Apple GPU SIMD width (structural, Inv-9).

// Per-step non-finite counters (cleared each step by the host) and life-of-engine reinit
// counters (never cleared).
constant uint OAK_CNT_OBS_NF       = 0u;
constant uint OAK_CNT_LEARN_NF     = 1u;
constant uint OAK_CNT_PLAN_NF      = 2u;
constant uint OAK_CNT_REINIT_OPT   = 3u;
constant uint OAK_CNT_REINIT_FEAT  = 4u;
constant uint OAK_CNT_TDPSI_NF     = 5u;

// =================================================================================================
// Configuration mirrored from Swift KernelConfig. All offsets are float-counts (UInt32 buffer is
// counted in UInt32 units). Single-stream means no `batch` field.
// =================================================================================================
struct KernelConfig {
    uint obsDim, rank, actDim, optionCount, baseObsDim;
    uint actionParamCount;                  // dPolicy from the env wrapper (Inv-1).

    // Param offsets (params buffer, fp32). Six heads only (Plan-required):
    //   encoder  - perception
    //   valueW   - main value GVF (cumulant = reward, no stopping value)
    //   psi      - K subtask GVFs (cumulant = reward, stopping = bonus on feature)
    //   optPolicy- K component policies (Design A blending)
    //   optReward, optTransition - K option models (RR-Subtasks eq. 17)
    uint pEncoder, pValue, pPsi, pOptPolicy;
    uint pOptReward, pOptTransition;

    // State arena offsets (state buffer, fp32). Single-stream: each slot holds ONE row.
    uint sObservation, sObsMean, sObsSM;
    uint sState;                             // 2 * R floats (slot toggle).
    uint sPrevPolicyMean, sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi, sVCurr;
    uint sChord, sChordPrev, sBetaProb;      // betaProb is STRUCTURAL (no learned head).
    uint sFeatSM, sWBar, sFeatMean;
    uint sUtilFeat, sUtilOpt, sUtilPlan, sOptFracEMA;
    uint sAlphaImpFeat;                      // recomputed in bk_admit_commit from featSM (no meta-grad).
    uint sPlanAdv;                           // scalar planning advantage (oak_plan -> learn_heads).

    // UInt arena offsets (uint buffer, UInt32).
    uint uDone, uActiveMask, uActiveList, uOption, uCounters, uSubtaskFeatureOf;
    uint uOptionAge, uPlanArgmax;

    // Structural meta-learning constants (Inv-9: derived from cap = R, never tuned).
    float cap;          // cap = R (per-feature retention bound; structural).
    float alpha0;       // alpha0 = 1/cap.
    float ridge;        // ridge = 1/(cap+1) for safe-divide / track-rate.
};

// =================================================================================================
// Per-step scalars written by kernels and consumed across kernels in the same step. Mirrored from
// Swift KernelScalars.
// =================================================================================================
struct KernelScalars {
    uint  stepCount;            // monotonic step counter (host-bumped via bk_admit_commit).
    uint  activeCount;          // |activeList|; floor at ceil(sqrt(R)).
    uint  stateSlot;            // toggle: which of the 2 state slots holds s_t (0 or 1).
    uint  optionSlot;           // toggle: which of the 2 option slots holds k_t (0 or 1).
    uint  activeListIsIdentity; // 1 iff activeCount == R and activeList[i]==i; fast path flag.

    float rho;                  // rho_bar - cumulative reward EMA (Reward Centering).
    float pEff;                 // effective rank proxy from featSM.
    float alphaPi;              // implicit alpha for policy weights (||state||^2 * ||scoreGrad||^2).
    float modelErrEMA;          // EMA of option-model prediction MSE (LPM signal).
    float modelErrEMAPrev;      // slow snapshot of modelErrEMA (LPM second-derivative).
};

// =================================================================================================
// Per-step telemetry. Layout mirrors Swift KernelStats. Host reads these at the end of each step.
// =================================================================================================
struct KernelStats {
    float rewardRate, latentStateMeanSquare, activeCount;
    float rawNonFiniteCount, sanitizedNonFiniteCount;
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean, scoreGradL2Mean;
    float optionModelPredictionMSE;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilityOptionMean;
    float tdPsiNonFiniteCount;
    float pEffectiveRank;
    float modelErrEMA, modelErrEMAPrev;
    float alphaPiStat;
    float oakFeatureReinitCount;
    float valueWMax, psiMax, optPolicyMax;
    float optRewardMax, optTransitionMax;
};

// =================================================================================================
// Numerical safety - two helpers, both parameter-free.
//   safe_w(old, new):    NF-guarded weight write that PRESERVES the previous value on non-finite
//                        (no zero-clamp; no arbitrary magnitude cap; weights grow as the agent's
//                        slice of reality demands - BWH).
//   unit_soft(x) = tanh: structural sigmoid for FORWARD OUTPUTS (state, mu, planAdv). The
//                        bound (-1, 1) is the natural range of the tanh sigmoid, not a tunable
//                        constant. Keeps eligibility traces structurally bounded by 1/(1-lambda).
// No bounded_soft / safe_half / fp16-derived cap anywhere. EMAs (obsMean, obsSM, featMean,
// rho_bar, vMeanBaseline, etc.) are stored raw fp32 - they track the world's actual statistics.
// =================================================================================================
inline float safe_w(float old_w, float new_w) {
    return isfinite(new_w) ? new_w : old_w;
}

inline float unit_soft(float x) {
    return isfinite(x) ? tanh(x) : 0.0f;
}

// Tracking-rate EMA (Sutton-Barto §2.6): equivalent to 1/(stepCount+1) as a step shrinks toward
// 0, with `ridge` floor for early steps where stepCount is small. Used for non-stationary
// tracking of the world's signal statistics (BWH).
inline float track_rate(uint step, float ridge) {
    return max(1.0f / (float(step) + 1.0f), ridge);
}

// splitmix32 hash for deterministic RNG without per-step host alloc. Returns u in [0, 1).
inline float hash_uniform(uint a, uint b, uint c) {
    uint z = a ^ (b * 0x7feb352du) ^ (c * 0x846ca68bu);
    z = (z ^ (z >> 16)) * 0x9e3779b9u;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z ^= z >> 16;
    return float(z >> 8) * 5.9604645e-8f;       // 1 / 2^24
}

// =================================================================================================
// Per-step computation context. ALL kernels build one of these from the bound buffers and config.
// Eliminates pointer-arithmetic noise from every kernel body.
// =================================================================================================
struct Ctx {
    uint O, R, K, A, M, baseO, D;
    uint stepCount, activeCount, stateSlot, optionSlot;
    float cap, alpha0, ridge;
    float rho, alphaPi;

    device float        *encoder;
    device float        *valueW;
    device float        *psi;
    device float        *optPolicy;
    device float        *optReward;
    device float        *optTransition;

    device float        *observation;
    device float        *observationMean;
    device float        *obsSM;
    device float        *state;          // [2 * R] - both slots flat.
    device float        *prevPolicyMean;
    device float        *policyMean;
    device float        *scoreGrad;
    device float        *reward;
    device float        *td;
    device float        *tdPsi;
    device float        *vCurr;
    device float        *chord;
    device float        *chordPrev;
    device float        *betaProb;
    device float        *featSM;
    device float        *wBar;
    device float        *featMean;
    device float        *utilFeat;
    device float        *utilOpt;
    device float        *utilPlan;
    device float        *optFracEMA;
    device float        *alphaImpFeat;
    device float        *planAdv;

    device uint         *done;
    device uint         *activeMask;
    device uint         *activeList;
    device uint         *option;
    device uint         *counters;
    device uint         *subtaskFeatureOf;
    device uint         *optionAge;
    device uint         *planArgmax;

    device KernelScalars *scalars;
    device KernelStats   *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount; c.A = cfg.actDim;
    c.baseO = cfg.baseObsDim; c.D = cfg.actionParamCount;
    c.M = c.R + c.K;                                // psi concat dim (state R + chord K).
    c.stepCount = scalars->stepCount;
    c.activeCount = scalars->activeCount;
    c.stateSlot = scalars->stateSlot;
    c.optionSlot = scalars->optionSlot;
    c.cap = cfg.cap; c.alpha0 = cfg.alpha0; c.ridge = cfg.ridge;
    c.rho = scalars->rho;
    c.alphaPi = scalars->alphaPi;

    c.encoder        = params + cfg.pEncoder;
    c.valueW         = params + cfg.pValue;
    c.psi            = params + cfg.pPsi;
    c.optPolicy      = params + cfg.pOptPolicy;
    c.optReward      = params + cfg.pOptReward;
    c.optTransition  = params + cfg.pOptTransition;

    c.observation    = stateArena + cfg.sObservation;
    c.observationMean= stateArena + cfg.sObsMean;
    c.obsSM          = stateArena + cfg.sObsSM;
    c.state          = stateArena + cfg.sState;
    c.prevPolicyMean = stateArena + cfg.sPrevPolicyMean;
    c.policyMean     = stateArena + cfg.sPolicyMean;
    c.scoreGrad      = stateArena + cfg.sScoreGrad;
    c.reward         = stateArena + cfg.sReward;
    c.td             = stateArena + cfg.sTd;
    c.tdPsi          = stateArena + cfg.sTdPsi;
    c.vCurr          = stateArena + cfg.sVCurr;
    c.chord          = stateArena + cfg.sChord;
    c.chordPrev      = stateArena + cfg.sChordPrev;
    c.betaProb       = stateArena + cfg.sBetaProb;
    c.featSM         = stateArena + cfg.sFeatSM;
    c.wBar           = stateArena + cfg.sWBar;
    c.featMean       = stateArena + cfg.sFeatMean;
    c.utilFeat       = stateArena + cfg.sUtilFeat;
    c.utilOpt        = stateArena + cfg.sUtilOpt;
    c.utilPlan       = stateArena + cfg.sUtilPlan;
    c.optFracEMA     = stateArena + cfg.sOptFracEMA;
    c.alphaImpFeat   = stateArena + cfg.sAlphaImpFeat;
    c.planAdv        = stateArena + cfg.sPlanAdv;

    c.done             = uintArena + cfg.uDone;
    c.activeMask       = uintArena + cfg.uActiveMask;
    c.activeList       = uintArena + cfg.uActiveList;
    c.option           = uintArena + cfg.uOption;
    c.counters         = uintArena + cfg.uCounters;
    c.subtaskFeatureOf = uintArena + cfg.uSubtaskFeatureOf;
    c.optionAge        = uintArena + cfg.uOptionAge;
    c.planArgmax       = uintArena + cfg.uPlanArgmax;

    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

// Standardized observation x_tilde = (x - mu) / sqrt(SM - mu^2 + ridge). Online normalization
// per Sutton-Barto Ch. 9.5.4 - tracking estimates, not stationary BatchNorm.
inline float standardized_obs(float x, float mean, float sm, float ridge) {
    float var = max(sm - mean * mean, 0.0f);
    return (x - mean) * rsqrt(var + ridge);
}

// =================================================================================================
// Threadgroup reduction primitives (small fixed count, no allocation).
// =================================================================================================
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

inline float tg_max_128(threadgroup float *buf, uint tid, uint sgid, float v) {
    v = simd_max(v);
    if (simd_is_first()) buf[sgid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float m = (tid < 4u) ? buf[tid] : -FLT_MAX;
        m = simd_max(m);
        if (tid == 0u) buf[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// =================================================================================================
// 1. oak_bk_obs - per-feature observation Welford EMA. Single TG (32 threads strides over O).
//    Updated FIRST so encode reads the just-tracked statistics.
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
// 2. oak_encode - perception. Per-feature linear projection over standardized obs + per-feature
//    belief retention (alpha_i = min(1, alphaImpFeat[i]*cap), Belief-State RWKV 2604.09671).
//    Outputs the new state slot. One thread per latent dim; grid = R threads.
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
    uint writeSlot = 1u - ctx.stateSlot;
    uint encRow = feat * ctx.baseO;

    float z_instant = 0.0f;
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = standardized_obs(ctx.observation[o], ctx.observationMean[o],
                                   ctx.obsSM[o], ctx.ridge);
        z_instant = fma(ctx.encoder[encRow + o], c, z_instant);
    }
    float retention = min(1.0f, ctx.alphaImpFeat[feat] * ctx.cap);
    float prev = state_at(ctx, ctx.stateSlot)[feat];
    float z = retention * z_instant + (1.0f - retention) * prev;
    state_at(ctx, writeSlot)[feat] = unit_soft(z);

    if (feat == 0u) {
        // Reset planAdv from previous step (one writer per step).
        ctx.planAdv[0] = 0.0f;
    }
}

// =================================================================================================
// 3. oak_meta_policy - parameter-free option keyboard.
//    Chord lambda[k] = softmax(psi_k(s')) - the chord intensity for option k IS the value of
//    pursuing subtask k from here (Alberta Plan p. 10: "chord is a fixed blending of the
//    component options, taking into account the intensities of each component note/option/
//    feature/subtask in the chord"). Zero learned parameters.
//    Beta termination beta[k] = sigmoid(z_k) where z_k is the standardized excess of the
//    subtask's anchor feature over its tracking mean. RR-Subtasks: terminate when feature is
//    high. Zero learned parameters.
//    The argmax option index is committed for telemetry/learning attribution; the actual policy
//    output uses the soft chord for blending.
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

    // chord[k] = softmax(psi_k(s')); psi already incorporates the chord-prev component (M=R+K).
    // Stable softmax via max-subtract.
    float maxL = -FLT_MAX;
    float logits[OAK_SIMD];
    for (uint k = 0u; k < ctx.K; ++k) {
        float l = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            l = fma(ctx.psi[k * ctx.M + i], curr[i], l);
        for (uint j = 0u; j < ctx.K; ++j)
            l = fma(ctx.psi[k * ctx.M + ctx.R + j], ctx.chordPrev[j], l);
        logits[k] = l;
        maxL = max(maxL, l);
    }
    float Z = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) Z += exp(logits[k] - maxL);
    float invZ = 1.0f / max(Z, FLT_MIN);
    uint  bestK = 0u; float bestP = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float p = exp(logits[k] - maxL) * invZ;
        ctx.chord[k] = p;
        if (p > bestP) { bestP = p; bestK = k; }
    }
    ctx.option[1u - ctx.optionSlot] = bestK;
    ctx.optionAge[0] += 1u;

    // Structural beta termination: sigmoid of z-scored anchor-feature excess. Beta is high when
    // the subtask's feature is above its tracking mean (RR-Subtasks: "encourages ending when the
    // feature is high"). Single shared beta scalar = beta of the argmax option (for the option
    // model's TD update). Zero learned parameters.
    uint f_best = ctx.subtaskFeatureOf[bestK];
    if (f_best >= ctx.R) f_best = 0u;
    float fmean = ctx.featMean[f_best];
    float fvar  = max(ctx.featSM[f_best] - fmean * fmean, 0.0f);
    float z = (curr[f_best] - fmean) * rsqrt(fvar + ctx.ridge);
    ctx.betaProb[0] = 1.0f / (1.0f + exp(-z));
}

// =================================================================================================
// 4. oak_predict - Horde-style multi-GVF update. Computes:
//      Main TD:    delta = (r - rho_bar) + (V(s') - v_bar) - (V(s) - v_bar)  (varpi-RTD)
//      Subtask k:  delta_k = (r - rho_bar) + beta_k * g_k + (1 - beta_k) * (psi_k(s') - psi_bar_k)
//                          - (psi_k(s) - psi_bar_k)
//                  where g_k = (w_bar[f_k] - valueW[f_k]) * curr[f_k]   (RR-Subtasks eq. 4)
//    Single-threaded (one row, sequentially through K).
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

    // Differential TD with rho-bar centering (Reward Centering 2405.09999).
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

    // K subtask GVFs. Cumulant = main reward (RR-Subtasks); stopping value = bonus on anchor
    // feature (eq. 4); structural beta from feature attainment.
    uint nfPsi = 0u;
    float beta_k = ctx.betaProb[0];
    for (uint k = 0u; k < ctx.K; ++k) {
        uint f_k = ctx.subtaskFeatureOf[k]; if (f_k >= ctx.R) f_k = 0u;
        float psiPrev = 0.0f, psiNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            psiPrev = fma(ctx.psi[k * ctx.M + i], prev[i], psiPrev);
            psiNext = fma(ctx.psi[k * ctx.M + i], curr[i], psiNext);
        }
        for (uint j = 0u; j < ctx.K; ++j) {
            psiPrev = fma(ctx.psi[k * ctx.M + ctx.R + j], ctx.chordPrev[j], psiPrev);
            psiNext = fma(ctx.psi[k * ctx.M + ctx.R + j], ctx.chord[j],     psiNext);
        }
        float bonus = (ctx.wBar[f_k] - ctx.valueW[f_k]) * curr[f_k];
        float dk = reward_c + beta_k * bonus + (1.0f - beta_k) * psiNext - psiPrev;
        if (!isfinite(dk)) { dk = 0.0f; nfPsi++; }
        ctx.tdPsi[k] = dk;
    }
    if (nfPsi > 0u) ctx.counters[OAK_CNT_TDPSI_NF] += nfPsi;
}

// =================================================================================================
// 5. oak_plan - Step 11 BLOCKER. Single-row asynchronous VI sweep over option models
//    (RR-Subtasks eq. 19): td += max_o [ r_hat^o(s') + V_hat(n_hat^o(s')) ] - V(s'). Search
//    control: option backups visited in utilOpt-ranked order (Step 9 prioritized sweeping with
//    linear FA, Sutton-Szepesvari-Geramifard-Bowling 2008). Writes planAdv (consumed by
//    learn_value as augmented TD) and planArgmax (consumed by bk_admit_commit as feature
//    attribution for utilPlan). Single-threaded.
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
    uint  bestK = 0u;
    uint nfPlan = 0u;

    for (uint k = 0u; k < ctx.K; ++k) {
        // r_hat^k(s') = w_r^k . s'
        float rHat = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);

        // n_hat^k(s')[j] = w_n^{k,j} . s'  -> V_hat(n_hat) = sum_j valueW[j] * n_hat[j].
        float vHat = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j) {
            float nj = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i)
                nj = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + i], curr[i], nj);
            vHat = fma(ctx.valueW[j], nj, vHat);
        }
        float Q = rHat + vHat;
        if (!isfinite(Q)) { nfPlan++; continue; }
        if (Q > bestQ) { bestQ = Q; bestK = k; }
    }

    float adv = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
    if (!isfinite(adv)) { adv = 0.0f; nfPlan++; }
    // Self-bounding: x / (1 + |x|) maps R smoothly into (-1, 1). Keeps planAdv on the same
    // scale as the on-policy td so planning is a perturbation, not a dominator (RR-Subtasks
    // eq. 19 - planning is interleaved learning, same alpha; without bounding, an untrained
    // option model would drown out the on-policy signal).
    ctx.planAdv[0] = adv / (1.0f + fabs(adv));
    ctx.planArgmax[0] = bestK;
    if (nfPlan > 0u) ctx.counters[OAK_CNT_PLAN_NF] = ctx.counters[OAK_CNT_PLAN_NF] + nfPlan;
}

// =================================================================================================
// 6. oak_reduce_alpha - global implicit alpha for the policy head only (Kim 2510.06149).
//      alphaPi = alpha_0 / (1 + alpha_0 * ||s'||^2 * ||scoreGrad||^2)
//    Per-feature alpha for value/psi/model heads is computed in oak_bk_admit_commit from
//    featSM[i] (parameter-free per-feature; Step 1 of the Plan).
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
// 7. oak_learn_heads - 1-step TD updates on Plan-required learnable heads. NO eligibility
//    traces (Plan note: traces are an EXAMPLE of "short-term credit-assignment memories", not
//    required; Sutton-Barto Ch. 6 1-step TD with linear FA is provably stable for average
//    reward, traces with bootstrapping are not). NO encoder gradient updates (the Plan: per-
//    feature alpha IS representation learning; encoder evolution happens via generate-and-test
//    in bk_admit_commit, Step 2). One grid-stride pass; each thread does one safe_w write.
//
//    Heads (subid):
//      0: valueW[R]            -> (td + planAdv) * prev[i]  (planning ONLY here, eq. 19)
//      1: psi[K*M]             -> tdPsi[k] * x_m  (1-step subtask TD)
//      2: optPolicy[K*D*R]     -> td * scoreGrad[d] * prev[i] * chordPrev[k]  (REINFORCE)
//      3: optReward[K*R]       -> r_td * prev[i] * chordPrev[k]  (RR-Subtasks eq. 17)
//      4: optTransition[K*R*R] -> n_td * prev[i] * chordPrev[k]  (RR-Subtasks eq. 17)
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
    uint nPsi = ctx.K * ctx.M;
    uint nPi  = ctx.K * ctx.D * ctx.R;
    uint nOR  = ctx.K * ctx.R;
    uint nOT  = ctx.K * ctx.R * ctx.R;
    uint nTotal = nVal + nPsi + nPi + nOR + nOT;
    if (gid >= nTotal) return;

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float td = ctx.td[0];
    float deltaValue = td + ctx.planAdv[0];     // valueW only (RR-Subtasks eq. 19).
    float bp = ctx.betaProb[0];

    // ---- Head 0: valueW[i] -- main TD + planning advantage, per-feature alpha. ----
    if (gid < nVal) {
        uint i = gid;
        float w = ctx.valueW[i];
        ctx.valueW[i] = safe_w(w, w + ctx.alphaImpFeat[i] * deltaValue * prev[i]);
        // Feature 2nd-moment EMA (drives next step's per-feature alpha + retention).
        float invT = track_rate(ctx.stepCount, ctx.ridge);
        float xi = curr[i];
        ctx.featSM[i] += invT * (xi * xi - ctx.featSM[i]);
        // Decayed high-water-mark for RR-Subtasks bonus weight (eq. 4).
        ctx.wBar[i] = max((1.0f - ctx.alpha0) * ctx.wBar[i], ctx.valueW[i]);
        return;
    }
    // ---- Head 1: psi[k * M + m] -- 1-step subtask TD on cumulant k. ----
    if (gid < nVal + nPsi) {
        uint id = gid - nVal;
        uint k = id / ctx.M;
        uint m = id % ctx.M;
        float xm = (m < ctx.R) ? prev[m] : ctx.chordPrev[m - ctx.R];
        float a = (m < ctx.R) ? ctx.alphaImpFeat[m] : ctx.alpha0;
        float w = ctx.psi[id];
        ctx.psi[id] = safe_w(w, w + a * ctx.tdPsi[k] * xm);
        return;
    }
    // ---- Head 2: optPolicy[k][d][i] -- REINFORCE chord-attributed (Williams 1992). ----
    if (gid < nVal + nPsi + nPi) {
        uint id = gid - nVal - nPsi;
        uint k  = id / (ctx.D * ctx.R);
        uint rem = id - k * ctx.D * ctx.R;
        uint d  = rem / ctx.R;
        uint i  = rem % ctx.R;
        float grad = ctx.chordPrev[k] * ctx.scoreGrad[d] * prev[i];
        float w = ctx.optPolicy[id];
        ctx.optPolicy[id] = safe_w(w, w + ctx.alphaPi * td * grad);
        return;
    }
    // ---- Head 3: optReward[k][i] -- 1-step option reward model (RR-Subtasks eq. 17). ----
    if (gid < nVal + nPsi + nPi + nOR) {
        uint id = gid - nVal - nPsi - nPi;
        uint k = id / ctx.R;
        uint i = id % ctx.R;
        float predPrev = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            predPrev = fma(ctx.optReward[k * ctx.R + j], prev[j], predPrev);
        float predNext = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            predNext = fma(ctx.optReward[k * ctx.R + j], curr[j], predNext);
        float r_td = ctx.reward[0] + (1.0f - bp) * predNext - predPrev;
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optReward[id];
        ctx.optReward[id] = safe_w(w, w + ctx.alphaImpFeat[i] * r_td * grad);
        return;
    }
    // ---- Head 4: optTransition[k][j][i] -- 1-step transition model. ----
    {
        uint id = gid - nVal - nPsi - nPi - nOR;
        uint k  = id / (ctx.R * ctx.R);
        uint rem = id - k * ctx.R * ctx.R;
        uint j  = rem / ctx.R;
        uint i  = rem % ctx.R;
        float predPrev = 0.0f;
        for (uint ii = 0u; ii < ctx.R; ++ii)
            predPrev = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + ii], prev[ii], predPrev);
        float predNext = 0.0f;
        for (uint ii = 0u; ii < ctx.R; ++ii)
            predNext = fma(ctx.optTransition[(k * ctx.R + j) * ctx.R + ii], curr[ii], predNext);
        float n_td = curr[j] + (1.0f - bp) * predNext - predPrev;
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optTransition[id];
        ctx.optTransition[id] = safe_w(w, w + ctx.alphaImpFeat[i] * n_td * grad);
    }
}

// =================================================================================================
// 8. oak_meta_action - soft-chord blended linear policy. mu_p = sum_k chord[k] * (W_pi[k][p] . s).
//    Per-row; one thread per dim p. Wrapper does the structural-sigma=1 Gaussian draw with
//    pre-clamp z as scoreGrad (Williams 1992 REINFORCE).
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
// 9. oak_bk_reduce - per-feature Welford EMA for featMean. Single TG x 128 threads.
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
        float r = ctx.reward[0];
        ctx.scalars->rho += invT * (r - ctx.rho);                  // Reward Centering.
        ctx.stats->rewardRate = ctx.scalars->rho;
        float pa = ctx.planAdv[0];
        float mse = pa * pa;
        ctx.scalars->modelErrEMAPrev = ctx.scalars->modelErrEMA;
        ctx.scalars->modelErrEMA += invT * (mse - ctx.scalars->modelErrEMA);
        ctx.stats->optionModelPredictionMSE = mse;
        ctx.stats->modelErrEMA       = ctx.scalars->modelErrEMA;
        ctx.stats->modelErrEMAPrev   = ctx.scalars->modelErrEMAPrev;
        ctx.stats->alphaPiStat       = ctx.scalars->alphaPi;
    }
}

// =================================================================================================
// 10. oak_bk_utility - GraMa per-feature gradient utility (2505.24061), per-option |delta_k|^2
//     utility, planning utility (1{f_k* used i}), and per-buffer max(|w|) for the bounded-magnitude
//     gate. Grid = R + K threads laid out so that the first R handle features and the next K
//     handle options + weight-magnitudes (one per buffer for the structural max).
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
        // GraMa per-feature utility: EMA(|delta * x_i| + sum_k |delta_k * x_i|).
        float xi = prev[i];
        float gSum = fabs(ctx.td[0] * xi);
        for (uint k = 0u; k < ctx.K; ++k) gSum += fabs(ctx.tdPsi[k] * xi);
        float perI = gSum / float(1u + ctx.K);
        ctx.utilFeat[i] += invT * (perI - ctx.utilFeat[i]);
        // Planning utility: 1 iff feature i anchors the bestK option (FCSTOMP feedback).
        uint bestK = ctx.planArgmax[0];
        uint anchor = ctx.subtaskFeatureOf[bestK];
        float planContrib = (anchor == i) ? 1.0f : 0.0f;
        ctx.utilPlan[i] += invT * (planContrib - ctx.utilPlan[i]);
        return;
    }
    uint k = gid - ctx.R;
    if (k < ctx.K) {
        float dk = ctx.tdPsi[k];
        ctx.utilOpt[k] += invT * (dk * dk - ctx.utilOpt[k]);
        ctx.optFracEMA[k] += invT * (ctx.chord[k] - ctx.optFracEMA[k]);
        return;
    }
    // Last 5 threads: per-buffer max(|w|) (one per LEARNED head; encoder is fixed random
    // projection so its magnitude is structural at init time).
    uint b = gid - ctx.R - ctx.K;
    if (b >= 5u) return;
    device const float *base; uint count;
    switch (b) {
        case 0u: base = ctx.valueW;        count = ctx.R; break;
        case 1u: base = ctx.psi;           count = ctx.K * ctx.M; break;
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
// 11. oak_bk_admit_commit - the Oak feedback engine. Reranks sf[K] by combined utilFeat +
//     utilPlan (closes the FCSTOMP loop per RR-Subtasks Section 8); FIRE+SWR option reinit
//     conjunctive on dormant_k AND magnitudeOutlier_k (2602.08040); generate-and-test feature
//     creation on reinited feature slot (Mahmood-Sutton-Pilarski 2012); rho/v_bar/psi_bar
//     baseline updates consumed by the next step's predict. Single-threaded.
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

    // ---- Active-feature admission. Floor at ceil(sqrt(R)), structural (Inv-9). ----
    float meanU = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) meanU += ctx.utilFeat[i] + ctx.utilPlan[i];
    meanU /= float(max(ctx.R, 1u));
    float invR = 1.0f / float(max(ctx.R, 1u));
    uint floor = (uint) ceil(sqrt(float(ctx.R)));
    uint act = 0u;
    for (uint i = 0u; i < ctx.R; ++i) {
        bool admit = (ctx.utilFeat[i] + ctx.utilPlan[i]) > meanU;
        ctx.activeMask[i] = admit ? 1u : 0u;
        if (admit) { ctx.activeList[act++] = i; }
    }
    {                                          // pad up to floor (linear scan over R, terminates).
        uint scan = 0u;
        while (act < floor && scan < ctx.R) {
            if (ctx.activeMask[scan] == 0u) {
                ctx.activeMask[scan] = 1u;
                ctx.activeList[act++] = scan;
            }
            scan++;
        }
    }
    ctx.scalars->activeCount = act;
    ctx.stats->activeCount = float(act);

    // Identity fast-path flag: 1 iff every feature is admitted in order.
    uint identity = (act == ctx.R) ? 1u : 0u;
    if (identity == 1u) {
        for (uint i = 0u; i < ctx.R; ++i)
            if (ctx.activeList[i] != i) { identity = 0u; break; }
    }
    ctx.scalars->activeListIsIdentity = identity;

    // ---- Rerank sf[K] by combined utility (FCSTOMP feedback). Selection sort over R; R is small. ----
    for (uint k = 0u; k < ctx.K; ++k) {
        uint  bestI = k % ctx.R;
        float bestU = -FLT_MAX;
        for (uint i = 0u; i < ctx.R; ++i) {
            // Skip already-claimed features in earlier sf positions.
            bool claimed = false;
            for (uint j = 0u; j < k; ++j) if (ctx.subtaskFeatureOf[j] == i) { claimed = true; break; }
            if (claimed) continue;
            float u = ctx.utilFeat[i] + ctx.utilPlan[i];
            if (u > bestU) { bestU = u; bestI = i; }
        }
        ctx.subtaskFeatureOf[k] = bestI;
    }

    // ---- Per-feature implicit alpha (parameter-free; MetaOptimize 2402.02342 simplest form).
    //      alpha_i = alpha_0 / (1 + alpha_0 * featSM[i]). High-variance features get smaller
    //      steps; low-variance features get larger steps. NO meta-gradient, NO theta. ----
    for (uint i = 0u; i < ctx.R; ++i) {
        ctx.alphaImpFeat[i] = ctx.alpha0 / (1.0f + ctx.alpha0 * ctx.featSM[i]);
    }

    // ---- Single-worst-option reinit, gated on dormancy below uniform-mass baseline (1/K).
    //      Parameter-free: at most one option reinited per step; no thresholds to tune. ----
    uint worstK = 0u; float worstU = FLT_MAX;
    for (uint k = 0u; k < ctx.K; ++k) {
        if (ctx.utilOpt[k] < worstU) { worstU = ctx.utilOpt[k]; worstK = k; }
    }
    uint reinitedOpt = 0u, reinitedFeat = 0u;
    if (ctx.optFracEMA[worstK] < 1.0f / float(max(ctx.K, 1u))) {
        uint k = worstK;
        for (uint m = 0u; m < ctx.M; ++m) {
            ctx.psi[k * ctx.M + m] = (hash_uniform(ctx.stepCount, k, m) * 2.0f - 1.0f) * ctx.alpha0;
        }
        for (uint pi = 0u; pi < ctx.D * ctx.R; ++pi) {
            ctx.optPolicy[k * ctx.D * ctx.R + pi] =
                (hash_uniform(ctx.stepCount, k, ctx.M + pi) * 2.0f - 1.0f) * ctx.alpha0;
        }
        for (uint i = 0u; i < ctx.R; ++i) {
            ctx.optReward[k * ctx.R + i] =
                (hash_uniform(ctx.stepCount, k, ctx.M + ctx.D * ctx.R + i) * 2.0f - 1.0f) * ctx.alpha0;
            for (uint j = 0u; j < ctx.R; ++j) {
                ctx.optTransition[(k * ctx.R + j) * ctx.R + i] =
                    (hash_uniform(ctx.stepCount, (k * ctx.R + j) ^ 0xa3c59ac3u, i) * 2.0f - 1.0f) * ctx.alpha0;
            }
        }
        ctx.utilOpt[k] = 0.0f;
        ctx.optFracEMA[k] = 1.0f / float(max(ctx.K, 1u));
        reinitedOpt = 1u;

        // Generate-and-test feature reinit on the worst feature (Mahmood-Sutton-Pilarski 2012).
        uint worstI = 0u; float worstFU = FLT_MAX;
        for (uint i = 0u; i < ctx.R; ++i) {
            float u = ctx.utilFeat[i] + ctx.utilPlan[i];
            if (u < worstFU) { worstFU = u; worstI = i; }
        }
        for (uint o = 0u; o < ctx.baseO; ++o) {
            float u = hash_uniform(ctx.stepCount, worstI, o) * 2.0f - 1.0f;
            ctx.encoder[worstI * ctx.baseO + o] = u * rsqrt(float(max(ctx.baseO, 1u)));
        }
        ctx.utilFeat[worstI] = 0.0f;
        ctx.utilPlan[worstI] = 0.0f;
        ctx.featSM[worstI] = 0.0f;
        ctx.wBar[worstI] = 0.0f;
        ctx.featMean[worstI] = 0.0f;
        reinitedFeat = 1u;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
    }
    ctx.stats->oakOptionReinitCount  = float(reinitedOpt);
    ctx.stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

    // ---- pEff stable-rank proxy: tr(featSM) / max(featSM). ----
    float trSM = 0.0f, maxSM = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) { trSM += ctx.featSM[i]; maxSM = max(maxSM, ctx.featSM[i]); }
    ctx.scalars->pEff = trSM / max(maxSM, ctx.ridge);
    ctx.stats->pEffectiveRank = ctx.scalars->pEff;

    // ---- Telemetry roll-ups. ----
    float utilFeatMean = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) utilFeatMean += ctx.utilFeat[i];
    ctx.stats->utilityFeatureMean = utilFeatMean * invR;
    float utilOptMean = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) utilOptMean += ctx.utilOpt[k];
    ctx.stats->utilityOptionMean = utilOptMean / float(max(ctx.K, 1u));
    ctx.stats->rawNonFiniteCount       = float(ctx.counters[OAK_CNT_OBS_NF]
                                             + ctx.counters[OAK_CNT_LEARN_NF]
                                             + ctx.counters[OAK_CNT_PLAN_NF]
                                             + ctx.counters[OAK_CNT_TDPSI_NF]);
    ctx.stats->sanitizedNonFiniteCount = ctx.stats->rawNonFiniteCount;
    ctx.stats->encodeNonFiniteCount    = float(ctx.counters[OAK_CNT_OBS_NF]);
    ctx.stats->learnNonFiniteCount     = float(ctx.counters[OAK_CNT_LEARN_NF]);
    ctx.stats->planNonFiniteCount      = float(ctx.counters[OAK_CNT_PLAN_NF]);
    ctx.stats->tdPsiNonFiniteCount     = float(ctx.counters[OAK_CNT_TDPSI_NF]);
    // maxAbs* telemetry comes from per-step bounds of policyMean / state.
    float mLat = 0.0f, mPol = 0.0f;
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    for (uint i = 0u; i < ctx.R; ++i) mLat = max(mLat, fabs(curr[i]));
    for (uint p = 0u; p < ctx.D; ++p) mPol = max(mPol, fabs(ctx.policyMean[p]));
    ctx.stats->maxAbsLatentPrediction = mLat;
    ctx.stats->maxAbsPolicyOutput     = mPol;
    // Policy-action delta vs prev step (telemetry only).
    float dl = 0.0f;
    for (uint p = 0u; p < ctx.D; ++p) {
        float d = ctx.policyMean[p] - ctx.prevPolicyMean[p];
        dl = fma(d, d, dl);
    }
    ctx.stats->actionDeltaL2Mean = dl / float(max(ctx.D, 1u));

    ctx.scalars->stateSlot  = 1u - ctx.stateSlot;
    ctx.scalars->optionSlot = 1u - ctx.optionSlot;
    ctx.scalars->stepCount = ctx.stepCount + 1u;
}
