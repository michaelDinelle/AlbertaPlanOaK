// =================================================================================================
// OakKernel.metal — Orthodox Step 11 Oak agent. Single-stream (B=1), Metal 4 ICB.
//
// What the Plan demands (Sutton-Bowling-Pilarski 2022 arXiv:2208.11173, Step 11):
//   Perception                       → oak_encode (deep Fourier features over std_obs)
//   Value function                   → valueW (main GVF, differential, ρ̄-centered)
//   K subtask GVFs (RR-Subtasks)     → psi (cumulant=reward, stopping=feature attainment)
//   K option policies                → optPolicy (Design A keyboard, blended by chord)
//   K option reward + transition     → optReward, optTransition (RR-Subtasks eq. 17, dense)
//   Planning over option models      → oak_plan + oak_plan_reduce (eq. 19, async VI)
//   Option keyboard chord            → oak_meta (chord = softmax_k(ψ(s')))
//   Feedback / utility / remove-replace → oak_post_learn (utility EMAs) + oak_bk_admit_commit (G&T)
//   Average-reward (no discount)     → ρ̄ EMA (Reward Centering 2405.09999)
//   Per-feature step sizes (Step 1)  → MetaOptimize L-approximation in oak_post_learn
//                                       (Sharifnassab-Salehkaleybar-Sutton 2024 arXiv:2402.02342)
//
// Pipeline (5 fused kernels per step, Metal 4 ICB, fixed grid widths for the engine's lifetime):
//   bk_encode(+meta_predict) → plan → plan_reduce → learn_heads → post_learn(+G&T commit)
//
// Telemetry kept to what the host can't observe: per-phase non-finite counters and cumulative
// reinit counts. RewardRate, chatter, and scoreGrad L2 are host-derived from env signals.
//
// Optimization phase contract (see docs/roadmap.md): bit-identical kernel changes (kernel splits,
// per-thread parallelism without cross-thread reductions, pre-computed constants in KernelConfig)
// ship freely; FP-reorder optimizations (simd_sum, threadgroup reductions where input order
// changes) require per-seed parity proof on the 10×1.5M paired bench. simd_max is associative
// and is bit-identical for max-reductions; simd_sum is not.
// =================================================================================================
#include <metal_stdlib>
#include <float.h>
using namespace metal;

// Counters: per-step (cleared by host) and cumulative.
constant uint OAK_CNT_OBS_NF      = 0u;
constant uint OAK_CNT_LEARN_NF    = 1u;
constant uint OAK_CNT_PLAN_NF     = 2u;
constant uint OAK_CNT_REINIT_OPT  = 3u;   // cumulative (life of engine).
constant uint OAK_CNT_REINIT_FEAT = 4u;   // cumulative.
constant uint OAK_CNT_TDPSI_NF    = 5u;

struct KernelConfig {
    uint obsDim, rank, optionCount, baseObsDim, actionParamCount;
    // Tier I1 (Plan Step 3 partial): count of NON-INSTANTANEOUS obsTrace levels =
    // M − 1, where M = floor(log2(baseObsDim)) + 1. Layout: encoder row reads M·baseO
    // input columns (col 0..baseO-1 = std_obs; col m·baseO..(m+1)·baseO-1 = trace level m
    // for m=1..M-1). HC baseObsDim=17 → M=5 → baseObsTraceLevels=4.
    uint baseObsTraceLevels;

    // Param offsets: 7 parameter blocks. `phase` is the cos/sin offset for the
    // deep Fourier features encoder (Lewandowski-Schuurmans-Machado 2024,
    // arXiv:2410.20634). Encoder rows now span R/2 pre-activations, each producing a
    // (cos, sin) pair into state[0..R).
    uint pEncoder, pPhase, pValue, pPsi, pOptPolicy, pOptReward, pOptTransition, pOptDuration;

    // State arena offsets.
    uint sObservation, sObsMean, sObsSM;
    uint sState;                // 2 * R floats (slot toggle).
    uint sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi;
    uint sChord, sChordPrev;
    uint sUtilFeat, sUtilOpt;
    uint sBonusW;
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
    // Per-option reward-model TD targets, K floats. Written once by oak_plan lane 0
    // as reward - optReward[k]·prev, consumed by oak_learn_heads Head 3 for every
    // optReward[k,i] weight. Avoids K*R duplicate dot products in learn_heads.
    uint sOptRewardTd;
    uint sDurationTd, sDurationHat;
    // Per-option Q^k cache, K floats. Written by oak_plan (lane 0 of TG k), consumed by
    // oak_plan_reduce for the simd_max + planAdv soft-bound.
    uint sQOption;
    // Tier I1 (Plan Step 3 partial): per-obs-dim multi-timescale traces, (M−1)·baseO floats.
    // Layout: `obsTraceMulti[m·baseO + o]` for m=1..M−1, o=0..baseO-1. Updated by
    // oak_bk_encode Phase 1 with M−1 parallel EMAs at rates 1/τ_m where τ_m = 2^m
    // (so trace level m gets `(1 − 1/2^m)·t + (1/2^m)·std_obs[o]`). Consumed by Phase 2
    // Fourier projection alongside std_obs (encoder row width is M·baseO total — col 0..baseO-1
    // multiplies std_obs, col m·baseO..(m+1)·baseO-1 multiplies trace level m).
    // Replaces Tier C1's single obsTrace at fixed rate 1/baseO.
    uint sObsTraceMulti;
    // Per-option planning/update priority, K floats.
    uint sPrioOpt;
    uint sContinuation;
    uint sReplacementAgeMean;
    // UInt arena offsets.
    uint uCounters, uOptFeature, uFeatureAge, uOptAge, uFeatureEverUsed;

    // No tunable scalars. alpha is implicit (Kim arXiv:2510.06149) per-head from instantaneous
    // signal magnitudes; ridges are unit-Bayesian (var + 1, not var + tunable_epsilon).
};

struct KernelScalars {
    uint  stepCount;
    uint  stateSlot;
    float rho;                  // rho-bar (Reward Centering).
    float alphaPi;              // implicit alpha for policy weights.
    // Tier D2 (Plan Step 9): mean(prioOpt[K]), the threshold for Moore-Atkeson 1993
    // prioritized sweeping in oak_learn_heads Head 4. Computed once per step in
    // oak_plan_reduce via one hardware-simdgroup reduction, then broadcast-cached for every
    // Head 4 thread to read (avoids repeating the same K-element reduction).
    float meanPrio;
};

// Telemetry the host literally cannot derive from env signals: per-phase non-finite counters
// (kernel-internal numerical health) and cumulative reinit counts (Plan Step 11 evidence).
struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
    // Chord effective-K = 1 / Σ_k chord[k]² ∈ [1, K]. Independent of
    // continuation; pure concentration measure. K = uniform chord; 1 = one-hot.
    // Written by fused meta/predict lane 0 after chord renormalization.
    float chordEffectiveK;
    float rrBonusMean, psiStopGapMean, terminationRate, optionDurationMean;
    float neverUsedFeatureCount, replacementAgeMean, stateEffectiveRank;
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
    uint baseObsTraceLevels;                   // Tier I1: M − 1 (count of stored trace levels).
    uint stepCount, stateSlot;
    float rho, alphaPi, meanPrio;              // meanPrio: Tier D2 broadcast scalar, see KernelScalars.

    device float *encoder, *phase, *valueW, *psi, *optPolicy, *optReward, *optTransition, *optDuration;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *scoreGrad;
    device float *reward, *td, *tdPsi;
    device float *chord, *chordPrev;
    device float *utilFeat, *utilOpt;
    device float *bonusW;
    device float *planAdv;
    device float *alpha, *beta, *hMeta;        // MetaOptimize L-approximation per-feature buffers.
    device float *nTdMat;                      // Dense optTransition TD targets, K*R floats.
    device float *optRewardTd;                 // Option reward-model TD targets, K floats.
    device float *durationTd, *durationHat;     // Average-reward option duration targets/preds, K floats each.
    device float *qOption;                     // Per-option Q^k cache, K floats.
    device float *obsTraceMulti;               // Tier I1: per-obs-dim multi-timescale traces,
                                                // (M−1)·baseO floats. Layout: trace_m[o] at offset
                                                // m·baseO + o for m=1..M−1.
    device float *prioOpt;                     // Tier D2: per-option priority, K floats.
    device float *continuation;                 // RR-derived per-option continuation indicator/probability, K floats.
    device float *replacementAgeMean;           // EMA of age at replacement, one float.
    device uint  *counters;
    device uint  *optFeature, *featureAge, *optAge, *featureEverUsed;
    device KernelScalars *scalars;
    device KernelStats   *stats;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, device KernelStats *stats,
                    constant KernelConfig &cfg) {
    Ctx c;
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount;
    c.baseO = cfg.baseObsDim; c.D = cfg.actionParamCount;
    c.baseObsTraceLevels = cfg.baseObsTraceLevels;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
    c.rho = scalars->rho; c.alphaPi = scalars->alphaPi; c.meanPrio = scalars->meanPrio;

    c.encoder       = params + cfg.pEncoder;
    c.phase         = params + cfg.pPhase;
    c.valueW        = params + cfg.pValue;
    c.psi           = params + cfg.pPsi;
    c.optPolicy     = params + cfg.pOptPolicy;
    c.optReward     = params + cfg.pOptReward;
    c.optTransition = params + cfg.pOptTransition;
    c.optDuration   = params + cfg.pOptDuration;

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
    c.bonusW          = stateArena + cfg.sBonusW;
    c.planAdv         = stateArena + cfg.sPlanAdv;
    c.alpha           = stateArena + cfg.sAlpha;
    c.beta            = stateArena + cfg.sBeta;
    c.hMeta           = stateArena + cfg.sH;
    c.nTdMat          = stateArena + cfg.sNTdMat;
    c.optRewardTd     = stateArena + cfg.sOptRewardTd;
    c.durationTd      = stateArena + cfg.sDurationTd;
    c.durationHat     = stateArena + cfg.sDurationHat;
    c.qOption         = stateArena + cfg.sQOption;
    c.obsTraceMulti   = stateArena + cfg.sObsTraceMulti;
    c.prioOpt         = stateArena + cfg.sPrioOpt;
    c.continuation    = stateArena + cfg.sContinuation;
    c.replacementAgeMean = stateArena + cfg.sReplacementAgeMean;

    c.counters = uintArena + cfg.uCounters;
    c.optFeature = uintArena + cfg.uOptFeature;
    c.featureAge = uintArena + cfg.uFeatureAge;
    c.optAge = uintArena + cfg.uOptAge;
    c.featureEverUsed = uintArena + cfg.uFeatureEverUsed;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }
inline float decode_duration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }
inline uint encoder_input_width(Ctx ctx) {
    return (ctx.baseObsTraceLevels + 1u) * ctx.baseO + ctx.D + 1u;
}

// =================================================================================================
// 1+2. oak_bk_encode - fused obs normalization/traces + deep Fourier features.
//      Encoder row width is M·baseO + D + 1: observation traces plus previous action/reward.
// =================================================================================================
kernel void oak_bk_encode(
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

    // ================= PHASE 1: oak_bk_obs =================
    float invT = track_rate(ctx.stepCount, ctx.baseO);
    uint nfLocal = 0u;
    // Tier I1 (Plan Step 3 partial): M − 1 multi-timescale traces at structural rates
    // 1/τ_m where τ_m = 2^m for m=1..M−1. Each obs-thread owns its `o` and updates all
    // (M−1) trace levels for that obs dim per step. Cost: (M−1) reads + (M−1) writes per
    // obs-thread per step (HC: 4 EMAs per o per step, ~20 fmas total — negligible).
    constexpr uint OAK_MAX_OBS_CACHE = 512u;
    threadgroup float tg_stdObs[OAK_MAX_OBS_CACHE];
    for (uint o = tid; o < ctx.O; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nfLocal++; continue; }
        ctx.observationMean[o] += invT * (x - ctx.observationMean[o]);
        ctx.obsSM[o]           += invT * (x * x - ctx.obsSM[o]);
        // EMAs are over *standardized* obs using the just-updated stats. Shares std_obs's
        // scale (RFF-1 init then puts the trace levels in the same regime).
        float s = standardized_obs(x, ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        if (o < OAK_MAX_OBS_CACHE) tg_stdObs[o] = s;
        // Update each non-instantaneous trace level. τ_m = 2^m → rate 1/2^m via shift.
        for (uint m = 1u; m <= ctx.baseObsTraceLevels; ++m) {
            uint slot = (m - 1u) * ctx.baseO + o;   // obsTraceMulti layout: m−1 stored slots
            float rate = 1.0f / float(1u << m);     // 1/2^m structural; m=1 → 1/2, m=2 → 1/4, etc.
            ctx.obsTraceMulti[slot] += rate * (s - ctx.obsTraceMulti[slot]);
        }
    }

    // NF count aggregation across simdgroups: per-simdgroup simd_sum, then cross-simdgroup
    // sum in threadgroup memory. tg_nf is sized for up to 8 simdgroups (256-thread TG),
    // which covers the current R ∈ {32, 64, 128} range used by real envs.
    constexpr uint OAK_MAX_TG_SIMDGROUPS = 8u;
    constexpr uint OAK_MAX_RANK = 256u;
    threadgroup uint tg_nf[OAK_MAX_TG_SIMDGROUPS];
    threadgroup float tg_prevState[OAK_MAX_RANK];
    threadgroup float tg_currState[OAK_MAX_RANK];
    uint sgSum = simd_sum(nfLocal);
    uint sgId  = tid / 32u;
    if (tid % 32u == 0u && sgId < OAK_MAX_TG_SIMDGROUPS) tg_nf[sgId] = sgSum;

    // ================= BARRIER (phase 1 → phase 2, device memory fence) =================
    // Phase 2 reads obsMean, obsSM, obsTrace from device memory. Need threadgroup_barrier
    // with mem_device scope so all writes from phase 1 are visible across all lanes of the
    // R-wide threadgroup before any phase 2 read. (simdgroup_barrier would only fence within
    // one simdgroup.)
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint nfTotal = 0u;
        uint nSgs = (tgW + 31u) / 32u;
        for (uint s = 0u; s < nSgs && s < OAK_MAX_TG_SIMDGROUPS; ++s) nfTotal += tg_nf[s];
        ctx.counters[OAK_CNT_OBS_NF] = nfTotal;
    }

    // ================= PHASE 2: oak_encode =================
    uint feat = tid;
    if (feat >= ctx.R) return;
    uint halfR  = ctx.R >> 1u;                 // ctx.R is even (host-enforced).
    uint pre    = (feat < halfR) ? feat : feat - halfR;
    // Layout: M observation blocks followed by previous action/reward.
    uint M_levels = ctx.baseObsTraceLevels + 1u;
    uint obsWidth = M_levels * ctx.baseO;
    uint encRow = pre * encoder_input_width(ctx);
    float z = ctx.phase[pre];
    // Region m=0: std_obs, cached once in Phase 1.
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = (o < OAK_MAX_OBS_CACHE)
                ? tg_stdObs[o]
                : standardized_obs(ctx.observation[o], ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        z = fma(ctx.encoder[encRow + o], c, z);
    }
    // Regions m=1..M−1: trace levels at decay 1/2^m. Storage slot is (m−1)·baseO + o.
    for (uint m = 1u; m < M_levels; ++m) {
        uint storeBase = (m - 1u) * ctx.baseO;     // obsTraceMulti slot for level m, dim 0
        uint encBase   = m * ctx.baseO;            // encoder col offset for region m
        for (uint o = 0u; o < ctx.baseO; ++o) {
            z = fma(ctx.encoder[encRow + encBase + o], ctx.obsTraceMulti[storeBase + o], z);
        }
    }
    // Experience-oriented Plan input: previous action descriptor and reward.
    uint arBase = obsWidth;
    for (uint p = 0u; p < ctx.D; ++p)
        z = fma(ctx.encoder[encRow + arBase + p], ctx.policyMean[p], z);
    z = fma(ctx.encoder[encRow + arBase + ctx.D], ctx.reward[0], z);
    float v = (feat < halfR) ? cos(z) : sin(z);
    device const float *prevState = state_at(ctx, ctx.stateSlot);
    device float *currState = state_at(ctx, 1u - ctx.stateSlot);
    // safe_w-style preservation: on non-finite, keep the previous slot's value for this feature.
    float prevFeat = prevState[feat];
    float currFeat = isfinite(v) ? v : prevFeat;
    currState[feat] = currFeat;
    tg_prevState[feat] = prevFeat;
    tg_currState[feat] = currFeat;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ================= PHASE 3+4: fused meta/predict =================
    // Meta/predict fits inside the first hardware simdgroup of this same R-thread
    // encode TG. Keeping it here removes a dispatch + host memoryBarrier while preserving
    // the same program order: encode writes curr state, then meta writes chord/rho, then
    // predict writes td/tdPsi for learn_heads in the next kernel.
    if (tid < 32u) {
        uint k = tid;
        bool active = (k < ctx.K);
        threadgroup const float *prev = tg_prevState;
        threadgroup const float *curr = tg_currState;

        // Save chordPrev and compute the per-option ψ logit in one R-wide pass over curr.
        float zPsi = 0.0f;
        if (active) {
            ctx.chordPrev[k] = ctx.chord[k];
            uint basePsi = k * ctx.R;
            for (uint i = 0u; i < ctx.R; ++i)
                zPsi = fma(ctx.psi[basePsi + i], curr[i], zPsi);
        }

        float vCurr0 = 0.0f;
        if (k == 0u) for (uint i = 0u; i < ctx.R; ++i) vCurr0 = fma(ctx.valueW[i], curr[i], vCurr0);
        float vCurr = simd_broadcast(vCurr0, 0u);

        uint featIdx = active ? (ctx.optFeature[k] % ctx.R) : 0u;
        float zStop = active
            ? (vCurr + (ctx.bonusW[featIdx] - ctx.valueW[featIdx]) * curr[featIdx])
            : 0.0f;
        float cont = (active && isfinite(zPsi) && isfinite(zStop) && zPsi > zStop) ? 1.0f : 0.0f;
        float stopGap = active && isfinite(zPsi - zStop) ? (zPsi - zStop) : 0.0f;
        if (active) ctx.continuation[k] = cont;

        float qLogit = active && isfinite(ctx.qOption[k]) ? ctx.qOption[k] : zPsi;
        float lForMax = active ? qLogit : -FLT_MAX;
        float maxL    = simd_max(lForMax);
        float expL    = active ? exp(qLogit - maxL) : 0.0f;

        float Zexp = 0.0f;
        for (uint kk = 0u; kk < ctx.K; ++kk) {
            float vSoft = simd_shuffle(expL, kk);
            if (k == 0u) Zexp += vSoft;
        }
        Zexp = simd_broadcast(Zexp, 0u);
        float invZ = 1.0f / max(Zexp, FLT_MIN);
        float softK = expL * invZ;

        float chordPrevK = active ? ctx.chordPrev[k] : 0.0f;
        float terminatedMass = simd_sum(active ? ((1.0f - cont) * chordPrevK) : 0.0f);
        float chordRaw = active ? (cont * chordPrevK + terminatedMass * softK) : 0.0f;
        if (!isfinite(chordRaw)) chordRaw = 0.0f;
        float chordSum = simd_sum(chordRaw);
        float chordK = chordRaw / max(chordSum, FLT_MIN);
        if (active) {
            float oldChord = ctx.chord[k];
            ctx.chord[k] = isfinite(chordK) ? chordK : oldChord;
        }

        float chordK_norm = chordRaw / max(chordSum, FLT_MIN);
        float chordSqSum  = simd_sum(chordK_norm * chordK_norm);
        float stopGapSum  = simd_sum(stopGap);
        float termSum     = simd_sum(active ? (1.0f - cont) : 0.0f);
        if (k == 0u) {
            float effK = 1.0f / max(chordSqSum, FLT_MIN);
            ctx.stats->chordEffectiveK =
                isfinite(effK) ? effK : ctx.stats->chordEffectiveK;
            float invK = 1.0f / float(max(ctx.K, 1u));
            float gapMean = stopGapSum * invK;
            float termRate = termSum * invK;
            ctx.stats->psiStopGapMean = isfinite(gapMean) ? gapMean : ctx.stats->psiStopGapMean;
            ctx.stats->terminationRate = isfinite(termRate) ? termRate : ctx.stats->terminationRate;
        }

        if (k == 0u) {
            float ss = 0.0f, gg = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) ss = fma(curr[i], curr[i], ss);
            for (uint p = 0u; p < ctx.D; ++p) gg = fma(ctx.scoreGrad[p], ctx.scoreGrad[p], gg);
            ctx.scalars->alphaPi = 1.0f / (float(max(ctx.baseO, 1u)) + ss * gg * chordSqSum);

            float invT2 = track_rate(ctx.stepCount, ctx.baseO);
            ctx.scalars->rho += invT2 * (ctx.reward[0] - ctx.rho);
        }

        simdgroup_barrier(mem_flags::mem_device);

        float reward_c = ctx.reward[0] - ctx.scalars->rho;
        if (k == 0u) {
            float vPrev = 0.0f;
            for (uint i = 0u; i < ctx.R; ++i) vPrev = fma(ctx.valueW[i], prev[i], vPrev);
            float delta = reward_c + vCurr - vPrev;
            if (!isfinite(delta)) { ctx.counters[OAK_CNT_LEARN_NF] += 1u; delta = 0.0f; }
            ctx.td[0] = delta;
        }

        uint nfLane = 0u;
        if (active) {
            uint  basePsi  = k * ctx.R;
            float psiPrev = 0.0f;
            float psiNext = zPsi;
            for (uint i = 0u; i < ctx.R; ++i) {
                float w = ctx.psi[basePsi + i];
                psiPrev = fma(w, prev[i], psiPrev);
            }
            float dk = reward_c + cont * psiNext + (1.0f - cont) * zStop - psiPrev;
            if (!isfinite(dk)) { dk = 0.0f; nfLane = 1u; }
            ctx.tdPsi[k] = dk;
        }
        uint nfTotal = simd_sum(nfLane);
        if (k == 0u && nfTotal > 0u) ctx.counters[OAK_CNT_TDPSI_NF] += nfTotal;
    }
}

// =================================================================================================
// 5. oak_plan - parallel asynchronous VI over DENSE option models (RR-Subtasks eq. 17 + 19)
//    + Tier D2 (Plan Step 9, prioritized sweeping; Moore-Atkeson 1993, Plan footnote 13)
//    in-line per-option priority computation.
//
//    K·R-thread parallel: K threadgroups, R threads each (one TG per option k, one thread
//    per output dim j). Each (k, j) thread computes one row of the dense option transition,
//    fuses the curr and prev passes into a single R-load over `optTransition[k][j][:R]`,
//    and writes nTdMat[k·R + j] = target_j − n̂^k_j(prev) directly. Each thread stores its
//    `nHatCurr` into threadgroup memory AND its `n_td_sq = n_td · n_td` into a parallel
//    threadgroup array — both writes go BEFORE the same single barrier, zero added barriers.
//
//    Lane 0 of each TG runs a single R-loop that interleaves three accumulators:
//      vHat^k     = Σ_j valueW[j] · nHatCurr_j      (strict j=0..R-1 fma order, bit-identical)
//      ntdSqSum^k = Σ_j n_td_j^2                    (Tier D2 priority L2² of model TD error)
//      rHat^k     = Σ_i optReward[k][i] · curr[i]   (separate i-loop, R fma)
//    Lane 0 then writes:
//      qOption[k]  = isfinite(Q) ? Q : -FLT_MAX,                      (existing)
//      prioOpt[k]  = (isfinite(ntdSqSum) && ntdSqSum > 0)              (Tier D2 NEW)
//                    ? chord[k] · sqrt(ntdSqSum)
//                    : 0.0f                                            (NF-clamp: dirty option
//                                                                       becomes lowest-priority,
//                                                                       gets skipped, others learn).
//
//    The cross-K mean (meanPrio = mean_k prioOpt[k]) and Q^k max (bestQ = max_k Q^k) both
//    run in the follow-up `oak_plan_reduce` kernel (K threads, single simdgroup, simd_sum +
//    simd_max — both are associative/bit-id-safe for their respective ops).
//
//    Compute budget is parametric in K and R. Per-thread inner cost is 2R FMA plus local
//    priority work; per-TG lane-0 reductions are R-wide and preserve the documented fma order.
// =================================================================================================
kernel void oak_plan(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    device KernelStats   *stats       [[buffer(4)]],
    constant KernelConfig &cfg        [[buffer(5)]],
    uint  j     [[thread_position_in_threadgroup]],
    uint  k     [[threadgroup_position_in_grid]],
    uint  tgW   [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    if (k >= ctx.K) return;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    bool active = (j < ctx.R);
    float contK = clamp(ctx.continuation[k], 0.0f, 1.0f);

    // Per-thread dense-transition row. Same R-load fuses curr and prev passes; each thread
    // also writes its own slot in nTdMat (no contention). Stores nHatCurr (not the fma'd
    // contribution) into threadgroup memory so the reduction below keeps the original
    // single-thread kernel's exact `vHat = fma(valueW[j], nHatCurr, vHat)` sequence —
    // separate multiply-then-add would lose the fma's single-rounding and drift. Tier D2:
    // also squares the local n_td (still in register) so the per-thread cost is +1 mul.
    float nHatCurrLocal = 0.0f;
    float nTdSqLocal    = 0.0f;
    if (active) {
        float nHatPrev = 0.0f;
        uint  base = k * ctx.R * ctx.R + j * ctx.R;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.optTransition[base + i];
            nHatCurrLocal = fma(w, curr[i], nHatCurrLocal);
            nHatPrev      = fma(w, prev[i], nHatPrev);
        }
        float target = (1.0f - contK) * curr[j] + contK * nHatCurrLocal;
        float n_td = target - nHatPrev;
        ctx.nTdMat[k * ctx.R + j] = n_td;
        nTdSqLocal = n_td * n_td;     // Tier D2: in-register, free with ILP.
    }

    // Cross-thread reduction stage. Both arrays are written before a SINGLE barrier — adding
    // tg_ntdSq does not introduce a second barrier (Inv-10 zero-intelligence-hit on existing
    // kernel timings). OAK_MAX_RANK = 256 bounds this kernel's threadgroup storage.
    constexpr uint OAK_MAX_RANK = 256u;
    threadgroup float tg_nHatCurr[OAK_MAX_RANK];
    threadgroup float tg_ntdSq[OAK_MAX_RANK];
    tg_nHatCurr[j] = active ? nHatCurrLocal : 0.0f;
    tg_ntdSq[j]    = active ? nTdSqLocal    : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j == 0u) {
        // Two independent fma chains in the SAME R-loop body: Apple GPU's ILP scheduler can
        // dual-issue them, so ntdSqSum costs ~0 added latency on top of vHat. vHat's fma
        // sequence is preserved bit-identical to the prior kernel; ntdSqSum is a new
        // reduction (no prior bit-id baseline applies).
        float vHat     = 0.0f;
        float ntdSqSum = 0.0f;
        for (uint jj = 0u; jj < ctx.R; ++jj) {
            vHat     = fma(ctx.valueW[jj], tg_nHatCurr[jj], vHat);
            ntdSqSum = ntdSqSum + tg_ntdSq[jj];
        }
        float rHat = 0.0f;
        float rHatPrev = 0.0f;
        float lHatRaw = 0.0f;
        float lHatPrevRaw = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            uint idx = k * ctx.R + i;
            float w = ctx.optReward[idx];
            rHat     = fma(w, curr[i], rHat);
            rHatPrev = fma(w, prev[i], rHatPrev);
            float dw = ctx.optDuration[idx];
            lHatRaw     = fma(dw, curr[i], lHatRaw);
            lHatPrevRaw = fma(dw, prev[i], lHatPrevRaw);
        }
        float lHat = decode_duration(lHatRaw);
        float lHatPrev = decode_duration(lHatPrevRaw);
        float rTd = ctx.reward[0] + contK * rHat - rHatPrev;
        float dTd = 1.0f + contK * lHat - lHatPrev;
        float rTdSafe = isfinite(rTd) ? rTd : 0.0f;
        float dTdSafe = isfinite(dTd) ? dTd : 0.0f;
        ctx.optRewardTd[k] = rTdSafe;
        ctx.durationTd[k] = dTdSafe;
        ctx.durationHat[k] = lHat;

        float Q = rHat - ctx.scalars->rho * lHat + vHat;
        bool finite = isfinite(Q);
        ctx.qOption[k] = finite ? Q : -FLT_MAX;
        if (!finite) ctx.counters[OAK_CNT_PLAN_NF] += 1u;

        // Tier D2 priority write: chord-weighted L2 of model TD error. NF-clamp at write so
        // a dirty option becomes lowest-priority (skipped by the Head 4 gate) — failure
        // isolation, the rest of the K options learn normally. Plan-orthodox: prio depends
        // only on agent-internal signals (chord, nTdMat) — no env-knowledge, no tunable.
        float modelErr = ntdSqSum + rTdSafe * rTdSafe + dTdSafe * dTdSafe;
        bool prioFinite = isfinite(modelErr) && (modelErr > 0.0f);
        ctx.prioOpt[k]  = prioFinite ? (ctx.chord[k] * sqrt(modelErr)) : 0.0f;
    }
}

// =================================================================================================
// 5b. oak_plan_reduce - cross-option max over qOption[K] + Tier D2 (Plan Step 9) cross-option
//    mean over prioOpt[K], then the planAdv soft-bound `(bestQ − V(s)) / (1 + |bestQ − V(s)|)`.
//    K threads = single simdgroup. Both reductions are simdgroup-wide:
//      bestQ    = simd_max(Q)         — bit-identical (max is associative + commutative)
//      meanPrio = simd_sum(prio) / K  — new reduction, no prior bit-id baseline applies
//    Lane 0 computes V(curr), planAdv, AND writes meanPrio to scalars (broadcast-cached for
//    every oak_learn_heads Head 4 thread to read in one cycle — vs the alternative of
//    making every Head 4 thread compute the same K-element reduction).
// =================================================================================================
kernel void oak_plan_reduce(
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
    float Q = active ? ctx.qOption[k] : -FLT_MAX;
    if (!isfinite(Q)) Q = -FLT_MAX;
    float bestQ = simd_max(Q);

    // Tier D2 (Plan Step 9): cross-option mean of priority. simd_sum across one hardware
    // simdgroup is a single butterfly-add. All lanes participate (Metal spec); only
    // lane 0 writes the broadcast scalar. Inactive lanes contribute 0 → mean denominator is
    // K (the host's optionCount), not the active count.
    float prio    = active ? ctx.prioOpt[k] : 0.0f;
    if (!isfinite(prio)) prio = 0.0f;
    float prioSum = simd_sum(prio);
    float dur = active ? ctx.durationHat[k] : 0.0f;
    if (!isfinite(dur)) dur = 0.0f;
    float durSum = simd_sum(dur);

    if (k == 0u) {
        device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
        float vCurr = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) vCurr = fma(ctx.valueW[i], curr[i], vCurr);
        float advRaw = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
        // x/(1+|x|) self-bound — the standard rational soft-clamp from Sutton's gradient-TD
        // line of work. Image is (-1, 1), continuous + monotone, no tunable scale.
        float adv = advRaw / (1.0f + fabs(advRaw));
        ctx.planAdv[0] = isfinite(adv) ? adv : ctx.planAdv[0];   // safe_w-style preserve on NF.

        // Tier D2: write meanPrio to the broadcast scalar. NF-clamp at write so the Head 4
        // gate has a well-defined comparison. Edge case (K = 0 should never happen but defend):
        // divide by max(K, 1).
        float invK = 1.0f / float(max(ctx.K, 1u));
        float meanPrio = prioSum * invK;
        ctx.scalars->meanPrio = isfinite(meanPrio) ? meanPrio : 0.0f;
        float durMean = durSum * invK;
        ctx.stats->optionDurationMean = isfinite(durMean) ? durMean : ctx.stats->optionDurationMean;
    }
}

// 6. oak_learn_heads - 1-step TD updates on 6 trainable linear heads.
//      0: valueW[R]              -> alpha[i] * (td + planAdv) * prev[i]   (planAdv added HERE ONLY)
//      1: psi[K*R]               -> alpha[i] * tdPsi[k] * prev[i] * max(chordPrev[k], 1/K)
//                                   Tier D3 (Plan Step 10 "conditional on the option, off-policy"):
//                                   chord-weighted attribution lets option k's psi specialize to
//                                   its own activation instead of converging to the chord mean.
//                                   1/K floor prevents self-reinforcing starvation when chord[k]→0.
//      2: optPolicy[K*D*R]       -> alphaPi * relevance(i) * td * scoreGrad[d] * prev[i] * chordPrev[k]
//      3: optReward[K*R]         -> alpha[i] * optRewardTd[k] * prev[i] * chordPrev[k]
//      4: optTransition[K*R*R]   -> alpha[i] * nTdMat[k*R+j] * prev[i] * chordPrev[k]
//                                   (DENSE: n_hat^k_j(s) = sum_i w^k_{j,i} * s[i], RR-Subtasks eq. 17;
//                                    n_td uses the RR stopping-aware recursive target from oak_plan)
//      5: optDuration[K*R]       -> alpha[i] * durationTd[k] * prev[i] * chordPrev[k]
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
    uint nOD  = ctx.K * ctx.R;
    if (gid >= nVal + nPsi + nPi + nOR + nOT + nOD) return;

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
        // Tier D3 (Plan Step 10 off-policy attribution): weight the gradient by the
        // chord mass option k carried at the experience source state (prev). Before D3,
        // every psi_k received full-rate updates from every step, so all K GVFs converged
        // toward the chord-level value — options had no identity. With chord-weighted
        // updates, option k's psi specializes to what happens when k is actually active.
        // The 1/K floor is a structural minimum that prevents self-reinforcing starvation
        // (chord[k]→0 → no updates → stale psi → chord stays 0). K is the Plan-defined
        // option count (host-capped at simdgroup width 32), so 1/K is not a tunable.
        uint id = gid - nVal;
        uint k = id / ctx.R, i = id % ctx.R;
        float chordW = max(ctx.chordPrev[k], 1.0f / float(max(ctx.K, 1u)));
        float w = ctx.psi[id];
        ctx.psi[id] = safe_w(w, w + ctx.alpha[i] * ctx.tdPsi[k] * prev[i] * chordW);
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
    if (gid < nVal + nPsi + nPi + nOR) {                       // Head 3: option reward model
        uint id = gid - nVal - nPsi - nPi;
        uint k = id / ctx.R, i = id % ctx.R;
        float r_td = ctx.optRewardTd[k];
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optReward[id];
        ctx.optReward[id] = safe_w(w, w + ctx.alpha[i] * r_td * grad);
        return;
    }
    if (gid < nVal + nPsi + nPi + nOR + nOT) {                 // Head 4: dense optTransition
        // (RR-Subtasks eq. 17). Per-weight gradient:
        //   ∂ ½(target_j − n_hat^k_j(prev))² / ∂ w^k_{j,i} = -n_td_j · prev[i]
        // n_td precomputed into ctx.nTdMat[k*R + j] by oak_plan (one shared inner sum per (k, j)
        // serves R weights along the i axis). chordPrev[k] is the on-chord weight (importance
        // ratio per RR-Subtasks eq. 17). α[i] (not α[j]) because the gradient direction involves
        // prev[i] — the i axis is the input dimension whose per-feature relevance gates the update.
        //
        // Tier D2 (Plan Step 9, Moore-Atkeson 1993 prioritized sweeping; Plan footnote 13):
        // gate the update by `prioOpt[k] >= meanPrio` so only above-mean-priority options
        // refresh their dense transition models this step. Below-mean options inherit their
        // existing weights via the implicit no-op (no safe_w call → no write → preserve).
        // Inv-SIMD: thread layout `(k, j, i)` with `R*R = 4096 ≫ 32` → all 32 simdgroup lanes
        // share the same `k` → uniform gate, zero divergence cost.
        // Fail-safe: if meanPrio is non-finite (shouldn't happen — oak_plan_reduce NF-clamps
        // at write), the gate falls through to "do the update" rather than starving Head 4.
        uint id  = gid - nVal - nPsi - nPi - nOR;
        uint k   = id / (ctx.R * ctx.R);
        if (isfinite(ctx.meanPrio) && ctx.prioOpt[k] < ctx.meanPrio) return;
        uint rem = id - k * ctx.R * ctx.R;
        uint j   = rem / ctx.R;
        uint i   = rem % ctx.R;
        float n_td = ctx.nTdMat[k * ctx.R + j];
        float grad = ctx.chordPrev[k] * prev[i];
        float w    = ctx.optTransition[id];
        ctx.optTransition[id] = safe_w(w, w + ctx.alpha[i] * n_td * grad);
        return;
    }
    if (gid < nVal + nPsi + nPi + nOR + nOT + nOD) {          // Head 5: optDuration
        uint id = gid - nVal - nPsi - nPi - nOR - nOT;
        uint k = id / ctx.R, i = id % ctx.R;
        float grad = ctx.chordPrev[k] * prev[i];
        float w = ctx.optDuration[id];
        ctx.optDuration[id] = safe_w(w, w + ctx.alpha[i] * ctx.durationTd[k] * grad);
        return;
    }
}

inline void oak_bk_admit_commit_body(
    device float         *params,
    device float         *stateArena,
    device uint          *uintArena,
    device KernelScalars *scalars,
    device KernelStats   *stats,
    constant KernelConfig &cfg,
    uint tid,
    uint tgW,
    threadgroup uint &worstK_tg,
    threadgroup uint &worstI_tg,
    threadgroup uint &stepCount_tg,
    threadgroup uint &preA_tg,
    threadgroup uint &preB_tg,
    threadgroup uint &compose_tg,
    threadgroup uint &candidateI_tg,
    threadgroup uint &replaceOpt_tg,
    threadgroup uint &replaceFeat_tg);

// =================================================================================================
// 8. oak_post_learn - three concurrent passes after learn_heads writes the new weights:
//      gid in [0, R)              -> per-feature GraMa utility EMA + MetaOptimize L-approx
//      gid in [R, R+K)            -> per-option |tdPsi|² utility EMA
//      gid in [R+K, R+K+D·K)      -> action mu_p = sum_k chord[k] · (W_pi[k][p] · s')
//    Folding all three into one kernel saves a launch.
//
//    PERF NOTE on region 3 (action). Old: one thread per action dim, each doing K·R
//    sequential fma in one simdgroup with most hardware lanes idle.
//    New: D simdgroups of hardware-lane threads. Lane k of TG p computes its own
//    `dot_k = Σ_i optPolicy[(k·D+p)·R+i] · curr[i]` in parallel. Lane 0 then reads each
//    lane's dot via `simd_shuffle(dot, kk)` and accumulates `mu = fma(chord[kk], d, mu)`
//    in the SAME `kk = 0..K-1` order as the prior single-thread code — bit-identical FP
//    (Inv-10 zero-intelligence-hit preserved by construction). K may be below the hardware
//    lane width; lanes K..31 stay inactive and every action dim still starts at lane 0.
//
//    Region 1 still computes `-log(baseO)` and `-log(R·baseO)` per thread per step. Tried
//    moving these to host-precomputed scalars but CPU `log()` and GPU `log()` are NOT
//    bit-identical for the same input (different transcendental approximations) — broke
//    Inv-10 (chatter shifted +2.8% in single-process diagnostic). The 2 GPU log() per
//    thread are ~10 cycles in 2 simdgroups in parallel = negligible (~0.01 µs/step).
// =================================================================================================
kernel void oak_post_learn(
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
    uint actBase   = ((ctx.R + ctx.K + 31u) / 32u) * 32u;
    uint actStride = 32u;
    uint actCount  = ctx.D * actStride;
    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    float invT = track_rate(ctx.stepCount, ctx.baseO);
    if (tid < ctx.R) {
        uint i = tid;
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
        // α-cap = 1/baseO (safe-update ceiling) ⇔ β-cap = −log baseO.
        // α-floor = 1/(R·baseO) (always-learning guarantee: BWH/temporal-uniformity says
        // OaK never stops adapting any feature; even after a streak of inconsistent gradients
        // the per-feature step floors at 1/(R·baseO) so the soft mechanism keeps tracking
        // alongside the hard G&T mechanism). At the floor, summed across R features the
        // total per-step learning rate equals the original global a = 1/baseO; at the cap,
        // it can amplify up to R · 1/baseO when every feature is fully informative.
        float aOld   = ctx.alpha[i];
        float hOld   = ctx.hMeta[i];
        float dw     = aOld * td * xi;
        float invR   = 1.0f / float(max(ctx.R, 1u));
        float gamma  = 1.0f - invR;
        float eta    = invR / float(max(ctx.baseO, 1u));
        float betaCap   = -log(float(max(ctx.baseO, 1u)));
        float betaFloor = -log(float(max(ctx.R * ctx.baseO, 1u)));
        float newBeta = ctx.beta[i] + eta * td * xi * hOld;
        if (!isfinite(newBeta)) newBeta = ctx.beta[i];
        newBeta = clamp(newBeta, betaFloor, betaCap);
        ctx.hMeta[i] = gamma * hOld + dw;
        ctx.beta[i]  = newBeta;
        ctx.alpha[i] = exp(newBeta);
        float vw = ctx.valueW[i];
        if (isfinite(vw) && vw > ctx.bonusW[i]) ctx.bonusW[i] = vw;
    } else if (tid >= ctx.R && tid < ctx.R + ctx.K) {
        uint k = tid - ctx.R;
        float dk = ctx.tdPsi[k];
        ctx.utilOpt[k] += invT * (dk * dk - ctx.utilOpt[k]);
    }

    // Action region: D simdgroups, one hardware-width lane group per action dim.
    // tid layout after padded actBase: (tid - actBase) = p · 32 + lane. actBase is rounded
    // up to a simdgroup boundary so each p starts at lane 0 even when K < 32.
    if (tid >= actBase && tid < actBase + actCount) {
        uint actId = tid - actBase;
        uint p     = actId / actStride;
        uint k     = actId - p * actStride;

        // Each lane computes its own dot product in parallel (R fma).
        float dot = 0.0f;
        if (k < ctx.K) {
            for (uint i = 0u; i < ctx.R; ++i)
                dot = fma(ctx.optPolicy[(k * ctx.D + p) * ctx.R + i], curr[i], dot);
        }

        // Lane 0 reads each lane's dot via simd_shuffle and applies chord with fma in the
        // ORIGINAL kk=0..K-1 sequence — bit-identical to the prior single-thread loop's
        // `mu = fma(chord[kk], dot, mu)`. ALL active action lanes execute the shuffle.
        float mu = 0.0f;
        for (uint kk = 0u; kk < ctx.K; ++kk) {
            float d = simd_shuffle(dot, kk);
            if (k == 0u) mu = fma(ctx.chord[kk], d, mu);
        }
        if (k == 0u) {
            float prevMu = ctx.policyMean[p];
            ctx.policyMean[p] = isfinite(mu) ? mu : prevMu;
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    threadgroup uint  worstK_tg    = 0u;
    threadgroup uint  worstI_tg    = 0u;
    threadgroup uint  stepCount_tg = 0u;
    threadgroup uint  preA_tg      = 0u;
    threadgroup uint  preB_tg      = 0u;
    threadgroup uint  compose_tg   = 0u;
    threadgroup uint  candidateI_tg = 0u;
    threadgroup uint  replaceOpt_tg = 0u;
    threadgroup uint  replaceFeat_tg = 0u;
    oak_bk_admit_commit_body(params, stateArena, uintArena, scalars, stats, cfg, tid, tgW,
                             worstK_tg, worstI_tg, stepCount_tg, preA_tg, preB_tg,
                             compose_tg, candidateI_tg, replaceOpt_tg, replaceFeat_tg);
}

// =================================================================================================
// 10. oak_bk_admit_commit - Oak feedback: every-step assessment plus age-gated remove/replace
//     for option bundles and unused feature candidates (Plan Step 11 + Step 2 generate-and-test).
//
//     Single threadgroup, multi-thread:
//       Phase 1 (lane 0):  find worstK + worstI via linear scan; sample two parent
//                          pre-activations from paired utility mass (Tier K2); bump
//                          cumulative counters. Writes worstK, worstI, preA/preB, and the
//                          compose/fallback flag into threadgroup memory.
//       Barrier            (mem_threadgroup)
//       Phase 2 (all lanes): cooperative parallel reseed of the worstK weight slices
//                          (psi, optReward, optPolicy, optTransition) plus the worstI
//                          Fourier encoder row, distributed by `tid`. Encoder row is either
//                          a stochastic composition of two sampled parents or the prior
//                          fresh-random fallback when utility mass is insufficient.
//       Barrier
//       Phase 3 (lane 0):  scalar resets (utilOpt, utilFeat, phase, MetaOptimize) +
//                          slot toggle + stepCount increment + stats surface.
//
//     Lane 0 captures `stepCount` BEFORE the increment so all reseed hashes match the
//     prior single-threaded kernel byte-for-byte (deterministic seed equivalence).
//     One Metal threadgroup keeps occupancy modest; the dominant optTransition slice is
//     K·R² writes, distributed cooperatively across active lanes.
// =================================================================================================
inline void oak_bk_admit_commit_body(
    device float         *params,
    device float         *stateArena,
    device uint          *uintArena,
    device KernelScalars *scalars,
    device KernelStats   *stats,
    constant KernelConfig &cfg,
    uint tid,
    uint tgW,
    threadgroup uint &worstK_tg,
    threadgroup uint &worstI_tg,
    threadgroup uint &stepCount_tg,
    threadgroup uint &preA_tg,
    threadgroup uint &preB_tg,
    threadgroup uint &compose_tg,
    threadgroup uint &candidateI_tg,
    threadgroup uint &replaceOpt_tg,
    threadgroup uint &replaceFeat_tg)
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);

    // Phase 1: lane 0 ages the registries, chooses eligible replacements, and samples
    // K2 parent rows only if a feature row is actually being replaced.
    if (tid == 0u) {
        uint stepC = ctx.stepCount;
        stepCount_tg = stepC;
        uint halfR = ctx.R >> 1u;

        for (uint i = 0u; i < ctx.R; ++i) if (ctx.featureAge[i] < 0xffffffffu) ctx.featureAge[i] += 1u;
        for (uint k = 0u; k < ctx.K; ++k) if (ctx.optAge[k] < 0xffffffffu) ctx.optAge[k] += 1u;

        float admitRate = 1.0f / float(max(ctx.R, 1u));
        uint tryOpt = (hash_uniform(stepC, ctx.R, 0x51f15eedu) < admitRate) ? 1u : 0u;
        uint tryFeat = (hash_uniform(stepC, ctx.R, 0xa17c9e3du) < admitRate) ? 1u : 0u;

        uint wK = 0u, replaceOpt = 0u;
        float worstOptScore = FLT_MAX;
        if (tryOpt != 0u) {
            for (uint k = 0u; k < ctx.K; ++k) {
                if (ctx.optAge[k] < ctx.R) continue;
                float s = ctx.utilOpt[k] + ctx.prioOpt[k];
                if (!isfinite(s)) s = -FLT_MAX;
                if (s < worstOptScore) { worstOptScore = s; wK = k; replaceOpt = 1u; }
            }
        }

        uint candidateI = replaceOpt ? (ctx.optFeature[wK] % ctx.R) : 0u;
        if (replaceOpt != 0u) {
            float bestFresh = -FLT_MAX;
            uint haveFresh = 0u;
            for (uint i = 0u; i < ctx.R; ++i) {
                bool assigned = false;
                for (uint k = 0u; k < ctx.K; ++k) assigned = assigned || ((ctx.optFeature[k] % ctx.R) == i);
                if (assigned || ctx.featureAge[i] < ctx.R || ctx.featureEverUsed[i] != 0u) continue;
                float u = ctx.utilFeat[i];
                if (isfinite(u) && u > bestFresh) { bestFresh = u; candidateI = i; haveFresh = 1u; }
            }
            if (haveFresh == 0u) {
                for (uint i = 0u; i < ctx.R; ++i) {
                    bool assigned = false;
                    for (uint k = 0u; k < ctx.K; ++k) assigned = assigned || ((ctx.optFeature[k] % ctx.R) == i);
                    if (assigned || ctx.featureAge[i] < ctx.R) continue;
                    float u = ctx.utilFeat[i];
                    if (isfinite(u) && u > bestFresh) { bestFresh = u; candidateI = i; }
                }
            }
        }

        uint wI = 0u, replaceFeat = 0u;
        float worstPair = FLT_MAX;
        if (tryFeat != 0u) {
            for (uint p = 0u; p < halfR; ++p) {
                bool pairAssigned = false;
                for (uint k = 0u; k < ctx.K; ++k) {
                    uint of = ctx.optFeature[k] % ctx.R;
                    pairAssigned = pairAssigned || (of == p) || (of == p + halfR);
                }
                uint candPre = (candidateI < halfR) ? candidateI : candidateI - halfR;
                if (pairAssigned || (replaceOpt != 0u && p == candPre)) continue;
                if (ctx.featureAge[p] < ctx.R && ctx.featureAge[p + halfR] < ctx.R) continue;
                float pairU = ctx.utilFeat[p] + ctx.utilFeat[p + halfR];
                if (!isfinite(pairU)) pairU = -FLT_MAX;
                if (pairU < worstPair) {
                    worstPair = pairU;
                    wI = (ctx.utilFeat[p] <= ctx.utilFeat[p + halfR]) ? p : p + halfR;
                    replaceFeat = 1u;
                }
            }
        }

        float ageSum = 0.0f;
        float ageN = 0.0f;
        if (replaceOpt != 0u) { ageSum += float(ctx.optAge[wK]); ageN += 1.0f; }
        if (replaceFeat != 0u) { ageSum += float(ctx.featureAge[wI]); ageN += 1.0f; }
        if (ageN > 0.0f) {
            float r = track_rate(stepC, ctx.R);
            ctx.replacementAgeMean[0] += r * ((ageSum / ageN) - ctx.replacementAgeMean[0]);
        }

        worstK_tg = wK;
        worstI_tg = wI;
        candidateI_tg = candidateI;
        replaceOpt_tg = replaceOpt;
        replaceFeat_tg = replaceFeat;

        uint preTarget = (wI < halfR) ? wI : wI - halfR;
        uint preA = 0u;
        uint preB = 0u;
        uint compose = 0u;
        float totalPairUtil = 0.0f;
        uint positiveParents = 0u;

        if (replaceFeat != 0u) {
            for (uint p = 0u; p < halfR; ++p) {
                if (p == preTarget) continue;
                float paired = ctx.utilFeat[p] + ctx.utilFeat[p + halfR];
                if (isfinite(paired) && paired > 0.0f) {
                    totalPairUtil += paired;
                    positiveParents += 1u;
                }
            }

            if (positiveParents >= 2u && isfinite(totalPairUtil) && totalPairUtil > 0.0f) {
                float uA = hash_uniform(stepC, preTarget, 0x6b325541u) * totalPairUtil;
                uint haveA = 0u;
                float pairUtilA = 0.0f;
                float runningA = 0.0f;
                for (uint p = 0u; p < halfR; ++p) {
                    if (p == preTarget) continue;
                    float paired = ctx.utilFeat[p] + ctx.utilFeat[p + halfR];
                    if (!(isfinite(paired) && paired > 0.0f)) continue;
                    runningA += paired;
                    if (runningA > uA) {
                        preA = p;
                        pairUtilA = paired;
                        haveA = 1u;
                        break;
                    }
                }

                float remainingPairUtil = totalPairUtil - pairUtilA;
                if (haveA != 0u && isfinite(remainingPairUtil) && remainingPairUtil > 0.0f) {
                    float uB = hash_uniform(stepC, preTarget, 0x2f8c7a19u) * remainingPairUtil;
                    float runningB = 0.0f;
                    for (uint p = 0u; p < halfR; ++p) {
                        if (p == preTarget || p == preA) continue;
                        float paired = ctx.utilFeat[p] + ctx.utilFeat[p + halfR];
                        if (!(isfinite(paired) && paired > 0.0f)) continue;
                        runningB += paired;
                        if (runningB > uB) {
                            preB = p;
                            compose = 1u;
                            break;
                        }
                    }
                }
            }
        }

        preA_tg = preA;
        preB_tg = preB;
        compose_tg = compose;
        if (replaceOpt != 0u) ctx.counters[OAK_CNT_REINIT_OPT]  += 1u;
        if (replaceFeat != 0u) ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint worstK = worstK_tg;
    uint worstI = worstI_tg;
    uint stepC  = stepCount_tg;
    uint preA   = preA_tg;
    uint preB   = preB_tg;
    uint compose = compose_tg;
    uint candidateI = candidateI_tg;
    uint replaceOpt = replaceOpt_tg;
    uint replaceFeat = replaceFeat_tg;
    uint halfR  = ctx.R >> 1u;
    uint pre    = (worstI < halfR) ? worstI : worstI - halfR;
    float invSqrtR = rsqrt(float(max(ctx.R, 1u)));

    // Phase 2: cooperative parallel reseed. Sub-grids stacked end-to-end into one virtual
    // index space; each lane strides through `total` writes. Branch-per-region is cheap
    // because all lanes in a simdgroup almost always hit the same region (consecutive id).
    uint nPsi = ctx.R;
    uint nOR  = ctx.R;
    uint nPi  = ctx.D * ctx.R;
    uint nOT  = ctx.R * ctx.R;
    uint nOD  = ctx.R;
    // Tier I1: encoder rows are M·baseO (std_obs ‖ M−1 trace levels). M = baseObsTraceLevels + 1.
    uint nEnc = encoder_input_width(ctx);
    uint o0   = 0u;
    uint o1   = o0 + nPsi;
    uint o2   = o1 + nOR;
    uint o3   = o2 + nPi;
    uint o4   = o3 + nOT;
    uint o5   = o4 + nOD;
    uint optTotal = (replaceOpt != 0u) ? o5 : 0u;
    uint total = optTotal + ((replaceFeat != 0u) ? nEnc : 0u);

    for (uint id = tid; id < total; id += tgW) {
        if (id < optTotal && id < o1) {                             // psi[worstK][i]
            uint i = id - o0;
            ctx.psi[worstK * ctx.R + i] =
                (hash_uniform(stepC, worstK, i) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < optTotal && id < o2) {                      // optReward[worstK][i]
            uint i = id - o1;
            ctx.optReward[worstK * ctx.R + i] =
                (hash_uniform(stepC, worstK, ctx.R + i) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < optTotal && id < o3) {                      // optPolicy[worstK][:D*R]
            uint pi = id - o2;
            ctx.optPolicy[worstK * ctx.D * ctx.R + pi] =
                (hash_uniform(stepC, worstK, 3u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < optTotal && id < o4) {                      // optTransition[worstK][:R*R]
            uint pi = id - o3;
            ctx.optTransition[worstK * ctx.R * ctx.R + pi] =
                (hash_uniform(stepC, worstK, 4u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < optTotal && id < o5) {                      // optDuration[worstK][i]
            uint i = id - o4;
            ctx.optDuration[worstK * ctx.R + i] = 0.0f;
        } else {                                                     // encoder[pre][:M·baseO]
            // Tier K2: compose the replacement row from two sampled parent pre-activations
            // without replacement, weighted by paired utility mass. Fallback stays the prior
            // fresh-random bandwidth-1 reseed when there are fewer than two positive parents.
            uint o = id - optTotal;
            uint rowStride = encoder_input_width(ctx);
            if (compose != 0u) {
                float rA = ctx.encoder[preA * rowStride + o];
                float rB = ctx.encoder[preB * rowStride + o];
                ctx.encoder[pre * rowStride + o] =
                    (rA + rB) * 0.7071067811865475f;                 // 1 / sqrt(2), structural.
            } else {
                float u = hash_uniform(stepC, pre, o + 0xdeadbeefu) * 2.0f - 1.0f;
                ctx.encoder[pre * rowStride + o] = u;                // RFF bandwidth-1.
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: lane 0 does the small scalar resets + slot toggle + stepCount increment +
    // surface stats. Done here (post-reseed) so the increment doesn't race the hash reads.
    if (tid == 0u) {
        float betaInit = -log(float(max(ctx.baseO, 1u)));
        float aInit    = exp(betaInit);
        if (replaceOpt != 0u) {
            ctx.utilOpt[worstK] = 0.0f;
            ctx.optAge[worstK] = 0u;
            ctx.optFeature[worstK] = candidateI;
            ctx.featureEverUsed[candidateI] = 1u;
            ctx.continuation[worstK] = 0.0f;
            ctx.durationTd[worstK] = 0.0f;
            ctx.durationHat[worstK] = 1.0f;
            ctx.psi[worstK * ctx.R + candidateI] =
                safe_w(ctx.psi[worstK * ctx.R + candidateI],
                       ctx.psi[worstK * ctx.R + candidateI] + invSqrtR);
        }

        if (replaceFeat != 0u) {
            float u = hash_uniform(stepC, pre, 0xfeedf00du);
            ctx.phase[pre] = u * 6.283185307179586f;                 // 2π, structural.
            ctx.utilFeat[pre]         = 0.0f;
            ctx.utilFeat[pre + halfR] = 0.0f;
            ctx.featureAge[pre] = 0u;
            ctx.featureAge[pre + halfR] = 0u;
            ctx.featureEverUsed[pre] = 0u;
            ctx.featureEverUsed[pre + halfR] = 0u;
            ctx.bonusW[pre] = ctx.valueW[pre];
            ctx.bonusW[pre + halfR] = ctx.valueW[pre + halfR];
            ctx.beta[pre]          = betaInit;
            ctx.beta[pre + halfR]  = betaInit;
            ctx.alpha[pre]         = aInit;
            ctx.alpha[pre + halfR] = aInit;
            ctx.hMeta[pre]         = 0.0f;
            ctx.hMeta[pre + halfR] = 0.0f;
        }

        device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
        float bonusGap = 0.0f, sum2 = 0.0f, sum4 = 0.0f;
        uint never = 0u;
        for (uint i = 0u; i < ctx.R; ++i) {
            bonusGap += fabs(ctx.bonusW[i] - ctx.valueW[i]);
            if (ctx.featureEverUsed[i] == 0u) never += 1u;
            float x = curr[i];
            float x2 = x * x;
            sum2 += x2;
            sum4 += x2 * x2;
        }
        float invR = 1.0f / float(max(ctx.R, 1u));
        ctx.stats->rrBonusMean = isfinite(bonusGap) ? bonusGap * invR : ctx.stats->rrBonusMean;
        ctx.stats->neverUsedFeatureCount = float(never);
        float effRank = (sum4 > 0.0f) ? ((sum2 * sum2) / sum4) : 0.0f;
        ctx.stats->stateEffectiveRank = isfinite(effRank) ? effRank : ctx.stats->stateEffectiveRank;
        ctx.stats->replacementAgeMean = ctx.replacementAgeMean[0];
        ctx.stats->encodeNonFiniteCount    = float(ctx.counters[OAK_CNT_OBS_NF]);
        ctx.stats->learnNonFiniteCount     = float(ctx.counters[OAK_CNT_LEARN_NF]);
        ctx.stats->planNonFiniteCount      = float(ctx.counters[OAK_CNT_PLAN_NF]);
        ctx.stats->tdPsiNonFiniteCount     = float(ctx.counters[OAK_CNT_TDPSI_NF]);
        ctx.stats->oakOptionReinitCount    = float(ctx.counters[OAK_CNT_REINIT_OPT]);
        ctx.stats->oakFeatureReinitCount   = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

        ctx.scalars->stateSlot = 1u - ctx.stateSlot;
        ctx.scalars->stepCount = stepC + 1u;
    }
}
