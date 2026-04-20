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
// Pipeline (9 kernels per step, Metal 4 ICB, fixed grid widths for the engine's lifetime):
//   bk_obs → encode → meta → predict → plan → plan_reduce → learn_heads → post_learn → bk_admit_commit
//
// Telemetry kept to the bare minimum the host can't observe: per-phase non-finite counters and
// cumulative reinit counts. Everything else (rewardRate, chatter, scoreGrad L2) is host-derived
// from env signals.
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
    // Per-option Q^k cache, K floats. Written by oak_plan (lane 0 of TG k), consumed by
    // oak_plan_reduce for the simd_max + planAdv soft-bound.
    uint sQOption;
    // Tier C1 (Plan Step 3 explicit): per-obs-dim low-pass trace of std_obs, O floats.
    // Updated by oak_bk_obs each step as `t_o ← (1 − 1/baseO)·t_o + (1/baseO)·std_obs[o]`;
    // consumed by oak_encode as the second half of every Fourier pre-activation's input
    // (encoder row width is 2·baseO — first baseO multiplies std_obs, second multiplies trace).
    uint sObsTrace;

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
    device float *qOption;                     // Per-option Q^k cache, K floats.
    device float *obsTrace;                    // Tier C1: per-obs-dim trace of std_obs, O floats.
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
    c.qOption         = stateArena + cfg.sQOption;
    c.obsTrace        = stateArena + cfg.sObsTrace;

    c.counters = uintArena + cfg.uCounters;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

// =================================================================================================
// 1. oak_bk_obs - per-feature observation Welford EMA + Tier C1 obsTrace EMA. First so
//    encode reads current stats. Per-thread, no cross-thread reduction over data — bit-id
//    against per-thread compute order. The trace EMA fires at a FIXED rate 1/baseO (not the
//    `track_rate(t, baseO)` warmup-aware rate used for obsMean/obsSM): the trace is meant
//    as a memory of the last ~baseO standardized observations, so it should not be biased
//    by an exact-mean warmup phase.
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
    float invT     = track_rate(ctx.stepCount, ctx.baseO);
    float traceRate = 1.0f / float(max(ctx.baseO, 1u));   // Tier C1: fixed structural decay.
    uint nfLocal = 0u;
    for (uint o = tid; o < ctx.O; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nfLocal++; continue; }
        ctx.observationMean[o] += invT * (x - ctx.observationMean[o]);
        ctx.obsSM[o]           += invT * (x * x - ctx.obsSM[o]);
        // Trace EMA over *standardized* obs using the just-updated stats. The trace is the
        // second half of every Fourier pre-activation's input in oak_encode, so it must
        // share std_obs's scale (RFF-1 init then puts the trace half in the same regime).
        float s = standardized_obs(x, ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        ctx.obsTrace[o] += traceRate * (s - ctx.obsTrace[o]);
    }
    uint nfTotal = simd_sum(nfLocal);
    if (tid == 0u) ctx.counters[OAK_CNT_OBS_NF] = nfTotal;
}

// =================================================================================================
// 2. oak_encode - perception. Deep Fourier features (Lewandowski-Schuurmans-Machado 2024,
//    arXiv:2410.20634): R/2 pre-activations, each producing a paired (cos, sin) channel.
//    Each pre-activation reads BOTH the standardized obs (instantaneous) AND the obsTrace
//    (Tier C1, Plan Step 3 explicit: low-pass of std_obs at decay 1/baseO, capturing the
//    last ~baseO steps of dynamics so POMDP signals like gait phase / joint accel become
//    encode-able). Encoder row width is therefore 2·baseO — first baseO multiplies std_obs,
//    next baseO multiplies obsTrace.
//
//      state[feat in [0, R/2)] = cos( phase[feat]  +  Σ_o W_enc[feat][o]·std_obs[o]
//                                                 +  Σ_o W_enc[feat][baseO+o]·obsTrace[o] )
//      state[feat in [R/2, R)] = sin( ... same as above with the paired pre-activation ... )
//
//    cos/sin are bounded in [−1, 1] (structural scale prior on the BASIS); linear weights
//    on top still grow without cap (BWH). Encoder + phase evolve only via generate-and-test
//    in bk_admit_commit (Plan Step 2 generate-and-test).
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
    uint encRow = pre * 2u * ctx.baseO;        // 2·baseO weights per pre-activation (Tier C1).
    float z = ctx.phase[pre];
    // First half: std_obs (instantaneous).
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = standardized_obs(ctx.observation[o], ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        z = fma(ctx.encoder[encRow + o], c, z);
    }
    // Second half: obsTrace (low-pass of std_obs, integrated over ~baseO steps).
    for (uint o = 0u; o < ctx.baseO; ++o) {
        z = fma(ctx.encoder[encRow + ctx.baseO + o], ctx.obsTrace[o], z);
    }
    float v = (feat < halfR) ? cos(z) : sin(z);
    // safe_w-style preservation: on non-finite, keep the previous slot's value for this feature.
    float prev = state_at(ctx, ctx.stateSlot)[feat];
    state_at(ctx, 1u - ctx.stateSlot)[feat] = isfinite(v) ? v : prev;
}

// =================================================================================================
// 3. oak_meta - chord softmax (Plan option keyboard, Design A) + alphaPi (Kim arXiv:2510.06149)
//    + ρ̄ EMA (Reward Centering arXiv:2405.09999). Runs BEFORE oak_predict so the new chord
//    drives this step's ψ TD.
//
//    K threads (one per option, fits one Apple SIMD group). Per-thread k cost: R FMA
//    (ψ·curr inner product) — fully parallel across K. Reductions are bit-identical to
//    the prior sequential `for k in K` kernel:
//      * `simd_max` for the softmax max (max is associative + commutative → bit-identical).
//      * `Z = Σ_k expL_k` summed sequentially on lane 0 via `simd_shuffle`, so the reduction
//        order matches the original `for (uint k = 0; k < ctx.K; ++k) Z += expL[k]` exactly.
//        Lane 0 broadcasts `Z` via `simd_broadcast`. Optimization-phase contract Inv-10:
//        FP-reorder kernel changes (the previous `simd_sum`) shifted per-seed chatter in
//        the validation bench — pinning reduction order restores zero-intelligence-hit.
//    Lane 0 also does the scalar alphaPi (R + D ops) and ρ̄ EMA.
// =================================================================================================
kernel void oak_meta(
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
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);

    // Save chordPrev and compute the per-option ψ logit in one R-wide pass over curr.
    float zPsi = 0.0f;
    if (active) {
        ctx.chordPrev[k] = ctx.chord[k];
        uint basePsi = k * ctx.R;
        for (uint i = 0u; i < ctx.R; ++i)
            zPsi = fma(ctx.psi[basePsi + i], curr[i], zPsi);
    }

    // softmax max — `simd_max` is associative for float max; bit-identical to sequential.
    float lForMax = active ? zPsi : -FLT_MAX;
    float maxL    = simd_max(lForMax);
    float expL    = active ? exp(zPsi - maxL) : 0.0f;

    // softmax denominator — lane 0 accumulates in original `for k=0..K-1` order via
    // simd_shuffle, then broadcasts. ALL lanes must call the shuffle (Metal spec: shuffle
    // result is undefined unless both source and destination lanes are active in the same
    // control flow path), so the loop runs uniformly and only the accumulator is gated.
    float Zexp = 0.0f;
    for (uint kk = 0u; kk < ctx.K; ++kk) {
        float v = simd_shuffle(expL, kk);
        if (k == 0u) Zexp += v;
    }
    Zexp = simd_broadcast(Zexp, 0u);
    float invZ = 1.0f / max(Zexp, FLT_MIN);
    if (active) ctx.chord[k] = expL * invZ;

    // Scalar updates (single-threaded; small) — αPi and ρ̄. Done by lane 0 after the
    // softmax broadcast so the work doesn't interleave with the reduction.
    if (k == 0u) {
        float ss = 0.0f, gg = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) ss = fma(curr[i], curr[i], ss);
        for (uint p = 0u; p < ctx.D; ++p) gg = fma(ctx.scoreGrad[p], ctx.scoreGrad[p], gg);
        ctx.scalars->alphaPi = 1.0f / (float(max(ctx.baseO, 1u)) + ss * gg);

        float invT = track_rate(ctx.stepCount, ctx.baseO);
        ctx.scalars->rho += invT * (ctx.reward[0] - ctx.rho);
    }
}

// =================================================================================================
// 4. oak_predict - Horde-style multi-GVF update (K threads, fits one Apple SIMD group).
//      Main TD (lane 0):    delta   = (r - ρ̄) + V(s') - V(s)        (differential TD, ρ̄ centered)
//      Subtask TD (lane k): delta_k = (r - ρ̄) + (1−β_k)·curr[k] + β_k·ψ_k(s') − ψ_k(s)
//                           with adaptive β_k = (1 − 1/R)·α₀/(α₀ + utilOpt[k]),  α₀ = 1/baseO.
//    utilOpt[k] (EMA of |tdPsi[k]|² from oak_post_learn) is the negative-feedback brake that
//    replaced the constant β = 1 − 1/R attempt that diverged 6/10 seeds. Initialized at
//    utilOpt[k] = R so β_k starts ≈ 0 (myopic); it grows toward the cap as ψ stabilizes.
//    NF counter aggregated via `simd_sum` so a single bad lane updates the counter without
//    atomics. Per-thread cost: 2R FMA on ψ (lane 0 also does 2R FMA on valueW).
// =================================================================================================
kernel void oak_predict(
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
    float reward_c = ctx.reward[0] - ctx.rho;

    // Main TD on lane 0. Cheap (2R FMA on the value head, single output).
    if (k == 0u) {
        float vPrev = 0.0f, vNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            vPrev = fma(ctx.valueW[i], prev[i], vPrev);
            vNext = fma(ctx.valueW[i], curr[i], vNext);
        }
        float delta = reward_c + vNext - vPrev;
        if (!isfinite(delta)) { ctx.counters[OAK_CNT_LEARN_NF] += 1u; delta = 0.0f; }
        ctx.td[0] = delta;
    }

    // Per-option subtask TD on lane k. Inactive lanes contribute 0 to the NF reduction.
    uint nfLane = 0u;
    if (active) {
        float beta_cap = 1.0f - 1.0f / float(max(ctx.R, 1u));
        float a0       = 1.0f / float(max(ctx.baseO, 1u));
        float beta_k   = beta_cap * a0 / (a0 + ctx.utilOpt[k]);
        uint  basePsi  = k * ctx.R;
        float psiPrev = 0.0f, psiNext = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.psi[basePsi + i];
            psiPrev = fma(w, prev[i], psiPrev);
            psiNext = fma(w, curr[i], psiNext);
        }
        float dk = reward_c + (1.0f - beta_k) * curr[k] + beta_k * psiNext - psiPrev;
        if (!isfinite(dk)) { dk = 0.0f; nfLane = 1u; }
        ctx.tdPsi[k] = dk;
    }
    uint nfTotal = simd_sum(nfLane);
    if (k == 0u && nfTotal > 0u) ctx.counters[OAK_CNT_TDPSI_NF] += nfTotal;
}

// =================================================================================================
// 5. oak_plan - parallel asynchronous VI over DENSE option models (RR-Subtasks eq. 17 + 19).
//
//    K·R-thread parallel: K threadgroups, R threads each (one TG per option k, one thread
//    per output dim j). Each (k, j) thread computes one row of the dense option transition,
//    fuses the curr and prev passes into a single R-load over `optTransition[k][j][:R]`,
//    and writes nTdMat[k·R + j] = curr[j] − n̂^k_j(prev) directly. Each thread stores its
//    `nHatCurr` into threadgroup memory; lane 0 sums vHat = Σ_j valueW[j]·nHatCurr_j via
//    `fma` in strict index order j=0..R-1, exactly matching the Tier B single-thread loop's
//    `vHat = fma(valueW[j], nHatCurr, vHat)` sequence — bit-identical FP, no `simd_sum`.
//    Lane 0 also computes r̂^k = optReward[k] · curr (R FMA) and emits Q^k = r̂^k + V(n̂^k)
//    into the qOption[K] cache.
//
//    The cross-K reduction (bestQ = max_k Q^k → planAdv soft-bound + valueW residual)
//    runs as a tiny follow-up kernel `oak_plan_reduce` (K threads, single simdgroup,
//    simd_max — bit-identical for max).
//
//    Compute budget at K=32 R=64: 32 simdgroups in flight per kernel (≈ 32× the previous
//    K-thread design which used a single simdgroup). Per-thread inner cost: 2R FMA = 128
//    FMA + 1 threadgroup_barrier; per-TG lane-0 reduction adds R fma's. Bit-identical
//    intelligence, parallel throughput.
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

    // Per-thread dense-transition row. Same R-load fuses curr and prev passes; each thread
    // also writes its own slot in nTdMat (no contention). Stores nHatCurr (not the fma'd
    // contribution) into threadgroup memory so the reduction below keeps the original
    // single-thread kernel's exact `vHat = fma(valueW[j], nHatCurr, vHat)` sequence —
    // separate multiply-then-add would lose the fma's single-rounding and drift.
    float nHatCurrLocal = 0.0f;
    if (active) {
        float nHatPrev = 0.0f;
        uint  base = k * ctx.R * ctx.R + j * ctx.R;
        for (uint i = 0u; i < ctx.R; ++i) {
            float w = ctx.optTransition[base + i];
            nHatCurrLocal = fma(w, curr[i], nHatCurrLocal);
            nHatPrev      = fma(w, prev[i], nHatPrev);
        }
        ctx.nTdMat[k * ctx.R + j] = curr[j] - nHatPrev;
    }

    // Cross-thread reduction: vHat^k = Σ_j valueW[j]·nHatCurr_j, accumulated via fma in
    // strict index order j=0..R-1, exactly matching the original single-thread loop.
    // tg_nHatCurr is sized for OAK_MAX_RANK = 256 (1 KB of threadgroup memory) — the
    // host-rounded `rank` is capped well below this for current envs (HC-v4: R = 64).
    constexpr uint OAK_MAX_RANK = 256u;
    threadgroup float tg_nHatCurr[OAK_MAX_RANK];
    tg_nHatCurr[j] = active ? nHatCurrLocal : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j == 0u) {
        float vHat = 0.0f;
        for (uint jj = 0u; jj < ctx.R; ++jj)
            vHat = fma(ctx.valueW[jj], tg_nHatCurr[jj], vHat);
        float rHat = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);
        float Q = rHat + vHat;
        bool finite = isfinite(Q);
        ctx.qOption[k] = finite ? Q : -FLT_MAX;
        if (!finite) ctx.counters[OAK_CNT_PLAN_NF] += 1u;
    }
}

// =================================================================================================
// 5b. oak_plan_reduce - cross-option max over qOption[K], then the planAdv soft-bound
//    `(bestQ − V(s)) / (1 + |bestQ − V(s)|)`. K threads = single simdgroup, simd_max for
//    the reduction. Lane 0 also computes V(curr) and writes planAdv (safe_w-preserve on NF).
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

    if (k == 0u) {
        device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
        float vCurr = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i) vCurr = fma(ctx.valueW[i], curr[i], vCurr);
        float advRaw = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
        // x/(1+|x|) self-bound — the standard rational soft-clamp from Sutton's gradient-TD
        // line of work. Image is (-1, 1), continuous + monotone, no tunable scale.
        float adv = advRaw / (1.0f + fabs(advRaw));
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
//
//     Single threadgroup, multi-thread:
//       Phase 1 (lane 0):  find worstK + worstI via linear scan; bump cumulative counters.
//                          Writes worstK, worstI into threadgroup memory.
//       Barrier            (mem_threadgroup)
//       Phase 2 (all lanes): cooperative parallel reseed of the worstK weight slices
//                          (psi, optReward, optPolicy, optTransition) plus the worstI
//                          Fourier encoder row, distributed by `tid`.
//       Barrier
//       Phase 3 (lane 0):  scalar resets (utilOpt, utilFeat, phase, MetaOptimize) +
//                          slot toggle + stepCount increment + stats surface.
//
//     Lane 0 captures `stepCount` BEFORE the increment so all reseed hashes match the
//     prior single-threaded kernel byte-for-byte (deterministic seed equivalence).
//     One TG of 256 threads keeps occupancy modest; the dominant K·R²=4096-write
//     optTransition slice gets ≥16 writes/lane = ~1 µs.
// =================================================================================================
kernel void oak_bk_admit_commit(
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

    // Default-initialized so the compiler's per-lane flow analysis is satisfied; lane 0
    // unconditionally overwrites all three before the barrier so the defaults never leak.
    threadgroup uint  worstK_tg    = 0u;
    threadgroup uint  worstI_tg    = 0u;
    threadgroup uint  stepCount_tg = 0u;

    // Phase 1: lane 0 picks worsts and bumps cumulative counters. Capture stepCount BEFORE
    // the post-step increment so the parallel reseed below uses the same hash key as the
    // prior single-threaded kernel.
    if (tid == 0u) {
        uint wK = 0u; float wUO = FLT_MAX;
        for (uint k = 0u; k < ctx.K; ++k)
            if (ctx.utilOpt[k] < wUO) { wUO = ctx.utilOpt[k]; wK = k; }
        worstK_tg = wK;
        uint wI = 0u; float wUF = FLT_MAX;
        for (uint i = 0u; i < ctx.R; ++i)
            if (ctx.utilFeat[i] < wUF) { wUF = ctx.utilFeat[i]; wI = i; }
        worstI_tg = wI;
        stepCount_tg = ctx.stepCount;
        ctx.counters[OAK_CNT_REINIT_OPT]  += 1u;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint worstK = worstK_tg;
    uint worstI = worstI_tg;
    uint stepC  = stepCount_tg;
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
    uint nEnc = 2u * ctx.baseO;        // Tier C1: encoder rows are 2·baseO (std_obs ‖ trace).
    uint o0   = 0u;
    uint o1   = o0 + nPsi;
    uint o2   = o1 + nOR;
    uint o3   = o2 + nPi;
    uint o4   = o3 + nOT;
    uint total = o4 + nEnc;

    for (uint id = tid; id < total; id += tgW) {
        if (id < o1) {                                              // psi[worstK][i]
            uint i = id - o0;
            ctx.psi[worstK * ctx.R + i] =
                (hash_uniform(stepC, worstK, i) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < o2) {                                       // optReward[worstK][i]
            uint i = id - o1;
            ctx.optReward[worstK * ctx.R + i] =
                (hash_uniform(stepC, worstK, ctx.R + i) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < o3) {                                       // optPolicy[worstK][:D*R]
            uint pi = id - o2;
            ctx.optPolicy[worstK * ctx.D * ctx.R + pi] =
                (hash_uniform(stepC, worstK, 3u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        } else if (id < o4) {                                       // optTransition[worstK][:R*R]
            uint pi = id - o3;
            ctx.optTransition[worstK * ctx.R * ctx.R + pi] =
                (hash_uniform(stepC, worstK, 4u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        } else {                                                     // encoder[pre][:2·baseO]
            uint o = id - o4;
            float u = hash_uniform(stepC, pre, o + 0xdeadbeefu) * 2.0f - 1.0f;
            ctx.encoder[pre * 2u * ctx.baseO + o] = u;               // RFF bandwidth-1, both halves.
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: lane 0 does the small scalar resets + slot toggle + stepCount increment +
    // surface stats. Done here (post-reseed) so the increment doesn't race the hash reads.
    if (tid == 0u) {
        ctx.utilOpt[worstK] = 0.0f;
        float u = hash_uniform(stepC, pre, 0xfeedf00du);
        ctx.phase[pre] = u * 6.283185307179586f;                     // 2π, structural.
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

        // Surface the only telemetry the host literally cannot derive from env signals.
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
