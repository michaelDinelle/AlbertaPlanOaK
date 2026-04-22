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
    // Tier D2 (Plan Step 9, prioritized sweeping; Moore-Atkeson 1993, Plan footnote 13):
    // per-option priority signal, K floats. prio[k] = chord[k] · ||model TD error_k||_L2.
    // Written by oak_plan lane 0 (in-line via fused tg_ntdSq write, zero added barriers);
    // consumed by oak_plan_reduce (mean → KernelScalars.meanPrio) and by oak_learn_heads
    // Head 4 gate (`prioOpt[k] >= meanPrio` → update; below → skip the K·R² optTransition
    // gradient for that option this step).
    uint sPrioOpt;
    // Tier H1 (Plan Step 10 STOMP multi-step options): per-option continuation probability
    // β^k = 1/(1 + ntdSqSum[k]) ∈ (0, 1], K floats. Written by oak_plan lane 0 at the end
    // of step t (piggybacks on the same lane-0 scope that writes prioOpt — zero added
    // barriers). Read by step t+1's oak_meta_predict phase 1 for chord hysteresis:
    //   chord_k ← β^k · chordPrev[k] + (1 − β^k) · softmax_k(ψ·s')     (renormalized)
    // High β (confident transition model) → option persists; low β (model surprised)
    // → option terminates and chord follows the fresh softmax. One-step β lag is
    // intentional (β tracks slow-moving optTransition quality).
    uint sPrevBeta;

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
    // Tier D2 (Plan Step 9): mean(prioOpt[K]), the threshold for Moore-Atkeson 1993
    // prioritized sweeping in oak_learn_heads Head 4. Computed once per step in
    // oak_plan_reduce via simd_sum across K=32 lanes (single simdgroup, ~5 cycles), then
    // broadcast-cached for every Head 4 thread to read (avoids 131k Head 4 threads each
    // computing the same K-element reduction = 4.2M wasted fma per step).
    float meanPrio;
};

// Telemetry the host literally cannot derive from env signals: per-phase non-finite counters
// (kernel-internal numerical health) and cumulative reinit counts (Plan Step 11 evidence).
struct KernelStats {
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount, tdPsiNonFiniteCount;
    float oakOptionReinitCount, oakFeatureReinitCount;
    // Tier H1 diagnostic: chord effective-K = 1 / Σ_k chord[k]² ∈ [1, K]. Independent of
    // chord step-to-step persistence; pure concentration measure. K = uniform chord;
    // 1 = one-hot. Written by oak_meta_predict lane 0 after chord renormalization.
    float chordEffectiveK;
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
    float rho, alphaPi, meanPrio;              // meanPrio: Tier D2 broadcast scalar, see KernelScalars.

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
    device float *prioOpt;                     // Tier D2: per-option priority, K floats.
    device float *prevBeta;                    // Tier H1: per-option continuation β, K floats.
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
    c.rho = scalars->rho; c.alphaPi = scalars->alphaPi; c.meanPrio = scalars->meanPrio;

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
    c.prioOpt         = stateArena + cfg.sPrioOpt;
    c.prevBeta        = stateArena + cfg.sPrevBeta;

    c.counters = uintArena + cfg.uCounters;
    c.scalars = scalars; c.stats = stats;
    return c;
}

inline device float *state_at(Ctx ctx, uint slot) { return ctx.state + slot * ctx.R; }

// =================================================================================================
// 1+2. oak_bk_encode - FUSED kernel combining oak_bk_obs (per-feature observation Welford +
//      obsTrace EMAs, Tier C1) and oak_encode (perception / deep Fourier features,
//      Lewandowski-Schuurmans-Machado 2024, arXiv:2410.20634). Dispatched as 1 TG of R=rank
//      threads. Phase 1 uses threads with tid < O to update obsMean/obsSM/obsTrace; phase 2
//      uses all R threads to compute one Fourier feature per thread. A single threadgroup_
//      barrier(mem_flags::mem_device) between the phases replaces the prior host-level
//      memoryBarrier between two separate kernel dispatches — saves 1 dispatch + 1 host
//      barrier per step (~25-40 µs on concurrent bench).
//
//      The trace EMA fires at a FIXED rate 1/baseO (not the `track_rate(t, baseO)` warmup-
//      aware rate used for obsMean/obsSM): the trace is meant as a memory of the last ~baseO
//      standardized observations, so it should not be biased by an exact-mean warmup phase.
//
//      Encoder row width is 2·baseO (Tier C1) — first baseO multiplies std_obs, next baseO
//      multiplies obsTrace:
//        state[feat in [0, R/2)] = cos( phase[pre] + Σ_o W[pre, o]·std_obs[o]
//                                                 + Σ_o W[pre, baseO+o]·obsTrace[o] )
//        state[feat in [R/2, R)] = sin( ... same pre-activation index as its cos partner ... )
//      cos/sin are bounded in [−1, 1]; linear weights on top grow without cap (BWH).
//      Encoder + phase evolve only via generate-and-test in oak_bk_admit_commit.
//
//      Tier C3 (recurrent self-feedback) was attempted in two forms and both regressed by
//      ~52% on the 10×1.5M paired bench (TD double non-stationarity); encoder remains C1
//      shape.
//
//      Inv-10 (bit-identical FP) preserved: same fma sequences in both phases; the NF
//      counter aggregation uses a per-simdgroup `simd_sum` + threadgroup-memory cross-
//      simdgroup sum (mathematically equivalent to the pre-fusion single-simdgroup
//      simd_sum in the original oak_bk_obs since total NF count is associative).
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
    float invT      = track_rate(ctx.stepCount, ctx.baseO);
    float traceRate = 1.0f / float(max(ctx.baseO, 1u));   // Tier C1: fixed structural decay.
    uint nfLocal = 0u;
    for (uint o = tid; o < ctx.O; o += tgW) {
        float x = ctx.observation[o];
        if (!isfinite(x)) { nfLocal++; continue; }
        ctx.observationMean[o] += invT * (x - ctx.observationMean[o]);
        ctx.obsSM[o]           += invT * (x * x - ctx.obsSM[o]);
        // Trace EMA over *standardized* obs using the just-updated stats. Shares std_obs's
        // scale (RFF-1 init then puts the trace half in the same regime).
        float s = standardized_obs(x, ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        ctx.obsTrace[o] += traceRate * (s - ctx.obsTrace[o]);
    }

    // NF count aggregation across simdgroups: per-simdgroup simd_sum, then cross-simdgroup
    // sum in threadgroup memory. tg_nf is sized for up to 8 simdgroups (256-thread TG),
    // which covers the current R ∈ {32, 64, 128} range used by real envs.
    constexpr uint OAK_MAX_TG_SIMDGROUPS = 8u;
    threadgroup uint tg_nf[OAK_MAX_TG_SIMDGROUPS];
    uint sgSum = simd_sum(nfLocal);
    uint sgId  = tid / 32u;
    if (tid % 32u == 0u && sgId < OAK_MAX_TG_SIMDGROUPS) tg_nf[sgId] = sgSum;

    // ================= BARRIER (phase 1 → phase 2, device memory fence) =================
    // Phase 2 reads obsMean, obsSM, obsTrace from device memory. Need threadgroup_barrier
    // with mem_device scope so all writes from phase 1 are visible across all lanes of
    // both simdgroups before any phase 2 read. (simdgroup_barrier would only fence within
    // one simdgroup; we have 2 simdgroups in the R=64 TG on HC.)
    threadgroup_barrier(mem_flags::mem_device);

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
// 3+4. oak_meta_predict - FUSED kernel combining oak_meta (chord keyboard) and oak_predict
//      (multi-GVF TD). Both kernels are K-thread single-simdgroup with a clear dependency
//      chain: predict reads ρ̄ (which meta updates). Fusing eliminates 1 kernel dispatch
//      and 1 host-level memoryBarrier per step (~25-40 µs/step on concurrent bench), at
//      the cost of one intra-kernel simdgroup_barrier (which is essentially free on Apple
//      GPU where lanes execute in lockstep).
//
//      Phase 1 (oak_meta):
//        * chord softmax (Plan option keyboard, Design A): chord[k] ← softmax_k(ψ(s'))
//        * alphaPi = 1/(baseO + ‖s'‖²·‖∇log π‖²)   (Kim arXiv:2510.06149)
//        * ρ̄ EMA                                  (Reward Centering arXiv:2405.09999)
//
//      Phase 2 (oak_predict):
//        * Main TD on lane 0: δ = (r − ρ̄) + V(s') − V(s)   (ρ̄ uses the POST-meta value)
//        * Per-option ψ-TD on lane k: δ_k = (r − ρ̄) + (1−β_k)·curr[k] + β_k·ψ_k(s') − ψ_k(s)
//          with adaptive β_k = (1 − 1/R)·α₀/(α₀ + utilOpt[k]), α₀ = 1/baseO.
//
//      Dependency carried across the barrier: phase 1 lane 0 updates `ctx.scalars->rho`;
//      phase 2 reads the updated ρ̄ by dereferencing `ctx.scalars->rho` (NOT the cached
//      `ctx.rho`, which was snapshotted pre-phase-1 in make_ctx). The
//      `simdgroup_barrier(mem_flags::mem_device)` fence guarantees lane 0's device-memory
//      write is visible to all other lanes after the barrier.
//
//      Inv-10 (bit-identical FP) preserved: the fma sequences in both phases are identical
//      to the pre-fusion kernels, and the only difference is that predict reads the just-
//      updated ρ̄ from scalars directly instead of from a kernel-start Ctx snapshot.
//      Both kernels previously executed as separate launches with a host memoryBarrier in
//      between, which produces the same program order (phase 1 writes → phase 2 reads).
// =================================================================================================
kernel void oak_meta_predict(
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

    // ================= PHASE 1: oak_meta =================

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
    float softK = expL * invZ;       // fresh softmax "intent" for option k this step.

    // Tier H1 (Plan Step 10 STOMP multi-step options) — chord hysteresis.
    // β^k = prevBeta[k] ∈ [0, 1] is last step's optTransition confidence (written by
    // oak_plan at end of step t-1). Per-option lerp between the just-saved chordPrev
    // and the fresh softmax "intent":
    //   chord_k_raw = β^k · chordPrev[k] + (1 − β^k) · softK
    // Fresh random optTransition → β ≈ 0 → chord = softK (1-step behavior, safe fallback).
    // Converged optTransition → β → 1 → chord ≈ chordPrev (multi-step persistence).
    // NF-safe: β clamped to [0,1] at read; output clamped at write via safe_w-style preserve.
    //
    // Renormalize via simd_sum across K lanes so Σ chord = 1 (required for downstream
    // heads that treat chord as a probability distribution over options). This is a NEW
    // reduction (no prior bit-id baseline applies per Inv-10; Tier H1 is algorithmic, not
    // perf-opt, so FP-reorder is allowed without per-seed validation).
    float betaK = active ? ctx.prevBeta[k] : 0.0f;
    betaK = clamp(betaK, 0.0f, 1.0f);
    float chordPrevK = active ? ctx.chordPrev[k] : 0.0f;
    float chordRaw = active ? (betaK * chordPrevK + (1.0f - betaK) * softK) : 0.0f;
    if (!isfinite(chordRaw)) chordRaw = 0.0f;
    float chordSum = simd_sum(chordRaw);
    float chordK = chordRaw / max(chordSum, FLT_MIN);
    if (active) {
        float oldChord = ctx.chord[k];
        ctx.chord[k] = isfinite(chordK) ? chordK : oldChord;
    }

    // Tier H1 diagnostic: chord effective-K = 1 / Σ_k chord[k]² (chord concentration).
    // Range [1, K]: K = uniform chord (all K options equally weighted, mechanism inactive
    // or chord-flat); 1 = one-hot chord (single option fully dominant); intermediate =
    // moderate concentration. With K=32 on HC: 32 = no concentration, ~10 = moderate
    // (3-4 dominant options), 1-3 = sharp concentration.
    //
    // Replaces an earlier diagnostic Σ chord·chordPrev which was confounded by chord
    // magnitude (uniform chord always returned 1/K regardless of whether β-hysteresis
    // was active or not). chordEffectiveK is a pure concentration measure independent
    // of step-to-step persistence; combined with the standard chatter signal it lets us
    // distinguish "mechanism inactive (β=0)" from "mechanism active but chord uniform".
    //
    // Computed in the same simd_sum framework: chord_sq = chord[k]² summed across K lanes.
    // chord_sq uses the POST-renormalization chord[k] = chordRaw[k]/chordSum, so we use
    // (chordRaw/chordSum)². Lane 0 inverts and writes (NF-safe clamp-at-write).
    float chordK_norm = chordRaw / max(chordSum, FLT_MIN);
    float chordSqSum  = simd_sum(chordK_norm * chordK_norm);
    if (k == 0u) {
        float effK = 1.0f / max(chordSqSum, FLT_MIN);
        ctx.stats->chordEffectiveK =
            isfinite(effK) ? effK : ctx.stats->chordEffectiveK;
    }

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

    // ================= BARRIER (device memory fence) =================
    // Lane 0's write to scalars->rho must be visible to all other lanes before phase 2
    // reads it. simdgroup_barrier with mem_device scope is the minimal fence for this;
    // within a single simdgroup (K=32 lanes) it's essentially free on Apple GPU (lanes
    // are already in lockstep execution), with only the memory-visibility cost.
    simdgroup_barrier(mem_flags::mem_device);

    // ================= PHASE 2: oak_predict =================
    // Re-read ρ̄ from the scalars buffer (lane 0 just updated it in phase 1). Using
    // ctx.rho here would read the kernel-start snapshot, NOT the meta-updated value,
    // and would break bit-id with the pre-fusion two-kernel order.
    float reward_c = ctx.reward[0] - ctx.scalars->rho;

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
    // Adaptive β_k = (1 − 1/R)·α₀/(α₀ + utilOpt[k]), α₀ = 1/baseO. utilOpt[k] is the EMA
    // of |tdPsi[k]|² from oak_post_learn (Kim implicit-step self-bound). Initialized
    // at utilOpt[k] = R so β_k starts ≈ 0 (myopic, safe at random init); grows toward
    // the cap (1 − 1/R) as ψ stabilizes. If |tdPsi|² spikes (divergence threat), β
    // automatically chokes. This is a STABILITY-adaptive β, not a Plan Step 10 STOMP
    // termination β — Tier C4 attempted structural termination β = (1+curr[k])/2 but
    // regressed -30% aggregate lift at 1.5M (options went myopic without bootstrap,
    // chatter collapsed to 0.02). See docs/roadmap.md Tier C4 for the negative-result
    // writeup. Plan Step 10 STOMP termination is structurally incompatible with the
    // current ψ-TD formulation at these scales; β stays stability-adaptive.
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
// 5. oak_plan - parallel asynchronous VI over DENSE option models (RR-Subtasks eq. 17 + 19)
//    + Tier D2 (Plan Step 9, prioritized sweeping; Moore-Atkeson 1993, Plan footnote 13)
//    in-line per-option priority computation.
//
//    K·R-thread parallel: K threadgroups, R threads each (one TG per option k, one thread
//    per output dim j). Each (k, j) thread computes one row of the dense option transition,
//    fuses the curr and prev passes into a single R-load over `optTransition[k][j][:R]`,
//    and writes nTdMat[k·R + j] = curr[j] − n̂^k_j(prev) directly. Each thread stores its
//    `nHatCurr` into threadgroup memory AND its `n_td_sq = n_td · n_td` into a parallel
//    threadgroup array — both writes go BEFORE the same single barrier, zero added barriers.
//
//    Lane 0 of each TG runs a single R-loop that interleaves three accumulators:
//      vHat^k     = Σ_j valueW[j] · nHatCurr_j      (strict j=0..R-1 fma order, bit-identical)
//      ntdSqSum^k = Σ_j n_td_j^2                    (Tier D2 priority L2² of model TD error)
//      rHat^k     = Σ_i optReward[k][i] · curr[i]   (separate i-loop, R fma)
//    The two j-loop accumulators are independent fma chains → Apple GPU's ILP scheduler runs
//    them in parallel, so the prio addition adds ~0 latency (lane 0's R fma chain cycle count
//    stays ~R cycles instead of 2R).
//
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
//    Compute budget at K=32 R=64: 32 simdgroups in flight per kernel. Per-thread inner cost:
//    2R FMA + 1 squaring = 129 ops + 1 threadgroup_barrier; per-TG lane-0 reduction is ~R
//    cycles (vHat + ntdSqSum interleaved) + R cycles (rHat). Bit-identical intelligence on
//    the existing reductions, in-line priority cost is < 1% of step.
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
        float n_td = curr[j] - nHatPrev;
        ctx.nTdMat[k * ctx.R + j] = n_td;
        nTdSqLocal = n_td * n_td;     // Tier D2: in-register, free with ILP.
    }

    // Cross-thread reduction stage. Both arrays are written before a SINGLE barrier — adding
    // tg_ntdSq does not introduce a second barrier (Inv-10 zero-intelligence-hit on existing
    // kernel timings). OAK_MAX_RANK = 256 (1 KB per array = 2 KB total per TG of TG memory).
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
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);
        float Q = rHat + vHat;
        bool finite = isfinite(Q);
        ctx.qOption[k] = finite ? Q : -FLT_MAX;
        if (!finite) ctx.counters[OAK_CNT_PLAN_NF] += 1u;

        // Tier D2 priority write: chord-weighted L2 of model TD error. NF-clamp at write so
        // a dirty option becomes lowest-priority (skipped by the Head 4 gate) — failure
        // isolation, the rest of the K options learn normally. Plan-orthodox: prio depends
        // only on agent-internal signals (chord, nTdMat) — no env-knowledge, no tunable.
        bool prioFinite = isfinite(ntdSqSum) && (ntdSqSum > 0.0f);
        ctx.prioOpt[k]  = prioFinite ? (ctx.chord[k] * sqrt(ntdSqSum)) : 0.0f;

        // Tier H1: per-option continuation probability β^k = 1/(1 + MSE^k) ∈ (0, 1] where
        // MSE^k = ntdSqSum[k] / R is the PER-DIMENSION model-TD-error (not raw L2). The /R
        // scaling is structural (R = representation size, already used in MetaOptimize's
        // η = 1/(R·baseO)) — it matches the natural scale of ntdSqSum across the learning
        // trajectory: random features give Var(n_td_j)~1 so MSE~1, β~0.5; well-learned
        // give MSE→0, β→1. Without /R, β is stuck near 1/K = no persistence (verified
        // empirically in 10×1.5M paired bench: persistence stayed at 1/K throughout).
        // NF-safe: non-finite ntdSqSum → β = 0 (option terminates, safe default).
        float invR    = 1.0f / float(max(ctx.R, 1u));
        float betaNew = isfinite(ntdSqSum) ? (1.0f / (1.0f + ntdSqSum * invR)) : 0.0f;
        ctx.prevBeta[k] = isfinite(betaNew) ? betaNew : 0.0f;
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
//    making 131k Head 4 threads each compute the same K-element reduction).
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

    // Tier D2 (Plan Step 9): cross-option mean of priority. simd_sum across K=32 lanes is a
    // single butterfly-add (~5 cycles on Apple GPU). All lanes participate (Metal spec); only
    // lane 0 writes the broadcast scalar. Inactive lanes contribute 0 → mean denominator is
    // K (the host's optionCount), not the active count.
    float prio    = active ? ctx.prioOpt[k] : 0.0f;
    if (!isfinite(prio)) prio = 0.0f;
    float prioSum = simd_sum(prio);

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
    }
}

// =================================================================================================
// 6. oak_learn_heads - 1-step TD updates on 5 learnable heads.
//      0: valueW[R]              -> alpha[i] * (td + planAdv) * prev[i]   (planAdv added HERE ONLY)
//      1: psi[K*R]               -> alpha[i] * tdPsi[k] * prev[i] * max(chordPrev[k], 1/K)
//                                   Tier D3 (Plan Step 10 "conditional on the option, off-policy"):
//                                   chord-weighted attribution lets option k's psi specialize to
//                                   its own activation instead of converging to the chord mean.
//                                   1/K floor prevents self-reinforcing starvation when chord[k]→0.
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
    }
}

// =================================================================================================
// 8. oak_post_learn - three concurrent passes after learn_heads writes the new weights:
//      gid in [0, R)              -> per-feature GraMa utility EMA + MetaOptimize L-approx
//      gid in [R, R+K)            -> per-option |tdPsi|² utility EMA
//      gid in [R+K, R+K+D·K)      -> action mu_p = sum_k chord[k] · (W_pi[k][p] · s')
//    Folding all three into one kernel saves a launch.
//
//    PERF NOTE on region 3 (action). Old: 6 threads (one per action dim) each doing K·R =
//    2048 sequential fma in 1 simdgroup with 26 IDLE lanes — ~110 µs/step, 55% of total GPU.
//    New: D·K threads, dispatch grid = R + K + D·K, tgSize = 32 (= simdgroup width = K).
//    Each `p` lands in exactly ONE simdgroup of K=32 lanes. Lane k of TG p computes its own
//    `dot_k = Σ_i optPolicy[(k·D+p)·R+i] · curr[i]` in parallel. Lane 0 then reads each
//    lane's dot via `simd_shuffle(dot, kk)` and accumulates `mu = fma(chord[kk], d, mu)`
//    in the SAME `kk = 0..K-1` order as the prior single-thread code — bit-identical FP
//    (Inv-10 zero-intelligence-hit preserved by construction). If a future env ever pushes
//    K below 32, lanes K..31 stay inactive (their `dot` is not read by the kk=0..K-1
//    reduction), so functional correctness holds.
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
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, stats, cfg);
    uint actBase   = ctx.R + ctx.K;
    uint actCount  = ctx.D * ctx.K;
    if (gid >= actBase + actCount) return;
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
    if (gid < actBase) {
        uint k = gid - ctx.R;
        float dk = ctx.tdPsi[k];
        ctx.utilOpt[k] += invT * (dk * dk - ctx.utilOpt[k]);
        return;
    }

    // Action region (D·K threads, organized as D simdgroups of K=32 lanes each).
    // gid layout: (gid - actBase) = p · K + k  →  p = block index, k = lane within block.
    // With tgSize = 32 = simdgroup width = K, lid IS k and the threadgroup IS one simdgroup
    // dedicated to one p — perfect for simd_shuffle reduction with no threadgroup memory.
    uint actId = gid - actBase;
    uint p     = actId / ctx.K;
    uint k     = lid;                      // = actId % K (since tgSize == K).

    // Each lane computes its own dot product in parallel (R fma).
    float dot = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i)
        dot = fma(ctx.optPolicy[(k * ctx.D + p) * ctx.R + i], curr[i], dot);

    // Lane 0 reads each lane's dot via simd_shuffle and applies chord with fma in the
    // ORIGINAL kk=0..K-1 sequence — bit-identical to the prior single-thread loop's
    // `mu = fma(chord[kk], dot, mu)`. ALL lanes must execute the shuffle (Metal spec:
    // shuffle is undefined unless source and destination are both in active control flow).
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
