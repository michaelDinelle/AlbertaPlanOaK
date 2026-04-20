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

    // Param offsets: 6 heads, all Plan-required.
    uint pEncoder, pValue, pPsi, pOptPolicy, pOptReward, pOptTransition;

    // State arena offsets.
    uint sObservation, sObsMean, sObsSM;
    uint sState;                // 2 * R floats (slot toggle).
    uint sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi;
    uint sChord, sChordPrev;
    uint sUtilFeat, sUtilOpt;
    uint sPlanAdv;

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

    device float *encoder, *valueW, *psi, *optPolicy, *optReward, *optTransition;
    device float *observation, *observationMean, *obsSM;
    device float *state;
    device float *policyMean, *scoreGrad;
    device float *reward, *td, *tdPsi;
    device float *chord, *chordPrev;
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
    c.O = cfg.obsDim; c.R = cfg.rank; c.K = cfg.optionCount;
    c.baseO = cfg.baseObsDim; c.D = cfg.actionParamCount;
    c.stepCount = scalars->stepCount; c.stateSlot = scalars->stateSlot;
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
// 2. oak_encode - perception. state[r] = W_enc[r] . standardized_obs. Encoder is a FIXED
//    random projection at init (1/sqrt(O) scale), so state magnitude is naturally O(1) without
//    artificial bounding (BWH). Evolves only via generate-and-test in bk_admit_commit.
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
        float c = standardized_obs(ctx.observation[o], ctx.observationMean[o], ctx.obsSM[o], ctx.baseO);
        z = fma(ctx.encoder[encRow + o], c, z);
    }
    // safe_w-style preservation: on non-finite, keep the previous slot's value for this feature
    // (no zero-clamp; BWH-orthodox).
    float prev = state_at(ctx, ctx.stateSlot)[feat];
    state_at(ctx, 1u - ctx.stateSlot)[feat] = isfinite(z) ? z : prev;
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

    // Subtask TDs: dk = (r - rho) + curr[k] - psi(s). Centering keeps tdPsi mean-zero so the
    // psi-weight gradient (tdPsi * prev[i]) doesn't accumulate a state-correlated bias.
    // K = min(R, OAK_SIMD) so curr[k] is always valid.
    uint nfPsi = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        float psiPrev = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            psiPrev = fma(ctx.psi[k * ctx.R + i], prev[i], psiPrev);
        float dk = reward_c + curr[k] - psiPrev;
        if (!isfinite(dk)) { dk = 0.0f; nfPsi++; }
        ctx.tdPsi[k] = dk;
    }
    if (nfPsi > 0u) ctx.counters[OAK_CNT_TDPSI_NF] += nfPsi;
}

// =================================================================================================
// 5. oak_plan - asynchronous VI over option models (RR-Subtasks eq. 19).
//   Q^k(s) = r_hat^k(s) + V(n_hat^k(s));  planAdv = max_k Q^k - V(s) (self-bounded).
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
    float vCurr = 0.0f;
    for (uint i = 0u; i < ctx.R; ++i) vCurr = fma(ctx.valueW[i], curr[i], vCurr);

    float bestQ = -FLT_MAX;
    uint nfPlan = 0u;
    for (uint k = 0u; k < ctx.K; ++k) {
        // r_hat^k(s) = sum_i optReward[k][i] * curr[i].
        float rHat = 0.0f;
        for (uint i = 0u; i < ctx.R; ++i)
            rHat = fma(ctx.optReward[k * ctx.R + i], curr[i], rHat);
        // Diagonal transition: n_hat^k_j(s) = optTransition[k][j] * curr[j]. Then
        // V(n_hat^k(s)) = sum_j valueW[j] * optTransition[k][j] * curr[j]. K*R cost vs K*R^2.
        float vHat = 0.0f;
        for (uint j = 0u; j < ctx.R; ++j)
            vHat = fma(ctx.valueW[j] * ctx.optTransition[k * ctx.R + j], curr[j], vHat);
        float Q = rHat + vHat;
        if (!isfinite(Q)) { nfPlan++; continue; }
        if (Q > bestQ) bestQ = Q;
    }
    // RR-Subtasks eq. 19 bracket term: planAdv = max_k Q^k - V(s). Raw, no scale prior.
    // If the option model diverges, planAdv becomes huge and learn_heads' valueW update will
    // notice (via safe_w preservation on overflow). The Plan doesn't bound the bracket; the
    // natural scale comes from V's own magnitude.
    float adv = (bestQ > -FLT_MAX) ? (bestQ - vCurr) : 0.0f;
    if (!isfinite(adv)) { ctx.counters[OAK_CNT_PLAN_NF] += 1u; }
    else if (nfPlan > 0u)             { ctx.counters[OAK_CNT_PLAN_NF] += nfPlan; }
    ctx.planAdv[0] = isfinite(adv) ? adv : ctx.planAdv[0];   // safe_w-style preserve on NF.
}

// =================================================================================================
// 6. oak_learn_heads - 1-step TD updates on 5 learnable heads.
//      0: valueW[R]            -> alpha * (td + planAdv) * prev[i]   (planAdv added HERE ONLY)
//      1: psi[K*R]             -> alpha * tdPsi[k] * prev[i]
//      2: optPolicy[K*D*R]     -> alphaPi * td * scoreGrad[d] * prev[i] * chordPrev[k]
//      3: optReward[K*R]       -> alpha * (reward - r_hat^k(prev)) * prev[i] * chordPrev[k]
//      4: optTransition[K*R]   -> alpha * (curr[j] - w_n^{k,j}*prev[j]) * prev[j] * chordPrev[k]
//                                 (DIAGONAL transition: n_hat^k_j(s) = w_n^{k,j} * s[j])
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
    uint nOT  = ctx.K * ctx.R;       // diagonal: K*R, was K*R^2.
    if (gid >= nVal + nPsi + nPi + nOR + nOT) return;

    device const float *prev = state_at(ctx, ctx.stateSlot);
    device const float *curr = state_at(ctx, 1u - ctx.stateSlot);
    float td = ctx.td[0];
    float deltaValue = td + ctx.planAdv[0];
    // Per-feature step size 1/baseO: structurally derived from env input dim, the only
    // env-given quantity. R features each contribute 1/baseO of the gradient → total step
    // bounded by (R/baseO) per parameter. Same value used for value/psi/optModel heads.
    float a = 1.0f / float(max(ctx.baseO, 1u));

    if (gid < nVal) {                                          // Head 0: valueW
        uint i = gid;
        float w = ctx.valueW[i];
        ctx.valueW[i] = safe_w(w, w + a * deltaValue * prev[i]);
        return;
    }
    if (gid < nVal + nPsi) {                                   // Head 1: psi
        uint id = gid - nVal;
        uint k = id / ctx.R, i = id % ctx.R;
        float w = ctx.psi[id];
        ctx.psi[id] = safe_w(w, w + a * ctx.tdPsi[k] * prev[i]);
        return;
    }
    if (gid < nVal + nPsi + nPi) {                             // Head 2: optPolicy (REINFORCE)
        uint id = gid - nVal - nPsi;
        uint k = id / (ctx.D * ctx.R);
        uint rem = id - k * ctx.D * ctx.R;
        uint d = rem / ctx.R, i = rem % ctx.R;
        float grad = ctx.chordPrev[k] * ctx.scoreGrad[d] * prev[i];
        float w = ctx.optPolicy[id];
        ctx.optPolicy[id] = safe_w(w, w + ctx.alphaPi * td * grad);
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
        ctx.optReward[id] = safe_w(w, w + a * r_td * grad);
        return;
    }
    {                                                          // Head 4: diagonal optTransition
        // Diagonal: w_n^{k,j} updates with TD on its own feature only.
        //   n_hat^k_j(prev) = w * prev[j];  target = curr[j];  grad w.r.t. w = prev[j].
        // chordPrev[k] is the on-chord weight (importance ratio, RR-Subtasks eq. 17).
        uint id = gid - nVal - nPsi - nPi - nOR;
        uint k = id / ctx.R, j = id % ctx.R;
        float w = ctx.optTransition[id];
        float n_td = curr[j] - w * prev[j];
        float grad = ctx.chordPrev[k] * prev[j];
        ctx.optTransition[id] = safe_w(w, w + a * n_td * grad);
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
        // GraMa per-feature gradient utility (arXiv:2505.24061). Sum, no normalization:
        // ranking is scale-invariant so dividing by (1+K) is purely cosmetic.
        float gSum = fabs(ctx.td[0] * xi);
        for (uint k = 0u; k < ctx.K; ++k) gSum += fabs(ctx.tdPsi[k] * xi);
        ctx.utilFeat[i] += invT * (gSum - ctx.utilFeat[i]);
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
            ctx.optTransition[k * ctx.R + i] =
                (hash_uniform(ctx.stepCount, k, 2u * ctx.R + i) * 2.0f - 1.0f) * invSqrtR;
        }
        for (uint pi = 0u; pi < ctx.D * ctx.R; ++pi) {
            ctx.optPolicy[k * ctx.D * ctx.R + pi] =
                (hash_uniform(ctx.stepCount, k, 3u * ctx.R + pi) * 2.0f - 1.0f) * invSqrtR;
        }
        ctx.utilOpt[k] = 0.0f;
        ctx.counters[OAK_CNT_REINIT_OPT] += 1u;
    }

    // Worst feature: lowest utilFeat. Reseed encoder column with random projection.
    uint worstI = 0u; float worstUF = FLT_MAX;
    for (uint i = 0u; i < ctx.R; ++i)
        if (ctx.utilFeat[i] < worstUF) { worstUF = ctx.utilFeat[i]; worstI = i; }
    {
        uint i = worstI;
        for (uint o = 0u; o < ctx.baseO; ++o) {
            float u = hash_uniform(ctx.stepCount, i, o + 0xdeadbeefu) * 2.0f - 1.0f;
            ctx.encoder[i * ctx.baseO + o] = u * rsqrt(float(max(ctx.baseO, 1u)));
        }
        ctx.utilFeat[i] = 0.0f;
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
