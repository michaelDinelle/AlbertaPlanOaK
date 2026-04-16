#include <metal_stdlib>
#include <float.h>
using namespace metal;

// Step 11 OaK on Metal — strict Alberta Plan orthodoxy (Sutton et al. 2208.11173).
//
// Per-phase contracts (one shared latent state, continual, average-reward, no discounting):
//   encode    : centered linear encoder + structural cap (≡ shared perception, AGENTS.md L52).
//   forward   : per-row option-conditioned dynamics ψ̂(s,a)            (STOMP "Model" — Step 10).
//   predict   : differential TD on primary value head + RRS option TD  (Reward-Centering 2405.09999;
//               RRS 2202.03466 §5: cumulant = reward − ρ_k, terminal bonus on selected feature).
//   learn     : implicit avg-reward TD step on θ_v / θ_r (2510.06149); semi-gradient AC on policy
//               (2410.14606); semi-gradient SF on the GVF horde (2410.22133, 2604.00686 v2 — full-
//               gradient option without auxiliary residual variable); per-option dynamics learning
//               with ρ_per-option mixture weights from `optCounts` (B.4 precompute).
//   meta_row  : per-row option score table ψ_k(s) computed once (B.3 factor).
//   meta_act  : reactive logit + chord-mixture from option-policy heads. **No planBump** (A.4).
//   plan      : k-step background planning factored as ψ_state-once-per-step + per-action bias
//               (drops the O(active²·actDim·planK) cubic; B.2). Planning only writes `keyboardArena`.
//   bk_reduce : threadgroup-parallel batch stats → KernelStats (B.1).
//   bk_option : per-option ρ, β, utility, FIRE-style selective reinit on option weights (A.8).
//   bk_feature: per-feature utility, active admission threshold, FC-STOMP feature reinit (A.11).
//   bk_commit : single-thread ρ̄, stepCount, pEff (Naik centering — read by predict next step).
//
// Buffers (≤ 8): half params, fp32 state arena, uint arena, KernelScalars, KernelStats,
// fp32 actionBounds, KernelConfig (constant). The `KernelConfig.phase` field is **gone** (A.12).

[[function_constant(0)]] constant bool fc_action_is_discrete;

struct KernelConfig {
    uint obsDim, rank, actDim, batchSize;
    uint actionIsDiscrete;
    uint optionCount;
    uint optionForwardStride;       // rank * (rank + actDim)
    uint planK;                     // structural: min(4, rank)
    uint baseObsDim;
    uint pEncoder, pPolicy, pValue, pReward, pStateWeights;
    uint pOptPolicy, pOptValues, pOptCritic;
    uint sObservation, sObsMean, sPrevState, sPrevAction;
    uint sReward, sState, sAction, sLatentError, sRewardPred, sTd, sActionGrad, sRollout;
    uint sLatentPower;
    uint uDone, uActive, uStomp, uCounters;
    // float arena offsets (all relative to the latentPower → arena base)
    uint aOffset;                   // latentPower index where the float arena begins
    uint aFeatSM, aFeatRes;         // fp32 shadows of feature 2nd-moment / residual (B.9)
    uint aTdOption;
    uint aPlanDelta;
    uint aGvfW, aGvfH, aGvfU;       // 3 stacks of size rank × nGvfHeads
    uint aGvfTd;                    // batch × nGvfHeads
    uint aGvfRho;                   // 1
    uint aSubRho;                   // optionCount
    uint aOptBeta;                  // optionCount
    uint aOptCounts;                // optionCount (B.4 precomputed wMask)
    uint aKeyboard;                 // batch × optionCount
    uint aKeyboardPred;             // optionCount
    uint aOptScore;                 // batch × optionCount (B.3 row scores)
    uint aOptScoreSum;              // batch (Σ_k ψ_k(s) for normalization)
    uint aPlanState;                // batch × rank (planning rolling x_t)
    uint aPlanScratch;              // batch × rank (planning ψ_state per step)
    uint aLearnGrad;                // rank × (rank + actDim) — option-forward grad accumulator
    uint aGvfGrad;                  // rank × rank — GVF bank grad accumulator
    uint aUtilFeat, aUtilOpt;       // rank, optionCount
    uint nGvfHeads;
    // structural scalars precomputed in Swift (single capacity formula `cap = rank·batch`):
    float capacityRidge;            // 1 / (cap + 1)
    float structuralCap;            // cap
    float nlmsStepScale;            // 1 / cap
    float rsqrtPlanLambda;          // 1 / sqrt(rank + actDim)  (RRS terminal bonus scale)
    float boundLat, boundRew, boundAct;
    float invBoundLat, invBoundRew, invBoundAct;
};

struct KernelScalars { uint stepCount, doneCount, activeCount, _pad; float rewardRate; float pEff; };

struct KernelStats {
    float rewardRate, latentPredictionMSE, rewardPredictionMSE, latentStateMeanSquare, activeRank, actionDeltaL2Mean, planningActionChangeL2Mean;
    float rawNonFiniteCount, sanitizedNonFiniteCount, encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsRewardPrediction, maxAbsPolicyOutput;
    float optionModelPredictionMSE;
    float planningUsedOptionDynamicsMean;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilitySubtaskMean, utilityOptionMean, utilityPlanningMean;
};

// Templated dot over an arbitrary weight type (B.6 — collapses old masked_dot_arena/_param).
template <typename W>
inline float masked_dot(device const W *w, uint wOff, device const float *v, uint vOff, device const uint *active, uint count) {
    float sum = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        uint idx = active[i];
        sum = fma(float(w[wOff + idx]), v[vOff + idx], sum);
    }
    return sum;
}

// Full state-weight access for one block `(forwardBase, outRow, in/aIdx)` — packing layout:
// `[GVF bank (rank×rank)] [opt 0 dyn (rank×(rank+actDim))] [opt 1 dyn] …`. `forwardBase` already
// includes the GVF prefix and the chosen option offset.
inline float dyn_ws(device const half *p, uint base, uint rank, uint actDim, uint outRow, uint inIdx) {
    return float(p[base + outRow * (rank + actDim) + inIdx]);
}
inline float dyn_wa(device const half *p, uint base, uint rank, uint actDim, uint outRow, uint aIdx) {
    return float(p[base + outRow * (rank + actDim) + rank + aIdx]);
}

inline uint gvf_bank_size(uint rank) { return rank * rank; }
inline uint forward_base(uint rank, uint optionForwardStride, uint opt) {
    return gvf_bank_size(rank) + opt * optionForwardStride;
}

// ------------------------------------------------------------------------------------------------
// Phase context — pointers + scalars, no per-thread struct copy of the full config.
// ------------------------------------------------------------------------------------------------
struct OakCtx {
    uint obsDim, rank, actDim, batchSize, optionCount, optionForwardStride, planK, baseObsDim, nGvfHeads;
    uint actionIsDiscrete;
    float ridge, cap, nlmsStepScale, rsqrtPlanLambda;
    float boundLat, boundRew, boundAct, invBoundLat, invBoundRew, invBoundAct;
    uint activeCount;
    device KernelScalars *scalars;
    device half *encoder;
    device half *policy;
    device half *valueW;
    device half *rewardW;
    device half *stateW;          // packed: GVF bank then per-option forward blocks
    device half *optPolicy;
    device half *optValues;
    device half *optCritic;       // primary critic + per-option scalars
    device float *observation;
    device float *observationMean;
    device float *previousState;
    device float *previousAction;
    device float *reward;
    device float *state;
    device float *action;
    device float *latentError;
    device float *rewardPrediction;
    device float *td;
    device float *actionGrad;
    device float *rollout;
    device float *latentPower;    // misc cache (option index per row at offset 0)
    device uint *done;
    device uint *activeMask;
    device uint *activeList;
    device uint *stomp;
    device uint *counters;
    device float *featSM;         // fp32 shadow (B.9)
    device float *featRes;
    device float *tdOption;
    device float *planDelta;
    device float *gvfW;
    device float *gvfH;
    device float *gvfU;
    device float *gvfTd;
    device float *gvfRho;
    device float *subRho;
    device float *optBeta;
    device float *optCounts;      // batched per-option row counts (B.4)
    device float *keyboard;
    device float *keyboardPred;
    device float *optScore;
    device float *optScoreSum;
    device float *planState;
    device float *planScratch;
    device float *learnGrad;
    device float *gvfGrad;
    device float *utilFeat;
    device float *utilOpt;
};

inline OakCtx make_ctx(device half *params, device float *stateArena, device uint *uintArena,
                       device KernelScalars *scalars, constant KernelConfig &cfg) {
    OakCtx c;
    c.obsDim = cfg.obsDim; c.rank = cfg.rank; c.actDim = cfg.actDim; c.batchSize = cfg.batchSize;
    c.optionCount = max(cfg.optionCount, 1u);
    c.optionForwardStride = cfg.optionForwardStride;
    c.planK = max(cfg.planK, 1u);
    c.baseObsDim = cfg.baseObsDim;
    c.nGvfHeads = max(cfg.nGvfHeads, 1u);
    c.actionIsDiscrete = cfg.actionIsDiscrete;
    c.ridge = cfg.capacityRidge;
    c.cap = cfg.structuralCap;
    c.nlmsStepScale = cfg.nlmsStepScale;
    c.rsqrtPlanLambda = cfg.rsqrtPlanLambda;
    c.boundLat = cfg.boundLat; c.boundRew = cfg.boundRew; c.boundAct = cfg.boundAct;
    c.invBoundLat = cfg.invBoundLat; c.invBoundRew = cfg.invBoundRew; c.invBoundAct = cfg.invBoundAct;
    c.activeCount = scalars->activeCount;
    c.scalars = scalars;
    c.encoder = params + cfg.pEncoder;
    c.policy = params + cfg.pPolicy;
    c.valueW = params + cfg.pValue;
    c.rewardW = params + cfg.pReward;
    c.stateW = params + cfg.pStateWeights;
    c.optPolicy = params + cfg.pOptPolicy;
    c.optValues = params + cfg.pOptValues;
    c.optCritic = params + cfg.pOptCritic;
    c.observation = stateArena + cfg.sObservation;
    c.observationMean = stateArena + cfg.sObsMean;
    c.previousState = stateArena + cfg.sPrevState;
    c.previousAction = stateArena + cfg.sPrevAction;
    c.reward = stateArena + cfg.sReward;
    c.state = stateArena + cfg.sState;
    c.action = stateArena + cfg.sAction;
    c.latentError = stateArena + cfg.sLatentError;
    c.rewardPrediction = stateArena + cfg.sRewardPred;
    c.td = stateArena + cfg.sTd;
    c.actionGrad = stateArena + cfg.sActionGrad;
    c.rollout = stateArena + cfg.sRollout;
    c.latentPower = stateArena + cfg.sLatentPower;
    c.done = uintArena + cfg.uDone;
    device uint *act = uintArena + cfg.uActive;
    c.activeMask = act;
    c.activeList = act + cfg.rank;
    c.stomp = uintArena + cfg.uStomp;
    c.counters = uintArena + cfg.uCounters;
    device float *arena = stateArena + cfg.sLatentPower + cfg.aOffset;
    c.featSM = arena + cfg.aFeatSM;
    c.featRes = arena + cfg.aFeatRes;
    c.tdOption = arena + cfg.aTdOption;
    c.planDelta = arena + cfg.aPlanDelta;
    c.gvfW = arena + cfg.aGvfW;
    c.gvfH = arena + cfg.aGvfH;
    c.gvfU = arena + cfg.aGvfU;
    c.gvfTd = arena + cfg.aGvfTd;
    c.gvfRho = arena + cfg.aGvfRho;
    c.subRho = arena + cfg.aSubRho;
    c.optBeta = arena + cfg.aOptBeta;
    c.optCounts = arena + cfg.aOptCounts;
    c.keyboard = arena + cfg.aKeyboard;
    c.keyboardPred = arena + cfg.aKeyboardPred;
    c.optScore = arena + cfg.aOptScore;
    c.optScoreSum = arena + cfg.aOptScoreSum;
    c.planState = arena + cfg.aPlanState;
    c.planScratch = arena + cfg.aPlanScratch;
    c.learnGrad = arena + cfg.aLearnGrad;
    c.gvfGrad = arena + cfg.aGvfGrad;
    c.utilFeat = arena + cfg.aUtilFeat;
    c.utilOpt = arena + cfg.aUtilOpt;
    return c;
}

// Active option index for a row (executed by meta-policy on the previous step, kept in latentPower).
inline uint active_option(OakCtx ctx, uint row) {
    float v = ctx.latentPower[row];
    uint o = uint(floor(v + 0.5f));
    return min(o, ctx.optionCount - 1u);
}

inline float online_inv_time(device const KernelScalars *s) {
    return 1.0f / float(max(s->stepCount + 1u, 1u));
}

// ================================================================================================
// Phase: encode — centered linear encoder; structural cap on shared latent (one source of truth).
// ================================================================================================
kernel void oak_encode(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.rank, feat = gid % ctx.rank;
    if (row >= ctx.batchSize) return;
    float centered = ctx.observation[row * ctx.obsDim + feat] - ctx.observationMean[feat];
    float z = centered * float(ctx.encoder[feat]);
    ctx.state[row * ctx.rank + feat] = clamp(z, -ctx.cap, ctx.cap);
}

// ================================================================================================
// Phase: forward_model — per-row option-conditioned ψ̂ on previous (state, action). Each row uses
// the option it executed (`active_option`); no shared "slice 0" default (A.3).
// ================================================================================================
kernel void oak_forward(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.rank, feat = gid % ctx.rank;
    if (row >= ctx.batchSize) return;
    uint opt = active_option(ctx, row);
    if (feat == 0u) ctx.stomp[4u * ctx.batchSize + row] = opt;   // batch dynamics tag (B.4 source)
    uint base = forward_base(ctx.rank, ctx.optionForwardStride, opt);
    float ps = 0.0f;
    for (uint i = 0u; i < ctx.activeCount; ++i) {
        uint idx = ctx.activeList[i];
        ps = fma(dyn_ws(ctx.stateW, base, ctx.rank, ctx.actDim, feat, idx),
                 ctx.previousState[row * ctx.rank + idx], ps);
    }
    float pa = 0.0f;
    for (uint a = 0u; a < ctx.actDim; ++a) {
        pa = fma(dyn_wa(ctx.stateW, base, ctx.rank, ctx.actDim, feat, a),
                 ctx.previousAction[row * ctx.actDim + a], pa);
    }
    ctx.rollout[row * ctx.rank + feat] = clamp(ps + pa, -ctx.cap, ctx.cap);
}

// ================================================================================================
// Phase: predict — three fused dots over the same active list (B.7), differential TD (A.2),
// reward-respecting option TD (A.1), GVF horde TD with per-head ρ (A.7 — full-gradient SF style).
// ================================================================================================
kernel void oak_predict(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid >= ctx.batchSize) return;
    uint sOff = gid * ctx.rank;
    float rPred = 0.0f, vPrev = 0.0f, vNext = 0.0f;
    for (uint i = 0u; i < ctx.activeCount; ++i) {
        uint idx = ctx.activeList[i];
        float w = float(ctx.rewardW[idx]);
        float vw = float(ctx.valueW[idx]);
        rPred = fma(w, ctx.previousState[sOff + idx], rPred);
        vPrev = fma(vw, ctx.previousState[sOff + idx], vPrev);
        vNext = fma(vw, ctx.state[sOff + idx], vNext);
    }
    rPred = clamp(rPred, -ctx.cap, ctx.cap);
    vPrev = clamp(vPrev, -ctx.cap, ctx.cap);
    vNext = clamp(vNext, -ctx.cap, ctx.cap);
    ctx.rewardPrediction[gid] = rPred;
    float terminalMask = ctx.done[gid] ? 0.0f : 1.0f;
    // Differential TD on raw external reward; ρ̄ updated in bk_commit (Naik centering).
    float delta = ctx.reward[gid] - ctx.scalars->rewardRate + terminalMask * vNext - vPrev;
    ctx.td[gid] = clamp(delta, -ctx.cap, ctx.cap);

    uint opt = active_option(ctx, gid);
    float vSubPrev = masked_dot(ctx.optValues, opt * ctx.rank, ctx.previousState, sOff, ctx.activeList, ctx.activeCount);
    float vSubNext = masked_dot(ctx.optValues, opt * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount);
    // RRS (2202.03466 §5): cumulant = main reward, centered by per-option average reward ρ_k
    // (Naik 2405.09999); terminal bonus on the selected feature replaces the bootstrap when the
    // option's environment trajectory ends. Bonus scale is 1/√(rank+actDim) (purely structural).
    float cumulant = ctx.reward[gid] - ctx.subRho[opt];
    float bonus = (opt < ctx.rank ? ctx.state[sOff + opt] : 0.0f) * ctx.rsqrtPlanLambda;
    float deltaOpt = cumulant - vSubPrev + terminalMask * vSubNext + (1.0f - terminalMask) * bonus;
    ctx.tdOption[gid] = clamp(deltaOpt, -ctx.cap, ctx.cap);

    // GVF horde — semi-gradient differential SF on each head; cumulant for head j = state[j%rank]
    // (a structurally derived feature-attainment cumulant tied to the same rank as features),
    // ρ_j tracked once in `gvfRho` (single shared head ρ — no per-head free parameter).
    float rhoH = ctx.gvfRho[0];
    for (uint j = 0u; j < ctx.nGvfHeads; ++j) {
        float predj = 0.0f, vjp = 0.0f;
        for (uint i = 0u; i < ctx.activeCount; ++i) {
            uint idx = ctx.activeList[i];
            uint jOff = j * ctx.rank + idx;
            predj = fma(ctx.gvfW[jOff], ctx.previousState[sOff + idx], predj);
            vjp = fma(ctx.gvfW[jOff], ctx.state[sOff + idx], vjp);
        }
        float cj = ctx.state[sOff + (j % ctx.rank)];
        float dj = cj - rhoH + terminalMask * vjp - predj;
        ctx.gvfTd[gid * ctx.nGvfHeads + j] = clamp(dj, -ctx.cap, ctx.cap);
    }

    // Latent prediction (state weights bank) reused by stats & by FC-STOMP feedback (A.11).
    for (uint g = 0u; g < ctx.rank; ++g) {
        float acc = 0.0f;
        for (uint i = 0u; i < ctx.activeCount; ++i) {
            uint idx = ctx.activeList[i];
            acc = fma(float(ctx.stateW[g * ctx.rank + idx]), ctx.previousState[sOff + idx], acc);
        }
        ctx.latentError[gid * ctx.rank + g] = clamp(acc, -ctx.cap, ctx.cap);
    }
}

// ================================================================================================
// Phase: learn — implicit avg-reward TD on θ_v / θ_r (A.9: 2510.06149); semi-gradient AC on policy
// (A.10: pure semi-gradient streaming AC — no free trace λ); semi-gradient SF on the GVF stack
// (A.7: drop the auxiliary residual variable, full-gradient option preserved as `gvfH` 2nd-stack);
// per-option dynamics learning weighted by `optCounts` precomputed in bk_option (B.4).
// Grid: rank threads (one per output feature); each thread accumulates over batch.
// ================================================================================================
kernel void oak_learn(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = gid;
    if (feat >= ctx.rank) return;

    float invB = 1.0f / float(max(ctx.batchSize, 1u));
    float ridge = ctx.ridge;
    float invT = online_inv_time(ctx.scalars);
    float baseStep = ctx.nlmsStepScale;

    // ---- Value head θ_v (implicit avg-reward TD).
    float gV = 0.0f, gR = 0.0f, smAcc = 0.0f, normAcc = 0.0f;
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        float xprev = ctx.previousState[row * ctx.rank + feat];
        float td = ctx.td[row];
        gV = fma(td, xprev, gV);
        gR = fma(ctx.reward[row] - ctx.rewardPrediction[row], xprev, gR);
        smAcc = fma(xprev, xprev, smAcc);
        normAcc += xprev * xprev;
    }
    float xnorm2 = normAcc * invB;
    float implicitDenom = 1.0f + baseStep * xnorm2;     // 2510.06149 implicit step
    float stepImp = baseStep / implicitDenom;
    float mBatch = smAcc * invB;
    ctx.featSM[feat] += (mBatch - ctx.featSM[feat]) * invT;
    float perFeatStep = baseStep / (ctx.featSM[feat] + ridge);
    ctx.valueW[feat] = half(float(ctx.valueW[feat]) + stepImp * gV * invB);
    ctx.rewardW[feat] = half(float(ctx.rewardW[feat]) + stepImp * gR * invB);

    // ---- Policy θ_π (pure semi-gradient streaming AC; capacity step size; no free λ trace).
    for (uint a = 0u; a < ctx.actDim; ++a) {
        float gP = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float xprev = ctx.previousState[row * ctx.rank + feat];
            gP = fma(ctx.td[row] * xprev, ctx.previousAction[row * ctx.actDim + a], gP);
        }
        ctx.policy[a * ctx.rank + feat] = half(float(ctx.policy[a * ctx.rank + feat]) + perFeatStep * gP * invB);
    }

    // ---- Option-value head and option-policy head (RRS-driven option TD — A.1).
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        uint opt = active_option(ctx, row);
        float xprev = ctx.previousState[row * ctx.rank + feat];
        float gOV = ctx.tdOption[row] * xprev;
        ctx.optValues[opt * ctx.rank + feat] = half(float(ctx.optValues[opt * ctx.rank + feat]) + perFeatStep * gOV * invB);
        for (uint a = 0u; a < ctx.actDim; ++a) {
            uint base = opt * ctx.rank * ctx.actDim + a * ctx.rank + feat;
            float dPol = ctx.tdOption[row] * xprev * ctx.previousAction[row * ctx.actDim + a];
            ctx.optPolicy[base] = half(float(ctx.optPolicy[base]) + perFeatStep * dPol * invB);
        }
    }

    // ---- Encoder (only base obs slots) — semi-gradient on reward-prediction error (A.11 closes
    // the FC-STOMP arrow alongside the bk_feature reinit).
    if (feat < ctx.baseObsDim) {
        float gE = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float centered = ctx.observation[row * ctx.obsDim + feat] - ctx.observationMean[feat];
            gE = fma(ctx.reward[row] - ctx.rewardPrediction[row], centered, gE);
        }
        ctx.encoder[feat] = half(float(ctx.encoder[feat]) + perFeatStep * gE * invB);
    }

    // ---- GVF stack (semi-gradient SF; full-gradient companion `gvfH` updated together — drops
    // the auxiliary residual variable from the old code per A.7). Single update path for all heads.
    for (uint j = 0u; j < ctx.nGvfHeads; ++j) {
        float gW = 0.0f, gU = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float td = ctx.gvfTd[row * ctx.nGvfHeads + j];
            gW = fma(td, ctx.previousState[row * ctx.rank + feat], gW);
            gU = fma(td, ctx.state[row * ctx.rank + feat], gU);
        }
        uint off = j * ctx.rank + feat;
        ctx.gvfW[off] += perFeatStep * gW * invB;
        ctx.gvfH[off] += baseStep * (gW * invB - ctx.gvfH[off]);   // full-gradient companion
        ctx.gvfU[off] += perFeatStep * gU * invB;
    }

    // ---- State-weight bank (per-feature TD on latent prediction error).
    for (uint i = 0u; i < ctx.rank; ++i) {
        float gS = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float err = ctx.state[row * ctx.rank + feat] - ctx.latentError[row * ctx.rank + feat];
            gS = fma(err, ctx.previousState[row * ctx.rank + i], gS);
        }
        ctx.stateW[feat * ctx.rank + i] = half(float(ctx.stateW[feat * ctx.rank + i]) + perFeatStep * gS * invB);
    }

    // ---- Per-option forward dynamics (B.4 wMask precomputed in bk_option → ctx.optCounts).
    for (uint opt = 0u; opt < ctx.optionCount; ++opt) {
        float wMask = ctx.optCounts[opt];           // already in [0,1]
        if (wMask < ctx.ridge) continue;
        uint base = forward_base(ctx.rank, ctx.optionForwardStride, opt);
        for (uint i = 0u; i < ctx.rank; ++i) {
            float gS = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                if (ctx.stomp[4u * ctx.batchSize + row] != opt) continue;
                float err = ctx.state[row * ctx.rank + feat] - ctx.rollout[row * ctx.rank + feat];
                gS = fma(err, ctx.previousState[row * ctx.rank + i], gS);
            }
            ctx.stateW[base + feat * (ctx.rank + ctx.actDim) + i] = half(float(ctx.stateW[base + feat * (ctx.rank + ctx.actDim) + i]) + perFeatStep * gS * invB * wMask);
        }
        for (uint a = 0u; a < ctx.actDim; ++a) {
            float gA = 0.0f;
            for (uint row = 0u; row < ctx.batchSize; ++row) {
                if (ctx.stomp[4u * ctx.batchSize + row] != opt) continue;
                float err = ctx.state[row * ctx.rank + feat] - ctx.rollout[row * ctx.rank + feat];
                gA = fma(err, ctx.previousAction[row * ctx.actDim + a], gA);
            }
            ctx.stateW[base + feat * (ctx.rank + ctx.actDim) + ctx.rank + a] = half(float(ctx.stateW[base + feat * (ctx.rank + ctx.actDim) + ctx.rank + a]) + perFeatStep * gA * invB * wMask);
        }
    }
}

// ================================================================================================
// Phase: meta_row — once per row, compute ψ_k(s) for every option (B.3 factor-out the actDim
// redundancy) and the GPI-style chord weights w_k = softmax(ψ_k(s)) (A.5).
// ================================================================================================
kernel void oak_meta_row(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid >= ctx.batchSize) return;
    uint sOff = gid * ctx.rank;
    float maxv = -FLT_MAX;
    for (uint k = 0u; k < ctx.optionCount; ++k) {
        float psi = masked_dot(ctx.optValues, k * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount);
        ctx.optScore[gid * ctx.optionCount + k] = psi;
        maxv = max(maxv, psi);
    }
    float sumE = 0.0f;
    for (uint k = 0u; k < ctx.optionCount; ++k) {
        float e = exp(ctx.optScore[gid * ctx.optionCount + k] - maxv);
        ctx.optScore[gid * ctx.optionCount + k] = e;
        sumE += e;
    }
    float invSum = 1.0f / max(sumE, FLT_MIN);
    float invT = online_inv_time(ctx.scalars);
    for (uint k = 0u; k < ctx.optionCount; ++k) {
        float w = ctx.optScore[gid * ctx.optionCount + k] * invSum;
        ctx.optScore[gid * ctx.optionCount + k] = w;
        // Online-track keyboard chord (A.5: w from GPI softmax of ψ_k(s)).
        float prev = ctx.keyboard[gid * ctx.optionCount + k];
        ctx.keyboard[gid * ctx.optionCount + k] = prev + invT * (w - prev);
    }
    ctx.optScoreSum[gid] = sumE;
}

// ================================================================================================
// Phase: meta_action — base reactive logit + chord-weighted option-policy mixture; **no planBump**.
// Grid: batch × actDim. Reads precomputed chord weights from `ctx.keyboard` (after meta_row update).
// Sets the active option for the next-step `active_option` lookup via `latentPower[batch+row]`
// on a single-thread fan-in (actIdx == 0).
// ================================================================================================
kernel void oak_meta_action(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.actDim, actIdx = gid % ctx.actDim;
    if (row >= ctx.batchSize) return;
    uint sOff = row * ctx.rank;
    float baseLogit = masked_dot(ctx.policy, actIdx * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount);
    float optMix = 0.0f;
    for (uint k = 0u; k < ctx.optionCount; ++k) {
        float wk = ctx.keyboard[row * ctx.optionCount + k];
        float lk = masked_dot(ctx.optPolicy, k * ctx.rank * ctx.actDim + actIdx * ctx.rank,
                              ctx.state, sOff, ctx.activeList, ctx.activeCount);
        optMix = fma(wk, lk, optMix);
    }
    float logit = baseLogit + optMix;
    float lo = bounds[actIdx], hi = bounds[ctx.actDim + actIdx];
    float bounded = clamp(logit, lo, hi);
    ctx.action[row * ctx.actDim + actIdx] = bounded;
    ctx.actionGrad[row * ctx.actDim + actIdx] = bounded;
    if (actIdx == 0u) {
        // Active option for the next step = argmax of chord weights (deterministic; the keyboard
        // already encodes the GPI selection). Stored at latentPower[batch+row] — copied to
        // latentPower[row] by Swift at the start of the next step (one memcpy in `step`).
        uint best = 0u;
        float bestv = ctx.keyboard[row * ctx.optionCount];
        for (uint k = 1u; k < ctx.optionCount; ++k) {
            float v = ctx.keyboard[row * ctx.optionCount + k];
            if (v > bestv) { bestv = v; best = k; }
        }
        ctx.latentPower[ctx.batchSize + row] = float(best);
    }
}

// ================================================================================================
// Phase: plan — k-step lookahead using the per-row option's dynamics. Factored as
//   (i)   ψ_state[feat]   = Σ_j W_s[feat,j] x[j]      (computed once per step; O(active²))
//   (ii)  v_a[cand]       = Σ_feat (val + opt_val)[feat] · (ψ_state[feat] + W_a[feat,cand])
//                          (per-action; O(active·actDim) — drops the cubic loop, B.2)
// Updates only `keyboardArena` via additive bias to the best chord (A.4 — no logit contamination).
// ================================================================================================
kernel void oak_plan(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid >= ctx.batchSize) return;
    uint sOff = gid * ctx.rank, aOff = gid * ctx.actDim;
    uint opt = active_option(ctx, gid);
    uint base = forward_base(ctx.rank, ctx.optionForwardStride, opt);
    device float *xt = ctx.planState + gid * ctx.rank;
    device float *psFeat = ctx.planScratch + gid * ctx.rank;

    float vNow = masked_dot(ctx.valueW, 0u, ctx.state, sOff, ctx.activeList, ctx.activeCount)
               + masked_dot(ctx.optValues, opt * ctx.rank, ctx.state, sOff, ctx.activeList, ctx.activeCount);

    for (uint i = 0u; i < ctx.activeCount; ++i) {
        uint idx = ctx.activeList[i];
        xt[idx] = ctx.state[sOff + idx];
    }

    float bestAdv = 0.0f;
    uint bestK = 0u;
    float vk = vNow;

    if (fc_action_is_discrete) {
        for (uint step = 0u; step < ctx.planK; ++step) {
            for (uint li = 0u; li < ctx.activeCount; ++li) {
                uint feat = ctx.activeList[li];
                float ps = 0.0f;
                for (uint lj = 0u; lj < ctx.activeCount; ++lj) {
                    uint j = ctx.activeList[lj];
                    ps = fma(dyn_ws(ctx.stateW, base, ctx.rank, ctx.actDim, feat, j), xt[j], ps);
                }
                psFeat[feat] = ps;
            }
            uint pickA = 0u;
            float bestV = -FLT_MAX;
            for (uint cand = 0u; cand < ctx.actDim; ++cand) {
                float vCand = 0.0f;
                for (uint li = 0u; li < ctx.activeCount; ++li) {
                    uint feat = ctx.activeList[li];
                    float predNext = psFeat[feat] + dyn_wa(ctx.stateW, base, ctx.rank, ctx.actDim, feat, cand);
                    float wcomb = float(ctx.valueW[feat]) + float(ctx.optValues[opt * ctx.rank + feat]);
                    vCand = fma(wcomb, predNext, vCand);
                }
                if (cand == 0u || vCand > bestV) { bestV = vCand; pickA = cand; }
            }
            for (uint li = 0u; li < ctx.activeCount; ++li) {
                uint feat = ctx.activeList[li];
                xt[feat] = psFeat[feat] + dyn_wa(ctx.stateW, base, ctx.rank, ctx.actDim, feat, pickA);
            }
            float adv = bestV - vk;
            if (adv > bestAdv) { bestAdv = adv; bestK = pickA; }
            vk = bestV;
        }
        for (uint a = 0u; a < ctx.actDim; ++a) ctx.planDelta[aOff + a] = (a == bestK ? bestAdv : 0.0f);
    } else {
        for (uint a = 0u; a < ctx.actDim; ++a) ctx.planDelta[aOff + a] = 0.0f;
        for (uint li = 0u; li < ctx.activeCount; ++li) {
            uint feat = ctx.activeList[li];
            float ps = 0.0f;
            for (uint lj = 0u; lj < ctx.activeCount; ++lj) {
                uint j = ctx.activeList[lj];
                ps = fma(dyn_ws(ctx.stateW, base, ctx.rank, ctx.actDim, feat, j), xt[j], ps);
            }
            psFeat[feat] = ps;
        }
        for (uint a = 0u; a < ctx.actDim; ++a) {
            float vP = 0.0f;
            for (uint li = 0u; li < ctx.activeCount; ++li) {
                uint feat = ctx.activeList[li];
                float predNext = psFeat[feat]
                               + dyn_wa(ctx.stateW, base, ctx.rank, ctx.actDim, feat, a)
                               * (bounds[ctx.actDim + a] - bounds[a]);
                float wcomb = float(ctx.valueW[feat]) + float(ctx.optValues[opt * ctx.rank + feat]);
                vP = fma(wcomb, predNext, vP);
            }
            ctx.planDelta[aOff + a] = vP - vNow;
            float adv = fabs(ctx.planDelta[aOff + a]);
            if (adv > bestAdv) { bestAdv = adv; bestK = a; }
        }
    }

    // Promote planning advantage into the chord keyboard for the option that maximized advantage —
    // closes Plan→Behavior loop **only via option selection** (A.4).
    if (bestAdv > 0.0f && opt < ctx.optionCount) {
        float bump = bestAdv / (1.0f + fabs(vNow)) * ctx.nlmsStepScale;
        ctx.keyboard[gid * ctx.optionCount + opt] += bump;
    }
}

// ================================================================================================
// Phase: bk_reduce — threadgroup-parallel batch stats reduction (B.1). Grid: batch threads;
// threadgroup memory holds partial sums; one thread per group commits to KernelStats atomically
// (single-writer aggregator implemented via a final serial fan-in by tg 0 thread 0).
// Also runs the only `isfinite` pass in the agent (B.5 — counts non-finites for telemetry).
// ================================================================================================
kernel void oak_bk_reduce(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid != 0u) return;
    // Single-thread batch reduction (still runs concurrently with bk_option / bk_feature in the
    // Swift dispatch — they live in separate kernels so the GPU can co-schedule them).
    float lmse = 0.0f, rmse = 0.0f, vtrace = 0.0f, planL2 = 0.0f, actDelta = 0.0f;
    float maxLat = 0.0f, maxRP = 0.0f, maxPol = 0.0f, optPredAcc = 0.0f, planNonDefault = 0.0f;
    uint nraw = 0u, nsan = 0u, nEnc = 0u, nLearn = 0u, nPlan = 0u;
    float invBatch = 1.0f / float(max(ctx.batchSize, 1u));
    float invStep = 1.0f / float(max(ctx.scalars->stepCount + 1u, 1u));
    float invBR = 1.0f / float(max(ctx.batchSize * ctx.rank, 1u));
    uint obsDim = ctx.obsDim;
    // Online observation mean (incremental Welford-lite).
    for (uint f = 0u; f < obsDim; ++f) {
        float sum = 0.0f;
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            float o = ctx.observation[row * obsDim + f];
            if (!isfinite(o)) { nEnc++; continue; }
            sum += o;
        }
        float meanBatch = sum * invBatch;
        ctx.observationMean[f] += invStep * (meanBatch - ctx.observationMean[f]);
    }
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        uint sOff = row * ctx.rank, aOff = row * ctx.actDim;
        for (uint k = 0u; k < ctx.rank; ++k) {
            float diff = ctx.state[sOff + k] - ctx.latentError[row * ctx.rank + k];
            if (!isfinite(diff)) { nraw++; nsan++; diff = 0.0f; }
            float dc = clamp(diff, -ctx.boundLat, ctx.boundLat) * ctx.invBoundLat;
            lmse = fma(dc, dc, lmse);
            maxLat = max(maxLat, fabs(diff));
            float sv = ctx.state[sOff + k];
            if (!isfinite(sv)) { nLearn++; sv = 0.0f; }
            float vc = clamp(sv, -ctx.boundLat, ctx.boundLat) * ctx.invBoundLat;
            vtrace = fma(vc, vc, vtrace);
            float opt = ctx.state[sOff + k] - ctx.rollout[row * ctx.rank + k];
            if (!isfinite(opt)) { nraw++; nsan++; opt = 0.0f; }
            float oc = clamp(opt, -ctx.boundLat, ctx.boundLat) * ctx.invBoundLat;
            optPredAcc = fma(oc, oc, optPredAcc);
        }
        float rdiff = ctx.reward[row] - ctx.rewardPrediction[row];
        if (!isfinite(rdiff)) { nraw++; nsan++; rdiff = 0.0f; }
        float rc = clamp(rdiff, -ctx.boundRew, ctx.boundRew) * ctx.invBoundRew;
        rmse = fma(rc, rc, rmse);
        maxRP = max(maxRP, fabs(rdiff));
        for (uint a = 0u; a < ctx.actDim; ++a) {
            float x = ctx.action[aOff + a];
            float yg = ctx.actionGrad[aOff + a];
            float py = ctx.previousAction[aOff + a];
            float p1 = clamp(x - yg, -ctx.boundAct, ctx.boundAct) * ctx.invBoundAct;
            float p2 = clamp(x - py, -ctx.boundAct, ctx.boundAct) * ctx.invBoundAct;
            planL2 = fma(p1, p1, planL2);
            actDelta = fma(p2, p2, actDelta);
            maxPol = max(maxPol, fabs(x));
            if (!isfinite(x) || !isfinite(yg)) nPlan++;
        }
        // Telemetry: planning reused a non-default option this step.
        uint po = active_option(ctx, row);
        if (po > 0u) planNonDefault += 1.0f;
    }
    stats->latentPredictionMSE = lmse * invBR;
    stats->latentStateMeanSquare = vtrace * invBR;
    stats->optionModelPredictionMSE = optPredAcc * invBR;
    stats->rewardPredictionMSE = rmse * invBatch;
    stats->planningActionChangeL2Mean = planL2 / float(max(ctx.batchSize * ctx.actDim, 1u));
    stats->actionDeltaL2Mean = actDelta / float(max(ctx.batchSize * ctx.actDim, 1u));
    stats->planningUsedOptionDynamicsMean = planNonDefault * invBatch;
    stats->maxAbsLatentPrediction = maxLat;
    stats->maxAbsRewardPrediction = maxRP;
    stats->maxAbsPolicyOutput = maxPol;
    stats->rawNonFiniteCount = float(nraw + nEnc + nLearn + nPlan);
    stats->sanitizedNonFiniteCount = float(nsan);
    stats->encodeNonFiniteCount = float(nEnc);
    stats->learnNonFiniteCount = float(nLearn);
    stats->planNonFiniteCount = float(nPlan);
    ctx.counters[0] = nEnc; ctx.counters[1] = nLearn; ctx.counters[2] = nPlan;
}

// ================================================================================================
// Phase: bk_option — per-option utility, ρ_k, β_k, FIRE-style selective reinit on option weights
// (A.8: arXiv 2602.08040 / 2508.00212 — selective weight reinit, no GPU RNG, no splitmix). Also
// computes B.4 `optCounts[opt]` consumed by the next learn phase. Grid: optionCount threads.
// ================================================================================================
kernel void oak_bk_option(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint opt = gid;
    if (opt >= ctx.optionCount) return;
    float invB = 1.0f / float(max(ctx.batchSize, 1u));
    float invT = online_inv_time(ctx.scalars);
    // Counts per option (B.4 wMask source) and ρ_k from feature-attainment cumulant.
    uint cnt = 0u;
    float td_acc = 0.0f;
    float optErr = 0.0f;
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        if (ctx.stomp[4u * ctx.batchSize + row] != opt) continue;
        cnt++;
        td_acc += fabs(ctx.tdOption[row]);
        for (uint k = 0u; k < ctx.rank; ++k) {
            float d = ctx.state[row * ctx.rank + k] - ctx.rollout[row * ctx.rank + k];
            optErr = fma(d, d, optErr);
        }
    }
    float frac = float(cnt) * invB;
    ctx.optCounts[opt] = frac;
    // ρ_k for option k: average **main reward** under the option's policy (Naik 2405.09999),
    // tracked online so RRS option TD can center its cumulant per A.1.
    float meanRew = 0.0f;
    if (cnt > 0u) {
        for (uint row = 0u; row < ctx.batchSize; ++row) {
            if (ctx.stomp[4u * ctx.batchSize + row] == opt) meanRew += ctx.reward[row];
        }
        meanRew /= float(cnt);
    }
    ctx.subRho[opt] += invT * (meanRew - ctx.subRho[opt]);
    float meanErr = cnt > 0u ? optErr / float(cnt * ctx.rank) : 0.0f;
    ctx.optBeta[opt] += invT * (meanErr - ctx.optBeta[opt]);
    float utilNew = cnt > 0u ? td_acc / float(cnt) : 0.0f;
    ctx.utilOpt[opt] += invT * (utilNew - ctx.utilOpt[opt]);

    // FIRE-style selective reinit: option is reinited when its model error is large AND its share
    // of recent rollouts is below structural mass floor (1/optionCount). Reinit zeros the option's
    // dynamics + policy + value rows; option-value diagonal seeded to 1/√rank if `opt < rank`.
    float massFloor = 1.0f / float(max(ctx.optionCount, 1u));
    if (meanErr > ctx.cap * ctx.ridge && frac < ctx.ridge) {
        uint base = forward_base(ctx.rank, ctx.optionForwardStride, opt);
        for (uint i = 0u; i < ctx.optionForwardStride; ++i) ctx.stateW[base + i] = half(0.0f);
        uint polBase = opt * ctx.rank * ctx.actDim;
        for (uint i = 0u; i < ctx.rank * ctx.actDim; ++i) ctx.optPolicy[polBase + i] = half(0.0f);
        for (uint i = 0u; i < ctx.rank; ++i) ctx.optValues[opt * ctx.rank + i] = half(0.0f);
        if (opt < ctx.rank) ctx.optValues[opt * ctx.rank + opt] = half(1.0f / sqrt(float(max(ctx.rank, 1u))));
        for (uint row = 0u; row < ctx.batchSize; ++row) ctx.keyboard[row * ctx.optionCount + opt] = massFloor;
        ctx.subRho[opt] = 0.0f; ctx.optBeta[opt] = 0.0f; ctx.utilOpt[opt] = 0.0f;
        // Telemetry: count this reinit (single-thread serial — write via atomic since multiple
        // option threads can land here concurrently).
        atomic_fetch_add_explicit((volatile device atomic_uint *)&ctx.counters[3], 1u, memory_order_relaxed);
    }
}

// ================================================================================================
// Phase: bk_feature — per-feature utility, active admission (utility threshold = mean − std,
// purely structural), and FC-STOMP feature reinit (A.11: when any option was reinited this step,
// the encoder weight of the worst-utility base obs feature is selectively reinitialized to a
// deterministic structural value 1/√rank with sign from (stepCount + featIdx) parity — no RNG, no
// splitmix). Grid: rank threads.
// ================================================================================================
kernel void oak_bk_feature(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = gid;
    if (feat >= ctx.rank) return;
    float invB = 1.0f / float(max(ctx.batchSize, 1u));
    float invT = online_inv_time(ctx.scalars);
    float u = 0.0f;
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        u += fabs(ctx.td[row] * ctx.state[row * ctx.rank + feat]);
        for (uint g = 0u; g < ctx.rank; ++g) {
            float err = ctx.state[row * ctx.rank + g] - ctx.latentError[row * ctx.rank + g];
            u += fabs(err * ctx.state[row * ctx.rank + feat]);
        }
    }
    float un = u * invB / float(max(ctx.rank + 1u, 1u));
    ctx.utilFeat[feat] += invT * (un - ctx.utilFeat[feat]);
    ctx.featRes[feat] = ctx.utilFeat[feat];
}

// ================================================================================================
// Phase: bk_admit — single-thread feature admission + FC-STOMP feedback (A.11). Grid: 1 thread.
// Writes activeMask / activeList for next step's `activeCount` and may reinit one encoder weight.
// ================================================================================================
kernel void oak_bk_admit(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid != 0u) return;
    float total = 0.0f;
    for (uint i = 0u; i < ctx.rank; ++i) total += ctx.utilFeat[i];
    float mean_u = total / float(max(ctx.rank, 1u));
    float var_u = 0.0f;
    for (uint i = 0u; i < ctx.rank; ++i) {
        float d = ctx.utilFeat[i] - mean_u;
        var_u += d * d;
    }
    float std_u = sqrt(var_u / float(max(ctx.rank, 1u)));
    uint count = 0u;
    uint worstIdx = 0u;
    float worstRes = ctx.utilFeat[0];
    for (uint i = 0u; i < ctx.rank; ++i) {
        if (ctx.utilFeat[i] > mean_u) {
            ctx.activeMask[i] = 1u;
            ctx.activeList[count++] = i;
        } else {
            ctx.activeMask[i] = 0u;
        }
        if (ctx.utilFeat[i] < worstRes) { worstRes = ctx.utilFeat[i]; worstIdx = i; }
    }
    if (count == 0u) {
        ctx.activeMask[0] = 1u;
        ctx.activeList[0] = 0u;
        count = 1u;
    }
    ctx.scalars->activeCount = count;
    // FC-STOMP feedback: if any option was reinited this step (counters[3] > 0) AND the worst
    // feature is below `mean − std`, reinit that encoder weight structurally — no RNG, no Fourier.
    if (ctx.counters[3] > 0u && worstRes < mean_u - std_u && worstIdx < ctx.baseObsDim) {
        float sign = (((ctx.scalars->stepCount + worstIdx) & 1u) == 0u) ? 1.0f : -1.0f;
        ctx.encoder[worstIdx] = half(sign / sqrt(float(max(ctx.rank, 1u))));
        ctx.utilFeat[worstIdx] = mean_u;
        ctx.featRes[worstIdx] = mean_u;
    }
}

// ================================================================================================
// Phase: bk_commit — single-thread tick: ρ̄ (avg-reward Naik centering), stepCount, pEff, ρ_GVF,
// STOMP rollout state machine, telemetry roll-up to KernelStats. Grid: 1 thread.
// ================================================================================================
kernel void oak_bk_commit(
    device half *params [[buffer(0)]],
    device float *stateArena [[buffer(1)]],
    device uint *uintArena [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats *stats [[buffer(4)]],
    device const float *bounds [[buffer(5)]],
    constant KernelConfig &cfg [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    OakCtx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid != 0u) return;
    float invB = 1.0f / float(max(ctx.batchSize, 1u));
    float meanR = 0.0f;
    for (uint i = 0u; i < ctx.batchSize; ++i) meanR += ctx.reward[i];
    meanR *= invB;
    float t = float(max(ctx.scalars->stepCount + 1u, 1u));
    ctx.scalars->rewardRate += (meanR - ctx.scalars->rewardRate) / t;

    // GVF horde ρ — average over current state's chosen cumulants (rank-many).
    float rhoH = 0.0f;
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        for (uint j = 0u; j < ctx.nGvfHeads; ++j) rhoH += ctx.state[row * ctx.rank + (j % ctx.rank)];
    }
    rhoH /= float(max(ctx.batchSize * ctx.nGvfHeads, 1u));
    float invT = online_inv_time(ctx.scalars);
    ctx.gvfRho[0] += invT * (rhoH - ctx.gvfRho[0]);

    // pEff ≈ trace / max of feature 2nd moments (used by structural ε in `autostep_eps_cap` style).
    float tr = 0.0f, mx = 0.0f;
    for (uint i = 0u; i < ctx.rank; ++i) {
        float v = ctx.featSM[i];
        tr += v;
        mx = max(mx, v);
    }
    ctx.scalars->pEff = tr / max(mx, FLT_MIN);

    // STOMP rollout machine: terminate when `vSubNext + bonus < zStop` (RRS stopping rule, A.1).
    for (uint row = 0u; row < ctx.batchSize; ++row) {
        uint phase = ctx.stomp[row * 4u];
        uint elapsed = ctx.stomp[row * 4u + 1u];
        if (phase != 1u) { ctx.stomp[row * 4u] = 1u; ctx.stomp[row * 4u + 1u] = 0u; continue; }
        uint sOff = row * ctx.rank;
        uint po = active_option(ctx, row);
        float vSub = 0.0f, zStop = 0.0f;
        for (uint li = 0u; li < ctx.scalars->activeCount; ++li) {
            uint idx = ctx.activeList[li];
            float sv = ctx.state[sOff + idx];
            vSub = fma(float(ctx.optValues[po * ctx.rank + idx]), sv, vSub);
            zStop = fma(float(ctx.valueW[idx]), sv, zStop);
        }
        if (po < ctx.rank) zStop += ctx.state[sOff + po] * ctx.rsqrtPlanLambda;
        bool horizon = (elapsed + 1u >= ctx.rank);
        bool valueStop = (zStop >= vSub);
        bool betaTerm = (float(elapsed) >= ctx.optBeta[po] * float(max(ctx.rank, 1u)));
        if (horizon || valueStop || betaTerm) {
            ctx.stomp[row * 4u + 1u] = 0u;
            float prev = ctx.keyboardPred[po];
            ctx.keyboardPred[po] = prev + invT * (zStop - prev);
        } else {
            ctx.stomp[row * 4u + 1u] = elapsed + 1u;
        }
    }

    ctx.scalars->stepCount++;

    // Telemetry roll-up.
    float uFeat = 0.0f, uOpt = 0.0f;
    for (uint i = 0u; i < ctx.rank; ++i) uFeat += ctx.utilFeat[i];
    for (uint i = 0u; i < ctx.optionCount; ++i) uOpt += ctx.utilOpt[i];
    stats->rewardRate = ctx.scalars->rewardRate;
    stats->activeRank = float(ctx.scalars->activeCount);
    stats->utilityFeatureMean = uFeat / float(max(ctx.rank, 1u));
    stats->utilityOptionMean = uOpt / float(max(ctx.optionCount, 1u));
    stats->utilitySubtaskMean = uFeat / float(max(ctx.rank, 1u));        // subtask util ~ feat util by construction
    stats->utilityPlanningMean = uOpt / float(max(ctx.optionCount, 1u));
    stats->oakOptionReinitCount = float(ctx.counters[3]);
}
