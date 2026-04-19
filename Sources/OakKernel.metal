#include <metal_stdlib>
#include <float.h>
using namespace metal;

// =================================================================================================
// Orthodox Step 11 OaK on Metal — one shared latent, continual, average-reward, no discounting.
// Invariants (.cursor/rules/alberta_plan.mdc §1–9) hold: no env branches, no tuned scalars, no
// episodic semantics, no train/eval split, no per-step allocation, no cosmetic normalization,
// no curiosity on the external reward, meta-constants structurally derived from {R, B, K}.
// Eighteen kernels: encode, predict_outcome, meta_policy, predict, plan,
// reduce_alpha, learn_value, learn_encoder, learn_outcome, learn_psi, learn_pi, learn_beta,
// learn_omega, meta_action, bk_reduce, bk_feature, bk_option, bk_admit_commit.
//   ed oak_predict_outcome + oak_learn_outcome for OPSR full state
//   augmentation (Vereecken 2604.07016): E = R + K extended state. The outcome head H[K · E]
//   writes K next-step cumulant predictions per row recursively from the previous step's
//   full E-dim state, giving the agent partial recurrence without an LSTM.
//   ed oak_meta_policy + oak_learn_beta + oak_learn_omega for temporal
//   abstraction (Bacon-Precup 1609.05140 + Wan-Naik-Sutton 2110.13855): persistent soft
//   chord, β termination, ω chord generator. All linear heads consume the OPSR-extended
//   E-dim state; cumulant ranking and admission still operate on the first R encoder
//   features only. (bk_reduce and bk_feature are kept separate because the per-feature
//   utility pass needs R parallel threadgroups to avoid 32× serialization at large R·B;
//   with MTLIndirectCommandBuffer the extra dispatch has ~zero host cost, so fusion is
//   not profitable here.)
// Citations (docs/papers/):
//   AlbertaPlan_2208.11173                      base agent; Step 11 Oak; STOMP progression
//   RethinkingFoundationsContinualRL_2504.08161 Bowling self-critique of episodic baggage
//   RewardRespectingSubtasks_2202.03466 §6.1    w̄ high-water-mark bonus weight (F1/F2)
//   RewardCentering_2405.09999                  differential TD w/ ρ̄ centering (main + per-opt)
//   BellmanErrorCentering_2502.03104            theoretical identity — VRC ≡ BEC, linear-FA proof
//   ImplicitUpdates_AvgRewardTD_2510.06149      αimp = α0/(1+α0·‖x‖²) — F10 on every linear head
//   ImplicitQLearningSarsa_2601.18907           implicit TD extended to control
//   StreamingDRLFinallyWorks_2410.14606         streaming AC (semi-grad; λ-trace rejected per Inv 9)
//     [known-superior-but-rejected: GradientEligibilityTraces_2507.09087]
//   FullGradientSuccessorFeatureRepresentations_2604.00686  ψ[k,j,i] vector SF Bellman
//   DifferentialTDConvergence_2602.16629        on-policy n-step differential TD a.s. converges
//                                                without a local clock — learn_psi gating justified
//   TaleOfTwoTDErrors_2603.21921                implicit-TD form is the stability-preserving choice
//   OptionKeyboard_2106.13105                   Alg 1 line 3: argmax_k ψ^{π_k}·w (GPI)
//   OptimalBehaviorBasis_OptionKeyboard_2505.00787  OBB: minimal behavior basis for GPI
//   SuccessorFeaturesKeyboard_2310.15940        SFK: ψ_k(s)·w option scoring
//   CompositionalPlanningJumpyWorldModels_2602.19634  GPI = α₁=1 limit of CompPlan; oak_plan sits here
//   SpectralAlignmentFB_2603.20103              temporal abstraction lowers effective SR rank → K
//   FIRE_2602.08040 / SWR_2508.00212 / SpectralCollapsePlasticity_2509.22335
//                                                selective weight reinit + effective-rank floor
//   MetaGradientSearchControl_2406.19561        planning TD on w_main (F4)
//   ValueHallucinationDynaPlanning_2006.04363   planning-only-from-current-state (avoid VH)
//   TowardAgentsThatReasonAboutTheirComputation_2510.22833  planK structurally bounded
//   WhichFeaturesBestForSuccessorFeatures_2502.10790        cumulant caveat (identity cumulants OK)
//   TTT-Options-Sutton-2025-slides              activate per step: O*_t = argmax_o q̂(s, o, w⁰)
//   RankGradientLost_SWD_2604.01913             Θ(1/k) gradient decay — motivates track_rate floor
//   RethinkingPlasticity_OCP_2603.21173         OCP hypothesis validates conjunctive reinit trigger
//   WhyNetworksForget_2603.04580                forgetting ↔ eRank collapse confirmation for pEff
//   ParameterFreeTD_2603.02577                  α_imp parameter-free validated (no ω, no τ_mix)
//   RelativeTD_Stability_2603.27874             Naik centering stable under FA for any baseline ≥ 0
//   GraMa_MeasureGradients_2505.24061           gradient-magnitude = correct utility signal (we use it)
// Params (fp16, packed):
//   encoder[R]                                  linear encoder weights W_e (diagonal)
//   valueW[R]                                   w_main (main-task linear value)
//   psi[K · M]                                  W_ψ[k] · concat[s, λ]: shared chord-conditional
//                                               SF (Design 2; OBB §2.4 Eq.4). M = E + K (
//                                               Tier 2: ψ's state weights span the full E-dim
//                                               extended state including outcome augmentations).
//   optPolicy[K · D · E]                        per-option policy parameters over E-dim extended
//                                               state. D = D_policy is the env-declared
//                                               action-spec parameter count.
//   outcomeHead[K · E]                          OPSR outcome head: K outcome predictors (one per
//                                               option-keyboard slot), each a linear projection
//                                               from the previous E-dim state. Writes
//                                               state[1-slot][r][R + k] = H[k] · state[slot][r].
// State arena (fp32):
//   observation[B·O]                            centered → state[slot · B · E][:R]
//   observationMean[O]                          online Welford mean for encoder centering
//   state[2 · B · E]                            extended state: [:R] = encoder output;
//                                               [R:E] = K outcome predictions (recursive from
//                                               previous step's E-dim state via outcomeHead).
//                                               Double-buffered (slot toggles each step; S5).
//   prevPolicyMean[B·D]                         previous step's policyMean (for actionDelta telemetry)
//   policyMean[B·D]                             current step's policy mean — written by oak_meta_action;
//                                               read by host wrapper which samples + computes scoreGrad
//   scoreGrad[B · D]                            host-written ∇log π(a|s) for the wrapper-sampled action
//   reward[B], td[B]
//   tdPsi[B · K]                                K cumulant TDs per row (Design 2 collapse)
// UInt arena:
//   done[B], activeMask[E], activeList[E]                (E = R + K; outcome dims R..E always-active)
//   option[2 · B]                               slot-toggled active/next option index (S4)
//   counters[8]                                 {encNF, learnNF, planNF, reinitOptCount,
//                                                reinitFeatCount, _, tdPsiNonFinite, _}
// Arena extras (fp32):
//   featSM[R]                                   batch 2nd-moment EMA of active x (utility norm)
//   wBar[R]                                     high-water-mark of |w_main| (RRS §6.1)
//   featMean[R]                                 per-feature running mean (cumulant centering ρ_k)
//   utilFeat[R], utilOpt[K]                     running utilities for admission / reinit
// =================================================================================================

// Structural SIMD-group width on Apple GPUs. Both an assertion boundary (K ≤ 32 guarded in host
// via OakMetalLayout.optionCount) and the TG-size unit used by all reduction kernels.
constant uint OAK_SIMD = 32u;

// Named offsets into counters[8]. Self-documenting replacement for hardcoded indices 0..7.
// All telemetry counters live here; layout is fixed in OakMetalLayout.uintLayout.
constant uint OAK_CNT_OBS_NF        = 0u;  // bk_reduce: non-finite obs count (per-step)
constant uint OAK_CNT_LEARN_NF      = 1u;  // bk_reduce: non-finite latent / TD-target count (per-step)
constant uint OAK_CNT_PLAN_NF       = 2u;  // bk_reduce: non-finite action / scoreGrad count (per-step)
constant uint OAK_CNT_REINIT_OPT    = 3u;  // bk_option: bitmask of options reinited this step
                                            //   (K ≤ OAK_SIMD = 32, so one uint suffices)
constant uint OAK_CNT_REINIT_FEAT   = 4u;  // bk_admit_commit: encoder-feature reinit count (cumulative)
// numerical-safety counters.
// Index 5 is reserved.
constant uint OAK_CNT_TDPSI_NF      = 6u;  // bk_reduce: tdPsi non-finite count (per-step). Was
                                            //   silently passing through to optionModelPredictionMSE
                                            //, contaminating telemetry. Now scanned +
                                            //   sanitized to zero before MSE roll-up.
// Index 7 is reserved.

// No function constants. : the agent has zero a-priori knowledge of action-
// space type. optPolicy is always K · 2·A · R (Gaussian μ_d, log_σ_d per action dim); the env-
// wrapper layer (EnvPoolEnv.step in Main.swift) is the SOLE site that interprets the agent's
// continuous-vector output (argmax for discrete envs, bound-clamp for continuous envs).
// Inv-1: no env-conditioned branches in agent; Inv-3: no env-type-conditioned widths.

struct KernelConfig {
    uint obsDim, rank, actDim, batch, optionCount, baseObsDim;
    // D_policy = env-declared sum of per-component policy-param widths.
    // For single-Discrete-N envs D = N; for single-Box-A envs D = A; multi-component envs sum.
    // The agent has no notion of WHAT the components are — only the integer width.
    uint actionParamCount;
    // Param offsets (in fp16 stride). betaW[E + K] (termination head)
    // and omegaW[K · E] (chord generator) for temporal abstraction.
    uint pEncoder, pValue, pPsi, pOptPolicy, pOutcomeHead, pBetaW, pOmegaW;
    // sprDecoder[K · R] chord-conditioned diagonal next-state predictor (SPR-Stream).
    uint pSprDecoder;
    // State offsets (in fp32 stride). policyMean[B · D] is written by oak_meta_action and
    // read by the host wrapper for sampling. scoreGrad[B · D] is HOST-written (after wrapper
    // samples per-component) and read by oak_learn_pi at the *next* step. prevPolicyMean[B · D]
    // holds the previous step's μ for actionDelta telemetry. vCurr[B] caches w_main · curr.
    // chordSlot[B · K], chordSlotPrev[B · K], betaProb[B] for temporal abstraction.
    uint sObservation, sObsMean, sState, sPrevPolicyMean, sPolicyMean, sScoreGrad;
    uint sReward, sTd, sTdPsi, sVCurr;
    uint sChordSlot, sChordSlotPrev, sBetaProb;
    // sSprPred removed (oak_spr_fused recomputes ẑ_N inline; saves ~3 MB).
    // UInt offsets. subtaskFeatureOf[K] = state-feature index that cumulant k tracks
    // (top-K by per-feature utility, re-ranked by oak_bk_admit_commit each step; Tier 7 RED).
    // optionAge[B] tracks steps since last termination for each row.
    uint uDone, uActiveMask, uActiveList, uOption, uCounters, uSubtaskFeatureOf, uOptionAge;
    // Arena offsets (in fp32 stride, past sState's tail). featMean[R] is the running EMA of
    // current-step feature values used as cumulant centering ρ_k = featMean[sf[k]] (RED).
    // optFracEMA[K] tracks soft-chord per-option mass for the dormancy gate.
    uint aFeatSM, aWBar, aFeatMean, aUtilFeat, aUtilOpt, aOptFracEMA;
    // Per-observation second-moment EMA E[obs²] paired with obsMean for Welford-variance
    // standardization of the encoder input (Sutton-Barto §9.5.4); updated BEFORE encode
    // each step by oak_bk_obs so the very first step sees correctly-scaled inputs.
    uint aObsSM;
    // per-feature TIDBD α (SwiftSarsa 2507.19539 + MetaOptimize 2402.02342).
    uint aAlphaImpFeat, aAlphaImpBeta, aAlphaImpH;
    // per-cumulant ψ-centering baseline = <ψ_k, featMean> over R encoder
    // dims. K floats updated in oak_bk_admit_commit; consumed by oak_predict.
    uint aPsiMeanBaseline;
    // TD(λ) eligibility traces (Sutton-Barto Ch 12.7). One float per linear-
    // FA weight. valueE[E] for w_main; psiE[K · M] for ψ. Engine-init zero; in-place updated
    // by oak_learn_value / oak_learn_psi at every step (B=1 path) via e ← λ · e + ∇V(s_t).
    uint aValueE, aPsiE;
    // Sutton-Barto §13.5 actor-critic-with-traces: backward-view eligibility trace
    // piE[K · D · E] decayed by lambdaTrace each step and accumulated with the per-step
    // ∂log π/∂θ. Integrates the policy gradient over the effective horizon 1/(1−λ),
    // restoring policy improvement when V is well-learned and per-step δ → 0.
    uint aPiE;
    // Derived scalars (all structurally derived from rank · batch).
    float cap;                // rank · batch (capacity volume; ridge / α₀ derived from it)
    float alpha0;             // 1 / cap (base step size)
    float ridge;              // 1 / (cap + 1) (Tikhonov ridge; tracking EMA floor)
};

struct KernelScalars {
    uint  stepCount, activeCount;
    uint  stateSlot, optionSlot;
    float rho;                // main-task average reward (Naik centering)
    float pEff;               // effective rank trace/max of featSM
    float alphaImp;           // implicit TD step: α₀ / (1 + α₀·⟨‖x_active‖²⟩_batch) — 2510.06149
    float alphaEnc;           // implicit TD step for encoder (centered obs)
    // REINFORCE policy implicit step. The per-row gradient is
    // δ · prev[i] · scoreGrad[d] (an outer product), so the BWH-orthodox implicit
    // denominator must include BOTH ‖prev‖² AND ‖scoreGrad‖² (Cauchy-Schwarz upper
    // bound on the per-parameter gradient norm; Kim 2510.06149 + Stream-X 2410.14606
    // ObGD §A4 generalized to outer-product features). per-thread n2 was
    // missing the cross-row scoreGrad norm-square aggregation, allowing 100×
    // scoreGrad blow-ups (mean 394 → unbounded) to drive 200M optionModelPredictionMSE
    // when scoreGrad was momentarily large but the implicit step did not compensate.
    //   alphaPi = α₀ / (1 + α₀ · ⟨‖x_active‖²⟩_batch · ⟨‖scoreGrad‖²⟩_batch)
    // Computed once in oak_reduce_alpha, consumed by oak_learn_pi.
    float alphaPi;
    float tdBaseline;         // REINFORCE running-mean TD baseline (Williams 1992)
    float modelErrEMA;        // Continual EMA of mean ψ TD error squared
                              // (= optionModelPredictionMSE). The agent's *current* model error.
    float modelErrEMAPrev;    // snapshot of modelErrEMA from N steps ago (slow EMA).
                              // lpmRate = max(0, modelErrEMAPrev - modelErrEMA) is the
                              // INFORMATION-GAIN proxy (Hou-An-Du 2509.25438 ICLR 2026 LPM).
                              // Positive when the model is improving; zero when stagnant.
                              // Used to RESCALE planning gate and β-termination so a wrong-but-
                              // improving model still gets credit (fixes absolute-scale
                              // regression on Dynamic-Obstacles family).
    uint  activeListIsIdentity; // 1 iff activeCount == E AND activeList[i] == i
                                // for all i in [0, E). Set by oak_bk_admit_commit; consumed by
                                // active_dot_half_auto to take the dense fast-path.
    float vMeanBaseline;        // (FULL Sakha §1 form): <valueW, featMean> — the
                                // expected V(s) under the empirical state distribution. Subtracted
                                // from the main-TD residual along with rhoBaseline. The plan §2.1
                                // names only rhoBaseline; the underlying paper §1 + research report
                                // §11 require this term too — without it `cond(Ā)` deteriorates
                                // unboundedly (200M MSE blowup root cause). Updated in
                                // oak_bk_admit_commit using the most-recent valueW.
    float lambdaTrace;          // TD(λ) eligibility-trace decay rate.
                                // Structural = 1 − 1/cap (Inv-9: derived from R·B; not tuned).
                                // Recovers variance reduction at B=1 (the BWH-orthodox setting
                                // per T3) by accumulating gradient depth INSIDE the TD target
                                // via Sutton-Barto Ch 12.7 backward-view eligibility traces.
                                // λ → 1−1/cap as cap → ∞ approaches the every-visit Monte-Carlo
                                // limit; λ → 0 recovers 1-step TD. The structural form gives
                                // λ ∈ [0.94, 0.999] for typical OaK runs (cap ≥ 17), well in
                                // the standard TD(λ) regime.
};

// KernelStats — telemetry overhaul. Layout must match Swift byte-for-byte.
// Three groups:
//   1. -14 baseline (legacy semantics — activeRank renamed to activeCount).
//   2. T0 numerical-safety (tdPsi NF + alphaPi/Imp/Enc).
//   3. T1 surfacing of KernelScalars + per-buffer weight-magnitude maxima
//      (canonical signal for the (C) bounded-magnitude gate; see oak_bk_weight_magnitudes).
struct KernelStats {
    float rewardRate, latentStateMeanSquare, activeCount;
    float rawNonFiniteCount, sanitizedNonFiniteCount;
    float encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsPolicyOutput;
    float actionDeltaL2Mean, scoreGradL2Mean;
    float optionModelPredictionMSE;
    float oakOptionReinitCount;
    float utilityFeatureMean, utilityOptionMean;
    float tdPsiNonFiniteCount;     // per-step NF in tdPsi (was untracked, contaminated MSE)
    float pEffectiveRank;          // real effective rank from KernelScalars.pEff
                                   //       (separate from activeCount; the trace/max stable-rank proxy)
    float modelErrEMA;             // fast EMA of mean ψ TD error² (consumer of LPM gate)
    float modelErrEMAPrev;         // slow lagged EMA (lpmRate = max(0, prev - cur))
    float tdBaselineStat;          // REINFORCE running-mean baseline (Williams 1992)
    float alphaImpStat;            // implicit α for value/ψ heads (Kim 2510.06149)
    float alphaEncStat;            // implicit α for encoder (centered-obs variant)
    float alphaPiStat;             // implicit α for policy (T0.3 ‖scoreGrad‖² form)
    float oakFeatureReinitCount;   // cumulative encoder feature reinits (was untracked)
    // weight-magnitude maxima (one per param buffer; canonical gate-(C) signal):
    float encoderMax, valueWMax, psiMax, optPolicyMax;
    float outcomeHeadMax, betaWMax, omegaWMax, sprDecoderMax;
};

// ------------------------------------------------------------------------------------------------
// Context: pointers + scalars, no per-thread copy of the full config.
// E = R + K is the OPSR-extended state width (encoder output + outcome head).
// M = E + K is the concat dim for ψ's chord-conditional input ([state | one-hot chord]).
// ------------------------------------------------------------------------------------------------
struct Ctx {
    uint R, A, B, K, O, baseO, M, D, E;
    uint activeCount, stepCount;
    uint stateSlot, optionSlot;
    float cap, alpha0, ridge, rho;

    device float  *encoder;
    device float  *valueW;         // [E] : value head over the OPSR-extended state
    device float  *psi;            // [K][M] : W_ψ[k] = (state weights, chord weights), M = E + K
    device float  *optPolicy;      // [K][D][E] : per-option policy parameters (D = D_policy)
    device float  *outcomeHead;    // [K][E] : OPSR outcome predictor (Tier 2)
    device float  *betaW;          // [E + K] : termination head (state weights + chord weights)
    device float  *omegaW;         // [K][E] : chord-generator (softmax over K from E-dim state)
    device float  *sprDecoder;     // [K][R] : chord-conditioned next-state predictor

    device float  *observation;    // [B][O]
    device float  *observationMean;// [O]
    device float  *obsSM;          // [O] — E[obs²] Welford EMA for input standardization
    device float  *state;          // [2][B][R]
    device float  *prevPolicyMean; // [B][D] : prev step's policyMean (host-written for telemetry)
    device float  *policyMean;     // [B][D] : current step's policy mean (kernel-written)
    device float  *scoreGrad;      // [B][D] : ∇log π(a|s) for wrapper-sampled action (host-written)
    device float  *reward;         // [B]
    device float  *td;             // [B]
    device float  *tdPsi;          // [B][K]  -- one cumulant TD per row per option
    device float  *vCurr;          // [B]     -- w_main · curr_state, cached by predict, read by plan
    device float  *chordSlot;      // [B][K]  -- current step's persistent soft chord (Tier 3)
    device float  *chordSlotPrev;  // [B][K]  -- prev step's chord (for learn_psi/_pi attribution)
    device float  *betaProb;       // [B]     -- β termination probability (telemetry + learn_beta)
    // sprPred[B][R] device pointer removed (oak_spr_fused inlines).

    device uint   *done;           // [B]
    device uint   *activeMask;     // [E]
    device uint   *activeList;     // [E]
    device uint   *option;         // [2][B]
    device uint   *counters;       // [6]
    device uint   *sf;             // [K] subtaskFeatureOf — dynamic top-K feature ranking
    device uint   *optionAge;      // [B] steps since last termination (Tier 3)

    device float  *featSM;         // [R]
    device float  *wBar;           // [R]
    device float  *featMean;       // [R]  -- per-feature running mean (cumulant centering ρ_k)
    device float  *utilFeat;       // [R]
    device float  *utilOpt;        // [K]
    device float  *optFracEMA;     // [K]  -- soft-chord per-option mass EMA (Tier 5 reinit gate)
    // per-feature TIDBD α — meta-learned per-feature implicit step-sizes.
    device float  *alphaImpFeat;   // [E]  -- α_i = exp(β_i) / (1 + exp(β_i) · featSM[i])
    device float  *alphaImpBeta;   // [E]  -- β_i meta-parameter (log-step), init log(α₀)
    device float  *alphaImpH;      // [E]  -- decay trace of past Δw per feature, init 0
    // per-cumulant ψ-centering baseline = <ψ_k, featMean> over R encoder dims.
    device float  *psiMeanBaseline; // [K] -- subtracted from each cumulant's TD residual
    // TD(λ) backward-view eligibility traces (Sutton-Barto Ch 12.7).
    device float  *valueE;          // [E]     value-head trace (e_t = λ·e_{t-1} + ∇V(s_t))
    device float  *psiE;            // [K * M] ψ traces (one per cumulant × concat dim)
    device float  *piE;             // [K·D·E] policy traces (Sutton-Barto §13.5)

    device KernelScalars *scalars;
};

inline Ctx make_ctx(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars, constant KernelConfig &cfg) {
    Ctx c;
    c.R = cfg.rank; c.A = cfg.actDim; c.B = cfg.batch; c.K = max(cfg.optionCount, 1u);
    c.O = cfg.obsDim; c.baseO = cfg.baseObsDim;
    c.E = c.R + c.K;                       // OPSR extended-state width
    c.M = c.E + c.K;                       // ψ concat dim: [extended state | one-hot chord]
    c.D = max(cfg.actionParamCount, 1u);   // env-declared D_policy
    c.activeCount = scalars->activeCount;
    c.stepCount   = scalars->stepCount;
    c.stateSlot   = scalars->stateSlot;
    c.optionSlot  = scalars->optionSlot;
    c.cap = cfg.cap; c.alpha0 = cfg.alpha0; c.ridge = cfg.ridge;
    c.rho = scalars->rho;

    c.encoder     = params + cfg.pEncoder;
    c.valueW      = params + cfg.pValue;
    c.psi         = params + cfg.pPsi;
    c.optPolicy   = params + cfg.pOptPolicy;
    c.outcomeHead = params + cfg.pOutcomeHead;
    c.betaW       = params + cfg.pBetaW;
    c.omegaW      = params + cfg.pOmegaW;
    c.sprDecoder  = params + cfg.pSprDecoder;

    c.observation     = stateArena + cfg.sObservation;
    c.observationMean = stateArena + cfg.sObsMean;
    c.obsSM           = stateArena + cfg.aObsSM;
    c.state           = stateArena + cfg.sState;
    c.prevPolicyMean  = stateArena + cfg.sPrevPolicyMean;
    c.policyMean      = stateArena + cfg.sPolicyMean;
    c.scoreGrad       = stateArena + cfg.sScoreGrad;
    c.reward          = stateArena + cfg.sReward;
    c.td              = stateArena + cfg.sTd;
    c.tdPsi           = stateArena + cfg.sTdPsi;
    c.vCurr           = stateArena + cfg.sVCurr;
    c.chordSlot       = stateArena + cfg.sChordSlot;
    c.chordSlotPrev   = stateArena + cfg.sChordSlotPrev;
    c.betaProb        = stateArena + cfg.sBetaProb;
    // c.sprPred removed.

    c.done       = uintArena + cfg.uDone;
    c.activeMask = uintArena + cfg.uActiveMask;
    c.activeList = uintArena + cfg.uActiveList;
    c.option     = uintArena + cfg.uOption;
    c.counters   = uintArena + cfg.uCounters;
    c.sf         = uintArena + cfg.uSubtaskFeatureOf;
    c.optionAge  = uintArena + cfg.uOptionAge;

    c.featSM     = stateArena + cfg.aFeatSM;
    c.wBar       = stateArena + cfg.aWBar;
    c.featMean   = stateArena + cfg.aFeatMean;
    c.utilFeat   = stateArena + cfg.aUtilFeat;
    c.utilOpt    = stateArena + cfg.aUtilOpt;
    c.optFracEMA = stateArena + cfg.aOptFracEMA;
    c.alphaImpFeat = stateArena + cfg.aAlphaImpFeat;
    c.alphaImpBeta = stateArena + cfg.aAlphaImpBeta;
    c.alphaImpH    = stateArena + cfg.aAlphaImpH;
    c.psiMeanBaseline = stateArena + cfg.aPsiMeanBaseline;
    c.valueE       = stateArena + cfg.aValueE;
    c.psiE         = stateArena + cfg.aPsiE;
    c.piE          = stateArena + cfg.aPiE;

    c.scalars = scalars;
    return c;
}

inline device float *state_slot(Ctx ctx, uint slot) { return ctx.state + slot * ctx.B * ctx.E; }
inline device uint  *option_slot(Ctx ctx, uint slot) { return ctx.option + slot * ctx.B; }
inline uint active_option(Ctx ctx, uint row) {
    device const uint *o = option_slot(ctx, ctx.optionSlot);
    return min(o[row], ctx.K - 1u);
}

// Deterministic counter-based RNG: splitmix32 of (step, row, dim) → uniform float in (0, 1].
// FLT_MIN floor avoids log(0) in Box-Muller. Used by oak_meta_policy and any kernel that needs
// per-thread reproducible randomness without host RNG state (Inv 9).
inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}
inline float hash_uniform(uint step, uint row, uint dim) {
    uint h = splitmix32(step ^ splitmix32(row ^ splitmix32(dim)));
    return max(float(h >> 8) * 5.9604645e-8f, FLT_MIN);
}

// Continual tracking rate. Warmup phase (~1/ridge steps) follows sample mean via 1/(t+1);
// afterwards floors at structural ridge = 1/(R·B+1). Guarantees tracking does NOT freeze
// for t → ∞ (Big World Hypothesis; Autostep-style meta-rate; 2604.01913 Θ(1/k) remedy).
// Inv-9 compliant: both rates are structurally derived from {R, B}, no scalar hyperparameter.
inline float track_rate(uint stepCount, float ridge) {
    return max(1.0f / float(stepCount + 1u), ridge);
}

// Faster timescale than track_rate, used by featMean / ψMeanBaseline / vMeanBaseline.
// Sakha-Kamalapurkar-Meyn arXiv:2603.27874v2 §1.3 / Eq. 18b requires β_n / α_n → ∞ for
// the ϖ-RTD Hurwitz proof to apply (the empirical mean must converge faster than the
// value weights so the baseline is approximately stationary from the value head's POV).
// At α_n = 1/n, β_n = 1/√n satisfies β_n/α_n = √n → ∞. Implemented as
// 1 / sqrt(stepCount+1) with the same ridge floor for tracking on infinite horizons.
// Inv-9 compliant: structural in {stepCount, ridge}, no tuned scalar.
inline float track_rate_fast(uint stepCount, float ridge) {
    return max(1.0f / sqrt(float(stepCount + 1u)), ridge);
}

inline uint ceil_sqrt(uint n) { float s = sqrt(float(max(n, 1u))); return uint(ceil(s)); }

// Inv-4 zero-bias passthrough: returns x if finite, else 0. Used at every device-array
// READ that flows into a scalar reduction (mean, sum, EMA), preventing one stale NaN cell
// from poisoning a long-lived running statistic. This is the ONLY non-finite guard the
// kernel uses — magnitudes that grow as the agent's encountered slice of reality demands
// are intentional (BWH; Sutton-Barto Ch 9). The implicit-α step (Kim 2510.06149) bounds
// per-step UPDATES; reward centering (Sakha 2603.27874v2) bounds TD signal in expectation;
// L2-ER (He 2509.22335) bounds long-run weight drift on the representation heads only.
// Together these are the standard linear-FA stability conditions (Tsitsiklis-Van Roy 1997).
inline float nf_safe(float x) { return isfinite(x) ? x : 0.0f; }

// Standardized observation (Sutton-Barto §9.5.4): (obs − E[obs]) / √Var(obs). Tracks the
// world's per-feature scale via the Welford EMAs obsMean and obsSM (= E[obs²]); variance
// = obsSM − obsMean² floored at ridge to avoid div-by-zero on constant-feature dims.
// Lets the encoder operate in tanh's linear region regardless of env-specific obs scale
// (joint angles ~±π vs velocities ~±50 in Mujoco). BWH-orthodox: the agent tracks the
// world's actual statistics; the standardization is structural in those statistics.
// Updated BEFORE encode each step (oak_bk_obs runs first in the pipeline) so the very
// first encode call sees correctly-scaled inputs — eliminates the Welford-bootstrap
// transient where raw obs at step 0 saturates the encoder.
inline float standardized_obs(float obs, float mean, float sm, float ridge) {
    float var = max(sm - mean * mean, ridge);
    return (obs - mean) / sqrt(var);
}


// Weights are stored as fp32 (BWH: magnitudes carry information, no arbitrary HALF_MAX
// = 65504 scale prior). Per-write NaN guard via nf_safe at the call sites — the
// implicit-α step + ϖ-RTD centering + L2-ER (anti-collapse only) provide the
// structural learning stability without imposing a stationary weight scale.

// L2-Effective-Rank regularizer (He et al. 2509.22335 ICLR 2026 "Spectral Collapse").
// L2-Effective-Rank decay (He et al. 2509.22335) with structural floor enforcing
// Sakha-Kamalapurkar-Meyn assumption (A1•) Σ(0) > ε_0·I. Theorem 2.4(ii)'s Hurwitz proof
// requires at least one direction to be strictly damped (ε_0 > 0); without the floor,
// when matrix-encoder features push pEff arbitrarily close to R, the gap → 0 and the
// damping vanishes — exposing the bootstrap V'-V positive-feedback divergence.
// Floor: pEff ≤ R − 1 (always treat as 1-deficient), guaranteeing a minimum effective
// damping rate of α₀ / R along all directions. Inv-9: floor is structural in R, no magic.
inline float l2er_decay(float alpha0, uint R, float pEff) {
    float pEff_floor = min(pEff, float(R) - 1.0f);
    float gap = max(0.0f, (float(R) - pEff_floor) / float(max(R, 1u)));
    return alpha0 * gap;
}

// Dot product over the active-feature list (gather).
inline float active_dot_half(device const float *w, uint wBase,
                             device const float *x, uint xBase,
                             device const uint *active, uint count) {
    float s = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        uint idx = active[i];
        s = fma(w[wBase + idx], x[xBase + idx], s);
    }
    return s;
}

// dense fast-path for `active_dot_half` when activeCount == E and the
// activeList is the identity ordering. Skips the indirection through activeList[i] — the
// fp16-fp32 fused dot becomes a tight contiguous loop the compiler can vectorize tightly.
// Caller checks `activeListIsIdentity` flag (cached in KernelScalars by oak_bk_admit_commit).
inline float active_dot_dense(device const float *w, uint wBase,
                              device const float *x, uint xBase, uint count) {
    float s = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        s = fma(w[wBase + i], x[xBase + i], s);
    }
    return s;
}

// dispatch helper — uses the dense path when activeListIsIdentity is set,
// otherwise falls back to gather. Keeps the call sites readable. The flag is computed once
// per step in oak_bk_admit_commit (negligible cost) so all callers within the step agree.
inline float active_dot_half_auto(device const float *w, uint wBase,
                                  device const float *x, uint xBase,
                                  device const uint *active, uint count,
                                  uint dense) {
    return dense ? active_dot_dense(w, wBase, x, xBase, count)
                 : active_dot_half(w, wBase, x, xBase, active, count);
}

// TG-shared variant. Same math but reads from threadgroup-local cached
// state. Used by oak_predict where ONE row's prev/curr state is loaded ONCE into TG memory
// and shared across the value-head dot + K cumulant-head dots (was: K+1 redundant scans of
// the same E-byte device-memory state row per thread).
inline float active_dot_half_tg(device const float *w, uint wBase,
                                threadgroup const float *x,
                                device const uint *active, uint count,
                                uint dense) {
    float s = 0.0f;
    if (dense) {
        for (uint i = 0u; i < count; ++i) {
            s = fma(w[wBase + i], x[i], s);
        }
    } else {
        for (uint i = 0u; i < count; ++i) {
            uint idx = active[i];
            s = fma(w[wBase + idx], x[idx], s);
        }
    }
    return s;
}

inline float active_norm2(device const float *x, uint xBase,
                          device const uint *active, uint count) {
    float s = 0.0f;
    for (uint i = 0u; i < count; ++i) {
        uint idx = active[i];
        float v = x[xBase + idx];
        s = fma(v, v, s);
    }
    return s;
}

// Hierarchical simd_sum for 128-thread threadgroups (4 SIMD groups × 32). Each lane
// contributes v_local; first lane of each SIMD stages to sg_buf; first SIMD group
// finalizes with one more simd_sum. Returns the global sum to lane 0 only.
inline float tg_sum_128(float v_local, threadgroup float *sg_buf, uint tid, uint sgid) {
    float v_sg = simd_sum(v_local);
    if (simd_is_first()) sg_buf[sgid] = v_sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (tid < 4u) ? sg_buf[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0u) sg_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sg_buf[0];
}

inline float tg_max_128(float v_local, threadgroup float *sg_buf, uint tid, uint sgid) {
    float v_sg = simd_max(v_local);
    if (simd_is_first()) sg_buf[sgid] = v_sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0u) {
        float v = (tid < 4u) ? sg_buf[tid] : -FLT_MAX;
        v = simd_max(v);
        if (tid == 0u) sg_buf[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sg_buf[0];
}

// =================================================================================================
// noop — ablation stub. Hot-swap target for the per-kernel ablation harness:
// OakMetalEngine.init accepts an `ablatedKernels: Set<String>`; for each phase whose name
// is in that set, the ICB encodes oak_noop's pipeline state instead of the real kernel.
// The arena slots that the ablated kernel would have written stay at their previous step's
// values (stale data) — this is the cleanest way to simulate "what if this kernel didn't run"
// without disturbing the layout. Same buffer signature as every other kernel so the ICB
// can swap pipelines without touching buffer bindings.
// =================================================================================================
kernel void oak_noop(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    (void)params; (void)stateArena; (void)uintArena;
    (void)scalars; (void)stats; (void)cfg; (void)gid;
}

// =================================================================================================
// encode — full R×O matrix encoder (Sutton-Barto Ch 9 linear FA): each latent feature i is a
// learned linear combination of all standardized obs dimensions. Decouples representation
// rank R from observation dim O so the agent can express cross-dim mixtures (joint angles ×
// velocities, etc.) that diagonal scaling cannot represent. Per-row, per-feat:
//   z_instant[i] = Σ_o encoder[i, o] · standardized_obs[o]
//   z_t[i]      = α_i · z_instant[i] + (1 − α_i) · z_{t-1}[i]   (Belief-State RWKV 2604.09671)
//   z_t[i]      = tanh(z_t[i])                                   (Tsitsiklis-Van Roy bounded features)
// α_i = min(1, alphaImpFeat[i]·cap) ties retention to TIDBD's per-feature step (no new params).
//   [Grid: R · B] One thread per (row, feat).
// =================================================================================================
kernel void oak_encode(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.R, feat = gid % ctx.R;
    if (row >= ctx.B) return;
    uint writeSlot = 1u - ctx.stateSlot;
    uint encRow = feat * ctx.baseO;
    float z_instant = 0.0f;
    for (uint o = 0u; o < ctx.baseO; ++o) {
        float c = standardized_obs(ctx.observation[row * ctx.O + o],
                                   ctx.observationMean[o], ctx.obsSM[o], ctx.ridge);
        z_instant = fma(ctx.encoder[encRow + o], c, z_instant);
    }
    float alpha = min(1.0f, ctx.alphaImpFeat[feat] * ctx.cap);
    float prev_belief = nf_safe(state_slot(ctx, ctx.stateSlot)[row * ctx.E + feat]);
    float z_raw = alpha * z_instant + (1.0f - alpha) * prev_belief;
    state_slot(ctx, writeSlot)[row * ctx.E + feat] = isfinite(z_raw) ? tanh(z_raw) : 0.0f;
}

// =================================================================================================
// predict_outcome — OPSR augmentation (Vereecken 2604.07016). Writes K outcome predictions per
// row recursively from the previous step's full E-dim state:
//   outcome[r, k] = H[k, :] · state_prev[r, :E]
// The OPSR head provides agent-side memory: the outcome dims at state[1-slot][r][R:E] mix the
// previous-step's encoder output AND previous outcome predictions, giving the linear heads
// (value, ψ, π) access to a 1-step-back compressed history.
// Recursive: depends on state_slot(ctx.stateSlot) which holds step N-1's writes (encoded
// observations + outcome predictions), so this kernel must run BEFORE any kernel that reads
// state[1-slot] (i.e., before predict / plan / meta_action / learn_*). Inserted immediately
// after oak_encode in the phase order.
//   [Grid: B · K — one thread per (row, outcome dim).]
// =================================================================================================
kernel void oak_predict_outcome(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.K, k = gid % ctx.K;
    if (row >= ctx.B) return;
    device const float *prev = state_slot(ctx, ctx.stateSlot) + row * ctx.E;
    device       float *curr = state_slot(ctx, 1u - ctx.stateSlot) + row * ctx.E;
    // Active-feature dot: outcome head sees only encoder-admitted features + outcome dims (which
    // are always-active by construction). active_dot_half iterates activeList[0..activeCount).
    float v = active_dot_half_auto(ctx.outcomeHead, k * ctx.E, prev, 0u,
                                   ctx.activeList, ctx.activeCount,
                                   ctx.scalars->activeListIsIdentity);
    // Outcome predictions feed state[R..E] which is consumed by ALL downstream linear
    // heads. The Tsitsiklis-Van Roy 1997 linear-FA stability proof requires BOUNDED
    // features. Encoder z[0..R) is tanh-bounded (oak_encode); outcome predictions are
    // predictions OF those features, so they must be on the same scale — tanh applied.
    // This is NOT a stationary scale prior on weights (BWH preserved); it's matching
    // the prediction target's natural range.
    curr[ctx.R + k] = isfinite(v) ? tanh(v) : 0.0f;
}

// =================================================================================================
// meta_policy — temporal abstraction (β termination + ω chord generator). Per
// Bacon-Precup 1609.05140 (Option-Critic) under Wan-Naik-Sutton 2110.13855 continuing
// average-reward formulation.
// Per row r (one thread per row, B threads total):
//   1. Save the current chord into chordSlotPrev[r] (= the chord that produced action_{N-1}).
//   2. Compute β(s, λ_prev) = σ(W_β · concat[s_E, λ_prev]). Bernoulli-sample termination
//      via splitmix32 deterministic hash (no host RNG).
//   3. If terminate (or optionAge == 0 at start): resample chord from softmax(ω(s_E)) over K
//      logits. Reset optionAge to 1.
//   4. Else: keep chord. Increment optionAge.
// `option_slot` is also written here (= argmax_k chord[r][k]) for telemetry compatibility
// with the bk_option / learn_psi / learn_pi attribution logic that still uses the discrete
// dominant-option index. Soft-chord weighting on top of the dominant-option attribution is
// the principled formulation; we run them BOTH for now.
//   [Grid: B — one thread per row.]
// =================================================================================================
kernel void oak_meta_policy(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid;
    if (row >= ctx.B) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot) + row * ctx.E;
    device       float *chord     = ctx.chordSlot     + row * ctx.K;
    device       float *chordPrev = ctx.chordSlotPrev + row * ctx.K;

    // (1) Save current chord into chordPrev. The current chord at meta_policy entry == the
    // chord that was active for step N-1's meta_action (no kernel between them mutates it).
    for (uint k = 0u; k < ctx.K; ++k) chordPrev[k] = chord[k];

    // (2) β(s_E, λ_prev) = σ(W_β · [s_E ; λ_prev]).
    float beta_logit = active_dot_half(ctx.betaW, 0u, curr, 0u,
                                       ctx.activeList, ctx.activeCount);
    for (uint k = 0u; k < ctx.K; ++k)
        beta_logit += ctx.betaW[ctx.E + k] * chordPrev[k];
    // β-on-surprise-MINUS-IG (CPD-Option-Critic 2510.24988 + LPM Hou-An-Du 2509.25438).
    // Normalized form: contribute (modelErrEMA − lpmRate) / max(modelErrEMA, ridge) which
    // is in [0, 1] (= 1 when surprise is high AND not improving, → 0 when fully improving).
    // The earlier raw form saturated β→1 immediately when modelErrEMA was O(K) (e.g. 3347
    // on HalfCheetah at B=1), forcing termination every step and preventing options from
    // accumulating. Normalizing by modelErrEMA itself keeps the contribution at its
    // structural [0, 1] sigmoid-sensitive range without introducing a tuned scalar.
    float lpmRate_meta = max(0.0f,
                             ctx.scalars->modelErrEMAPrev - ctx.scalars->modelErrEMA);
    float surprise_norm = (ctx.scalars->modelErrEMA - lpmRate_meta)
                        / max(ctx.scalars->modelErrEMA, ctx.ridge);
    beta_logit += surprise_norm;
    float beta_prob = 1.0f / (1.0f + exp(-beta_logit));
    ctx.betaProb[row] = beta_prob;

    // (3) Per-row Bernoulli termination via splitmix32(stepCount, row).
    float u = hash_uniform(ctx.stepCount, row, 0u);
    bool terminate = (ctx.optionAge[row] == 0u) || (u < beta_prob);

    if (terminate) {
        // Resample chord ~ softmax(ω(s_E)). K ≤ OAK_SIMD = 32 (host-capped), so on-stack.
        float maxLogit = -FLT_MAX;
        float logits[OAK_SIMD];
        for (uint k = 0u; k < ctx.K; ++k) {
            float l = active_dot_half(ctx.omegaW, k * ctx.E, curr, 0u,
                                      ctx.activeList, ctx.activeCount);
            logits[k] = l;
            maxLogit = max(maxLogit, l);
        }
        float Z = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) Z += exp(logits[k] - maxLogit);
        float invZ = 1.0f / max(Z, FLT_MIN);
        uint  bestK = 0u; float bestP = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float p = exp(logits[k] - maxLogit) * invZ;
            chord[k] = p;
            if (p > bestP) { bestP = p; bestK = k; }
        }
        ctx.optionAge[row] = 1u;
        // Telemetry-compatible argmax-chord written into option_slot's NEW slot.
        option_slot(ctx, 1u - ctx.optionSlot)[row] = bestK;
    } else {
        ctx.optionAge[row] += 1u;
        // No chord change; preserve the option_slot dominant-index from prev step's plan.
        uint prevBest = option_slot(ctx, ctx.optionSlot)[row];
        option_slot(ctx, 1u - ctx.optionSlot)[row] = prevBest;
    }
}

// =================================================================================================
// ( oak_predict_next_latent fused into oak_spr_fused at Tier 1 — see below.)
// The standalone predictor kernel was removed because the per-(row, e) predictions are now
// recomputed on-the-fly inside the fused per-feat encoder/decoder update. Saves: 1 ICB
// dispatch + 1 barrier + 1 device-memory write/read of sprPred[B·R] (~3 MB at B=2048, R=376).
// Pipeline 20 → 19 kernels.
// =================================================================================================

// =================================================================================================
// predict — main differential TD δ_main (A.2) + K per-cumulant SF TDs δ_ψ[k] under the row's
// active option e_{k*} (Alberta Plan Design 2: shared ψ_k(s, λ) = W_ψ[k] · concat[s, λ]).
// ψ_k is a SCALAR SF for cumulant k under chord λ — not an R-vector per option. Subtask feature
// sf[] is dynamic top-K via bk_admit_commit re-rank; seeded as identity
// sf[k] = k at engine init. Cumulant
// centering ρ_k = featMean[sf[k]] tracks the per-feature running mean (2405.09999 differential
// TD; 2603.27874 stable under FA for any baseline).
//   [Grid: B]
// =================================================================================================
// per-thread dispatchThreads, one thread
// per row computes both the value-head TD AND all K cumulant SF TDs serially.
// briefly tried B threadgroups × 32 threads with TG-shared row state, but the resulting
// SIMD lane waste (only K=17 of 32 lanes do tdPsi work; only 1 does value work) cost more
// agent_ms than the device-memory bandwidth saved by caching. Apple's L1 cache makes the
// K-fold reads of the same row cheap in steady state, and the per-thread dispatch fills
// 32-wide SIMD groups without idle lanes. Empirical: B-thread dispatch wins by ~2× vs
// TG-cooperative on MiniGrid.
//   [Grid: B — one thread per row.]
kernel void oak_predict(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid >= ctx.B) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint pOff = gid * ctx.E, cOff = gid * ctx.E;
    uint M = ctx.M;

    // STRICT ϖ-Relative TD (Sakha-Kamalapurkar-Meyn arXiv:2603.27874v2 Eq. 24, Theorem 2.4(ii)).
    // The TD residual here is the STANDARD differential TD with NO centering inside the scalar
    // — the centering is applied by oak_learn_value as a separate rank-1 vector update
    //   w[i] -= α · δ_r · vMeanBaseline · featMean[i]
    // along the empirical-mean direction ψ̄. The two forms have the same mean flow Ā(λ;δ_r),
    // but the strict form's noise is rank-1 along ψ̄ (always damped by the perturbation),
    // whereas the partial form's noise is full-rank O(‖w‖²) and dominates the Hurwitz
    // damping when feature covariance Σ(0) is full-rank (matrix encoder regime). With this
    // form, Theorem 2.4(ii) guarantees the mean flow is uniformly Hurwitz for any δ_r ≥ 0.
    uint dense = ctx.scalars->activeListIsIdentity;
    float vPrev = active_dot_half_auto(ctx.valueW, 0u, prev, pOff,
                                       ctx.activeList, ctx.activeCount, dense);
    float vNext = active_dot_half_auto(ctx.valueW, 0u, curr, cOff,
                                       ctx.activeList, ctx.activeCount, dense);
    ctx.vCurr[gid] = vNext;
    float terminalMask = (ctx.done[gid] != 0u) ? 0.0f : 1.0f;
    float rho       = ctx.scalars->rho;
    ctx.td[gid] = (ctx.reward[gid] - rho) + terminalMask * vNext - vPrev;

    // Per-cumulant SF TD: same strict ϖ-RTD form (K parallel linear-FA TD instances).
    // The cumulant centering psiMeanBaseline[k] is applied as a separate rank-1 vector
    // update by oak_learn_psi along the meanBasis = concat[featMean, chordMean] direction.
    device const float *chord_prev = ctx.chordSlotPrev + gid * ctx.K;
    float reward_centered = ctx.reward[gid] - rho;
    for (uint k = 0u; k < ctx.K; ++k) {
        uint psiBase = k * M;
        float chord_part = 0.0f;
        for (uint k2 = 0u; k2 < ctx.K; ++k2) {
            chord_part += ctx.psi[psiBase + ctx.E + k2] * chord_prev[k2];
        }
        float psi_prev = active_dot_half_auto(ctx.psi, psiBase, prev, pOff,
                                              ctx.activeList, ctx.activeCount, dense)
                       + chord_part;
        float psi_next = active_dot_half_auto(ctx.psi, psiBase, curr, cOff,
                                              ctx.activeList, ctx.activeCount, dense)
                       + chord_part;
        uint sf_k = ctx.sf[k];
        float feat_centered = (sf_k < ctx.R) ? (curr[cOff + sf_k] - ctx.featMean[sf_k]) : 0.0f;
        float c_RED = reward_centered + feat_centered;
        ctx.tdPsi[gid * ctx.K + k] = c_RED + terminalMask * psi_next - psi_prev;
    }
}

// =================================================================================================
// plan — one-step option-model lookahead (F4) under Design 2 shared chord-conditional ψ.
//   Q_k(s) = w_main[sf[k]] · ψ_k(s, e_k) + max(0, (w̄[sf[k]] − w_main[sf[k]]) · s[sf[k]])
// where ψ_k(s, e_k) = Σ_i W_ψ[k][i] · s[i] + W_ψ[k][R + k] (one-hot chord λ = e_k).
// SIMD tournament argmax across K lanes (one TG = one SIMD group; K ≤ OAK_SIMD = 32 by host
// OakMetalLayout.optionCount). RRS optimism (§6.1) on the cumulant's own feature is the
// planning-time bonus that drives feature attainment. Citations: OBB 2505.00787 §2.4 Eq.4
// (chord-conditional Q); OK 2106.13105 Alg 1 line 3 (GPI argmax); RRS 2202.03466 §6.1.
//   [Dispatch: B threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_plan(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = tgid;
    if (row >= ctx.B) return;
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint xOff = row * ctx.E;
    uint M = ctx.M;

    // Plan still does a per-OPTION GPI argmax (one-hot chord) to compute the planning
    // advantage td += Q_max − vCurr — i.e., "what if we replaced the current chord with the
    // single best option". This gives the option-keyboard greedy improvement direction.
    float qk = -FLT_MAX;
    uint  k  = tid;
    if (k < ctx.K) {
        uint psiBase = k * M;
        // ψ_k(s, e_k) under one-hot chord at index k.
        float psi_k = active_dot_half_auto(ctx.psi, psiBase, curr, xOff,
                                           ctx.activeList, ctx.activeCount,
                                           ctx.scalars->activeListIsIdentity)
                    + ctx.psi[psiBase + ctx.E + k];
        uint sf_k = ctx.sf[k];  // dynamic feature ranking (Tier 7)
        float wk  = (sf_k < ctx.R) ? ctx.valueW[sf_k] : 0.0f;
        float val = wk * psi_k;
        if (sf_k < ctx.R) {
            float bonus = (ctx.wBar[sf_k] - wk) * curr[xOff + sf_k];
            val += max(bonus, 0.0f);
        }
        qk = val;
    }

    // SIMD tournament argmax in log2(OAK_SIMD) shuffle rounds. No threadgroup memory needed.
    uint k_lane = tid;
    for (uint off = OAK_SIMD >> 1; off > 0u; off >>= 1u) {
        float q2 = simd_shuffle_down(qk, off);
        uint  k2 = simd_shuffle_down(k_lane, off);
        bool  wins = (q2 > qk);
        qk     = wins ? q2 : qk;
        k_lane = wins ? k2 : k_lane;
    }
    if (tid == 0u) {
        // LPM normalized planning gate (Hou-An-Du 2509.25438): trust the planning bonus
        // proportional to how much the model is improving relative to its current error.
        // gateScale = (ridge + lpmRate) / (ridge + modelErrEMA) — bounded in [0, 1] under
        // the natural ridge floor; → 1 when lpmRate ≈ modelErrEMA (model converging fast),
        // → 0 when lpmRate << modelErrEMA (stagnant high-error regime). The earlier
        // (1 + lpmRate)/(1 + modelErrEMA) form shrunk the bonus to ≈0 when modelErrEMA
        // was O(K) (HalfCheetah B=1: gateScale ≈ 1/3348 = 0.0003), making planning a no-op.
        float lpmRate = max(0.0f,
                            ctx.scalars->modelErrEMAPrev - ctx.scalars->modelErrEMA);
        float gateScale = (ctx.ridge + lpmRate)
                        / (ctx.ridge + ctx.scalars->modelErrEMA);
        // Add the planning advantage to td. NF guards keep stale NaN cells from poisoning
        // the running EMAs in oak_bk_admit_commit; magnitudes are intentionally unbounded
        // per BWH — the implicit-α step + ϖ-RTD centering keep the LEARNING dynamics stable
        // (Tsitsiklis-Van Roy 1997, Sakha 2603.27874v2 Theorem 2.4(ii)).
        ctx.td[row] = nf_safe(ctx.td[row]) + gateScale * (qk - nf_safe(ctx.vCurr[row]));
    }
}

// =================================================================================================
// meta_action — + Tier 3: emit ONLY policyMean[B · D]; the env-wrapper layer
// (EnvPoolEnv.sampleAndStep) samples per-component (Gumbel-max categorical for Discrete-N
// components; structural-σ-1 Gaussian for Box-A components) and computes the matching score
// function (Williams 1992; Maddison-Tarlow-Minka 2014; Sutton-Barto Ch.13). The agent has
// zero a-priori knowledge of action-space type — D is the env-declared D_policy width
// (Inv-1; Bitter Lesson).
// Per (row, policy-param p ∈ [0, D)):
//   μ_p = Σ_k λ_k · (W_π[k][p] · s_curr)
// where λ = chordSlot[r] is the persistent soft chord from oak_meta_policy.
// K-outer-E-inner accumulator (pattern restored) WITHOUT
// the dynamically-indexed `w_tilde` stack array. Key insight: optPolicy[k * D * E + p * E + e]
// has stride 1 across e for fixed k (consecutive fp16 cache-friendly reads per option),
// whereas the Tier-6.1 e-outer-k-inner pattern reads optPolicy with stride D*E
// across k (cache-line miss per k-iteration). The original K-outer was empirically
// fastest because it maximizes optPolicy spatial locality. The w_tilde[128] stack
// array was the actual bug (private-memory spill), NOT the loop ordering. keeps
// the K-outer order and accumulates dot_k in a register, then mu via outer FMA.
// Per thread: K * E FMAs. curr is read K times per (row, p) but cached
// in L1 (curr is ≤ 1 KB per row, fits comfortably across K iterations). optPolicy reads
// E consecutive fp16 per k-iteration = 786 bytes per k = perfectly cache-friendly.
//   [Grid: B · D — one thread per (row, policy-param).]
// =================================================================================================
kernel void oak_meta_action(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint row = gid / ctx.D, p = gid % ctx.D;
    if (row >= ctx.B) return;

    device const float *curr  = state_slot(ctx, 1u - ctx.stateSlot);
    device const float *chord = ctx.chordSlot + row * ctx.K;
    uint xOff = row * ctx.E;
    uint dense = ctx.scalars->activeListIsIdentity;

    float mu = 0.0f;
    for (uint k = 0u; k < ctx.K; ++k) {
        float lambda_k = chord[k];
        if (lambda_k < FLT_MIN) continue;
        uint optBase = k * ctx.D * ctx.E + p * ctx.E;
        float dot_k = active_dot_half_auto(ctx.optPolicy, optBase, curr, xOff,
                                           ctx.activeList, ctx.activeCount, dense);
        mu = fma(lambda_k, dot_k, mu);
    }
    ctx.policyMean[row * ctx.D + p] = isfinite(mu) ? tanh(mu) : 0.0f;
}

// =================================================================================================
// reduce_alpha — compute α_imp = α₀ / (1 + α₀ · ⟨‖x_active‖²⟩) and α_enc (centered-obs variant)
// once per step. 128-thread TG with hierarchical simd_sum. Saves K·R·R recomputations in
// learn_psi and R recomputations in learn_value.
// also REFRESHES per-feature TIDBD α_i = exp(β_i) / (1 + exp(β_i) · featSM[i])
// for i ∈ [0, E). Encoder features (i < R) use featSM[i] as the second-moment proxy; outcome
// dims (i ≥ R) fall back to the global α (no per-feature SM available — they're predictions,
// not directly-driven activations). β_i meta-update lives in oak_learn_value (the only place
// with per-feature δ aggregated across the batch). SwiftSarsa 2507.19539 + MetaOptimize
// 2402.02342: per-feature implicit step shrinks high-noise features and grows informative
// ones — the BWH-orthodox version of an Adam-style per-parameter step-size.
//   [Dispatch: 1 threadgroup × 128 threads.]
// =================================================================================================
kernel void oak_reduce_alpha(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float n2_l = 0.0f, obs_l = 0.0f, sg2_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        n2_l += active_norm2(prev, r * ctx.E, ctx.activeList, ctx.activeCount);
        for (uint f = 0u; f < ctx.baseO; ++f) {
            float c = standardized_obs(ctx.observation[r * ctx.O + f],
                                       ctx.observationMean[f], ctx.obsSM[f], ctx.ridge);
            obs_l = fma(c, c, obs_l);
        }
        // per-row ‖scoreGrad‖² aggregated for alpha_pi (REINFORCE implicit step,
        // Cauchy-Schwarz upper bound on the outer-product gradient norm).
        for (uint d = 0u; d < ctx.D; ++d) {
            float sg = ctx.scoreGrad[r * ctx.D + d];
            sg2_l = fma(sg, sg, sg2_l);
        }
    }
    threadgroup float sg_n2[4], sg_obs[4], sg_sg2[4];
    float n2  = tg_sum_128(n2_l,  sg_n2,  tid, sgid);
    float obs = tg_sum_128(obs_l, sg_obs, tid, sgid);
    float sg2 = tg_sum_128(sg2_l, sg_sg2, tid, sgid);
    if (tid == 0u) {
        float invB = 1.0f / float(max(ctx.B, 1u));
        scalars->alphaImp = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invB);
        scalars->alphaEnc = ctx.alpha0 / (1.0f + ctx.alpha0 * obs * invB);
        // alphaPi denominator includes ⟨‖x‖²⟩·⟨‖scoreGrad‖²⟩ (Cauchy-Schwarz on outer
        // product) — Kim 2510.06149 generalized; Stream-X 2410.14606 ObGD §A4.
        scalars->alphaPi = ctx.alpha0
                         / (1.0f + ctx.alpha0 * (n2 * invB) * (sg2 * invB));
    }
    // Per-feature TIDBD α (SwiftSarsa 2507.19539 Sec 3.2 + MetaOptimize 2402.02342).
    // α_i = exp(β_i) / (1 + exp(β_i) · max(featSM[i], ridge))
    //     = 1 / (exp(-β_i) + max(featSM[i], ridge))
    // Log-space form is overflow-safe: exp(-β_i) underflows for β_i → +∞ giving
    // α_i → 1/sm_i (natural upper bound from the implicit-α condition); saturates to +Inf
    // for β_i → -∞ giving α_i → 0 cleanly via 1/Inf. featSM floored at ridge (= 1/(cap+1))
    // structurally prevents the early-training case where featSM ≈ 0 makes α_i unbounded
    // (and gMeta·h_old in oak_learn_value drives β_i runaway). Outcome dims R..E use the
    // batch-mean ‖x‖² as the per-feature SM proxy.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float n2_mean = n2 / float(max(ctx.B, 1u));
    for (uint i = tid; i < ctx.E; i += tgW) {
        float beta_i = ctx.alphaImpBeta[i];
        float sm_i = (i < ctx.R) ? ctx.featSM[i] : n2_mean;
        sm_i = max(sm_i, ctx.ridge);
        ctx.alphaImpFeat[i] = 1.0f / (exp(-beta_i) + sm_i);
    }
}

// =================================================================================================
// learn_value — update w_main[feat] with δ_total · x[feat] reduction, via simd_sum across batch.
// Uses precomputed α_imp (single source of truth; no per-feature recomputation). Also updates
// featSM[feat] (EMA of x²) and w̄[feat] (slow high-water-mark).
//   [Dispatch: R threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_learn_value(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = tgid;
    if (feat >= ctx.E) return;     // valueW spans the OPSR-extended state
    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float g_l = 0.0f, sm_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float xp = prev[r * ctx.E + feat];
        g_l  = fma(ctx.td[r], xp, g_l);
        sm_l = fma(xp, xp, sm_l);
    }
    float g  = simd_sum(g_l);
    float sm = simd_sum(sm_l);
    if (tid == 0u) {
        float invB = 1.0f / float(max(ctx.B, 1u));
        float invT = track_rate(ctx.stepCount, ctx.ridge);
        // Per-feature TIDBD step (SwiftSarsa 2507.19539 Alg 1 + MetaOptimize 2402.02342).
        // alphaImpFeat refreshed in oak_reduce_alpha; meta-update β_i ← β_i + α₀·g·h_i;
        // h trace decays via max(0, 1 − α·sm_i) which IMPLICITLY self-bounds when α·sm
        // exceeds 1 (the SwiftSarsa Sec 3.2 self-clipping mechanism — no explicit clamp).
        float step = ctx.alphaImpFeat[feat];
        if (feat < ctx.R) {
            float fmNew = ctx.featSM[feat] + (sm * invB - ctx.featSM[feat]) * invT;
            ctx.featSM[feat] = nf_safe(fmNew);
        }
        float lambda_L2 = l2er_decay(ctx.alpha0, ctx.R, ctx.scalars->pEff);
        // Backward-view TD(λ) eligibility trace at B=1 (Sutton-Barto Ch 12.7) with the
        // true-online-TD(λ) step scaling (Seijen-Sutton 2014 ICML §3): eff_step = α·(1-λ)
        // bounds the per-step weight update by α·(1-λ)·td·e ≤ α·td·ϕ.
        float gradStep;
        if (ctx.B == 1u) {
            float xp = nf_safe(prev[feat]);
            ctx.valueE[feat] = nf_safe(ctx.scalars->lambdaTrace * ctx.valueE[feat] + xp);
            float eff_step = step * (1.0f - ctx.scalars->lambdaTrace);
            gradStep = eff_step * nf_safe(ctx.td[0]) * ctx.valueE[feat];
        } else {
            gradStep = step * (g * invB);
        }
        // Strict ϖ-RTD rank-1 PSD perturbation along ψ̄ (Sakha-Kamalapurkar-Meyn Eq. 24,
        // Theorem 2.4(ii)). δ_r = 1 is the paper-validated structural choice — for ANY
        // δ_r ≥ 0 the mean flow Ā(λ;δ_r) = Ā(λ;0) − δ_r · ψ̄ψ̄ᵀ is uniformly Hurwitz when
        // μ = ϖ (i.e. when ψ̄ = featMean is the empirical state distribution mean, which
        // it is by construction). The perturbation acts as guaranteed contraction along the
        // ψ̄ direction without disturbing the mean flow's other eigenvalues. Critical for
        // stability with rich features where Σ(0) is full-rank and the L2-ER decay weakens.
        float vMeanBase   = ctx.scalars->vMeanBaseline;
        float psi_bar_i   = (feat < ctx.R) ? ctx.featMean[feat] : 0.0f;
        float rtd_decay   = step * vMeanBase * psi_bar_i;
        float wNew = ctx.valueW[feat] * (1.0f - lambda_L2) + gradStep - rtd_decay;
        ctx.valueW[feat] = nf_safe(wNew);
        if (feat < ctx.R) {
            float decay = 1.0f - ctx.alpha0;
            ctx.wBar[feat] = max(ctx.wBar[feat] * decay, fabs(wNew));
        }
        // TIDBD β + h update (SwiftSarsa Alg 1 lines 9,11). h self-clips via max(0,...).
        float gMeta = (step > 0.0f) ? gradStep / step : 0.0f;
        float sm_i = (feat < ctx.R) ? max(ctx.featSM[feat], ctx.ridge) : (sm * invB);
        float h_old = ctx.alphaImpH[feat];
        ctx.alphaImpBeta[feat] = nf_safe(ctx.alphaImpBeta[feat] + ctx.alpha0 * gMeta * h_old);
        ctx.alphaImpH[feat]    = nf_safe(h_old * max(0.0f, 1.0f - step * sm_i) + step * gMeta);
    }
}

// =================================================================================================
// learn_encoder — semi-gradient TD update on encoder[i, o]. Uses the heuristic td · obs[o]
// gradient (no valueW chain factor) to avoid amplifying with value-head magnitude — the
// strict ϖ-RTD form in oak_learn_value handles bootstrap stability separately. Per-row
// differentiation comes from oak_spr_fused's per-i SPR delta. Two-timescale (Borkar 1997):
// encoder learns slower than value head, both stable.
// L2-ER decay (He 2509.22335) with structural rank floor (always damp ≥ 1 direction).
//   [Grid: R · baseObsDim — one thread per (latent, obs) parameter pair.]
// =================================================================================================
kernel void oak_learn_encoder(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint i = gid / ctx.baseO;
    uint o = gid % ctx.baseO;
    if (i >= ctx.R) return;
    float step = ctx.scalars->alphaEnc;
    float g = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        float c = standardized_obs(ctx.observation[r * ctx.O + o],
                                   ctx.observationMean[o], ctx.obsSM[o], ctx.ridge);
        g = fma(ctx.td[r], c, g);
    }
    float lambda_L2 = l2er_decay(ctx.alpha0, ctx.R, ctx.scalars->pEff);
    uint pIdx = i * ctx.baseO + o;
    float eNew = ctx.encoder[pIdx] * (1.0f - lambda_L2)
               + step * g / float(max(ctx.B, 1u));
    ctx.encoder[pIdx] = nf_safe(eNew);
}

// =================================================================================================
// spr_fused — FUSED predict + learn for SPR-Stream (Nilaksh 2602.09396) on the matrix encoder.
// Per (latent i, obs o), uses the per-i SPR delta to provide row-differentiated gradient
// (the missing piece for matrix-encoder rank diversity since oak_learn_encoder's td · obs[o]
// gradient is shared across all i in a column).
// Per-row math:
//   ẑ_N[r, i] = tanh((Σ_k chordPrev[r, k] · sprDecoder[k, i]) · z_{N-1}[r, i])
//   δ_r       = z_N[r, i] − ẑ_N[r, i]
//   encoder[i, o] gradient: -2 · δ_r · obs[r, o]   (no tanh' chain on z — paired with the
//                                                   strict ϖ-RTD stability in oak_learn_value)
//   sprDecoder[k, i]: -2 · δ_r · (1 − ẑ²) · chord[r, k] · z_{N-1}[r, i]
// Decoder update only on the o == 0 thread of each i to avoid redundant work + atomics.
//   [Grid: R · baseObsDim — one thread per encoder parameter; decoder dispatched on o == 0.]
// =================================================================================================
kernel void oak_spr_fused(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint i = gid / ctx.baseO;
    uint o = gid % ctx.baseO;
    if (i >= ctx.R) return;

    device const float *currState = state_slot(ctx, 1u - ctx.stateSlot);
    device const float *prevState = state_slot(ctx, ctx.stateSlot);
    float invB = 1.0f / float(max(ctx.B, 1u));

    float g_RL = 0.0f, g_SPR = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        float c     = standardized_obs(ctx.observation[r * ctx.O + o],
                                       ctx.observationMean[o], ctx.obsSM[o], ctx.ridge);
        float z     = currState[r * ctx.E + i];
        float zPrev = prevState[r * ctx.E + i];

        float gain = 0.0f;
        for (uint k = 0u; k < ctx.K; ++k) {
            float chord_k = ctx.chordSlotPrev[r * ctx.K + k];
            gain = fma(chord_k, ctx.sprDecoder[k * ctx.R + i], gain);
        }
        float zhatRaw = gain * zPrev;
        float zhat    = isfinite(zhatRaw) ? tanh(zhatRaw) : 0.0f;
        float delta   = z - zhat;

        g_RL  = fma(ctx.td[r], c, g_RL);
        g_SPR = fma(-2.0f * delta, c, g_SPR);
    }
    g_RL  *= invB;
    g_SPR *= invB;

    float denom = g_RL * g_RL + ctx.ridge;
    float coef  = (g_RL * g_SPR) / denom;
    float g_SPR_damped = g_SPR - coef * g_RL;

    float step = ctx.scalars->alphaEnc;
    uint pIdx = i * ctx.baseO + o;
    ctx.encoder[pIdx] = nf_safe(ctx.encoder[pIdx] + step * g_SPR_damped);

    if (o == 0u) {
        float gD[OAK_SIMD];
        for (uint k = 0u; k < ctx.K; ++k) gD[k] = 0.0f;
        for (uint r = 0u; r < ctx.B; ++r) {
            float z     = currState[r * ctx.E + i];
            float zPrev = prevState[r * ctx.E + i];
            float gain = 0.0f;
            for (uint k = 0u; k < ctx.K; ++k) {
                float chord_k = ctx.chordSlotPrev[r * ctx.K + k];
                gain = fma(chord_k, ctx.sprDecoder[k * ctx.R + i], gain);
            }
            float zhatRaw = gain * zPrev;
            float zhat    = isfinite(zhatRaw) ? tanh(zhatRaw) : 0.0f;
            float dTanhZhat = 1.0f - zhat * zhat;
            float delta = z - zhat;
            for (uint k = 0u; k < ctx.K; ++k) {
                float chord_k = ctx.chordSlotPrev[r * ctx.K + k];
                gD[k] = fma(chord_k * zPrev * dTanhZhat, delta, gD[k]);
            }
        }
        for (uint k = 0u; k < ctx.K; ++k) {
            float dDecoder = -2.0f * gD[k] * invB;
            uint dIdx = k * ctx.R + i;
            ctx.sprDecoder[dIdx] = nf_safe(ctx.sprDecoder[dIdx]
                                           + ctx.scalars->alphaImp * dDecoder);
        }
    }
}

// =================================================================================================
// learn_outcome — OPSR head update (; Vereecken 2604.07016). Each outcome
// dimension k predicts the next-step value of the cumulant feature sf[k] (the same feature
// the chord-conditional ψ_k tracks for option k). Squared-error TD on the ROLLED-FORWARD
// state[curr][r][R + k]:
//   target_k(r)  = state_curr[r][sf[k]]                                  // 1-step lookahead
//   pred_k(r)    = state_curr[r][R + k] = H[k] · state_prev[r][:E]        // outcome head
//   δ_outcome    = target_k − pred_k
//   ∇_{H[k][e]} L = − Σ_r δ_outcome · state_prev[r][e]
// Update: H[k][e] += step · Σ_r δ_outcome[r] · state_prev[r][e] / B (positive sign to descend).
// Per-update implicit α stabilizes against state-norm spikes (Kim 2510.06149).
//   [Grid: K · E]
// =================================================================================================
kernel void oak_learn_outcome(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = gid / ctx.E, e = gid % ctx.E;
    if (k >= ctx.K || e >= ctx.E) return;
    if (ctx.activeMask[e] == 0u) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    uint sf_k = ctx.sf[k];
    if (sf_k >= ctx.R) return;   // outcomes track encoder features only (sf[] ranges over [0, R))

    // pred = tanh(outcomeHead·prev) (oak_predict_outcome applies the tanh squash for
    // matched-scale prediction). The MSE gradient w.r.t. outcomeHead must include the
    // tanh' = (1 − pred²) chain factor; without it the gradient keeps pushing outcomeHead
    // larger even when the output saturates, killing outcome features as a representation
    // (they collapse to ±1 binary). With the chain factor the update self-gates: as pred
    // saturates, gradient → 0, outcomeHead stops growing.
    float g = 0.0f, n2 = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        float target = curr[r * ctx.E + sf_k];
        float pred   = curr[r * ctx.E + ctx.R + k];
        float delta  = target - pred;
        float dTanh  = 1.0f - pred * pred;          // tanh'(·) = 1 − tanh²(·) = 1 − pred²
        float xPrev  = prev[r * ctx.E + e];
        float term   = delta * dTanh * xPrev;
        g  = fma(1.0f, term, g);
        // Implicit-α denominator includes the same dTanh factor (Cauchy-Schwarz on the
        // effective per-row gradient norm), so αeff stays valid for the chained update.
        float scale  = dTanh * xPrev;
        n2 = fma(scale, scale, n2);
    }
    float invB = 1.0f / float(max(ctx.B, 1u));
    float alpha_eff = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invB);
    uint pIdx = k * ctx.E + e;
    ctx.outcomeHead[pIdx] = nf_safe(ctx.outcomeHead[pIdx] + alpha_eff * g * invB);
}

// =================================================================================================
// learn_psi — update shared ψ[k, m] for m ∈ [0, M = E + K). Cumulant k's gradient w.r.t.
// weight m:  ∂L/∂W_ψ[k][m] = − (1/B) Σ_r δ_k[r] · concat[s_r, λ_prev_r][m]
// where concat[r][m] = prev_state[r][m] for m < E, else λ_prev[r][m - E] (
// SOFT chord weighting via chordSlotPrev — replaces the pre-Tier-3 one-hot indicator).
// Every row contributes to every cumulant's parameters via its own δ_k[r] (chord-conditional
// formulation; OBB §2.4). Implicit α from oak_reduce_alpha provides per-step contraction
// independent of input scale (Kim 2510.06149); no clamp needed.
//   [Grid: K · M = K · (E + K)]
// =================================================================================================
kernel void oak_learn_psi(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint M = ctx.M;
    uint k = gid / M, m = gid % M;
    if (k >= ctx.K) return;
    // M = E + K. State weights are at [0, E); chord weights at [E, M). State-weight gating
    // uses activeMask[m] (outcome dims R..E are always-active by init).
    if (m < ctx.E && ctx.activeMask[m] == 0u) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    // state weights use per-feature TIDBD α[m]; chord weights (m ≥ E) fall back
    // to the global scalar α (no per-chord SM tracker — chord weights are dense and uniform-
    // scale by simplex constraint).
    // Per-feature TIDBD α for state weights; chord weights use scalar α (no per-chord SM).
    float step = (m < ctx.E) ? ctx.alphaImpFeat[m] : ctx.scalars->alphaImp;
    float g = 0.0f;
    if (m < ctx.E) {
        // State weight: every row contributes via its own δ_k.
        for (uint r = 0u; r < ctx.B; ++r) {
            g = fma(ctx.tdPsi[r * ctx.K + k], prev[r * ctx.E + m], g);
        }
    } else {
        // chord weight W_ψ[k][E + cidx] gets soft-chord-weighted contribution.
        // ψ_k uses chordPrev (the chord active during transition); ∂ψ_k/∂W_ψ[k][E+cidx] = λ_prev[cidx].
        uint cidx = m - ctx.E;
        for (uint r = 0u; r < ctx.B; ++r) {
            float lambda_cidx = ctx.chordSlotPrev[r * ctx.K + cidx];
            g = fma(ctx.tdPsi[r * ctx.K + k] * lambda_cidx, 1.0f, g);
        }
    }
    float invB = 1.0f / float(max(ctx.B, 1u));
    uint  pIdx = k * M + m;
    // L2-Effective-Rank regularizer on ψ (Spectral Collapse 2509.22335).
    float lambda_L2 = l2er_decay(ctx.alpha0, ctx.R, ctx.scalars->pEff);
    // Backward-view TD(λ) trace at B=1 (one trace per ψ weight). e[k][m] = λ·e[k][m]
    // + ∇ψ_k(s_t)[m] where ∇ψ_k = prev[m] (state weight) or chordPrev[m−E] (chord weight).
    float gradStep;
    if (ctx.B == 1u) {
        float gradV = (m < ctx.E) ? nf_safe(prev[m]) : nf_safe(ctx.chordSlotPrev[m - ctx.E]);
        ctx.psiE[pIdx] = nf_safe(ctx.scalars->lambdaTrace * ctx.psiE[pIdx] + gradV);
        float eff_step = step * (1.0f - ctx.scalars->lambdaTrace);
        gradStep = eff_step * nf_safe(ctx.tdPsi[k]) * ctx.psiE[pIdx];
    } else {
        gradStep = step * g * invB;
    }
    // Strict ϖ-RTD rank-1 PSD perturbation (Sakha-Kamalapurkar-Meyn Eq. 24, Theorem 2.4(ii))
    // applied per-cumulant. δ_r · ψMeanBaseline_k · meanBasis[m]: meanBasis = featMean for
    // state weights (m < E), uniform-chord 1/K for chord weights (m ≥ E). The chord prior
    // 1/K is structural (matches the engine's seed chord distribution and the long-run mean
    // under uniform option dynamics) — not a tuned scalar. δ_r = 1 paper-validated.
    float psiMeanBase = ctx.psiMeanBaseline[k];
    float meanBasis;
    if (m < ctx.E) {
        meanBasis = (m < ctx.R) ? ctx.featMean[m] : 0.0f;  // outcome dims have no Welford
    } else {
        meanBasis = 1.0f / float(max(ctx.K, 1u));
    }
    float rtd_decay = step * psiMeanBase * meanBasis;
    ctx.psi[pIdx] = nf_safe(ctx.psi[pIdx] * (1.0f - lambda_L2) + gradStep - rtd_decay);
}

// =================================================================================================
// learn_pi — REAL REINFORCE on the per-option policy. chord-weighted gradient:
//   g[k,d,e] = Σ_r λ_prev[r,k] · (td[r] − tdBaseline) · scoreGrad[r,d] · prev_state[r,e]
// since meta_action emits μ_d = Σ_k λ_k · (W_π[k][d] · s) (chord-weighted soft mixture), the
// per-option contribution scales by chord_prev[r,k]. scoreGrad[r,d] is wrapper-written
// (Tier 1: Gumbel-max categorical for Discrete components, Gaussian-σ=1 for Box components).
// implicit step uses scalars->alphaPi (computed once in oak_reduce_alpha).
// alphaPi = α₀ / (1 + α₀ · ⟨‖x‖²⟩_batch · ⟨‖scoreGrad‖²⟩_batch) — the Cauchy-Schwarz upper
// bound on the per-parameter outer-product gradient norm. per-thread n2 missed the
// cross-row scoreGrad norm-square aggregation (proximate cause of 200M MSE blowup
// stability research §11). The shared alphaPi denominator is the structurally-correct fix and
// also removes the per-thread n2 reduction (saves O(B) FMA per thread × K·D·E threads).
//   [Grid: K · D · E — K options × D_policy params × E extended-state features]
// =================================================================================================
kernel void oak_learn_pi(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = gid / (ctx.D * ctx.E);
    uint rem = gid - k * ctx.D * ctx.E;
    uint d = rem / ctx.E, i = rem % ctx.E;
    if (k >= ctx.K || d >= ctx.D || i >= ctx.E) return;
    if (ctx.activeMask[i] == 0u) return;   // outcome dims R..E are always-active by init

    // ACTOR-CRITIC WITH ELIGIBILITY TRACES (Sutton-Barto §13.5). Per-step gradient of
    // log π(a|s) w.r.t. W_π[k][d][e] for a soft-chord-weighted Gaussian policy is
    //   ∇_{W} log π = λ_k(s) · prev_state[e] · scoreGrad[d]
    // The backward-view trace e_θ ← λ_trace · e_θ + ∇log π integrates this gradient over
    // the effective horizon 1/(1−λ_trace); update is then optPolicy ← optPolicy + αpi·δ·e_θ.
    // Subtract tdBaseline so the trace integrates a zero-mean signal (Sutton-Barto §13.4).
    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float baseline = ctx.scalars->tdBaseline;
    float g = 0.0f, deltaSum = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        float lambda_k = ctx.chordSlotPrev[r * ctx.K + k];
        float pi_term  = prev[r * ctx.E + i] * ctx.scoreGrad[r * ctx.D + d];
        g  = fma(lambda_k, pi_term, g);
        deltaSum += nf_safe(ctx.td[r]) - baseline;
    }
    float invB = 1.0f / float(max(ctx.B, 1u));
    float alpha_pi = ctx.scalars->alphaPi;       // shared denominator includes ‖scoreGrad‖²
    uint  pIdx = k * ctx.D * ctx.E + d * ctx.E + i;
    // Trace decay-and-accumulate (backward view, vanilla Sutton-Barto §13.5 form): NO
    // (1−λ) scaling on the actor update — that's a Seijen-Sutton (2014) bias correction
    // specific to value/return learning, not actor gradients. The trace integrates ∇log π
    // over horizon 1/(1−λ) so accumulated piE has magnitude g/(1−λ) at steady state, and
    // αpi already absorbs the per-parameter bound through its implicit denominator.
    float lambdaTrace = ctx.scalars->lambdaTrace;
    ctx.piE[pIdx] = nf_safe(lambdaTrace * ctx.piE[pIdx] + g * invB);
    float gradStep = alpha_pi * (deltaSum * invB) * ctx.piE[pIdx];
    // No L2 decay on the policy head (it's an action emitter, not a representation —
    // Sutton-Barto Ch 13). Boundedness comes from the implicit-α step.
    ctx.optPolicy[pIdx] = nf_safe(ctx.optPolicy[pIdx] + gradStep);
}

// =================================================================================================
// learn_beta — termination-gradient update on betaW[E + K]. Per Bacon-Precup
// 1609.05140 (Option-Critic) under Wan-Naik-Sutton 2110.13855 (continuing avg-reward):
//   ∇_β J = (β − 1{terminated}) · A(s_prev, λ_prev)
// where A is the differential-TD advantage td[r] (already includes the planning bonus from
// oak_plan). Per-weight gradient on betaW[m] (state-or-chord index m ∈ [0, M_β = E + K)):
//   g[m] = (1/B) Σ_r (β(s_prev, λ_prev) − terminated(r)) · td[r] · concat[s_prev, λ_prev][m]
// Update: betaW[m] -= alpha_eff · g[m]  (NEGATIVE: descend the loss = β·A increasing β when A < 0,
// decreasing when A > 0 — i.e., terminate sooner when the option is bad).
// `terminated(r)` is recovered from the SHIFT of optionAge: optionAge[r] == 1u iff
// meta_policy resampled the chord at the *current* step (which corresponds to terminating
// out of the prev step's option).
//   [Grid: E + K — one thread per betaW weight.]
// =================================================================================================
kernel void oak_learn_beta(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint mBetaW = ctx.E + ctx.K;
    uint m = gid;
    if (m >= mBetaW) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float baseline = ctx.scalars->tdBaseline;
    float g = 0.0f, n2 = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        float beta = ctx.betaProb[r];
        // 1{terminated}: optionAge[r] == 1u means meta_policy resampled at this step.
        float terminated = (ctx.optionAge[r] == 1u) ? 1.0f : 0.0f;
        float adv = ctx.td[r] - baseline;
        float feat = (m < ctx.E) ? prev[r * ctx.E + m]
                                 : ctx.chordSlotPrev[r * ctx.K + (m - ctx.E)];
        float term = (beta - terminated) * adv * feat;
        g  += term;
        n2 = fma(feat, feat, n2);
    }
    float invB = 1.0f / float(max(ctx.B, 1u));
    float alpha_eff = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invB);
    // Sign: minimize J = β · A → descend g (negative gradient direction).
    ctx.betaW[m] = nf_safe(ctx.betaW[m] - alpha_eff * g * invB);
}

// =================================================================================================
// learn_omega — chord-generator update via softmax policy gradient. omegaW
// produces K logits from E-dim state; chord = softmax(logits). Per-weight gradient:
//   g[k][e] = (1/B) Σ_{r: terminated this step} (1{k chosen at r} − chord_prev[r][k]) · adv[r] · prev[r][e]
// Only TERMINATION rows contribute (the standard option-critic formulation per Bacon-Precup
// 1609.05140): on rows where chord persists across the step, ω did not produce a sample so
// its gradient signal is zero. comment fixed to match code semantics
// (prior version claimed soft attribution on non-term rows; code never did that).
//   [Grid: K · E — one thread per (k, e).]
// =================================================================================================
kernel void oak_learn_omega(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = gid / ctx.E, e = gid % ctx.E;
    if (k >= ctx.K || e >= ctx.E) return;
    if (ctx.activeMask[e] == 0u) return;

    device const float *prev = state_slot(ctx, ctx.stateSlot);
    float baseline = ctx.scalars->tdBaseline;
    float g = 0.0f, n2 = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) {
        bool terminatedHere = (ctx.optionAge[r] == 1u);
        if (!terminatedHere) continue;  // chord only resampled at termination → grad only on those rows
        float lambda_k = ctx.chordSlot[r * ctx.K + k];
        float chosen = (ctx.option[(1u - ctx.optionSlot) * ctx.B + r] == k) ? 1.0f : 0.0f;
        float adv = ctx.td[r] - baseline;
        float feat = prev[r * ctx.E + e];
        float term = (chosen - lambda_k) * adv * feat;
        g += term;
        n2 = fma(feat, feat, n2);
    }
    float invB = 1.0f / float(max(ctx.B, 1u));
    float alpha_eff = ctx.alpha0 / (1.0f + ctx.alpha0 * n2 * invB);
    uint pIdx = k * ctx.E + e;
    // Sign: ascend J = log π(k|s) · A → +g direction.
    ctx.omegaW[pIdx] = nf_safe(ctx.omegaW[pIdx] + alpha_eff * g * invB);
}

// =================================================================================================
// bk_obs — per-feature observation Welford EMAs (mean + 2nd moment). Runs FIRST in the pipeline
// so encode and downstream learners see correctly-scaled inputs from step 0 (eliminating the
// Welford-bootstrap transient that previously saturated tanh on the very first step). One TG of
// 32 threads strided across the O obs dims; per-step NF count stashed in counters[OAK_CNT_OBS_NF]
// for the (B) numerical-safety rollup that bk_reduce performs at end-of-step.
//   [Dispatch: 1 threadgroup × 32 threads.]
// =================================================================================================
kernel void oak_bk_obs(
    device float         *params      [[buffer(0)]],
    device float         *stateArena  [[buffer(1)]],
    device uint          *uintArena   [[buffer(2)]],
    device KernelScalars *scalars     [[buffer(3)]],
    constant KernelConfig &cfg        [[buffer(4)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    uint nfLocal = 0u;
    for (uint f = tid; f < ctx.O; f += tgW) {
        float sum = 0.0f, sumSq = 0.0f;
        uint nf = 0u;
        for (uint r = 0u; r < ctx.B; ++r) {
            float o = ctx.observation[r * ctx.O + f];
            if (isfinite(o)) { sum += o; sumSq = fma(o, o, sumSq); }
            else nf++;
        }
        float invB = 1.0f / float(max(ctx.B, 1u));
        float meanBatch = sum   * invB;
        float smBatch   = sumSq * invB;
        ctx.observationMean[f] = nf_safe(ctx.observationMean[f]
                                       + invT * (meanBatch - ctx.observationMean[f]));
        ctx.obsSM[f]           = nf_safe(ctx.obsSM[f]
                                       + invT * (smBatch   - ctx.obsSM[f]));
        nfLocal += nf;
    }
    uint nfTotal = simd_sum(nfLocal);
    if (tid == 0u) ctx.counters[OAK_CNT_OBS_NF] = nfTotal;
}

// =================================================================================================
// bk_reduce — global telemetry. One 128-thread TG with hierarchical simd_sum for the batch-wide
// (row,feat) and (row,a) reductions. Per-feature utility lives in a separate kernel
// (oak_bk_feature, R TGs × 32) to preserve R-way parallelism at large R·B.
//   [Dispatch: 1 threadgroup × 128 threads.]
// =================================================================================================
kernel void oak_bk_reduce(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);

    // (1) Per-feature obs Welford EMAs (mean + 2nd-moment) are updated by oak_bk_obs
    // BEFORE encode each step, so encode sees correctly-scaled inputs from step 0. Here
    // we just rebroadcast the NF count it stashed into counters[OAK_CNT_OBS_NF] so the
    // (B) numerical-safety rollup picks it up.
    float obsNF = float(ctx.counters[OAK_CNT_OBS_NF]);

    // (1b) Per-feature mean EMA on encoder features (running ρ_k for cumulant centering
    // in oak_predict). Outcome dims R..E are bounded by oak_predict_outcome's bounded_soft;
    // their centering is implicit in the prediction target. NF-guarded reads keep the EMA
    // bounded under any propagated NaN/Inf (Inv-4 zero-bias substitution).
    // Two-timescale: featMean uses the FAST rate β_n = 1/√n so β/α = √n → ∞ (Sakha §1.3).
    float invT_fast = track_rate_fast(ctx.stepCount, ctx.ridge);
    for (uint f = tid; f < ctx.R; f += tgW) {
        float sum = 0.0f;
        for (uint r = 0u; r < ctx.B; ++r) sum += nf_safe(curr[r * ctx.E + f]);
        float meanBatch = sum / float(max(ctx.B, 1u));
        float fmNew = ctx.featMean[f] + invT_fast * (meanBatch - ctx.featMean[f]);
        ctx.featMean[f] = nf_safe(fmNew);
    }

    // (2) Latent telemetry: mean-square + max|latent| + non-finite count over the FULL
    // E-wide state (B · E cells; runaway outcomeHead would show up here).
    // state stride is ctx.E (was ctx.R — bug since Tier 2
    // when state grew from R to E cols). Telemetry was reading wrong cells for r >= 1.
    float lms_l = 0.0f, maxLat_l = 0.0f, nLearn_l = 0.0f;
    uint total_rf = ctx.B * ctx.E;
    for (uint idx = tid; idx < total_rf; idx += tgW) {
        uint r = idx / ctx.E, f = idx % ctx.E;
        float v = curr[r * ctx.E + f];
        if (!isfinite(v)) { nLearn_l += 1.0f; v = 0.0f; }
        lms_l = fma(v, v, lms_l);
        maxLat_l = max(maxLat_l, fabs(v));
    }
    // (2b) ψ TD telemetry over B·K cells (Design 2: K cumulants per row, not R).
    // NF scan + zero-replacement on tdPsi BEFORE the MSE roll-up.
    // silently passed non-finite tdPsi values into optionModelPredictionMSE,
    // contaminating the canonical learning-progress signal at deep workerDepth (200M
    // MSE traced to NaN poisoning, not real model error). Zero-bias replacement per
    // Inv-4: a non-finite tdPsi means the prediction failed entirely, equivalent to
    // a missed update; treating it as zero in the MSE accumulator gives the unbiased
    // mean-square over only the well-defined predictions.
    float psi_l = 0.0f, tdPsiNF_l = 0.0f;
    uint total_rk = ctx.B * ctx.K;
    for (uint idx = tid; idx < total_rk; idx += tgW) {
        float dp = ctx.tdPsi[idx];
        if (!isfinite(dp)) { tdPsiNF_l += 1.0f; dp = 0.0f; ctx.tdPsi[idx] = 0.0f; }
        psi_l = fma(dp, dp, psi_l);
    }
    threadgroup float sg_lms[4], sg_psi[4], sg_maxLat[4], sg_nLearn[4], sg_tdPsiNF[4];
    float lms      = tg_sum_128(lms_l,      sg_lms,      tid, sgid);
    float psi      = tg_sum_128(psi_l,      sg_psi,      tid, sgid);
    float maxLat   = tg_max_128(maxLat_l,   sg_maxLat,   tid, sgid);
    float nLearn   = tg_sum_128(nLearn_l,   sg_nLearn,   tid, sgid);
    float tdPsiNF  = tg_sum_128(tdPsiNF_l,  sg_tdPsiNF,  tid, sgid);

    // (3) Per-(row, p) reductions on policyMean[B·D]: actuation change vs prev step,
    // max|μ|, non-finite. With wrapper-side sampling, prevPolicyMean and policyMean
    // both sit in arena (host-written for prev, kernel-written for this step), so the delta
    // measures POLICY churn (μ_t − μ_{t-1}), independent of wrapper-side noise.
    float ad_l = 0.0f, sgL2_l = 0.0f, maxPol_l = 0.0f, nPlan_l = 0.0f;
    uint total_rd = ctx.B * ctx.D;
    for (uint idx = tid; idx < total_rd; idx += tgW) {
        float x  = ctx.policyMean[idx];
        float py = ctx.prevPolicyMean[idx];
        bool nf = !isfinite(x);
        if (nf) { nPlan_l += 1.0f; x = 0.0f; }
        float p2 = x - py;
        ad_l = fma(p2, p2, ad_l);
        maxPol_l = max(maxPol_l, fabs(x));
        float sg = ctx.scoreGrad[idx];
        if (!isfinite(sg)) sg = 0.0f;
        sgL2_l = fma(sg, sg, sgL2_l);
    }
    threadgroup float sg_ad[4], sg_sgL2[4], sg_maxPol[4], sg_nPlan[4];
    float ad     = tg_sum_128(ad_l,     sg_ad,     tid, sgid);
    float sgL2   = tg_sum_128(sgL2_l,   sg_sgL2,   tid, sgid);
    float maxPol = tg_max_128(maxPol_l, sg_maxPol, tid, sgid);
    float nPlan  = tg_sum_128(nPlan_l,  sg_nPlan,  tid, sgid);

    if (tid == 0u) {
        float invBE = 1.0f / float(max(ctx.B * ctx.E, 1u));   // OPSR-extended state width
        float invBK = 1.0f / float(max(ctx.B * ctx.K, 1u));
        float invBD = 1.0f / float(max(ctx.B * ctx.D, 1u));
        stats->latentStateMeanSquare      = lms * invBE;
        stats->optionModelPredictionMSE   = psi * invBK;
        stats->scoreGradL2Mean            = sgL2 * invBD;  // mean ‖∇log π‖² across rows × params
        stats->actionDeltaL2Mean          = ad  * invBD;   // μ-churn over D policy params
        stats->maxAbsLatentPrediction     = maxLat;
        stats->maxAbsPolicyOutput         = maxPol;
        // tdPsiNonFiniteCount surfaces the previously-silent NF
        // poisoning into rawNonFiniteCount, so the (B) numerical-safety gate sees it.
        stats->tdPsiNonFiniteCount        = tdPsiNF;
        stats->rawNonFiniteCount          = obsNF + nLearn + nPlan + tdPsiNF;
        stats->sanitizedNonFiniteCount    = nLearn + nPlan + tdPsiNF;
        stats->encodeNonFiniteCount       = obsNF;
        stats->learnNonFiniteCount        = nLearn;
        stats->planNonFiniteCount         = nPlan;
        ctx.counters[OAK_CNT_OBS_NF]    = uint(obsNF);
        ctx.counters[OAK_CNT_LEARN_NF]  = uint(nLearn);
        ctx.counters[OAK_CNT_PLAN_NF]   = uint(nPlan);
        ctx.counters[OAK_CNT_TDPSI_NF]  = uint(tdPsiNF);
    }
}

// =================================================================================================
// bk_feature — per-feature utility EMA: util_feat[i] ← track_rate·(mean|δ·x_i| + Σ_j|δψ_j·x_i|).
// GraMa-aligned gradient-magnitude signal (2505.24061). R threadgroups × 32 threads gives
// R-way outer parallelism so this scales with R even at large B; each TG uses simd_sum across
// its 32 lanes to reduce over B·(R+1) inner terms.
//   [Dispatch: R threadgroups × 32 threads.]
// =================================================================================================
kernel void oak_bk_feature(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint feat = tgid;
    if (feat >= ctx.R) return;   // utility ranking on encoder features only (sf[] picks here)
    device const float *curr = state_slot(ctx, 1u - ctx.stateSlot);
    float invB = 1.0f / float(max(ctx.B, 1u));
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    // Per-feature utility = mean over batch of |δ_main · x| + Σ_k |δ_ψ_k · x|. 1+K terms per
    // row. K cumulants now (was R features) under Design 2 — divisor is structural 1+K.
    float u_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float x = nf_safe(curr[r * ctx.E + feat]);
        u_l += fabs(nf_safe(ctx.td[r]) * x);
        for (uint k = 0u; k < ctx.K; ++k) u_l += fabs(nf_safe(ctx.tdPsi[r * ctx.K + k]) * x);
    }
    float u = simd_sum(u_l);
    if (tid == 0u) {
        float un = u * invB / float(1u + ctx.K);
        float ufNew = ctx.utilFeat[feat] + invT * (un - ctx.utilFeat[feat]);
        ctx.utilFeat[feat] = nf_safe(ufNew);
    }
}

// =================================================================================================
// bk_option — per-option utility tracking + FIRE+SWR-style reinit (Design 2 scalar cumulants).
// Cumulant centering ρ_k now lives in featMean[sf[k]] (tracked in bk_reduce); per-option
// reward-mean rhoK was telemetry-only and has been deleted. utilOpt[k] tracks |δ_k|² over rows
// where option k is active. OCP hypothesis (2603.21173) validates the conjunctive reinit gate.
//   [Dispatch: K threadgroups × 128 threads.]
// =================================================================================================
kernel void oak_bk_option(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    uint k = tgid;
    if (k >= ctx.K) return;
    float invB = 1.0f / float(max(ctx.B, 1u));
    float invT = track_rate(ctx.stepCount, ctx.ridge);

    // Per-option counts and squared-cumulant-error. SOFT chord weighting —
    // option k's "share" of each row's contribution is chordSlotPrev[r][k] (the chord that
    // was active during the transition that produced tdPsi[r][k]).
    float cnt_l = 0.0f, psiErr_l = 0.0f;
    for (uint r = tid; r < ctx.B; r += tgW) {
        float lambda_k = nf_safe(ctx.chordSlotPrev[r * ctx.K + k]);
        cnt_l += lambda_k;
        float d = nf_safe(ctx.tdPsi[r * ctx.K + k]);
        psiErr_l = fma(d * d, lambda_k, psiErr_l);
    }
    threadgroup float sg_cnt[4], sg_p[4];
    float cnt        = tg_sum_128(cnt_l,    sg_cnt, tid, sgid);
    float pErr       = tg_sum_128(psiErr_l, sg_p,   tid, sgid);
    float frac       = cnt * invB;
    float meanPsiErr = (cnt > 0.0f) ? pErr / cnt : 0.0f;

    if (tid == 0u) {
        float uoNew = ctx.utilOpt[k] + invT * (meanPsiErr - ctx.utilOpt[k]);
        ctx.utilOpt[k] = nf_safe(uoNew);
        float ofNew = ctx.optFracEMA[k] + invT * (frac - ctx.optFracEMA[k]);
        ctx.optFracEMA[k] = nf_safe(ofNew);

        // Reinit gate fires when option k is BOTH dormant (gets < uniform chord share)
        // AND stale (its TD-error² is a positive 1σ cross-option outlier). Both
        // conditions are structural cross-option statistics derived from K and σ
        // (Chebyshev one-sigma rule); no tuned scalar.
        float meanU = 0.0f;
        for (uint k2 = 0u; k2 < ctx.K; ++k2) meanU += nf_safe(ctx.utilOpt[k2]);
        meanU /= float(max(ctx.K, 1u));
        float varU = 0.0f;
        for (uint k2 = 0u; k2 < ctx.K; ++k2) {
            float dU = nf_safe(ctx.utilOpt[k2]) - meanU;
            varU += dU * dU;
        }
        float stdU = sqrt(varU / float(max(ctx.K, 1u)));
        // Structural dormancy: 1/K is the uniform chord share (no preference). An option
        // averaging below uniform is by definition under-utilized.
        // Structural staleness: utilOpt[k] is a 1σ positive outlier (Chebyshev / one-sigma
        // rule) — its TD-error² is unusually high vs the cross-option distribution.
        bool dormant = ctx.optFracEMA[k] < 1.0f / float(max(ctx.K, 1u));
        bool stale   = ctx.utilOpt[k] > meanU + stdU;
        if (dormant && stale) {
            // Targeted column reinit: zero the chord-bias W_ψ[k][E + k] (so plan stops picking
            // this option for its specific ψ chord-weight contribution) AND the cumulant column
            // W_ψ[k][sf[k]] (the feature this cumulant tracks). Preserves cross-option transfer
            // (other state weights stay). Aligned with ReinitWeightsVsUnits 2508.00212.
            uint psiBase = k * ctx.M;
            ctx.psi[psiBase + ctx.E + k] = 0.0f;
            uint sf_k = ctx.sf[k];
            if (sf_k < ctx.R) ctx.psi[psiBase + sf_k] = half(1.0f / sqrt(float(max(ctx.R, 1u))));
            // Reset the option's policy + outcome head: with stale stats they're all noise.
            uint polBase = k * ctx.D * ctx.E;
            for (uint idx = 0u; idx < ctx.D * ctx.E; ++idx) ctx.optPolicy[polBase + idx] = 0.0f;
            uint outBase = k * ctx.E;
            for (uint e = 0u; e < ctx.E; ++e) ctx.outcomeHead[outBase + e] = 0.0f;
            // Reinit option k's chord-generator row + termination chord-weight so meta_policy
            // doesn't keep re-selecting this just-reinited option immediately.
            for (uint e = 0u; e < ctx.E; ++e) ctx.omegaW[k * ctx.E + e] = 0.0f;
            ctx.betaW[ctx.E + k] = 0.0f;
            ctx.utilOpt[k] = meanU;       // reset to neutral so the gate doesn't fire repeatedly
            ctx.optFracEMA[k] = 1.0f / float(max(ctx.K, 1u));   // reset to uniform-share
            // K ≤ OAK_SIMD by host-side cap (OakMetalLayout.optionCount), so one uint
            // bitmask covers all options. Atomic OR for race-free per-option signaling
            // when multiple TGs trigger reinit in the same step.
            atomic_fetch_or_explicit((device atomic_uint *)(ctx.counters + OAK_CNT_REINIT_OPT),
                                     1u << k, memory_order_relaxed);
        }
    }
}

// =================================================================================================
// bk_admit_commit — FUSED bk_admit + bk_commit. Serial on one thread:
//   (admit) refresh activeMask/activeList/activeCount; enforce effective-rank floor ⌈√R⌉;
//           FC-STOMP encoder reinit on worst feature when any option reinited this step.
//   (commit) ρ̄ Naik update, pEff effective-rank proxy, telemetry roll-up, slot toggles,
//            stepCount++, clear per-step reinit bitmasks.
//   [Grid: 1]
// =================================================================================================
kernel void oak_bk_admit_commit(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    if (gid != 0u) return;

    // -- admit phase ---
    uint R = ctx.R;
    float sum = 0.0f;
    for (uint i = 0u; i < R; ++i) sum += ctx.utilFeat[i];
    float mean = sum / float(max(R, 1u));
    float var = 0.0f;
    for (uint i = 0u; i < R; ++i) { float d = ctx.utilFeat[i] - mean; var += d * d; }
    float std = sqrt(var / float(max(R, 1u)));

    uint count = 0u;
    uint worstIdx = 0u;
    float worstU = FLT_MAX;
    for (uint i = 0u; i < R; ++i) {
        if (ctx.utilFeat[i] > mean) {
            ctx.activeMask[i] = 1u;
            ctx.activeList[count++] = i;
        } else {
            ctx.activeMask[i] = 0u;
        }
        if (ctx.utilFeat[i] < worstU) { worstU = ctx.utilFeat[i]; worstIdx = i; }
    }
    // Effective-rank floor = ⌈√R⌉ (F11 — 2509.22335).
    uint floorR = ceil_sqrt(R);
    while (count < floorR) {
        uint pick = R; float pickU = -FLT_MAX;
        for (uint i = 0u; i < R; ++i) {
            if (ctx.activeMask[i] == 0u && ctx.utilFeat[i] > pickU) { pickU = ctx.utilFeat[i]; pick = i; }
        }
        if (pick >= R) break;
        ctx.activeMask[pick] = 1u;
        ctx.activeList[count++] = pick;
    }
    if (count == 0u) { ctx.activeMask[0] = 1u; ctx.activeList[0] = 0u; count = 1u; }
    // append the K OPSR outcome dims (R..E) — always-active by construction.
    for (uint k = 0u; k < ctx.K; ++k) {
        ctx.activeMask[ctx.R + k] = 1u;
        ctx.activeList[count++] = ctx.R + k;
    }
    ctx.scalars->activeCount = count;

    // dense-fast-path flag. Set when activeCount == E (full-rank, all
    // encoder features admitted plus all K outcome dims) AND activeList is the identity
    // ordering 0, 1, ..., E-1. In steady state after warmup this is the common case and lets
    // active_dot_half_auto skip the gather indirection.
    bool isIdentity = (count == ctx.E);
    if (isIdentity) {
        for (uint i = 0u; i < ctx.E; ++i) {
            if (ctx.activeList[i] != i) { isIdentity = false; break; }
        }
    }
    ctx.scalars->activeListIsIdentity = isIdentity ? 1u : 0u;

    // Tier 7: dynamic top-K subtask-feature ranking. Cumulant k tracks the feature with the
    // k-th highest utilFeat. BWH-orthodox structural meta-learning: the agent picks its own
    // most-informative cumulants, no prior knowledge of which features matter.
    // O(K · R) selection-sort top-K (K is hardware-capped at 32, so this is small).
    for (uint k = 0u; k < ctx.K; ++k) {
        uint best = R; float bestU = -FLT_MAX;
        for (uint i = 0u; i < R; ++i) {
            bool taken = false;
            for (uint k2 = 0u; k2 < k; ++k2) if (ctx.sf[k2] == i) { taken = true; break; }
            if (taken) continue;
            if (ctx.utilFeat[i] > bestU) { bestU = ctx.utilFeat[i]; best = i; }
        }
        ctx.sf[k] = (best < R) ? best : min(k, R - 1u);
    }

    // FC-STOMP feedback: if any option reinited this step (set by oak_bk_option) AND the
    // worst feature is more than 1σ below mean utility, reinit its base-obs encoder weight
    // with a structural ±1/√R unit (OCP hypothesis 2603.21173 conjunctive-gate validation).
    if (ctx.counters[OAK_CNT_REINIT_OPT] != 0u && worstU < mean - std && worstIdx < ctx.baseO) {
        float sign = (((ctx.stepCount + worstIdx) & 1u) == 0u) ? 1.0f : -1.0f;
        ctx.encoder[worstIdx] = half(sign / sqrt(float(max(R, 1u))));
        ctx.utilFeat[worstIdx] = mean;
        ctx.counters[OAK_CNT_REINIT_FEAT] += 1u;
    }

    // -- commit phase ---

    // Average reward ρ̄ — single tracking EMA at track_rate (floors at ridge = 1/(cap+1)).
    // Sutton-Barto §6.6 continuing avg-reward + Alberta Plan continual-learning principle:
    // baselines TRACK with non-vanishing rate so the agent never freezes. The same ρ̄ is
    // used both as stats->rewardRate (fast EMA reporting) AND as the centering baseline μ
    // in oak_predict's ϖ-RTD form (Sakha-Kamalapurkar-Meyn 2603.27874v2 Theorem 2.4(ii) —
    // Hurwitz under any tracking estimator of the asymptotic ϖ). No dual scalar; no magic
    // half-rate; the mean-flow centering and the reward stat are the SAME quantity.
    float meanR = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) meanR += nf_safe(ctx.reward[r]);
    meanR /= float(max(ctx.B, 1u));
    // ρ̄ is part of the empirical-distribution baseline ϖ; uses the FAST timescale per
    // Sakha §1.3 so β/α → ∞ holds for the same reasons featMean does (the reward stat
    // and the state-mean stat must both converge faster than the value weights).
    float invT_fast_rho = track_rate_fast(ctx.stepCount, ctx.ridge);
    ctx.scalars->rho = nf_safe(ctx.scalars->rho + (meanR - ctx.scalars->rho) * invT_fast_rho);
    float invT = track_rate(ctx.stepCount, ctx.ridge);
    // TD(λ) trace decay rate — structural λ = 1 − 1/cap (Inv-9, derived
    // from R·B). Idempotent set per step (cap is run-invariant). Used by oak_learn_value
    // and oak_learn_psi at B=1 for backward-view eligibility-trace updates.
    ctx.scalars->lambdaTrace = 1.0f - 1.0f / ctx.cap;

    // per-step recompute of the V-side and
    // ψ-side centering baselines from the just-updated weights and the current
    // featMean. Single-threaded — at K=32, R≤512 this is ~16K FMAs (negligible
    // vs the 200K+ FMAs in oak_meta_action). Inv-9 compliant: derived structurally
    // from the existing featMean Welford EMA + the just-updated linear-head weights;
    // no tuned scalar.
    float vMean = 0.0f;
    for (uint i = 0u; i < R; ++i) {
        vMean = fma(ctx.valueW[i], nf_safe(ctx.featMean[i]), vMean);
    }
    ctx.scalars->vMeanBaseline = nf_safe(vMean);
    uint M = ctx.M;
    for (uint k = 0u; k < ctx.K; ++k) {
        float pMean = 0.0f;
        uint psiBase = k * M;
        for (uint i = 0u; i < R; ++i) {
            pMean = fma(ctx.psi[psiBase + i], nf_safe(ctx.featMean[i]), pMean);
        }
        ctx.psiMeanBaseline[k] = nf_safe(pMean);
    }
    // REINFORCE running-mean baseline. Tracks mean(td) so policy gradient
    // updates use (td - tdBaseline) instead of raw td. Pure variance reduction (Williams 1992,
    // Sutton-Barto Ch.13.3). The baseline is unbiased: E[scoreGrad · b] = 0 for any constant b
    // since E[scoreGrad] = 0 (it's the gradient of log-probability normalization).
    float meanTd = 0.0f;
    for (uint r = 0u; r < ctx.B; ++r) meanTd += nf_safe(ctx.td[r]);
    meanTd /= float(max(ctx.B, 1u));
    float tdbNew = ctx.scalars->tdBaseline + (meanTd - ctx.scalars->tdBaseline) * invT;
    ctx.scalars->tdBaseline = nf_safe(tdbNew);

    // + track BOTH a fast EMA of mean ψ TD error squared
    // (modelErrEMA) AND a slow lagged snapshot (modelErrEMAPrev). The slow snapshot updates
    // at half the fast rate so lpmRate = max(0, modelErrEMAPrev - modelErrEMA) is positive
    // when the model is IMPROVING. This is Hou-An-Du 2509.25438 (LPM, ICLR 2026): the
    // monotone indicator of Information Gain. Used by oak_plan to UNGATE the planning bonus
    // when the model improves (fixes Dynamic-Obstacles regression caused by absolute
    // gating) and by oak_meta_policy as `β += modelErrEMA - lpmRate` (terminate when wrong
    // AND not improving; keep when wrong but improving).
    float meanPsiSq = 0.0f;
    uint psiCount = ctx.B * ctx.K;
    for (uint i = 0u; i < psiCount; ++i) {
        float dp = nf_safe(ctx.tdPsi[i]);
        meanPsiSq = fma(dp, dp, meanPsiSq);
    }
    meanPsiSq /= float(max(psiCount, 1u));
    float meNew = ctx.scalars->modelErrEMA + (meanPsiSq - ctx.scalars->modelErrEMA) * invT;
    ctx.scalars->modelErrEMA = nf_safe(meNew);
    // Slow EMA tracks at half the rate so lpmRate = max(0, prev − cur) is positive when the
    // model is improving (Hou-An-Du 2509.25438 Information-Gain proxy).
    float mePrevNew = ctx.scalars->modelErrEMAPrev
                    + (ctx.scalars->modelErrEMA - ctx.scalars->modelErrEMAPrev) * (invT * 0.5f);
    ctx.scalars->modelErrEMAPrev = nf_safe(mePrevNew);

    // Effective-rank via trace/max of featSM (stable-rank proxy; He et al. 2509.22335).
    float tr = 0.0f, mx = 0.0f;
    for (uint i = 0u; i < R; ++i) {
        float v = nf_safe(ctx.featSM[i]);
        tr += v;
        mx = max(mx, v);
    }
    ctx.scalars->pEff = nf_safe(tr / max(mx, FLT_MIN));

    // Telemetry roll-up. surface KernelScalars + cumulative counters
    // into KernelStats for host-side bench JSON consumption.
    float uF = 0.0f, uO = 0.0f;
    for (uint i = 0u; i < R; ++i) uF += ctx.utilFeat[i];
    for (uint i = 0u; i < ctx.K; ++i) uO += ctx.utilOpt[i];
    stats->rewardRate         = ctx.scalars->rho;
    stats->activeCount        = float(ctx.scalars->activeCount);
    stats->utilityFeatureMean = uF / float(max(R, 1u));
    stats->utilityOptionMean  = uO / float(max(ctx.K, 1u));
    stats->oakOptionReinitCount = float(popcount(ctx.counters[OAK_CNT_REINIT_OPT]));
    // surface KernelScalars (consumer-facing telemetry).
    stats->pEffectiveRank   = ctx.scalars->pEff;
    stats->modelErrEMA      = ctx.scalars->modelErrEMA;
    stats->modelErrEMAPrev  = ctx.scalars->modelErrEMAPrev;
    stats->tdBaselineStat   = ctx.scalars->tdBaseline;
    stats->alphaImpStat     = ctx.scalars->alphaImp;
    stats->alphaEncStat     = ctx.scalars->alphaEnc;
    stats->alphaPiStat      = ctx.scalars->alphaPi;
    // surface cumulative encoder-feature reinit count (was in
    // counters[OAK_CNT_REINIT_FEAT] but never read by the host).
    stats->oakFeatureReinitCount = float(ctx.counters[OAK_CNT_REINIT_FEAT]);

    // Slot toggles + step + clear option-reinit bitmask for next step.
    ctx.scalars->stateSlot  = 1u - ctx.scalars->stateSlot;
    ctx.scalars->optionSlot = 1u - ctx.scalars->optionSlot;
    ctx.scalars->stepCount += 1u;
    ctx.counters[OAK_CNT_REINIT_OPT] = 0u;
}

// =================================================================================================
// bk_weight_magnitudes — Per-buffer max(|w|) telemetry surfaced into KernelStats so
// host-side health checks can monitor weight magnitudes per training-progress signal.
// 8 threadgroups × 32 threads. Each TG stride-loops over one fp32 weight buffer with
// simd_max reduction. Cost ≪ 1 ms even on the largest buffer.
// =================================================================================================
kernel void oak_bk_weight_magnitudes(
    device float  *params     [[buffer(0)]],
    device float  *stateArena [[buffer(1)]],
    device uint   *uintArena  [[buffer(2)]],
    device KernelScalars *scalars [[buffer(3)]],
    device KernelStats   *stats   [[buffer(4)]],
    constant KernelConfig &cfg    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgW  [[threads_per_threadgroup]])
{
    Ctx ctx = make_ctx(params, stateArena, uintArena, scalars, cfg);
    device const float *base;
    uint count;
    switch (tgid) {
        case 0u: base = ctx.encoder;     count = ctx.R;                  break;
        case 1u: base = ctx.valueW;      count = ctx.E;                  break;
        case 2u: base = ctx.psi;         count = ctx.K * ctx.M;          break;
        case 3u: base = ctx.optPolicy;   count = ctx.K * ctx.D * ctx.E;  break;
        case 4u: base = ctx.outcomeHead; count = ctx.K * ctx.E;          break;
        case 5u: base = ctx.betaW;       count = ctx.E + ctx.K;          break;
        case 6u: base = ctx.omegaW;      count = ctx.K * ctx.E;          break;
        case 7u: base = ctx.sprDecoder;  count = ctx.K * ctx.R;          break;
        default: return;
    }
    float m_l = 0.0f;
    for (uint i = tid; i < count; i += tgW) {
        m_l = max(m_l, fabs(float(base[i])));
    }
    float m = simd_max(m_l);
    if (tid == 0u) {
        switch (tgid) {
            case 0u: stats->encoderMax     = m; break;
            case 1u: stats->valueWMax      = m; break;
            case 2u: stats->psiMax         = m; break;
            case 3u: stats->optPolicyMax   = m; break;
            case 4u: stats->outcomeHeadMax = m; break;
            case 5u: stats->betaWMax       = m; break;
            case 6u: stats->omegaWMax      = m; break;
            case 7u: stats->sprDecoderMax  = m; break;
        }
    }
}
