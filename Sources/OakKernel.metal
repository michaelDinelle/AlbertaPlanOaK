#pragma METAL fp math_mode(safe)

constant bool OAK_ABLATE_FEATURE_CONSTRUCTION [[function_constant(0)]];
constant bool OAK_ABLATE_GVF_PREDICTION [[function_constant(1)]];
constant bool OAK_ABLATE_RR_SUBTASKS [[function_constant(2)]];
constant bool OAK_ABLATE_OPTION_POLICIES [[function_constant(3)]];
constant bool OAK_ABLATE_OPTION_MODELS [[function_constant(4)]];
constant bool OAK_ABLATE_OPTION_KEYBOARD [[function_constant(6)]];
constant bool OAK_ABLATE_KEYBOARD_TO_POLICY [[function_constant(8)]];
constant bool OAK_ABLATE_SEARCH_CONTROL [[function_constant(9)]];
constant bool OAK_ABLATE_SEARCH_TO_GENERATE_TEST [[function_constant(10)]];
constant bool OAK_ABLATE_GENERATE_AND_TEST [[function_constant(11)]];

constant bool OAK_ABLATE_JOINT_GEOMETRY [[function_constant(27)]];

constant uint OAK_MEGA_G [[function_constant(12)]];

constant uint OAK_SPEC_OBS_DIM [[function_constant(13)]];
constant uint OAK_SPEC_RANK [[function_constant(14)]];
constant uint OAK_SPEC_OPTION_COUNT [[function_constant(15)]];
constant uint OAK_SPEC_ACTION_PARAMS [[function_constant(16)]];
constant uint OAK_SPEC_TRACE_LEVELS [[function_constant(17)]];
constant uint OAK_SPEC_ANCHOR_COUNT [[function_constant(18)]];
constant uint OAK_SPEC_PARAM_STRIDE [[function_constant(19)]];
constant uint OAK_SPEC_STATE_STRIDE [[function_constant(20)]];
constant uint OAK_SPEC_UINT_STRIDE [[function_constant(21)]];
constant float OAK_SPEC_INV_RANK [[function_constant(22)]];
constant float OAK_SPEC_INV_OPTION_COUNT [[function_constant(23)]];
constant float OAK_SPEC_META_ETA [[function_constant(24)]];

constant uint OAK_SPEC_NURSERY [[function_constant(28)]];

constant bool OAK_PROBE_FIXEDIDX_FC [[function_constant(30)]];
constant bool OAK_PROBE_FIXEDIDX = is_function_constant_defined(OAK_PROBE_FIXEDIDX_FC)
    ? OAK_PROBE_FIXEDIDX_FC : false;
inline uint activeIndex(device const uint *act, uint a) {
    return OAK_PROBE_FIXEDIDX ? a : act[a];
}

constant bool OAK_PROBE_FTZ_FC [[function_constant(31)]];
constant bool OAK_PROBE_FTZ = is_function_constant_defined(OAK_PROBE_FTZ_FC)
    ? OAK_PROBE_FTZ_FC : false;
inline float ftz_probe(float value) {
    return (OAK_PROBE_FTZ && fabs(value) < 1.1754944e-38f) ? 0.0f : value;
}

constant uint OAK_MEGA_STAGE_MASK [[function_constant(25)]];
constant uint OAK_MEGA_STAGES = is_function_constant_defined(OAK_MEGA_STAGE_MASK)
    ? OAK_MEGA_STAGE_MASK : 0xffffffffu;

struct KernelConfig {
    uint obsDim, rank, optionCount, actionParamCount;
    uint obsTraceLevels;
    uint linearAnchorCount;
    float invRank, invOptionCount;
    float metaEta;
    uint agentCount, paramStride, stateStride, uintStride;
    uint nurseryCount;
    float sampleSigma;
    float invTwoSigmaSquared;
    float actorAlphaDenom;
};

constant uint OAK_CORE = 32u;
constant uint OAK_GATE_BANK = 7u;
constant uint OAK_GATE_GROUPS = 2u;
constant uint OAK_ACT = OAK_CORE * (1u + OAK_GATE_GROUPS);
constant uint PLAN_INV_SQRT_FEATURES = 2u;
static_assert(OAK_DIAG_RANK == OAK_CORE + OAK_CORE * OAK_GATE_BANK * OAK_GATE_GROUPS,
              "OakDiagRecord rank drifted from the derived feature count");
static_assert(OAK_DIAG_K == 2u * OAK_CORE, "sketch width is two sign words per feature");

constant uint FEATURE_PRODUCT = 0u;
constant uint FEATURE_NORMALIZED_SUM = 1u;
constant uint FEATURE_RESIDUAL_PRODUCT = 2u;
constant uint FEATURE_OBSERVATION_TIMESCALE = 3u;

inline float keepFiniteRaw(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }
#define keepFinite(o, n) ftz_probe(keepFiniteRaw((o), (n)))

inline float track_rate(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

struct Agent {
    uint observationCount, featureCount, optionCount, actionCount, traceLevels, anchorCount, nurseryCount;
    uint envObservationCount, envActionCount;
    uint stepCount, stateSlot;
    float invFeatureCount, invOptionCount, metaEta;
    float sampleSigma, invTwoSigmaSquared, actorAlphaDenom;
    device float *valueWeight, *optionValue, *optionPolicy, *optionReward, *optionDuration;
    device float *optionValueAux, *optionNextValue, *optionNextValueAux;
    device float *observation, *observationMean, *observationSecondMoment;
    device float *observationAlpha, *observationBeta, *observationH, *observationGradEnergy;
    device float *state;
    device float *policyMean, *policyMeanRaw, *actionSignal, *optionActorCredit, *optionPolicyMean;
    device float *importance;
    device float *reward, *td, *optionTd;
    device float *featureUtility, *optionUtility, *bonusWeight, *featureMean, *featureSecondMoment;
    device float *valueAlpha, *valueBeta, *valueH;
    device float *valueGradEnergy;
    device float *optionValueAlpha, *optionValueBeta, *optionValueH, *optionValueGradEnergy;
    device float *modelAlpha, *modelBeta, *modelH, *modelGradEnergy;
    device float *actorAdvantage;
    device float *optionRewardTd, *optionDurationTd;
    device float *observationTimescale, *optionPriority, *continuation;
    device float *keyboard;
    device float *rewardCenter;
    device float *optionValueScale, *valueScale, *previousFeatureNormSq;
    device float *optionValueDot, *optionValueAuxScale, *optionAverageReward, *optionModelAverageReward, *optionFitness;
    device float *actorAlpha, *actorBeta, *actorH, *actorGradEnergy;
    device float *importanceStaging, *actorCreditStaging, *sampleScratch;
    device float *predictedReward, *predictedDuration, *predictedNextValue;
    device float *twinDisagreement;
    device float *optionNextValueTd, *optionNextValueDot, *optionNextValueScale, *optionNextValueAuxScale;
    device float *nurseryStats;
    device float *optionLogit, *optionPolicyCoef, *optionUseByFeature, *nurseryScore, *replaceScore;
    device float *stageScratch;
    device float *planScratch;
    device float *optionNextValueTwin;
    device uint *optionFeature, *featureAge, *optionAge;
    device uint *featureParentA, *featureParentB, *featureKind, *featureBirthCount;
    device uint *selectedOption;
    device uint *lastStopAge;
    device uint *nurseryParentA, *nurseryParentB, *nurseryKind, *nurseryAge;
    device uint *activeFeature;
    device uint *replaceSlot;
    device uint *parked;
    device KernelScalars *scalars;
};

inline Agent make_agent(device float *params, device float *stateArena, device uint *uintArena,
                    device KernelScalars *scalars,
                    KernelConfig cfg, uint agentID, uint agentGlobal,
                    uint envObservationCount, uint envActionCount) {
#define functionConstant(fc, fb) (is_function_constant_defined(fc) ? (fc) : (fb))
    uint paramStride = functionConstant(OAK_SPEC_PARAM_STRIDE, cfg.paramStride);
    uint stateStride = functionConstant(OAK_SPEC_STATE_STRIDE, cfg.stateStride);
    uint uintStride = functionConstant(OAK_SPEC_UINT_STRIDE, cfg.uintStride);
    params += agentID * paramStride;
    stateArena += agentID * stateStride;
    uintArena += agentID * uintStride;
    scalars += agentGlobal;
    Agent agent;
    agent.observationCount = functionConstant(OAK_SPEC_OBS_DIM, cfg.obsDim);
    agent.featureCount = functionConstant(OAK_SPEC_RANK, cfg.rank);
    agent.optionCount = functionConstant(OAK_SPEC_OPTION_COUNT, cfg.optionCount);
    agent.actionCount = functionConstant(OAK_SPEC_ACTION_PARAMS, cfg.actionParamCount);
    agent.traceLevels = functionConstant(OAK_SPEC_TRACE_LEVELS, cfg.obsTraceLevels);
    agent.anchorCount = functionConstant(OAK_SPEC_ANCHOR_COUNT, cfg.linearAnchorCount);
    agent.nurseryCount = functionConstant(OAK_SPEC_NURSERY, cfg.nurseryCount);
    agent.envObservationCount = (envObservationCount != 0u) ? envObservationCount : agent.observationCount;
    agent.envActionCount = (envActionCount != 0u) ? envActionCount : max(agent.actionCount / 2u, 1u);
    agent.stepCount = scalars->stepCount; agent.stateSlot = scalars->stateSlot;
    agent.invFeatureCount = functionConstant(OAK_SPEC_INV_RANK, cfg.invRank);
    agent.invOptionCount = functionConstant(OAK_SPEC_INV_OPTION_COUNT, cfg.invOptionCount);
    agent.metaEta = functionConstant(OAK_SPEC_META_ETA, cfg.metaEta);
    agent.sampleSigma = cfg.sampleSigma;
    agent.invTwoSigmaSquared = cfg.invTwoSigmaSquared;
    agent.actorAlphaDenom = cfg.actorAlphaDenom;
    uint paramOffset = 0u;
    agent.valueWeight = params + paramOffset; paramOffset += agent.featureCount;
    agent.optionValue = params + paramOffset; paramOffset += agent.optionCount * agent.featureCount;
    agent.optionPolicy = params + paramOffset; paramOffset += agent.optionCount * max(agent.actionCount / 2u, 1u) * agent.featureCount;
    agent.optionReward = params + paramOffset; paramOffset += agent.optionCount * agent.featureCount;
    agent.optionDuration = params + paramOffset; paramOffset += agent.optionCount * agent.featureCount;
    agent.optionValueAux = params + paramOffset; paramOffset += agent.optionCount * agent.featureCount;
    agent.optionNextValue = params + paramOffset; paramOffset += agent.optionCount * agent.featureCount;
    agent.optionNextValueAux = params + paramOffset; paramOffset += agent.optionCount * agent.featureCount;
    agent.optionNextValueTwin = params + paramOffset;

    uint stateOffset = 0u;
    agent.observation = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.observationMean = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.observationSecondMoment = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.observationAlpha = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.observationBeta = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.observationH = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.observationGradEnergy = stateArena + stateOffset; stateOffset += agent.observationCount;
    agent.state = stateArena + stateOffset; stateOffset += 2u * agent.featureCount;
    agent.policyMean = stateArena + stateOffset; stateOffset += agent.actionCount;
    agent.policyMeanRaw = stateArena + stateOffset; stateOffset += agent.actionCount;
    agent.actionSignal = stateArena + stateOffset; stateOffset += agent.actionCount;
    agent.optionActorCredit = stateArena + stateOffset; stateOffset += agent.optionCount * agent.actionCount;
    agent.optionPolicyMean = stateArena + stateOffset; stateOffset += agent.optionCount * agent.actionCount;
    agent.importance = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.reward = stateArena + stateOffset; stateOffset += 1u;
    agent.td = stateArena + stateOffset; stateOffset += 1u;
    agent.optionTd = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.featureUtility = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.optionUtility = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.bonusWeight = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.featureMean = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.featureSecondMoment = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.valueAlpha = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.valueBeta = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.valueH = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.valueGradEnergy = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.optionValueAlpha = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.optionValueBeta = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.optionValueH = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.optionValueGradEnergy = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.modelAlpha = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.modelBeta = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.modelH = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.modelGradEnergy = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.actorAdvantage = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionRewardTd = stateArena + stateOffset; stateOffset += agent.optionCount;

    agent.optionDurationTd = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.observationTimescale = stateArena + stateOffset; stateOffset += agent.traceLevels * agent.observationCount;
    agent.optionPriority = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.continuation = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.keyboard = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.rewardCenter = stateArena + stateOffset; stateOffset += 1u;
    agent.optionValueScale = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.valueScale = stateArena + stateOffset; stateOffset += 1u;
    agent.previousFeatureNormSq = stateArena + stateOffset; stateOffset += 1u;
    agent.optionValueDot = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionValueAuxScale = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionAverageReward = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionModelAverageReward = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionFitness = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.actorAlpha = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.actorBeta = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.actorH = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.actorGradEnergy = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.importanceStaging = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.actorCreditStaging = stateArena + stateOffset; stateOffset += agent.optionCount * agent.actionCount;
    agent.sampleScratch = stateArena + stateOffset; stateOffset += 8u;
    agent.predictedReward = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.predictedDuration = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.predictedNextValue = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.twinDisagreement = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionNextValueTd = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionNextValueDot = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionNextValueScale = stateArena + stateOffset; stateOffset += agent.optionCount;
    agent.optionNextValueAuxScale = stateArena + stateOffset; stateOffset += agent.optionCount;
    stateOffset += 1u;
    stateOffset = (stateOffset + 3u) & ~3u;

    agent.nurseryStats = stateArena + stateOffset; stateOffset += 4u * agent.nurseryCount;
    agent.optionLogit = stateArena + stateOffset; stateOffset += agent.optionCount * agent.actionCount;
    agent.optionPolicyCoef = stateArena + stateOffset; stateOffset += agent.optionCount * max(agent.actionCount / 2u, 1u);
    agent.optionUseByFeature = stateArena + stateOffset; stateOffset += agent.featureCount;
    agent.nurseryScore = stateArena + stateOffset; stateOffset += agent.nurseryCount;
    stateOffset += 3u;
    stateOffset &= ~3u;
    agent.replaceScore = stateArena + stateOffset; stateOffset += 4u * agent.featureCount;
    agent.stageScratch = stateArena + stateOffset; stateOffset += 16u;

    agent.planScratch = stateArena + stateOffset; stateOffset += 4u;

    uint uintOffset = 0u;
    agent.optionFeature = uintArena + uintOffset; uintOffset += agent.optionCount;
    agent.featureAge = uintArena + uintOffset; uintOffset += agent.featureCount;
    agent.optionAge = uintArena + uintOffset; uintOffset += agent.optionCount;
    agent.featureParentA = uintArena + uintOffset; uintOffset += agent.featureCount;
    agent.featureParentB = uintArena + uintOffset; uintOffset += agent.featureCount;
    agent.featureKind = uintArena + uintOffset; uintOffset += agent.featureCount;
    agent.featureBirthCount = uintArena + uintOffset; uintOffset += agent.featureCount;
    agent.selectedOption = uintArena + uintOffset; uintOffset += 1u;
    agent.lastStopAge = uintArena + uintOffset; uintOffset += agent.optionCount;
    agent.nurseryParentA = uintArena + uintOffset; uintOffset += agent.nurseryCount;
    agent.nurseryParentB = uintArena + uintOffset; uintOffset += agent.nurseryCount;
    agent.nurseryKind = uintArena + uintOffset; uintOffset += agent.nurseryCount;
    agent.nurseryAge = uintArena + uintOffset; uintOffset += agent.nurseryCount;
    agent.activeFeature = uintArena + uintOffset; uintOffset += 2u * OAK_ACT;
    agent.replaceSlot = uintArena + uintOffset; uintOffset += 4u;
    agent.parked = uintArena + uintOffset; uintOffset += agent.optionCount;
    agent.scalars = scalars;
    return agent;
}

inline device float *state_at(Agent agent, uint slot) { return agent.state + slot * agent.featureCount; }

constant uint OAK_SPEC_ENV_KIND [[function_constant(29)]];

inline uint envKind(uint stored) {
    return is_function_constant_defined(OAK_SPEC_ENV_KIND) ? OAK_SPEC_ENV_KIND : stored;
}

inline KernelConfig deriveConfig(uint obsDim, uint actionParams) {
    KernelConfig cfgOut;
    cfgOut.obsDim = obsDim;
    cfgOut.actionParamCount = actionParams;
    cfgOut.rank = OAK_CORE + OAK_CORE * OAK_GATE_BANK * 2u;
    cfgOut.optionCount = cfgOut.linearAnchorCount = OAK_CORE;
    cfgOut.obsTraceLevels = cfgOut.rank / (2u * OAK_CORE);
    cfgOut.nurseryCount = cfgOut.rank;
    uint featureTimeScale = OAK_CORE + OAK_CORE * OAK_GATE_BANK;
    cfgOut.invRank = 1.0f / float(featureTimeScale);
    cfgOut.invOptionCount = 1.0f / float(cfgOut.optionCount);
    cfgOut.metaEta = 1.0f / float(featureTimeScale) / float(max(featureTimeScale + cfgOut.actionParamCount, 1u));
    cfgOut.sampleSigma = 1.0f / float(max(cfgOut.actionParamCount, 1u));
    cfgOut.invTwoSigmaSquared = 1.0f / (2.0f * cfgOut.sampleSigma * cfgOut.sampleSigma);
    cfgOut.actorAlphaDenom = float(max(cfgOut.actionParamCount * cfgOut.optionCount, 1u));
    cfgOut.agentCount = cfgOut.paramStride = cfgOut.stateStride = cfgOut.uintStride = 0u;
    return cfgOut;
}

struct OakShapeProbe {
    uint rank, optionCount, nursery, traceLevels, anchorCount;
    uint paramStride, stateStride, uintStride;
    uint pPsi, pOptPolicy, stUtilOpt;
    uint stAlpha, stBeta, stAlphaPsi, stBetaPsi, stAlphaModel, stBetaModel;
    uint stAlphaAct, stBetaAct, uOptFeature, uParentA, uParentB, uKind, uNurParentA;
    float invRank, invOptionCount, metaEta, sampleSigma, invTwoSigmaSquared, actorAlphaDenom;
    uint stKeep0, stKeep0n, stKeep1, stKeep1n, uKeep0, uKeep0n;
};
static_assert(sizeof(OakShapeProbe) == 144, "OakShapeProbe layout drifted from the Swift mirror");

kernel void oak_shape_probe(device OakShapeProbe *out [[buffer(0)]],
                            device float *params [[buffer(1)]],
                            device float *stateArena [[buffer(2)]],
                            device uint *uintArena [[buffer(3)]],
                            device KernelScalars *scalars [[buffer(4)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    uint obsDim = functionConstant(OAK_SPEC_OBS_DIM, 0u);
    uint actionParams = functionConstant(OAK_SPEC_ACTION_PARAMS, 0u);
    KernelConfig cfg = deriveConfig(obsDim, actionParams);
    Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg, 0u, 0u,
                     cfg.obsDim, cfg.actionParamCount / 2u);
    out->rank = agent.featureCount;
    out->optionCount = agent.optionCount;
    out->nursery = agent.nurseryCount;
    out->traceLevels = agent.traceLevels;
    out->anchorCount = agent.anchorCount;
    out->paramStride = uint(agent.optionNextValueTwin - params) + agent.optionCount * agent.featureCount;
    out->stateStride = ((uint(agent.planScratch - stateArena) + 4u) + 3u) & ~3u;
    out->uintStride = uint(agent.parked - uintArena) + agent.optionCount;
    out->pPsi = uint(agent.optionValue - params);
    out->pOptPolicy = uint(agent.optionPolicy - params);
    out->stUtilOpt = uint(agent.optionUtility - stateArena);
    out->stAlpha = uint(agent.valueAlpha - stateArena);
    out->stBeta = uint(agent.valueBeta - stateArena);
    out->stAlphaPsi = uint(agent.optionValueAlpha - stateArena);
    out->stBetaPsi = uint(agent.optionValueBeta - stateArena);
    out->stAlphaModel = uint(agent.modelAlpha - stateArena);
    out->stBetaModel = uint(agent.modelBeta - stateArena);
    out->stAlphaAct = uint(agent.actorAlpha - stateArena);
    out->stBetaAct = uint(agent.actorBeta - stateArena);
    out->uOptFeature = uint(agent.optionFeature - uintArena);
    out->uParentA = uint(agent.featureParentA - uintArena);
    out->uParentB = uint(agent.featureParentB - uintArena);
    out->uKind = uint(agent.featureKind - uintArena);
    out->uNurParentA = uint(agent.nurseryParentA - uintArena);
    out->invRank = cfg.invRank;
    out->invOptionCount = cfg.invOptionCount;
    out->metaEta = cfg.metaEta;
    out->sampleSigma = cfg.sampleSigma;
    out->invTwoSigmaSquared = cfg.invTwoSigmaSquared;
    out->actorAlphaDenom = cfg.actorAlphaDenom;
    out->stKeep0 = 0u;
    out->stKeep0n = uint(agent.sampleScratch - stateArena);
    out->stKeep1 = uint(agent.nurseryStats - stateArena);
    out->stKeep1n = 4u * agent.nurseryCount;
    out->uKeep0 = 0u;
    out->uKeep0n = uint(agent.parked - uintArena) + agent.optionCount;
}

constant uint OAK_ENV_TABLE_STRIDE = 7u + OAK_POSE_LEN;

kernel void oak_env_table_probe(device uint *out [[buffer(0)]],
                                uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    out[0] = OAK_ENV_COUNT;
    for (uint i = 0u; i < OAK_ENV_COUNT; ++i) {
        device uint *row = out + 1u + i * OAK_ENV_TABLE_STRIDE;
        row[0] = OAK_ENV_SPECS[i].nameHash;
        row[1] = OAK_ENV_SPECS[i].obsDimNatural;
        row[2] = OAK_ENV_SPECS[i].actionKind;
        row[3] = OAK_ENV_SPECS[i].actionCount;
        row[4] = OAK_ENV_SPECS[i].gridW;
        row[5] = OAK_ENV_SPECS[i].gridH;
        row[6] = OAK_ENV_SPECS[i].controlHz;
        for (uint s = 0u; s < OAK_POSE_LEN; ++s) row[7u + s] = OAK_ENV_SPECS[i].slots[s];
    }
}

inline device uint *activeFeatures(Agent agent, uint slot) { return agent.activeFeature + slot * OAK_ACT; }

inline uint actionMeanCount(Agent agent) { return max(agent.actionCount / 2u, 1u); }

inline uint featureTile(float activation) {
    if (!isfinite(activation)) return 3u;
    return uint(clamp(floor(activation + 3.5f), 0.0f, 6.0f));
}

inline uint tileSourceIndex(Agent agent, uint source) {
    return (source < agent.envObservationCount) ? source
         : (source < agent.envObservationCount + agent.envActionCount) ? agent.observationCount + (source - agent.envObservationCount)
                                              : agent.observationCount + agent.actionCount;
}

inline uint zoomedFeatureTile(float activation, uint zoomLevel) {
    if (!isfinite(activation)) return 3u;
    float zoom = float(1u << min(zoomLevel, 2u));
    return uint(clamp(floor(activation * zoom + 3.5f), 0.0f, 6.0f));
}

inline uint jointFeatureTile(float activationA, float activationB, uint sourceReuse) {
    if (!isfinite(activationA) || !isfinite(activationB) || fabs(activationA) >= 3.5f || fabs(activationB) >= 3.5f) return 6u;
    activationA += ((sourceReuse & 1u) == 0u) ? (7.0f / 6.0f) : 0.0f;
    uint tileA = uint(clamp(floor((activationA + 3.5f) * (3.0f / 7.0f)), 0.0f, 2.0f));
    uint tileB = uint(clamp(floor((activationB + 3.5f) * (2.0f / 7.0f)), 0.0f, 1.0f));
    return tileA * 2u + tileB;
}

inline bool wasActive(device const uint *previousActive, uint i) {
    return i < OAK_CORE || previousActive[OAK_CORE + (i - OAK_CORE) / OAK_GATE_BANK] == i;
}
inline float heldDuration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }

inline float normalizedObservation(Agent agent, uint observationIndex) {
    float mean = agent.observationMean[observationIndex];
    float secondMoment = agent.observationSecondMoment[observationIndex];
    float var = max(secondMoment - mean * mean, 0.0f);
    return (agent.observation[observationIndex] - mean) * rsqrt(var + 1.0f / float(2u * OAK_CORE));
}

inline void metaOptimizeUpdate(device float *alpha,
                                device float *beta,
                                device float *h,
                                device float *energy,
                                uint i,
                                float signal,
                                float x,
                                uint age,
                                float metaEta,
                                float decay) {
    if (!isfinite(signal) || !isfinite(x)) return;
    float oldAlpha = alpha[i];
    float oldH = h[i];
    float grad = signal * x;
    float metaGradient = grad * oldH;
    float gradEnergyRate = 1.0f / (float(age) + 1.0f);
    float oldGradEnergy = energy[i];
    float newGradEnergy = oldGradEnergy + gradEnergyRate * (metaGradient * metaGradient - oldGradEnergy);
    energy[i] = keepFinite(oldGradEnergy, newGradEnergy);
    float ridge = 1.0f / (float(age) + 1.0f);
    float normalizedMetaGradient = metaGradient * rsqrt(max(energy[i], 0.0f) + ridge);
    float oldBeta = beta[i];
    float newBeta = oldBeta + metaEta * normalizedMetaGradient;

    beta[i] = keepFinite(oldBeta, clamp(newBeta, 2.0f * log(decay), 0.0f));
    alpha[i] = keepFinite(oldAlpha, exp(beta[i]));
    float featureCurvature = x * x;

    float hDecay = (1.0f - decay) * max(0.0f, 1.0f - oldAlpha * featureCurvature);
    h[i] = keepFinite(oldH, hDecay * oldH + oldAlpha * grad);
}

inline void storeLearnScales(Agent agent, uint tid) {

    device const uint *previousActive = activeFeatures(agent, agent.stateSlot);
    device const float *previousFeatures = state_at(agent, agent.stateSlot);
    float valueStepDenominator = 1.0f, previousFeatureNormSq = 0.0f;
    for (uint base = 0u; base < OAK_ACT; base += OAK_CORE) {
        uint a = base + tid;
        uint ip = previousActive[a];
        float previousFeature = previousFeatures[ip];
        float previousAlpha = agent.valueAlpha[ip];
        for (uint t = 0u; t < OAK_CORE; ++t) {
            float x = simd_shuffle(previousFeature, (ushort)t);
            float previousAlphaBroadcast = simd_shuffle(previousAlpha, (ushort)t);
            if (isfinite(previousAlphaBroadcast) && previousAlphaBroadcast > 0.0f) valueStepDenominator += previousAlphaBroadcast * (x * x);
            previousFeatureNormSq = fma(x, x, previousFeatureNormSq);
        }
    }
    if (tid == 0u) {
        agent.valueScale[0] = (isfinite(valueStepDenominator) && valueStepDenominator > 0.0f) ? (1.0f / valueStepDenominator) : 0.0f;
        agent.previousFeatureNormSq[0] = previousFeatureNormSq;
    }
}

inline void predictAndControl(Agent agent, uint tid,
                                      threadgroup float *tgCont,
                                      threadgroup float *tgStash) {

    device float *stageScalars = agent.stageScratch;
    device const float *previousFeatures = state_at(agent, agent.stateSlot);
    device const float *currentFeatures = state_at(agent, 1u - agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, agent.stateSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);
    float previousValue = 0.0f, currentValue = 0.0f;
    {

        float previousValuePartial = 0.0f, currentValuePartial = 0.0f;
        for (uint a = tid; a < OAK_ACT; a += OAK_CORE) {
            uint ip = activeIndex(previousActive, a), ic = activeIndex(currentActive, a);
            previousValuePartial = fma(agent.valueWeight[ip], previousFeatures[ip], previousValuePartial);
            currentValuePartial = fma(agent.valueWeight[ic], currentFeatures[ic], currentValuePartial);
        }
        previousValue = simd_sum(previousValuePartial);
        currentValue = simd_sum(currentValuePartial);
        if (OAK_ABLATE_GVF_PREDICTION) {
            previousValue = 0.0f;
            currentValue = 0.0f;
        }
        if (tid == 0u) {
            stageScalars[1] = currentValue;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float gauge = isfinite(agent.rewardCenter[0]) ? agent.rewardCenter[0] : 0.0f;
    float previousDifferentialValue = previousValue - gauge;
    float currentDifferentialValue = currentValue - gauge;

    float averageRewardOld = agent.scalars->averageReward;
    float centeredReward = agent.reward[0] - averageRewardOld;
    if (tid == 0u) {
        float delta = OAK_ABLATE_GVF_PREDICTION ? 0.0f : (centeredReward + currentDifferentialValue - previousDifferentialValue);
        if (!isfinite(delta)) delta = 0.0f;
        agent.td[0] = delta;
        float averageRewardNew = averageRewardOld + agent.invFeatureCount * delta;
        float averageRewardUpdated = (!OAK_ABLATE_GVF_PREDICTION && isfinite(averageRewardNew)) ? averageRewardNew : averageRewardOld;
        agent.scalars->averageReward = averageRewardUpdated;
    }

    {
        float optionValuePrevEnergyPartial = 0.0f, optionValueCrossEnergyPartial = 0.0f, optionValueNextEnergyPartial = 0.0f;
        float modelPrevEnergyPartial = 0.0f, modelCrossEnergyPartial = 0.0f, modelNextEnergyPartial = 0.0f;
        for (uint a = tid; a < OAK_ACT; a += OAK_CORE) {
            uint ip = activeIndex(previousActive, a), ic = activeIndex(currentActive, a);
            float xp = previousFeatures[ip], xc = currentFeatures[ic];
            float ap = agent.optionValueAlpha[ip];
            if (isfinite(ap) && ap > 0.0f) {
                optionValuePrevEnergyPartial = fma(ap * xp, xp, optionValuePrevEnergyPartial);
                optionValueCrossEnergyPartial = fma(ap * xp, currentFeatures[ip], optionValueCrossEnergyPartial);
            }
            float acps = agent.optionValueAlpha[ic];
            if (isfinite(acps) && acps > 0.0f) optionValueNextEnergyPartial = fma(acps * xc, xc, optionValueNextEnergyPartial);
            float am = agent.modelAlpha[ip];
            if (isfinite(am) && am > 0.0f) {
                modelPrevEnergyPartial = fma(am * xp, xp, modelPrevEnergyPartial);
                modelCrossEnergyPartial = fma(am * xp, currentFeatures[ip], modelCrossEnergyPartial);
            }
            float amc = agent.modelAlpha[ic];
            if (isfinite(amc) && amc > 0.0f) modelNextEnergyPartial = fma(amc * xc, xc, modelNextEnergyPartial);
        }
        stageScalars[5] = simd_sum(optionValuePrevEnergyPartial);
        stageScalars[6] = simd_sum(optionValueCrossEnergyPartial);
        stageScalars[7] = simd_sum(optionValueNextEnergyPartial);
        agent.stageScratch[8] = simd_sum(modelPrevEnergyPartial);
        agent.stageScratch[9] = simd_sum(modelCrossEnergyPartial);
        agent.stageScratch[10] = simd_sum(modelNextEnergyPartial);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < agent.optionCount && agent.parked[tid] != 0u) {
        agent.continuation[tid] = 0.0f;
        agent.optionTd[tid] = 0.0f;
        agent.keyboard[tid] = 0.0f;
        tgCont[tid] = 0.0f;
    } else if (tid < agent.optionCount) {
        uint optionFeatureIndex = agent.optionFeature[tid] % agent.featureCount;
        float optionValuePrev = 0.0f, optionValueNext = 0.0f, optionValueAuxDot = 0.0f;
        float rewardHat = 0.0f, durationHat = 0.0f, nextValueHat = 0.0f, twinNextValueHat = 0.0f;
        float previousNextValue = 0.0f, previousNextValueAuxDot = 0.0f, previousRewardHat = 0.0f, previousDurationHat = 0.0f;
        for (uint a = 0u; a < OAK_ACT; ++a) {
            uint ip = activeIndex(previousActive, a), ic = activeIndex(currentActive, a);
            float xp = previousFeatures[ip], xc = currentFeatures[ic];
            uint idxP = ip * agent.optionCount + tid, idxC = ic * agent.optionCount + tid;
            optionValuePrev = fma(agent.optionValue[idxP], xp, optionValuePrev);
            optionValueNext = fma(agent.optionValue[idxC], xc, optionValueNext);
            optionValueAuxDot = fma(agent.optionValueAux[idxP], xp, optionValueAuxDot);
            if (!OAK_ABLATE_OPTION_MODELS) {
                rewardHat = fma(agent.optionReward[idxC], xc, rewardHat);
                durationHat = fma(agent.optionDuration[idxC], xc, durationHat);
                nextValueHat = fma(agent.optionNextValue[idxC], xc, nextValueHat);
                twinNextValueHat = fma(agent.optionNextValueTwin[idxC], xc, twinNextValueHat);
            }
            previousNextValue = fma(agent.optionNextValue[idxP], xp, previousNextValue);
            previousNextValueAuxDot = fma(agent.optionNextValueAux[idxP], xp, previousNextValueAuxDot);
            previousRewardHat = fma(agent.optionReward[idxP], xp, previousRewardHat);
            previousDurationHat = fma(agent.optionDuration[idxP], xp, previousDurationHat);
        }
        tgStash[tid * 4u + 0u] = previousNextValue;
        tgStash[tid * 4u + 1u] = previousNextValueAuxDot;
        tgStash[tid * 4u + 2u] = previousRewardHat;
        tgStash[tid * 4u + 3u] = previousDurationHat;

        float optionValuePrevEnergy = stageScalars[5], optionValueCrossEnergy = stageScalars[6], optionValueNextEnergy = stageScalars[7];

        float stoppingBonus = agent.bonusWeight[optionFeatureIndex] - agent.valueWeight[optionFeatureIndex];
        if (!isfinite(stoppingBonus) || stoppingBonus < 0.0f) stoppingBonus = 0.0f;
        float stoppingValue = currentDifferentialValue + (1.0f + stoppingBonus) * currentFeatures[optionFeatureIndex];
        bool subtaskFinite = isfinite(optionValuePrev) && isfinite(optionValueNext) && isfinite(stoppingValue);


        float stop = (isfinite(optionValueNext) && isfinite(stoppingValue) && stoppingValue >= optionValueNext) ? 1.0f : 0.0f;

        if (tid == 0u) stop = 0.0f;
        if (OAK_ABLATE_RR_SUBTASKS) {
            subtaskFinite = true;
            stop = (tid == 0u) ? 0.0f : 1.0f;
        }
        float continuation = 1.0f - stop;
        agent.continuation[tid] = continuation;
        if (stop > 0.0f) agent.lastStopAge[tid] = 0u;

        float optionAverageRewardOld = agent.optionAverageReward[tid];
        if (!isfinite(optionAverageRewardOld)) optionAverageRewardOld = 0.0f;
        float optionTdError = (!OAK_ABLATE_RR_SUBTASKS && subtaskFinite)
            ? (centeredReward - optionAverageRewardOld + continuation * optionValueNext + stop * stoppingValue - optionValuePrev) : 0.0f;
        if (!isfinite(optionTdError)) {
            optionTdError = 0.0f;
        }

        if (tid == 0u) optionTdError = agent.td[0] - optionAverageRewardOld;
        agent.optionTd[tid] = optionTdError;
        {
            float optionAverageRewardNew = optionAverageRewardOld + agent.invFeatureCount * optionTdError;
            agent.optionAverageReward[tid] = keepFinite(optionAverageRewardOld, optionAverageRewardNew);
        }
        float importanceRatio = agent.importance[tid];
        if (!isfinite(importanceRatio) || importanceRatio < 0.0f) importanceRatio = 0.0f;
        {

            float previousOptionFitness = agent.optionFitness[tid];
            if (!isfinite(previousOptionFitness)) previousOptionFitness = 0.0f;
            float observedReward = agent.reward[0];
            if (!isfinite(observedReward)) observedReward = 0.0f;
            agent.optionFitness[tid] = keepFinite(previousOptionFitness, previousOptionFitness + agent.invFeatureCount * importanceRatio * (observedReward - previousOptionFitness));
        }

        float optionValueBootstrapEnergy = optionValuePrevEnergy - 2.0f * continuation * optionValueCrossEnergy + continuation * continuation * optionValueNextEnergy;
        float denom = fma(importanceRatio, optionValueBootstrapEnergy, 1.0f);

        float optionValueStepDenom = (isfinite(denom) && denom > 0.0f) ? denom : 0.0f;
        agent.optionValueScale[tid] = (isfinite(optionValueStepDenom) && optionValueStepDenom > 0.0f) ? (1.0f / optionValueStepDenom) : 0.0f;
        {
            agent.optionValueDot[tid] = isfinite(optionValueAuxDot) ? optionValueAuxDot : 0.0f;

            float optionValueAuxDenom = fma(importanceRatio, optionValuePrevEnergy, 1.0f);
            agent.optionValueAuxScale[tid] = (isfinite(optionValueAuxDenom) && optionValueAuxDenom > 0.0f)
                ? (1.0f / optionValueAuxDenom) : 0.0f;
        }
        tgCont[tid] = continuation;
        agent.predictedReward[tid] = rewardHat;
        agent.predictedDuration[tid] = durationHat;
        agent.predictedNextValue[tid] = nextValueHat;
        float twinAbsDiff = fabs(nextValueHat - twinNextValueHat);
        agent.twinDisagreement[tid] = isfinite(twinAbsDiff) ? twinAbsDiff : 0.0f;

        if (tid == 0u) nextValueHat = currentValue;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float gaugeRate = track_rate(agent.stepCount, agent.featureCount);
        float gaugeNew = agent.rewardCenter[0] + gaugeRate * (currentValue - agent.rewardCenter[0]);
        agent.rewardCenter[0] = keepFinite(agent.rewardCenter[0], gaugeNew);
    }
    if (tid < agent.optionCount) {

        float advantage = OAK_ABLATE_OPTION_POLICIES ? 0.0f : agent.optionTd[tid];
        if (!isfinite(advantage)) advantage = 0.0f;
        agent.actorAdvantage[tid] = advantage;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    storeLearnScales(agent, tid);
    uint prevSel = agent.selectedOption[0] % agent.optionCount;
    bool continuing = (prevSel != 0u) && tgCont[prevSel] > 0.0f;
    uint selected = continuing ? prevSel : 0u;
    float previousKeyboard = 0.0f, keyboardTwinDisagreement = 0.0f, keyboardDuration = 0.0f;
    if (tid < agent.optionCount) {
        previousKeyboard = agent.keyboard[tid];
        keyboardTwinDisagreement = agent.twinDisagreement[tid];
        keyboardDuration = agent.predictedDuration[tid];
    }
    float previousKeyboardMass = 0.0f;
    for (uint t = 0u; t < agent.optionCount; ++t) {
        float w = simd_shuffle(previousKeyboard, (ushort)t);
        if (isfinite(w) && w > 0.0f) previousKeyboardMass += w;
    }
    if (OAK_ABLATE_OPTION_KEYBOARD) {
        if (tid < agent.optionCount) agent.keyboard[tid] = 0.0f;
        if (tid == 0u) agent.selectedOption[0] = selected;
        return;
    }

    float keyboardWeight = 0.0f;
    bool refreshKeyboard = (!continuing && prevSel != 0u) || previousKeyboardMass <= 0.0f;
    if (refreshKeyboard) {
        float bestScore = -FLT_MAX;
        uint probe = 0u;
        for (uint t = 0u; t < agent.optionCount; ++t) {
            float twinDis = simd_shuffle(keyboardTwinDisagreement, (ushort)t);
            float predictedDur = heldDuration(simd_shuffle(keyboardDuration, (ushort)t));
            float probeScore = twinDis * predictedDur;
            if (isfinite(probeScore) && probeScore > bestScore) { bestScore = probeScore; probe = t; }
        }
        keyboardWeight = (tid == probe) ? 1.0f : 0.0f;
        if (tid < agent.optionCount) agent.keyboard[tid] = keyboardWeight;
    }

    if (tid == 0u) agent.selectedOption[0] = selected;
}

inline void constructFeatures(Agent agent,
                                uint tid,
                                uint tgW) {
    if (!OAK_ABLATE_FEATURE_CONSTRUCTION) {
        for (uint observationIndex = tid; observationIndex < agent.observationCount; observationIndex += tgW) {
            float x = agent.observation[observationIndex];
            if (!isfinite(x)) continue;
            float observationMean = agent.observationMean[observationIndex];
            metaOptimizeUpdate(agent.observationAlpha, agent.observationBeta, agent.observationH, agent.observationGradEnergy,
                                observationIndex, x - observationMean, 1.0f, min(agent.stepCount, agent.featureCount), agent.metaEta,
                                agent.invFeatureCount);
            float obsRate = max(1.0f / (float(agent.stepCount) + 1.0f), agent.observationAlpha[observationIndex]);
            agent.observationMean[observationIndex] += obsRate * (x - observationMean);
            agent.observationSecondMoment[observationIndex] += obsRate * (x * x - agent.observationSecondMoment[observationIndex]);
            float standardized = normalizedObservation(agent, observationIndex);
            for (uint m = 1u; m <= agent.traceLevels; ++m) {
                uint slot = (m - 1u) * agent.observationCount + observationIndex;
                float rate = 1.0f / float(1u << m);
                agent.observationTimescale[slot] += rate * (standardized - agent.observationTimescale[slot]);
            }
        }
    }
    for (uint k = tid; k < agent.optionCount; k += OAK_CORE) agent.importance[k] = agent.importanceStaging[k];
    for (uint j = tid; j < agent.optionCount * agent.actionCount; j += tgW)
        agent.optionActorCredit[j] = agent.actorCreditStaging[j];

    device const float *prev = state_at(agent, agent.stateSlot);
    device float *curr = state_at(agent, 1u - agent.stateSlot);
    device uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);

    simdgroup_barrier(mem_flags::mem_device);

    {
        uint i = tid;
        float x;
        if (OAK_ABLATE_FEATURE_CONSTRUCTION) {
            x = (i == 0u) ? 1.0f : 0.0f;
        } else if (i < agent.observationCount) {
            x = normalizedObservation(agent, i);
        } else if (i < agent.observationCount + agent.actionCount) {
            x = agent.actionSignal[i - agent.observationCount];
        } else if (i == agent.observationCount + agent.actionCount) {
            x = agent.reward[0];
        } else if (i == agent.observationCount + agent.actionCount + 1u) {
            x = 1.0f;
        } else {
            uint j = i - (agent.observationCount + agent.actionCount + 2u);
            uint lvl = j / max(agent.envObservationCount, 1u);
            x = (lvl < agent.traceLevels)
                ? agent.observationTimescale[lvl * agent.observationCount + (j - lvl * agent.envObservationCount)]
                : 0.0f;
        }
        if (!isfinite(x)) x = 0.0f;
        curr[i] = x;
        currentActive[tid] = i;
        if (agent.featureAge[i] < 0xffffffffu && !OAK_ABLATE_FEATURE_CONSTRUCTION)
            agent.featureAge[i] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);

    uint tile;
    uint tile2;
    {
        uint sourceCount = agent.envObservationCount + agent.envActionCount + 1u;
        uint sourceIndex = tid % sourceCount, occupancy = tid / sourceCount;
        if (occupancy == 0u || OAK_ABLATE_JOINT_GEOMETRY) {
            tile = featureTile(curr[tileSourceIndex(agent, sourceIndex)]);
        } else if ((tid & 1u) == 0u) {
            tile = zoomedFeatureTile(curr[tileSourceIndex(agent, sourceIndex)], occupancy);
        } else {
            uint channelA = tileSourceIndex(agent, sourceIndex);
            uint channelB = (sourceIndex + occupancy) % agent.envObservationCount;
            tile = jointFeatureTile(curr[channelA], curr[channelB], occupancy);
        }
        tile2 = zoomedFeatureTile(curr[tileSourceIndex(agent, sourceIndex)], occupancy + 1u);
    }
    for (uint g = 0u; g < OAK_GATE_GROUPS; ++g) {
    uint groupTile = (g == 0u) ? tile : tile2;
    uint bankLo = OAK_CORE + (tid + g * OAK_CORE) * OAK_GATE_BANK, bankHi = bankLo + OAK_GATE_BANK;
    uint winner = bankLo + groupTile;
    float winningValue = 0.0f;
    for (uint i = bankLo; i < bankHi; ++i) {
        float x = 0.0f;
        if (!OAK_ABLATE_FEATURE_CONSTRUCTION) {
            uint parentA = agent.featureParentA[i] % agent.featureCount;
            uint parentB = agent.featureParentB[i] % agent.featureCount;
            uint constructor = agent.featureKind[i] % 4u;
            float constructed;
            if (constructor == FEATURE_OBSERVATION_TIMESCALE) {

                constructed = agent.observationTimescale[agent.featureParentA[i] % (agent.traceLevels * agent.observationCount)];
            } else {
                float a = prev[parentA], b = prev[parentB];
                constructed = (constructor == FEATURE_PRODUCT) ? a * b
                    : (constructor == FEATURE_NORMALIZED_SUM) ? (a + b) * 0.7071067811865475f
                    : (constructor == FEATURE_RESIDUAL_PRODUCT) ? a * (b - a)
                    : a * (b - a);
            }
            uint age = agent.featureAge[i];
            float rate = max(1.0f / (float(age) + 1.0f), agent.invFeatureCount);
            agent.featureMean[i] += rate * (constructed - agent.featureMean[i]);
            agent.featureSecondMoment[i] += rate * (constructed * constructed - agent.featureSecondMoment[i]);
            float var = max(agent.featureSecondMoment[i] - agent.featureMean[i] * agent.featureMean[i], 0.0f);

            float ridge = max(1.0f / (float(age) + 1.0f), 1.0f / float(2u * OAK_CORE));
            x = (constructed - agent.featureMean[i]) * rsqrt(var + ridge);
            if (!isfinite(x)) x = 0.0f;
        }
        curr[i] = 0.0f;
        if (i == winner) winningValue = x;
    }
    curr[winner] = winningValue;
    currentActive[OAK_CORE + tid + g * OAK_CORE] = winner;
    if (agent.featureAge[winner] < 0xffffffffu && !OAK_ABLATE_FEATURE_CONSTRUCTION)
        agent.featureAge[winner] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);
}

static inline void encodeObservation(Agent agent, uint tid, uint tgW,
                                    threadgroup float *tgCont,
                                    threadgroup float *tgStash) {
    constructFeatures(agent, tid, tgW);
    predictAndControl(agent, tid, tgCont, tgStash);
}

static inline void planOption(Agent agent, uint k,
                                threadgroup const float *stash,
                                bool stashDirty, uint pSlot) {
    float continuation = agent.continuation[k];
    if (OAK_ABLATE_OPTION_MODELS) {
        agent.optionRewardTd[k] = 0.0f;
        agent.optionPriority[k] = 0.0f;
        agent.optionNextValueTd[k] = 0.0f;
        agent.optionNextValueDot[k] = 0.0f;
        agent.optionNextValueScale[k] = 0.0f;
        agent.optionNextValueAuxScale[k] = 0.0f;
        return;
    }

        float previousNextValueDot, optionNextValueDot, previousRewardHat, previousDurationHat;
        if (!stashDirty) {
            previousNextValueDot = stash[k * 4u + 0u];
            optionNextValueDot = stash[k * 4u + 1u];
            previousRewardHat = stash[k * 4u + 2u];
            previousDurationHat = stash[k * 4u + 3u];
        } else {
            device const uint *previousActive = activeFeatures(agent, pSlot);
            device const float *prev = state_at(agent, pSlot);
            previousNextValueDot = 0.0f; optionNextValueDot = 0.0f; previousRewardHat = 0.0f; previousDurationHat = 0.0f;
            for (uint a = 0u; a < OAK_ACT; ++a) {
                uint ip = activeIndex(previousActive, a);
                uint idxP = ip * agent.optionCount + k;
                float xp = prev[ip];
                previousNextValueDot = fma(agent.optionNextValue[idxP], xp, previousNextValueDot);
                optionNextValueDot = fma(agent.optionNextValueAux[idxP], xp, optionNextValueDot);
                previousRewardHat = fma(agent.optionReward[idxP], xp, previousRewardHat);
                previousDurationHat = fma(agent.optionDuration[idxP], xp, previousDurationHat);
            }
        }

        float rewardHat = agent.predictedReward[k], durationHat = agent.predictedDuration[k], currentNextValueHat = agent.predictedNextValue[k];
        float currentValue = agent.stageScratch[1];

        float modelPrevEnergy = agent.stageScratch[8];
        float modelBootstrapNorm = agent.stageScratch[8] - 2.0f * continuation * agent.stageScratch[9]
                       + continuation * continuation * agent.stageScratch[10];
        modelBootstrapNorm = max(modelBootstrapNorm, 0.0f);
        if (!isfinite(modelBootstrapNorm)) modelBootstrapNorm = 0.0f;
        if (!isfinite(modelPrevEnergy) || modelPrevEnergy < 0.0f) modelPrevEnergy = 0.0f;

        float modelAverageRewardOld = agent.optionModelAverageReward[k];
        if (!isfinite(modelAverageRewardOld)) modelAverageRewardOld = 0.0f;
        float rewardModelTd = (k == 0u) ? 0.0f : ((agent.planScratch[0] - modelAverageRewardOld) + continuation * rewardHat - previousRewardHat);
        float importanceRatio = agent.importance[k];
        if (!isfinite(importanceRatio) || importanceRatio < 0.0f) importanceRatio = 0.0f;
        if (isfinite(agent.planScratch[0]))
            agent.optionModelAverageReward[k] = modelAverageRewardOld + track_rate(agent.stepCount, agent.featureCount) * importanceRatio * (agent.planScratch[0] - modelAverageRewardOld);
        float durationTd = (k == 0u) ? 0.0f : (1.0f + continuation * durationHat - previousDurationHat);

        float optionNextValueTd = continuation * currentNextValueHat + (1.0f - continuation) * currentValue - previousNextValueDot;
        bool modelFinite = isfinite(rewardModelTd) && isfinite(durationTd) && isfinite(durationHat) && isfinite(optionNextValueTd);
        if (!isfinite(rewardModelTd)) rewardModelTd = 0.0f;
        if (!isfinite(durationTd)) durationTd = 0.0f;
        if (!isfinite(optionNextValueTd)) optionNextValueTd = 0.0f;
        agent.optionRewardTd[k] = rewardModelTd;
        agent.optionDurationTd[k] = durationTd;
        agent.optionNextValueTd[k] = optionNextValueTd;
        agent.optionNextValueDot[k] = isfinite(optionNextValueDot) ? optionNextValueDot : 0.0f;
        float nextValueStepDenom = fma(importanceRatio, modelBootstrapNorm, 1.0f);
        agent.optionNextValueScale[k] = (isfinite(nextValueStepDenom) && nextValueStepDenom > 0.0f) ? (1.0f / nextValueStepDenom) : 0.0f;
        float nextValueAuxDenom = fma(importanceRatio, modelPrevEnergy, 1.0f);
        agent.optionNextValueAuxScale[k] = (isfinite(nextValueAuxDenom) && nextValueAuxDenom > 0.0f) ? (1.0f / nextValueAuxDenom) : 0.0f;
        float modelSquaredError = optionNextValueTd * optionNextValueTd + rewardModelTd * rewardModelTd;
        if (OAK_ABLATE_SEARCH_CONTROL) {
            agent.optionPriority[k] = 0.0f;
        } else if (modelFinite && isfinite(modelSquaredError)) {
            agent.optionPriority[k] = sqrt(max(modelSquaredError, 0.0f));
        } else {
            float debt = fabs(agent.optionPriority[k]);
            if (!isfinite(debt) || debt <= 0.0f) debt = 1.0f;
            agent.optionPriority[k] = -debt;
        }
    }

static inline void planAllOptions(Agent agent, uint tid,
                                      threadgroup const float *stash,
                                      bool stashDirty, uint pSlot) {
    for (uint kk = tid; kk < agent.optionCount; kk += OAK_CORE) {
    if (agent.parked[kk] != 0u) {
        agent.optionRewardTd[kk] = 0.0f;
        agent.optionDurationTd[kk] = 0.0f;
        agent.optionNextValueTd[kk] = 0.0f;
        agent.optionNextValueDot[kk] = 0.0f;
        agent.optionNextValueScale[kk] = 0.0f;
        agent.optionNextValueAuxScale[kk] = 0.0f;
        agent.optionPriority[kk] = 0.0f;
        continue;
    }
    planOption(agent, kk, stash, stashDirty, pSlot);

    {
        uint k = kk;
        float importanceRatio = agent.importance[k];
        if (!isfinite(importanceRatio)) importanceRatio = 0.0f;
        float advantage = agent.actorAdvantage[k];
        if (!isfinite(advantage)) advantage = 0.0f;

        metaOptimizeUpdate(agent.actorAlpha, agent.actorBeta, agent.actorH, agent.actorGradEnergy,
                            k, advantage, 1.0f, agent.optionAge[k], agent.metaEta, agent.invFeatureCount);
        float actorAlpha = agent.actorAlpha[k] / agent.actorAlphaDenom;
        float previousFeatureNormSq = agent.previousFeatureNormSq[0];
        for (uint d = 0u; d < actionMeanCount(agent); ++d) {
            float targetGrad = agent.optionActorCredit[k * agent.actionCount + d];
            if (!isfinite(targetGrad)) targetGrad = 0.0f;
            float gradScale = importanceRatio * targetGrad;

            float denom = 1.0f + actorAlpha * advantage * advantage * gradScale * gradScale * previousFeatureNormSq;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            float policyCoef = actorAlpha * advantage * gradScale * scale;
            agent.optionPolicyCoef[k * actionMeanCount(agent) + d] = isfinite(policyCoef) ? policyCoef : 0.0f;
        }
    }
    }
}

static inline void learnFromTransition(Agent agent, device const float *prev,
                                  device const float *curr, uint tid, uint tw,
                                  uint pSlot) {
    device const uint *previousActive = activeFeatures(agent, pSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - pSlot);
    const uint activeFeatureCount = OAK_ACT;
    uint optionN = agent.optionCount;
    uint policyN = actionMeanCount(agent);
    uint optionPolicyStride = optionN * policyN;

    uint k = tid;
    const bool optionLive = (agent.parked[k] == 0u);
    float importanceRatio = agent.importance[k];
    if (!isfinite(importanceRatio) || importanceRatio < 0.0f) importanceRatio = 0.0f;
    float continuation = agent.continuation[k];
    if (!isfinite(continuation)) continuation = 0.0f;
    float optionValueAuxDot = agent.optionValueDot[k], optionValueStepScale = agent.optionValueScale[k], td = agent.optionTd[k];
    float nextValueAuxDot = agent.optionNextValueDot[k], nextValueStepScale = agent.optionNextValueScale[k], nextValueTdError = agent.optionNextValueTd[k];
    float rewardModelTdError = agent.optionRewardTd[k], durationTdError = agent.optionDurationTd[k];
    float optionValueAuxResidual = td - optionValueAuxDot, optionValueAuxStepScale = agent.optionValueAuxScale[k];
    float nextValueAuxResidual = nextValueTdError - nextValueAuxDot, nextValueAuxStepScale = agent.optionNextValueAuxScale[k];
    float twinPrevDot = 0.0f;
    for (uint a0 = 0u; a0 < activeFeatureCount; ++a0) {
        uint i0 = activeIndex(previousActive, a0);
        twinPrevDot = fma(agent.optionNextValueTwin[i0 * optionN + k], prev[i0], twinPrevDot);
    }
    if (!isfinite(twinPrevDot)) twinPrevDot = 0.0f;
    float twinResidual = rewardModelTdError - twinPrevDot;
    float actorCoefficient[16];
    for (uint j = 0u; j < policyN; ++j) actorCoefficient[j] = agent.optionPolicyCoef[tid * policyN + j];
    for (uint cb = 0u; cb < activeFeatureCount; cb += OAK_CORE) {
        uint aH = cb + tid;
        uint iH = activeIndex(previousActive, aH);
        float xpH = prev[iH], xcH = curr[iH];
        float aPH = agent.optionValueAlpha[iH], aMH = agent.modelAlpha[iH];
        uint lEnd = min(OAK_CORE, activeFeatureCount - cb);
        for (uint l = 0u; l < lEnd; ++l) {
            uint i = simd_shuffle(iH, (ushort)l);
            float xp = simd_shuffle(xpH, (ushort)l);
            float xc = simd_shuffle(xcH, (ushort)l);
            float aP = simd_shuffle(aPH, (ushort)l);
            float aM = simd_shuffle(aMH, (ushort)l);
            if (!optionLive) continue;
            if (!OAK_ABLATE_OPTION_POLICIES) {
                device float *optionPolicyWeights = &agent.optionPolicy[i * optionPolicyStride + tid * policyN];
                for (uint j = 0u; j < policyN; ++j)
                    optionPolicyWeights[j] = keepFinite(optionPolicyWeights[j], optionPolicyWeights[j] + actorCoefficient[j] * xp);
            }
            if (!OAK_ABLATE_RR_SUBTASKS && k != 0u) {
                float optionValueGradient = (continuation > 0.0f) ? (optionValueAuxDot * (xp - continuation * xc)) : (td * xp);
                device float *optionValueWeight = &agent.optionValue[i * optionN + k];
                *optionValueWeight = keepFinite(*optionValueWeight, *optionValueWeight + aP * importanceRatio * optionValueGradient * optionValueStepScale);
                device float *optionValueAuxWeight = &agent.optionValueAux[i * optionN + k];
                *optionValueAuxWeight = keepFinite(*optionValueAuxWeight, *optionValueAuxWeight + aP * importanceRatio * optionValueAuxResidual * xp * optionValueAuxStepScale);
            }
            if (!OAK_ABLATE_OPTION_MODELS) {
                float nextValueGradient = (continuation > 0.0f) ? (nextValueAuxDot * (xp - continuation * xc)) : (nextValueTdError * xp);
                device float *nextValueWeight = &agent.optionNextValue[i * optionN + k];
                *nextValueWeight = keepFinite(*nextValueWeight, *nextValueWeight + aM * importanceRatio * nextValueGradient * nextValueStepScale);
                device float *nextValueAuxWeight = &agent.optionNextValueAux[i * optionN + k];
                *nextValueAuxWeight = keepFinite(*nextValueAuxWeight, *nextValueAuxWeight + aM * importanceRatio * nextValueAuxResidual * xp * nextValueAuxStepScale);
                device float *durationWeight = &agent.optionDuration[i * optionN + k];
                *durationWeight = keepFinite(*durationWeight, *durationWeight + aM * importanceRatio * durationTdError * xp * nextValueStepScale * 0.05f);
                float rewardModelGradient = (continuation > 0.0f) ? (twinPrevDot * (xp - continuation * xc)) : (rewardModelTdError * xp);
                device float *rewardWeight = &agent.optionReward[i * optionN + k];
                *rewardWeight = keepFinite(*rewardWeight, *rewardWeight + aM * importanceRatio * rewardModelGradient * nextValueStepScale);
                device float *twinWeight = &agent.optionNextValueTwin[i * optionN + k];
                *twinWeight = keepFinite(*twinWeight, *twinWeight + aM * importanceRatio * twinResidual * xp * nextValueAuxStepScale);
            }
        }
    }

    {
        uint newFeatureH = activeIndex(currentActive, OAK_CORE + tid);
        uint wasPreviouslyActiveH = wasActive(previousActive, newFeatureH) ? 1u : 0u;
        float newFeatureValueH = curr[newFeatureH];
        float newOptionAlphaH = agent.optionValueAlpha[newFeatureH], newModelAlphaH = agent.modelAlpha[newFeatureH];
        for (uint l = 0u; l < OAK_CORE; ++l) {
            uint i = simd_shuffle(newFeatureH, (ushort)l);
            uint wasPreviouslyActive = simd_shuffle(wasPreviouslyActiveH, (ushort)l);
            float xc = simd_shuffle(newFeatureValueH, (ushort)l);
            float aP = simd_shuffle(newOptionAlphaH, (ushort)l);
            float aM = simd_shuffle(newModelAlphaH, (ushort)l);
            if (!optionLive || wasPreviouslyActive != 0u) continue;
            if (continuation > 0.0f && importanceRatio > 0.0f) {
                if (!OAK_ABLATE_RR_SUBTASKS && k != 0u) {
                    device float *optionValueWeight = &agent.optionValue[i * optionN + k];
                    *optionValueWeight = keepFinite(*optionValueWeight, *optionValueWeight + aP * importanceRatio * (-optionValueAuxDot * continuation * xc) * optionValueStepScale);
                }
                if (!OAK_ABLATE_OPTION_MODELS) {
                    device float *nextValueWeight = &agent.optionNextValue[i * optionN + k];
                    *nextValueWeight = keepFinite(*nextValueWeight, *nextValueWeight + aM * importanceRatio * (-nextValueAuxDot * continuation * xc) * nextValueStepScale);
                    device float *rewardWeight = &agent.optionReward[i * optionN + k];
                    *rewardWeight = keepFinite(*rewardWeight, *rewardWeight + aM * importanceRatio * (-twinPrevDot * continuation * xc) * nextValueStepScale);
                }
            }
        }
    }

    if (!OAK_ABLATE_GVF_PREDICTION) {
        float valueTdError = agent.td[0];
        float valueStepScale = agent.valueScale[0];
        for (uint cb = 0u; cb < activeFeatureCount; cb += OAK_CORE) {
            uint eH = cb + tid;
            uint iH = activeIndex(previousActive, eH);
            for (uint l = 0u; l < OAK_CORE; ++l) {
                uint i = simd_shuffle(iH, (ushort)l);
                if (i % tw != tid) continue;
                float w = agent.valueWeight[i];
                float delta = valueTdError * prev[i] * valueStepScale;
                agent.valueWeight[i] = keepFinite(w, w + agent.valueAlpha[i] * delta);
            }
        }
    }
}

inline void emitOptionActions(Agent agent, uint tid) {

    uint policyN = actionMeanCount(agent);
    uint optionPolicyStride = agent.optionCount * policyN;
    device const uint *active = activeFeatures(agent, agent.stateSlot);
    device const float *currS = state_at(agent, agent.stateSlot);
    if (OAK_ABLATE_OPTION_POLICIES) {
        for (uint k = tid; k < agent.optionCount; k += OAK_CORE)
            for (uint j = 0u; j < agent.actionCount; ++j) {
                agent.optionPolicyMean[k * agent.actionCount + j] = 0.0f;
                agent.optionLogit[k * agent.actionCount + j] = 0.0f;
            }
        if (tid < agent.actionCount) agent.policyMean[tid] = 0.0f;
        return;
    }
    for (uint k = tid; k < agent.optionCount; k += OAK_CORE) {
    if (agent.parked[k] != 0u) {
        for (uint j = 0u; j < agent.actionCount; ++j) {
            agent.optionPolicyMean[k * agent.actionCount + j] = 0.0f;
            agent.optionLogit[k * agent.actionCount + j] = 0.0f;
        }
        continue;
    }
    float policyLogit[16];
    for (uint j = 0u; j < policyN; ++j) policyLogit[j] = 0.0f;
    for (uint a = 0u; a < OAK_ACT; ++a) {
        uint i = activeIndex(active, a);
        float x = currS[i];
        device const float *row = &agent.optionPolicy[i * optionPolicyStride + k * policyN];
        for (uint j = 0u; j < policyN; ++j) policyLogit[j] = fma(row[j], x, policyLogit[j]);
    }
    for (uint j = 0u; j < agent.actionCount; ++j) {
        uint optionActionIndex = k * agent.actionCount + j;

        float actionMean = (j < policyN) ? tanh(policyLogit[j]) : 0.0f;
        agent.optionLogit[optionActionIndex] = (j < policyN) ? policyLogit[j] : 0.0f;
        if (!isfinite(actionMean)) actionMean = agent.optionPolicyMean[optionActionIndex];
        agent.optionPolicyMean[optionActionIndex] = actionMean;
    }
    }
}

inline void blendKeyboard(Agent agent, uint tid) {
    if (tid >= agent.actionCount) return;
    uint d = tid;
    uint selected = agent.selectedOption[0] % agent.optionCount;
    float blendedMean = 0.0f, keyboardNorm = 0.0f, selectedMean = 0.0f;
    float blendedLogit = 0.0f, selectedLogit = 0.0f;
    for (uint ok = 0u; ok < agent.optionCount; ++ok) {
        float optionMeanValue = agent.optionPolicyMean[ok * agent.actionCount + d];
        if (!isfinite(optionMeanValue)) optionMeanValue = 0.0f;
        if (ok == selected) selectedMean = optionMeanValue;
        float optionLogitValue = agent.optionLogit[ok * agent.actionCount + d];
        if (!isfinite(optionLogitValue)) optionLogitValue = 0.0f;
        if (ok == selected) selectedLogit = optionLogitValue;
        float w = agent.keyboard[ok];
        if (!isfinite(w)) w = 0.0f;
        blendedMean = fma(w, optionMeanValue, blendedMean);
        blendedLogit = fma(w, optionLogitValue, blendedLogit);
        keyboardNorm += fabs(w);
    }

    {
        float baseOptionMean = agent.optionPolicyMean[d];
        if (isfinite(baseOptionMean)) { blendedMean += baseOptionMean; keyboardNorm += 1.0f; }
        float baseOptionLogit = agent.optionLogit[d];
        if (isfinite(baseOptionLogit)) blendedLogit += baseOptionLogit;
    }
    agent.policyMean[d] = (keyboardNorm > 0.0f && !OAK_ABLATE_KEYBOARD_TO_POLICY)
        ? blendedMean / keyboardNorm : selectedMean;
    agent.policyMeanRaw[d] = (keyboardNorm > 0.0f && !OAK_ABLATE_KEYBOARD_TO_POLICY)
        ? blendedLogit / keyboardNorm : selectedLogit;
}

static inline void replaceWorstFeature(Agent agent, uint tid, threadgroup float *tgStash,
                                      threadgroup float *tgDirty) {
    if (tid == 0u) tgDirty[0] = 0.0f;
    device const float *curr = state_at(agent, 1u - agent.stateSlot);
    if (tid == 0u) agent.planScratch[0] = agent.reward[0];
    simdgroup_barrier(mem_flags::mem_device);

    uint bestNursery;
    float bestNurseryBound;
    {
        float bestBound = -FLT_MAX;
        uint bestIndex = 0xffffffffu;
        for (uint n = tid; n < agent.nurseryCount; n += OAK_CORE) {
            float lowerConfidenceBound = agent.nurseryScore[n];
            if (lowerConfidenceBound > bestBound) { bestBound = lowerConfidenceBound; bestIndex = n; }
        }
        bestNurseryBound = simd_max(bestBound);
        bestNursery = simd_min((bestBound == bestNurseryBound) ? bestIndex : 0xffffffffu);
    }

    uint haveWorst = 0u, worstFeature = 0u, worstAge = 0u;
    float worstUtility = 0.0f, worstRedundancy = 0.0f, worstDownstream = 0.0f;
    for (uint i = agent.anchorCount + tid; i < agent.featureCount; i += OAK_CORE) {
        float4 e = *(device const float4 *)&agent.replaceScore[i * 4u];
        if (e.w <= 0.0f) continue;
        float util = e.x, redundancy = e.y, downstream = e.z;
        uint age = agent.featureAge[i];
        bool worse = (haveWorst == 0u)
            || downstream < worstDownstream
            || (downstream == worstDownstream && util < worstUtility)
            || (downstream == worstDownstream && util == worstUtility && redundancy > worstRedundancy)
            || (downstream == worstDownstream && util == worstUtility && redundancy == worstRedundancy && age > worstAge)
            || (downstream == worstDownstream && util == worstUtility && redundancy == worstRedundancy
                && age == worstAge && i < worstFeature);
        if (worse) {
            haveWorst = 1u; worstFeature = i; worstAge = age;
            worstUtility = util; worstRedundancy = redundancy; worstDownstream = downstream;
        }
    }

    for (uint t = 1u; t < OAK_CORE; ++t) {
        uint otherHave = simd_shuffle(haveWorst, (ushort)t);
        if (otherHave == 0u) continue;
        uint otherWorst = simd_shuffle(worstFeature, (ushort)t);
        uint otherAge = simd_shuffle(worstAge, (ushort)t);
        float otherUtility = simd_shuffle(worstUtility, (ushort)t);
        float otherRedundancy = simd_shuffle(worstRedundancy, (ushort)t);
        float otherDownstream = simd_shuffle(worstDownstream, (ushort)t);
        bool worse = (haveWorst == 0u)
            || otherDownstream < worstDownstream
            || (otherDownstream == worstDownstream && otherUtility < worstUtility)
            || (otherDownstream == worstDownstream && otherUtility == worstUtility && otherRedundancy > worstRedundancy)
            || (otherDownstream == worstDownstream && otherUtility == worstUtility && otherRedundancy == worstRedundancy && otherAge > worstAge)
            || (otherDownstream == worstDownstream && otherUtility == worstUtility && otherRedundancy == worstRedundancy
                && otherAge == worstAge && otherWorst < worstFeature);
        if (worse) {
            haveWorst = 1u; worstFeature = otherWorst; worstAge = otherAge;
            worstUtility = otherUtility; worstRedundancy = otherRedundancy; worstDownstream = otherDownstream;
        }
    }

    for (uint k = tid; k < agent.optionCount; k += OAK_CORE) {
        if (agent.optionAge[k] < 0xffffffffu) agent.optionAge[k] += 1u;
        if (agent.lastStopAge[k] < 0xffffffffu) agent.lastStopAge[k] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
    uint candidateStart = agent.anchorCount;
    uint eligible = 0u;
    float eligibleUtilSum = 0.0f, eligibleRedundancySum = 0.0f, eligibleDownstreamSum = 0.0f;
    for (uint i = candidateStart; i < agent.featureCount; ++i) {
        float4 e = *(device const float4 *)&agent.replaceScore[i * 4u];
        if (e.w > 0.0f) {
            eligible += 1u;
            eligibleUtilSum += e.x;
            eligibleRedundancySum += e.y;
            eligibleDownstreamSum += e.z;
        }
    }
    uint worst = (haveWorst != 0u) ? worstFeature : candidateStart;

    if (!OAK_ABLATE_GENERATE_AND_TEST) {

        uint staleLimit = agent.featureCount * agent.optionCount;
        uint staleOption = 0u;
        for (uint k2 = 1u; k2 < agent.optionCount; ++k2) {
            if (agent.parked[k2] == 0u || agent.lastStopAge[k2] < staleLimit) continue;
            staleOption = k2; break;
        }
        if (staleOption == 0u) {
            for (uint k2 = 1u; k2 < agent.optionCount; ++k2) {
                if (agent.parked[k2] != 0u || agent.lastStopAge[k2] < staleLimit) continue;
                staleOption = k2; break;
            }
        }
        if (staleOption != 0u) {
            uint k = staleOption;
            agent.optionAge[k] = 0u;
            agent.lastStopAge[k] = 0u;
            agent.optionUtility[k] = float(agent.featureCount);
            agent.actorAdvantage[k] = 0.0f;
            agent.optionAverageReward[k] = 0.0f;
            agent.optionModelAverageReward[k] = 0.0f;
            agent.parked[k] = 0u;
            uint feat = agent.optionFeature[k] % agent.featureCount;
            for (uint i = 0u; i < agent.featureCount; ++i) {
                agent.optionValue[(i) * agent.optionCount + k] = 0.0f;
                agent.optionValueAux[(i) * agent.optionCount + k] = 0.0f;
                agent.optionReward[(i) * agent.optionCount + k] = 0.0f;
                agent.optionDuration[(i) * agent.optionCount + k] = 0.0f;
                agent.optionNextValue[(i) * agent.optionCount + k] = 0.0f;
                agent.optionNextValueAux[(i) * agent.optionCount + k] = 0.0f;
            }
            agent.optionValue[(feat) * agent.optionCount + k] = agent.planScratch[PLAN_INV_SQRT_FEATURES];
            tgStash[k * 4u + 0u] = 0.0f;
            tgStash[k * 4u + 1u] = 0.0f;
            tgStash[k * 4u + 2u] = 0.0f;
            tgStash[k * 4u + 3u] = 0.0f;
        }
    }

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRedundancy = eligibleRedundancySum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtility < meanEligibleUtil;
    bool highRedundancy = worstRedundancy > meanEligibleRedundancy;
    bool lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
    bool structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);

    bool doReplace = !OAK_ABLATE_GENERATE_AND_TEST
        && structurallyReplaceable && worst < agent.featureCount
        && bestNursery != 0xffffffffu
        && bestNurseryBound > worstUtility + worstDownstream;
    if (doReplace && worst < agent.featureCount) {
        uint pa = agent.nurseryParentA[bestNursery] % agent.featureCount;
        uint pb = agent.nurseryParentB[bestNursery] % agent.featureCount;
        uint kind = agent.nurseryKind[bestNursery] % 4u;
        uint birth = agent.featureBirthCount[worst] + 1u;
        agent.featureBirthCount[worst] = birth;
        agent.featureParentA[worst] = pa;
        agent.featureParentB[worst] = pb;
        agent.featureKind[worst] = kind;

        agent.featureAge[worst] = agent.nurseryAge[bestNursery];
        agent.featureMean[worst] = agent.nurseryStats[(bestNursery) * 4u];
        agent.featureSecondMoment[worst] = agent.nurseryStats[(bestNursery) * 4u + 1u];
        agent.featureUtility[worst] = agent.nurseryStats[(bestNursery) * 4u + 2u];
        agent.valueWeight[worst] = 0.0f; agent.bonusWeight[worst] = 0.0f;

        uint seedParentA = 0u, seedParentB = (agent.featureCount > 1u) ? 1u : 0u;
        float bestA = -FLT_MAX, bestB = -FLT_MAX;
        uint featureLimit = agent.featureCount;
        for (uint i = 0u; i < featureLimit; ++i) {
            if (i == worst) continue;
            float parentUtility = agent.featureUtility[i];
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestA) { bestA = parentUtility; seedParentA = i; }
        }
        for (uint i = 0u; i < featureLimit; ++i) {
            if (i == worst || i == seedParentA) continue;
            float decor = 1.0f / (1.0f + fabs(curr[i] * curr[seedParentA]));
            float parentUtility = agent.featureUtility[i] * decor;
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestB) { bestB = parentUtility; seedParentB = i; }
        }
        agent.nurseryParentA[bestNursery] = seedParentA;
        agent.nurseryParentB[bestNursery] = seedParentB;
        agent.nurseryKind[bestNursery] = (agent.nurseryKind[bestNursery] + 1u) % 4u;
        agent.nurseryAge[bestNursery] = 0u;
        agent.nurseryStats[(bestNursery) * 4u] = 0.0f;
        agent.nurseryStats[(bestNursery) * 4u + 1u] = 0.0f;
        agent.nurseryStats[(bestNursery) * 4u + 2u] = 0.0f;
        agent.nurseryStats[(bestNursery) * 4u + 3u] = 0.0f;

        float initialStepSize = 1.0f / float(OAK_ACT);
        agent.valueAlpha[worst] = initialStepSize;
        agent.valueBeta[worst] = log(agent.valueAlpha[worst]); agent.valueH[worst] = 0.0f; agent.valueGradEnergy[worst] = 0.0f;
        agent.optionValueAlpha[worst] = initialStepSize;
        agent.optionValueBeta[worst] = log(agent.optionValueAlpha[worst]); agent.optionValueH[worst] = 0.0f; agent.optionValueGradEnergy[worst] = 0.0f;
        agent.modelAlpha[worst] = initialStepSize;
        agent.modelBeta[worst] = log(agent.modelAlpha[worst]); agent.modelH[worst] = 0.0f; agent.modelGradEnergy[worst] = 0.0f;

        uint k = 1u;
        float worstOptionDemand = FLT_MAX;
        for (uint ok = 1u; ok < agent.optionCount; ++ok) {
            float use = agent.optionUtility[ok] + agent.optionPriority[ok];
            if (!isfinite(use)) use = FLT_MAX;
            if (agent.optionFeature[ok] == worst) { k = ok; worstOptionDemand = -FLT_MAX; break; }
            if (use < worstOptionDemand) { worstOptionDemand = use; k = ok; }
        }
        agent.optionAge[k] = 0u; agent.optionFeature[k] = worst; agent.optionUtility[k] = float(agent.featureCount);
        agent.actorAdvantage[k] = 0.0f;
        agent.optionAverageReward[k] = 0.0f;
        agent.optionModelAverageReward[k] = 0.0f;
        agent.optionFitness[k] = 0.0f;
        agent.lastStopAge[k] = 0u;

        agent.replaceSlot[0] = 1u; agent.replaceSlot[1] = worst;
        agent.replaceSlot[2] = k; agent.replaceSlot[3] = birth;
    } else {
        agent.replaceSlot[0] = 0u;
    }
    if (agent.replaceSlot[0] == 0u && !OAK_ABLATE_GENERATE_AND_TEST
        && (agent.stepCount % max(agent.optionCount, 1u)) == 0u) {

        float meanOptionFitness = 0.0f;
        for (uint k2 = 0u; k2 < agent.optionCount; ++k2) {
            float f = agent.optionFitness[k2];
            meanOptionFitness += isfinite(f) ? f : 0.0f;
        }
        meanOptionFitness *= agent.invOptionCount;
        uint replacedOption = 0u; float worstFitness = FLT_MAX;
        for (uint k2 = 1u; k2 < agent.optionCount; ++k2) {
            if (agent.optionAge[k2] < agent.featureCount) continue;
            float f = agent.optionFitness[k2];
            if (!isfinite(f)) f = 0.0f;
            if (f < worstFitness) { worstFitness = f; replacedOption = k2; }
        }
        if (replacedOption != 0u && worstFitness < 0.0f && worstFitness < meanOptionFitness) {
            agent.optionAge[replacedOption] = 0u; agent.lastStopAge[replacedOption] = 0u;
            agent.optionFitness[replacedOption] = 0.0f; agent.optionAverageReward[replacedOption] = 0.0f; agent.optionModelAverageReward[replacedOption] = 0.0f;
            agent.actorAdvantage[replacedOption] = 0.0f;
            agent.replaceSlot[0] = 2u; agent.replaceSlot[2] = replacedOption;
            agent.replaceSlot[3] = agent.stepCount;
        }
    }
    agent.scalars->stateSlot = 1u - agent.stateSlot;
    agent.scalars->stepCount = agent.stepCount + 1u;
    }

    simdgroup_barrier(mem_flags::mem_device);
    if (agent.replaceSlot[0] != 0u) {
        uint worst = agent.replaceSlot[1], k = agent.replaceSlot[2], birth = agent.replaceSlot[3];
        float policyPerturbScale = 1.0f / float(max(agent.actionCount, 1u));
        uint policyN = actionMeanCount(agent);
        uint optionPolicyStride = agent.optionCount * policyN;

        if (tid == 0u) {
            tgStash[k * 4u + 0u] = 0.0f;
            tgStash[k * 4u + 1u] = 0.0f;
            tgStash[k * 4u + 2u] = 0.0f;
            tgStash[k * 4u + 3u] = 0.0f;
            agent.parked[k] = 0u;
            if (agent.replaceSlot[0] == 1u) tgDirty[0] = 1.0f;
        }

        for (uint i = tid; i < agent.featureCount; i += OAK_CORE) {
            agent.optionValue[(i) * agent.optionCount + k] = 0.0f;
            agent.optionValueAux[(i) * agent.optionCount + k] = 0.0f;
            agent.optionReward[(i) * agent.optionCount + k] = 0.0f;
            agent.optionDuration[(i) * agent.optionCount + k] = 0.0f;
            agent.optionNextValue[(i) * agent.optionCount + k] = 0.0f;
            agent.optionNextValueAux[(i) * agent.optionCount + k] = 0.0f;
            for (uint d = 0u; d < policyN; ++d) {
                uint h = (i * 2654435761u) ^ (d * 40503u)
                       ^ (k * 2246822519u) ^ (birth * 3266489917u)
                       ^ agent.stepCount;
                h ^= h >> 13; h *= 2654435761u; h ^= h >> 16;
                float hashedNoise = float(h & 0xffffu) / 32768.0f - 1.0f;

                float w0 = agent.optionPolicy[i * optionPolicyStride + d];
                if (!isfinite(w0)) w0 = 0.0f;
                agent.optionPolicy[i * optionPolicyStride + k * policyN + d] =
                    w0 + hashedNoise * policyPerturbScale * rsqrt(float(OAK_ACT));
            }
        }
        simdgroup_barrier(mem_flags::mem_device);

        if (tid < agent.optionCount && agent.replaceSlot[0] == 1u) {
            uint ok = tid;
            agent.optionValue[worst * agent.optionCount + ok] = 0.0f;
            agent.optionValueAux[worst * agent.optionCount + ok] = 0.0f;
            agent.optionReward[worst * agent.optionCount + ok] = 0.0f;
            agent.optionDuration[worst * agent.optionCount + ok] = 0.0f;
            agent.optionNextValue[worst * agent.optionCount + ok] = 0.0f;
            agent.optionNextValueAux[worst * agent.optionCount + ok] = 0.0f;
            agent.optionNextValueTwin[worst * agent.optionCount + ok] = 0.0f;
            for (uint d = 0u; d < policyN; ++d)
                agent.optionPolicy[worst * optionPolicyStride + ok * policyN + d] = 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) {
            uint optionFeatureIndex = (agent.replaceSlot[0] == 1u) ? worst : (agent.optionFeature[k] % agent.featureCount);
            agent.optionValue[optionFeatureIndex * agent.optionCount + k] = agent.planScratch[PLAN_INV_SQRT_FEATURES];
        }
    }
}

static inline void updateUtilities(Agent agent, uint tid) {
    const uint tgW = OAK_CORE;
    uint tPrev = agent.stepCount - 1u;
    device const float *prev = state_at(agent, 1u - agent.stateSlot);
    device const float *curr = state_at(agent, agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, 1u - agent.stateSlot);
    float trackingRate = track_rate(tPrev, agent.featureCount);

    const uint activeSlotsPerThread = OAK_ACT / OAK_CORE;
    uint iS[activeSlotsPerThread];
    float optionTdMagnitude[activeSlotsPerThread], keyboardSearchSignal[activeSlotsPerThread];
    for (uint s = 0u; s < activeSlotsPerThread; ++s) {
        iS[s] = previousActive[tid + s * OAK_CORE];
        optionTdMagnitude[s] = 0.0f; keyboardSearchSignal[s] = 0.0f;
    }
    float valueTdError = agent.td[0];
    for (uint m = 0u; m < agent.optionCount; m += OAK_CORE) {
        uint kH = m + tid;
        bool kLive = kH < agent.optionCount;
        uint   ofH = kLive ? agent.optionFeature[kH] : 0u;
        float tdH = kLive ? agent.optionTd[kH] : 0.0f;
        float kbH = kLive ? agent.keyboard[kH] : 0.0f;
        float aaH = kLive ? agent.actorAdvantage[kH] : 0.0f;
        float imH = kLive ? agent.importance[kH] : 0.0f;
        uint kEnd = min(OAK_CORE, agent.optionCount - m);
        for (uint l = 0u; l < kEnd; ++l) {
            uint   ofV = simd_shuffle(ofH, (ushort)l);
            float tdV = simd_shuffle(tdH, (ushort)l);
            float kbV = simd_shuffle(kbH, (ushort)l);
            float aaV = simd_shuffle(aaH, (ushort)l);
            float imV = simd_shuffle(imH, (ushort)l);
            for (uint s = 0u; s < activeSlotsPerThread; ++s) {
                if ((ofV % agent.featureCount) != iS[s]) continue;
                float optionTdUse = fabs(tdV);
                float keyboardMass = fabs(kbV);
                float keyboardUse = OAK_ABLATE_SEARCH_TO_GENERATE_TEST
                    ? 0.0f : keyboardMass * (fabs(aaV) + max(imV, 0.0f));
                if (isfinite(optionTdUse)) optionTdMagnitude[s] += optionTdUse;
                if (isfinite(keyboardUse)) keyboardSearchSignal[s] += keyboardUse;
            }
        }
    }
    uint sel = agent.selectedOption[0] % agent.optionCount;
    for (uint s = 0u; s < activeSlotsPerThread; ++s) {
        uint activeSlot = tid + s * OAK_CORE;
        uint i = iS[s];
        float optionValueSignal = optionTdMagnitude[s], keyboardSignal = keyboardSearchSignal[s];
        float planningSignal = fabs(valueTdError);

        float selectedPolicyLoad = 0.0f;
        for (uint d = 0u; d < actionMeanCount(agent); ++d)
            selectedPolicyLoad += fabs(agent.optionPolicy[i * agent.optionCount * actionMeanCount(agent) + sel * actionMeanCount(agent) + d]);
        float load = fabs(prev[i]) * (fabs(agent.valueWeight[i]) + selectedPolicyLoad);
        if (!isfinite(load)) load = 0.0f;
        float featureUtilitySignal = fabs(planningSignal * prev[i])
                + fabs((optionValueSignal + keyboardSignal) * prev[i])
                + load;
        agent.featureUtility[i] += trackingRate * (featureUtilitySignal - agent.featureUtility[i]);

        {
            float bonusHighWater = agent.bonusWeight[i];
            if (!isfinite(bonusHighWater)) bonusHighWater = 0.0f;
            float valueWeight = agent.valueWeight[i];
            if (isfinite(valueWeight) && valueWeight > bonusHighWater) agent.bonusWeight[i] = valueWeight;
        }
        uint age = agent.featureAge[i];
        metaOptimizeUpdate(agent.valueAlpha, agent.valueBeta, agent.valueH, agent.valueGradEnergy,
                            i, valueTdError, prev[i], age, agent.metaEta, agent.invFeatureCount);
        uint metaOptimizeSlot = tPrev % OAK_CORE;
        if (activeSlot == metaOptimizeSlot) {

            uint metaOptimizeOption = tPrev % max(agent.optionCount, 1u);
            float optionTdSigned = agent.optionTd[metaOptimizeOption];
            float modelSigned = agent.optionRewardTd[metaOptimizeOption];
            metaOptimizeUpdate(agent.optionValueAlpha, agent.optionValueBeta, agent.optionValueH, agent.optionValueGradEnergy,
                                i, optionTdSigned, prev[i], age, agent.metaEta, agent.invFeatureCount);
            metaOptimizeUpdate(agent.modelAlpha, agent.modelBeta, agent.modelH, agent.modelGradEnergy,
                                i, modelSigned, prev[i], age, agent.metaEta, agent.invFeatureCount);
        }
    }
    for (uint k = tid; k < agent.optionCount; k += tgW) {
        float err = agent.optionPriority[k];
        if (!isfinite(err) || OAK_ABLATE_SEARCH_CONTROL) err = 0.0f;
        float keyboardMass = OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(agent.keyboard[k]);
        float optionUtilitySignal = agent.optionTd[k] * agent.optionTd[k]
                + keyboardMass * fabs(agent.actorAdvantage[k])
                - err;
        agent.optionUtility[k] += trackingRate * (optionUtilitySignal - agent.optionUtility[k]);
    }

    if (tid == 0u) {
        float utilSum = 0.0f;
        uint maturedLive = 0u;
        for (uint k = 0u; k < agent.optionCount; ++k) {
            if (agent.parked[k] != 0u || agent.optionAge[k] < agent.featureCount) continue;
            float u = agent.optionUtility[k];
            utilSum += isfinite(u) ? u : 0.0f;
            maturedLive += 1u;
        }
        float meanLiveUtil = (maturedLive > 0u) ? utilSum / float(maturedLive) : 0.0f;
        uint selectedForPark = agent.selectedOption[0] % agent.optionCount;
        for (uint k = 1u; k < agent.optionCount; ++k) {
            if (agent.parked[k] == 0u) continue;
            float u = agent.optionUtility[k];
            bool revived = (k == selectedForPark) || (agent.keyboard[k] > 0.0f)
                || (isfinite(u) && u >= meanLiveUtil);
            if (revived) { agent.parked[k] = 0u; break; }
        }
        uint optionToPark = 0xffffffffu;
        float parkUtility = 0.0f;
        for (uint k = 1u; k < agent.optionCount; ++k) {
            if (agent.parked[k] != 0u || agent.optionAge[k] < agent.featureCount || k == selectedForPark) continue;
            float kb = agent.keyboard[k];
            if (!isfinite(kb) || kb > 0.0f) continue;
            float u = agent.optionUtility[k];
            if (!isfinite(u)) u = 0.0f;
            if (u < meanLiveUtil && (optionToPark == 0xffffffffu || u < parkUtility)) {
                optionToPark = k; parkUtility = u;
            }
        }
        if (optionToPark != 0xffffffffu) agent.parked[optionToPark] = 1u;
    }

    if (!OAK_ABLATE_OPTION_POLICIES && !OAK_ABLATE_GENERATE_AND_TEST) {

        float fH = (tid < agent.optionCount) ? agent.optionFitness[tid] : 0.0f;
        float meanOptionFitness = 0.0f;
        for (uint k = 0u; k < agent.optionCount; ++k) {
            float f = simd_shuffle(fH, (ushort)k);
            meanOptionFitness += isfinite(f) ? f : 0.0f;
        }
        meanOptionFitness *= agent.invOptionCount;
        float previousFeatureNormSq = agent.previousFeatureNormSq[0];
        if (!isfinite(previousFeatureNormSq) || previousFeatureNormSq < 0.0f) previousFeatureNormSq = 0.0f;
        float rate = agent.invFeatureCount * agent.invOptionCount / (1.0f + previousFeatureNormSq);

        const uint policyN = max(agent.actionCount / 2u, 1u);
        const uint optionPolicyStride = agent.optionCount * policyN;
        float fitnessSearchDirection = 0.0f;
        float baseOptionMean = 0.0f;
        if (tid < policyN) {
            baseOptionMean = agent.optionPolicyMean[tid];
            if (!isfinite(baseOptionMean)) baseOptionMean = 0.0f;
        }
        float fitnessDirection = 0.0f;
        for (uint k = 1u; k < agent.optionCount; ++k) {
            float f = simd_shuffle(fH, (ushort)k);
            if (agent.parked[k] != 0u) continue;
            if (tid < policyN) {
                float fitnessCentered = (isfinite(f) ? f : 0.0f) - meanOptionFitness;
                float mk = agent.optionPolicyMean[k * agent.actionCount + tid];
                fitnessDirection = fma(fitnessCentered, (isfinite(mk) ? mk : 0.0f) - baseOptionMean, fitnessDirection);
            }
        }
        if (tid < policyN)
            fitnessSearchDirection = isfinite(fitnessDirection) ? (fitnessDirection / (1.0f + fabs(fitnessDirection))) : 0.0f;
        for (uint policyWeightIndex = tid; policyWeightIndex < policyN * OAK_ACT; policyWeightIndex += tgW) {
            uint d = policyWeightIndex / OAK_ACT;
            uint a = policyWeightIndex % OAK_ACT;
            float fitnessDirection = simd_shuffle(fitnessSearchDirection, (ushort)d);

            uint i = previousActive[a];
            float w0 = agent.optionPolicy[i * optionPolicyStride + d];
            agent.optionPolicy[i * optionPolicyStride + d] = keepFinite(w0, w0 + rate * fitnessDirection * prev[i]);
        }
    }
    if (!OAK_ABLATE_GENERATE_AND_TEST) {

        for (uint n = tid + (((tPrev >> 1u) & 1u) != 0u ? tgW : 0u); n < agent.nurseryCount; n += 2u * tgW) {
            uint parentA = agent.nurseryParentA[n] % agent.featureCount;
            uint parentB = agent.nurseryParentB[n] % agent.featureCount;
            uint constructor = agent.nurseryKind[n] % 4u;
            float constructed;
            if (constructor == FEATURE_OBSERVATION_TIMESCALE) {
                constructed = agent.observationTimescale[agent.nurseryParentA[n] % (agent.traceLevels * agent.observationCount)];
            } else {
                float a = prev[parentA], b = prev[parentB];
                constructed = (constructor == FEATURE_PRODUCT) ? a * b
                    : (constructor == FEATURE_NORMALIZED_SUM) ? (a + b) * 0.7071067811865475f
                    : (constructor == FEATURE_RESIDUAL_PRODUCT) ? a * (b - a)
                    : a * (b - a);
            }
            uint age = agent.nurseryAge[n];
            float rate = max(1.0f / (float(age) + 1.0f), agent.invFeatureCount);

            device float4 *nurseryStatsVec = (device float4 *)&agent.nurseryStats[(n) * 4u];
            float4 st = *nurseryStatsVec;
            st.x += rate * (constructed - st.x);
            st.y += rate * (constructed * constructed - st.y);
            float var = max(st.y - st.x * st.x, 0.0f);

            float ridge = max(1.0f / (float(age) + 1.0f), 1.0f / float(2u * OAK_CORE));
            float standardizedCandidate = (constructed - st.x) * rsqrt(var + ridge);
            float candidateUtilitySignal = fabs(agent.td[0] * standardizedCandidate);
            if (isfinite(candidateUtilitySignal)) {
                st.z += trackingRate * (candidateUtilitySignal - st.z);
                st.w += trackingRate * (candidateUtilitySignal * candidateUtilitySignal - st.w);
            }
            *nurseryStatsVec = st;
            if (age < 0xffffffffu) agent.nurseryAge[n] = age + 1u;
        }
    }
    if (!OAK_ABLATE_GENERATE_AND_TEST) {

        for (uint i = tid; i < agent.featureCount; i += tgW) agent.optionUseByFeature[i] = 0.0f;
        simdgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) {

            for (uint k = 0u; k < agent.optionCount; ++k) {
                float use = agent.optionUtility[k]
                    + (OAK_ABLATE_SEARCH_TO_GENERATE_TEST ? 0.0f : fabs(agent.keyboard[k]));
                if (isfinite(use)) agent.optionUseByFeature[agent.optionFeature[k] % agent.featureCount] += use;
            }
        }

        for (uint n = tid; n < agent.nurseryCount; n += tgW) {
            float nurseryUtility = agent.nurseryStats[(n) * 4u + 2u];
            float lowerConfidenceBound = -FLT_MAX;
            if (agent.nurseryAge[n] >= agent.featureCount && isfinite(nurseryUtility)) {
                float nurseryUtilityVariance = max(agent.nurseryStats[(n) * 4u + 3u] - nurseryUtility * nurseryUtility, 0.0f);
                float nurseryUtilityRate = max(1.0f / (float(agent.nurseryAge[n]) + 1.0f), agent.invFeatureCount);
                float v = nurseryUtility - sqrt(nurseryUtilityVariance * nurseryUtilityRate);
                if (isfinite(v)) lowerConfidenceBound = v;
            }
            agent.nurseryScore[n] = lowerConfidenceBound;
        }
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (!OAK_ABLATE_GENERATE_AND_TEST) {

        device const float *currS = curr;
        for (uint i = agent.anchorCount + tid + (((tPrev >> 1u) & 1u) != 0u ? tgW : 0u);
             i < agent.featureCount; i += 2u * tgW) {
            device float4 *replaceScoreVec = (device float4 *)&agent.replaceScore[i * 4u];
            if (agent.featureAge[i] < agent.featureCount) { *replaceScoreVec = float4(0.0f, 0.0f, 0.0f, 0.0f); continue; }
            uint parentA = agent.featureParentA[i] % agent.featureCount;
            float util = agent.featureUtility[i];
            if (!isfinite(util)) util = 0.0f;
            float redundancy = fabs(currS[i] * currS[parentA]);
            if (!isfinite(redundancy)) redundancy = 0.0f;
            float downstreamUse = agent.optionUseByFeature[i] + fabs(agent.valueWeight[i]);
            if (!isfinite(downstreamUse)) downstreamUse = 0.0f;
            *replaceScoreVec = float4(util, redundancy, downstreamUse, 1.0f);
        }
    }
}


kernel void oak_fleet_init(device float *params [[buffer(0)]],
                           device float *stateArena [[buffer(1)]],
                           device uint *uintArena [[buffer(2)]],
                           device KernelScalars *scalars [[buffer(3)]],
                           constant KernelConfig &cfg [[buffer(4)]],
                           device OakFleetLane *lanes [[buffer(6)]],
                           device const uint *laneSeeds [[buffer(7)]],
                           constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                           device const uint *laneEnvKinds [[buffer(9)]],
                           constant uint &laneOffset [[buffer(10)]],
                           uint gid [[thread_position_in_grid]]) {
    uint gidLocal = gid;
    gid += laneOffset;
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    uint storedEnvKind = laneEnvKinds[gid];
    HalfCheetahState halfCheetah;
    uint envWords[10];
    resetLane(lane, storedEnvKind, laneSeeds[gid], envWords, halfCheetah);
    {
        Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg, gidLocal, gid,
                           lane->envObservationCount, lane->envActionCount);
        for (uint o = 0u; o < agent.observationCount; ++o) {
            agent.observationAlpha[o] = 1.0f / float(2u * OAK_CORE);
            agent.observationBeta[o] = log(1.0f / float(2u * OAK_CORE));
        }
        float observation[17];
        env_observe(storedEnvKind, envWords, halfCheetah, observation);
        for (uint i = 0u; i < agent.observationCount && i < 17u; ++i) agent.observation[i] = observation[i];
        agent.reward[0] = 0.0f;
        for (uint d = 0u; d < agent.actionCount; ++d) agent.actionSignal[d] = 0.0f;
        for (uint i = 0u; i < agent.optionCount * agent.actionCount; ++i) agent.optionActorCredit[i] = 0.0f;
        for (uint k = 0u; k < agent.optionCount; ++k) {
            agent.importance[k] = 1.0f;
            agent.importanceStaging[k] = 1.0f;
        }

        agent.planScratch[PLAN_INV_SQRT_FEATURES] = rsqrt(float(max(agent.featureCount, 1u)));
        for (uint k = 0u; k < agent.optionCount; ++k) agent.parked[k] = 0u;
    }
}

static inline void sampleDiscreteAction(Agent agent,
                                                 device OakFleetLane *lane,
                                                 device const uint *laneSeeds,
                                                 uint gid, uint tid) {
    uint discreteActionCount = OAK_ENV_SPECS[envKind(lane->envKind)].actionCount;
    uint stepIdx = lane->steps;
    uint4 rw = philox4x32(uint2(laneSeeds[gid], RNG_STREAM_ACT),
                             uint4(stepIdx, 0u, 0u, 0u));
    uint4 rw1 = philox4x32(uint2(laneSeeds[gid], RNG_STREAM_ACT),
                              uint4(stepIdx, 1u, 0u, 0u));
    uint words[8] = { rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w };

    float behaviorLogitMax = -FLT_MAX;
    for (uint d = 0u; d < discreteActionCount; ++d) behaviorLogitMax = max(behaviorLogitMax, finiteOrZero(agent.policyMeanRaw[d]));
    float behaviorSoftmaxSum = 0.0f;
    for (uint d = 0u; d < discreteActionCount; ++d) behaviorSoftmaxSum += exp(finiteOrZero(agent.policyMeanRaw[d]) - behaviorLogitMax);
    uint sampledAction = 0u;
    float bestGumbelScore = -FLT_MAX;
    for (uint d = 0u; d < discreteActionCount; ++d) {
        float mu = finiteOrZero(agent.policyMeanRaw[d]);

        float uniform = min(u01open(words[d & 7u]), 0x1.fffffep-1f);
        float gumbelScore = mu - log(-log(uniform));
        if (gumbelScore > bestGumbelScore) { bestGumbelScore = gumbelScore; sampledAction = d; }
    }

    simdgroup_barrier(mem_flags::mem_device);
    if (tid < 6u) lane->pendingAction[tid] = (tid == 0u) ? float(sampledAction) : 0.0f;
    if (tid < agent.actionCount) {
        float actionSignal = 0.0f;
        if (tid < discreteActionCount)
            actionSignal = ((tid == sampledAction) ? 1.0f : 0.0f) - exp(finiteOrZero(agent.policyMeanRaw[tid]) - behaviorLogitMax) / behaviorSoftmaxSum;
        agent.actionSignal[tid] = actionSignal;
    }
    if (tid == 0u) lane->steps = stepIdx + 1u;
    if (tid < agent.optionCount) {
        uint off = tid * agent.actionCount;
        float optionLogitMax = -FLT_MAX;
        for (uint d = 0u; d < discreteActionCount; ++d) optionLogitMax = max(optionLogitMax, finiteOrZero(agent.optionLogit[off + d]));
        float optionSoftmaxSum = 0.0f;
        for (uint d = 0u; d < discreteActionCount; ++d)
            optionSoftmaxSum += exp(finiteOrZero(agent.optionLogit[off + d]) - optionLogitMax);
        for (uint d = 0u; d < agent.actionCount; ++d) {
            float credit = 0.0f;
            if (d < discreteActionCount)
                credit = ((d == sampledAction) ? 1.0f : 0.0f)
                   - exp(finiteOrZero(agent.optionLogit[off + d]) - optionLogitMax) / optionSoftmaxSum;
            agent.actorCreditStaging[off + d] = credit;
        }

        float behaviorActionProb = exp(finiteOrZero(agent.policyMeanRaw[sampledAction]) - behaviorLogitMax) / behaviorSoftmaxSum;
        float optionActionProb = exp(finiteOrZero(agent.optionLogit[off + sampledAction]) - optionLogitMax) / optionSoftmaxSum;
        float ratio = optionActionProb / behaviorActionProb;
        agent.importanceStaging[tid] = isfinite(ratio) ? min(ratio, float(agent.optionCount)) : 0.0f;
    }
}

static inline void sampleAction(Agent agent,
                                             device OakFleetLane *lanes,
                                             device const uint *laneSeeds,
                                             uint gid, uint tid) {
    device OakFleetLane *lane = &lanes[gid];
    if (OAK_ENV_SPECS[envKind(lane->envKind)].actionKind == 1u) {
        sampleDiscreteAction(agent, lane, laneSeeds, gid, tid);
        return;
    }
    device float *scratch = agent.sampleScratch;
    uint envActionCount = OAK_ENV_SPECS[envKind(lane->envKind)].actionCount;
    float sampleSigma = agent.sampleSigma;
    uint stepIdx = lane->steps;

    if (tid < envActionCount) {
        uint d = tid;
        uint2 key = uint2(laneSeeds[gid], RNG_STREAM_ACT);
        uint pair = d >> 1u;
        uint4 r = philox4x32(key, uint4(stepIdx, pair >> 1u, 0u, 0u));

        uint w0, w1;
        if (pair == 0u) { w0 = r.x; w1 = r.y; }
        else if (pair == 1u) { w0 = r.z; w1 = r.w; }
        else { w0 = r.x; w1 = r.y; }
        float uniform1 = u01open(w0);
        float uniform2 = u01(w1);
        float boxMullerRadius = sqrt(-2.0f * log(uniform1));
        float boxMullerAngle = 2.0f * M_PI_F * uniform2;
        float standardNormal = ((d & 1u) == 0u) ? (boxMullerRadius * cos(boxMullerAngle)) : (boxMullerRadius * sin(boxMullerAngle));
        float mu = finiteOrZero(agent.policyMean[d]);
        float sample = finiteOrZero(mu + sampleSigma * standardNormal);
        float clippedAction = max(-1.0f, min(1.0f, sample));
        lane->pendingAction[d] = clippedAction;
        agent.actionSignal[d] = clippedAction;
        agent.actionSignal[envActionCount + d] = 0.0f;
        scratch[d] = sample;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) lane->steps = stepIdx + 1u;
    if (tid < agent.optionCount) {
        uint off = tid * agent.actionCount;
        float logImportance = 0.0f;
        for (uint d = 0u; d < envActionCount; ++d) {
            float sample = scratch[d];
            float optionMeanValue = finiteOrZero(agent.optionPolicyMean[off + d]);

            agent.actorCreditStaging[off + d] = (sample - optionMeanValue) * (1.0f - optionMeanValue * optionMeanValue);
            agent.actorCreditStaging[off + envActionCount + d] = 0.0f;
            float behaviorResidual = sample - finiteOrZero(agent.policyMean[d]);
            float optionResidual = sample - optionMeanValue;
            logImportance += behaviorResidual * behaviorResidual - optionResidual * optionResidual;
        }

        float ratio = exp(logImportance * agent.invTwoSigmaSquared);
        agent.importanceStaging[tid] = isfinite(ratio) ? min(ratio, float(agent.optionCount)) : 0.0f;
    }
}

static inline void stepEnvironment(device float *params,
                                       device float *stateArena,
                                       device uint *uintArena,
                                       device KernelScalars *scalars,
                                       constant KernelConfig &cfg,
                                       device OakFleetLane *lanes,
                                       device const uint *laneSeeds,
                                       constant OakFleetEnvCfg &ecfg,
                                       device const int *recordSlot,
                                       device float *recordBuf,
                                       uint gid, uint gidLocal) {
    device OakFleetLane *lane = &lanes[gid];
    uint envK = envKind(lane->envKind);
    float ctrl[6];
    uint done = 0u;
    float obs[17];
    uint envWords[10];
    HalfCheetahState halfCheetah;
    float reward = advanceEnvironment(lane, envK, laneSeeds[gid], ecfg, ctrl, obs,
                                      envWords, halfCheetah, done);
    uint stepNum = lane->steps;
    accumulateReward(lane, reward, stepNum, ecfg);
    {
        Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg, gidLocal, gid,
                           lane->envObservationCount, lane->envActionCount);
        if (lane->deathStep == OAK_DEATH_NONE) {
            for (uint i = 0u; i < agent.actionCount; ++i) {
                if (!isfinite(agent.policyMean[i])) {
                    lane->deathStep = stepNum;
                    lane->deathDetector = OAK_DET_POLICY_MEAN;
                    break;
                }
            }
        }
        for (uint i = 0u; i < agent.observationCount && i < 17u; ++i) {
            agent.observation[i] = obs[i];
            if (!isfinite(obs[i]) && lane->deathStep == OAK_DEATH_NONE) {
                lane->deathStep = stepNum;
                lane->deathDetector = OAK_DET_ENV_OBS;
            }
        }
        agent.reward[0] = reward;
    }
    recordPose(envK, envWords, halfCheetah, ctrl, stepNum, done, ecfg, recordSlot, recordBuf, gid);
}

static inline void laneFence() {
    simdgroup_barrier(mem_flags::mem_device);
}

inline uint2 diagSketchBits(uint featureIndex) {
    uint2 h = uint2(featureIndex * 0x9E3779B9u, (featureIndex + 0x632BE59Bu) * 0x85EBCA6Bu);
    h ^= h >> 16u;
    h *= 0x21F0AAADu;
    h ^= h >> 15u;
    h *= 0x735A2D97u;
    h ^= h >> 15u;
    return h;
}

inline float diagSketchSign(uint2 bits, uint j) {
    uint word = j < OAK_CORE ? bits.x : bits.y;
    return ((word >> (j % OAK_CORE)) & 1u) == 0u ? 1.0f : -1.0f;
}

static inline void accumulateFeatureSketch(Agent agent, uint tid, device OakDiagAccum *acc) {
    device const float *curr = state_at(agent, 1u - agent.stateSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);
    float zLo = 0.0f, zHi = 0.0f;
    for (uint a = 0u; a < OAK_ACT; ++a) {
        uint i = currentActive[a];
        float x = curr[i];
        uint2 bits = diagSketchBits(i);
        zLo = fma(diagSketchSign(bits, tid), x, zLo);
        zHi = fma(diagSketchSign(bits, tid + OAK_CORE), x, zHi);
    }
    device float *rowLo = acc->gram + tid * OAK_DIAG_K;
    device float *rowHi = acc->gram + (tid + OAK_CORE) * OAK_DIAG_K;
    for (uint c = 0u; c < OAK_DIAG_K; ++c) {
        float zc = simd_shuffle(c < OAK_CORE ? zLo : zHi, (ushort)(c % OAK_CORE));
        rowLo[c] += zLo * zc;
        rowHi[c] += zHi * zc;
    }
}

kernel void oak_megastep(device float *params [[buffer(0)]],
                         device float *stateArena [[buffer(1)]],
                         device uint *uintArena [[buffer(2)]],
                         device KernelScalars *scalars [[buffer(3)]],
                         constant KernelConfig &cfg [[buffer(4)]],
                         device OakFleetLane *lanes [[buffer(6)]],
                         device const uint *laneSeeds [[buffer(7)]],
                         constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                         device const int *recordSlot [[buffer(9)]],
                         device float *recordBuf [[buffer(10)]],
                         constant uint3 &dispatchSpan [[buffer(11)]],
                         device OakDiagAccum *diagAccum [[buffer(12)]],
                         threadgroup float *tgCont [[threadgroup(0)]],
                         uint tgIdx [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint slane [[thread_index_in_simdgroup]],
                         uint sgrp [[simdgroup_index_in_threadgroup]]) {
    uint laneBase = (tgIdx + dispatchSpan.z) * OAK_MEGA_G;
    uint laneShift = dispatchSpan.z * OAK_MEGA_G;
    if (laneBase >= ecfg.laneCount) return;
    uint lanesThisTg = min(OAK_MEGA_G, ecfg.laneCount - laneBase);
    uint laneSg = sgrp;
    bool wideActive = laneSg < lanesThisTg;
    uint wideLane = laneBase + laneSg;
    bool packedActive = tid < lanesThisTg;
    uint packedLane = laneBase + tid;
    uint tgK = is_function_constant_defined(OAK_SPEC_OPTION_COUNT)
        ? OAK_SPEC_OPTION_COUNT : cfg.optionCount;
    threadgroup float *lCont = tgCont + laneSg * tgK;
    threadgroup float *lStash = tgCont + OAK_MEGA_G * tgK + laneSg * 4u * tgK;
    threadgroup float *lDirty = tgCont + 5u * OAK_MEGA_G * tgK + laneSg;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {

        if (wideActive && (OAK_MEGA_STAGES & 1u) != 0u) {
            Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg,
                               wideLane - laneShift, wideLane,
                               lanes[wideLane].envObservationCount, lanes[wideLane].envActionCount);
            encodeObservation(agent, slane, OAK_CORE, lCont, lStash);
        }
        laneFence();
        if (wideActive && (OAK_MEGA_STAGES & 128u) != 0u) {
            Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg,
                               wideLane - laneShift, wideLane,
                               lanes[wideLane].envObservationCount, lanes[wideLane].envActionCount);
            accumulateFeatureSketch(agent, slane, &diagAccum[wideLane]);
        }
        if (wideActive && (OAK_MEGA_STAGES & 8u) != 0u) {
            Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg,
                               wideLane - laneShift, wideLane,
                               lanes[wideLane].envObservationCount, lanes[wideLane].envActionCount);
            replaceWorstFeature(agent, slane, lStash, lDirty);
        }
        laneFence();
        if (wideActive && (OAK_MEGA_STAGES & 16u) != 0u) {
            Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg,
                               wideLane - laneShift, wideLane,
                               lanes[wideLane].envObservationCount, lanes[wideLane].envActionCount);
            emitOptionActions(agent, slane);
            simdgroup_barrier(mem_flags::mem_device);
            blendKeyboard(agent, slane);
        }
        laneFence();
        if (wideActive && (OAK_MEGA_STAGES & 32u) != 0u) {
            Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg,
                               wideLane - laneShift, wideLane,
                               lanes[wideLane].envObservationCount, lanes[wideLane].envActionCount);
            sampleAction(agent, lanes, laneSeeds, wideLane, slane);
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if ((OAK_MEGA_STAGES & 64u) != 0u) {
            if (packedActive && sgrp == 0u)
                stepEnvironment(params, stateArena, uintArena, scalars, cfg,
                                    lanes, laneSeeds, ecfg, recordSlot, recordBuf, packedLane,
                                    packedLane - laneShift);
        }
        if (sgrp > 0u && lanesThisTg > 0u) {
            uint idx = sgrp - 1u;
            if (lanesThisTg > OAK_MEGA_G - 1u) {

                uint T = dispatchSpan.y + it + 1u;
                uint skip = T % lanesThisTg;
                if (idx >= skip) idx += 1u;
            }
            if (idx < lanesThisTg) {
                uint wLane = laneBase + idx;
                Agent planAgent = make_agent(params, stateArena, uintArena, scalars, cfg,
                                    wLane - laneShift, wLane,
                                    lanes[wLane].envObservationCount, lanes[wLane].envActionCount);
                uint pSlot = 1u - planAgent.stateSlot;
                threadgroup const float *pStash = tgCont + OAK_MEGA_G * tgK + idx * 4u * tgK;
                bool stashDirty = tgCont[5u * OAK_MEGA_G * tgK + idx] != 0.0f;
                if ((OAK_MEGA_STAGES & 2u) != 0u)
                    planAllOptions(planAgent, slane, pStash, stashDirty, pSlot);
                if ((OAK_MEGA_STAGES & 4u) != 0u) {
                    device const float *prevS = state_at(planAgent, pSlot);
                    device const float *currS = state_at(planAgent, 1u - pSlot);
                    learnFromTransition(planAgent, prevS, currS, slane, OAK_CORE, pSlot);
                }
                simdgroup_barrier(mem_flags::mem_device);
                if ((OAK_MEGA_STAGES & 8u) != 0u)
                    updateUtilities(planAgent, slane);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void oak_fleet_snapshot(device float *params [[buffer(0)]],
                               device float *stateArena [[buffer(1)]],
                               device uint *uintArena [[buffer(2)]],
                               device KernelScalars *scalars [[buffer(3)]],
                               constant KernelConfig &cfg [[buffer(4)]],
                               device OakFleetLane *lanes [[buffer(6)]],
                               device const uint *laneSeeds [[buffer(7)]],
                               constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                               device OakLaneSnapshot *snap [[buffer(9)]],
                               constant uint2 &snapCtl [[buffer(10)]],
                               constant uint &fullScan [[buffer(11)]],
                               device OakDiagAccum *diagAccum [[buffer(12)]],
                               device float *diagOut [[buffer(13)]],
                               uint slane [[thread_index_in_simdgroup]],
                               uint sgid [[threadgroup_position_in_grid]]) {

    uint sgidL = sgid;
    sgid += snapCtl.y;
    if (sgid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[sgid];
    device OakLaneSnapshot *o = &snap[snapCtl.x + sgid];
    device OakDiagRecord *rec = (device OakDiagRecord *)(diagOut + ulong(snapCtl.x + sgid) * OAK_DIAG_OUT_FLOATS);
    if (slane == 0u) o->scalars = scalars[sgid];
    float weightAbsMean = NAN, weightAbsMeanValue = NAN;
    if (fullScan != 0u) {
        device const float *lp = params + sgidL * cfg.paramStride;

        uint firstNF = 0xffffffffu;
        for (uint i = slane; i < cfg.paramStride; i += OAK_CORE) {
            if (!isfinite(lp[i])) { firstNF = i; break; }
        }
        firstNF = simd_min(firstNF);
        uint lim = min(firstNF, cfg.paramStride);
        float mx = 0.0f, sumAbs = 0.0f, sumAbsValue = 0.0f;
        for (uint i = slane; i < lim; i += OAK_CORE) {
            float a = fabs(lp[i]);
            mx = max(mx, a);
            sumAbs += a;
            if (i < cfg.rank) sumAbsValue += a;
        }
        mx = simd_max(mx);
        sumAbs = simd_sum(sumAbs);
        sumAbsValue = simd_sum(sumAbsValue);
        if (firstNF == 0xffffffffu) {
            weightAbsMean = sumAbs / float(cfg.paramStride);
            weightAbsMeanValue = sumAbsValue / float(cfg.rank);
        }
        if (slane == 0u) {
            o->paramAbsMax = mx;
            o->paramNonFinite = (firstNF == 0xffffffffu) ? 0u : 1u;
        }
    } else if (slane == 0u) {
        o->paramAbsMax = -1.0f;
        o->paramNonFinite = 0u;
    }
    {
        Agent agent = make_agent(params, stateArena, uintArena, scalars, cfg, sgidL, sgid, 0u, 0u);
        device OakDiagAccum *acc = &diagAccum[sgid];
        uint births = 0u;
        for (uint i = slane; i < cfg.rank; i += OAK_CORE) {
            rec->featureAge[i] = as_type<float>(agent.featureAge[i]);
            rec->featureUtility[i] = agent.featureUtility[i];
            rec->valueAlpha[i] = agent.valueAlpha[i];
            rec->optionValueAlpha[i] = agent.optionValueAlpha[i];
            rec->modelAlpha[i] = agent.modelAlpha[i];
            births += agent.featureBirthCount[i];
        }
        births = simd_sum(births);
        for (uint g = slane; g < OAK_DIAG_GRAM; g += OAK_CORE) {
            rec->gram[g] = acc->gram[g];
            acc->gram[g] = 0.0f;
        }
        if (slane == 0u) {
            rec->featureBirths = float(births);
            rec->weightAbsMean = weightAbsMean;
            rec->weightAbsMeanValue = weightAbsMeanValue;
        }
    }
    if (slane != 0u) return;
    o->rewardSum = lane->rewardSum;
    o->rho = lane->averageReward;
    o->deathStep = lane->deathStep;
    o->deathDetector = lane->deathDetector;
}
