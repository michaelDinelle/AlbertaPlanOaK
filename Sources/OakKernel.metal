#pragma METAL fp math_mode(safe)

constant uint OAK_SPEC_OBS_DIM [[function_constant(0)]];
constant uint OAK_SPEC_ENV_KIND [[function_constant(2)]];
constant uint OAK_MEGA_G [[function_constant(3)]];
constant bool OAK_PROBE_FIXEDIDX_FC [[function_constant(5)]];
constant bool OAK_PROBE_FIXEDIDX = is_function_constant_defined(OAK_PROBE_FIXEDIDX_FC)
    ? OAK_PROBE_FIXEDIDX_FC : false;
inline uint activeIndex(device const uint *act, uint a) {
    return OAK_PROBE_FIXEDIDX ? a : act[a];
}

inline void simdSyncDev() { simdgroup_barrier(mem_flags::mem_device); }

inline void simdSyncTg() { simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); }

constant uint PLAN_SCRATCH_INV_SQRT_FEATURES = 2u;

constant uint FEATURE_PRODUCT = 0u;
constant uint FEATURE_NORMALIZED_SUM = 1u;
constant uint FEATURE_OBSERVATION_TIMESCALE = 3u;

constant uint RECORD_VALUE = 0u;
constant uint RECORD_AUX = 1u;
constant uint RECORD_REWARD = 2u;
constant uint RECORD_DURATION = 3u;
constant uint RECORD_NEXT = 4u;
constant uint RECORD_NEXT_AUX = 5u;
constant uint RECORD_TWIN = 6u;
constant uint RECORD_POLICY = 7u;
constant uint OAK_RECORD_VEC_MAX = ((RECORD_POLICY + OAK_POLICY_MAX) + 3u) / 4u;

inline float keepFinite(float old_w, float new_w) { return isfinite(new_w) ? new_w : old_w; }

inline float track_rate(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

struct Shape {
    uint observationCount, actionCount, perceptCount, featureCount, optionCount, nurseryCount, traceLevels;
    uint policyCount, recordFloats, envActionCount;
    float invFeatureTimeScale, invOptionCount, metaEta, sampleSigma, invTwoSigmaSquared, actorAlphaDenom;
};

inline Shape shapeFor(uint obsDim, uint envActionCount) {
    Shape s;
    s.observationCount = obsDim;
    s.actionCount = OAK_ACTION_PARAMS;
    s.perceptCount = perceptsForStream(obsDim, OAK_ACTION_PARAMS);
    s.featureCount = featureCountForPercepts(s.perceptCount);
    s.optionCount = OAK_OPTION_COUNT;
    s.nurseryCount = nurseryForPercepts(s.perceptCount);
    s.traceLevels = OAK_CANDIDATES_PER_SLOT;
    s.policyCount = max(OAK_ACTION_PARAMS / 2u, 1u);
    s.recordFloats = ((RECORD_POLICY + s.policyCount) + 3u) & ~3u;
    s.envActionCount = envActionCount;
    uint featureTimeScale = featureTimeScaleForPercepts(s.perceptCount);
    s.invFeatureTimeScale = 1.0f / float(featureTimeScale);
    s.invOptionCount = 1.0f / float(s.optionCount);
    s.metaEta = 1.0f / float(featureTimeScale) / float(max(featureTimeScale + s.actionCount, 1u));
    s.sampleSigma = 1.0f / float(max(s.actionCount, 1u));
    s.invTwoSigmaSquared = 1.0f / (2.0f * s.sampleSigma * s.sampleSigma);
    s.actorAlphaDenom = float(max(s.actionCount * s.optionCount, 1u));
    return s;
}

inline Shape shape() {
    return shapeFor(OAK_SPEC_OBS_DIM, OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionCount);
}

struct Layout {
    uint paramStride, stateStride, uintStride;
    uint option;
    uint observation, observationMean, observationSecondMoment;
    uint observationMeta;
    uint state, policyMean, policyMeanRaw, actionSignal, optionActorCredit, optionPolicyMean, importance;
    uint reward, td, optionTd, featureUtility, optionUtility, bonusWeight, featureMoments;
    uint valueMeta, optionValueMeta, modelMeta;
    uint actorAdvantage, optionRewardTd, optionDurationTd, observationTimescale, optionPriority, continuation, keyboard;
    uint rewardCenter, optionValueScale, valueScale, previousFeatureNormSq;
    uint optionValueDot, optionValueAuxScale, optionAverageReward, optionModelAverageReward, optionFitness;
    uint actorMeta;
    uint importanceStaging, actorCreditStaging, sampleScratch;
    uint predictedReward, predictedDuration, predictedNextValue, twinDisagreement;
    uint optionNextValueTd, optionNextValueDot, optionNextValueScale, optionNextValueAuxScale;
    uint nurseryStats, optionLogit, optionPolicyCoef, optionUseByFeature, nurseryScore, replaceScore, stageScratch, planScratch;
    uint optionFeature, optionAge, featureMeta, featureBirthCount;
    uint selectedOption, lastStopAge, nurseryMeta;
    uint activeFeature, replaceSlot, parked;
};

inline Layout layoutFor(Shape s) {
    Layout L;
    uint F = s.featureCount, K = s.optionCount, A = s.actionCount, N = s.nurseryCount, O = s.observationCount;
    uint p = 0u;
    p += F;
    L.option = p; p += F * K * s.recordFloats;
    L.paramStride = p;

    uint o = 0u;
    L.observation = o; o += O;
    L.observationMean = o; o += O;
    L.observationSecondMoment = o; o += O;
    o = (o + 3u) & ~3u;
    L.observationMeta = o; o += 4u * O;
    L.state = o; o += 2u * F;
    L.policyMean = o; o += A;
    L.policyMeanRaw = o; o += A;
    L.actionSignal = o; o += A;
    L.optionActorCredit = o; o += K * A;
    L.optionPolicyMean = o; o += K * A;
    L.importance = o; o += K;
    L.reward = o; o += 1u;
    L.td = o; o += 1u;
    L.optionTd = o; o += K;
    L.featureUtility = o; o += F;
    L.optionUtility = o; o += K;
    L.bonusWeight = o; o += F;
    o = (o + 3u) & ~3u;
    L.featureMoments = o; o += 2u * F;
    L.valueMeta = o; o += 4u * F;
    L.optionValueMeta = o; o += 4u * F;
    L.modelMeta = o; o += 4u * F;
    L.actorAdvantage = o; o += K;
    L.optionRewardTd = o; o += K;
    L.optionDurationTd = o; o += K;
    L.observationTimescale = o; o += s.traceLevels * O;
    L.optionPriority = o; o += K;
    L.continuation = o; o += K;
    L.keyboard = o; o += K;
    L.rewardCenter = o; o += 1u;
    L.optionValueScale = o; o += K;
    L.valueScale = o; o += 1u;
    L.previousFeatureNormSq = o; o += 1u;
    L.optionValueDot = o; o += K;
    L.optionValueAuxScale = o; o += K;
    L.optionAverageReward = o; o += K;
    L.optionModelAverageReward = o; o += K;
    L.optionFitness = o; o += K;
    o = (o + 3u) & ~3u;
    L.actorMeta = o; o += 4u * K;
    L.importanceStaging = o; o += K;
    L.actorCreditStaging = o; o += K * A;
    L.sampleScratch = o; o += OAK_SAMPLE_SCRATCH;
    L.predictedReward = o; o += K;
    L.predictedDuration = o; o += K;
    L.predictedNextValue = o; o += K;
    L.twinDisagreement = o; o += K;
    L.optionNextValueTd = o; o += K;
    L.optionNextValueDot = o; o += K;
    L.optionNextValueScale = o; o += K;
    L.optionNextValueAuxScale = o; o += K;
    o = (o + 3u) & ~3u;
    L.nurseryStats = o; o += 4u * N;
    L.optionLogit = o; o += K * A;
    L.optionPolicyCoef = o; o += K * s.policyCount;
    L.optionUseByFeature = o; o += F;
    L.nurseryScore = o; o += N;
    o = (o + 3u) & ~3u;
    L.replaceScore = o; o += 4u * F;
    L.stageScratch = o; o += 16u;
    L.planScratch = o; o += 4u;
    L.stateStride = (o + 3u) & ~3u;

    uint u = 0u;
    L.optionFeature = u; u += K;
    L.optionAge = u; u += K;
    L.featureBirthCount = u; u += F;
    L.selectedOption = u; u += 1u;
    L.lastStopAge = u; u += K;
    L.activeFeature = u; u += 2u * activeFeaturesForPercepts(s.perceptCount);
    L.replaceSlot = u; u += 4u;
    L.parked = u; u += K;
    u = (u + 3u) & ~3u;
    L.featureMeta = u; u += 4u * F;
    L.nurseryMeta = u; u += 4u * N;
    L.uintStride = u;
    return L;
}

struct Agent {
    uint observationCount, featureCount, optionCount, actionCount, traceLevels, perceptCount, nurseryCount;
    uint envActionCount, policyCount, recordFloats;
    uint stepCount, stateSlot;
    float invFeatureTimeScale, invOptionCount, metaEta;
    float sampleSigma, invTwoSigmaSquared, actorAlphaDenom;
    device float *valueWeight, *option;
    device float *observation, *observationMean, *observationSecondMoment;
    device float4 *observationMeta;
    device float *state;
    device float *policyMean, *policyMeanRaw, *actionSignal, *optionActorCredit, *optionPolicyMean;
    device float *importance;
    device float *reward, *td, *optionTd;
    device float *featureUtility, *optionUtility, *bonusWeight;
    device float2 *featureMoments;
    device float4 *valueMeta, *optionValueMeta, *modelMeta;
    device float *actorAdvantage, *optionRewardTd, *optionDurationTd;
    device float *observationTimescale, *optionPriority, *continuation, *keyboard;
    device float *rewardCenter, *optionValueScale, *valueScale, *previousFeatureNormSq;
    device float *optionValueDot, *optionValueAuxScale, *optionAverageReward, *optionModelAverageReward, *optionFitness;
    device float4 *actorMeta;
    device float *importanceStaging, *actorCreditStaging, *sampleScratch;
    device float *predictedReward, *predictedDuration, *predictedNextValue, *twinDisagreement;
    device float *optionNextValueTd, *optionNextValueDot, *optionNextValueScale, *optionNextValueAuxScale;
    device float *nurseryStats, *optionLogit, *optionPolicyCoef, *optionUseByFeature, *nurseryScore, *replaceScore;
    device float *stageScratch, *planScratch;
    device uint *optionFeature, *optionAge, *featureBirthCount;
    device uint *selectedOption, *lastStopAge;
    device uint4 *featureMeta, *nurseryMeta;
    device uint *activeFeature, *replaceSlot, *parked;
    device KernelScalars *scalars;
};

inline uint activeCount(uint perceptCount) { return activeFeaturesForPercepts(perceptCount); }

inline Agent make_agent(device float *params, device float *stateArena, device uint *uintArena,
                        device KernelScalars *scalars, Shape s, uint agentID, uint agentGlobal) {
    Layout L = layoutFor(s);
    params += agentID * L.paramStride;
    stateArena += agentID * L.stateStride;
    uintArena += agentID * L.uintStride;
    Agent agent;
    agent.observationCount = s.observationCount;
    agent.featureCount = s.featureCount;
    agent.optionCount = s.optionCount;
    agent.actionCount = s.actionCount;
    agent.traceLevels = s.traceLevels;
    agent.perceptCount = s.perceptCount;
    agent.nurseryCount = s.nurseryCount;
    agent.envActionCount = s.envActionCount;
    agent.policyCount = s.policyCount;
    agent.recordFloats = s.recordFloats;
    agent.stepCount = scalars[agentGlobal].stepCount;
    agent.stateSlot = scalars[agentGlobal].stateSlot;
    agent.invFeatureTimeScale = s.invFeatureTimeScale;
    agent.invOptionCount = s.invOptionCount;
    agent.metaEta = s.metaEta;
    agent.sampleSigma = s.sampleSigma;
    agent.invTwoSigmaSquared = s.invTwoSigmaSquared;
    agent.actorAlphaDenom = s.actorAlphaDenom;
    agent.valueWeight = params;
    agent.option = params + L.option;
    agent.observation = stateArena + L.observation;
    agent.observationMean = stateArena + L.observationMean;
    agent.observationSecondMoment = stateArena + L.observationSecondMoment;
    agent.observationMeta = (device float4 *)(stateArena + L.observationMeta);
    agent.state = stateArena + L.state;
    agent.policyMean = stateArena + L.policyMean;
    agent.policyMeanRaw = stateArena + L.policyMeanRaw;
    agent.actionSignal = stateArena + L.actionSignal;
    agent.optionActorCredit = stateArena + L.optionActorCredit;
    agent.optionPolicyMean = stateArena + L.optionPolicyMean;
    agent.importance = stateArena + L.importance;
    agent.reward = stateArena + L.reward;
    agent.td = stateArena + L.td;
    agent.optionTd = stateArena + L.optionTd;
    agent.featureUtility = stateArena + L.featureUtility;
    agent.optionUtility = stateArena + L.optionUtility;
    agent.bonusWeight = stateArena + L.bonusWeight;
    agent.featureMoments = (device float2 *)(stateArena + L.featureMoments);
    agent.valueMeta = (device float4 *)(stateArena + L.valueMeta);
    agent.optionValueMeta = (device float4 *)(stateArena + L.optionValueMeta);
    agent.modelMeta = (device float4 *)(stateArena + L.modelMeta);
    agent.actorAdvantage = stateArena + L.actorAdvantage;
    agent.optionRewardTd = stateArena + L.optionRewardTd;
    agent.optionDurationTd = stateArena + L.optionDurationTd;
    agent.observationTimescale = stateArena + L.observationTimescale;
    agent.optionPriority = stateArena + L.optionPriority;
    agent.continuation = stateArena + L.continuation;
    agent.keyboard = stateArena + L.keyboard;
    agent.rewardCenter = stateArena + L.rewardCenter;
    agent.optionValueScale = stateArena + L.optionValueScale;
    agent.valueScale = stateArena + L.valueScale;
    agent.previousFeatureNormSq = stateArena + L.previousFeatureNormSq;
    agent.optionValueDot = stateArena + L.optionValueDot;
    agent.optionValueAuxScale = stateArena + L.optionValueAuxScale;
    agent.optionAverageReward = stateArena + L.optionAverageReward;
    agent.optionModelAverageReward = stateArena + L.optionModelAverageReward;
    agent.optionFitness = stateArena + L.optionFitness;
    agent.actorMeta = (device float4 *)(stateArena + L.actorMeta);
    agent.importanceStaging = stateArena + L.importanceStaging;
    agent.actorCreditStaging = stateArena + L.actorCreditStaging;
    agent.sampleScratch = stateArena + L.sampleScratch;
    agent.predictedReward = stateArena + L.predictedReward;
    agent.predictedDuration = stateArena + L.predictedDuration;
    agent.predictedNextValue = stateArena + L.predictedNextValue;
    agent.twinDisagreement = stateArena + L.twinDisagreement;
    agent.optionNextValueTd = stateArena + L.optionNextValueTd;
    agent.optionNextValueDot = stateArena + L.optionNextValueDot;
    agent.optionNextValueScale = stateArena + L.optionNextValueScale;
    agent.optionNextValueAuxScale = stateArena + L.optionNextValueAuxScale;
    agent.nurseryStats = stateArena + L.nurseryStats;
    agent.optionLogit = stateArena + L.optionLogit;
    agent.optionPolicyCoef = stateArena + L.optionPolicyCoef;
    agent.optionUseByFeature = stateArena + L.optionUseByFeature;
    agent.nurseryScore = stateArena + L.nurseryScore;
    agent.replaceScore = stateArena + L.replaceScore;
    agent.stageScratch = stateArena + L.stageScratch;
    agent.planScratch = stateArena + L.planScratch;
    agent.optionFeature = uintArena + L.optionFeature;
    agent.optionAge = uintArena + L.optionAge;
    agent.featureBirthCount = uintArena + L.featureBirthCount;
    agent.selectedOption = uintArena + L.selectedOption;
    agent.lastStopAge = uintArena + L.lastStopAge;
    agent.activeFeature = uintArena + L.activeFeature;
    agent.replaceSlot = uintArena + L.replaceSlot;
    agent.parked = uintArena + L.parked;
    agent.featureMeta = (device uint4 *)(uintArena + L.featureMeta);
    agent.nurseryMeta = (device uint4 *)(uintArena + L.nurseryMeta);
    agent.scalars = scalars + agentGlobal;
    return agent;
}

inline device float *record(Agent agent, uint feature, uint option) {
    return agent.option + (feature * agent.optionCount + option) * agent.recordFloats;
}

inline device float *state_at(Agent agent, uint slot) { return agent.state + slot * agent.featureCount; }

inline uint perceptSourceCount(Agent agent) { return agent.observationCount + agent.envActionCount + 1u; }

struct OakShapeProbe {
    uint featureCount, optionCount, nursery, traceLevels, perceptCount, actionCount;
    uint paramStride, stateStride, uintStride;
    uint pOption, recordFloats, recordValue, recordPolicy;
    uint stUtilOpt, stValueMeta, stOptionValueMeta, stModelMeta, stActorMeta;
    uint uOptFeature, uFeatureMeta, uNurseryMeta;
    uint simdWidth, threadgroupLanes, activeFeatures, featureTimeScale, optionsPerThread, actorAlphaDenom;
    uint laneThreadgroupFloats;
};
static_assert(sizeof(OakShapeProbe) == 112, "OakShapeProbe layout drifted from the Swift mirror");

kernel void oak_shape_probe(device OakShapeProbe *out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    Shape s = shape();
    Layout L = layoutFor(s);
    out->featureCount = s.featureCount;
    out->optionCount = s.optionCount;
    out->nursery = s.nurseryCount;
    out->traceLevels = s.traceLevels;
    out->perceptCount = s.perceptCount;
    out->actionCount = s.actionCount;
    out->paramStride = L.paramStride;
    out->stateStride = L.stateStride;
    out->uintStride = L.uintStride;
    out->pOption = L.option;
    out->recordFloats = s.recordFloats;
    out->recordValue = RECORD_VALUE;
    out->recordPolicy = RECORD_POLICY;
    out->stUtilOpt = L.optionUtility;
    out->stValueMeta = L.valueMeta;
    out->stOptionValueMeta = L.optionValueMeta;
    out->stModelMeta = L.modelMeta;
    out->stActorMeta = L.actorMeta;
    out->uOptFeature = L.optionFeature;
    out->uFeatureMeta = L.featureMeta;
    out->uNurseryMeta = L.nurseryMeta;
    out->simdWidth = OAK_SIMD_WIDTH;
    out->threadgroupLanes = OAK_THREADGROUP_LANES;
    out->activeFeatures = activeCount(s.perceptCount);
    out->featureTimeScale = featureTimeScaleForPercepts(s.perceptCount);
    out->optionsPerThread = OAK_OPTIONS_PER_THREAD;
    out->actorAlphaDenom = uint(s.actorAlphaDenom);
    out->laneThreadgroupFloats = 6u * s.optionCount + 1u;
}

constant uint OAK_ENV_TABLE_STRIDE = 9u + OAK_POSE_LEN;

kernel void oak_env_table_probe(device uint *out [[buffer(0)]],
                                uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    out[0] = OAK_ENV_COUNT;
    out[1] = OAK_ENV_OBS_MAX;
    out[2] = OAK_ENV_ACT_MAX;
    out[3] = OAK_ENV_WORDS_MAX;
    out[4] = OAK_ENV_CONSTANT_WORDS;
    for (uint i = 0u; i < OAK_ENV_COUNT; ++i) {
        device uint *row = out + 5u + i * OAK_ENV_TABLE_STRIDE;
        row[0] = OAK_ENV_SPECS[i].nameHash;
        row[1] = OAK_ENV_SPECS[i].obsDimNatural;
        row[2] = OAK_ENV_SPECS[i].actionKind;
        row[3] = OAK_ENV_SPECS[i].actionCount;
        row[4] = OAK_ENV_SPECS[i].controlHz;
        row[5] = OAK_ENV_SPECS[i].envStateWords;
        row[6] = OAK_ENV_SPECS[i].mjIndex;
        row[7] = OAK_ENV_SPECS[i].laneThreads;
        for (uint s = 0u; s < OAK_POSE_LEN; ++s) row[8u + s] = OAK_ENV_SPECS[i].slots[s];
        row[8u + OAK_POSE_LEN] = OAK_ENV_SPECS[i].persistentEntryWords;
    }
}

kernel void oak_env_constants_init(device float *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) jbw_buildResponse(out);
}

inline device uint *activeFeatures(Agent agent, uint slot) { return agent.activeFeature + slot * activeCount(agent.perceptCount); }

inline uint featureTile(float activation) {
    return uint(clamp(floor(activation + 3.5f), 0.0f, 6.0f));
}

inline uint tileSourceIndex(Agent agent, uint source) {
    return (source < agent.observationCount + agent.envActionCount) ? source
         : agent.observationCount + agent.actionCount;
}

inline uint zoomedFeatureTile(float activation, uint zoomLevel) {
    float zoom = float(1u << min(zoomLevel, 2u));
    return uint(clamp(floor(activation * zoom + 3.5f), 0.0f, 6.0f));
}

inline uint jointFeatureTile(float activationA, float activationB, uint sourceReuse) {
    if (fabs(activationA) >= 3.5f || fabs(activationB) >= 3.5f) return 6u;
    activationA += ((sourceReuse & 1u) == 0u) ? (7.0f / 6.0f) : 0.0f;
    uint tileA = uint(clamp(floor((activationA + 3.5f) * (3.0f / 7.0f)), 0.0f, 2.0f));
    uint tileB = uint(clamp(floor((activationB + 3.5f) * (2.0f / 7.0f)), 0.0f, 1.0f));
    return tileA * 2u + tileB;
}

inline bool wasActive(device const uint *previousActive, uint percepts, uint i) {
    return i < percepts || previousActive[percepts + (i - percepts) / OAK_CANDIDATES_PER_SLOT] == i;
}
inline float heldDuration(float raw) { return isfinite(raw) ? max(1.0f, raw) : 1.0f; }

inline float normalizedObservation(Agent agent, uint observationIndex) {
    float mean = agent.observationMean[observationIndex];
    float secondMoment = agent.observationSecondMoment[observationIndex];
    float var = max(secondMoment - mean * mean, 0.0f);
    return (agent.observation[observationIndex] - mean) * rsqrt(var + 1.0f / float(2u * agent.perceptCount));
}

inline void metaOptimizeUpdate(device float4 *meta, uint i, float signal, float x, uint age, float metaEta, float decay) {
    if (!isfinite(signal) || !isfinite(x)) return;
    float4 old = meta[i];
    float oldAlpha = old.x, oldBeta = old.y, oldH = old.z, oldGradEnergy = old.w;
    float grad = signal * x;
    float metaGradient = grad * oldH;
    float gradEnergyRate = 1.0f / (float(age) + 1.0f);
    float gradEnergy = keepFinite(oldGradEnergy, oldGradEnergy + gradEnergyRate * (metaGradient * metaGradient - oldGradEnergy));
    float ridge = 1.0f / (float(age) + 1.0f);
    float normalizedMetaGradient = metaGradient * rsqrt(max(gradEnergy, 0.0f) + ridge);
    float newBeta = oldBeta + metaEta * normalizedMetaGradient;
    float beta = keepFinite(oldBeta, clamp(newBeta, 2.0f * log(decay), 0.0f));
    float alpha = keepFinite(oldAlpha, exp(beta));
    float featureCurvature = x * x;
    float hDecay = (1.0f - decay) * max(0.0f, 1.0f - oldAlpha * featureCurvature);
    meta[i] = float4(alpha, beta, keepFinite(oldH, hDecay * oldH + oldAlpha * grad), gradEnergy);
}

inline void storeLearnScales(Agent agent, uint tid) {
    device const uint *previousActive = activeFeatures(agent, agent.stateSlot);
    device const float *previousFeatures = state_at(agent, agent.stateSlot);
    float valueStepDenominator = 1.0f;
    float previousFeatureNormSq = 0.0f;
    for (uint base = 0u; base < activeCount(agent.perceptCount); base += OAK_SIMD_WIDTH) {
        uint a = base + tid;
        uint ip = previousActive[a];
        float previousFeature = previousFeatures[ip];
        float previousAlpha = agent.valueMeta[ip].x;
        for (uint t = 0u; t < OAK_SIMD_WIDTH; ++t) {
            float x = simd_shuffle(previousFeature, (ushort)t);
            float previousAlphaBroadcast = simd_shuffle(previousAlpha, (ushort)t);
            valueStepDenominator += previousAlphaBroadcast * (x * x);
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
                              threadgroup float *tgStash,
                              threadgroup float *tgDirty) {
    if (tid == 0u) tgDirty[0] = 0.0f;
    device float *stageScalars = agent.stageScratch;
    device const float *previousFeatures = state_at(agent, agent.stateSlot);
    device const float *currentFeatures = state_at(agent, 1u - agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, agent.stateSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);
    float previousValue = 0.0f, currentValue = 0.0f;
    {
        float previousValuePartial = 0.0f, currentValuePartial = 0.0f;
        for (uint a = tid; a < activeCount(agent.perceptCount); a += OAK_SIMD_WIDTH) {
            uint ip = activeIndex(previousActive, a), ic = activeIndex(currentActive, a);
            previousValuePartial = fma(agent.valueWeight[ip], previousFeatures[ip], previousValuePartial);
            currentValuePartial = fma(agent.valueWeight[ic], currentFeatures[ic], currentValuePartial);
        }
        previousValue = simd_sum(previousValuePartial);
        currentValue = simd_sum(currentValuePartial);
        if (tid == 0u) stageScalars[1] = currentValue;
    }
    simdSyncTg();
    float gauge = agent.rewardCenter[0];
    float previousDifferentialValue = previousValue - gauge;
    float currentDifferentialValue = currentValue - gauge;

    float averageRewardOld = agent.scalars->averageReward;
    float centeredReward = agent.reward[0] - averageRewardOld;
    if (tid == 0u) {
        float delta = centeredReward + currentDifferentialValue - previousDifferentialValue;
        if (!isfinite(delta)) delta = 0.0f;
        agent.td[0] = delta;
        float averageRewardNew = averageRewardOld + agent.invFeatureTimeScale * delta;
        agent.scalars->averageReward = isfinite(averageRewardNew) ? averageRewardNew : averageRewardOld;
    }

    float optionValuePrevEnergy, optionValueCrossEnergy, optionValueNextEnergy;
    {
        float optionValuePrevEnergyPartial = 0.0f, optionValueCrossEnergyPartial = 0.0f, optionValueNextEnergyPartial = 0.0f;
        float modelPrevEnergyPartial = 0.0f, modelCrossEnergyPartial = 0.0f, modelNextEnergyPartial = 0.0f;
        for (uint a = tid; a < activeCount(agent.perceptCount); a += OAK_SIMD_WIDTH) {
            uint ip = activeIndex(previousActive, a), ic = activeIndex(currentActive, a);
            float xp = previousFeatures[ip], xc = currentFeatures[ic];
            float ap = agent.optionValueMeta[ip].x;
            optionValuePrevEnergyPartial = fma(ap * xp, xp, optionValuePrevEnergyPartial);
            optionValueCrossEnergyPartial = fma(ap * xp, currentFeatures[ip], optionValueCrossEnergyPartial);
            float acps = agent.optionValueMeta[ic].x;
            optionValueNextEnergyPartial = fma(acps * xc, xc, optionValueNextEnergyPartial);
            float am = agent.modelMeta[ip].x;
            modelPrevEnergyPartial = fma(am * xp, xp, modelPrevEnergyPartial);
            modelCrossEnergyPartial = fma(am * xp, currentFeatures[ip], modelCrossEnergyPartial);
            float amc = agent.modelMeta[ic].x;
            modelNextEnergyPartial = fma(amc * xc, xc, modelNextEnergyPartial);
        }
        optionValuePrevEnergy = simd_sum(optionValuePrevEnergyPartial);
        optionValueCrossEnergy = simd_sum(optionValueCrossEnergyPartial);
        optionValueNextEnergy = simd_sum(optionValueNextEnergyPartial);
        stageScalars[8] = simd_sum(modelPrevEnergyPartial);
        stageScalars[9] = simd_sum(modelCrossEnergyPartial);
        stageScalars[10] = simd_sum(modelNextEnergyPartial);
    }
    simdSyncTg();
    float observedReward = agent.reward[0];
    if (!isfinite(observedReward)) observedReward = 0.0f;

    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        if (agent.parked[k] == 0u) continue;
        agent.continuation[k] = 0.0f;
        agent.optionTd[k] = 0.0f;
        agent.actorAdvantage[k] = 0.0f;
        agent.keyboard[k] = 0.0f;
        tgCont[k] = 0.0f;
    }
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        bool slotLive = agent.parked[k] == 0u;
        uint slot = k;
        float optionValuePrev = 0.0f, optionValueNext = 0.0f, optionValueAuxDot = 0.0f;
        float rewardHat = 0.0f, durationHat = 0.0f, nextValueHat = 0.0f, twinNextValueHat = 0.0f;
        float previousNextValue = 0.0f, previousNextValueAuxDot = 0.0f, previousRewardHat = 0.0f, previousDurationHat = 0.0f;
        float twinPrevDot = 0.0f;
        float policyLogit[OAK_POLICY_MAX];
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) policyLogit[j] = 0.0f;
        for (uint base = 0u; base < activeCount(agent.perceptCount); base += OAK_SIMD_WIDTH) {
            uint ipH = activeIndex(previousActive, base + tid), icH = activeIndex(currentActive, base + tid);
            float xpH = previousFeatures[ipH], xcH = currentFeatures[icH];
            for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                uint ip = simd_shuffle(ipH, (ushort)l), ic = simd_shuffle(icH, (ushort)l);
                float xp = simd_shuffle(xpH, (ushort)l), xc = simd_shuffle(xcH, (ushort)l);
                if (slotLive && xp != 0.0f) {
                    device const float *rp = record(agent, ip, slot);
                    optionValuePrev = fma(rp[RECORD_VALUE], xp, optionValuePrev);
                    optionValueAuxDot = fma(rp[RECORD_AUX], xp, optionValueAuxDot);
                    previousNextValue = fma(rp[RECORD_NEXT], xp, previousNextValue);
                    previousNextValueAuxDot = fma(rp[RECORD_NEXT_AUX], xp, previousNextValueAuxDot);
                    previousRewardHat = fma(rp[RECORD_REWARD], xp, previousRewardHat);
                    previousDurationHat = fma(rp[RECORD_DURATION], xp, previousDurationHat);
                    twinPrevDot = fma(rp[RECORD_TWIN], xp, twinPrevDot);
                }
                if (slotLive && xc != 0.0f) {
                    device const float *rc = record(agent, ic, slot);
                    optionValueNext = fma(rc[RECORD_VALUE], xc, optionValueNext);
                    rewardHat = fma(rc[RECORD_REWARD], xc, rewardHat);
                    durationHat = fma(rc[RECORD_DURATION], xc, durationHat);
                    nextValueHat = fma(rc[RECORD_NEXT], xc, nextValueHat);
                    twinNextValueHat = fma(rc[RECORD_TWIN], xc, twinNextValueHat);
                    for (uint j = 0u; j < agent.policyCount; ++j) policyLogit[j] = fma(rc[RECORD_POLICY + j], xc, policyLogit[j]);
                }
            }
        }
        if (!slotLive) continue;
        uint optionFeatureIndex = agent.optionFeature[k] % agent.featureCount;
        tgStash[k * 5u + 0u] = previousNextValue;
        tgStash[k * 5u + 1u] = previousNextValueAuxDot;
        tgStash[k * 5u + 2u] = previousRewardHat;
        tgStash[k * 5u + 3u] = previousDurationHat;
        tgStash[k * 5u + 4u] = twinPrevDot;
        for (uint j = 0u; j < agent.policyCount; ++j) agent.optionLogit[k * agent.actionCount + j] = policyLogit[j];

        float stoppingBonus = agent.bonusWeight[optionFeatureIndex] - agent.valueWeight[optionFeatureIndex];
        if (!isfinite(stoppingBonus) || stoppingBonus < 0.0f) stoppingBonus = 0.0f;
        float stoppingValue = currentDifferentialValue + (1.0f + stoppingBonus) * currentFeatures[optionFeatureIndex];
        bool subtaskFinite = isfinite(optionValuePrev) && isfinite(optionValueNext) && isfinite(stoppingValue);

        float stop = (isfinite(optionValueNext) && isfinite(stoppingValue) && stoppingValue >= optionValueNext) ? 1.0f : 0.0f;

        if (k == 0u) stop = 0.0f;
        float continuation = 1.0f - stop;
        agent.continuation[k] = continuation;
        if (stop > 0.0f) agent.lastStopAge[k] = 0u;

        float optionAverageRewardOld = agent.optionAverageReward[k];
        float optionTdError = subtaskFinite
            ? (centeredReward - optionAverageRewardOld + continuation * optionValueNext + stop * stoppingValue - optionValuePrev) : 0.0f;
        if (!isfinite(optionTdError)) optionTdError = 0.0f;

        if (k == 0u) optionTdError = agent.td[0] - optionAverageRewardOld;
        agent.optionTd[k] = optionTdError;
        agent.actorAdvantage[k] = isfinite(optionTdError) ? optionTdError : 0.0f;
        {
            float optionAverageRewardNew = optionAverageRewardOld + agent.invFeatureTimeScale * optionTdError;
            agent.optionAverageReward[k] = keepFinite(optionAverageRewardOld, optionAverageRewardNew);
        }
        float importanceRatio = agent.importance[k];
        {
            float previousOptionFitness = agent.optionFitness[k];
            agent.optionFitness[k] = keepFinite(previousOptionFitness, previousOptionFitness + agent.invFeatureTimeScale * importanceRatio * (observedReward - previousOptionFitness));
        }

        float optionValueBootstrapEnergy = optionValuePrevEnergy - 2.0f * continuation * optionValueCrossEnergy + continuation * continuation * optionValueNextEnergy;
        float denom = fma(importanceRatio, optionValueBootstrapEnergy, 1.0f);

        agent.optionValueScale[k] = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        {
            agent.optionValueDot[k] = isfinite(optionValueAuxDot) ? optionValueAuxDot : 0.0f;

            float optionValueAuxDenom = fma(importanceRatio, optionValuePrevEnergy, 1.0f);
            agent.optionValueAuxScale[k] = (isfinite(optionValueAuxDenom) && optionValueAuxDenom > 0.0f)
                ? (1.0f / optionValueAuxDenom) : 0.0f;
        }
        tgCont[k] = continuation;
        agent.predictedReward[k] = rewardHat;
        agent.predictedDuration[k] = durationHat;
        agent.predictedNextValue[k] = nextValueHat;
        float twinAbsDiff = fabs(nextValueHat - twinNextValueHat);
        agent.twinDisagreement[k] = isfinite(twinAbsDiff) ? twinAbsDiff : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float gaugeRate = track_rate(agent.stepCount, agent.featureCount);
        float gaugeNew = agent.rewardCenter[0] + gaugeRate * (currentValue - agent.rewardCenter[0]);
        agent.rewardCenter[0] = keepFinite(agent.rewardCenter[0], gaugeNew);
    }
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    storeLearnScales(agent, tid);
    uint prevSel = agent.selectedOption[0] % agent.optionCount;
    bool continuing = (prevSel != 0u) && tgCont[prevSel] > 0.0f;
    uint selected = continuing ? prevSel : 0u;
    bool keyboardMass = false;
    for (uint t = tid; t < agent.optionCount; t += OAK_SIMD_WIDTH) keyboardMass = keyboardMass || agent.keyboard[t] > 0.0f;

    bool refreshKeyboard = (!continuing && prevSel != 0u) || !simd_any(keyboardMass);
    if (refreshKeyboard) {
        float bestScore = -FLT_MAX;
        uint probe = 0xffffffffu;
        for (uint t = tid; t < agent.optionCount; t += OAK_SIMD_WIDTH) {
            float twinDis = agent.twinDisagreement[t];
            float predictedDur = heldDuration(agent.predictedDuration[t]);
            float probeScore = twinDis * predictedDur;
            if (isfinite(probeScore) && probeScore > bestScore) { bestScore = probeScore; probe = t; }
        }
        float bestScoreAll = simd_max(bestScore);
        probe = (bestScoreAll == -FLT_MAX) ? 0u : simd_min(bestScore == bestScoreAll ? probe : 0xffffffffu);
        simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) agent.keyboard[k] = (k == probe) ? 1.0f : 0.0f;
    }

    if (tid == 0u) agent.selectedOption[0] = selected;
}

inline float constructedValue(Agent agent, device const float *features, uint parentA, uint parentB, uint kind) {
    kind %= 4u;
    if (kind == FEATURE_OBSERVATION_TIMESCALE)
        return agent.observationTimescale[parentA % (agent.traceLevels * agent.observationCount)];
    float a = features[parentA % agent.featureCount], b = features[parentB % agent.featureCount];
    return kind == FEATURE_PRODUCT ? a * b
        : kind == FEATURE_NORMALIZED_SUM ? (a + b) * 0.7071067811865475f : a * (b - a);
}

inline void constructFeatures(Agent agent, uint tid) {
    float standardizedObservation = 0.0f;
    if (tid < agent.observationCount) {
        uint observationIndex = tid;
        float x = agent.observation[observationIndex];
        if (isfinite(x)) {
            float observationMean = agent.observationMean[observationIndex];
            metaOptimizeUpdate(agent.observationMeta, observationIndex, x - observationMean, 1.0f, min(agent.stepCount, agent.featureCount), agent.metaEta,
                               agent.invFeatureTimeScale);
            float obsRate = max(1.0f / (float(agent.stepCount) + 1.0f), agent.observationMeta[observationIndex].x);
            agent.observationMean[observationIndex] += obsRate * (x - observationMean);
            agent.observationSecondMoment[observationIndex] += obsRate * (x * x - agent.observationSecondMoment[observationIndex]);
            standardizedObservation = normalizedObservation(agent, observationIndex);
            for (uint m = 1u; m <= agent.traceLevels; ++m) {
                uint slot = (m - 1u) * agent.observationCount + observationIndex;
                float rate = 1.0f / float(1u << m);
                agent.observationTimescale[slot] += rate * (standardizedObservation - agent.observationTimescale[slot]);
            }
        } else {
            standardizedObservation = normalizedObservation(agent, observationIndex);
        }
    }
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) agent.importance[k] = agent.importanceStaging[k];
    for (uint j = tid; j < agent.optionCount * agent.actionCount; j += OAK_SIMD_WIDTH)
        agent.optionActorCredit[j] = agent.actorCreditStaging[j];

    device const float *prev = state_at(agent, agent.stateSlot);
    device float *curr = state_at(agent, 1u - agent.stateSlot);
    device uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);

    simdSyncDev();

    for (uint ablk = 0u; ablk < agent.perceptCount; ablk += OAK_SIMD_WIDTH) {
        uint i = ablk + tid;
        float x;
        if (i < agent.observationCount) {
            x = standardizedObservation;
        } else if (i < agent.observationCount + agent.actionCount) {
            x = agent.actionSignal[i - agent.observationCount];
        } else if (i == agent.observationCount + agent.actionCount) {
            x = agent.reward[0];
        } else if (i == agent.observationCount + agent.actionCount + 1u) {
            x = 1.0f;
        } else {
            uint j = i - (agent.observationCount + agent.actionCount + 2u);
            uint lvl = j / max(agent.observationCount, 1u);
            x = (lvl < agent.traceLevels)
                ? agent.observationTimescale[lvl * agent.observationCount + (j - lvl * agent.observationCount)]
                : 0.0f;
        }
        if (!isfinite(x)) x = 0.0f;
        curr[i] = x;
        currentActive[i] = i;
        if (agent.featureMeta[i].w < 0xffffffffu) agent.featureMeta[i].w += 1u;
    }
    simdSyncDev();

    for (uint ablk = 0u; ablk < agent.perceptCount; ablk += OAK_SIMD_WIDTH) {
        uint ai = ablk + tid;
        uint tile;
        uint tile2;
        {
            uint sourceCount = perceptSourceCount(agent);
            uint sourceIndex = ai % sourceCount, occupancy = ai / sourceCount;
            if (occupancy == 0u) {
                tile = featureTile(curr[tileSourceIndex(agent, sourceIndex)]);
            } else if ((ai & 1u) == 0u) {
                tile = zoomedFeatureTile(curr[tileSourceIndex(agent, sourceIndex)], occupancy);
            } else {
                uint channelA = tileSourceIndex(agent, sourceIndex);
                uint channelB = (sourceIndex + occupancy) % agent.observationCount;
                tile = jointFeatureTile(curr[channelA], curr[channelB], occupancy);
            }
            tile2 = zoomedFeatureTile(curr[tileSourceIndex(agent, sourceIndex)], occupancy + 1u);
        }
        for (uint g = 0u; g < OAK_CONSTRUCTED_PER_PERCEPT; ++g) {
            uint groupTile = (g == 0u) ? tile : tile2;
            uint bankLo = agent.perceptCount + (ai + g * agent.perceptCount) * OAK_CANDIDATES_PER_SLOT, bankHi = bankLo + OAK_CANDIDATES_PER_SLOT;
            uint winner = bankLo + groupTile;
            float winningValue = 0.0f;
            for (uint i = bankLo; i < bankHi; ++i) {
                uint4 meta = agent.featureMeta[i];
                float constructed = constructedValue(agent, prev, meta.x, meta.y, meta.z);
                float rate = max(1.0f / (float(meta.w) + 1.0f), agent.invFeatureTimeScale);
                float2 moments = agent.featureMoments[i];
                moments.x += rate * (constructed - moments.x);
                moments.y += rate * (constructed * constructed - moments.y);
                agent.featureMoments[i] = moments;
                if (i == winner) winningValue = constructed;
                curr[i] = 0.0f;
            }
            float2 winnerMoments = agent.featureMoments[winner];
            float var = max(winnerMoments.y - winnerMoments.x * winnerMoments.x, 0.0f);
            float ridge = max(1.0f / (float(agent.featureMeta[winner].w) + 1.0f), 1.0f / float(2u * agent.perceptCount));
            winningValue = (winningValue - winnerMoments.x) * rsqrt(var + ridge);
            curr[winner] = isfinite(winningValue) ? winningValue : 0.0f;
            currentActive[agent.perceptCount + ai + g * agent.perceptCount] = winner;
            if (agent.featureMeta[winner].w < 0xffffffffu) agent.featureMeta[winner].w += 1u;
        }
    }
    simdSyncDev();
}

static inline void planOption(Agent agent, uint k,
                              float previousNextValueDot,
                              float optionNextValueDot,
                              float previousRewardHat,
                              float previousDurationHat) {
    float continuation = agent.continuation[k];
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
    if (modelFinite && isfinite(modelSquaredError)) {
        agent.optionPriority[k] = sqrt(max(modelSquaredError, 0.0f));
    } else {
        float debt = fabs(agent.optionPriority[k]);
        if (debt <= 0.0f) debt = 1.0f;
        agent.optionPriority[k] = -debt;
    }
}

static inline void planAllOptions(Agent agent, uint tid,
                                  threadgroup const float *stash,
                                  bool stashDirty, uint pSlot) {
    device const uint *previousActive = activeFeatures(agent, pSlot);
    device const float *prev = state_at(agent, pSlot);
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        if (agent.parked[k] == 0u) continue;
        agent.optionRewardTd[k] = 0.0f;
        agent.optionDurationTd[k] = 0.0f;
        agent.optionNextValueTd[k] = 0.0f;
        agent.optionNextValueDot[k] = 0.0f;
        agent.optionNextValueScale[k] = 0.0f;
        agent.optionNextValueAuxScale[k] = 0.0f;
        agent.optionPriority[k] = 0.0f;
    }
    simdSyncDev();
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        if (agent.parked[k] != 0u) continue;
        uint slot = k;
        float previousNextValueDot, optionNextValueDot, previousRewardHat, previousDurationHat;
        if (stashDirty) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            for (uint a = 0u; a < activeCount(agent.perceptCount); a += 1u) {
                uint ip = activeIndex(previousActive, a);
                float xp = prev[ip];
                if (xp == 0.0f) continue;
                device const float *rp = record(agent, ip, slot);
                p0 = fma(rp[RECORD_NEXT], xp, p0);
                p1 = fma(rp[RECORD_NEXT_AUX], xp, p1);
                p2 = fma(rp[RECORD_REWARD], xp, p2);
                p3 = fma(rp[RECORD_DURATION], xp, p3);
            }
            previousNextValueDot = p0;
            optionNextValueDot = p1;
            previousRewardHat = p2;
            previousDurationHat = p3;
        } else {
            previousNextValueDot = stash[k * 5u + 0u];
            optionNextValueDot = stash[k * 5u + 1u];
            previousRewardHat = stash[k * 5u + 2u];
            previousDurationHat = stash[k * 5u + 3u];
        }
        planOption(agent, k, previousNextValueDot, optionNextValueDot, previousRewardHat, previousDurationHat);

        float importanceRatio = agent.importance[k];
        float advantage = agent.actorAdvantage[k];
        if (!isfinite(advantage)) advantage = 0.0f;

        metaOptimizeUpdate(agent.actorMeta, k, advantage, 1.0f, agent.optionAge[k], agent.metaEta, agent.invFeatureTimeScale);
        float actorAlpha = agent.actorMeta[k].x / agent.actorAlphaDenom;
        float previousFeatureNormSq = agent.previousFeatureNormSq[0];
        for (uint d = 0u; d < agent.policyCount; ++d) {
            float targetGrad = agent.optionActorCredit[k * agent.actionCount + d];
            float gradScale = importanceRatio * targetGrad;

            float denom = 1.0f + actorAlpha * advantage * advantage * gradScale * gradScale * previousFeatureNormSq;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            float policyCoef = actorAlpha * advantage * gradScale * scale;
            agent.optionPolicyCoef[k * agent.policyCount + d] = isfinite(policyCoef) ? policyCoef : 0.0f;
        }
    }
}

static inline void learnFromTransition(Agent agent, device const float *prev,
                                       device const float *curr, uint tid, uint pSlot,
                                       threadgroup const float *stash, bool stashDirty) {
    device const uint *previousActive = activeFeatures(agent, pSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - pSlot);
    const uint activeFeatureCount = activeCount(agent.perceptCount);
    uint policyN = agent.policyCount;

    for (uint slotBase = 0u; slotBase < agent.optionCount; slotBase += OAK_SIMD_WIDTH) {
        uint slot = slotBase + tid;
        uint k = slot;
        bool optionLive = agent.parked[k] == 0u;
        float importanceRatio = 0.0f, continuation = 0.0f;
        float optionValueAuxDot = 0.0f, optionValueStepScale = 0.0f, td = 0.0f;
        float nextValueAuxDot = 0.0f, nextValueStepScale = 0.0f, nextValueTdError = 0.0f;
        float rewardModelTdError = 0.0f, durationTdError = 0.0f;
        float optionValueAuxResidual = 0.0f, optionValueAuxStepScale = 0.0f;
        float nextValueAuxResidual = 0.0f, nextValueAuxStepScale = 0.0f;
        float twinPrevDot = 0.0f, twinResidual = 0.0f;
        float actorCoefficient[OAK_POLICY_MAX];
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) actorCoefficient[j] = 0.0f;
        {
            float dot = 0.0f;
            if (stashDirty) {
                for (uint base = 0u; base < activeFeatureCount; base += OAK_SIMD_WIDTH) {
                    uint iH = activeIndex(previousActive, base + tid);
                    float xH = prev[iH];
                    for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                        uint i0 = simd_shuffle(iH, (ushort)l);
                        float x0 = simd_shuffle(xH, (ushort)l);
                        if (!optionLive || x0 == 0.0f) continue;
                        dot = fma(record(agent, i0, slot)[RECORD_TWIN], x0, dot);
                    }
                }
            } else if (optionLive) {
                dot = stash[k * 5u + 4u];
            }
            if (!isfinite(dot)) dot = 0.0f;
            twinPrevDot = dot;
        }
        if (optionLive) {
            importanceRatio = agent.importance[k];
            continuation = agent.continuation[k];
            optionValueAuxDot = agent.optionValueDot[k]; optionValueStepScale = agent.optionValueScale[k]; td = agent.optionTd[k];
            nextValueAuxDot = agent.optionNextValueDot[k]; nextValueStepScale = agent.optionNextValueScale[k]; nextValueTdError = agent.optionNextValueTd[k];
            rewardModelTdError = agent.optionRewardTd[k]; durationTdError = agent.optionDurationTd[k];
            optionValueAuxResidual = td - optionValueAuxDot; optionValueAuxStepScale = agent.optionValueAuxScale[k];
            nextValueAuxResidual = nextValueTdError - nextValueAuxDot; nextValueAuxStepScale = agent.optionNextValueAuxScale[k];
            twinResidual = rewardModelTdError - twinPrevDot;
            for (uint j = 0u; j < policyN; ++j) actorCoefficient[j] = agent.optionPolicyCoef[k * policyN + j];
        }
        bool baseOption = k == 0u;
        for (uint cb = 0u; cb < activeFeatureCount; cb += OAK_SIMD_WIDTH) {
            uint aH = cb + tid;
            uint iH = activeIndex(previousActive, aH);
            float xpH = prev[iH], xcH = curr[iH];
            float aPH = agent.optionValueMeta[iH].x, aMH = agent.modelMeta[iH].x;
            uint lEnd = min(OAK_SIMD_WIDTH, activeFeatureCount - cb);
            for (uint l = 0u; l < lEnd; ++l) {
                uint i = simd_shuffle(iH, (ushort)l);
                float xp = simd_shuffle(xpH, (ushort)l);
                float xc = simd_shuffle(xcH, (ushort)l);
                if (xp == 0.0f && xc == 0.0f) continue;
                float aP = simd_shuffle(aPH, (ushort)l);
                float aM = simd_shuffle(aMH, (ushort)l);
                if (!optionLive) continue;
                device float4 *recv = (device float4 *)record(agent, i, slot);
                uint qn = agent.recordFloats >> 2u;
                float4 q[OAK_RECORD_VEC_MAX];
                for (uint v = 0u; v < qn; ++v) q[v] = recv[v];
                thread float *rec = (thread float *)q;
                for (uint j = 0u; j < policyN; ++j) {
                    float w = rec[RECORD_POLICY + j];
                    rec[RECORD_POLICY + j] = keepFinite(w, w + actorCoefficient[j] * xp);
                }
                if (!baseOption) {
                    float optionValueGradient = (continuation > 0.0f) ? (optionValueAuxDot * (xp - continuation * xc)) : (td * xp);
                    float value = rec[RECORD_VALUE], aux = rec[RECORD_AUX];
                    rec[RECORD_VALUE] = keepFinite(value, value + aP * importanceRatio * optionValueGradient * optionValueStepScale);
                    rec[RECORD_AUX] = keepFinite(aux, aux + aP * importanceRatio * optionValueAuxResidual * xp * optionValueAuxStepScale);
                }
                float nextValueGradient = (continuation > 0.0f) ? (nextValueAuxDot * (xp - continuation * xc)) : (nextValueTdError * xp);
                float next = rec[RECORD_NEXT], nextAux = rec[RECORD_NEXT_AUX], duration = rec[RECORD_DURATION], rewardWeight = rec[RECORD_REWARD], twin = rec[RECORD_TWIN];
                rec[RECORD_NEXT] = keepFinite(next, next + aM * importanceRatio * nextValueGradient * nextValueStepScale);
                rec[RECORD_NEXT_AUX] = keepFinite(nextAux, nextAux + aM * importanceRatio * nextValueAuxResidual * xp * nextValueAuxStepScale);
                rec[RECORD_DURATION] = keepFinite(duration, duration + aM * importanceRatio * durationTdError * xp * nextValueStepScale * 0.05f);
                float rewardModelGradient = (continuation > 0.0f) ? (twinPrevDot * (xp - continuation * xc)) : (rewardModelTdError * xp);
                rec[RECORD_REWARD] = keepFinite(rewardWeight, rewardWeight + aM * importanceRatio * rewardModelGradient * nextValueStepScale);
                rec[RECORD_TWIN] = keepFinite(twin, twin + aM * importanceRatio * twinResidual * xp * nextValueAuxStepScale);
                for (uint v = 0u; v < qn; ++v) recv[v] = q[v];
            }
        }

        bool learnsNewFeatures = optionLive && continuation > 0.0f && importanceRatio > 0.0f;
        for (uint wb = 0u; wb < agent.perceptCount; wb += OAK_SIMD_WIDTH) {
            uint newFeatureH = activeIndex(currentActive, agent.perceptCount + wb + tid);
            uint wasPreviouslyActiveH = wasActive(previousActive, agent.perceptCount, newFeatureH) ? 1u : 0u;
            float newFeatureValueH = curr[newFeatureH];
            float newOptionAlphaH = agent.optionValueMeta[newFeatureH].x, newModelAlphaH = agent.modelMeta[newFeatureH].x;
            for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                uint wasPreviouslyActive = simd_shuffle(wasPreviouslyActiveH, (ushort)l);
                if (wasPreviouslyActive != 0u) continue;
                uint i = simd_shuffle(newFeatureH, (ushort)l);
                float xc = simd_shuffle(newFeatureValueH, (ushort)l);
                float aP = simd_shuffle(newOptionAlphaH, (ushort)l);
                float aM = simd_shuffle(newModelAlphaH, (ushort)l);
                if (learnsNewFeatures) {
                    device float *rec = record(agent, i, slot);
                    if (!baseOption) {
                        float value = rec[RECORD_VALUE];
                        rec[RECORD_VALUE] = keepFinite(value, value + aP * importanceRatio * (-optionValueAuxDot * continuation * xc) * optionValueStepScale);
                    }
                    float next = rec[RECORD_NEXT], rewardWeight = rec[RECORD_REWARD];
                    rec[RECORD_NEXT] = keepFinite(next, next + aM * importanceRatio * (-nextValueAuxDot * continuation * xc) * nextValueStepScale);
                    rec[RECORD_REWARD] = keepFinite(rewardWeight, rewardWeight + aM * importanceRatio * (-twinPrevDot * continuation * xc) * nextValueStepScale);
                }
            }
        }
    }

    float valueTdError = agent.td[0];
    float valueStepScale = agent.valueScale[0];
    for (uint a = 0u; a < activeFeatureCount; ++a) {
        uint i = activeIndex(previousActive, a);
        if (i % OAK_SIMD_WIDTH != tid) continue;
        float w = agent.valueWeight[i];
        float delta = valueTdError * prev[i] * valueStepScale;
        agent.valueWeight[i] = keepFinite(w, w + agent.valueMeta[i].x * delta);
    }
}

inline void emitOptionActions(Agent agent, uint tid, bool recomputeLogits) {
    uint policyN = agent.policyCount;
    device const uint *active = activeFeatures(agent, agent.stateSlot);
    device const float *currS = state_at(agent, agent.stateSlot);
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        if (agent.parked[k] == 0u) continue;
        for (uint j = 0u; j < agent.actionCount; ++j) {
            agent.optionPolicyMean[k * agent.actionCount + j] = 0.0f;
            agent.optionLogit[k * agent.actionCount + j] = 0.0f;
        }
    }
    simdSyncDev();
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        bool slotLive = agent.parked[k] == 0u;
        uint slot = k;
        float policyLogit[OAK_POLICY_MAX];
        for (uint j = 0u; j < policyN; ++j) policyLogit[j] = 0.0f;
        if (recomputeLogits) {
            for (uint base = 0u; base < activeCount(agent.perceptCount); base += OAK_SIMD_WIDTH) {
                uint iH = activeIndex(active, base + tid);
                float xH = currS[iH];
                for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                    uint i = simd_shuffle(iH, (ushort)l);
                    float x = simd_shuffle(xH, (ushort)l);
                    if (!slotLive || x == 0.0f) continue;
                    device const float *row = record(agent, i, slot) + RECORD_POLICY;
                    for (uint j = 0u; j < policyN; ++j) policyLogit[j] = fma(row[j], x, policyLogit[j]);
                }
            }
        } else if (slotLive) {
            for (uint j = 0u; j < policyN; ++j) policyLogit[j] = agent.optionLogit[k * agent.actionCount + j];
        }
        if (!slotLive) continue;
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
    agent.policyMean[d] = (keyboardNorm > 0.0f) ? blendedMean / keyboardNorm : selectedMean;
    agent.policyMeanRaw[d] = (keyboardNorm > 0.0f) ? blendedLogit / keyboardNorm : selectedLogit;
}

inline void zeroOptionColumn(Agent agent, uint tid, uint k) {
    uint slot = k;
    for (uint i = tid; i < agent.featureCount; i += OAK_SIMD_WIDTH) {
        device float *rec = record(agent, i, slot);
        rec[RECORD_VALUE] = 0.0f;
        rec[RECORD_AUX] = 0.0f;
        rec[RECORD_REWARD] = 0.0f;
        rec[RECORD_DURATION] = 0.0f;
        rec[RECORD_NEXT] = 0.0f;
        rec[RECORD_NEXT_AUX] = 0.0f;
    }
}

static inline void replaceWorstFeature(Agent agent, uint tid, threadgroup float *tgDirty) {
    device const float *curr = state_at(agent, 1u - agent.stateSlot);
    if (tid == 0u) agent.planScratch[0] = agent.reward[0];
    simdgroup_barrier(mem_flags::mem_device);

    uint bestNursery;
    float bestNurseryBound;
    {
        float bestBound = -FLT_MAX;
        uint bestIndex = 0xffffffffu;
        for (uint n = tid; n < agent.nurseryCount; n += OAK_SIMD_WIDTH) {
            float lowerConfidenceBound = agent.nurseryScore[n];
            if (lowerConfidenceBound > bestBound) { bestBound = lowerConfidenceBound; bestIndex = n; }
        }
        bestNurseryBound = simd_max(bestBound);
        bestNursery = simd_min((bestBound == bestNurseryBound) ? bestIndex : 0xffffffffu);
    }

    uint haveWorst = 0u, worstFeature = 0u, worstAge = 0u;
    float worstUtility = 0.0f, worstRedundancy = 0.0f, worstDownstream = 0.0f;
    uint eligiblePartial = 0u;
    float eligibleUtilPartial = 0.0f, eligibleRedundancyPartial = 0.0f, eligibleDownstreamPartial = 0.0f;
    for (uint i = agent.perceptCount + tid; i < agent.featureCount; i += OAK_SIMD_WIDTH) {
        float4 e = *(device const float4 *)&agent.replaceScore[i * 4u];
        if (e.w <= 0.0f) continue;
        eligiblePartial += 1u;
        eligibleUtilPartial += e.x;
        eligibleRedundancyPartial += e.y;
        eligibleDownstreamPartial += e.z;
        float util = e.x, redundancy = e.y, downstream = e.z;
        uint age = agent.featureMeta[i].w;
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

    bool finalist = haveWorst != 0u;
    float downstream = simd_min(finalist ? worstDownstream : FLT_MAX);
    finalist = finalist && worstDownstream == downstream;
    float utility = simd_min(finalist ? worstUtility : FLT_MAX);
    finalist = finalist && worstUtility == utility;
    float redundancy = simd_max(finalist ? worstRedundancy : -FLT_MAX);
    finalist = finalist && worstRedundancy == redundancy;
    uint age = simd_max(finalist ? worstAge : 0u);
    worstFeature = simd_min(finalist && worstAge == age ? worstFeature : 0xffffffffu);
    haveWorst = worstFeature != 0xffffffffu;
    worstDownstream = haveWorst ? downstream : 0.0f;
    worstUtility = haveWorst ? utility : 0.0f;
    worstRedundancy = haveWorst ? redundancy : 0.0f;

    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        if (agent.optionAge[k] < 0xffffffffu) agent.optionAge[k] += 1u;
        if (agent.lastStopAge[k] < 0xffffffffu) agent.lastStopAge[k] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);
    uint candidateStart = agent.perceptCount;
    uint eligible = simd_sum(eligiblePartial);
    float eligibleUtilSum = simd_sum(eligibleUtilPartial);
    float eligibleRedundancySum = simd_sum(eligibleRedundancyPartial);
    float eligibleDownstreamSum = simd_sum(eligibleDownstreamPartial);
    uint worst = (haveWorst != 0u) ? worstFeature : candidateStart;

    {
        uint staleLimit = staleOptionSteps(agent.featureCount, agent.optionCount);
        uint parkedStale = 0xffffffffu, unparkedStale = 0xffffffffu;
        for (uint k2 = 1u + tid; k2 < agent.optionCount; k2 += OAK_SIMD_WIDTH) {
            if (agent.lastStopAge[k2] < staleLimit) continue;
            if (agent.parked[k2] != 0u) parkedStale = min(parkedStale, k2);
            else unparkedStale = min(unparkedStale, k2);
        }
        parkedStale = simd_min(parkedStale);
        unparkedStale = simd_min(unparkedStale);
        uint staleOption = (parkedStale != 0xffffffffu) ? parkedStale
            : ((unparkedStale != 0xffffffffu) ? unparkedStale : 0u);
        if (staleOption != 0u) {
            uint k = staleOption;
            zeroOptionColumn(agent, tid, k);
            if (tid == 0u) {
                tgDirty[0] = 1.0f;
                agent.optionAge[k] = 0u;
                agent.lastStopAge[k] = 0u;
                agent.optionUtility[k] = float(agent.featureCount);
                agent.actorAdvantage[k] = 0.0f;
                agent.optionAverageReward[k] = 0.0f;
                agent.optionModelAverageReward[k] = 0.0f;
                agent.parked[k] = 0u;
                uint feat = agent.optionFeature[k] % agent.featureCount;
                record(agent, feat, k)[RECORD_VALUE] = agent.planScratch[PLAN_SCRATCH_INV_SQRT_FEATURES];
            }
        }
    }
    simdSyncDev();

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRedundancy = eligibleRedundancySum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtility < meanEligibleUtil;
    bool highRedundancy = worstRedundancy > meanEligibleRedundancy;
    bool lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
    bool structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);

    bool doReplace = structurallyReplaceable && worst < agent.featureCount
        && bestNursery != 0xffffffffu
        && bestNurseryBound > worstUtility + worstDownstream;
    if (doReplace) {
        uint featureLimit = agent.featureCount;
        float bestA = -FLT_MAX;
        uint bestAIndex = 0u;
        for (uint i = tid; i < featureLimit; i += OAK_SIMD_WIDTH) {
            if (i == worst) continue;
            float parentUtility = agent.featureUtility[i];
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestA) { bestA = parentUtility; bestAIndex = i; }
        }
        float bestAAll = simd_max(bestA);
        uint seedParentA = (bestAAll == -FLT_MAX) ? 0u
            : simd_min(bestA == bestAAll ? bestAIndex : 0xffffffffu);
        float bestB = -FLT_MAX;
        uint bestBIndex = 0u;
        for (uint i = tid; i < featureLimit; i += OAK_SIMD_WIDTH) {
            if (i == worst || i == seedParentA) continue;
            float decor = 1.0f / (1.0f + fabs(curr[i] * curr[seedParentA]));
            float parentUtility = agent.featureUtility[i] * decor;
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestB) { bestB = parentUtility; bestBIndex = i; }
        }
        float bestBAll = simd_max(bestB);
        uint seedParentB = (bestBAll == -FLT_MAX) ? ((agent.featureCount > 1u) ? 1u : 0u)
            : simd_min(bestB == bestBAll ? bestBIndex : 0xffffffffu);
        uint trackingOption = 0xffffffffu, leastDemandOption = 0xffffffffu;
        float leastDemand = FLT_MAX;
        for (uint ok = 1u + tid; ok < agent.optionCount; ok += OAK_SIMD_WIDTH) {
            float use = agent.optionUtility[ok] + agent.optionPriority[ok];
            if (!isfinite(use)) use = FLT_MAX;
            if (agent.optionFeature[ok] == worst) trackingOption = min(trackingOption, ok);
            if (use < leastDemand) { leastDemand = use; leastDemandOption = ok; }
        }
        trackingOption = simd_min(trackingOption);
        float leastDemandAll = simd_min(leastDemand);
        uint replacedOption = (trackingOption != 0xffffffffu) ? trackingOption
            : ((leastDemandAll == FLT_MAX) ? 1u : simd_min(leastDemand == leastDemandAll ? leastDemandOption : 0xffffffffu));
        if (tid == 0u) {
            uint4 nursery = agent.nurseryMeta[bestNursery];
            uint birth = agent.featureBirthCount[worst] + 1u;
            agent.featureBirthCount[worst] = birth;
            agent.featureMeta[worst] = uint4(nursery.x % agent.featureCount, nursery.y % agent.featureCount, nursery.z % 4u, nursery.w);
            agent.featureMoments[worst] = float2(agent.nurseryStats[(bestNursery) * 4u], agent.nurseryStats[(bestNursery) * 4u + 1u]);
            agent.featureUtility[worst] = agent.nurseryStats[(bestNursery) * 4u + 2u];
            agent.valueWeight[worst] = 0.0f; agent.bonusWeight[worst] = 0.0f;

            agent.nurseryMeta[bestNursery] = uint4(seedParentA, seedParentB, (nursery.z + 1u) % 4u, 0u);
            agent.nurseryStats[(bestNursery) * 4u] = 0.0f;
            agent.nurseryStats[(bestNursery) * 4u + 1u] = 0.0f;
            agent.nurseryStats[(bestNursery) * 4u + 2u] = 0.0f;
            agent.nurseryStats[(bestNursery) * 4u + 3u] = 0.0f;

            float initialStepSize = 1.0f / float(activeCount(agent.perceptCount));
            float4 freshMeta = float4(initialStepSize, log(initialStepSize), 0.0f, 0.0f);
            agent.valueMeta[worst] = freshMeta;
            agent.optionValueMeta[worst] = freshMeta;
            agent.modelMeta[worst] = freshMeta;

            uint k = replacedOption;
            agent.optionAge[k] = 0u; agent.optionFeature[k] = worst; agent.optionUtility[k] = float(agent.featureCount);
            agent.actorAdvantage[k] = 0.0f;
            agent.optionAverageReward[k] = 0.0f;
            agent.optionModelAverageReward[k] = 0.0f;
            agent.optionFitness[k] = 0.0f;
            agent.lastStopAge[k] = 0u;

            agent.replaceSlot[0] = 1u; agent.replaceSlot[1] = worst;
            agent.replaceSlot[2] = k; agent.replaceSlot[3] = birth;
        }
    } else if (tid == 0u) {
        agent.replaceSlot[0] = 0u;
    }
    if (tid == 0u) {
        if (agent.replaceSlot[0] == 0u && (agent.stepCount % max(agent.optionCount, 1u)) == 0u) {
            float meanOptionFitness = 0.0f;
            for (uint k2 = 0u; k2 < agent.optionCount; ++k2) meanOptionFitness += agent.optionFitness[k2];
            meanOptionFitness *= agent.invOptionCount;
            uint replacedOption = 0u; float worstFitness = FLT_MAX;
            for (uint k2 = 1u; k2 < agent.optionCount; ++k2) {
                if (agent.optionAge[k2] < agent.featureCount) continue;
                float f = agent.optionFitness[k2];
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
    uint replacement = agent.replaceSlot[0];
    if (replacement != 0u) {
        uint worst = agent.replaceSlot[1], k = agent.replaceSlot[2], birth = agent.replaceSlot[3];
        float policyPerturbScale = 1.0f / float(max(agent.actionCount, 1u));
        uint policyN = agent.policyCount;

        if (tid == 0u) {
            tgDirty[0] = 1.0f;
            agent.parked[k] = 0u;
        }
        uint slot = k;
        uint baseSlot = 0u;
        for (uint i = tid; i < agent.featureCount; i += OAK_SIMD_WIDTH) {
            device float *rec = record(agent, i, slot);
            rec[RECORD_VALUE] = 0.0f;
            rec[RECORD_AUX] = 0.0f;
            rec[RECORD_REWARD] = 0.0f;
            rec[RECORD_DURATION] = 0.0f;
            rec[RECORD_NEXT] = 0.0f;
            rec[RECORD_NEXT_AUX] = 0.0f;
            device const float *base = record(agent, i, baseSlot);
            for (uint d = 0u; d < policyN; ++d) {
                uint h = (i * 2654435761u) ^ (d * 40503u)
                       ^ (k * 2246822519u) ^ (birth * 3266489917u)
                       ^ agent.stepCount;
                h ^= h >> 13; h *= 2654435761u; h ^= h >> 16;
                float hashedNoise = float(h & 0xffffu) / 32768.0f - 1.0f;

                float w0 = base[RECORD_POLICY + d];
                rec[RECORD_POLICY + d] = w0 + hashedNoise * policyPerturbScale * rsqrt(float(activeCount(agent.perceptCount)));
            }
        }
        simdgroup_barrier(mem_flags::mem_device);

        if (replacement == 1u) for (uint ok = tid; ok < agent.optionCount; ok += OAK_SIMD_WIDTH) {
            device float *rec = record(agent, worst, ok);
            for (uint f = 0u; f < agent.recordFloats; ++f) rec[f] = 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_device);
        if (tid == 0u) {
            uint optionFeatureIndex = (replacement == 1u) ? worst : (agent.optionFeature[k] % agent.featureCount);
            record(agent, optionFeatureIndex, slot)[RECORD_VALUE] = agent.planScratch[PLAN_SCRATCH_INV_SQRT_FEATURES];
        }
    }
}

static inline void updateUtilities(Agent agent, uint tid) {
    uint tPrev = agent.stepCount - 1u;
    device const float *prev = state_at(agent, 1u - agent.stateSlot);
    device const float *curr = state_at(agent, agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, 1u - agent.stateSlot);
    float trackingRate = track_rate(tPrev, agent.featureCount);

    const uint activeSlotsPerThread = activeCount(agent.perceptCount) / OAK_SIMD_WIDTH;
    uint iS[OAK_ACTIVE_SLOTS_MAX];
    float optionTdMagnitude[OAK_ACTIVE_SLOTS_MAX], keyboardSearchSignal[OAK_ACTIVE_SLOTS_MAX];
    for (uint s = 0u; s < activeSlotsPerThread; s += 1u) {
        iS[s] = previousActive[tid + s * OAK_SIMD_WIDTH];
        optionTdMagnitude[s] = 0.0f; keyboardSearchSignal[s] = 0.0f;
    }
    float valueTdError = agent.td[0];
    for (uint m = 0u; m < agent.optionCount; m += OAK_SIMD_WIDTH) {
        uint kH = m + tid;
        bool kLive = kH < agent.optionCount;
        uint   ofH = kLive ? agent.optionFeature[kH] : 0u;
        float tdH = kLive ? agent.optionTd[kH] : 0.0f;
        float kbH = kLive ? agent.keyboard[kH] : 0.0f;
        float aaH = kLive ? agent.actorAdvantage[kH] : 0.0f;
        float imH = kLive ? agent.importance[kH] : 0.0f;
        uint kEnd = min(OAK_SIMD_WIDTH, agent.optionCount - m);
        for (uint l = 0u; l < kEnd; ++l) {
            uint   ofV = simd_shuffle(ofH, (ushort)l);
            float tdV = simd_shuffle(tdH, (ushort)l);
            float kbV = simd_shuffle(kbH, (ushort)l);
            float aaV = simd_shuffle(aaH, (ushort)l);
            float imV = simd_shuffle(imH, (ushort)l);
            uint ofFeature = ofV % agent.featureCount;
            for (uint s = 0u; s < activeSlotsPerThread; s += 1u) {
                if (ofFeature != iS[s]) continue;
                float optionTdUse = fabs(tdV);
                float keyboardMass = fabs(kbV);
                float keyboardUse = keyboardMass * (fabs(aaV) + max(imV, 0.0f));
                if (isfinite(optionTdUse)) optionTdMagnitude[s] += optionTdUse;
                if (isfinite(keyboardUse)) keyboardSearchSignal[s] += keyboardUse;
            }
        }
    }
    uint sel = agent.selectedOption[0] % agent.optionCount;
    uint selectedSlot = sel;
    for (uint s = 0u; s < activeSlotsPerThread; s += 1u) {
        uint activeSlot = tid + s * OAK_SIMD_WIDTH;
        uint i = iS[s];
        float optionValueSignal = optionTdMagnitude[s], keyboardSignal = keyboardSearchSignal[s];
        float planningSignal = fabs(valueTdError);

        float selectedPolicyLoad = 0.0f;
        device const float *selectedPolicy = record(agent, i, selectedSlot) + RECORD_POLICY;
        for (uint d = 0u; d < agent.policyCount; ++d) selectedPolicyLoad += fabs(selectedPolicy[d]);
        float load = fabs(prev[i]) * (fabs(agent.valueWeight[i]) + selectedPolicyLoad);
        if (!isfinite(load)) load = 0.0f;
        float featureUtilitySignal = fabs(planningSignal * prev[i])
                + fabs((optionValueSignal + keyboardSignal) * prev[i])
                + load;
        agent.featureUtility[i] += trackingRate * (featureUtilitySignal - agent.featureUtility[i]);

        {
            float bonusHighWater = agent.bonusWeight[i];
            float valueWeight = agent.valueWeight[i];
            if (valueWeight > bonusHighWater) agent.bonusWeight[i] = valueWeight;
        }
        uint age = agent.featureMeta[i].w;
        metaOptimizeUpdate(agent.valueMeta, i, valueTdError, prev[i], age, agent.metaEta, agent.invFeatureTimeScale);
        uint metaOptimizeSlot = tPrev % OAK_SIMD_WIDTH;
        if (activeSlot == metaOptimizeSlot) {
            uint metaOptimizeOption = tPrev % max(agent.optionCount, 1u);
            float optionTdSigned = agent.optionTd[metaOptimizeOption];
            float modelSigned = agent.optionRewardTd[metaOptimizeOption];
            metaOptimizeUpdate(agent.optionValueMeta, i, optionTdSigned, prev[i], age, agent.metaEta, agent.invFeatureTimeScale);
            metaOptimizeUpdate(agent.modelMeta, i, modelSigned, prev[i], age, agent.metaEta, agent.invFeatureTimeScale);
        }
    }
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        float err = agent.optionPriority[k];
        float keyboardMass = fabs(agent.keyboard[k]);
        float optionUtilitySignal = agent.optionTd[k] * agent.optionTd[k]
                + keyboardMass * fabs(agent.actorAdvantage[k])
                - err;
        agent.optionUtility[k] += trackingRate * (optionUtilitySignal - agent.optionUtility[k]);
    }

    {
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
        uint revivedOption = 0xffffffffu;
        for (uint k = 1u + tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
            if (agent.parked[k] == 0u) continue;
            float u = agent.optionUtility[k];
            bool revived = (k == selectedForPark) || (agent.keyboard[k] > 0.0f)
                || (isfinite(u) && u >= meanLiveUtil);
            if (revived) revivedOption = min(revivedOption, k);
        }
        revivedOption = simd_min(revivedOption);
        uint optionToPark = 0xffffffffu;
        float parkUtility = FLT_MAX;
        for (uint k = 1u + tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
            bool live = agent.parked[k] == 0u || k == revivedOption;
            if (!live || agent.optionAge[k] < agent.featureCount || k == selectedForPark) continue;
            float kb = agent.keyboard[k];
            if (kb > 0.0f) continue;
            float u = agent.optionUtility[k];
            if (!isfinite(u)) u = 0.0f;
            if (u < meanLiveUtil && u < parkUtility) { optionToPark = k; parkUtility = u; }
        }
        float parkUtilityAll = simd_min(parkUtility);
        optionToPark = (parkUtilityAll == FLT_MAX) ? 0xffffffffu
            : simd_min(parkUtility == parkUtilityAll ? optionToPark : 0xffffffffu);
        if (tid == 0u) {
            if (revivedOption != 0xffffffffu) agent.parked[revivedOption] = 0u;
            if (optionToPark != 0xffffffffu) agent.parked[optionToPark] = 1u;
        }
    }
    simdSyncDev();

    {
        float meanOptionFitness = 0.0f;
        for (uint k = 0u; k < agent.optionCount; ++k) meanOptionFitness += agent.optionFitness[k];
        meanOptionFitness *= agent.invOptionCount;
        float previousFeatureNormSq = agent.previousFeatureNormSq[0];
        if (!isfinite(previousFeatureNormSq) || previousFeatureNormSq < 0.0f) previousFeatureNormSq = 0.0f;
        float rate = agent.invFeatureTimeScale * agent.invOptionCount / (1.0f + previousFeatureNormSq);

        const uint policyN = agent.policyCount;
        float fitnessSearchDirection = 0.0f;
        float baseOptionMean = 0.0f;
        if (tid < policyN) {
            baseOptionMean = agent.optionPolicyMean[tid];
            if (!isfinite(baseOptionMean)) baseOptionMean = 0.0f;
        }
        float fitnessDirection = 0.0f;
        for (uint k = 1u; k < agent.optionCount; ++k) {
            float f = agent.optionFitness[k];
            if (agent.parked[k] != 0u) continue;
            if (tid < policyN) {
                float fitnessCentered = f - meanOptionFitness;
                float mk = agent.optionPolicyMean[k * agent.actionCount + tid];
                fitnessDirection = fma(fitnessCentered, (isfinite(mk) ? mk : 0.0f) - baseOptionMean, fitnessDirection);
            }
        }
        if (tid < policyN)
            fitnessSearchDirection = isfinite(fitnessDirection) ? (fitnessDirection / (1.0f + fabs(fitnessDirection))) : 0.0f;
        uint activeCnt = activeCount(agent.perceptCount);
        float direction[OAK_POLICY_MAX];
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d) direction[d] = simd_shuffle(fitnessSearchDirection, (ushort)d);
        for (uint a = tid; a < activeCnt; a += OAK_SIMD_WIDTH) {
            uint i = previousActive[a];
            float x = prev[i];
            device float *weights = record(agent, i, 0u) + RECORD_POLICY;
            for (uint d = 0u; d < policyN; ++d) {
                float w0 = weights[d];
                weights[d] = keepFinite(w0, w0 + rate * direction[d] * x);
            }
        }
    }
    {
        for (uint n = tid + (((tPrev >> 1u) & 1u) != 0u ? OAK_SIMD_WIDTH : 0u); n < agent.nurseryCount; n += 2u * OAK_SIMD_WIDTH) {
            uint4 nursery = agent.nurseryMeta[n];
            float constructed = constructedValue(agent, prev, nursery.x, nursery.y, nursery.z);
            uint age = nursery.w;
            float rate = max(1.0f / (float(age) + 1.0f), agent.invFeatureTimeScale);

            device float4 *nurseryStatsVec = (device float4 *)&agent.nurseryStats[(n) * 4u];
            float4 st = *nurseryStatsVec;
            st.x += rate * (constructed - st.x);
            st.y += rate * (constructed * constructed - st.y);
            float var = max(st.y - st.x * st.x, 0.0f);

            float ridge = max(1.0f / (float(age) + 1.0f), 1.0f / float(2u * agent.perceptCount));
            float standardizedCandidate = (constructed - st.x) * rsqrt(var + ridge);
            float candidateUtilitySignal = fabs(agent.td[0] * standardizedCandidate);
            if (isfinite(candidateUtilitySignal)) {
                st.z += trackingRate * (candidateUtilitySignal - st.z);
                st.w += trackingRate * (candidateUtilitySignal * candidateUtilitySignal - st.w);
            }
            *nurseryStatsVec = st;
            if (age < 0xffffffffu) agent.nurseryMeta[n].w = age + 1u;
        }
    }
    {
        for (uint i = tid; i < agent.featureCount; i += OAK_SIMD_WIDTH) agent.optionUseByFeature[i] = 0.0f;
        simdSyncDev();
        if (tid == 0u) {
            for (uint k = 0u; k < agent.optionCount; ++k) {
                float use = agent.optionUtility[k] + fabs(agent.keyboard[k]);
                if (isfinite(use)) agent.optionUseByFeature[agent.optionFeature[k] % agent.featureCount] += use;
            }
        }

        for (uint n = tid; n < agent.nurseryCount; n += OAK_SIMD_WIDTH) {
            float nurseryUtility = agent.nurseryStats[(n) * 4u + 2u];
            float lowerConfidenceBound = -FLT_MAX;
            uint nurseryAge = agent.nurseryMeta[n].w;
            if (nurseryAge >= agent.featureCount && isfinite(nurseryUtility)) {
                float nurseryUtilityVariance = max(agent.nurseryStats[(n) * 4u + 3u] - nurseryUtility * nurseryUtility, 0.0f);
                float nurseryUtilityRate = max(1.0f / (float(nurseryAge) + 1.0f), agent.invFeatureTimeScale);
                float v = nurseryUtility - sqrt(nurseryUtilityVariance * nurseryUtilityRate);
                if (isfinite(v)) lowerConfidenceBound = v;
            }
            agent.nurseryScore[n] = lowerConfidenceBound;
        }
    }
    simdSyncDev();
    {
        for (uint i = agent.perceptCount + tid + (((tPrev >> 1u) & 1u) != 0u ? OAK_SIMD_WIDTH : 0u);
             i < agent.featureCount; i += 2u * OAK_SIMD_WIDTH) {
            device float4 *replaceScoreVec = (device float4 *)&agent.replaceScore[i * 4u];
            uint4 meta = agent.featureMeta[i];
            if (meta.w < agent.featureCount) { *replaceScoreVec = float4(0.0f, 0.0f, 0.0f, 0.0f); continue; }
            uint parentA = meta.x % agent.featureCount;
            float util = agent.featureUtility[i];
            if (!isfinite(util)) util = 0.0f;
            float redundancy = fabs(curr[i] * curr[parentA]);
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
                           device OakFleetLane *lanes [[buffer(6)]],
                           device const uint *laneSeeds [[buffer(7)]],
                           constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                           constant uint &laneOffset [[buffer(10)]],
                           device uint *envArena [[buffer(14)]],
                           device const uint *envOff [[buffer(15)]],
                           constant uint *mjModels [[buffer(16)]],
                           device const float *envConstants [[buffer(19)]],
                           uint gid [[thread_position_in_grid]]) {
    uint gidLocal = gid;
    gid += laneOffset;
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    device uint *envState = laneEnvState(envArena, envOff, gid);
    resetLane(lane, envState, mjModels, OAK_SPEC_ENV_KIND, laneSeeds[gid]);
    Agent agent = make_agent(params, stateArena, uintArena, scalars, shape(), gidLocal, gid);
    for (uint o = 0u; o < agent.observationCount; ++o) {
        agent.observationMeta[o] = float4(1.0f / float(2u * agent.perceptCount), log(1.0f / float(2u * agent.perceptCount)), 0.0f, 0.0f);
    }
    observeLane(OAK_SPEC_ENV_KIND, laneSeeds[gid], mjModels, envConstants, ecfg, envState, agent.observation);
    agent.reward[0] = 0.0f;
    for (uint d = 0u; d < agent.actionCount; ++d) agent.actionSignal[d] = 0.0f;
    for (uint i = 0u; i < agent.optionCount * agent.actionCount; ++i) agent.optionActorCredit[i] = 0.0f;
    for (uint k = 0u; k < agent.optionCount; ++k) {
        agent.importance[k] = 1.0f;
        agent.importanceStaging[k] = 1.0f;
        agent.parked[k] = 0u;
    }
    agent.planScratch[PLAN_SCRATCH_INV_SQRT_FEATURES] = rsqrt(float(max(agent.featureCount, 1u)));
}

static inline void sampleDiscreteAction(Agent agent,
                                        device OakFleetLane *lane,
                                        device float *pending,
                                        device const uint *laneSeeds,
                                        uint gid, uint tid) {
    uint discreteActionCount = OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionCount;
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
    if (tid < discreteActionCount) pending[tid] = (tid == 0u) ? float(sampledAction) : 0.0f;
    if (tid < agent.actionCount) {
        float actionSignal = 0.0f;
        if (tid < discreteActionCount)
            actionSignal = ((tid == sampledAction) ? 1.0f : 0.0f) - exp(finiteOrZero(agent.policyMeanRaw[tid]) - behaviorLogitMax) / behaviorSoftmaxSum;
        agent.actionSignal[tid] = actionSignal;
    }
    if (tid == 0u) lane->steps = stepIdx + 1u;
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        uint off = k * agent.actionCount;
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
        agent.importanceStaging[k] = isfinite(ratio) ? min(ratio, float(agent.optionCount)) : 0.0f;
    }
}

static inline void sampleAction(Agent agent,
                                device OakFleetLane *lanes,
                                device uint *envArena,
                                device const uint *envOff,
                                device const uint *laneSeeds,
                                uint gid, uint tid) {
    device OakFleetLane *lane = &lanes[gid];
    device float *pending = lanePendingAction(laneEnvState(envArena, envOff, gid));
    if (OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionKind == 1u) {
        sampleDiscreteAction(agent, lane, pending, laneSeeds, gid, tid);
        return;
    }
    device float *scratch = agent.sampleScratch;
    uint envActionCount = OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionCount;
    float sampleSigma = agent.sampleSigma;
    uint stepIdx = lane->steps;

    if (tid < envActionCount) {
        uint d = tid;
        uint2 key = uint2(laneSeeds[gid], RNG_STREAM_ACT);
        uint pair = d >> 1u;
        uint4 r = philox4x32(key, uint4(stepIdx, pair >> 1u, 0u, 0u));

        uint w0 = (pair & 1u) ? r.z : r.x;
        uint w1 = (pair & 1u) ? r.w : r.y;
        float uniform1 = u01open(w0);
        float uniform2 = u01(w1);
        float boxMullerRadius = sqrt(-2.0f * log(uniform1));
        float boxMullerAngle = 2.0f * M_PI_F * uniform2;
        float standardNormal = ((d & 1u) == 0u) ? (boxMullerRadius * cos(boxMullerAngle)) : (boxMullerRadius * sin(boxMullerAngle));
        float mu = finiteOrZero(agent.policyMean[d]);
        float sample = finiteOrZero(mu + sampleSigma * standardNormal);
        float clippedAction = max(-1.0f, min(1.0f, sample));
        pending[d] = clippedAction;
        agent.actionSignal[d] = clippedAction;
        agent.actionSignal[envActionCount + d] = 0.0f;
        scratch[d] = sample;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) lane->steps = stepIdx + 1u;
    for (uint k = tid; k < agent.optionCount; k += OAK_SIMD_WIDTH) {
        uint off = k * agent.actionCount;
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
        agent.importanceStaging[k] = isfinite(ratio) ? min(ratio, float(agent.optionCount)) : 0.0f;
    }
}

static inline void stepEnvironment(device float *params,
                                   device float *stateArena,
                                   device uint *uintArena,
                                   device KernelScalars *scalars,
                                   device OakFleetLane *lanes,
                                   device uint *envArena,
                                   device const uint *envOff,
                                   constant uint *mjModels,
                                   device float *mjScratch,
                                   device const uint *mjScratchOff,
                                   device const float *envConstants,
                                   device const uint *laneSeeds,
                                   constant OakFleetEnvCfg &ecfg,
                                   device const int *recordSlot,
                                   device float *recordBuf,
                                   uint gid, uint gidLocal, uint slane, bool wide) {
    device OakFleetLane *lane = &lanes[gid];
    uint envK = OAK_SPEC_ENV_KIND;
    float ctrl[OAK_ENV_ACT_MAX];
    uint done = 0u;
    device uint *envState = laneEnvState(envArena, envOff, gid);
    device float *obsOut = laneEnvObs(envState, envK);
    float reward = advanceEnvironment(lane, envState, mjModels, mjScratch + mjScratchOff[gid], envConstants,
                                      envK, laneSeeds[gid], ecfg, ctrl, obsOut, done, slane, wide);
    if (slane != 0u) return;
    uint stepNum = lane->steps;
    accumulateReward(lane, reward, stepNum, ecfg);
    {
        Agent agent = make_agent(params, stateArena, uintArena, scalars, shape(), gidLocal, gid);
        if (lane->deathStep == OAK_DEATH_NONE) {
            for (uint i = 0u; i < agent.actionCount; ++i) {
                if (!isfinite(agent.policyMean[i])) {
                    lane->deathStep = stepNum;
                    lane->deathDetector = OAK_DET_POLICY_MEAN;
                    break;
                }
            }
        }
        for (uint i = 0u; i < agent.observationCount && i < OAK_ENV_OBS_MAX; ++i) {
            agent.observation[i] = obsOut[i];
            if (!isfinite(obsOut[i]) && lane->deathStep == OAK_DEATH_NONE) {
                lane->deathStep = stepNum;
                lane->deathDetector = OAK_DET_ENV_OBS;
            }
        }
        agent.reward[0] = reward;
    }
    recordPose(envK, mjModels, envState, ctrl, stepNum, done, reward, ecfg, recordSlot, recordBuf, gid);
}

kernel void oak_megastep(device float *params [[buffer(0)]],
                         device float *stateArena [[buffer(1)]],
                         device uint *uintArena [[buffer(2)]],
                         device KernelScalars *scalars [[buffer(3)]],
                         device OakFleetLane *lanes [[buffer(6)]],
                         device const uint *laneSeeds [[buffer(7)]],
                         constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                         device const int *recordSlot [[buffer(9)]],
                         device float *recordBuf [[buffer(10)]],
                         constant uint3 &dispatchSpan [[buffer(11)]],
                         device uint *envArena [[buffer(14)]],
                         device const uint *envOff [[buffer(15)]],
                         constant uint *mjModels [[buffer(16)]],
                         device float *mjScratch [[buffer(17)]],
                         device const uint *mjScratchOff [[buffer(18)]],
                         device const float *envConstants [[buffer(19)]],
                         threadgroup float *tgCont [[threadgroup(0)]],
                         uint tgIdx [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint slane [[thread_index_in_simdgroup]],
                         uint sgrp [[simdgroup_index_in_threadgroup]]) {
    uint laneBase = (tgIdx + dispatchSpan.z) * OAK_MEGA_G;
    uint laneShift = dispatchSpan.z * OAK_MEGA_G;
    uint wideLane = laneBase + sgrp;
    Shape s = shape();
    uint tgK = s.optionCount;
    threadgroup float *lCont = tgCont + sgrp * tgK;
    threadgroup float *lStash = tgCont + OAK_MEGA_G * tgK + sgrp * 5u * tgK;
    threadgroup float *lDirty = tgCont + 6u * OAK_MEGA_G * tgK + sgrp;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {
        Agent agent = make_agent(params, stateArena, uintArena, scalars, s, wideLane - laneShift, wideLane);
        if ((OAK_MEGA_STAGES & 1u) != 0u) {
            constructFeatures(agent, slane);
            predictAndControl(agent, slane, lCont, lStash, lDirty);
        }
        simdSyncDev();
        if ((OAK_MEGA_STAGES & 8u) != 0u)
            replaceWorstFeature(agent, slane, lDirty);
        simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        agent.stepCount = scalars[wideLane].stepCount;
        agent.stateSlot = scalars[wideLane].stateSlot;
        if ((OAK_MEGA_STAGES & 16u) != 0u) {
            emitOptionActions(agent, slane, lDirty[0] != 0.0f);
            simdgroup_barrier(mem_flags::mem_device);
            blendKeyboard(agent, slane);
        }
        simdSyncDev();
        if ((OAK_MEGA_STAGES & 32u) != 0u)
            sampleAction(agent, lanes, envArena, envOff, laneSeeds, wideLane, slane);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if ((OAK_MEGA_STAGES & 64u) != 0u) {
            if (OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].laneThreads > 1u) {
                stepEnvironment(params, stateArena, uintArena, scalars,
                                lanes, envArena, envOff, mjModels, mjScratch, mjScratchOff, envConstants,
                                laneSeeds, ecfg, recordSlot, recordBuf, wideLane,
                                wideLane - laneShift, slane, true);
            } else if (tid < OAK_MEGA_G) {
                uint packedLane = laneBase + tid;
                stepEnvironment(params, stateArena, uintArena, scalars,
                                lanes, envArena, envOff, mjModels, mjScratch, mjScratchOff, envConstants,
                                laneSeeds, ecfg, recordSlot, recordBuf, packedLane,
                                packedLane - laneShift, 0u, false);
            }
        }
        uint planStep = dispatchSpan.y + it + 1u;
        uint planSkipPos = ((planStep % OAK_PLAN_PERIOD) + OAK_PLAN_PERIOD
               - (laneBase % OAK_PLAN_PERIOD)) % OAK_PLAN_PERIOD;
        bool skipInThisGroup = planSkipPos < OAK_MEGA_G;
        if (!skipInThisGroup || sgrp > 0u) {
            uint idx = skipInThisGroup ? (sgrp - 1u) : sgrp;
            if (skipInThisGroup && idx >= planSkipPos) idx += 1u;
            uint wLane = laneBase + idx;
            Agent planAgent = make_agent(params, stateArena, uintArena, scalars, s, wLane - laneShift, wLane);
            uint pSlot = 1u - planAgent.stateSlot;
            threadgroup const float *pStash = tgCont + OAK_MEGA_G * tgK + idx * 5u * tgK;
            bool stashDirty = tgCont[6u * OAK_MEGA_G * tgK + idx] != 0.0f;
            if ((OAK_MEGA_STAGES & 2u) != 0u)
                planAllOptions(planAgent, slane, pStash, stashDirty, pSlot);
            if ((OAK_MEGA_STAGES & 4u) != 0u) {
                device const float *prevS = state_at(planAgent, pSlot);
                device const float *currS = state_at(planAgent, 1u - pSlot);
                learnFromTransition(planAgent, prevS, currS, slane, pSlot, pStash, stashDirty);
            }
            simdgroup_barrier(mem_flags::mem_device);
            if ((OAK_MEGA_STAGES & 8u) != 0u)
                updateUtilities(planAgent, slane);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

kernel void oak_fleet_snapshot(device float *params [[buffer(0)]],
                               device float *stateArena [[buffer(1)]],
                               device uint *uintArena [[buffer(2)]],
                               device KernelScalars *scalars [[buffer(3)]],
                               device OakFleetLane *lanes [[buffer(6)]],
                               constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                               device OakLaneSnapshot *snap [[buffer(9)]],
                               constant uint2 &snapCtl [[buffer(10)]],
                               uint slane [[thread_index_in_simdgroup]],
                               uint sgid [[threadgroup_position_in_grid]]) {
    uint sgidL = sgid;
    sgid += snapCtl.y;
    if (sgid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[sgid];
    device OakLaneSnapshot *o = &snap[snapCtl.x + sgid];
    uint paramStride = layoutFor(shape()).paramStride;
    device const float *lp = params + sgidL * paramStride;
    uint nonFinite = 0u;
    float mx = 0.0f;
    for (uint i = slane; i < paramStride; i += OAK_SIMD_WIDTH) {
        float v = lp[i];
        if (!isfinite(v)) nonFinite = 1u;
        else mx = max(mx, fabs(v));
    }
    nonFinite = simd_max(nonFinite);
    mx = simd_max(mx);
    if (slane != 0u) return;
    o->scalars = scalars[sgid];
    o->paramAbsMax = mx;
    o->paramNonFinite = nonFinite;
    o->rewardSum = lane->rewardSum;
    o->rho = lane->averageReward;
    o->deathStep = lane->deathStep;
    o->deathDetector = lane->deathDetector;
}
