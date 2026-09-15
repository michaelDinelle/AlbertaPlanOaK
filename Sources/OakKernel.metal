#pragma METAL fp math_mode(safe)

constant uint OAK_SPEC_ENV_KIND [[function_constant(2)]];
constant uint OAK_MEGA_G [[function_constant(3)]];

inline void syncMemory() {
    simdgroup_barrier(mem_flags::mem_device);
}

inline void syncSharedMemory() {
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}

constant uint FEATURE_PRODUCT = 0u;
constant uint FEATURE_NORMALIZED_SUM = 1u;
constant uint FEATURE_OBSERVATION_TIMESCALE = 3u;

inline float keepFinite(float old_w, float new_w) {
    return isfinite(new_w) ? new_w : old_w;
}

inline float trackingRateFor(uint step, uint cap) {
    return max(1.0f / (float(step) + 1.0f), 1.0f / float(max(cap, 1u)));
}

struct AgentParameters {
    float valueWeight[OAK_FEATURE_CAPACITY];
    Record option[OAK_FEATURE_CAPACITY * OAK_OPTION_COUNT];
};

struct AgentMemory {
    float observation[OAK_OBSERVATION_CAPACITY];
    float observationMean[OAK_OBSERVATION_CAPACITY];
    float observationSecondMoment[OAK_OBSERVATION_CAPACITY];
    float4 observationMeta[OAK_OBSERVATION_CAPACITY];
    float features[OAK_FEATURE_STATES];
    float policyMean[OAK_POLICY_MAX];
    float policyMeanRaw[OAK_POLICY_MAX];
    float actionSignal[OAK_ACTION_PARAMS];
    float optionActorCredit[OAK_POLICY_BANK];
    float optionPolicyMean[OAK_POLICY_BANK];
    float importance[OAK_OPTION_COUNT];
    float reward;
    float td;
    float optionTd[OAK_OPTION_COUNT];
    float featureUtility[OAK_FEATURE_CAPACITY];
    float optionUtility[OAK_OPTION_COUNT];
    float bonusWeight[OAK_FEATURE_CAPACITY];
    alignas(float4) float2 featureMoments[OAK_FEATURE_CAPACITY];
    float4 valueMeta[OAK_FEATURE_CAPACITY];
    float4 optionValueMeta[OAK_FEATURE_CAPACITY];
    float4 modelMeta[OAK_FEATURE_CAPACITY];
    float actorAdvantage[OAK_OPTION_COUNT];
    float optionRewardTd[OAK_OPTION_COUNT];
    float optionDurationTd[OAK_OPTION_COUNT];
    float observationTimescale[OAK_CANDIDATES_PER_SLOT * OAK_OBSERVATION_CAPACITY];
    float optionPriority[OAK_OPTION_COUNT];
    float continuation[OAK_OPTION_COUNT];
    float rewardCenter;
    float optionValueScale[OAK_OPTION_COUNT];
    float valueScale;
    float previousFeatureNormSq;
    float optionValueDot[OAK_OPTION_COUNT];
    float optionValueAuxScale[OAK_OPTION_COUNT];
    float optionAverageReward[OAK_OPTION_COUNT];
    float optionModelAverageReward[OAK_OPTION_COUNT];
    float optionFitness[OAK_OPTION_COUNT];
    float4 actorMeta[OAK_OPTION_COUNT];
    float importanceStaging[OAK_OPTION_COUNT];
    float actorCreditStaging[OAK_POLICY_BANK];
    float sampleScratch[OAK_SAMPLE_SCRATCH];
    float predictedReward[OAK_OPTION_COUNT];
    float predictedDuration[OAK_OPTION_COUNT];
    float predictedNextValue[OAK_OPTION_COUNT];
    float optionNextValueTd[OAK_OPTION_COUNT];
    float optionNextValueDot[OAK_OPTION_COUNT];
    float optionNextValueScale[OAK_OPTION_COUNT];
    float optionNextValueAuxScale[OAK_OPTION_COUNT];
    float nurseryStats[OAK_FEATURE_STATS];
    float optionLogit[OAK_POLICY_BANK];
    float optionPolicyCoef[OAK_POLICY_BANK];
    float optionUseByFeature[OAK_FEATURE_CAPACITY];
    float nurseryScore[OAK_FEATURE_CAPACITY];
    float replaceScore[OAK_FEATURE_STATS];
    float currentValue;
    float previousModelEnergy;
    float crossModelEnergy;
    float currentModelEnergy;
    float planningReward;
    float inverseFeatureNorm;
    float averageReward;
};

struct AgentIndices {
    uint optionFeature[OAK_OPTION_COUNT];
    uint optionAge[OAK_OPTION_COUNT];
    uint featureBirthCount[OAK_FEATURE_CAPACITY];
    uint lastStopAge[OAK_OPTION_COUNT];
    uint activeFeature[OAK_ACTIVE_STATES];
    uint replaceSlot[4u];
    uint parked[OAK_OPTION_COUNT];
    uint4 featureMeta[OAK_FEATURE_CAPACITY];
    uint4 nurseryMeta[OAK_FEATURE_CAPACITY];
};

struct Agent {
    uint observationCount, envActionCount, stepCount, stateSlot;
    device AgentParameters *parameters;
    device AgentMemory *memory;
    device AgentIndices *indices;
};

inline Agent bindAgent(device float *parameters, device float *memory, device uint *indices, uint lane,
                       uint step) {
    return {OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].obsDimNatural,
            OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionCount,
            step,
            step & 1u,
            (device AgentParameters *)parameters + lane,
            (device AgentMemory *)memory + lane,
            (device AgentIndices *)indices + lane};
}

inline device Record *optionRecord(Agent agent, uint feature, uint option) {
    return agent.parameters->option + feature * OAK_OPTION_COUNT + option;
}

inline device float *featureValues(Agent agent, uint slot) {
    return agent.memory->features + slot * OAK_FEATURE_CAPACITY;
}

inline uint perceptSourceCount(Agent agent) {
    return agent.observationCount + agent.envActionCount + 1u;
}

struct OakShapeProbe {
    uint featureCount, perceptCount, actionCount;
    uint parameterStride, memoryStride, indexStride;
    uint simdWidth, threadgroupLanes, laneThreadgroupFloats, laneSimdgroups;
};

kernel void oak_shape_probe(device OakShapeProbe *out [[buffer(0)]]) {
    *out = {OAK_FEATURE_CAPACITY,
            OAK_PERCEPT_COUNT,
            OAK_ACTION_PARAMS,
            sizeof(AgentParameters) / sizeof(float),
            sizeof(AgentMemory) / sizeof(float),
            sizeof(AgentIndices) / sizeof(uint),
            OAK_SIMD_WIDTH,
            OAK_THREADGROUP_LANES,
            OAK_LANE_SHARED_FLOATS,
            OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].laneThreads > 1u ? OAK_WIDE_LANE_SIMDGROUPS : 1u};
}

constant uint OAK_ENV_TABLE_HEADER = 9u;
constant uint OAK_ENV_TABLE_STRIDE = 10u + OAK_POSE_LEN;

inline uint mjWorkWords(uint envK, device float *base) {
    if (OAK_ENV_SPECS[envK].mjIndex != 0u) return 0u;
    E0::Work w = E0::bindWork(base);
    return uint((device float *)w.conRows - base) + uint(w.conRows - w.conDim);
}

kernel void oak_env_table_probe(device uint *out [[buffer(0)]],
                                uint threadIndex [[thread_position_in_grid]]) {
    if (threadIndex != 0u) return;
    out[0] = OAK_ENV_COUNT;
    out[1] = OAK_ENV_OBS_MAX;
    out[2] = OAK_ENV_ACT_MAX;
    out[3] = OAK_ENV_WORDS_MAX;
    out[4] = OAK_ENV_CONSTANT_WORDS;
    out[5] = PPO_PARAM_STRIDE;
    out[6] = PPO_ROLL_STRIDE;
    out[7] = PPO_SCRATCH_STRIDE;
    out[8] = PPO_STATE_STRIDE;
    for (uint i = 0u; i < OAK_ENV_COUNT; ++i) {
        device uint *row = out + OAK_ENV_TABLE_HEADER + i * OAK_ENV_TABLE_STRIDE;
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
        row[9u + OAK_POSE_LEN] = mjWorkWords(i, (device float *)out);
    }
}

kernel void oak_env_constants_init(device float *out [[buffer(0)]],
                                   uint threadIndex [[thread_position_in_grid]]) {
    if (threadIndex == 0u) jbw_buildResponse(out);
}

inline device uint *activeFeatures(Agent agent, uint slot) {
    return agent.indices->activeFeature + slot * OAK_ACTIVE_CAPACITY;
}

inline uint featureTile(float activation) {
    return uint(clamp(floor(activation + 3.5f), 0.0f, 6.0f));
}

inline uint tileSourceIndex(Agent agent, uint source) {
    return (source < agent.observationCount + agent.envActionCount)
               ? source
               : agent.observationCount + OAK_ACTION_PARAMS;
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
inline float normalizedObservation(Agent agent, uint observationIndex) {
    float mean = agent.memory->observationMean[observationIndex];
    float secondMoment = agent.memory->observationSecondMoment[observationIndex];
    float var = max(secondMoment - mean * mean, 0.0f);
    return (agent.memory->observation[observationIndex] - mean) *
           rsqrt(var + 1.0f / float(2u * OAK_PERCEPT_COUNT));
}

inline void metaOptimizeUpdate(device float4 *meta, uint i, float signal, float x, uint age, float metaEta,
                               float decay) {
    if (!isfinite(signal) || !isfinite(x)) return;
    float4 old = meta[i];
    float oldAlpha = old.x, oldBeta = old.y, oldH = old.z, oldGradEnergy = old.w;
    float grad = signal * x;
    float metaGradient = grad * oldH;
    float gradEnergyRate = 1.0f / (float(age) + 1.0f);
    float gradEnergy = keepFinite(
        oldGradEnergy, oldGradEnergy + gradEnergyRate * (metaGradient * metaGradient - oldGradEnergy));
    float ridge = 1.0f / (float(age) + 1.0f);
    float normalizedMetaGradient = metaGradient * rsqrt(max(gradEnergy, 0.0f) + ridge);
    float newBeta = oldBeta + metaEta * normalizedMetaGradient;
    float beta = keepFinite(oldBeta, clamp(newBeta, 2.0f * log(decay), 0.0f));
    float alpha = keepFinite(oldAlpha, exp(beta));
    float featureCurvature = x * x;
    float hDecay = (1.0f - decay) * max(0.0f, 1.0f - oldAlpha * featureCurvature);
    meta[i] = float4(alpha, beta, keepFinite(oldH, hDecay * oldH + oldAlpha * grad), gradEnergy);
}

inline void storeLearnScales(Agent agent, uint threadIndex) {
    device const uint *previousActive = activeFeatures(agent, agent.stateSlot);
    device const float *previousFeatures = featureValues(agent, agent.stateSlot);
    float valueStepDenominator = 1.0f;
    float previousFeatureNormSq = 0.0f;
    for (uint base = 0u; base < OAK_ACTIVE_CAPACITY; base += OAK_SIMD_WIDTH) {
        uint a = base + threadIndex;
        uint previousIndex = previousActive[a];
        float previousFeature = previousFeatures[previousIndex];
        float previousAlpha = agent.memory->valueMeta[previousIndex].x;
        for (uint t = 0u; t < OAK_SIMD_WIDTH; ++t) {
            float x = simd_shuffle(previousFeature, (ushort)t);
            float previousAlphaBroadcast = simd_shuffle(previousAlpha, (ushort)t);
            valueStepDenominator += previousAlphaBroadcast * (x * x);
            previousFeatureNormSq = fma(x, x, previousFeatureNormSq);
        }
    }
    if (threadIndex == 0u) {
        agent.memory->valueScale = (isfinite(valueStepDenominator) && valueStepDenominator > 0.0f)
                                       ? (1.0f / valueStepDenominator)
                                       : 0.0f;
        agent.memory->previousFeatureNormSq = previousFeatureNormSq;
    }
}

inline void predictAndControl(Agent agent, uint threadIndex, threadgroup float *predictions,
                              threadgroup float *predictionsChanged) {
    if (threadIndex == 0u) predictionsChanged[0] = 0.0f;
    device const float *previousFeatures = featureValues(agent, agent.stateSlot);
    device const float *currentFeatures = featureValues(agent, 1u - agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, agent.stateSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);
    float previousValue = 0.0f, currentValue = 0.0f;
    {
        float previousValuePartial = 0.0f, currentValuePartial = 0.0f;
        for (uint a = threadIndex; a < OAK_ACTIVE_CAPACITY; a += OAK_SIMD_WIDTH) {
            uint previousIndex = previousActive[a], currentIndex = currentActive[a];
            previousValuePartial = fma(agent.parameters->valueWeight[previousIndex],
                                       previousFeatures[previousIndex], previousValuePartial);
            currentValuePartial = fma(agent.parameters->valueWeight[currentIndex],
                                      currentFeatures[currentIndex], currentValuePartial);
        }
        previousValue = simd_sum(previousValuePartial);
        currentValue = simd_sum(currentValuePartial);
        if (threadIndex == 0u) agent.memory->currentValue = currentValue;
    }
    syncSharedMemory();
    float gauge = agent.memory->rewardCenter;
    float previousDifferentialValue = previousValue - gauge;
    float currentDifferentialValue = currentValue - gauge;

    float averageRewardOld = agent.memory->averageReward;
    float centeredReward = agent.memory->reward - averageRewardOld;
    if (threadIndex == 0u) {
        float delta = centeredReward + currentDifferentialValue - previousDifferentialValue;
        delta = finiteOrZero(delta);
        agent.memory->td = delta;
        float averageRewardNew = averageRewardOld + OAK_TRACKING_RATE * delta;
        agent.memory->averageReward = isfinite(averageRewardNew) ? averageRewardNew : averageRewardOld;
    }

    float optionValuePrevEnergy, optionValueCrossEnergy, optionValueNextEnergy;
    {
        float optionValuePrevEnergyPartial = 0.0f, optionValueCrossEnergyPartial = 0.0f,
              optionValueNextEnergyPartial = 0.0f;
        float modelPrevEnergyPartial = 0.0f, modelCrossEnergyPartial = 0.0f, modelNextEnergyPartial = 0.0f;
        for (uint a = threadIndex; a < OAK_ACTIVE_CAPACITY; a += OAK_SIMD_WIDTH) {
            uint previousIndex = previousActive[a], currentIndex = currentActive[a];
            float previousActivation = previousFeatures[previousIndex],
                  currentActivation = currentFeatures[currentIndex];
            float ap = agent.memory->optionValueMeta[previousIndex].x;
            optionValuePrevEnergyPartial =
                fma(ap * previousActivation, previousActivation, optionValuePrevEnergyPartial);
            optionValueCrossEnergyPartial =
                fma(ap * previousActivation, currentFeatures[previousIndex], optionValueCrossEnergyPartial);
            float acps = agent.memory->optionValueMeta[currentIndex].x;
            optionValueNextEnergyPartial =
                fma(acps * currentActivation, currentActivation, optionValueNextEnergyPartial);
            float am = agent.memory->modelMeta[previousIndex].x;
            modelPrevEnergyPartial = fma(am * previousActivation, previousActivation, modelPrevEnergyPartial);
            modelCrossEnergyPartial =
                fma(am * previousActivation, currentFeatures[previousIndex], modelCrossEnergyPartial);
            float amc = agent.memory->modelMeta[currentIndex].x;
            modelNextEnergyPartial = fma(amc * currentActivation, currentActivation, modelNextEnergyPartial);
        }
        optionValuePrevEnergy = simd_sum(optionValuePrevEnergyPartial);
        optionValueCrossEnergy = simd_sum(optionValueCrossEnergyPartial);
        optionValueNextEnergy = simd_sum(optionValueNextEnergyPartial);
        agent.memory->previousModelEnergy = simd_sum(modelPrevEnergyPartial);
        agent.memory->crossModelEnergy = simd_sum(modelCrossEnergyPartial);
        agent.memory->currentModelEnergy = simd_sum(modelNextEnergyPartial);
    }
    syncSharedMemory();
    float observedReward = agent.memory->reward;
    observedReward = finiteOrZero(observedReward);

    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        if (agent.indices->parked[k] == 0u) continue;
        agent.memory->continuation[k] = 0.0f;
        agent.memory->optionTd[k] = 0.0f;
        agent.memory->actorAdvantage[k] = 0.0f;
    }
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        bool slotLive = agent.indices->parked[k] == 0u;
        uint slot = k;
        float optionValuePrev = 0.0f, optionValueNext = 0.0f, optionValueAuxDot = 0.0f;
        float rewardHat = 0.0f, durationHat = 0.0f, nextValueHat = 0.0f;
        float previousNextValue = 0.0f, previousNextValueAuxDot = 0.0f, previousRewardHat = 0.0f,
              previousDurationHat = 0.0f;
        float twinPrevDot = 0.0f;
        float policyLogit[OAK_POLICY_MAX];
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) policyLogit[j] = 0.0f;
        for (uint base = 0u; base < OAK_ACTIVE_CAPACITY; base += OAK_SIMD_WIDTH) {
            uint localPreviousIndex = previousActive[base + threadIndex],
                 localCurrentIndex = currentActive[base + threadIndex];
            float localPreviousActivation = previousFeatures[localPreviousIndex],
                  localCurrentActivation = currentFeatures[localCurrentIndex];
            for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                uint previousIndex = simd_shuffle(localPreviousIndex, (ushort)l),
                     currentIndex = simd_shuffle(localCurrentIndex, (ushort)l);
                float previousActivation = simd_shuffle(localPreviousActivation, (ushort)l),
                      currentActivation = simd_shuffle(localCurrentActivation, (ushort)l);
                if (slotLive && previousActivation != 0.0f) {
                    device const Record *rp = optionRecord(agent, previousIndex, slot);
                    optionValuePrev = fma(rp->value, previousActivation, optionValuePrev);
                    optionValueAuxDot = fma(rp->aux, previousActivation, optionValueAuxDot);
                    previousNextValue = fma(rp->next, previousActivation, previousNextValue);
                    previousNextValueAuxDot = fma(rp->nextAux, previousActivation, previousNextValueAuxDot);
                    previousRewardHat = fma(rp->reward, previousActivation, previousRewardHat);
                    previousDurationHat = fma(rp->duration, previousActivation, previousDurationHat);
                    twinPrevDot = fma(rp->twin, previousActivation, twinPrevDot);
                }
                if (slotLive && currentActivation != 0.0f) {
                    device const Record *rc = optionRecord(agent, currentIndex, slot);
                    optionValueNext = fma(rc->value, currentActivation, optionValueNext);
                    rewardHat = fma(rc->reward, currentActivation, rewardHat);
                    durationHat = fma(rc->duration, currentActivation, durationHat);
                    nextValueHat = fma(rc->next, currentActivation, nextValueHat);
                    for (uint j = 0u; j < OAK_POLICY_MAX; ++j)
                        policyLogit[j] = fma(rc->policy[j], currentActivation, policyLogit[j]);
                }
            }
        }
        if (!slotLive) continue;
        uint optionFeatureIndex = agent.indices->optionFeature[k] % OAK_FEATURE_CAPACITY;
        predictions[k * 5u + 0u] = previousNextValue;
        predictions[k * 5u + 1u] = previousNextValueAuxDot;
        predictions[k * 5u + 2u] = previousRewardHat;
        predictions[k * 5u + 3u] = previousDurationHat;
        predictions[k * 5u + 4u] = twinPrevDot;
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j)
            agent.memory->optionLogit[k * OAK_POLICY_MAX + j] = policyLogit[j];

        float stoppingBonus =
            agent.memory->bonusWeight[optionFeatureIndex] - agent.parameters->valueWeight[optionFeatureIndex];
        if (!isfinite(stoppingBonus) || stoppingBonus < 0.0f) stoppingBonus = 0.0f;
        float stoppingValue =
            currentDifferentialValue + (1.0f + stoppingBonus) * currentFeatures[optionFeatureIndex];
        bool subtaskFinite =
            isfinite(optionValuePrev) && isfinite(optionValueNext) && isfinite(stoppingValue);

        float stop =
            (isfinite(optionValueNext) && isfinite(stoppingValue) && stoppingValue >= optionValueNext) ? 1.0f
                                                                                                       : 0.0f;

        if (k == 0u) stop = 0.0f;
        float continuation = 1.0f - stop;
        agent.memory->continuation[k] = continuation;
        if (stop > 0.0f) agent.indices->lastStopAge[k] = 0u;

        float optionAverageRewardOld = agent.memory->optionAverageReward[k];
        float optionTdError = subtaskFinite
                                  ? (centeredReward - optionAverageRewardOld +
                                     continuation * optionValueNext + stop * stoppingValue - optionValuePrev)
                                  : 0.0f;
        optionTdError = finiteOrZero(optionTdError);

        if (k == 0u) optionTdError = agent.memory->td - optionAverageRewardOld;
        agent.memory->optionTd[k] = optionTdError;
        agent.memory->actorAdvantage[k] = finiteOrZero(optionTdError);
        {
            float optionAverageRewardNew = optionAverageRewardOld + OAK_TRACKING_RATE * optionTdError;
            agent.memory->optionAverageReward[k] = keepFinite(optionAverageRewardOld, optionAverageRewardNew);
        }
        float importanceRatio = agent.memory->importance[k];
        {
            float previousOptionFitness = agent.memory->optionFitness[k];
            agent.memory->optionFitness[k] = keepFinite(
                previousOptionFitness, previousOptionFitness + OAK_TRACKING_RATE * importanceRatio *
                                                                   (observedReward - previousOptionFitness));
        }

        float optionValueBootstrapEnergy = optionValuePrevEnergy -
                                           2.0f * continuation * optionValueCrossEnergy +
                                           continuation * continuation * optionValueNextEnergy;
        float denom = fma(importanceRatio, optionValueBootstrapEnergy, 1.0f);

        agent.memory->optionValueScale[k] = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        {
            agent.memory->optionValueDot[k] = finiteOrZero(optionValueAuxDot);

            float optionValueAuxDenom = fma(importanceRatio, optionValuePrevEnergy, 1.0f);
            agent.memory->optionValueAuxScale[k] =
                (isfinite(optionValueAuxDenom) && optionValueAuxDenom > 0.0f) ? (1.0f / optionValueAuxDenom)
                                                                              : 0.0f;
        }
        agent.memory->predictedReward[k] = rewardHat;
        agent.memory->predictedDuration[k] = durationHat;
        agent.memory->predictedNextValue[k] = nextValueHat;
    }
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (threadIndex == 0u) {
        float gaugeRate = trackingRateFor(agent.stepCount, OAK_FEATURE_CAPACITY);
        float gaugeNew = agent.memory->rewardCenter + gaugeRate * (currentValue - agent.memory->rewardCenter);
        agent.memory->rewardCenter = keepFinite(agent.memory->rewardCenter, gaugeNew);
    }
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    storeLearnScales(agent, threadIndex);
}

inline float constructedValue(Agent agent, device const float *features, uint parentA, uint parentB,
                              uint kind) {
    kind %= 4u;
    if (kind == FEATURE_OBSERVATION_TIMESCALE)
        return agent.memory->observationTimescale[parentA % (OAK_TRACE_LEVELS * agent.observationCount)];
    float a = features[parentA % OAK_FEATURE_CAPACITY], b = features[parentB % OAK_FEATURE_CAPACITY];
    return kind == FEATURE_PRODUCT          ? a * b
           : kind == FEATURE_NORMALIZED_SUM ? (a + b) * 0.7071067811865475f
                                            : a * (b - a);
}

inline void constructFeatures(Agent agent, uint threadIndex) {
    float standardizedObservation = 0.0f;
    if (threadIndex < agent.observationCount) {
        uint observationIndex = threadIndex;
        float x = agent.memory->observation[observationIndex];
        if (isfinite(x)) {
            float observationMean = agent.memory->observationMean[observationIndex];
            metaOptimizeUpdate(agent.memory->observationMeta, observationIndex, x - observationMean, 1.0f,
                               min(agent.stepCount, OAK_FEATURE_CAPACITY), OAK_META_RATE, OAK_TRACKING_RATE);
            float obsRate = max(1.0f / (float(agent.stepCount) + 1.0f),
                                agent.memory->observationMeta[observationIndex].x);
            agent.memory->observationMean[observationIndex] += obsRate * (x - observationMean);
            agent.memory->observationSecondMoment[observationIndex] +=
                obsRate * (x * x - agent.memory->observationSecondMoment[observationIndex]);
            standardizedObservation = normalizedObservation(agent, observationIndex);
            for (uint m = 1u; m <= OAK_TRACE_LEVELS; ++m) {
                uint slot = (m - 1u) * agent.observationCount + observationIndex;
                float rate = 1.0f / float(1u << m);
                agent.memory->observationTimescale[slot] +=
                    rate * (standardizedObservation - agent.memory->observationTimescale[slot]);
            }
        } else {
            standardizedObservation = normalizedObservation(agent, observationIndex);
        }
    }
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH)
        agent.memory->importance[k] = agent.memory->importanceStaging[k];
    for (uint j = threadIndex; j < OAK_OPTION_COUNT * OAK_POLICY_MAX; j += OAK_SIMD_WIDTH)
        agent.memory->optionActorCredit[j] = agent.memory->actorCreditStaging[j];

    device const float *previousFeatures = featureValues(agent, agent.stateSlot);
    device float *currentFeatures = featureValues(agent, 1u - agent.stateSlot);
    device uint *currentActive = activeFeatures(agent, 1u - agent.stateSlot);

    syncMemory();

    for (uint ablk = 0u; ablk < OAK_PERCEPT_COUNT; ablk += OAK_SIMD_WIDTH) {
        uint i = ablk + threadIndex;
        float x;
        if (i < agent.observationCount) {
            x = standardizedObservation;
        } else if (i < agent.observationCount + OAK_ACTION_PARAMS) {
            x = agent.memory->actionSignal[i - agent.observationCount];
        } else if (i == agent.observationCount + OAK_ACTION_PARAMS) {
            x = agent.memory->reward;
        } else if (i == agent.observationCount + OAK_ACTION_PARAMS + 1u) {
            x = 1.0f;
        } else {
            uint j = i - (agent.observationCount + OAK_ACTION_PARAMS + 2u);
            uint lvl = j / max(agent.observationCount, 1u);
            x = (lvl < OAK_TRACE_LEVELS)
                    ? agent.memory->observationTimescale[lvl * agent.observationCount +
                                                         (j - lvl * agent.observationCount)]
                    : 0.0f;
        }
        x = finiteOrZero(x);
        currentFeatures[i] = x;
        currentActive[i] = i;
        if (agent.indices->featureMeta[i].w < 0xffffffffu) agent.indices->featureMeta[i].w += 1u;
    }
    syncMemory();

    for (uint ablk = 0u; ablk < OAK_PERCEPT_COUNT; ablk += OAK_SIMD_WIDTH) {
        uint ai = ablk + threadIndex;
        uint tile;
        uint tile2;
        {
            uint sourceCount = perceptSourceCount(agent);
            uint sourceIndex = ai % sourceCount, occupancy = ai / sourceCount;
            if (occupancy == 0u) {
                tile = featureTile(currentFeatures[tileSourceIndex(agent, sourceIndex)]);
            } else if ((ai & 1u) == 0u) {
                tile = zoomedFeatureTile(currentFeatures[tileSourceIndex(agent, sourceIndex)], occupancy);
            } else {
                uint channelA = tileSourceIndex(agent, sourceIndex);
                uint channelB = (sourceIndex + occupancy) % agent.observationCount;
                tile = jointFeatureTile(currentFeatures[channelA], currentFeatures[channelB], occupancy);
            }
            tile2 = zoomedFeatureTile(currentFeatures[tileSourceIndex(agent, sourceIndex)], occupancy + 1u);
        }
        for (uint g = 0u; g < OAK_CONSTRUCTED_PER_PERCEPT; ++g) {
            uint groupTile = (g == 0u) ? tile : tile2;
            uint bankLo = OAK_PERCEPT_COUNT + (ai + g * OAK_PERCEPT_COUNT) * OAK_CANDIDATES_PER_SLOT,
                 bankHi = bankLo + OAK_CANDIDATES_PER_SLOT;
            uint winner = bankLo + groupTile;
            float winningValue = 0.0f;
            for (uint i = bankLo; i < bankHi; ++i) {
                uint4 meta = agent.indices->featureMeta[i];
                float constructed = constructedValue(agent, previousFeatures, meta.x, meta.y, meta.z);
                float rate = max(1.0f / (float(meta.w) + 1.0f), OAK_TRACKING_RATE);
                float2 moments = agent.memory->featureMoments[i];
                moments.x += rate * (constructed - moments.x);
                moments.y += rate * (constructed * constructed - moments.y);
                agent.memory->featureMoments[i] = moments;
                if (i == winner) winningValue = constructed;
                currentFeatures[i] = 0.0f;
            }
            float2 winnerMoments = agent.memory->featureMoments[winner];
            float var = max(winnerMoments.y - winnerMoments.x * winnerMoments.x, 0.0f);
            float ridge = max(1.0f / (float(agent.indices->featureMeta[winner].w) + 1.0f),
                              1.0f / float(2u * OAK_PERCEPT_COUNT));
            winningValue = (winningValue - winnerMoments.x) * rsqrt(var + ridge);
            currentFeatures[winner] = finiteOrZero(winningValue);
            currentActive[OAK_PERCEPT_COUNT + ai + g * OAK_PERCEPT_COUNT] = winner;
            if (agent.indices->featureMeta[winner].w < 0xffffffffu)
                agent.indices->featureMeta[winner].w += 1u;
        }
    }
    syncMemory();
}

static inline void planOption(Agent agent, uint k, float previousNextValueDot, float optionNextValueDot,
                              float previousRewardHat, float previousDurationHat) {
    float continuation = agent.memory->continuation[k];
    float rewardHat = agent.memory->predictedReward[k], durationHat = agent.memory->predictedDuration[k],
          currentNextValueHat = agent.memory->predictedNextValue[k];
    float currentValue = agent.memory->currentValue;

    float modelPrevEnergy = agent.memory->previousModelEnergy;
    float modelBootstrapNorm = agent.memory->previousModelEnergy -
                               2.0f * continuation * agent.memory->crossModelEnergy +
                               continuation * continuation * agent.memory->currentModelEnergy;
    modelBootstrapNorm = max(modelBootstrapNorm, 0.0f);
    modelBootstrapNorm = finiteOrZero(modelBootstrapNorm);
    if (!isfinite(modelPrevEnergy) || modelPrevEnergy < 0.0f) modelPrevEnergy = 0.0f;

    float modelAverageRewardOld = agent.memory->optionModelAverageReward[k];
    modelAverageRewardOld = finiteOrZero(modelAverageRewardOld);
    float rewardModelTd = (k == 0u) ? 0.0f
                                    : ((agent.memory->planningReward - modelAverageRewardOld) +
                                       continuation * rewardHat - previousRewardHat);
    float importanceRatio = agent.memory->importance[k];
    if (isfinite(agent.memory->planningReward))
        agent.memory->optionModelAverageReward[k] =
            modelAverageRewardOld + trackingRateFor(agent.stepCount, OAK_FEATURE_CAPACITY) * importanceRatio *
                                        (agent.memory->planningReward - modelAverageRewardOld);
    float durationTd = (k == 0u) ? 0.0f : (1.0f + continuation * durationHat - previousDurationHat);

    float optionNextValueTd =
        continuation * currentNextValueHat + (1.0f - continuation) * currentValue - previousNextValueDot;
    bool modelFinite = isfinite(rewardModelTd) && isfinite(durationTd) && isfinite(durationHat) &&
                       isfinite(optionNextValueTd);
    rewardModelTd = finiteOrZero(rewardModelTd);
    durationTd = finiteOrZero(durationTd);
    optionNextValueTd = finiteOrZero(optionNextValueTd);
    agent.memory->optionRewardTd[k] = rewardModelTd;
    agent.memory->optionDurationTd[k] = durationTd;
    agent.memory->optionNextValueTd[k] = optionNextValueTd;
    agent.memory->optionNextValueDot[k] = finiteOrZero(optionNextValueDot);
    float nextValueStepDenom = fma(importanceRatio, modelBootstrapNorm, 1.0f);
    agent.memory->optionNextValueScale[k] =
        (isfinite(nextValueStepDenom) && nextValueStepDenom > 0.0f) ? (1.0f / nextValueStepDenom) : 0.0f;
    float nextValueAuxDenom = fma(importanceRatio, modelPrevEnergy, 1.0f);
    agent.memory->optionNextValueAuxScale[k] =
        (isfinite(nextValueAuxDenom) && nextValueAuxDenom > 0.0f) ? (1.0f / nextValueAuxDenom) : 0.0f;
    float modelSquaredError = optionNextValueTd * optionNextValueTd + rewardModelTd * rewardModelTd;
    if (modelFinite && isfinite(modelSquaredError)) {
        agent.memory->optionPriority[k] = sqrt(max(modelSquaredError, 0.0f));
    } else {
        float debt = fabs(agent.memory->optionPriority[k]);
        if (debt <= 0.0f) debt = 1.0f;
        agent.memory->optionPriority[k] = -debt;
    }
}

static inline void planAllOptions(Agent agent, uint threadIndex, threadgroup const float *stash,
                                  bool stashDirty, uint previousSlot) {
    device const uint *previousActive = activeFeatures(agent, previousSlot);
    device const float *previousFeatures = featureValues(agent, previousSlot);
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        if (agent.indices->parked[k] == 0u) continue;
        agent.memory->optionRewardTd[k] = 0.0f;
        agent.memory->optionDurationTd[k] = 0.0f;
        agent.memory->optionNextValueTd[k] = 0.0f;
        agent.memory->optionNextValueDot[k] = 0.0f;
        agent.memory->optionNextValueScale[k] = 0.0f;
        agent.memory->optionNextValueAuxScale[k] = 0.0f;
        agent.memory->optionPriority[k] = 0.0f;
    }
    syncMemory();
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        if (agent.indices->parked[k] != 0u) continue;
        uint slot = k;
        float previousNextValueDot, optionNextValueDot, previousRewardHat, previousDurationHat;
        if (stashDirty) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            for (uint a = 0u; a < OAK_ACTIVE_CAPACITY; a += 1u) {
                uint previousIndex = previousActive[a];
                float previousActivation = previousFeatures[previousIndex];
                if (previousActivation == 0.0f) continue;
                device const Record *rp = optionRecord(agent, previousIndex, slot);
                p0 = fma(rp->next, previousActivation, p0);
                p1 = fma(rp->nextAux, previousActivation, p1);
                p2 = fma(rp->reward, previousActivation, p2);
                p3 = fma(rp->duration, previousActivation, p3);
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
        planOption(agent, k, previousNextValueDot, optionNextValueDot, previousRewardHat,
                   previousDurationHat);

        float importanceRatio = agent.memory->importance[k];
        float advantage = agent.memory->actorAdvantage[k];
        advantage = finiteOrZero(advantage);

        metaOptimizeUpdate(agent.memory->actorMeta, k, advantage, 1.0f, agent.indices->optionAge[k],
                           OAK_META_RATE, OAK_TRACKING_RATE);
        float actorAlpha = agent.memory->actorMeta[k].x / OAK_ACTOR_ALPHA_DENOM;
        float previousFeatureNormSq = agent.memory->previousFeatureNormSq;
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d) {
            float targetGrad = agent.memory->optionActorCredit[k * OAK_POLICY_MAX + d];
            float gradScale = importanceRatio * targetGrad;

            float denom =
                1.0f + actorAlpha * advantage * advantage * gradScale * gradScale * previousFeatureNormSq;
            float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
            float policyCoef = actorAlpha * advantage * gradScale * scale;
            agent.memory->optionPolicyCoef[k * OAK_POLICY_MAX + d] = finiteOrZero(policyCoef);
        }
    }
}

static inline void learnFromTransition(Agent agent, device const float *previousFeatures,
                                       device const float *currentFeatures, uint threadIndex,
                                       uint previousSlot, threadgroup const float *stash, bool stashDirty) {
    device const uint *previousActive = activeFeatures(agent, previousSlot);
    device const uint *currentActive = activeFeatures(agent, 1u - previousSlot);
    const uint activeFeatureCount = OAK_ACTIVE_CAPACITY;
    uint policyN = OAK_POLICY_MAX;

    for (uint slotBase = 0u; slotBase < OAK_OPTION_COUNT; slotBase += OAK_SIMD_WIDTH) {
        uint slot = slotBase + threadIndex;
        uint k = slot;
        bool optionLive = agent.indices->parked[k] == 0u;
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
                    uint iH = previousActive[base + threadIndex];
                    float xH = previousFeatures[iH];
                    for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                        uint i0 = simd_shuffle(iH, (ushort)l);
                        float x0 = simd_shuffle(xH, (ushort)l);
                        if (!optionLive || x0 == 0.0f) continue;
                        dot = fma(optionRecord(agent, i0, slot)->twin, x0, dot);
                    }
                }
            } else if (optionLive) {
                dot = stash[k * 5u + 4u];
            }
            dot = finiteOrZero(dot);
            twinPrevDot = dot;
        }
        if (optionLive) {
            importanceRatio = agent.memory->importance[k];
            continuation = agent.memory->continuation[k];
            optionValueAuxDot = agent.memory->optionValueDot[k];
            optionValueStepScale = agent.memory->optionValueScale[k];
            td = agent.memory->optionTd[k];
            nextValueAuxDot = agent.memory->optionNextValueDot[k];
            nextValueStepScale = agent.memory->optionNextValueScale[k];
            nextValueTdError = agent.memory->optionNextValueTd[k];
            rewardModelTdError = agent.memory->optionRewardTd[k];
            durationTdError = agent.memory->optionDurationTd[k];
            optionValueAuxResidual = td - optionValueAuxDot;
            optionValueAuxStepScale = agent.memory->optionValueAuxScale[k];
            nextValueAuxResidual = nextValueTdError - nextValueAuxDot;
            nextValueAuxStepScale = agent.memory->optionNextValueAuxScale[k];
            twinResidual = rewardModelTdError - twinPrevDot;
            for (uint j = 0u; j < policyN; ++j)
                actorCoefficient[j] = agent.memory->optionPolicyCoef[k * policyN + j];
        }
        bool baseOption = k == 0u;
        for (uint cb = 0u; cb < activeFeatureCount; cb += OAK_SIMD_WIDTH) {
            uint aH = cb + threadIndex;
            uint iH = previousActive[aH];
            float localPreviousActivation = previousFeatures[iH],
                  localCurrentActivation = currentFeatures[iH];
            float aPH = agent.memory->optionValueMeta[iH].x, aMH = agent.memory->modelMeta[iH].x;
            uint lEnd = min(OAK_SIMD_WIDTH, activeFeatureCount - cb);
            for (uint l = 0u; l < lEnd; ++l) {
                uint i = simd_shuffle(iH, (ushort)l);
                float previousActivation = simd_shuffle(localPreviousActivation, (ushort)l);
                float currentActivation = simd_shuffle(localCurrentActivation, (ushort)l);
                if (previousActivation == 0.0f && currentActivation == 0.0f) continue;
                float aP = simd_shuffle(aPH, (ushort)l);
                float aM = simd_shuffle(aMH, (ushort)l);
                if (!optionLive) continue;
                device Record *recv = optionRecord(agent, i, slot);
                Record rec = *recv;
                for (uint j = 0u; j < policyN; ++j) {
                    float w = rec.policy[j];
                    rec.policy[j] = keepFinite(w, w + actorCoefficient[j] * previousActivation);
                }
                if (!baseOption) {
                    float optionValueGradient =
                        (continuation > 0.0f)
                            ? (optionValueAuxDot * (previousActivation - continuation * currentActivation))
                            : (td * previousActivation);
                    float value = rec.value, aux = rec.aux;
                    rec.value = keepFinite(value, value + aP * importanceRatio * optionValueGradient *
                                                             optionValueStepScale);
                    rec.aux = keepFinite(aux, aux + aP * importanceRatio * optionValueAuxResidual *
                                                  previousActivation * optionValueAuxStepScale);
                }
                float nextValueGradient =
                    (continuation > 0.0f)
                        ? (nextValueAuxDot * (previousActivation - continuation * currentActivation))
                        : (nextValueTdError * previousActivation);
                float next = rec.next, nextAux = rec.nextAux, duration = rec.duration,
                      rewardWeight = rec.reward, twin = rec.twin;
                rec.next =
                    keepFinite(next, next + aM * importanceRatio * nextValueGradient * nextValueStepScale);
                rec.nextAux = keepFinite(nextAux, nextAux + aM * importanceRatio * nextValueAuxResidual *
                                                                previousActivation * nextValueAuxStepScale);
                rec.duration = keepFinite(duration, duration + aM * importanceRatio * durationTdError *
                                                                   previousActivation * nextValueStepScale *
                                                                   0.05f);
                float rewardModelGradient =
                    (continuation > 0.0f)
                        ? (twinPrevDot * (previousActivation - continuation * currentActivation))
                        : (rewardModelTdError * previousActivation);
                rec.reward = keepFinite(rewardWeight, rewardWeight + aM * importanceRatio *
                                                              rewardModelGradient * nextValueStepScale);
                rec.twin = keepFinite(twin, twin + aM * importanceRatio * twinResidual * previousActivation *
                                                nextValueAuxStepScale);
                *recv = rec;
            }
        }

        bool learnsNewFeatures = optionLive && continuation > 0.0f && importanceRatio > 0.0f;
        for (uint wb = 0u; wb < OAK_PERCEPT_COUNT; wb += OAK_SIMD_WIDTH) {
            uint newFeatureH = currentActive[OAK_PERCEPT_COUNT + wb + threadIndex];
            uint wasPreviouslyActiveH = wasActive(previousActive, OAK_PERCEPT_COUNT, newFeatureH) ? 1u : 0u;
            float newFeatureValueH = currentFeatures[newFeatureH];
            float newOptionAlphaH = agent.memory->optionValueMeta[newFeatureH].x,
                  newModelAlphaH = agent.memory->modelMeta[newFeatureH].x;
            for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                uint wasPreviouslyActive = simd_shuffle(wasPreviouslyActiveH, (ushort)l);
                if (wasPreviouslyActive != 0u) continue;
                uint i = simd_shuffle(newFeatureH, (ushort)l);
                float currentActivation = simd_shuffle(newFeatureValueH, (ushort)l);
                float aP = simd_shuffle(newOptionAlphaH, (ushort)l);
                float aM = simd_shuffle(newModelAlphaH, (ushort)l);
                if (learnsNewFeatures) {
                    device Record *rec = optionRecord(agent, i, slot);
                    if (!baseOption) {
                        float value = rec->value;
                        rec->value = keepFinite(
                            value, value + aP * importanceRatio *
                                               (-optionValueAuxDot * continuation * currentActivation) *
                                               optionValueStepScale);
                    }
                    float next = rec->next, rewardWeight = rec->reward;
                    rec->next =
                        keepFinite(next, next + aM * importanceRatio *
                                                    (-nextValueAuxDot * continuation * currentActivation) *
                                                    nextValueStepScale);
                    rec->reward = keepFinite(
                        rewardWeight, rewardWeight + aM * importanceRatio *
                                                         (-twinPrevDot * continuation * currentActivation) *
                                                         nextValueStepScale);
                }
            }
        }
    }

    float valueTdError = agent.memory->td;
    float valueStepScale = agent.memory->valueScale;
    for (uint a = threadIndex; a < activeFeatureCount; a += OAK_SIMD_WIDTH) {
        uint i = previousActive[a];
        float w = agent.parameters->valueWeight[i];
        float delta = valueTdError * previousFeatures[i] * valueStepScale;
        agent.parameters->valueWeight[i] = keepFinite(w, w + agent.memory->valueMeta[i].x * delta);
    }
}

inline void emitOptionActions(Agent agent, uint threadIndex, bool recomputeLogits) {
    uint policyN = OAK_POLICY_MAX;
    device const uint *active = activeFeatures(agent, agent.stateSlot);
    device const float *currentFeatures = featureValues(agent, agent.stateSlot);
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        if (agent.indices->parked[k] == 0u) continue;
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) {
            agent.memory->optionPolicyMean[k * OAK_POLICY_MAX + j] = 0.0f;
            agent.memory->optionLogit[k * OAK_POLICY_MAX + j] = 0.0f;
        }
    }
    syncMemory();
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        bool slotLive = agent.indices->parked[k] == 0u;
        uint slot = k;
        float policyLogit[OAK_POLICY_MAX];
        for (uint j = 0u; j < policyN; ++j) policyLogit[j] = 0.0f;
        if (recomputeLogits) {
            for (uint base = 0u; base < OAK_ACTIVE_CAPACITY; base += OAK_SIMD_WIDTH) {
                uint iH = active[base + threadIndex];
                float xH = currentFeatures[iH];
                for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                    uint i = simd_shuffle(iH, (ushort)l);
                    float x = simd_shuffle(xH, (ushort)l);
                    if (!slotLive || x == 0.0f) continue;
                    device const Record *row = optionRecord(agent, i, slot);
                    for (uint j = 0u; j < policyN; ++j)
                        policyLogit[j] = fma(row->policy[j], x, policyLogit[j]);
                }
            }
        } else if (slotLive) {
            for (uint j = 0u; j < policyN; ++j)
                policyLogit[j] = agent.memory->optionLogit[k * OAK_POLICY_MAX + j];
        }
        if (!slotLive) continue;
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) {
            uint optionActionIndex = k * OAK_POLICY_MAX + j;

            float actionMean = tanh(policyLogit[j]);
            agent.memory->optionLogit[optionActionIndex] = policyLogit[j];
            if (!isfinite(actionMean)) actionMean = agent.memory->optionPolicyMean[optionActionIndex];
            agent.memory->optionPolicyMean[optionActionIndex] = actionMean;
        }
    }
}

inline void writePrimaryPolicy(Agent agent, uint action) {
    if (action >= OAK_POLICY_MAX) return;
    float mean = agent.memory->optionPolicyMean[action];
    float logit = finiteOrZero(agent.memory->optionLogit[action]);
    agent.memory->policyMean[action] = (fma(1.0f, mean, 0.0f) + mean) / 2.0f;
    agent.memory->policyMeanRaw[action] = (fma(1.0f, logit, 0.0f) + logit) / 2.0f;
}

inline void zeroOptionColumn(Agent agent, uint threadIndex, uint k) {
    uint slot = k;
    for (uint i = threadIndex; i < OAK_FEATURE_CAPACITY; i += OAK_SIMD_WIDTH) {
        device Record *rec = optionRecord(agent, i, slot);
        rec->value = 0.0f;
        rec->aux = 0.0f;
        rec->reward = 0.0f;
        rec->duration = 0.0f;
        rec->next = 0.0f;
        rec->nextAux = 0.0f;
    }
}

static inline void replaceWorstFeature(Agent agent, uint threadIndex, threadgroup float *predictionsChanged) {
    device const float *currentFeatures = featureValues(agent, 1u - agent.stateSlot);
    if (threadIndex == 0u) agent.memory->planningReward = agent.memory->reward;
    simdgroup_barrier(mem_flags::mem_device);

    uint bestNursery;
    float bestNurseryBound;
    {
        float bestBound = -FLT_MAX;
        uint bestIndex = 0xffffffffu;
        for (uint n = threadIndex; n < OAK_FEATURE_CAPACITY; n += OAK_SIMD_WIDTH) {
            float lowerConfidenceBound = agent.memory->nurseryScore[n];
            if (lowerConfidenceBound > bestBound) {
                bestBound = lowerConfidenceBound;
                bestIndex = n;
            }
        }
        bestNurseryBound = simd_max(bestBound);
        bestNursery = simd_min((bestBound == bestNurseryBound) ? bestIndex : 0xffffffffu);
    }

    uint haveWorst = 0u, worstFeature = 0u, worstAge = 0u;
    float worstUtility = 0.0f, worstRedundancy = 0.0f, worstDownstream = 0.0f;
    uint eligiblePartial = 0u;
    float eligibleUtilPartial = 0.0f, eligibleRedundancyPartial = 0.0f, eligibleDownstreamPartial = 0.0f;
    for (uint i = OAK_PERCEPT_COUNT + threadIndex; i < OAK_FEATURE_CAPACITY; i += OAK_SIMD_WIDTH) {
        float4 e = *(device const float4 *)&agent.memory->replaceScore[i * 4u];
        if (e.w <= 0.0f) continue;
        eligiblePartial += 1u;
        eligibleUtilPartial += e.x;
        eligibleRedundancyPartial += e.y;
        eligibleDownstreamPartial += e.z;
        float util = e.x, redundancy = e.y, downstream = e.z;
        uint age = agent.indices->featureMeta[i].w;
        bool worse =
            (haveWorst == 0u) || downstream < worstDownstream ||
            (downstream == worstDownstream && util < worstUtility) ||
            (downstream == worstDownstream && util == worstUtility && redundancy > worstRedundancy) ||
            (downstream == worstDownstream && util == worstUtility && redundancy == worstRedundancy &&
             age > worstAge) ||
            (downstream == worstDownstream && util == worstUtility && redundancy == worstRedundancy &&
             age == worstAge && i < worstFeature);
        if (worse) {
            haveWorst = 1u;
            worstFeature = i;
            worstAge = age;
            worstUtility = util;
            worstRedundancy = redundancy;
            worstDownstream = downstream;
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

    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        if (agent.indices->optionAge[k] < 0xffffffffu) agent.indices->optionAge[k] += 1u;
        if (agent.indices->lastStopAge[k] < 0xffffffffu) agent.indices->lastStopAge[k] += 1u;
    }
    simdgroup_barrier(mem_flags::mem_device);
    uint candidateStart = OAK_PERCEPT_COUNT;
    uint eligible = simd_sum(eligiblePartial);
    float eligibleUtilSum = simd_sum(eligibleUtilPartial);
    float eligibleRedundancySum = simd_sum(eligibleRedundancyPartial);
    float eligibleDownstreamSum = simd_sum(eligibleDownstreamPartial);
    uint worst = (haveWorst != 0u) ? worstFeature : candidateStart;

    {
        uint staleLimit = (OAK_FEATURE_CAPACITY * OAK_OPTION_COUNT);
        uint parkedStale = 0xffffffffu, unparkedStale = 0xffffffffu;
        for (uint k2 = 1u + threadIndex; k2 < OAK_OPTION_COUNT; k2 += OAK_SIMD_WIDTH) {
            if (agent.indices->lastStopAge[k2] < staleLimit) continue;
            if (agent.indices->parked[k2] != 0u)
                parkedStale = min(parkedStale, k2);
            else
                unparkedStale = min(unparkedStale, k2);
        }
        parkedStale = simd_min(parkedStale);
        unparkedStale = simd_min(unparkedStale);
        uint staleOption = (parkedStale != 0xffffffffu)
                               ? parkedStale
                               : ((unparkedStale != 0xffffffffu) ? unparkedStale : 0u);
        if (staleOption != 0u) {
            uint k = staleOption;
            zeroOptionColumn(agent, threadIndex, k);
            if (threadIndex == 0u) {
                predictionsChanged[0] = 1.0f;
                agent.indices->optionAge[k] = 0u;
                agent.indices->lastStopAge[k] = 0u;
                agent.memory->optionUtility[k] = float(OAK_FEATURE_CAPACITY);
                agent.memory->actorAdvantage[k] = 0.0f;
                agent.memory->optionAverageReward[k] = 0.0f;
                agent.memory->optionModelAverageReward[k] = 0.0f;
                agent.indices->parked[k] = 0u;
                uint feat = agent.indices->optionFeature[k] % OAK_FEATURE_CAPACITY;
                optionRecord(agent, feat, k)->value = agent.memory->inverseFeatureNorm;
            }
        }
    }
    syncMemory();

    float invEligible = 1.0f / float(max(eligible, 1u));
    float meanEligibleUtil = eligibleUtilSum * invEligible;
    float meanEligibleRedundancy = eligibleRedundancySum * invEligible;
    float meanEligibleDownstream = eligibleDownstreamSum * invEligible;
    bool lowUtility = worstUtility < meanEligibleUtil;
    bool highRedundancy = worstRedundancy > meanEligibleRedundancy;
    bool lowDownstreamUse = worstDownstream <= meanEligibleDownstream;
    bool structurallyReplaceable = (eligible > 1u) && lowDownstreamUse && (lowUtility || highRedundancy);

    bool doReplace = structurallyReplaceable && worst < OAK_FEATURE_CAPACITY && bestNursery != 0xffffffffu &&
                     bestNurseryBound > worstUtility + worstDownstream;
    if (doReplace) {
        uint featureLimit = OAK_FEATURE_CAPACITY;
        float bestA = -FLT_MAX;
        uint bestAIndex = 0u;
        for (uint i = threadIndex; i < featureLimit; i += OAK_SIMD_WIDTH) {
            if (i == worst) continue;
            float parentUtility = agent.memory->featureUtility[i];
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestA) {
                bestA = parentUtility;
                bestAIndex = i;
            }
        }
        float bestAAll = simd_max(bestA);
        uint seedParentA =
            (bestAAll == -FLT_MAX) ? 0u : simd_min(bestA == bestAAll ? bestAIndex : 0xffffffffu);
        float bestB = -FLT_MAX;
        uint bestBIndex = 0u;
        for (uint i = threadIndex; i < featureLimit; i += OAK_SIMD_WIDTH) {
            if (i == worst || i == seedParentA) continue;
            float decor = 1.0f / (1.0f + fabs(currentFeatures[i] * currentFeatures[seedParentA]));
            float parentUtility = agent.memory->featureUtility[i] * decor;
            if (!isfinite(parentUtility)) parentUtility = -FLT_MAX;
            if (parentUtility > bestB) {
                bestB = parentUtility;
                bestBIndex = i;
            }
        }
        float bestBAll = simd_max(bestB);
        uint seedParentB = (bestBAll == -FLT_MAX) ? ((OAK_FEATURE_CAPACITY > 1u) ? 1u : 0u)
                                                  : simd_min(bestB == bestBAll ? bestBIndex : 0xffffffffu);
        uint trackingOption = 0xffffffffu, leastDemandOption = 0xffffffffu;
        float leastDemand = FLT_MAX;
        for (uint ok = 1u + threadIndex; ok < OAK_OPTION_COUNT; ok += OAK_SIMD_WIDTH) {
            float use = agent.memory->optionUtility[ok] + agent.memory->optionPriority[ok];
            if (!isfinite(use)) use = FLT_MAX;
            if (agent.indices->optionFeature[ok] == worst) trackingOption = min(trackingOption, ok);
            if (use < leastDemand) {
                leastDemand = use;
                leastDemandOption = ok;
            }
        }
        trackingOption = simd_min(trackingOption);
        float leastDemandAll = simd_min(leastDemand);
        uint replacedOption =
            (trackingOption != 0xffffffffu)
                ? trackingOption
                : ((leastDemandAll == FLT_MAX)
                       ? 1u
                       : simd_min(leastDemand == leastDemandAll ? leastDemandOption : 0xffffffffu));
        if (threadIndex == 0u) {
            uint4 nursery = agent.indices->nurseryMeta[bestNursery];
            uint birth = agent.indices->featureBirthCount[worst] + 1u;
            agent.indices->featureBirthCount[worst] = birth;
            agent.indices->featureMeta[worst] =
                uint4(nursery.x % OAK_FEATURE_CAPACITY, nursery.y % OAK_FEATURE_CAPACITY, nursery.z % 4u,
                      nursery.w);
            agent.memory->featureMoments[worst] = float2(agent.memory->nurseryStats[(bestNursery) * 4u],
                                                         agent.memory->nurseryStats[(bestNursery) * 4u + 1u]);
            agent.memory->featureUtility[worst] = agent.memory->nurseryStats[(bestNursery) * 4u + 2u];
            agent.parameters->valueWeight[worst] = 0.0f;
            agent.memory->bonusWeight[worst] = 0.0f;

            agent.indices->nurseryMeta[bestNursery] =
                uint4(seedParentA, seedParentB, (nursery.z + 1u) % 4u, 0u);
            agent.memory->nurseryStats[(bestNursery) * 4u] = 0.0f;
            agent.memory->nurseryStats[(bestNursery) * 4u + 1u] = 0.0f;
            agent.memory->nurseryStats[(bestNursery) * 4u + 2u] = 0.0f;
            agent.memory->nurseryStats[(bestNursery) * 4u + 3u] = 0.0f;

            float initialStepSize = 1.0f / float(OAK_ACTIVE_CAPACITY);
            float4 freshMeta = float4(initialStepSize, log(initialStepSize), 0.0f, 0.0f);
            agent.memory->valueMeta[worst] = freshMeta;
            agent.memory->optionValueMeta[worst] = freshMeta;
            agent.memory->modelMeta[worst] = freshMeta;

            uint k = replacedOption;
            agent.indices->optionAge[k] = 0u;
            agent.indices->optionFeature[k] = worst;
            agent.memory->optionUtility[k] = float(OAK_FEATURE_CAPACITY);
            agent.memory->actorAdvantage[k] = 0.0f;
            agent.memory->optionAverageReward[k] = 0.0f;
            agent.memory->optionModelAverageReward[k] = 0.0f;
            agent.memory->optionFitness[k] = 0.0f;
            agent.indices->lastStopAge[k] = 0u;

            agent.indices->replaceSlot[0] = 1u;
            agent.indices->replaceSlot[1] = worst;
            agent.indices->replaceSlot[2] = k;
            agent.indices->replaceSlot[3] = birth;
        }
    } else if (threadIndex == 0u) {
        agent.indices->replaceSlot[0] = 0u;
    }
    if (threadIndex == 0u) {
        if (agent.indices->replaceSlot[0] == 0u && (agent.stepCount % max(OAK_OPTION_COUNT, 1u)) == 0u) {
            float meanOptionFitness = 0.0f;
            for (uint k2 = 0u; k2 < OAK_OPTION_COUNT; ++k2)
                meanOptionFitness += agent.memory->optionFitness[k2];
            meanOptionFitness *= OAK_INV_OPTION_COUNT;
            uint replacedOption = 0u;
            float worstFitness = FLT_MAX;
            for (uint k2 = 1u; k2 < OAK_OPTION_COUNT; ++k2) {
                if (agent.indices->optionAge[k2] < OAK_FEATURE_CAPACITY) continue;
                float f = agent.memory->optionFitness[k2];
                if (f < worstFitness) {
                    worstFitness = f;
                    replacedOption = k2;
                }
            }
            if (replacedOption != 0u && worstFitness < 0.0f && worstFitness < meanOptionFitness) {
                agent.indices->optionAge[replacedOption] = 0u;
                agent.indices->lastStopAge[replacedOption] = 0u;
                agent.memory->optionFitness[replacedOption] = 0.0f;
                agent.memory->optionAverageReward[replacedOption] = 0.0f;
                agent.memory->optionModelAverageReward[replacedOption] = 0.0f;
                agent.memory->actorAdvantage[replacedOption] = 0.0f;
                agent.indices->replaceSlot[0] = 2u;
                agent.indices->replaceSlot[2] = replacedOption;
                agent.indices->replaceSlot[3] = agent.stepCount;
            }
        }
    }

    simdgroup_barrier(mem_flags::mem_device);
    uint replacement = agent.indices->replaceSlot[0];
    if (replacement != 0u) {
        uint worst = agent.indices->replaceSlot[1], k = agent.indices->replaceSlot[2],
             birth = agent.indices->replaceSlot[3];
        float policyPerturbScale = 1.0f / float(max(OAK_ACTION_PARAMS, 1u));
        uint policyN = OAK_POLICY_MAX;

        if (threadIndex == 0u) {
            predictionsChanged[0] = 1.0f;
            agent.indices->parked[k] = 0u;
        }
        uint slot = k;
        uint baseSlot = 0u;
        for (uint i = threadIndex; i < OAK_FEATURE_CAPACITY; i += OAK_SIMD_WIDTH) {
            device Record *rec = optionRecord(agent, i, slot);
            rec->value = 0.0f;
            rec->aux = 0.0f;
            rec->reward = 0.0f;
            rec->duration = 0.0f;
            rec->next = 0.0f;
            rec->nextAux = 0.0f;
            device const Record *base = optionRecord(agent, i, baseSlot);
            for (uint d = 0u; d < policyN; ++d) {
                uint h = (i * 2654435761u) ^ (d * 40503u) ^ (k * 2246822519u) ^ (birth * 3266489917u) ^
                         agent.stepCount;
                h ^= h >> 13;
                h *= 2654435761u;
                h ^= h >> 16;
                float hashedNoise = float(h & 0xffffu) / 32768.0f - 1.0f;

                float w0 = base->policy[d];
                rec->policy[d] =
                    w0 + hashedNoise * policyPerturbScale * rsqrt(float(OAK_ACTIVE_CAPACITY));
            }
        }
        simdgroup_barrier(mem_flags::mem_device);

        if (replacement == 1u)
            for (uint ok = threadIndex; ok < OAK_OPTION_COUNT; ok += OAK_SIMD_WIDTH) {
                *optionRecord(agent, worst, ok) = Record{};
            }
        simdgroup_barrier(mem_flags::mem_device);
        if (threadIndex == 0u) {
            uint optionFeatureIndex =
                (replacement == 1u) ? worst : (agent.indices->optionFeature[k] % OAK_FEATURE_CAPACITY);
            optionRecord(agent, optionFeatureIndex, slot)->value = agent.memory->inverseFeatureNorm;
        }
    }
}

static inline void updateUtilities(Agent agent, uint threadIndex) {
    uint tPrev = agent.stepCount - 1u;
    device const float *previousFeatures = featureValues(agent, 1u - agent.stateSlot);
    device const float *currentFeatures = featureValues(agent, agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, 1u - agent.stateSlot);
    float trackingRate = trackingRateFor(tPrev, OAK_FEATURE_CAPACITY);

    const uint activeSlotsPerThread = OAK_ACTIVE_CAPACITY / OAK_SIMD_WIDTH;
    uint iS[OAK_ACTIVE_SLOTS_MAX];
    float optionTdMagnitude[OAK_ACTIVE_SLOTS_MAX], keyboardSearchSignal[OAK_ACTIVE_SLOTS_MAX];
    for (uint s = 0u; s < activeSlotsPerThread; s += 1u) {
        iS[s] = previousActive[threadIndex + s * OAK_SIMD_WIDTH];
        optionTdMagnitude[s] = 0.0f;
        keyboardSearchSignal[s] = 0.0f;
    }
    float valueTdError = agent.memory->td;
    for (uint m = 0u; m < OAK_OPTION_COUNT; m += OAK_SIMD_WIDTH) {
        uint kH = m + threadIndex;
        bool kLive = kH < OAK_OPTION_COUNT;
        uint ofH = kLive ? agent.indices->optionFeature[kH] : 0u;
        float tdH = kLive ? agent.memory->optionTd[kH] : 0.0f;
        float kbH = kH == 0u ? 1.0f : 0.0f;
        float aaH = kLive ? agent.memory->actorAdvantage[kH] : 0.0f;
        float imH = kLive ? agent.memory->importance[kH] : 0.0f;
        uint kEnd = min(OAK_SIMD_WIDTH, OAK_OPTION_COUNT - m);
        for (uint l = 0u; l < kEnd; ++l) {
            uint ofV = simd_shuffle(ofH, (ushort)l);
            float tdV = simd_shuffle(tdH, (ushort)l);
            float kbV = simd_shuffle(kbH, (ushort)l);
            float aaV = simd_shuffle(aaH, (ushort)l);
            float imV = simd_shuffle(imH, (ushort)l);
            uint ofFeature = ofV % OAK_FEATURE_CAPACITY;
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
    for (uint s = 0u; s < activeSlotsPerThread; s += 1u) {
        uint activeSlot = threadIndex + s * OAK_SIMD_WIDTH;
        uint i = iS[s];
        float optionValueSignal = optionTdMagnitude[s], keyboardSignal = keyboardSearchSignal[s];
        float planningSignal = fabs(valueTdError);

        float selectedPolicyLoad = 0.0f;
        device const Record *selected = optionRecord(agent, i, 0u);
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d) selectedPolicyLoad += fabs(selected->policy[d]);
        float load =
            fabs(previousFeatures[i]) * (fabs(agent.parameters->valueWeight[i]) + selectedPolicyLoad);
        load = finiteOrZero(load);
        float featureUtilitySignal = fabs(planningSignal * previousFeatures[i]) +
                                     fabs((optionValueSignal + keyboardSignal) * previousFeatures[i]) + load;
        agent.memory->featureUtility[i] +=
            trackingRate * (featureUtilitySignal - agent.memory->featureUtility[i]);

        {
            float bonusHighWater = agent.memory->bonusWeight[i];
            float valueWeight = agent.parameters->valueWeight[i];
            if (valueWeight > bonusHighWater) agent.memory->bonusWeight[i] = valueWeight;
        }
        uint age = agent.indices->featureMeta[i].w;
        metaOptimizeUpdate(agent.memory->valueMeta, i, valueTdError, previousFeatures[i], age, OAK_META_RATE,
                           OAK_TRACKING_RATE);
        uint metaOptimizeSlot = tPrev % OAK_SIMD_WIDTH;
        if (activeSlot == metaOptimizeSlot) {
            uint metaOptimizeOption = tPrev % max(OAK_OPTION_COUNT, 1u);
            float optionTdSigned = agent.memory->optionTd[metaOptimizeOption];
            float modelSigned = agent.memory->optionRewardTd[metaOptimizeOption];
            metaOptimizeUpdate(agent.memory->optionValueMeta, i, optionTdSigned, previousFeatures[i], age,
                               OAK_META_RATE, OAK_TRACKING_RATE);
            metaOptimizeUpdate(agent.memory->modelMeta, i, modelSigned, previousFeatures[i], age,
                               OAK_META_RATE, OAK_TRACKING_RATE);
        }
    }
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        float err = agent.memory->optionPriority[k];
        float keyboardMass = (k == 0u ? 1.0f : 0.0f);
        float optionUtilitySignal = agent.memory->optionTd[k] * agent.memory->optionTd[k] +
                                    keyboardMass * fabs(agent.memory->actorAdvantage[k]) - err;
        agent.memory->optionUtility[k] +=
            trackingRate * (optionUtilitySignal - agent.memory->optionUtility[k]);
    }

    {
        float utilSum = 0.0f;
        uint maturedLive = 0u;
        for (uint k = 0u; k < OAK_OPTION_COUNT; ++k) {
            if (agent.indices->parked[k] != 0u || agent.indices->optionAge[k] < OAK_FEATURE_CAPACITY)
                continue;
            float u = agent.memory->optionUtility[k];
            utilSum += finiteOrZero(u);
            maturedLive += 1u;
        }
        float meanLiveUtil = (maturedLive > 0u) ? utilSum / float(maturedLive) : 0.0f;
        uint revivedOption = 0xffffffffu;
        for (uint k = 1u + threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
            if (agent.indices->parked[k] == 0u) continue;
            float u = agent.memory->optionUtility[k];
            bool revived = (isfinite(u) && u >= meanLiveUtil);
            if (revived) revivedOption = min(revivedOption, k);
        }
        revivedOption = simd_min(revivedOption);
        uint optionToPark = 0xffffffffu;
        float parkUtility = FLT_MAX;
        for (uint k = 1u + threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
            bool live = agent.indices->parked[k] == 0u || k == revivedOption;
            if (!live || agent.indices->optionAge[k] < OAK_FEATURE_CAPACITY) continue;
            float u = agent.memory->optionUtility[k];
            u = finiteOrZero(u);
            if (u < meanLiveUtil && u < parkUtility) {
                optionToPark = k;
                parkUtility = u;
            }
        }
        float parkUtilityAll = simd_min(parkUtility);
        optionToPark = (parkUtilityAll == FLT_MAX)
                           ? 0xffffffffu
                           : simd_min(parkUtility == parkUtilityAll ? optionToPark : 0xffffffffu);
        if (threadIndex == 0u) {
            if (revivedOption != 0xffffffffu) agent.indices->parked[revivedOption] = 0u;
            if (optionToPark != 0xffffffffu) agent.indices->parked[optionToPark] = 1u;
        }
    }
    syncMemory();

    {
        float meanOptionFitness = 0.0f;
        for (uint k = 0u; k < OAK_OPTION_COUNT; ++k) meanOptionFitness += agent.memory->optionFitness[k];
        meanOptionFitness *= OAK_INV_OPTION_COUNT;
        float previousFeatureNormSq = agent.memory->previousFeatureNormSq;
        if (!isfinite(previousFeatureNormSq) || previousFeatureNormSq < 0.0f) previousFeatureNormSq = 0.0f;
        float rate = OAK_TRACKING_RATE * OAK_INV_OPTION_COUNT / (1.0f + previousFeatureNormSq);

        const uint policyN = OAK_POLICY_MAX;
        float fitnessSearchDirection = 0.0f;
        float baseOptionMean = 0.0f;
        if (threadIndex < policyN) {
            baseOptionMean = agent.memory->optionPolicyMean[threadIndex];
            baseOptionMean = finiteOrZero(baseOptionMean);
        }
        float fitnessDirection = 0.0f;
        for (uint k = 1u; k < OAK_OPTION_COUNT; ++k) {
            float f = agent.memory->optionFitness[k];
            if (agent.indices->parked[k] != 0u) continue;
            if (threadIndex < policyN) {
                float fitnessCentered = f - meanOptionFitness;
                float mk = agent.memory->optionPolicyMean[k * OAK_POLICY_MAX + threadIndex];
                fitnessDirection =
                    fma(fitnessCentered, finiteOrZero(mk) - baseOptionMean, fitnessDirection);
            }
        }
        if (threadIndex < policyN)
            fitnessSearchDirection =
                isfinite(fitnessDirection) ? (fitnessDirection / (1.0f + fabs(fitnessDirection))) : 0.0f;
        uint activeCnt = OAK_ACTIVE_CAPACITY;
        float direction[OAK_POLICY_MAX];
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d)
            direction[d] = simd_shuffle(fitnessSearchDirection, (ushort)d);
        for (uint a = threadIndex; a < activeCnt; a += OAK_SIMD_WIDTH) {
            uint i = previousActive[a];
            float x = previousFeatures[i];
            device Record *weights = optionRecord(agent, i, 0u);
            for (uint d = 0u; d < policyN; ++d) {
                float w0 = weights->policy[d];
                weights->policy[d] = keepFinite(w0, w0 + rate * direction[d] * x);
            }
        }
    }
    {
        for (uint n = threadIndex + (((tPrev >> 1u) & 1u) != 0u ? OAK_SIMD_WIDTH : 0u);
             n < OAK_FEATURE_CAPACITY; n += 2u * OAK_SIMD_WIDTH) {
            uint4 nursery = agent.indices->nurseryMeta[n];
            float constructed = constructedValue(agent, previousFeatures, nursery.x, nursery.y, nursery.z);
            uint age = nursery.w;
            float rate = max(1.0f / (float(age) + 1.0f), OAK_TRACKING_RATE);

            device float4 *nurseryStatsVec = (device float4 *)&agent.memory->nurseryStats[(n) * 4u];
            float4 st = *nurseryStatsVec;
            st.x += rate * (constructed - st.x);
            st.y += rate * (constructed * constructed - st.y);
            float var = max(st.y - st.x * st.x, 0.0f);

            float ridge = max(1.0f / (float(age) + 1.0f), 1.0f / float(2u * OAK_PERCEPT_COUNT));
            float standardizedCandidate = (constructed - st.x) * rsqrt(var + ridge);
            float candidateUtilitySignal = fabs(agent.memory->td * standardizedCandidate);
            if (isfinite(candidateUtilitySignal)) {
                st.z += trackingRate * (candidateUtilitySignal - st.z);
                st.w += trackingRate * (candidateUtilitySignal * candidateUtilitySignal - st.w);
            }
            *nurseryStatsVec = st;
            if (age < 0xffffffffu) agent.indices->nurseryMeta[n].w = age + 1u;
        }
    }
    {
        for (uint i = threadIndex; i < OAK_FEATURE_CAPACITY; i += OAK_SIMD_WIDTH)
            agent.memory->optionUseByFeature[i] = 0.0f;
        syncMemory();
        if (threadIndex == 0u) {
            for (uint k = 0u; k < OAK_OPTION_COUNT; ++k) {
                float use = agent.memory->optionUtility[k] + (k == 0u ? 1.0f : 0.0f);
                if (isfinite(use))
                    agent.memory
                        ->optionUseByFeature[agent.indices->optionFeature[k] % OAK_FEATURE_CAPACITY] += use;
            }
        }

        for (uint n = threadIndex; n < OAK_FEATURE_CAPACITY; n += OAK_SIMD_WIDTH) {
            float nurseryUtility = agent.memory->nurseryStats[(n) * 4u + 2u];
            float lowerConfidenceBound = -FLT_MAX;
            uint nurseryAge = agent.indices->nurseryMeta[n].w;
            if (nurseryAge >= OAK_FEATURE_CAPACITY && isfinite(nurseryUtility)) {
                float nurseryUtilityVariance =
                    max(agent.memory->nurseryStats[(n) * 4u + 3u] - nurseryUtility * nurseryUtility, 0.0f);
                float nurseryUtilityRate = max(1.0f / (float(nurseryAge) + 1.0f), OAK_TRACKING_RATE);
                float v = nurseryUtility - sqrt(nurseryUtilityVariance * nurseryUtilityRate);
                if (isfinite(v)) lowerConfidenceBound = v;
            }
            agent.memory->nurseryScore[n] = lowerConfidenceBound;
        }
    }
    syncMemory();
    {
        for (uint i = OAK_PERCEPT_COUNT + threadIndex + (((tPrev >> 1u) & 1u) != 0u ? OAK_SIMD_WIDTH : 0u);
             i < OAK_FEATURE_CAPACITY; i += 2u * OAK_SIMD_WIDTH) {
            device float4 *replaceScoreVec = (device float4 *)&agent.memory->replaceScore[i * 4u];
            uint4 meta = agent.indices->featureMeta[i];
            if (meta.w < OAK_FEATURE_CAPACITY) {
                *replaceScoreVec = float4(0.0f, 0.0f, 0.0f, 0.0f);
                continue;
            }
            uint parentA = meta.x % OAK_FEATURE_CAPACITY;
            float util = agent.memory->featureUtility[i];
            util = finiteOrZero(util);
            float redundancy = fabs(currentFeatures[i] * currentFeatures[parentA]);
            redundancy = finiteOrZero(redundancy);
            float downstreamUse =
                agent.memory->optionUseByFeature[i] + fabs(agent.parameters->valueWeight[i]);
            downstreamUse = finiteOrZero(downstreamUse);
            *replaceScoreVec = float4(util, redundancy, downstreamUse, 1.0f);
        }
    }
}

kernel void oak_fleet_init(device float *params [[buffer(0)]], device float *stateArena [[buffer(1)]],
                           device uint *uintArena [[buffer(2)]], device OakFleetLane *lanes [[buffer(6)]],
                           device const uint *laneSeeds [[buffer(7)]],
                           constant OakFleetEnvCfg &ecfg [[buffer(8)]],
                           constant uint &laneOffset [[buffer(10)]], constant float4 &initial [[buffer(12)]],
                           device uint *envArena [[buffer(14)]], device const uint *envOff [[buffer(15)]],
                           constant uint *mjModels [[buffer(16)]],
                           device const float *envConstants [[buffer(19)]],
                           uint gid [[thread_position_in_grid]]) {
    uint gidLocal = gid;
    gid += laneOffset;
    if (gid >= ecfg.laneCount) return;
    device OakFleetLane *lane = &lanes[gid];
    device uint *envState = (envArena + envOff[gid]);
    resetLane(lane, envState, mjModels, OAK_SPEC_ENV_KIND, laneSeeds[gid]);
    Agent agent = bindAgent(params, stateArena, uintArena, gidLocal, lane->steps);
    uint traceChannels = max(OAK_TRACE_LEVELS * agent.observationCount, 1u);
    for (uint feature = 0; feature < OAK_FEATURE_CAPACITY; feature++) {
        uint parent = feature % (feature % 4u == 3u ? traceChannels : OAK_PERCEPT_COUNT);
        agent.indices->nurseryMeta[feature] =
            uint4(parent, (feature * 7u + 3u) % OAK_PERCEPT_COUNT, feature % 4u, 0u);
        if (feature >= OAK_PERCEPT_COUNT) {
            uint source = feature - OAK_PERCEPT_COUNT;
            agent.indices->featureMeta[feature] =
                source < traceChannels ? uint4(source, 0u, 3u, 0u)
                                       : uint4(source % OAK_PERCEPT_COUNT,
                                               (source * 7u + 3u) % OAK_PERCEPT_COUNT, source % 3u, 0u);
        }
        float4 meta = float4(initial.y, initial.z, 0.0f, 0.0f);
        agent.memory->valueMeta[feature] = meta;
        agent.memory->optionValueMeta[feature] = meta;
        agent.memory->modelMeta[feature] = meta;
    }
    for (uint option = 0; option < OAK_OPTION_COUNT; option++) {
        agent.indices->optionFeature[option] = option;
        optionRecord(agent, option, option)->value = initial.x;
        agent.memory->optionUtility[option] = float(OAK_FEATURE_CAPACITY);
        agent.memory->actorMeta[option].x = 1.0f;
        uint hash = gid * 2654435761u + option * 40503u + 977u;
        uint action = hash % OAK_POLICY_MAX;
        float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
        optionRecord(agent, 0u, option)->policy[action] = sign * initial.w;
    }

    for (uint o = 0u; o < agent.observationCount; ++o) {
        agent.memory->observationMeta[o] = float4(1.0f / float(2u * OAK_PERCEPT_COUNT),
                                                  log(1.0f / float(2u * OAK_PERCEPT_COUNT)), 0.0f, 0.0f);
    }
    observeLane(OAK_SPEC_ENV_KIND, laneSeeds[gid], mjModels, envConstants, ecfg, envState,
                agent.memory->observation);
    agent.memory->reward = 0.0f;
    for (uint d = 0u; d < OAK_ACTION_PARAMS; ++d) agent.memory->actionSignal[d] = 0.0f;
    for (uint i = 0u; i < OAK_OPTION_COUNT * OAK_POLICY_MAX; ++i) agent.memory->optionActorCredit[i] = 0.0f;
    for (uint k = 0u; k < OAK_OPTION_COUNT; ++k) {
        agent.memory->importance[k] = 1.0f;
        agent.memory->importanceStaging[k] = 1.0f;
        agent.indices->parked[k] = 0u;
    }
    agent.memory->inverseFeatureNorm = rsqrt(float(max(OAK_FEATURE_CAPACITY, 1u)));
}

static inline void sampleDiscreteAction(Agent agent, device OakFleetLane *lane, device float *pending,
                                        device const uint *laneSeeds, uint gid, uint threadIndex) {
    uint discreteActionCount = OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionCount;
    uint stepIdx = lane->steps;
    uint4 rw = philox4x32(uint2(laneSeeds[gid], RNG_STREAM_ACT), uint4(stepIdx, 0u, 0u, 0u));
    uint4 rw1 = philox4x32(uint2(laneSeeds[gid], RNG_STREAM_ACT), uint4(stepIdx, 1u, 0u, 0u));
    uint words[8] = {rw.x, rw.y, rw.z, rw.w, rw1.x, rw1.y, rw1.z, rw1.w};

    float behaviorLogitMax = -FLT_MAX;
    for (uint d = 0u; d < discreteActionCount; ++d)
        behaviorLogitMax = max(behaviorLogitMax, finiteOrZero(agent.memory->policyMeanRaw[d]));
    float behaviorSoftmaxSum = 0.0f;
    for (uint d = 0u; d < discreteActionCount; ++d)
        behaviorSoftmaxSum += exp(finiteOrZero(agent.memory->policyMeanRaw[d]) - behaviorLogitMax);
    uint sampledAction = 0u;
    float bestGumbelScore = -FLT_MAX;
    for (uint d = 0u; d < discreteActionCount; ++d) {
        float mu = finiteOrZero(agent.memory->policyMeanRaw[d]);

        float uniform = min(u01open(words[d & 7u]), 0x1.fffffep-1f);
        float gumbelScore = mu - log(-log(uniform));
        if (gumbelScore > bestGumbelScore) {
            bestGumbelScore = gumbelScore;
            sampledAction = d;
        }
    }

    simdgroup_barrier(mem_flags::mem_device);
    if (threadIndex < discreteActionCount)
        pending[threadIndex] = (threadIndex == 0u) ? float(sampledAction) : 0.0f;
    if (threadIndex < OAK_ACTION_PARAMS) {
        float actionSignal = 0.0f;
        if (threadIndex < discreteActionCount)
            actionSignal = ((threadIndex == sampledAction) ? 1.0f : 0.0f) -
                           exp(finiteOrZero(agent.memory->policyMeanRaw[threadIndex]) - behaviorLogitMax) /
                               behaviorSoftmaxSum;
        agent.memory->actionSignal[threadIndex] = actionSignal;
    }
    if (threadIndex == 0u) lane->steps = stepIdx + 1u;
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        uint off = k * OAK_POLICY_MAX;
        float optionLogitMax = -FLT_MAX;
        for (uint d = 0u; d < discreteActionCount; ++d)
            optionLogitMax = max(optionLogitMax, finiteOrZero(agent.memory->optionLogit[off + d]));
        float optionSoftmaxSum = 0.0f;
        for (uint d = 0u; d < discreteActionCount; ++d)
            optionSoftmaxSum += exp(finiteOrZero(agent.memory->optionLogit[off + d]) - optionLogitMax);
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d) {
            float credit = 0.0f;
            if (d < discreteActionCount)
                credit =
                    ((d == sampledAction) ? 1.0f : 0.0f) -
                    exp(finiteOrZero(agent.memory->optionLogit[off + d]) - optionLogitMax) / optionSoftmaxSum;
            agent.memory->actorCreditStaging[off + d] = credit;
        }

        float behaviorActionProb =
            exp(finiteOrZero(agent.memory->policyMeanRaw[sampledAction]) - behaviorLogitMax) /
            behaviorSoftmaxSum;
        float optionActionProb =
            exp(finiteOrZero(agent.memory->optionLogit[off + sampledAction]) - optionLogitMax) /
            optionSoftmaxSum;
        float ratio = optionActionProb / behaviorActionProb;
        agent.memory->importanceStaging[k] = isfinite(ratio) ? min(ratio, float(OAK_OPTION_COUNT)) : 0.0f;
    }
}

static inline void sampleAction(Agent agent, device OakFleetLane *lanes, device uint *envArena,
                                device const uint *envOff, device const uint *laneSeeds, uint gid,
                                uint threadIndex) {
    device OakFleetLane *lane = &lanes[gid];
    device float *pending = ((device float *)(envArena + envOff[gid]));
    if (OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionKind == 1u) {
        sampleDiscreteAction(agent, lane, pending, laneSeeds, gid, threadIndex);
        return;
    }
    device float *scratch = agent.memory->sampleScratch;
    uint envActionCount = OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].actionCount;
    float sampleSigma = OAK_SAMPLE_SIGMA;
    uint stepIdx = lane->steps;

    if (threadIndex < envActionCount) {
        uint d = threadIndex;
        uint2 key = uint2(laneSeeds[gid], RNG_STREAM_ACT);
        uint pair = d >> 1u;
        uint4 r = philox4x32(key, uint4(stepIdx, pair >> 1u, 0u, 0u));

        uint w0 = (pair & 1u) ? r.z : r.x;
        uint w1 = (pair & 1u) ? r.w : r.y;
        float uniform1 = u01open(w0);
        float uniform2 = u01(w1);
        float boxMullerRadius = sqrt(-2.0f * log(uniform1));
        float boxMullerAngle = 2.0f * M_PI_F * uniform2;
        float standardNormal = ((d & 1u) == 0u) ? (boxMullerRadius * cos(boxMullerAngle))
                                                : (boxMullerRadius * sin(boxMullerAngle));
        float mu = finiteOrZero(agent.memory->policyMean[d]);
        float sample = finiteOrZero(mu + sampleSigma * standardNormal);
        float clippedAction = max(-1.0f, min(1.0f, sample));
        pending[d] = clippedAction;
        agent.memory->actionSignal[d] = clippedAction;
        agent.memory->actionSignal[envActionCount + d] = 0.0f;
        scratch[d] = sample;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (threadIndex == 0u) lane->steps = stepIdx + 1u;
    for (uint k = threadIndex; k < OAK_OPTION_COUNT; k += OAK_SIMD_WIDTH) {
        uint off = k * OAK_POLICY_MAX;
        float logImportance = 0.0f;
        for (uint d = 0u; d < envActionCount; ++d) {
            float sample = scratch[d];
            float optionMeanValue = finiteOrZero(agent.memory->optionPolicyMean[off + d]);

            agent.memory->actorCreditStaging[off + d] =
                (sample - optionMeanValue) * (1.0f - optionMeanValue * optionMeanValue);
            float behaviorResidual = sample - finiteOrZero(agent.memory->policyMean[d]);
            float optionResidual = sample - optionMeanValue;
            logImportance += behaviorResidual * behaviorResidual - optionResidual * optionResidual;
        }

        float ratio = exp(logImportance * OAK_INV_TWO_SIGMA_SQUARED);
        agent.memory->importanceStaging[k] = isfinite(ratio) ? min(ratio, float(OAK_OPTION_COUNT)) : 0.0f;
    }
}

static inline void stepEnvironment(device float *params, device float *stateArena, device uint *uintArena,
                                   device OakFleetLane *lanes, device uint *envArena,
                                   device const uint *envOff, constant uint *mjModels,
                                   device float *mjScratch, device const uint *mjScratchOff,
                                   device const float *envConstants, device const uint *laneSeeds,
                                   constant OakFleetEnvCfg &ecfg, device float *recordBuf, uint gid,
                                   uint gidLocal, uint threadIndexInWave, bool wide) {
    device OakFleetLane *lane = &lanes[gid];
    uint envK = OAK_SPEC_ENV_KIND;
    float ctrl[OAK_ENV_ACT_MAX];
    uint done = 0u;
    device uint *envState = (envArena + envOff[gid]);
    device float *obsOut = laneEnvObs(envState, envK);
    float reward =
        advanceEnvironment(lane, envState, mjModels, mjScratch + mjScratchOff[gid], envConstants, envK,
                           laneSeeds[gid], ecfg, ctrl, obsOut, done, threadIndexInWave, wide);
    if (threadIndexInWave != 0u) return;
    uint stepNum = lane->steps;
    accumulateReward(lane, reward, stepNum, ecfg);
    {
        Agent agent = bindAgent(params, stateArena, uintArena, gidLocal, lane->steps);
        for (uint i = 0u; i < agent.observationCount && i < OAK_ENV_OBS_MAX; ++i) {
            agent.memory->observation[i] = obsOut[i];
            if (!isfinite(obsOut[i]) && lane->deathStep == OAK_DEATH_NONE) {
                lane->deathStep = stepNum;
                lane->deathDetector = OAK_DET_ENV_OBS;
            }
        }
        agent.memory->reward = reward;
    }
    recordPose(envK, mjModels, envState, ctrl, stepNum, done, reward, ecfg, recordBuf, gid);
}

kernel void
oak_megastep(device float *params [[buffer(0)]], device float *stateArena [[buffer(1)]],
             device uint *uintArena [[buffer(2)]], device OakFleetLane *lanes [[buffer(6)]],
             device const uint *laneSeeds [[buffer(7)]], constant OakFleetEnvCfg &ecfg [[buffer(8)]],
             device float *recordBuf [[buffer(10)]], constant uint3 &dispatchSpan [[buffer(11)]],
             device uint *envArena [[buffer(14)]], device const uint *envOff [[buffer(15)]],
             constant uint *mjModels [[buffer(16)]], device float *mjScratch [[buffer(17)]],
             device const uint *mjScratchOff [[buffer(18)]], device const float *envConstants [[buffer(19)]],
             threadgroup float *continuations [[threadgroup(0)]], uint tgIdx [[threadgroup_position_in_grid]],
             uint threadIndex [[thread_position_in_threadgroup]],
             uint threadIndexInWave [[thread_index_in_simdgroup]],
             uint waveIndex [[simdgroup_index_in_threadgroup]]) {
    const bool wideKind = OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].laneThreads > 1u;
    const bool planOnly = waveIndex >= OAK_MEGA_G;
    uint laneBase = (tgIdx + dispatchSpan.z) * OAK_MEGA_G;
    uint laneShift = dispatchSpan.z * OAK_MEGA_G;
    uint laneSg = planOnly ? waveIndex - OAK_MEGA_G : waveIndex;
    uint wideLane = laneBase + laneSg;
    uint optionCount = OAK_OPTION_COUNT;
    threadgroup float *lStash = continuations + laneSg * 5u * optionCount;
    threadgroup float *lDirty = continuations + 5u * OAK_MEGA_G * optionCount + laneSg;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {
        if (!planOnly) {
            Agent agent = bindAgent(params, stateArena, uintArena, wideLane - laneShift, dispatchSpan.y + it);
            if ((OAK_MEGA_STAGES & 1u) != 0u) {
                constructFeatures(agent, threadIndexInWave);
                predictAndControl(agent, threadIndexInWave, lStash, lDirty);
            }
            syncMemory();
            if ((OAK_MEGA_STAGES & 8u) != 0u) replaceWorstFeature(agent, threadIndexInWave, lDirty);
            simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            agent.stepCount += 1u;
            agent.stateSlot ^= 1u;
            if ((OAK_MEGA_STAGES & 16u) != 0u) {
                emitOptionActions(agent, threadIndexInWave, lDirty[0] != 0.0f);
                simdgroup_barrier(mem_flags::mem_device);
                writePrimaryPolicy(agent, threadIndexInWave);
            }
            syncMemory();
            if ((OAK_MEGA_STAGES & 32u) != 0u)
                sampleAction(agent, lanes, envArena, envOff, laneSeeds, wideLane, threadIndexInWave);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if ((OAK_MEGA_STAGES & 64u) != 0u) {
            if (wideKind) {
                if (!planOnly)
                    stepEnvironment(params, stateArena, uintArena, lanes, envArena, envOff, mjModels,
                                    mjScratch, mjScratchOff, envConstants, laneSeeds, ecfg, recordBuf,
                                    wideLane, wideLane - laneShift, threadIndexInWave, true);
            } else if (threadIndex < OAK_MEGA_G) {
                uint packedLane = laneBase + threadIndex;
                stepEnvironment(params, stateArena, uintArena, lanes, envArena, envOff, mjModels, mjScratch,
                                mjScratchOff, envConstants, laneSeeds, ecfg, recordBuf, packedLane,
                                packedLane - laneShift, 0u, false);
            }
        }
        uint planStep = dispatchSpan.y + it + 1u;
        uint planSkipPos =
            ((planStep % OAK_PLAN_PERIOD) + OAK_PLAN_PERIOD - (laneBase % OAK_PLAN_PERIOD)) % OAK_PLAN_PERIOD;
        bool skipInThisGroup = planSkipPos < OAK_MEGA_G;
        uint idx = laneSg;
        bool plans = planOnly && !(skipInThisGroup && laneSg == planSkipPos);
        if (!wideKind) {
            plans = !skipInThisGroup || waveIndex > 0u;
            idx = skipInThisGroup ? (waveIndex - 1u) : waveIndex;
            if (skipInThisGroup && idx >= planSkipPos) idx += 1u;
        }
        if (plans) {
            uint wLane = laneBase + idx;
            Agent planAgent = bindAgent(params, stateArena, uintArena, wLane - laneShift, planStep);
            uint previousSlot = 1u - planAgent.stateSlot;
            threadgroup const float *pStash = continuations + idx * 5u * optionCount;
            bool stashDirty = continuations[5u * OAK_MEGA_G * optionCount + idx] != 0.0f;
            if ((OAK_MEGA_STAGES & 2u) != 0u)
                planAllOptions(planAgent, threadIndexInWave, pStash, stashDirty, previousSlot);
            if ((OAK_MEGA_STAGES & 4u) != 0u) {
                device const float *prevS = featureValues(planAgent, previousSlot);
                device const float *currentFeatures = featureValues(planAgent, 1u - previousSlot);
                learnFromTransition(planAgent, prevS, currentFeatures, threadIndexInWave, previousSlot,
                                    pStash, stashDirty);
            }
            simdgroup_barrier(mem_flags::mem_device);
            if ((OAK_MEGA_STAGES & 8u) != 0u) updateUtilities(planAgent, threadIndexInWave);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
