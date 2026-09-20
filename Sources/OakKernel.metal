#pragma METAL fp math_mode(safe)

constant uint OAK_SPEC_ENV_KIND [[function_constant(2)]];
constant uint OAK_MEGA_G [[function_constant(3)]];

inline void syncMemory() {
    simdgroup_barrier(mem_flags::mem_device);
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
    float policy[OAK_POLICY_BANK];
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
    float actorCredit[OAK_POLICY_MAX];
    float actorCreditStaging[OAK_POLICY_MAX];
    float logit[OAK_POLICY_MAX];
    float policyCoef[OAK_POLICY_MAX];
    float reward;
    float td;
    float advantage;
    float advantageCenter;
    float4 actorMeta;
    float featureUtility[OAK_FEATURE_CAPACITY];
    alignas(float4) float2 featureMoments[OAK_FEATURE_CAPACITY];
    float4 valueMeta[OAK_FEATURE_CAPACITY];
    float observationTimescale[OAK_CANDIDATES_PER_SLOT * OAK_OBSERVATION_CAPACITY];
    float rewardCenter;
    float valueScale;
    float previousFeatureNormSq;
    float averageReward;
    float sampleScratch[OAK_SAMPLE_SCRATCH];
    alignas(float4) float nurseryStats[OAK_FEATURE_STATS];
    float nurseryScore[OAK_FEATURE_CAPACITY];
    alignas(float4) float replaceScore[OAK_FEATURE_STATS];
};

struct AgentIndices {
    uint actorAge;
    uint replaced;
    uint replacedFeature;
    uint activeFeature[OAK_ACTIVE_STATES];
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

inline device float *policyRow(Agent agent, uint feature) {
    return agent.parameters->policy + feature * OAK_POLICY_MAX;
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
    uint simdWidth, threadgroupLanes;
};

kernel void oak_shape_probe(device OakShapeProbe *out [[buffer(0)]]) {
    *out = {OAK_FEATURE_CAPACITY,
            OAK_PERCEPT_COUNT,
            OAK_ACTION_PARAMS,
            sizeof(AgentParameters) / sizeof(float),
            sizeof(AgentMemory) / sizeof(float),
            sizeof(AgentIndices) / sizeof(uint),
            OAK_SIMD_WIDTH,
            OAK_THREADGROUP_LANES};
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

inline void predictAndControl(Agent agent, uint threadIndex) {
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
    }
    syncMemory();
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
    syncMemory();

    float policyLogit[OAK_POLICY_MAX];
    for (uint j = 0u; j < OAK_POLICY_MAX; ++j) policyLogit[j] = 0.0f;
    for (uint base = 0u; base < OAK_ACTIVE_CAPACITY; base += OAK_SIMD_WIDTH) {
        uint localCurrentIndex = currentActive[base + threadIndex];
        float localCurrentActivation = currentFeatures[localCurrentIndex];
        for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
            uint currentIndex = simd_shuffle(localCurrentIndex, (ushort)l);
            float currentActivation = simd_shuffle(localCurrentActivation, (ushort)l);
            if (currentActivation != 0.0f) {
                device const float *row = policyRow(agent, currentIndex);
                for (uint j = 0u; j < OAK_POLICY_MAX; ++j)
                    policyLogit[j] = fma(row[j], currentActivation, policyLogit[j]);
            }
        }
    }
    if (threadIndex == 0u) {
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) agent.memory->logit[j] = policyLogit[j];
        float advantageCenterOld = agent.memory->advantageCenter;
        float advantageError = agent.memory->td - advantageCenterOld;
        agent.memory->advantage = finiteOrZero(advantageError);
        agent.memory->advantageCenter =
            keepFinite(advantageCenterOld, advantageCenterOld + OAK_TRACKING_RATE * advantageError);
    }
    syncMemory();
    if (threadIndex == 0u) {
        float gaugeRate = trackingRateFor(agent.stepCount, OAK_FEATURE_CAPACITY);
        float gaugeNew = agent.memory->rewardCenter + gaugeRate * (currentValue - agent.memory->rewardCenter);
        agent.memory->rewardCenter = keepFinite(agent.memory->rewardCenter, gaugeNew);
    }
    syncMemory();

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
    for (uint j = threadIndex; j < OAK_POLICY_MAX; j += OAK_SIMD_WIDTH)
        agent.memory->actorCredit[j] = agent.memory->actorCreditStaging[j];

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

static inline void actorCoefficients(Agent agent, uint threadIndex) {
    if (threadIndex != 0u) return;
    float advantage = finiteOrZero(agent.memory->advantage);
    metaOptimizeUpdate(&agent.memory->actorMeta, 0u, advantage, 1.0f, agent.indices->actorAge, OAK_META_RATE,
                       OAK_TRACKING_RATE);
    float actorAlpha = agent.memory->actorMeta.x / OAK_ACTOR_ALPHA_DENOM;
    float previousFeatureNormSq = agent.memory->previousFeatureNormSq;
    for (uint d = 0u; d < OAK_POLICY_MAX; ++d) {
        float gradScale = agent.memory->actorCredit[d];
        float denom = 1.0f + actorAlpha * advantage * advantage * gradScale * gradScale * previousFeatureNormSq;
        float scale = (isfinite(denom) && denom > 0.0f) ? (1.0f / denom) : 0.0f;
        float policyCoef = actorAlpha * advantage * gradScale * scale;
        agent.memory->policyCoef[d] = finiteOrZero(policyCoef);
    }
}

static inline void learnFromTransition(Agent agent, uint threadIndex, uint previousSlot) {
    device const float *previousFeatures = featureValues(agent, previousSlot);
    device const float *currentFeatures = featureValues(agent, 1u - previousSlot);
    device const uint *previousActive = activeFeatures(agent, previousSlot);
    float coefficient[OAK_POLICY_MAX];
    for (uint j = 0u; j < OAK_POLICY_MAX; ++j) coefficient[j] = agent.memory->policyCoef[j];
    for (uint a = threadIndex; a < OAK_ACTIVE_CAPACITY; a += OAK_SIMD_WIDTH) {
        uint i = previousActive[a];
        float previousActivation = previousFeatures[i];
        if (previousActivation == 0.0f && currentFeatures[i] == 0.0f) continue;
        device float *row = policyRow(agent, i);
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) {
            float w = row[j];
            row[j] = keepFinite(w, w + coefficient[j] * previousActivation);
        }
    }

    float valueTdError = agent.memory->td;
    float valueStepScale = agent.memory->valueScale;
    for (uint a = threadIndex; a < OAK_ACTIVE_CAPACITY; a += OAK_SIMD_WIDTH) {
        uint i = previousActive[a];
        float w = agent.parameters->valueWeight[i];
        float delta = valueTdError * previousFeatures[i] * valueStepScale;
        agent.parameters->valueWeight[i] = keepFinite(w, w + agent.memory->valueMeta[i].x * delta);
    }
}

inline void emitPolicy(Agent agent, uint threadIndex, bool recompute) {
    float policyLogit[OAK_POLICY_MAX];
    for (uint j = 0u; j < OAK_POLICY_MAX; ++j) policyLogit[j] = 0.0f;
    if (recompute) {
        device const uint *active = activeFeatures(agent, agent.stateSlot);
        device const float *currentFeatures = featureValues(agent, agent.stateSlot);
        for (uint base = 0u; base < OAK_ACTIVE_CAPACITY; base += OAK_SIMD_WIDTH) {
            uint iH = active[base + threadIndex];
            float xH = currentFeatures[iH];
            for (uint l = 0u; l < OAK_SIMD_WIDTH; ++l) {
                uint i = simd_shuffle(iH, (ushort)l);
                float x = simd_shuffle(xH, (ushort)l);
                if (x == 0.0f) continue;
                device const float *row = policyRow(agent, i);
                for (uint j = 0u; j < OAK_POLICY_MAX; ++j)
                    policyLogit[j] = fma(row[j], x, policyLogit[j]);
            }
        }
    } else {
        for (uint j = 0u; j < OAK_POLICY_MAX; ++j) policyLogit[j] = agent.memory->logit[j];
    }
    if (threadIndex != 0u) return;
    for (uint j = 0u; j < OAK_POLICY_MAX; ++j) {
        float actionMean = tanh(policyLogit[j]);
        agent.memory->logit[j] = policyLogit[j];
        if (!isfinite(actionMean)) actionMean = agent.memory->policyMean[j];
        agent.memory->policyMean[j] = actionMean;
        agent.memory->policyMeanRaw[j] = finiteOrZero(policyLogit[j]);
    }
}

static inline void replaceWorstFeature(Agent agent, uint threadIndex) {
    device const float *currentFeatures = featureValues(agent, 1u - agent.stateSlot);

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

    if (threadIndex == 0u && agent.indices->actorAge < 0xffffffffu) agent.indices->actorAge += 1u;
    simdgroup_barrier(mem_flags::mem_device);
    uint candidateStart = OAK_PERCEPT_COUNT;
    uint eligible = simd_sum(eligiblePartial);
    float eligibleUtilSum = simd_sum(eligibleUtilPartial);
    float eligibleRedundancySum = simd_sum(eligibleRedundancyPartial);
    float eligibleDownstreamSum = simd_sum(eligibleDownstreamPartial);
    uint worst = (haveWorst != 0u) ? worstFeature : candidateStart;

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
        if (threadIndex == 0u) {
            uint4 nursery = agent.indices->nurseryMeta[bestNursery];
            agent.indices->featureMeta[worst] =
                uint4(nursery.x % OAK_FEATURE_CAPACITY, nursery.y % OAK_FEATURE_CAPACITY, nursery.z % 4u,
                      nursery.w);
            agent.memory->featureMoments[worst] = float2(agent.memory->nurseryStats[(bestNursery) * 4u],
                                                         agent.memory->nurseryStats[(bestNursery) * 4u + 1u]);
            agent.memory->featureUtility[worst] = agent.memory->nurseryStats[(bestNursery) * 4u + 2u];
            agent.parameters->valueWeight[worst] = 0.0f;

            agent.indices->nurseryMeta[bestNursery] =
                uint4(seedParentA, seedParentB, (nursery.z + 1u) % 4u, 0u);
            agent.memory->nurseryStats[(bestNursery) * 4u] = 0.0f;
            agent.memory->nurseryStats[(bestNursery) * 4u + 1u] = 0.0f;
            agent.memory->nurseryStats[(bestNursery) * 4u + 2u] = 0.0f;
            agent.memory->nurseryStats[(bestNursery) * 4u + 3u] = 0.0f;

            float initialStepSize = 1.0f / float(OAK_ACTIVE_CAPACITY);
            agent.memory->valueMeta[worst] = float4(initialStepSize, log(initialStepSize), 0.0f, 0.0f);

            agent.indices->replaced = 1u;
            agent.indices->replacedFeature = worst;
        }
    } else if (threadIndex == 0u) {
        agent.indices->replaced = 0u;
    }

    simdgroup_barrier(mem_flags::mem_device);
    if (agent.indices->replaced != 0u) {
        if (threadIndex < OAK_POLICY_MAX) policyRow(agent, agent.indices->replacedFeature)[threadIndex] = 0.0f;
        simdgroup_barrier(mem_flags::mem_device);
    }
}

static inline void updateUtilities(Agent agent, uint threadIndex) {
    uint tPrev = agent.stepCount - 1u;
    device const float *previousFeatures = featureValues(agent, 1u - agent.stateSlot);
    device const float *currentFeatures = featureValues(agent, agent.stateSlot);
    device const uint *previousActive = activeFeatures(agent, 1u - agent.stateSlot);
    float trackingRate = trackingRateFor(tPrev, OAK_FEATURE_CAPACITY);
    float valueTdError = agent.memory->td;
    float advantage = agent.memory->advantage;
    float planningSignal = fabs(valueTdError);
    for (uint a = threadIndex; a < OAK_ACTIVE_CAPACITY; a += OAK_SIMD_WIDTH) {
        uint i = previousActive[a];
        float actorSignal = (i == 0u) ? (fabs(advantage) + (fabs(advantage) + 1.0f)) : 0.0f;
        float selectedPolicyLoad = 0.0f;
        device const float *row = policyRow(agent, i);
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d) selectedPolicyLoad += fabs(row[d]);
        float load =
            fabs(previousFeatures[i]) * (fabs(agent.parameters->valueWeight[i]) + selectedPolicyLoad);
        load = finiteOrZero(load);
        float featureUtilitySignal = fabs(planningSignal * previousFeatures[i]) +
                                     fabs(actorSignal * previousFeatures[i]) + load;
        agent.memory->featureUtility[i] +=
            trackingRate * (featureUtilitySignal - agent.memory->featureUtility[i]);
        uint age = agent.indices->featureMeta[i].w;
        metaOptimizeUpdate(agent.memory->valueMeta, i, valueTdError, previousFeatures[i], age, OAK_META_RATE,
                           OAK_TRACKING_RATE);
    }
    syncMemory();
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
    syncMemory();
    {
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
            float downstreamUse = fabs(agent.parameters->valueWeight[i]);
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
        agent.memory->valueMeta[feature] = float4(initial.y, initial.z, 0.0f, 0.0f);
    }
    agent.memory->actorMeta.x = 1.0f;
    uint hash = gid * 2654435761u + 977u;
    uint action = hash % OAK_POLICY_MAX;
    float sign = ((hash >> 16u) & 1u) == 0u ? 1.0f : -1.0f;
    policyRow(agent, 0u)[action] = sign * initial.w;

    for (uint o = 0u; o < agent.observationCount; ++o) {
        agent.memory->observationMeta[o] = float4(1.0f / float(2u * OAK_PERCEPT_COUNT),
                                                  log(1.0f / float(2u * OAK_PERCEPT_COUNT)), 0.0f, 0.0f);
    }
    observeLane(OAK_SPEC_ENV_KIND, laneSeeds[gid], mjModels, envConstants, ecfg, envState,
                agent.memory->observation);
    agent.memory->reward = 0.0f;
    for (uint d = 0u; d < OAK_ACTION_PARAMS; ++d) agent.memory->actionSignal[d] = 0.0f;
    for (uint i = 0u; i < OAK_POLICY_MAX; ++i) agent.memory->actorCredit[i] = 0.0f;
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
    if (threadIndex == 0u) {
        lane->steps = stepIdx + 1u;
        float logitMax = -FLT_MAX;
        for (uint d = 0u; d < discreteActionCount; ++d)
            logitMax = max(logitMax, finiteOrZero(agent.memory->logit[d]));
        float softmaxSum = 0.0f;
        for (uint d = 0u; d < discreteActionCount; ++d)
            softmaxSum += exp(finiteOrZero(agent.memory->logit[d]) - logitMax);
        for (uint d = 0u; d < OAK_POLICY_MAX; ++d) {
            float credit = 0.0f;
            if (d < discreteActionCount)
                credit = ((d == sampledAction) ? 1.0f : 0.0f) -
                         exp(finiteOrZero(agent.memory->logit[d]) - logitMax) / softmaxSum;
            agent.memory->actorCreditStaging[d] = credit;
        }
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
    if (threadIndex == 0u) {
        lane->steps = stepIdx + 1u;
        for (uint d = 0u; d < envActionCount; ++d) {
            float sample = scratch[d];
            float meanValue = finiteOrZero(agent.memory->policyMean[d]);
            agent.memory->actorCreditStaging[d] = (sample - meanValue) * (1.0f - meanValue * meanValue);
        }
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
             uint tgIdx [[threadgroup_position_in_grid]],
             uint threadIndexInWave [[thread_index_in_simdgroup]],
             uint waveIndex [[simdgroup_index_in_threadgroup]]) {
    const bool wideKind = OAK_ENV_SPECS[OAK_SPEC_ENV_KIND].laneThreads > 1u;
    uint laneShift = dispatchSpan.z * OAK_MEGA_G;
    uint lane = (tgIdx + dispatchSpan.z) * OAK_MEGA_G + waveIndex;
    uint local = lane - laneShift;
    for (uint it = 0u; it < dispatchSpan.x; ++it) {
        Agent agent = bindAgent(params, stateArena, uintArena, local, dispatchSpan.y + it);
        if ((OAK_MEGA_STAGES & 1u) != 0u) {
            constructFeatures(agent, threadIndexInWave);
            predictAndControl(agent, threadIndexInWave);
        }
        syncMemory();
        if ((OAK_MEGA_STAGES & 8u) != 0u) replaceWorstFeature(agent, threadIndexInWave);
        syncMemory();
        agent.stepCount += 1u;
        agent.stateSlot ^= 1u;
        if ((OAK_MEGA_STAGES & 16u) != 0u) emitPolicy(agent, threadIndexInWave, agent.indices->replaced != 0u);
        syncMemory();
        if ((OAK_MEGA_STAGES & 32u) != 0u)
            sampleAction(agent, lanes, envArena, envOff, laneSeeds, lane, threadIndexInWave);
        syncMemory();
        uint previousSlot = 1u - agent.stateSlot;
        if ((OAK_MEGA_STAGES & 2u) != 0u) actorCoefficients(agent, threadIndexInWave);
        syncMemory();
        if ((OAK_MEGA_STAGES & 4u) != 0u) learnFromTransition(agent, threadIndexInWave, previousSlot);
        syncMemory();
        if ((OAK_MEGA_STAGES & 8u) != 0u) updateUtilities(agent, threadIndexInWave);
        syncMemory();
        if ((OAK_MEGA_STAGES & 64u) != 0u)
            stepEnvironment(params, stateArena, uintArena, lanes, envArena, envOff, mjModels, mjScratch,
                            mjScratchOff, envConstants, laneSeeds, ecfg, recordBuf, lane, local,
                            threadIndexInWave, wideKind);
        syncMemory();
    }
}
