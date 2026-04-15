#include <metal_stdlib>
using namespace metal;

struct KernelConfig { uint phase, obsDim, rank, actDim, batchSize, actionIsDiscrete; };
struct KernelScalars { uint stepCount, rewardCount, activeCount; float rewardRate; };
struct KernelStats {
    float rewardRate, latentPredictionMSE, rewardPredictionMSE, latentVarianceTrace, activeRank, actionDeltaL2Mean, planningActionChangeL2Mean;
    float rawNonFiniteCount, sanitizedNonFiniteCount, encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsRewardPrediction, maxAbsPolicyOutput;
};

inline float oak_sanitize(float value, thread uint& raw, thread uint& sanitized) {
    if (isfinite(value)) return value;
    raw += 1u;
    sanitized += 1u;
    return 0.0f;
}

inline void oak_flush(device atomic_uint *counters, uint phase, uint raw, uint sanitized) {
    if (raw) atomic_fetch_add_explicit(&counters[0], raw, memory_order_relaxed);
    if (sanitized) atomic_fetch_add_explicit(&counters[1], sanitized, memory_order_relaxed);
    if (phase < 3u && raw) atomic_fetch_add_explicit(&counters[2u + phase], raw, memory_order_relaxed);
}

inline float oak_masked_dot(
    device const float *weights,
    uint weightOffset,
    device const float *values,
    uint valueOffset,
    device const uint *activeList,
    uint activeCount,
    thread uint& raw,
    thread uint& sanitized
) {
    float sum = 0.0f;
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint index = activeList[slot];
        sum += weights[weightOffset + index] * oak_sanitize(values[valueOffset + index], raw, sanitized);
    }
    return oak_sanitize(sum, raw, sanitized);
}

inline uint oak_discrete_index(
    device const float *actionValues,
    uint actionOffset,
    uint actDim,
    thread uint& raw,
    thread uint& sanitized
) {
    uint best = 0u;
    float bestValue = -FLT_MAX;
    for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
        const float value = oak_sanitize(actionValues[actionOffset + actionIndex], raw, sanitized);
        if (value > bestValue) {
            bestValue = value;
            best = actionIndex;
        }
    }
    return best;
}

inline float oak_reward_prediction_continuous(
    device const float *rewardWeights,
    device const float *stateValues,
    uint stateOffset,
    device const float *actionValues,
    uint actionOffset,
    device const uint *activeList,
    uint activeCount,
    uint rank,
    uint actDim,
    thread uint& raw,
    thread uint& sanitized
) {
    float sum = oak_masked_dot(rewardWeights, 0u, stateValues, stateOffset, activeList, activeCount, raw, sanitized);
    for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) sum += rewardWeights[rank + actionIndex] * oak_sanitize(actionValues[actionOffset + actionIndex], raw, sanitized);
    return oak_sanitize(sum, raw, sanitized);
}

inline float oak_reward_prediction_discrete(
    device const float *rewardWeights,
    device const float *stateValues,
    uint stateOffset,
    device const uint *activeList,
    uint activeCount,
    uint rank,
    uint actionIndex,
    thread uint& raw,
    thread uint& sanitized
) {
    const uint weightOffset = actionIndex * (rank + 1u);
    return oak_sanitize(
        oak_masked_dot(rewardWeights, weightOffset, stateValues, stateOffset, activeList, activeCount, raw, sanitized) + rewardWeights[weightOffset + rank],
        raw,
        sanitized
    );
}

inline float oak_feature_transition(
    device const float *weights,
    uint weightOffset,
    float featureValue,
    device const float *actionValues,
    uint actionOffset,
    uint discreteIndex,
    bool actionIsDiscrete,
    uint actDim,
    thread uint& raw,
    thread uint& sanitized
) {
    const float recurrentWeight = oak_sanitize(weights[weightOffset], raw, sanitized);
    const float feature = oak_sanitize(featureValue, raw, sanitized);
    float norm = recurrentWeight * recurrentWeight;
    if (actionIsDiscrete) {
        const float actionWeight = oak_sanitize(weights[weightOffset + 1u + discreteIndex], raw, sanitized);
        const float inverseNorm = rsqrt(max(norm + actionWeight * actionWeight, FLT_EPSILON));
        return oak_sanitize(recurrentWeight * inverseNorm * feature + actionWeight * inverseNorm, raw, sanitized);
    }
    float sum = recurrentWeight * feature;
    for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
        const float actionWeight = oak_sanitize(weights[weightOffset + 1u + actionIndex], raw, sanitized);
        norm += actionWeight * actionWeight;
    }
    const float inverseNorm = rsqrt(max(norm, FLT_EPSILON));
    sum = recurrentWeight * inverseNorm * feature;
    for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
        const float actionWeight = oak_sanitize(weights[weightOffset + 1u + actionIndex], raw, sanitized);
        sum += actionWeight * inverseNorm * oak_sanitize(actionValues[actionOffset + actionIndex], raw, sanitized);
    }
    return oak_sanitize(sum, raw, sanitized);
}

inline void oak_predict_state(
    device const float *stateWeights,
    device const float *sourceState,
    uint sourceOffset,
    device float *targetState,
    uint targetOffset,
    device const float *actionValues,
    uint actionOffset,
    uint discreteIndex,
    bool actionIsDiscrete,
    device const uint *activeList,
    uint activeCount,
    uint rank,
    uint actDim,
    thread uint& raw,
    thread uint& sanitized
) {
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint feature = activeList[slot];
        targetState[targetOffset + feature] = oak_sanitize(
            oak_feature_transition(
                stateWeights,
                feature * (1u + actDim),
                sourceState[sourceOffset + feature],
                actionValues,
                actionOffset,
                discreteIndex,
                actionIsDiscrete,
                actDim,
                raw,
                sanitized
            ),
            raw,
            sanitized
        );
    }
}

inline float oak_residual_trace(
    device const float *featureResidual,
    device const float *stateValues,
    uint stateOffset,
    device const uint *activeList,
    uint activeCount,
    thread uint& raw,
    thread uint& sanitized
) {
    float residualMean = 0.0f;
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint feature = activeList[slot];
        residualMean += oak_sanitize(featureResidual[feature], raw, sanitized);
    }
    const float residualScale = max(
        activeCount > 0u ? residualMean / float(activeCount) : 0.0f,
        FLT_EPSILON
    );
    float weighted = 0.0f;
    float energy = 0.0f;
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint feature = activeList[slot];
        const float value = oak_sanitize(stateValues[stateOffset + feature], raw, sanitized);
        const float contribution = value * value;
        energy += contribution;
        weighted += (oak_sanitize(featureResidual[feature], raw, sanitized) / residualScale) * contribution;
    }
    return oak_sanitize(energy > FLT_EPSILON ? weighted / energy : 0.0f, raw, sanitized);
}

inline uint oak_argmax(device const float *values, uint offset, uint count, thread uint& raw, thread uint& sanitized) {
    uint best = 0u;
    float bestValue = -FLT_MAX;
    for (uint index = 0; index < count; ++index) {
        const float candidate = oak_sanitize(values[offset + index], raw, sanitized);
        if (candidate > bestValue) {
            bestValue = candidate;
            best = index;
        }
    }
    return best;
}

inline float oak_active_delta_energy(
    device const float *source,
    uint sourceOffset,
    device const float *target,
    uint targetOffset,
    device const uint *activeList,
    uint activeCount,
    thread uint& raw,
    thread uint& sanitized
) {
    float energy = 0.0f;
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint feature = activeList[slot];
        const float delta = oak_sanitize(target[targetOffset + feature] - source[sourceOffset + feature], raw, sanitized);
        energy += delta * delta;
    }
    return oak_sanitize(activeCount > 0u ? energy / float(activeCount) : 0.0f, raw, sanitized);
}

inline float oak_standardized_feature(
    device const float *featureMean,
    device const float *featureSecondMoment,
    device const float *stateValues,
    uint stateOffset,
    device const float *latentPower,
    uint row,
    uint feature,
    thread uint& raw,
    thread uint& sanitized
) {
    const float value = oak_sanitize(stateValues[stateOffset + feature], raw, sanitized) * sqrt(max(latentPower[row], FLT_EPSILON));
    const float mean = oak_sanitize(featureMean[feature], raw, sanitized);
    const float secondMoment = oak_sanitize(featureSecondMoment[feature], raw, sanitized);
    const float variance = max(secondMoment - mean * mean, FLT_EPSILON);
    return oak_sanitize((value - mean) * rsqrt(variance), raw, sanitized);
}

inline void oak_normalize_active_state(
    device float *values,
    uint offset,
    device const uint *activeList,
    uint activeCount,
    thread uint& raw,
    thread uint& sanitized
) {
    float norm = 0.0f;
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint feature = activeList[slot];
        const float value = oak_sanitize(values[offset + feature], raw, sanitized);
        norm += value * value;
    }
    if (activeCount == 0u) return;
    const float scale = rsqrt(max(norm / float(activeCount), FLT_EPSILON));
    for (uint slot = 0; slot < activeCount; ++slot) {
        const uint feature = activeList[slot];
        values[offset + feature] = oak_sanitize(values[offset + feature] * scale, raw, sanitized);
    }
}

kernel void oak_step(
    device const float *observation [[buffer(0)]],
    device float *observationMean [[buffer(1)]],
    device float *encoderWeights [[buffer(2)]],
    device float *policyWeights [[buffer(3)]],
    device float *valueWeights [[buffer(4)]],
    device float *rewardWeights [[buffer(5)]],
    device float *stateWeights [[buffer(6)]],
    device float *optionPolicyWeights [[buffer(7)]],
    device float *optionValues [[buffer(8)]],
    device float *previousState [[buffer(9)]],
    device float *previousAction [[buffer(10)]],
    device const float *reward [[buffer(11)]],
    device uint *done [[buffer(12)]],
    device float *state [[buffer(13)]],
    device float *action [[buffer(14)]],
    device float *latentError [[buffer(15)]],
    device float *rewardPrediction [[buffer(16)]],
    device float *td [[buffer(17)]],
    device float *actionGrad [[buffer(18)]],
    device float *obsScale [[buffer(19)]],
    device float *featureMean [[buffer(20)]],
    device KernelScalars *scalars [[buffer(21)]],
    device KernelStats *statsOut [[buffer(22)]],
    device float *featureSecondMoment [[buffer(23)]],
    device float *featureResidual [[buffer(24)]],
    device uint *activeState [[buffer(25)]],
    device float *rolloutState [[buffer(26)]],
    device float *latentPower [[buffer(27)]],
    device const float *actionBounds [[buffer(28)]],
    device atomic_uint *counters [[buffer(29)]],
    constant KernelConfig &config [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint obsDim = max(config.obsDim, 1u);
    const uint rank = max(config.rank, 1u);
    const uint actDim = max(config.actDim, 1u);
    const uint batchSize = max(config.batchSize, 1u);
    const bool actionIsDiscrete = config.actionIsDiscrete != 0u;
    device uint *activeMask = activeState;
    device uint *activeList = activeState + rank;
    const uint activeCount = min(max(scalars->activeCount, 1u), rank);
    uint raw = 0u, sanitized = 0u;

    if (config.phase == 0u) {
        const uint row = gid / rank;
        const uint feature = gid % rank;
        if (row < batchSize) {
            const uint obsOffset = row * obsDim;
            const uint stateOffset = row * rank;
            const float centered = oak_sanitize(observation[obsOffset + feature] - observationMean[feature], raw, sanitized);
            const float projected = oak_sanitize(centered * encoderWeights[feature], raw, sanitized);
            state[stateOffset + feature] = projected;
            rolloutState[stateOffset + feature] = centered * centered;
            latentError[stateOffset + feature] = projected * projected;
        }
        oak_flush(counters, 0u, raw, sanitized);
        return;
    }

    if (config.phase == 1u) {
        if (gid < batchSize) {
            const uint stateOffset = gid * rank;
            const uint actionOffset = gid * actDim;
            float norm = 0.0f;
            float obsEnergy = 0.0f;
            for (uint feature = 0; feature < rank; ++feature) {
                norm += oak_sanitize(latentError[stateOffset + feature], raw, sanitized);
                obsEnergy += oak_sanitize(rolloutState[stateOffset + feature], raw, sanitized);
            }
            latentPower[gid] = max(norm / float(rank), FLT_EPSILON);
            const float stateScale = rsqrt(max(norm / float(rank), FLT_EPSILON));
            for (uint feature = 0; feature < rank; ++feature) state[stateOffset + feature] = oak_sanitize(state[stateOffset + feature] * stateScale, raw, sanitized);
            obsScale[gid] = rsqrt(max(obsEnergy / float(obsDim), FLT_EPSILON));

            if (scalars->stepCount == 0u) {
                rewardPrediction[gid] = 0.0f;
                td[gid] = 0.0f;
                for (uint feature = 0; feature < rank; ++feature) latentError[stateOffset + feature] = 0.0f;
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) actionGrad[actionOffset + actionIndex] = 0.0f;
                oak_flush(counters, 0u, raw, sanitized);
                return;
            }

            const bool terminal = done[gid] != 0u;
            const uint previousDiscreteIndex = actionIsDiscrete ? oak_discrete_index(previousAction, actionOffset, actDim, raw, sanitized) : 0u;
            rewardPrediction[gid] = actionIsDiscrete
                ? oak_reward_prediction_discrete(rewardWeights, previousState, stateOffset, activeList, activeCount, rank, previousDiscreteIndex, raw, sanitized)
                : oak_reward_prediction_continuous(rewardWeights, previousState, stateOffset, previousAction, actionOffset, activeList, activeCount, rank, actDim, raw, sanitized);
            if (!terminal) {
                oak_predict_state(
                    stateWeights,
                    previousState,
                    stateOffset,
                    rolloutState,
                    stateOffset,
                    previousAction,
                    actionOffset,
                    previousDiscreteIndex,
                    actionIsDiscrete,
                    activeList,
                    activeCount,
                    rank,
                    actDim,
                    raw,
                    sanitized
                );
                oak_normalize_active_state(rolloutState, stateOffset, activeList, activeCount, raw, sanitized);
            }
            float latentEnergy = 0.0f;
            for (uint slot = 0; slot < activeCount; ++slot) {
                const uint feature = activeList[slot];
                const float predicted = terminal ? 0.0f : rolloutState[stateOffset + feature];
                const float error = oak_sanitize(terminal ? 0.0f : state[stateOffset + feature] - predicted, raw, sanitized);
                latentError[stateOffset + feature] = error;
                latentEnergy += error * error;
            }
            const float latentScale = terminal ? 1.0f : rsqrt(max(latentEnergy / float(rank), FLT_EPSILON));
            obsScale[gid] *= latentScale;

            const float rewardNow = oak_sanitize(reward[gid], raw, sanitized);
            const float valuePrev = oak_masked_dot(valueWeights, 0u, previousState, stateOffset, activeList, activeCount, raw, sanitized);
            const float valueNow = terminal ? 0.0f : oak_masked_dot(valueWeights, 0u, state, stateOffset, activeList, activeCount, raw, sanitized);
            const float tdRaw = rewardNow - scalars->rewardRate + valueNow - valuePrev;
            td[gid] = oak_sanitize(tdRaw / max(fabs(rewardNow) + fabs(scalars->rewardRate) + fabs(valueNow) + fabs(valuePrev), FLT_EPSILON), raw, sanitized);

            if (!actionIsDiscrete) {
                float gradNorm = 0.0f;
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                    float grad = rewardWeights[rank + actionIndex];
                    for (uint slot = 0; slot < activeCount; ++slot) {
                        const uint feature = activeList[slot];
                        grad += valueWeights[feature] * stateWeights[feature * (1u + actDim) + 1u + actionIndex];
                    }
                    const float sanitizedGrad = oak_sanitize(grad, raw, sanitized);
                    actionGrad[actionOffset + actionIndex] = sanitizedGrad;
                    gradNorm += sanitizedGrad * sanitizedGrad;
                }
                const float gradScale = rsqrt(max(gradNorm / float(actDim), FLT_EPSILON));
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) actionGrad[actionOffset + actionIndex] = oak_sanitize(actionGrad[actionOffset + actionIndex] * gradScale, raw, sanitized);
            } else {
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) actionGrad[actionOffset + actionIndex] = 0.0f;
            }
            for (uint slot = 0; slot < activeCount; ++slot) {
                const uint feature = activeList[slot];
                rolloutState[stateOffset + feature] = oak_standardized_feature(
                    featureMean,
                    featureSecondMoment,
                    state,
                    stateOffset,
                    latentPower,
                    gid,
                    feature,
                    raw,
                    sanitized
                );
            }
        }
        oak_flush(counters, 0u, raw, sanitized);
        return;
    }

    if (config.phase == 2u) {
        if (scalars->stepCount == 0u) {
            oak_flush(counters, 1u, raw, sanitized);
            return;
        }
        const uint encoderSpan = obsDim;
        const uint rewardSpan = actionIsDiscrete ? actDim * (rank + 1u) : rank + actDim;
        const uint stateSpan = rank * (1u + actDim);
        const uint valueSpan = rank;
        const uint policySpan = actDim * rank;
        const uint optionValueSpan = 2u * rank;
        const uint optionPolicySpan = 2u * rank * actDim;
        if (gid < encoderSpan) {
            const uint feature = gid;
            if (activeMask[feature] != 0u) {
                float sum = 0.0f;
                for (uint row = 0; row < batchSize; ++row) {
                    const uint obsOffset = row * obsDim;
                    sum += obsScale[row] * latentError[row * rank + feature] * oak_sanitize(observation[obsOffset + feature] - observationMean[feature], raw, sanitized);
                }
                encoderWeights[gid] = oak_sanitize(encoderWeights[gid] + sum / float(batchSize), raw, sanitized);
            }
        } else if (gid < encoderSpan + rewardSpan) {
            const uint index = gid - encoderSpan;
            float sum = 0.0f;
            if (!actionIsDiscrete && (index >= rank || activeMask[index] != 0u)) {
                for (uint row = 0; row < batchSize; ++row) {
                    const uint stateOffset = row * rank;
                    const uint actionOffset = row * actDim;
                    const uint discreteIndex = actionIsDiscrete ? oak_discrete_index(previousAction, actionOffset, actDim, raw, sanitized) : 0u;
                    const float input = index < rank
                        ? oak_sanitize(previousState[stateOffset + index], raw, sanitized)
                        : (actionIsDiscrete
                            ? float(index - rank == discreteIndex)
                            : oak_sanitize(previousAction[actionOffset + index - rank], raw, sanitized));
                    const float rewardValue = oak_sanitize(reward[row], raw, sanitized);
                    const float rewardError = rewardValue - rewardPrediction[row];
                    sum += rewardError * input / max(fabs(rewardValue) + fabs(rewardPrediction[row]), FLT_EPSILON);
                }
                rewardWeights[index] = oak_sanitize(rewardWeights[index] + sum / float(batchSize), raw, sanitized);
            } else if (actionIsDiscrete) {
                const uint actionIndex = index / (rank + 1u);
                const uint feature = index % (rank + 1u);
                if (feature == rank || activeMask[feature] != 0u) {
                    for (uint row = 0; row < batchSize; ++row) {
                        const uint stateOffset = row * rank;
                        const uint actionOffset = row * actDim;
                        const uint discreteIndex = oak_discrete_index(previousAction, actionOffset, actDim, raw, sanitized);
                        if (discreteIndex != actionIndex) continue;
                        const float input = feature == rank
                            ? 1.0f
                            : oak_sanitize(previousState[stateOffset + feature], raw, sanitized);
                        const float rewardValue = oak_sanitize(reward[row], raw, sanitized);
                        const float rewardError = rewardValue - rewardPrediction[row];
                        sum += rewardError * input / max(fabs(rewardValue) + fabs(rewardPrediction[row]), FLT_EPSILON);
                    }
                    rewardWeights[index] = oak_sanitize(rewardWeights[index] + sum / float(batchSize), raw, sanitized);
                }
            }
        } else if (gid < encoderSpan + rewardSpan + stateSpan) {
            const uint local = gid - encoderSpan - rewardSpan;
            const uint feature = local / (1u + actDim);
            const uint inputIndex = local % (1u + actDim);
            if (activeMask[feature] != 0u) {
                float sum = 0.0f;
                for (uint row = 0; row < batchSize; ++row) {
                    const uint stateOffset = row * rank;
                    const uint actionOffset = row * actDim;
                    const uint discreteIndex = actionIsDiscrete ? oak_discrete_index(previousAction, actionOffset, actDim, raw, sanitized) : 0u;
                    const float input = inputIndex == 0u
                        ? oak_sanitize(previousState[stateOffset + feature], raw, sanitized)
                        : (actionIsDiscrete
                            ? float(inputIndex - 1u == discreteIndex)
                            : oak_sanitize(previousAction[actionOffset + inputIndex - 1u], raw, sanitized));
                    sum += obsScale[row] * latentError[stateOffset + feature] * input;
                }
                stateWeights[local] = oak_sanitize(stateWeights[local] + sum / float(batchSize), raw, sanitized);
            }
        } else if (gid < encoderSpan + rewardSpan + stateSpan + valueSpan) {
            const uint feature = gid - encoderSpan - rewardSpan - stateSpan;
            if (activeMask[feature] != 0u) {
                float sum = 0.0f;
                for (uint row = 0; row < batchSize; ++row) sum += obsScale[row] * td[row] * oak_sanitize(previousState[row * rank + feature], raw, sanitized);
                valueWeights[feature] = oak_sanitize(valueWeights[feature] + sum / float(batchSize), raw, sanitized);
            }
        } else if (gid < encoderSpan + rewardSpan + stateSpan + valueSpan + policySpan) {
            const uint local = gid - encoderSpan - rewardSpan - stateSpan - valueSpan;
            const uint actionIndex = local / rank;
            const uint feature = local % rank;
            if (activeMask[feature] != 0u) {
                float sum = 0.0f;
                for (uint row = 0; row < batchSize; ++row) {
                    const uint actionOffset = row * actDim;
                    const uint discreteIndex = actionIsDiscrete ? oak_discrete_index(previousAction, actionOffset, actDim, raw, sanitized) : 0u;
                    const float policySignal = actionIsDiscrete
                        ? float(actionIndex == discreteIndex)
                        : oak_sanitize(actionGrad[actionOffset + actionIndex], raw, sanitized);
                    sum += obsScale[row] * td[row] * policySignal * oak_sanitize(previousState[row * rank + feature], raw, sanitized);
                }
                policyWeights[local] = oak_sanitize(policyWeights[local] + sum / float(batchSize), raw, sanitized);
            }
        } else if (gid < encoderSpan + rewardSpan + stateSpan + valueSpan + policySpan + optionValueSpan) {
            const uint optionIndex = gid - encoderSpan - rewardSpan - stateSpan - valueSpan - policySpan;
            const uint targetFeature = optionIndex / 2u;
            const float direction = optionIndex % 2u == 0u ? -1.0f : 1.0f;
            if (activeMask[targetFeature] != 0u) {
                float sum = 0.0f;
                const float currentValue = oak_sanitize(optionValues[optionIndex], raw, sanitized);
                for (uint row = 0; row < batchSize; ++row) {
                    const uint stateOffset = row * rank;
                    const float standardized = oak_sanitize(rolloutState[stateOffset + targetFeature], raw, sanitized);
                    const float subtaskReturn = oak_sanitize(
                        oak_sanitize(reward[row], raw, sanitized) - scalars->rewardRate + direction * standardized,
                        raw,
                        sanitized
                    );
                    sum += subtaskReturn - currentValue;
                }
                optionValues[optionIndex] = oak_sanitize(currentValue + sum / float(batchSize), raw, sanitized);
            }
        } else if (gid < encoderSpan + rewardSpan + stateSpan + valueSpan + policySpan + optionValueSpan + optionPolicySpan) {
            const uint local = gid - encoderSpan - rewardSpan - stateSpan - valueSpan - policySpan - optionValueSpan;
            const uint optionIndex = local / actDim;
            const uint actionIndex = local % actDim;
            const uint targetFeature = optionIndex / 2u;
            const float direction = optionIndex % 2u == 0u ? -1.0f : 1.0f;
            if (activeMask[targetFeature] != 0u) {
                float sum = 0.0f;
                for (uint row = 0; row < batchSize; ++row) {
                    const uint stateOffset = row * rank;
                    const uint actionOffset = row * actDim;
                    const float standardized = oak_sanitize(rolloutState[stateOffset + targetFeature], raw, sanitized);
                    const float optionSignal = oak_sanitize(obsScale[row] * (td[row] + direction * standardized), raw, sanitized);
                    const float actionSignal = actionIsDiscrete
                        ? float(actionIndex == oak_discrete_index(previousAction, actionOffset, actDim, raw, sanitized))
                        : oak_sanitize(previousAction[actionOffset + actionIndex], raw, sanitized);
                    sum += optionSignal * actionSignal;
                }
                optionPolicyWeights[local] = oak_sanitize(optionPolicyWeights[local] + sum / float(batchSize), raw, sanitized);
            }
        }
        oak_flush(counters, 1u, raw, sanitized);
        return;
    }

    if (config.phase == 3u) {
        const uint row = gid / actDim;
        const uint actionIndex = gid % actDim;
        if (row < batchSize) {
            const uint stateOffset = row * rank;
            const uint actionOffset = row * actDim;
            action[actionOffset + actionIndex] = oak_masked_dot(policyWeights, actionIndex * rank, state, stateOffset, activeList, activeCount, raw, sanitized);
        }
        oak_flush(counters, 2u, raw, sanitized);
        return;
    }

    if (config.phase == 4u) {
        if (gid < batchSize) {
            const uint stateOffset = gid * rank;
            const uint actionOffset = gid * actDim;
            const uint rolloutOffset = gid * rank;

            if (actionIsDiscrete) {
                const uint reactiveIndex = oak_argmax(action, actionOffset, actDim, raw, sanitized);
                oak_predict_state(stateWeights, state, stateOffset, rolloutState, rolloutOffset, action, actionOffset, reactiveIndex, true, activeList, activeCount, rank, actDim, raw, sanitized);
                oak_normalize_active_state(rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                const float reactiveValue = oak_masked_dot(valueWeights, 0u, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                const float reactiveVariance = oak_residual_trace(featureResidual, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                const float reactiveReward = oak_reward_prediction_discrete(rewardWeights, state, stateOffset, activeList, activeCount, rank, reactiveIndex, raw, sanitized);
                const float reactiveScore = oak_sanitize(reactiveReward - scalars->rewardRate + reactiveValue, raw, sanitized);
                float bestScore = reactiveScore;
                float bestControl = oak_active_delta_energy(state, stateOffset, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                uint bestIndex = reactiveIndex;

                for (uint candidateIndex = 0; candidateIndex < actDim; ++candidateIndex) {
                    oak_predict_state(stateWeights, state, stateOffset, rolloutState, rolloutOffset, action, actionOffset, candidateIndex, true, activeList, activeCount, rank, actDim, raw, sanitized);
                    oak_normalize_active_state(rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                    const float valuePrediction = oak_masked_dot(valueWeights, 0u, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                    const float varianceTrace = oak_residual_trace(featureResidual, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                    const float controlEnergy = oak_active_delta_energy(state, stateOffset, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                    const float rewardPredictionValue = oak_reward_prediction_discrete(rewardWeights, state, stateOffset, activeList, activeCount, rank, candidateIndex, raw, sanitized);
                    const float candidateScore = oak_sanitize(rewardPredictionValue - scalars->rewardRate + valuePrediction, raw, sanitized);
                    if (varianceTrace > reactiveVariance) continue;
                    if (
                        candidateScore > bestScore ||
                        (candidateScore == bestScore && controlEnergy > bestControl)
                    ) {
                        bestIndex = candidateIndex;
                        bestScore = candidateScore;
                        bestControl = controlEnergy;
                    }
                }

                for (uint slot = 0; slot < activeCount; ++slot) {
                    const uint feature = activeList[slot];
                    for (uint signIndex = 0u; signIndex < 2u; ++signIndex) {
                        const uint optionIndex = feature * 2u + signIndex;
                        uint optionAction = 0u;
                        float optionPreference = -FLT_MAX;
                        for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                            const float preference = oak_sanitize(optionPolicyWeights[optionIndex * actDim + actionIndex], raw, sanitized);
                            if (preference > optionPreference) {
                                optionPreference = preference;
                                optionAction = actionIndex;
                            }
                        }
                        oak_predict_state(stateWeights, state, stateOffset, rolloutState, rolloutOffset, action, actionOffset, optionAction, true, activeList, activeCount, rank, actDim, raw, sanitized);
                        oak_normalize_active_state(rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                        const float varianceTrace = oak_residual_trace(featureResidual, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                        if (varianceTrace > reactiveVariance) continue;
                        const float controlEnergy = oak_active_delta_energy(state, stateOffset, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                        const float rewardPredictionValue = oak_reward_prediction_discrete(rewardWeights, state, stateOffset, activeList, activeCount, rank, optionAction, raw, sanitized);
                        const float optionScore = oak_sanitize(
                            rewardPredictionValue - scalars->rewardRate + oak_sanitize(optionValues[optionIndex], raw, sanitized),
                            raw,
                            sanitized
                        );
                        if (optionScore > bestScore || (optionScore == bestScore && controlEnergy > bestControl)) {
                            bestIndex = optionAction;
                            bestScore = optionScore;
                            bestControl = controlEnergy;
                        }
                    }
                }

                float deltaSum = 0.0f;
                float planningSum = 0.0f;
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                    const float reactive = actionIndex == reactiveIndex ? 1.0f : 0.0f;
                    const float finalAction = actionIndex == bestIndex ? 1.0f : 0.0f;
                    const float previous = done[gid] != 0u ? 0.0f : previousAction[actionOffset + actionIndex];
                    const float planDelta = finalAction - reactive;
                    const float actionDelta = finalAction - previous;
                    planningSum += planDelta * planDelta;
                    deltaSum += actionDelta * actionDelta;
                    action[actionOffset + actionIndex] = finalAction;
                    previousAction[actionOffset + actionIndex] = finalAction;
                }
                td[gid] = sqrt(deltaSum / float(actDim));
                obsScale[gid] = sqrt(planningSum / float(actDim));
            } else {
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                    action[actionOffset + actionIndex] = clamp(
                        oak_sanitize(action[actionOffset + actionIndex], raw, sanitized),
                        actionBounds[actionIndex],
                        actionBounds[actDim + actionIndex]
                    );
                }
                oak_predict_state(stateWeights, state, stateOffset, rolloutState, rolloutOffset, action, actionOffset, 0u, false, activeList, activeCount, rank, actDim, raw, sanitized);
                oak_normalize_active_state(rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                const float baselineValue = oak_masked_dot(valueWeights, 0u, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                const float baselineVariance = oak_residual_trace(featureResidual, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                const float baselineReward = oak_reward_prediction_continuous(rewardWeights, state, stateOffset, action, actionOffset, activeList, activeCount, rank, actDim, raw, sanitized);
                const float baselineScore = oak_sanitize(baselineReward - scalars->rewardRate + baselineValue, raw, sanitized);
                float bestImprovement = 0.0f;
                bool improved = false;
                for (uint slot = 0; slot < activeCount; ++slot) {
                    const uint feature = activeList[slot];
                    for (uint signIndex = 0u; signIndex < 2u; ++signIndex) {
                        const uint optionIndex = feature * 2u + signIndex;
                        for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                            actionGrad[actionOffset + actionIndex] = clamp(
                                action[actionOffset + actionIndex] + oak_sanitize(optionPolicyWeights[optionIndex * actDim + actionIndex], raw, sanitized),
                                actionBounds[actionIndex],
                                actionBounds[actDim + actionIndex]
                            );
                        }
                        oak_predict_state(stateWeights, state, stateOffset, rolloutState, rolloutOffset, actionGrad, actionOffset, 0u, false, activeList, activeCount, rank, actDim, raw, sanitized);
                        oak_normalize_active_state(rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                        const float varianceTrace = oak_residual_trace(featureResidual, rolloutState, rolloutOffset, activeList, activeCount, raw, sanitized);
                        const float rewardPredictionValue = oak_reward_prediction_continuous(rewardWeights, state, stateOffset, actionGrad, actionOffset, activeList, activeCount, rank, actDim, raw, sanitized);
                        const float improvement = oak_sanitize(
                            rewardPredictionValue - scalars->rewardRate + oak_sanitize(optionValues[optionIndex], raw, sanitized) - baselineScore,
                            raw,
                            sanitized
                        );
                        if (varianceTrace <= baselineVariance && improvement > bestImprovement) {
                            bestImprovement = improvement;
                            improved = true;
                            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) action[actionOffset + actionIndex] = actionGrad[actionOffset + actionIndex];
                        }
                    }
                }

                float deltaSum = 0.0f;
                float planningSum = 0.0f;
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                    const float reactive = action[actionOffset + actionIndex];
                    const float finalAction = improved ? action[actionOffset + actionIndex] : reactive;
                    const float previous = done[gid] != 0u ? 0.0f : previousAction[actionOffset + actionIndex];
                    const float planDelta = finalAction - reactive;
                    const float actionDelta = finalAction - previous;
                    planningSum += planDelta * planDelta;
                    deltaSum += actionDelta * actionDelta;
                    action[actionOffset + actionIndex] = finalAction;
                    previousAction[actionOffset + actionIndex] = finalAction;
                }
                td[gid] = sqrt(deltaSum / float(actDim));
                obsScale[gid] = sqrt(planningSum / float(actDim));
            }

            for (uint feature = 0; feature < rank; ++feature) previousState[stateOffset + feature] = state[stateOffset + feature];
        }
        oak_flush(counters, 2u, raw, sanitized);
        return;
    }

    if (config.phase == 5u) {
        const uint oldCount = scalars->rewardCount;
        const uint newCount = oldCount + batchSize;
        const float countScale = float(batchSize) / float(max(newCount, 1u));
        if (gid < obsDim) {
            const uint obsIndex = gid;
            float batchMean = 0.0f;
            for (uint row = 0; row < batchSize; ++row) batchMean += oak_sanitize(observation[row * obsDim + obsIndex], raw, sanitized);
            batchMean /= float(batchSize);
            observationMean[obsIndex] += (batchMean - observationMean[obsIndex]) * countScale;
        }
        if (gid < rank) {
            const uint feature = gid;
            float mean = 0.0f;
            float meanSq = 0.0f;
            float residual = 0.0f;
            for (uint row = 0; row < batchSize; ++row) {
                const float value = oak_sanitize(state[row * rank + feature] * sqrt(max(latentPower[row], FLT_EPSILON)), raw, sanitized);
                mean += value;
                meanSq += value * value;
                const float error = oak_sanitize(latentError[row * rank + feature], raw, sanitized);
                residual += error * error;
            }
            mean /= float(batchSize);
            meanSq /= float(batchSize);
            residual /= float(batchSize);
            featureMean[feature] += (mean - featureMean[feature]) * countScale;
            featureSecondMoment[feature] += (meanSq - featureSecondMoment[feature]) * countScale;
            featureResidual[feature] += ((activeMask[feature] != 0u ? residual : 0.0f) - featureResidual[feature]) * countScale;
        }
        oak_flush(counters, 5u, raw, sanitized);
        return;
    }

    if (config.phase == 6u && gid == 0u) {
        float rewardSum = 0.0f;
        float varianceTrace = 0.0f;
        float traceSquared = 0.0f;
        float bestVariance = -FLT_MAX;
        uint bestFeature = 0u;
        const uint oldCount = scalars->rewardCount;
        const uint newCount = oldCount + batchSize;
        const float countScale = float(batchSize) / float(max(newCount, 1u));
        for (uint row = 0; row < batchSize; ++row) rewardSum += oak_sanitize(reward[row], raw, sanitized);
        for (uint feature = 0; feature < rank; ++feature) {
            const float variance = max(featureSecondMoment[feature] - featureMean[feature] * featureMean[feature], 0.0f);
            varianceTrace += variance;
            traceSquared += variance * variance;
            activeMask[feature] = 0u;
            if (variance > bestVariance) {
                bestVariance = variance;
                bestFeature = feature;
            }
        }
        uint activeCount = uint(ceil((varianceTrace * varianceTrace) / max(traceSquared, FLT_EPSILON)));
        activeCount = min(max(activeCount, 1u), rank);
        for (uint slot = 0u; slot < activeCount; ++slot) {
            float currentBest = -FLT_MAX;
            uint currentIndex = bestFeature;
            for (uint feature = 0; feature < rank; ++feature) {
                if (activeMask[feature] != 0u) continue;
                const float variance = max(featureSecondMoment[feature] - featureMean[feature] * featureMean[feature], 0.0f);
                if (variance > currentBest) {
                    currentBest = variance;
                    currentIndex = feature;
                }
            }
            activeMask[currentIndex] = 1u;
            activeList[slot] = currentIndex;
        }
        for (uint slot = activeCount; slot < rank; ++slot) activeList[slot] = 0u;
        const float batchRewardMean = rewardSum / float(batchSize);
        scalars->rewardCount = newCount;
        scalars->activeCount = activeCount;
        scalars->rewardRate += (batchRewardMean - scalars->rewardRate) * countScale;
        scalars->stepCount += 1u;
        scalars->rewardRate = oak_sanitize(scalars->rewardRate, raw, sanitized);
        statsOut[0].latentVarianceTrace = varianceTrace;
        statsOut[0].activeRank = float(activeCount);
        oak_flush(counters, 6u, raw, sanitized);
        return;
    }

    if (config.phase == 7u) {
        if (gid < batchSize) {
            const uint stateOffset = gid * rank;
            const uint actionOffset = gid * actDim;
            float latentErrorMean = 0.0f;
            float maxLatentPredictionAbs = 0.0f;
            float maxPolicyOutputAbs = 0.0f;
            for (uint slot = 0; slot < activeCount; ++slot) {
                const uint feature = activeList[slot];
                const float error = oak_sanitize(latentError[stateOffset + feature], raw, sanitized);
                latentErrorMean += error * error;
                maxLatentPredictionAbs = max(
                    maxLatentPredictionAbs,
                    fabs(oak_sanitize(state[stateOffset + feature] - error, raw, sanitized))
                );
            }
            latentPower[gid] = oak_sanitize(activeCount > 0u ? latentErrorMean / float(activeCount) : 0.0f, raw, sanitized);
            rolloutState[stateOffset] = maxLatentPredictionAbs;
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                maxPolicyOutputAbs = max(
                    maxPolicyOutputAbs,
                    fabs(oak_masked_dot(policyWeights, actionIndex * rank, state, stateOffset, activeList, activeCount, raw, sanitized))
                );
            }
            actionGrad[actionOffset] = maxPolicyOutputAbs;
        }
        oak_flush(counters, 7u, raw, sanitized);
        return;
    }

    if (config.phase == 8u && gid == 0u) {
        float rewardSum = 0.0f;
        float rewardErrorSum = 0.0f;
        float latentErrorSum = 0.0f;
        float actionDeltaSum = 0.0f;
        float planningDeltaSum = 0.0f;
        float maxLatentPredictionAbs = 0.0f;
        float maxRewardPredictionAbs = 0.0f;
        float maxPolicyOutputAbs = 0.0f;
        const float varianceTrace = statsOut[0].latentVarianceTrace;
        const uint activeCount = uint(statsOut[0].activeRank);
        for (uint row = 0; row < batchSize; ++row) {
            const uint stateOffset = row * rank;
            const uint actionOffset = row * actDim;
            const float rewardValue = oak_sanitize(reward[row], raw, sanitized);
            const float rewardEstimate = oak_sanitize(rewardPrediction[row], raw, sanitized);
            const float rewardError = rewardValue - rewardEstimate;
            rewardSum += rewardValue;
            rewardErrorSum += rewardError * rewardError;
            latentErrorSum += oak_sanitize(latentPower[row], raw, sanitized);
            actionDeltaSum += oak_sanitize(td[row], raw, sanitized);
            planningDeltaSum += oak_sanitize(obsScale[row], raw, sanitized);
            maxLatentPredictionAbs = max(maxLatentPredictionAbs, fabs(oak_sanitize(rolloutState[stateOffset], raw, sanitized)));
            maxRewardPredictionAbs = max(maxRewardPredictionAbs, fabs(rewardEstimate));
            maxPolicyOutputAbs = max(maxPolicyOutputAbs, fabs(oak_sanitize(actionGrad[actionOffset], raw, sanitized)));
        }

        statsOut[0] = {
            scalars->rewardRate,
            latentErrorSum / float(batchSize),
            rewardErrorSum / float(batchSize),
            varianceTrace,
            float(activeCount),
            actionDeltaSum / float(batchSize),
            planningDeltaSum / float(batchSize),
            float(atomic_load_explicit(&counters[0], memory_order_relaxed)),
            float(atomic_load_explicit(&counters[1], memory_order_relaxed)),
            float(atomic_load_explicit(&counters[2], memory_order_relaxed)),
            float(atomic_load_explicit(&counters[3], memory_order_relaxed)),
            float(atomic_load_explicit(&counters[4], memory_order_relaxed)),
            maxLatentPredictionAbs,
            maxRewardPredictionAbs,
            maxPolicyOutputAbs
        };
        oak_flush(counters, 8u, raw, sanitized);
    }
}
