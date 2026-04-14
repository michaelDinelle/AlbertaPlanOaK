#include <metal_stdlib>
using namespace metal;

struct KernelConfig {
    uint obsDim;
    uint rank;
    uint actDim;
    uint batchSize;
    float offeredNs;
};

struct KernelScalars {
    uint activeRank;
    uint rewardCount;
    uint reserved0;
    uint reserved1;
    float rewardRate;
};

struct KernelStats {
    float residualMean;
    float totalMean;
    float rewardRate;
    float tdAbsMean;
    float actionDeltaL2;
    float gvfErrorMean;
    float gvfPredictionAbsMean;
    float modelErrorMean;
    float rewardPredictionAbsMean;
    float statePredictionAbsMean;
    float optionSelectionFraction;
    float optionTrainingSignalMean;
    float optionScoreMean;
    float planningCandidateCount;
    float planningScoreMean;
    float planningScoreMax;
    float planningActionChangeL2;
    float rawNonFiniteCount;
    float sanitizedNonFiniteCount;
    float fatalNonFiniteCount;
};

inline float oak_sanitize(float value, thread float& rawCount, thread float& sanitizedCount) {
    if (isfinite(value)) {
        return value;
    }
    rawCount += 1.0f;
    sanitizedCount += 1.0f;
    return 0.0f;
}

inline float oak_safe_tanh(float value, thread float& rawCount, thread float& sanitizedCount) {
    return oak_sanitize(tanh(value), rawCount, sanitizedCount);
}

inline float oak_inverse_rms(
    device const float *values,
    uint offset,
    uint count,
    thread float& rawCount,
    thread float& sanitizedCount
) {
    float sumSquares = 0.0f;
    for (uint index = 0; index < count; ++index) {
        const float value = oak_sanitize(values[offset + index], rawCount, sanitizedCount);
        sumSquares += value * value;
    }
    const float meanSquare = sumSquares / float(max(count, 1u));
    return rsqrt(max(meanSquare, FLT_EPSILON));
}

kernel void oak_step(
    device const float *observation [[buffer(0)]],
    device float *encoderWeights [[buffer(1)]],
    device float *policyWeights [[buffer(2)]],
    device float *valueWeights [[buffer(3)]],
    device float *gvfWeights [[buffer(4)]],
    device float *rewardModelWeights [[buffer(5)]],
    device float *stateModelWeights [[buffer(6)]],
    device float *optionPolicyWeights [[buffer(7)]],
    device float *optionValueWeights [[buffer(8)]],
    device float *previousState [[buffer(9)]],
    device float *previousAction [[buffer(10)]],
    device const float *reward [[buffer(11)]],
    device const float *done [[buffer(12)]],
    device float *state [[buffer(13)]],
    device float *action [[buffer(14)]],
    device KernelScalars *scalars [[buffer(15)]],
    device KernelStats *statsOut [[buffer(16)]],
    constant KernelConfig &config [[buffer(17)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) {
        return;
    }

    const uint obsDim = config.obsDim;
    const uint rank = max(config.rank, 1u);
    const uint actDim = max(config.actDim, 1u);
    const uint batchSize = max(config.batchSize, 1u);
    uint activeRank = min(max(scalars->activeRank, 1u), rank);

    float rawNonFiniteCount = 0.0f;
    float sanitizedNonFiniteCount = 0.0f;
    float fatalNonFiniteCount = 0.0f;

    float residualSum = 0.0f;
    float totalEnergySum = 0.0f;

    for (uint row = 0; row < batchSize; ++row) {
        const uint obsOffset = row * obsDim;
        const uint stateOffset = row * rank;

        for (uint feature = 0; feature < rank; ++feature) {
            float projection = 0.0f;
            const uint weightOffset = feature * obsDim;
            for (uint index = 0; index < obsDim; ++index) {
                const float observed = oak_sanitize(observation[obsOffset + index], rawNonFiniteCount, sanitizedNonFiniteCount);
                projection += encoderWeights[weightOffset + index] * observed;
            }
            state[stateOffset + feature] = oak_sanitize(projection, rawNonFiniteCount, sanitizedNonFiniteCount);
        }

        const float stateScale = oak_inverse_rms(state, stateOffset, rank, rawNonFiniteCount, sanitizedNonFiniteCount);
        for (uint feature = 0; feature < rank; ++feature) {
            state[stateOffset + feature] = oak_sanitize(state[stateOffset + feature] * stateScale, rawNonFiniteCount, sanitizedNonFiniteCount);
        }

        float residual = 0.0f;
        float total = 0.0f;
        for (uint index = 0; index < obsDim; ++index) {
            const float observed = oak_sanitize(observation[obsOffset + index], rawNonFiniteCount, sanitizedNonFiniteCount);
            float rebuilt = 0.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                rebuilt += state[stateOffset + feature] * encoderWeights[feature * obsDim + index];
            }
            const float delta = observed - oak_sanitize(rebuilt, rawNonFiniteCount, sanitizedNonFiniteCount);
            residual += delta * delta;
            total += observed * observed;
        }
        residualSum += residual;
        totalEnergySum += total;
    }

    const float residualMean = residualSum / float(batchSize);
    const float totalMean = totalEnergySum / float(batchSize);
    const float encodeAlpha = 1.0f / float(max(obsDim, 1u));
    for (uint row = 0; row < batchSize; ++row) {
        const uint obsOffset = row * obsDim;
        const uint stateOffset = row * rank;
        for (uint index = 0; index < obsDim; ++index) {
            const float observed = oak_sanitize(observation[obsOffset + index], rawNonFiniteCount, sanitizedNonFiniteCount);
            float rebuilt = 0.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                rebuilt += state[stateOffset + feature] * encoderWeights[feature * obsDim + index];
            }
            const float delta = observed - oak_sanitize(rebuilt, rawNonFiniteCount, sanitizedNonFiniteCount);
            for (uint feature = 0; feature < activeRank; ++feature) {
                const uint weightIndex = feature * obsDim + index;
                const float updated = encoderWeights[weightIndex] + encodeAlpha * delta * state[stateOffset + feature];
                encoderWeights[weightIndex] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
            }
        }
    }

    const float unexplainedFraction = totalMean > FLT_EPSILON ? clamp(residualMean / totalMean, 0.0f, 1.0f) : 0.0f;
    activeRank = min(max(uint(ceil(float(rank) * sqrt(unexplainedFraction))), 1u), rank);
    scalars->activeRank = activeRank;

    float batchReward = 0.0f;
    for (uint row = 0; row < batchSize; ++row) {
        batchReward += oak_sanitize(reward[row], rawNonFiniteCount, sanitizedNonFiniteCount);
    }
    batchReward /= float(batchSize);

    const bool hasPreviousTransition = scalars->rewardCount > 0u;
    scalars->rewardCount += batchSize;
    const float rewardCount = float(max(scalars->rewardCount, 1u));
    const float previousRewardRate = scalars->rewardRate;
    scalars->rewardRate += (batchReward - scalars->rewardRate) / rewardCount;
    const float rewardRate = oak_sanitize(scalars->rewardRate, rawNonFiniteCount, sanitizedNonFiniteCount);

    float tdAbsSum = 0.0f;
    float gvfErrorSum = 0.0f;
    float gvfPredictionAbsSum = 0.0f;
    float gvfSampleCount = 0.0f;
    float modelErrorSum = 0.0f;
    float rewardPredictionAbsSum = 0.0f;
    float statePredictionAbsSum = 0.0f;
    float modelSampleCount = 0.0f;
    const float optionAlpha = 1.0f / float(max(activeRank, 1u));

    if (hasPreviousTransition) {
        const float featureAlpha = 1.0f / float(max(activeRank, 1u));
        const float transitionAlpha = 1.0f / float(max(activeRank + actDim, 1u));

        for (uint row = 0; row < batchSize; ++row) {
            const uint stateOffset = row * rank;
            const uint actionOffset = row * actDim;
            const bool terminated = oak_sanitize(done[row], rawNonFiniteCount, sanitizedNonFiniteCount) > 0.5f;
            const float rowReward = oak_sanitize(reward[row], rawNonFiniteCount, sanitizedNonFiniteCount);
            const float previousScale = oak_inverse_rms(previousState, stateOffset, activeRank, rawNonFiniteCount, sanitizedNonFiniteCount);
            const float currentScale = oak_inverse_rms(state, stateOffset, activeRank, rawNonFiniteCount, sanitizedNonFiniteCount);

            float valuePrev = 0.0f;
            float valueNow = 0.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                valuePrev += valueWeights[feature] * previousState[stateOffset + feature] * previousScale;
                if (!terminated) {
                    valueNow += valueWeights[feature] * state[stateOffset + feature] * currentScale;
                }
            }
            const float td = oak_sanitize(rowReward - previousRewardRate + valueNow - valuePrev, rawNonFiniteCount, sanitizedNonFiniteCount);
            const float tdUpdate = oak_safe_tanh(td, rawNonFiniteCount, sanitizedNonFiniteCount);
            tdAbsSum += fabs(td);

            for (uint feature = 0; feature < activeRank; ++feature) {
                const float updatedValue = valueWeights[feature] + featureAlpha * tdUpdate * previousState[stateOffset + feature] * previousScale;
                valueWeights[feature] = oak_sanitize(updatedValue, rawNonFiniteCount, sanitizedNonFiniteCount);
            }
            float actionGradSquareSum = 0.0f;
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                float actionGrad = rewardModelWeights[rank + actionIndex];
                for (uint predictedFeature = 0; predictedFeature < activeRank; ++predictedFeature) {
                    const uint rowOffset = predictedFeature * (rank + actDim);
                    actionGrad += valueWeights[predictedFeature] * stateModelWeights[rowOffset + rank + actionIndex];
                }
                actionGrad = oak_sanitize(actionGrad, rawNonFiniteCount, sanitizedNonFiniteCount);
                actionGradSquareSum += actionGrad * actionGrad;
            }
            const float actionGradScale = rsqrt(max(actionGradSquareSum / float(max(actDim, 1u)), FLT_EPSILON));
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                const uint rowOffset = actionIndex * rank;
                float actionGrad = rewardModelWeights[rank + actionIndex];
                for (uint predictedFeature = 0; predictedFeature < activeRank; ++predictedFeature) {
                    const uint modelOffset = predictedFeature * (rank + actDim);
                    actionGrad += valueWeights[predictedFeature] * stateModelWeights[modelOffset + rank + actionIndex];
                }
                actionGrad = oak_sanitize(actionGrad * actionGradScale, rawNonFiniteCount, sanitizedNonFiniteCount);
                const float behaviorAction = oak_sanitize(previousAction[actionOffset + actionIndex], rawNonFiniteCount, sanitizedNonFiniteCount);
                const float target = oak_safe_tanh(behaviorAction + actionGrad, rawNonFiniteCount, sanitizedNonFiniteCount);
                float policyLogit = 0.0f;
                for (uint feature = 0; feature < activeRank; ++feature) {
                    policyLogit += policyWeights[rowOffset + feature] * previousState[stateOffset + feature] * previousScale;
                }
                const float policyAction = oak_safe_tanh(policyLogit, rawNonFiniteCount, sanitizedNonFiniteCount);
                const float policyError = oak_sanitize(target - policyAction, rawNonFiniteCount, sanitizedNonFiniteCount);
                for (uint feature = 0; feature < activeRank; ++feature) {
                    const uint index = rowOffset + feature;
                    const float updatedPolicy = policyWeights[index] + featureAlpha * policyError * previousState[stateOffset + feature] * previousScale;
                    policyWeights[index] = oak_sanitize(updatedPolicy, rawNonFiniteCount, sanitizedNonFiniteCount);
                }
            }

            for (uint question = 0; question < activeRank + 1u; ++question) {
                const uint weightOffset = question * rank;
                float prediction = 0.0f;
                for (uint feature = 0; feature < activeRank; ++feature) {
                    prediction += gvfWeights[weightOffset + feature] * previousState[stateOffset + feature] * previousScale;
                }
                prediction = oak_sanitize(prediction, rawNonFiniteCount, sanitizedNonFiniteCount);
                const float target = question == 0u ? rowReward : (terminated ? 0.0f : state[stateOffset + question - 1u] * currentScale);
                const float error = oak_sanitize(target - prediction, rawNonFiniteCount, sanitizedNonFiniteCount);
                const float errorUpdate = oak_safe_tanh(error, rawNonFiniteCount, sanitizedNonFiniteCount);
                gvfErrorSum += fabs(error);
                gvfPredictionAbsSum += fabs(prediction);
                gvfSampleCount += 1.0f;
                for (uint feature = 0; feature < activeRank; ++feature) {
                    const uint index = weightOffset + feature;
                    const float updated = gvfWeights[index] + featureAlpha * errorUpdate * previousState[stateOffset + feature] * previousScale;
                    gvfWeights[index] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
                }
            }

            float rewardPrediction = 0.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                rewardPrediction += rewardModelWeights[feature] * previousState[stateOffset + feature] * previousScale;
            }
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                rewardPrediction += rewardModelWeights[rank + actionIndex] * previousAction[actionOffset + actionIndex];
            }
            rewardPrediction = oak_sanitize(rewardPrediction, rawNonFiniteCount, sanitizedNonFiniteCount);
            const float rewardError = oak_sanitize(rowReward - rewardPrediction, rawNonFiniteCount, sanitizedNonFiniteCount);
            const float rewardUpdate = oak_safe_tanh(rewardError, rawNonFiniteCount, sanitizedNonFiniteCount);
            rewardPredictionAbsSum += fabs(rewardPrediction);
            modelErrorSum += fabs(rewardError);
            modelSampleCount += 1.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                const float updated = rewardModelWeights[feature] + transitionAlpha * rewardUpdate * previousState[stateOffset + feature] * previousScale;
                rewardModelWeights[feature] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
            }
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                const uint index = rank + actionIndex;
                const float updated = rewardModelWeights[index] + transitionAlpha * rewardUpdate * previousAction[actionOffset + actionIndex];
                rewardModelWeights[index] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
            }

            if (!terminated) {
                for (uint stateIndex = 0; stateIndex < activeRank; ++stateIndex) {
                    const uint rowOffset = stateIndex * (rank + actDim);
                    float prediction = 0.0f;
                    for (uint feature = 0; feature < activeRank; ++feature) {
                        prediction += stateModelWeights[rowOffset + feature] * previousState[stateOffset + feature] * previousScale;
                    }
                    for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                        prediction += stateModelWeights[rowOffset + rank + actionIndex] * previousAction[actionOffset + actionIndex];
                    }
                    prediction = oak_sanitize(prediction, rawNonFiniteCount, sanitizedNonFiniteCount);
                    const float target = state[stateOffset + stateIndex] * currentScale;
                    const float error = oak_sanitize(target - prediction, rawNonFiniteCount, sanitizedNonFiniteCount);
                    const float errorUpdate = oak_safe_tanh(error, rawNonFiniteCount, sanitizedNonFiniteCount);
                    statePredictionAbsSum += fabs(prediction);
                    modelErrorSum += fabs(error);
                    modelSampleCount += 1.0f;
                    for (uint feature = 0; feature < activeRank; ++feature) {
                        const uint index = rowOffset + feature;
                        const float updated = stateModelWeights[index] + transitionAlpha * errorUpdate * previousState[stateOffset + feature] * previousScale;
                        stateModelWeights[index] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
                    }
                    for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                        const uint index = rowOffset + rank + actionIndex;
                        const float updated = stateModelWeights[index] + transitionAlpha * errorUpdate * previousAction[actionOffset + actionIndex];
                        stateModelWeights[index] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
                    }
                }
            }

        }
    }

    float actionDeltaSquares = 0.0f;
    float optionSelectionCount = 0.0f;
    float optionTrainingSignalSum = 0.0f;
    float optionScoreSum = 0.0f;
    float planningCandidateSum = 0.0f;
    float planningScoreSum = 0.0f;
    float planningScoreMax = -FLT_MAX;
    float planningActionChangeSquares = 0.0f;
    const uint planningWorkScale = max(obsDim + actDim + activeRank, 1u);
    const uint candidateLimit = config.offeredNs > 0.0f
        ? min(activeRank, max(uint(config.offeredNs / float(planningWorkScale)), 1u))
        : 0u;

    for (uint row = 0; row < batchSize; ++row) {
        const uint stateOffset = row * rank;
        const uint actionOffset = row * actDim;
        const float currentScale = oak_inverse_rms(state, stateOffset, activeRank, rawNonFiniteCount, sanitizedNonFiniteCount);

        for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
            float logit = 0.0f;
            const uint rowOffset = actionIndex * rank;
            for (uint feature = 0; feature < activeRank; ++feature) {
                logit += policyWeights[rowOffset + feature] * state[stateOffset + feature] * currentScale;
            }
            action[actionOffset + actionIndex] = oak_safe_tanh(logit, rawNonFiniteCount, sanitizedNonFiniteCount);
        }

        float baselineRewardPrediction = 0.0f;
        for (uint feature = 0; feature < activeRank; ++feature) {
            baselineRewardPrediction += rewardModelWeights[feature] * state[stateOffset + feature] * currentScale;
        }
        for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
            baselineRewardPrediction += rewardModelWeights[rank + actionIndex] * action[actionOffset + actionIndex];
        }
        float baselineValue = 0.0f;
        for (uint predictedFeature = 0; predictedFeature < activeRank; ++predictedFeature) {
            const uint rowOffset = predictedFeature * (rank + actDim);
            float predictedState = 0.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                predictedState += stateModelWeights[rowOffset + feature] * state[stateOffset + feature] * currentScale;
            }
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                predictedState += stateModelWeights[rowOffset + rank + actionIndex] * action[actionOffset + actionIndex];
            }
            baselineValue += oak_sanitize(predictedState, rawNonFiniteCount, sanitizedNonFiniteCount) * valueWeights[predictedFeature];
        }
        const float baselineScore = oak_safe_tanh(baselineRewardPrediction, rawNonFiniteCount, sanitizedNonFiniteCount)
            + oak_safe_tanh(baselineValue, rawNonFiniteCount, sanitizedNonFiniteCount);

        float candidateScoreTotal = 0.0f;
        float bestImprovement = -FLT_MAX;
        int bestOption = -1;

        for (uint option = 0; option < candidateLimit; ++option) {
            float rewardPrediction = 0.0f;
            for (uint feature = 0; feature < activeRank; ++feature) {
                rewardPrediction += rewardModelWeights[feature] * state[stateOffset + feature] * currentScale;
            }
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                const float candidateAction = oak_safe_tanh(
                    action[actionOffset + actionIndex] + optionPolicyWeights[option * actDim + actionIndex] * state[stateOffset + option] * currentScale,
                    rawNonFiniteCount,
                    sanitizedNonFiniteCount
                );
                rewardPrediction += rewardModelWeights[rank + actionIndex] * candidateAction;
            }
            float predictedValue = 0.0f;
            float predictedOptionState = 0.0f;
            for (uint predictedFeature = 0; predictedFeature < activeRank; ++predictedFeature) {
                const uint rowOffset = predictedFeature * (rank + actDim);
                float predictedState = 0.0f;
                for (uint feature = 0; feature < activeRank; ++feature) {
                    predictedState += stateModelWeights[rowOffset + feature] * state[stateOffset + feature] * currentScale;
                }
                for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                    const float candidateAction = oak_safe_tanh(
                        action[actionOffset + actionIndex] + optionPolicyWeights[option * actDim + actionIndex] * state[stateOffset + option] * currentScale,
                        rawNonFiniteCount,
                        sanitizedNonFiniteCount
                    );
                    predictedState += stateModelWeights[rowOffset + rank + actionIndex] * candidateAction;
                }
                predictedState = oak_sanitize(predictedState, rawNonFiniteCount, sanitizedNonFiniteCount);
                if (predictedFeature == option) {
                    predictedOptionState = predictedState;
                }
                predictedValue += predictedState * valueWeights[predictedFeature];
            }
            float optionValue = 0.0f;
            const uint optionValueOffset = option * rank;
            for (uint feature = 0; feature < activeRank; ++feature) {
                optionValue += optionValueWeights[optionValueOffset + feature] * state[stateOffset + feature] * currentScale;
            }
            optionValue = oak_safe_tanh(optionValue, rawNonFiniteCount, sanitizedNonFiniteCount);
            const float optionProgress = oak_safe_tanh(predictedOptionState - state[stateOffset + option] * currentScale, rawNonFiniteCount, sanitizedNonFiniteCount);
            const float candidateScore = oak_sanitize(
                oak_safe_tanh(rewardPrediction, rawNonFiniteCount, sanitizedNonFiniteCount)
                    + oak_safe_tanh(predictedValue, rawNonFiniteCount, sanitizedNonFiniteCount)
                    + optionValue
                    + optionProgress,
                rawNonFiniteCount,
                sanitizedNonFiniteCount
            );
            const float candidateImprovement = oak_sanitize(candidateScore - baselineScore, rawNonFiniteCount, sanitizedNonFiniteCount);
            candidateScoreTotal += candidateImprovement;
            if (bestOption < 0 || candidateImprovement > bestImprovement) {
                bestImprovement = candidateImprovement;
                bestOption = int(option);
            }
            planningScoreMax = max(planningScoreMax, candidateImprovement);
        }

        planningCandidateSum += float(candidateLimit);
        planningScoreSum += candidateScoreTotal;
        optionScoreSum += bestOption >= 0 ? bestImprovement : 0.0f;

        if (bestOption >= 0 && bestImprovement > 0.0f) {
            optionSelectionCount += 1.0f;
        }
        const float optionSignal = bestOption >= 0
            ? oak_safe_tanh(bestImprovement, rawNonFiniteCount, sanitizedNonFiniteCount)
            : 0.0f;
        optionTrainingSignalSum += fabs(optionSignal);

        float rowPlanningChangeSquares = 0.0f;
        for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
            const float reactiveAction = action[actionOffset + actionIndex];
            float finalAction = reactiveAction;
            if (bestOption >= 0 && bestImprovement > 0.0f) {
                finalAction = oak_safe_tanh(
                    reactiveAction + optionSignal * optionPolicyWeights[uint(bestOption) * actDim + actionIndex] * state[stateOffset + uint(bestOption)] * currentScale,
                    rawNonFiniteCount,
                    sanitizedNonFiniteCount
                );
            }
            const float deltaReactive = finalAction - reactiveAction;
            const float previous = oak_sanitize(done[row], rawNonFiniteCount, sanitizedNonFiniteCount) > 0.5f
                ? 0.0f
                : oak_sanitize(previousAction[actionOffset + actionIndex], rawNonFiniteCount, sanitizedNonFiniteCount);
            const float deltaPrevious = finalAction - previous;
            rowPlanningChangeSquares += deltaReactive * deltaReactive;
            actionDeltaSquares += deltaPrevious * deltaPrevious;
            action[actionOffset + actionIndex] = finalAction;
        }
        planningActionChangeSquares += rowPlanningChangeSquares;

        if (hasPreviousTransition && bestOption >= 0) {
            const uint option = uint(bestOption);
            const float scale = state[stateOffset + option] * currentScale;
            for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
                const uint index = option * actDim + actionIndex;
                const float updated = optionPolicyWeights[index] + optionAlpha * optionSignal * scale * action[actionOffset + actionIndex];
                optionPolicyWeights[index] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
            }
            const uint valueOffset = option * rank;
            for (uint feature = 0; feature < activeRank; ++feature) {
                const uint index = valueOffset + feature;
                const float updated = optionValueWeights[index] + optionAlpha * optionSignal * scale * state[stateOffset + feature] * currentScale;
                optionValueWeights[index] = oak_sanitize(updated, rawNonFiniteCount, sanitizedNonFiniteCount);
            }
        }
    }

    for (uint row = 0; row < batchSize; ++row) {
        const uint stateOffset = row * rank;
        const uint actionOffset = row * actDim;
        for (uint feature = 0; feature < rank; ++feature) {
            previousState[stateOffset + feature] = state[stateOffset + feature];
        }
        for (uint actionIndex = 0; actionIndex < actDim; ++actionIndex) {
            previousAction[actionOffset + actionIndex] = action[actionOffset + actionIndex];
        }
    }

    KernelStats stats;
    stats.residualMean = residualMean;
    stats.totalMean = totalMean;
    stats.rewardRate = rewardRate;
    stats.tdAbsMean = batchSize > 0 ? tdAbsSum / float(batchSize) : 0.0f;
    stats.actionDeltaL2 = sqrt(actionDeltaSquares / float(max(batchSize * actDim, 1u)));
    stats.gvfErrorMean = gvfSampleCount > 0.0f ? gvfErrorSum / gvfSampleCount : 0.0f;
    stats.gvfPredictionAbsMean = gvfSampleCount > 0.0f ? gvfPredictionAbsSum / gvfSampleCount : 0.0f;
    stats.modelErrorMean = modelSampleCount > 0.0f ? modelErrorSum / modelSampleCount : 0.0f;
    stats.rewardPredictionAbsMean = float(batchSize) > 0.0f ? rewardPredictionAbsSum / float(batchSize) : 0.0f;
    stats.statePredictionAbsMean = float(max(batchSize * activeRank, 1u)) > 0.0f ? statePredictionAbsSum / float(max(batchSize * activeRank, 1u)) : 0.0f;
    stats.optionSelectionFraction = optionSelectionCount / float(batchSize);
    stats.optionTrainingSignalMean = optionTrainingSignalSum / float(batchSize);
    stats.optionScoreMean = optionScoreSum / float(batchSize);
    stats.planningCandidateCount = planningCandidateSum / float(batchSize);
    stats.planningScoreMean = planningCandidateSum > 0.0f ? planningScoreSum / planningCandidateSum : 0.0f;
    stats.planningScoreMax = planningScoreMax == -FLT_MAX ? 0.0f : planningScoreMax;
    stats.planningActionChangeL2 = sqrt(planningActionChangeSquares / float(max(batchSize * actDim, 1u)));
    stats.rawNonFiniteCount = rawNonFiniteCount;
    stats.sanitizedNonFiniteCount = sanitizedNonFiniteCount;
    stats.fatalNonFiniteCount = fatalNonFiniteCount;
    statsOut[0] = stats;
}
