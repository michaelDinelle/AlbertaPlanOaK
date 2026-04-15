#include <metal_stdlib>
using namespace metal;

struct KernelConfig { uint phase, obsDim, rank, actDim, batchSize, actionIsDiscrete; };
struct KernelScalars { uint stepCount, rewardCount, activeCount; float rewardRate; };
struct KernelStats {
    float rewardRate, latentPredictionMSE, rewardPredictionMSE, latentVarianceTrace, activeRank, actionDeltaL2Mean, planningActionChangeL2Mean;
    float rawNonFiniteCount, sanitizedNonFiniteCount, encodeNonFiniteCount, learnNonFiniteCount, planNonFiniteCount;
    float maxAbsLatentPrediction, maxAbsRewardPrediction, maxAbsPolicyOutput;
};

// --- SANITIZATION ---
inline float sanitize(float v, thread uint& raw, thread uint& san) {
    if (isfinite(v)) return v;
    raw++; san++;
    return 0.0f;
}

inline float masked_dot(device const float *w, uint wOff, device const float *v, uint vOff, device const uint *active, uint count, thread uint& r, thread uint& s) {
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        uint idx = active[i];
        sum += w[wOff + idx] * sanitize(v[vOff + idx], r, s);
    }
    return sanitize(sum, r, s);
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
    device float *featureResidual [[buffer(24)]], // Utility Tracker
    device uint *activeState [[buffer(25)]],
    device float *rolloutState [[buffer(26)]],
    device float *latentPower [[buffer(27)]],
    device const float *actionBounds [[buffer(28)]],
    device atomic_uint *counters [[buffer(29)]],
    constant KernelConfig &config [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint obsDim = config.obsDim;
    const uint rank = config.rank;
    const uint actDim = config.actDim;
    const uint batchSize = config.batchSize;
    const bool isDisc = config.actionIsDiscrete != 0u;
    device uint *activeMask = activeState;
    device uint *activeList = activeState + rank;
    const uint activeCount = scalars->activeCount;
    uint raw = 0, san = 0;

    // PHASE 0: Representation (Sutton Step 1-2)
    if (config.phase == 0u) {
        uint row = gid / rank;
        uint feat = gid % rank;
        if (row < batchSize) {
            float centered = sanitize(observation[row * obsDim + feat] - observationMean[feat], raw, san);
            state[row * rank + feat] = sanitize(centered * encoderWeights[feat], raw, san);
        }
        return;
    }

    // PHASE 1: Prediction & TD Error (Sutton Step 3-5: GVFs)
    if (config.phase == 1u) {
        if (gid < batchSize) {
            uint sOff = gid * rank;
            uint aOff = gid * actDim;
            
            // Predict Reward
            float rPred = masked_dot(rewardWeights, 0, previousState, sOff, activeList, activeCount, raw, san);
            rewardPrediction[gid] = rPred;

            // Average Reward TD Error: delta = r - rho + V(s') - V(s)
            float vPrev = masked_dot(valueWeights, 0, previousState, sOff, activeList, activeCount, raw, san);
            float vNow = done[gid] ? 0.0f : masked_dot(valueWeights, 0, state, sOff, activeList, activeCount, raw, san);
            float delta = reward[gid] - scalars->rewardRate + vNow - vPrev;
            td[gid] = sanitize(delta / (1.0f + fabs(vNow)), raw, san);
        }
        return;
    }

    // PHASE 2: Continual Learning & Utility tracking (Sutton Step 11)
    if (config.phase == 2u) {
        if (gid < rank) {
            uint feat = gid;
            float weightUpdate = 0;
            float utility = 0;
            
            for (uint row = 0; row < batchSize; ++row) {
                float sVal = state[row * rank + feat];
                float tdVal = td[row];
                weightUpdate += tdVal * sVal;
                // The Alberta Utility: Absolute contribution to the learning signal
                utility += fabs(tdVal * sVal);
            }
            
            // Update Value Weight
            valueWeights[feat] += weightUpdate / float(batchSize);
            // Update Utility (Exponential Moving Average)
            featureResidual[feat] = 0.99f * featureResidual[feat] + 0.01f * (utility / float(batchSize));
        }
        return;
    }

    // PHASE 3: Policy Control (Sutton Step 4, 6)
    if (config.phase == 3u) {
        uint row = gid / actDim;
        uint actIdx = gid % actDim;
        if (row < batchSize) {
            action[row * actDim + actIdx] = masked_dot(policyWeights, actIdx * rank, state, row * rank, activeList, activeCount, raw, san);
        }
        return;
    }

    // PHASE 4: Action Refinement & Planning
    if (config.phase == 4u) {
        if (gid < batchSize) {
            uint aOff = gid * actDim;
            for (uint a = 0; a < actDim; ++a) {
                action[aOff + a] = clamp(action[aOff + a], actionBounds[a], actionBounds[actDim + a]);
            }
        }
        return;
    }

    // PHASE 5: Oak Pruning (The Core of Step 11)
    if (config.phase == 5u && gid == 0u) {
        float totalUtility = 0;
        for(uint i=0; i<rank; ++i) totalUtility += featureResidual[i];
        
        // Prune features that contribute less than 2% of the average utility
        float threshold = (totalUtility / float(rank)) * 0.02f;
        uint count = 0;
        for (uint i=0; i<rank; ++i) {
            if (featureResidual[i] > threshold) {
                activeMask[i] = 1;
                activeList[count++] = i;
            } else {
                activeMask[i] = 0;
            }
        }
        scalars->activeCount = count;
    }
    
    // PHASE 6: Average Reward Update (Non-episodic stability)
    if (config.phase == 6u && gid == 0u) {
        float batchR = 0;
        for(uint i=0; i<batchSize; ++i) batchR += reward[i];
        float meanR = batchR / float(batchSize);
        scalars->rewardRate = 0.99f * scalars->rewardRate + 0.01f * meanR;
        scalars->stepCount++;
    }
}