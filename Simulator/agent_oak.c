#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "agent.h"
#include "rng.h"

#define OBSERVATION_CAPACITY 9u
#define ACTION_CAPACITY 3u
#define SYMBOLS 16u
#define FEATURE_TIMESCALE 2048u
#define TRACKING_RATE (1.0f / (float)FEATURE_TIMESCALE)
#define OAK_AGENT_STREAM 0u

typedef struct { float reward, effect; uint32_t seen; } Predict;

typedef struct {
    uint32_t seed, step;

    const float *observation;
    float logit[ACTION_CAPACITY];

    float symbolValue[OBSERVATION_CAPACITY][SYMBOLS];
    uint32_t symbolCount[OBSERVATION_CAPACITY];
    uint32_t symbol[OBSERVATION_CAPACITY];
    uint32_t previousSymbol[OBSERVATION_CAPACITY];
    uint32_t translate[OBSERVATION_CAPACITY][OBSERVATION_CAPACITY][SYMBOLS + 1u];
    uint32_t dictionarySize;

    Predict model[SYMBOLS][ACTION_CAPACITY][OBSERVATION_CAPACITY][SYMBOLS];
    uint32_t bank, bankCount, gate, predictedChannel;

    float actionMean[ACTION_CAPACITY][OBSERVATION_CAPACITY];
    float channelMean[OBSERVATION_CAPACITY];
    float channelSquare[OBSERVATION_CAPACITY];

    float context, canonical[SYMBOLS], rewardRate;
    uint32_t stepsSinceReward, longestDrought;

    float alignment[ACTION_CAPACITY][OBSERVATION_CAPACITY];
    uint32_t sourceChannel[ACTION_CAPACITY];
    uint32_t alignmentAction;
    int haveAlignment;

    uint32_t previousAction;
    int havePreviousAction;
} OakAgent;

static float keepFinite(float old, float fresh){
    return isfinite(fresh) ? fresh : old;
}

static uint32_t encodeSymbol(OakAgent *a, uint32_t channel, float value){
    uint32_t count = a->symbolCount[channel];
    if (count == SYMBOLS) return SYMBOLS;
    for (uint32_t code = 0u; code < count; ++code)
        if (value == a->symbolValue[channel][code]) return code;
    a->symbolValue[channel][count] = value;
    a->symbolCount[channel] = count + 1u;
    return count;
}

static uint32_t convertSymbol(OakAgent *a, uint32_t from, uint32_t code, uint32_t to){
    if (code >= a->symbolCount[from]) return SYMBOLS;
    float value = a->symbolValue[from][code];
    for (uint32_t other = 0u; other < a->symbolCount[to]; ++other)
        if (a->symbolValue[to][other] == value) return other;
    return SYMBOLS;
}

static void rebuildTranslation(OakAgent *a){
    for (uint32_t from = 0u; from < OBSERVATION_CAPACITY; ++from)
        for (uint32_t to = 0u; to < OBSERVATION_CAPACITY; ++to)
            for (uint32_t code = 0u; code <= SYMBOLS; ++code)
                a->translate[from][to][code] = convertSymbol(a, from, code, to);
}

static void constructFeatures(OakAgent *a){
    for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel)
        a->symbol[channel] = encodeSymbol(a, channel, a->observation[channel]);

    uint32_t total = 0u;
    for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel)
        total += a->symbolCount[channel];
    if (total != a->dictionarySize){
        a->dictionarySize = total;
        rebuildTranslation(a);
    }
}

static void selectGate(OakAgent *a){
    float bestSpread = 0.0f;
    for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel){
        if (a->symbolCount[channel] == SYMBOLS) continue;
        float spread = 0.0f;
        for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action)
            for (uint32_t code = 0u; code < a->symbolCount[channel]; ++code)
                for (uint32_t bank = 0u; bank < a->bankCount; ++bank){
                    float weight = a->model[bank][action][channel][code].reward;
                    spread += weight * weight;
                }
        if (spread > bestSpread){
            bestSpread = spread;
            a->gate = channel;
        }
    }
}

static void trackAlignment(OakAgent *a){
    uint32_t gateSymbol = a->symbol[a->gate];
    uint32_t best = a->gate;
    float *row = a->alignment[a->alignmentAction];
    float bestMatch = -1.0f;
    for (uint32_t source = 0u; source < OBSERVATION_CAPACITY; ++source){
        row[source] = row[source] * (1.0f - TRACKING_RATE)
                    + (gateSymbol == a->previousSymbol[source] ? TRACKING_RATE : 0.0f);
        if (row[source] > bestMatch){
            bestMatch = row[source];
            best = source;
        }
    }
    a->sourceChannel[a->alignmentAction] = best;
}

static uint32_t mostActionDiscriminableChannel(OakAgent *a){
    uint32_t best = 0u;
    float bestScore = -1.0f;
    for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel){
        float value = a->observation[channel];
        if (a->haveAlignment)
            a->actionMean[a->alignmentAction][channel]
                += TRACKING_RATE * (value - a->actionMean[a->alignmentAction][channel]);
        a->channelMean[channel]   += TRACKING_RATE * (value - a->channelMean[channel]);
        a->channelSquare[channel] += TRACKING_RATE * (value * value - a->channelSquare[channel]);

        float sum = 0.0f;
        float between = 0.0f;
        for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action){
            float mean = a->actionMean[action][channel];
            sum += mean;
            between += mean * mean;
        }
        float within = a->channelSquare[channel] - a->channelMean[channel] * a->channelMean[channel];
        float score = (between - sum * sum / (float)ACTION_CAPACITY)
                    / (within > 0.0f ? within : 1.0f);
        if (score > bestScore && a->symbol[channel] < SYMBOLS){
            bestScore = score;
            best = channel;
        }
    }
    return best;
}

static void learnEffect(OakAgent *a, uint32_t target){
    for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel){
        if (a->previousSymbol[channel] >= SYMBOLS) continue;
        Predict *entry = &a->model[a->bank][a->alignmentAction][channel][a->previousSymbol[channel]];
        float rate = entry->seen < FEATURE_TIMESCALE
                   ? 1.0f / (float)++entry->seen
                   : TRACKING_RATE;
        entry->effect = keepFinite(entry->effect,
                                   entry->effect + rate * (a->observation[target] - entry->effect));
    }
}

static void learnReward(OakAgent *a, float centeredReward){
    float predicted = 0.0f;
    float strongest = -1.0f;
    uint32_t credit = OBSERVATION_CAPACITY;
    for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel){
        if (a->symbol[channel] >= SYMBOLS) continue;
        float weight = a->model[a->bank][a->previousAction][channel][a->symbol[channel]].reward;
        predicted += weight;
        if (fabsf(weight) > strongest){
            strongest = fabsf(weight);
            credit = channel;
        }
    }
    if (credit == OBSERVATION_CAPACITY) return;
    Predict *entry = &a->model[a->bank][a->previousAction][credit][a->symbol[credit]];
    entry->reward = keepFinite(entry->reward,
                               entry->reward + TRACKING_RATE * (centeredReward - predicted));
}

static void learn(OakAgent *a, float reward){
    a->stepsSinceReward++;
    if (reward != 0.0f){
        selectGate(a);
        if (a->symbol[a->gate] < SYMBOLS){
            if (a->stepsSinceReward > a->longestDrought) a->longestDrought = a->stepsSinceReward;
            a->stepsSinceReward = 0u;
            uint32_t gateSymbol = a->symbol[a->gate];
            float canonical = a->canonical[gateSymbol];
            if (canonical != 0.0f)
                a->context = keepFinite(a->context, reward / canonical);
            else
                a->canonical[gateSymbol] = keepFinite(0.0f, reward / a->context);
        }
    }

    float centeredReward = keepFinite(0.0f, reward / a->context);
    a->rewardRate += TRACKING_RATE * (reward - a->rewardRate);

    if (a->haveAlignment && a->symbol[a->gate] < SYMBOLS) trackAlignment(a);

    uint32_t target = mostActionDiscriminableChannel(a);
    if (a->haveAlignment) learnEffect(a, target);

    a->bank = a->previousSymbol[target] < SYMBOLS ? a->previousSymbol[target] : 0u;
    a->bankCount = a->symbolCount[target];
    a->predictedChannel = target;

    memcpy(a->previousSymbol, a->symbol, sizeof a->previousSymbol);
    a->alignmentAction = a->previousAction;
    a->haveAlignment = 1;

    learnReward(a, centeredReward);
}

static void scoreActions(OakAgent *a){
    uint32_t mover = 0u;
    for (uint32_t action = 1u; action < ACTION_CAPACITY; ++action)
        if (a->actionMean[action][a->predictedChannel] > a->actionMean[mover][a->predictedChannel])
            mover = action;

    if (a->stepsSinceReward > a->longestDrought){
        for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action) a->logit[action] = 0.0f;
        return;
    }

    float trust = 1.0f / (1.0f + (float)a->stepsSinceReward * TRACKING_RATE);
    float gateValue = a->symbol[a->gate] < SYMBOLS ? a->canonical[a->symbol[a->gate]] : 0.0f;

    float value[ACTION_CAPACITY];
    float mean = 0.0f;
    for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action){
        uint32_t source = a->sourceChannel[action];
        uint32_t arriving = a->translate[source][a->gate][a->symbol[source]];
        float ahead = (arriving < SYMBOLS && action != mover) ? a->canonical[arriving] : 0.0f;

        float effect = FLT_MAX;
        for (uint32_t channel = 0u; channel < OBSERVATION_CAPACITY; ++channel){
            if (a->symbol[channel] >= SYMBOLS) continue;
            float predicted = a->model[a->bank][action][channel][a->symbol[channel]].effect;
            if (predicted < effect) effect = predicted;
        }

        float changes = (float)(action == mover ? effect > 0.0f
                                                : a->sourceChannel[action] != a->gate);
        value[action] = keepFinite(0.0f,
                                   effect * (a->context * gateValue + a->rewardRate)
                                   + fmaxf(a->context * ahead, 0.0f)
                                   + changes * a->rewardRate);
        mean += value[action] / (float)ACTION_CAPACITY;
    }

    float spread = 0.0f;
    for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action){
        float deviation = value[action] - mean;
        spread += deviation * deviation;
    }
    float deviation = sqrtf(spread);
    float gain = logf((float)(ACTION_CAPACITY - 1u) * (float)FEATURE_TIMESCALE)
               / (deviation > 0.0f ? deviation : 1.0f);

    for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action)
        a->logit[action] = keepFinite(0.0f, gain * (value[action] - mean) * trust);
}

static uint32_t sampleAction(OakAgent *a){
    u4 draw = philox4x32(a->seed, RNG_KEY_AGENT,
                         (u4){ a->step++, OAK_AGENT_STREAM, 0u, 0u });
    uint32_t word[ACTION_CAPACITY] = { draw.x, draw.y, draw.z };

    uint32_t chosen = 0u;
    float bestScore = -FLT_MAX;
    for (uint32_t action = 0u; action < ACTION_CAPACITY; ++action){
        float uniform = ((float)(word[action] >> 8) + 1.0f) * 0x1p-24f;
        if (uniform > 0x1.fffffep-1f) uniform = 0x1.fffffep-1f;
        float score = a->logit[action] - logf(-logf(uniform));
        if (score > bestScore){
            bestScore = score;
            chosen = action;
        }
    }
    return chosen;
}

static OakAgent oakAgent;

void *agentCreate(AgentShape shape, uint32_t seed){
    if (shape.observationCount != OBSERVATION_CAPACITY) abort();
    if (shape.actionCount != ACTION_CAPACITY) abort();
    oakAgent.seed = seed;
    oakAgent.context = 1.0f;
    return &oakAgent;
}

uint32_t agentStep(void *handle, const float *observation, float reward){
    OakAgent *a = handle;
    if (a->havePreviousAction) learn(a, reward);
    a->observation = observation;
    constructFeatures(a);
    scoreActions(a);
    a->previousAction = sampleAction(a);
    a->havePreviousAction = 1;
    return a->previousAction;
}
