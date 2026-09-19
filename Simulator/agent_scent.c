#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "agent.h"

#define OBSERVATIONS 12
#define ACTIONS 3
#define SYMBOLS 16
#define PLAN_NODES 256
#define PLAN_DEPTH 12

#define RING_CHANNELS 8
#define SCENT_CHANNEL RING_CHANNELS
#define MOVED_CHANNEL (OBSERVATIONS - 1)
#define SCENT_DIMENSIONS 3

#define SEARCH_RADIUS 8
#define TRACK_LENGTH 32
#define TRACK_AGE 16
#define TRACK_SAMPLES 4
#define SOURCE_STRENGTH 0.002
#define AMPLITUDE_MIN 0.6
#define AMPLITUDE_MAX 1.6
#define AMPLITUDE_PENALTY 0.01
#define DISTANCE_PENALTY 0.001
#define FIT_QUALITY 0.995

#define KERNEL_SIDE 25
#define LATTICE_SIDE 51
#define LATTICE_CENTRE 25
#define TAIL_AGES 257
#define TAIL_RADIUS 16
#define TAIL_CELLS 153
#define SETTLE_STEPS 2000

#define MAP_SLOTS 1024
#define MAP_MASK 31
#define EATEN_SLOTS 64
#define PROBE_SAMPLES 4
#define PROBE_SPACING 32
#define EMISSIONS_WANTED 8
#define CENTROIDS_BEFORE_NAVIGATION 32

typedef struct {
    float value[SYMBOLS];
    uint32_t count[SYMBOLS];
    int size, continuous, mode;
} Alphabet;

typedef struct {
    uint64_t visits, rewarded, unchanged;
} Outcome;

typedef struct {
    Alphabet alphabet[OBSERVATIONS];
    uint64_t transportSeen[ACTIONS][OBSERVATIONS];
    uint64_t transportMatched[ACTIONS][OBSERVATIONS][OBSERVATIONS];
    int sourceOf[ACTIONS][OBSERVATIONS];
    int categorical[OBSERVATIONS], categoricalCount;
    Outcome outcome[ACTIONS][OBSERVATIONS][SYMBOLS];
    uint64_t actionSeen[ACTIONS];
    double novelty[ACTIONS];

    int previousCode[OBSERVATIONS], previousAction, started;
    int gateChannel[ACTIONS], exploreAction;
    float previousObservation[OBSERVATIONS];

    double canonical[ACTIONS][OBSERVATIONS][SYMBOLS];
    unsigned char known[ACTIONS][OBSERVATIONS][SYMBOLS];
    double polarity, rewardBase;

    double rewardValue[SYMBOLS];
    uint64_t rewardCount[SYMBOLS];
    int rewardKinds, rewardMode;

    int plan[PLAN_DEPTH], planLength, planAt;
    uint64_t steps, lastRewardStep, longestGap;
    int alarm;
    uint64_t rng;
} SymbolCore;

typedef struct {
    SymbolCore core;
    int x, y, heading, lastAction, started;

    double centroid[SYMBOLS][SCENT_DIMENSIONS];
    uint64_t centroidCount[SYMBOLS], centroidUpdates;
    double emission[SYMBOLS][SCENT_DIMENSIONS];
    uint64_t emissionCount[SYMBOLS];
    double decode[SYMBOLS][SCENT_DIMENSIONS];

    float previousScent[SCENT_DIMENSIONS];
    double cleanScent[SCENT_DIMENSIONS];

    int trackX[TRACK_LENGTH], trackY[TRACK_LENGTH];
    double trackScent[TRACK_LENGTH][SCENT_DIMENSIONS];
    uint64_t trackStep[TRACK_LENGTH];
    int trackCount;

    int mapX[MAP_SLOTS], mapY[MAP_SLOTS];
    float mapValue[MAP_SLOTS];
    uint64_t mapStamp[MAP_SLOTS];

    double kernel[KERNEL_SIDE][KERNEL_SIDE], kernelTotal;
    float tailMass[TAIL_AGES][TAIL_CELLS];
    double retention, diffusion, physicsLoss;
    int physicsKnown;

    int probeActive, probeCode, probeIndex;
    double probeDelta[PROBE_SAMPLES][SCENT_DIMENSIONS];
    uint64_t lastConsumeStep;

    int eatenX[EATEN_SLOTS], eatenY[EATEN_SLOTS], eatenCode[EATEN_SLOTS];
    uint64_t eatenStep[EATEN_SLOTS];
    int eatenCount;

    int targetX, targetY;
} ScentAgent;

static const int STEP_X[4] = { 0, 1, 0, -1 };
static const int STEP_Y[4] = { -1, 0, 1, 0 };
static const int RING_X[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
static const int RING_Y[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };

static uint64_t nextRandom(SymbolCore *core){
    uint64_t z = (core->rng += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int encodeValue(Alphabet *alphabet, float value){
    if (alphabet->continuous) return -1;
    for (int code = 0; code < alphabet->size; code++)
        if (value == alphabet->value[code]){
            if (++alphabet->count[code] > alphabet->count[alphabet->mode]) alphabet->mode = code;
            return code;
        }
    if (alphabet->size == SYMBOLS){
        alphabet->continuous = 1;
        return -1;
    }
    int code = alphabet->size++;
    alphabet->value[code] = value;
    alphabet->count[code] = 1;
    return code;
}

static int translateCode(const SymbolCore *core, int from, int code, int to){
    if (code < 0) return -1;
    float value = core->alphabet[from].value[code];
    for (int other = 0; other < core->alphabet[to].size; other++)
        if (core->alphabet[to].value[other] == value) return other;
    return -1;
}

static void refreshModels(SymbolCore *core){
    core->categoricalCount = 0;
    for (int channel = 0; channel < OBSERVATIONS; channel++){
        core->categorical[channel] = !core->alphabet[channel].continuous
                                   && core->alphabet[channel].size >= 2;
        core->categoricalCount += core->categorical[channel];
    }

    for (int action = 0; action < ACTIONS; action++){
        double bestGate = 0;
        int gate = -1;
        for (int channel = 0; channel < OBSERVATIONS; channel++){
            if (!core->categorical[channel]) continue;
            double score = 0;
            for (int code = 0; code < core->alphabet[channel].size; code++){
                Outcome *cell = &core->outcome[action][channel][code];
                if (cell->visits && code != core->alphabet[channel].mode)
                    score += (double)cell->rewarded * cell->rewarded / cell->visits;
            }
            if (score > bestGate){
                bestGate = score;
                gate = channel;
            }
        }
        core->gateChannel[action] = gate;

        for (int target = 0; target < OBSERVATIONS; target++){
            int source = -1;
            double best = 0.5;
            if (core->categorical[target])
                for (int channel = 0; channel < OBSERVATIONS; channel++){
                    if (!core->categorical[channel]) continue;
                    if (core->transportSeen[action][channel] < ACTIONS) continue;
                    if (core->alphabet[channel].value[core->alphabet[channel].mode]
                     != core->alphabet[target].value[core->alphabet[target].mode]) continue;
                    double share = (double)core->transportMatched[action][channel][target]
                                 / core->transportSeen[action][channel];
                    if (share > best){
                        best = share;
                        source = channel;
                    }
                }
            core->sourceOf[action][target] = source;
        }
    }

    int best = -1;
    double bestValue = -1;
    for (int action = 0; action < ACTIONS; action++){
        double value = core->actionSeen[action]
            ? core->novelty[action] / (OBSERVATIONS * core->actionSeen[action])
              + sqrt(2.0 * log((double)core->steps + ACTIONS) / core->actionSeen[action])
            : DBL_MAX;
        if (value > bestValue
         || (value == bestValue && (best < 0 || core->actionSeen[action] < core->actionSeen[best]))){
            bestValue = value;
            best = action;
        }
    }
    core->exploreAction = best;
}

static void trackRewardBase(SymbolCore *core, double reward){
    int kind = -1;
    for (int code = 0; code < core->rewardKinds; code++)
        if (core->rewardValue[code] == reward){
            kind = code;
            break;
        }
    if (kind < 0 && core->rewardKinds < SYMBOLS){
        kind = core->rewardKinds++;
        core->rewardValue[kind] = reward;
    }
    if (kind < 0) return;

    core->rewardCount[kind]++;
    if (core->rewardCount[kind] > core->rewardCount[core->rewardMode]) core->rewardMode = kind;

    double base = core->rewardValue[core->rewardMode];
    if (base == core->rewardBase) return;
    core->rewardBase = base;
    core->polarity = 1;
    memset(core->known, 0, sizeof core->known);
    memset(core->canonical, 0, sizeof core->canonical);
}

static void learnTransition(SymbolCore *core, const int *current, double reward){
    int action = core->previousAction;
    trackRewardBase(core, reward);

    int changed = 0, informative = 0;
    for (int channel = 0; channel < OBSERVATIONS; channel++){
        if (!core->categorical[channel]) continue;
        if (current[channel] != core->previousCode[channel]) changed = 1;
        if (core->previousCode[channel] != core->alphabet[channel].mode) informative = 1;
    }
    core->actionSeen[action]++;

    int rewarded = reward != core->rewardBase;
    if (rewarded){
        uint64_t gap = core->steps - core->lastRewardStep;
        if (gap > core->longestGap) core->longestGap = gap;
        core->lastRewardStep = core->steps;
        core->alarm = 0;
    }

    if (informative)
        for (int channel = 0; channel < OBSERVATIONS; channel++){
            if (!core->categorical[channel]) continue;
            if (core->previousCode[channel] < 0) continue;
            if (core->previousCode[channel] == core->alphabet[channel].mode) continue;
            core->transportSeen[action][channel]++;
            float value = core->alphabet[channel].value[core->previousCode[channel]];
            for (int target = 0; target < OBSERVATIONS; target++)
                if (core->categorical[target] && current[target] >= 0
                 && core->alphabet[target].value[current[target]] == value)
                    core->transportMatched[action][channel][target]++;
        }

    for (int channel = 0; channel < OBSERVATIONS; channel++){
        if (core->previousCode[channel] < 0) continue;
        if (core->alphabet[channel].continuous) continue;
        Outcome *cell = &core->outcome[action][channel][core->previousCode[channel]];
        cell->visits++;
        cell->rewarded += rewarded;
        cell->unchanged += (!changed && informative);
    }

    if (!rewarded) return;
    refreshModels(core);
    int gate = core->gateChannel[action];
    if (gate < 0) return;

    int code = core->previousCode[gate];
    double offset = reward - core->rewardBase;
    if (core->known[action][gate][code] && core->canonical[action][gate][code] != 0)
        core->polarity = offset / core->canonical[action][gate][code];
    else {
        core->known[action][gate][code] = 1;
        core->canonical[action][gate][code] = offset / core->polarity;
    }
}

static double predictedReward(const SymbolCore *core, const signed char *codes, int action){
    int gate = core->gateChannel[action];
    if (gate < 0) return 0;
    int code = codes[gate];
    if (code < 0) return 0;
    return core->known[action][gate][code] ? core->polarity * core->canonical[action][gate][code] : 0;
}

static int predictedBlocked(const SymbolCore *core, const signed char *codes, int action){
    int gate = core->gateChannel[action];
    if (gate < 0) return 0;
    int code = codes[gate];
    if (code < 0) return 0;
    const Outcome *cell = &core->outcome[action][gate][code];
    return cell->visits > 0 && cell->rewarded == 0 && cell->unchanged * 2 > cell->visits;
}

typedef struct {
    signed char codes[OBSERVATIONS];
    double value;
    short depth;
    signed char first, sequence[PLAN_DEPTH];
} PlanNode;

#define QUEUED_SLOTS 1024

static uint64_t queuedKey[QUEUED_SLOTS];
static double queuedBest[QUEUED_SLOTS];
static uint32_t queuedVisit[QUEUED_SLOTS], queuedGeneration;

static uint64_t codesKey(const signed char *codes){
    uint64_t key = 1469598103934665603ULL;
    for (int channel = 0; channel < OBSERVATIONS; channel++){
        key ^= (uint64_t)(unsigned char)codes[channel];
        key *= 1099511628211ULL;
    }
    return key ? key : 1ULL;
}

static double *queuedEntryFor(const signed char *codes){
    uint64_t key = codesKey(codes);
    for (uint32_t probe = (uint32_t)key & (QUEUED_SLOTS - 1u); ;
         probe = (probe + 1u) & (QUEUED_SLOTS - 1u)){
        if (queuedVisit[probe] != queuedGeneration){
            queuedVisit[probe] = queuedGeneration;
            queuedKey[probe] = key;
            queuedBest[probe] = -DBL_MAX;
            return &queuedBest[probe];
        }
        if (queuedKey[probe] == key) return &queuedBest[probe];
    }
}

static int queuedValueAtLeast(const signed char *codes, double value){
    return *queuedEntryFor(codes) >= value;
}

static void rememberQueued(const signed char *codes, double value){
    double *best = queuedEntryFor(codes);
    if (value > *best) *best = value;
}

static int planAction(SymbolCore *core, const int *current){
    if (core->steps < OBSERVATIONS * ACTIONS || core->categoricalCount < ACTIONS)
        return (int)(nextRandom(core) % ACTIONS);

    int informative = 0;
    for (int channel = 0; channel < OBSERVATIONS; channel++)
        if (core->categorical[channel] && current[channel] >= 0
         && current[channel] != core->alphabet[channel].mode) informative = 1;
    if (!informative && core->exploreAction >= 0) return core->exploreAction;

    PlanNode queue[PLAN_NODES];
    memset(&queue[0], 0, sizeof queue[0]);
    ++queuedGeneration;
    for (int channel = 0; channel < OBSERVATIONS; channel++)
        queue[0].codes[channel] = (signed char)current[channel];
    queue[0].first = -1;
    rememberQueued(queue[0].codes, queue[0].value);

    int count = 1, head = 0, best = -1;
    double bestRate = 0;
    PlanNode bestNode = queue[0], edgeNode = queue[0];
    int edge = -1, edgeDepth = PLAN_DEPTH + 1;
    double edgeValue = -DBL_MAX;
    uint64_t edgeTies = 0;

    while (head < count){
        PlanNode node = queue[head++];
        if (node.depth == PLAN_DEPTH) continue;

        for (int action = 0; action < ACTIONS; action++){
            if (predictedBlocked(core, node.codes, action)) continue;

            PlanNode next = node;
            next.sequence[node.depth] = action;
            next.depth++;
            next.first = node.depth ? node.first : (signed char)action;
            next.value += predictedReward(core, node.codes, action);

            int missing = 0, differs = 0;
            for (int target = 0; target < OBSERVATIONS; target++){
                if (!core->categorical[target]) continue;
                int source = core->sourceOf[action][target];
                if (source < 0){
                    next.codes[target] = (signed char)core->alphabet[target].mode;
                    missing++;
                } else {
                    next.codes[target] = (signed char)translateCode(core, source, node.codes[source], target);
                }
                if (next.codes[target] != node.codes[target]) differs = 1;
            }

            double rate = next.value / next.depth;
            if (rate > bestRate){
                bestRate = rate;
                best = next.first;
                bestNode = next;
            }

            if (missing){
                int nonNegative = next.value >= 0, wasNonNegative = edgeValue >= 0;
                if (edge < 0 || nonNegative > wasNonNegative
                 || (nonNegative == wasNonNegative
                     && (next.depth < edgeDepth
                         || (next.depth == edgeDepth && next.value > edgeValue)))){
                    edge = next.first;
                    edgeDepth = next.depth;
                    edgeValue = next.value;
                    edgeTies = 1;
                    edgeNode = next;
                } else if (nonNegative == wasNonNegative && next.depth == edgeDepth
                        && next.value == edgeValue){
                    if (nextRandom(core) % ++edgeTies == 0){
                        edge = next.first;
                        edgeNode = next;
                    }
                }
            }

            if (!differs) continue;
            if (queuedValueAtLeast(next.codes, next.value)) continue;
            if (count < PLAN_NODES){
                rememberQueued(next.codes, next.value);
                queue[count++] = next;
            }
        }
    }

    if (best < 0 && edge < 0) return (int)(nextRandom(core) % ACTIONS);

    PlanNode *selected = best >= 0 ? &bestNode : &edgeNode;
    core->planLength = selected->depth;
    core->planAt = 1;
    for (int index = 0; index < selected->depth; index++)
        core->plan[index] = selected->sequence[index];
    return selected->sequence[0];
}

static int coreStep(SymbolCore *core, const float *observation, double reward){
    int current[OBSERVATIONS];
    for (int channel = 0; channel < OBSERVATIONS; channel++)
        current[channel] = encodeValue(&core->alphabet[channel], observation[channel]);

    if (core->started)
        for (int channel = 0; channel < OBSERVATIONS; channel++){
            if (!core->alphabet[channel].continuous) continue;
            double scale = fabs(observation[channel]) + fabs(core->previousObservation[channel]);
            if (scale > 0)
                core->novelty[core->previousAction]
                    += fabs(observation[channel] - core->previousObservation[channel]) / scale;
        }
    memcpy(core->previousObservation, observation, sizeof core->previousObservation);

    if (core->started){
        if (core->planLength && core->planAt == core->planLength) core->planLength = 0;
        learnTransition(core, current, reward);
    }
    if (core->steps % OBSERVATIONS == 0 || reward != core->rewardBase) refreshModels(core);

    int action = core->planLength ? core->plan[core->planAt++] : planAction(core, current);

    if (core->steps - core->lastRewardStep > core->longestGap){
        core->alarm = 1;
        action = (int)(nextRandom(core) % ACTIONS);
        core->planLength = 0;
        core->planAt = 0;
    }

    memcpy(core->previousCode, current, sizeof current);
    core->previousAction = action;
    core->started = 1;
    core->steps++;
    return action;
}

static void buildDefaultKernel(ScentAgent *agent){
    for (int i = 0; i < KERNEL_SIDE; i++)
        for (int j = 0; j < KERNEL_SIDE; j++){
            double distance = sqrt(i * i + j * j);
            agent->kernel[i][j] = distance > 0 ? exp(-.5 * (distance - 1)) / sqrt(distance) : 2.5;
        }
}

static void buildPhysicalKernel(ScentAgent *agent){
    static double field[LATTICE_SIDE][LATTICE_SIDE], next[LATTICE_SIDE][LATTICE_SIDE];
    memset(field, 0, sizeof field);
    memset(agent->tailMass, 0, sizeof agent->tailMass);
    field[LATTICE_CENTRE][LATTICE_CENTRE] = 1;

    for (int age = 0; age < SETTLE_STEPS; age++){
        if (age <= TAIL_AGES - 1)
            for (int i = 0; i <= TAIL_RADIUS; i++)
                for (int j = 0; j <= i; j++)
                    agent->tailMass[age][i * (i + 1) / 2 + j] =
                        (float)field[LATTICE_CENTRE + i][LATTICE_CENTRE + j];

        for (int i = 0; i < LATTICE_SIDE; i++)
            for (int j = 0; j < LATTICE_SIDE; j++)
                next[i][j] = agent->retention * field[i][j]
                           + agent->diffusion * ((i ? field[i - 1][j] : 0)
                                               + (i < LATTICE_SIDE - 1 ? field[i + 1][j] : 0)
                                               + (j ? field[i][j - 1] : 0)
                                               + (j < LATTICE_SIDE - 1 ? field[i][j + 1] : 0));
        next[LATTICE_CENTRE][LATTICE_CENTRE] += 1;
        memcpy(field, next, sizeof field);
    }

    agent->kernelTotal = field[LATTICE_CENTRE + 1][LATTICE_CENTRE];
    for (int i = 0; i < KERNEL_SIDE; i++)
        for (int j = 0; j < KERNEL_SIDE; j++)
            agent->kernel[i][j] = field[LATTICE_CENTRE + i][LATTICE_CENTRE + j] / agent->kernelTotal;

    for (int age = 0; age <= TAIL_AGES - 1; age++)
        for (int i = 0; i <= TAIL_RADIUS; i++)
            for (int j = 0; j <= i; j++){
                int cell = i * (i + 1) / 2 + j;
                agent->tailMass[age][cell] =
                    (float)(field[LATTICE_CENTRE + i][LATTICE_CENTRE + j] - agent->tailMass[age][cell]);
            }
    agent->trackCount = 0;
}

static double physicsLossAt(double retention, double ratio, double fourth, double fifth,
                            double *diffusion){
    if (3 * retention <= ratio || ratio <= retention) return 1e20;
    double spread = retention * retention * (ratio - retention) / (3 * retention - ratio);
    double rate = sqrt(spread) / 2;
    if (retention + 4 * rate >= 1 || rate <= 0) return 1e20;

    double second = retention * retention + spread;
    double predictedFourth = (retention * retention * retention * retention
                            + 6 * retention * retention * spread + 2.25 * spread * spread) / second;
    double predictedFifth = (retention * retention * retention * retention * retention
                           + 10 * retention * retention * retention * spread
                           + 11.25 * retention * spread * spread) / second;
    *diffusion = rate;
    return (predictedFourth - fourth) * (predictedFourth - fourth)
         + (predictedFifth - fifth) * (predictedFifth - fifth);
}

static void fitPhysics(ScentAgent *agent){
    double moment[PROBE_SAMPLES] = { 0 };
    for (int sample = 0; sample < PROBE_SAMPLES; sample++)
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            moment[sample] += agent->probeDelta[sample][axis];
    if (moment[0] <= 1e-8) return;

    double ratio = moment[1] / moment[0];
    double fourth = moment[2] / moment[0];
    double fifth = moment[3] / moment[0];

    double best = 1e20, bestRetention = 0, rate;
    for (int grid = 1; grid < 1000; grid++){
        double retention = grid / 1000.0;
        double loss = physicsLossAt(retention, ratio, fourth, fifth, &rate);
        if (loss < best){
            best = loss;
            bestRetention = retention;
        }
    }
    if (bestRetention == 0) return;

    double low = bestRetention - .002, high = bestRetention + .002;
    for (int refine = 0; refine < 64; refine++){
        double a = (2 * low + high) / 3, b = (low + 2 * high) / 3;
        if (physicsLossAt(a, ratio, fourth, fifth, &rate)
          < physicsLossAt(b, ratio, fourth, fifth, &rate)) high = b;
        else low = a;
    }

    double retention = (low + high) / 2;
    best = physicsLossAt(retention, ratio, fourth, fifth, &rate);
    if (best >= agent->physicsLoss) return;

    agent->physicsLoss = best;
    agent->retention = retention;
    agent->diffusion = rate;
    agent->physicsKnown = 1;
    buildPhysicalKernel(agent);
}

static void rebuildDecoder(ScentAgent *agent){
    SymbolCore *core = &agent->core;
    int columns[SCENT_DIMENSIONS], found = 0;

    for (int slot = 0; slot < SCENT_DIMENSIONS; slot++){
        int best = -1;
        for (int code = 0; code < core->alphabet[0].size; code++){
            if (agent->centroidCount[code] < 3 || agent->emissionCount[code] < 1) continue;
            int used = 0;
            for (int taken = 0; taken < found; taken++)
                if (columns[taken] == code) used = 1;
            if (used) continue;
            if (best < 0 || agent->centroidCount[code] > agent->centroidCount[best]) best = code;
        }
        if (best >= 0) columns[found++] = best;
    }

    memset(agent->decode, 0, sizeof agent->decode);
    if (found != SCENT_DIMENSIONS) return;

    double system[SCENT_DIMENSIONS][2 * SCENT_DIMENSIONS];
    memset(system, 0, sizeof system);
    for (int row = 0; row < SCENT_DIMENSIONS; row++){
        for (int column = 0; column < SCENT_DIMENSIONS; column++)
            system[row][column] = agent->emission[columns[column]][row];
        system[row][SCENT_DIMENSIONS + row] = 1;
    }

    for (int column = 0; column < SCENT_DIMENSIONS; column++){
        int pivot = column;
        for (int row = column + 1; row < SCENT_DIMENSIONS; row++)
            if (fabs(system[row][column]) > fabs(system[pivot][column])) pivot = row;
        if (fabs(system[pivot][column]) < 1e-9) return;

        for (int entry = 0; entry < 2 * SCENT_DIMENSIONS; entry++){
            double swap = system[column][entry];
            system[column][entry] = system[pivot][entry];
            system[pivot][entry] = swap;
        }
        double scale = system[column][column];
        for (int entry = 0; entry < 2 * SCENT_DIMENSIONS; entry++) system[column][entry] /= scale;
        for (int row = 0; row < SCENT_DIMENSIONS; row++){
            if (row == column) continue;
            double factor = system[row][column];
            for (int entry = 0; entry < 2 * SCENT_DIMENSIONS; entry++)
                system[row][entry] -= factor * system[column][entry];
        }
    }

    for (int row = 0; row < SCENT_DIMENSIONS; row++){
        double scale = 0;
        if (agent->physicsKnown){
            double second = agent->retention * agent->retention
                          + 4 * agent->diffusion * agent->diffusion;
            scale = agent->kernelTotal / second;
        } else {
            for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
                scale += system[row][SCENT_DIMENSIONS + axis] * agent->centroid[columns[row]][axis];
        }
        if (scale <= 0) continue;
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            agent->decode[columns[row]][axis] = system[row][SCENT_DIMENSIONS + axis] / scale;
    }
}

static void subtractEatenScent(ScentAgent *agent, const float *observation){
    for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
        agent->cleanScent[axis] = observation[SCENT_CHANNEL + axis];
    if (!agent->physicsKnown) return;

    double second = agent->retention * agent->retention
                  + 4 * agent->diffusion * agent->diffusion;
    for (int back = 0; back < agent->eatenCount && back < EATEN_SLOTS; back++){
        int slot = (agent->eatenCount - 1 - back) % EATEN_SLOTS;
        uint64_t age = agent->core.steps - agent->eatenStep[slot];
        if (age > TAIL_AGES - 1) continue;

        int major = abs(agent->x - agent->eatenX[slot]);
        int minor = abs(agent->y - agent->eatenY[slot]);
        if (minor > major){
            int swap = major;
            major = minor;
            minor = swap;
        }
        if (major > TAIL_RADIUS) continue;

        int code = agent->eatenCode[slot];
        if (!agent->emissionCount[code]) continue;
        double mass = agent->tailMass[age][major * (major + 1) / 2 + minor] / second;
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            agent->cleanScent[axis] -= mass * agent->emission[code][axis];
    }
}

static double kernelAt(const ScentAgent *agent, int dx, int dy){
    int major = abs(dx), minor = abs(dy);
    return major < KERNEL_SIDE && minor < KERNEL_SIDE ? agent->kernel[major][minor] : 0;
}

static unsigned mapSlot(int x, int y){
    return (((unsigned)x & MAP_MASK) << 5) | ((unsigned)y & MAP_MASK);
}

static void rememberCell(ScentAgent *agent, int x, int y, float value){
    unsigned slot = mapSlot(x, y);
    agent->mapX[slot] = x;
    agent->mapY[slot] = y;
    agent->mapValue[slot] = value;
    agent->mapStamp[slot] = agent->core.steps + 1;
}

static int cellDiffers(const ScentAgent *agent, int x, int y, float value){
    unsigned slot = mapSlot(x, y);
    return agent->mapStamp[slot] && agent->mapX[slot] == x && agent->mapY[slot] == y
        && agent->mapValue[slot] != value;
}

static int collectBlockingValues(const ScentAgent *agent, float *values){
    const SymbolCore *core = &agent->core;
    int count = 0;
    for (int code = 0; code < core->alphabet[0].size; code++){
        int blocks;
        if (core->known[0][0][code] && core->polarity * core->canonical[0][0][code] < 0) blocks = 1;
        else {
            const Outcome *cell = &core->outcome[0][0][code];
            blocks = cell->visits > 0 && !cell->rewarded && cell->unchanged * 2 > cell->visits
                  && code != core->alphabet[0].mode;
        }
        if (blocks) values[count++] = core->alphabet[0].value[code];
    }
    return count;
}

static int cellBlocks(const ScentAgent *agent, const float *blocking, int blockingCount,
                      int x, int y){
    unsigned slot = mapSlot(x, y);
    if (!agent->mapStamp[slot] || agent->mapX[slot] != x || agent->mapY[slot] != y) return 0;
    for (int index = 0; index < blockingCount; index++)
        if (agent->mapValue[slot] == blocking[index]) return 1;
    return 0;
}

static int routeFirstAction(const ScentAgent *agent, int targetX, int targetY){
    enum { SIDE = 33, CENTRE = 16, CAPACITY = SIDE * SIDE * 4 };
    static short queue[CAPACITY];
    static signed char firstMove[CAPACITY];
    static uint32_t seen[CAPACITY];
    static uint32_t visit;
    ++visit;

    float blocking[SYMBOLS];
    int blockingCount = collectBlockingValues(agent, blocking);

    int start = ((CENTRE * SIDE + CENTRE) * 4 + agent->heading), head = 0, tail = 1;
    queue[0] = (short)start;
    firstMove[start] = -1;
    seen[start] = visit;

    while (head < tail){
        int node = queue[head++];
        int heading = node % 4, x = (node / 4) / SIDE, y = (node / 4) % SIDE;
        if (x - CENTRE + agent->x == targetX && y - CENTRE + agent->y == targetY)
            return firstMove[node];

        for (int action = 0; action < ACTIONS; action++){
            int nextX = x, nextY = y, nextHeading = heading;
            if (action == 1) nextHeading = (heading + 3) % 4;
            else if (action == 2) nextHeading = (heading + 1) % 4;
            else {
                nextX += STEP_X[heading];
                nextY += STEP_Y[heading];
                if (nextX < 0 || nextX >= SIDE || nextY < 0 || nextY >= SIDE) continue;
                if (cellBlocks(agent, blocking, blockingCount,
                               nextX - CENTRE + agent->x, nextY - CENTRE + agent->y)) continue;
            }
            int target = (nextX * SIDE + nextY) * 4 + nextHeading;
            if (seen[target] == visit) continue;
            seen[target] = visit;
            firstMove[target] = node == start ? (signed char)action : firstMove[node];
            queue[tail++] = (short)target;
        }
    }
    return -1;
}

static int locateSource(ScentAgent *agent, int code, int *targetX, int *targetY){
    double projection[TRACK_SAMPLES];
    int slotOf[TRACK_SAMPLES];
    int used = 0;
    double energy = 0;

    for (int back = 0; back < agent->trackCount && used < TRACK_SAMPLES; back++){
        int slot = (agent->trackCount - 1 - back) % TRACK_LENGTH;
        if (agent->core.steps - agent->trackStep[slot] > TRACK_AGE) break;
        double value = 0;
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            value += agent->trackScent[slot][axis] * agent->decode[code][axis];
        if (value < 0) value = 0;
        projection[used] = value;
        slotOf[used] = slot;
        energy += value * value;
        used++;
    }
    if (used < 3 || energy < 1e-12) return 0;

    double best = 0;
    int found = 0;
    float gateValue = agent->core.alphabet[0].value[code];

    for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++)
        for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++){
            int distance = abs(dx) + abs(dy);
            if (distance < 2 || distance > SEARCH_RADIUS + SEARCH_RADIUS / 2) continue;
            int x = agent->x + dx, y = agent->y + dy;
            if (cellDiffers(agent, x, y, gateValue)) continue;

            double square = 0, cross = 0;
            for (int sample = 0; sample < used; sample++){
                int slot = slotOf[sample];
                double weight = kernelAt(agent, x - agent->trackX[slot], y - agent->trackY[slot]);
                square += weight * weight;
                cross += weight * projection[sample];
            }
            if (square < 1e-16) continue;

            double amplitude = cross / square;
            double quality = cross * cross / (square * energy);
            if (amplitude < AMPLITUDE_MIN || amplitude > AMPLITUDE_MAX) continue;
            if (quality < FIT_QUALITY) continue;

            double score = quality - AMPLITUDE_PENALTY * fabs(log(amplitude))
                         - DISTANCE_PENALTY * distance;
            if (score > best){
                best = score;
                *targetX = x;
                *targetY = y;
                found = 1;
            }
        }
    return found;
}

static void advanceProbe(ScentAgent *agent, const float *observation){
    for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
        agent->probeDelta[agent->probeIndex][axis] =
            agent->previousScent[axis] - observation[SCENT_CHANNEL + axis];
    agent->probeIndex++;
    if (agent->probeIndex != PROBE_SAMPLES) return;

    int code = agent->probeCode;
    ++agent->emissionCount[code];
    for (int axis = 0; axis < SCENT_DIMENSIONS; axis++){
        double value = agent->probeDelta[0][axis];
        if (agent->emissionCount[code] == 1 || value < agent->emission[code][axis])
            agent->emission[code][axis] = value;
    }
    fitPhysics(agent);
    rebuildDecoder(agent);
    agent->probeActive = 0;
}

static void followLastAction(ScentAgent *agent, const float *observation){
    SymbolCore *core = &agent->core;
    if (agent->lastAction == 1) agent->heading = (agent->heading + 3) % 4;
    else if (agent->lastAction == 2) agent->heading = (agent->heading + 1) % 4;
    else if (observation[MOVED_CHANNEL] > 0.5f){
        agent->x += STEP_X[agent->heading];
        agent->y += STEP_Y[agent->heading];
    }

    if (agent->lastAction != 0 || observation[MOVED_CHANNEL] <= 0.5f) return;
    int code = core->previousCode[0];
    if (code < 0 || code == core->alphabet[0].mode) return;

    double rate = 1.0 / (++agent->centroidCount[code]);
    for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
        agent->centroid[code][axis] += rate * (agent->previousScent[axis] - agent->centroid[code][axis]);
    agent->centroidUpdates++;
    rebuildDecoder(agent);

    int slot = agent->eatenCount++ % EATEN_SLOTS;
    agent->eatenX[slot] = agent->x;
    agent->eatenY[slot] = agent->y;
    agent->eatenCode[slot] = code;
    agent->eatenStep[slot] = core->steps - 1;

    if (agent->emissionCount[code] < EMISSIONS_WANTED && agent->probeActive == 0
     && core->steps - agent->lastConsumeStep >= PROBE_SPACING){
        agent->probeActive = 1;
        agent->probeIndex = 0;
        agent->probeCode = code;
    }
    agent->lastConsumeStep = core->steps;
}

static int bestRewardingCode(const ScentAgent *agent){
    const SymbolCore *core = &agent->core;
    int best = -1;
    double bestValue = 0;
    for (int code = 0; code < core->alphabet[0].size; code++){
        if (!core->known[0][0][code]) continue;
        double value = core->polarity * core->canonical[0][0][code];
        if (value > bestValue){
            bestValue = value;
            best = code;
        }
    }
    return best;
}

static int decoderIsClean(const ScentAgent *agent){
    const SymbolCore *core = &agent->core;
    double positive = 0, negative = 0;
    for (int code = 0; code < core->alphabet[0].size; code++){
        double value = 0;
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            value += agent->decode[code][axis] * agent->cleanScent[axis];
        if (value > 0) positive += value;
        else negative -= value;
    }
    return negative <= positive / 4;
}

static int scentStep(ScentAgent *agent, const float *observation, double reward){
    SymbolCore *core = &agent->core;

    if (agent->probeActive) advanceProbe(agent, observation);
    if (agent->started) followLastAction(agent, observation);
    subtractEatenScent(agent, observation);

    int action = coreStep(core, observation, reward);

    for (int cell = 0; cell < RING_CHANNELS; cell++){
        int direction = (2 * agent->heading + cell) & 7;
        rememberCell(agent, agent->x + RING_X[direction], agent->y + RING_Y[direction],
                     observation[cell]);
    }
    rememberCell(agent, agent->x, agent->y, core->alphabet[0].value[core->alphabet[0].mode]);

    int rewarding = bestRewardingCode(agent);
    double strength = 0;
    if (rewarding >= 0)
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            strength += agent->decode[rewarding][axis] * agent->cleanScent[axis];

    if (agent->trackCount == 0 || agent->lastAction == 0){
        int slot = agent->trackCount % TRACK_LENGTH;
        agent->trackX[slot] = agent->x;
        agent->trackY[slot] = agent->y;
        for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
            agent->trackScent[slot][axis] = agent->cleanScent[axis];
        agent->trackStep[slot] = core->steps;
        agent->trackCount++;
    }

    int visible = 0;
    if (rewarding >= 0)
        for (int cell = 0; cell < RING_CHANNELS; cell++)
            if (observation[cell] == core->alphabet[0].value[rewarding]) visible = 1;

    if (rewarding >= 0 && !visible && !core->alarm
     && agent->centroidUpdates > CENTROIDS_BEFORE_NAVIGATION
     && strength > SOURCE_STRENGTH && decoderIsClean(agent)
     && locateSource(agent, rewarding, &agent->targetX, &agent->targetY)){
        int routed = routeFirstAction(agent, agent->targetX, agent->targetY);
        if (routed >= 0) action = routed;
    }

    if (agent->probeActive) action = 1;
    if (action != core->previousAction){
        core->planLength = 0;
        core->planAt = 0;
    }

    agent->lastAction = action;
    core->previousAction = action;
    for (int axis = 0; axis < SCENT_DIMENSIONS; axis++)
        agent->previousScent[axis] = observation[SCENT_CHANNEL + axis];
    agent->started = 1;
    return action;
}

void *agentCreate(AgentShape shape, uint32_t seed){
    if (shape.observationCount != OBSERVATIONS) abort();
    if (shape.actionCount != ACTIONS) abort();

    ScentAgent *agent = calloc(1, sizeof *agent);
    agent->core.rng = seed;
    agent->core.polarity = 1;
    agent->core.previousAction = -1;
    agent->core.longestGap = OBSERVATIONS * SYMBOLS;
    for (int action = 0; action < ACTIONS; action++){
        agent->core.gateChannel[action] = -1;
        for (int channel = 0; channel < OBSERVATIONS; channel++)
            agent->core.sourceOf[action][channel] = -1;
    }
    agent->lastAction = -1;
    agent->physicsLoss = 1e20;
    buildDefaultKernel(agent);
    return agent;
}

uint32_t agentStep(void *handle, const float *observation, float reward){
    return (uint32_t)scentStep(handle, observation, (double)reward);
}
