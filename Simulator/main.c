#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include "jbw.h"
#include "agent.h"

#ifndef STEPS
#define STEPS 1000000
#endif

#define SEEDS 8
#define FIRST_SEED 101u
#define RUNGS 5

#define RING_CELLS 8
#ifdef WITH_SCENT
#define SCENT_CHANNELS 3
#else
#define SCENT_CHANNELS 0
#endif
#define OBSERVATION_COUNT (RING_CELLS + SCENT_CHANNELS + 1)
#define ACTION_COUNT 3u
#define ACTION_MOVE 0u
#define ACTION_TURN_LEFT 1u
#define ACTION_TURN_RIGHT 2u

#define HEADINGS 4
#define REWARD_POLARITY_PERIOD 100000L
#define REWARDING_CELL_DENSITY 0.0077064

static const int FORWARD[HEADINGS][2] = {
    { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 }
};

static const int NEIGHBOUR[RING_CELLS][2] = {
    { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 },
    { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }
};

static const long RUNG[RUNGS] = {
    1000000L, 4000000L, 16000000L, 64000000L, 256000000L
};

typedef struct {
    double rate[RUNGS], coverage[RUNGS];
    long drought[RUNGS];
    long positive, negative, chose[ACTION_COUNT], moves;
} Result;

static uint32_t *allocateDestructionTable(uint32_t *mask){
    size_t needed = (size_t)STEPS * JBW_BLOCK_ITEMS * 2u
                  / ((size_t)JBW_BLOCK * JBW_BLOCK) + 2u;
    size_t slots = 2u;
    while (slots < needed) slots <<= 1;
    *mask = (uint32_t)(slots - 1u);
    return calloc(slots, 3u * sizeof(uint32_t));
}

static void runSeed(uint32_t seed, Result *out){
#ifdef WITH_SCENT
    float *scentResponse = malloc(sizeof(float) * (size_t)JBW_SCENT_CELLS * JBW_DESTRUCTION_LIFETIME);
    jbw_buildScentResponse(scentResponse);
#endif
    uint32_t mask;
    uint32_t *destroyed = allocateDestructionTable(&mask);
    void *agent = agentCreate((AgentShape){ OBSERVATION_COUNT, ACTION_COUNT }, seed);

    float buffer[2][OBSERVATION_COUNT];
    int x = 0;
    int y = 0;
    int heading = 0;
    int moved = 0;
    int rung = 0;
    int spanLowX = 0, spanHighX = 0, spanLowY = 0, spanHighY = 0;
    long drought = 0, windowDrought = 0, windowEvents = 0;
    float reward = 0.0f;
    double total = 0.0, windowStart = 0.0;
    long windowFirstStep = 0;

    for (long step = 0; step < STEPS; ++step){
        uint32_t aheadItem = jbw_itemAt(seed, x + FORWARD[heading][0], y + FORWARD[heading][1]);
        uint32_t aheadDestroyed = jbw_collectible(aheadItem)
                                ? jbw_destroyedAt(destroyed, mask,
                                                  x + FORWARD[heading][0], y + FORWARD[heading][1])
                                : 0u;

        float *observation = buffer[step & 1];
        for (int cell = 0; cell < RING_CELLS; ++cell){
            int direction = (2 * heading + cell) & (RING_CELLS - 1);
            int cellX = x + NEIGHBOUR[direction][0];
            int cellY = y + NEIGHBOUR[direction][1];
            uint32_t item = jbw_itemAt(seed, cellX, cellY);
            if (jbw_collectible(item) && jbw_destroyedAt(destroyed, mask, cellX, cellY) != 0u)
                item = JBW_EMPTY;
            observation[cell] = JBW_ENERGY[item];
        }
#ifdef WITH_SCENT
        jbw_scentAt(seed, x, y, (uint32_t)step, destroyed, mask, scentResponse,
                    observation + RING_CELLS);
#endif
        observation[OBSERVATION_COUNT - 1] = (float)moved;

        uint32_t action = agentStep(agent, observation, reward);

        reward = 0.0f;
        moved = 0;

        if (action == ACTION_TURN_LEFT){
            heading = (heading + HEADINGS - 1) & (HEADINGS - 1);
        } else if (action == ACTION_TURN_RIGHT){
            heading = (heading + 1) & (HEADINGS - 1);
        } else if (!jbw_blocks(aheadItem)){
            x += FORWARD[heading][0];
            y += FORWARD[heading][1];
            moved = 1;
            if (x < spanLowX) spanLowX = x;
            if (x > spanHighX) spanHighX = x;
            if (y < spanLowY) spanLowY = y;
            if (y > spanHighY) spanHighY = y;

            if (jbw_collectible(aheadItem) && aheadDestroyed == 0u){
                jbw_destroy(destroyed, mask, x, y, (uint32_t)step);
                float polarity = ((step / REWARD_POLARITY_PERIOD) & 1) == 0 ? 1.0f : -1.0f;
                if (aheadItem == JBW_JELLY) reward = polarity;
                else if (aheadItem == JBW_ONION) reward = -polarity;
            }
        }

        out->chose[action]++;
        out->moves += moved;
        total += reward;
        if (reward != 0.0f){
            windowEvents++;
            if (drought > windowDrought) windowDrought = drought;
            drought = 0;
        } else {
            drought++;
        }
        if (reward > 0.0f) out->positive++;
        else if (reward < 0.0f) out->negative++;

        if (rung < RUNGS && step + 1 == RUNG[rung]){
            double span = (double)(spanHighX - spanLowX + 1) * (double)(spanHighY - spanLowY + 1);
            out->rate[rung] = (total - windowStart) / (double)(RUNG[rung] - windowFirstStep);
            out->coverage[rung] = (double)windowEvents / (span * REWARDING_CELL_DENSITY);
            out->drought[rung] = windowDrought > drought ? windowDrought : drought;
            windowStart = total;
            windowFirstStep = RUNG[rung];
            windowDrought = 0;
            windowEvents = 0;
            rung++;
        }
    }
}

int main(void){
    struct timespec start, finish;
    Result *results = mmap(NULL, SEEDS * sizeof(Result), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANON, -1, 0);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int lane = 0; lane < SEEDS; ++lane)
        if (fork() == 0){
            runSeed(FIRST_SEED + (uint32_t)lane, &results[lane]);
            _exit(0);
        }
    while (wait(NULL) > 0){ }
    clock_gettime(CLOCK_MONOTONIC, &finish);

    for (int level = 0; level < RUNGS && RUNG[level] <= (long)STEPS; ++level){
        double sum = 0.0, coverage = 0.0;
        double worst = results[0].rate[level];
        long drought = 0;
        for (int lane = 0; lane < SEEDS; ++lane){
            sum += results[lane].rate[level];
            coverage += results[lane].coverage[level];
            if (results[lane].rate[level] < worst) worst = results[lane].rate[level];
            if (results[lane].drought[level] > drought) drought = results[lane].drought[level];
        }
        printf("  %4ldM  mean %+.9f  worst %+.9f   harvest %.3f   drought %ld\n",
               RUNG[level] / 1000000L, sum / SEEDS, worst, coverage / SEEDS, drought);
    }

    long positive = 0, negative = 0, moves = 0, chose[ACTION_COUNT] = { 0 };
    for (int lane = 0; lane < SEEDS; ++lane){
        positive += results[lane].positive;
        negative += results[lane].negative;
        moves += results[lane].moves;
        for (uint32_t action = 0u; action < ACTION_COUNT; ++action)
            chose[action] += results[lane].chose[action];
    }

    double elapsed = (finish.tv_sec - start.tv_sec)
                   + 1e-9 * (finish.tv_nsec - start.tv_nsec);
    double lifetime = (double)STEPS * SEEDS;
    double correct = (positive + negative) ? 100.0 * positive / (positive + negative) : 0.0;

    double entropy = 0.0;
    for (uint32_t action = 0u; action < ACTION_COUNT; ++action){
        double share = (double)chose[action] / lifetime;
        if (share > 0.0) entropy -= share * log(share) / log((double)ACTION_COUNT);
    }

    printf("  reward events %ld positive / %ld negative (%.1f%% positive)   moves/step %.4f\n",
           positive, negative, correct, (double)moves / lifetime);
    printf("  actions");
    for (uint32_t action = 0u; action < ACTION_COUNT; ++action)
        printf(" %.4f", (double)chose[action] / lifetime);
    printf("   entropy %.4f   %.2fM SPS/lane   %.1fM SPS\n",
           entropy, STEPS / elapsed / 1e6, lifetime / elapsed / 1e6);
    return 0;
}
