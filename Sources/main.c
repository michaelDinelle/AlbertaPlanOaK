#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#include "geometry.h"
#include "lane.h"
#include "../.build/catalog.h"

_Static_assert(OAK_LANES == 1u, "");

typedef struct {
    uint32_t policy, steps, firstSeed, seeds, windows, laneCount, stateWords;
    size_t cacheLine;
    Lane *lanes;
    const EnvSpec **specs;
    uint32_t *extra;
    float *shared, *rewardSum;
    double *laneSeconds;
    _Atomic uint32_t next;
} Run;

static const char *POLICY_NAMES[POLICY_COUNT] = { "oak", "ppo", "random", "oracle", "constant", "stream", "oak-permuted", "oak-rescaled" };

static volatile sig_atomic_t pausing;
static struct timespec pausedAt;
static double pausedSeconds;

static double secondsBetween(const struct timespec *from, const struct timespec *to) {
    return (double)(to->tv_sec - from->tv_sec) + 1e-9 * (double)(to->tv_nsec - from->tv_nsec);
}

static void onPause(int sig) {
    (void)sig;
    clock_gettime(CLOCK_MONOTONIC, &pausedAt);
    pausing = 1;
    raise(SIGSTOP);
}

static void onResume(int sig) {
    (void)sig;
    if (!pausing) return;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    pausedSeconds += secondsBetween(&pausedAt, &now);
    pausing = 0;
}

static void flushToZero(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= 1ull << 24;
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
}

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static uint32_t policyNamed(const char *name) {
    for (uint32_t p = 0u; p < POLICY_COUNT; p++) if (strcmp(POLICY_NAMES[p], name) == 0) return p;
    die("unknown policy");
    return 0u;
}

static void allocateRun(Run *run) {
    run->windows = (run->steps + OAK_WINDOW - 1u) / OAK_WINDOW;
    size_t cacheLineSize = sizeof run->cacheLine;
    if (sysctlbyname("hw.cachelinesize", &run->cacheLine, &cacheLineSize, 0, 0) != 0 ||
        run->cacheLine < sizeof(void *) || (run->cacheLine & (run->cacheLine - 1u)) != 0u) die("cannot determine cache-line alignment");
    uint32_t wordsPerLine = (uint32_t)(run->cacheLine / sizeof(uint32_t));
    run->stateWords = ((policy_stateWords(run->policy) + wordsPerLine - 1u) / wordsPerLine) * wordsPerLine;
    uint64_t extraTotal = 0u;
    for (uint32_t i = 0u; i < run->laneCount; i++) {
        run->lanes[i].extraOffset = (uint32_t)extraTotal;
        extraTotal += run->specs[i] ? env_extraWords(run->lanes[i].env, run->steps) : 0u;
    }
    if (extraTotal > 0xffffffffull) die("fleet exceeds the 32-bit word address space");
    run->extra = calloc((size_t)extraTotal + 1u, sizeof(uint32_t));
    run->shared = calloc((size_t)env_sharedWords() + 1u, sizeof(float));
    run->rewardSum = calloc((size_t)run->windows * run->laneCount, sizeof(float));
    run->laneSeconds = calloc(run->laneCount, sizeof(double));
    if (!run->extra || !run->shared || !run->rewardSum || !run->laneSeconds) die("out of memory");
    env_buildShared(run->shared, run->specs, run->laneCount);
}

static uint32_t *laneState(const Run *run) {
    uint32_t *state;
    if (posix_memalign((void **)&state, run->cacheLine, ((size_t)run->stateWords + 1u) * sizeof(uint32_t)) != 0) die("out of memory");
    return state;
}

static void runLane(Run *run, uint32_t laneIndex, uint32_t *state) {
    uint32_t lane = 0u;
    Lane working = run->lanes[laneIndex];
    Lane *l = &working;
    uint32_t *extra = run->extra + l->extraOffset;
    const float *shared = run->shared + env_sharedOffset(l->env);
    float ctrl[OAK_ENV_ACT_MAX];
    lane_init(l, lane, run->steps, state, extra, shared, run->policy);
    for (uint32_t window = 0u; window < run->windows; window++) {
        uint32_t end = (window + 1u) * OAK_WINDOW < run->steps ? (window + 1u) * OAK_WINDOW : run->steps;
        lane_window(l, lane, state, extra, shared, run->policy, end, ctrl);
        run->rewardSum[(size_t)window * run->laneCount + laneIndex] = l->tally.sum;
    }
    run->lanes[laneIndex] = working;
}

static void *worker(void *arg) {
    Run *run = (Run *)arg;
    uint32_t *state = laneState(run);
    flushToZero();
    for (;;) {
        uint32_t lane = atomic_fetch_add(&run->next, 1u);
        if (lane >= run->laneCount) break;
        struct timespec from, to;
        clock_gettime(CLOCK_MONOTONIC, &from);
        runLane(run, lane, state);
        clock_gettime(CLOCK_MONOTONIC, &to);
        run->laneSeconds[lane] = secondsBetween(&from, &to);
    }
    free(state);
    return 0;
}

static void reportLanes(const Run *run, long cores) {
    double total = 0.0;
    uint32_t slowest = 0u;
    for (uint32_t i = 0u; i < run->laneCount; i++) {
        total += run->laneSeconds[i];
        if (run->laneSeconds[i] > run->laneSeconds[slowest]) slowest = i;
    }
    double balanced = total / (double)cores;
    printf("lane seconds: %.1f in all, %.1f s balanced over %ld cores, slowest %.1f s (%s seed %u)\n", total, balanced, cores,
           run->laneSeconds[slowest], run->specs[slowest]->name, run->lanes[slowest].seed);
    for (uint32_t i = 0u; i < run->laneCount; i++)
        if (run->laneSeconds[i] > balanced) printf("  longer than balanced: %s seed %u, %.1f s\n", run->specs[i]->name, run->lanes[i].seed, run->laneSeconds[i]);
}

static double runWorkers(Run *run, long *cores) {
    *cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (*cores < 1) *cores = 1;
    pthread_t *threads = calloc((size_t)*cores, sizeof(pthread_t));
    signal(SIGTSTP, onPause);
    signal(SIGCONT, onResume);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < *cores; i++) pthread_create(&threads[i], 0, worker, run);
    for (long i = 0; i < *cores; i++) pthread_join(threads[i], 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    free(threads);
    return secondsBetween(&t0, &t1) - pausedSeconds;
}

static void timestamp(char *out, size_t size) {
    time_t now = time(0);
    struct tm utc;
    gmtime_r(&now, &utc);
    strftime(out, size, "%Y-%m-%dT%H:%M:%SZ", &utc);
}

static void writeMeta(const char *root, Run *run, const char *started, const char *finished, double stepsPerSecond) {
    char path[4096];
    snprintf(path, sizeof path, "%s/meta.json", root);
    FILE *f = fopen(path, "w");
    if (!f) die("cannot write meta.json");
    fprintf(f, "{\n \"batch\": %u,\n \"steps\": %u,\n \"window\": %u,\n \"policy\": \"%s\",\n \"startedAt\": \"%s\",\n",
            run->laneCount, run->steps, OAK_WINDOW, POLICY_NAMES[run->policy], started);
    if (finished) fprintf(f, " \"finishedAt\": \"%s\",\n \"stepsPerSecond\": %.3f,\n", finished, stepsPerSecond);
    fprintf(f, " \"seeds\": [");
    for (uint32_t i = 0u; i < run->laneCount; i++) fprintf(f, "%s%u", i ? ", " : "", run->lanes[i].seed);
    fprintf(f, "],\n \"envSchedule\": [\n");
    for (uint32_t i = 0u; i < run->laneCount; i += run->seeds) {
        const EnvSpec *env = run->specs[i];
        fprintf(f, "  {\"envId\": \"%s\", \"seedStart\": %u, \"seedEnd\": %u, \"obs\": %u, \"actions\": %u, \"box\": %u}%s\n",
                env->name, run->lanes[i].seed, run->lanes[i].seed + run->seeds - 1u, env->obs, env->actions, env->box,
                i + run->seeds < run->laneCount ? "," : "");
    }
    fprintf(f, " ]\n}\n");
    fclose(f);
}

static void writeFile(const char *root, const char *name, const void *data, size_t bytes) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", root, name);
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write recording file");
    if (fwrite(data, 1u, bytes, f) != bytes) die("short write");
    fclose(f);
}

static void scheduleLanes(Run *run, const EnvSpec *const *envs, uint32_t envCount) {
    run->laneCount = envCount * run->seeds;
    run->lanes = calloc(run->laneCount, sizeof(Lane));
    run->specs = calloc(run->laneCount, sizeof(const EnvSpec *));
    uint32_t lane = 0u;
    for (uint32_t e = 0u; e < envCount; e++)
        for (uint32_t s = 0u; s < run->seeds; s++, lane++) {
            run->lanes[lane].env = envs[e]->id;
            run->lanes[lane].seed = envs[e]->seedStart + run->firstSeed + s;
            run->specs[lane] = envs[e];
        }
}

static int record(int argc, char **argv, uint32_t firstSeed, uint32_t seeds, int rootArg) {
    if (argc <= rootArg) die("usage: oak record <policy> <steps> <root> [env...] | oak try <policy> <steps> <first seed> <seeds> <root>");
    Run run;
    memset(&run, 0, sizeof run);
    run.firstSeed = firstSeed;
    run.seeds = seeds;
    if (seeds == 0u || firstSeed + seeds > OAK_ENV_SEEDS) die("seeds out of range");
    run.policy = policyNamed(argv[2]);
    long steps = atol(argv[3]);
    if (steps <= 0 || steps > 0xffffffffL) die("steps out of range");
    run.steps = (uint32_t)steps;
    const char *root = argv[rootArg];
    struct stat st;
    if (stat(root, &st) == 0) die("recording already exists");
    env_checkTable();
    const EnvSpec *envs[OAK_ENV_TABLE_MAX];
    uint32_t envCount = 0u;
    if (argc > rootArg + 1) {
        for (int i = rootArg + 1; i < argc; i++) {
            const EnvSpec *env = env_named(argv[i]);
            if (!env) die("unknown environment");
            envs[envCount++] = env;
        }
    } else {
        for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) envs[envCount++] = env_at(i);
    }
    scheduleLanes(&run, envs, envCount);
    allocateRun(&run);
    char started[32], finished[32];
    timestamp(started, sizeof started);
    char series[4096];
    snprintf(series, sizeof series, "%s/series", root);
    if (mkdir(root, 0755) != 0 || mkdir(series, 0755) != 0) die("cannot create recording directory");
    writeMeta(root, &run, started, 0, 0.0);
    long cores = 1;
    double seconds = runWorkers(&run, &cores);
    uint32_t *ends = calloc(run.windows, sizeof(uint32_t));
    uint64_t *hash = calloc(run.laneCount, sizeof(uint64_t));
    uint32_t *health = calloc(run.laneCount, sizeof(uint32_t));
    uint32_t *instruments = calloc((size_t)run.laneCount * 4u, sizeof(uint32_t));
    for (uint32_t w = 0u; w < run.windows; w++) ends[w] = (w + 1u) * OAK_WINDOW < run.steps ? (w + 1u) * OAK_WINDOW : run.steps;
    for (uint32_t i = 0u; i < run.laneCount; i++) {
        const LaneTally *t = &run.lanes[i].tally;
        hash[i] = t->hash;
        health[i] = t->health;
        instruments[4u * i] = t->episodes;
        instruments[4u * i + 1u] = t->positive;
        instruments[4u * i + 2u] = t->negative;
        instruments[4u * i + 3u] = t->firstPositive;
    }
    writeFile(root, "ends.u32", ends, run.windows * sizeof(uint32_t));
    writeFile(root, "series/rewardSum.f32", run.rewardSum, (size_t)run.windows * run.laneCount * sizeof(float));
    writeFile(root, "hash.u64", hash, run.laneCount * sizeof(uint64_t));
    writeFile(root, "health.u32", health, run.laneCount * sizeof(uint32_t));
    writeFile(root, "instruments.u32", instruments, (size_t)run.laneCount * 4u * sizeof(uint32_t));
    timestamp(finished, sizeof finished);
    double stepsPerSecond = (double)run.steps * (double)run.laneCount / seconds;
    writeMeta(root, &run, started, finished, stepsPerSecond);
    printf("%u lanes x %u steps in %.1f s: %.0f steps/s\n", run.laneCount, run.steps, seconds, stepsPerSecond);
    reportLanes(&run, cores);
    return 0;
}

static void catalogEntry(FILE *out, const EnvSpec *env) {
    fprintf(out, "{\"envId\":\"%s\",\"family\":\"%s\",\"category\":\"%s\",\"instance\":%u,\"obs\":%u,\"actions\":%u,\"box\":%u,\"seedStart\":%u,\"seedEnd\":%u,\"words\":%u,\"view\":%u",
            env->name, env->family, env->category, env->instance, env->obs, env->actions, env->box, env->seedStart,
            env->seedStart + OAK_ENV_SEEDS - 1u, env->words, env_view(env->id, 0, 0, 0, 0));
    catalogDetails(out, env);
    fputc('}', out);
}

static const char *metaField(const char *meta, const char *field) {
    const char *found = strstr(meta, field);
    if (!found) die("recording metadata is incomplete");
    return found;
}

static void replayWrite(const void *data, size_t bytes) {
    if (fwrite(data, 1u, bytes, stdout) != bytes) die("replay stream closed");
}

static void replayFrame(const EnvSpec *spec, const Lane *lane, const uint32_t *extra, const float *shared, const float *ctrl) {
    for (uint32_t k = spec->words; k < OAK_ENV_WORDS_MAX; k++)
        if (lane->words[k]) { fprintf(stderr, "%s writes state word %u beyond its declared width of %u\n", spec->name, k, spec->words); exit(1); }
    uint32_t row[OAK_ENV_WORDS_MAX + OAK_ENV_VIEW_MAX + OAK_ENV_OBS_MAX + OAK_ENV_ACT_MAX + 2u];
    uint32_t at = 0u, controls = spec->box ? spec->actions : 1u;
    for (uint32_t k = 0u; k < spec->words; k++) row[at++] = lane->words[k];
    at += env_view(spec->id, lane->words, extra, shared, row + at);
    for (uint32_t k = 0u; k < spec->obs; k++) row[at++] = floatWord(lane->obs[k]);
    for (uint32_t k = 0u; k < controls; k++) row[at++] = floatWord(ctrl[k]);
    row[at++] = floatWord(lane->tally.reward);
    row[at++] = floatWord((float)lane->tally.done);
    replayWrite(row, at * sizeof(uint32_t));
}

static int replay(int argc, char **argv) {
    if (argc != 4) die("usage: oak replay <recording> <lane>");
    const char *root = argv[2];
    char *end;
    long laneIndex = strtol(argv[3], &end, 10);
    if (*end || laneIndex < 0 || laneIndex > 0xffffffffL) die("lane out of range");
    char path[4096];
    snprintf(path, sizeof path, "%s/meta.json", root);
    FILE *f = fopen(path, "r");
    if (!f) die("cannot read meta.json");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0) die("empty metadata");
    rewind(f);
    char *meta = calloc((size_t)size + 1u, 1u);
    if (!meta || fread(meta, 1u, (size_t)size, f) != (size_t)size) die("short metadata read");
    fclose(f);
    char policyName[32];
    Run run = {0};
    if (sscanf(metaField(meta, "\"steps\""), "\"steps\": %u", &run.steps) != 1 || !run.steps || run.steps == 0xffffffffu) die("invalid horizon");
    if (sscanf(metaField(meta, "\"policy\""), "\"policy\": \"%31[^\"]\"", policyName) != 1) die("invalid policy");
    metaField(meta, "\"finishedAt\"");
    run.policy = policyNamed(policyName);
    const EnvSpec *spec = 0;
    uint32_t seed = 0u, base = 0u;
    for (const char *cursor = metaField(meta, "\"envSchedule\""); (cursor = strstr(cursor, "\"envId\"")) != 0; cursor += 7) {
        char name[64];
        uint32_t first, last;
        if (sscanf(cursor, "\"envId\": \"%63[^\"]\", \"seedStart\": %u, \"seedEnd\": %u", name, &first, &last) != 3 || last < first) die("bad envSchedule entry");
        uint32_t count = last - first + 1u;
        if ((uint32_t)laneIndex >= base && (uint32_t)laneIndex - base < count) {
            spec = env_named(name);
            seed = first + (uint32_t)laneIndex - base;
            break;
        }
        base += count;
    }
    if (!spec) die("lane environment is not in the table");
    free(meta);
    snprintf(path, sizeof path, "%s/hash.u64", root);
    FILE *h = fopen(path, "rb");
    uint64_t recorded;
    if (!h || fseek(h, laneIndex * 8, SEEK_SET) || fread(&recorded, 8u, 1u, h) != 1u) die("missing recorded reward hash");
    fclose(h);
    Lane lane = { .env = spec->id, .seed = seed };
    run.laneCount = 1u;
    run.lanes = &lane;
    run.specs = &spec;
    allocateRun(&run);
    uint32_t *state = laneState(&run);
    flushToZero();
    const float *shared = run.shared + env_sharedOffset(spec->id);
    lane_init(&lane, 0u, run.steps, state, run.extra, shared, run.policy);
    char *descriptor = 0;
    size_t descriptorSize = 0u;
    FILE *header = open_memstream(&descriptor, &descriptorSize);
    if (!header) die("cannot allocate replay descriptor");
    fputs("{\"environment\":", header);
    catalogEntry(header, spec);
    fprintf(header, ",\"seed\":%u,\"horizon\":%u,\"policy\":\"%s\",\"lane\":%ld,\"controls\":%u",
            seed, run.steps, policyName, laneIndex, spec->box ? spec->actions : 1u);
    fputc('}', header);
    fclose(header);
    uint32_t prefix[8] = {0};
    memcpy(prefix, "OAKTAPE", 8);
    prefix[2] = (uint32_t)descriptorSize;
    prefix[3] = spec->words + env_view(spec->id, 0, 0, 0, 0) + spec->obs + (spec->box ? spec->actions : 1u) + 2u;
    prefix[4] = run.steps + 1u;
    prefix[5] = 256u;
    replayWrite(prefix, sizeof prefix);
    replayWrite(descriptor, descriptorSize);
    uint32_t padding = 0u;
    replayWrite(&padding, (4u - descriptorSize % 4u) % 4u);
    free(descriptor);
    float ctrl[OAK_ENV_ACT_MAX] = {0};
    replayFrame(spec, &lane, run.extra, shared, ctrl);
    if (fflush(stdout)) die("replay stream closed");
    while (lane.tally.step < run.steps) {
        lane_window(&lane, 0u, state, run.extra, shared, run.policy, lane.tally.step + 1u, ctrl);
        replayFrame(spec, &lane, run.extra, shared, ctrl);
        if ((lane.tally.step == 1u || lane.tally.step % prefix[5] == 0u) && fflush(stdout)) die("replay stream closed");
    }
    replayWrite("OAKHASH", 8u);
    replayWrite(&lane.tally.hash, 8u);
    replayWrite(&recorded, 8u);
    replayWrite(&prefix[4], 4u);
    replayWrite(&lane.tally.health, 4u);
    if (fflush(stdout)) die("replay stream closed");
    return lane.tally.hash == recorded ? 0 : 2;
}

static void checkRun(Run *run, uint32_t policy, uint32_t steps, const EnvSpec *const *envs, uint32_t envCount) {
    memset(run, 0, sizeof *run);
    run->policy = policy;
    run->steps = steps;
    run->seeds = 1u;
    scheduleLanes(run, envs, envCount);
    allocateRun(run);
    long cores = 1;
    runWorkers(run, &cores);
}

static int envcheck(int argc, char **argv) {
    long steps = argc > 2 ? atol(argv[2]) : 200000L;
    if (steps <= 0 || steps > 0xffffffffL) die("steps out of range");
    env_checkTable();
    const EnvSpec *envs[OAK_ENV_TABLE_MAX];
    for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) envs[i] = env_at(i);
    static const uint32_t CHECKED[4] = { POLICY_RANDOM, POLICY_CONSTANT, POLICY_ORACLE, POLICY_ORACLE };
    Run runs[4];
    for (uint32_t p = 0u; p < 4u; p++) checkRun(&runs[p], CHECKED[p], (uint32_t)steps, envs, OAK_ENV_TABLE_COUNT);
    for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) {
        printf("{\"env\":\"%s\",\"steps\":%ld", envs[i]->name, steps);
        for (uint32_t p = 0u; p < 3u; p++) {
            const LaneTally *t = &runs[p].lanes[i].tally;
            printf(",\"%s\":{\"reward\":%.9g,\"episodes\":%u,\"positive\":%u,\"negative\":%u,\"firstPositive\":%u,\"healthy\":%u}", POLICY_NAMES[CHECKED[p]],
                   (double)t->sum / (double)steps, t->episodes, t->positive, t->negative, t->firstPositive, t->health == OAK_HEALTH_ALIVE);
        }
        printf(",\"deterministic\":%u}\n", runs[2].lanes[i].tally.hash == runs[3].lanes[i].tally.hash);
    }
    return 0;
}

#define FUZZ_STEPS 4096u
#define FUZZ_SCENARIOS 7u
static const char *FUZZ_NAMES[FUZZ_SCENARIOS] = { "normal", "nonfinite", "huge", "denormal", "offset", "constant", "reward-huge" };

static float fuzzValue(uint32_t scenario, uint32_t step, uint32_t index) {
    u4 r = philox4x32(9173u, 0x2F22u, step, index, scenario, 0u);
    float normal = normalFrom(r.x, r.y);
    uint32_t pick = r.z % 40u;
    switch (scenario) {
        case 1u: return pick == 0u ? NAN : pick == 1u ? INFINITY : pick == 2u ? -INFINITY : normal;
        case 2u: return normal * 1e20f;
        case 3u: return wordFloat((r.x & 0x807FFFFFu) | 1u);
        case 4u: return 1e6f + normal;
        case 5u: return (index & 1u) ? 3.0f : -7.0f;
        default: return normal;
    }
}

static float fuzzReward(uint32_t scenario, uint32_t step) {
    u4 r = philox4x32(9173u, 0x2F23u, step, scenario, 0u, 0u);
    float normal = normalFrom(r.x, r.y);
    if (scenario == 1u) return r.z % 64u == 0u ? NAN : normal;
    if (scenario == 6u) return normal * 1e30f;
    return normal;
}

static int fuzz(void) {
    env_checkTable();
    flushToZero();
    uint32_t *state;
    if (posix_memalign((void **)&state, 128u, (size_t)oak_stateWords() * sizeof(uint32_t)) != 0) die("out of memory");
    uint32_t seen[OAK_ENV_TABLE_MAX][3], kinds = 0u;
    for (uint32_t e = 0u; e < OAK_ENV_TABLE_COUNT; e++) {
        const EnvSpec *env = env_at(e);
        uint32_t known = 0u;
        for (uint32_t k = 0u; k < kinds; k++) known |= seen[k][0] == env->obs && seen[k][1] == env->actions && seen[k][2] == env->box;
        if (known) continue;
        seen[kinds][0] = env->obs;
        seen[kinds][1] = env->actions;
        seen[kinds][2] = env->box;
        kinds++;
        for (uint32_t scenario = 0u; scenario < FUZZ_SCENARIOS; scenario++) {
            _Alignas(16) float obs[OAK_ENV_OBS_MAX] = { 0.0f };
            oak_init(state, 0u, env->id, 7919u, obs);
            uint32_t invalid = 0u;
            for (uint32_t step = 0u; step < FUZZ_STEPS; step++) {
                for (uint32_t o = 0u; o < env->obs; o++) obs[o] = fuzzValue(scenario, step, o);
                float ctrl[OAK_ENV_ACT_MAX] = { 0.0f };
                oak_act(state, 0u, env->id, 7919u, step, obs, step ? fuzzReward(scenario, step) : 0.0f, 0u, ctrl);
                if (env->box) {
                    for (uint32_t d = 0u; d < env->actions; d++) invalid += !(fabsf(ctrl[d]) <= 1.0f);
                } else {
                    invalid += !(ctrl[0] >= 0.0f && ctrl[0] < (float)env->actions && ctrl[0] == floorf(ctrl[0]));
                }
            }
            printf("{\"env\":\"%s\",\"obs\":%u,\"actions\":%u,\"box\":%u,\"scenario\":\"%s\",\"nonfinite\":%u,\"invalidControls\":%u}\n", env->name, env->obs,
                   env->actions, env->box, FUZZ_NAMES[scenario], oak_nonfinite(state), invalid);
        }
    }
    free(state);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "record") == 0) return record(argc, argv, 0u, OAK_ENV_SEEDS, 4);
    if (argc == 7 && strcmp(argv[1], "try") == 0) return record(argc, argv, (uint32_t)atol(argv[4]), (uint32_t)atol(argv[5]), 6);
    if (argc >= 2 && strcmp(argv[1], "replay") == 0) return replay(argc, argv);
    if (argc >= 2 && argc <= 3 && strcmp(argv[1], "envcheck") == 0) return envcheck(argc, argv);
    if (argc == 2 && strcmp(argv[1], "fuzz") == 0) return fuzz();
    if (argc == 2 && strcmp(argv[1], "envs") == 0) {
        env_checkTable();
        fputc('[', stdout);
        for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) {
            if (i) fputc(',', stdout);
            catalogEntry(stdout, env_at(i));
        }
        puts("]");
        return 0;
    }
    die("usage: oak record <policy> <steps> <root> [env...] | oak try <policy> <steps> <first seed> <seeds> <root> | oak replay <recording> <lane> | oak envs | oak envcheck [steps] | oak fuzz");
    return 1;
}
