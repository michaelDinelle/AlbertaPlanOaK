#include <pthread.h>
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

typedef struct {
    uint32_t policy, steps, windows, laneCount, stateWords, width;
    Lane *lanes;
    const EnvSpec **specs;
    uint32_t *state, *extra;
    float *shared, *rewardSum;
    _Atomic uint32_t next;
} Run;

static const char *POLICY_NAMES[POLICY_COUNT] = { "oak", "ppo", "random", "oracle" };

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

static void allocateRun(Run *run, uint32_t width) {
    run->windows = (run->steps + OAK_WINDOW - 1u) / OAK_WINDOW;
    size_t cacheLine = 0u, cacheLineSize = sizeof cacheLine;
    if (sysctlbyname("hw.cachelinesize", &cacheLine, &cacheLineSize, 0, 0) != 0 ||
        cacheLine < sizeof(void *) || (cacheLine & (cacheLine - 1u)) != 0u) die("cannot determine cache-line alignment");
    uint32_t wordsPerLine = (uint32_t)(cacheLine / sizeof(uint32_t));
    run->stateWords = ((policy_stateWords(run->policy) + wordsPerLine - 1u) / wordsPerLine) * wordsPerLine;
    run->width = width;
    uint64_t extraTotal = 0u;
    for (uint32_t i = 0u; i < run->laneCount; i++) {
        run->lanes[i].extraOffset = (uint32_t)extraTotal;
        extraTotal += run->specs[i] ? env_extraWords(run->lanes[i].env, run->steps) : 0u;
    }
    uint64_t groups = (run->laneCount + width - 1u) / width;
    if (groups * width * run->stateWords > 0xffffffffull || extraTotal > 0xffffffffull) die("fleet exceeds the 32-bit word address space");
    if (posix_memalign((void **)&run->state, cacheLine,
                      (size_t)(groups * width * run->stateWords + 1u) * sizeof(uint32_t)) != 0) die("out of memory");
    run->extra = calloc((size_t)extraTotal + 1u, sizeof(uint32_t));
    run->shared = calloc(jbw_sharedWords(), sizeof(float));
    run->rewardSum = calloc((size_t)run->windows * run->laneCount, sizeof(float));
    if (!run->state || !run->extra || !run->shared || !run->rewardSum) die("out of memory");
    jbw_buildShared(run->shared);
}

static void runLane(Run *run, uint32_t laneIndex, FILE *replay) {
    const EnvSpec *spec = run->specs[laneIndex];
    uint32_t lane = laneIndex % OAK_LANES;
    uint32_t *state = run->state + (laneIndex / OAK_LANES) * OAK_LANES * run->stateWords;
    Lane working = run->lanes[laneIndex];
    Lane *l = &working;
    uint32_t *extra = run->extra + l->extraOffset;
    lane_init(l, lane, run->steps, state, extra, run->shared, run->policy);
    float ctrl[OAK_ENV_ACT_MAX];
    for (uint32_t window = 0u; window < run->windows; window++) {
        uint32_t end = (window + 1u) * OAK_WINDOW < run->steps ? (window + 1u) * OAK_WINDOW : run->steps;
        if (replay) {
            while (l->step < end) {
                lane_step(l, lane, state, extra, run->shared, run->policy, ctrl);
                float row[OAK_POSE_LEN + OAK_ENV_OBS_MAX + 2u];
                env_pose(l->env, l->words, ctrl, row);
                for (uint32_t k = 0u; k < spec->obs; k++) row[spec->poseLen + k] = l->obs[k];
                row[spec->poseLen + spec->obs] = l->reward;
                row[spec->poseLen + spec->obs + 1u] = (float)l->done;
                fwrite(row, sizeof(float), spec->poseLen + spec->obs + 2u, replay);
            }
        } else {
            lane_window(l, lane, state, extra, run->shared, run->policy, end);
        }
        run->rewardSum[(size_t)window * run->laneCount + laneIndex] = l->sum;
    }
    run->lanes[laneIndex] = working;
}

static void *worker(void *arg) {
    Run *run = (Run *)arg;
    flushToZero();
    for (;;) {
        uint32_t lane = atomic_fetch_add(&run->next, 1u);
        if (lane >= run->laneCount) return 0;
        runLane(run, lane, 0);
    }
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
    fprintf(f, "{\n \"batch\": %u,\n \"steps\": %u,\n \"window\": %u,\n \"policy\": \"%s\",\n \"preset\": \"%s\",\n \"startedAt\": \"%s\",\n",
            run->laneCount, run->steps, OAK_WINDOW, POLICY_NAMES[run->policy], OAK_PRESET_NAME, started);
    if (finished) fprintf(f, " \"finishedAt\": \"%s\",\n \"stepsPerSecond\": %.3f,\n", finished, stepsPerSecond);
    fprintf(f, " \"seeds\": [");
    for (uint32_t i = 0u; i < run->laneCount; i++) fprintf(f, "%s%u", i ? ", " : "", run->lanes[i].seed);
    fprintf(f, "],\n \"envSchedule\": [\n");
    for (uint32_t i = 0u; i < run->laneCount; i += run->specs[i]->seeds) {
        const EnvSpec *env = run->specs[i];
        fprintf(f, "  {\"envId\": \"%s\", \"seedStart\": %u, \"seedEnd\": %u, \"obs\": %u, \"actions\": %u, \"box\": %u, \"pose\": %u}%s\n",
                env->name, run->lanes[i].seed, run->lanes[i].seed + env->seeds - 1u, env->obs, env->actions, env->box, env->poseLen,
                i + env->seeds < run->laneCount ? "," : "");
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
    run->laneCount = 0u;
    for (uint32_t e = 0u; e < envCount; e++) run->laneCount += envs[e]->seeds;
    run->lanes = calloc(run->laneCount, sizeof(Lane));
    run->specs = calloc(run->laneCount, sizeof(const EnvSpec *));
    uint32_t lane = 0u;
    for (uint32_t e = 0u; e < envCount; e++)
        for (uint32_t s = 0u; s < envs[e]->seeds; s++, lane++) {
            run->lanes[lane].env = envs[e]->id;
            run->lanes[lane].seed = envs[e]->seedStart + s;
            run->specs[lane] = envs[e];
        }
}

static int record(int argc, char **argv) {
    if (argc < 5) die("usage: oak record oak|ppo|random|oracle <steps> <root> [env...]");
    Run run;
    memset(&run, 0, sizeof run);
    run.policy = policyNamed(argv[2]);
    long steps = atol(argv[3]);
    if (steps <= 0 || steps > 0xffffffffL) die("steps out of range");
    run.steps = (uint32_t)steps;
    const char *root = argv[4];
    struct stat st;
    if (stat(root, &st) == 0) die("recording already exists");
    env_checkTable();
    const EnvSpec *envs[OAK_ENV_TABLE_MAX];
    uint32_t envCount = 0u;
    if (argc > 5) {
        for (int i = 5; i < argc; i++) {
            const EnvSpec *env = env_named(argv[i]);
            if (!env) die("unknown environment");
            envs[envCount++] = env;
        }
    } else {
        for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) envs[envCount++] = env_at(i);
    }
    scheduleLanes(&run, envs, envCount);
    allocateRun(&run, OAK_LANES);
    char started[32], finished[32];
    timestamp(started, sizeof started);
    char series[4096];
    snprintf(series, sizeof series, "%s/series", root);
    if (mkdir(root, 0755) != 0 || mkdir(series, 0755) != 0) die("cannot create recording directory");
    writeMeta(root, &run, started, 0, 0.0);
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) cores = 1;
    pthread_t *threads = calloc((size_t)cores, sizeof(pthread_t));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < cores; i++) pthread_create(&threads[i], 0, worker, &run);
    for (long i = 0; i < cores; i++) pthread_join(threads[i], 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double seconds = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    uint32_t *ends = calloc(run.windows, sizeof(uint32_t));
    uint64_t *hash = calloc(run.laneCount, sizeof(uint64_t));
    uint32_t *health = calloc(run.laneCount, sizeof(uint32_t));
    for (uint32_t w = 0u; w < run.windows; w++) ends[w] = (w + 1u) * OAK_WINDOW < run.steps ? (w + 1u) * OAK_WINDOW : run.steps;
    for (uint32_t i = 0u; i < run.laneCount; i++) { hash[i] = run.lanes[i].hash; health[i] = run.lanes[i].health; }
    writeFile(root, "ends.u32", ends, run.windows * sizeof(uint32_t));
    writeFile(root, "series/rewardSum.f32", run.rewardSum, (size_t)run.windows * run.laneCount * sizeof(float));
    writeFile(root, "hash.u64", hash, run.laneCount * sizeof(uint64_t));
    writeFile(root, "health.u32", health, run.laneCount * sizeof(uint32_t));
    timestamp(finished, sizeof finished);
    double stepsPerSecond = (double)run.steps * (double)run.laneCount / seconds;
    writeMeta(root, &run, started, finished, stepsPerSecond);
    printf("%u lanes x %u steps in %.1f s: %.0f steps/s\n", run.laneCount, run.steps, seconds, stepsPerSecond);
    return 0;
}

static int replay(int argc, char **argv) {
    if (argc != 4) die("usage: oak replay <recording> <lane>");
    const char *root = argv[2];
    long laneIndex = atol(argv[3]);
    char path[4096];
    snprintf(path, sizeof path, "%s/meta.json", root);
    FILE *f = fopen(path, "r");
    if (!f) die("cannot read meta.json");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *meta = calloc((size_t)size + 1u, 1u);
    if (fread(meta, 1u, (size_t)size, f) != (size_t)size) die("short read");
    fclose(f);
    char policyName[32];
    unsigned steps;
    if (sscanf(strstr(meta, "\"steps\""), "\"steps\": %u", &steps) != 1) die("meta.json lacks steps");
    if (sscanf(strstr(meta, "\"policy\""), "\"policy\": \"%31[^\"]\"", policyName) != 1) die("meta.json lacks policy");
    Run run;
    memset(&run, 0, sizeof run);
    run.policy = policyNamed(policyName);
    run.steps = steps;
    const EnvSpec *envs[OAK_ENV_TABLE_MAX];
    uint32_t seedStarts[OAK_ENV_TABLE_MAX], seedEnds[OAK_ENV_TABLE_MAX];
    uint32_t envCount = 0u;
    for (char *cursor = strstr(meta, "\"envSchedule\""); (cursor = strstr(cursor, "\"envId\"")) != 0;) {
        char name[64];
        if (sscanf(cursor, "\"envId\": \"%63[^\"]\", \"seedStart\": %u, \"seedEnd\": %u", name, &seedStarts[envCount], &seedEnds[envCount]) != 3) die("bad envSchedule entry");
        run.laneCount += seedEnds[envCount] - seedStarts[envCount] + 1u;
        envs[envCount++] = env_named(name);
        cursor += 7;
    }
    run.lanes = calloc(run.laneCount, sizeof(Lane));
    run.specs = calloc(run.laneCount, sizeof(const EnvSpec *));
    for (uint32_t e = 0u, lane = 0u; e < envCount; e++)
        for (uint32_t seed = seedStarts[e]; seed <= seedEnds[e]; seed++, lane++) {
            run.lanes[lane].env = envs[e] ? envs[e]->id : 0u;
            run.lanes[lane].seed = seed;
            run.specs[lane] = envs[e];
        }
    if (laneIndex < 0 || (uint32_t)laneIndex >= run.laneCount) die("lane out of range");
    if (!run.specs[laneIndex]) die("this lane's environment no longer exists");
    env_checkTable();
    allocateRun(&run, OAK_LANES);
    snprintf(path, sizeof path, "%s/replay", root);
    mkdir(path, 0755);
    snprintf(path, sizeof path, "%s/replay/%ld.f32", root, laneIndex);
    FILE *out = fopen(path, "wb");
    if (!out) die("cannot write replay");
    flushToZero();
    runLane(&run, (uint32_t)laneIndex, out);
    fclose(out);
    snprintf(path, sizeof path, "%s/hash.u64", root);
    FILE *h = fopen(path, "rb");
    uint64_t recorded = 0u;
    if (h) {
        fseek(h, (long)laneIndex * 8, SEEK_SET);
        if (fread(&recorded, 8u, 1u, h) != 1u) recorded = 0u;
        fclose(h);
    }
    uint64_t computed = run.lanes[laneIndex].hash;
    printf("%s lane %ld seed %u: hash %016llx recorded %016llx %s\n", run.specs[laneIndex]->name, laneIndex,
           run.lanes[laneIndex].seed, (unsigned long long)computed, (unsigned long long)recorded, computed == recorded ? "identical" : "different");
    return computed == recorded ? 0 : 2;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "record") == 0) return record(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "replay") == 0) return replay(argc, argv);
    if (argc == 2 && strcmp(argv[1], "envs") == 0) {
        for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) {
            const EnvSpec *env = env_at(i);
            printf("%s %u %u %u %u %u %u\n", env->name, env->obs, env->actions, env->box, env->seedStart, env->seeds, env->poseLen);
        }
        return 0;
    }
    die("usage: oak record oak|ppo|random|oracle <steps> <root> [env...] | oak replay <recording> <lane> | oak envs");
    return 1;
}
