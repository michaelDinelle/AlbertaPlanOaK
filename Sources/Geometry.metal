#pragma METAL fp math_mode(safe)

constant uint OAK_SIMD_WIDTH = 32u;
constant uint OAK_THREADGROUP_LANES = 4u;
static_assert(OAK_THREADGROUP_LANES > 0u && OAK_THREADGROUP_LANES <= OAK_SIMD_WIDTH
              && (OAK_THREADGROUP_LANES & (OAK_THREADGROUP_LANES - 1u)) == 0u,
              "packed environment steps must fit one wave");

constant uint OAK_CANDIDATES_PER_SLOT = 7u;
constant uint OAK_CONSTRUCTED_PER_PERCEPT = 2u;
#if defined(OAK_PRESET_BASE)
constant uint OAK_OPTION_COUNT = 32u;
constant uint OAK_PERCEPT_COUNT = 32u;
kernel void oak_preset_base() {}
#elif defined(OAK_PRESET_PRO)
constant uint OAK_OPTION_COUNT = 64u;
constant uint OAK_PERCEPT_COUNT = 64u;
kernel void oak_preset_pro() {}
#elif defined(OAK_PRESET_MAX)
constant uint OAK_OPTION_COUNT = 128u;
constant uint OAK_PERCEPT_COUNT = 128u;
kernel void oak_preset_max() {}
#elif defined(OAK_PRESET_ULTRA)
constant uint OAK_OPTION_COUNT = 160u;
constant uint OAK_PERCEPT_COUNT = 160u;
kernel void oak_preset_ultra() {}
#else
#error "compile with exactly one geometry preset: -DOAK_PRESET_BASE, -DOAK_PRESET_PRO, -DOAK_PRESET_MAX or -DOAK_PRESET_ULTRA"
#endif
constant uint OAK_OPTIONS_PER_THREAD = OAK_OPTION_COUNT / OAK_SIMD_WIDTH;
constant uint OAK_PLAN_PERIOD = 32u;

inline uint featureCountForPercepts(uint percepts) { return percepts * (1u + OAK_CONSTRUCTED_PER_PERCEPT * OAK_CANDIDATES_PER_SLOT); }
inline uint activeFeaturesForPercepts(uint percepts) { return percepts * (1u + OAK_CONSTRUCTED_PER_PERCEPT); }
inline uint nurseryForPercepts(uint percepts) { return featureCountForPercepts(percepts); }
inline uint featureTimeScaleForPercepts(uint percepts) { return percepts * (1u + OAK_CANDIDATES_PER_SLOT); }

constant uint OAK_FEATURE_COUNT = OAK_PERCEPT_COUNT * (1u + OAK_CONSTRUCTED_PER_PERCEPT * OAK_CANDIDATES_PER_SLOT);
constant uint OAK_ACTIVE_FEATURES_PER_STEP = OAK_PERCEPT_COUNT * (1u + OAK_CONSTRUCTED_PER_PERCEPT);
constant uint OAK_CONSTRUCTED_FEATURES_PER_STEP = OAK_ACTIVE_FEATURES_PER_STEP - OAK_PERCEPT_COUNT;
constant uint OAK_NURSERY_COUNT = OAK_FEATURE_COUNT;
constant uint OAK_FEATURE_TIME_SCALE = OAK_PERCEPT_COUNT * (1u + OAK_CANDIDATES_PER_SLOT);
constant uint OAK_SEARCH_POPULATION = OAK_OPTION_COUNT - 1u;
inline uint staleOptionSteps(uint featureCount, uint options) { return featureCount * options; }
constant uint OAK_STALE_OPTION_STEPS = OAK_FEATURE_COUNT * OAK_OPTION_COUNT;

static_assert(OAK_FEATURE_COUNT == OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_FEATURES_PER_STEP * OAK_CANDIDATES_PER_SLOT,
              "every constructed slot draws from exactly one candidate pool");
static_assert(OAK_OPTIONS_PER_THREAD * OAK_SIMD_WIDTH == OAK_OPTION_COUNT,
              "every wave thread owns the same number of options");

inline uint perceptsForStream(uint obsDim, uint actionParams) {
    uint sources = ((obsDim + actionParams + 2u + OAK_SIMD_WIDTH - 1u) / OAK_SIMD_WIDTH) * OAK_SIMD_WIDTH;
    return max(sources, OAK_PERCEPT_COUNT);
}

constant uint OAK_PERCEPT_SOURCES_MAX = OAK_ENV_OBS_MAX + 2u * OAK_ENV_ACT_MAX + 2u;
constant uint OAK_PERCEPT_SOURCES_TILED =
    ((OAK_PERCEPT_SOURCES_MAX + OAK_SIMD_WIDTH - 1u) / OAK_SIMD_WIDTH) * OAK_SIMD_WIDTH;
constant uint OAK_PERCEPTS_MAX =
    (OAK_PERCEPT_SOURCES_TILED > OAK_PERCEPT_COUNT) ? OAK_PERCEPT_SOURCES_TILED : OAK_PERCEPT_COUNT;
constant uint OAK_ACTIVE_SLOTS_MAX =
    OAK_PERCEPTS_MAX * (1u + OAK_CONSTRUCTED_PER_PERCEPT) / OAK_SIMD_WIDTH;
constant uint OAK_POLICY_MAX = OAK_ENV_ACT_MAX;
constant uint OAK_DIAG_K = 2u * OAK_SIMD_WIDTH;
constant uint OAK_DIAG_GRAM = OAK_DIAG_K * OAK_DIAG_K;

static_assert(OAK_OPTION_COUNT % OAK_SIMD_WIDTH == 0u,
              "options must tile the simd width");
static_assert(OAK_PERCEPT_COUNT % OAK_SIMD_WIDTH == 0u,
              "declared percepts must tile the simd width");
static_assert(OAK_PERCEPTS_MAX % OAK_SIMD_WIDTH == 0u,
              "percepts must tile the simd width");
static_assert(OAK_PERCEPTS_MAX >= OAK_PERCEPT_SOURCES_MAX,
              "percepts must cover every source the widest environment emits");
static_assert(OAK_PERCEPTS_MAX * (1u + OAK_CONSTRUCTED_PER_PERCEPT * OAK_CANDIDATES_PER_SLOT) - OAK_PERCEPTS_MAX
              == (OAK_PERCEPTS_MAX * (1u + OAK_CONSTRUCTED_PER_PERCEPT) - OAK_PERCEPTS_MAX) * OAK_CANDIDATES_PER_SLOT,
              "every constructed slot must draw from exactly one candidate pool");
