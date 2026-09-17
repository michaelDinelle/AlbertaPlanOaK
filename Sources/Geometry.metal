#pragma METAL fp math_mode(safe)

#if !defined(OAK_PERCEPTS) || !defined(OAK_SLOTS) || !defined(OAK_CANDIDATES) || !defined(OAK_OPTIONS) \
    || !defined(OAK_UPDATES) \
    || !defined(OAK_TRACES)
#error "Declare the geometry: ./setup.sh p<percepts>_s<slots>_c<candidates>_k<options>_u<updates>_t<traces>"
#endif
kernel void OAK_GEOMETRY_MARKER() {}

constant uint OAK_SIMD_WIDTH = 32u;
constant uint OAK_THREADGROUP_LANES = 4u;
constant uint OAK_WIDE_LANE_SIMDGROUPS = 2u;
constant uint OAK_PLAN_PERIOD = 32u;
constant uint OAK_CANDIDATES_PER_SLOT = OAK_CANDIDATES;
constant uint OAK_TRACE_LEVELS = OAK_TRACES;
constant uint OAK_TILE_LAST = OAK_CANDIDATES_PER_SLOT - 1u;
constant uint OAK_JOINT_ROWS = OAK_TILE_LAST / 2u;
constant float OAK_TILE_HALF = 0.5f * float(OAK_CANDIDATES_PER_SLOT);
constant uint OAK_ACTION_PARAMS = 2u * OAK_ENV_ACT_MAX;
constant uint OAK_POLICY_MAX = OAK_ENV_ACT_MAX;
constant uint OAK_OBSERVATION_CAPACITY = OAK_ENV_OBS_MAX;
constant uint OAK_SOURCE_COUNT =
    (1u + OAK_TRACE_LEVELS) * OAK_OBSERVATION_CAPACITY + OAK_ACTION_PARAMS + 2u;
constant uint OAK_SOURCE_CAPACITY =
    ((OAK_SOURCE_COUNT + OAK_SIMD_WIDTH - 1u) / OAK_SIMD_WIDTH) * OAK_SIMD_WIDTH;
constant uint OAK_PERCEPT_COUNT = OAK_PERCEPTS;
constant uint OAK_CONSTRUCTED_SLOTS = OAK_SLOTS;
constant uint OAK_FEATURE_CAPACITY =
    OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_SLOTS * OAK_CANDIDATES_PER_SLOT;
constant uint OAK_ACTIVE_CAPACITY = OAK_PERCEPT_COUNT + OAK_CONSTRUCTED_SLOTS;
constant uint OAK_OPTION_COUNT = OAK_OPTIONS;
constant uint OAK_OPTION_UPDATES = OAK_UPDATES;
constant uint OAK_OPTION_BLOCKS = OAK_OPTION_COUNT / OAK_SIMD_WIDTH;
constant uint OAK_UPDATE_BLOCKS = OAK_OPTION_UPDATES / OAK_SIMD_WIDTH;
constant uint OAK_FEATURE_TIMESCALE = OAK_PERCEPT_COUNT * (1u + OAK_CANDIDATES_PER_SLOT);
constant uint OAK_LANE_SHARED_FLOATS = 5u * OAK_OPTION_COUNT + 1u;
constant float OAK_TRACKING_RATE = 1.0f / float(OAK_FEATURE_TIMESCALE);
constant float OAK_SAMPLE_SIGMA = 1.0f / float(OAK_ACTION_PARAMS);
constant float OAK_INV_TWO_SIGMA_SQUARED = 1.0f / (2.0f * OAK_SAMPLE_SIGMA * OAK_SAMPLE_SIGMA);
constant float OAK_INV_OPTION_COUNT = 1.0f / float(OAK_OPTION_COUNT);
constant float OAK_META_RATE = OAK_TRACKING_RATE / float(OAK_FEATURE_TIMESCALE + OAK_ACTION_PARAMS);
constant float OAK_ACTOR_ALPHA_DENOM = float(OAK_ACTION_PARAMS * OAK_OPTION_COUNT);

struct alignas(16) Record {
    float value, aux, reward, duration, next, nextAux, twin;
    float policy[OAK_POLICY_MAX];
};

constant uint OAK_FEATURE_STATES = 2u * OAK_FEATURE_CAPACITY;
constant uint OAK_POLICY_BANK = OAK_OPTION_COUNT * OAK_POLICY_MAX;
constant uint OAK_FEATURE_STATS = 4u * OAK_FEATURE_CAPACITY;
constant uint OAK_ACTIVE_STATES = 2u * OAK_ACTIVE_CAPACITY;
