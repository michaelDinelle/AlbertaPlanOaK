#pragma METAL fp math_mode(safe)

constant uint OAK_SIMD_WIDTH = 32u;
constant uint OAK_SPL = 1u;

constant uint OAK_CANDIDATES_PER_SLOT = 7u;
constant uint OAK_CONSTRUCTED_PER_ANCHOR = 2u;
constant uint OAK_OPTION_COUNT = 32u;
constant uint OAK_PLAN_PERIOD = 32u;

inline uint anchorsForStream(uint obsDim, uint actionParams) {
    return ((obsDim + actionParams + 2u + OAK_SIMD_WIDTH - 1u) / OAK_SIMD_WIDTH) * OAK_SIMD_WIDTH;
}

constant uint OAK_ANCHOR_SOURCES_MAX = OAK_ENV_OBS_MAX + 2u * OAK_ENV_ACT_MAX + 2u;
constant uint OAK_ANCHORS_MAX =
    ((OAK_ANCHOR_SOURCES_MAX + OAK_SIMD_WIDTH - 1u) / OAK_SIMD_WIDTH) * OAK_SIMD_WIDTH;
constant uint OAK_ACTIVE_SLOTS_MAX =
    OAK_ANCHORS_MAX * (1u + OAK_CONSTRUCTED_PER_ANCHOR) / OAK_SIMD_WIDTH;
constant uint OAK_POLICY_MAX = OAK_ENV_ACT_MAX;
constant uint OAK_DIAG_K = 2u * OAK_SIMD_WIDTH;
constant uint OAK_DIAG_GRAM = OAK_DIAG_K * OAK_DIAG_K;

static_assert(OAK_ANCHORS_MAX % OAK_SIMD_WIDTH == 0u,
              "anchors must tile the simd width");
static_assert(OAK_ANCHORS_MAX >= OAK_ANCHOR_SOURCES_MAX,
              "anchors must cover every source the widest environment emits");
static_assert(OAK_ANCHORS_MAX * (1u + OAK_CONSTRUCTED_PER_ANCHOR * OAK_CANDIDATES_PER_SLOT) - OAK_ANCHORS_MAX
              == (OAK_ANCHORS_MAX * (1u + OAK_CONSTRUCTED_PER_ANCHOR) - OAK_ANCHORS_MAX) * OAK_CANDIDATES_PER_SLOT,
              "every constructed slot must draw from exactly one candidate pool");
