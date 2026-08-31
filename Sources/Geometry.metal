#pragma METAL fp math_mode(safe)

constant uint OAK_SIMD_WIDTH = 32u;
constant uint OAK_SPL = 1u;

constant uint OAK_PERCEPT_BUDGET = 32u;
constant uint OAK_GATE_BANK = 7u;
constant uint OAK_GATE_GROUPS = 2u;
constant uint OAK_OPTION_COUNT = 32u;
constant uint OAK_PLAN_PERIOD = 32u;

constant uint OAK_ANCHORS_MAX = OAK_PERCEPT_BUDGET;
constant uint OAK_ACTIVE_SLOTS_MAX = OAK_ANCHORS_MAX * (1u + OAK_GATE_GROUPS) / OAK_SIMD_WIDTH;
constant uint OAK_POLICY_MAX = OAK_ENV_ACT_MAX;
constant uint OAK_DIAG_K = 2u * OAK_SIMD_WIDTH;
constant uint OAK_DIAG_GRAM = OAK_DIAG_K * OAK_DIAG_K;

static_assert(OAK_PERCEPT_BUDGET % OAK_SIMD_WIDTH == 0u,
              "perception budget must tile the simd width");
static_assert(OAK_PERCEPT_BUDGET >= OAK_ENV_OBS_MAX + 2u * OAK_ENV_ACT_MAX + 2u,
              "perception budget must cover every anchor source the widest environment emits");
static_assert(OAK_ANCHORS_MAX * (1u + OAK_GATE_GROUPS * OAK_GATE_BANK) - OAK_ANCHORS_MAX
              == (OAK_ANCHORS_MAX * (1u + OAK_GATE_GROUPS) - OAK_ANCHORS_MAX) * OAK_GATE_BANK,
              "every gate slot must cover exactly one bank of constructed features");
