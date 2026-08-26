#pragma METAL fp math_mode(safe)

constant uint OAK_ENV_FOURROOMS   = 1u;
constant uint OAK_ENV_DEEPSEA     = 2u;
constant uint OAK_ENV_KEYCORRIDOR = 3u;
constant uint OAK_ENV_ACROBOT     = 4u;
constant uint OAK_ENV_MOUNTAINCAR = 5u;
constant uint OAK_ENV_CARTPOLE    = 6u;
constant uint OAK_ENV_PENDULUM    = 7u;
constant uint OAK_ENV_CHAIN       = 8u;
constant uint OAK_ENV_LARGEDEEPSEA = 9u;
constant uint OAK_ENV_MULTIKEYCORRIDOR = 10u;
constant uint OAK_ENV_MEMORYCHAIN = 11u;
constant uint OAK_ENV_PROCEDURALMAZE = 12u;
constant uint OAK_ENV_HIGHDIMMAZE = 13u;
constant uint OAK_ENV_POMAZE      = 14u;
constant uint OAK_ENV_DELAYEDCHAIN = 15u;
constant uint OAK_ENV_COUNT       = 26u;

constant uint OAK_ENV_OBS_MAX   = 348u;
constant uint OAK_ENV_CLASSIC_OBS_MAX = 17u;
constant uint OAK_ENV_ACT_MAX   = 17u;
constant uint OAK_ENV_WORDS_MAX = 10u;
constant uint OAK_SAMPLE_SCRATCH = ((OAK_ENV_ACT_MAX + 3u) / 4u) * 4u;

constant uint OAK_SLOT_NONE    =  0u;
constant uint OAK_SLOT_X       =  1u;
constant uint OAK_SLOT_Y       =  2u;
constant uint OAK_SLOT_GOAL_X  =  3u;
constant uint OAK_SLOT_GOAL_Y  =  4u;
constant uint OAK_SLOT_KEY_X   =  5u;
constant uint OAK_SLOT_KEY_Y   =  6u;
constant uint OAK_SLOT_CARRY   =  7u;
constant uint OAK_SLOT_GATES   =  8u;
constant uint OAK_SLOT_TOKENS  =  9u;
constant uint OAK_SLOT_CUE     = 10u;
constant uint OAK_SLOT_HEADING = 11u;
constant uint OAK_SLOT_EFFORT  = 12u;
constant uint OAK_SLOT_Q       = 16u;

constant uint OAK_POSE_LEN = 24u;
constant uint OAK_RECORD_FLOATS = 1u + OAK_POSE_LEN;

struct OakEnvSpec {
    uint nameHash; uint obsDimNatural; uint actionKind; uint actionCount; uint actionParams;
    uint gridW; uint gridH; uint controlHz; uint stepFlops; uint envStateWords; uint mjIndex;
    uint slots[OAK_POSE_LEN];
};
constant OakEnvSpec OAK_ENV_SPECS[OAK_ENV_COUNT] = {
    { 0x9d7fd44bu, 17u, 0u, 6u, 12u,  0u,  0u, 20u, 33214u, 33u, 0u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u, OAK_SLOT_Q + 4u,
        OAK_SLOT_Q + 5u, OAK_SLOT_Q + 6u, OAK_SLOT_Q + 7u, OAK_SLOT_Q + 8u, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x8014a187u, 17u, 1u, 4u, 12u, 11u, 11u,  8u, 60u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_Y, OAK_SLOT_GOAL_X, OAK_SLOT_GOAL_Y, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x56af5bf2u,  8u, 1u, 2u, 12u,  8u,  8u,  8u, 30u, 10u, 0xffffffffu,
      { OAK_SLOT_Y, OAK_SLOT_X, OAK_SLOT_GOAL_Y, OAK_SLOT_GOAL_X, OAK_SLOT_EFFORT,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xdd537ebau, 14u, 1u, 4u, 12u,  9u,  7u,  8u, 70u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_Y, OAK_SLOT_GOAL_X, OAK_SLOT_GOAL_Y, OAK_SLOT_KEY_X,
        OAK_SLOT_KEY_Y, OAK_SLOT_CARRY, OAK_SLOT_GATES, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x93531833u,  7u, 1u, 3u, 12u,  0u,  0u,  5u, 400u, 10u, 0xffffffffu,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u, OAK_SLOT_EFFORT,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xf5b3ddc8u,  3u, 1u, 3u, 12u,  0u,  0u, 30u, 40u, 10u, 0xffffffffu,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_NONE, OAK_SLOT_EFFORT, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x41f9937du,  5u, 1u, 2u, 12u,  0u,  0u, 50u, 60u, 10u, 0xffffffffu,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u, OAK_SLOT_EFFORT,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x69bbbb73u,  4u, 0u, 1u, 12u,  0u,  0u, 20u, 60u, 10u, 0xffffffffu,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_EFFORT, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x7b267422u,  2u, 1u, 2u, 12u, 16u,  1u,  8u, 25u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xb96ddc3du,  8u, 1u, 2u, 12u, 16u, 16u,  8u, 30u, 10u, 0xffffffffu,
      { OAK_SLOT_Y, OAK_SLOT_X, OAK_SLOT_GOAL_Y, OAK_SLOT_GOAL_X, OAK_SLOT_EFFORT,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x4c3c2d3fu, 17u, 1u, 4u, 12u, 17u,  7u,  8u, 80u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_Y, OAK_SLOT_GOAL_X, OAK_SLOT_GOAL_Y, OAK_SLOT_TOKENS,
        OAK_SLOT_GATES, OAK_SLOT_EFFORT, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xcf293fd7u,  6u, 1u, 2u, 12u, 32u,  1u,  8u, 40u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_NONE, OAK_SLOT_CUE, OAK_SLOT_EFFORT, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xd75dd5f9u, 17u, 1u, 4u, 12u, 11u, 11u,  8u, 90u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_Y, OAK_SLOT_GOAL_X, OAK_SLOT_GOAL_Y, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xbfeb2b7au, 16u, 1u, 6u, 12u, 11u, 11u,  8u, 70u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_Y, OAK_SLOT_GOAL_X, OAK_SLOT_GOAL_Y, OAK_SLOT_HEADING,
        OAK_SLOT_EFFORT, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x0582e383u, 15u, 1u, 4u, 12u, 11u, 11u,  8u, 70u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_Y, OAK_SLOT_GOAL_X, OAK_SLOT_GOAL_Y, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x56184b96u,  3u, 1u, 2u, 12u, 32u,  1u,  8u, 30u, 10u, 0xffffffffu,
      { OAK_SLOT_X, OAK_SLOT_NONE, OAK_SLOT_CUE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x934afb1cu, 4u, 0u, 1u, 2u, 0u, 0u, 25u, 423u, 7u, 1u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x7db1edd7u, 9u, 0u, 1u, 2u, 0u, 0u, 20u, 2380u, 10u, 2u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x8cb879dfu, 10u, 0u, 2u, 4u, 0u, 0u, 50u, 1693u, 14u, 3u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xc3212447u, 8u, 0u, 2u, 4u, 0u, 0u, 25u, 5291u, 17u, 4u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x1546028fu, 11u, 0u, 3u, 6u, 0u, 0u, 125u, 35809u, 21u, 5u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_Q + 5u, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0xb41030a9u, 17u, 0u, 6u, 12u, 0u, 0u, 125u, 106285u, 33u, 6u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_Q + 5u, OAK_SLOT_Q + 6u, OAK_SLOT_Q + 7u,
        OAK_SLOT_Q + 8u, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x1101b258u, 23u, 0u, 7u, 14u, 0u, 0u, 20u, 163253u, 40u, 7u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_Q + 5u, OAK_SLOT_Q + 6u, OAK_SLOT_Q + 7u,
        OAK_SLOT_Q + 8u, OAK_SLOT_Q + 9u, OAK_SLOT_Q + 10u, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x956ec6e0u, 105u, 0u, 8u, 16u, 0u, 0u, 20u, 560000u, 51u, 8u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_Q + 5u, OAK_SLOT_Q + 6u, OAK_SLOT_Q + 7u,
        OAK_SLOT_Q + 8u, OAK_SLOT_Q + 9u, OAK_SLOT_Q + 10u, OAK_SLOT_Q + 11u,
        OAK_SLOT_Q + 12u, OAK_SLOT_Q + 13u, OAK_SLOT_Q + 14u, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE,
        OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE, OAK_SLOT_NONE } },
    { 0x8d2e84ccu, 348u, 0u, 17u, 34u, 0u, 0u, 67u, 7962989u, 87u, 9u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_Q + 5u, OAK_SLOT_Q + 6u, OAK_SLOT_Q + 7u,
        OAK_SLOT_Q + 8u, OAK_SLOT_Q + 9u, OAK_SLOT_Q + 10u, OAK_SLOT_Q + 11u,
        OAK_SLOT_Q + 12u, OAK_SLOT_Q + 13u, OAK_SLOT_Q + 14u, OAK_SLOT_Q + 15u,
        OAK_SLOT_Q + 16u, OAK_SLOT_Q + 17u, OAK_SLOT_Q + 18u, OAK_SLOT_Q + 19u,
        OAK_SLOT_Q + 20u, OAK_SLOT_Q + 21u, OAK_SLOT_Q + 22u, OAK_SLOT_Q + 23u } },
    { 0xb3aa307bu, 348u, 0u, 17u, 34u, 0u, 0u, 67u, 7962989u, 87u, 10u,
      { OAK_SLOT_Q + 0u, OAK_SLOT_Q + 1u, OAK_SLOT_Q + 2u, OAK_SLOT_Q + 3u,
        OAK_SLOT_Q + 4u, OAK_SLOT_Q + 5u, OAK_SLOT_Q + 6u, OAK_SLOT_Q + 7u,
        OAK_SLOT_Q + 8u, OAK_SLOT_Q + 9u, OAK_SLOT_Q + 10u, OAK_SLOT_Q + 11u,
        OAK_SLOT_Q + 12u, OAK_SLOT_Q + 13u, OAK_SLOT_Q + 14u, OAK_SLOT_Q + 15u,
        OAK_SLOT_Q + 16u, OAK_SLOT_Q + 17u, OAK_SLOT_Q + 18u, OAK_SLOT_Q + 19u,
        OAK_SLOT_Q + 20u, OAK_SLOT_Q + 21u, OAK_SLOT_Q + 22u, OAK_SLOT_Q + 23u } },
};

inline void env_reset(uint envKind, uint seed,
                      thread uint *words, thread uint &elapsed, thread uint &resets) {
    for (uint i = 0u; i < OAK_ENV_WORDS_MAX; i++) words[i] = 0u;
    if (envKind == OAK_ENV_FOURROOMS) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 9u; words[3] = 9u;
        words[4] = 0u; words[5] = 0u;
    } else if (envKind == OAK_ENV_DEEPSEA) {
        uint4 rm = philox4x32(uint2(seed, RNG_STREAM_RESET),
                                 uint4(0u, 0u, 0u, 0u));
        words[0] = 0u; words[1] = 0u; words[2] = 0u;
        words[3] = rm.x; words[4] = rm.y; words[5] = 0u;
    } else if (envKind == OAK_ENV_KEYCORRIDOR) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 3u; words[3] = 5u;
        words[4] = 7u; words[5] = 1u;
    } else if (envKind == OAK_ENV_MOUNTAINCAR) {
        words[0] = as_type<uint>(-0.5f);
    } else if (envKind == OAK_ENV_PENDULUM) {
        words[0] = as_type<uint>(M_PI_F);
    } else if (envKind == OAK_ENV_LARGEDEEPSEA) {
        uint4 rm0 = philox4x32(uint2(seed, RNG_STREAM_RESET),
                                  uint4(0u, 0u, 0u, 0u));
        uint4 rm1 = philox4x32(uint2(seed, RNG_STREAM_RESET),
                                  uint4(1u, 0u, 0u, 0u));
        words[0] = 0u;
        words[1] = rm0.x; words[2] = rm0.y;
        words[3] = rm0.z; words[4] = rm0.w;
        words[5] = rm1.x; words[6] = rm1.y;
        words[7] = rm1.z; words[8] = rm1.w;
    } else if (envKind == OAK_ENV_MULTIKEYCORRIDOR) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 3u | (5u << 4u);
        words[3] = 7u | (5u << 4u);
        words[4] = 11u | (5u << 4u);
        words[5] = 15u; words[6] = 1u;
    } else if (envKind == OAK_ENV_PROCEDURALMAZE) {
        uint mazeWalls[3];
        proceduralMaze_generate(seed, mazeWalls);
        words[6] = mazeWalls[0]; words[7] = mazeWalls[1]; words[8] = mazeWalls[2];
        words[0] = 1u; words[1] = 1u;
        words[2] = 9u; words[3] = 9u;
    } else if (envKind == OAK_ENV_HIGHDIMMAZE || envKind == OAK_ENV_POMAZE) {
        words[0] = 1u; words[1] = 1u;
        words[2] = 9u; words[3] = 9u;
        words[6] = 1u;
    }
}

inline void env_observe(uint envKind, thread const uint *words, thread float *obs) {
    if (envKind == OAK_ENV_FOURROOMS) {
        fourRooms_obs(1u, 1u, 9u, 9u, 0u, obs);
    } else if (envKind == OAK_ENV_DEEPSEA) {
        deepSea_obs(0u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_KEYCORRIDOR) {
        keyCorridor_obs(1u, 1u, 3u, 5u, 7u, 1u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_ACROBOT) {
        acrobot_obs(0.0f, 0.0f, 0.0f, 0.0f, obs);
    } else if (envKind == OAK_ENV_MOUNTAINCAR) {
        mountainCar_obs(-0.5f, 0.0f, obs);
    } else if (envKind == OAK_ENV_CARTPOLE) {
        cartPole_obs(0.0f, 0.0f, 0.0f, 0.0f, obs);
    } else if (envKind == OAK_ENV_PENDULUM) {
        pendulum_obs(M_PI_F, 0.0f, obs);
    } else if (envKind == OAK_ENV_CHAIN) {
        obs[0] = 0.0f; obs[1] = 1.0f;
        for (uint i = 2u; i < OAK_ENV_OBS_MAX; i++) obs[i] = 0.0f;
    } else if (envKind == OAK_ENV_LARGEDEEPSEA) {
        largeDeepSea_obs(0u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_MULTIKEYCORRIDOR) {
        multiKeyCorridor_obs(1u, 1u, 3u | (5u << 4u), 7u | (5u << 4u), 11u | (5u << 4u),
                   15u, 1u, 0u, obs);
    } else if (envKind == OAK_ENV_MEMORYCHAIN) {
        memoryChain_obs(0u, 0u, 0u, obs);
    } else if (envKind == OAK_ENV_PROCEDURALMAZE) {
        proceduralMaze_obs(1u, 1u, 9u, 9u, 0u, words[6], words[7], words[8], obs);
    } else if (envKind == OAK_ENV_HIGHDIMMAZE) {
        highDimMaze_obs(1u, 1u, 9u, 9u, obs);
    } else if (envKind == OAK_ENV_POMAZE) {
        poMaze_obs(1u, 1u, 9u, 9u, obs);
    } else {
        obs[0] = 0.0f; obs[1] = 0.0f; obs[2] = 1.0f;
        for (uint i = 3u; i < OAK_ENV_OBS_MAX; i++) obs[i] = 0.0f;
    }
}

inline float env_step(uint envKind, uint seed, thread const float *ctrl,
                      thread uint *words,
                      thread float *obs, thread uint &done) {
    done = 0u;
    uint discreteAction = uint(ctrl[0]);
    if (envKind == OAK_ENV_FOURROOMS) return fourRooms_step(words, seed, discreteAction, obs);
    if (envKind == OAK_ENV_DEEPSEA) return deepSea_step(words, discreteAction, obs);
    if (envKind == OAK_ENV_KEYCORRIDOR) return keyCorridor_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_CHAIN) return chain_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_LARGEDEEPSEA) return largeDeepSea_step(words, discreteAction, obs);
    if (envKind == OAK_ENV_MULTIKEYCORRIDOR) return multiKeyCorridor_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_MEMORYCHAIN) return memoryChain_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_PROCEDURALMAZE) return proceduralMaze_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_HIGHDIMMAZE) return highDimMaze_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_POMAZE) return poMaze_step(words, seed, discreteAction, obs, &done);
    if (envKind == OAK_ENV_DELAYEDCHAIN) return delayedChain_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_ACROBOT) return acrobot_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_MOUNTAINCAR) return mountainCar_step(words, discreteAction, obs, &done);
    if (envKind == OAK_ENV_CARTPOLE) return cartPole_step(words, discreteAction, obs, &done);
    return pendulum_step(words, ctrl[0], obs);
}

inline void env_pose(uint envKind, thread const uint *words,
                     thread const float *ctrl, thread float *pose) {
    if (envKind == OAK_ENV_FOURROOMS) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_DEEPSEA) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = 7.0f; pose[3] = 7.0f;
        pose[4] = ctrl[0];
        for (int i = 5; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_KEYCORRIDOR) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[4]); pose[3] = float(words[5]);
        pose[4] = float(words[2]); pose[5] = float(words[3]);
        pose[6] = float(words[6] & 1u); pose[7] = float((words[6] >> 1u) & 1u);
        pose[8] = 0.0f;
    } else if (envKind == OAK_ENV_CHAIN) {
        pose[0] = float(words[0]); pose[1] = 15.0f;
        for (int i = 2; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_LARGEDEEPSEA) {
        pose[0] = float(words[0] & 15u); pose[1] = float((words[0] >> 4u) & 15u);
        pose[2] = 15.0f; pose[3] = 15.0f;
        pose[4] = ctrl[0];
        for (int i = 5; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_MULTIKEYCORRIDOR) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[5]); pose[3] = float(words[6]);
        pose[4] = float(words[7] & 7u);
        pose[5] = float((words[7] >> 3u) & 7u);
        pose[6] = ctrl[0];
        for (int i = 7; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_MEMORYCHAIN) {
        pose[0] = float(words[0]); pose[1] = 31.0f;
        pose[2] = float(words[1]); pose[3] = ctrl[0];
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_PROCEDURALMAZE) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_HIGHDIMMAZE) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        pose[4] = float(words[6]); pose[5] = ctrl[0];
        for (int i = 6; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_POMAZE) {
        pose[0] = float(words[0]); pose[1] = float(words[1]);
        pose[2] = float(words[2]); pose[3] = float(words[3]);
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_DELAYEDCHAIN) {
        pose[0] = float(words[0]); pose[1] = 31.0f;
        pose[2] = float(words[1]);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_ACROBOT) {
        float theta1 = as_type<float>(words[0]), theta2 = as_type<float>(words[1]);
        pose[0] = theta1; pose[1] = theta2;
        pose[2] = as_type<float>(words[2]); pose[3] = as_type<float>(words[3]);
        pose[4] = ctrl[0];
        for (int i = 5; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_MOUNTAINCAR) {
        pose[0] = as_type<float>(words[0]); pose[1] = as_type<float>(words[1]);
        pose[2] = 0.5f; pose[3] = ctrl[0];
        for (int i = 4; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_CARTPOLE) {
        pose[0] = as_type<float>(words[0]); pose[1] = as_type<float>(words[1]);
        pose[2] = as_type<float>(words[2]); pose[3] = as_type<float>(words[3]);
        pose[4] = ctrl[0];
        for (int i = 5; i < 9; i++) pose[i] = 0.0f;
    } else if (envKind == OAK_ENV_PENDULUM) {
        float theta = as_type<float>(words[0]), thetaDot = as_type<float>(words[1]);
        pose[0] = theta; pose[1] = thetaDot; pose[2] = clamp(2.0f * ctrl[0], -2.0f, 2.0f);
        for (int i = 3; i < 9; i++) pose[i] = 0.0f;
    } else {
        for (int i = 0; i < 9; i++) pose[i] = 0.0f;
    }
}
