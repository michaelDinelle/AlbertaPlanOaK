#include "geometry.h"

#define OAK_FAMILY_LIST(F) \
    F(access) \
    F(acrobot) \
    F(aliased) \
    F(bandit) \
    F(blocking) \
    F(calibration) \
    F(cartpole) \
    F(catch) \
    F(cellular) \
    F(walk) \
    F(cliff) \
    F(coinswap) \
    F(conditioning) \
    F(contextual) \
    F(count) \
    F(covariate) \
    F(deepsea) \
    F(forager) \
    F(hall) \
    F(garnet) \
    F(hashed) \
    F(hiddenmode) \
    F(jbw) \
    F(keydoor) \
    F(tmaze) \
    F(mountainCar) \
    F(nexting) \
    F(operant) \
    F(parity) \
    F(password) \
    F(patience) \
    F(pendulum) \
    F(permuted) \
    F(puddle) \
    F(recite) \
    F(regression) \
    F(repeat) \
    F(ring) \
    F(riverswim) \
    F(rps) \
    F(scentpuck) \
    F(sokoban) \
    F(subsetparity) \
    F(taxi) \
    F(tiger) \
    F(tracking) \
    F(umbrella) \
    F(walkbet)

#define OAK_ENV_LIST(X) \
    X(ACCESS_FIXED, access, 0u, "AccessControl-cont-v0", 6u, 2u, 0u, 6144u, 16u, 0u, noPose, noExtra) \
    X(ACCESS_DRIFT, access, 1u, "AccessControl-drift-cont-v0", 6u, 2u, 0u, 6208u, 16u, 0u, noPose, noExtra) \
    X(ACROBOT, acrobot, 0u, "Acrobot-cont-v0", 7u, 3u, 0u, 256u, 32u, 5u, acrobot_pose, noExtra) \
    X(ACROBOT_SLIP, acrobot, 1u, "Acrobot-slip-cont-v0", 7u, 3u, 0u, 8320u, 64u, 5u, acrobot_pose, noExtra) \
    X(ACROBOT_POSITION, acrobot, 2u, "Acrobot-position-cont-v0", 7u, 3u, 0u, 8384u, 64u, 5u, acrobot_pose, noExtra) \
    X(ALIASED_8, aliased, 0u, "AliasedRing-8-cont-v0", 2u, 2u, 0u, 5824u, 16u, 0u, noPose, noExtra) \
    X(ALIASED_32, aliased, 1u, "AliasedRing-32-cont-v0", 2u, 2u, 0u, 5888u, 64u, 0u, noPose, noExtra) \
    X(ALIASED_128, aliased, 2u, "AliasedRing-128-cont-v0", 2u, 2u, 0u, 5952u, 64u, 0u, noPose, noExtra) \
    X(BANDIT_3, bandit, 0u, "Bandit-3-cont-v0", 1u, 3u, 0u, 1088u, 64u, 0u, noPose, noExtra) \
    X(BANDIT_6, bandit, 1u, "Bandit-6-cont-v0", 1u, 6u, 0u, 1152u, 64u, 0u, noPose, noExtra) \
    X(BANDIT_3_DRIFT, bandit, 2u, "Bandit-3-drift-cont-v0", 1u, 3u, 0u, 1216u, 64u, 0u, noPose, noExtra) \
    X(BANDIT_6_DRIFT, bandit, 3u, "Bandit-6-drift-cont-v0", 1u, 6u, 0u, 1280u, 64u, 0u, noPose, noExtra) \
    X(BLOCKINGMAZE, blocking, 0u, "BlockingMaze-cont-v0", 8u, 4u, 0u, 3840u, 16u, 0u, noPose, noExtra) \
    X(SHORTCUTMAZE, blocking, 1u, "ShortcutMaze-cont-v0", 8u, 4u, 0u, 3904u, 64u, 0u, noPose, noExtra) \
    X(CALIBRATION_30, calibration, 0u, "Calibration-30-cont-v0", 2u, 2u, 0u, 2112u, 64u, 0u, noPose, noExtra) \
    X(CALIBRATION_80, calibration, 1u, "Calibration-80-cont-v0", 2u, 2u, 0u, 2176u, 64u, 0u, noPose, noExtra) \
    X(CARTPOLE, cartpole, 0u, "CartPoleSwingUp-cont-v0", 6u, 3u, 0u, 7168u, 32u, 0u, noPose, noExtra) \
    X(CARTPOLE_SLIP, cartpole, 1u, "CartPoleSwingUp-slip-cont-v0", 6u, 3u, 0u, 7232u, 64u, 0u, noPose, noExtra) \
    X(CATCH_5_10, catch, 0u, "Catch-5x10-cont-v0", 4u, 3u, 0u, 3136u, 27u, 0u, noPose, noExtra) \
    X(CATCH_10_20, catch, 1u, "Catch-10x20-cont-v0", 4u, 3u, 0u, 3200u, 16u, 0u, noPose, noExtra) \
    X(CATCH_NOISE, catch, 2u, "Catch-noise-cont-v0", 4u, 3u, 0u, 3264u, 41u, 0u, noPose, noExtra) \
    X(CATCH_SCALE, catch, 3u, "Catch-scale-cont-v0", 4u, 3u, 0u, 3328u, 64u, 0u, noPose, noExtra) \
    X(CELLULAR, cellular, 0u, "Cellular-cont-v0", 17u, 2u, 0u, 7808u, 16u, 0u, noPose, noExtra) \
    X(CHAIN, walk, 0u, "Chain-cont-v0", 2u, 2u, 0u, 512u, 16u, 2u, walk_pose, noExtra) \
    X(CLIFF, cliff, 0u, "CliffWalk-cont-v0", 7u, 4u, 0u, 6464u, 64u, 0u, noPose, noExtra) \
    X(CLIFF_SLIP, cliff, 1u, "CliffWalk-slip-cont-v0", 7u, 4u, 0u, 6528u, 64u, 0u, noPose, noExtra) \
    X(COINSWAP_SLOW, coinswap, 0u, "CoinSwap-slow-cont-v0", 1u, 2u, 0u, 1344u, 25u, 0u, noPose, noExtra) \
    X(COINSWAP_FAST, coinswap, 1u, "CoinSwap-fast-cont-v0", 1u, 2u, 0u, 1408u, 17u, 0u, noPose, noExtra) \
    X(TRACE_10, conditioning, 0u, "TraceConditioning-10-cont-v0", 5u, 2u, 0u, 4672u, 16u, 0u, noPose, noExtra) \
    X(TRACE_100, conditioning, 1u, "TraceConditioning-100-cont-v0", 5u, 2u, 0u, 4736u, 16u, 0u, noPose, noExtra) \
    X(PATTERNING, conditioning, 2u, "TracePatterning-cont-v0", 5u, 2u, 0u, 4800u, 16u, 0u, noPose, noExtra) \
    X(CONTEXTUAL_8, contextual, 0u, "Contextual-8-cont-v0", 9u, 4u, 0u, 1472u, 32u, 0u, noPose, contextual_extraWords) \
    X(CONTEXTUAL_16, contextual, 1u, "Contextual-16-cont-v0", 17u, 4u, 0u, 1536u, 16u, 0u, noPose, contextual_extraWords) \
    X(CONTEXTUAL_8_DRIFT, contextual, 2u, "Contextual-8-drift-cont-v0", 9u, 4u, 0u, 1600u, 39u, 0u, noPose, contextual_extraWords) \
    X(CONTEXTUAL_16_DRIFT, contextual, 3u, "Contextual-16-drift-cont-v0", 17u, 4u, 0u, 1664u, 20u, 0u, noPose, contextual_extraWords) \
    X(COUNT_8, count, 0u, "CountRecall-8-cont-v0", 3u, 4u, 0u, 5440u, 17u, 0u, noPose, noExtra) \
    X(COUNT_32, count, 1u, "CountRecall-32-cont-v0", 3u, 4u, 0u, 5504u, 16u, 0u, noPose, noExtra) \
    X(COVARIATE_FIXED, covariate, 0u, "Covariate-fixed-cont-v0", 5u, 2u, 0u, 4544u, 16u, 0u, noPose, noExtra) \
    X(COVARIATE_SHIFT, covariate, 1u, "Covariate-shift-cont-v0", 5u, 2u, 0u, 4608u, 47u, 0u, noPose, noExtra) \
    X(DEEPSEA, deepsea, 0u, "DeepSea-cont-v0", 8u, 2u, 0u, 128u, 16u, 5u, deepsea_pose, noExtra) \
    X(LARGEDEEPSEA, deepsea, 1u, "LargeDeepSea-cont-v0", 8u, 2u, 0u, 576u, 64u, 5u, deepsea_pose, noExtra) \
    X(DEEPSEA_32, deepsea, 2u, "DeepSea-32-cont-v0", 8u, 2u, 0u, 8000u, 16u, 5u, deepsea_pose, noExtra) \
    X(DEEPSEA_16_SLIP, deepsea, 3u, "DeepSea-16-slip-cont-v0", 8u, 2u, 0u, 8064u, 42u, 5u, deepsea_pose, noExtra) \
    X(FORAGER_16_1, forager, 0u, "Forager-16-1-cont-v0", 10u, 4u, 0u, 7296u, 19u, 0u, noPose, forager_extraWords) \
    X(FORAGER_64_1, forager, 1u, "Forager-64-1-cont-v0", 10u, 4u, 0u, 7360u, 61u, 0u, noPose, forager_extraWords) \
    X(FORAGER_16_2, forager, 2u, "Forager-16-2-cont-v0", 26u, 4u, 0u, 7424u, 16u, 0u, noPose, forager_extraWords) \
    X(FORAGER_16_SWITCH, forager, 3u, "Forager-16-1-switch-cont-v0", 10u, 4u, 0u, 7488u, 18u, 0u, noPose, forager_extraWords) \
    X(HALL_FIXED, hall, 0u, "FourRoomsHall-cont-v0", 14u, 4u, 0u, 6592u, 16u, 4u, hall_pose, noExtra) \
    X(HALL_MOVING, hall, 1u, "FourRoomsHall-moving-cont-v0", 14u, 4u, 0u, 6656u, 64u, 4u, hall_pose, noExtra) \
    X(GARNET_SMALL, garnet, 0u, "Garnet-10-2-2-cont-v0", 5u, 2u, 0u, 6784u, 64u, 0u, noPose, garnet_extraWords) \
    X(GARNET_LARGE, garnet, 1u, "Garnet-50-4-3-cont-v0", 7u, 4u, 0u, 6848u, 64u, 0u, noPose, garnet_extraWords) \
    X(GARNET_SMALL_RESAMPLE, garnet, 2u, "Garnet-10-2-2-resample-cont-v0", 5u, 2u, 0u, 6912u, 64u, 0u, noPose, garnet_extraWords) \
    X(GARNET_LARGE_RESAMPLE, garnet, 3u, "Garnet-50-4-3-resample-cont-v0", 7u, 4u, 0u, 6976u, 64u, 0u, noPose, garnet_extraWords) \
    X(HASHED_8_LOCAL, hashed, 0u, "HashedMDP-8-local-cont-v0", 9u, 4u, 0u, 7552u, 64u, 0u, noPose, noExtra) \
    X(HASHED_16_LOCAL, hashed, 1u, "HashedMDP-16-local-cont-v0", 17u, 4u, 0u, 7616u, 64u, 0u, noPose, noExtra) \
    X(HASHED_8_GLOBAL, hashed, 2u, "HashedMDP-8-global-cont-v0", 9u, 4u, 0u, 7680u, 64u, 0u, noPose, noExtra) \
    X(HASHED_16_GLOBAL, hashed, 3u, "HashedMDP-16-global-cont-v0", 17u, 4u, 0u, 7744u, 64u, 0u, noPose, noExtra) \
    X(HIDDENMODE_SLOW, hiddenmode, 0u, "HiddenMode-slow-cont-v0", 1u, 2u, 0u, 6016u, 38u, 0u, noPose, noExtra) \
    X(HIDDENMODE_FAST, hiddenmode, 1u, "HiddenMode-fast-cont-v0", 1u, 2u, 0u, 6080u, 52u, 0u, noPose, noExtra) \
    X(JELLYBEANWORLD, jbw, 0u, "JellyBeanWorld-cont-v0", 16u, 3u, 0u, 1024u, 64u, 7u, jbw_pose, jbw_extraWords) \
    X(JELLYBEANWORLD_8, jbw, 1u, "JellyBeanWorld-8-cont-v0", 16u, 3u, 0u, 8576u, 64u, 7u, jbw_pose, jbw_extraWords) \
    X(KEYDOOR_8, keydoor, 0u, "KeyDoor-8-cont-v0", 4u, 2u, 0u, 2944u, 16u, 0u, noPose, noExtra) \
    X(KEYDOOR_64, keydoor, 1u, "KeyDoor-64-cont-v0", 4u, 2u, 0u, 3008u, 16u, 0u, noPose, noExtra) \
    X(KEYDOOR_512, keydoor, 2u, "KeyDoor-512-cont-v0", 4u, 2u, 0u, 3072u, 64u, 0u, noPose, noExtra) \
    X(TMAZE_4_1, tmaze, 0u, "TMaze-4-1-cont-v0", 6u, 2u, 0u, 4992u, 16u, 4u, tmaze_pose, noExtra) \
    X(TMAZE_16_1, tmaze, 1u, "TMaze-16-1-cont-v0", 6u, 2u, 0u, 5056u, 44u, 4u, tmaze_pose, noExtra) \
    X(MEMORYCHAIN, tmaze, 2u, "MemoryChain-cont-v0", 6u, 2u, 0u, 704u, 38u, 4u, tmaze_pose, noExtra) \
    X(TMAZE_64_1, tmaze, 3u, "TMaze-64-1-cont-v0", 6u, 2u, 0u, 5120u, 23u, 4u, tmaze_pose, noExtra) \
    X(TMAZE_16_2, tmaze, 4u, "TMaze-16-2-cont-v0", 7u, 4u, 0u, 5184u, 64u, 4u, tmaze_pose, noExtra) \
    X(TMAZE_64_2, tmaze, 5u, "TMaze-64-2-cont-v0", 7u, 4u, 0u, 5248u, 26u, 4u, tmaze_pose, noExtra) \
    X(MOUNTAINCAR, mountainCar, 0u, "MountainCar-cont-v0", 3u, 3u, 0u, 320u, 45u, 4u, mountainCar_pose, noExtra) \
    X(MOUNTAINCAR_SLIP, mountainCar, 1u, "MountainCar-slip-cont-v0", 3u, 3u, 0u, 8192u, 64u, 4u, mountainCar_pose, noExtra) \
    X(MOUNTAINCAR_POSITION, mountainCar, 2u, "MountainCar-position-cont-v0", 3u, 3u, 0u, 8256u, 64u, 4u, mountainCar_pose, noExtra) \
    X(NEXTING_1, nexting, 0u, "Nexting-1-cont-v0", 4u, 2u, 0u, 4864u, 16u, 0u, noPose, noExtra) \
    X(NEXTING_10, nexting, 1u, "Nexting-10-cont-v0", 4u, 2u, 0u, 4928u, 16u, 0u, noPose, noExtra) \
    X(OPERANT_16, operant, 0u, "Operant-16-cont-v0", 2u, 2u, 0u, 1728u, 16u, 0u, noPose, noExtra) \
    X(OPERANT_64, operant, 1u, "Operant-64-cont-v0", 2u, 2u, 0u, 1792u, 16u, 0u, noPose, noExtra) \
    X(PARITY_4, parity, 0u, "Parity-4-cont-v0", 2u, 2u, 0u, 3392u, 16u, 0u, noPose, noExtra) \
    X(PARITY_64, parity, 1u, "Parity-64-cont-v0", 2u, 2u, 0u, 3456u, 16u, 0u, noPose, noExtra) \
    X(PASSWORD_2, password, 0u, "Password-2-cont-v0", 2u, 4u, 0u, 3648u, 35u, 0u, noPose, noExtra) \
    X(PASSWORD_4, password, 1u, "Password-4-cont-v0", 2u, 4u, 0u, 3712u, 45u, 0u, noPose, noExtra) \
    X(PASSWORD_6, password, 2u, "Password-6-cont-v0", 2u, 4u, 0u, 3776u, 64u, 0u, noPose, noExtra) \
    X(PATIENCE_5, patience, 0u, "Patience-5-cont-v0", 2u, 2u, 0u, 2624u, 64u, 0u, noPose, noExtra) \
    X(PATIENCE_50, patience, 1u, "Patience-50-cont-v0", 2u, 2u, 0u, 2688u, 64u, 0u, noPose, noExtra) \
    X(PATIENCE_500, patience, 2u, "Patience-500-cont-v0", 2u, 2u, 0u, 2752u, 64u, 0u, noPose, noExtra) \
    X(PENDULUM, pendulum, 0u, "Pendulum-cont-v0", 4u, 1u, 1u, 448u, 64u, 3u, pendulum_pose, noExtra) \
    X(PENDULUM_SLIP, pendulum, 1u, "Pendulum-slip-cont-v0", 4u, 1u, 1u, 8448u, 64u, 3u, pendulum_pose, noExtra) \
    X(PENDULUM_POSITION, pendulum, 2u, "Pendulum-position-cont-v0", 4u, 1u, 1u, 8512u, 64u, 3u, pendulum_pose, noExtra) \
    X(PERMUTED_FIXED, permuted, 0u, "Permuted-fixed-cont-v0", 9u, 4u, 0u, 4416u, 16u, 0u, noPose, noExtra) \
    X(PERMUTED_RECODE, permuted, 1u, "Permuted-recode-cont-v0", 9u, 4u, 0u, 4480u, 64u, 0u, noPose, noExtra) \
    X(PUDDLE, puddle, 0u, "PuddleWorld-cont-v0", 3u, 4u, 0u, 6400u, 16u, 0u, noPose, noExtra) \
    X(RECITE_4, recite, 0u, "Recite-4-cont-v0", 7u, 4u, 0u, 5568u, 16u, 0u, noPose, noExtra) \
    X(RECITE_16, recite, 1u, "Recite-16-cont-v0", 7u, 4u, 0u, 5632u, 16u, 0u, noPose, noExtra) \
    X(REGRESSION_8, regression, 0u, "Regression-8-cont-v0", 9u, 5u, 0u, 4032u, 64u, 0u, noPose, noExtra) \
    X(REGRESSION_16, regression, 1u, "Regression-16-cont-v0", 17u, 5u, 0u, 4096u, 64u, 0u, noPose, noExtra) \
    X(REPEAT_4, repeat, 0u, "Repeat-4-cont-v0", 5u, 4u, 0u, 5312u, 16u, 0u, noPose, noExtra) \
    X(REPEAT_32, repeat, 1u, "Repeat-32-cont-v0", 5u, 4u, 0u, 5376u, 16u, 0u, noPose, noExtra) \
    X(RING_2, ring, 0u, "Ring-2-cont-v0", 2u, 2u, 0u, 1984u, 16u, 0u, noPose, noExtra) \
    X(RING_8, ring, 1u, "Ring-8-cont-v0", 2u, 2u, 0u, 2048u, 16u, 0u, noPose, noExtra) \
    X(RIVERSWIM_6, riverswim, 0u, "RiverSwim-6-cont-v0", 4u, 2u, 0u, 3520u, 16u, 0u, noPose, noExtra) \
    X(RIVERSWIM_24, riverswim, 1u, "RiverSwim-24-cont-v0", 6u, 2u, 0u, 3584u, 16u, 0u, noPose, noExtra) \
    X(RPS_10, rps, 0u, "RockPaperScissors-10-cont-v0", 4u, 3u, 0u, 7872u, 16u, 0u, noPose, noExtra) \
    X(RPS_100, rps, 1u, "RockPaperScissors-100-cont-v0", 4u, 3u, 0u, 7936u, 16u, 0u, noPose, noExtra) \
    X(SCENTPUCK_ONE, scentpuck, 0u, "ScentPuck-cont-v0", 5u, 5u, 0u, 6272u, 64u, 0u, noPose, noExtra) \
    X(SCENTPUCK_TWO, scentpuck, 1u, "ScentPuck-competing-cont-v0", 5u, 5u, 0u, 6336u, 64u, 0u, noPose, noExtra) \
    X(SOKOBAN, sokoban, 0u, "Sokoban-cont-v0", 7u, 4u, 0u, 3968u, 35u, 0u, noPose, noExtra) \
    X(SUBSETPARITY_8_1, subsetparity, 0u, "SubsetParity-8-1-cont-v0", 9u, 2u, 0u, 4160u, 16u, 0u, noPose, noExtra) \
    X(SUBSETPARITY_16_1, subsetparity, 1u, "SubsetParity-16-1-cont-v0", 17u, 2u, 0u, 4224u, 16u, 0u, noPose, noExtra) \
    X(SUBSETPARITY_8_2, subsetparity, 2u, "SubsetParity-8-2-cont-v0", 9u, 2u, 0u, 4288u, 21u, 0u, noPose, noExtra) \
    X(SUBSETPARITY_16_2, subsetparity, 3u, "SubsetParity-16-2-cont-v0", 17u, 2u, 0u, 4352u, 16u, 0u, noPose, noExtra) \
    X(TAXI, taxi, 0u, "Taxi-cont-v0", 16u, 6u, 0u, 6720u, 16u, 0u, noPose, noExtra) \
    X(TIGER_85, tiger, 0u, "Tiger-85-cont-v0", 3u, 3u, 0u, 5696u, 64u, 0u, noPose, noExtra) \
    X(TIGER_60, tiger, 1u, "Tiger-60-cont-v0", 3u, 3u, 0u, 5760u, 64u, 0u, noPose, noExtra) \
    X(TRACKING_99, tracking, 0u, "Tracking-99-cont-v0", 2u, 1u, 1u, 1856u, 27u, 0u, noPose, noExtra) \
    X(TRACKING_999, tracking, 1u, "Tracking-999-cont-v0", 2u, 1u, 1u, 1920u, 53u, 0u, noPose, noExtra) \
    X(UMBRELLA_4, umbrella, 0u, "Umbrella-4-cont-v0", 4u, 2u, 0u, 2240u, 16u, 0u, noPose, noExtra) \
    X(UMBRELLA_16, umbrella, 1u, "Umbrella-16-cont-v0", 4u, 2u, 0u, 2304u, 64u, 0u, noPose, noExtra) \
    X(UMBRELLA_64, umbrella, 2u, "Umbrella-64-cont-v0", 4u, 2u, 0u, 2368u, 64u, 0u, noPose, noExtra) \
    X(UMBRELLA_4_NOISE, umbrella, 3u, "Umbrella-4-noise-cont-v0", 16u, 2u, 0u, 2432u, 64u, 0u, noPose, noExtra) \
    X(UMBRELLA_16_NOISE, umbrella, 4u, "Umbrella-16-noise-cont-v0", 16u, 2u, 0u, 2496u, 64u, 0u, noPose, noExtra) \
    X(UMBRELLA_64_NOISE, umbrella, 5u, "Umbrella-64-noise-cont-v0", 16u, 2u, 0u, 2560u, 64u, 0u, noPose, noExtra) \
    X(WALKBET_FIXED, walkbet, 0u, "WalkBet-cont-v0", 2u, 2u, 0u, 7040u, 16u, 0u, noPose, noExtra) \
    X(WALKBET_DRIFT, walkbet, 1u, "WalkBet-drift-cont-v0", 2u, 2u, 0u, 7104u, 60u, 0u, noPose, noExtra)

#define OAK_FAMILY_PROTOTYPES(family) \
    void family##_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps); \
    float family##_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared); \
    void family##_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared); \
    void family##_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared);
OAK_FAMILY_LIST(OAK_FAMILY_PROTOTYPES)
void acrobot_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void walk_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void deepsea_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void hall_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void jbw_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void tmaze_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void mountainCar_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
void pendulum_pose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose);
uint32_t contextual_extraWords(uint32_t instance, uint32_t steps);
uint32_t forager_extraWords(uint32_t instance, uint32_t steps);
uint32_t garnet_extraWords(uint32_t instance, uint32_t steps);
uint32_t jbw_extraWords(uint32_t instance, uint32_t steps);

static inline void noPose(uint32_t instance, const uint32_t *words, const float *ctrl, float *pose) { (void)instance; (void)words; (void)ctrl; (void)pose; }
static inline uint32_t noExtra(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 0u; }

#define OAK_ENV_ENUM(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, pose, extra) ENV_##sym,
enum { OAK_ENV_LIST(OAK_ENV_ENUM) OAK_ENV_COUNT };

void env_reset(uint32_t env, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
#define OAK_ENV_RESET(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, pose, extra_) case ENV_##sym: family##_reset(instance, seed, words, extra, steps); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_RESET) default: return; }
}

float env_step(uint32_t env, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
#define OAK_ENV_STEP(sym, family, instance, name, obs_, actions, box, seedStart, seeds, poseLen, pose, extra_) case ENV_##sym: return family##_step(instance, seed, step, ctrl, words, extra, obs, done, shared);
    switch (env) { OAK_ENV_LIST(OAK_ENV_STEP) default: *done = 0u; return 0.0f; }
}

void env_observe(uint32_t env, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
#define OAK_ENV_OBSERVE(sym, family, instance, name, obs_, actions, box, seedStart, seeds, poseLen, pose, extra_) case ENV_##sym: family##_observe(instance, seed, words, extra, step, obs, shared); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_OBSERVE) default: return; }
}

void env_pose(uint32_t env, const uint32_t *words, const float *ctrl, float *pose) {
#define OAK_ENV_POSE(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, poseFn, extra) case ENV_##sym: poseFn(instance, words, ctrl, pose); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_POSE) default: return; }
}

void env_oracle(uint32_t env, uint32_t seed, uint32_t step, const uint32_t *words, const uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
#define OAK_ENV_ORACLE(sym, family, instance, name, obs_, actions, box, seedStart, seeds, poseLen, pose, extra_) case ENV_##sym: family##_oracle(instance, seed, step, words, extra, obs, ctrl, shared); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_ORACLE) default: return; }
}

uint32_t env_obs(uint32_t env) {
#define OAK_ENV_OBS(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, pose, extra) case ENV_##sym: return obs;
    switch (env) { OAK_ENV_LIST(OAK_ENV_OBS) default: return 0u; }
}

uint32_t env_actions(uint32_t env) {
#define OAK_ENV_ACTIONS(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, pose, extra) case ENV_##sym: return actions;
    switch (env) { OAK_ENV_LIST(OAK_ENV_ACTIONS) default: return 0u; }
}

uint32_t env_box(uint32_t env) {
#define OAK_ENV_BOX(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, pose, extra) case ENV_##sym: return box;
    switch (env) { OAK_ENV_LIST(OAK_ENV_BOX) default: return 0u; }
}

uint32_t env_extraWords(uint32_t env, uint32_t steps) {
#define OAK_ENV_EXTRA(sym, family, instance, name, obs, actions, box, seedStart, seeds, poseLen, pose, extraFn) case ENV_##sym: return extraFn(instance, steps);
    switch (env) { OAK_ENV_LIST(OAK_ENV_EXTRA) default: return 0u; }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OAK_ENV_SPEC(sym, family, instance, name_, obs_, actions_, box_, seedStart_, seeds_, poseLen_, pose, extra) { name_, #family, instance, ENV_##sym, obs_, actions_, box_, seedStart_, seeds_, poseLen_ },
static const EnvSpec TABLE[] = { OAK_ENV_LIST(OAK_ENV_SPEC) };

const uint32_t OAK_ENV_TABLE_COUNT = sizeof TABLE / sizeof TABLE[0];

const EnvSpec *env_at(uint32_t index) { return &TABLE[index]; }

const EnvSpec *env_named(const char *name) {
    for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++)
        if (strcmp(TABLE[i].name, name) == 0) return &TABLE[i];
    return 0;
}

void env_checkTable(void) {
    if (OAK_ENV_TABLE_COUNT > OAK_ENV_TABLE_MAX) { fprintf(stderr, "the environment table holds %u entries, above the engine's capacity of %u\n", OAK_ENV_TABLE_COUNT, OAK_ENV_TABLE_MAX); exit(1); }
    for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) {
        const EnvSpec *env = &TABLE[i];
        if (env->obs == 0u || env->obs > OAK_OBSERVATION_LIMIT || env->actions == 0u || env->actions > OAK_ENV_ACT_MAX || env->poseLen > OAK_POSE_LEN || env->seeds == 0u || env->seeds > 64u) {
            fprintf(stderr, "%s exceeds the engine's stream capacity (obs %u/%u, actions %u/%u, pose %u/%u)\n", env->name,
                    env->obs, OAK_OBSERVATION_LIMIT, env->actions, OAK_ENV_ACT_MAX, env->poseLen, OAK_POSE_LEN);
            exit(1);
        }
        for (uint32_t j = 0u; j < i; j++)
            if (strcmp(TABLE[j].name, env->name) == 0 || (TABLE[j].seedStart < env->seedStart + env->seeds && env->seedStart < TABLE[j].seedStart + TABLE[j].seeds)) {
                fprintf(stderr, "%s and %s collide in name or seed block\n", TABLE[j].name, env->name);
                exit(1);
            }
    }
}
