#include "geometry.h"

#define OAK_FAMILY_LIST(F) \
    F(access, decision, noShared, noView) \
    F(acrobot, control, noShared, noView) \
    F(alchemy, puzzles, noShared, noView) \
    F(aliased, memory, noShared, aliased_view) \
    F(ancients, games, noShared, ancients_view) \
    F(backgammon, games, backgammon_shared, noView) \
    F(arm2, control, noShared, noView) \
    F(bandit, decision, noShared, noView) \
    F(blocking, navigation, noShared, noView) \
    F(calibration, decision, noShared, noView) \
    F(cartpole, control, noShared, noView) \
    F(catch, control, noShared, noView) \
    F(cellular, prediction, noShared, noView) \
    F(chess, games, chess_shared, noView) \
    F(walk, navigation, noShared, noView) \
    F(cliff, navigation, noShared, noView) \
    F(coinswap, decision, noShared, noView) \
    F(conditioning, prediction, noShared, noView) \
    F(connect4, games, noShared, noView) \
    F(contextual, decision, noShared, noView) \
    F(corridor, navigation, noShared, noView) \
    F(count, memory, noShared, noView) \
    F(covariate, prediction, noShared, noView) \
    F(craftax, world, craftax_shared, craftax_view) \
    F(darkkey, memory, noShared, noView) \
    F(deepsea, navigation, noShared, noView) \
    F(eleven, games, noShared, eleven_view) \
    F(dungeon, world, noShared, dungeon_view) \
    F(forager, world, noShared, forager_view) \
    F(hall, navigation, noShared, noView) \
    F(frost, prediction, noShared, noView) \
    F(g2048, puzzles, g2048_shared, noView) \
    F(garnet, decision, noShared, garnet_view) \
    F(generals, games, noShared, generals_view) \
    F(hanabi, games, noShared, noView) \
    F(hanoi, puzzles, noShared, noView) \
    F(hashed, decision, noShared, noView) \
    F(hiddenmode, decision, noShared, noView) \
    F(homeostasis, world, noShared, noView) \
    F(jbw, world, jbw_shared, noView) \
    F(keydoor, memory, noShared, noView) \
    F(lightcycles, games, noShared, noView) \
    F(lbf, games, noShared, noView) \
    F(maze, navigation, noShared, noView) \
    F(microrts, games, noShared, microrts_view) \
    F(tmaze, memory, noShared, noView) \
    F(minatar, control, noShared, minatar_view) \
    F(minesweeper, puzzles, noShared, minesweeper_view) \
    F(mountainCar, control, noShared, noView) \
    F(nexting, prediction, noShared, noView) \
    F(operant, decision, noShared, noView) \
    F(overcooked, games, overcooked_shared, noView) \
    F(parity, memory, noShared, noView) \
    F(password, memory, password_shared, password_view) \
    F(patience, memory, patience_shared, noView) \
    F(pendulum, control, noShared, noView) \
    F(permuted, prediction, noShared, permuted_view) \
    F(planar, control, planar_shared, noView) \
    F(pocketcube, puzzles, pocketcube_shared, noView) \
    F(poker, games, poker_shared, noView) \
    F(pommerman, games, noShared, pommerman_view) \
    F(puddle, control, noShared, noView) \
    F(quadrotor, control, quadrotor_shared, noView) \
    F(recite, memory, noShared, recite_view) \
    F(regression, prediction, noShared, noView) \
    F(relicfog, games, relicfog_shared, relicfog_view) \
    F(repeat, memory, noShared, noView) \
    F(reversi, games, noShared, noView) \
    F(ring, prediction, noShared, noView) \
    F(riverswim, navigation, noShared, noView) \
    F(rps, decision, noShared, noView) \
    F(rware, games, rware_shared, noView) \
    F(scentpuck, control, noShared, noView) \
    F(slime, games, noShared, noView) \
    F(skirmish, games, noShared, skirmish_view) \
    F(sokoban, navigation, sokoban_shared, sokoban_view) \
    F(subsetparity, prediction, noShared, noView) \
    F(supervised, prediction, noShared, supervised_view) \
    F(taxi, navigation, noShared, noView) \
    F(tetris, puzzles, noShared, noView) \
    F(tiger, decision, noShared, noView) \
    F(tracking, control, noShared, noView) \
    F(umbrella, memory, noShared, noView) \
    F(walkbet, prediction, walkbet_shared, noView)

#define OAK_ENV_LIST(X) \
    X(ACCESS_FIXED, access, 0u, "AccessControl-cont-v0", 6u, 2u, 0u, 6144u, 2u, access_extraWords) \
    X(ACCESS_DRIFT, access, 1u, "AccessControl-drift-cont-v0", 6u, 2u, 0u, 6208u, 2u, access_extraWords) \
    X(ACROBOT, acrobot, 0u, "Acrobot-cont-v0", 7u, 3u, 0u, 256u, 4u, acrobot_extraWords) \
    X(ACROBOT_SLIP, acrobot, 1u, "Acrobot-slip-cont-v0", 7u, 3u, 0u, 8320u, 4u, acrobot_extraWords) \
    X(ACROBOT_POSITION, acrobot, 2u, "Acrobot-position-cont-v0", 7u, 3u, 0u, 8384u, 4u, acrobot_extraWords) \
    X(ALCHEMY_FIXED, alchemy, 0u, "SymbolicAlchemy-fixed-cont-v0", 18u, 6u, 0u, 15680u, 12u, alchemy_extraWords) \
    X(ALCHEMY_ROTATION, alchemy, 1u, "SymbolicAlchemy-rotation-cont-v0", 18u, 6u, 0u, 15744u, 12u, alchemy_extraWords) \
    X(ALIASED_8, aliased, 0u, "AliasedRing-8-cont-v0", 2u, 2u, 0u, 5824u, 2u, aliased_extraWords) \
    X(ALIASED_32, aliased, 1u, "AliasedRing-32-cont-v0", 2u, 2u, 0u, 5888u, 2u, aliased_extraWords) \
    X(ALIASED_128, aliased, 2u, "AliasedRing-128-cont-v0", 2u, 2u, 0u, 5952u, 2u, aliased_extraWords) \
    X(ANCIENTS_1V1, ancients, 0u, "Ancients-1v1-cont-v0", 18u, 6u, 1u, 8704u, 6u, ancients_extraWords) \
    X(ANCIENTS_1V1_FOG, ancients, 1u, "Ancients-1v1-fog-cont-v0", 18u, 6u, 1u, 8768u, 6u, ancients_extraWords) \
    X(ANCIENTS_3V3, ancients, 2u, "Ancients-3v3-cont-v0", 18u, 6u, 1u, 8832u, 6u, ancients_extraWords) \
    X(ANCIENTS_5V5, ancients, 3u, "Ancients-5v5-cont-v0", 18u, 6u, 1u, 8896u, 6u, ancients_extraWords) \
    X(ANCIENTS_5V5_LEAGUE, ancients, 4u, "Ancients-5v5-league-cont-v0", 18u, 6u, 1u, 8960u, 6u, ancients_extraWords) \
    X(BACKGAMMON_RANDOM, backgammon, 0u, "Backgammon-random-cont-v0", 18u, 5u, 0u, 11456u, 18u, backgammon_extraWords) \
    X(BACKGAMMON_PUBEVAL, backgammon, 1u, "Backgammon-pubeval-cont-v0", 18u, 5u, 0u, 11520u, 18u, backgammon_extraWords) \
    X(BACKGAMMON_LEAGUE, backgammon, 2u, "Backgammon-league-cont-v0", 18u, 5u, 0u, 11584u, 18u, backgammon_extraWords) \
    X(ARM2_REACH, arm2, 0u, "Arm2-reach-cont-v0", 11u, 2u, 1u, 13632u, 8u, arm2_extraWords) \
    X(ARM2_PAYLOAD, arm2, 1u, "Arm2-payload-cont-v0", 11u, 2u, 1u, 13696u, 8u, arm2_extraWords) \
    X(BANDIT_3, bandit, 0u, "Bandit-3-cont-v0", 1u, 3u, 0u, 1088u, 3u, bandit_extraWords) \
    X(BANDIT_6, bandit, 1u, "Bandit-6-cont-v0", 1u, 6u, 0u, 1152u, 6u, bandit_extraWords) \
    X(BANDIT_3_DRIFT, bandit, 2u, "Bandit-3-drift-cont-v0", 1u, 3u, 0u, 1216u, 3u, bandit_extraWords) \
    X(BANDIT_6_DRIFT, bandit, 3u, "Bandit-6-drift-cont-v0", 1u, 6u, 0u, 1280u, 6u, bandit_extraWords) \
    X(BLOCKINGMAZE, blocking, 0u, "BlockingMaze-cont-v0", 8u, 4u, 0u, 3840u, 2u, blocking_extraWords) \
    X(SHORTCUTMAZE, blocking, 1u, "ShortcutMaze-cont-v0", 8u, 4u, 0u, 3904u, 2u, blocking_extraWords) \
    X(CALIBRATION_30, calibration, 0u, "Calibration-30-cont-v0", 2u, 2u, 0u, 2112u, 1u, noExtra) \
    X(CALIBRATION_80, calibration, 1u, "Calibration-80-cont-v0", 2u, 2u, 0u, 2176u, 1u, noExtra) \
    X(CARTPOLE, cartpole, 0u, "CartPoleSwingUp-cont-v0", 6u, 3u, 0u, 7168u, 4u, cartpole_extraWords) \
    X(CARTPOLE_SLIP, cartpole, 1u, "CartPoleSwingUp-slip-cont-v0", 6u, 3u, 0u, 7232u, 4u, cartpole_extraWords) \
    X(CATCH_5_10, catch, 0u, "Catch-5x10-cont-v0", 4u, 3u, 0u, 3136u, 3u, catch_extraWords) \
    X(CATCH_10_20, catch, 1u, "Catch-10x20-cont-v0", 4u, 3u, 0u, 3200u, 3u, catch_extraWords) \
    X(CATCH_NOISE, catch, 2u, "Catch-noise-cont-v0", 4u, 3u, 0u, 3264u, 3u, catch_extraWords) \
    X(CATCH_SCALE, catch, 3u, "Catch-scale-cont-v0", 4u, 3u, 0u, 3328u, 3u, catch_extraWords) \
    X(CELLULAR, cellular, 0u, "Cellular-cont-v0", 17u, 2u, 0u, 7808u, 1u, cellular_extraWords) \
    X(CHESS_KQK, chess, 0u, "Chess-kqk-cont-v0", 18u, 6u, 0u, 11008u, 14u, chess_extraWords) \
    X(CHESS_KRK, chess, 1u, "Chess-krk-cont-v0", 18u, 6u, 0u, 11072u, 14u, chess_extraWords) \
    X(CHESS_RANDOM, chess, 2u, "Chess-random-cont-v0", 18u, 6u, 0u, 11136u, 14u, chess_extraWords) \
    X(CHESS_ALPHABETA, chess, 3u, "Chess-alphabeta-cont-v0", 18u, 6u, 0u, 11200u, 14u, chess_extraWords) \
    X(CHAIN, walk, 0u, "Chain-cont-v0", 2u, 2u, 0u, 512u, 2u, noExtra) \
    X(DELAYEDCHAIN, walk, 1u, "DelayedChain-cont-v0", 3u, 2u, 0u, 960u, 2u, noExtra) \
    X(CLIFF, cliff, 0u, "CliffWalk-cont-v0", 7u, 4u, 0u, 6464u, 2u, cliff_extraWords) \
    X(CLIFF_SLIP, cliff, 1u, "CliffWalk-slip-cont-v0", 7u, 4u, 0u, 6528u, 2u, cliff_extraWords) \
    X(COINSWAP_SLOW, coinswap, 0u, "CoinSwap-slow-cont-v0", 1u, 2u, 0u, 1344u, 1u, coinswap_extraWords) \
    X(COINSWAP_FAST, coinswap, 1u, "CoinSwap-fast-cont-v0", 1u, 2u, 0u, 1408u, 1u, coinswap_extraWords) \
    X(TRACE_10, conditioning, 0u, "TraceConditioning-10-cont-v0", 5u, 2u, 0u, 4672u, 3u, conditioning_extraWords) \
    X(TRACE_100, conditioning, 1u, "TraceConditioning-100-cont-v0", 5u, 2u, 0u, 4736u, 3u, conditioning_extraWords) \
    X(PATTERNING, conditioning, 2u, "TracePatterning-cont-v0", 5u, 2u, 0u, 4800u, 3u, conditioning_extraWords) \
    X(CONNECT4_RANDOM, connect4, 0u, "ConnectFour-random-cont-v0", 17u, 3u, 0u, 10368u, 9u, connect4_extraWords) \
    X(CONNECT4_GREEDY, connect4, 1u, "ConnectFour-greedy-cont-v0", 17u, 3u, 0u, 10432u, 9u, connect4_extraWords) \
    X(CONNECT4_SEARCH, connect4, 2u, "ConnectFour-search-cont-v0", 17u, 3u, 0u, 10496u, 9u, connect4_extraWords) \
    X(CONNECT4_LEAGUE, connect4, 3u, "ConnectFour-league-cont-v0", 17u, 3u, 0u, 10560u, 9u, connect4_extraWords) \
    X(CONTEXTUAL_8, contextual, 0u, "Contextual-8-cont-v0", 9u, 4u, 0u, 1472u, 0u, contextual_extraWords) \
    X(CONTEXTUAL_16, contextual, 1u, "Contextual-16-cont-v0", 17u, 4u, 0u, 1536u, 0u, contextual_extraWords) \
    X(CONTEXTUAL_8_DRIFT, contextual, 2u, "Contextual-8-drift-cont-v0", 9u, 4u, 0u, 1600u, 0u, contextual_extraWords) \
    X(CONTEXTUAL_16_DRIFT, contextual, 3u, "Contextual-16-drift-cont-v0", 17u, 4u, 0u, 1664u, 0u, contextual_extraWords) \
    X(KEYCORRIDOR_1, corridor, 0u, "KeyCorridor-1-cont-v0", 14u, 4u, 0u, 192u, 8u, corridor_extraWords) \
    X(KEYCORRIDOR_3, corridor, 1u, "KeyCorridor-3-cont-v0", 17u, 4u, 0u, 640u, 10u, corridor_extraWords) \
    X(COUNT_8, count, 0u, "CountRecall-8-cont-v0", 3u, 4u, 0u, 5440u, 4u, count_extraWords) \
    X(COUNT_32, count, 1u, "CountRecall-32-cont-v0", 3u, 4u, 0u, 5504u, 4u, count_extraWords) \
    X(COVARIATE_FIXED, covariate, 0u, "Covariate-fixed-cont-v0", 5u, 2u, 0u, 4544u, 5u, covariate_extraWords) \
    X(COVARIATE_SHIFT, covariate, 1u, "Covariate-shift-cont-v0", 5u, 2u, 0u, 4608u, 5u, covariate_extraWords) \
    X(CRAFTAX_CLASSIC, craftax, 0u, "Craftax-Classic-cont-v0", 18u, 6u, 0u, 14784u, 25u, craftax_extraWords) \
    X(DARKKEY_9, darkkey, 0u, "DarkKeyToDoor-9-cont-v0", 2u, 5u, 0u, 15808u, 4u, noExtra) \
    X(DEEPSEA, deepsea, 0u, "DeepSea-8-cont-v0", 8u, 2u, 0u, 128u, 5u, deepsea_extraWords) \
    X(LARGEDEEPSEA, deepsea, 1u, "DeepSea-16-cont-v0", 8u, 2u, 0u, 576u, 11u, deepsea_extraWords) \
    X(DEEPSEA_32, deepsea, 2u, "DeepSea-32-cont-v0", 8u, 2u, 0u, 8000u, 35u, deepsea_extraWords) \
    X(DEEPSEA_16_SLIP, deepsea, 3u, "DeepSea-16-slip-cont-v0", 8u, 2u, 0u, 8064u, 11u, deepsea_extraWords) \
    X(ELEVEN_DRILL, eleven, 0u, "Eleven-drill-cont-v0", 18u, 6u, 1u, 13824u, 7u, eleven_extraWords) \
    X(ELEVEN_1V1, eleven, 1u, "Eleven-1v1-cont-v0", 18u, 6u, 1u, 13888u, 7u, eleven_extraWords) \
    X(ELEVEN_3V3, eleven, 2u, "Eleven-3v3-cont-v0", 18u, 6u, 1u, 13952u, 7u, eleven_extraWords) \
    X(ELEVEN_11V11_HARD, eleven, 3u, "Eleven-11v11-hard-cont-v0", 18u, 6u, 1u, 14016u, 7u, eleven_extraWords) \
    X(ELEVEN_11V11_LEAGUE, eleven, 4u, "Eleven-11v11-league-cont-v0", 18u, 6u, 1u, 14080u, 7u, eleven_extraWords) \
    X(DUNGEON_6_LIT, dungeon, 0u, "Dungeon-6-lit-cont-v0", 18u, 6u, 0u, 14848u, 29u, dungeon_extraWords) \
    X(DUNGEON_6_DARK, dungeon, 1u, "Dungeon-6-dark-cont-v0", 18u, 6u, 0u, 14912u, 29u, dungeon_extraWords) \
    X(DUNGEON_6_SHUFFLE, dungeon, 2u, "Dungeon-6-shuffle-cont-v0", 18u, 6u, 0u, 14976u, 29u, dungeon_extraWords) \
    X(DUNGEON_9_SHUFFLE, dungeon, 3u, "Dungeon-9-shuffle-cont-v0", 18u, 6u, 0u, 15040u, 29u, dungeon_extraWords) \
    X(FORAGER_16_1, forager, 0u, "Forager-16-1-cont-v0", 10u, 4u, 0u, 7296u, 3u, forager_extraWords) \
    X(FORAGER_64_1, forager, 1u, "Forager-64-1-cont-v0", 10u, 4u, 0u, 7360u, 3u, forager_extraWords) \
    X(FORAGER_16_2, forager, 2u, "Forager-16-2-cont-v0", 26u, 4u, 0u, 7424u, 3u, forager_extraWords) \
    X(FORAGER_16_SWITCH, forager, 3u, "Forager-16-1-switch-cont-v0", 10u, 4u, 0u, 7488u, 3u, forager_extraWords) \
    X(HALL_FIXED, hall, 0u, "FourRoomsHall-cont-v0", 14u, 4u, 0u, 6592u, 5u, hall_extraWords) \
    X(HALL_MOVING, hall, 1u, "FourRoomsHall-moving-cont-v0", 14u, 4u, 0u, 6656u, 5u, hall_extraWords) \
    X(G2048, g2048, 0u, "2048-cont-v0", 17u, 4u, 0u, 9984u, 4u, noExtra) \
    X(FOURROOMS, hall, 2u, "FourRooms-cont-v0", 17u, 4u, 0u, 64u, 6u, hall_extraWords) \
    X(FROSTHOLLOW_FIXED, frost, 0u, "FrostHollow-fixed-cont-v0", 10u, 3u, 0u, 15488u, 7u, frost_extraWords) \
    X(FROSTHOLLOW_RANDOM, frost, 1u, "FrostHollow-random-cont-v0", 10u, 3u, 0u, 15552u, 7u, frost_extraWords) \
    X(FROSTHOLLOW_DRIFT, frost, 2u, "FrostHollow-drift-cont-v0", 10u, 3u, 0u, 15616u, 7u, frost_extraWords) \
    X(GARNET_SMALL, garnet, 0u, "Garnet-10-2-2-cont-v0", 5u, 2u, 0u, 6784u, 1u, garnet_extraWords) \
    X(GARNET_LARGE, garnet, 1u, "Garnet-50-4-3-cont-v0", 7u, 4u, 0u, 6848u, 1u, garnet_extraWords) \
    X(GARNET_SMALL_RESAMPLE, garnet, 2u, "Garnet-10-2-2-resample-cont-v0", 5u, 2u, 0u, 6912u, 1u, garnet_extraWords) \
    X(GARNET_LARGE_RESAMPLE, garnet, 3u, "Garnet-50-4-3-resample-cont-v0", 7u, 4u, 0u, 6976u, 1u, garnet_extraWords) \
    X(GENERALS_2_12, generals, 0u, "Generals-2-12-cont-v0", 18u, 6u, 0u, 9024u, 9u, generals_extraWords) \
    X(GENERALS_2_23, generals, 1u, "Generals-2-23-cont-v0", 18u, 6u, 0u, 9088u, 9u, generals_extraWords) \
    X(GENERALS_4_28, generals, 2u, "Generals-4-28-cont-v0", 18u, 6u, 0u, 9152u, 9u, generals_extraWords) \
    X(GENERALS_8_32, generals, 3u, "Generals-8-32-cont-v0", 18u, 6u, 0u, 9216u, 9u, generals_extraWords) \
    X(GENERALS_8_32_LEAGUE, generals, 4u, "Generals-8-32-league-cont-v0", 18u, 6u, 0u, 9280u, 9u, generals_extraWords) \
    X(HANABI_2P_LITERAL, hanabi, 0u, "Hanabi-2p-literal-cont-v0", 18u, 6u, 0u, 13056u, 32u, hanabi_extraWords) \
    X(HANABI_2P_FOCUS, hanabi, 1u, "Hanabi-2p-focus-cont-v0", 18u, 6u, 0u, 13120u, 32u, hanabi_extraWords) \
    X(HANABI_4P_FOCUS, hanabi, 2u, "Hanabi-4p-focus-cont-v0", 18u, 6u, 0u, 13184u, 32u, hanabi_extraWords) \
    X(HANOI_4, hanoi, 0u, "Hanoi-4-cont-v0", 6u, 6u, 0u, 9664u, 4u, hanoi_extraWords) \
    X(HANOI_8, hanoi, 1u, "Hanoi-8-cont-v0", 10u, 6u, 0u, 9728u, 4u, hanoi_extraWords) \
    X(HANOI_8_REMAP, hanoi, 2u, "Hanoi-8-remap-cont-v0", 10u, 6u, 0u, 9792u, 5u, hanoi_extraWords) \
    X(HASHED_8_LOCAL, hashed, 0u, "HashedMDP-8-local-cont-v0", 9u, 4u, 0u, 7552u, 1u, hashed_extraWords) \
    X(HASHED_16_LOCAL, hashed, 1u, "HashedMDP-16-local-cont-v0", 17u, 4u, 0u, 7616u, 1u, hashed_extraWords) \
    X(HASHED_8_GLOBAL, hashed, 2u, "HashedMDP-8-global-cont-v0", 9u, 4u, 0u, 7680u, 1u, hashed_extraWords) \
    X(HASHED_16_GLOBAL, hashed, 3u, "HashedMDP-16-global-cont-v0", 17u, 4u, 0u, 7744u, 1u, hashed_extraWords) \
    X(HIDDENMODE_SLOW, hiddenmode, 0u, "HiddenMode-slow-cont-v0", 1u, 2u, 0u, 6016u, 1u, hiddenmode_extraWords) \
    X(HIDDENMODE_FAST, hiddenmode, 1u, "HiddenMode-fast-cont-v0", 1u, 2u, 0u, 6080u, 1u, hiddenmode_extraWords) \
    X(HOMEOSTASIS_2, homeostasis, 0u, "Homeostasis-2-cont-v0", 17u, 4u, 0u, 15872u, 13u, noExtra) \
    X(HOMEOSTASIS_2_HIDDEN, homeostasis, 1u, "Homeostasis-2-hidden-cont-v0", 15u, 4u, 0u, 15936u, 13u, noExtra) \
    X(JELLYBEANWORLD, jbw, 0u, "JellyBeanWorld-cont-v0", 16u, 3u, 0u, 1024u, 9u, jbw_extraWords) \
    X(JELLYBEANWORLD_8, jbw, 1u, "JellyBeanWorld-8-cont-v0", 16u, 3u, 0u, 8576u, 9u, jbw_extraWords) \
    X(KEYDOOR_8, keydoor, 0u, "KeyDoor-8-cont-v0", 4u, 2u, 0u, 2944u, 2u, keydoor_extraWords) \
    X(KEYDOOR_64, keydoor, 1u, "KeyDoor-64-cont-v0", 4u, 2u, 0u, 3008u, 2u, keydoor_extraWords) \
    X(KEYDOOR_512, keydoor, 2u, "KeyDoor-512-cont-v0", 4u, 2u, 0u, 3072u, 2u, keydoor_extraWords) \
    X(LIGHTCYCLES_RANDOM, lightcycles, 0u, "LightCycles-random-cont-v0", 17u, 3u, 0u, 11648u, 22u, lightcycles_extraWords) \
    X(LIGHTCYCLES_SPACE, lightcycles, 1u, "LightCycles-space-cont-v0", 17u, 3u, 0u, 11712u, 22u, lightcycles_extraWords) \
    X(LIGHTCYCLES_MINIMAX, lightcycles, 2u, "LightCycles-minimax-cont-v0", 17u, 3u, 0u, 11776u, 22u, lightcycles_extraWords) \
    X(LBF_8X8_2P_2F, lbf, 0u, "LBF-8x8-2p-2f-cont-v0", 12u, 6u, 0u, 12608u, 11u, lbf_extraWords) \
    X(LBF_8X8_2P_2F_COOP, lbf, 1u, "LBF-8x8-2p-2f-coop-cont-v0", 12u, 6u, 0u, 12672u, 11u, lbf_extraWords) \
    X(LBF_2S_8X8_2P_2F_COOP, lbf, 2u, "LBF-2s-8x8-2p-2f-coop-cont-v0", 12u, 6u, 0u, 12736u, 11u, lbf_extraWords) \
    X(LBF_10X10_3P_3F_PEN, lbf, 3u, "LBF-10x10-3p-3f-pen-cont-v0", 18u, 6u, 0u, 12800u, 11u, lbf_extraWords) \
    X(PROCEDURALMAZE_5, maze, 0u, "ProceduralMaze-5-cont-v0", 17u, 4u, 0u, 768u, 10u, maze_extraWords) \
    X(PROCEDURALMAZE_15, maze, 1u, "ProceduralMaze-15-cont-v0", 17u, 4u, 0u, 8640u, 34u, maze_extraWords) \
    X(POMAZE, maze, 2u, "POMaze-cont-v0", 15u, 4u, 0u, 896u, 7u, maze_extraWords) \
    X(HIGHDIMMAZE, maze, 3u, "HighDimMaze-cont-v0", 16u, 6u, 0u, 832u, 7u, maze_extraWords) \
    X(MICRORTS_8_RANDOM, microrts, 0u, "MicroRTS-8-random-cont-v0", 18u, 6u, 0u, 9344u, 14u, microrts_extraWords) \
    X(MICRORTS_8_WORKERRUSH, microrts, 1u, "MicroRTS-8-workerrush-cont-v0", 18u, 6u, 0u, 9408u, 14u, microrts_extraWords) \
    X(MICRORTS_16_LIGHTRUSH, microrts, 2u, "MicroRTS-16-lightrush-cont-v0", 18u, 6u, 0u, 9472u, 14u, microrts_extraWords) \
    X(MICRORTS_16_FOG, microrts, 3u, "MicroRTS-16-fog-cont-v0", 18u, 6u, 0u, 9536u, 14u, microrts_extraWords) \
    X(MICRORTS_16_LEAGUE, microrts, 4u, "MicroRTS-16-league-cont-v0", 18u, 6u, 0u, 9600u, 14u, microrts_extraWords) \
    X(TMAZE_4_1, tmaze, 0u, "TMaze-4-1-cont-v0", 6u, 2u, 0u, 4992u, 4u, tmaze_extraWords) \
    X(TMAZE_16_1, tmaze, 1u, "TMaze-16-1-cont-v0", 6u, 2u, 0u, 5056u, 4u, tmaze_extraWords) \
    X(MEMORYCHAIN, tmaze, 2u, "TMaze-32-1-cont-v0", 6u, 2u, 0u, 704u, 4u, tmaze_extraWords) \
    X(TMAZE_64_1, tmaze, 3u, "TMaze-64-1-cont-v0", 6u, 2u, 0u, 5120u, 4u, tmaze_extraWords) \
    X(TMAZE_16_2, tmaze, 4u, "TMaze-16-2-cont-v0", 7u, 4u, 0u, 5184u, 4u, tmaze_extraWords) \
    X(TMAZE_64_2, tmaze, 5u, "TMaze-64-2-cont-v0", 7u, 4u, 0u, 5248u, 4u, tmaze_extraWords) \
    X(MINATAR_BREAKOUT, minatar, 0u, "MinAtar-Breakout-cont-v0", 15u, 3u, 0u, 14464u, 4u, minatar_extraWords) \
    X(MINATAR_SPACE_INVADERS, minatar, 1u, "MinAtar-SpaceInvaders-cont-v0", 14u, 4u, 0u, 14528u, 14u, minatar_extraWords) \
    X(MINATAR_FREEWAY, minatar, 2u, "MinAtar-Freeway-cont-v0", 17u, 3u, 0u, 14592u, 8u, minatar_extraWords) \
    X(MINATAR_ASTERIX, minatar, 3u, "MinAtar-Asterix-cont-v0", 18u, 5u, 0u, 14656u, 5u, minatar_extraWords) \
    X(MINATAR_SEAQUEST, minatar, 4u, "MinAtar-Seaquest-cont-v0", 30u, 6u, 0u, 14720u, 6u, minatar_extraWords) \
    X(MINESWEEPER_9_9_10, minesweeper, 0u, "Minesweeper-9x9-10-cont-v0", 16u, 6u, 0u, 10176u, 6u, minesweeper_extraWords) \
    X(MINESWEEPER_16_16_40, minesweeper, 1u, "Minesweeper-16x16-40-cont-v0", 16u, 6u, 0u, 10240u, 6u, minesweeper_extraWords) \
    X(MINESWEEPER_30_16_99, minesweeper, 2u, "Minesweeper-30x16-99-cont-v0", 16u, 6u, 0u, 10304u, 6u, minesweeper_extraWords) \
    X(MOUNTAINCAR, mountainCar, 0u, "MountainCar-cont-v0", 3u, 3u, 0u, 320u, 2u, mountainCar_extraWords) \
    X(MOUNTAINCAR_SLIP, mountainCar, 1u, "MountainCar-slip-cont-v0", 3u, 3u, 0u, 8192u, 2u, mountainCar_extraWords) \
    X(MOUNTAINCAR_POSITION, mountainCar, 2u, "MountainCar-position-cont-v0", 3u, 3u, 0u, 8256u, 2u, mountainCar_extraWords) \
    X(NEXTING_1, nexting, 0u, "Nexting-1-cont-v0", 4u, 2u, 0u, 4864u, 1u, nexting_extraWords) \
    X(NEXTING_10, nexting, 1u, "Nexting-10-cont-v0", 4u, 2u, 0u, 4928u, 1u, nexting_extraWords) \
    X(OPERANT_16, operant, 0u, "Operant-16-cont-v0", 2u, 2u, 0u, 1728u, 1u, operant_extraWords) \
    X(OPERANT_64, operant, 1u, "Operant-64-cont-v0", 2u, 2u, 0u, 1792u, 1u, operant_extraWords) \
    X(OVERCOOKED_CRAMPED, overcooked, 0u, "OvercookedAI-cramped-cont-v0", 17u, 6u, 0u, 12352u, 8u, noExtra) \
    X(OVERCOOKED_ASYMMETRIC, overcooked, 1u, "OvercookedAI-asymmetric-cont-v0", 17u, 6u, 0u, 12416u, 8u, noExtra) \
    X(OVERCOOKED_RING, overcooked, 2u, "OvercookedAI-ring-cont-v0", 17u, 6u, 0u, 12480u, 8u, noExtra) \
    X(OVERCOOKED_CIRCUIT, overcooked, 3u, "OvercookedAI-circuit-cont-v0", 17u, 6u, 0u, 12544u, 8u, noExtra) \
    X(PARITY_4, parity, 0u, "Parity-4-cont-v0", 2u, 2u, 0u, 3392u, 3u, parity_extraWords) \
    X(PARITY_64, parity, 1u, "Parity-64-cont-v0", 2u, 2u, 0u, 3456u, 3u, parity_extraWords) \
    X(PASSWORD_2, password, 0u, "Password-2-cont-v0", 2u, 4u, 0u, 3648u, 1u, password_extraWords) \
    X(PASSWORD_4, password, 1u, "Password-4-cont-v0", 2u, 4u, 0u, 3712u, 1u, password_extraWords) \
    X(PASSWORD_6, password, 2u, "Password-6-cont-v0", 2u, 4u, 0u, 3776u, 1u, password_extraWords) \
    X(PATIENCE_5, patience, 0u, "Patience-5-cont-v0", 2u, 2u, 0u, 2624u, 1u, noExtra) \
    X(PATIENCE_50, patience, 1u, "Patience-50-cont-v0", 2u, 2u, 0u, 2688u, 1u, noExtra) \
    X(PATIENCE_500, patience, 2u, "Patience-500-cont-v0", 2u, 2u, 0u, 2752u, 1u, noExtra) \
    X(PENDULUM, pendulum, 0u, "Pendulum-cont-v0", 4u, 1u, 1u, 448u, 2u, pendulum_extraWords) \
    X(PENDULUM_SLIP, pendulum, 1u, "Pendulum-slip-cont-v0", 4u, 1u, 1u, 8448u, 2u, pendulum_extraWords) \
    X(PENDULUM_POSITION, pendulum, 2u, "Pendulum-position-cont-v0", 4u, 1u, 1u, 8512u, 2u, pendulum_extraWords) \
    X(PERMUTED_FIXED, permuted, 0u, "Permuted-fixed-cont-v0", 9u, 4u, 0u, 4416u, 1u, permuted_extraWords) \
    X(PERMUTED_RECODE, permuted, 1u, "Permuted-recode-cont-v0", 9u, 4u, 0u, 4480u, 1u, permuted_extraWords) \
    X(HALFCHEETAH, planar, 0u, "HalfCheetah-cont-v0", 17u, 6u, 1u, 0u, 29u, noExtra) \
    X(HALFCHEETAH_FRICTION, planar, 1u, "HalfCheetah-friction-cont-v0", 17u, 6u, 1u, 13248u, 29u, noExtra) \
    X(REACHER, planar, 2u, "Reacher-cont-v0", 10u, 2u, 1u, 13312u, 32u, noExtra) \
    X(INVERTED_DOUBLE_PENDULUM, planar, 3u, "InvertedDoublePendulum-cont-v0", 9u, 1u, 1u, 13376u, 36u, noExtra) \
    X(HOPPER, planar, 4u, "Hopper-cont-v0", 11u, 3u, 1u, 13440u, 37u, noExtra) \
    X(POCKETCUBE_3, pocketcube, 0u, "PocketCube-3-cont-v0", 15u, 6u, 0u, 9856u, 3u, pocketcube_extraWords) \
    X(POCKETCUBE_RANDOM, pocketcube, 1u, "PocketCube-random-cont-v0", 15u, 6u, 0u, 9920u, 3u, pocketcube_extraWords) \
    X(KUHNPOKER_RANDOM, poker, 0u, "KuhnPoker-random-cont-v0", 16u, 2u, 0u, 11264u, 3u, poker_extraWords) \
    X(LEDUCPOKER_CFR, poker, 1u, "LeducPoker-cfr-cont-v0", 16u, 3u, 0u, 11328u, 3u, poker_extraWords) \
    X(LEDUCPOKER_FICTITIOUS, poker, 2u, "LeducPoker-fictitious-cont-v0", 16u, 3u, 0u, 11392u, 4u, poker_extraWords) \
    X(POMMERMAN_1V1, pommerman, 0u, "Pommerman-1v1-cont-v0", 18u, 6u, 0u, 12032u, 27u, pommerman_extraWords) \
    X(POMMERMAN_FFA, pommerman, 1u, "Pommerman-FFA-cont-v0", 18u, 6u, 0u, 12096u, 27u, pommerman_extraWords) \
    X(POMMERMAN_TEAM, pommerman, 2u, "Pommerman-Team-cont-v0", 18u, 6u, 0u, 12160u, 27u, pommerman_extraWords) \
    X(POMMERMAN_FFA_COLLAPSE, pommerman, 3u, "Pommerman-FFA-collapse-cont-v0", 18u, 6u, 0u, 12224u, 27u, pommerman_extraWords) \
    X(POMMERMAN_FFA_LEAGUE, pommerman, 4u, "Pommerman-FFA-league-cont-v0", 18u, 6u, 0u, 12288u, 27u, pommerman_extraWords) \
    X(PUDDLE, puddle, 0u, "PuddleWorld-cont-v0", 3u, 4u, 0u, 6400u, 2u, puddle_extraWords) \
    X(QUADROTOR2D_HOVER, quadrotor, 0u, "Quadrotor2D-hover-cont-v0", 10u, 2u, 1u, 13504u, 8u, quadrotor_extraWords) \
    X(QUADROTOR2D_GUST, quadrotor, 1u, "Quadrotor2D-gust-cont-v0", 10u, 2u, 1u, 13568u, 10u, quadrotor_extraWords) \
    X(RECITE_4, recite, 0u, "Recite-4-cont-v0", 7u, 4u, 0u, 5568u, 2u, recite_extraWords) \
    X(RECITE_16, recite, 1u, "Recite-16-cont-v0", 7u, 4u, 0u, 5632u, 2u, recite_extraWords) \
    X(REGRESSION_8, regression, 0u, "Regression-8-cont-v0", 9u, 5u, 0u, 4032u, 8u, regression_extraWords) \
    X(REGRESSION_16, regression, 1u, "Regression-16-cont-v0", 17u, 5u, 0u, 4096u, 16u, regression_extraWords) \
    X(RELICFOG_SCOUT, relicfog, 0u, "RelicFog-scout-cont-v0", 18u, 6u, 0u, 15104u, 14u, relicfog_extraWords) \
    X(RELICFOG_SAPPER, relicfog, 1u, "RelicFog-sapper-cont-v0", 18u, 6u, 0u, 15168u, 14u, relicfog_extraWords) \
    X(RELICFOG_LEAGUE, relicfog, 2u, "RelicFog-league-cont-v0", 18u, 6u, 0u, 15232u, 14u, relicfog_extraWords) \
    X(REPEAT_4, repeat, 0u, "Repeat-4-cont-v0", 5u, 4u, 0u, 5312u, 3u, repeat_extraWords) \
    X(REPEAT_32, repeat, 1u, "Repeat-32-cont-v0", 5u, 4u, 0u, 5376u, 3u, repeat_extraWords) \
    X(REVERSI_6_GREEDY, reversi, 0u, "Reversi-6-greedy-cont-v0", 17u, 5u, 0u, 10752u, 11u, reversi_extraWords) \
    X(REVERSI_8_SWH, reversi, 1u, "Reversi-8-swh-cont-v0", 17u, 5u, 0u, 10816u, 11u, reversi_extraWords) \
    X(REVERSI_8_ALPHABETA, reversi, 2u, "Reversi-8-alphabeta-cont-v0", 17u, 5u, 0u, 10880u, 11u, reversi_extraWords) \
    X(REVERSI_8_LEAGUE, reversi, 3u, "Reversi-8-league-cont-v0", 17u, 5u, 0u, 10944u, 11u, reversi_extraWords) \
    X(RING_2, ring, 0u, "Ring-2-cont-v0", 2u, 2u, 0u, 1984u, 1u, noExtra) \
    X(RING_8, ring, 1u, "Ring-8-cont-v0", 2u, 2u, 0u, 2048u, 1u, noExtra) \
    X(RIVERSWIM_6, riverswim, 0u, "RiverSwim-6-cont-v0", 4u, 2u, 0u, 3520u, 1u, riverswim_extraWords) \
    X(RIVERSWIM_24, riverswim, 1u, "RiverSwim-24-cont-v0", 6u, 2u, 0u, 3584u, 1u, riverswim_extraWords) \
    X(RPS_10, rps, 0u, "RockPaperScissors-10-cont-v0", 4u, 3u, 0u, 7872u, 13u, noExtra) \
    X(RPS_100, rps, 1u, "RockPaperScissors-100-cont-v0", 4u, 3u, 0u, 7936u, 13u, noExtra) \
    X(RWARE_TINY_2AG, rware, 0u, "RWARE-tiny-2ag-cont-v0", 15u, 5u, 0u, 12864u, 10u, rware_extraWords) \
    X(RWARE_TINY_4AG, rware, 1u, "RWARE-tiny-4ag-cont-v0", 15u, 5u, 0u, 12928u, 13u, rware_extraWords) \
    X(RWARE_MEDIUM_4AG_HARD, rware, 2u, "RWARE-medium-4ag-hard-cont-v0", 15u, 5u, 0u, 12992u, 16u, rware_extraWords) \
    X(SCENTPUCK_ONE, scentpuck, 0u, "ScentPuck-cont-v0", 5u, 5u, 0u, 6272u, 6u, scentpuck_extraWords) \
    X(SCENTPUCK_TWO, scentpuck, 1u, "ScentPuck-competing-cont-v0", 5u, 5u, 0u, 6336u, 6u, scentpuck_extraWords) \
    X(SLIME_EVOLVED, slime, 0u, "SlimeVolley-evolved-cont-v0", 12u, 6u, 0u, 11840u, 24u, slime_extraWords) \
    X(SLIME_POSITION, slime, 1u, "SlimeVolley-position-cont-v0", 6u, 6u, 0u, 11904u, 24u, slime_extraWords) \
    X(SLIME_LEAGUE, slime, 2u, "SlimeVolley-league-cont-v0", 12u, 6u, 0u, 11968u, 24u, slime_extraWords) \
    X(SKIRMISH_5V5, skirmish, 0u, "Skirmish-5v5-cont-v0", 18u, 6u, 0u, 14144u, 14u, skirmish_extraWords) \
    X(SKIRMISH_5V5_RANDOM, skirmish, 1u, "Skirmish-5v5-random-cont-v0", 18u, 6u, 0u, 14208u, 14u, skirmish_extraWords) \
    X(SKIRMISH_10V11, skirmish, 2u, "Skirmish-10v11-cont-v0", 18u, 6u, 0u, 14272u, 14u, skirmish_extraWords) \
    X(SKIRMISH_20V23, skirmish, 3u, "Skirmish-20v23-cont-v0", 18u, 6u, 0u, 14336u, 14u, skirmish_extraWords) \
    X(SKIRMISH_10V10_LEAGUE, skirmish, 4u, "Skirmish-10v10-league-cont-v0", 18u, 6u, 0u, 14400u, 14u, skirmish_extraWords) \
    X(SOKOBAN, sokoban, 0u, "Sokoban-cont-v0", 7u, 4u, 0u, 3968u, 7u, sokoban_extraWords) \
    X(SOKOBAN_7X7_2, sokoban, 1u, "Sokoban-7x7-2-cont-v0", 17u, 5u, 0u, 10624u, 5u, sokoban_extraWords) \
    X(SOKOBAN_10X10_4, sokoban, 2u, "Sokoban-10x10-4-cont-v0", 17u, 5u, 0u, 10688u, 5u, sokoban_extraWords) \
    X(SUBSETPARITY_8_1, subsetparity, 0u, "SubsetParity-8-1-cont-v0", 9u, 2u, 0u, 4160u, 2u, subsetparity_extraWords) \
    X(SUBSETPARITY_16_1, subsetparity, 1u, "SubsetParity-16-1-cont-v0", 17u, 2u, 0u, 4224u, 2u, subsetparity_extraWords) \
    X(SUBSETPARITY_8_2, subsetparity, 2u, "SubsetParity-8-2-cont-v0", 9u, 2u, 0u, 4288u, 2u, subsetparity_extraWords) \
    X(SUBSETPARITY_16_2, subsetparity, 3u, "SubsetParity-16-2-cont-v0", 17u, 2u, 0u, 4352u, 2u, subsetparity_extraWords) \
    X(IDBD_20_5, supervised, 0u, "IDBD-20-5-cont-v0", 21u, 1u, 1u, 15296u, 3u, supervised_extraWords) \
    X(SLOWREGRESSION_20_15, supervised, 1u, "SlowRegression-20-15-cont-v0", 21u, 1u, 1u, 15360u, 3u, supervised_extraWords) \
    X(RATETRACKING, supervised, 2u, "RateTracking-cont-v0", 2u, 1u, 1u, 15424u, 3u, supervised_extraWords) \
    X(TAXI, taxi, 0u, "Taxi-cont-v0", 16u, 6u, 0u, 6720u, 4u, taxi_extraWords) \
    X(TETRIS_A0, tetris, 0u, "Tetris-NES-A0-cont-v0", 18u, 4u, 1u, 10048u, 20u, tetris_extraWords) \
    X(TETRIS_A19, tetris, 1u, "Tetris-NES-A19-cont-v0", 18u, 4u, 1u, 10112u, 20u, tetris_extraWords) \
    X(TIGER_85, tiger, 0u, "Tiger-85-cont-v0", 3u, 3u, 0u, 5696u, 4u, tiger_extraWords) \
    X(TIGER_60, tiger, 1u, "Tiger-60-cont-v0", 3u, 3u, 0u, 5760u, 4u, tiger_extraWords) \
    X(TRACKING_99, tracking, 0u, "Tracking-99-cont-v0", 2u, 1u, 1u, 1856u, 1u, tracking_extraWords) \
    X(TRACKING_999, tracking, 1u, "Tracking-999-cont-v0", 2u, 1u, 1u, 1920u, 1u, tracking_extraWords) \
    X(UMBRELLA_4, umbrella, 0u, "Umbrella-4-cont-v0", 4u, 2u, 0u, 2240u, 3u, umbrella_extraWords) \
    X(UMBRELLA_16, umbrella, 1u, "Umbrella-16-cont-v0", 4u, 2u, 0u, 2304u, 3u, umbrella_extraWords) \
    X(UMBRELLA_64, umbrella, 2u, "Umbrella-64-cont-v0", 4u, 2u, 0u, 2368u, 3u, umbrella_extraWords) \
    X(UMBRELLA_4_NOISE, umbrella, 3u, "Umbrella-4-noise-cont-v0", 16u, 2u, 0u, 2432u, 3u, umbrella_extraWords) \
    X(UMBRELLA_16_NOISE, umbrella, 4u, "Umbrella-16-noise-cont-v0", 16u, 2u, 0u, 2496u, 3u, umbrella_extraWords) \
    X(UMBRELLA_64_NOISE, umbrella, 5u, "Umbrella-64-noise-cont-v0", 16u, 2u, 0u, 2560u, 3u, umbrella_extraWords) \
    X(WALKBET_FIXED, walkbet, 0u, "WalkBet-cont-v0", 2u, 2u, 0u, 7040u, 1u, walkbet_extraWords) \
    X(WALKBET_DRIFT, walkbet, 1u, "WalkBet-drift-cont-v0", 2u, 2u, 0u, 7104u, 1u, walkbet_extraWords)

#define OAK_FAMILY_PROTOTYPES(family, category, shared, view) \
    void family##_reset(uint32_t instance, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps); \
    float family##_step(uint32_t instance, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared); \
    void family##_observe(uint32_t instance, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared); \
    void family##_oracle(uint32_t instance, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared);
OAK_FAMILY_LIST(OAK_FAMILY_PROTOTYPES)
#define OAK_FAMILY_CATEGORY(family, category, shared, view) static const char *const CATEGORY_##family = #category;
OAK_FAMILY_LIST(OAK_FAMILY_CATEGORY)
uint32_t access_extraWords(uint32_t instance, uint32_t steps);
uint32_t acrobot_extraWords(uint32_t instance, uint32_t steps);
uint32_t alchemy_extraWords(uint32_t instance, uint32_t steps);
uint32_t aliased_extraWords(uint32_t instance, uint32_t steps);
uint32_t ancients_extraWords(uint32_t instance, uint32_t steps);
uint32_t arm2_extraWords(uint32_t instance, uint32_t steps);
uint32_t backgammon_extraWords(uint32_t instance, uint32_t steps);
uint32_t bandit_extraWords(uint32_t instance, uint32_t steps);
uint32_t blocking_extraWords(uint32_t instance, uint32_t steps);
uint32_t cartpole_extraWords(uint32_t instance, uint32_t steps);
uint32_t catch_extraWords(uint32_t instance, uint32_t steps);
uint32_t cellular_extraWords(uint32_t instance, uint32_t steps);
uint32_t chess_extraWords(uint32_t instance, uint32_t steps);
uint32_t cliff_extraWords(uint32_t instance, uint32_t steps);
uint32_t coinswap_extraWords(uint32_t instance, uint32_t steps);
uint32_t conditioning_extraWords(uint32_t instance, uint32_t steps);
uint32_t connect4_extraWords(uint32_t instance, uint32_t steps);
uint32_t contextual_extraWords(uint32_t instance, uint32_t steps);
uint32_t corridor_extraWords(uint32_t instance, uint32_t steps);
uint32_t count_extraWords(uint32_t instance, uint32_t steps);
uint32_t covariate_extraWords(uint32_t instance, uint32_t steps);
uint32_t craftax_extraWords(uint32_t instance, uint32_t steps);
uint32_t deepsea_extraWords(uint32_t instance, uint32_t steps);
uint32_t dungeon_extraWords(uint32_t instance, uint32_t steps);
uint32_t eleven_extraWords(uint32_t instance, uint32_t steps);
uint32_t forager_extraWords(uint32_t instance, uint32_t steps);
uint32_t frost_extraWords(uint32_t instance, uint32_t steps);
uint32_t garnet_extraWords(uint32_t instance, uint32_t steps);
uint32_t generals_extraWords(uint32_t instance, uint32_t steps);
uint32_t hall_extraWords(uint32_t instance, uint32_t steps);
uint32_t hanabi_extraWords(uint32_t instance, uint32_t steps);
uint32_t hanoi_extraWords(uint32_t instance, uint32_t steps);
uint32_t hashed_extraWords(uint32_t instance, uint32_t steps);
uint32_t hiddenmode_extraWords(uint32_t instance, uint32_t steps);
uint32_t jbw_extraWords(uint32_t instance, uint32_t steps);
uint32_t keydoor_extraWords(uint32_t instance, uint32_t steps);
uint32_t lbf_extraWords(uint32_t instance, uint32_t steps);
uint32_t lightcycles_extraWords(uint32_t instance, uint32_t steps);
uint32_t maze_extraWords(uint32_t instance, uint32_t steps);
uint32_t microrts_extraWords(uint32_t instance, uint32_t steps);
uint32_t minatar_extraWords(uint32_t instance, uint32_t steps);
uint32_t minesweeper_extraWords(uint32_t instance, uint32_t steps);
uint32_t mountainCar_extraWords(uint32_t instance, uint32_t steps);
uint32_t nexting_extraWords(uint32_t instance, uint32_t steps);
uint32_t operant_extraWords(uint32_t instance, uint32_t steps);
uint32_t parity_extraWords(uint32_t instance, uint32_t steps);
uint32_t password_extraWords(uint32_t instance, uint32_t steps);
uint32_t pendulum_extraWords(uint32_t instance, uint32_t steps);
uint32_t permuted_extraWords(uint32_t instance, uint32_t steps);
uint32_t pocketcube_extraWords(uint32_t instance, uint32_t steps);
uint32_t poker_extraWords(uint32_t instance, uint32_t steps);
uint32_t pommerman_extraWords(uint32_t instance, uint32_t steps);
uint32_t puddle_extraWords(uint32_t instance, uint32_t steps);
uint32_t quadrotor_extraWords(uint32_t instance, uint32_t steps);
uint32_t recite_extraWords(uint32_t instance, uint32_t steps);
uint32_t regression_extraWords(uint32_t instance, uint32_t steps);
uint32_t relicfog_extraWords(uint32_t instance, uint32_t steps);
uint32_t repeat_extraWords(uint32_t instance, uint32_t steps);
uint32_t reversi_extraWords(uint32_t instance, uint32_t steps);
uint32_t riverswim_extraWords(uint32_t instance, uint32_t steps);
uint32_t rware_extraWords(uint32_t instance, uint32_t steps);
uint32_t scentpuck_extraWords(uint32_t instance, uint32_t steps);
uint32_t skirmish_extraWords(uint32_t instance, uint32_t steps);
uint32_t slime_extraWords(uint32_t instance, uint32_t steps);
uint32_t sokoban_extraWords(uint32_t instance, uint32_t steps);
uint32_t subsetparity_extraWords(uint32_t instance, uint32_t steps);
uint32_t supervised_extraWords(uint32_t instance, uint32_t steps);
uint32_t taxi_extraWords(uint32_t instance, uint32_t steps);
uint32_t tetris_extraWords(uint32_t instance, uint32_t steps);
uint32_t tiger_extraWords(uint32_t instance, uint32_t steps);
uint32_t tmaze_extraWords(uint32_t instance, uint32_t steps);
uint32_t tracking_extraWords(uint32_t instance, uint32_t steps);
uint32_t umbrella_extraWords(uint32_t instance, uint32_t steps);
uint32_t walkbet_extraWords(uint32_t instance, uint32_t steps);
uint32_t ancients_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t craftax_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t dungeon_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t eleven_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t generals_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t microrts_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t minesweeper_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t pommerman_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t relicfog_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t skirmish_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t sokoban_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t supervised_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t aliased_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t forager_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t garnet_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t minatar_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t password_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t permuted_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint32_t recite_view(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out);
uint64_t backgammon_shared(float *table);
uint64_t chess_shared(float *table);
uint64_t craftax_shared(float *table);
uint64_t g2048_shared(float *table);
uint64_t jbw_shared(float *table);
uint64_t overcooked_shared(float *table);
uint64_t password_shared(float *table);
uint64_t patience_shared(float *table);
uint64_t planar_shared(float *table);
uint64_t pocketcube_shared(float *table);
uint64_t poker_shared(float *table);
uint64_t quadrotor_shared(float *table);
uint64_t relicfog_shared(float *table);
uint64_t rware_shared(float *table);
uint64_t sokoban_shared(float *table);
uint64_t walkbet_shared(float *table);

static inline uint32_t noExtra(uint32_t instance, uint32_t steps) { (void)instance; (void)steps; return 0u; }
static inline uint64_t noShared(float *table) { (void)table; return 0u; }
static inline uint32_t noView(uint32_t instance, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) { (void)instance; (void)words; (void)extra; (void)shared; (void)out; return 0u; }

#define OAK_FAMILY_ENUM(family, category, shared, view) FAMILY_##family,
enum { OAK_FAMILY_LIST(OAK_FAMILY_ENUM) OAK_FAMILY_COUNT };

#define OAK_ENV_ENUM(sym, family, instance, name, obs, actions, box, seedStart, words_, extra) ENV_##sym,
enum { OAK_ENV_LIST(OAK_ENV_ENUM) OAK_ENV_COUNT };

void env_reset(uint32_t env, uint32_t seed, uint32_t *words, uint32_t *extra, uint32_t steps) {
#define OAK_ENV_RESET(sym, family, instance, name, obs, actions, box, seedStart, words_, extra_) case ENV_##sym: family##_reset(instance, seed, words, extra, steps); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_RESET) default: return; }
}

float env_step(uint32_t env, uint32_t seed, uint32_t step, const float *ctrl, uint32_t *words, uint32_t *extra, float *obs, uint32_t *done, const float *shared) {
#define OAK_ENV_STEP(sym, family, instance, name, obs_, actions, box, seedStart, words_, extra_) case ENV_##sym: return family##_step(instance, seed, step, ctrl, words, extra, obs, done, shared);
    switch (env) { OAK_ENV_LIST(OAK_ENV_STEP) default: *done = 0u; return 0.0f; }
}

void env_observe(uint32_t env, uint32_t seed, const uint32_t *words, const uint32_t *extra, uint32_t step, float *obs, const float *shared) {
#define OAK_ENV_OBSERVE(sym, family, instance, name, obs_, actions, box, seedStart, words_, extra_) case ENV_##sym: family##_observe(instance, seed, words, extra, step, obs, shared); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_OBSERVE) default: return; }
}

void env_oracle(uint32_t env, uint32_t seed, uint32_t step, const uint32_t *words, uint32_t *extra, const float *obs, float *ctrl, const float *shared) {
#define OAK_ENV_ORACLE(sym, family, instance, name, obs_, actions, box, seedStart, words_, extra_) case ENV_##sym: family##_oracle(instance, seed, step, words, extra, obs, ctrl, shared); return;
    switch (env) { OAK_ENV_LIST(OAK_ENV_ORACLE) default: return; }
}

uint32_t env_obs(uint32_t env) {
#define OAK_ENV_OBS(sym, family, instance, name, obs, actions, box, seedStart, words_, extra) case ENV_##sym: return obs;
    switch (env) { OAK_ENV_LIST(OAK_ENV_OBS) default: return 0u; }
}

uint32_t env_actions(uint32_t env) {
#define OAK_ENV_ACTIONS(sym, family, instance, name, obs, actions, box, seedStart, words_, extra) case ENV_##sym: return actions;
    switch (env) { OAK_ENV_LIST(OAK_ENV_ACTIONS) default: return 0u; }
}

uint32_t env_box(uint32_t env) {
#define OAK_ENV_BOX(sym, family, instance, name, obs, actions, box, seedStart, words_, extra) case ENV_##sym: return box;
    switch (env) { OAK_ENV_LIST(OAK_ENV_BOX) default: return 0u; }
}

static uint64_t familyShared(uint32_t family, float *table) {
#define OAK_FAMILY_SHARED(family, category, shared, view) case FAMILY_##family: return shared(table);
    switch (family) { OAK_FAMILY_LIST(OAK_FAMILY_SHARED) default: return 0u; }
}

static uint32_t envFamily(uint32_t env) {
#define OAK_ENV_FAMILY(sym, family, instance, name, obs, actions, box, seedStart, words_, extra) case ENV_##sym: return FAMILY_##family;
    switch (env) { OAK_ENV_LIST(OAK_ENV_FAMILY) default: return OAK_FAMILY_COUNT; }
}

static uint32_t envInstance(uint32_t env) {
#define OAK_ENV_INSTANCE(sym, family, instance, name, obs, actions, box, seedStart, words_, extra) case ENV_##sym: return instance;
    switch (env) { OAK_ENV_LIST(OAK_ENV_INSTANCE) default: return 0u; }
}

uint32_t env_view(uint32_t env, const uint32_t *words, const uint32_t *extra, const float *shared, uint32_t *out) {
#define OAK_FAMILY_VIEW(family, category, shared_, view) case FAMILY_##family: return view(envInstance(env), words, extra, shared, out);
    switch (envFamily(env)) { OAK_FAMILY_LIST(OAK_FAMILY_VIEW) default: return 0u; }
}

uint64_t env_sharedWords(void) {
    uint64_t words = 0u;
    for (uint32_t family = 0u; family < OAK_FAMILY_COUNT; family++) words += familyShared(family, 0);
    return words;
}

uint64_t env_sharedOffset(uint32_t env) {
    uint64_t offset = 0u;
    for (uint32_t family = 0u; family < envFamily(env); family++) offset += familyShared(family, 0);
    return offset;
}

uint32_t env_extraWords(uint32_t env, uint32_t steps) {
#define OAK_ENV_EXTRA(sym, family, instance, name, obs, actions, box, seedStart, words_, extraFn) case ENV_##sym: return extraFn(instance, steps);
    switch (env) { OAK_ENV_LIST(OAK_ENV_EXTRA) default: return 0u; }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OAK_ENV_SPEC(sym, family, instance, name_, obs_, actions_, box_, seedStart_, words_, extra) { name_, #family, CATEGORY_##family, instance, ENV_##sym, obs_, actions_, box_, seedStart_, words_ },
static const EnvSpec TABLE[] = { OAK_ENV_LIST(OAK_ENV_SPEC) };

const uint32_t OAK_ENV_TABLE_COUNT = sizeof TABLE / sizeof TABLE[0];

const EnvSpec *env_at(uint32_t index) { return &TABLE[index]; }

void env_buildShared(float *shared, const EnvSpec *const *specs, uint32_t count) {
    uint8_t built[OAK_FAMILY_COUNT] = {0};
    for (uint32_t i = 0u; i < count; i++) {
        uint32_t family = envFamily(specs[i]->id);
        if (built[family]) continue;
        built[family] = 1u;
        familyShared(family, shared + env_sharedOffset(specs[i]->id));
    }
}

const EnvSpec *env_named(const char *name) {
    for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++)
        if (strcmp(TABLE[i].name, name) == 0) return &TABLE[i];
    return 0;
}

void env_checkTable(void) {
    if (OAK_ENV_TABLE_COUNT > OAK_ENV_TABLE_MAX) { fprintf(stderr, "the environment table holds %u entries, above the engine's capacity of %u\n", OAK_ENV_TABLE_COUNT, OAK_ENV_TABLE_MAX); exit(1); }
    for (uint32_t i = 0u; i < OAK_ENV_TABLE_COUNT; i++) {
        const EnvSpec *env = &TABLE[i];
        if (env->obs == 0u || env->obs > OAK_OBSERVATION_LIMIT || env->actions == 0u || env->actions > OAK_ENV_ACT_MAX || env->words > OAK_ENV_WORDS_MAX) {
            fprintf(stderr, "%s exceeds the engine's stream capacity (obs %u/%u, actions %u/%u, words %u/%u)\n", env->name,
                    env->obs, OAK_OBSERVATION_LIMIT, env->actions, OAK_ENV_ACT_MAX, env->words, OAK_ENV_WORDS_MAX);
            exit(1);
        }
        for (uint32_t j = 0u; j < i; j++)
            if (strcmp(TABLE[j].name, env->name) == 0 || (TABLE[j].seedStart < env->seedStart + OAK_ENV_SEEDS && env->seedStart < TABLE[j].seedStart + OAK_ENV_SEEDS)) {
                fprintf(stderr, "%s and %s collide in name or seed block\n", TABLE[j].name, env->name);
                exit(1);
            }
    }
}
