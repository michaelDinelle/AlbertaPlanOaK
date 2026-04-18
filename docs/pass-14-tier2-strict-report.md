# Pass 14 strict-gate report

Bench dir: `.oak-benchmarks/pass14-tier2-ablation/baseline`
Baselines (multi-baseline strict gate): `.oak-benchmarks/20260418-151021-12678`, `.oak-benchmarks/20260418-164936-51887`, `.oak-benchmarks/pass14-tier0-ablation/baseline`
Envs: 84 | RR mode: final-heartbeat | Effect-size floor: 0.0003

## Strict-rule pass/fail (per the per-tier acceptance gate)

- (A) **Zero envs strictly worse vs ANY baseline** at p<0.05 (|Δ| ≥ 0.0003): **FAIL** (1 envs)
  - (Below-floor noise-level apparent regressions: 3 envs (|Δ| < 0.0003); not counted toward strict gate.)
- (C) **>=1 channel improves at p<0.05**: rR_win=9/84, LPM_pos=81/84, util_pos=0/84
- (D) LPM positive trend on >=85% of envs: **PASS**

## Headline counts

- rewardRate strict win (Δ vs random > 0 at p<0.05): **9/84** (0.107)
- rewardRate strict loss: 0/84
- P(Δ vs random > 0): 27/84 (0.321)
- LPM positive trend (p<0.05): 81/84 (0.964)
- utilFeat positive (p<0.05): 0/84 | utilFeat negative: 0/84
- Multi-signal Pareto PASS: 84/84 (1.000)
- **Strict regression vs ANY baseline**: 1/84
- **Non-zero rewardRate** (|finalRR| > 1e-3): 26/84

Non-zero envs (mean final rR): Humanoid-v4=+4.5052, HalfCheetah-v4=-0.3099, MiniGrid-Dynamic-Obstacles-5x5-v0=-0.1467, MiniGrid-Dynamic-Obstacles-Random-5x5-v0=-0.1278, MiniGrid-Dynamic-Obstacles-6x6-v0=-0.1272, MiniGrid-Dynamic-Obstacles-Random-6x6-v0=-0.1115, MiniGrid-Dynamic-Obstacles-8x8-v0=-0.0994, MiniGrid-Dynamic-Obstacles-16x16-v0=-0.0778, Walker2d-v4=+0.0523, MiniGrid-GoToObject-6x6-N2-v0=+0.0210, MiniGrid-GoToDoor-5x5-v0=+0.0115, MiniGrid-GoToObject-8x8-N2-v0=+0.0102, MiniGrid-GoToDoor-6x6-v0=+0.0086, MiniGrid-Fetch-5x5-N2-v0=+0.0086, MiniGrid-Empty-Random-5x5-v0=+0.0070, MiniGrid-GoToDoor-8x8-v0=+0.0061, MiniGrid-Fetch-6x6-N2-v0=+0.0056, MiniGrid-Empty-Random-6x6-v0=+0.0041, MiniGrid-Fetch-8x8-N3-v0=+0.0029, MiniGrid-MemoryS7-v0=+0.0025

## Per-env table

| env | rR Δ | rR p | LPM ρ | LPM p | util Δ | util p | pEff min | agent_ms | strict_regression |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v4 | +0.01728 | 0.007 | -1.000 | 1.000 | +0 | 1.000 | 11.0 | 2.448 | no |
| Humanoid-v4 | +0.00935 | 0.000 | -0.179 | 0.975 | +0 | 1.000 | 60.0 | 8.495 | YES (20260418-164936-51887: Δ-0.004875 p0.01, baseline: Δ-0.004724 p0.014) |
| MiniGrid-BlockedUnlockPickup-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.075 | no |
| MiniGrid-DistShift1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 33.0 | 3.045 | no |
| MiniGrid-DistShift2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 3.149 | no |
| MiniGrid-DoorKey-16x16-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 3.104 | no |
| MiniGrid-DoorKey-5x5-v0 | -0.00000 | 0.666 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.192 | no |
| MiniGrid-DoorKey-6x6-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.194 | no |
| MiniGrid-DoorKey-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 3.289 | no |
| MiniGrid-Dynamic-Obstacles-16x16-v0 | +0.00172 | 0.119 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 2.182 | no |
| MiniGrid-Dynamic-Obstacles-5x5-v0 | +0.00273 | 0.000 | +1.000 | 0.000 | +0 | 1.000 | 27.4 | 2.181 | no |
| MiniGrid-Dynamic-Obstacles-6x6-v0 | +0.00218 | 0.000 | +1.000 | 0.000 | +0 | 1.000 | 40.0 | 2.196 | no |
| MiniGrid-Dynamic-Obstacles-8x8-v0 | +0.00179 | 0.044 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 2.222 | no |
| MiniGrid-Dynamic-Obstacles-Random-5x5-v0 | -0.00038 | 0.641 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.251 | no |
| MiniGrid-Dynamic-Obstacles-Random-6x6-v0 | -0.00061 | 0.758 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 2.196 | no |
| MiniGrid-Empty-16x16-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 33.0 | 3.208 | no |
| MiniGrid-Empty-5x5-v0 | -0.00013 | 1.000 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 3.164 | no |
| MiniGrid-Empty-6x6-v0 | -0.00009 | 1.000 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 3.208 | no |
| MiniGrid-Empty-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 3.151 | no |
| MiniGrid-Empty-Random-5x5-v0 | +0.00020 | 0.038 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 3.183 | no |
| MiniGrid-Empty-Random-6x6-v0 | +0.00013 | 0.164 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 3.286 | no |
| MiniGrid-Fetch-5x5-N2-v0 | +0.00016 | 0.207 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 3.172 | no |
| MiniGrid-Fetch-6x6-N2-v0 | +0.00016 | 0.062 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 3.182 | no |
| MiniGrid-Fetch-8x8-N3-v0 | -0.00008 | 0.863 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 3.270 | no |
| MiniGrid-FourRooms-v0 | +0.00000 | 0.465 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 3.123 | no |
| MiniGrid-GoToDoor-5x5-v0 | +0.00039 | 0.043 | +1.000 | 0.000 | +0 | 1.000 | 25.2 | 3.034 | no |
| MiniGrid-GoToDoor-6x6-v0 | +0.00001 | 0.485 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 3.095 | no |
| MiniGrid-GoToDoor-8x8-v0 | +0.00006 | 0.317 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.122 | no |
| MiniGrid-GoToObject-6x6-N2-v0 | +0.00006 | 0.402 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.005 | no |
| MiniGrid-GoToObject-8x8-N2-v0 | +0.00021 | 0.203 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.108 | no |
| MiniGrid-KeyCorridorS3R1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 23.3 | 3.141 | no |
| MiniGrid-KeyCorridorS3R2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.1 | 3.039 | no |
| MiniGrid-KeyCorridorS3R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.040 | no |
| MiniGrid-KeyCorridorS4R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.145 | no |
| MiniGrid-KeyCorridorS5R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.3 | 3.221 | no |
| MiniGrid-KeyCorridorS6R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.1 | 3.100 | no |
| MiniGrid-LavaCrossingS11N5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 3.150 | no |
| MiniGrid-LavaCrossingS9N1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 3.044 | no |
| MiniGrid-LavaCrossingS9N2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 42.0 | 3.058 | no |
| MiniGrid-LavaCrossingS9N3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 40.0 | 3.118 | no |
| MiniGrid-LavaGapS5-v0 | -0.00007 | 0.951 | +1.000 | 0.000 | +0 | 1.000 | 42.0 | 3.057 | no |
| MiniGrid-LavaGapS6-v0 | -0.00002 | 0.967 | +1.000 | 0.000 | +0 | 1.000 | 42.0 | 3.053 | no |
| MiniGrid-LavaGapS7-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 41.0 | 3.088 | no |
| MiniGrid-LockedRoom-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 61.9 | 3.118 | no |
| MiniGrid-MemoryS11-v0 | +0.00000 | 0.488 | +1.000 | 0.000 | +0 | 1.000 | 48.0 | 3.047 | no |
| MiniGrid-MemoryS13-v0 | +0.00007 | 0.191 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.121 | no |
| MiniGrid-MemoryS13Random-v0 | +0.00011 | 0.175 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 3.143 | no |
| MiniGrid-MemoryS17Random-v0 | +0.00002 | 0.442 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.038 | no |
| MiniGrid-MemoryS7-v0 | +0.00009 | 0.311 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 3.276 | no |
| MiniGrid-MemoryS9-v0 | -0.00006 | 0.696 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 3.144 | no |
| MiniGrid-MultiRoom-N2-S4-v0 | -0.00003 | 0.903 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 3.059 | no |
| MiniGrid-MultiRoom-N4-S5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 3.054 | no |
| MiniGrid-MultiRoom-N6-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 3.159 | no |
| MiniGrid-ObstructedMaze-1Dl-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.082 | no |
| MiniGrid-ObstructedMaze-1Dlh-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.069 | no |
| MiniGrid-ObstructedMaze-1Dlhb-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.128 | no |
| MiniGrid-ObstructedMaze-1Q-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.100 | no |
| MiniGrid-ObstructedMaze-1Q-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 3.075 | no |
| MiniGrid-ObstructedMaze-2Dl-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.170 | no |
| MiniGrid-ObstructedMaze-2Dlh-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.137 | no |
| MiniGrid-ObstructedMaze-2Dlhb-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 3.052 | no |
| MiniGrid-ObstructedMaze-2Dlhb-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.106 | no |
| MiniGrid-ObstructedMaze-2Q-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 3.087 | no |
| MiniGrid-ObstructedMaze-2Q-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.042 | no |
| MiniGrid-ObstructedMaze-Full-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.073 | no |
| MiniGrid-ObstructedMaze-Full-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 3.158 | no |
| MiniGrid-Playground-v0 | +0.00000 | 0.500 | +0.620 | 0.001 | +0 | 1.000 | 67.8 | 3.051 | no |
| MiniGrid-PutNear-6x6-N2-v0 | -0.00001 | 0.656 | +1.000 | 0.000 | +0 | 1.000 | 36.7 | 3.064 | no |
| MiniGrid-PutNear-8x8-N3-v0 | -0.00000 | 0.557 | +1.000 | 0.000 | +0 | 1.000 | 35.1 | 3.170 | no |
| MiniGrid-RedBlueDoors-6x6-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 3.023 | no |
| MiniGrid-RedBlueDoors-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 39.0 | 3.041 | no |
| MiniGrid-SimpleCrossingS11N5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.173 | no |
| MiniGrid-SimpleCrossingS9N1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 3.092 | no |
| MiniGrid-SimpleCrossingS9N2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.051 | no |
| MiniGrid-SimpleCrossingS9N3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 3.201 | no |
| MiniGrid-Unlock-v0 | -0.00000 | 0.569 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 3.146 | no |
| MiniGrid-UnlockPickup-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.103 | no |
| MiniGrid-WFC-DungeonMazeScaled-v0 | -0.00003 | 0.745 | +1.000 | 0.000 | +0 | 1.000 | 36.0 | 3.215 | no |
| MiniGrid-WFC-MazeSimple-v0 | +0.00006 | 0.040 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 3.153 | no |
| MiniGrid-WFC-ObstaclesAngular-v0 | -0.00001 | 0.614 | +1.000 | 0.000 | +0 | 1.000 | 36.0 | 3.115 | no |
| MiniGrid-WFC-ObstaclesBlackdots-v0 | -0.00000 | 0.546 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 3.122 | no |
| MiniGrid-WFC-ObstaclesHogs3-v0 | +0.00001 | 0.402 | +1.000 | 0.000 | +0 | 1.000 | 37.0 | 3.165 | no |
| MiniGrid-WFC-RoomsFabric-v0 | +0.00001 | 0.425 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 3.085 | no |
| Walker2d-v4 | +0.05915 | 0.000 | -0.993 | 1.000 | +0 | 1.000 | 11.0 | 2.462 | no |