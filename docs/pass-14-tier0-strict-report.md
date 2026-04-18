# Pass 14 strict-gate report

Bench dir: `.oak-benchmarks/pass14-tier0-ablation/baseline`
Baselines (multi-baseline strict gate): `.oak-benchmarks/20260418-151021-12678`, `.oak-benchmarks/20260418-164936-51887`
Envs: 84 | RR mode: final-heartbeat | Effect-size floor: 0.0003

## Strict-rule pass/fail (per the per-tier acceptance gate)

- (A) **Zero envs strictly worse vs ANY baseline** at p<0.05 (|Δ| ≥ 0.0003): **PASS**
  - (Below-floor noise-level apparent regressions: 6 envs (|Δ| < 0.0003); not counted toward strict gate.)
- (C) **>=1 channel improves at p<0.05**: rR_win=9/84, LPM_pos=81/84, util_pos=0/84
- (D) LPM positive trend on >=85% of envs: **PASS**

## Headline counts

- rewardRate strict win (Δ vs random > 0 at p<0.05): **9/84** (0.107)
- rewardRate strict loss: 0/84
- P(Δ vs random > 0): 25/84 (0.298)
- LPM positive trend (p<0.05): 81/84 (0.964)
- utilFeat positive (p<0.05): 0/84 | utilFeat negative: 0/84
- Multi-signal Pareto PASS: 84/84 (1.000)
- **Strict regression vs ANY baseline**: 0/84
- **Non-zero rewardRate** (|finalRR| > 1e-3): 26/84

Non-zero envs (mean final rR): Humanoid-v4=+4.5099, HalfCheetah-v4=-0.3085, MiniGrid-Dynamic-Obstacles-5x5-v0=-0.1465, MiniGrid-Dynamic-Obstacles-Random-5x5-v0=-0.1279, MiniGrid-Dynamic-Obstacles-6x6-v0=-0.1270, MiniGrid-Dynamic-Obstacles-Random-6x6-v0=-0.1114, MiniGrid-Dynamic-Obstacles-8x8-v0=-0.0991, MiniGrid-Dynamic-Obstacles-16x16-v0=-0.0775, Walker2d-v4=+0.0519, MiniGrid-GoToObject-6x6-N2-v0=+0.0210, MiniGrid-GoToDoor-5x5-v0=+0.0115, MiniGrid-GoToObject-8x8-N2-v0=+0.0102, MiniGrid-GoToDoor-6x6-v0=+0.0086, MiniGrid-Fetch-5x5-N2-v0=+0.0086, MiniGrid-Empty-Random-5x5-v0=+0.0070, MiniGrid-GoToDoor-8x8-v0=+0.0060, MiniGrid-Fetch-6x6-N2-v0=+0.0056, MiniGrid-Empty-Random-6x6-v0=+0.0040, MiniGrid-Fetch-8x8-N3-v0=+0.0030, MiniGrid-MemoryS7-v0=+0.0024

## Per-env table

| env | rR Δ | rR p | LPM ρ | LPM p | util Δ | util p | pEff min | agent_ms | strict_regression |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v4 | +0.01869 | 0.004 | -0.985 | 1.000 | +0 | 1.000 | 11.0 | 2.418 | no |
| Humanoid-v4 | +0.01407 | 0.000 | -0.688 | 1.000 | +0 | 1.000 | 49.5 | 15.948 | no |
| MiniGrid-BlockedUnlockPickup-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 5.306 | no |
| MiniGrid-DistShift1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 33.0 | 5.407 | no |
| MiniGrid-DistShift2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 5.308 | no |
| MiniGrid-DoorKey-16x16-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.312 | no |
| MiniGrid-DoorKey-5x5-v0 | -0.00000 | 0.666 | +1.000 | 0.000 | +0 | 1.000 | 39.0 | 5.304 | no |
| MiniGrid-DoorKey-6x6-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.215 | no |
| MiniGrid-DoorKey-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 47.0 | 5.227 | no |
| MiniGrid-Dynamic-Obstacles-16x16-v0 | +0.00203 | 0.034 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 3.263 | no |
| MiniGrid-Dynamic-Obstacles-5x5-v0 | +0.00288 | 0.000 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 3.268 | no |
| MiniGrid-Dynamic-Obstacles-6x6-v0 | +0.00242 | 0.000 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 3.261 | no |
| MiniGrid-Dynamic-Obstacles-8x8-v0 | +0.00208 | 0.006 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 3.220 | no |
| MiniGrid-Dynamic-Obstacles-Random-5x5-v0 | -0.00049 | 0.713 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 3.229 | no |
| MiniGrid-Dynamic-Obstacles-Random-6x6-v0 | -0.00053 | 0.775 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 3.206 | no |
| MiniGrid-Empty-16x16-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 5.281 | no |
| MiniGrid-Empty-5x5-v0 | -0.00011 | 0.997 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 5.435 | no |
| MiniGrid-Empty-6x6-v0 | -0.00008 | 0.999 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 5.750 | no |
| MiniGrid-Empty-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 33.0 | 5.422 | no |
| MiniGrid-Empty-Random-5x5-v0 | +0.00018 | 0.063 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 5.333 | no |
| MiniGrid-Empty-Random-6x6-v0 | +0.00007 | 0.292 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 5.363 | no |
| MiniGrid-Fetch-5x5-N2-v0 | +0.00013 | 0.237 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 5.414 | no |
| MiniGrid-Fetch-6x6-N2-v0 | +0.00018 | 0.048 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 5.402 | no |
| MiniGrid-Fetch-8x8-N3-v0 | -0.00002 | 0.584 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 5.390 | no |
| MiniGrid-FourRooms-v0 | -0.00001 | 0.563 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 5.394 | no |
| MiniGrid-GoToDoor-5x5-v0 | +0.00039 | 0.038 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.351 | no |
| MiniGrid-GoToDoor-6x6-v0 | +0.00003 | 0.434 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.372 | no |
| MiniGrid-GoToDoor-8x8-v0 | +0.00004 | 0.388 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 5.344 | no |
| MiniGrid-GoToObject-6x6-N2-v0 | +0.00013 | 0.298 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.380 | no |
| MiniGrid-GoToObject-8x8-N2-v0 | +0.00018 | 0.222 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.389 | no |
| MiniGrid-KeyCorridorS3R1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.3 | 5.345 | no |
| MiniGrid-KeyCorridorS3R2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 29.0 | 5.362 | no |
| MiniGrid-KeyCorridorS3R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 5.353 | no |
| MiniGrid-KeyCorridorS4R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 5.355 | no |
| MiniGrid-KeyCorridorS5R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 29.0 | 5.359 | no |
| MiniGrid-KeyCorridorS6R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.399 | no |
| MiniGrid-LavaCrossingS11N5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.392 | no |
| MiniGrid-LavaCrossingS9N1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 42.0 | 5.369 | no |
| MiniGrid-LavaCrossingS9N2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 39.0 | 5.676 | no |
| MiniGrid-LavaCrossingS9N3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.292 | no |
| MiniGrid-LavaGapS5-v0 | -0.00007 | 0.957 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 5.269 | no |
| MiniGrid-LavaGapS6-v0 | -0.00002 | 0.966 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 5.272 | no |
| MiniGrid-LavaGapS7-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 41.0 | 5.256 | no |
| MiniGrid-LockedRoom-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 62.3 | 5.424 | no |
| MiniGrid-MemoryS11-v0 | -0.00002 | 0.594 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.361 | no |
| MiniGrid-MemoryS13-v0 | +0.00005 | 0.259 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 5.305 | no |
| MiniGrid-MemoryS13Random-v0 | +0.00009 | 0.225 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.415 | no |
| MiniGrid-MemoryS17Random-v0 | -0.00001 | 0.530 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 5.402 | no |
| MiniGrid-MemoryS7-v0 | +0.00002 | 0.450 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.384 | no |
| MiniGrid-MemoryS9-v0 | -0.00012 | 0.843 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.275 | no |
| MiniGrid-MultiRoom-N2-S4-v0 | -0.00004 | 0.970 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 5.326 | no |
| MiniGrid-MultiRoom-N4-S5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 5.318 | no |
| MiniGrid-MultiRoom-N6-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 33.0 | 5.319 | no |
| MiniGrid-ObstructedMaze-1Dl-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.268 | no |
| MiniGrid-ObstructedMaze-1Dlh-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.272 | no |
| MiniGrid-ObstructedMaze-1Dlhb-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.294 | no |
| MiniGrid-ObstructedMaze-1Q-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.272 | no |
| MiniGrid-ObstructedMaze-1Q-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.264 | no |
| MiniGrid-ObstructedMaze-2Dl-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.294 | no |
| MiniGrid-ObstructedMaze-2Dlh-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.259 | no |
| MiniGrid-ObstructedMaze-2Dlhb-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.270 | no |
| MiniGrid-ObstructedMaze-2Dlhb-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.243 | no |
| MiniGrid-ObstructedMaze-2Q-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.277 | no |
| MiniGrid-ObstructedMaze-2Q-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.263 | no |
| MiniGrid-ObstructedMaze-Full-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 5.271 | no |
| MiniGrid-ObstructedMaze-Full-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 5.260 | no |
| MiniGrid-Playground-v0 | +0.00000 | 0.500 | +0.835 | 0.000 | +0 | 1.000 | 68.0 | 5.257 | no |
| MiniGrid-PutNear-6x6-N2-v0 | -0.00001 | 0.746 | +1.000 | 0.000 | +0 | 1.000 | 37.3 | 5.323 | no |
| MiniGrid-PutNear-8x8-N3-v0 | -0.00000 | 0.509 | +1.000 | 0.000 | +0 | 1.000 | 35.5 | 5.326 | no |
| MiniGrid-RedBlueDoors-6x6-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 5.320 | no |
| MiniGrid-RedBlueDoors-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 5.278 | no |
| MiniGrid-SimpleCrossingS11N5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 41.0 | 5.252 | no |
| MiniGrid-SimpleCrossingS9N1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 5.238 | no |
| MiniGrid-SimpleCrossingS9N2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 44.0 | 5.320 | no |
| MiniGrid-SimpleCrossingS9N3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 5.452 | no |
| MiniGrid-Unlock-v0 | -0.00001 | 0.689 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 5.432 | no |
| MiniGrid-UnlockPickup-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 5.349 | no |
| MiniGrid-WFC-DungeonMazeScaled-v0 | -0.00003 | 0.772 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 5.336 | no |
| MiniGrid-WFC-MazeSimple-v0 | +0.00004 | 0.119 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 5.216 | no |
| MiniGrid-WFC-ObstaclesAngular-v0 | -0.00002 | 0.755 | +1.000 | 0.000 | +0 | 1.000 | 36.0 | 5.364 | no |
| MiniGrid-WFC-ObstaclesBlackdots-v0 | +0.00000 | 0.471 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 5.307 | no |
| MiniGrid-WFC-ObstaclesHogs3-v0 | +0.00001 | 0.329 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 5.295 | no |
| MiniGrid-WFC-RoomsFabric-v0 | -0.00001 | 0.573 | +1.000 | 0.000 | +0 | 1.000 | 39.0 | 5.832 | no |
| Walker2d-v4 | +0.05872 | 0.000 | -0.982 | 1.000 | +0 | 1.000 | 11.0 | 2.453 | no |