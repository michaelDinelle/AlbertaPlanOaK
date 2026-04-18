# Pass 14 strict-gate report

Bench dir: `.oak-benchmarks/pass14-tier1-ablation/baseline`
Baselines (multi-baseline strict gate): `.oak-benchmarks/20260418-151021-12678`, `.oak-benchmarks/20260418-164936-51887`, `.oak-benchmarks/pass14-tier2-ablation/baseline`
Envs: 84 | RR mode: final-heartbeat | Effect-size floor: 0.0003

## Strict-rule pass/fail (per the per-tier acceptance gate)

- (A) **Zero envs strictly worse vs ANY baseline** at p<0.05 (|Δ| ≥ 0.0003): **FAIL** (2 envs)
  - (Below-floor noise-level apparent regressions: 2 envs (|Δ| < 0.0003); not counted toward strict gate.)
- (C) **>=1 channel improves at p<0.05**: rR_win=7/84, LPM_pos=81/84, util_pos=0/84
- (D) LPM positive trend on >=85% of envs: **PASS**

## Headline counts

- rewardRate strict win (Δ vs random > 0 at p<0.05): **7/84** (0.083)
- rewardRate strict loss: 0/84
- P(Δ vs random > 0): 28/84 (0.333)
- LPM positive trend (p<0.05): 81/84 (0.964)
- utilFeat positive (p<0.05): 0/84 | utilFeat negative: 0/84
- Multi-signal Pareto PASS: 84/84 (1.000)
- **Strict regression vs ANY baseline**: 2/84
- **Non-zero rewardRate** (|finalRR| > 1e-3): 26/84

Non-zero envs (mean final rR): Humanoid-v4=+4.5049, HalfCheetah-v4=-0.3109, MiniGrid-Dynamic-Obstacles-5x5-v0=-0.1467, MiniGrid-Dynamic-Obstacles-Random-5x5-v0=-0.1279, MiniGrid-Dynamic-Obstacles-6x6-v0=-0.1271, MiniGrid-Dynamic-Obstacles-Random-6x6-v0=-0.1115, MiniGrid-Dynamic-Obstacles-8x8-v0=-0.0994, MiniGrid-Dynamic-Obstacles-16x16-v0=-0.0779, Walker2d-v4=+0.0520, MiniGrid-GoToObject-6x6-N2-v0=+0.0210, MiniGrid-GoToDoor-5x5-v0=+0.0115, MiniGrid-GoToObject-8x8-N2-v0=+0.0102, MiniGrid-GoToDoor-6x6-v0=+0.0086, MiniGrid-Fetch-5x5-N2-v0=+0.0086, MiniGrid-Empty-Random-5x5-v0=+0.0070, MiniGrid-GoToDoor-8x8-v0=+0.0061, MiniGrid-Fetch-6x6-N2-v0=+0.0056, MiniGrid-Empty-Random-6x6-v0=+0.0041, MiniGrid-Fetch-8x8-N3-v0=+0.0030, MiniGrid-MemoryS7-v0=+0.0025

## Per-env table

| env | rR Δ | rR p | LPM ρ | LPM p | util Δ | util p | pEff min | agent_ms | strict_regression |
|---|---|---|---|---|---|---|---|---|---|
| HalfCheetah-v4 | +0.01629 | 0.009 | -1.000 | 1.000 | +0 | 1.000 | 11.0 | 2.288 | YES (20260418-151021-12678: Δ-0.001622 p0.017) |
| Humanoid-v4 | +0.00905 | 0.000 | -0.161 | 0.968 | +0 | 1.000 | 58.0 | 6.356 | YES (20260418-151021-12678: Δ-0.0005119 p0.00397, 20260418-164936-51887: Δ-0.00517 p0.00717) |
| MiniGrid-BlockedUnlockPickup-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.637 | no |
| MiniGrid-DistShift1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 33.0 | 2.695 | no |
| MiniGrid-DistShift2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.665 | no |
| MiniGrid-DoorKey-16x16-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 2.672 | no |
| MiniGrid-DoorKey-5x5-v0 | -0.00000 | 0.666 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 2.681 | no |
| MiniGrid-DoorKey-6x6-v0 | +0.00000 | 0.067 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 2.701 | no |
| MiniGrid-DoorKey-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 47.0 | 2.655 | no |
| MiniGrid-Dynamic-Obstacles-16x16-v0 | +0.00165 | 0.121 | +1.000 | 0.000 | +0 | 1.000 | 57.4 | 2.069 | no |
| MiniGrid-Dynamic-Obstacles-5x5-v0 | +0.00270 | 0.000 | +1.000 | 0.000 | +0 | 1.000 | 28.6 | 2.052 | no |
| MiniGrid-Dynamic-Obstacles-6x6-v0 | +0.00223 | 0.000 | +1.000 | 0.000 | +0 | 1.000 | 39.8 | 2.064 | no |
| MiniGrid-Dynamic-Obstacles-8x8-v0 | +0.00178 | 0.038 | +1.000 | 0.000 | +0 | 1.000 | 52.8 | 2.044 | no |
| MiniGrid-Dynamic-Obstacles-Random-5x5-v0 | -0.00048 | 0.684 | +1.000 | 0.000 | +0 | 1.000 | 29.6 | 1.970 | no |
| MiniGrid-Dynamic-Obstacles-Random-6x6-v0 | -0.00059 | 0.787 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 2.097 | no |
| MiniGrid-Empty-16x16-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.660 | no |
| MiniGrid-Empty-5x5-v0 | -0.00011 | 0.989 | +1.000 | 0.000 | +0 | 1.000 | 29.0 | 2.701 | no |
| MiniGrid-Empty-6x6-v0 | -0.00009 | 1.000 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.676 | no |
| MiniGrid-Empty-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.686 | no |
| MiniGrid-Empty-Random-5x5-v0 | +0.00022 | 0.016 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.693 | no |
| MiniGrid-Empty-Random-6x6-v0 | +0.00008 | 0.263 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.669 | no |
| MiniGrid-Fetch-5x5-N2-v0 | +0.00015 | 0.212 | +1.000 | 0.000 | +0 | 1.000 | 32.0 | 2.653 | no |
| MiniGrid-Fetch-6x6-N2-v0 | +0.00014 | 0.073 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 2.655 | no |
| MiniGrid-Fetch-8x8-N3-v0 | -0.00002 | 0.596 | +1.000 | 0.000 | +0 | 1.000 | 31.0 | 2.666 | no |
| MiniGrid-FourRooms-v0 | +0.00001 | 0.428 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 2.656 | no |
| MiniGrid-GoToDoor-5x5-v0 | +0.00036 | 0.060 | +1.000 | 0.000 | +0 | 1.000 | 25.2 | 2.670 | no |
| MiniGrid-GoToDoor-6x6-v0 | +0.00007 | 0.365 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 2.637 | no |
| MiniGrid-GoToDoor-8x8-v0 | +0.00007 | 0.324 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.701 | no |
| MiniGrid-GoToObject-6x6-N2-v0 | +0.00014 | 0.298 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.682 | no |
| MiniGrid-GoToObject-8x8-N2-v0 | +0.00022 | 0.198 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 2.679 | no |
| MiniGrid-KeyCorridorS3R1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 23.2 | 2.707 | no |
| MiniGrid-KeyCorridorS3R2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.8 | 2.674 | no |
| MiniGrid-KeyCorridorS3R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.790 | no |
| MiniGrid-KeyCorridorS4R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.739 | no |
| MiniGrid-KeyCorridorS5R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 2.718 | no |
| MiniGrid-KeyCorridorS6R3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.1 | 2.746 | no |
| MiniGrid-LavaCrossingS11N5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 2.742 | no |
| MiniGrid-LavaCrossingS9N1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 41.0 | 2.759 | no |
| MiniGrid-LavaCrossingS9N2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 41.0 | 2.707 | no |
| MiniGrid-LavaCrossingS9N3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 42.0 | 2.742 | no |
| MiniGrid-LavaGapS5-v0 | -0.00008 | 0.974 | +1.000 | 0.000 | +0 | 1.000 | 40.0 | 2.718 | no |
| MiniGrid-LavaGapS6-v0 | -0.00002 | 0.928 | +1.000 | 0.000 | +0 | 1.000 | 41.0 | 2.714 | no |
| MiniGrid-LavaGapS7-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 2.711 | no |
| MiniGrid-LockedRoom-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 62.0 | 2.728 | no |
| MiniGrid-MemoryS11-v0 | +0.00003 | 0.381 | +1.000 | 0.000 | +0 | 1.000 | 47.0 | 2.716 | no |
| MiniGrid-MemoryS13-v0 | +0.00010 | 0.119 | +1.000 | 0.000 | +0 | 1.000 | 47.0 | 2.695 | no |
| MiniGrid-MemoryS13Random-v0 | +0.00012 | 0.142 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 2.636 | no |
| MiniGrid-MemoryS17Random-v0 | +0.00003 | 0.397 | +1.000 | 0.000 | +0 | 1.000 | 48.0 | 2.616 | no |
| MiniGrid-MemoryS7-v0 | +0.00012 | 0.240 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 2.631 | no |
| MiniGrid-MemoryS9-v0 | -0.00008 | 0.744 | +1.000 | 0.000 | +0 | 1.000 | 46.0 | 2.655 | no |
| MiniGrid-MultiRoom-N2-S4-v0 | -0.00003 | 0.917 | +1.000 | 0.000 | +0 | 1.000 | 33.7 | 2.679 | no |
| MiniGrid-MultiRoom-N4-S5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 34.0 | 2.689 | no |
| MiniGrid-MultiRoom-N6-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 40.0 | 2.621 | no |
| MiniGrid-ObstructedMaze-1Dl-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.625 | no |
| MiniGrid-ObstructedMaze-1Dlh-v0 | +0.00000 | 0.159 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.642 | no |
| MiniGrid-ObstructedMaze-1Dlhb-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.679 | no |
| MiniGrid-ObstructedMaze-1Q-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.817 | no |
| MiniGrid-ObstructedMaze-1Q-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.865 | no |
| MiniGrid-ObstructedMaze-2Dl-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.850 | no |
| MiniGrid-ObstructedMaze-2Dlh-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.829 | no |
| MiniGrid-ObstructedMaze-2Dlhb-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 2.836 | no |
| MiniGrid-ObstructedMaze-2Dlhb-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 2.842 | no |
| MiniGrid-ObstructedMaze-2Q-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.829 | no |
| MiniGrid-ObstructedMaze-2Q-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 25.0 | 2.829 | no |
| MiniGrid-ObstructedMaze-Full-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 26.0 | 2.848 | no |
| MiniGrid-ObstructedMaze-Full-v1 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 27.0 | 2.848 | no |
| MiniGrid-Playground-v0 | +0.00000 | 0.500 | +0.569 | 0.000 | +0 | 1.000 | 23.0 | 2.842 | no |
| MiniGrid-PutNear-6x6-N2-v0 | -0.00001 | 0.689 | +1.000 | 0.000 | +0 | 1.000 | 36.6 | 2.841 | no |
| MiniGrid-PutNear-8x8-N3-v0 | -0.00000 | 0.565 | +1.000 | 0.000 | +0 | 1.000 | 35.7 | 2.824 | no |
| MiniGrid-RedBlueDoors-6x6-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 2.881 | no |
| MiniGrid-RedBlueDoors-8x8-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 2.861 | no |
| MiniGrid-SimpleCrossingS11N5-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 43.0 | 2.846 | no |
| MiniGrid-SimpleCrossingS9N1-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 2.839 | no |
| MiniGrid-SimpleCrossingS9N2-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 45.0 | 2.875 | no |
| MiniGrid-SimpleCrossingS9N3-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 30.0 | 2.887 | no |
| MiniGrid-Unlock-v0 | -0.00001 | 0.698 | +1.000 | 0.000 | +0 | 1.000 | 23.0 | 2.879 | no |
| MiniGrid-UnlockPickup-v0 | +0.00000 | 0.500 | +1.000 | 0.000 | +0 | 1.000 | 28.0 | 2.870 | no |
| MiniGrid-WFC-DungeonMazeScaled-v0 | -0.00003 | 0.758 | +1.000 | 0.000 | +0 | 1.000 | 38.0 | 2.832 | no |
| MiniGrid-WFC-MazeSimple-v0 | +0.00005 | 0.062 | +1.000 | 0.000 | +0 | 1.000 | 37.0 | 2.828 | no |
| MiniGrid-WFC-ObstaclesAngular-v0 | -0.00000 | 0.533 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 2.771 | no |
| MiniGrid-WFC-ObstaclesBlackdots-v0 | +0.00000 | 0.496 | +1.000 | 0.000 | +0 | 1.000 | 36.0 | 2.700 | no |
| MiniGrid-WFC-ObstaclesHogs3-v0 | +0.00001 | 0.370 | +1.000 | 0.000 | +0 | 1.000 | 37.0 | 2.742 | no |
| MiniGrid-WFC-RoomsFabric-v0 | -0.00001 | 0.617 | +1.000 | 0.000 | +0 | 1.000 | 35.0 | 2.714 | no |
| Walker2d-v4 | +0.05877 | 0.000 | -0.990 | 1.000 | +0 | 1.000 | 11.0 | 2.263 | no |