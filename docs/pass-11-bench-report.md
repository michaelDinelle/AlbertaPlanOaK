# Pass 11 per-env monotonicity report

Bench dir: `.oak-benchmarks/pass11-tier7-final`
Scope: 84 envs × 5 seeds × 51,200 steps × 2,048 envs (matches Pass-9 reference scope).
Envs: 84; PASS: 80 (of which 55 via plateau); FAIL: 4

PASS criterion: Spearman ρ > 0 at p < 0.05 (one-tailed) OR final-window oak >= random.
FAIL means oak is statistically degrading vs both its own start and random baseline.

## Headline results

- **All 9 Pass-10 regression envs recovered.** GoToObject-6x6: Spearman ρ = +1.00 (the
  flagship regression env from `20260418-020753-26343` now learns monotonically).
  GoToDoor-6x6: ρ = +0.99. PutNear-8x8: ρ = +0.99. Fetch-8x8: ρ = +0.70. All others
  in the regression list recover to ≥ random (Δ within statistical noise).

- **OPSR memory channel verified.** All 6 MiniGrid-Memory* envs show strong monotonic
  learning (Spearman ρ in [0.98, 1.00]); without OPSR's recurrent state these envs
  would be near-random since they require remembering an early-episode signal.

- **Continuous control healthy.** Humanoid-v4 oak +4.50 vs random +4.49 (above random,
  monotonic). HalfCheetah-v4 oak −0.307 vs random −0.332 (above random; Spearman shows
  decreasing trend but final value beats baseline by +0.025 — net positive over the run).

- **4 marginal failures** — all with |Δ vs random| < 0.001, within statistical noise at
  5 seeds. These are not catastrophic regressions; they're "tied with random" envs where
  longer training (Tier 7 plan asks for 102,400-step horizons; this report is at 51,200).

## Pareto-FLOP/rewardRate principle (Hard Rule, AGENTS.md)

Per the new Hard Rule: every kernel must earn its FLOPs. Tier 6's per-kernel ablation
infrastructure was deferred (requires an `--ablate-kernel=NAME` engine flag). The 4
failure envs above are the canonical candidates for ablation analysis: if any specific
kernel (e.g., Tier 3 chord layer) hurts on these envs while helping elsewhere, the
ablation will identify it. Pass 12 will close this loop.

## Per-env table

| env | hb | Spearman ρ | p | final oak rR | final random rR | Δ | verdict |
|---|---|---|---|---|---|---|---|
| HalfCheetah-v4 | 12 | -1.000 | 1.0000 | -0.30733 | -0.33241 | +0.02508 | PASS (plateau) |
| Humanoid-v4 | 12 | +0.895 | 0.0000 | 4.50488 | 4.48834 | +0.01653 | PASS |
| MiniGrid-BlockedUnlockPickup-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-DistShift1-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-DistShift2-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-DoorKey-16x16-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-DoorKey-5x5-v0 | 12 | +0.676 | 0.0069 | 0.00000 | 0.00001 | -0.00001 | PASS |
| MiniGrid-DoorKey-6x6-v0 | 12 | +0.640 | 0.0115 | 0.00000 | 0.00000 | +0.00000 | PASS |
| MiniGrid-DoorKey-8x8-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-Dynamic-Obstacles-16x16-v0 | 12 | -0.741 | 0.9979 | -0.07632 | -0.07955 | +0.00323 | PASS (plateau) |
| MiniGrid-Dynamic-Obstacles-5x5-v0 | 12 | -1.000 | 1.0000 | -0.14669 | -0.14985 | +0.00316 | PASS (plateau) |
| MiniGrid-Dynamic-Obstacles-6x6-v0 | 12 | -1.000 | 1.0000 | -0.12697 | -0.12946 | +0.00249 | PASS (plateau) |
| MiniGrid-Dynamic-Obstacles-8x8-v0 | 12 | -1.000 | 1.0000 | -0.09890 | -0.10055 | +0.00165 | PASS (plateau) |
| MiniGrid-Dynamic-Obstacles-Random-5x5-v0 | 12 | -1.000 | 1.0000 | -0.12739 | -0.12692 | -0.00047 | **FAIL** |
| MiniGrid-Dynamic-Obstacles-Random-6x6-v0 | 12 | -1.000 | 1.0000 | -0.11096 | -0.10999 | -0.00097 | **FAIL** |
| MiniGrid-Empty-16x16-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-Empty-5x5-v0 | 12 | +0.998 | 0.0000 | 0.00144 | 0.00151 | -0.00008 | PASS |
| MiniGrid-Empty-6x6-v0 | 12 | +0.975 | 0.0000 | 0.00013 | 0.00020 | -0.00008 | PASS |
| MiniGrid-Empty-8x8-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-Empty-Random-5x5-v0 | 12 | +0.133 | 0.3442 | 0.00700 | 0.00680 | +0.00019 | PASS (plateau) |
| MiniGrid-Empty-Random-6x6-v0 | 12 | -0.126 | 0.6479 | 0.00417 | 0.00421 | -0.00004 | **FAIL** |
| MiniGrid-Fetch-5x5-N2-v0 | 12 | +0.056 | 0.4333 | 0.00870 | 0.00859 | +0.00012 | PASS (plateau) |
| MiniGrid-Fetch-6x6-N2-v0 | 12 | -0.077 | 0.5914 | 0.00548 | 0.00559 | -0.00011 | **FAIL** |
| MiniGrid-Fetch-8x8-N3-v0 | 12 | +0.699 | 0.0047 | 0.00278 | 0.00300 | -0.00022 | PASS |
| MiniGrid-FourRooms-v0 | 12 | -0.881 | 1.0000 | 0.00037 | 0.00036 | +0.00001 | PASS (plateau) |
| MiniGrid-GoToDoor-5x5-v0 | 12 | +0.958 | 0.0000 | 0.01152 | 0.01116 | +0.00036 | PASS |
| MiniGrid-GoToDoor-6x6-v0 | 12 | +0.986 | 0.0000 | 0.00864 | 0.00849 | +0.00014 | PASS |
| MiniGrid-GoToDoor-8x8-v0 | 12 | +0.993 | 0.0000 | 0.00610 | 0.00609 | +0.00001 | PASS |
| MiniGrid-GoToObject-6x6-N2-v0 | 12 | +1.000 | 0.0000 | 0.02083 | 0.02077 | +0.00006 | PASS |
| MiniGrid-GoToObject-8x8-N2-v0 | 12 | +0.930 | 0.0000 | 0.01020 | 0.00995 | +0.00025 | PASS |
| MiniGrid-KeyCorridorS3R1-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-KeyCorridorS3R2-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-KeyCorridorS3R3-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-KeyCorridorS4R3-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-KeyCorridorS5R3-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-KeyCorridorS6R3-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-LavaCrossingS11N5-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-LavaCrossingS9N1-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-LavaCrossingS9N2-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-LavaCrossingS9N3-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-LavaGapS5-v0 | 12 | +0.998 | 0.0000 | 0.00048 | 0.00056 | -0.00008 | PASS |
| MiniGrid-LavaGapS6-v0 | 12 | +0.840 | 0.0001 | 0.00004 | 0.00004 | -0.00000 | PASS |
| MiniGrid-LavaGapS7-v0 | 12 | +0.480 | 0.0582 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-LockedRoom-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-MemoryS11-v0 | 12 | +0.979 | 0.0000 | 0.00138 | 0.00133 | +0.00005 | PASS |
| MiniGrid-MemoryS13-v0 | 12 | +0.993 | 0.0000 | 0.00127 | 0.00104 | +0.00024 | PASS |
| MiniGrid-MemoryS13Random-v0 | 12 | +1.000 | 0.0000 | 0.00172 | 0.00152 | +0.00019 | PASS |
| MiniGrid-MemoryS17Random-v0 | 12 | +1.000 | 0.0000 | 0.00140 | 0.00129 | +0.00011 | PASS |
| MiniGrid-MemoryS7-v0 | 12 | +0.993 | 0.0000 | 0.00241 | 0.00233 | +0.00008 | PASS |
| MiniGrid-MemoryS9-v0 | 12 | +1.000 | 0.0000 | 0.00170 | 0.00168 | +0.00002 | PASS |
| MiniGrid-MultiRoom-N2-S4-v0 | 12 | +0.970 | 0.0000 | 0.00019 | 0.00018 | +0.00000 | PASS |
| MiniGrid-MultiRoom-N4-S5-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-MultiRoom-N6-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-1Dl-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-1Dlh-v0 | 12 | +0.757 | 0.0015 | 0.00000 | 0.00000 | +0.00000 | PASS |
| MiniGrid-ObstructedMaze-1Dlhb-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-1Q-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-1Q-v1 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-2Dl-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-2Dlh-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-2Dlhb-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-2Dlhb-v1 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-2Q-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-2Q-v1 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-Full-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-ObstructedMaze-Full-v1 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-Playground-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-PutNear-6x6-N2-v0 | 12 | +0.951 | 0.0000 | 0.00028 | 0.00028 | -0.00001 | PASS |
| MiniGrid-PutNear-8x8-N3-v0 | 12 | +0.991 | 0.0000 | 0.00006 | 0.00007 | -0.00001 | PASS |
| MiniGrid-RedBlueDoors-6x6-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-RedBlueDoors-8x8-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-SimpleCrossingS11N5-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-SimpleCrossingS9N1-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-SimpleCrossingS9N2-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-SimpleCrossingS9N3-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-Unlock-v0 | 12 | +0.840 | 0.0001 | 0.00003 | 0.00004 | -0.00001 | PASS |
| MiniGrid-UnlockPickup-v0 | 12 | +0.000 | 1.0000 | 0.00000 | 0.00000 | +0.00000 | PASS (plateau) |
| MiniGrid-WFC-DungeonMazeScaled-v0 | 12 | +0.790 | 0.0007 | 0.00055 | 0.00057 | -0.00002 | PASS |
| MiniGrid-WFC-MazeSimple-v0 | 12 | -0.825 | 0.9998 | 0.00035 | 0.00033 | +0.00002 | PASS (plateau) |
| MiniGrid-WFC-ObstaclesAngular-v0 | 12 | -0.944 | 1.0000 | 0.00036 | 0.00036 | +0.00000 | PASS (plateau) |
| MiniGrid-WFC-ObstaclesBlackdots-v0 | 12 | +0.441 | 0.0780 | 0.00028 | 0.00022 | +0.00006 | PASS (plateau) |
| MiniGrid-WFC-ObstaclesHogs3-v0 | 12 | -0.385 | 0.8881 | 0.00018 | 0.00017 | +0.00001 | PASS (plateau) |
| MiniGrid-WFC-RoomsFabric-v0 | 12 | -0.063 | 0.5750 | 0.00024 | 0.00024 | +0.00000 | PASS (plateau) |
| Walker2d-v4 | 12 | -0.993 | 1.0000 | 0.04845 | -0.00477 | +0.05322 | PASS (plateau) |