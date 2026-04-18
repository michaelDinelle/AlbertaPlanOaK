# Pass 12 Tier 3: per-kernel keep/remove decision (multi-signal Pareto)

Baseline: `.oak-benchmarks/pass12-ablation/baseline`
Ablation root: `.oak-benchmarks/pass12-ablation`
Significance threshold: p < 0.05

## Decision rule (AGENTS.md Multi-signal Pareto Hard Rule)

A kernel is KEPT iff its ablation HURTS at least one of the four agent-side channels
(rewardRate, LPM_traj_mean, util_traj_mean, rank_min) at p < threshold on at least one
env. Otherwise REMOVED — the kernel is not earning its FLOPs.

## Per-kernel decision table

| kernel | decision | net earned envs | earned (rR/LPM/util/rank) | hurt (rR/LPM/util/rank) | mean Δagent_ms (ms; positive = ablation saves FLOPs) |
|---|---|---|---|---|---|
| oak_bk_admit_commit | **KEEP** | +115 | 30/1/0/84 | 0/0/0/0 | -0.1416 |
| oak_encode | **KEEP** | +82 | 1/0/0/81 | 0/0/0/0 | -0.1061 |
| oak_bk_feature | **KEEP** | +81 | 0/0/0/81 | 0/0/0/0 | -0.5117 |
| oak_plan | **KEEP** | +78 | 0/0/0/78 | 0/0/0/0 | +0.0040 |
| oak_predict | **KEEP** | +19 | 0/0/0/19 | 0/0/0/0 | +0.0339 |
| oak_bk_option | **KEEP** | +17 | 0/4/0/13 | 0/0/0/0 | -0.1241 |
| oak_meta_action | **KEEP** | +17 | 1/0/0/16 | 0/0/0/0 | -0.1699 |
| oak_bk_reduce | **KEEP** | +15 | 1/0/0/14 | 0/0/0/0 | +0.2477 |
| oak_learn_omega | **KEEP** | +10 | 0/1/0/9 | 0/0/0/0 | -0.1672 |
| oak_learn_pi | **KEEP** | +8 | 0/0/0/8 | 0/0/0/0 | -0.1580 |
| oak_predict_outcome | **KEEP** | +7 | 0/3/0/4 | 0/0/0/0 | +0.0014 |
| oak_learn_beta | **KEEP** | +6 | 0/0/0/6 | 0/0/0/0 | -0.1673 |
| oak_learn_psi | **KEEP** | +6 | 0/1/0/5 | 0/0/0/0 | -0.1958 |
| oak_meta_policy | **KEEP** | +6 | 0/1/0/5 | 0/0/0/0 | +0.0099 |
| oak_reduce_alpha | **KEEP** | +6 | 0/2/0/4 | 0/0/0/0 | -0.2119 |
| oak_learn_outcome | **KEEP** | +5 | 0/2/0/3 | 0/0/0/0 | -0.2056 |
| oak_learn_encoder | **KEEP** | +4 | 0/4/0/0 | 0/0/0/0 | -0.2399 |
| oak_learn_value | **KEEP** | +3 | 0/1/0/2 | 0/0/0/0 | -0.2062 |

## Summary

- KEEP: 18 kernels: oak_bk_admit_commit, oak_bk_feature, oak_bk_option, oak_bk_reduce, oak_encode, oak_learn_beta, oak_learn_encoder, oak_learn_omega, oak_learn_outcome, oak_learn_pi, oak_learn_psi, oak_learn_value, oak_meta_action, oak_meta_policy, oak_plan, oak_predict, oak_predict_outcome, oak_reduce_alpha
- REMOVE: 0 kernels: 