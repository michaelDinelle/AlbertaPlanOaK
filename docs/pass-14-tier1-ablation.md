# Pass 14: per-kernel keep/remove decision (>=2-of-5-channels rule)

Baseline: `.oak-benchmarks/pass14-tier1-ablation/baseline`
Ablation root: `.oak-benchmarks/pass14-tier1-ablation`
Significance: p < 0.05 | Min harmed channels: 2/5

## Decision rule

A kernel is KEPT iff its ablation HURTS at least **2 of 5** channels
(rewardRate, LPM, utilFeat, pEff, agent_ms) at p<0.05 on at least one env. 
Otherwise REMOVED — single-channel kernels do not earn their FLOPs under Pass 14's
strict rule (Pass 12 used a lenient >=1 rule that let SPR-like kernels through
with only LPM contribution).

## Per-kernel decision table

| kernel | decision | channels harmed | rR envs | LPM envs | utilFeat envs | pEff envs | agent_ms envs |
|---|---|---|---|---|---|---|---|
| oak_spr_fused | **KEEP** | 3/5 | 0 | 1 | 0 | 13 | 77 |

## Summary

- KEEP (1): oak_spr_fused
- REMOVE (0): 