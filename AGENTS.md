# AGENTS

## The Alberta Plan

### Roadmap (steps 1–11)

1. Representation I: Continual supervised learning with given features.
2. Representation II: Supervised feature finding.
3. Prediction I: Continual GVF prediction learning.
4. Control I: Continual actor-critic control.
5. Prediction II: Average-reward GVF learning.
6. Control II: Continuing control problems.
7. Planning I: Planning with average reward.
8. Prototype-AI I: One-step model-based RL with continual function approximation.
9. Planning II: Search control and exploration.
10. Prototype-AI II: The STOMP progression.
11. Prototype-AI III: Oak.

### Research vision (Sutton-Bowling-Pilarski 2022, [arXiv:2208.11173](https://arxiv.org/abs/2208.11173))

- **Experience-oriented.** Only `(observation, action, reward)` time series.
- **Temporally uniform.** Same algorithms run every step. No special training periods, no train/eval split, no episodes inside the agent.
- **Compute cost first-class.** Small time step; reactive action prioritized.
- **Big World Hypothesis.** Env vastly more complex than agent → tracking, non-stationary approximations. Magnitudes carry information; **no caps that impose a stationary scale prior**.
- **Bitter Lesson.** Search and learning scale with compute; human-engineered domain knowledge does not.

### The base agent (Plan Figure 2)

Four components interconnected by a single shared state representation:

- **Perception** → state.
- **Reactive policies** (primary + others).
- **Value functions** (one per policy).
- **Transition model** (predicts next state and reward; may be temporally abstract over options).

Planning runs continually over the transition model. Eligibility traces are mentioned as one example of "short-term credit-assignment memories" — not required.

### Step 11 Oak additions to STOMP

Strict current-status audit lives in `docs/roadmap.md`. The literature target is:

- Subtasks defined by features (cumulant = reward, stopping value = feature attainment per Reward-Respecting Subtasks [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4). **Current rebuild OaK has the RR stopping-value branch live and has produced bridge-measured positive walking in some seeds; stable promotion is still open.**
- Options = policies + termination. **Current rebuild continuation is derived from the RR stopping comparison, not H1 model-confidence hysteresis.**
- Option models r̂(s,o), n̂(s,o), and for average-reward options duration l̂(s,o), used for planning. **Current L3-L7 keeps the orthodox raw option-reward-plus-duration form: `Q = rHat - rho*lHat + V(nHat)`. A coherent differential option-reward alternative was tested and rejected by the 10k paired gate, so do not revive it without materially new evidence.**
- Planning over option models (RR-Subtasks eq. 19 / average-reward option backup).
- **Option keyboard** = real-valued vector λ ∈ R^K, one component per subtask. Design A: chord blends K component options; model treats chord as opaque descriptor. Design B: chord defines a joint subtask. **Current OaK has a live scalar chord vector used to blend option policies, but it is not a fully operational Option Keyboard.** The unproven `psiCross`/GPE-GPI surrogate was deleted; current behavior is selected by RR continuation plus model-backed average-reward option scores, and emitted action means are a chord blend with selected-option fallback. A fully operational FC-STOMP keyboard/chord-descriptor cycle is an open architecture gap, not a hidden live feature.
- **Feedback / utility / remove-replace.** Continually assess utility of features, subtasks, options, and option models. Replace what isn't pulling weight (Plan Step 11 + RR-Subtasks §8 FCSTOMP). **Current rebuild has age-gated feature/option-bundle replacement, `optFeature`, feature age, option age, and feature-ever-used metadata; richer utility remains an active research frontier.**
- **Named FC-STOMP loop / ablations.** `Sources/OakKernel.metal` exposes the live loop vocabulary directly in code: `FeatureConstruction`, `GVFPrediction`, `RewardRespectingSubtasks`, `OptionPolicies`, `OptionModels`, `Planning`, `OptionKeyboard`, `SearchControl`, `GenerateAndTest`, and `ActionEmission`. `OAK_ABLATE` compiles specialized Metal pipeline variants for diagnostic ablations of those sections and couplings; canonical promotion runs use `OAK_ABLATE=none`.
- **Ablation lesson.** The first 50k focus matrix (`.oak-benchmarks/step11-ablation-focus50k-*`) confirms the live walking mode depends on the whole FC-STOMP loop. The largest losses come from `keyboard-to-policy`, `option-keyboard`, and `model-to-keyboard`; `search-control` can help a quiet seed while killing the rocket. Treat this as evidence for a richer cumulant/value-space keyboard and bundle-utility loop, not as permission to delete organs.

### Current status

OaK produces real HalfCheetah locomotion in a thin slice of seeds, but most seeds diverge and the population delivered rate is still negative. Current numbers come from fleet runs, not from this file — run `tools/fleet.py report`, and see the evidence sections in `docs/roadmap.md` (Walking Checkpoint Evidence, Current Evidence Ledger). The full April CPU-bridge walking-loop history — every status snapshot and every rejected stabilization mechanism — is archived in `docs/roadmap.md` under Exhausted Families (Keyboard / search-control / arbiter era).

Durable lessons from that work, independent of instrument or seed:

- **Preserve the high-return walking mode while improving coverage.** Changes that calm actions, reduce commitment, refresh chords more often, or delete model-backed selection have repeatedly killed the strong mode. That is a regression, not a simplification.
- **Interruption, termination, and duration-bounding are a dead end for plasticity** — every form tested killed the strong seed. Pursue representational plasticity instead (feature/representation rank maintenance, step-size adaptation, actor-entropy maintenance) in FeatureConstruction / GVFPrediction / GenerateAndTest.
- **The missing organ is a coherent keyboard / search-control / utility loop** — cumulant/value-space composition (GPE-GPI or a chord-descriptor model) plus search-control effectiveness feeding back into which features, subtasks, options, and models survive. Isolated chord refreshes, damping rules, fallbacks, and raw reward-TD actor pressure are patch-and-pray.
- **Judge the population, not a single seed.** Seeds 3 and 8 are inspectable lanes, not gates; promotion is the paired population gate (`tools/fleet.py gate`).

### What the Plan eschews

- Nothing about the human world (vision, space, objects, language, geometry, other agents).
- Learning first, domain knowledge last.
- No discounting (Step 5+).
- Actions and observations are signals with no intrinsic meaning.

## Hard rules (project invariants)

These are the non-negotiables. Every kernel and every line of host code must respect them. When an invariant drifts from the cited research, update the invariant instead of preserving folklore.

1. **Greenfield.** No legacy code, no compatibility fallbacks. Replace or refactor.
2. **Inv-1: env-type knowledge boundary.** The agent (`OakMetalEngine`, `OakKernel.metal`, `KernelConfig`) contains ZERO action-space-type or env-identity branches. Action space (discrete vs continuous), bounds, env id, discrete count live exclusively in the env-wrapper layer (`EnvPoolEnv` and below). Agent reads only `obsDim` and `D_policy`.
3. **Inv-4: no episodic semantics inside the agent.** Resets are wrapper events; the agent runs continually. The `done` signal is intentionally not plumbed through to the engine.
4. **Inv-5: no special training phases.** Same updates every step.
5. **Inv-6: no per-step heap allocation on the hot path.** Pre-allocated arenas only.
6. **Inv-7: temporal uniformity** (Plan research-vision pillar #2; verbatim: *"all times are the same with respect to the algorithms running on the agent. There are no special training periods when training information is available or when rewards are counted more or less than others. If training information is provided... it is provided on every time step. If the agent learns or plans, then it learns or plans on every time step. If the agent constructs its own representations or subtasks, then the meta-algorithms for constructing them operate on every time step."*). Every Plan mechanism in OaK runs every step: perception (encode), prediction/control, planning (full-bank `oak_plan`), learning (learn_heads), meta-learning (post_learn MetaOptimize), and every-step G&T assessment with capacity/age-gated commits. Reward is centered (ρ̄) the same way every step. `track_rate(t, obsDim) = max(1/(t+1), 1/obsDim)` is the current structural tracking choice, not a Plan mandate; replace it if paired evidence or primary research shows a more faithful parameter-free tracker. Subsumes Inv-4 + Inv-5 in their consolidated Plan-citing form. **Forbidden constructs**: warmup-gated branches (`if stepCount > N`), epoch counters, train/eval splits, frozen weights post-N steps, special "exploration" vs "exploitation" phases. `OAK_WARMUP_STEPS` is host-side bench metric collection only — the agent's per-step computation is identical regardless.
7. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
8. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity, stream shape, or device execution width (cap = R, K = R, α₀ = 1/cap, ridge = 1/(age+1)). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
9. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
10. **Fleet stream semantics.** Each agent's experience is one observation → one action → one reward per step — a single time series, exactly as the Alberta Plan describes. The live engine batches **B semantically independent agents** (one per seed) into one process: per-lane disjoint state (agent-strided buffers, agentID on the grid's `.y` axis), zero cross-stream reductions, identical per-agent updates. Lane b runs seed b; a B=1 lane is bitwise self-reproducible run-to-run, and lane-disjointness is the standing acceptance test for any engine change (state-hash streams via `OAK_DUMP_STATE_EVERY` + `actions.bin` `cmp`). The environment is GPU-resident (`Sources/OakPhysics.metal`, fp32 port of MuJoCo certified population-equivalent to fp64 CPU MuJoCo by the null-calibrated TOST gate at N=8192 — see docs/roadmap.md F4 verdict); streams are CONTINUING by default (`OAK_EPISODE_STEPS=0`, never auto-reset — matching all CPU-era data), with gym 1000-step semantics available for external comparability and flagged in any SOTA claim.
11. **Inv-10: zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits and megakernel stage fusion separated by `threadgroup_barrier(mem_device)`, grid-stride or simdgroup/threadgroup remaps of independent elementwise work at any width ≥ 32, sequential re-ordering of writes-disjoint threadgroup work, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/function constants (with `-fno-fast-math`, which forbids reassociation, shape-specialized unrolling is in this class), `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with **bitwise A/B proof** (`actions.bin` `cmp` + `OAK_DUMP_STATE_EVERY` state-hash streams) + Tier-S smoke. **FP-reorder** (`simd_sum`, `simdgroup_matrix` ops, threadgroup reductions where the accumulation order changes from sequential, `fma` decomposed into separate multiply + add): ship only on a `tools/fleet.py gate` **PROMOTE at Tier-M (N=1024 × 100k, paired by seed) for screening and Tier-L for promotion** against the current keeper. The population law replaces the retired 10-seed ±3% per-seed-lift rule: per-seed reshuffling that averages to zero now surfaces as a paired divergence/IQM tail at N=1024 instead of anecdotal seed inspection. Historical lesson retained: the first C2-perf attempt used `simd_sum` and showed aggregate parity while per-seed chatter std doubled (0.020 → 0.041) — exactly the failure mode the paired population gate is built to catch.

## Architecture

The agent is a focused-kernel single-stream Oak prototype over the Plan steps. K is the active option/subtask count and now equals R; kernel threadgroup width is derived from the Metal pipeline execution width rather than an option page scheduler. Every learnable head and every state slot must be justified against the Plan; strict completion status is in `docs/roadmap.md`.

### Learnable parameters

| Head | Shape | Role |
|---|---|---|
| `valueW` | R | Main value GVF (cumulant = reward, differential, ρ̄-centered). |
| `psi` | K · R | K=R feature-indexed RR subtask GVFs. Cumulant is shared average-reward-centered reward; stopping value is RR Eq. 4: `V(s') - valueW[i]·x_i(s') + bonusW[i]·x_i(s')`. |
| `optPolicy` | K · D · R | K component policies. The wrapper samples from the emitted behavior descriptor and returns exact one-step importance ratios for each target option policy in the active bank. |
| `optReward` | K · R | K stopping-aware option reward models r̂^k. |
| `optTransition` | K · R · R | K dense stopping-aware transition models `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]`. |
| `optDuration` | K · R | K average-reward option-duration models l̂^k with a structural one-step lower bound at decode. |

State representation is a finite adaptive predictive state, not a full reconstruction of the world: direct normalized experience channels (observation coordinates, multi-timescale observation traces, previous abstract action descriptor, previous reward, and an affine channel) plus constructed feature slots. Deep Fourier, random Fourier, and random-projection encoders are optional experimental modes, not Alberta Plan law. Constructed slots remain replaceable signed products/sums/residuals of useful parents. Plasticity is driven by MetaOptimize-style `alpha/beta/h/gradEnergy` families plus age/utility replacement; raw IDBD/Autostep are historical relatives, not the implementation target.

### Pipeline (5 Metal dispatches per step, direct command encoding)

```
oak_encode → oak_plan → oak_learn_heads → oak_replace → oak_action_emit
```

| Kernel | What it does |
|---|---|
| `oak_encode` | Updates observation/traces, builds normalized anchor/constructed state features, then runs prediction/control: value TD, RR subtask TD, RR continuation, model-backed option scoring, and chord refresh. |
| `oak_plan` | Updates the full option bank's stopping-aware model TDs, duration TDs, raw-reward option model, and model-priority statistics every step. |
| `oak_learn_heads` | Updates value, psi, option policy, and option reward/transition/duration using exact one-step ratios and wrapper-provided target-option score gradients. |
| `oak_replace` → `oak_action_emit` | Commits utility/meta-learning diagnostics and age/capacity-gated replacements, then emits all option policy means after a dispatch boundary so action emission cannot race replacement writes. |

The dispatch shape stays fixed, but the internal design pattern is the named FC-STOMP loop: `FeatureConstruction -> GVFPrediction -> RewardRespectingSubtasks -> OptionPolicies -> OptionModels -> Planning -> OptionKeyboard -> SearchControl -> GenerateAndTest -> ActionEmission`.

### Action interface (Inv-1)

Each env declares one structured action spec: `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = component_param_count` is the agent's only window into the env's action structure.

The agent emits `policyMean[D_policy]` from `oak_action_emit`. The env-wrapper layer (`HCEnv::step_oak` in the native bridge) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax); target-option actor scores use `1{a=d} − π_d` (Maddison 2014; Williams 1992).
- Box A: currently exposes `2A` policy coordinates to keep `D_policy`, default rank, and K stable. The first A coordinates are means; the second A coordinates are wrapper-reserved/inert. The wrapper samples with structural fixed `σ = 1/D_policy`, emits residual-style mean credit (`raw − μ`) for Box mean slots, writes zero score/action-signal entries for reserved scale slots, and clamps only the env-facing action. Full `D_policy = A` mean-only deletion failed the 1M gate and remains rejected unless a different actor geometry earns fresh evidence.

Adding a new env or action-space shape requires adding to `ActionSpec` in Swift; no agent or kernel changes.
`doneCount` is a wrapper random-restart metric, not an episode signal to OaK; the engine state is never reset on `done`.

## Fleet run output (fleetSchemaVersion: 1)

A fleet run root (`.oak-benchmarks/<run-id>/`, override with `OAK_METRICS_STORAGE`) contains:

- **`meta.json`** — run config: `fleetSchemaVersion`, `envId`, `batch` (B), `seeds`, `steps`, `policies`, `ablation`/`ablationLabel`, `startedAt`.
- **`fleet.jsonl`** — one compact-JSON row per (lane, heartbeat) and per (lane, final): `v`, `kind` (`hb`|`final`), `lane`, `seed`, `step`, `wallSeconds`, `stepsPerSecond`, `rewardSum`, `doneCount`, `firstNonFiniteStep`/`firstNonFiniteDetector` (the death certificate), `kernelRho` (the kernel's internal ρ — a divergence diagnostic, distinct from the host EMA), `paramAbsMax` (running max |param|; dates 1e36 blowups before NF counters fire), and the full `health` block (same fields as before: `rewardRate` host ρ̄, `actionDeltaL2Mean`/`Peak`, `continuingRestartCount`, `envInfo`, `nonFiniteByPhase` cumulative `{encode, learn, plan, tdPsi}`, search/model/keyboard diagnostics, actor diagnostics).
- **`deaths.jsonl`** — one row per lane death event: `seed`, `lane`, `step`, `detector` (`envReward` | `policyMean` | `paramNonFinite` | `paramExplosion` | `kernelRho` | `envError`), `nfByPhase`, `rhoAtDeath`, `paramAbsMax`, `actorSupportEntropy`, `terminationRate`, `optionDurationMean`, `rewardSumAtDeath`, `lastGoodRewardRate`. A "dead" lane keeps stepping in the GPU fleet (one continual stream, never reset; recovery is data); the certificate dates the first event.
- **`aggregate.jsonl`** — one row per heartbeat per policy: `step`, `alive`/`dead`, delivered-rate P10/P50/P90, `aggStepsPerSecond`, `wallSeconds`. This is the live dashboard (`tools/fleet.py watch`).
- **`<envId>/seed<N>/`** — per-lane `actions.bin` + `meta.json` (+ optional `stats.bin`) for lanes selected by `OAK_RECORD_SEEDS=3,8` (or all lanes with `OAK_RECORD_ACTIONS=1`); `state-hashes.jsonl` when `OAK_DUMP_STATE_EVERY=N` is set (byte-identity validation).

**Tools** (numpy; `requirements.txt`):

- `.venv/bin/python tools/fleet.py report <run-root>` — population report: Kaplan-Meier survival + RMST, divergence rate with Wilson 95% CI (overall + per detector), IQM delivered reward rate (`rewardSum/steps`) with bootstrap CI over ALL lanes (headline; never condition on survival) and survivors-only (diagnostic), plasticity quartiles (zenith step/rate, decay ratio), entropy-collapse rate. Markdown output pastes into `docs/roadmap.md`; also writes `summary.json`.
- `.venv/bin/python tools/fleet.py gate <keeper-root> <candidate-root>` — **the promotion law.** Same seed set + horizon required (hard error); all bootstraps paired by seed. PASS iff ALL four hold (AND-gate ⇒ each one-sided α=0.05, no multiplicity correction): (1) divergence non-inferiority, 95th pct of paired Δp ≤ δ (default 0.64/√N: 0.02 @ N=1024); (2) IQM delivered-rate non-inferiority, 5th pct of ΔIQM ≥ −5%·|IQM_keeper|, with a non-blocking IMPROVEMENT flag (what SOTA claims must cite); (3) RMST survival non-inferiority, 5th pct of ΔRMST ≥ −5%·horizon; (4) per-phase NF prevalence non-inferiority — for each phase, 95th pct of paired Δ(fraction of lanes with any non-finite in that phase) ≤ δ (per-lane PREVALENCE, not fleet-sum totals: a few already-diverged lanes spray non-finites post-death and inflate totals without changing how many lanes fail, which double-counts what criterion 1 already measures; consistent with the equivalence gate's `nf_prevalence`). Exit 0 = PROMOTE, 1 = REJECT. Seed-3/seed-8 folklore is not encoded anywhere — those seeds are ordinary lanes (lane 3, lane 8 of every fleet) whose trajectories remain inspectable in `fleet.jsonl`.
- `.venv/bin/python tools/fleet.py watch <run-root> [--follow]` — live aggregate view.

## The SOTA bar (HalfCheetah-v4, external)

Published results report **episodic return** over 1000-step episodes; divide by
1000 to get per-step reward rate, directly comparable to our delivered
`rewardSum/steps`. Values at 1M env steps (evaluation of the final policy):

| Algorithm | Episode return @1M | Reward/step |
|---|---|---|
| TD7 | ~17,433 | ~17.4 |
| TD-MPC2 | ~15,078 | ~15.1 |
| MR.Q | ~13,305 | ~13.3 |
| TD3 | ~10,051 | ~10.1 |
| SAC | ~9,871 | ~9.9 |

Sources: TD7/TD-MPC2/MR.Q table in [arXiv:2502.15280](https://arxiv.org/pdf/2502.15280);
SAC/TD3 HalfCheetah-v4 @1M in [arXiv:2208.00755](https://arxiv.org/pdf/2208.00755).

Reference point (first fleet keeper, 2026-06-11, B=1024 @250k): best lane
(seed 3) delivered ~0.37/step; top-10 lanes ~0.14–0.22; population median
−0.023. Raw gap to TD7 ≈ 47× at the best lane; population median is negative.

Comparison rules:

1. **Eval vs delivered.** Published numbers evaluate the final policy on free
   episodes; OaK's number is online delivered reward over the entire stream,
   flailing included. SAC's delivered-during-training average over 1M is
   roughly 5–7/step, so the delivered-vs-delivered gap is ~15–20×, not 47×.
   Never mix the two framings in one claim.
2. **Per-FLOP.** The project's actual SOTA target is reward-rate-per-FLOP
   (SAC/TD7 spend large replay-batch gradient updates per env step; OaK spends
   a few small linear heads). No per-FLOP claim is allowed until FLOP counts
   are instrumented on both sides.
3. The gap decomposes into two measured problems: (a) make the strong-walker
   mode (~0.2–0.4/step, currently a ~1–2% population mode) the typical
   outcome — this alone moves population IQM from ≈−0.02 to ≈+0.3; (b) raise
   the mode's ceiling from ~0.4 toward 10+ — bound by the post-zenith
   plasticity frontier.

## Streamlined research workflow

The working goal is **general online robustness**: any seed, any horizon, population-level delivered reward rate that beats SOTA — measured on N=1024 lanes, not 10 anecdotes. The loop is architecture-first, evidence-fast:

1. State the FC-STOMP section being changed before editing: feature construction, GVF prediction, RR subtasks, option policies, option models, planning, option keyboard, search control, generate-and-test, or action emission. A candidate that cannot name the section is patch-and-pray.
2. Run every candidate up the tier ladder. **One fleet size (B=1024), one horizon ladder, one measured throughput constant**; tiers are prefix-nested (seeds 0–1023 everywhere) so every tier is a paired extension of the one below:

   | Tier | B × horizon | Wall-clock (measured, GPU instrument) | Use | Resolves |
   |---|---|---|---|---|
   | S | 1024 × 10k | ~39 s | every candidate | hard fails (NF, explosion) + divergence Δp ≳ 4% |
   | M | 1024 × 100k | ~6.4 min | candidates passing S | divergence ±2–3%, IQM screen |
   | L | 1024 × 1M | ~64 min | keeper promotion, frontier studies | survival curves through the ~1M plasticity frontier, Δp ≳ 2% |
   | M-8192 | 8192 × 100k | ~34 min | instrument equivalence, SOTA-grade CIs | divergence ±1pp, IQM at the strict ±5% band |
   | XL | 1024 × 10M | F5-opt-gated | decay-constant program (not a gate) | post-zenith time constants, 10M survival |

   **Hard rule: no protocol step exceeds ~2 h wall-clock.** Wall-clock derives from the measured constants in `docs/roadmap.md` (GPU loop: 265.7K agg @ B=1024, 401.9K @ B=16384) — never from estimates. XL's 10M horizon divides by the per-lane rate (~259 steps/s at B=1024), not aggregate throughput; it unlocks when agent-kernel fusion/occupancy work (the open F5-opt item — agent kernels are 77% of step wall) raises per-lane rate ~4×.
3. Promote only on a Tier-L `fleet.py gate` PROMOTE against the keeper baseline. The gate is population law (divergence Δp, IQM delivered rate, RMST survival, NF phases — all paired by seed); a SOTA claim additionally requires the gate's IMPROVEMENT flag.
4. Per-seed collapse is population data, not an averaging input: `fleet.py report` reports it as divergence rate, survival curves, and detector counts. Individual seeds (3, 8, …) remain inspectable as lanes in every fleet, but no single seed blocks or licenses promotion.
5. Name run roots by hypothesis and tier, for example `.oak-benchmarks/closure-arbiter-tierM-YYYYMMDD`, and inspect live runs with `tools/fleet.py watch --follow` rather than waiting for final rows.
6. Use subagents when the work can truly run in parallel: a Metrics Archaeologist mines benchmark roots, a Plan/Invariant Reviewer checks Alberta/FC-STOMP fit, a Mechanism Designer proposes closed-loop organs, a Benchmark Steward watches gates, and a Metal Auditor handles speed/FP-order risk. The main thread integrates; subagents do not define project doctrine.
7. Archive rejected mechanisms in `docs/roadmap.md`/`AGENTS.md` when they teach a reusable lesson. Keep the live code lean; diagnostics earn their place only when they shorten the next decision.

## CLI

| Command | Use |
|---|---|
| `OaK benchmark` | One fleet per policy: B independent (agent, env, seed) lanes in one process — the FC-STOMP loop, action sampling, and HalfCheetah physics all run GPU-resident in multi-step command buffers. `OAK_BENCH_ENV` is currently `HalfCheetah-v4`; `OAK_BENCH_STEPS`, `OAK_BENCH_SEED_COUNT` (default 1024) / `OAK_BENCH_SEEDS` / `OAK_BENCH_SEED_START`, and `OAK_BENCH_POLICIES` define the workload. `OAK_FLEET_TIMING=1` prints heartbeat throughput. One launcher owns the workspace at a time; nonempty requested roots get a fresh suffixed output dir. |

`OAK_ABLATE` defaults to `none`. It accepts a comma-separated diagnostic list: `feature-construction`, `gvf-prediction`, `rr-subtasks`, `option-policies`, `option-models`, `planning`, `option-keyboard`, `model-to-keyboard`, `keyboard-to-policy`, `search-control`, `search-to-generate-test`, and `generate-and-test`. These are specialized Metal function-constant variants, not a promoted behavior mode.

There is no deep-diagnostics shadow path in the live engine. Add audit heads only when they earn a Plan role and pass the same speed/learning gates as behavior code.

## Performance: Metal trace

Use Instruments → Metal System Trace with `tools/instruments/ShaderTemplate.tracetemplate` (Shader Timeline ON, no counters). The live CLI no longer has per-phase diagnostic JSON; use Instruments for phase shares.

## Citations (all primary papers in `docs/papers/`)

- Alberta Plan: [2208.11173](docs/papers/AlbertaPlanForAIResearch_2208.11173.pdf)
- Reward-Respecting Subtasks: [2202.03466](docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf)
- Reward Centering: [2405.09999](docs/papers/RewardCentering_2405.09999.pdf)
- Option Keyboard Basis: [2505.00787](docs/papers/OptionKeyboardBasis_2505.00787.pdf)
- Implicit α: [2510.06149](docs/papers/ImplicitUpdates_AvgRewardTD_2510.06149.pdf)
- GraMa feature utility: [2505.24061](docs/papers/GraMa_MeasureGradients_2505.24061.pdf)
- Generate-and-test: Mahmood-Sutton-Pilarski 2012 ([03-mahmood-sutton-autostep](docs/papers/03-mahmood-sutton-autostep-ICML-2012.pdf)), Rahman 2020 thesis, Samani 2022 thesis
- Deep Fourier features: Lewandowski-Schuurmans-Machado 2024 ([2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf))
- MetaOptimize per-feature step sizes: Sharifnassab-Salehkaleybar-Sutton 2024 ([2402.02342](docs/papers/MetaOptimize_StepSizes_2402.02342.pdf)). Treat this as the controlling source for `alpha/beta/h/gradEnergy`; IDBD/Autostep are historical special cases, not the implementation target.
- Big World Hypothesis: [2512.23419](docs/papers/WorldIsBigger_ComputationallyEmbeddedBWH_2512.23419.pdf)
- Three Dogmas of RL: [2407.10583](docs/papers/ThreeDogmasRL_2407.10583.pdf)
- Prioritized sweeping (Plan Step 9, Plan footnote 13): Moore-Atkeson 1993 (*Machine Learning* 13(1):103–130), Peng-Williams 1993 (*Adaptive Behavior* 1(4):437–454), Sutton-Szepesvári-Geramifard-Bowling 2008 (UAI). Current priority is option-model TD magnitude over the full live option bank; true state/update search control remains open.
- Meta-gradient search control: [2406.19561](docs/papers/MetaGradientSearchControl_DynaPlanning_2406.19561.pdf)
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §8.4 (prioritized sweeping with Dyna), §9.5.4 (online normalization)
