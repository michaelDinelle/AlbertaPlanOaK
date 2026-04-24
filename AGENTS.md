# AGENTS

## The Alberta Plan (slide law)

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

- Subtasks defined by features (cumulant = reward, stopping value = feature attainment per Reward-Respecting Subtasks [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4). **Current L3-L7 OaK has the RR stopping-value branch live; benchmark promotion is still open.**
- Options = policies + termination. **Current L3-L7 continuation is derived from the RR stopping comparison, not H1 model-confidence hysteresis.**
- Option models r̂(s,o), n̂(s,o), and for average-reward options duration l̂(s,o), used for planning. **Current L3-L7 has stopping-aware reward/transition recurrences plus an `optDuration` head; acceptance still depends on reward-rate evidence.**
- Planning over option models (RR-Subtasks eq. 19 / average-reward option backup).
- **Option keyboard** = real-valued vector λ ∈ R^K, one component per subtask. Design A: chord blends K component options; model treats chord as opaque descriptor. Design B: chord defines a joint subtask. **Current OaK blends policies but does not yet learn a chord-descriptor model.**
- **Feedback / utility / remove-replace.** Continually assess utility of features, subtasks, options, and option models. Replace what isn't pulling weight (Plan Step 11 + RR-Subtasks §8 FCSTOMP). **Current L3-L7 has age-gated option/feature replacement, `optFeature`, feature age, option age, and feature-ever-used metadata; richer utility remains an active research frontier.**

### What the Plan eschews

- Nothing about the human world (vision, space, objects, language, geometry, other agents).
- Learning first, domain knowledge last.
- No discounting (Step 5+).
- Actions and observations are signals with no intrinsic meaning.

## Hard rules (project invariants)

These are the non-negotiables. Every kernel and every line of host code must respect them.

1. **Greenfield.** No legacy code, no compatibility fallbacks. Replace or refactor.
2. **Inv-1: env-type knowledge boundary.** The agent (`OakMetalEngine`, `OakKernel.metal`, `KernelConfig`) contains ZERO action-space-type or env-identity branches. Action space (discrete vs continuous), bounds, env id, discrete count live exclusively in the env-wrapper layer (`EnvPoolEnv` and below). Agent reads only `obsDim` and `D_policy`.
3. **Inv-4: no episodic semantics inside the agent.** Resets are wrapper events; the agent runs continually. The `done` signal is intentionally not plumbed through to the engine.
4. **Inv-5: no special training phases.** Same updates every step.
5. **Inv-6: no per-step heap allocation on the hot path.** Pre-allocated arenas only.
6. **Inv-7: temporal uniformity** (Plan research-vision pillar #2; verbatim: *"all times are the same with respect to the algorithms running on the agent. There are no special training periods when training information is available or when rewards are counted more or less than others. If training information is provided... it is provided on every time step. If the agent learns or plans, then it learns or plans on every time step. If the agent constructs its own representations or subtasks, then the meta-algorithms for constructing them operate on every time step."*). Every Plan mechanism in OaK runs every step at the same rate: perception (encode), prediction (meta_predict), planning (plan + plan_reduce), learning (learn_heads), meta-learning (post_learn MetaOptimize), G&T option/feature replacement (bk_admit_commit). Reward is centered (ρ̄) the same way every step. **The Plan also says** *"if the agent can reduce its speed of learning about parts of the environment when they appear stable, then it can also increase its speed of learning when they start to change"* — `track_rate(t, baseO) = max(1/(t+1), 1/baseO)` honors this by slowing the EMA rate as data accumulates while flooring at `1/baseO` so adaptation never freezes. Subsumes Inv-4 + Inv-5 in their consolidated Plan-citing form. **Forbidden constructs**: warmup-gated branches (`if stepCount > N`), epoch counters, train/eval splits, frozen weights post-N steps, special "exploration" vs "exploitation" phases. `OAK_WARMUP_STEPS` is host-side bench metric collection only — the agent's per-step computation is identical regardless.
7. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
8. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity (cap = R, α₀ = 1/cap, ridge = 1/(cap+1), K ≤ 32 = SIMD width). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
9. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
10. **Single-stream (B = 1).** One observation → one action → one reward per step. Multi-env benchmarking is N independent agent instances over N envs (OperationQueue), never one batched agent.
11. **Inv-10: zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/`KernelScalars`, `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with smoke + 1.5M paired bench. **FP-reorder** (`simd_sum`, threadgroup reductions where the input order changes from sequential, `fma` decomposed into separate multiply + add): ship only after a 10-seed paired bench shows **per-seed parity within ±3 % of the prior commit's per-seed lift** — aggregate parity is necessary but insufficient; per-seed reshuffling that averages to zero is still considered an intelligence hit. The first attempt at C2-perf used `simd_sum` for `oak_meta`/`oak_plan` and showed aggregate parity but per-seed chatter std doubled (0.020 → 0.041); the follow-up restored bit-identical with lane-0 sequential reductions for ~4 µs cost.

## Architecture

The agent is a 5-dispatch single-stream Oak prototype over 9 logical Plan phases. Every learnable head and every state slot must be justified against the Plan; strict completion status is in `docs/roadmap.md`.

### Learnable parameters (8 buffers)

| Head | Shape | Role |
|---|---|---|
| `encoder` | (R/2) · (M·O + D_policy + 1) | Perception (Fourier features W). Each pre-activation row reads standardized observation, multi-timescale observation traces, previous action descriptor, and previous reward. **Fixed random projection at init** (RFF bandwidth-1 over all input regions); evolves via generate-and-test reseeding (Plan Step 2). |
| `phase` | R/2 | Per-pre-activation Fourier offset (uniform on [0, 2π) at init). Reseeded alongside its `encoder` row in generate-and-test. Together with `encoder`, parameterizes the deep Fourier features basis (Lewandowski-Schuurmans-Machado 2024, [arXiv:2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf)). |
| `valueW` | R | Main value GVF (cumulant = reward, differential, ρ̄-centered). |
| `psi` | K · R | K feature-indexed RR subtask GVFs. Cumulant is centered reward; stopping value uses `V(s') - valueW[i]·x_i(s') + bonusW[i]·x_i(s')` for the option's registered feature. |
| `optPolicy` | K · D · R | K component policies, blended by chord (Design A keyboard). |
| `optReward` | K · R | K stopping-aware option reward models r̂^k. |
| `optTransition` | K · R · R | K dense stopping-aware transition models `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]`. |
| `optDuration` | K · R | K average-reward option-duration models l̂^k with a structural one-step lower bound at decode. |

Per-feature alpha is achieved structurally by MetaOptimize step sizes plus KEEPING relevant features alive and REPLACING irrelevant ones via `oak_post_learn`'s fused admit/commit phase. R is rounded up to the nearest even value at host init because each Fourier pre-activation produces a paired `(cos, sin)` channel into the R-wide state. Cap K ≤ 32 is the Apple SIMD width (structural, hardware-derived) and is what lets cross-option reductions fit in one simdgroup. **Tier C2 (option termination via `termW` + closed-form geometric Q^k + hold-or-refresh chord) was attempted three times and regressed every time**; do not revive that family.

### Pipeline (5 Metal dispatches per step, 9 logical phases, Metal 4 ICB)

```
bk_encode(+meta_predict) → plan → plan_reduce → learn_heads → post_learn(+bk_admit_commit)
```

| Kernel | What it does |
|---|---|
| `oak_bk_encode` | Fuses observation normalization/traces, Fourier feature construction, chord update, reward centering, value TD, and RR subtask TD. Chord continuation comes from `psi(s') > z_i(s')`, where `z_i` is the reward-respecting stopping value for the option's registered feature. |
| `oak_plan` | K threadgroups x R threads. Computes stopping-aware option transition TDs, reward TDs, duration TDs, differential option Q `rHat - rho*lHat + V(nHat)`, and a value-relevant priority signal. |
| `oak_plan_reduce` | K-thread cross-option reduction. Writes soft-bounded `planAdv`, `meanPrio`, and compact duration diagnostics. |
| `oak_learn_heads` | Updates value, psi, option policy, option reward, option transition, and option duration heads. Psi/reward/transition/duration use structural `1/K` off-policy floors; policy remains on-chord because it is the actual gradient of the blended action. |
| `oak_post_learn` | Fuses feature/option utility, MetaOptimize alpha updates, chord-blended action emission, and age-gated Step-11 admit/commit replacement. Feature replacement is a clean birth that clears downstream weights tied to the replaced Fourier pair. |

### Action interface (Inv-1)

Each env declares a structured action spec — a list of components, each `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = Σ_components(component_param_count)` is the agent's only window into the env's action structure.

The agent emits `policyMean[D_policy]` from `oak_meta_action`. The env-wrapper layer (`EnvPoolEnv.sampleAndStep`) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax); scoreGrad = `1{a=d} − π_d` (Maddison 2014; Williams 1992).
- Box A: exposes `2A` policy coordinates, `[μ_0...μ_A, logσ_0...logσ_A]`. The wrapper samples `μ + σ·N(0,1)` and clamps only the env-facing action. `σ` starts at one policy-coordinate unit, `1/D_policy`, then the actor can adapt it. The mean score uses the compatible/natural variance-scaled form `z·σ`; the scale score is `z²−1`. This replaces the old `1/√A` rule, which is empirical history, not an invariant.

Adding a new env or action-space component type requires adding to `ActionComponent` in Swift; no agent or kernel changes.

## Run metrics JSON

Each `run`, `diagnose`, and `benchmark` writes one pretty-printed JSON file with stable alphabetical key order (`schemaVersion: 3`).

**Top-level keys:** `command`, `envId`, `policy`, `seed`, `numEnvs` (always 1 — single-stream), `warmupSteps`, `stopAfterMeasuredSteps`, `observationDim`, `actionDim`, `actionDiscreteCount` (when discrete), `measuredStepCount`, `measuredBatchCount`, `measuredSeconds`, `workerDepth = measuredStepCount`, `stepsPerSecond`, `rewardSum`, `doneCount`, `stopReason`, `agentStepMsMean`, `envStepMsMean`, `discreteAction` (when discrete), `continuousAction` (when continuous), `health`, `diagnostic` (only with `--diag` or for heartbeats).

**`health` block.** Symmetric across policies. Host-computable fields are populated for ANY policy (oak or random):

- `rewardRate` — host-tracked ρ̄ via incremental EMA (Reward Centering).
- `actionDeltaL2Mean` — mean L2 of consecutive `policyMean` deltas.
- `scoreGradL2Mean` — mean L2 of `scoreGrad`.

Kernel-side fields are populated only for `oak`:

- `oakOptionReinitCountFinal`, `oakFeatureReinitCountFinal` — cumulative age-gated generate-and-test replacement counts.
- `nonFiniteByPhase` = `{"encode": N, "learn": N, "plan": N, "tdPsi": N}` — per-phase NF counts. **Healthy = all zeros across the run.**

**Storage.** Default directory `.oak-benchmarks/<run-id>/`, with `<run-root>/<envId>/seed<N>/<stem>.json` for finals and `<stem>.heartbeat-NNNNNN.json` for pulses. Override the run root with `OAK_METRICS_STORAGE`.

## CLI

| Command | Use |
|---|---|
| `OaK run <env-id>` | Long single-env run with heartbeats. `OAK_STOP_AFTER_STEPS` controls measured-window length. |
| `OaK diagnose <env-id> [--policy oak\|random] [--steps N]` | Short timed run with `diagnostic` block (per-phase GPU timings). |
| `OaK benchmark` | Concurrent multi-env multi-seed sweep. `OAK_BENCH_STEPS`, `OAK_BENCH_SEED_COUNT`, `OAK_BENCH_POLICIES`, `OAK_BENCH_CONCURRENCY` (default 8). |

## Performance: Metal trace

Use Instruments → Metal System Trace with `tools/instruments/ShaderTemplate.tracetemplate` (Shader Timeline ON, no counters). Cross-check phase shares against `diagnostic.metalPhaseGpuMsMean` from a `--diag` JSON.

## Citations (all primary papers in `docs/papers/`)

- Alberta Plan: [2208.11173](docs/papers/AlbertaPlanForAIResearch_2208.11173.pdf)
- Reward-Respecting Subtasks: [2202.03466](docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf)
- Reward Centering: [2405.09999](docs/papers/RewardCentering_2405.09999.pdf)
- Implicit α: [2510.06149](docs/papers/ImplicitUpdates_AvgRewardTD_2510.06149.pdf)
- GraMa feature utility: [2505.24061](docs/papers/GraMa_MeasureGradients_2505.24061.pdf)
- Generate-and-test: Mahmood-Sutton-Pilarski 2012 ([03-mahmood-sutton-autostep](docs/papers/03-mahmood-sutton-autostep-ICML-2012.pdf)), Rahman 2020 thesis, Samani 2022 thesis
- Deep Fourier features: Lewandowski-Schuurmans-Machado 2024 ([2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf))
- MetaOptimize per-feature step sizes: Sharifnassab-Salehkaleybar-Sutton 2024 ([2402.02342](docs/papers/MetaOptimize_StepSizes_2402.02342.pdf))
- Big World Hypothesis: [2512.23419](docs/papers/WorldIsBigger_ComputationallyEmbeddedBWH_2512.23419.pdf)
- Three Dogmas of RL: [2407.10583](docs/papers/ThreeDogmasRL_2407.10583.pdf)
- Prioritized sweeping (Plan Step 9, Plan footnote 13): Moore-Atkeson 1993 (*Machine Learning* 13(1):103–130), Peng-Williams 1993 (*Adaptive Behavior* 1(4):437–454), Sutton-Szepesvári-Geramifard-Bowling 2008 (UAI). Tier D2 uses chord-weighted L2 of model TD error as the per-option priority signal.
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §8.4 (prioritized sweeping with Dyna), §9.5.4 (online normalization)
