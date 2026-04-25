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

- Subtasks defined by features (cumulant = reward, stopping value = feature attainment per Reward-Respecting Subtasks [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4). **Current rebuild OaK has the RR stopping-value branch live; benchmark promotion is still open.**
- Options = policies + termination. **Current rebuild continuation is derived from the RR stopping comparison, not H1 model-confidence hysteresis.**
- Option models r̂(s,o), n̂(s,o), and for average-reward options duration l̂(s,o), used for planning. **Current L3-L7 keeps the orthodox raw option-reward-plus-duration form: `Q = rHat - rho*lHat + V(nHat)`. A coherent differential option-reward alternative was tested and rejected by the 10k paired gate, so do not revive it without materially new evidence.**
- Planning over option models (RR-Subtasks eq. 19 / average-reward option backup).
- **Option keyboard** = real-valued vector λ ∈ R^K, one component per subtask. Design A: chord blends K component options; model treats chord as opaque descriptor. Design B: chord defines a joint subtask. **Current OaK has deleted the unproven signed-λ/`psiCross` keeper path and chooses one behavior option by RR continuation plus direct `psiNext`, preserving exact one-step off-policy ratios while a true chord-descriptor model remains open.**
- **Feedback / utility / remove-replace.** Continually assess utility of features, subtasks, options, and option models. Replace what isn't pulling weight (Plan Step 11 + RR-Subtasks §8 FCSTOMP). **Current rebuild has age-gated feature/option-bundle replacement, `optFeature`, feature age, option age, and feature-ever-used metadata; richer utility remains an active research frontier.**

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
6. **Inv-7: temporal uniformity** (Plan research-vision pillar #2; verbatim: *"all times are the same with respect to the algorithms running on the agent. There are no special training periods when training information is available or when rewards are counted more or less than others. If training information is provided... it is provided on every time step. If the agent learns or plans, then it learns or plans on every time step. If the agent constructs its own representations or subtasks, then the meta-algorithms for constructing them operate on every time step."*). Every Plan mechanism in OaK runs every step: perception (encode), prediction (meta_predict), planning (plan + plan_reduce), learning (learn_heads), meta-learning (post_learn MetaOptimize), and every-step G&T assessment with capacity/age-gated commits. Reward is centered (ρ̄) the same way every step. `track_rate(t, obsDim) = max(1/(t+1), 1/obsDim)` is the current structural tracking choice, not slide law; replace it if paired evidence or primary research shows a more faithful parameter-free tracker. Subsumes Inv-4 + Inv-5 in their consolidated Plan-citing form. **Forbidden constructs**: warmup-gated branches (`if stepCount > N`), epoch counters, train/eval splits, frozen weights post-N steps, special "exploration" vs "exploitation" phases. `OAK_WARMUP_STEPS` is host-side bench metric collection only — the agent's per-step computation is identical regardless.
7. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
8. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity or stream shape (cap = R, K = R, page width = hardware SIMD width, α₀ = 1/cap, ridge = 1/(age+1)). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
9. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
10. **Current single-agent stream semantics.** The live engine is one observation → one action → one reward per agent step (`B = 1`) because the Alberta Plan describes an agent's experience as a single time series. This is an implementation stance, not a sacred batch-size law: future batched kernels are allowed only if they are semantically N independent agents with disjoint state, no cross-stream reductions, and identical per-agent updates. Benchmark concurrency remains N independent worker processes for one environment/seed/policy grid.
11. **Inv-10: zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/`KernelScalars`, `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with smoke + 1.5M paired bench. **FP-reorder** (`simd_sum`, threadgroup reductions where the input order changes from sequential, `fma` decomposed into separate multiply + add): ship only after a 10-seed paired bench shows **per-seed parity within ±3 % of the prior commit's per-seed lift** — aggregate parity is necessary but insufficient; per-seed reshuffling that averages to zero is still considered an intelligence hit. The first attempt at C2-perf used `simd_sum` for `oak_meta`/`oak_plan` and showed aggregate parity but per-seed chatter std doubled (0.020 → 0.041); the follow-up restored bit-identical with lane-0 sequential reductions for ~4 µs cost.

## Architecture

The agent is a focused-kernel single-stream Oak prototype over the Plan steps. K is the active option/subtask count and now equals R; 32 is only the SIMD option-page width for bounded compute. Every learnable head and every state slot must be justified against the Plan; strict completion status is in `docs/roadmap.md`.

### Learnable parameters

| Head | Shape | Role |
|---|---|---|
| `valueW` | R | Main value GVF (cumulant = reward, differential, ρ̄-centered). |
| `psi` | K · R | K=R feature-indexed RR subtask GVFs. Cumulant is shared average-reward-centered reward; stopping value is RR Eq. 4: `V(s') - valueW[i]·x_i(s') + bonusW[i]·x_i(s')`. |
| `optPolicy` | K · D · R | K component policies. The wrapper samples the selected behavior option and returns exact one-step importance ratios for the active bank. |
| `optReward` | K · R | K stopping-aware option reward models r̂^k. |
| `optTransition` | K · R · R | K dense stopping-aware transition models `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]`. |
| `optDuration` | K · R | K average-reward option-duration models l̂^k with a structural one-step lower bound at decode. |

State representation is a finite adaptive predictive state, not a full reconstruction of the world: direct normalized experience channels (observation coordinates, multi-timescale observation traces, previous abstract action descriptor, previous reward, and an affine channel) plus constructed feature slots. Deep Fourier, random Fourier, and random-projection encoders are optional experimental modes, not Alberta Plan law. Constructed slots remain replaceable signed products/sums/residuals of useful parents. Plasticity is driven by MetaOptimize-style `alpha/beta/h/gradEnergy` families plus age/utility replacement; raw IDBD/Autostep are historical relatives, not the implementation target.

### Pipeline (6 Metal dispatches per step, direct command encoding)

```
oak_encode → oak_plan → oak_plan_reduce → oak_learn_heads → oak_replace → oak_action_emit
```

| Kernel | What it does |
|---|---|
| `oak_encode` | Updates observation/traces, builds normalized anchor/constructed state features, then runs prediction/control: value TD, RR subtask TD, RR continuation, and direct RR option selection. |
| `oak_plan` | Updates the scheduled option page/slots' stopping-aware model TDs, duration TDs, and raw-reward option model. |
| `oak_plan_reduce` | Reduces model Q/priority and selects high-priority/stale model pages for fair search control. When K fits one SIMD page, the whole option bank is planned. |
| `oak_learn_heads` | Updates value, psi, option policy, and option reward/transition/duration using exact one-step ratios and wrapper-provided target-option score gradients. |
| `oak_replace` → `oak_action_emit` | Commits utility/meta-learning diagnostics and age/capacity-gated replacements, then emits all option policy means after a dispatch boundary so action emission cannot race replacement writes. |

### Action interface (Inv-1)

Each env declares one structured action spec: `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = component_param_count` is the agent's only window into the env's action structure.

The agent emits `policyMean[D_policy]` from `oak_meta_action`. The env-wrapper layer (`EnvPoolEnv.sampleAndStep`) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax); target-option actor scores use `1{a=d} − π_d` (Maddison 2014; Williams 1992).
- Box A: currently exposes `2A` policy coordinates to keep `D_policy`, default rank, and K stable. The first A coordinates are means; the second A coordinates are wrapper-reserved/inert. The wrapper samples with structural fixed `σ = 1/D_policy`, emits residual-style mean credit (`raw − μ`) for Box mean slots, writes zero score/action-signal entries for reserved scale slots, and clamps only the env-facing action. Full `D_policy = A` mean-only deletion failed the 1M gate and remains rejected unless a different actor geometry earns fresh evidence.

Adding a new env or action-space shape requires adding to `ActionSpec` in Swift; no agent or kernel changes.
`doneCount` is a wrapper random-restart metric, not an episode signal to OaK; the engine state is never reset on `done`.

## Run metrics JSON

Each benchmark worker writes one pretty-printed JSON file with stable alphabetical key order (`schemaVersion: 5`).

**Top-level keys:** `command`, `envId`, `policy`, `seed`, `numEnvs` (always 1 — single-stream), `stopAfterMeasuredSteps`, `observationDim`, `actionDim`, `actionDiscreteCount` (when discrete), `measuredStepCount`, `measuredSeconds`, `stepsPerSecond`, `rewardSum`, `doneCount`, `stopReason`, `heartbeatSequence`, and `health`.

**`health` block.** Symmetric across policies. Host-computable fields are populated for ANY policy (oak or random):

- `rewardRate` — host-tracked ρ̄ via incremental EMA (Reward Centering).
- `actionDeltaL2Mean` — mean L2 of consecutive `policyMean` deltas.
- `continuingRestartCount` — mirror of wrapper `doneCount`, named to make clear that terminal events are random restarts outside the agent.

Kernel-side fields are populated only for `oak`:

- `oakOptionReinitCountFinal`, `oakFeatureReinitCountFinal` — cumulative age-gated generate-and-test replacement counts.
- `nonFiniteByPhase` = `{"encode": N, "learn": N, "plan": N, "tdPsi": N}` — per-phase NF counts. **Healthy = all zeros across the run.**
- Search/model diagnostics: `modelPriorityMax`, `optionDurationMean`, `terminationRate`.
- Actor/support diagnostics: `importanceEss`, `actorUpdateNormMean`, `actorSupportEntropy`.

**Storage.** Default directory `.oak-benchmarks/<run-id>/`, with `<run-root>/<envId>/seed<N>/<stem>.json` for finals and `<stem>.heartbeat-NNNNNN.json` for pulses. Override the run root with `OAK_METRICS_STORAGE`.

## CLI

| Command | Use |
|---|---|
| `OaK benchmark` | Concurrent single-env multi-seed sweep. Each item runs in its own worker process so native env/global state cannot leak across independent streams. `OAK_BENCH_ENV` selects the env (default `HalfCheetah-v4`); `OAK_BENCH_STEPS`, `OAK_BENCH_SEED_COUNT`, `OAK_BENCH_POLICIES`, `OAK_BENCH_CONCURRENCY` (default/max = all logical cores; accepts `auto`/`max`). One benchmark launcher owns the workspace at a time; nonempty requested roots get a fresh suffixed output dir, and finished worker slots are refilled deterministically from the canonical work list. |

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
- Prioritized sweeping (Plan Step 9, Plan footnote 13): Moore-Atkeson 1993 (*Machine Learning* 13(1):103–130), Peng-Williams 1993 (*Adaptive Behavior* 1(4):437–454), Sutton-Szepesvári-Geramifard-Bowling 2008 (UAI). Current priority is expected option-model improvement with page fairness.
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §8.4 (prioritized sweeping with Dyna), §9.5.4 (online normalization)
