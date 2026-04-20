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

- Subtasks defined by features (cumulant = reward, stopping value = feature attainment per Reward-Respecting Subtasks [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4).
- Options = policies + termination.
- Option models r̂(s,o), n̂(s,o) used for planning (RR-Subtasks eq. 17 expectation models).
- Planning over option models (RR-Subtasks eq. 19, asynchronous VI).
- **Option keyboard** = real-valued vector λ ∈ R^K, one component per subtask. Design A: chord blends K component options; model treats chord as opaque descriptor. Design B: chord defines a joint subtask.
- **Feedback / utility / remove-replace.** Continually assess utility of features, subtasks, options, and option models. Replace what isn't pulling weight (Plan Step 11 + RR-Subtasks §8 FCSTOMP).

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
6. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
7. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity (cap = R, α₀ = 1/cap, ridge = 1/(cap+1), K ≤ 32 = SIMD width). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
8. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
9. **Single-stream (B = 1).** One observation → one action → one reward per step. Multi-env benchmarking is N independent agent instances over N envs (OperationQueue), never one batched agent.
10. **Zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/`KernelScalars`, `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with smoke + 1.5M paired bench. **FP-reorder** (`simd_sum`, threadgroup reductions where the input order changes from sequential, `fma` decomposed into separate multiply + add): ship only after a 10-seed paired bench shows **per-seed parity within ±3 % of the prior commit's per-seed lift** — aggregate parity is necessary but insufficient; per-seed reshuffling that averages to zero is still considered an intelligence hit. The first attempt at C2-perf used `simd_sum` for `oak_meta`/`oak_plan` and showed aggregate parity but per-seed chatter std doubled (0.020 → 0.041); the follow-up restored bit-identical with lane-0 sequential reductions for ~4 µs cost.

## Architecture

The agent is a 10-kernel single-stream Step-11 Oak instance. Every learnable head and every state slot is Plan-required.

### Learnable parameters (7 buffers)

| Head | Shape | Role |
|---|---|---|
| `encoder` | (R/2) · 2·O | Perception (Fourier features W). **Tier C1**: each pre-activation row is `[W_std_obs ‖ W_trace]` — first O cols multiply `std_obs[o]` (instantaneous), next O cols multiply `obsTrace[o]` (low-pass at decay 1/baseO, integrated over ~baseO steps). Together every Fourier feature can encode both instantaneous and slow-frequency dynamics (POMDP coverage: gait phase, joint accel, sustained drift). **Fixed random projection at init** (RFF bandwidth-1 over both halves); evolves via generate-and-test reseeding (Plan Step 2). |
| `phase` | R/2 | Per-pre-activation Fourier offset (uniform on [0, 2π) at init). Reseeded alongside its `encoder` row in generate-and-test. Together with `encoder`, parameterizes the deep Fourier features basis (Lewandowski-Schuurmans-Machado 2024, [arXiv:2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf)). |
| `valueW` | R | Main value GVF (cumulant = reward, differential, ρ̄-centered). |
| `psi` | K · R | K subtask GVFs (cumulant = reward, stopping value = feature attainment per RR-Subtasks eq. 4). |
| `optPolicy` | K · D · R | K component policies, blended by chord (Design A keyboard). |
| `optReward` | K · R | K option reward models r̂^k (RR-Subtasks eq. 17). |
| `optTransition` | K · R · R | K dense option transition models `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]` (RR-Subtasks Sutton-Machado-Holland-Timbers-Tanner-White 2022 eq. 17, [arXiv:2202.03466](docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf)). |

Per-feature alpha is currently achieved structurally by KEEPING relevant features alive and REPLACING irrelevant ones via `oak_bk_admit_commit`. R is rounded up to the nearest even value at host init because each Fourier pre-activation produces a paired `(cos, sin)` channel into the R-wide state. Total Plan-required GVFs: **1 + 2K + KR²** (1 main + K subtasks + K option-reward + KR² option-transition), all learned in parallel from the single experience stream. Cap K ≤ 32 is the Apple SIMD width (structural, hardware-derived) and is what lets `oak_plan`'s K-thread parallel reduction fit in one simdgroup via `simd_max`.

### Pipeline (9 kernels per step, Metal 4 ICB)

```
bk_obs → encode → meta → predict → plan → plan_reduce
       → learn_heads → post_learn → bk_admit_commit
```

| Kernel | What it does |
|---|---|
| `oak_bk_obs` | Per-feature observation Welford EMA (Sutton-Barto §9.5.4) **plus Tier C1 obsTrace EMA** at fixed rate `1/baseO` over the just-standardized obs (`obsTrace[o] ← (1 − 1/baseO)·obsTrace[o] + (1/baseO)·std_obs[o]`). Updated FIRST so encode reads current stats + current trace. |
| `oak_encode` | Deep Fourier features (Lewandowski et al. 2024 [arXiv:2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf)). For `pre = feat % (R/2)`, `z = phase[pre] + Σ_o W_enc[pre, o]·std_obs[o] + Σ_o W_enc[pre, baseO+o]·obsTrace[o]`; `state[feat]` = `cos(z)` for `feat ∈ [0, R/2)` and `sin(z)` for `feat ∈ [R/2, R)`. Each pre-activation reads BOTH instantaneous (std_obs) AND integrated (obsTrace, Tier C1) — Plan Step 3 explicit "incorporation of older signals and traces". One thread per state slot. NF-preserve from previous slot. |
| `oak_meta` | K-thread parallel (one per option, fits one Apple SIMD group). Each lane computes `logit_k = ψ_k · s'` (R FMA), saves `chordPrev[k] ← chord[k]`, then `chord[k] = softmax_k(ψ(s'))` via `simd_max + simd_sum`. Lane 0 also computes `α_π = 1/(baseO + ‖s'‖²·‖∇log π‖²)` (Kim implicit-step, [arXiv:2510.06149](https://arxiv.org/abs/2510.06149)) and the ρ̄ EMA (Reward Centering [arXiv:2405.09999](https://arxiv.org/abs/2405.09999)). Zero learned termination parameters: chord is recomputed every step from ψ, the keyboard "plays" the relative subtask value with no per-option lifetime. |
| `oak_predict` | Main TD `δ = (r − ρ̄) + V(s') − V(s)` (Reward Centering [arXiv:2405.09999](https://arxiv.org/abs/2405.09999)) plus K subtask TDs `δ_k = (r − ρ̄) + (1−β_k)·curr[k] + β_k·ψ_k(s') − ψ_k(s)` (RR-Subtasks ψ-GVF) with **per-option adaptive β_k = (1 − 1/R) · α₀ / (α₀ + utilOpt[k])**, α₀ = 1/baseO. utilOpt[k] is the EMA of `|δ_k|²` from `oak_post_learn` (Kim implicit-step style self-bound, [arXiv:2510.06149](https://arxiv.org/abs/2510.06149)). Initialized at utilOpt[k] = R so β_k starts ≈ 0 (myopic, safe at random init); β grows toward the cap as ψ stabilizes; if `|δ_k|²` spikes (divergence threat), β automatically chokes. |
| `oak_plan` | Parallel RR-Subtasks asynchronous VI ([arXiv:2202.03466](docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf) eq. 17 + 19). **K threadgroups × R threads each** (one TG per option, one thread per output dim — 32 simdgroups in flight on Apple Silicon vs the prior 1). Each `(k, j)` thread computes one row of the dense option transition `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]` with `curr` and `prev` fused into a single R-load pass, writes `nTdMat[k·R + j] = curr[j] − n̂^k_j(prev)` for `oak_learn_heads`, and contributes `valueW[j]·n̂^k_j(curr)` to the per-option `vHat^k`. Per-TG reduction: `simd_sum` across each simdgroup → 2 partials per TG (R = 64 = 2 simdgroups) reduced in `threadgroup memory + barrier`. Lane 0 of TG k computes `r̂^k = optReward[k]·curr` (R FMA) and emits `Q^k = r̂^k + vHat^k` into the `qOption[K]` cache. |
| `oak_plan_reduce` | K-thread cross-option reduction (single simdgroup): `bestQ = simd_max(qOption)`; lane 0 computes `V(curr)` and writes `planAdv = (bestQ − V(curr)) / (1 + |bestQ − V(curr)|)` self-bounded into (−1, 1) (the standard rational soft-clamp from Sutton's gradient-TD work; structural, no tunable scale). |
| `oak_learn_heads` | One grid-stride pass updating all 5 weight buffers via 1-step TD. PlanAdv added ONLY to the value head (RR-Subtasks eq. 19 scope). All writes through `safe_w`. |
| `oak_post_learn` | Per-feature: GraMa gradient utility EMA ([arXiv:2505.24061](https://arxiv.org/abs/2505.24061)) **and** MetaOptimize L-approximation per-feature step-size update (Sharifnassab-Salehkaleybar-Sutton 2024, [arXiv:2402.02342](docs/papers/MetaOptimize_StepSizes_2402.02342.pdf), Algorithm 3, SGD base + SGD meta, Hessian-free). Constants structurally derived: γ = 1 − 1/R, η = 1/(R · baseO), α-cap = 1/baseO. Per-option: `|tdPsi|²` utility EMA. Action: μ_p = Σ_k chord[k]·(W_π[k][p]·s′). |
| `oak_bk_admit_commit` | Generate-and-test (Mahmood-Sutton-Pilarski 2012, Rahman 2020, Samani 2022): reseed worst-utility option AND worst-utility feature each step. Slot toggle, stepCount bump. |

### Action interface (Inv-1)

Each env declares a structured action spec — a list of components, each `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = Σ_components(component_param_count)` is the agent's only window into the env's action structure.

The agent emits `policyMean[D_policy]` from `oak_meta_action`. The env-wrapper layer (`EnvPoolEnv.sampleAndStep`) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax); scoreGrad = `1{a=d} − π_d` (Maddison 2014; Williams 1992).
- Box A: structural-σ=1 Gaussian (`μ + N(0,1)` clamped to env bounds); scoreGrad = pre-clamp `z` (Williams 1992 REINFORCE; the env-side clamp doesn't change the score function the agent's policy parameterized).

Adding a new env or action-space component type requires adding to `ActionComponent` in Swift; no agent or kernel changes.

## Run metrics JSON

Each `run`, `diagnose`, and `benchmark` writes one pretty-printed JSON file with stable alphabetical key order (`schemaVersion: 3`).

**Top-level keys:** `command`, `envId`, `policy`, `seed`, `numEnvs` (always 1 — single-stream), `warmupSteps`, `stopAfterMeasuredSteps`, `observationDim`, `actionDim`, `actionDiscreteCount` (when discrete), `measuredStepCount`, `measuredBatchCount`, `measuredSeconds`, `workerDepth = measuredStepCount`, `stepsPerSecond`, `rewardSum`, `doneCount`, `stopReason`, `agentStepMsMean`, `envStepMsMean`, `discreteAction` (when discrete), `continuousAction` (when continuous), `health`, `diagnostic` (only with `--diag` or for heartbeats).

**`health` block.** Symmetric across policies. Host-computable fields are populated for ANY policy (oak or random):

- `rewardRate` — host-tracked ρ̄ via incremental EMA (Reward Centering).
- `actionDeltaL2Mean` — mean L2 of consecutive `policyMean` deltas.
- `scoreGradL2Mean` — mean L2 of `scoreGrad`.
- `maxAbsPolicyOutput` — max |μ| over the window.

Kernel-side fields are populated only for `oak`:

- `oakOptionReinitCountFinal`, `oakFeatureReinitCountFinal` — cumulative generate-and-test count (one each per step under the current unconditional reinit policy).
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
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §9.5.4 (online normalization)
