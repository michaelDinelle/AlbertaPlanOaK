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
6. **Inv-7: temporal uniformity** (Plan research-vision pillar #2; verbatim: *"all times are the same with respect to the algorithms running on the agent. There are no special training periods when training information is available or when rewards are counted more or less than others. If training information is provided... it is provided on every time step. If the agent learns or plans, then it learns or plans on every time step. If the agent constructs its own representations or subtasks, then the meta-algorithms for constructing them operate on every time step."*). Every Plan mechanism in OaK runs every step at the same rate: perception (encode), prediction (meta_predict), planning (plan + plan_reduce), learning (learn_heads), meta-learning (post_learn MetaOptimize), G&T option/feature replacement (bk_admit_commit). Reward is centered (ρ̄) the same way every step. **The Plan also says** *"if the agent can reduce its speed of learning about parts of the environment when they appear stable, then it can also increase its speed of learning when they start to change"* — `track_rate(t, baseO) = max(1/(t+1), 1/baseO)` honors this by slowing the EMA rate as data accumulates while flooring at `1/baseO` so adaptation never freezes. Subsumes Inv-4 + Inv-5 in their consolidated Plan-citing form. **Forbidden constructs**: warmup-gated branches (`if stepCount > N`), epoch counters, train/eval splits, frozen weights post-N steps, special "exploration" vs "exploitation" phases. `OAK_WARMUP_STEPS` is host-side bench metric collection only — the agent's per-step computation is identical regardless.
7. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
8. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity (cap = R, α₀ = 1/cap, ridge = 1/(cap+1), K ≤ 32 = SIMD width). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
9. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
10. **Single-stream (B = 1).** One observation → one action → one reward per step. Multi-env benchmarking is N independent agent instances over N envs (OperationQueue), never one batched agent.
11. **Inv-10: zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/`KernelScalars`, `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with smoke + 1.5M paired bench. **FP-reorder** (`simd_sum`, threadgroup reductions where the input order changes from sequential, `fma` decomposed into separate multiply + add): ship only after a 10-seed paired bench shows **per-seed parity within ±3 % of the prior commit's per-seed lift** — aggregate parity is necessary but insufficient; per-seed reshuffling that averages to zero is still considered an intelligence hit. The first attempt at C2-perf used `simd_sum` for `oak_meta`/`oak_plan` and showed aggregate parity but per-seed chatter std doubled (0.020 → 0.041); the follow-up restored bit-identical with lane-0 sequential reductions for ~4 µs cost.

## Architecture

The agent is a 10-kernel single-stream Step-11 Oak instance. Every learnable head and every state slot is Plan-required.

### Learnable parameters (7 buffers)

| Head | Shape | Role |
|---|---|---|
| `encoder` | (R/2) · 2·O | Perception (Fourier features W). **Tier C1**: each pre-activation row is `[W_std_obs ‖ W_trace]` — first O cols multiply `std_obs[o]` (instantaneous), next O cols multiply `obsTrace[o]` (low-pass at decay 1/baseO, integrated over ~baseO steps). Together every Fourier feature can encode both instantaneous and slow-frequency dynamics (POMDP coverage: gait phase, joint accel, sustained drift). **Fixed random projection at init** (RFF bandwidth-1 over both halves); evolves via generate-and-test reseeding (Plan Step 2). Tier C3 recurrent self-feedback was attempted (encoder reads `state[t-1]` as a third region) and regressed by ~52% on the 10×1.5M paired bench in both RFF-1 and variance-balanced inits — see `docs/roadmap.md` for the negative result; recurrent dynamics create double non-stationarity that OaK's TD bootstrap cannot track. |
| `phase` | R/2 | Per-pre-activation Fourier offset (uniform on [0, 2π) at init). Reseeded alongside its `encoder` row in generate-and-test. Together with `encoder`, parameterizes the deep Fourier features basis (Lewandowski-Schuurmans-Machado 2024, [arXiv:2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf)). |
| `valueW` | R | Main value GVF (cumulant = reward, differential, ρ̄-centered). |
| `psi` | K · R | K subtask GVFs (cumulant = reward, stopping value = feature attainment per RR-Subtasks eq. 4). |
| `optPolicy` | K · D · R | K component policies, blended by chord (Design A keyboard). |
| `optReward` | K · R | K option reward models r̂^k (RR-Subtasks eq. 17). |
| `optTransition` | K · R · R | K dense option transition models `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]` (RR-Subtasks Sutton-Machado-Holland-Timbers-Tanner-White 2022 eq. 17, [arXiv:2202.03466](docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf)). |

Per-feature alpha is currently achieved structurally by KEEPING relevant features alive and REPLACING irrelevant ones via `oak_bk_admit_commit`. R is rounded up to the nearest even value at host init because each Fourier pre-activation produces a paired `(cos, sin)` channel into the R-wide state. Total Plan-required GVFs: **1 + 2K + KR²** (1 main + K subtasks + K option-reward + KR² option-transition), all learned in parallel from the single experience stream. Cap K ≤ 32 is the Apple SIMD width (structural, hardware-derived) and is what lets `oak_plan`'s K-thread parallel reduction fit in one simdgroup via `simd_max`. **Tier C2 (option termination via `termW` + closed-form geometric Q^k + hold-or-refresh chord) was attempted three times — pre-C1 v1 (β₀ = 1/R), pre-C1 v2 (β₀ = 0.5), and post-C1 retry — and regressed every time** (-99% to -100% mean lift vs C1). See `docs/roadmap.md` Tier C entry for the full negative-result writeup; the failure mode appears structurally incompatible with OaK's TD-bootstrapped fixed-random-feature setup regardless of substrate richness.

### Pipeline (9 kernels per step, Metal 4 ICB)

```
bk_obs → encode → meta → predict → plan → plan_reduce
       → learn_heads → post_learn → bk_admit_commit
```

| Kernel | What it does |
|---|---|
| `oak_bk_encode` | **FUSED** (oak_bk_obs + oak_encode). 1 TG of R=rank threads. **Phase 1 (tid < O)**: Per-feature observation Welford EMA (Sutton-Barto §9.5.4) **plus Tier I1 multi-timescale obsTrace EMAs** at structural rates `1/2^m` for `m=1..M-1` where `M = floor(log2(baseO)) + 1` (HC: M=5 → 4 trace levels at τ ∈ {2,4,8,16}). Each obs-thread updates all M-1 trace levels for its `o`. Replaces Tier C1's single trace at fixed rate `1/baseO`. **threadgroup_barrier(mem_device)**. **Phase 2 (all R threads)**: Deep Fourier features (Lewandowski et al. 2024 [arXiv:2410.20634](docs/papers/PlasticLearningDeepFourierFeatures_2410.20634.pdf)). For `pre = feat % (R/2)`, encoder row width is M·baseO with M concatenated regions: `z = phase[pre] + Σ_o W_enc[pre, o]·std_obs[o] + Σ_{m=1..M-1} Σ_o W_enc[pre, m·baseO+o]·obsTraceMulti[(m-1)·baseO + o]`; `state[feat]` = `cos(z)` for `feat ∈ [0, R/2)` and `sin(z)` for `feat ∈ [R/2, R)`. NF-preserve from previous slot. Plan Step 3 literal: *"incorporation of older signals and traces"* — temporal coverage spans gait-cycle (τ=2-8) through baseO-scale slow drift (τ=16). Saves 1 dispatch + 1 host memoryBarrier per step (encode kernels still fused). |
| `oak_meta_predict` | **FUSED** (oak_meta + oak_predict). K threads (1 simdgroup). **Phase 1**: chord softmax (`logit_k = ψ_k · s'` then `softK = softmax_k`) via `simd_max` + lane-0 `simd_shuffle` reduction. **Tier H1 (Plan Step 10 STOMP multi-step options) chord hysteresis**: `chord_raw = β^k · chordPrev[k] + (1 − β^k) · softK` where `β^k = prevBeta[k] = 1/(1 + ntdSqSum[k])` from previous step's `oak_plan` (model-confidence signal; high β ⇔ confident ⇔ option continues, low β ⇔ model surprised ⇔ 1-step fallback). Renormalized via `simd_sum(chord_raw)` so `Σ chord = 1`. Diagnostic `chordPersistenceMeanAbs = Σ_k chord[k]·chordPrev[k]` surfaced to KernelStats. At init (fresh optTransition, large ntdSqSum, β ≈ 0) chord = softK → 1-step behavior preserved; options gain multi-step persistence automatically as optTransition learns. Sidesteps the ψ-coupled termination family exhausted by C2/C4/Step-11 (β derives from forward-model-prediction error, not ψ/backward-value). Lane 0 updates `α_π = 1/(baseO + ‖s'‖²·‖∇log π‖²)` (Kim implicit-step, [arXiv:2510.06149](https://arxiv.org/abs/2510.06149)) and ρ̄ EMA (Reward Centering [arXiv:2405.09999](https://arxiv.org/abs/2405.09999)). **simdgroup_barrier(mem_device)** to flush lane-0's ρ̄ write before phase 2 reads it. **Phase 2**: main TD `δ = (r − ρ̄) + V(s') − V(s)` on lane 0 + per-option subtask TDs `δ_k = (r − ρ̄) + (1−β_k)·curr[k] + β_k·ψ_k(s') − ψ_k(s)` (RR-Subtasks eq. 17) on lane k with **stability-adaptive β_k = (1 − 1/R)·α₀/(α₀ + utilOpt[k])** (α₀ = 1/baseO). `utilOpt[k]` is the EMA of `|tdPsi[k]|²` from `oak_post_learn` (Kim implicit-step self-bound); β_k is independent from Tier H1's chord-hysteresis β^k (different slots, different roles — phase-2 β_k controls ψ bootstrap horizon, H1 β^k controls chord persistence). Uses the POST-phase-1 ρ̄ via direct `scalars->rho` dereference. Saves 1 dispatch + 1 host memoryBarrier per step. |
| `oak_plan` | Parallel RR-Subtasks asynchronous VI ([arXiv:2202.03466](docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf) eq. 17 + 19) + **Tier D2** in-line per-option priority signal (Plan Step 9, Moore-Atkeson 1993). **K threadgroups × R threads each** (one TG per option, one thread per output dim — 32 simdgroups in flight on Apple Silicon vs the prior 1). Each `(k, j)` thread computes one row of the dense option transition `n̂^k_j(s) = Σ_i w^k_{j,i}·s[i]` with `curr` and `prev` fused into a single R-load pass, writes `nTdMat[k·R + j] = curr[j] − n̂^k_j(prev)`, AND keeps `n_td_sq = n_td · n_td` in-register (Tier D2 free with ILP). Two parallel TG arrays (`tg_nHatCurr`, `tg_ntdSq`) written before the **single** existing barrier (zero added barriers). Lane 0 runs ONE R-loop interleaving `vHat = Σ_j valueW[j] · nHatCurr_j` (bit-id fma chain preserved) and `ntdSqSum = Σ_j n_td_j²` (Apple GPU dual-issues independent fma chains), then computes `r̂^k = optReward[k] · curr` (R FMA), emits `Q^k = r̂^k + vHat^k` into `qOption[K]`, AND writes two per-option signals from `ntdSqSum`: `prioOpt[k] = isfinite(ntdSqSum) && ntdSqSum > 0 ? chord[k] · sqrt(ntdSqSum) : 0` (Tier D2 prioritized sweeping) and `prevBeta[k] = 1/(1 + ntdSqSum)` (**Tier H1 chord-hysteresis continuation probability** — read by NEXT step's oak_meta_predict phase 1; 1-step lag is intentional). |
| `oak_plan_reduce` | K-thread cross-option reduction (single simdgroup): `bestQ = simd_max(qOption)` (bit-id; max is associative) + Tier D2 `prioSum = simd_sum(prioOpt)` (new reduction, no prior baseline applies). Lane 0 computes `V(curr)`, writes `planAdv = (bestQ − V(curr)) / (1 + |bestQ − V(curr)|)` self-bounded into (−1, 1), AND writes `meanPrio = prioSum / K` to `KernelScalars.meanPrio` (broadcast-cached for the 131k Head 4 threads to read in one cycle each). |
| `oak_learn_heads` | One grid-stride pass updating all 5 weight buffers via 1-step TD. PlanAdv added ONLY to the value head (RR-Subtasks eq. 19 scope). **Tier D3 Head 1 off-policy attribution** (Plan Step 10: *"The learning processes are conditional on the option, and so will need to be done off-policy."*): the ψ update is chord-weighted, `Δψ[k,i] = α[i]·tdPsi[k]·prev[i]·max(chordPrev[k], 1/K)`. Prior to D3, ψ was the only head with no chord weighting, so every ψ_k converged to the chord-level value regardless of option identity. The 1/K floor is a structural minimum (K host-capped at simdgroup width 32) that prevents self-reinforcing starvation when chord[k]→0. Heads 2–4 already chord-weight `chordPrev[k]` in their gradients. **Tier D2 Head 4 gate**: dense `optTransition` update (K·R² FMA = ~70% of kernel cost) is gated by `prioOpt[k] >= meanPrio` — about half of K options skip per step on average (Moore-Atkeson 1993 prioritized backups). The gate is uniform across each simdgroup since the thread layout `gid → (k, j, i)` with `R² = 4096 ≫ 32` puts all 32 lanes on the same `k` → zero divergence cost. All writes through `safe_w`. Tier G1 TD(λ) attempt (both normalized and true-online forms, on valueW and ψ) was rolled back — see `docs/roadmap.md` for the stability-analysis negative-result writeup. |
| `oak_post_learn` | Three concurrent regions in one kernel (R + K + D·K threads, tgSize = 32). **Region 1 (R threads)**: per-feature GraMa gradient utility EMA ([arXiv:2505.24061](https://arxiv.org/abs/2505.24061)) **and** MetaOptimize L-approximation per-feature step-size update (Sharifnassab-Salehkaleybar-Sutton 2024, [arXiv:2402.02342](docs/papers/MetaOptimize_StepSizes_2402.02342.pdf), Algorithm 3, SGD base + SGD meta, Hessian-free). Constants structurally derived: γ = 1 − 1/R, η = 1/(R · baseO), α-cap = 1/baseO, α-floor = 1/(R·baseO). **Region 2 (K threads)**: per-option `|tdPsi|²` utility EMA. **Region 3 (D·K threads, D simdgroups of K=32 lanes each)**: action μ_p = Σ_k chord[k]·(W_π[k][p]·s′). Each (p, k) is its own thread; lane k of TG p computes `dot_k = Σ_i W_π[k][p][i]·s′[i]` in parallel, then lane 0 reads each lane's dot via `simd_shuffle(dot, kk)` and accumulates `μ = fma(chord[kk], d, μ)` in the original kk=0..K-1 order — bit-identical to the prior single-thread loop (Inv-10 preserved by construction). Old: 6 threads × 2048 sequential fma in one simdgroup with 26 idle lanes (~110 µs/step, 55% of total GPU). New: 6 simdgroups × 32 lanes × ~64 parallel cycles → **−90% kernel time, −50% total step GPU time**. |
| `oak_bk_admit_commit` | Generate-and-test (Mahmood-Sutton-Pilarski 2012, Rahman 2020, Samani 2022): reseed worst-utility option AND worst-utility feature each step. `worstK = argmin_k utilOpt[k]` (EMA(|tdPsi|²); low ⇒ converged, safe to replace); `worstI = argmin_i utilFeat[i]` (GraMa-based). Slot toggle, stepCount bump. Step-11-sharpening (argmin `planUtil = EMA(Q^k − V)`) rolled back — 10M `−16%` regression. Tier J1 mean-utility gate rolled back — argmin is mathematically always multiple σ below mean for K≈32 samples (central-statistic gate family exhausted). Tier K1 Plan-Step-2 feature combinations (top-2 row-sum × 1/√2 encoder reseed) rolled back — 1.5M screen clean (+449k, 10/10 wins), but 10M collapsed to `−43%` vs F1 with 2/10 seeds catastrophically negative and effK still uniform (31.5) — compositional reseed inherits from the same top-2 every step, collapsing feature diversity after early convergence. See `docs/roadmap.md` exhausted-families table. |

### Action interface (Inv-1)

Each env declares a structured action spec — a list of components, each `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = Σ_components(component_param_count)` is the agent's only window into the env's action structure.

The agent emits `policyMean[D_policy]` from `oak_meta_action`. The env-wrapper layer (`EnvPoolEnv.sampleAndStep`) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax); scoreGrad = `1{a=d} − π_d` (Maddison 2014; Williams 1992).
- Box A: **structural-σ = 1/√A Gaussian** (`μ + (1/√A)·N(0,1)` clamped to env bounds); scoreGrad = `z·√A` (Williams 1992 REINFORCE for σ-parameterized Gaussian, `d log π/dμ = (a−μ)/σ² = z/σ`). Tier F1 (plateau-breaker): the prior σ=1 gave per-step action noise magnitude ≈ √A across the box, which is ~100% of HC's [-1, 1] action range for A=6 — committed option policies were drowned by exploration noise. F1 scales σ with action dim (HC: σ=0.41), so per-component noise stays bounded while the total action-vector-length noise remains ≈ 1. No tunable; A is the Plan-supplied env action-dim. Env-side clamp doesn't change the score function the agent's policy parameterized.

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
- Prioritized sweeping (Plan Step 9, Plan footnote 13): Moore-Atkeson 1993 (*Machine Learning* 13(1):103–130), Peng-Williams 1993 (*Adaptive Behavior* 1(4):437–454), Sutton-Szepesvári-Geramifard-Bowling 2008 (UAI). Tier D2 uses chord-weighted L2 of model TD error as the per-option priority signal.
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §8.4 (prioritized sweeping with Dyna), §9.5.4 (online normalization)
