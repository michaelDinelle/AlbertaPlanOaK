# AGENTS

## Alberta Plan Slide Law

### The Alberta Plan roadmap (steps 1–11)

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

### The Alberta Plan's Research vision: Intelligence as signal processing over time

Four key features of DeepMind Alberta's AI research

1. Experience oriented. Experience is the data of AI
2. Online, temporally uniform, continually running & learning
3. Concern for computational cost, response time
4. Agents interacting with environments containing other agents

### Outline (take-home messages)

- Understanding intelligence is a grand scientific prize that may soon be within reach; the glory will go to the bold
- The Alberta Plan is a direct run at this great prize
- emphasizing intelligence as the acquisition of new knowledge
- My motivation for the plan stems from the Oak architecture, a complete conceptual design for a Model-based RL agent, a Proto-AI
- The Alberta Plan is a massive retreat; going back to the most basic algorithms; re-vamping them for continual and meta-learning along the roadmap steps above

### First, the Big World Perspective

- The world is vastly more complex than the agent
- After all, the world contains many other agents, and what is going on in many of their minds matters for yours
- It follows that you can never accurately know the state of the world, much less its transition dynamics; your knowledge is a gross approximation of reality
- This alone makes prior knowledge a never-ending fool's errand, a mistake
- The agent must adapt its approximations to the particular part of the world that it encounters
- And do this continually as the part encountered changes
- The agent's approximations will be non-stationary; they must track; this is continual learning
- When learning is continual, extensive repetitive experience is obtained with learning; this makes meta-learning-learning how to learn-possible and essential
- Thus continual learning and meta-learning (e.g., representation learning) are naturally linked

### The Alberta Plan's model of the intelligent agent

- Adds SubTasks (≈feature-control subtasks)
- Multiple policies and value functions

### The common model of the intelligent agent

The agent comprises four components:

- Perception produces the state representation used by all components
- Reactive Policy quickly produces an action appropriate to the state
- Value Function evaluates how well things are going, and changes the policy (learning)
- Transition model predicts the consequences of alternate actions, and changes the policy (planning)

### The Alberta Plan eschews domain knowledge

- The Plan should contain nothing specific to the human world
- nothing about vision, space, objects, language, or other agents
- nothing that distinguishes people from other animals
- nothing about specific ecological niches
- No content, just algorithms for acquiring and organizing knowledge

Learning first, domain knowledge last.
Why?

- To keep the problem small, the methods scalable
- To keep the focus on acquiring new knowledge

### The Alberta Plan takes the conventional RL interface seriously

- A complex computational agent interacts with a vastly more complex environment to maximize its reward
- optimality is impossible
- approximation is required
- tracking is required
- The interaction is continual and temporally uniform
- there are no special training periods
- there are no episodes, no discounting
- Actions and observations are just signals with no intrinsic meaning; they are just information
- no language, vision, space, objects, or geometry

### The Oak Architecture (Options And Knowledge) is based on a cycle of discovery

- Models of options
- State features
- Subtasks to attain features
- Options/values that solve (sub)tasks
- Arrows show the direction of primary flow
- Each arrow has a slower backward flow of credit
- "I use you, so don't change too much"
- The consumers of state and time abstractions support, evaluate, and shape the abstractions
- Cycle of discovery produces abstractions tailored to the environment and ultimately tied to reward

## Literature anchors (Alberta Plan & reward-respecting STOMP)

Primary sources in `docs/papers/`: *The Alberta Plan for AI Research* ([arXiv:2208.11173](https://arxiv.org/abs/2208.11173)); *Reward-Respecting Subtasks for Model-Based Reinforcement Learning* ([arXiv:2202.03466](https://arxiv.org/abs/2202.03466), Artificial Intelligence 2023). The following is implementation-agnostic content from those documents—constraints on this codebase remain the **Paper-Grounded Corollaries** below.

- **Base agent (Alberta Plan).** Perception builds **state** shared by all parts; **reactive policies** (the primary policy and possibly others); **value functions** for learning those policies; **transition model** learned from experience. The model may be **temporally abstract**: it can take an **option** (a way of behaving plus a **termination condition**) and predict state at termination and cumulative reward along the way. **Planning** uses the model to imagine outcomes evaluated by value functions; it is **continual** and treated as ongoing background work while perception, policy, and value updates run on every step in the foreground.
- **STOMP progression (both sources).** **Subtask → Option → Model → Planning**: pose a subtask, learn an option solving it, learn that option’s transition model, then use the model in planning. The papers present the steps sequentially for clarity but state they can also run **in parallel** in the long-run architecture.
- **Reward-respecting subtasks (Reward-Respecting Subtasks paper).** Subsidiary tasks should be chosen so option models are **useful to planning**, not only easy to discover. A **reward-respecting** subtask uses the **main task reward** as the cumulant and defines **stopping values** using the agent’s **value estimates**, including **feature-attainment** variants where stopping is encouraged when a selected feature is **high** (via a **bonus** on top of the estimated value for that feature). That sharply narrows the space of options compared to subtasks that optimize unrelated cumulants. Learning is framed with **GVFs** and **off-policy** updates where appropriate.
- **Oak (Prototype-AI III in the Alberta Plan).** Extends STOMP with **feedback**: continually assess the **utility** of **features, subtasks, options, and option models**; **remove or replace** elements that are not pulling their weight (e.g. an option model **unused in planning** should not persist indefinitely with its option and subtask). **Features** are **reranked** by usefulness to learning and planning so the set of feature-based subtasks can change. **Option keyboard:** a real-valued vector with one component per subtask-linked option; a **chord** (several nonzero components) refers to a combination. The Plan describes **two** keyboard designs: (1) learn component options in the usual way and treat a chord as a **fixed blend** of those options, with the model learning **chord** dynamics as played; (2) interpret the vector as defining a **single** multi-component subtask (main reward plus a terminal value tied to all active keys), with options learned toward that joint subtask while the model treats the vector as an opaque **descriptor** of the option.
- **FC-STOMP (Reward-Respecting Subtasks, §8).** **Feature construction** feeding STOMP, with **feedback from later stages to earlier**—e.g. planning effectiveness and **search control** influencing which **features** warrant **feature-attainment** options and models—is identified with the **Oak** architecture in that paper’s terminology. Open directions there include **search control** over which states receive planning updates, planning to improve **subtask** value functions, and **compositional** modeling (higher-level transition structure built from lower-level models)—all subject to the same **no domain knowledge / no benchmark patching** rules as the rest of this project.

## Hard Rules

- Project is greenfield. It should never have legacy code or legacy fallbacks. Replace or refactor old paths instead of adding compatibility fallbacks.
- **Env-type knowledge boundary (Inv-1):** the agent (`OakMetalEngine`, `OakKernel.metal`, `KernelConfig`) must contain ZERO action-space-type or env-identity branches. Action-space type (discrete vs continuous), per-dim bounds, env id, and any other env-specific metadata live exclusively in the env-wrapper layer (`EnvPoolEnv` and below). The agent's contract with the wrapper is: env declares a structured action spec `[(Discrete N | Box A, low, high), …]`; agent reads only the integer `D_policy` (sum of per-component policy-param widths) and `obsDim`; wrapper composes per-component sampling and computes the matching score function. Adding new env suites or new action-space component types must NOT require touching the agent.
- **Multi-signal Pareto gate is the primary acceptance signal (Pass 12).** The agent is "learning" iff its bench performance shows: (1) NO rewardRate regression at p < 0.05, AND (2) at least ONE of the five principled channels improves at p < 0.05 — `rewardRate↑` (outcome; Alberta Plan), `LPM↑` (decreasing |outcomeError| trend over training; learning progress monitoring; Hou-An-Du [`2509.25438`](docs/papers/LPM_NoiseRobustExploration_LearningProgressMonitoring_2509.25438.pdf) ICLR 2026), `utilFeat↑` (`bk_feature` gradient-magnitude utility; GraMa [`2505.24061`](docs/papers/GraMa_MeasureGradients_2505.24061.pdf)), `pEff↑` (effective-rank floor; RankGradientLost [`2604.01913`](docs/papers/RankGradientLost_SWD_2604.01913.pdf)), `agent_ms↓` (FLOPs/reward Pareto efficiency). Backed by Three Dogmas of RL [`2407.10583`](docs/papers/ThreeDogmasRL_2407.10583.pdf) (Bowling, Alberta-Plan co-author): reward maximization alone is not the sole objective; representation quality matters independently. Single-channel rewardRate is provably insufficient because 36/84 bench envs are sparse-reward zero-equivalent (rewardRate gives no discrimination signal). The multi-signal gate captures learning across all five principled axes; any commit that fails multi-signal Pareto on any env is a regression. Pass-11's monotonic-rewardRate gate is a STRICT SUBSET of the new multi-signal gate (rewardRate ↑ implies multi-signal PASS), so Pass-11 verifications remain valid.
- **Maximum FLOP/rewardRate (Pareto principle).** Every kernel in the pipeline must earn its FLOPs by demonstrated multi-signal contribution. No kernel may be added in any future pass without an ablation proving paired-t Δ > 0 at p < 0.05 on at least ONE of the five channels across the full bench scope (84 envs × ≥10 seeds × ≥50,000 steps), AND no rewardRate regression. Symmetrically, any existing kernel whose ablation shows zero help on any of the four agent-side channels (rewardRate, LPM, utilFeat, pEff — agent_ms doesn't count as an EARNING vote since keeping a kernel COSTS FLOPs) is a candidate for removal. Pipeline size is a Pareto-frontier outcome, not a target — Pass 12 will likely shrink it from 18 → 14-16 kernels via this rule.

## Paper-Grounded Corollaries

- Benchmark-driven patching is forbidden.
- OaK must stay online, temporally uniform, and average-reward.
- No train/eval split, no discounting, and no episodic special cases beyond target gating at termination.
- The agent must keep one shared state representation used by perception, reactive policy, value functions, transition models, subtasks, options, and planning.
- Oak must include reward-respecting subtasks, multiple policies and value functions, and models of options used by planning.
- Domain knowledge is forbidden. No environment-specific heuristics, no benchmark-specific shaping, and no task-identity-conditioned control logic.
- Computational cost and response time are first-class. Do not grow repo-resident diagnostic frameworks in the hot path.

## Performance: run metrics JSON (throughput and host timing)

Each **`run`**, **`diagnose`**, and **`benchmark`** writes one **pretty-printed** JSON file with **stable, alphabetical key order** (encoder **`sortedKeys`**): **`schemaVersion`** (see emitted JSON; extended fields use **`2`**), **`command`**, and one flat run description—identity and config (**`envId`**, **`policy`**, **`seed`**, **`numEnvs`**, **`warmupSteps`**, **`stopAfterMeasuredSteps`**, **`observationDim`**, **`actionDim`**, optional **`actionDiscreteCount`**), measured window (**`measuredStepCount`**, **`measuredBatchCount`**, **`measuredSeconds`**, **`stepsPerSecond`**, **`rewardSum`**, **`doneCount`**, **`stopReason`**), per-batch timing means (**`agentStepMsMean`**, **`envStepMsMean`**), optional **`discreteAction`** when **`actionDiscreteCount`** is set (**`histogram`**, **`uniqueCount`**), optional **`continuousAction`** when actions are continuous (**`meanPerDim`**, **`stdPerDim`**, **`meanL2`**, **`stdL2`** over env-step samples in the measured window—near-zero **`stdPerDim`** / **`stdL2`** suggests collapsed or near-constant actuation), optional **`health`** (kernel stats; omitted for **`random`** when there is nothing to record), optional **`diagnostic`** (host/GPU timing shares; **`diagnose`** enables GPU phase timing in **`diagnostic`** when configured). Keys with no value are **omitted** (not `null`), aside from numeric fields that are legitimately zero.

- **Default directory:** **`.oak-benchmarks/<run-id>/`** as the run root, with one subfolder per environment and one subfolder per seed inside it: **`<run-root>/<envId>/seed<N>/<stem>.json`** for finals and **`<run-root>/<envId>/seed<N>/<stem>.heartbeat-NNNNNN.json`** for pulses (stem = **`envId-policy-seedN`**; when no seed is set the leaf folder is **`seedn`**). Override the run root with **`OAK_METRICS_STORAGE`** (directory path; env/seed subfolders are still created inside).
- **`benchmark`:** snapshots go to **`<bench-dir>/<envId>/seed<N>/`** unless **`OAK_METRICS_STORAGE`** is set (then that directory is the run root). **`*.log`** stems live at the run root and match the **`*.json`** snapshot stem. Without **`OAK_BENCH_DIR`**, the working directory is the latest-modified incomplete **auto-generated** **`.oak-benchmarks/<run-id>/`** (fewer primary **`<envId>/seed<N>/<stem>.json`** finals than env×seeds×policies; only directory names matching **`yyyyMMdd-HHmmss-<pid>`** are eligible for auto-resume so user-curated labels like **`composerMuse`** or **`gpt 5.4`** are never reused), or **`OAK_METRICS_STORAGE`** itself if that run root is incomplete, or a new run id. Existing **`<envId>/seed<N>/<stem>.json`** files are skipped; delete them to re-run a task.
- **`diagnose`:** **stderr** prints the snapshot file path after a successful write.
- **`run`:** **stderr** prints the run-root path when the run starts (same **`.oak-benchmarks/<run-id>/`** tree as the final snapshot) and the final **`<envId>/seed<N>/<stem>.json`** path when the run finishes.

**Stop / batch semantics:** The measured window counts **aggregate environment steps** (`measuredStepCount` ≈ batches × `num_envs`). **`measuredBatchCount`** is `measuredStepCount / num_envs`. **`OAK_BENCH_STEPS`** sets the measured-step target for **`benchmark`**.

**Env vs agent (serial MDP):** Each iteration runs **Metal `oak_step` then native `env.step`**. Without an async batched env API, throughput is bounded by this split (see **`diagnostic`** shares). Native env code is built by **`setup.sh`** into **`.build/oak-native/liboakenv.dylib`**.

**Action interface (Pass 11 generalization):** each env declares a structured action spec — a list of action components, each either `Discrete N` (categorical with N choices) or `Box A` (A-dimensional continuous with per-dim `[low, high]` bounds). The integer `D_policy = Σ_components(component_param_count)` (N for each Discrete, A for each Box) is the agent's only window into the env's action structure. The agent emits `policyMean[B · D_policy]` from `oak_meta_action` (which is the SOFT-CHORD-WEIGHTED sum `μ_p = Σ_k λ_k · (W_π[k][p] · s)` over the K options' policies, with λ = chordSlot from `oak_meta_policy`).

The env-wrapper layer (`EnvPoolEnv.sampleAndStep` in `Sources/Main.swift`) is the **sole** site that samples actions and computes score functions — Inv-1 boundary. Per Discrete-N component: Gumbel-max categorical sample of `softmax(policyMean[range])`; categorical scoreGrad `1{a=d} − π_d` (Maddison-Tarlow-Minka 2014; Williams 1992). Per Box-A component: structural-σ-1 Gaussian sample `μ + N(0,1)` clamped to env bounds; Gaussian scoreGrad `z` (since σ=1 by wrapper choice). The wrapper writes scoreGrad back to the arena for `oak_learn_pi` to consume on the next step. Action-bounds and discrete-count metadata never enter `OakMetalEngine`, `KernelConfig`, or any kernel — there are no env-conditioned function constants and no env-type-conditioned widths. Adding a new env suite (StarCraft 2 multi-discrete; mixed continuous+discrete) requires adding a new component type to `ActionComponent` in Swift; no agent or kernel changes.

**Pipeline (Pass 11; 18 kernels):** `encode → predict_outcome → meta_policy → predict → plan → reduce_alpha → learn_value → learn_encoder → learn_outcome → learn_psi → learn_pi → learn_beta → learn_omega → meta_action → bk_reduce → bk_feature → bk_option → bk_admit_commit`. The five new Pass-11 kernels are `predict_outcome` and `learn_outcome` (Tier 2 OPSR; Vereecken 2604.07016), `meta_policy`, `learn_beta`, `learn_omega` (Tier 3 temporal abstraction; Bacon-Precup 1609.05140 + Wan-Naik-Sutton 2110.13855). State width `E = R + K` (encoder + outcome augmentation); `M = E + K` (ψ concat dim with one-hot chord). Active-feature mask spans `E`; outcome dims `R..E` are always-active by construction.

### Long runs (depth + width at high SPS)

Default **`benchmark`** sweeps all native env ids with modest **`OAK_BENCH_STEPS`** and **`OAK_BENCH_SEED_COUNT`** — good for coverage, **not** for maximizing per-trajectory learning depth. For one env and long horizons, use **`OaK run`** with explicit limits.

| Quantity | Role |
|----------|------|
| **`numEnvs`** | Parallel width (see **`directNumEnvs`** in `Main.swift`; override with **`OAK_NUM_ENVS`** or **`OAK_BENCH_NUM_ENVS`** for **`benchmark`** only). |
| **`OAK_STOP_AFTER_STEPS`** | **Aggregate** measured step target (same counter as JSON **`measuredStepCount`**). |
| **`measuredBatchCount`** | **`measuredStepCount / numEnvs`** ≈ **measured steps per env** in the window. |

**Throughput calibration:** run **`diagnose`** on the target env with the **`OAK_NUM_ENVS`** you intend for the long job; read **`diagnostic.agentShare`**, **`envShare`**, **`hostShare`** and adjust width so neither agent nor native env dominates if the goal is maximum aggregate **`stepsPerSecond`**.

**Sizing:** aggregate steps available in wall time \(T\) ≈ **`stepsPerSecond × T`** (from a recent JSON). Per-env depth ≈ **`OAK_STOP_AFTER_STEPS / numEnvs`**. Example: ~2M steps per env with **`numEnvs=512`** → **`OAK_STOP_AFTER_STEPS ≈ 512 × 2_000_000`**. Set **`OAK_WARMUP_STEPS`** to a smaller aggregate prefix so most time is measured learning.

**Helper:** **`scripts/oak_longrun.sh`** requires **`OAK_STOP_AFTER_STEPS`** and runs **`OaK run`** with env id as **`$1`** or **`OAK_ENV`**. Release binary: **`swift build -c release`** first.

**Progress:** **`OAK_HEARTBEAT_STEPS`** writes **`<envId>/seed<N>/<stem>.heartbeat-NNNNNN.json`** alongside the final **`<envId>/seed<N>/<stem>.json`** (same schema; **`stopReason`** **`heartbeat`**). Default cadence for **`run`** and **`benchmark`** is **`stopAfter/10`** (≈10 pulses per run) rounded down to a **`numEnvs`**-aligned boundary; set **`OAK_HEARTBEAT_STEPS=0`** to disable, any positive integer to override. **`diagnose`** stays off by default.

**Baseline:** **`OAK_BASELINE_RANDOM_STEPS`** > 0 with **`OAK_POLICY=oak`** prepends a matching **`random`** snapshot (**`*-random-*.json`**) into the same **`<envId>/seed<N>/`** folder.

**Continual signal scale (BWH-orthodox stability):** zero clamps anywhere. Per-step boundedness comes from two compounding mechanisms, both of which TRACK the world's actual statistics rather than imposing a fixed prior:

- **Implicit-α step (Kim 2510.06149):** `α_imp = α₀ / (1 + α₀·⟨‖x_active‖²⟩_batch)` computed once per step in `oak_reduce_alpha`. As input magnitude grows, the step size shrinks proportionally; the per-step weight update `Δw = α_imp · δ · x` cannot blow up the parameter regardless of input scale.
- **Differential ρ̄ centering (Naik / 2603.27874):** `δ = (r − ρ̄) + V(s') − V(s)` with ρ̄ tracked via `track_rate(stepCount, ridge)`. Keeps the TD error in a bounded range over the long run regardless of reward magnitude.

Forward outputs (encoder, value, ψ, π logits, σ) are FREE to grow as the world demands — Big World requires that magnitudes carry information about the agent's encountered slice of reality. Layer normalization is rejected as an anti-BWH stationary scale prior. Telemetry uses non-finite guards (zero out NaN/Inf, count occurrences) instead of saturating clamps.

## Performance: Metal trace (GPU truth surface)

Claims about **where time goes inside the Metal kernel** need a **`.trace`** from **Metal System Trace–class** capture plus **release** (`swift build -c release`). CPU-only profilers are insufficient: they mostly show `waitUntilCompleted` / driver glue, not **encode vs learn vs planning** inside `oak_step`.

**Template:** Stock `xctrace --template 'Metal System Trace'` leaves **Shader Timeline off** in the saved session. Use the repo copy **`tools/instruments/ShaderTemplate.tracetemplate`** (Instruments: Metal Application → **Enable Shader Timeline**, **Counter set: None** to avoid “counter profile not supported”). Pass **`--template "$PWD/tools/instruments/ShaderTemplate.tracetemplate"`** (absolute path) so automation never depends on `~/Library/...`.

**Signposts:** Set **`OAK_METAL_SIGNPOSTS=1`** so `insertDebugSignpost` labels phases (`encode`, `learn`, `planning`, …) in `Main.swift`.

**Record** (always **repo root** so `.build/oak-native` resolves):

```bash
OAK_METAL_SIGNPOSTS=1 xctrace record --no-prompt \
  --template "$PWD/tools/instruments/ShaderTemplate.tracetemplate" \
  --time-limit 30s --output /tmp/oak.trace \
  --launch -- .build/release/OaK diagnose '<env-id>' --steps 50000
```

**Export:** `xctrace export --input … --toc`; then `--xpath` tables: **`metal-driver-intervals`**, **`metal-application-intervals`**, **`metal-shader-profiler-shader-list`**. Treat **`metal-shader-profiler-intervals`** as best-effort in CLI export (often empty even with Shader Timeline on). Prefer **short `--time-limit`** and **`--launch`** one process (not `--all-processes`) to keep traces small.

**Consistent summary:** After recording, run **`python3 scripts/oak_trace_report.py /path/to/capture.trace`** (add **`--json`** for machine-readable output). Cross-check **agent / env** splits against **`diagnose`** JSON (**`diagnostic.agentShare`**, **`envShare`**, **`metalPhaseGpuMsMean`**) with the same env configuration. `xctrace export` remains supplementary.

**Do not:** argue dominant kernel phase from **`sample` / Time Profiler alone**; treat tiny **encode** rows in application intervals as the whole step (total work includes GPU execution on driver/GPU tracks).

## OaK kernel checklist (Step 11 architecture)

13 kernels in `OakKernel.metal`, executed in order via Metal 4 Indirect Command Buffer. Each step: encode → predict → plan → reduce_alpha → learn_value → learn_encoder → learn_psi → learn_pi → meta_action → bk_reduce → bk_feature → bk_option → bk_admit_commit. `meta_action` runs LAST among learn-time phases so `learn_pi` reads `scoreGrad` from the previous step (= score function for the action that produced the current TD).

Notation: `R` = rank, `B` = batch, `K` = optionCount (≤ 32 = Apple SIMD width), `M` = `R + K` (concat dim for chord-conditional ψ), `A` = actDim. The agent always emits a Gaussian over `A` dims, so the action-parameter width is structurally `2·A` (μ_d, log_σ_d per dim) — there is no env-type-conditioned `D`.

| Kernel | Math | Invariant | Param shapes touched |
|---|---|---|---|
| `oak_encode` | `s'_i = (obs_i − obsMean_i) · W_enc[i]` per (row, feat) | New state slot is overwritten; only `W_enc[0..baseO)` are non-identity | reads `encoder[R]`, `obs[B·O]`, `obsMean[O]`; writes `state[1−slot][B·R]` |
| `oak_predict` | `δ_main = (r − ρ̄) + V(s') − V(s)`; for k ∈ [0, K): `δ_k = c_k + V_k(s', e_{k*}) − V_k(s, e_{k*})` with RED cumulant `c_k = (r − ρ̄) + (state[sf[k]] − featMean[sf[k]])` | `td[B]` and `tdPsi[B·K]` finite; ρ̄ centering ensures mean-zero δ over time | reads `valueW[R]`, `psi[K·M]`, `featMean[R]`, `sf[K]`, `state` slots, `reward[B]`, `done[B]`; writes `td[B]`, `tdPsi[B·K]`, `vCurr[B]` |
| `oak_plan` | `Q_k(s) = w_main[sf[k]] · ψ_k(s, e_k) + max(0, (w̄[sf[k]] − w_main[sf[k]]) · s[sf[k]])`; SIMD argmax over K lanes; `td[r] += Q_{k*} − vCurr[r]` | One TG per row × OAK_SIMD threads; `option_slot[1−slot]` updated to k*; planning advantage added to main TD | reads `psi`, `valueW`, `wBar[R]`, `sf[K]`, `vCurr[B]`; writes `option_slot`, `td` |
| `oak_reduce_alpha` | `α_imp = α₀/(1 + α₀ · ⟨‖x‖²⟩_batch)`; `α_enc = α₀/(1 + α₀ · ⟨‖obs−obsMean‖²⟩_batch)` | Single TG × 128 threads; `scalars->alphaImp/alphaEnc` are positive and finite | reads `state`, `obs`, `obsMean`; writes `scalars->alphaImp/alphaEnc` |
| `oak_learn_value` | `w_main[i] += α_imp · ⟨δ · prev[i]⟩_batch` (active features only); update `featSM[i]` (EMA of x²) and `wBar[i]` (decayed high-water-mark) | R TGs × OAK_SIMD threads; SIMD-sum reduction over B per feature | reads/writes `valueW[R]`, `featSM[R]`, `wBar[R]` |
| `oak_learn_encoder` | `W_enc[f] += α_enc · ⟨δ · (obs[f] − obsMean[f])⟩_batch` for f ∈ [0, baseO) | One thread per base-obs feat; learns per-feature scaling tied to main TD | reads/writes `encoder[baseO]` |
| `oak_learn_psi` | For each (k, m) ∈ K × M: `W_ψ[k][m] += α_imp · ⟨δ_k · concat[s, λ][m]⟩_batch`; chord-part contribution gated on row's active option | Grid K · M; every row contributes to every cumulant via its own δ_k | reads/writes `psi[K·M]` |
| `oak_learn_pi` | For each (k, d, i) ∈ K × 2·A × R, gated on opt(r) == k: `W_π[k][d][i] += α_imp · ⟨δ · prev[i] · scoreGrad[d]⟩_{rows: opt==k}` | REAL REINFORCE on Gaussian parameter space — replaces broken `td · prev · prevAction` heuristic | reads/writes `optPolicy[K · 2·A · R]` |
| `oak_meta_action` | Per (row, dim): μ_d, log_σ_d from W_π[k*][2d, 2d+1] · curr; σ = exp(log_σ); Box-Muller `z ~ N(0,1)`; `a_d = μ_d + σ_d·z`; score `(z/σ, (z²−1)/2)`. ALWAYS Gaussian — no env-type branch (Pass 10) | scoreGrad has nonzero variance across rows; agent treats action space as opaque real-valued vector | reads `optPolicy[K · 2·A · R]`; writes `action[B·A]`, `scoreGrad[B · 2·A]` |
| `oak_bk_reduce` | One 128-thread TG: Welford EMA of `obsMean[O]` and `featMean[R]`; mean-square / max telemetry over B·R latent and B·A action; `scoreGradL2Mean` over B · 2·A | Updates `featMean` (= ρ_k for cumulant centering); accumulates non-finite counters | reads everything; writes `obsMean`, `featMean`, `stats`, `counters[OBS_NF, LEARN_NF, PLAN_NF]` |
| `oak_bk_feature` | Per feature i: `utilFeat[i] += track_rate · (mean_{rows} \|δ · x[i]\| + Σ_k \|δ_k · x[i]\| − utilFeat[i])` (GraMa gradient-magnitude utility, divisor `1 + K`) | R TGs × OAK_SIMD threads; per-feature utility for top-K rerank in admit_commit | reads `td`, `tdPsi`, `state`; writes `utilFeat[R]` |
| `oak_bk_option` | Per option k (TG of 128): `utilOpt[k] += track_rate · (mean_{r: opt==k}\|δ_k\|² − utilOpt[k])`; FIRE+SWR reinit triggers on conjunctive (high stale TD ∧ dormant fraction) — clears `psi[k][:M]` and `optPolicy[k][:D·R]`, sets `OAK_CNT_REINIT_OPT` bitmask | utilOpt evolves over time; reinit bitmask atomic-OR'd race-free | reads `tdPsi`, `option_slot`; writes `utilOpt[K]`, `psi`, `optPolicy`, `counters[REINIT_OPT]` |
| `oak_bk_admit_commit` | One thread: refresh `activeMask/activeList/activeCount` (admit features above mean utility, floor at ⌈√R⌉); rerank `sf[K]` by top-K utilFeat (RED dynamic feature ranking); ρ̄ Naik update; pEff stable-rank proxy from featSM; FC-STOMP encoder reinit on worst feature when option reinited; toggle slots, increment stepCount | sf[] is permutation of distinct feature indices; activeCount ≥ ⌈√R⌉ unless R = 0; ρ̄ tracks batch-mean reward via track_rate | reads everything; writes `activeMask/List`, `sf[K]`, `scalars->{rho, pEff, slots, stepCount}`, `counters[REINIT_OPT]` cleared |

All kernels share a single 7-buffer signature (params, stateArena, uintArena, scalars, stats, bounds, config) for ICB pre-encoding even though some kernels don't read every buffer. Probe slots (Tier 0 harness, removed in Tier 12.3): `state, td, tdPsi, action, scoreGrad, valueW, encoderW, featMean, subtaskFeatureOf, psi_total_count, psi_bytes, concat_dim, actionParamCount, chord_slot, rho, activeRank, rewardRate, utilFeat, utilOpt`.
