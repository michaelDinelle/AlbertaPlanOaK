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

- **Default directory:** **`.oak-benchmarks/<run-id>/metrics/`** (override with **`OAK_METRICS_STORAGE`** to a directory path).
- **`benchmark`:** snapshots go to **`<bench-dir>/metrics/`** unless **`OAK_METRICS_STORAGE`** is set (then all benchmark snapshots use that directory). **`*.log`** stems match the **`*.json`** snapshot stem (**`env-policy-seed`**). Without **`OAK_BENCH_DIR`**, the working directory is the latest-modified incomplete **`.oak-benchmarks/<run-id>/`** (fewer primary **`*.json`** than env×seeds×policies), or **`OAK_METRICS_STORAGE`**’s parent if that metrics folder is incomplete, or a new run id. Existing **`<stem>.json`** files are skipped; delete them to re-run a task.
- **`diagnose`:** **stderr** prints the snapshot file path after a successful write.
- **`run`:** **stderr** prints the **`metrics/`** directory path when the run starts (same **`.oak-benchmarks/<run-id>/metrics/`** tree as the final snapshot) and the final **`*.json`** path when the run finishes.

**Stop / batch semantics:** The measured window counts **aggregate environment steps** (`measuredStepCount` ≈ batches × `num_envs`). **`measuredBatchCount`** is `measuredStepCount / num_envs`. **`OAK_BENCH_STEPS`** sets the measured-step target for **`benchmark`**.

**Env vs agent (serial MDP):** Each iteration runs **Metal `oak_step` then native `env.step`**. Without an async batched env API, throughput is bounded by this split (see **`diagnostic`** shares). Native env code is built by **`setup.sh`** into **`.build/oak-native/liboakenv.dylib`**.

**Discrete control:** the vector sent to **`env.step`** uses **indices sampled from softmax(logits)** produced in the meta-policy phase (not deterministic argmax), and the host **`floats`** are set to the matching **one-hot** so **`previousAction`** matches what was executed. Logits are **clamped** to **`±min(√(rank·actDim), log(max finite float32))`** (`OakMetalLayout.discreteLogitHalfWidth`) so Metal’s `clamp` is not a no-op on ±`greatestFiniteMagnitude` and host **`exp`** stays in range.

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

**Progress:** **`OAK_HEARTBEAT_STEPS`** writes **`stem.heartbeat-NNNNNN.json`** in the same **`metrics/`** tree as **`stem.json`** (same schema; **`stopReason`** **`heartbeat`**; default **off**).

**Baseline:** **`OAK_BASELINE_RANDOM_STEPS`** > 0 with **`OAK_POLICY=oak`** prepends a matching **`random`** snapshot (**`*-random-*.json`**) in the same **`metrics/`** dir.

**Continual signal scale:** encoder output, GVF **`latentError`**, reward prediction, and **TD** are **clamped** in the Metal kernel to **`±(rank × batchSize)`** — the same capacity volume term used in **`capacity_ridge`** (`structural_activation_cap` in `OakKernel.metal`) — so forward quantities cannot diverge toward float overflow on an infinite horizon.

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
