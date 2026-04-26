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

- Subtasks defined by features (cumulant = reward, stopping value = feature attainment per Reward-Respecting Subtasks [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4). **Current rebuild OaK has the RR stopping-value branch live and has produced bridge-measured positive walking in some seeds; stable promotion is still open.**
- Options = policies + termination. **Current rebuild continuation is derived from the RR stopping comparison, not H1 model-confidence hysteresis.**
- Option models r̂(s,o), n̂(s,o), and for average-reward options duration l̂(s,o), used for planning. **Current L3-L7 keeps the orthodox raw option-reward-plus-duration form: `Q = rHat - rho*lHat + V(nHat)`. A coherent differential option-reward alternative was tested and rejected by the 10k paired gate, so do not revive it without materially new evidence.**
- Planning over option models (RR-Subtasks eq. 19 / average-reward option backup).
- **Option keyboard** = real-valued vector λ ∈ R^K, one component per subtask. Design A: chord blends K component options; model treats chord as opaque descriptor. Design B: chord defines a joint subtask. **Current OaK has a live scalar chord vector used to blend option policies, but it is not a fully operational Option Keyboard.** The unproven `psiCross`/GPE-GPI surrogate was deleted; current behavior is selected by RR continuation plus model-backed average-reward option scores, and emitted action means are a chord blend with selected-option fallback. A fully operational FC-STOMP keyboard/chord-descriptor cycle is an open architecture gap, not a hidden live feature.
- **Feedback / utility / remove-replace.** Continually assess utility of features, subtasks, options, and option models. Replace what isn't pulling weight (Plan Step 11 + RR-Subtasks §8 FCSTOMP). **Current rebuild has age-gated feature/option-bundle replacement, `optFeature`, feature age, option age, and feature-ever-used metadata; richer utility remains an active research frontier.**

### Current empirical status (2026-04-26)

OaK is no longer only a negative-return random-beating control. The bridge has now shown real HalfCheetah walking:

- `tools/bridge_oracle_cem.py` through the same C bridge, fixed open-loop sinusoid, 5 seeds x 5k steps: mean return `+4428.30`, mean `info:x_velocity=+0.983`.
- `.oak-benchmarks/fresh-postaudit-300k-20260426-030016`, 10 seeds x `{oak,zero,random}` at 300k: Oak mean return `+5365.61`, max seed `+109689.46`, mean `info:x_velocity=+0.0379`, positive velocity `6/10`, positive return `2/10`, corrected Oak NF total `4`.
- `.oak-benchmarks/deep-seed3-10m-20260426-030016`, Oak seed 3 at 10M: the gait loses its 1M snowball but never loses cumulative forward momentum (`info:x_velocity` mean `+0.011783`, x-position delta `+5891.636`, final return `+64937.78`). Terminal health shows the FC-STOMP cycle going quiet (`terminationRate=0`, `modelPriorityMax=0`, `optionDurationMean=1`, actor entropy/update norm `0`) after one learn NF and one TD-psi NF.
- `.oak-benchmarks/microscope-seeds0-5-1m-20260426` plus `.oak-benchmarks/microscope-seeds6-9-1m-20260426`, Oak-only 1M coverage: mean return `-2783.91`, mean `info:x_velocity=+0.020291`, positive return/velocity `3/10`, NF total `8`. Seeds 3, 6, and 9 are real forward snowballs by 1M; seed 8 is the large backward attractor; seeds 0/1 are quiet failures.
- `.oak-benchmarks/short-broad-surprise-oak-300k-20260426`, Oak-only 300k with a structurally gated active-backward keyboard refresh: Oak mean return `+9988.57`, max seed `+109689.46`, mean `info:x_velocity=+0.0509`, positive velocity `8/10`, positive return `3/10`, Oak NF total `3`. It preserves 9/10 seeds bit-identically versus grounded dead-credit and repairs seed 8 from `-37294.86`, `info:x_velocity=-0.0867` to `+6837.42`, `info:x_velocity=+0.0356`.
- The same gate still fails stability: seeds 0/1/2/4/5/6/7 are not strong positive-return walkers, seed 1 still has non-finite health counts, and Oak is not 10/10 better than `zero` by the promotion standard.

Operational consequence: every stabilization change must preserve the high-return walking mode while improving seed coverage. Changes that merely calm actions, reduce commitment, refresh chords more often, or delete model-backed selection have repeatedly killed the strong seed and are not progress.

Post-walking audit consequence: the current walking checkpoint appears to rely on an accidental delayed-plasticity/layout coupling in the Swift arena. Directly aligning the host arena with the kernel is not a keeper yet: it is more correct mechanically, but it kills the strong gait unless paired with a better FC-STOMP feedback/search-control mechanism. Q-per-duration aligned selection proved the system can still snowball a strong gait, but it can also snowball backward; do not promote it, selected-option execution/model credit, chord-basis model credit, signed utility, decoded duration TD, or reward-TD actor/score patches without fresh paired evidence.

Post-10M audit consequence: exact chord-responsibility bridge credit, every-step live keyboard refresh, all-continue keyboard refresh, reward-signed resource utility, and stale-search refresh were tested and rejected on 10k/50k gates. Keeping actor support alive or waking silent options is not enough if it erases the hard commitment that discovers the gait or wakes bad commitments too early. Long-run memory drift was host-side; `Sources/Main.swift` now wraps each measured benchmark step in `autoreleasepool`, and `.oak-benchmarks/autorelease-seed3-300k-20260426` reproduced the seed-3 300k reward while worker RSS stayed flat around `19.5 MB`.

Latest FC-STOMP audit consequence: forced RR termination on non-finite stopping comparisons woke a quiet seed into a control-cost explosion; uniform off-policy TD-energy normalization of option reward/transition/duration model updates calmed 10k but killed the seed-3 rocket by 50k. The live code keeps only the truthful search-control correction: non-finite option-model TD arithmetic is counted and no longer silently converts model priority to zero.

ArXiv FC-STOMP search consequence: Reward-Respecting Subtasks says the missing feedback loop is specifically later-stage planning/search-control effectiveness feeding back into which features, subtasks, options, and models survive. Average-reward options say interruption is justified by finite option values (`q_mu(s,o) < v_mu(s)`), not by invalid arithmetic. Option Keyboard says compositions happen in cumulant/value space through GPE/GPI or an equivalent chord descriptor, not by merely blending already-emitted policies. Meta-gradient search-control work says useful planning must learn to avoid inaccurate or credit-stalling model queries. Therefore the next serious stabilization mechanism should be a coherent keyboard/search-control/utility repair, not another isolated chord refresh, damping rule, or non-finite termination patch.

Latest closed-loop probe consequence: fixed-budget pagewise cross pressure, causal reward-closed search-use traces, duration-consistent priority protection, and "preserve last finite RR/actor signal" were all rejected. Page/search-use variants killed waking or the seed-3 rocket; duration-consistency and preserved actor pressure changed diagnostics without changing behavior. The current lock is downstream of actor pressure: continuation/keyboard emission can stay frozen even when hidden actor updates become enormous. Do not add hidden utility or pressure unless it changes emitted behavior while preserving the strong walking seed.

Targeted-screen consequence: finite value interruption is a real lever but not a keeper. `value-interrupt-target50k-20260426` woke seed 1 and improved seed 8, yet killed seed 3's rocket; the positive-TD guarded variant still killed seed 3. Interruption cannot be bolted onto RR continuation alone. The next attempt must distinguish productive hard commitment from stale continuation inside the keyboard/emission loop.
Model-priority-gated interruption also failed the targeted gate (`prio-gated-interrupt-target50k-20260426`), killing seed 3 and making seed 1 much worse. Surprise alone is not the productive/stale discriminator.
Stale-emission fallback was rejected too (`stale-fallback-target50k-20260426`): it improved seed 8 and seed 0, but cut seed 3 from `+12351.97` to `+1368.63` at 50k. Fallback/emission repair cannot merely replace low-movement chord blends with selected-option execution; it must preserve the snowballing blended gait.
All-continue dead-option generate-and-test was rejected (`dead-option-gt-target50k-20260426`): it preserved seed 3 exactly, but worsened seed 0 and left seed 1/8 unchanged. Replacing mature zero-actor bundles is not enough to wake quiet seeds; the missing loop is upstream in keyboard/search-control credit, not only bundle churn.
Low-rank chord-progress keyboard projection was rejected in raw, agreement-only, model-error-gated, and `1/K` basis-normalized forms. It can wake seed 8 and seed 1, proving the missing keyboard organ is real, but it steals the seed-3 rocket. Do not reintroduce cross-cumulant behavior weights without a stronger arbitration mechanism that preserves the identity/reward path when it is about to snowball.
Signed behavior-responsibility main-TD actor pressure was rejected in full and degenerate-only forms. Full responsibility credit woke some quiet behavior but killed seed 3; degenerate-only preserved seed 3 but did not wake seed 1/8 and worsened seed 0. The primary-control gap remains real, but direct reward TD inside the option actor is not the keeper.
Search-failure-gated main-TD actor pressure was rejected too. It rescued seed 1 and partially seed 8, but still killed seed 3 even with `rho < -1/R` and model-priority/duration gates. Do not add reward-TD option actor pressure without a real arbitration state that can protect the identity/reward snowball.
Mature-selected-option search-failure actor pressure is also rejected. At 300k it produced a strong seed-1 gait and mostly repaired seed 8, but destroyed the seed-3 high mode and worsened the target mean. This is evidence for a future arbiter, not a keeper.
The first slow-arbiter rescue actor was rejected (`slowarb-tdactor-target50k-20260426`): the tiny slow reward trace still stole too much from seed 3 and missed seed 1. Arbitration likely needs a richer state than a scalar reward trace.
Origin-lock rescue actor was rejected (`originlock-tdactor-target50k-20260426`): the intervention itself pushed seed 3 into the origin-locked quiet path. Origin-lock is a diagnostic symptom, not a safe trigger for actor rescue.
Grounded dead-credit keyboard refresh is the current keeper substrate improvement. It refreshes the keyboard under all-continue/zero-actor-credit only when option durations are still grounded by lived stream length. It improves seed 0 at 300k, reduces NF, and preserves the seed-3 rocket bit-identically; it does not solve seed 8 or 10/10 stability.
Search-use closure arbitration was tested and rejected in signed-utility and behavior-only forms (`closure-arbiter-target50k-20260426`, `closure-arbiter-noutil-target50k-20260426`). Both improved seed 8 but stole the seed-3 snowball by 50k, proving that scalar closure/use traces are still too blunt. Future arbiters must preserve the incumbent keyboard identity through the early rocket window while using richer active-backward and quiet-lock discriminators.
Active-short keyboard refresh was rejected too (`active-short-keyrefresh-target50k-20260426`, `active-short-only-target50k-20260426`). Low keyboard mass plus selected duration below bank mean is not enough to identify backward commitment: it improves seed 8 and seed 0, but prevents seed 3 from changing identity into the high-return snowball.
Negative-rate active-short refresh (`negrate-active-short-target50k-20260426`) and quiet-ungrounded refresh (`quiet-ungrounded-target50k-20260426`) are rejected. The former still kills seed 3; the latter preserves seed 3 bit-identically but turns seed 1 into a control-cost catastrophe. Quiet-lock repair must reduce bad model duration/control pressure, not merely refresh the keyboard under ungrounded durations.
Duration-grounded option scoring plus quiet refresh (`duration-grounded-score-target50k-20260426`) is rejected for the same reason: seed 3 is bit-identical, but seed 1's control cost explodes. The quiet-lock fix must intervene before or inside the actor/model pressure that creates the cost sink, not after the keyboard has gone quiet.
Duration-grounded score-only model use (`duration-grounded-scoreonly-target50k-20260426`) passed the 50k target but failed 300k (`duration-grounded-scoreonly-oak-300k-20260426`): seed 3 dropped from `+109689.46` to `+65102.79` and gained three non-finites. Duration grounding can be a diagnostic, but it is not a keeper unless it preserves the long rocket, not merely the 50k prefix.
Short/broad/surprising active-backward refresh is the current keeper. It refreshes the keyboard only when the live reward rate is structurally negative, model priority exceeds `K^2`, RR termination is broad (`termSum^2 > K`), the existing keyboard has low mass, and the selected duration is shorter than `K`. This is the first active-backward repair that preserves seed 3 and seed 0/1 while flipping seed 8 forward at 300k. It is still not 10/10 stability.

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
6. **Inv-7: temporal uniformity** (Plan research-vision pillar #2; verbatim: *"all times are the same with respect to the algorithms running on the agent. There are no special training periods when training information is available or when rewards are counted more or less than others. If training information is provided... it is provided on every time step. If the agent learns or plans, then it learns or plans on every time step. If the agent constructs its own representations or subtasks, then the meta-algorithms for constructing them operate on every time step."*). Every Plan mechanism in OaK runs every step: perception (encode), prediction/control, planning (full-bank `oak_plan`), learning (learn_heads), meta-learning (post_learn MetaOptimize), and every-step G&T assessment with capacity/age-gated commits. Reward is centered (ρ̄) the same way every step. `track_rate(t, obsDim) = max(1/(t+1), 1/obsDim)` is the current structural tracking choice, not slide law; replace it if paired evidence or primary research shows a more faithful parameter-free tracker. Subsumes Inv-4 + Inv-5 in their consolidated Plan-citing form. **Forbidden constructs**: warmup-gated branches (`if stepCount > N`), epoch counters, train/eval splits, frozen weights post-N steps, special "exploration" vs "exploitation" phases. `OAK_WARMUP_STEPS` is host-side bench metric collection only — the agent's per-step computation is identical regardless.
7. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
8. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity, stream shape, or device execution width (cap = R, K = R, α₀ = 1/cap, ridge = 1/(age+1)). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
9. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
10. **Current single-agent stream semantics.** The live engine is one observation → one action → one reward per agent step (`B = 1`) because the Alberta Plan describes an agent's experience as a single time series. This is an implementation stance, not a sacred batch-size law: future batched kernels are allowed only if they are semantically N independent agents with disjoint state, no cross-stream reductions, and identical per-agent updates. Benchmark concurrency remains N independent worker processes for one environment/seed/policy grid.
11. **Inv-10: zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/`KernelScalars`, `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with smoke + 1.5M paired bench. **FP-reorder** (`simd_sum`, threadgroup reductions where the input order changes from sequential, `fma` decomposed into separate multiply + add): ship only after a 10-seed paired bench shows **per-seed parity within ±3 % of the prior commit's per-seed lift** — aggregate parity is necessary but insufficient; per-seed reshuffling that averages to zero is still considered an intelligence hit. The first attempt at C2-perf used `simd_sum` for `oak_meta`/`oak_plan` and showed aggregate parity but per-seed chatter std doubled (0.020 → 0.041); the follow-up restored bit-identical with lane-0 sequential reductions for ~4 µs cost.

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

### Action interface (Inv-1)

Each env declares one structured action spec: `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = component_param_count` is the agent's only window into the env's action structure.

The agent emits `policyMean[D_policy]` from `oak_action_emit`. The env-wrapper layer (`HCEnv::step_oak` in the native bridge) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax); target-option actor scores use `1{a=d} − π_d` (Maddison 2014; Williams 1992).
- Box A: currently exposes `2A` policy coordinates to keep `D_policy`, default rank, and K stable. The first A coordinates are means; the second A coordinates are wrapper-reserved/inert. The wrapper samples with structural fixed `σ = 1/D_policy`, emits residual-style mean credit (`raw − μ`) for Box mean slots, writes zero score/action-signal entries for reserved scale slots, and clamps only the env-facing action. Full `D_policy = A` mean-only deletion failed the 1M gate and remains rejected unless a different actor geometry earns fresh evidence.

Adding a new env or action-space shape requires adding to `ActionSpec` in Swift; no agent or kernel changes.
`doneCount` is a wrapper random-restart metric, not an episode signal to OaK; the engine state is never reset on `done`.

## Run metrics JSON

Each benchmark worker writes one pretty-printed JSON file with stable alphabetical key order (`schemaVersion: 6`).

**Top-level keys:** `command`, `envId`, `policy`, `seed`, `numEnvs` (always 1 — single-stream), `stopAfterMeasuredSteps`, `observationDim`, `actionDim`, `actionDiscreteCount` (when discrete), `measuredStepCount`, `measuredSeconds`, `stepsPerSecond`, `rewardSum`, `doneCount`, `stopReason`, `heartbeatSequence`, and `health`.

**`health` block.** Symmetric across policies. Host-computable fields are populated for ANY policy (`oak`, `zero`, or `random`):

- `rewardRate` — host-tracked ρ̄ via incremental EMA (Reward Centering).
- `actionDeltaL2Mean` — mean L2 of consecutive `policyMean` deltas.
- `continuingRestartCount` — mirror of wrapper `doneCount`, named to make clear that terminal events are random restarts outside the agent.
- `envInfo` — generic wrapper audit telemetry keyed by raw env-provided `info:*` signal names, each with `count`, `finiteCount`, `nonFiniteCount`, `sum`, `mean`, `last`, and `delta`. These diagnostics are never agent inputs and must not be promoted into env-semantic benchmark contract fields such as "forward reward".

Kernel-side fields are populated only for `oak`:

- `oakOptionReinitCountFinal`, `oakFeatureReinitCountFinal` — cumulative age-gated generate-and-test replacement counts.
- `nonFiniteByPhase` = `{"encode": N, "learn": N, "plan": N, "tdPsi": N}` — per-phase NF counts. **Healthy = all zeros across the run.**
- Search/model diagnostics: `modelPriorityMax`, `optionDurationMean`, `terminationRate`, `selectedOptionAge`, `selectedOptionDuration`, `selectedOptionPriority`, `keyboardL1`.
- Actor/support diagnostics: `importanceEss`, `actorUpdateNormMean`, `actorSupportEntropy`, `actorAdvantageL1`.

**Storage.** Default directory `.oak-benchmarks/<run-id>/`, with `<run-root>/<envId>/seed<N>/<stem>.json` for finals and `<stem>.heartbeat-NNNNNN.json` for pulses. Override the run root with `OAK_METRICS_STORAGE`.
`OAK_BENCH_HEARTBEAT_STEPS=N` lowers or raises the host-only pulse stride for diagnostics; this does not change agent computation.
Use `python3 tools/bench_watch.py <run-root>` to continuously summarize heartbeat/final JSON into per-seed reward, velocity, control, non-finite, model, actor, and replacement columns.
`OAK_BENCH_SEEDS=0,1,3,8` overrides the contiguous seed range for targeted paired screens. Use `python3 tools/bench_compare.py <keeper-root> <candidate-root> --heartbeats` to locate first divergence before spending a full 10-seed gate.
Use `python3 tools/bench_gate.py <keeper-root> <candidate-root>` as the default targeted gate. Add `--at-step 10000` or `--at-step 20000` when comparing a short candidate run against heartbeat records in a longer keeper root. It encodes the current empirical law: preserve the rocket seed, do not regress the backward attractor seed, and do not hide new non-finites behind a better mean.

## Streamlined research workflow

The working goal is stable 10/10 HalfCheetah walking without bloating the policy or violating FC-STOMP. The loop is architecture-first, evidence-fast:

1. State the FC-STOMP organ being changed before editing: feedback/search control, option keyboard/chord descriptor, option model, utility/remove-replace, or actor emission. A candidate that cannot name the organ is patch-and-pray.
2. Keep each behavior patch paired with a keeper-root comparison. Default target screen: `OAK_BENCH_POLICIES=oak`, `OAK_BENCH_SEEDS=0,1,3,8`, 10k with frequent heartbeats, then 50k if not obviously dead. Seed 3 is the rocket-preservation sentinel; seed 8 is the active-backward sentinel; seeds 0/1 are quiet-lock sentinels.
3. Promote only after the target screen passes `tools/bench_gate.py`, then run Oak-only 300k x 10 seeds. Run `{oak,zero,random}` only for keeper/promotion evidence, bridge audits, and claims about beating baselines.
4. Reject early if seed 3 loses the high-return path by 50k, seed 8 becomes more backward, or non-finite counts increase, unless the candidate produces broad 10/10 evidence in the same run. Do not average away per-seed collapse.
5. Name run roots by hypothesis and scale, for example `.oak-benchmarks/closure-arbiter-target50k-YYYYMMDD`, and inspect live runs with `tools/bench_watch.py` rather than waiting for final JSON.
6. Use subagents when the work can truly run in parallel: a Metrics Archaeologist mines benchmark roots, a Plan/Invariant Reviewer checks Alberta/FC-STOMP fit, a Mechanism Designer proposes closed-loop organs, a Benchmark Steward watches gates, and a Metal Auditor handles speed/FP-order risk. The main thread integrates; subagents do not define project doctrine.
7. Archive rejected mechanisms in `docs/roadmap.md`/`AGENTS.md` when they teach a reusable lesson. Keep the live code lean; diagnostics earn their place only when they shorten the next decision.

## CLI

| Command | Use |
|---|---|
| `OaK benchmark` | Concurrent single-env multi-seed sweep. Each item runs in its own worker process so native env/global state cannot leak across independent streams. `OAK_BENCH_ENV` selects the env (default `HalfCheetah-v4`); `OAK_BENCH_STEPS`, `OAK_BENCH_SEED_COUNT`/`OAK_BENCH_SEEDS`, and `OAK_BENCH_POLICIES` define the workload. `OAK_BENCH_CONCURRENCY` caps worker count; otherwise concurrency is all logical cores, with deterministic refill from the canonical work list. One benchmark launcher owns the workspace at a time; nonempty requested roots get a fresh suffixed output dir. |

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
