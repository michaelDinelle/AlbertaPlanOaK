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

- Subtasks defined by features (cumulant = reward, stopping value = feature attainment per Reward-Respecting Subtasks [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4). **Mark IV: the eq.-4 bonus weight is LIVE — `bonusW[i]` is the high-water mark of `valueW[i]` (RR §6 optimism) plus a structural unit bonus, and stopping follows eq. 9's ≥ tie direction. Option 0 is RR p.12's subtask 0 (cumulant = reward, β = 0): the main-task key.**
- Options = policies + termination. **Current rebuild continuation is derived from the RR stopping comparison, not H1 model-confidence hysteresis.**
- Option models r̂(s,o), n̂(s,o), and for average-reward options duration l̂(s,o), used for planning. **Mark IV (2026-07-04) moved to the centered differential form `Q = r̂_diff + V(stop)` with per-head differential centering (`optRho`, the psiRho pattern — Mark III law (d)); the materially new evidence for reviving it was the five-agent audit (raw-form staleness blow-ups, modelErr drip). Duration l̂ is a diagnostic/staleness head only; option 0 is the never-stopping main-task key.**
- Planning over option models (RR-Subtasks eq. 19 / average-reward option backup).
- **Option keyboard** = real-valued vector λ ∈ R^K, one component per subtask. Design A: chord blends K component options; model treats chord as opaque descriptor. Design B: chord defines a joint subtask. **Current OaK has a live scalar chord vector used to blend option policies, but it is not a fully operational Option Keyboard.** The unproven `psiCross`/GPE-GPI surrogate was deleted; current behavior is selected by RR continuation plus model-backed average-reward option scores, and emitted action means are a chord blend with selected-option fallback. A fully operational FC-STOMP keyboard/chord-descriptor cycle is an open architecture gap, not a hidden live feature.
- **Feedback / utility / remove-replace.** Continually assess utility of features, subtasks, options, and option models. Replace what isn't pulling weight (Plan Step 11 + RR-Subtasks §8 FCSTOMP). **Current rebuild has age-gated feature/option-bundle replacement, `optFeature`, feature age, option age, and feature-ever-used metadata; richer utility remains an active research frontier.**
- **Named FC-STOMP loop / ablations.** `Sources/OakKernel.metal` exposes the live loop vocabulary directly in code: `FeatureConstruction`, `GVFPrediction`, `RewardRespectingSubtasks`, `OptionPolicies`, `OptionModels`, `Planning`, `OptionKeyboard`, `SearchControl`, `GenerateAndTest`, and `ActionEmission`. `OAK_ABLATE` compiles specialized Metal pipeline variants for diagnostic ablations of those sections and couplings; canonical promotion runs use `OAK_ABLATE=none`.
- **Ablation lesson.** The first 50k focus matrix (`.oak-benchmarks/step11-ablation-focus50k-*`) confirms the live walking mode depends on the whole FC-STOMP loop. The largest losses come from `keyboard-to-policy`, `option-keyboard`, and `model-to-keyboard`; `search-control` can help a quiet seed while killing the rocket. Treat this as evidence for a richer cumulant/value-space keyboard and bundle-utility loop, not as permission to delete organs.

### Current status

As of Mark IV (2026-07-04): divergence is essentially extinct on current substrates (0/1024 @100k) — the binding failure mode is chronic slightly-negative delivery with mode extinction ("never dies, never walks"), not numerical death. The last legally promoted keeper is `.oak-benchmarks/mk8-tierL-20260705` (Tier-L IQM +0.4433, 98.1% positive, div 1.37%, sustain 0.664), which ships `OAK_ABLATE=planning`. (Superseded history: `.oak-benchmarks/gated-10-tierL`, Tier-L IQM −0.0103; the mk2-D1/A4 substrate that followed it was gate-REJECTED at both tiers and is not a keeper baseline.) The June strong-mode ceiling was 0.58–0.71/step (probe-rho1, valuescale, gated-9 — all rejected for divergence under the old criteria); any 0.88 figure traces to an incomplete run (honest number: 0.451). Current numbers come from fleet runs, not from this file — run `tools/fleet.py report` and see `docs/roadmap.md`. The April CPU-bridge history is archived under Exhausted Families.

**Current keeper root: `.oak-benchmarks/tape-reference-tierL-20260805`** — the pmaze-perfect kernel re-baselined onto the METRICS-TAPE instrument (0/1024 final-row mismatches vs the pre-tape root; bit-identical streams; pre-tape roots are archives every reader refuses).

**Keeper lineage (2026-08-05): `.oak-benchmarks/pmaze-perfect-tierL-20260805`** — the ProceduralMaze perfect-maze ENV fix (agent untouched; randomized-DFS spanning tree over the 5×5 room lattice, integer-only Philox, all 49 free cells mutually reachable by construction — the old generator left disconnected pockets in 999/1024 seeds and respawns stranded agent or goal). Gate PROMOTE vs `wall-reference-tierL-20260804` (balanced 16-env schedule, B=1024 × 1M): 15/16 env slices bit-identical, ProceduralMaze slice IQM 2.8e-06 → 4.1e-04 (+14,784%, slice gate IMPROVEMENT), live positivity 85.9% vs 3.1%. Env law: procedural-env solvability must be structural, never probabilistic; suspect a flat env slice as an env bug before an agent limitation.

**Superseded keeper (2026-07-21): `.oak-benchmarks/p93-tierL-20260721`** (K-parking; ablation `model-to-keyboard`). Tier-L gate vs `p7s12-tierL-20260721` PROMOTE with IMPROVEMENT: **IQM +0.38725 vs +0.32484 (+19.2%)**, divergence **eliminated** (0 deaths vs 18), sustainers 851 vs 729 (+17%), HalfCheetah median +0.497 (+13%), live +0.492 (+20%), P10 +0.255 (+50%), positivity 99.5%. K-parking hibernates degenerate options (matured `optAge≥R`, below live-matured-mean utility, not selected/chorded/probe; ≤1 park/step, reversible) — the K-bank (OptionPolicies+OptionModels+RR ≈ 89% of agent FLOPs) is skipped when inert, rows kept frozen not zeroed. Lineage: `p4m-tierL-20260719` (Mark IV planner un-poisoned) → `p7s12-tierL-20260721` (Phase 7 representation) → `p93-tierL-20260721` (Phase 9.1 priority index + Phase 9.3 K-parking). Also shipped this cycle: **Phase 7 representation geometry** (geometry derived from stream shape `naturalO/naturalA` via `OAK_ENV_SPECS`, Inv-1 shape-read not env-identity; `legacy` predicate `(naturalO==17 && 2*naturalA==12)` → cheetah bitwise A/B; joint/zoom tile banks + derived-trace core for sparse envs; the §3 generalized-G&T enumeration was CONVICTED (KeyCorridor feature explosion) and permanently deleted, not ablated). **Phase 8 per-organ reward/GFLOP instrument** (`organFlops`+`organCostVersion` in meta.json, per-env stratification in `fleet.py report`, `organ-report.json`; bit-identical). First instrument read: Planning = 37% of agent FLOPs with planTd=0, OptionPolicies 24%, OptionModels 20%, RR 10%.

Durable lessons from that work, independent of instrument or seed:

- **Preserve the high-return walking mode while improving coverage.** Changes that calm actions, reduce commitment, refresh chords more often, or delete model-backed selection have repeatedly killed the strong mode. That is a regression, not a simplification.
- **Interruption, termination, and duration-bounding are a dead end for plasticity** — every form tested killed the strong seed. Pursue representational plasticity instead (feature/representation rank maintenance, step-size adaptation, actor-entropy maintenance) in FeatureConstruction / GVFPrediction / GenerateAndTest.
- **The missing organ is a coherent keyboard / search-control / utility loop** — cumulant/value-space composition (GPE-GPI or a chord-descriptor model) plus search-control effectiveness feeding back into which features, subtasks, options, and models survive. Isolated chord refreshes, damping rules, fallbacks, and raw reward-TD actor pressure are patch-and-pray.
- **Mark III audit law (2026-07-04, five-agent audit — supersedes any older record that says otherwise).** (a) High-rate generate-and-test replacement is the ignition engine, not a bug: throttling it (the D1 promotion bar) collapsed igniters 64→~5 (correction 2026-07-04: sustain-of-igniters was 1–2/5, not 0%). (b) Magnitude-competition WTA is forbidden as a sparsifier: it blinded the actor (≤3/17 obs channels visible/step), spike-standardized constructed features, and amplified replacement damage ~13×; sparsity must be GEOMETRY-gated (stationary function of state) with all raw inputs always visible. (c) Tier-M IQM alone is blind to walking-mode kills: igniters, sustainers, best-lane-class, and %positive are BLOCKING criteria in `fleet.py gate` (paired non-inferiority, shipped Mark IV). (d) Uncentered stopping-aware cumulants are structurally divergent for slow-stopping options — every option-model head carries per-head differential centering (the psiRho pattern).
- **Judge the population, not a single seed.** Seeds 3 and 8 are inspectable lanes, not gates; promotion is the paired population gate (`tools/fleet.py gate`).
- **Mark IV–VIII laws (2026-07-04/05, all gate- or forensics-measured; see `.claude/skills/organ-bisect` and `keeper-cycle` for the full list).** (a) Model estimates may gate/select; they must NEVER be written into `valueW` or ρ̄ (the planning value-backup was the population anti-learner: optimism-inflated ρ̄ made the agent judge real experience against model fantasy — ablating it took positivity 24.6%→97.8%). (b) THE ZENITH LAW: emission geometry must encode actuator bounds — tanh action means with the exact score gradient `(a−m)(1−m²)`; unbounded means past the env clamp caused every death, every ceiling turnover at ~400k, and the population plateau. (c) Every bootstrapped/cumulant head carries per-head differential centering (psiRho/optRho/vsRho — the one uncentered family was the one that died). (d) Never freeze an organ; fix its target/scale geometry (freezing measured −IQM and moved explosions). (e) Aux model heads regularize the shared representation even with consumers dead (deletion gate-REJECTED at −31% IQM). (f) Workflow: one diagnosed change → `tools/check.sh` at 100k → ONE Tier-L via `tools/run_fleet.sh` gated vs the keeper → commit on PROMOTE (the keeper-cycle skill).
- **Mark IX law (2026-07-21, seven mechanisms measured, all reverted): the Mark IV (a) trap is structural — there is NO stable-and-unbiased way to feed the planner's long-horizon credit into `valueW` in OaK's γ=1, non-stationary, improving-stream setting.** The `v(x_frontier)` ABSOLUTE bootstrap is the amplifier (live-bootstrap real-return relay: exponential explosion, paramAbsMax 1e8 by 9k; lap-frozen tail only delays it to 60k, 327 deaths — the sync re-closes the loop). Removing the bootstrap (pure-MC centered relay, `realU[s]=Σ(r_j−ρ̄)` over the ring) IS stable — but the truncated window return is **biased low** (no tail) and drags `valueW` down ~40–60% on every env regardless of centering (ring-mean or ρ̄_global). Bootstrap → explodes (unbiased, unstable); no-bootstrap → stable but biased (truncated). The long credit is real but must stay **sealed in the planner (`planU`)**; the working levers for sparse/long-horizon competence are geometry (Phase 7 representation) and allocation (K-parking), never value-feeding. **Selection channels are a dead end under the frozen chord** (`model-to-keyboard` ⇒ `tgG=NAN` ⇒ `gpiBest=0` ⇒ RR continuation dead): greedy-Q selection is toxic (Phase 6, IQM −48%), novelty commitment-break is disruptive (Phase 11, IQM −24% but MountainCar positivity 18.8%→45.3% — the one partial sparse signal, preserved as a future lane-conditional refinement), and the safe negative-selection veto AND the RR-latch are both INERT (nothing to act on — measured A/B-identical to keeper). Do not retry value-feeding relays or selection-dynamics revivals without a genuinely new mechanism; the organ-bisect/keeper-cycle evidence says they are closed.

### What the Plan eschews

- Nothing about the human world (vision, space, objects, language, geometry, other agents).
- Learning first, domain knowledge last.
- No discounting (Step 5+).
- Actions and observations are signals with no intrinsic meaning.

## Hard rules (project invariants)

These are the non-negotiables. Every kernel and every line of host code must respect them. When an invariant drifts from the cited research, update the invariant instead of preserving folklore.

1. **Greenfield.** No legacy code, no compatibility fallbacks. Replace or refactor.
2. **Inv-1: env-type knowledge boundary.** The agent (`OakMetalEngine`, `OakKernel.metal`, `KernelConfig`) contains ZERO action-space-type or env-identity branches. Action space (discrete vs continuous), bounds, env id, discrete count live exclusively in the env-wrapper layer — now the megakernel's sample/env stage (`oak_stage_sample_act_wide` / `oak_stage_fleet_env`); the CPU wrapper types (`EnvPoolEnv` and below) were deleted per the F4 verdict. Agent reads only `obsDim` and `D_policy`.
3. **Inv-4: no episodic semantics inside the agent.** Resets are wrapper events; the agent runs continually. The `done` signal is intentionally not plumbed through to the engine.
4. **Inv-5: no special training phases.** Same updates every step.
5. **Inv-6: no per-step heap allocation on the hot path.** Pre-allocated arenas only.
6. **Inv-7: temporal uniformity** (Plan research-vision pillar #2; verbatim: *"all times are the same with respect to the algorithms running on the agent. There are no special training periods when training information is available or when rewards are counted more or less than others. If training information is provided... it is provided on every time step. If the agent learns or plans, then it learns or plans on every time step. If the agent constructs its own representations or subtasks, then the meta-algorithms for constructing them operate on every time step."*). Every Plan mechanism in OaK runs every step: perception (encode), prediction/control, planning (full-bank `oak_plan`), learning (learn_heads), meta-learning (post_learn MetaOptimize), and every-step G&T assessment with capacity/age-gated commits. Reward is centered (ρ̄) the same way every step. `track_rate(t, obsDim) = max(1/(t+1), 1/obsDim)` is the current structural tracking choice, not a Plan mandate; replace it if paired evidence or primary research shows a more faithful parameter-free tracker. Subsumes Inv-4 + Inv-5 in their consolidated Plan-citing form. **Forbidden constructs**: warmup-gated branches (`if stepCount > N`), epoch counters, train/eval splits, frozen weights post-N steps, special "exploration" vs "exploitation" phases. `OAK_WARMUP_STEPS` is host-side bench metric collection only — the agent's per-step computation is identical regardless.
7. **Inv-8: strict objective isolation.** No intrinsic curiosity, no learning-progress bonus added to the main reward stream.
8. **Inv-9: parameter-free meta-learning.** Every constant is structurally derived from agent capacity, stream shape, or device execution width (cap = R, α₀ = 1/cap, ridge = 1/(age+1); the option count K = 32 is a hard-pinned structural constant, NOT derived to equal R — the design's "K=R" was stale). No tunable knobs. **If you have a number to tweak, the underlying mechanism probably isn't Plan-orthodox** — strip the mechanism, not the number.
9. **No magnitude caps on weights.** `safe_w(old, new) = isfinite(new) ? new : old` preserves on non-finite; weights grow as the agent's slice of reality demands. No `bounded_soft`, no `safe_half`, no fp16-derived `M = 256`.
10. **Fleet stream semantics.** Each agent's experience is one observation → one action → one reward per step — a single time series, exactly as the Alberta Plan describes. The live engine batches **B semantically independent agents** (one per seed) into one process: per-lane disjoint state (agent-strided buffers, agentID on the grid's `.y` axis), zero cross-stream reductions, identical per-agent updates. Lane b runs seed b; a B=1 lane is bitwise self-reproducible run-to-run, and lane-disjointness is the standing acceptance test for any engine change (state-hash streams via `OAK_DUMP_STATE_EVERY` + `actions.bin` `cmp`). The environment is GPU-resident (`Sources/OakPhysics.metal`, fp32 port of MuJoCo certified population-equivalent to fp64 CPU MuJoCo by the null-calibrated TOST gate at N=8192 — see docs/roadmap.md F4 verdict); streams are CONTINUING by default (`OAK_EPISODE_STEPS=0`, never auto-reset — matching all CPU-era data), with gym 1000-step semantics available for external comparability and flagged in any SOTA claim.
11. **Inv-10: zero-intelligence-hit optimization rule.** Kernel optimizations split into two classes. **Bit-identical** (kernel splits and megakernel stage fusion separated by `threadgroup_barrier(mem_device)`, grid-stride or simdgroup/threadgroup remaps of independent elementwise work at any width ≥ 32, sequential re-ordering of writes-disjoint threadgroup work, per-thread parallelism without cross-thread reductions, pre-computed constants moved into `KernelConfig`/function constants (with `-fno-fast-math`, which forbids reassociation, shape-specialized unrolling is in this class), `simd_max` / `simd_min` reductions, lane-0 sequential reductions via `simd_shuffle` or threadgroup memory that preserve the original `fma` sequence in original index order): ship with **bitwise A/B proof** (`actions.bin` `cmp` + `OAK_DUMP_STATE_EVERY` state-hash streams) + Tier-S smoke. **FP-reorder** (`simd_sum`, `simdgroup_matrix` ops, threadgroup reductions where the accumulation order changes from sequential, `fma` decomposed into separate multiply + add): ship only on a `tools/fleet.py gate` **PROMOTE at Tier-M (N=1024 × 100k, paired by seed) for screening and Tier-L for promotion** against the current keeper. The population law replaces the retired 10-seed ±3% per-seed-lift rule: per-seed reshuffling that averages to zero now surfaces as a paired divergence/IQM tail at N=1024 instead of anecdotal seed inspection. Historical lesson retained: the first C2-perf attempt used `simd_sum` and showed aggregate parity while per-seed chatter std doubled (0.020 → 0.041) — exactly the failure mode the paired population gate is built to catch.

## Architecture

The agent is a focused-kernel single-stream Oak prototype over the Plan steps. K is the active option/subtask count (hard-pinned at 32, a structural constant — not derived to equal R); R = 256 is the stream-shape-derived capacity. Kernel threadgroup width is derived from the Metal pipeline execution width rather than an option page scheduler. Every learnable head and every state slot must be justified against the Plan; strict completion status is in `docs/roadmap.md`.

### Learnable parameters

| Head | Shape | Role |
|---|---|---|
| `valueW` | R | Main value GVF (cumulant = reward, differential, ρ̄-centered). |
| `psi` | K · R | K feature-indexed RR subtask GVFs (option 0 = main-task key, β=0). Cumulant is centered reward with per-head psiRho; stopping value is RR eq. 4 with the live high-water `bonusW` optimism term; eq.-9 ≥ ties stop. |
| `optPolicy` | K · D · R | K component policies. The wrapper samples from the emitted behavior descriptor and returns exact one-step importance ratios for each target option policy in the active bank. |
| `optReward` | K · R | K CENTERED differential reward-to-stop models r̂^k (per-head `optRho` centering; `q = r̂ + v(stop)`). |
| `vStop`, `vStop2` | 2 · K · R | Twin stop-value GVFs `E[v(s_stop)]` (value-equivalent partial model replacing the K·R·R transition tensor); twin disagreement gates planning trust, never the target. |
| `optDuration` | K · R | K average-reward option-duration models l̂^k with a structural one-step lower bound at decode. |

State representation (Mark IV geometry-gated sparse): a 32-slot always-active CORE (17 standardized obs, 12 prev-action, prev reward, bias — all raw inputs visible every step, Mark III law (b)) plus 224 gated slots in 32 banks of 7, exactly one active per bank selected by a STATIONARY tile of a designated core channel (tile-coding geometry, never magnitude competition). Gated slots hold multi-timescale obs traces and constructed features; active width A = 64. Standardization ridges are floored at 1/O (no spike amplification). Deep Fourier, random Fourier, and random-projection encoders are optional experimental modes, not Alberta Plan law. Constructed slots remain replaceable signed products/sums/residuals of useful parents. Plasticity is driven by MetaOptimize-style `alpha/beta/h/gradEnergy` families plus age/utility replacement; raw IDBD/Autostep are historical relatives, not the implementation target.

### Pipeline (single megakernel, multi-step command buffers)

`oak_megastep` runs the whole per-step loop GPU-resident (encode → plan → learn → replace → emit → sample → env), with the eq.-19 prioritized-sweeping planning fan-out riding the warps that idle during the physics chain (wall-free). The discrete per-stage kernels were deleted in Mark IV (they were compiled but never dispatched).

| Stage | What it does |
|---|---|
| encode | Observation/traces, core + geometry-gated features, value TD, RR subtask TDs, RR continuation, GPI selection, keyboard refresh, trust-gated one-step planning backup (planTd, fed into ρ̄ per differential planning). |
| plan | Full option bank's centered reward-model TDs, duration TDs, twin stop-value TDs, model-priority statistics. |
| learn | Value (decoupled td/plan normalizers), psi, option policies, option reward/duration, twin aux heads, and the prioritized-sweeping backup accumulator apply. |
| replace → emit | Utility/meta commits, bonusW high-water, ring write + priorities, G&T replacement (pre-D1 rate), then option means + chord blend emission. |
| env window | Physics chain on one warp; prioritized-sweeping eq.-19 backups + twin copy-2 updates on the idle warps. |

The dispatch shape stays fixed, but the internal design pattern is the named FC-STOMP loop: `FeatureConstruction -> GVFPrediction -> RewardRespectingSubtasks -> OptionPolicies -> OptionModels -> Planning -> OptionKeyboard -> SearchControl -> GenerateAndTest -> ActionEmission`.

### Action interface (Inv-1)

Each env declares one structured action spec: `Discrete N` (categorical) or `Box A` (continuous with `[low, high]` bounds). The integer `D_policy = component_param_count` is the agent's only window into the env's action structure.

**Universal I/O contract (mixed-env fleets, 2026-07-19; Phase-3 multi-shape, 2026-07-19).** The AGENT kernel stays env-blind: obsDim=17 and D_policy=12 are the function constants for EVERY env. The compile-time `OAK_ENV_SPECS[envKind] → {obsDimNatural, actionKind (0=Box,1=Discrete), actionCount}` table (`Sources/Gridworld.metal`) is the single source of truth the four sanctioned sites read; envs write their natural obs channels and zero-pad to 17. Integer gridworlds (`FourRooms-cont-v0`, `DeepSea-cont-v0`, `KeyCorridor-cont-v0` (key→door→goal, D4), `Chain-cont-v0` (bsuite L=16, D2), `LargeDeepSea-cont-v0` (N=16 DeepSea, D2), `MultiKeyCorridor-cont-v0` (3 sequential keys/doors, D4), `MemoryChain-cont-v0` (bsuite Memory Chain L=32, D2 — cue visible only at pos 0, choice at pos 31), `ProceduralMaze-cont-v0` (per-lane Philox-generated wall layout, D4), `HighDimMaze-cont-v0` (serpentine maze, D6 with stay+dash, 16-ch rich obs), `POMaze-cont-v0` (view radius 1, D4), `DelayedChain-cont-v0` (bare corridor L=32, reward only at the far end, D2) — `Sources/Gridworld.metal`) are bitwise-certified against `tools/gridworld_ref.py` by the `OaK gridworld-probe` case gate; classic-control envs (`Acrobot-cont-v0` D3, `MountainCar-cont-v0` D3, `CartPole-cont-v0` D2, `Pendulum-cont-v0` Box 1 — `Sources/ClassicControl.metal`) are certified by `tools/physics_parity.py --classic-tost` (F1-style one-step parity + F4-style null-calibrated population TOST) plus the `--classic-probe-*` single-step boundary gate. All are CONTINUING: completion respawns state inside the stream (`done` is a wrapper metric only). Discrete envs consume the first N of the 12 policy coords as Gumbel-max logits; Box envs the first actionCount as means. `OakFleetLane.envKind` (0=cheetah, 1=FourRooms, 2=DeepSea, 3=KeyCorridor, 4=Acrobot, 5=MountainCar, 6=CartPole, 7=Pendulum, 8=Chain, 9=LargeDeepSea, 10=MultiKeyCorridor, 11=MemoryChain, 12=ProceduralMaze, 13=HighDimMaze, 14=POMaze, 15=DelayedChain; stride 384→400 with the `gw[10]` env-state union) is read ONLY in the env window (`oak_stage_fleet_env` switch), the sample stage (Box vs Discrete-N dispatch), fleet init, and the record path — never by agent stages. `OAK_BENCH_ENVS="EnvId:count,..."` assigns contiguous seed ranges per env (seed b → lane b); run-root meta.json gains `envSchedule` (absent for single-env runs, which stay byte-comparable with legacy roots). The default schedule is `HalfCheetah-v4:1024,KeyCorridor-cont-v0:128,FourRooms-cont-v0:64,DeepSea-cont-v0:64,Acrobot-cont-v0:64,MountainCar-cont-v0:64` (= 1408 lanes).

The agent emits `policyMean[D_policy]` from the emit stage (`emit_all_options` / `emit_blend_grid`). The megakernel's sample stage (`oak_stage_sample_act_wide` + `oak_stage_sample_act_discrete`; the CPU bridge's `HCEnv::step_oak` was deleted per the F4 verdict) is the **sole** site that samples actions and computes score functions:

- Discrete N: Gumbel-max categorical (`-log(-log(U))` + argmax over the first N policy means, ACT-stream uniforms); target-option actor scores use `1{a=d} − π_d` with π the softmax over the N means (Maddison 2014; Williams 1992); importance ratio is the exact categorical `π_o(a)/π_b(a)` with the same structural clamp at K; remaining policy coords get zero score/action-signal entries. The zero/random baselines draw uniform-categorical actions.
- Box A: currently exposes `2A` policy coordinates to keep `D_policy`, default rank, and K stable. The first A coordinates are means; the second A coordinates are wrapper-reserved/inert. The wrapper samples with structural fixed `σ = 1/D_policy`, emits residual-style mean credit (`raw − μ`) for Box mean slots, writes zero score/action-signal entries for reserved scale slots, and clamps only the env-facing action. Full `D_policy = A` mean-only deletion failed the 1M gate and remains rejected unless a different actor geometry earns fresh evidence.

Adding a new env or action-space shape requires adding to `ActionSpec` in Swift; no agent or kernel changes.
`doneCount` is a wrapper random-restart metric, not an episode signal to OaK; the engine state is never reset on `done`.

## Fleet run output (fleetSchemaVersion: 1)

A fleet run root (`.oak-benchmarks/<run-id>/`, override with `OAK_METRICS_STORAGE`) contains:

- **`meta.json`** — run config: `fleetSchemaVersion`, `envId`, `batch` (B), `seeds`, `steps`, `policies`, `ablation`/`ablationLabel`, `startedAt`. Mixed-env runs (`OAK_BENCH_ENVS`) add `envSchedule`: `[{envId, seedStart, seedEnd, flopsPerLaneStep}]` (contiguous seed ranges; `tools/table.py` hard-refuses schedule mismatches).
- **`metrics.<policy>.oakt`** — **the continuous metrics tape (METRICS LAW 2026-08-05): the primary population time-series artifact.** One row per lane per command-buffer window (~1024 steps — ~1000 points/lane at Tier-L, 100× the old heartbeat resolution), 26 curated fields (rewardSum, doneCount, rewardRate ρ̄, kernelRho, action deltas, mainTd, entropy, termination, keyboard L1s, model/importance/search diagnostics, reinit + NF counters, paramAbsMax — the last real only on heartbeat windows, −1 sentinel between). Self-describing binary (OAKT v1: header carries the field table as JSON; raw-DEFLATE segments flushed per heartbeat; torn tails drop by crc like record-v4 chunks); ~15–25 MB per Tier-L. Written by `MetricsTapeWriter` (Sources/Main.swift) from a slim per-CB GPU snapshot (the O(paramStride) param scan stays heartbeat-only); readers: `tools/metrics_tape.py` (Python; `synth_hb_rows` reconstructs fleet.jsonl-shaped heartbeat rows so fleet.py/table.py are decision-identical across eras) and `frontend/server/metricsTape.ts` (injected into the fleet columnar reduction, capped at 256 windows/query). The dead weight the tape replaced was removed AT THE SOURCE, not hidden: `envInfo` (per-step Kahan accumulators feeding 28 unread JSON keys/lane/hb, one channel a bit-identical duplicate), the five structurally-zero stats (`selectedOptionDuration`, `keyboardOrigin`, `optionDurationMean` — which was also a ghost Tier-0 ranking signal, `dynaBackupMean`, `learningMature`), and the never-read fields (`keyboardSketchDim`, `selectedOptionFeature`, `continuingRestartCount`, snapshot `resets`).
- **`fleet.jsonl`** — tape-era roots: one compact-JSON row per (lane, **final only**) — the per-lane certificate: `v`, `kind` (`final`), `lane`, `seed`, `envId` (per-LANE under a mixed schedule), `step`, `wallSeconds`, `stepsPerSecond`, `rewardSum`, `doneCount`, `firstNonFiniteStep`/`firstNonFiniteDetector` (the death certificate), `kernelRho` (the kernel's internal ρ — a divergence diagnostic, distinct from the host EMA), `paramAbsMax` (full-scan value; dates 1e36 blowups before NF counters fire), and the trimmed `health` block (`rewardRate` host ρ̄, `actionDeltaL2Mean`/`Peak`, `nonFiniteByPhase` cumulative `{encode, learn, plan, tdPsi}`, search/model/keyboard/actor diagnostics). Pre-tape roots keep their per-heartbeat rows and every reader accepts both eras.
- **`deaths.jsonl`** — one row per lane death event: `seed`, `lane`, `step`, `detector` (`envReward` | `policyMean` | `paramNonFinite` | `paramExplosion` | `kernelRho` | `envError`), `nfByPhase`, `rhoAtDeath`, `paramAbsMax`, `actorSupportEntropy`, `terminationRate`, `optionDurationMean`, `rewardSumAtDeath`, `lastGoodRewardRate`. A "dead" lane keeps stepping in the GPU fleet (one continual stream, never reset; recovery is data); the certificate dates the first event.
- **`aggregate.jsonl`** — one row per heartbeat per policy: `step`, `alive`/`dead`, delivered-rate P10/P50/P90, `aggStepsPerSecond`, `wallSeconds`. This is the live dashboard (`tools/fleet.py watch`).
- **`summary.json`** — written by `tools/fleet.py report` (`summarySchemaVersion: 2`, plus `solvedThreshold`). VOCABULARY LAW: a fleet lane IS a seed, so every population figure counts seeds — `seeds`, `rewardRateIQM` (+`rewardRateIQMCI95`, `rewardRateIQMNonDiverged`), `recentRewardRateIQM`/`Max`, `bestSeed`/`bestSeedRewardRate`/`bestSeedTotalReward`, `positiveSeeds`, `rewardRatePercentiles`, `thresholdFractions`, `peak` (`peakStepQ`/`peakRewardRateQ`/`retentionQ`), `solvedSeeds`/`solvedSeedsRetained`/`solvedRetentionFraction`, `retentionFraction`, `seedStepsTotal`, `divergence.diverged`, `bestSeedRewardPerGFLOP` — and the PERSISTED per-seed vectors `rewardRateBySeed`, `recentRewardRateBySeed`, `retentionBySeed`. v1 roots (the ~600 historical ones, keyed `lanes`/`iqmDeliveredAll`/`plasticity`/`igniters`/`sustainingFraction`/…) are NOT regenerated; `frontend/server/summaryAdapter.ts` maps them key-for-key on read and synthesizes nothing. Per-env `envSlices` carry their own `rewardRateIQMCI95` (the slice's own population, never the pooled one). **STRICT-JSON LAW: every artifact fleet.py writes goes through `write_json` (sanitize non-finite → `null`, then `allow_nan=False` as a permanent tripwire).** RFC 8259 has no NaN/Infinity, and `json.dumps` emits them bare: one seed with an undefined `retentionBySeed` (peak windowed rate ≤ 0 ⇒ `final/peak` undefined) used to make the WHOLE file unparseable, and `readJsonFile` in `frontend/server/runIndex.ts` turns that into `summary: null` — silently blanking the population panel for the best kernel itself. `null` is the honest encoding of "undefined for this seed"; readers already render it as "—". Mixed-env roots hit this constantly (Pendulum/LargeDeepSea can be 64/64 seeds).
- **`.oak-benchmarks/ROLES.json`** (bench root, not per run) — `{rolesSchemaVersion: 2, bestKernel, candidate, parent, featured?}`, values are run-root directory NAMES. `bestKernel` replaced `keeper`; every reader (`tools/table.py`, `tools/check.sh`, `frontend/server/roles.ts`) accepts BOTH spellings for one migration cycle and writes v2.
- **`<envId>/seed<N>/`** — per-lane `actions.bin` + `meta.json`; `state-hashes.jsonl` when `OAK_DUMP_STATE_EVERY=N` is set (byte-identity validation). **RECORDING LAW (2026-08-05): every fleet run records EVERY lane in record v4 by default** — `run_fleet.sh` and the binary default to `OAK_RECORD_ACTIONS=1` + `OAK_RECORD_FORMAT=4`; a run without recordings is a wall with no scene, and re-running a Tier-L just to get streams costs 40 minutes (this happened; never again). Recording is passive (agent stream bit-identity proven by state-hash `cmp` and 0/1024 `rewardSum` mismatches across recorded/unrecorded re-runs) and must stay at **SPS parity** — a recording-induced throughput regression is a bug to fix on the drain path (background thread, chunked DEFLATE), never a reason to disable recording. Formats: **v4** (`recordVersion: 4`, the frozen columnar/quantized/DEFLATE codec of `frontend/src/sim/RECORD_V4_SPEC.md`, ~22× smaller — 2.7 GiB vs 61 GiB for a 1024×1M fleet; decoded by `frontend/src/sim/record4.ts` and `tools/record4.py::load_rows`, which dispatches on `recordVersion` so consumers accept both) is the fleet default; discrete actions/done bits are bit-exact, quantized floats are display-exact (poses 1e-4, ctrl 1/255). **v3** (`recordVersion: 3`, raw 16 f32 per step — `ctrl[6], done, pose[9]`; cheetah pose = post-step qpos9) is the EXPLICIT certification format (`OAK_RECORD_FORMAT=3`) for gates whose tolerances sit below the v4 quantization contract: `tools/physics_parity.py --classic-tost` (1e-4 one-step parity) and `frontend/test/fixtures/gen_classic.py` (1e-6 cheetah replay) refuse v4 lanes with a clear error. The directory's `<envId>` level is the lane's own env under a mixed schedule. Feeds the `frontend/` replay theater (served over `/api/runs/:id/recordings/...`).

**Tools** (numpy; `requirements.txt`):

- `.venv/bin/python tools/fleet.py report <run-root>` — population report: Kaplan-Meier survival + restricted mean survival time (RMST), divergence rate with Wilson 95% CI (overall + per detector), IQM average reward rate (`rewardSum/steps`) with bootstrap CI over ALL seeds (headline; never condition on survival) and non-diverged-only (diagnostic), peak & retention quartiles (peak step / peak reward rate / retention = final ÷ peak), entropy-collapse rate. Markdown output pastes into `docs/roadmap.md`; also writes `summary.json` (v2) and `organ-report.json`.
- `.venv/bin/python tools/fleet.py gate <best-kernel-root> <candidate-root>` — **the promotion law.** Same seed set + horizon required (hard error); all bootstraps paired by seed. Criterion labels carry their pre-v2 name in parens (`rewardRateIQM (was iqmDelivered)`, `solvedSeeds (was igniters)`, `retainedSeeds (was sustainers)`, `positiveSeedFraction (was positiveFrac)`); the tests, thresholds and PASS/FAIL semantics behind them are UNCHANGED — the rename is decision-identical, proven by re-running a real Tier-L pair before and after. PASS iff ALL four hold (AND-gate ⇒ each one-sided α=0.05, no multiplicity correction): (1) divergence non-inferiority, 95th pct of paired Δp ≤ δ (default 0.64/√N: 0.02 @ N=1024); (2) IQM average-reward-rate non-inferiority, 5th pct of ΔIQM ≥ −5%·|IQM_bestKernel|, with a non-blocking IMPROVEMENT flag (what SOTA claims must cite); (3) RMST survival non-inferiority, 5th pct of ΔRMST ≥ −5%·horizon; (4) per-phase NF prevalence non-inferiority — for each phase, 95th pct of paired Δ(fraction of lanes with any non-finite in that phase) ≤ δ (per-lane PREVALENCE, not fleet-sum totals: a few already-diverged lanes spray non-finites post-death and inflate totals without changing how many lanes fail, which double-counts what criterion 1 already measures; consistent with the equivalence gate's `nf_prevalence`). Exit 0 = PROMOTE, 1 = REJECT. Seed-3/seed-8 folklore is not encoded anywhere — those are ordinary seeds (seed 3, seed 8 of every fleet) whose trajectories remain inspectable in `fleet.jsonl`. On a mixed-env pair the gate also prints `[info] perEnvDelta` — paired per-env ΔIQM, worst first, **diagnostic only, it never votes**: the pooled IQM is dominated by whichever env brings the most seeds and the biggest reward scale (HalfCheetah is 1024 of 1408 lanes at ~100× the gridworlds' scale), so an env can collapse without moving the headline. For a per-env PROMOTE/REJECT use `gate --env-slice <envId>`, which runs EVERY criterion paired on that env's own lanes. A zero-tolerance `perEnvNoRegression` criterion did run in July 2026 (`.oak-benchmarks/boost-rhat-tierL-20260725.log`) but was never committed and is deliberately NOT revived as law: it rejected `stepcap1M-20260724` for "Acrobot −2.0%, Chain −0.0%" while every other criterion passed with an IQM IMPROVEMENT, and re-measured on the real p7s12→p93 pair it would have REJECTED the current keeper (MountainCar −5.4%, KeyCorridor −3.2%, Acrobot −1.0%) despite +19.2% IQM and divergence eliminated.
- `.venv/bin/python tools/fleet.py watch <run-root> [--follow]` — live aggregate view.
- `.venv/bin/python tools/fleet.py lanes <run-root> --at <step>` — per-seed pack
  forensics at a heartbeat: crashed/negative/low/mid/leader health-median diff
  (entropy, termination, duration, keyboard L1, feature reinits, action delta,
  paramAbsMax, ρ̄). The organ-bisect loop's step 2 as one command.
- `tools/check.sh <name> [steps=100000] [ablate=none]` — the standard VISIBLE
  verification run (B=1024, fixed CB=1024, output in `.oak-benchmarks/<name>`)
  with the full heartbeat table printed. Use for every pre-launch check; never
  run checks in a scratchpad the user can't watch.
- `tools/ablate.py --steps N --base <ablations> <organ> [organ...]` — the
  organ-bisect matrix: runs baseline + each organ ablated on top of --base,
  prints the conviction table. (Named ablate.py deliberately: a tools/bisect.py
  shadows Python's stdlib `bisect` and hijacks fleet.py's argparse.)
- **`frontend/` — the visualization app** (Bun + Vite + Three.js; replaced the
  retired `tools/replay_html.py` 2D viewer, 2026-08-04): run gallery over every
  `.oak-benchmarks/` root, per-run population dashboard, live watch (SSE), and a
  3D replay theater with one artisanal, accuracy-locked nature scene per env
  (all 16; sim decoders certified bitwise vs `tools/gridworld_ref.py`, cheetah
  FK to 1e-9). `cd frontend && bun run dev` → http://localhost:5173. See
  `frontend/README.md`; scene law in `frontend/src/scenes/STYLE.md`.
- `tools/table.py <candidate-root> <best-kernel-root> <step>` — THE CANONICAL
  heartbeat table. **MANDATORY at every Tier-L heartbeat report**: never
  hand-compose heartbeat tables (hand tables drift and drop rows — SPS was
  silently lost once). Fixed rows, fixed order: config (run names + ablation +
  B + horizon + start time, self-identifying — never trust a bare
  "candidate/best-kernel" label) | deaths | median (cumulative) |
  median (live window) | P10 | best seed | live max | positivity | SPS |
  reward/GFLOP (median) | fleet total (all seeds). The LIVE rows (median live
  window, live max, positivity) are MEASURED — `rewardSum` differenced over the
  heartbeat window — NOT `health.rewardRate`, which is the agent's internal ρ̄
  estimate (a learned parameter and a divergence diagnostic, not delivered
  reward); that mislabel was fixed 2026-08-04 and moved the numbers it had been
  flattering. It prints ⚠️ on candidate/best-kernel ablation mismatch
  (the config error that poisoned a 1M Tier-L on 2026-07-05) and it REFUSES
  (hard error, no output) when either root lacks a row at the requested step —
  apples-to-apples is structural: compare ONLY at steps present in BOTH roots
  (user law 2026-07-05; mid-run peeks wait for the next common milestone).
  Every table shown to the user is pasted VERBATIM into the reply text (no
  retyping, no dropped rows), and the comparator is ALWAYS the current best
  kernel root. fleet total (sum of rewardSum across ALL seeds) is the population
  objective; reward/GFLOP is the SOTA-target metric (internal comparison only
  until FLOPs are instrumented on both sides).
- **Heartbeat cadence + kill discipline (user law, 2026-07-05).** For every
  run ≥ 300k: print the canonical table IN CHAT at EVERY 100k milestone (a
  Tier-L = 10 tables), via chained harness-notified waiters blocking on
  `aggregate.jsonl` reaching the next milestone (never poll, never go
  silent). Pre-register kill rules BEFORE launch (e.g., live median < ½
  keeper's at the matching step; positivity collapse; death wave) and execute
  them immediately — a run that is clearly not an upgrade is killed at the
  milestone that shows it, not run to horizon. Screens exist to find gems
  and discard stinkers early.
- **The organ-bisect law** (`.claude/skills/organ-bisect/SKILL.md`): for ANY
  population pathology, convict before you fix — trajectory shape, `lanes`
  diff, death certificates (`explosionElem` → head+column), then the ablation
  matrix, then fix at the source and verify at the pathology's own depth.
  One ablation bisect found in 10 minutes what eight theorized fixes missed.

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
   | L | 1024 × 1M | ~36–40 min (mk8-era) | keeper promotion, frontier studies | survival curves through the ~1M plasticity frontier, Δp ≳ 2% |
   | M-8192 | 8192 × 100k | ~34 min | instrument equivalence, SOTA-grade CIs | divergence ±1pp, IQM at the strict ±5% band |
   | XL | 1024 × 10M | F5-opt-gated | decay-constant program (not a gate) | post-zenith time constants, 10M survival |

   **Hard rule: no protocol step exceeds ~2 h wall-clock.** Wall-clock derives from the measured constants in `docs/roadmap.md` (GPU loop: ~470K agg @ B=1024 mk8-era — Tier-L ≈ 36–40 min; the older 265.7K agg @ B=1024 / 401.9K @ B=16384 constants are pre-Mark-IV) — never from estimates. XL's 10M horizon divides by the per-lane rate (~259 steps/s at B=1024), not aggregate throughput; it unlocks when agent-kernel fusion/occupancy work (the open F5-opt item — agent kernels are 77% of step wall) raises per-lane rate ~4×.
3. Promote only on a Tier-L `fleet.py gate` PROMOTE against the keeper baseline. The gate is population law (divergence Δp, IQM delivered rate, RMST survival, NF phases — all paired by seed); a SOTA claim additionally requires the gate's IMPROVEMENT flag.
4. Per-seed collapse is population data, not an averaging input: `fleet.py report` reports it as divergence rate, survival curves, and detector counts. Individual seeds (3, 8, …) remain inspectable as lanes in every fleet, but no single seed blocks or licenses promotion.
5. Name run roots by hypothesis and tier, for example `.oak-benchmarks/closure-arbiter-tierM-YYYYMMDD`, and inspect live runs with `tools/fleet.py watch --follow` rather than waiting for final rows.
6. Use subagents when the work can truly run in parallel: a Metrics Archaeologist mines benchmark roots, a Plan/Invariant Reviewer checks Alberta/FC-STOMP fit, a Mechanism Designer proposes closed-loop organs, a Benchmark Steward watches gates, and a Metal Auditor handles speed/FP-order risk. The main thread integrates; subagents do not define project doctrine.
7. Archive rejected mechanisms in `docs/roadmap.md`/`AGENTS.md` when they teach a reusable lesson. Keep the live code lean; diagnostics earn their place only when they shorten the next decision.

### Reporting & reliability standard (non-negotiable — baked into the tools, not left to memory)

- **Run every benchmark via `tools/run_fleet.sh <steps> <out-name> [keeper-root]`**, never a bare `OaK benchmark`. It launches the job FULLY DETACHED — a new session via Python `start_new_session` (macOS has no `setsid`) — so it survives Claude Code **context compaction** and idle-timeout, which SIGTERM `run_in_background` tasks ([anthropics/claude-code #25188](https://github.com/anthropics/claude-code/issues/25188), [#32050](https://github.com/anthropics/claude-code/issues/32050)). THAT is what silently killed our long runs — short (~6 min) ones finish before a compaction, 64-min ones do not; it was **never** laptop sleep (a wrong guess). Call `run_fleet.sh` as a NORMAL foreground Bash command — it returns immediately after detaching; **never wrap it in `run_in_background`** (that reintroduces the bug). The detached job runs benchmark → population report → gate and `touch`es `<root>/.done`; poll the log + `.done` (a thin `run_in_background` waiter that blocks on `.done` is an acceptable notifier, but the durable work is the detached job, not the waiter). `caffeinate -dimsu` is kept as cheap insurance only.
- **The population is the deliverable — never a single seed.** The headline is `fleet.py report`'s "Population progress" row: reward rate P10/P50/P90, the fraction of seeds above each threshold (`%>0` walking, `%>1` the milestone, `%>5` ~SAC-delivered, `%>SOTA(10)`), and `retention` (`retentionFraction`: seeds holding ≥½ their peak reward rate). The goal is the WHOLE population at SOTA **and staying there forever** (continuous online) — judge by the threshold fractions + retention, not by the best seed. A hand-picked rocket seed is a *diagnostic*, never the claim; reporting one as the result is the retired seed-focusing trap.
- **An incomplete run is poison.** A run that did not reach its declared horizon (laptop sleep / OOM / harness stop) is flagged loudly by `fleet.py report` (`⚠️ INCOMPLETE RUN`) and **refused** by `fleet.py gate` (`GATE ERROR`). Never trust, report, or gate a partial run. When a long run stops early, a silent death is the default suspect — check completeness FIRST.
- **Strong-seed / peak inspection is a REQUIRED diagnostic, not a headline.** IQM is robust to the tails, so a change that crushes the strong mode (or a silent death) can pass the IQM screen — always check the best seed and `retentionFraction` for ceiling damage. But the promotion claim is always the paired population gate.

## CLI

| Command | Use |
|---|---|
| `OaK benchmark` | One fleet per policy: B independent (agent, env, seed) lanes in one process — the FC-STOMP loop, action sampling, and env physics all run GPU-resident in multi-step command buffers. `OAK_BENCH_ENV` is currently `HalfCheetah-v4` (also `FourRooms-cont-v0`, `DeepSea-cont-v0`); `OAK_BENCH_ENVS="HalfCheetah-v4:768,FourRooms-cont-v0:192,DeepSea-cont-v0:64"` runs a mixed-env fleet with contiguous seed ranges per env (per-lane `envId` in fleet.jsonl, `envSchedule` in meta.json). `OAK_BENCH_STEPS`, `OAK_BENCH_SEED_COUNT` (default 1024) / `OAK_BENCH_SEEDS` / `OAK_BENCH_SEED_START`, and `OAK_BENCH_POLICIES` define the workload. `OAK_FLEET_TIMING=1` prints heartbeat throughput. One launcher owns the workspace at a time; nonempty requested roots get a fresh suffixed output dir. |
| `OaK gridworld-probe <cases.bin> <out.bin>` | The gridworld case-gate probe: runs a `tools/gridworld_ref.py` case file's in-records through the Metal env step (the same functions the fleet env window calls) and writes out-records for `gridworld_ref.py --compare` (bitwise certification of `Sources/Gridworld.metal`). |

`OAK_ABLATE` defaults to `none`. It accepts a comma-separated diagnostic list: `feature-construction`, `gvf-prediction`, `rr-subtasks`, `option-policies`, `option-models`, `planning`, `option-keyboard`, `model-to-keyboard`, `keyboard-to-policy`, `search-control`, `search-to-generate-test`, and `generate-and-test`. These are specialized Metal function-constant variants, not a promoted behavior mode.

There is no deep-diagnostics shadow path in the live engine. Add audit heads only when they earn a Plan role and pass the same speed/learning gates as behavior code.

## Performance: profiling (HEADLESS — verified 2026-07-04)

The repo template records without the Instruments GUI. Never trace (or rebuild the metallib) while a fleet run is in flight.

```
xctrace record --template tools/instruments/ShaderTemplate.tracetemplate \
  --output out.trace --time-limit 15s \
  --env OAK_BENCH_ENV=HalfCheetah-v4 --env OAK_BENCH_POLICIES=oak \
  --env OAK_BENCH_STEPS=200000 --env OAK_BENCH_SEED_COUNT=1024 \
  --env OAK_METRICS_STORAGE=/tmp/trace-run \
  --launch -- .build/release/OaK benchmark
xctrace export --input out.trace --toc                     # list tables
xctrace export --input out.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="SCHEMA"]' --output t.xml
```

Useful schemas: `metal-gpu-intervals` (per-dispatch GPU times), `gpu-performance-state-intervals` (clock derating, "Minimum … due to active device conditions"), `device-thermal-state-intervals` (thermal axis for SPS-decay questions). Caveat: `metal-shader-profiler-*` and `gpu-counter-*` tables exist but record ZERO rows with this template (shader-profiler/counter profiles are unarmed); arming them needs a one-time GUI template edit — bare `--instrument 'Metal GPU Counters'` records nothing in attach OR launch mode.

Complementary in-code instruments: `OAK_MEGA_STAGE_MASK` (per-stage timing; single-stage-only masks isolate cost but DEGENERATE the data — prefer `encode+X` pairs under live dynamics), `OAK_PROBE_FIXEDIDX=1` (fixed contiguous gather indices — separates memory-scatter cost from issue cost). Both are timing instruments; trajectories are garbage. The per-stage/falsification evidence log lives in docs/roadmap.md (SPS program).

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
