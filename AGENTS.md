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
- **Named FC-STOMP loop / ablations.** `Sources/OakKernel.metal` exposes the live loop vocabulary directly in code: `FeatureConstruction`, `GVFPrediction`, `RewardRespectingSubtasks`, `OptionPolicies`, `OptionModels`, `Planning`, `OptionKeyboard`, `SearchControl`, `GenerateAndTest`, and `ActionEmission`. `OAK_ABLATE` compiles specialized Metal pipeline variants for diagnostic ablations of those sections and couplings; canonical promotion runs use `OAK_ABLATE=none`.
- **Ablation lesson.** The first 50k focus matrix (`.oak-benchmarks/step11-ablation-focus50k-*`) confirms the live walking mode depends on the whole FC-STOMP loop. The largest losses come from `keyboard-to-policy`, `option-keyboard`, and `model-to-keyboard`; `search-control` can help a quiet seed while killing the rocket. Treat this as evidence for a richer cumulant/value-space keyboard and bundle-utility loop, not as permission to delete organs.

### Current empirical status (2026-04-26)

OaK is no longer only a negative-return random-beating control. The bridge has now shown real HalfCheetah walking:

- `tools/bridge_oracle_cem.py` through the same C bridge, fixed open-loop sinusoid, 5 seeds x 5k steps: mean return `+4428.30`, mean `info:x_velocity=+0.983`.
- `.oak-benchmarks/fresh-postaudit-300k-20260426-030016`, 10 seeds x `{oak,zero,random}` at 300k: Oak mean return `+5365.61`, max seed `+109689.46`, mean `info:x_velocity=+0.0379`, positive velocity `6/10`, positive return `2/10`, corrected Oak NF total `4`.
- `.oak-benchmarks/deep-seed3-10m-20260426-030016`, Oak seed 3 at 10M: the gait loses its 1M snowball but never loses cumulative forward momentum (`info:x_velocity` mean `+0.011783`, x-position delta `+5891.636`, final return `+64937.78`). Terminal health shows the FC-STOMP cycle going quiet (`terminationRate=0`, `modelPriorityMax=0`, `optionDurationMean=1`, actor entropy/update norm `0`) after one learn NF and one TD-psi NF.
- `.oak-benchmarks/microscope-seeds0-5-1m-20260426` plus `.oak-benchmarks/microscope-seeds6-9-1m-20260426`, Oak-only 1M coverage: mean return `-2783.91`, mean `info:x_velocity=+0.020291`, positive return/velocity `3/10`, NF total `8`. Seeds 3, 6, and 9 are real forward snowballs by 1M; seed 8 is the large backward attractor; seeds 0/1 are quiet failures.
- `.oak-benchmarks/short-broad-surprise-full-300k-20260426`, 10 seeds x `{oak,zero,random}` at 300k with a structurally gated active-backward keyboard refresh: Oak mean return `+9988.57`, max seed `+109689.46`, mean `info:x_velocity=+0.0509`, positive velocity `8/10`, positive return/zero-beating seeds `3/10`, Oak NF total `3`. Random mean is `-78032.28`; zero mean is `+0.01`. This reproduces the Oak-only keeper profile and repairs seed 8 from `-37294.86`, `info:x_velocity=-0.0867` to `+6837.42`, `info:x_velocity=+0.0356`.
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
Quiet-invalid option rebirth (`quiet-invalid-rebirth-target300k-20260426`) is rejected as behavior-dead: it repeatedly resets seed 1's selected option and drops selected-option age, but final reward/velocity remain bit-identical while replacement churn explodes. RR/psi implicit TD-energy normalization (`psi-implicit-target10k-20260426`) is rejected because it cools the waking surface immediately, regressing seeds 0/1/8 at 10k. Preserving last finite RR continuation (`rr-preserve-cont-target50k-20260426`) is rejected because it wakes seed 1 into a `-14506.99` control-cost sink. Invalid-RR main-TD fallback (`rr-invalid-maintd-target50k-20260426`) is behavior-identical and therefore not worth carrying.
Heavy-losing selected-option fallback (`heavy-losing-fallback-target50k-20260426`) is rejected: it preserves seed 3 but turns seed 1 into `-7244.57` and seed 8 into `-7579.48`, proving selected-option execution is still too blunt even under the high-mass negative-rate discriminator. Query-valid/invalid-duration keyboard probes (`queryvalid-cont-target50k-20260426`, `invalid-quiet-refresh-fixed-target50k-20260426`, `invalid-duration-probe-target50k-20260426`) preserve seed 3 and seed 8 but wake seed 1 into a large control-cost sink (`reward_ctrl` about `-12950` to `-14600`). Heavy-long reward-TD actor arbitration (`heavy-long-tdactor-only-target50k-20260426`) is safer and NF-clean, but still regresses seed 1 to `-467.51` and negative velocity. The live lesson is that the seed-1 discriminator is close, but the missing organ is a direction/effort arbiter, not raw fallback, raw invalid-query refresh, or raw reward-TD actor pressure.
Direction/effort keyboard arbitration was tested and rejected as a keeper in raw, relative, refresh-arbiter, active-short-protected, and split-score forms (`direff-keyboard-oak300k-20260426`, `direff-refresharb-oak300k-20260426`, `direff-relative-oak300k-20260426`, `direff-relative-noshort-oak300k-20260426`, `direff-splitscore-target50k-20260426`). It is a real FC-STOMP organ: relative direction/effort produced NF-clean positive seed 1 (`+24706.23`) and seed 2 (`+2616.03`) at 300k while preserving seed 3, and raw direction/effort preserved the seed-8 snowball (`+15702.38`) and woke seed 4 (`+1843.16`). The current keeper direction organ is narrower: persistent keyboard provenance routes raw direction/effort only through the active-backward path and leaves quiet rescue unpromoted (`provenance-direff-v7-oak300k-20260426`). It preserves seeds 0-7/9 bit-identically, improves seed 8 from `+6837.42` to `+23119.76` with `info:x_velocity +0.0950`, and keeps the old sampled NF total at `3`. Quiet seed 1 and seed 2 remain open; future work must find the relative quiet-rescue discriminator without stealing seed 8 or disturbing seed 3.
Cumulative-NF audit (`cumulative-nf-arbgate-oak300k-20260426-165800`) is behavior-identical to `provenance-agefree-oak300k-20260426` on every seed's reward and velocity, but reveals the old NF gate was sampled, not cumulative: seed 1 has `871,952` cumulative non-finite events by 300k (`learn=274,413`, `plan=323,126`, `tdPsi=274,413`). All other seeds remain NF-clean. The same patch gates direction/effort scoring so the serial lane-0 score passes run only when raw/relative arbitration is eligible, raising mean Oak SPS from about `2030` to `2068` without changing behavior.

Late-switch audit (`seed3-selected-bundle-trace420k-20260426-193108`) shows the seed-3 rocket does not fade gradually. It switches from selected option 18 with positive reward rate and finite health to selected option 0 with negative model-priority debt, duration `1`, hidden keyboard mass around `4.33e20`, and catastrophic action deltas. The keeper candidate `ungrounded-invalid-rescue-oak300k-20260426-195038` is 300k behavior-identical to `relative-td-or-broad-lock-oak300k-20260426-185147`; its only intended behavior change is after an invalid selected option's hidden keyboard mass exceeds lived stream length. Focused 450k evidence (`ungrounded-invalid-rescue-target450k-20260426-194018`) improves seed 3 from `+77899.15` to `+118129.84` and reduces focus NF without changing seeds 1/5/7/8 final reward, but it does not solve the action-delta/NF pathology or quiet seeds. Focused 1M evidence keeps seed 3 high (`+115363.16`) while seeds 7/8 decay badly, so this is a useful search-control rescue, not a promotion.

Rejected late-switch refinements: stored chord normalization kills the seed-3 rocket at 100k; raw-duration trust kills the rocket and floods NF; positive-reward active-bundle protection kills discovery; selected-behavior utility wakes seed 1 but destroys seed 3; effort-weighted model scoring wakes seed 7 but destroys seed 3; invalid-continuation interruption hurts seed 7 and does not save seed 3; ungrounded invalid option reset contaminates seed 5 with NF. The current lesson is precise: the late switch needs search-control handling of hidden invalid commitment without removing the magnitude memory that discovers the rocket.

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
10. **Fleet stream semantics.** Each agent's experience is one observation → one action → one reward per step — a single time series, exactly as the Alberta Plan describes. The live engine batches **B semantically independent agents** (one per seed) into one process: per-lane disjoint state (agent-strided buffers, agentID on the grid's `.y` axis), zero cross-stream reductions, identical per-agent updates. Lane b runs seed b; a B=1 lane is bitwise self-reproducible run-to-run, and lane-disjointness is the standing acceptance test for any engine change (state-hash streams via `OAK_DUMP_STATE_EVERY` + `actions.bin` `cmp`). The environment is GPU-resident (`Sources/OakPhysics.metal`, fp32 port of MuJoCo certified population-equivalent to fp64 CPU MuJoCo by the null-calibrated TOST gate at N=8192 — see docs/roadmap.md F4 verdict); streams are CONTINUING by default (`OAK_EPISODE_STEPS=0`, never auto-reset — matching all CPU-era data), with gym 1000-step semantics available for external comparability and flagged in any SOTA claim.
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
- **`deaths.jsonl`** — one row per lane death event: `seed`, `lane`, `step`, `detector` (`envReward` | `policyMean` | `paramNonFinite` | `paramExplosion` | `kernelRho` | `envError`), `nfByPhase`, `rhoAtDeath`, `paramAbsMax`, `actorSupportEntropy`, `terminationRate`, `optionDurationMean`, `rewardSumAtDeath`, `lastGoodRewardRate`. A "dead" lane keeps stepping by default (one continual stream, never reset; recovery is data) — the certificate dates the first event. `OAK_FLEET_FREEZE_DEAD=1` opts into freezing certified lanes (gate tiers only; never for science runs where post-divergence behavior is the data).
- **`aggregate.jsonl`** — one row per heartbeat per policy: `step`, `alive`/`dead`, delivered-rate P10/P50/P90, `aggStepsPerSecond`, `wallSeconds`. This is the live dashboard (`tools/fleet_watch.py`).
- **`<envId>/seed<N>/`** — per-lane `actions.bin` + `meta.json` (+ optional `stats.bin`) for lanes selected by `OAK_RECORD_SEEDS=3,8` (or all lanes with `OAK_RECORD_ACTIONS=1`); `state-hashes.jsonl` when `OAK_DUMP_STATE_EVERY=N` is set (byte-identity validation).

**Tools** (numpy; `requirements.txt`):

- `python3 tools/fleet_stats.py <run-root>` — population report: Kaplan-Meier survival + RMST, divergence rate with Wilson 95% CI (overall + per detector), IQM delivered reward rate (`rewardSum/steps`) with bootstrap CI over ALL lanes (headline; never condition on survival) and survivors-only (diagnostic), plasticity quartiles (zenith step/rate, decay ratio), entropy-collapse rate. Markdown output pastes into `docs/roadmap.md`; also writes `summary.json`.
- `python3 tools/fleet_gate.py <keeper-root> <candidate-root>` — **the promotion law.** Same seed set + horizon required (hard error); all bootstraps paired by seed. PASS iff ALL four hold (AND-gate ⇒ each one-sided α=0.05, no multiplicity correction): (1) divergence non-inferiority, 95th pct of paired Δp ≤ δ (default 0.64/√N: 0.02 @ N=1024); (2) IQM delivered-rate non-inferiority, 5th pct of ΔIQM ≥ −5%·|IQM_keeper|, with a non-blocking IMPROVEMENT flag (what SOTA claims must cite); (3) RMST survival non-inferiority, 5th pct of ΔRMST ≥ −5%·horizon; (4) no new NF phases and per-phase totals ≤ max(2× keeper, 100·N/10). Exit 0 = PROMOTE, 1 = REJECT. Seed-3/seed-8 folklore is not encoded anywhere — those seeds are ordinary lanes (lane 3, lane 8 of every fleet) whose trajectories remain inspectable in `fleet.jsonl`.
- `python3 tools/fleet_watch.py <run-root> [--follow]` — live aggregate view.

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

Honest-comparison rules:

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
3. Promote only on a Tier-L `fleet_gate.py` PROMOTE against the keeper baseline. The gate is population law (divergence Δp, IQM delivered rate, RMST survival, NF phases — all paired by seed); a SOTA claim additionally requires the gate's IMPROVEMENT flag.
4. Per-seed collapse is population data, not an averaging input: `fleet_stats.py` reports it as divergence rate, survival curves, and detector counts. Individual seeds (3, 8, …) remain inspectable as lanes in every fleet, but no single seed blocks or licenses promotion.
5. Name run roots by hypothesis and tier, for example `.oak-benchmarks/closure-arbiter-tierM-YYYYMMDD`, and inspect live runs with `tools/fleet_watch.py --follow` rather than waiting for final rows.
6. Use subagents when the work can truly run in parallel: a Metrics Archaeologist mines benchmark roots, a Plan/Invariant Reviewer checks Alberta/FC-STOMP fit, a Mechanism Designer proposes closed-loop organs, a Benchmark Steward watches gates, and a Metal Auditor handles speed/FP-order risk. The main thread integrates; subagents do not define project doctrine.
7. Archive rejected mechanisms in `docs/roadmap.md`/`AGENTS.md` when they teach a reusable lesson. Keep the live code lean; diagnostics earn their place only when they shorten the next decision.

## CLI

| Command | Use |
|---|---|
| `OaK benchmark` | One fleet per policy: B independent (agent, env, seed) lanes in one process — a single batched kernel dispatch per step, env stepping sharded across CPU threads. `OAK_BENCH_ENV` selects the env (default `HalfCheetah-v4`); `OAK_BENCH_STEPS`, `OAK_BENCH_SEED_COUNT` (default 1024) / `OAK_BENCH_SEEDS` / `OAK_BENCH_SEED_START`, and `OAK_BENCH_POLICIES` define the workload. `OAK_FLEET_ENV_THREADS` caps env-step shards (default: all cores); `OAK_FLEET_TIMING=1` prints per-phase wall shares each heartbeat; `OAK_FLEET_FREEZE_DEAD=1` stops stepping certified-dead lanes (gate tiers only). One launcher owns the workspace at a time; nonempty requested roots get a fresh suffixed output dir. |

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
