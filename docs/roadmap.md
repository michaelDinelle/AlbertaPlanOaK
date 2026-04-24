# OaK roadmap - Plan-orthodox continual improvement

Last audited: 2026-04-24, during the L3-L7 integrated repair branch. L1 remains
rolled back. The live branch now includes RR stopping-value subtasks,
RR-derived continuation, average-reward option duration, age-gated Step-11
replacement, and feature-candidate diagnostics; promotion is still gated by
paired smoke, 1.5M screen, and 10M acceptance evidence.

This roadmap is the control document for moving OaK toward the Alberta Plan's
Step-11 Oak architecture. It is deliberately stricter than the old tier log:
claims are marked complete only when the mechanism matches the literature, not
merely when a nearby surrogate exists in the kernel.

Primary sources:

- Alberta Plan: `docs/papers/AlbertaPlanForAIResearch_2208.11173.pdf`
- Reward-respecting subtasks / STOMP: `docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf`
- Average-reward options: `docs/papers/WanNaikSutton_AvgRewardOptions_2110.13855.pdf`
- Option keyboard: `docs/papers/OptionKeyboard_CombiningSkills_2106.13105.pdf`
- Supporting current mechanisms: Reward Centering, MetaOptimize, GraMa,
  Deep Fourier Features, generate-and-test, and prioritized sweeping papers in
  `docs/papers/`.

## Audit Rules

Every roadmap entry must answer four questions.

| Question | Required standard |
|---|---|
| What does the Plan require? | Cite the Plan step or supporting paper, and distinguish the literal mechanism from an implementation convenience. |
| What does OaK actually implement? | Name the kernel/buffer/host path. No "close enough" status labels. |
| What does the data say? | Name the artifact directory and recompute summary statistics from JSON, not memory. |
| What is the next orthodox move? | One mechanism, one expected diagnostic, one acceptance gate. |

Terms used below:

- **Strong**: mechanism is present and matches the literature closely enough to
  serve as a foundation.
- **Partial**: useful substrate exists, but a literature-required part is absent
  or simplified.
- **Not implemented**: no live code path satisfies the requirement.
- **Neutral**: empirically safe but not a benchmark improvement.
- **Rollback**: mechanism should not be revived without a materially different
  formulation.

## Active L3-L7 Note

The previous roadmap sections below are retained as historical audit text where
they describe why L3-L7 was needed. For the current code, these items are no
longer "not implemented":

- L3: `oak_bk_encode` trains psi with the RR stopping value
  `V(s') - valueW[i]·x_i(s') + bonusW[i]·x_i(s')`.
- L4/L5: `oak_plan` learns reward, transition, and duration models using
  stopping-aware recurrences. Duration TD uses the raw duration estimate from
  the Bellman recurrence; the structural one-step lower bound is applied only
  when decoding duration for planning. Planning uses
  `Q(s,o) = rHat(s,o) - rho*lHat(s,o) + V(nHat(s,o))`.
- L6: `oak_post_learn` runs age-gated option/feature replacement with
  `optFeature`, feature age, option age, and feature-ever-used metadata.
- L7-lite: health includes `rrBonusMean`, `psiStopGapMean`,
  `terminationRate`, `optionDurationMean`, `neverUsedFeatureCount`,
  `replacementAgeMean`, and `stateEffectiveRank`.

Current empirical read: short paired HalfCheetah tests are finite and now beat
random comfortably, but the reward-rate still hovers near zero and does not yet
meet the sustained positive-acceleration bar.

## Alberta Plan Commitments

These are non-negotiable. A candidate that violates them is not a roadmap item.

| Commitment | Consequence for OaK |
|---|---|
| Experience only | The agent learns from observation, action, reward, and internal state derived from them. No env id, action type, object, geometry, or episodic labels inside the agent. |
| Temporal uniformity | No training phases, no eval phases, no warmup-gated learning, no frozen weights, no special exploration period. If a mechanism exists, it runs every step. |
| Compute is first-class | Reactive action must stay cheap; planning and learning run continually under a bounded per-step compute budget. |
| Continuing objective | Use average-reward / differential formulations from Step 5 onward. Discounted episodic semantics are out. |
| Big World discipline | No magnitude caps that impose a stationary scale prior. Preserve finite old values on non-finite proposals; do not clamp intelligence. |
| Bitter Lesson priority | Prefer learning, search, planning, and scalable resource allocation over hand-coded domain structure. |

## Current Evidence Ledger

The table below was recomputed from local JSON artifacts in
`.oak-benchmarks/*-10m` on 2026-04-23. Each run is 10 seeds x paired
`oak`/`random`, 10M measured steps per policy.

| Run | Mean lift | Median lift | Seeds > random | Chatter final med | Peak reward-rate max | Positive reward-rate HBs | EffK final med | SPS mean | NF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `halfcheetah-c1-10m` | `+668.1k` | `+678.3k` | `10/10` | `0.1697` | `+0.3097` | `6/500` | n/a | `1552` | `0` |
| `halfcheetah-d2-10m` | `+656.0k` | `+669.2k` | `10/10` | `0.1818` | `+0.1349` | `11/500` | n/a | `2544` | `0` |
| `halfcheetah-d3-10m` | `+620.7k` | `+629.4k` | `10/10` | `0.1172` | `+0.1636` | `16/500` | n/a | `2526` | `0` |
| `halfcheetah-f1-10m` | `+1.673M` | `+1.681M` | `10/10` | `0.0374` | `+0.1090` | `20/500` | n/a | `2530` | `0` |
| `halfcheetah-step11-10m` | `+1.397M` | `+1.483M` | `10/10` | `0.1404` | `+0.1252` | `12/500` | n/a | `2544` | `0` |
| `halfcheetah-h1-10m` | `+1.661M` | `+1.658M` | `10/10` | `0.0391` | `+0.1005` | `16/500` | `30.86` | `2507` | `0` |
| `halfcheetah-i1-10m` | `+1.596M` | `+1.593M` | `10/10` | `0.0288` | `+0.0850` | `11/500` | `31.42` | `2370` | `0` |
| `halfcheetah-k1-10m` | `+1.585M` | `+1.584M` | `10/10` | `0.0272` | `+0.1264` | `12/500` | `31.51` | `2370` | `0` |
| `halfcheetah-k2-10m` | `+1.601M` | `+1.604M` | `10/10` | `0.0273` | `+0.0776` | `7/500` | `31.48` | `2389` | `0` |

Post-F1 paired comparison:

| Run | Mean delta vs F1 | Seeds beating F1 | Read |
|---|---:|---:|---|
| `step11` | `-276.0k` | `1/10` | rollback |
| `h1` | `-12.4k` | `2/10` | neutral, closest to F1 |
| `i1` | `-77.4k` | `3/10` | neutral, smoother but slower |
| `k1` | `-88.0k` | `3/10` | rollback/non-winning |
| `k2` | `-72.8k` | `3/10` | neutral, best Step-2 variant on disk, not baseline |

K2 trajectory:

| Window | Mean incremental lift |
|---|---:|
| `0-2M` | `+321.3k` |
| `2-4M` | `+322.6k` |
| `4-6M` | `+320.1k` |
| `6-8M` | `+319.6k` |
| `8-10M` | `+316.9k` |

Interpretation: K2 is stable and low-chatter, but it does not improve the
learning slope. Its final `chordEffectiveK` stays near uniform (`31.48/32`),
so the stochastic compositional reseed did not activate differentiated
options.

L1 rollback evidence:

The L1 action-conditioned one-step model was compared heartbeat-by-heartbeat
against `.oak-benchmarks/halfcheetah-k1-15m`, 10 paired oak seeds, 30 heartbeats
per seed, 50k steps per heartbeat. Higher reward is less negative.

| Metric | K1 | L1 | L1 - K1 |
|---|---:|---:|---:|
| Final cumulative reward mean | `-212.785k` | `-219.198k` | `-6.412k` |
| Final cumulative reward median | `-214.166k` | `-220.359k` | `-4.284k` |
| Seeds L1 beats K1 | n/a | `2/10` | n/a |
| Final rewardRate mean | `-0.1050` | `-0.1157` | `-0.0107` |
| Final actionDeltaL2Mean | `0.02119` | `0.02111` | `-0.00008` |
| Final scoreGradL2Mean | `5.99777` | `5.99777` | `0.00000` |
| Final chordEffectiveK mean | `31.621` | `31.618` | `-0.003` |
| Final SPS mean | `2366.6` | `2515.3` | `+148.8` |
| Final agentStepMsMean | `0.3777` | `0.3515` | `-0.0262` |
| NF max | `0` | `0` | n/a |

L1 cumulative reward trajectory:

| Checkpoint | Mean L1 - K1 | Wins |
|---:|---:|---:|
| `50k` | `+283` | `7/10` |
| `250k` | `-4,746` | `2/10` |
| `500k` | `-5,856` | `2/10` |
| `1.0M` | `-5,920` | `2/10` |
| `1.5M` | `-6,412` | `2/10` |

Incremental windows show the same conclusion: L1 loses early, then runs near
parity but never earns back the gap. Window diffs were `-4,746` for heartbeats
`1-5`, `-1,110` for `6-10`, `+5` for `11-15`, `-69` for `16-20`, `-257` for
`21-25`, and `-235` for `26-30`.

The one-step diagnostics were finite and did improve
(`oneStepRewardTdAbsMean 0.3995 -> 0.3115`,
`oneStepStateTdL2Mean 6.9577 -> 6.4937`,
`oneStepValueWeightedModelErrorMean 0.0685 -> 0.0380`), but the learned model
did not translate into reward or option differentiation. `oneStepRawAdvantage`
also drifted more negative (`-0.0300 -> -0.0510`), which is a bad sign for
using this model as the next planning substrate.

Decision: rollback L1. Do not run the 10M acceptance. Do not start L2 from this
one-step model. The retained speed work is infrastructure only: fused
`oak_bk_encode(+meta_predict)`, fused `oak_post_learn(+G&T commit)`,
`optRewardTd` precompute in `oak_plan`, and unretained command buffers.
Post-rollback hot smoke `.oak-benchmarks/l1-rollback-hot` ran `10k` steps at
`3945.7` SPS, `0.2259` agent ms, and NF zero with no `oneStep` health bucket.

## Current Priority List

This is the active priority order. Do not skip downward because a later item is
more interesting; the Plan order matters.

| Priority | Item | Status now | Why it comes here | Next decision |
|---:|---|---|---|---|
| 0 | Measurement hygiene | Active rule | A stale Metal library can completely corrupt health metrics. | Rebuild Metal after every kernel edit; never interpret a run whose metallib predates `Sources/OakKernel.metal`. |
| 1 | L1 one-step expectation model | **Rolled back** | The model learned finite predictions but reduced K1 reward on `8/10` seeds and did not create a usable planning signal. | Do not run 10M; any Step-8 retry must be materially different and isolated from feature-utility/reward-path damage. |
| 2 | L2 state-feature ring + asynchronous one-step planning | **Blocked** | L2 depends on a one-step model worth spending through search. L1 failed that screen. | Do not start from rolled-back L1. |
| 3 | L3 exact reward-respecting stopping value for psi | **Live, screening** | The RR stopping-value target is implemented in `oak_bk_encode`. | Keep only if paired screens show finite health and improving reward-rate. |
| 4 | L4 average-reward option duration + differential option backup | **Live, screening** | `optDuration` and `rHat - rho*lHat + V(nHat)` are implemented. | Duration must remain non-degenerate without suppressing reward improvement. |
| 5 | L5 true option termination | **Live, screening** | Continuation is derived from the RR stopping comparison. | Reject variants that revert to H1-style confidence persistence. |
| 6 | L6 Step-11 registry and age-aware replacement | **Live, screening** | `optFeature`, ages, and feature-ever-used metadata are implemented. | Replacement counts must stay structural and no longer equal measured steps. |
| 7 | L7 feature diagnostics/candidate pool | **Partial live** | Unused active features serve as candidates; cheap state participation rank is reported. | Add richer decorrelation only if it is needed by the next failing diagnostic. |
| 8 | L8 eligibility traces retry | Blocked on L7 evidence | Prior trace attempts failed from correlated Fourier features. | Retry only after effective-rank/stability diagnostics justify it. |
| 9 | L9 option keyboard descriptor model / Design B | Last | Descriptor models are only meaningful after true options exist. | Compare Design A/B after L5. |

## Strict Status Against Plan Steps

| Plan step | Status | Current implementation | Strict audit |
|---|---|---|---|
| 1. Representation I | **Partial/strong substrate** | Welford observation normalization, per-feature alpha via MetaOptimize, GraMa utility, continual updates. | The core Step-1 pieces are real. The base-agent text also mentions short-term credit-assignment memories; TD(lambda) attempts diverged on current correlated Fourier features, so traces are an open gap, not a completed item. |
| 2. Representation II | **Partial** | Deep Fourier features plus G&T reseeding; K2 composes a replacement row from two utility-sampled parent rows. | Step 2 is not just random projection replacement. It asks for smart construction, testing, ranking, and replacement of new features made from existing features in a multi-task setting. K2 installs a narrow composition surrogate but is empirically neutral. |
| 3. Prediction I | **Partial** | Main value plus K psi GVFs; multi-timescale obs traces in perception. | Continual GVF learning exists. Off-policy, stable multi-step prediction with learned feature finding and richer recurrent perception is incomplete. |
| 4. Control I | **Strong, with one caveat** | Continual actor-critic via compatible score gradients and chord-blended option policies; Box actions expose mean/log-sigma policy coordinates in the wrapper. | The actor-critic loop is useful and stable. The caveat is that action parameterization is wrapper-side research machinery, not a Plan step; keep only variants that survive paired baselines. |
| 5. Prediction II | **Partial/active** | Average reward centering for main TD and psi TD; L3-L7 adds average-reward option duration. | Differential value learning is present. Duration is live but still under reward-rate screening. |
| 6. Control II | **Strong** | Continuing wrappers; `done` is not plumbed into the engine; no episodic agent semantics. | This matches the Plan discipline. Benchmark resets are wrapper events only. |
| 7. Planning I | **Partial/active** | `oak_plan` computes differential average-reward backups over K stopping-aware option models for the current state. | OaK plans every step, but still lacks a state/update queue and model uncertainty. |
| 8. Prototype-AI I | **Not implemented as a reward-useful model** | L1's action-conditioned one-step model was implemented and then rolled back after the 1.5M screen. Live code has the base option-model scaffolds, not an explicit `[state, action] -> next state/reward` Step-8 model. | Step 8 remains a real gap. The failed L1 formulation learned finite predictions but pushed reward farther from K1, so the next attempt must not simply re-add the same heads/utility coupling. |
| 9. Planning II | **Partial** | Priority signal from chord-weighted option-model TD magnitude; dense transition update gate. | Prioritized sweeping lineage is present. The Plan also calls for uncertainty-aware control of planning order; OaK still has no model-uncertainty estimate and no state/update queue. |
| 10. STOMP | **Active, not promoted** | Feature-indexed RR psi heads, option policies, stopping-aware reward/transition/duration heads, planning over K heads, RR-derived continuation. | The STOMP pieces are live; the open question is whether they now produce sustained positive reward-rate. |
| 11. Oak | **Active, incomplete** | Utility estimates, age-gated option/feature replacement, feature registry metadata, and chord-blended policy keyboard. | Step 11 feedback is no longer unconditional churn, but richer utility assessment and full chord-descriptor option-keyboard modeling remain open. |
| 12. IA | Out of scope | None. | Keep out of this roadmap until Step 11 is credible. |

## Overclaims Removed

The following pre-L3-L7 claims were removed. They remain here only as a guard
against resurrecting stale descriptions.

| Old claim | Corrected statement |
|---|---|
| "Prototype-AI I is strong." | False. L1 showed that the naive action-conditioned one-step model learns finite predictions but harms reward; live code does not currently have a promoted Step-8 model. |
| "STOMP is present." | Pre-L3-L7 this was only a scaffold. Current L3-L7 has exact RR stopping, RR-derived continuation, and stopping-aware option models, but it is not promoted until reward-rate evidence clears the gates. |
| "H1 closes Step 10." | False. H1 is a chord-persistence mechanism. It may be useful, but it is not an option termination condition as defined in the options/STOMP literature. |
| "psi implements RR Eq. 4." | Pre-L3-L7 this was false. Current L3-L7 uses the RR stopping value with an optimistic per-feature bonus; the remaining question is empirical promotion. |
| "optReward/optTransition are full RR option models." | Pre-L3-L7 they were one-step surrogates. Current L3-L7 learns stopping-aware reward, transition, and duration heads; richer uncertainty/search-control is still open. |
| "Step 11 feedback is implemented." | Pre-L3-L7 replacement was unconditional churn. Current L3-L7 is age-gated with registry metadata; richer utility assessment remains open. |
| "K2 is in flight." | K2 has completed 10M characterization: neutral, not a new baseline. |

## Orthodoxy Gaps Ranked By Leverage

### 1. Complete and characterize Plan Step 8

Plan Step 8 comes before temporal abstraction for a reason: the agent needs a
real one-step model before option models can be trusted.

Current state:

- L1 attempted a wrapper-produced action signal plus `oneReward` and
  `oneTransition` heads over `[state, action]`.
- The model diagnostics improved and NF stayed zero, but paired K1 heartbeats
  showed `-6.412k` mean reward at 1.5M and only `2/10` seeds beating K1.
- The live code has rolled this back. No `prevActionSignal`, `oneReward`,
  `oneTransition`, one-step diagnostics, or one-step feature-utility term remain.

Remaining gap:

- Planning uses only the current state-feature vector and option-model
  scaffolds, not a controlled stream of action-conditioned one-step predictions.
- Model uncertainty/search-control is not yet represented.
- The failed L1 formulation suggests that bolting a learned model onto feature
  replacement can move the system farther from the reward objective even when
  the model loss improves.

Next orthodox mechanism:

- Do not start L2 from the rolled-back L1 heads.
- A Step-8 retry must first isolate the model from replacement utility and
  reward-path updates, or use an uncertainty/search-control formulation that
  can be measured without perturbing the policy/value loop.
- The first acceptance screen for any retry is K1/F1 parity at 1.5M, not a
  10M acceptance run.

Diagnostics required:

- one-step reward TD, one-step state TD L2, value-weighted model error,
  planning source mix if planning is active, model-feature utility contribution
  if any, and per-step planning backup count.

Acceptance:

- 10k smoke: NF zero, model TD finite, no action/chord collapse.
- 10x1.5M screen: beat K1 in at least `5/10` paired seeds and no worse than F1
  by more than 3% mean lift, NF zero, no worse than F1 final chatter by more
  than 2x.
- 10x10M acceptance: mean lift >= F1, at least 5/10 seeds beat F1, and last-2M
  incremental lift >= F1 last-2M. If mean lift ties but last-2M improves across
  at least 7/10 seeds, keep as a velocity win.

### 2. Make reward-respecting subtasks exact before more option tricks

Current gap:

- `tdPsi[k] = reward_c + (1-beta_k)*curr[k] + beta_k*psiNext - psiPrev`.
- That is a useful feature-attainment surrogate, but it is not the RR subtask
  stopping-value definition.

Next orthodox mechanism:

- Track a structurally derived optimistic bonus weight per selected feature.
  It must come from the main value head's own history/utility, not from an
  external knob.
- Define stopping value for feature `i` as the main value estimate with the
  selected feature's weight replaced by its bonus weight, matching the
  reward-respecting feature-attainment construction.
- Learn psi toward that stopping value while preserving average-reward
  centering and continual updates.

Diagnostics required:

- bonus-weight tracker value, stopping bonus magnitude, psi TD distribution,
  selected feature identity, and correlation between stopping bonus and later
  planning usefulness.

Acceptance:

- Must not reduce 10M mean lift below F1 by more than 10%.
- Must improve at least one STOMP-specific diagnostic: lower effective chord
  uniformity, higher option identity specialization, or stronger correlation
  between selected feature utility and planning advantage.

### 3. Add true termination and average-reward option models

Current gap:

- H1 smooths or persists the chord, but does not define an option termination
  condition.
- Current models predict next state/reward under one-step responsibility, not
  cumulative reward and state at option termination.
- Average-reward option models need duration; OaK has no `optDuration` head.

Next orthodox mechanism:

- Add an explicit continuation/termination variable per option derived from
  the exact subtask stopping comparison, not from the exhausted psi/backward
  value shortcut family.
- Add `optDuration[K*R]` or an equivalent capacity-derived duration model.
- Plan with differential option backup:
  `Q(s,o) = r_hat(s,o) - rho_bar * l_hat(s,o) + V(n_hat(s,o))`.

Diagnostics required:

- mean option duration, duration TD, termination rate, option-model reward TD,
  option-model transition TD, and planning advantage by option.

Acceptance:

- Option duration must be measurably non-degenerate in successful seeds
  (reported, not hard-gated).
- 10M mean lift >= H1 and no worse than F1 by more than 3%.
- If duration is non-degenerate but lift is neutral, keep only if last-2M
  learning velocity improves over H1.

### 4. Repair Step 11 feedback: "eventually" is not "every step"

Current gap:

- `oak_bk_admit_commit` always replaces the minimum-utility option and feature
  every step.
- J1 mean/sigma gates failed because the minimum of K utilities is an
  extreme-value statistic.
- There is no feature registry and no downstream accounting of which feature
  has been used as a subtask basis.

Next orthodox mechanism:

- Add age/tenure and "ever used as subtask basis" metadata for features and
  options.
- Replacement priority must combine utility, age, and novelty without a
  tunable phase schedule.
- Prefer replacing what has had enough continual opportunity to prove useful.
  This is still temporally uniform if the assessment runs every step.

Diagnostics required:

- replacement age distribution, never-used feature count, repeated-feature
  reuse rate, utility-at-replacement percentile, and option survival curves.

Acceptance:

- Reinit count must no longer equal step count.
- No NF and no catastrophic loss at 1.5M.
- At 10M, either beat F1 mean lift or improve last-2M lift slope with lower
  churn and non-uniform option identity.

### 5. Feature finding and decorrelation before eligibility traces

Current gap:

- K1 and K2 are feature-combination attempts, but neither creates a benchmark
  win or differentiated option keyboard.
- TD(lambda) diverged because correlated Fourier features compound same-sign
  traces under Plan-structural step sizes.

Next orthodox mechanism:

- Treat Step 2 as a resource-allocation problem, not only a reseed formula.
- Add feature decorrelation/isometry diagnostics over the active state vector:
  feature covariance, pair cosine, effective rank, and trace stability margin.
- Candidate feature generation should include combinations of existing useful
  features, but selection must preserve diversity and downstream utility.

Diagnostics required:

- feature covariance/effective rank, parent reuse entropy, candidate survival,
  utility transfer to value/model/policy heads, and trace stability margin.

Acceptance:

- Do not reattempt TD(lambda) until feature effective rank improves and the
  trace stability margin is finite across all seeds.
- A Step-2 mechanism must beat F1 at 10M or produce a clear enabling diagnostic
  for traces/options without degrading mean lift more than 3%.

### 6. Search control with uncertainty, not just TD magnitude

Current gap:

- D2 priority is model TD magnitude weighted by chord.
- The Plan explicitly points at search control that accounts for model
  uncertainty.

Next orthodox mechanism:

- Track uncertainty or variance of model TD error for one-step and option
  models.
- Use it to choose which state/model/option backups receive the fixed planning
  budget each step.

Diagnostics required:

- uncertainty mean/variance, backup priority entropy, selected backup source,
  and planning value improvement per backup.

Acceptance:

- Must improve planning efficiency diagnostics first; reward lift is required
  at 10M before replacing F1/H1 as baseline.

## Next Implementation Sequence

The next work should stop adding nearby surrogates to STOMP and repair the
pieces whose measured signals actually move the reward objective. L1 is no
longer in the sequence.

| Order | Tier | Mechanism | Status | Promotion gate |
|---:|---|---|---|---|
| 1 | L3 | Exact RR stopping value for psi subtasks | Next STOMP repair. | STOMP diagnostics improve without >10% 10M regression. |
| 2 | L4 | Average-reward option duration head and differential option backup | Blocked on L3. | H1 parity and non-degenerate duration diagnostics. |
| 3 | L5 | True option termination from stopping comparison | Blocked on L3/L4. | >= F1 mean lift or >= H1 lift plus improved last-2M velocity. |
| 4 | L6 | Step-11 feature/option registry and age-aware utility replacement | Blocked on better utility semantics. | Churn falls, utility rises, no 10M regression. |
| 5 | L7 | Feature decorrelation + candidate feature pool | Enabling substrate. | Effective-rank gain and F1 parity. |
| 6 | L8 | Reattempt eligibility traces on decorrelated features | Blocked on L7 diagnostics. | 10 seeds stable, no trace blow-up, 10M lift/velocity win. |
| 7 | L9 | Option keyboard descriptor model / Design B comparison | Last. | Beat Design A or produce cleaner planning diagnostics. |
| parked | Step-8 retry | Action-conditioned one-step model, redesigned | Rolled back until a materially different design exists. | Must beat K1 at 1.5M before any 10M run. |

## L1 Decision

The L1 decision is closed.

```text
L1 paired evidence
|
+-- NF zero and diagnostics improved?
|      yes
|
+-- Reward trajectory beats K1 or shows late velocity?
|      no: final mean -6.412k vs K1, 2/10 seed wins,
|          last 500k mean -492, 2/10 wins.
|
+-- Continue to 10M?
|      no. Roll back the algorithmic model path.
|
+-- Keep SPS work?
       yes. Keep dispatch fusion, optRewardTd precompute, and command-buffer
       allocation improvements because they are infrastructure, not the
       failed Step-8 mechanism.
```

## Exhausted Families

Do not revive these without a materially different mechanism.

| Family | Attempts | Failure signature |
|---|---|---|
| Termination from psi/backward value | C2 v1, C2 v2, C2 retry, C4, Step-11 planUtil | Myopic or unstable options; -30% to -99% regressions, or 10M -16% with high chatter. |
| Recurrent encoder feedback into Fourier input | C3 v1/v2 | Around -52% at 1.5M; double non-stationarity in TD bootstrap. |
| TD(lambda) on current Fourier features | G1 normalized, G1 true-online | One or more seeds diverge catastrophically at Plan-structural alpha/lambda. |
| Central-tendency G&T gates | J1 mean and sigma gates | Replacement still fires every step because argmin is an extreme-value statistic. |
| Greedy compositional reseed | K1 | Safe but non-winning: 10M -88.0k vs F1, EffK uniform. |
| Stochastic utility compositional reseed alone | K2 | Safe but neutral: 10M -72.8k vs F1, EffK uniform. Keep only as part of a fuller Step-2 system. |
| Naive action-conditioned one-step model | L1 | Numerically clean and model TD decreased, but 1.5M reward moved away from K1: mean `-6.412k`, median `-4.284k`, `2/10` seed wins. Do not revive without a materially different design. |

## Benchmark Protocol

The benchmark protocol is part of the roadmap because false positives have
repeatedly come from short runs and peak metrics.

| Stage | Purpose | Gate |
|---|---|---|
| 10k smoke | Numerical and semantic sanity. | NF zero; no action, value, model, or trace explosion; diagnostics populated. |
| 10 x 1.5M paired | Cheap screen only. | Mean lift sane, no negative-seed cluster, no obvious chatter explosion. Never use this alone for acceptance. |
| 10 x 10M paired | Acceptance. | Mean lift, per-seed lift, last-2M incremental lift, NF, and mechanism-specific diagnostics. |
| 10 x 15M paired | Tie-breaker. | Use only when 10M mean lift and last-window velocity disagree. |

Primary acceptance metric:

- mean paired lift vs random over all 10 seeds.

Secondary acceptance metrics:

- number of seeds beating the prior baseline,
- last-2M incremental lift,
- NF zero,
- final and heartbeat chatter,
- mechanism-specific diagnostics.

Avoid using max peak reward-rate as a promotion criterion. It is useful for
storytelling and failure analysis, but it has repeatedly misread the main
learning signal.

Kernel build hygiene:

- `swift build` does not rebuild `Sources/OakKernel.metal`; `Package.swift`
  excludes the Metal source.
- After every kernel edit, run:
  `xcrun -sdk macosx metal -fno-fast-math -c Sources/OakKernel.metal -o .build/OakKernel.air`
  and `xcrun -sdk macosx metallib .build/OakKernel.air -o .build/OakKernel.metallib`.
- Before interpreting any smoke or paired run, verify `.build/OakKernel.metallib`
  is newer than `Sources/OakKernel.metal`.

## Optimization Contract

Kernel optimizations are allowed only when they do not change the agent's
intelligence by accident.

| Optimization type | Requirement |
|---|---|
| Bit-identical | Smoke plus 1.5M sanity bench. Preserve the original `fma` sequence when reductions feed agent state. |
| FP reorder | 10-seed paired proof. Aggregate parity is insufficient; per-seed lift and chatter must stay within the accepted tolerance. |
| New algorithmic reduction | Treat as algorithmic, not a perf patch. Needs the full mechanism benchmark. |

Compute improvements matter because the Plan treats compute as first-class,
but speed is infrastructure. It cannot promote a non-orthodox mechanism.

## Current Baseline Decision

Baseline for the next algorithmic change:

- **F1/H1 substrate**, not K2.
- F1 remains the reward-lift leader.
- H1 is close enough and architecturally useful as a persistence diagnostic,
  but it must not be described as true Step-10 termination.
- K2 is neutral evidence for Step-2 composition, not a baseline.

The roadmap objective is now sharper: repair the missing Plan-order pieces
that prevent continuously improving learning, especially exact subtask/option
machinery and any future Step-8 model design that can prove reward relevance
before it is allowed into planning. More feature churn without those repairs is
unlikely to break the current post-F1 slope.
