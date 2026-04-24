# OaK roadmap - Plan-orthodox continual improvement

Last audited: 2026-04-24, during the research-first L3-L7 regroup. L1 remains
rolled back. The live branch includes RR stopping-value subtasks, RR-derived
continuation, raw option reward plus average-reward option duration,
age-gated Step-11 replacement, every-step replacement assessment, and
feature/search-control diagnostics; promotion is still gated by paired smoke,
1.5M screen, and 10M acceptance evidence.

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
  `V(s') - valueW[i]·x_i(s') + bonusW[i]·x_i(s')`. Each ψ GVF now
  has its own chord-responsibility reward-rate trace instead of sharing only
  the behavior-policy `rho`.
- L4/L5: `oak_plan` learns raw reward, transition, and duration models using
  stopping-aware recurrences. Duration TD uses the raw duration estimate from
  the Bellman recurrence; the structural one-step lower bound is applied only
  when decoding duration for planning. Planning uses the coherent
  average-reward option backup
  `Q(s,o) = rHat(s,o) - rho*lHat(s,o) + V(nHat(s,o))`. A coherent
  differential reward-model alternative
  `rDiff <- (reward-rho)+continue*rDiffNext-rDiffPrev`, `Q=rDiff+V(nHat)`
  was tested in `l3-l7-diffreward-paired-10k-225607` and rejected: NF zero and
  10/10 random wins, but mean reward-rate `-0.003614`, only `2/10` positive
  seeds, and no last-5k monotonic acceleration.
- L6: `oak_post_learn` runs every-step replacement assessment and age-gated
  option/feature commits with `optFeature`, feature age, option age, and
  feature-ever-used metadata.
- L7-lite: health includes `rrBonusMean`, `psiStopGapMean`,
  `terminationRate`, `optionDurationMean`, `neverUsedFeatureCount`,
  `replacementAgeMean`, `stateEffectiveRank`, `eligibleOptionCount`,
  `eligibleFeatureCount`, `replacementCommitRate`, `modelUpdateRate`,
  `modelPriorityEntropy`, and `optionDiffTdMean`.

Current empirical read: `l3-l7-feature-slowclock-paired-300k-232452` is finite,
beats random `10/10`, and improves the 300k mean reward sum from the rebuilt
keeper's `-711.154` to `-489.611` while cutting feature reinitializations from
`4717.7` to `149.5`. It is not promoted: final reward-rate is still near zero
(`-0.001257` mean, `5/10` positive seeds) and late heartbeats still show
negative drift rather than sustained acceleration.

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
| 4 | L4 average-reward option duration + raw reward backup | **Live, screening** | `optDuration` and `rHat - rho*lHat + V(nHat)` are implemented; the direct differential reward-model branch was rejected at 10k. | Duration must remain non-degenerate without suppressing reward improvement. |
| 5 | L5 true option termination | **Live, screening** | Continuation is derived from the RR stopping comparison. | Reject variants that revert to H1-style confidence persistence. |
| 6 | L6 Step-11 registry and age-aware replacement | **Live, screening** | `optFeature`, ages, feature-ever-used metadata, every-step eligible-candidate assessment, and age/capacity-gated commits are implemented. | Replacement counts must stay structural and no longer equal measured steps. |
| 7 | L7 feature/search diagnostics | **Partial live** | Unused active features serve as candidates; cheap state participation rank, eligible counts, commit rate, search-control entropy, model update rate, and option differential TD are reported. | Add richer decorrelation/uncertainty only if these diagnostics identify starvation or rank collapse. |
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

### 2. Reward-respecting subtasks are live; promote only with reward evidence

Current status:

- `oak_bk_encode` now computes RR stopping values for each option's registered
  feature:
  `z_i(s') = V(s') - valueW[i]*x_i(s') + bonusW[i]*x_i(s')`.
- Continuation is derived from the RR comparison, and psi cumulants use
  per-option chord-responsibility reward centering.

Next orthodox mechanism:

- Do not add another nearby stopping surrogate. The next move is diagnostic:
  determine whether feature utility, stopping gap, or chord concentration is
  failing before changing the subtask definition again.

Diagnostics required:

- bonus-weight tracker value, stopping bonus magnitude, psi TD distribution,
  selected feature identity, and correlation between stopping bonus and later
  planning usefulness.

Acceptance:

- Must not reduce 10M mean lift below F1 by more than 10%.
- Must improve at least one STOMP-specific diagnostic: lower effective chord
  uniformity, higher option identity specialization, or stronger correlation
  between selected feature utility and planning advantage.

### 3. True termination and average-reward option models are live

Current status:

- `continuation[k]` is derived from the RR stopping comparison, not H1
  confidence persistence.
- `optReward`, `optTransition`, and `optDuration` use stopping-aware recursive
  TD targets. The live average-reward backup is
  `Q(s,o) = r_hat(s,o) - rho_bar*l_hat(s,o) + V(n_hat(s,o))`.
- The coherent differential-reward-model alternative was tested and rejected
  at 10k; raw reward plus duration remains the keeper formulation because it
  is both literature-valid and empirically better on the paired gate.

Next orthodox mechanism:

- Use `optionDiffTdMean`, `optionDurationMean`, `terminationRate`,
  `modelUpdateRate`, and `modelPriorityEntropy` to decide whether the next
  change belongs in search control, utility assessment, or representation.

Diagnostics required:

- mean option duration, duration TD, termination rate, option-model reward TD,
  option-model transition TD, and planning advantage by option.

Acceptance:

- Option duration must be measurably non-degenerate in successful seeds
  (reported, not hard-gated).
- 10M mean lift >= H1 and no worse than F1 by more than 3%.
- If duration is non-degenerate but lift is neutral, keep only if last-2M
  learning velocity improves over H1.

### 4. Step 11 feedback: assess every step, commit by capacity

Current status:

- `oak_bk_admit_commit` now assesses worst eligible option and feature
  candidates every step.
- Actual replacement remains structural and capacity/age gated; reinit counts
  no longer equal measured steps.
- The feature registry and downstream subtask accounting are live via
  `optFeature`, `featureAge`, `optAge`, and `featureEverUsed`.

Next orthodox mechanism:

- Improve utility only from diagnostics that show a real failure. The current
  open question is not "does replacement run?", but whether low utility,
  search priority, and never-used feature status are the right replacement
  sufficient statistics.

Diagnostics required:

- replacement age distribution, never-used feature count, repeated-feature
  reuse rate, utility-at-replacement percentile, option survival curves,
  eligible option/feature counts, and replacement commit rate.

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
- OaK now reports `modelPriorityEntropy`, `modelUpdateRate`, and
  `optionDiffTdMean`, but it does not yet learn uncertainty or use
  meta-gradient search control.
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
| 1 | L3 | Exact RR stopping value for psi subtasks | Live, screening. | STOMP diagnostics improve without >10% 10M regression. |
| 2 | L4 | Average-reward option duration with raw reward backup | Live, screening. Differential reward-model branch rejected at 10k. | H1 parity and non-degenerate duration diagnostics. |
| 3 | L5 | True option termination from stopping comparison | Live, screening. | >= F1 mean lift or >= H1 lift plus improved last-2M velocity. |
| 4 | L6 | Step-11 feature/option registry and every-step assessment | Live, screening. | Churn falls, utility rises, no 10M regression. |
| 5 | L7 | Feature/search diagnostics and candidate feature pool | Partial live. | Diagnose rank/search starvation before adding heavier covariance or uncertainty heads. |
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

### Rejected By Paired Evidence

These are recent L3-L7 regroup variants that should not be rediscovered by
future audits as "obvious" fixes.

| Variant family | Evidence | Decision |
|---|---|---|
| Differential option reward model | `l3-l7-diffreward-paired-10k-225607`: NF zero and Oak beat random `10/10`, but mean reward-rate `-0.003614`, only `2/10` positive reward-rate seeds, no monotonic last-5k acceleration, and worse cumulative reward than the raw-reward keeper (`-23.381` vs `-15.588`). | Rejected; keep raw reward plus duration backup. |
| All-option model updates / raw plan advantage / current-Q keyboard / direct TD actor / meta-psi credit | 2026-04-23 rapid screens all failed 10k or 50k paired gates despite NF health; signatures were lower reward sum, fewer positive reward-rate seeds, or chatter/action damage. | Do not revive without a new literature-backed search-control or actor-critic mechanism. |
| Direct main-TD option actor | `l3-l7-step4-reward-actor-paired-10k-042108`: restoring the actor update to the main value TD was NF-clean but fell to reward sum `-80.608`, reward-rate `-0.009061`, and only `1/10` positive-rate seeds. | Rejected; keep the live RR subtask actor unless a fuller actor-critic mechanism changes the credit signal. |
| Separate additive primary reward actor | `l3-l7-primary-reward-actor-paired-10k-044233`: adding a standalone `D_policy*R` primary actor updated by main TD over-drove actions (`actL2 2.143`), exploded score gradients (`1.275e11`), collapsed duration to one, and lost to random `0/10`. | Rejected; the missing primary-policy concern needs a stable actor-critic formulation, not a raw additive TD actor. |
| Post-plan/current-Q keyboard | `l3-l7-postplan-keyboard-slowclock-paired-10k-042522`: using current planned option Q for the chord was NF-clean but missed the 10k gate with reward sum `-20.410`, reward-rate `-0.003470`, only `1/10` positive-rate seeds, and over-concentrated chords (`EffK 7.54`). | Rejected; do not move the keyboard to current-Q ordering without a real chord-descriptor model. |
| Option-policy inheritance on replacement | `l3-l7-option-policy-inherit-paired-10k-042848`: copying policy from the most-used option worsened reward sum to `-20.802`, drove termination to `0.0906`, and inflated duration to `192.6`. | Rejected; option birth remains a clean policy reset. |
| Actor-policy eligibility trace only | `l3-l7-policy-eligibility-trace-paired-10k-043219`: Oak lost to random `0/10` with mean reward `-6605.465` versus random `-2658.035`; action/score gradients exploded while duration collapsed to one step. | Rejected; do not add actor traces without a complete, stable actor-critic trace formulation. |
| RR stopping-bonus feature utility | `l3-l7-rrbonus-feature-utility-paired-10k-043925`: adding the RR bonus gap into registry feature utility was NF-clean and beat random, but reward sum regressed to `-22.907`, reward-rate was negative in all seeds, and option duration inflated to `157.7`. | Rejected; keep RR bonus in stopping values/diagnostics, not as a direct replacement score. |
| Unit-norm Gaussian exploration scale | `l3-l7-unitnorm-sigma-paired-10k-044454`: changing continuous-action `sigmaBase` from `1/D_policy` to `1/sqrt(D_policy)` was NF-clean but raised control cost, regressing reward sum to `-425.529` with only `2/10` positive-rate seeds. | Rejected; the plateau is not fixed by larger initial exploration alone. |
| Priority floor for option-model search control | `l3-l7-priority-floor-current-paired-10k-044909`: using `max(chord,1/K)` in the priority signal was NF-clean but regressed reward sum to `-22.661`, left only `1/10` final positive-rate seeds, and introduced early score spikes. | Rejected; keep the pure chord-weighted priority gate until a richer uncertainty mechanism is added. |
| Pre-update reward-rate TD centering | `l3-l7-preupdate-rho-td-paired-10k-045035`: using the previous `rho`/`rhoOpt` in the same step's value/psi TD errors was research-clean and NF-clean, but regressed reward sum to `-21.363`, left only `2/10` final positive-rate seeds, and inflated option duration to `72.8`. | Rejected; keep the current update-first centering until a fuller value-based reward-centering mechanism wins paired gates. |
| Fixed-sigma continuous policy | `l3-l7-fixed-sigma-paired-10k-045540`: freezing the Gaussian scale coordinates removed score-gradient spikes (`3.46e-3`) but starved the actor, regressing reward sum to `-115.259` with only `1/10` positive-rate seeds. | Rejected; keep adaptive log-sigma unless a different bounded continuous-policy formulation is justified and wins paired gates. |
| Source-chord option-model priority | `l3-l7-source-priority-paired-10k-045727`: weighting model priority by `chordPrev` instead of the current chord was NF-clean but regressed reward sum to `-25.159`, left only `1/10` positive-rate seeds, and inflated duration to `128.8`. | Rejected; keep current-chord priority until uncertainty/meta-gradient search control replaces the simple TD-priority gate. |
| Tanh-squashed Box policy | `l3-l7-tanh-box-paired-10k-045950`: mapping continuous Gaussian samples smoothly into Box bounds was wrapper-clean and NF-clean, but regressed reward sum to `-25.538`, inflated duration to `86.0`, and reduced the random baseline's action energy without improving Oak acceleration. | Rejected; keep the current clamp wrapper until a complete bounded-policy score formulation wins paired gates. |
| Gaussian log-scale natural Fisher factor | `l3-l7-logsigma-natural-half-paired-10k-051123`: changing the Box log-scale score to the exact natural-gradient `0.5*(z^2-1)` was NF-clean and reduced score magnitude, but reward regressed to `-41.760`, final reward-rate stayed negative, and option duration inflated to `34.8`. | Rejected; scale-drift repair needs more than shrinking the log-scale score. |
| Fixed F1-style Box actor on L3-L7 | `l3-l7-fixed-f1-box-paired-10k-051306`: reverting Box actions to fixed `sigma=1/sqrt(A)` and `D_policy=A` was NF-clean and beat random, but reward collapsed to `-2028.958`, mean reward-rate `-0.183469`, and action L2 `1.03`. | Rejected for current L3-L7; the old F1 actor cannot be transplanted without rebuilding the surrounding control objective. |
| Reserved primary option 0 | `l3-l7-primary-option0-paired-10k-051503`: making option 0 a non-replaceable one-step primary reward actor/model was NF-clean and beat random, but regressed reward to `-29.009`, concentrated the keyboard (`EffK 7.64`), and inflated duration to `140.8`. | Rejected; the base-policy gap is real, but this simple reservation is not the right integration. |
| Capacity-only rank changes | `l3-l7-keeper-r128-paired-10k-051732` raised effective rank but regressed reward to `-88.967`, concentrated the keyboard, and created score spikes; `l3-l7-keeper-r32-paired-10k-051808` improved speed but regressed reward to `-29.235` with no positive-rate seeds. | Rejected as a standalone fix; the plateau is not just too little or too much `R`. |
| Chord/duration-weighted option utility | `l3-l7-chordweighted-option-utility-paired-50k-051937` improved 50k reward sum to `-66.474` but failed the gate with `0/10` positive-rate seeds, near-never termination (`0.05`), and duration `591.9`; `l3-l7-durationweighted-option-utility-paired-10k-052223` still regressed 10k and left duration high. | Rejected for promotion; keep the evidence that option utility needs behavior responsibility without letting duration collapse. |
| Hard option interruption | `l3-l7-option-interrupt-utility-paired-10k-052129`: forcing continuation only for the current best-Q option fixed duration (`1.04`) but collapsed to all-stop termination (`0.997`) and reward `-68.446`. | Rejected; option interruption must be softer than hard best-option gating. |
| Standard Gaussian mean score | `l3-l7-standard-gaussian-mean-score-paired-10k-052400`: using the raw Williams mean score `z/sigma` was NF-clean but made score magnitude `80x`, raised action L2 to `0.273`, and regressed reward to `-107.868`. | Rejected; the current variance-scaled mean score remains the stable actor interface. |
| TD-error reward-rate update | `l3-l7-td-rho-paired-10k-052532`: updating `rho` from the main TD error instead of the reward EMA produced more positive-rate seeds (`5/10`) but regressed reward to `-34.753` and did not accelerate. | Rejected; keep simple reward-centering EMA unless a full differential-control branch wins paired gates. |
| Naive intra-option Q keyboard | `l3-l7-optvalue-keyboard-paired-10k-050453`: adding an average-reward `Q(s,o)` head and using it directly as the keyboard logit was NF-clean but made the chord too uniform, exploded score gradients in one seed, regressed reward sum to `-389.147`, and lost to random in `1/10` seeds. | Rejected; an option-value head remains a Plan gap, but it must not replace model-backed keyboard logits before it has a stable integration path. |
| Pure RR-activation keyboard on current keeper | `l3-l7-activation-keyboard-current-paired-10k-050638`: re-testing `zPsi` as the keyboard logit after slow feature reseeding was NF-clean but regressed reward sum to `-48.532`, left `0/10` positive-rate seeds, and inflated duration to `221.8`. | Rejected; keep model-backed keyboard logits in the current L3-L7 keeper. |
| Wrapper score/action scaling variants | Gaussian-score standardization, per-coordinate actor alpha, per-coordinate standard score, and behavior-policy birth variants regressed 10k/50k paired screens. | Keep the current wrapper score interface unless paired evidence says otherwise. |
| Tracking-rate substitutions | `rho`, global-rho, and utility-timescale variants tied to `R` did not improve early reward-rate or 50k reward evidence. | `track_rate(t, baseO)` remains the current structural tracker, not a research invariant. |
| Raw observation passthrough | `l3-l7-rawobs-paired-10k-230923`: NF zero but worse cumulative reward (`-20.933`) and only `1/10` positive reward-rate seeds. | Rejected; keep standardized observation input to the Fourier encoder. |
| Chord-consistent value backup | `l3-l7-chordmodel-plan-paired-10k-231520`: replacing best-option planning advantage with chord-weighted Q regressed reward sum to `-29.654`. | Rejected; Design A policy blending does not imply chord-weighted value backup here. |
| Current-state planAdv gradient | `l3-l7-curr-planadv-paired-10k-231730`: NF zero but reward sum `-23.689`, `0/10` positive reward-rate seeds, and SPS collapse. | Rejected; keep value update on the source-state gradient. |
| Tracking RR bonus | `l3-l7-tracking-rrbonus-paired-50k-232052`: 10k was superficially close, but 50k regressed to reward sum `-247.071` and duration collapse. | Rejected; keep RR bonus as the high-water value weight from the feature's current identity. |
| Full option/feature capacity clock | `l3-l7-capacityclock-replace-paired-10k-232222`: feature and option replacement both slowed to full capacity, but termination fell to zero and reward regressed. | Rejected; feature reseeding is the destructive clock to slow, not option replacement wholesale. |
| Unused-only option replacement / candidate-pressure option clock | `l3-l7-unusedonly-replace-paired-10k-233410` and `l3-l7-candidatepressure-replace-paired-10k-233538`: both reduced option churn but drove termination near zero and worsened 10k reward. | Rejected; the next utility repair needs richer planning usefulness, not a harder candidate gate. |
| Centered reward encoder input | `l3-l7-centered-reward-encoder-paired-10k-005316`: research-clean average-reward perception repair, but 10k reward sum fell to `-26.474` with only `1/10` positive-rate seeds. | Rejected; keep raw reward as part of the experience input unless a broader perception model changes. |
| Direct policy output without the action trace | `l3-l7-direct-policy-output-paired-10k-005406`: removing the `0.5` output trace worsened reward sum to `-27.419` and drove score-gradient mean to `3.206e3`. | Rejected; direct output is not the plateau fix. |
| Decoded duration TD | `l3-l7-decoded-duration-td-paired-50k-005528`: 10k had a positive-rate pulse, but 50k reward sum regressed to `-80.617` versus the raw-duration keeper's `-72.800`. | Rejected; keep raw Bellman duration TD with the lower bound only at planning decode. |
| Stratified initial option-feature registry | `l3-l7-stratified-registry-birth-paired-10k-005737`: spreading initial anchors across `R` worsened reward sum to `-23.227` and `0/10` positive-rate seeds. | Rejected; keep the current `optFeature[k] = k` birth rule. |
| Implicit TD denominator plus raw planning advantage | `l3-l7-implicit-raw-planadv-paired-10k-005940`: replacing the soft planning advantage with raw advantage and an implicit value update regressed to `-44.286` and score-gradient mean `2.737e1`. | Rejected; raw planning pressure remains damaging even with implicit shrinkage. |
| Fixed current Gaussian sigma | `l3-l7-fixed-current-sigma-paired-10k-010112`: freezing scale coordinates at the current structural sigma collapsed reward to `-115.259` and removed useful score signal. | Rejected; adaptive scale is unstable long-horizon but cannot simply be disabled. |
| Saturation-aware Gaussian score | `l3-l7-saturation-aware-score-paired-300k-011021`: ignoring score gradients for clamped continuous samples fixed the long-horizon score spike, but 300k reward regressed to `-526.692` versus the keeper's `-489.611`. | Rejected; the score spike needs a repair that does not reduce reward learning. |
| Value-aware transition update weighting | `l3-l7-valueaware-transition-update-paired-10k-010253`: weighting transition TD by `|valueW[j]|` caused action magnitude growth and reward collapse to `-1054.454`. | Rejected; keep value-aware priority, not value-weighted transition updates. |
| Registry-state fast feature clock | `l3-l7-unused-fast-feature-clock-paired-50k-010430`: fast feature testing while any never-used feature exists improved 10k to `-15.606`, but 50k regressed to `-113.198` and feature churn rose to `699.6`. | Rejected; early feature churn helps short screens but destroys the 50k gate. |

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
