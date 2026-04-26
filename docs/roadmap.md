# OaK roadmap - Plan-orthodox continual improvement

Last audited: 2026-04-26, after commit `f44a27bf3` (`Record 1M walking
microscope`), the corrected 300k walking gate, the first 10M seed-3
microscope, 1M coverage microscopes, the 10k-pulse live diagnosis pass, and
the grounded dead-credit refresh screen.

The live branch has moved from "negative but beats random" into a new factual
regime: OaK can produce real HalfCheetah locomotion through the bridge, but the
walking mode is not yet stable across seeds. The current Oak-only 300k keeper
evidence is `.oak-benchmarks/grounded-deadrefresh-oak-300k-20260426`: 10
HalfCheetah seeds. Oak mean reward is `+5575.338`, max seed reward is
`+109689.456`, mean `info:x_velocity` is `+0.038705`, positive velocity is
`7/10`, positive return is `2/10`, and Oak NF total is `3`. The strongest seed
is not marginal:
seed 3 reaches reward `+109689.456`, run reward `+119046.372`, and mean
velocity `+0.396821` by 300k. The same gate still fails stability: seed 8 is
strongly backward (`-37294.861`, mean velocity `-0.086680`), seed 1 still has
non-finite health counts, and the promotion standard remains unmet.

The first deep microscope is
`.oak-benchmarks/deep-seed3-10m-20260426-030016` (Oak seed 3, 10M steps).
The seed keeps positive bridge-measured forward momentum over the full run
(`info:x_velocity` mean `+0.011783`, `info:x_position` delta `+5891.636`) and
finishes with reward `+64937.781`, but it loses the 300k/1M snowball. The first
1M heartbeat is already `+117455.284`; the remaining 9M steps drift downward
as control cost accumulates and run reward decays. Terminal diagnostics show
the FC-STOMP feedback cycle going quiet rather than catastrophically reversing:
`terminationRate=0`, `modelPriorityMax=0`, `optionDurationMean=1`,
`actorSupportEntropy=0`, `actorUpdateNormMean=0`, huge finite
`actionDeltaL2Mean=3.62e49`, and non-finites limited to
`learn=1, tdPsi=1`. This is a better failure than collapse, but it is still a
no-termination/actor-support fixed point that cannot be promoted.

The follow-up 1M microscopes
`.oak-benchmarks/microscope-seeds0-5-1m-20260426` and
`.oak-benchmarks/microscope-seeds6-9-1m-20260426` show that the 300k headline
understates OaK's long-run upside but not its instability. Across seeds 0-9
at 1M, Oak has mean reward `-2783.908`, mean `info:x_velocity=+0.020291`,
positive reward/velocity `3/10`, and NF total `8`. Seeds 3, 6, and 9 are
forward snowballs by 1M (`+117455.284`, `+28838.810`, `+43641.635`), while
seed 8 is the large backward attractor (`-109670.777`, mean velocity
`-0.073140`). Seeds 0 and 1 remain quiet low-motion failures; seeds 4 and 5
become costly bad commitments.

A bridge oracle now exists as a non-agent witness, not doctrine:
`.venv/bin/python tools/bridge_oracle_cem.py --cem-iters 0 --eval-horizon 5000
--seed-count 5` runs a fixed open-loop sinusoid through the same C bridge and
returns mean reward `+4428.300`, mean `info:reward_run=+4914.978`, mean
`info:reward_ctrl=-486.678`, and mean `info:x_velocity=+0.982996`. This proves
the bridge can carry positive locomotion and gives a scale target; none of its
parameters or behavior enter OaK.

The live architecture at this audit: default perception is direct normalized
experience channels plus constructed feature slots, K=R options/subtasks,
wrapper-side one-step importance ratios, RR Eq. 4 continuation with a
differential-value gauge, model-backed average-reward option scoring,
scalar chord refresh/blended policy emission, raw reward plus duration option
models, full-bank option-model planning, and age/capacity-gated Oak
replacement. Box actions keep the `2A` wrapper interface for capacity stability,
but the scale half is wrapper-reserved/inert and sampling uses structural fixed
`sigma = 1/D_policy` with residual-style mean credit (`raw - mu`). `optValue`,
`rhoOpt`, `oneTransition`, predictive-only GVF banks, model-residual GVF
utility, learned Box log-scale, `optionPlanningUse`, stored primitive-channel
tables, `oneReward`, `psiCross`, signed-keyboard lambda machinery, cached
keyboard scores, and deep diagnostic ownership remain deleted because they were
finite scale paths or shadow bookkeeping without earned control value.

Operational lesson from the first walking loop: changes that reduce commitment
or "calm" the controller can look cleaner at 10k while deleting the high-return
gait by 50k/300k. Pure-Q behavior selection, score-based interruption,
negative-TD chord refresh, chord-weighted actor baseline, decoded duration TD,
positive-only keyboard output, option-only G&T, and reward-aligned actor probes
were all rejected because they either killed the strong walking seed or failed
the 300k coverage gate. Future changes must preserve the high-return seed while
improving seed coverage and non-finite health.

Post-walking loop audit, also rejected as standalone keepers:
Plan-clean Swift arena alignment (`fcstomp-arena-align-*`) exposed that the live
walking checkpoint depends on an accidental delayed-plasticity/layout coupling;
directly correcting the layout made the run NF-clean but killed the strong gait
(`50k` mean `-398.017`, max `+796.612`). Q-per-duration behavior selection with
the aligned layout gave a real architectural clue but failed promotion:
`fcstomp-aligned-qperdur-300k` produced one strong forward seed (`+60705.313`,
mean velocity `+0.221417`) while another seed snowballed backward (`-88300.394`,
mean velocity `-0.159385`), for mean reward `-7422.100`, positive velocity
`5/10`, and NF total `12`. Q-per-duration interruption, decoded-duration TD,
TD-corrected scores, selected-option execution/model credit, chord-basis model
credit, signed replacement utility, neutral utility birth, and diagnostic
`alpha=1` all regressed 10k/50k gates. The next accepted fix must replace the
accidental snowball with a Plan-honest FC-STOMP feedback/search-control
mechanism; do not revive these probes by name without materially new evidence.

Post-10M FC-STOMP probes rejected on 2026-04-26: exact chord-responsibility
bridge credit kept full actor support but starved behavior (`10k` Oak mean
`-58.869`, zero positive-velocity seeds); every-step live keyboard refresh was
also rejected (`10k` mean `-92.077`); and the narrower all-continue refresh
failed the 50k screen (`-498.482` mean, `2/10` positive return) versus the
walking checkpoint's 60k heartbeat (`+596.453` mean, `4/10` positive return).
Reward-signed resource utility was also rejected (`50k` mean `-771.168`, seed
3 only `+548.251`), as was stale-search refresh on all-continue plus zero
model-priority (`50k` mean `-1199.252`, NF total `2`). The lesson is sharper
now: keeping support alive or waking silent options is not enough if it erases
the hard commitment that discovers the gait or wakes bad commitments too early.

Live diagnosis pass: `OAK_BENCH_HEARTBEAT_STEPS=10000` plus
`tools/bench_watch.py` was added so heartbeats can be watched as a rolling
stream instead of only ten coarse snapshots. The Oak-only 300k witness
`.oak-benchmarks/live-diagnosis-oak-300k-20260426` is behavior-identical to the
fresh post-audit 300k final, but the 10k windows expose distinct failure modes:
seeds 0/1 enter a learn+TD-psi non-finite quiet trap and then carry zero
actor/model pressure while duration predictions become enormous; seed 3 is a
clean fast finite snowball through 300k; seed 8 is the same hard-commit
machinery pointed backward with huge actor/model pressure and no NF; seeds
2/6/9 remain connected but not yet strong. The quiet-seed limiter is therefore
not simply missing exploration: replacement can churn without coupling back
into actor pressure, while invalid model/subtask arithmetic can make FC-STOMP
go silent.

Efficiency rule after the closed-loop probe failures: new behavior candidates
must first run a targeted seed screen before full grids. Use
`OAK_BENCH_SEEDS=0,1,3,8` to cover quiet collapse, the strong seed-3 rocket, and
the seed-8 backward attractor, then compare with `tools/bench_compare.py`. A
candidate that cannot move seeds 0/1, preserve seed 3, and avoid worsening seed
8 at 50k is rejected before a 300k spend.

Two follow-up FC-STOMP probes were rejected. RR non-finite termination treated
invalid RR stopping comparisons as forced option termination; it woke seed 1
into a control-cost explosion (`rr-nf-terminate-50k-20260426`, Oak mean
`-1016.360`, seed 1 reward `-14363.9`, `actionDeltaL2Mean=2.9e33`), so invalid
continuation cannot be solved by forced refresh alone. Off-policy TD-energy
normalization for option reward/transition/duration model updates improved the
10k mean (`-75.155`) but killed the seed-3 rocket by 50k and created a seed-8
control blowup (`model-implicit-energy-50k-20260426`, Oak mean `-1616.230`);
fast model plasticity is part of the walking pathway and cannot be damped
uniformly.

One narrow truthfulness fix remains live: `oak_plan` now counts non-finite
model TD arithmetic and preserves the previous option-model priority instead
of silently converting a broken model update into zero priority. This did not
change the 300k behavior (`prio-preserve-oak-300k-20260426` matches the fresh
post-audit final rewards) but reveals hidden plan non-finites in quiet seeds
instead of reporting false search-control certainty.

ArXiv FC-STOMP search pass, 2026-04-26: the primary sources line up with the
current empirical failures. Reward-Respecting Subtasks defines STOMP as
subtask -> option -> model -> planning and explicitly calls out the missing
FC-STOMP feedback: planning effectiveness judged by search control should
influence which models, features, and feature-attainment options survive. The
average-reward options paper supports reward/dynamics/duration option models
and interruption only when finite option values show `q_mu(s,o) < v_mu(s)`;
it does not support forcing termination on broken RR arithmetic. Option
Keyboard combines intentions in cumulant/value space through cross-cumulant
GPE/GPI (or an equivalent chord-descriptor model), not by merely blending
component policy means. Meta-gradient search-control work is directly relevant
to the quiet/backward split: useful planning should learn to avoid inaccurate
or credit-stalling model queries. Operationally, the next stabilization change
must repair keyboard/search-control/utility as one loop; isolated wakeups,
uniform damping, and raw-TD priority substitutions are now treated as already
disfavored unless they come with materially new paired evidence.

The first post-search probe confirmed the warning. Replacing option-bundle
survival priority with downstream utility plus keyboard mass, and clearing
stale keyboard/search-control metadata on option rebirth, improved the 10k
screen (`useful-reinit-10k-20260426`, Oak mean `-51.671`, positive velocity
`5/10`, NF zero) but failed by 50k (`useful-reinit-50k-20260426`, Oak mean
`-754.780`, positive velocity `3/10`): it killed the seed-3 rocket path and
created a seed-6 pressure/duration blowup despite NF zero. Do not promote this
simple "stale cleanup" variant; raw-priority protection is entangled with the
current walking pathway until a better search-control signal replaces it.

A more coherent but still incomplete planning-use probe was also rejected. It
tracked per-option previous planned contribution times next real TD and fed
that signal through keyboard scores, feature/model utility, option utility,
and a heartbeat metric. This was closer to the FC-STOMP feedback target, but
still not a complete architecture: `planning-use-50k-20260426` reached positive
velocity `6/10` and positive reward `4/10`, yet mean reward was `-430.887`,
seed 3 did not rocket (`-290.158`), seed 5 blew up in duration/planning-use
with one NF, and the known walking mode was not preserved. Do not reattempt
single-scalar planning-use credit; the next change must define the whole
keyboard/search-control/utility contract before touching behavior.

The first explicit cross-cumulant keyboard implementation was rejected on both
learning and compute grounds. It added a dense `psiCross[K*K*R]` head, separated
the chord descriptor from a behavior-responsibility vector, and used a
GPI-style cross-cumulant score for action emission, planning pressure, utility,
and replacement. The 10k screen looked alive (`cross-keyboard-10k-20260426`,
Oak mean `-41.974`, positive reward `3/10`, positive velocity `5/10`, NF zero),
but the 50k screen collapsed (`cross-keyboard-50k-20260426`, Oak mean
`-2849.527`, positive reward `0/10`, mean velocity `-0.03064`, seed 8
`-11198.767`). It also added O(K^2 R) hot-path work and visibly slowed the
50k gate. Do not revive dense `psiCross`/GPI. A keeper FC-STOMP keyboard must
be fixed-budget and low-rank/pagewise, with no more than the current order of
per-step work unless Instruments evidence justifies the cost.

A fixed-budget page-loop variant was then rejected on the 10k gate. It used no
dense cross head and no new learner span, instead rotating one RR subtask page
through the existing option-model decode and feeding that page signal into
keyboard refresh, source-feature utility, option utility, and replacement.
`.oak-benchmarks/page-loop-10k-20260426` was NF-clean, but Oak mean reward was
`-84.616`, positive reward was `0/10`, mean velocity was `-0.003253`, positive
velocity was `0/10`, and mean Oak SPS was `2292`. This failed both waking and
compute criteria; do not revive pagewise cross pressure unless it is part of a
fuller fixed-budget design that preserves seed-3 walking evidence.

A causal reward-closed search-use vector was also rejected. It added one K-sized
trace updated every step from the previous chord's normalized absolute
responsibility, exact wrapper importance, and realized centered reward TD, then
fed that same trace into keyboard scoring, feature utility, option utility, and
replacement. This was the right kind of closed-loop test, but not a keeper:
`.oak-benchmarks/searchuse-10k-20260426` was NF-clean with one positive-return
seed, then `.oak-benchmarks/searchuse-50k-20260426` killed the seed-3 rocket and
finished with Oak mean reward `-775.868`, positive reward `0/10`, mean velocity
`-0.006057`, positive velocity `4/10`, NF zero, and mean SPS `2232`. The result
reinforces that realized reward feedback cannot simply be added to model-backed
keyboard scores; it must preserve the hard commitment pathway before it is
allowed to redirect search or replacement.

A duration-consistent priority-protection audit was rejected as behavior-dead.
It stopped raw model priority from protecting an option/model bundle during
replacement when its decoded duration exceeded the agent's entire lived stream.
This was structurally cleaner and preserved the seed-3 rocket, but it did not
change any 50k or 300k final reward/velocity/replacement result:
`.oak-benchmarks/duration-consistency-50k-20260426` and
`.oak-benchmarks/duration-consistency-oak-300k-20260426` matched the keeper
per-seed rewards exactly. Do not keep self-consistency checks that only improve
bookkeeping; the fix must change the FC-STOMP loop's behavior.

Preserving the last finite RR subtask TD / actor advantage on non-finite
subtask arithmetic was also rejected as behavior-dead. The hypothesis was that
zeroing transient subtask pressure made quiet seeds collapse into do-nothing.
`.oak-benchmarks/preserve-tdpsi-50k-20260426` and
`.oak-benchmarks/preserve-tdpsi-oak-300k-20260426` did revive enormous finite
actor pressure in the quiet seeds, but every final reward and velocity matched
the keeper exactly. The lock is downstream of actor pressure: the continuation /
keyboard emission path can remain behaviorally frozen even when actor updates
become huge. Do not keep hidden pressure that does not move the policy.

The first targeted-seed screen after adding `OAK_BENCH_SEEDS` rejected
theorem-aligned value interruption as a standalone fix. Pure finite
`q_mu(s,o) >= v_mu(s)` continuation woke seed 1 and greatly improved seed 8,
but killed seed 3's rocket by 50k
(`.oak-benchmarks/value-interrupt-target50k-20260426`: seed 3 delta
`-12797.589`). Adding a live positive-TD guard still killed seed 3 and regressed
seed 1 (`.oak-benchmarks/tdguard-interrupt-target50k-20260426`). Interruption is
a real lever for the lock, but it cannot be allowed to break hard commitment;
future attempts need a keyboard/emission mechanism that distinguishes productive
commitment from stale continuation, not only a finite option-value comparison.
Model-priority-gated interruption was also rejected by the same screen:
`.oak-benchmarks/prio-gated-interrupt-target50k-20260426` still killed seed 3
and drove seed 1 hard backward. Model surprise is not the missing discriminator
by itself.
The following stale-emission fallback was rejected as another too-blunt
behavior repair. It used the previous emitted-action movement to fall back to
selected-option execution when the chord blend was nearly stationary. The
targeted screen `.oak-benchmarks/stale-fallback-target50k-20260426` improved
seed 8 (`-4427.302 -> -120.987`) and seed 0 (`-610.384 -> -344.045`), but cut
seed 3 from `+12351.967` to `+1368.633` at 50k. The lesson matches the
interruption probes: low movement is not stale by itself; the stable repair must
keep the productive blended snowball while giving quiet seeds a way into it.
An all-continue dead-option generate-and-test repair was then rejected. It only
replaced a mature option bundle when the whole option bank was continuing, the
bundle had exactly zero actor pressure, and its utility had gone non-positive.
`.oak-benchmarks/dead-option-gt-target50k-20260426` preserved seed 3 exactly,
but worsened seed 0 (`-610.384 -> -935.493`) and left seed 1/8 unchanged.
Bundle churn after actor credit is dead is too late; the repair has to change
keyboard/search-control credit before the behavior path freezes.
Low-rank chord-progress keyboard projection was also rejected as a direct
behavior path. The raw projection
`.oak-benchmarks/chord-progress-target50k-20260426` woke seed 0 and flipped
seed 8 positive, but killed seed 3 (`+12351.967 -> -1771.834`) and drove seed 1
backward. The agreement-only projection
`.oak-benchmarks/chord-agree-target50k-20260426` improved seed 1 and seed 8,
but still killed seed 3 (`-810.945`). The model-error-gated agreement variant
`.oak-benchmarks/chord-erroragree-target50k-20260426` was worse. A `1/K`
basis-normalized version (`.oak-benchmarks/chord-basisnorm-target50k-20260426`)
still cut seed 3 to `+2103.403` and worsened seed 0. This is useful
architectural evidence: cross-cumulant keyboard semantics are relevant, but
direct low-rank behavior-weight overrides steal the identity/reward pathway
that discovers the high gait.
Signed behavior-responsibility main-TD actor pressure was rejected in two
forms. The full version
`.oak-benchmarks/responsibility-td-actor-target50k-20260426` woke seed 0 and
partially rescued seed 8, but killed seed 3 (`+12351.967 -> -1377.770`). The
degenerate-only version
`.oak-benchmarks/degenerate-td-actor-target50k-20260426` preserved seed 3
exactly, but did not wake seed 1/8 and worsened seed 0. The actor-collapse
diagnosis is valid, but direct reward TD inside the option actor is either too
invasive or too late.
Search-failure-gated versions of the same actor idea were also rejected:
`.oak-benchmarks/searchfail-tdactor-target50k-20260426`,
`.oak-benchmarks/searchfail-rfloor-tdactor-target50k-20260426`, and
`.oak-benchmarks/searchfail-negrho-tdactor-target50k-20260426`. They rescued
seed 1 and partly seed 8, but still erased seed 3 even with structural
model-priority/duration and `rho < -1/R` gates. The conclusion is now firm:
main-reward actor pressure needs a learned arbitration state, not another
inline behavior gate.
The mature-selected-option variant
`.oak-benchmarks/mature-searchfail-tdactor-target300k-20260426` is rejected for
the same reason. It produced a large seed-1 gait (`+40979.575`) and mostly
rescued seed 8 (`-37294.861 -> -1220.894`, positive velocity), but destroyed
the seed-3 high mode (`+109689.456 -> -1338.779`) and worsened the four-seed
target mean. This is the clearest evidence so far that the rescue actor is a
useful organ, but it needs arbitration with the identity/reward pathway before
it can be part of the body.
The first scalar slow-arbiter attempt
`.oak-benchmarks/slowarb-tdactor-target50k-20260426` was also rejected. A
capacity-derived slow reward trace reduced the violence of the rescue actor,
but still cut seed 3 from `+12351.967` to `+281.231` at 50k and missed seed 1.
Arbitration likely needs to remember more than persistent negative reward; it
must identify when the identity path is building a delayed snowball.
Origin-lock rescue actor (`.oak-benchmarks/originlock-tdactor-target50k-20260426`)
was rejected too. The selected-age diagnostics showed seed 8 and the quiet
locks as origin-lock failures under the keeper, but using that symptom as an
actor trigger pushed seed 3 into the same quiet origin-lock basin. Origin-lock
is a diagnostic, not yet a behavior-safe control signal.

The keeper from this loop is grounded dead-credit keyboard refresh. It refreshes
the keyboard only when every RR option continues, actor credit is exactly dead,
and all decoded option durations remain no larger than the agent's lived stream
length. The ungrounded version
`.oak-benchmarks/deadcredit-refresh-target50k-20260426` preserved seed 3 and
helped seed 0, but blew up seed 1's control cost. Adding the duration-grounding
check avoided that failure. Full 300k Oak-only evidence:
`.oak-benchmarks/grounded-deadrefresh-oak-300k-20260426` improved mean return
from `+5365.610` to `+5575.338`, positive velocity from `6/10` to `7/10`, and
NF total from `8` to `3`, while preserving seed 3 bit-identically at
`+109689.456`. This is a keeper substrate improvement, not the solution: seed 8
is still the large backward attractor and only `2/10` seeds have positive total
return.

Host health note: long runs exposed resident-memory drift in the benchmark
worker. `Sources/Main.swift` now wraps each measured worker step in an
`autoreleasepool`; the behavior-identical seed-3 300k check
`.oak-benchmarks/autorelease-seed3-300k-20260426` reproduced reward
`+109689.456` and sampled worker RSS stayed flat around `19.5 MB` from
10s-90s.

Host simplification gate: the Swift host and native bridge were collapsed to a
HalfCheetah-only benchmark path, `ActionTransducer.swift` and `EnvBridge.swift`
were deleted, and the benchmark parent now launches workers with direct
`posix_spawn`/`waitpid`. The attempted Box `D_policy=A` deletion repeatedly
collapsed support by 120k-450k steps, so the live simplified host keeps the
`2A` wrapper code width, inert scale half, and `R=max(obsDim,2*D_policy,2)`
rank relationship. Gate artifacts:
`.oak-benchmarks/codex-host-rewrite-rank2d-600k` and
`.oak-benchmarks/codex-host-rewrite-rank2d-1.5m`; Oak NF stayed zero `10/10`
and actor support stayed near uniform at the 1.5M gate.

Dead-code rule after baseline promotion: GPU `scoreGrad` plumbing was removed
because the only kernel consumer wrote an unread threadgroup scalar; behavior
score-gradient telemetry is now also removed from host metrics. The promoted
aggressive simplification pass deletes host run/diagnose tooling, diagnostic-only
health gauges, offset-shuttle config fields, generic multi-component action
wrapping, `oneReward`, and `psiCross`. Gate:
`.oak-benchmarks/baseline-1m`, 10 paired HalfCheetah seeds at 1M. Oak NF stayed
zero `10/10`, Oak beats random `10/10`, mean Oak reward is `-5869.101`, median
Oak reward is `-5860.469`, mean paired lift is `256759.957`, min ESS is
`0.999221`, min support entropy is `0.999878`, and replacement churn is zero.

Full-bank planning cleanup: `oak_plan_reduce`, `highModelOption`,
`modelAge`, and the option-page scheduler were deleted after the 1M paired gate
showed bit-identical per-seed Oak reward, Oak NF zero `10/10`, Oak beats random
`10/10`, and Oak mean SPS improved from `3654.563` to `3771.714` (`+3.21%`).
This is an optimization/orthodoxy cleanup only: it does not complete the Option
Keyboard or FC-STOMP feedback cycle.

Baseline action-interface evidence note: the exact fixed-sigma-score seed-1
gate failed by 100k, with actor support collapsed to zero and psi magnitudes
exploding despite the scale half being inert. The later full `D_policy=A`
mean-only deletion failed the 1M gate: Oak lost to random `0/10`, NF total was
`70`, final support entropy collapsed to `0`, and action deltas reached
`2.67e70`. Do not promote exact off-policy Box mean scores or mean-only Box
capacity shrinkage without a different actor geometry. The promoted baseline
keeps scale inert but uses residual-style mean credit.

This roadmap is the control document for moving OaK toward the Alberta Plan's
Step-11 Oak architecture. It is deliberately stricter than the old tier log:
claims are marked complete only when the mechanism matches the literature, not
merely when a nearby surrogate exists in the kernel.

Primary sources:

- Alberta Plan: `docs/papers/AlbertaPlanForAIResearch_2208.11173.pdf`
- Reward-respecting subtasks / STOMP: `docs/papers/RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf`
- Average-reward options: `docs/papers/WanNaikSutton_AvgRewardOptions_2110.13855.pdf`
- Option keyboard: `docs/papers/OptionKeyboard_CombiningSkills_2106.13105.pdf`
- Option Keyboard Basis: `docs/papers/OptionKeyboardBasis_2505.00787.pdf`
- Meta-gradient search control: `docs/papers/MetaGradientSearchControl_DynaPlanning_2406.19561.pdf`
- Supporting current mechanisms: Reward Centering, MetaOptimize, GraMa,
  generate-and-test, prioritized sweeping, and plasticity papers in
  `docs/papers/`. Deep Fourier/random-projection encoders are experimental
  references only, not the orthodox default.

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

## Step 1-11 Rebuild Status

The 2026-04-24 audit was accepted as controlling guidance, with one correction:
Reward-Respecting Subtasks Eq. 4 stays exactly
`z_i(s)=V(s)-valueW[i]x_i(s)+bonusW[i]x_i(s)`. The report suggestion to replace
that terminal value with a plain feature bonus is rejected by the primary paper.

| Audit item | Status | Live response / evidence |
|---|---|---|
| `K <= 32` as invariant | Accepted drift | Removed as an invariant. K now equals R; the old 32-wide option page scheduler is deleted. Threadgroup width is derived from Metal pipeline execution width for hardware occupancy. R is stream-derived (`max(obsDim, 2*D_policy)`, R=24 for the current HalfCheetah Box mean/reserved-scale wrapper); the rank override knob was deleted as a structural fork. |
| B=1 as sacred law | Accepted drift | Reworded as current single-agent stream semantics. Future batched kernels are allowed only for N independent agents with no cross-stream reductions. |
| Deep Fourier / random-projection default | Accepted | Default state construction is now direct normalized experience channels plus constructed feature slots. Fourier, deep Fourier, and random projection remain experimental modes only; they are not Alberta Plan law. |
| Feature finding | Partial live | Constructed features use signed product, signed sum, and parent-gated residual forms with parent utility/decorrelation diagnostics. Replacement is still young and needs long-run evidence. |
| Off-policy floors | Accepted | `max(chord,1/K)` floors are removed from live learning. Wrapper emits exact one-step ratios from behavior-option sampling; ratio diagnostics are live. |
| Average-reward option values | Accepted, removed from keeper | `optValue[K*R]` and `rhoOpt` were removed from the live keeper after they became behavior-relevant scale paths without earned 300k/1M value. Do not revive them until the option-value role is derived and gated independently. |
| Predictive-only GVF bank | Removed | `gvfRate`, `gvfValue`, `gvfTd`, `gvfAge`, GVF step-size/meta buffers, and `gvfTdMean` were deleted because the bank remained diagnostic-only and never earned a behavior, replacement, or health-gate role. Main value and RR `psi` GVFs remain live. |
| True keyboard / FC-STOMP cycle | Open architecture gap with live scalar chord | `psiCross[K*K*R]`, dense λ, and cached/pagewise GPE scores are deleted as unproven behavior complexity. Live code computes model-backed average-reward option scores, refreshes a scalar chord when RR continuation breaks, and emits a chord-blended policy with selected-option fallback. This has now produced real walking in some seeds, but it is still not the paper's full cumulant-space Option Keyboard: chord-descriptor models and coherent feature/subtask/option/model feedback remain future work. |
| One-step model | Removed from keeper | `oneReward` and `oneTransition` are both removed from default layout because they were shadow/descriptor paths without earned control value. |
| Search control | Partial live | Planning covers the full option bank every step, and model-backed option scoring is behavior-relevant. The stale option-page scheduler is deleted after a bit-identical 1M promotion with +3.2% Oak SPS. Full state-search and uncertainty-aware backup selection remain open, and the 2026-04-26 walking loop showed that simply subtracting priority/error differently or deleting the model-backed score kills the high gait. |
| Continuing testbeds | Deferred by scope | A temporary toy-env attempt was removed because it introduced hand-coded dynamics constants. Add real research-backed continuing wrappers only as separate env-layer work, never inside OaK. |
| Step 12 / IA | Deferred | Out of scope until Step 11 earns promotion. |

Initial smoke evidence: `rebuild-smoke-10k-default` (seed 0, paired random)
rebuilt Metal and Swift cleanly, ran NF zero, beat random reward sum
`-64.998` vs `-2587.504`, kept `importanceEss=0.805`, and restored hot-path SPS
to `2897`. It is not promoted because reward-rate was still negative
(`-0.02475`) and the heartbeat curve was not sustained positive. Performance
regression note: the first K=R R=64 attempt was only about `760` SPS at 1k; the
stream-derived default R=24 plus sparse intent-key GPE restored the old speed
target without removing the Plan-required heads.

Rejected rebuild follow-ups:

- `rebuild-smoke-10k-policytd`: adding `optValueTd` directly to the option
  actor TD caused infinite policy deltas before 10k. Keep `optValue` as a
  learned average-reward diagnostic/planning value until a principled actor
  coupling is audited.
- `rebuild-smoke-10k-replacegate`: committing replacements only when the
  utility-minus-redundancy score was negative cut reinitializations but worsened
  10k reward sum to `-79.915` and reward-rate to `-0.03053`. Keep the current
  age/capacity-gated replacement while the broader feature-finding design is
  reviewed.
- `nextwave-toy-continuing-envs`: a first attempt to add Drift/RiverSwim/
  Access/Jellybean toy wrappers was removed before testing because it embedded
  arbitrary dynamics constants. The Plan wants continuing tasks, but not
  hand-tuned tasklets smuggled into the benchmark layer.

Next-wave evidence so far:

- `nextwave-smoke-1k-v3`: seed-0 paired HalfCheetah, NF zero, Oak `-8.431`
  versus random `-277.377`, `importanceEss=0.994`, dense λ effective rank
  `18.35`, compact `gvfCount=12`, and `2196` SPS.
- `nextwave-smoke-10k-v3`: seed-0 paired HalfCheetah, NF zero and sane GVFs
  (`gvfTdMean=0.552`), but cumulative reward `-69.838` trails the current
  keeper's 10k `-64.998`. Do not promote on 10k evidence alone.
- `nextwave-smoke-50k`: seed-0 paired HalfCheetah, NF zero, Oak `-303.337`
  versus random `-13177.211`, slightly better than the keeper's 50k `-305.32`.
  Reward-rate remains negative (`-0.009296`), so this is architecture evidence,
  not promotion evidence.

Research-correct collapse-repair evidence (live branch after the April 2026
review):

- MetaOptimize is the controlling source for per-feature step sizes, not raw
  IDBD. The live `alpha/beta/h/gradEnergy` path should be understood as a
  low-complexity MetaOptimize L/Hessian-free approximation over feature blocks:
  the base update is the ordinary linear TD step, `h` tracks the causal
  sensitivity trace, and `gradEnergy` provides a structural RMS-style meta
  normalization. IDBD is only a special case/approximation inside
  MetaOptimize, per arXiv:2402.02342v6.
- `research-correct-stats-sparse-10k-hb`: seed-0 HalfCheetah, rebuilt Metal and
  Swift, NF zero, `3022` SPS with normal heartbeats, reward `-90.424`,
  `actorUpdateEss=0.621`, `actorSupportEntropy=0.999`, calm score/action
  telemetry (`actionDeltaL2Mean=0.0249`). Stats are now sampled
  on heartbeat/final only; OaK learning still runs every step.
- `research-correct-stats-sparse-10seed-100k`: 10 paired seeds, Oak beats random
  `10/10`, NF zero `10/10`, mean Oak `-767.19` versus random `-26423.56`, final
  reward-rate positive `5/10`, but final-10k reward slope positive `0/10`.
  Support is stable (`actorUpdateEss` mean `0.573`, option-support entropy mean
  `0.987`), score/action stay calm, and replacement remains zero.
- `research-correct-stats-sparse-10seed-300k`: 10 paired seeds, Oak beats random
  `10/10`, NF zero `10/10`, mean Oak `-2358.92` versus random `-79297.35`.
  The old collapse signature is largely repaired: no score/action runaway,
  no replacement churn, no systematic one-option floor (`actorUpdateEss` mean
  `0.560`, entropy mean `0.974`). It is not promotion evidence because final
  reward-rate is positive only `1/10` and final-50k slope is positive `0/10`.
- `research-correct-current-10seed-100k`: rebuilt current source after fixing
  `rhoOpt` to use the current-step implicit denominator. Oak beats random
  `10/10`, NF zero, mean Oak `-764.73` versus random `-26423.56`, positive
  final reward-rate `5/10`, but final-10k slope `0/10`. Full heartbeat metrics
  show calm score/action and zero replacement, but RR/psi magnitudes remain very
  large and actor updates are conservative.
- `research-correct-current-10seed-300k`: current source, 10 paired seeds. Oak
  beats random `10/10`, NF zero, mean Oak `-2517.45` versus random
  `-79297.35`, positive final reward-rate `1/10`, final-10k and final-50k slope
  `0/10`. This is not promotion evidence: seed 1 reaches the one-option support
  floor by 300k (`actorUpdateEss≈0.042`, entropy≈0.010), basis use collapses on
  two seeds, and RR/psi/keyboard margins remain too large despite calm
  primitive action statistics.
- `research-correct-rhoopt-rate-10seed-100k` and
  `research-correct-rhoopt-rate-10seed-300k`: removing the extra implicit
  denominator from the live `rhoOpt` rate update was more literal to the latest
  plan text and reduced RR/psi scale. It improved over the denominator-order
  branch at 100k/300k (`-738.66` mean at 100k; `-2401.29` mean at 300k) but did
  not beat the best stabilized pre-denominator evidence (`-2358.92` mean at
  300k), and seed support still touched the floor. Do not promote it as keeper
  until it beats the best known long-horizon reward/support evidence.
- `research-correct-cowfix-speed-50k`: host hot-path COW fix for importance
  arrays restored the single-stream speed guard without changing seed-0
  trajectory (`-403.083914` reward, NF zero, `3015` SPS versus `2964` before the
  fix). This is a zero-intelligence-hit efficiency fix.
- `research-correct-metaopt-hdecay-50k`: rejected as an isolated per-feature
  MetaOptimize 2x2/diagonal-curvature experiment. It was NF-clean and slightly
  improved seed-0 reward (`-393.51`), but it missed the speed guard (`2968` SPS)
  and increased RR/psi scale (`rrBonusMean≈27779`, `psiAbsMean≈8610`). The next
  MetaOptimize repair should use the paper's lower-complexity framework more
  carefully, not revive this diagonal-curvature trace unchanged. This does not
  demote MetaOptimize: the live per-feature step-size path remains part of the
  architecture and should be strengthened from MetaOptimize, not replaced by raw
  IDBD/Autostep or removed.
- `research-correct-restored-speed-50k`: final current source restored to the
  best-known long-horizon `rhoOpt` behavior while keeping the host COW speed fix,
  rebuilt Metal/Swift. Seed-0 HalfCheetah is over the speed guard (`3011` SPS),
  NF zero, and matches the stabilized reward trajectory (`-404.064262` at 50k).
  The long-horizon comparison for this math remains
  `research-correct-stats-sparse-10seed-100k` / `300k`; the COW fix is a
  hot-path copy change and does not alter learning.
- `architectural-leap-normal50k-sampled-meta-stats`: default encoder no longer
  uses hashed/random projections. It uses direct experience-channel anchors,
  constructed features, a gauged RR stopping value, centered actor advantage,
  live `rhoOpt`, signed cumulant-space keyboard, expanded GVFs, and
  MetaOptimize family state with the squared-error curvature trace term from
  the low-complexity MetaOptimize/IDBD special case. Seed-0 HalfCheetah is
  NF-zero and improves 50k reward over the restored keeper (`-303.34` versus
  `-404.06`), with high support (`importanceEss=0.995`, entropy `0.999`) and
  calm score/action statistics. It is not yet a promoted keeper: SPS is
  `2973`, just under the 3k guard, and the 10-seed 100k/long-horizon evidence
  must decide whether the direct predictive-state architecture improves across
  seeds.
- `architectural-leap-seed0-100k-cross-denom`: seed 0 exposed the actual
  collapse trigger. Before repair, heartbeats were calm through 60k but
  `keyboardGpiMargin` grew to trillions, then `psiCross`/keyboard selection
  fell into one-option support and action magnitudes overflowed. Applying the
  same implicit one-step denominator used by `psi`/option values to `psiCross`
  kept the formal GPE/GPI keyboard finite: seed 0 finished 100k NF-zero with
  reward `-605.33`, `importanceEss=0.984`, entropy `0.997`, calm score/action,
  and `keyboardGpiMargin=7.99`.
- `architectural-leap-10seed-100k-cross-denom`: 10 paired seeds, rebuilt
  Metal/Swift. Oak beats random `10/10`, NF zero `10/10`, mean Oak reward
  `-602.20` versus random `-26423.56`, improving over the previous stabilized
  100k mean (`-767.19`). Mean support is healthy (`importanceEss=0.953`,
  entropy `0.993`) and score/action remain calm. Positive final reward-rate is
  `4/10`, so this is a real architecture improvement but not a promotion to
  sustained positive learning. 300k and 1M evidence remain required.
- `architectural-leap-seed0-300k-cross-denom`: seed-0 long-horizon screen
  passed the previous collapse signature. Reward is `-1809.68`, NF zero,
  `importanceEss=0.997`, option-support entropy `0.9996`, calm action/score,
  zero replacement churn, and `keyboardGpiMargin=14.0`. The run does not show
  continuous positive reward-rate (`rewardRate=-0.0197`), but it avoids the
  earlier seed-0 failure mode where the branch fell into one-option support and
  huge action movement after a short positive window. 10-seed 300k is the next
  required judge.
- `bigworld-audit-current-1m`: current architectural-leap checkpoint after the
  Big World stability audit, rebuilt Metal/Swift, process-isolated concurrent
  10 seeds x `{oak,random}` at 1M. Oak is materially better than the previous
  `current-keeper-hc-10seed-1m` failure but is not solved: mean Oak reward
  improved from `-74449.58` to `-31360.39`, mean lift over random is about
  `+232936`, mean ESS improved from `0.0417` to `0.530`, entropy from `0.0009`
  to `0.557`, and concurrent SPS from `1398` to `2028`. NF remained zero.
  However several seeds still collapse to the one-option floor or huge
  action/score movement (`actionDeltaL2Mean` up to `4.49e301` and former
  behavior-score telemetry up to `1.39e69`), planning/model diagnostics still show
  finite hallucination-scale residuals (`planningBackupTdMean` up to
  `3.38e38`, mixed model-improvement scalars up to `8.10e15`), and option
  duration can drift to `2.00e22`. This is a better research checkpoint, not a
  keeper promotion; next cleanup must remove remaining Big World violations and
  replace live model-value coupling with a projected/GTD-style model-planning
  path before another promotion attempt.
- `deslop-single-stream-10seed-1m`: aggressive cleanup checkpoint after deleting
  unearned planner/stat surfaces, collapsing the wrapper to the true single-agent
  stream path, and replacing the mixed-unit `modelExpectedImprovement` feedback
  with direct option-model TD priority. Rebuilt Metal/Swift, process-isolated
  concurrent 10 seeds x `{oak,random}` at 1M. Oak beat random `10/10`, NF was
  zero on every final and heartbeat, mean Oak reward was `-5861.16` versus
  random `-264296.39`, mean lift `+258435.23`, and mean SPS was `3087.77`
  (`2912.72` min under worker contention). Support did not repeat the prior
  floor collapse: final `importanceEss` min `0.9842`, actor-support entropy min
  `0.9968`, no feature/option reinitialization, calm action telemetry
  (`actionDeltaL2Mean` max `3.95e-4`). This is
  a cleaner keeper floor, not a solved agent: option termination is still zero
  and duration remains too large, so the next cleanup should simplify the
  termination/duration substrate rather than add audit heads.
- `architectural-leap` rejected integration attempts: raw actor MetaOptimize
  without centered subtask advantage collapsed ESS/action support before 10k;
  full-rank non-value MetaOptimize plus primitive-anchor candidate scans
  suppressed speed; basis Gram diagnostics doubled `oak_replace`. These were
  repaired by centering actor advantage across options, adding the curvature
  term to the MetaOptimize sensitivity trace, servicing non-value family
  MetaOptimize on a structural feature slot each step, and moving expensive
  diagnostics out of the hot path. RR bonus remains stopping-value machinery
  only (`rrBonusLeakCount=0`).
- `research-correct-actornorm-10k`: rejected immediately. The square-root actor
  normalization produced NF zero but one-option floor from heartbeat 1, enormous
  score/action magnitudes, reward `-5997`, and `2877` SPS.
- `research-correct-valueimplicit-10k`: rejected immediately. Adding an
  implicit denominator to the main value update fixed RR/psi magnitudes
  (`rrBonusMean=10.85`, `psiAbsMean=10.25`) but collapsed option support
  (`actorUpdateEss=0.0417`, entropy `0.0021`) and fell below the 3k guard.
  Keep the live value path unchanged until a broader RR-bonus/meta-learning
  repair is available.

Collapse-repair baseline truth table:

| Review claim / subsystem | Current keeper truth | Next action |
|---|---|---|
| Actor update scope | The keeper updates every option policy every step in `oak_learn_heads`; it is not a selected-only actor. The current cleanup removed the invented actor MetaOptimize/exponential scale path and uses the structural capacity actor step. | Do not revive selected-only actor, ESS throttles, direct main-TD primitive actor shortcuts, or unearned actor-scale amplifiers. |
| Target-option actor gradients | Raw target-option scores and Fisher/natural target scores failed in isolation (`review-response-50k-natural-133322`, `review-response-50k-fisher-133540`). Exact wrapper target gradients are still supplied by the native bridge; actor scale remains structural rather than a learned gain. | Keep ratio/support/action health visible; next actor work must be derived, not damped. |
| Global `rho` | Global average reward remains the live control/model-planning rate: `Q = rHat - rho*lHat + V(nHat)`. | Keep live; compare against shadow per-option control rates before any behavior change. |
| `rhoOpt` status | Removed from the live keeper with `optValue`; global `rho` drives main value, RR `psi`, and model backup. | Reintroduce only with a separately justified option-value branch that passes 300k/1M support gates. |
| λ formula | Full signed/GPE λ is removed from keeper. Live behavior computes a scalar model-backed chord from average-reward option scores and preserves the current option while RR continuation is positive. | Do not revive `psiCross`/GPE-GPI or alternate chord formulas as isolated swaps. Any keyboard repair must preserve the current high-return gait and add explicit cumulant-space/chord-descriptor semantics. |
| GVF coverage | The predictive-only `G=R+K+2` GVF bank is removed. Main value and K RR `psi` GVFs remain the live prediction heads. | Do not re-add diagnostic prediction banks until they have a direct Plan role and pass the same benchmark gate as behavior code. |
| Replacement behavior | Live age/capacity-gated replacement remains, with `badFeatureAge` retained as anti-churn state after immediate replacement caused 100k support collapse. Shadow replacement metrics, pseudo-basis usage caches, `optionPlanningUse`, and stored primitive-channel tables are removed. | Next replacement work must simplify or replace the actual generate-and-test rule, not add more audit-only counters. |
| Search control behavior | `oak_plan` updates the full option bank every step. The page reducer, stale-page age state, and `highModelOption` scheduler are deleted; model priority remains a health/search diagnostic. | Next search-control work must add a real state/update queue or model uncertainty, not revive page fairness scaffolding. |

Unearned-scale cleanup evidence:

- `deslop-gvf-utility-delete-10seed-100k`: HalfCheetah 10 seeds x
  `{oak,random}`, Oak beat random `10/10`, NF zero, mean reward
  `-585.127` versus random `-26423.561`, mean ESS `0.999326`, min actor
  support entropy `0.999397`, and no feature replacement. Removing GVF utility
  did not regress the 100k learning/support trajectory and removed the prior
  finite `~7.9e35` GVF utility spike from reported health.
- `deslop-gvf-utility-delete-seed0-1m`: serial seed-0 HalfCheetah, NF zero,
  `3321.865` SPS, reward `-5764.547`, ESS `0.998883`, actor support entropy
  `0.999827`, no feature replacements, and `gvfTdMean=1.88e-8`. This improves
  the previous cleanup 1M seed-0 reward (`-5869.005`) while preserving the
  3k+ SPS guard.
- `deslop-planningring-bulk-10seed-100k`: removed stale keyboard/replacement/
  basis audit metrics, the pseudo-basis usage cache, and the recent-state
  planning backup/q-cache surface. HalfCheetah 10 seeds x `{oak,random}` stayed
  NF-zero, beat random `10/10`, improved mean Oak reward slightly to `-585.032`,
  kept min ESS/support high (`0.995978` / `0.999357`), and raised concurrent SPS
  to `2898.541`. This confirms the deleted planner backup surface was not a
  live intelligence path.

Collapse-repair smoke evidence:

- `collapse-repair-auxgvf-10k-oak-v2`: seed-0 HalfCheetah, NF zero,
  reward exactly matches the keeper sanity trajectory (`-44.609847`), random
  baseline remains `-2587.504`, `importanceEss=0.694`, `actorSupportEntropy=0.924`,
  `gvfCount=51`, and SPS `2039`. This is diagnostic/shadow parity, not a reward
  promotion. The first expanded-GVF attempt reused the compact live GVF head and
  regressed 10k reward (`-99.198`); it was corrected by separating the expanded
  GVFs into a shadow bank while preserving compact live GVF semantics.
- `collapse-repair-auxgvf-50k-oak`: seed-0 HalfCheetah preserved and improved
  the positive window, reaching `+172.773` cumulative reward versus random
  `-13177.211`, NF zero, `importanceEss=0.120`, `actorSupportEntropy=0.456`,
  `rewardRate=0.0429`, and about `2024` SPS. This confirms the shadow
  per-option-rate, formal-keyboard, actor-support, replacement, and expanded-GVF
  diagnostics did not suppress the keeper's early breakthrough.
- `collapse-repair-auxgvf-300k-oak`: seed-0 was stopped after the collapse
  signature reappeared. The run was still positive at 50k (`+172.283`) but had
  fallen to `-379.308` by 75k and about `-1036.906` by 125k; ESS was already
  near the one-option floor by 50k. Therefore the diagnostics are neutral, not a
  long-horizon repair.
- `collapse-repair-auxgvf-hc-10seed-50k`: HalfCheetah-only 10 seeds x
  `{oak,random}`. Oak beat random `10/10` with NF zero `10/10`, mean cumulative
  reward `-835.117` versus random `-13213.478`, mean lift `+12378.361`, final
  reward-rate positive `6/10`, but positive cumulative reward only `1/10`.
  Final ESS/entropy did not correlate strongly with reward across seeds, so the
  next architecture decision must use all-seed cumulative reward and late-window
  slope, not only seed-0 support collapse.
- `collapse-repair-badgate-10k-oak`: a live persistent bad-feature replacement
  gate drove 10k reward to `-118.271`, ESS to the one-option floor, and reinit
  count to `0`. Rejected: early replacement is doing useful work even though
  late replacement churn remains a collapse symptom.
- `collapse-repair-actorscale-50k-oak`: a live actor implicit-denominator scale
  preserved support (`importanceEss=0.517`, `actorSupportEntropy=0.892`) but
  destroyed the positive window (`-489.917` at 50k). Rejected: support repair
  alone is not intelligence if cumulative reward and reward-rate degrade.
- `filter-smoke-hc-2seed-1k`: the HalfCheetah-only benchmark path verified the
  concurrent benchmark env filter: exactly 4 work items for 2 seeds x
  `{oak,random}`, all under HalfCheetah. Future multi-seed screens should use
  `OaK benchmark` with this filter instead of sequential shell loops or
  accidental all-env sweeps.
- `current-keeper-hc-10seed-1m`: HalfCheetah-only 10 seeds x `{oak,random}` at
  1M steps on commit `baa0e9282`, rebuilt Metal and Swift first. Oak beat random
  `10/10` by paired cumulative lift, NF was zero `10/10`, but Oak final
  cumulative reward was negative `10/10`: mean Oak `-74449.6`, mean random
  `-264260.4`, mean lift `+189810.8`, median Oak `-73531.5`, and final
  reward-rate positive `0/10`. The final support signature was broad collapse:
  mean `importanceEss=0.042`, mean `actorSupportEntropy=0.001`, and mean option
  reinitializations `4963`. Heartbeat aggregates show the short-window signal
  dies out: at 50k mean reward was `-835.2` with `1/10` positive cumulative
  seeds and `5/10` positive reward-rate seeds; by 300k mean reward was
  `-11737.1` with `0/10` positive cumulative seeds and `1/10` positive
  reward-rate seed; by 1M mean reward was `-74449.4` with no positive final
  reward-rates. This confirms that the collapse is systematic, not only seed 0.
  Performance note: per-agent Oak SPS averaged `1398` under `N=8` benchmark
  contention; seeds that ran after random/earlier Oak jobs completed reached
  about `2046` SPS, so future speed analysis must separate single-stream kernel
  efficiency from multi-process GPU contention and use Instruments/diagnostic
  phase timing rather than concurrency guesswork.
- `perf-final-run-10k` / `perf-final-diagnose-10k`: performance repair after
  the 1M report kept seed-0 10k reward bit-for-bit at `-44.609847`, NF zero,
  and lifted single-stream HalfCheetah Oak to `3038` SPS. The profiling-backed
  changes were: default-gate shadow-only deep diagnostics behind
  `OAK_DEEP_DIAGNOSTICS=1`, parallelize independent planning row decodes, and
  widen action emission. Xcode Instruments trace artifact:
  `.build/perf-final-shader-1k.trace`.
- `perf-replacewide-run-50k-nohb` / `perf-random-run-50k-nohb`: seed-0 50k
  preservation check kept Oak positive at `+172.773` versus random
  `-13177.211`, with Oak at `3029` SPS and NF zero. A small concurrent paired
  check, `perf-final-concurrent-2seed-10k` (`N=4`), reported Oak finals at
  about `3454` SPS per agent for seeds 0 and 1 while beating paired random
  baselines, confirming that the single-stream 3k repair also behaves well
  under independent-agent concurrency.

Current keeper evidence after the ψ stability repair:

- `psi-gap-split-50k` showed that huge `psiStopGapMean` values were ψ
  divergence, not RR Eq. 4 stop-value explosion: final `psiAbsMean` was about
  `8.35e8`, while `rrStopAbsMean` was only about `1.84` and `valueAbsMean`
  about `0.11`. That diagnostic split was later stripped from live metrics
  after the promotion gate moved to support/action/model health only.
- `psi-implicit-only-10k` applied an implicit one-step denominator only to the
  RR ψ head. It improved seed-0 reward from `-62.028` to `-44.610`, kept NF
  zero, and kept ψ/stop magnitudes bounded (`psiAbsMean=0.54`,
  `rrStopAbsMean=0.56`). The same denominator on both ψ and `psiCross` had
  already failed, so the cross-GVF head remains explicit.
- `psi-implicit-only-50k` is the first mission-critical positive cumulative
  HalfCheetah datapoint: seed 0 reached `+117.037` cumulative reward versus
  random `-13177.211`, with a positive reward-rate pulse from about 25k to
  45k, NF zero, `terminationRate=0.708`, `optionDurationMean=1.043`,
  `importanceEss=0.812`, `lambdaEffectiveRank=12.47`, and complete queue
  coverage near the first positive crossing.
- `psi-implicit-only-50k-10seed` beat random `10/10` with NF zero `10/10`, but
  only seed 0 had positive cumulative reward by 50k. Three seeds ended with
  positive final reward-rate and eight seeds had at least one positive
  heartbeat reward-rate. This is a real signal, not promotion evidence.
- `psi-implicit-only-300k-seed0` did not sustain the 50k breakthrough:
  cumulative reward fell from `+116.81` at 50k to `-8144.016` by 300k. The
  collapse signature was importance ESS falling to the one-option floor
  (`~0.0417`), rising action/score gradients, bounded but noisier ψ/stop
  magnitudes, and continued replacement churn (`3368` option reinitializations).
  The next orthodox target is therefore the actor/keyboard/replacement/search
  interaction that preserves broad option support after early reward learning.

Rejected post-repair variants:

- Target-option score gradients were more exact in isolation but regressed the
  positive seed-0 50k run to `-576.639` and collapsed action movement. Do not
  re-add target-score actor credit without a fuller actor-critic redesign.
- ESS-scaled actor learning regressed seed-0 50k to `-525.604`; it suppressed
  the breakthrough instead of preserving it.
- Selected-only actor updates regressed seed-0 50k to `-309.030`; stable but
  too inert for early motor learning.
- Raw keyboard-score/lambda replacement protection regressed seed-0 50k to
  `-1522.678` and drove ESS to the floor. Replacement utility must not simply
  protect currently high-scoring keyboard rows.
- Option-only replacement for permanent-anchor options regressed seed-0 50k to
  `-533.364` with excessive option churn (`2083` reinitializations). The
  option-bundle renewal gap is real, but blunt anchor-option resets are not the
  repair.
- Review-response next-wave integrated rebuilds were tested and reverted rather
  than promoted. `review-response-50k-natural-133322` used exact target-option
  Gaussian score gradients and regressed to `-497.755`, with late behavior-score
  telemetry above `200`. `review-response-50k-fisher-133540` applied
  Fisher-preconditioned target scores and kept score gradients near `1`, but
  still regressed to `-461.998`. `review-response-50k-rewardlambda-133841`
  corrected the keyboard vector from differential `valueW` to the learned
  reward-cumulant descriptor (`oneReward`) and matched the 10k rebuild smoke,
  but collapsed by 50k to `-697.886`. The structural replacement gate variants
  reduced churn (`94` 10k reinitializations and `583` 50k reinitializations in
  `review-response-50k-gated-rhoopt-134244`) but suppressed the positive window
  harder (`-1047.078`). Keep the primary-source insight: Option Keyboard
  weights are cumulant/reward-feature weights, not differential value weights.
  Do not revive target-option actor gradients, combined-stop Design-B behavior,
  per-option ψ reward-rate centering, or aggressive lexicographic replacement
  as isolated swaps; they need a fuller actor/GVF/search-control derivation.

## Walking Checkpoint Evidence

The April 26 walking checkpoint changes the interpretation of all future gates.
Before this point, most evidence was about avoiding collapse while staying less
negative than random. After `.oak-benchmarks/fcstomp-encode-sync-clean-300k`,
the system has demonstrated bridge-measured locomotion with large positive
return in at least one seed. Promotion now asks whether the architecture can
make that mode reliable, not whether it can beat the random control-cost
catastrophe.

- `bc8cade80` (`Checkpoint bridge-verified FC-STOMP walking`) established the
  first walking checkpoint. `36d10e8ea` stabilized option duration updates.
  `b2d25f9fa` synchronized encode-time option selection by publishing the
  current reward-rate into threadgroup memory and adding the missing
  threadgroup barrier before lane-0 selection.
- `.oak-benchmarks/fcstomp-threadgroup-rho-50k`: 5 seeds x
  `{oak,zero,random}`. Oak mean reward `+2194.392`, max `+12351.967`,
  positive reward `2/5`, positive velocity `3/5`, mean velocity `+0.054737`.
  Seed 3 already shows a real gait by 50k: reward `+12351.967`, run reward
  `+13293.422`, mean velocity `+0.265868`.
- `.oak-benchmarks/fcstomp-encode-sync-clean-300k`: 10 seeds x
  `{oak,zero,random}`. Oak mean reward `+5365.610`, max `+109689.456`,
  positive reward `2/10`, positive velocity `6/10`, mean velocity
  `+0.037900`. Seed 3 reaches reward `+109689.456`, run reward
  `+119046.372`, and mean velocity `+0.396821`; seed 8 is the main collapse
  counterexample at `-37294.861`, run reward `-26004.084`, and mean velocity
  `-0.086680`.
- `.oak-benchmarks/grounded-deadrefresh-oak-300k-20260426`: Oak-only 10-seed
  keeper improvement. Mean reward `+5575.338`, positive velocity `7/10`, NF
  total `3`; seed 3 is bit-identical to the first walking checkpoint and seed
  0 improves from `-2553.895` to `-456.618`.
- Bridge oracle witness: `.venv/bin/python tools/bridge_oracle_cem.py
  --cem-iters 0 --eval-horizon 5000 --seed-count 5` returns mean reward
  `+4428.300`, mean run reward `+4914.978`, mean control cost `-486.678`, and
  mean velocity `+0.982996` through the same C bridge. The oracle is a witness
  and scale target only; it is not agent knowledge and must not enter the
  reward stream or action policy.
- Rejected after walking because they killed the gait or failed the 300k gate:
  pure average-reward Q selection (`fcstomp-pureq-50k`), score-based soft
  interruption (`fcstomp-softinterrupt-50k`), negative-TD chord refresh
  (`fcstomp-negtd-refresh-50k`), chord-weighted actor baseline
  (`fcstomp-chordbaseline-50k`), equality-triggered flat-bad replacement
  (`fcstomp-flatbad-300k`, exact checkpoint profile), decoded duration TD,
  positive-only keyboard output, option-only G&T, reward-aligned actor, local
  pre-update rho, and non-finite termination/key-zero patches.
- Rejected after this audit: model-score actor fallback only when RR actor
  advantages are exactly flat (`fcstomp-model-fallback-300k`). It preserved the
  seed-3 gait and slightly raised mean reward (`+5406.925` versus
  `+5365.610`) while reducing NF total (`20` versus `34`), but it did not
  improve coverage (`6/10` positive velocity, `2/10` positive return) and made
  support/model diagnostics worse (`actorSupportEntropy` mean `0.064`,
  `modelPriorityMax` mean `7.37e15`). Not a stability repair.
- Also rejected after this audit: absolute/no-baseline RR actor retest
  (`fcstomp-absolute-actor-50k`). It addressed the concern that "least bad"
  options can become positive under option-centering, but it deleted the strong
  gait: max 50k reward fell from `+12351.967` to `+779.763`, mean reward fell
  to `-399.283`, and `modelPriorityMax` mean rose to `1.10e12`.

Current strategic read: the high-return gait depends on sustained commitment
and model-backed chord pressure. The bad seeds are not merely "do nothing";
some commit to low-motion or backward modes. The next FC-STOMP work should
improve feedback/search/utility so more seeds find the same high-return
commitment, while preserving the seed-3 walking mode as a hard gate.

## Historical L3-L7 Note

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
  option/feature commits with `optFeature`, feature age, and option age.
- L7-lite: health now keeps only promotion-gate support signals:
  `terminationRate`, `optionDurationMean`, `importanceEss`,
  `actorSupportEntropy`, `actorUpdateNormMean`, reinit counts,
  non-finite counts, and model priority. RR/psi/value/lambda diagnostic gauges
  were stripped because they added hot-path surface without changing learning.

Current empirical read: `l3-l7-feature-slowclock-paired-300k-232452` is finite,
beats random `10/10`, and improves the 300k mean reward sum from the rebuilt
keeper's `-711.154` to `-489.611` while cutting feature reinitializations from
`4717.7` to `149.5`. It is not promoted: final reward-rate is still near zero
(`-0.001257` mean, `5/10` positive seeds) and late heartbeats still show
negative drift rather than sustained acceleration.

## Zoom-Out Architecture Diagnosis

The 2026-04-24 step-by-step audit changes the next-work stance. The live
L3-L7 branch is no longer missing the obvious RR stopping/duration/registry
pieces, so more nearby edits to termination thresholds, utility sums, reward
centering order, or action scaling are likely to be trinkets. The remaining
load-bearing gaps are earlier and larger.

| Plan area | Large missing piece | Evidence that this is not a local tweak |
|---|---|---|
| Step 2 feature finding | Smart feature construction/ranking across tasks, including diversity/decorrelation and downstream model/planning utility, not just random Fourier reseeding or one-row composition. | K1/K2 and utility variants were safe or neutral, but never differentiated the keyboard or improved 10M learning slope. Trace attempts blew up on correlated features. |
| Step 4 actor-critic | A clean continuing actor-critic credit path tied to the main control objective while preserving option specialization. | Direct main-TD actor, primary actor, primary option, sampled option execution, and actor traces all regressed badly; the live ψ actor is stable but not yet enough for expert learning. |
| Step 8 Prototype-AI I | A reward-useful one-step expectation/sample model with feature-ranking feedback, isolated until it proves value relevance. | L1 learned finite one-step predictions but harmed paired reward by 1.5M, so a Step-8 retry must be materially different from re-adding the same heads. |
| Step 9 search control | A state/update queue with uncertainty-aware or meta-gradient search control under a fixed per-step planning budget. | Current priority is only chord-weighted model TD magnitude. All-model updates, scalar Bellman priority, trusted-only planning, and state-ring AVI each failed without richer search control. |
| Step 10 subtask GVF planning | Planning should eventually improve subtask GVFs as well as the main value, but only with stable off-policy/GTD-style control. | Raw ψ model-planning briefly made reward-rate positive at 10k then collapsed by 50k; bounded ψ planning was stable but too weak. |
| Step 11 option keyboard | A true cumulant-space keyboard: cross-cumulant option values / GPE-GPI or a chord-descriptor model, not only scalar model-backed option/chord selection. | The Option Keyboard paper combines intentions in cumulant space. The live scalar chord can walk in some seeds, but deleted GPE-GPI/surrogate variants regressed or destabilized. Future keyboard work must be explicit and benchmarked against the walking checkpoint. |

Decision: do not spend more runs on single-line variants unless they implement
one of these architecture chunks or falsify a specific diagnostic. The next
implementation wave should be a coherent Plan-order repair, with new state or
heads admitted only when they are tied to Step 2/8/9/11 semantics and have
paired random-baseline gates.

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
against the historical K1 artifact. Higher reward is less negative.

| Metric | K1 | L1 | L1 - K1 |
|---|---:|---:|---:|
| Final cumulative reward mean | `-212.785k` | `-219.198k` | `-6.412k` |
| Final cumulative reward median | `-214.166k` | `-220.359k` | `-4.284k` |
| Seeds L1 beats K1 | n/a | `2/10` | n/a |
| Final rewardRate mean | `-0.1050` | `-0.1157` | `-0.0107` |
| Final actionDeltaL2Mean | `0.02119` | `0.02111` | `-0.00008` |
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
| 1 | Walking-stability contract | **Active hard gate** | OaK now has a real high-return walking mode. The next risk is accidentally deleting it while chasing calmer diagnostics. | Any candidate must preserve the seed-3-scale high gait at 50k/300k and improve positive-velocity/positive-return seed coverage. "Less negative than random" is no longer enough. |
| 2 | Step 2 feature resource audit | **Open architecture gap** | Feature discovery precedes stable traces, models, and option selection in the Plan. | Add diversity/decorrelation and utility-transfer diagnostics before adding more feature churn. |
| 3 | Step 8 one-step model retry | **Rolled back, redesign required** | The model learned finite predictions but reduced K1 reward on `8/10` seeds and did not create a usable planning signal. | Retry only as an isolated value-relevance/search-control substrate, not as the old L1 heads coupled directly into utility and planning. |
| 4 | Step 9 search control | **Partial** | Current priority is a scalar TD-magnitude gate; the Plan calls out state update ordering and model uncertainty. | Implement uncertainty/state-queue diagnostics before letting a redesigned model drive value. |
| 5 | L3-L6 STOMP core | **Live, walking but unstable** | RR stopping, continuation, option reward/transition/duration, registry, and scalar chord behavior are present and have produced real walking. | Keep as the keeper substrate unless paired evidence says a large architecture repair improves 300k coverage without killing the high gait. |
| 6 | L7 feature/search diagnostics | **Reduced live** | Diagnostic-only state rank, keyboard gauges, and option differential TD were deleted; promotion health keeps only support/action/model gates. | Extend only toward the Step 2/8/9 diagnostics above. |
| 7 | L8 eligibility traces retry | **Blocked** | Prior trace attempts failed from correlated Fourier features and unstable actor/model credit. | Retry only after feature diversity and a trace-stability diagnostic improve. |
| 8 | Step 11 option keyboard descriptor / GPE-GPI | **Open architecture gap** | The live scalar chord is useful but not the full cumulant-space Option Keyboard. | Reintroduce cross-cumulant/chord-descriptor machinery only with explicit value semantics and a benchmark gate against the current walking substrate. |

## Strict Status Against Plan Steps

| Plan step | Status | Current implementation | Strict audit |
|---|---|---|---|
| 1. Representation I | **Partial/strong substrate** | Welford observation normalization, per-feature alpha via MetaOptimize, GraMa utility, continual updates. | The core Step-1 pieces are real. The base-agent text also mentions short-term credit-assignment memories; TD(lambda)/implicit-trace attempts diverged on current correlated Fourier features, so traces are an open gap, not a completed item. |
| 2. Representation II | **Partial** | Direct experience anchors plus replaceable constructed feature slots; G&T composes a replacement row from two utility/decorrelation-selected parents using signed products, signed sums, or residual forms. | Step 2 is not just local feature churn. It asks for smart construction, testing, ranking, and replacement of new features made from existing features in a multi-task setting. The current substrate is live but not yet a stable walking-coverage solution. |
| 3. Prediction I | **Partial** | Main value plus K psi GVFs; multi-timescale obs traces in perception. | Continual GVF learning exists. Off-policy, stable multi-step prediction with learned feature finding and richer recurrent perception is incomplete. |
| 4. Control I | **Strong, unstable across seeds** | Continual actor-critic via wrapper score gradients, model-backed option/chord selection, and chord-blended behavior policy; Box actions keep a `2A` wrapper interface but only the mean half is live. | The actor-critic loop is now behaviorally real: it can produce high-return walking. The caveat is stability, not existence. Action parameterization remains wrapper-side research machinery; learned log-scale is reserved/inert, Box mean credit is residual-style, and full `D_policy=A` mean-only deletion failed the 1M gate. |
| 5. Prediction II | **Partial/active** | Average reward centering for main TD and psi TD; L3-L7 adds average-reward option duration. | Differential value learning is present. Duration is live but still under reward-rate screening. |
| 6. Control II | **Strong** | Continuing wrappers; `done` is not plumbed into the engine; no episodic agent semantics. | This matches the Plan discipline. Benchmark resets are wrapper events only. |
| 7. Planning I | **Partial/active** | `oak_plan` computes differential average-reward backups over K stopping-aware option models for the current state; the resulting model-backed score is behavior-relevant through chord/option selection. | OaK plans every step and this pathway has now produced walking, but still lacks a state/update queue and model uncertainty. |
| 8. Prototype-AI I | **Not implemented as a reward-useful model** | L1's action-conditioned one-step model was implemented and then rolled back after the 1.5M screen. Live code has the base option-model scaffolds, not an explicit `[state, action] -> next state/reward` Step-8 model. | Step 8 remains a real gap. The failed L1 formulation learned finite predictions but pushed reward farther from K1, so the next attempt must not simply re-add the same heads/utility coupling. |
| 9. Planning II | **Partial** | Priority signal from option-model TD magnitude; dense transition updates now cover the full option bank every step. | Prioritized-sweeping lineage is present only as a scalar priority diagnostic. The Plan also calls for uncertainty-aware control of planning order; OaK still has no model-uncertainty estimate and no state/update queue. |
| 10. STOMP | **Active, walking, incomplete** | Feature-indexed RR psi heads, option policies, stopping-aware reward/transition/duration heads, full-bank option planning, RR-derived continuation, and scalar chord blending. | These pieces are sufficient to produce real walking in some seeds, so the substrate is no longer merely speculative. It is still not a complete FC-STOMP cycle: a true keyboard/chord objective and coherent feedback over features, subtasks, options, and option models remain open. |
| 11. Oak | **Active, incomplete** | Utility estimates, age-gated option/feature replacement, feature registry metadata, model-backed scalar chord selection, and bridge-verified walking in some seeds. | Step 11 feedback is no longer unconditional churn, but richer utility assessment and full chord-descriptor Option Keyboard/FC-STOMP modeling remain open. Stable 10/10 walking is the live promotion target. |
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

- Use `optionDurationMean`, `terminationRate`, and coherent model priority to
  decide whether the next change belongs in search control, utility assessment,
  or representation.

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

- D2 priority is model TD magnitude over the live option bank.
- OaK now reports option/model priority signals, but it does not yet learn
  uncertainty or use meta-gradient search control.
- The Plan explicitly points at search control that accounts for model
  uncertainty.
- The current walking evidence splits into three cases that raw TD priority
  cannot distinguish: clean forward snowball, quiet non-finite stall, and
  high-pressure backward snowball.

Next orthodox mechanism:

- Track planning usefulness as a downstream signal, not raw surprise alone:
  whether an option-model backup improves the value/keyboard decision without
  producing non-finite arithmetic or persistent credit stalls.
- Use that signal to decide which feature/subtask/option/model bundles keep
  the fixed planning budget and survive replacement.
- Keep interruption theorem-aligned: only finite model-backed `q` versus `v`
  comparisons may break an option early; invalid RR/model arithmetic is a
  health event and search-control signal, not a behavior command.

Diagnostics required:

- uncertainty mean/variance, backup priority entropy, selected backup source,
  and planning value improvement per backup.

Acceptance:

- Must improve planning efficiency diagnostics first; reward lift is required
  at 10M before replacing F1/H1 as baseline.

Whole FC-STOMP contract before the next behavior edit:

- **Feature Construction** produces replaceable state features with parentage,
  age, activity, and downstream utility. It must not churn just because a local
  feature statistic looks bad; its utility must be credited from subtasks,
  option policies, option models, and planning.
- **SubTasks** remain RR feature-attainment GVFs: original reward cumulant, RR
  Eq. 4 stopping value, and no env-semantic shaping. The missing piece is not a
  different cumulant; it is evaluating how each learned option behaves under
  other live RR cumulants.
- **Options** keep their own RR-learned policy and termination. Interruption
  must be value-theorem aligned and finite. Invalid continuation arithmetic is
  health/search-control evidence, not a direct behavior command.
- **Models** remain average-reward option models with reward, transition, and
  duration. Raw model TD priority is a truth signal, not by itself usefulness
  or replacement protection.
- **Planning/Search Control** must judge whether option-model backups improve
  the downstream value/keyboard decision under the fixed per-step planning
  budget. It must distinguish useful surprise, useless surprise, quiet stalls,
  and confidently bad backward snowballs.
- **Option Keyboard** must become cumulant/value-space composition: either
  cross-cumulant option values plus GPE/GPI-style behavior selection, or an
  equivalent chord-descriptor model. A scalar chord over already-emitted policy
  means is only a walking substrate, not the full architecture.
- **Feedback / Remove-Replace** consumes the same bundle-level evidence used by
  search control: feature usefulness, RR subtask usefulness, option policy
  usefulness, model usefulness, and keyboard/planning usefulness. No future
  keeper may replace only one of these scores while leaving the others on the
  old contract.

Implementation consequence: the next serious FC-STOMP patch should introduce a
coherent bundle descriptor, likely centered on a fixed-budget/low-rank
approximation of cross-cumulant option values and search-control usefulness,
and route it through keyboard selection, model planning, feature utility,
option utility, and replacement together. A patch that only changes refresh
timing, damping, raw priority, interruption, or replacement scoring is now
rejected on design grounds before benchmarking. A patch that adds dense
O(K^2 R) keyboard machinery is rejected on compute grounds unless it is first
made sparse/pagewise under the existing per-step budget.

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
| 6 | L8 | Reattempt eligibility traces on decorrelated features | Blocked on L7 diagnostics; latest implicit trace attempts rejected. | 10 seeds stable, no trace blow-up, 10M lift/velocity win. |
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
| Planning-advantage actor pressure | `l3-l7-planadv-actor-paired-10k-054613`: adding the existing soft planning advantage to the option-policy TD signal was NF-clean, but regressed reward to `-29.514`, left only `1/10` positive-rate seeds, and increased action drift. | Rejected; do not add scalar planning pressure to the actor without a fuller model/control integration. |
| Structural actor responsibility floor | `l3-l7-actor-floor-paired-10k-090434`: applying the same `1/K` off-policy floor used by ψ/model heads to option-policy gradients was NF-clean and reduced chatter, but regressed reward to `-21.868`, over-concentrated the keyboard (`EffK 4.87`), lowered priority entropy (`0.502`), and inflated duration to `253.7`. | Rejected; do not floor actor responsibility without a true per-option off-policy policy-gradient/importance-ratio formulation. |
| Implicit value+ψ prediction traces | `l3-l7-value-psi-trace-paired-10k-084001` improved 10k cumulative reward to `-8.633` with NF zero, but did not produce positive final reward-rate (`0/10`) and over-concentrated the chord (`EffK 1.08`, duration `524.6`). The 50k screen `l3-l7-value-psi-trace-paired-50k-084045` failed with one seed's score-gradient explosion (`7.60e10` mean score), duration blow-up (`68k` mean), and final reward-rate `-0.075291`. | Rejected; accumulating RR ψ traces smear termination credit into the keyboard and starve search control. Retry traces only with a materially different true-online/off-policy formulation and trace-stability evidence. |
| ψ-only stopping-gated UWT trace | `l3-l7-psi-uwt-trace-paired-10k-091054`: adding RR-style ψ traces gated by continuation and off-policy chord responsibility was NF-clean but regressed reward to `-30.371`, with reward-rate still negative and no stable early acceleration. | Rejected; the useful 10k trace signal was not recovered by ψ-only stopping-gated traces. |
| Implicit value-only trace | `l3-l7-value-trace-paired-10k-084239`: removing ψ traces made the critic trace unstable by 10k, with Oak reward `-1566.129`, `0/10` positive reward-rate seeds, score-gradient `1.037e70`, and duration/model TD blow-up. The cleaner `l3-l7-tdtrace-planstep-paired-10k-084802`, which traced only reward-centered TD and kept `planAdv` one-step, failed harder: Oak reward `-5490.841`, lost to random `0/10`, action L2 `2.148`, and score-gradient `2.616e70`. | Rejected; critic traces cannot be bolted onto the current value/planning path without a stronger gradient-TD/actor-critic redesign. |
| Separate additive primary reward actor | `l3-l7-primary-reward-actor-paired-10k-044233`: adding a standalone `D_policy*R` primary actor updated by main TD over-drove actions (`actL2 2.143`), exploded score gradients (`1.275e11`), collapsed duration to one, and lost to random `0/10`. | Rejected; the missing primary-policy concern needs a stable actor-critic formulation, not a raw additive TD actor. |
| Post-plan/current-Q keyboard | `l3-l7-postplan-keyboard-slowclock-paired-10k-042522`: using current planned option Q for the chord was NF-clean but missed the 10k gate with reward sum `-20.410`, reward-rate `-0.003470`, only `1/10` positive-rate seeds, and over-concentrated chords (`EffK 7.54`). | Rejected; do not move the keyboard to current-Q ordering without a real chord-descriptor model. |
| Option-policy inheritance on replacement | `l3-l7-option-policy-inherit-paired-10k-042848`: copying policy from the most-used option worsened reward sum to `-20.802`, drove termination to `0.0906`, and inflated duration to `192.6`. | Rejected; option birth remains a clean policy reset. |
| Random option-policy birth | `l3-l7-random-policy-birth-paired-10k-085834`: initializing and replacing option-policy rows with structural random weights (`1/R`) kept NF zero and beat random, but regressed Oak reward to `-30.241`, reward-rate to `-0.003911`, and raised action magnitude to `0.154`. | Rejected; option-policy birth remains a clean zero reset until a complete keyboard/control mechanism justifies inherited or sampled behavior. |
| Actor-policy eligibility trace only | `l3-l7-policy-eligibility-trace-paired-10k-043219`: Oak lost to random `0/10` with mean reward `-6605.465` versus random `-2658.035`; action/score gradients exploded while duration collapsed to one step. | Rejected; do not add actor traces without a complete, stable actor-critic trace formulation. |
| RR stopping-bonus feature utility | `l3-l7-rrbonus-feature-utility-paired-10k-043925`: adding the RR bonus gap into registry feature utility was NF-clean and beat random, but reward sum regressed to `-22.907`, reward-rate was negative in all seeds, and option duration inflated to `157.7`. | Rejected; keep RR bonus in stopping values/diagnostics, not as a direct replacement score. |
| RR value-variation option candidate selection | `l3-l7-rrvariation-candidate-paired-10k-085246` was NF-clean and slightly improved 10k cumulative reward to `-17.345`, but the 50k gate `l3-l7-rrvariation-candidate-paired-50k-085320` regressed to `-79.288` versus the raw-duration keeper's roughly `-72.8`, with only `1/10` positive final reward-rate seeds. | Rejected; value-weight variation is a useful diagnostic, but choosing option features by the RR bonus gap alone does not create accelerating reward learning. |
| All-head feature utility | `l3-l7-allhead-feature-utility-paired-10k-053427`: extending feature utility to value planning pressure plus psi/reward/duration/model TD responsibility was NF-clean, but reward regressed to `-21.386`, chord effective-K collapsed to `6.91`, and duration inflated to `209.1`. | Rejected; richer Step-11 feature utility cannot simply sum every live head's TD magnitude. |
| Unit-norm Gaussian exploration scale | `l3-l7-unitnorm-sigma-paired-10k-044454`: changing continuous-action `sigmaBase` from `1/D_policy` to `1/sqrt(D_policy)` was NF-clean but raised control cost, regressing reward sum to `-425.529` with only `2/10` positive-rate seeds. | Rejected; the plateau is not fixed by larger initial exploration alone. |
| Priority floor for option-model search control | `l3-l7-priority-floor-current-paired-10k-044909`: using `max(chord,1/K)` in the priority signal was NF-clean but regressed reward sum to `-22.661`, left only `1/10` final positive-rate seeds, and introduced early score spikes. | Rejected; keep the pure chord-weighted priority gate until a richer uncertainty mechanism is added. |
| Pre-update reward-rate TD centering | `l3-l7-preupdate-rho-td-paired-10k-045035`: using the previous `rho`/`rhoOpt` in the same step's value/psi TD errors was research-clean and NF-clean, but regressed reward sum to `-21.363`, left only `2/10` final positive-rate seeds, and inflated option duration to `72.8`. | Rejected; keep the current update-first centering until a fuller value-based reward-centering mechanism wins paired gates. |
| Global reward-rate centering for ψ subtasks | `l3-l7-global-psi-rho-paired-10k-090745`: using the main global `rho` instead of per-option `rhoOpt` in RR ψ TDs was NF-clean but regressed reward to `-26.224`, left only `1/10` positive-rate seed, and inflated duration to `115.2`. | Rejected; per-option reward-rate traces remain useful subtask specialization in the current average-reward RR implementation. |
| Value-based per-option ψ reward-rate traces | `l3-l7-valuebased-rhoopt-paired-10k-093442`: updating `rhoOpt[k]` from the ψ TD error instead of reward EMA was NF-clean but regressed reward to `-79.801`, drove all-stop one-step behavior (`terminationRate 0.847`, duration `1.32`), and increased action energy without sustained positive reward-rate. | Rejected; keep per-option reward EMA centering for the current RR ψ heads unless a fuller off-policy GVF reward-rate estimator is added. |
| Fixed-sigma continuous policy on old L3-L7 | `l3-l7-fixed-sigma-paired-10k-045540`: freezing the old two-coordinate Gaussian scale path removed score-gradient spikes (`3.46e-3`) but starved the actor, regressing reward sum to `-115.259` with only `1/10` positive-rate seeds. | Historical rejection, not the live branch. The baseline Box action interface keeps `D_policy=2A` for capacity stability, makes the scale half inert, and uses residual-style mean credit. |
| Source-chord option-model priority | `l3-l7-source-priority-paired-10k-045727`: weighting model priority by `chordPrev` instead of the current chord was NF-clean but regressed reward sum to `-25.159`, left only `1/10` positive-rate seeds, and inflated duration to `128.8`. | Rejected; keep current-chord priority until uncertainty/meta-gradient search control replaces the simple TD-priority gate. |
| Tanh-squashed Box policy | `l3-l7-tanh-box-paired-10k-045950`: mapping continuous Gaussian samples smoothly into Box bounds was wrapper-clean and NF-clean, but regressed reward sum to `-25.538`, inflated duration to `86.0`, and reduced the random baseline's action energy without improving Oak acceleration. | Rejected; keep the current clamp wrapper until a complete bounded-policy score formulation wins paired gates. |
| Exact censored-Gaussian Box score | `l3-l7-censored-box-score-paired-10k-092013`: using the mathematically correct point-mass score for clipped Box samples was wrapper-clean and NF-clean, but regressed reward to `-20.175`, left `0/10` positive reward-rate seeds, and inflated duration to `127.7`. | Rejected; the long-run scale issue is real, but changing the Box score alone removes useful early credit. |
| Gaussian log-scale natural Fisher factor | `l3-l7-logsigma-natural-half-paired-10k-051123`: changing the Box log-scale score to the exact natural-gradient `0.5*(z^2-1)` was NF-clean and reduced score magnitude, but reward regressed to `-41.760`, final reward-rate stayed negative, and option duration inflated to `34.8`. | Rejected; scale-drift repair needs more than shrinking the log-scale score. |
| Fixed F1-style Box actor on L3-L7 | `l3-l7-fixed-f1-box-paired-10k-051306`: reverting Box actions to fixed `sigma=1/sqrt(A)` and `D_policy=A` was NF-clean and beat random, but reward collapsed to `-2028.958`, mean reward-rate `-0.183469`, and action L2 `1.03`. | Rejected as a transplant. Any future mean-only action interface must preserve cumulative reward and long-run support, not only remove the scale head. |
| Reserved primary option 0 | `l3-l7-primary-option0-paired-10k-051503`: making option 0 a non-replaceable one-step primary reward actor/model was NF-clean and beat random, but regressed reward to `-29.009`, concentrated the keyboard (`EffK 7.64`), and inflated duration to `140.8`. | Rejected; the base-policy gap is real, but this simple reservation is not the right integration. |
| Primary behavior policy option | `l3-l7-primary-behavior-policy-paired-10k-052751`: emitting only option 0 as the behavior policy, protecting it from replacement, and driving it from main TD lost to random `0/10`, with Oak reward `-3836.000`, reward-rate `-0.468356`, action L2 `1.7025`, score-gradient `4.577e2`, and duration collapse to `1.009`. | Rejected; behavior cannot be uncoupled to a raw primary actor this way. |
| Sampled one-option behavior | `l3-l7-sampled-option-behavior-paired-10k-054341`: sampling the chord into a one-hot executing option aligned behavior credit but over-drove actions (`actL2 0.541`, `delta 0.540`), collapsed duration to `1.22`, and regressed reward to `-526.937`. | Rejected; the keyboard needs value/cumulant composition, not hard one-option execution. |
| Deterministic GPI option execution | `l3-l7-deterministic-gpi-option-paired-10k-092626`: emitting the max-`qOption` base policy and assigning actor credit only to that executed option was NF-clean, but regressed reward to `-348.326`, raised action energy (`actL2 0.416`), collapsed duration to `1.82`, and kept reward-rate negative. | Rejected; GPI-style behavior needs cross-cumulant successor values/keyboard query learning, not only max main-reward option selection. |
| L2-normalized keyboard policy output | `l3-l7-l2-keyboard-output-paired-10k-092410`: interpreting the chord as a unit-L2 real keyboard vector preserved policy energy, but over-drove behavior (`actL2 0.210`, `scoreGrad 5.27e4`) and regressed reward to `-65.583`. | Rejected; the policy-blend weakness is real, but output scaling without a matching keyboard value/cumulant model is not stable. |
| Capacity-only rank changes | `l3-l7-keeper-r128-paired-10k-051732` raised effective rank but regressed reward to `-88.967`, concentrated the keyboard, and created score spikes; `l3-l7-keeper-r32-paired-10k-051808` improved speed but regressed reward to `-29.235` with no positive-rate seeds. | Rejected as a standalone fix; the plateau is not just too little or too much `R`. |
| Chord/duration-weighted option utility | `l3-l7-chordweighted-option-utility-paired-50k-051937` improved 50k reward sum to `-66.474` but failed the gate with `0/10` positive-rate seeds, near-never termination (`0.05`), and duration `591.9`; `l3-l7-durationweighted-option-utility-paired-10k-052223` still regressed 10k and left duration high. | Rejected for promotion; keep the evidence that option utility needs behavior responsibility without letting duration collapse. |
| Hard option interruption | `l3-l7-option-interrupt-utility-paired-10k-052129`: forcing continuation only for the current best-Q option fixed duration (`1.04`) but collapsed to all-stop termination (`0.997`) and reward `-68.446`. | Rejected; option interruption must be softer than hard best-option gating. |
| Standard Gaussian mean score | `l3-l7-standard-gaussian-mean-score-paired-10k-052400`: using the raw Williams mean score `z/sigma` was NF-clean but made score magnitude `80x`, raised action L2 to `0.273`, and regressed reward to `-107.868`. | Rejected; the current variance-scaled mean score remains the stable actor interface. |
| TD-error reward-rate update | `l3-l7-td-rho-paired-10k-052532`: updating `rho` from the main TD error instead of the reward EMA produced more positive-rate seeds (`5/10`) but regressed reward to `-34.753` and did not accelerate. | Rejected; keep simple reward-centering EMA unless a full differential-control branch wins paired gates. |
| Scalar Bellman-residual search priority | `l3-l7-bellman-priority-paired-10k-053224`: replacing the L2 option-model priority with the scalar differential Bellman residual lowered priority entropy (`0.721`) but regressed Oak reward to `-46.253`, left only `1/10` positive-rate seeds, and inflated duration to `91.2`. | Rejected; value-aware search control still needs richer uncertainty/credit, not only a scalar residual gate. |
| Trusted-only model planning backup | `l3-l7-trusted-model-plan-paired-10k-093212` reached a near-zero mean reward-rate at 10k, but the 50k screen `l3-l7-trusted-model-plan-paired-50k-093247` regressed reward to `-86.165` versus the keeper's roughly `-72.8`, left only `2/10` positive-rate seeds, and drove near-never termination/duration inflation. | Rejected; high-priority model error should inform search control, but simply excluding above-mean-priority options from planning starves useful backups. |
| All-option model learning with priority-gated planning pressure | `l3-l7-priority-plan-allmodel-paired-10k-094002` improved 10k cumulative reward to `-14.398` with NF zero, but final reward-rate stayed negative in all seeds and duration inflated (`297.7`). The 50k screen `l3-l7-priority-plan-allmodel-paired-50k-094041` kept cumulative reward better than the keeper (`-44.749`) but still failed positive-rate evidence (`-0.000326`, `1/10` positive), drove near-never termination (`0.022`), duration `838.5`, and late score-gradient spikes. | Rejected for source; keep the clue that experience-wide model learning can reduce control cost, but it needs uncertainty/termination repair before promotion. |
| All-option model learning with uncertainty-trusted planning | `l3-l7-step9-trust-allmodel-paired-10k` installed a real Step-9 uncertainty signal (`modelTrustMean 0.686`, `modelUncertaintyMean 0.314`) and produced positive mean final reward-rate (`+0.000821`, `6/10` positive-rate seeds), but cumulative reward regressed to `-25.784` versus the keeper's roughly `-17.936`. The 50k screen was stopped early after seed 2 showed score-gradient explosion (`2.17e16`) despite NF zero and the partial Oak mean reward-rate was negative (`-0.00210`). | Rejected; coefficient-of-variation trust is not enough to make all-option forward-model learning safe. Search control still needs a value-relevance/real-state queue or gradient-TD style safeguard before model learning is ungated. |
| Model-planned RR subtask ψ backups | `l3-l7-subtask-model-plan-paired-10k-094342` briefly produced positive mean reward-rate (`0.001765`) with non-inflated duration (`2.38`), but cumulative reward regressed to `-58.934` and score gradients spiked. The 50k screen `l3-l7-subtask-model-plan-paired-50k-094412` collapsed to `-506.674` with score-gradient `8.42e55` despite NF zero. | Rejected; planning subtask GVFs is a real RR/Oak gap, but direct additive ψ model-planning needs gradient-TD/uncertainty control before it is safe. |
| Bounded model-planned RR ψ backups | `l3-l7-bounded-psiplan-paired-10k-094750`: applying the same scale-free rational bound used by main `planAdv` stabilized the ψ-planning branch (score-gradient `1.275`, duration `9.93`, NF zero) but missed the 10k gate with reward `-20.365` versus the keeper's `-17.936` and mean reward-rate still slightly negative (`-0.000172`). | Rejected; bounded additive ψ planning is stable but too weak/noisy to promote. Revisit only with a fuller gradient-TD or uncertainty-weighted GVF planning design. |
| Capacity-normalized value-aware transition updates | `l3-l7-valueaware-transition-paired-10k-095151`: scaling transition-row updates by `R*valueW[j]^2` was inspired by value-aware model learning, but it regressed reward to `-87.954`, collapsed priority entropy to `0`, forced `modelUpdateRate 1.0`, and introduced NF counts (`28513`). | Rejected; do not weight dense transition learning directly by current value-weight magnitude without a calibrated VAML objective and stability proof/diagnostics. |
| Implicit option-model head updates | `l3-l7-implicit-modelheads-paired-10k-085109`: adding a fixed-point denominator to optReward/optTransition/optDuration updates was NF-clean and preserved action scale, but regressed Oak reward to `-22.048`, left `0/10` positive reward-rate seeds, reduced model-priority entropy to `0.645`, and kept duration high (`151.6`). | Rejected; model-head stabilization alone slows useful planning more than it helps. |
| Implicit main value update with soft planning advantage | `l3-l7-implicit-value-softplan-paired-10k-090603`: applying the average-reward implicit-TD denominator only to the value head while keeping the live soft `planAdv` was NF-clean but regressed reward to `-62.038`, raised action magnitude to `0.177`, and produced score-gradient spikes (`2.70e3` mean). | Rejected; implicit TD cannot be bolted onto the current blended value/planning target without a fuller average-reward TD formulation. |
| Neutral option-model birth | `l3-l7-neutral-model-birth-paired-10k-053655`: initializing and replacing option reward/transition models at zero avoided random model priors but regressed reward to `-28.247`, left `0/10` positive-rate seeds, and inflated duration to `124.8`. | Rejected; model hallucination is real risk, but zero birth starves the current keyboard/planning path. |
| Zero main-value prior | `l3-l7-zero-value-prior-paired-10k-053854`: initializing the main value head and RR bonus weights at zero was closer to the RR paper's tabular setup, but regressed reward to `-27.923`, suppressed termination (`0.028`), concentrated the keyboard (`EffK 7.46`), and inflated duration to `150.8`. | Rejected for the current Fourier/keyboard substrate; do not remove critic asymmetry without a broader initialization redesign. |
| Naive intra-option Q keyboard | `l3-l7-optvalue-keyboard-paired-10k-050453`: adding an average-reward `Q(s,o)` head and using it directly as the keyboard logit was NF-clean but made the chord too uniform, exploded score gradients in one seed, regressed reward sum to `-389.147`, and lost to random in `1/10` seeds. | Rejected; an option-value head remains a Plan gap, but it must not replace model-backed keyboard logits before it has a stable integration path. |
| Pure RR-activation keyboard on current keeper | `l3-l7-activation-keyboard-current-paired-10k-050638`: re-testing `zPsi` as the keyboard logit after slow feature reseeding was NF-clean but regressed reward sum to `-48.532`, left `0/10` positive-rate seeds, and inflated duration to `221.8`. | Rejected; keep model-backed keyboard logits in the current L3-L7 keeper. |
| Model-backed RR subtask keyboard logits | `l3-l7-rr-model-keyboard-paired-10k-092922`: separating main-reward planning Q from RR-subtask keyboard Q was more faithful to STOMP, but regressed reward to `-67.518`, lowered chord effective-K to `6.97`, and increased action energy without reward-rate acceleration. | Rejected; do not drive the keyboard from RR model logits until option models/cross-cumulant values are reliable enough. |
| Wrapper score/action scaling variants | Gaussian-score standardization, per-coordinate actor alpha, per-coordinate standard score, and behavior-policy birth variants regressed 10k/50k paired screens. | Do not add score normalizers or actor-gain knobs. The baseline action change deletes learned scale; it is not a new tuning layer. |
| Tracking-rate substitutions | `rho`, global-rho, and utility-timescale variants tied to `R` did not improve early reward-rate or 50k reward evidence. | `track_rate(t, obsDim)` remains the current structural tracker, not a research invariant. |
| Raw observation passthrough | `l3-l7-rawobs-paired-10k-230923`: NF zero but worse cumulative reward (`-20.933`) and only `1/10` positive reward-rate seeds. | Rejected; keep standardized observation input to the Fourier encoder. |
| Chord-consistent value backup | `l3-l7-chordmodel-plan-paired-10k-231520`: replacing best-option planning advantage with chord-weighted Q regressed reward sum to `-29.654`. | Rejected; Design A policy blending does not imply chord-weighted value backup here. |
| Current-state planAdv gradient | `l3-l7-curr-planadv-paired-10k-231730`: NF zero but reward sum `-23.689`, `0/10` positive reward-rate seeds, and SPS collapse. | Rejected; keep value update on the source-state gradient. |
| Source-state planning Q cache | `l3-l7-source-planq-paired-10k-090211`: splitting current-state `qOption` for the keyboard from source-state `planQOption` for value planning was a cleaner AVI mapping, but regressed reward to `-26.874`, lowered priority entropy to `0.711`, and inflated duration to `120.9`. | Rejected; a true state-search planner needs a broader state/update queue and keyboard model, not only source-aligned value pressure. |
| Sampled state-ring AVI planning | `l3-l7-state-ring-avi-paired-10k-091520`: adding an `R`-state feature ring, sampled-state option backups, and value pressure on the sampled planning state was NF-clean and beat random `10/10`, but regressed reward to `-25.695` versus the keeper's `-17.936`, left only `1/10` positive reward-rate seed, and showed no last-5k acceleration. | Rejected; do not add state-search AVI without a richer search-control/model queue and a keyboard/chord model that can consume the sampled backups. |
| Tracking RR bonus | `l3-l7-tracking-rrbonus-paired-50k-232052`: 10k was superficially close, but 50k regressed to reward sum `-247.071` and duration collapse. | Rejected; keep RR bonus as the high-water value weight from the feature's current identity. |
| Full option/feature capacity clock | `l3-l7-capacityclock-replace-paired-10k-232222`: feature and option replacement both slowed to full capacity, but termination fell to zero and reward regressed. | Rejected; feature reseeding is the destructive clock to slow, not option replacement wholesale. |
| Unused-only option replacement / candidate-pressure option clock | `l3-l7-unusedonly-replace-paired-10k-233410` and `l3-l7-candidatepressure-replace-paired-10k-233538`: both reduced option churn but drove termination near zero and worsened 10k reward. | Rejected; the next utility repair needs richer planning usefulness, not a harder candidate gate. |
| Centered reward encoder input | `l3-l7-centered-reward-encoder-paired-10k-005316`: research-clean average-reward perception repair, but 10k reward sum fell to `-26.474` with only `1/10` positive-rate seeds. | Rejected; keep raw reward as part of the experience input unless a broader perception model changes. |
| Standardized reward encoder input | `l3-l7-standardized-reward-encoder-paired-10k-054106`: adding online reward variance normalization to the encoder reward channel was NF-clean but regressed reward to `-21.045`, left only `1/10` positive-rate seeds, concentrated the chord (`EffK 8.89`), and inflated duration to `99.0`. | Rejected; reward-scale perception repair is not enough and should not replace the raw experience reward channel in this substrate. |
| No reward encoder input | `l3-l7-no-reward-encoder-paired-10k-092159`: removing the previous-reward coordinate from the Fourier encoder was NF-clean but regressed reward to `-24.099`, left only `1/10` positive-rate seed, and weakened continuation (`terminationRate 0.047`, duration `109.0`). | Rejected; keep raw reward in the experience representation while reward still drives the TD targets. |
| Direct policy output without the action trace | `l3-l7-direct-policy-output-paired-10k-005406`: removing the `0.5` output trace worsened reward sum to `-27.419` and drove score-gradient mean to `3.206e3`. | Rejected; direct output is not the plateau fix. |
| Decoded duration TD | `l3-l7-decoded-duration-td-paired-50k-005528`: 10k had a positive-rate pulse, but 50k reward sum regressed to `-80.617` versus the raw-duration keeper's `-72.800`. | Rejected; keep raw Bellman duration TD with the lower bound only at planning decode. |
| Duration-scaled planning advantage | `l3-l7-duration-scaled-planadv-paired-10k-054503`: dividing best-option planning advantage by the selected option duration was average-reward motivated, but regressed reward to `-27.136`, left only `1/10` positive-rate seeds, and kept duration inflated around `101`. | Rejected; duration scaling cannot be applied only at the value-head pressure point in the current planner. |
| Stratified initial option-feature registry | `l3-l7-stratified-registry-birth-paired-10k-005737`: spreading initial anchors across `R` worsened reward sum to `-23.227` and `0/10` positive-rate seeds. | Rejected; keep the current `optFeature[k] = k` birth rule. |
| Implicit TD denominator plus raw planning advantage | `l3-l7-implicit-raw-planadv-paired-10k-005940`: replacing the soft planning advantage with raw advantage and an implicit value update regressed to `-44.286` and score-gradient mean `2.737e1`. | Rejected; raw planning pressure remains damaging even with implicit shrinkage. |
| Fixed current Gaussian sigma | `l3-l7-fixed-current-sigma-paired-10k-010112`: freezing scale coordinates at the then-current structural sigma collapsed reward to `-115.259` and removed useful score signal. | Historical rejection. The baseline differs by keeping capacity shape, zeroing reserved scale scores, and using residual-style mean credit; the promoted paired artifact is `.oak-benchmarks/baseline-1m`. |
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

Concurrent seed sweeps:

- Use `OaK benchmark` with `OAK_BENCH_ENV=HalfCheetah-v4` for HalfCheetah-only
  seed sweeps. The parent owns a workspace lock, launches independent worker
  processes at structural max concurrency (all logical cores), and
  deterministically refills finished slots from the canonical seed/policy work
  list.
- Do not fall back to sequential seed loops for ordinary multi-seed evidence
  unless a profiler or crash investigation requires serialized isolation.

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

- **Current behavioral baseline**: grounded dead-credit refresh in the live
  working tree, artifact `.oak-benchmarks/grounded-deadrefresh-oak-300k-20260426`.
- HalfCheetah 300k, 10 Oak seeds: mean reward `+5575.338`, max
  `+109689.456`, positive velocity `7/10`, positive return `2/10`, and Oak
  non-finite total `3`. This is an improved walking checkpoint, not a stable
  promotion.
- **Historical negative floor**: `.oak-benchmarks/baseline-1m` remains useful
  for host/performance comparison only. It is no longer the behavioral target.
- K1 and rebuild-era runs remain historical comparison artifacts, not baselines.

The roadmap objective is now sharper: stabilize and generalize the bridge-seen
walking mode while repairing the missing Plan-order pieces, especially
feature/resource feedback, search control, exact subtask/option machinery, and
any future Step-8 model design that can prove reward relevance before it is
allowed into planning. More feature churn or calmer actor scaling without those
repairs is unlikely to turn a 2/10 positive-return gait into 10/10 walking.
