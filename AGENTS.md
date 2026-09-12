# AGENTS

## The Alberta Plan


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

### What the Plan eschews

- Nothing about the human world (vision, space, objects, language, geometry, other agents).
- Learning first, domain knowledge last.
- No discounting 
- Actions and observations are signals with no intrinsic meaning.

## Hard rules (project invariants)

When a rule drifts from the cited research, fix the rule — never preserve folklore.

1. **The agent knows nothing about its environment.** Observations and actions are meaningless signals (Alberta Plan). Everything env-specific — identity, action-space type, bounds — lives in the environment layer; the agent sees only dimensions.
2. **No episodes.** The agent lives one continuing stream; resets are environment events it never sees.
3. **Every step is the same.** All mechanisms — perception, planning, learning, meta-learning, generate-and-test — run every step, identically (Alberta Plan, pillar 2). No warmup, no phases, no train/eval split, no frozen weights.
4. **One reward stream.** No discounting — reward is centered instead (Reward Centering). No curiosity or progress bonuses; subtasks respect the main reward (Reward-Respecting Subtasks).
5. **No tunable knobs; capacity is not a knob.** Behavioural constants are derived from capacity, stream shape, or hardware; step sizes are meta-learned (MetaOptimize). A behavioural number you want to tweak means a mechanism you should remove. Model geometry is different in kind: it is the declared capacity those derivations consume, not a dial that tunes behaviour. Enlarging the agent is legitimate engineering, and this rule is never an argument against it.
6. **No caps on magnitudes.** The world is bigger than the agent, so magnitudes carry information (Big World Hypothesis). Never clamp; preserve on non-finite.
7. **Greenfield. There is only one way things work.** Never keep a second way of doing something around to cover for the first — replace and delete. If data that should be there is missing, fail loudly; never quietly try something else. Every file has exactly one layout, with no version numbers and no readers that check "which kind is this?": when a layout changes, convert the existing data once and delete every trace of the old one, code and words alike. Code that exists to perform that one-time conversion is itself a trace: it is written, run over everything, and deleted within the same change — one-time code that survives is a second way. Nothing anywhere — no code, no setting, no message shown to a person — may mention or handle an "old" or "legacy" anything, and no writing about the system may attach version labels to anything. Old data, once converted, is just data.
8. **Fleet lanes are independent agents.** One stream per seed, zero cross-lane coupling. Lane-disjointness is the acceptance test for every engine change.
9. **Proof matches risk.** Changes that preserve every floating-point op ship on a bitwise A/B check against the best fleet. Changes that reorder arithmetic ship only on the paired population gate against the best fleet.
10. **No commentary, only code.** No comments, no docstrings, no doc files of any kind. Names and structure carry all meaning; if code needs a comment to be understood, rewrite the code. Knowledge that can't live in code lives in git history, skills, or memory — never in the repo as prose.
11. **No environment cherry-picking under any circunstances** Use every environment and every seed the best uses. Using only some of them does not count. Do not write agent code that only works for one environment. To match the best, all of them have to match. It doesn't matter if you are doing an intermediate test or just a "quick check". Slicing arbitrary enviroment subsets are never acceptable. Every decision must be made holistically with all of the enviroments in mind. No matter the circumstances. 
12. **No branches or commits without explicit user authorization.** Work only in the single current working tree so the user can skim it live in VS Code. Never create a branch, never commit, and never propose a plan that contains branch or commit stages. Only branch or commit when the user has explicitly authorized it in that turn.
13. **Reuse the recorded best and matching baselines.** Every comparison uses the existing official best/keeper recording as its reference; never launch a fresh reference or source-machine run merely for comparison. If a candidate cannot match the keeper’s required source/schedule, report the mismatch rather than manufacture a new reference. Candidate runs still use `./fleet.py run`. Never re-record matching `random` and `ppo` baselines. They are the yardstick: recorded once for a given environment, seed, horizon, and lane schedule, then reused by every science run on that schedule. Do not re-record them as part of a science run — reuse is the point of a baseline. Record only when no complete matching baseline exists, and a baseline changes only when the environment it measures changes: the harness files every baseline under its schedule and the digest of the environment sources, and every run on that schedule links to it. Missing or mismatched baselines fail the run.
14. **No slice cherry-picking by the agent. Sub-1M tests say nothing about reward.** Reward is judged only at the best's 1,000,000-step horizon on the full fleet. Never gate, promote, decide, or volunteer a prefix, a peak window, a late window, or any other time slice as the result. A run shorter than 1M steps can tell you the engine started; it cannot tell you about reward. This rule binds the agent, not the user: when the user asks to see a slice, show it immediately, in full, without argument or preamble, labelled once as the prefix it is. The user asking is not an invitation to re-litigate the rule, and one label is enough.
15. **Model geometry is data, declared once.** Every number that sets the agent's shape — lane width, perception budget, gate banks and groups, option count, planning period, and everything derived from them — lives in `Sources/Geometry.metal` and nowhere else. No consumer restates a derivation: the host reads shape back through the shape probe rather than recomputing it, and a `static_assert` pins every identity so drift fails the build. A geometry number that merely equals a hardware number must derive from it explicitly and visibly; coincidence is not derivation, and the wave width is never permitted to set the agent's capacity by accident.
16. **Never opt out of a recording.** `./fleet.py run` is the only way to exercise a change: it records the `oak` candidate at the best's schedule and horizon, reuses matching `ppo` and `random` baselines, reports, and gates against the recorded best. Never hand-roll a benchmark request to OaK to check whether something worked, never point a run root outside `.oak-benchmarks/`, and never let a scratch run, a decoded pose tape, a rebuilt older tree, or a reference simulator stand in for the recording. A scratch run is not a smaller gate; it is no gate. When the harness itself is broken, fix the harness and rerun the gate — never route around it. The single exception is the timing instruments under Performance, which are wall-clock probes and make no claim about behaviour or reward.
17. **Never push without the user's explicit approval in that turn.** Benchmark recordings stay local and are ignored by Git. Nothing uploads them.


## Performance: profiling (HEADLESS — verified 2026-07-04)

The repo template records without the Instruments GUI. Never trace (or rebuild the metallib) while a fleet run is in flight.

```
xctrace record --template ShaderTemplate.tracetemplate \
  --output out.trace --time-limit 15s \
  --env OAK_BENCH_POLICY=oak \
  --env OAK_BENCH_STEPS=200000 --env OAK_BENCH_SEED_COUNT=1024 \
  --env OAK_METRICS_STORAGE=/tmp/trace-run \
  --launch -- .build/release/OaK
xctrace export --input out.trace --toc                     # list tables
xctrace export --input out.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="SCHEMA"]' --output t.xml
```

Useful schemas: `metal-gpu-intervals` (per-dispatch GPU times), `gpu-performance-state-intervals` (clock derating, "Minimum … due to active device conditions"), `device-thermal-state-intervals` (thermal axis for SPS-decay questions). Caveat: `metal-shader-profiler-*` and `gpu-counter-*` tables exist but record ZERO rows with this template (shader-profiler/counter profiles are unarmed); arming them needs a one-time GUI template edit — bare `--instrument 'Metal GPU Counters'` records nothing in attach OR launch mode.

Complementary in-code instruments: `OAK_MEGA_STAGE_MASK` (per-stage timing; single-stage-only masks isolate cost but DEGENERATE the data — prefer `encode+X` pairs under live dynamics), `OAK_PROBE_FIXEDIDX=1` (fixed contiguous gather indices — separates memory-scatter cost from issue cost). Both are timing instruments; trajectories are garbage. The per-stage/falsification evidence log (SPS program) lives in git history: `git log --all -- docs/roadmap.md`.

## Citations (arXiv; fetch on demand)

- Alberta Plan: [2208.11173](https://arxiv.org/abs/2208.11173)
- Reward-Respecting Subtasks: [2202.03466](https://arxiv.org/abs/2202.03466)
- Reward Centering: [2405.09999](https://arxiv.org/abs/2405.09999)
- Option Keyboard Basis: [2505.00787](https://arxiv.org/abs/2505.00787)
- Generate-and-test: Mahmood-Sutton 2012 Autostep (ICML), Rahman 2020 thesis, Samani 2022 thesis
- MetaOptimize per-feature step sizes: Sharifnassab-Salehkaleybar-Sutton 2024 ([2402.02342](https://arxiv.org/abs/2402.02342)). Treat this as the controlling source for `alpha/beta/h/gradEnergy`; IDBD/Autostep are historical special cases, not the implementation target.
- Big World Hypothesis: [2512.23419](https://arxiv.org/abs/2512.23419)
- Three Dogmas of RL: [2407.10583](https://arxiv.org/abs/2407.10583)
- Meta-gradient search control: [2406.19561](https://arxiv.org/abs/2406.19561)
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §8.4 (prioritized sweeping with Dyna), §9.5.4 (online normalization)
