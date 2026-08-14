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
5. **No tunable knobs.** Constants are derived from capacity, stream shape, or hardware; step sizes are meta-learned (MetaOptimize). A number you want to tweak means a mechanism you should remove.
6. **No caps on magnitudes.** The world is bigger than the agent, so magnitudes carry information (Big World Hypothesis). Never clamp; preserve on non-finite.
7. **Greenfield. There is only one way things work.** Never keep a second way of doing something around to cover for the first — replace and delete. If data that should be there is missing, fail loudly; never quietly try something else. Every file has exactly one layout, with no version numbers and no readers that check "which kind is this?": when a layout changes, convert the existing data once and delete every trace of the old one, code and words alike. Nothing anywhere — no code, no setting, no message shown to a person — may mention or handle an "old" or "legacy" anything. Old data, once converted, is just data.
8. **Fleet lanes are independent agents.** One stream per seed, zero cross-lane coupling. Lane-disjointness is the acceptance test for every engine change.
9. **Proof matches risk.** Changes that preserve every floating-point op ship on a bitwise A/B check against the keeper fleet. Changes that reorder arithmetic ship only on the paired population gate against the keeper fleet.
10. **No commentary, only code.** No comments, no docstrings, no doc files of any kind. Names and structure carry all meaning; if code needs a comment to be understood, rewrite the code. Knowledge that can't live in code lives in git history, skills, or memory — never in the repo as prose.
11. **No environment cherry-picking under any circunstances** Use every environment and every seed the keeper uses. Using only some of them does not count. Do not write agent code that only works for one environment. To match the keeper, all of them have to match. It doesn't matter if you are doing an intermediate test or just a "quick check". Slicing arbitrary enviroment subsets are never acceptable. Every decision must be made holistically with all of the enviroments in mind. No matter the circumstances. 


## Performance: profiling (HEADLESS — verified 2026-07-04)

The repo template records without the Instruments GUI. Never trace (or rebuild the metallib) while a fleet run is in flight.

```
xctrace record --template ShaderTemplate.tracetemplate \
  --output out.trace --time-limit 15s \
  --env OAK_BENCH_POLICIES=oak \
  --env OAK_BENCH_STEPS=200000 --env OAK_BENCH_SEED_COUNT=1024 \
  --env OAK_METRICS_STORAGE=/tmp/trace-run \
  --launch -- .build/release/OaK benchmark
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
