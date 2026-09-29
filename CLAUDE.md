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
4. **One reward stream, plus learning feels good.** No discounting — reward is centered instead (Reward Centering). The agent may reward itself for its own learning progress, measured as weight change under meta-learned step sizes (Sutton 2025 options talk; Linke et al. 2020). No novelty, count or prediction-error bonuses; subtasks respect the main reward (Reward-Respecting Subtasks). Posterior sampling and optimism in value estimates are not bonuses and are allowed: they choose actions and never touch the reward.
5. **No tunable knobs; capacity is not a knob.** Behavioural constants are derived from stream shape or hardware, or meta-learned; step sizes are meta-learned (MetaOptimize). Capacity sets width, never time: no rate, block length or budget derives from it, so enlarging the agent never slows its learning. A behavioural number you want to tweak means a mechanism you should remove. Model geometry is different in kind: it is the declared capacity that sets the agent's size, not a dial that tunes behaviour. Enlarging the agent is legitimate engineering, and this rule is never an argument against it.
6. **No caps on magnitudes.** The world is bigger than the agent, so magnitudes carry information (Big World Hypothesis). Never clamp; preserve on non-finite.
7. **Greenfield. There is only one way things work.** Never keep a second way of doing something around to cover for the first — replace and delete. If data that should be there is missing, fail loudly; never quietly try something else. Every file has exactly one layout, with no version numbers and no readers that check "which kind is this?": when a layout changes, convert the existing data once and delete every trace of the old one, code and words alike. Code that exists to perform that one-time conversion is itself a trace: it is written, run over everything, and deleted within the same change — one-time code that survives is a second way. Nothing anywhere — no code, no setting, no message shown to a person — may mention or handle an "old" or "legacy" anything, and no writing about the system may attach version labels to anything. Old data, once converted, is just data.
8. **Fleet lanes are independent agents.** One stream per seed, zero cross-lane coupling. Lane-disjointness is the acceptance test for every engine change. A try's lanes must be bitwise equal to the same binary's lanes in the full fleet; nothing a lane allocates or computes depends on how many lanes run.
9. **Proof matches risk.** Changes that preserve every floating-point op ship on a bitwise A/B check against the best fleet. Changes that reorder arithmetic ship only on the paired population gate against the best fleet.
10. **No commentary, only code.** No comments, no docstrings, no doc files of any kind. Names and structure carry all meaning; if code needs a comment to be understood, rewrite the code. Knowledge that can't live in code lives in git history, skills, or memory — never in the repo as prose.
11. **No environment cherry-picking under any circumstances.** Every run uses every environment. `./fleet.py try` runs the first seed of every environment's block at the full horizon against the recorded best; when that screen is BETTER or a close call (no signal while mean, IQM and median all improve), it confirms on the next three seeds of every block, still against the best, and it becomes champion only if the confirmation is BETTER and not worse than the champion's own confirmation run. The judge tests seeds 4 to 31, leaving out the seeds tries select on; every other run uses every seed the best uses. Seeds are never chosen or changed after seeing results. Do not write agent code that only works for one environment. Every verdict uses one statistic: per-environment normalized scores, a sign test over the suite and the eight catalog categories as one pre-declared family with Holm correction, and for promotions the per-environment Holm–Wilcoxon regression guard. No other subset is ever tested; every decision weighs all environments.
12. **No branches or commits without explicit user authorization.** Work only in the single current working tree so the user can skim it live in VS Code. Never create a branch, never commit, and never propose a plan that contains branch or commit stages. Only branch or commit when the user has explicitly authorized it in that turn.
13. **Reuse the recorded best and matching baselines.** A promotion compares against the recorded best, and a speed promotion also against the reward anchor, the last reward promotion or adoption, so small losses cannot creep in one speed promotion at a time. A try compares against the best's own lanes on the same seeds, reading the best's baselines at those lanes; a candidate champion is also compared against the champion's saved confirmation run. Never launch a fresh reference, source-machine run or baseline merely for comparison. If a candidate cannot match the keeper’s required source/schedule, report the mismatch rather than manufacture a new reference. Candidate runs still use `./fleet.py`. Never re-record matching `random`, `constant`, `ppo` and `oracle` baselines. They are the yardstick: recorded once for a given environment, seed, horizon, and lane schedule, then reused by every science run on that schedule. Normalization runs from the better of Random and the constant do-nothing policy up to the Oracle. Do not re-record them as part of a science run — reuse is the point of a baseline. Record only when no complete matching baseline exists, and a baseline changes only when the environment it measures changes: the harness files every baseline per environment, under that environment's seeds, the horizon, the policy's source and the digest of the environment's sources, and every run links to it. Missing or mismatched baselines fail the run.
14. **No slice cherry-picking by the agent. Sub-1M tests say nothing about reward.** Reward is judged only at the best's 1,000,000-step horizon: a try on the first seed of every environment steers iteration; only the full fleet promotes. Never gate, promote, decide, or volunteer a prefix, a peak window, a late window, or any other time slice as the result. A run shorter than 1M steps can tell you the engine started; it cannot tell you about reward. Longer runs of every environment may be recorded for report, never gated, but only once throughput makes them cheap and a proven case says they are worth the machine time. This rule binds the agent, not the user: when the user asks to see a slice, show it immediately, in full, without argument or preamble, labelled once as the prefix it is. The user asking is not an invitation to re-litigate the rule, and one label is enough.
15. **Model geometry is data, declared once.** Every number that sets the agent's shape — lane width, perception budget, gate banks and groups, option count, planning period, and everything derived from them — lives in `Sources/geometry.h` and nowhere else. No consumer restates a derivation: the host reads shape back through the shape probe rather than recomputing it, and a `static_assert` pins every identity so drift fails the build. A geometry number that merely equals a hardware number must derive from it explicitly and visibly; coincidence is not derivation, and the wave width is never permitted to set the agent's capacity by accident.
16. **Never opt out of a recording.** `./fleet.py` is the only way to exercise a change. It snapshots the working tree into a queue and returns at once; one background runner builds each snapshot, times it against the recorded best with the paired probe, and rejects without recording any candidate that can never promote — a binary `cmp`-identical to the best's, or one whose throughput falls below half the budget anchor's (the best recorded when the edicts were sealed, `232cf8f18`; each recording's `speed.json` chains its ratio to that anchor). Every candidate for promotion is recorded at the best's schedule and horizon, reuses matching baselines, is reported, and is promoted when it is BETTER on reward with the regression guard silent, or at least 1% faster and not worse against both the best and the reward anchor, or bitwise identical to the best on every lane's reward stream and not measurably slower (Rule 9), or not worse against both the best and the reward anchor, not measurably slower, and smaller in the learner's source (a simplification), or not worse against both with fewer failing fuzz runs than the best (a repair); a candidate with more failing fuzz runs than the best is rejected, and a judge that rejects the champion dethrones it; verdicts land in `.oak-benchmarks/verdicts.log`. `./fleet.py try` is the one smaller run: the first seed of every environment, confirmed on three more seeds when it screens BETTER or close, full horizon, never timed, never promotes. `./fleet.py check` (every environment's functional self-check and the learner's fuzz test) and `./fleet.py invariance` (permuted channels, relabeled actions, rescaled channels, compared with the plain try) are instruments that make no claim about reward. Never hand-roll a benchmark request to OaK to check whether something worked, never point a run root outside `.oak-benchmarks/`, and never let a scratch run, a decoded pose tape, a rebuilt older tree, or a reference simulator stand in for the recording. Any other scratch run is no gate. When the harness itself is broken, fix the harness and rerun the gate — never route around it. The exceptions to recording are the timing instruments under Performance, which are wall-clock probes and make no claim about reward, and the try, which may report IDENTICAL only for the lanes it ran; the paired probe may report that the candidate's reward streams diverge from the best's, never that they match.
17. **Never push without the user's explicit approval in that turn.** Benchmark recordings stay local and are ignored by Git. Nothing uploads them.
18. **Never call `clear_session`** (`mcp__ccd_session_mgmt__clear_session`). It wipes the whole conversation.


## Performance: timing (verified 2026-09-21)

Throughput drifts with temperature, so **sequential before/after timing is inadmissible**. Ten consecutive 20k-step probes of one identical binary measured 3.41M-3.78M steps/s: 10.6% peak-to-peak, but adjacent runs agreed to 0.5%. A speed promotion needs 1%; a reward promotion needs only to stay within the compute budget, at least half the budget anchor's throughput.

The instrument is an interleaved paired probe: both binaries alternate `oak record oak 20000` in ABBA blocks and the median of per-block ratios decides. It is the speed stage of every gate (the ratio against the best, times the best's own ratio to the budget anchor, is the candidate's budget ratio), and runs on its own without recording:

```
./fleet.py probe
```

7840 lanes, ~37 s per probe, all 245 environments, no claim about reward. After the first pair it reports how many lanes' reward streams already diverge from the best's. The stage is stored per binary pair, so a gate that follows its probe reuses it. Keep the machine quiescent: the runner is the only thing that builds, times or records; it pauses a fleet run while a probe job runs and stops every other `oak` process, such as a theater replay, for each timed run. Never let a subagent build or run anything during a probe. `./fleet.py trace` queues the same kind of job to profile the working tree's engine with Instruments' Time Profiler for 10 s of a probe-length run and print its hottest functions and instruction addresses; it makes no claim about behaviour or reward. A runner refuses to start at reduced priority, because other processes then steal cycles unevenly from a probe: zsh nices background jobs, so launch `./fleet.py` in the background only after `setopt no_bg_nice`.

Every recording carries a `source/` snapshot so its binary can be rebuilt and timed against a candidate in the same moment. A recording without one cannot serve as a timing reference.

A paired probe compares binaries, not sources. A cold-code edit that moved byte-identical hot functions 216 bytes measured 0.968x over three blocks, three times the margin. Before believing a speed verdict, diff the hot functions between candidate and reference with `objdump -d --disassemble-symbols=_oak_act`; identical bodies mean the verdict is layout, not code. A candidate whose binary is `cmp`-identical to the best's needs no probe and no recording. `setup.sh` aligns every function to `hw.cachelinesize`: 64-byte alignment slowed Pendulum-slip 43%. Absolute addresses still move with every relink: linking Craftax slowed byte-identical Acrobot 5% in all six paired blocks.

The per-stage falsification evidence log (SPS program) lives in git history: `git log --all -- docs/roadmap.md`. It describes the deleted GPU engine; its absolute numbers do not transfer, but its measurement law does.

## Citations (arXiv; fetch on demand)

- Alberta Plan: [2208.11173](https://arxiv.org/abs/2208.11173)
- Reward-Respecting Subtasks: [2202.03466](https://arxiv.org/abs/2202.03466)
- Reward Centering: [2405.09999](https://arxiv.org/abs/2405.09999)
- Planning and Action Selection in Options-based Agents (Sutton 2025): [slides](http://incompleteideas.net/Talks/TTT-options-2025.pdf) — learning-progress reward; activate options, never execute them
- Adapting Behavior via Intrinsic Reward (Linke-Ady-White-Degris-White 2020): [1906.07865](https://arxiv.org/abs/1906.07865)
- Option Keyboard Basis: [2505.00787](https://arxiv.org/abs/2505.00787)
- Generate-and-test: Mahmood-Sutton 2012 Autostep (ICML), Rahman 2020 thesis, Samani 2022 thesis
- MetaOptimize per-feature step sizes: Sharifnassab-Salehkaleybar-Sutton 2024 ([2402.02342](https://arxiv.org/abs/2402.02342)). Treat this as the controlling source for `alpha/beta/h/gradEnergy`; IDBD/Autostep are historical special cases, not the implementation target.
- Big World Hypothesis: [2512.23419](https://arxiv.org/abs/2512.23419)
- Three Dogmas of RL: [2407.10583](https://arxiv.org/abs/2407.10583)
- Meta-gradient search control: [2406.19561](https://arxiv.org/abs/2406.19561)
- Sutton-Barto 2nd ed: §2.6 (tracking-rate EMA), §6 (1-step TD stability), §8.4 (prioritized sweeping with Dyna), §9.5.4 (online normalization)
