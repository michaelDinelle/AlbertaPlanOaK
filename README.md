# OaK — Options And Knowledge (ROUGH DRAFT)

## This is an implementation of [Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)(https://arxiv.org/abs/2208.11173) (Sutton, Bowling & Pilarski,2022):



A native Apple Silicon implementation of **Step 11** of the [Alberta Plan for
AI Research] the **Oak architecture** — *Options And Knowledge* — built around a
continually-running **FC-STOMP** loop. Behavior emerges from the loop running
every time step on the live `(observation, action, reward)` stream; there is
no training/eval split and no episode boundary inside the agent. The code in the current state was derived from over a month of trial and error with various language models from MuseSpark , Gemini 3.1 pro, Gemma4, Claude and gpt 5.5, gpt 5.5 pro.

## Compared to modern Language Models. the goal of the OaK architecture is to keep our [`EyesOnThePrize.pdf`](docs/papers/EyesOnThePrize.pdf)


The Alberta Plan exists in service of a single long-term goal Sutton calls
*the Prize*: understanding the principles of intelligence well enough to
create (or become) beings of greater intelligence (see
. In that framing, intelligence is
*"the computational part of an agent's ability to predict and control a
stream of sensations, particularly a designated numerical sensation (called
reward), while interacting with a vastly more complex world."* Everything
the agent ever knows arrives through the **narrow aperture of experience** —
the alternating stream of actions, sensations, and reward — and it must do
its work in real time, against a world it can never fully contain.

OaK is a working bet on the four commitments that follow from that goal:

1. **Intelligence as real-time signal processing to maximize reward** under
   bounded compute — not offline optimization on a curated dataset.
2. **Continual learning from ordinary, unprepared experience** — no
   training/eval split, no episode reset inside the agent, no demonstrations
   or special training distributions.
3. **[The Big-World Hypothesis](https://openreview.net/pdf?id=Sv7DazuCn8)** —
   the agent is and always will be vastly less complex than its world, so
   every component must approximate, track, and be revisable, even in the
   limit.
4. **Continual construction of abstractions** — features, subtasks, options,
   GVFs — so the agent can think simply about long spans of time without
   collapsing them into a fixed scale.

Concretely: the FC-STOMP loop runs every step on the live stream; features,
subtasks, options, and option models are generated, scored by utility, and
removed-and-replaced when they fail to pull weight; planning is average-reward
so there is no privileged episode horizon. The point of the repo is not a
benchmark number — it is to keep these commitments load-bearing in actual
code on a real continuous-control stream, and to fail loudly when any one of
them quietly slips.

### Sutton and Bowlings Architecture adddresses "The Problem With AI Nobody Wants to Face"

[![Watch on YouTube](https://img.youtube.com/vi/qqXSpWYSxBw/hqdefault.jpg)](https://www.youtube.com/shorts/qqXSpWYSxBw)

### `HalfCheetah-v4` Currently Oak Beats the Random baseline 10/10 how ever only recieves positve reward in a hand full of seeds

Environment: [`HalfCheetah-v4`](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)
(MuJoCo, Gymnasium spec) — driven through our envpool-backed C++ bridge (see
[The envpool bridge](#the-envpool-bridge) below).

Seed three has been observed to learn to be a remarkably robust learner however it suffers from the classic catastrophic loss of placicity/knowlege in the current form of the code base.

#### Three replays, same setup

All three clips below are **seed 3** on `HalfCheetah-v4`, **500,000 steps**,
rendered through `tools/replay_render.py` at 60 fps. Only the policy changes —
so the contrast between them is purely a contrast between *what each policy
does to the cheetah*.

| Policy | File | What you should see |
|---|---|---|
| **OaK** (this repo, seed-3 zenith at 500k of a 1.5M run) | [`media/rocket-seed3-zenith-500k.mp4`](media/rocket-seed3-zenith-500k.mp4) | The "rocket" gait at its sharpest — sustained forward locomotion learned online from the live `(obs, action, reward)` stream. |
| **Random baseline** (uniform action) | [`media/random-seed3-500k.mp4`](media/random-seed3-500k.mp4) | The random-walk floor OaK has to clear. Twitchy thrashing, no directed motion. |
| **Zero baseline** (no-op, all actuators 0) | [`media/zero-seed3-500k.mp4`](media/zero-seed3-500k.mp4) | Raw HalfCheetah dynamics with no control signal — the cheetah falls and lies still. |

> **⚠️Physics drift caveat — read before watching.⚠️** All three clips are
> *replays* of `actions.bin` through `tools/replay_render.py`, which steps
> a vanilla Gymnasium `HalfCheetah-v4` env on the recorded action stream.
> The actions themselves were produced against our envpool-backed C++
> bridge ([The envpool bridge](#the-envpool-bridge)), and envpool's MuJoCo
> integration is **not bit-identical** to vanilla Gymnasium — small
> floating-point differences in contact resolution and integration
> compound over hundreds of thousands of steps. So over a 500k-step clip
> the on-screen cheetah will visibly diverge from the trajectory the
> agent actually saw at training time. **What you're watching is the
> faithful action stream applied to a near-identical physics, not the
> exact rollout the agent observed.** The reward / x-position / velocity
> numbers in the [Numbers behind the clips](#numbers-behind-the-clips)
> table come from the live envpool run, not the replay, so they *do*
> reflect what the agent actually experienced.

<!--
  GitHub renders an inline player when the .mp4 URL points at
  user-attachments/assets/... To get a URL, drag the file into any GitHub
  issue/PR comment box, copy the resulting URL, and replace the matching
  `src` below. Repo-relative links act as download fallbacks.
-->

### OaK — seed-3 @ 500k "Its Alive!"

[![Watch on YouTube](https://img.youtube.com/vi/dViNYDHtED4/hqdefault.jpg)](https://www.youtube.com/watch?v=dViNYDHtED4)

▶ Watch on YouTube: <https://www.youtube.com/watch?v=dViNYDHtED4>
Local copy (not tracked in git, see `.gitignore`): [`media/rocket-seed3-zenith-500k.mp4`](media/rocket-seed3-zenith-500k.mp4)

### Random baseline — uniform action, seed 3 @ 500k

[![Watch on YouTube](https://img.youtube.com/vi/f0NfvXHVDmc/hqdefault.jpg)](https://youtu.be/f0NfvXHVDmc)

▶ Watch on YouTube: <https://youtu.be/f0NfvXHVDmc>
Local copy (not tracked in git, see `.gitignore`): [`media/random-seed3-500k.mp4`](media/random-seed3-500k.mp4)

### Zero baseline — no-op, seed 3 @ 500k

[![Watch on YouTube](https://img.youtube.com/vi/UPK3DUp6YMQ/hqdefault.jpg)](https://youtu.be/UPK3DUp6YMQ)

▶ Watch on YouTube: <https://youtu.be/UPK3DUp6YMQ>
Local copy (not tracked in git, see `.gitignore`): [`media/zero-seed3-500k.mp4`](media/zero-seed3-500k.mp4)

#### Numbers behind the clips

Cumulative env stats pulled directly from the benchmark output JSON. The
random and zero runs are reported at their full 500k step budget. OaK
emits heartbeats every 200k steps, so for the same run that produced the
zenith video we report the bracket around the 500k mark — heartbeat 002
(400k cumulative) and heartbeat 003 (600k cumulative).

| Policy | Steps | Cumulative reward | Mean `x_velocity` | Final `x_position` | Mean `reward_run` | Mean `reward_ctrl` | `rewardRate` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OaK** — heartbeat 002, zenith neighbourhood | 400,000 | **+119,099.91** | **+0.334** | **+6,681.19** | +0.3341 | −0.0363 | −0.0297 |
| **OaK** — heartbeat 003, post-zenith | 600,000 | +117,304.54 | +0.222 | +6,669.00 | +0.2223 | −0.0268 | −0.0059 |
| **Random** — uniform action | 500,000 | −129,132.35 | −0.058 | −1,453.67 | −0.0581 | −0.2001 | −0.5231 |
| **Zero** — no-op (all actuators 0) | 500,000 | +0.66 | ≈ 0 | −0.05 | ≈ 0 | 0 | ≈ 0 |

Three things worth pointing at:

- **OaK is the only policy with strongly positive cumulative reward and a
  positive forward velocity/x-position.** Random thrashes backward; zero
  stays put.
- **Random pays a heavy control cost** (`reward_ctrl.mean = −0.20`). The
  env penalizes wild actuator commands, so uniform-action thrashing
  accumulates a deep negative reward even though `reward_run` alone isn't
  that bad — most of the −129k is paid in control.
- **The zenith bracket is visible in OaK's own numbers.** Mean
  `x_velocity` drops from **+0.334 → +0.222** between 400k and 600k
  cumulative, even though `x_position.last` barely changes. The cheetah
  is still on the ground at roughly the right place but has already lost
  the rocket cadence by 600k — exactly why the canonical visual is the
  500k checkpoint and not a later one.


## Where this sits in the Alberta Plan

The Alberta Plan lays out a 12-step roadmap, building from continual supervised
learning (Step 1) up through prototype-AI agents (Steps 8–11) and on to
Intelligence Amplification (Step 12). The progression is the architectural
content; [The Big-World Hypothesis](https://openreview.net/pdf?id=Sv7DazuCn8) — that the environment is vastly more
complex than the agent, so representations must track rather than converge — is
the *motivating* commitment of the Plan, not an architecture in itself. It is
why the Plan is experience-oriented, temporally uniform, compute-cost
first-class, and unwilling to impose caps that would force a stationary scale
prior. It is the north star, not the design.

This repo is specifically **Step 11: Prototype-AI III: Oak**. Step 10 is the
**STOMP progression** — *SubTask → Option → Model → Planning* — which builds
temporal abstraction on top of given features. Step 11 (Oak) extends Step 10
with two things:

1. Continual **utility assessment and remove-replace feedback** over every
   element (features, subtasks, options, option models). Anything that fails
   to pull weight is eventually deleted and replaced.
2. The **option keyboard**: a real-valued vector λ ∈ R^K, one component per
   subtask, where playing multiple non-zero components ("a chord") references
   a combined option.

## The FC-STOMP cycle / OaK Architecture

![Oak architecture diagram](media/OaK%20Architecture.png)

Diagram source: [`docs/papers/AlbertaPlanSlides.pdf`](docs/papers/AlbertaPlanSlides.pdf)

> *"Understanding intelligence is a grand scientific prize that may soon be
> within reach; the glory will go to the bold."*
> — [`AlbertaPlanSlides.pdf`](docs/papers/AlbertaPlanSlides.pdf), p. 12

The architecture for the Step-11 agent is the FC-STOMP cycle: the paper names
the cycle STOMP and the architecture Oak; FC (Feature Construction) is
prepended here because under the Big-World commitment features themselves must
be continually constructed (Plan Steps 1–2) rather than given. The full live
loop is exposed verbatim as named stages in
[`Sources/OakKernel.metal`](Sources/OakKernel.metal):

| Stage | Role |
|---|---|
| **F**eature **C**onstruction | Continual feature finding + GVFPrediction over feature cumulants; features are generated, tested, and replaced by utility (Plan Steps 1–3, 5; Step-11 feedback) |
| **S**ub**T**ask | Reward-Respecting subtasks (stopping-value per [arXiv:2202.03466](https://arxiv.org/abs/2202.03466) eq. 4); low-utility subtasks are removed and replaced (Step-11 feedback) |
| **O**ption | Option policies + RR-derived continuation; the **option keyboard** (chord λ ∈ R^K) blends component options into the emitted action (Step-11 addition) |
| **M**odel | Option models r̂(s,o), n̂(s,o), and average-reward duration l̂(s,o); models that never pay off in planning are deleted with their option (Step-11 feedback) |
| **P**lanning | Average-reward option backup `Q = rHat − ρ·lHat + V(nHat)`, driven by search-control prioritization |

Every stage runs every step. `OAK_ABLATE` compiles diagnostic Metal pipeline
variants that knock out individual sections or couplings; canonical promotion
runs use `OAK_ABLATE=none`. The current chord is a real-valued blend over
component options; a fully operational keyboard/chord-descriptor cycle is
still an open architecture gap (see `AGENTS.md`).

## Current state

OaK is past "negative-return random-beating control" and into real HalfCheetah
walking, but stability across all 10 seeds is still open research:

- **Seed 3 zenith at 500k (`media/rocket-seed3-zenith-500k.mp4`):** the rocket
  gait, captured at its sharpest. Seed 3 over the longer 10M horizon keeps
  cumulative forward momentum (final return `+64,937.78`, x-position delta
  `+5,891.6`, mean `info:x_velocity = +0.0118`) but the gait dulls past the
  zenith, which is why the 500k checkpoint is the canonical visual.
- **Seeds 3, 6, 9** become real forward snowballs by 1M.
- **Seed 8** is the large backward attractor (preserved as a sentinel).
- **Seeds 0/1** are quiet failures.
- 10/10 stability and clean non-finite health remain unsolved.

The strict, dated audit trail lives in [`AGENTS.md`](AGENTS.md) and the keeper
log in [`docs/roadmap.md`](docs/roadmap.md). The literature corpus is in
[`docs/papers/`](docs/papers/).

## Install / build

### Prerequisites (must be installed before `./setup.sh`)

- **macOS on Apple Silicon** (Metal compute pipeline).
- **Xcode command-line tools** — provides `swift` and the Metal toolchain
  (`xcrun -sdk macosx metal`, `metallib`):
  ```bash
  xcode-select --install
  ```
- **Python 3** on `PATH` (validation/replay tooling only; the hot loop has
  zero Python and zero C++ dependencies).

### One-shot build

```bash
./setup.sh
```

`setup.sh` creates `.venv` and installs `requirements.txt` (`mujoco`, `numpy`
— used only by the parity oracle and replay tools), compiles
`Sources/OakKernel.metal` to a metallib, and runs `swift build -c release`.
The resulting binary is `./.build/release/OaK`.

For Swift / Metal-only changes, `scripts/halfcheetah_rebuild_10m.sh` recompiles
just those parts before launching a benchmark — no need to rerun `setup.sh`.

## The GPU environment (no bridge, no envpool, no C++)

The environment lives in the same Metal command buffer as the agent:
[`Sources/OakPhysics.metal`](Sources/OakPhysics.metal) is a literal fp32 port
of MuJoCo 3.6.0's `mj_step` for this exact model — planar CRB mass matrix,
RNE, analytic capsule-vs-plane contacts, pyramidal Newton solver with
Cholesky rank-1 updates and exact linesearch, `eulerdamp` implicit
integration, warmstart, joint limits. The port contract with file:line
citations into MuJoCo source is
[`docs/gpu-physics-port-spec.md`](docs/gpu-physics-port-spec.md).

**Fidelity is proven, not assumed** (`tools/physics_parity.py` is the fp64
oracle — Python MuJoCo on the vendored
[`assets/half_cheetah_envpool.xml`](assets/half_cheetah_envpool.xml)):

- Single-step parity vs fp64 over 2000 golden cases (multi-contact and
  joint-limit states included): median rel err 6.4e-07, max 5.2e-05.
- Rollout divergence rides the fp64 chaotic twin envelope (fp32 behaves like
  a ~1e-7 perturbation, not a bias); energy traces agree to <0.0001%.
- Per-lane fixed op order + no cross-lane operations ⇒ **bitwise run-to-run
  reproducibility on the GPU**.
- Population-level equivalence vs CPU MuJoCo certified by a null-calibrated
  TOST gate at N=8192 lanes/side: divergence, IQM delivered rate, survival,
  NF prevalence, and KS distribution tests all EQUIVALENT
  (`tools/fleet_gate.py --equivalence`).

> **fp32 caveat** (the honest counterpart of the old envpool≠gymnasium replay
> caveat): Apple GPUs have no fp64, so trajectories are not bit-identical to
> CPU MuJoCo — chaotic contact dynamics make bit-match meaningless past
> ~100–1000 steps even between fp64 runs. What is preserved, and proven
> above, is the *physics* (same algorithm, same constants, same op order)
> and the *population statistics* (the science).

One env-step is ~10–30 KFLOP of thread-local register math per lane: the
environment is effectively free next to the learner, and the whole step loop
runs in multi-step command buffers with no per-step CPU round-trip.

## Wait? No PyTorch? Lets Talk about that.

Since this repo is mostly maintained by agents. We need to go **fast**. Why rely on an unoptimized framework when we can craft accelerated GPU kernels with claude directy? We can use XCODE traces and comparing the outputs of OaK bit for bit so we can losslessly upgrade and optimize the kernel without large dependencies or frameworks getting in our way. The Code is the algorithmic logic for OaK a thin launch script and thats it.

![PyTorch crossed out](media/pytorch%20crossed%20out.png)


## John Carmack "I am not a madman for saying that it is likely for Artificial General Inteligence is going to be tens of thousands of lines of code not millions"

Carmacks Ideas line up with sutton and is even working with AMII.

https://www.amii.ca/updates-insights/john-carmack-and-rich-sutton-agi

I follow John Carmacks philosophy of AGI "I am not a madman for saying that it is likely for Artificial General Inteligence is going to be tens of thousands of lines of code not millions" 

[![Watch on YouTube](https://img.youtube.com/vi/6gi8TGNpS_E/hqdefault.jpg)](https://www.youtube.com/watch?v=6gi8TGNpS_E&t=73s)

▶ Watch on YouTube (jumps to 1:13): <https://www.youtube.com/watch?v=6gi8TGNpS_E&t=73s>


PS. i think even this is pesemistic. OaK Works already.
 i think its likely 1500 lines of our metal kernel at most ;)
  We may very well just be a few hundred lines away from the grand prize.

## Run a benchmark (the fleet)

The benchmark is a **fleet**: B independent (agent, env, seed) lanes in one
process — one batched Metal dispatch per step for all B agents, env stepping
sharded across CPU threads. Each lane is still one continual stream (one
observation → one action → one reward per step, never reset); the fleet is
N independent agents, not one agent with vectorized envs.

```bash
./benchmark.sh
```

Defaults are configurable via env vars (read in `Sources/Main.swift`):

| Var | Default | Notes |
|---|---|---|
| `OAK_BENCH_ENV` | `HalfCheetah-v4` | Only env wired through the bridge today |
| `OAK_BENCH_STEPS` | `500000` | Per-lane step budget (horizon) |
| `OAK_BENCH_POLICIES` | `oak,zero,random` | Comma list; one fleet per policy |
| `OAK_BENCH_SEEDS` | — | Explicit seed list, overrides count |
| `OAK_BENCH_SEED_COUNT` | `1024` | B = lane count; lane b runs seed b |
| `OAK_BENCH_HEARTBEAT_STEPS` | steps/10 | Lower for live diagnostics |
| `OAK_FLEET_ENV_THREADS` | all cores | CPU shards for env stepping |
| `OAK_FLEET_TIMING` | unset | `1` prints per-phase wall shares per heartbeat |
| `OAK_FLEET_FREEZE_DEAD` | unset | `1` stops stepping certified-dead lanes (gate runs only) |
| `OAK_METRICS_STORAGE` | `.oak-benchmarks/...` | Output root |
| `OAK_RECORD_SEEDS` | unset | Comma list of lanes that record `actions.bin` |
| `OAK_RECORD_ACTIONS` | unset | `1` records ALL lanes (small fleets only) |
| `OAK_DUMP_STATE_EVERY` | unset | N = per-lane state-hash stride (byte-identity validation) |
| `OAK_ABLATE` | `none` | Diagnostic FC-STOMP ablations |

Outputs are population-scale files in the run root, not per-seed trees:
`fleet.jsonl` (one row per lane per heartbeat), `deaths.jsonl` (per-lane death
certificates: first non-finite step, detector, ρ and max |param| at death),
`aggregate.jsonl` (live dashboard rows), and per-lane `actions.bin` for
recorded lanes.

The tier ladder (one fleet size, horizons in powers of ten, seeds prefix-nested
so every tier is a paired extension of the one below; no protocol step exceeds
~2 h wall-clock):

| Tier | B × horizon | Use |
|---|---|---|
| S | 1024 × 10k | every candidate (hard fails) |
| M | 1024 × 100k | candidates passing S |
| L | 1024 × 1M | keeper promotion + frontier studies |
| XL | 128 × 10M | plasticity decay-constant program |

Watch a live fleet, report a finished one, and gate a candidate:

```bash
python3 tools/fleet_watch.py <run-root> --follow
python3 tools/fleet_stats.py <run-root>          # KM survival, Wilson CIs, IQM, plasticity
python3 tools/fleet_gate.py  <keeper> <candidate> # population promotion law (paired by seed)
```

`fleet_gate.py` is the promotion law: divergence-rate, IQM delivered-rate, and
RMST-survival non-inferiority plus no new non-finite phases — all paired by
seed across the full population. Single seeds are lanes in the data, not law.

## Reproducing the rocket (seed 3)

The 1.5M-step run that produces `media/rocket-seed3-zenith-500k.mp4`:

```bash
OAK_RECORD_ACTIONS=1 \
OAK_BENCH_POLICIES=oak \
OAK_BENCH_SEEDS=3 \
OAK_METRICS_STORAGE="$PWD/.oak-benchmarks/seed3-zenith-$(date +%Y%m%d-%H%M%S)" \
./scripts/halfcheetah_rebuild_15m.sh
```

> **Heads-up on the script names.** Despite the suffixes, the rebuild scripts
> count steps, not millions of steps, and the suffix is *abbreviated step
> count*, not "Nx million":
>
> | Script | `OAK_BENCH_STEPS` | i.e. |
> |---|---|---|
> | `halfcheetah_rebuild_15m.sh` | `1500000` | **1.5M** steps |
> | `halfcheetah_rebuild_10m.sh` | `10000000` | **10M** steps |
>
> The 500k zenith snapshot lives inside the 1.5M run; the 10M script is the
> long-horizon variant used for the cumulative-return numbers in
> [Current state](#current-state).

When it finishes, render the recording:

```bash
RUN=$(ls -dt .oak-benchmarks/seed3-zenith-* | head -1)
python3 tools/replay_render.py "$RUN/HalfCheetah-v4/seed3" --video media/rocket-seed3-zenith-500k.mp4 --fps 30
```

Recording is opt-in (`OAK_RECORD_ACTIONS=1`): one `fwrite` per step, no
measurable benchmark overhead.

## Watching the agent

`tools/replay_render.py <run-dir>` opens a pygame window (frames pulled with
`render_mode="rgb_array"` to sidestep gymnasium 0.29's broken human-mode mujoco
overlay). Esc or close the window to exit.

| Flag | Default | Notes |
|---|---|---|
| `--steps N` | all | Cap replay length |
| `--fps N` | `30` | Live playback speed; under `--video` it becomes the encoded frame rate |
| `--video FILE` | off | Skip the window, encode to `FILE` (mp4) via piped ffmpeg (`libx264`, crf 20, yuv420p) |

> **Physics drift caveat.** The envpool backend is not bit-identical to vanilla
> gymnasium, so over long horizons the replayed cheetah will diverge from the
> recorded trajectory. The agent's actions are faithful — what you see is the
> policy that ran, applied to a near-identical physics — which is what you want
> for "watch what the agent learned to do."

## Repo layout

```
Sources/        Swift host (fleet engine) + OakKernel.metal (B-lane FC-STOMP loop)
scripts/        HalfCheetah rebuild scripts (10M, 15M)
tools/          fleet_watch / fleet_stats / fleet_gate / replay_render / bridge_oracle_cem
docs/papers/    Literature corpus (Alberta Plan, RR-Subtasks, STOMP, plasticity, etc.)
docs/roadmap.md Strict keeper / rejection log + fleet tier protocol
AGENTS.md       Live audit trail and empirical lessons
```
