# OaK

Big-World, Alberta-Plan agent on Apple Silicon.

## Setup

```bash
./setup.sh
```

## Run

```bash
swift run -c release -- run <env-id>
./benchmark.sh
```

For live benchmark diagnostics, lower the heartbeat stride and watch the JSON
stream:

```bash
OAK_BENCH_HEARTBEAT_STEPS=10000 ./benchmark.sh
python3 tools/bench_watch.py --policy oak
```

For paired candidate screens, run only the diagnostic seeds first and compare
against the current keeper:

```bash
OAK_BENCH_SEEDS=0,1,3,8 OAK_BENCH_STEPS=50000 OAK_BENCH_POLICIES=oak ./benchmark.sh
python3 tools/bench_compare.py .oak-benchmarks/grounded-deadrefresh-50k-20260426 <candidate-root>
```

## Watching the agent

To see the agent move in the env, record an action stream during a benchmark
and replay it through standalone gymnasium (Box action spaces only). Recording
is opt-in via `OAK_RECORD_ACTIONS=1` and writes `meta.json` + `actions.bin`
alongside the JSON metrics; it adds a single per-step `fwrite` and does not
perturb benchmark timing.

```bash
OAK_RECORD_ACTIONS=1 OAK_BENCH_SEEDS=3 OAK_BENCH_STEPS=200000 \
  OAK_BENCH_POLICIES=oak OAK_BENCH_CONCURRENCY=1 ./benchmark.sh
python3 tools/replay_render.py .oak-benchmarks/<run-id>/HalfCheetah-v4/seed3
```

`tools/replay_render.py <run-dir>` opens a pygame window (frames pulled with
`render_mode="rgb_array"` to sidestep gymnasium 0.29's broken human-mode mujoco
overlay). Esc or close the window to bail.

| Flag | Default | Notes |
|---|---|---|
| `--steps N` | all | Cap replay length |
| `--fps N`   | `30` | Live playback speed; ignored under `--video` for the loop sleep, but still written as the encoded video's frame rate |
| `--video FILE` | off | Skip the window, encode to `FILE` (mp4) via piped ffmpeg (`libx264`, crf 20, yuv420p). Runs as fast as the renderer + ffmpeg can chew through it; progress is printed to stderr |

Render-only seed 3 example (the rocket-preservation sentinel) and an offline
encode of it:

```bash
python3 tools/replay_render.py .oak-benchmarks/<run>/HalfCheetah-v4/seed3 --fps 60
python3 tools/replay_render.py .oak-benchmarks/<run>/HalfCheetah-v4/seed3 --video rocket-seed3.mp4 --fps 60
```

Physics drift caveat: the underlying envpool backend is not bit-identical to
vanilla gymnasium, so over long horizons the replayed cheetah will diverge from
the recorded trajectory. The agent's actions are faithful — what you see is the
policy that ran, applied to a near-identical physics — which is what you want
for "watch what the agent learned to do".
