# OaK Preserve — frontend

The visualization app for OaK fleet runs: a run gallery over every root in
`.oak-benchmarks/`, a population dashboard per run, a live watch wall, and a
Three.js replay theater with one artisanal nature scene per environment (all 16).

It reads the fleet's own on-disk output and nothing else. No database, no build
step for data, no npm dependencies beyond `three` and `uplot`.

## Prerequisites

- [bun](https://bun.sh) ≥ 1.3
- A populated `../.oak-benchmarks/` (any fleet run roots; per-lane recordings are
  needed for the theater — see the recipe below)

## Commands

```bash
bun install
bun run dev          # API :5174 + Vite :5173 (with an /api proxy) — the working mode
bun run build        # vite build -> dist/
bun run start        # prod: ONE server on :5174 serving dist/ + /api
bun test             # the whole suite (accuracy goldens, kit, scenes, server, client)
bun run typecheck    # client + server tsconfigs
```

Everything is local-only: the server binds `127.0.0.1` and never reaches the
network. `bun run dev` starts the API in-process and then spawns Vite, so one
Ctrl-C stops both.

## Architecture map

```
frontend/
  server/          Bun HTTP server: the fleet store reader (no framework)
    app.ts           routes + static serving; createApp() is mountable without a port
    runIndex.ts      scans .oak-benchmarks, caches the index, decides complete/in-flight
    fleetDerive.ts   streams fleet.jsonl -> columnar per-lane series (disk + LRU cached)
    live.ts          SSE: tails aggregate.jsonl for the watch view
    recordings.ts    discovers per-lane actions.bin; symlink-hardened path resolution
    roles.ts         reads ROLES.json (keeper / candidate / parent)
    jsonl.ts         the streaming JSONL reader everything above shares
    config.ts        paths, ports, cache keys, tuning constants
  shared/          FROZEN contracts shared by server and client
    types.ts         on-disk fleet schemas + API DTOs
    envs.ts          the 16 env ids, names, slugs, action specs, categories
  src/
    sim/           THE GROUND TRUTH: decoders that turn a recorded pose stream into
                   exact env state (grid cells, walls, masks, FK joints). One file per
                   env under sim/envs/, plus philox.ts, fk/cheetah.ts, decode.ts, events.ts
    kit/           the shared Three.js "nature kit": tokens, materials, sky, lighting,
                   terrain, grid, wisp, vegetation, particles, water, post, rigs, rng,
                   celebrate, harness. Scenes may only use what kit/index.ts exports
    scenes/        16 artisan scenes, one directory each (index/manifest/world/actors/
                   fixtures). STYLE.md in this directory is law
    theater/       the replay host: loader.ts (fetch + decode), player.ts (transport,
                   scrubbing, speed), sceneHost.ts (renderer, camera, tier switching)
    views/         wall.ts (THE page), watch.ts (live), harness.ts (scene dev),
                   run.ts (the science module the wall embeds), gallery.ts +
                   theater.ts (redirect stubs + shared helpers). Each route
                   module exports mount(el, params) => { destroy() }
    components/    the DOM layer views share: table.ts (element helpers, header,
                   breadcrumb, empty/error states, sortable table), chart.ts (uPlot
                   wrappers), bars.ts, sparkline.ts
    api.ts         typed fetch client over shared/types.ts; throws ApiError
    state.ts       zero-DOM data layer: store, TTL resources, formatters, reshaping
    router.ts      hash router
    styles/main.css  all app chrome (scene canvases own their own color)
  test/            bun test — goldens in test/fixtures/
```

Routes:

| Route | View |
|---|---|
| `#/` | THE WALL — the one page: one run playing (hero + ghosts, leaderboard rail, env chips, run picker), the learning-curve strip with a playhead, and the run's full science below (per-env stats, collapsible per-seed detail with click-to-watch, collapsible whole-run statistics). `?run=&env=&seed=&at=&speed=` deep-link every part of it |
| `#/watch/:id` | one still-running fleet, live (SSE) |
| `#/harness/:slug` | the artisan scene dev page |
| `#/runs`, `#/run/:id`, `#/theater/...` | redirects into `#/` — kept so old links keep working |

API (all under `/api`, all JSON unless noted):

| Endpoint | Returns |
|---|---|
| `GET /api/health` | `{ ok, benchRoot, production }` |
| `GET /api/roles` | `{ keeper, candidate, parent }` from `ROLES.json` |
| `GET /api/runs` | the run index |
| `GET /api/runs/:id` | `RunDetail` — meta, summary, aggregate rows, deaths count, recordings |
| `GET /api/runs/:id/fleet?fields=&envId=&policy=` | columnar per-lane series |
| `GET /api/runs/:id/live` | SSE stream of aggregate heartbeats |
| `GET /api/runs/:id/recordings` | replayable lanes on this root |
| `GET /api/runs/:id/recordings/:envId/:seed` | one lane's descriptor |
| `GET /api/runs/:id/recordings/:envId/:seed/actions.bin` | the raw stream (range requests supported) |

## The accuracy contract

This is the point of the whole app. Scenes are stylized; they are never inaccurate.

1. **`src/sim/` is the single source of geometric and semantic truth.** It must
   match the Python oracles **bitwise** for the gridworlds (`tools/gridworld_ref.py`)
   and to **1e-9** for the cheetah FK. Golden fixtures live in `test/fixtures/` and
   are regenerated manually with `python3 test/fixtures/gen.py` (checked in).
2. **Every state-bearing mesh is positioned only from `ctx.geometry` + `FrameState`**
   — `cellToWorld`, `fk()`, `heightAt`, anchors. Decoration hangs off exact bones as
   a child: delete the decoration and what remains is still exactly correct.
3. **`probes()` registers every state-bearing object**, `fixtures.ts` ships ≥ 3
   canonical poses, and `bun test` runs `assertAligned` headless — |world position −
   truth| < 1e-3 on every probe of every scene.
4. **Honesty rules.** Never show what the agent cannot know outside the debug
   overlay (POMaze's full map, MemoryChain's cue after position 0, ProceduralMaze's
   carved path). Never hide what the environment defines (CartPole's failure arcs,
   Acrobot's success line, DeepSea's mask currents) — show them, subtly.
5. **Celebrations fire only from `frame.events`.** Nothing is synthesized from a
   value crossing a threshold.
6. **Determinism.** All randomness comes from `kit/rng` seeded with `ctx.seed`; all
   animation time comes from `ctx.time()`. Scrubbing to step N twice draws the same
   frame, and pausing freezes the world coherently.

`setDebug(true)` (the `g` key in the theater and harness) overlays the kit
`GroundTruthOverlay`: raw grid, walls, agent cross, goal, FK stick skeleton,
terrain polyline. Review screenshots use it.

The full scene law — stylization language, light budget, world scale, celebration
grammar, quality tiers, and the 22 measured canon lessons — is `src/scenes/STYLE.md`.

## Demo recording data

The theater plays per-lane `actions.bin` recordings (record v3: 16 float32 per
step — `ctrl[6], done, pose[9]`). Fleet roots written without recording enabled
show up in the gallery but have nothing to replay.

Generate a fresh 16-env demo root, from the **repo root**:

```bash
OAK_RECORD_ACTIONS=1 OAK_BENCH_SEED_COUNT=128 \
OAK_BENCH_ENVS="HalfCheetah-v4:8,FourRooms-cont-v0:8,DeepSea-cont-v0:8,KeyCorridor-cont-v0:8,Acrobot-cont-v0:8,MountainCar-cont-v0:8,CartPole-cont-v0:8,Pendulum-cont-v0:8,Chain-cont-v0:8,LargeDeepSea-cont-v0:8,MultiKeyCorridor-cont-v0:8,MemoryChain-cont-v0:8,ProceduralMaze-cont-v0:8,HighDimMaze-cont-v0:8,POMaze-cont-v0:8,DelayedChain-cont-v0:8" \
tools/run_fleet.sh 100000 frontend-demo-16env-$(date +%Y%m%d)
```

`OAK_RECORD_ACTIONS=1` records every lane; `OAK_RECORD_SEEDS=3,8` records just
those two, which is enough for a spot check and far smaller on disk. Recordings
appear in the app as soon as the files land — the run index rescans on demand.

## Watching a lane

Everything plays on the wall. Pick the run from the header picker, pick the
environment from the chips, and pick a seed either from the leaderboard rail or
by clicking its column in **Total reward by seed** (inside the per-seed detail
block) — the wall promotes that seed to hero. `#/?run=<id>&env=<envId>&seed=<n>`
deep-links a specific lane; old `#/theater/...` URLs redirect there.

Transport keys: `space` play/pause · `← →` step one frame · `[ ]` speed ·
`g` ground-truth overlay · `t` quality tier.

Quality tiers are `hero` (full art), `medium` (halved particles/vegetation, smaller
shadow, no MSAA) and `thumb` (no post, no shadows, fixed camera) — see STYLE.md for
the exact budgets.

## Harness usage

`#/harness/:slug` mounts one scene standalone against its own fixtures — the
artisan dev page and the way review screenshots are taken. The slug is the scene
directory name (`mountain-car`, `four-rooms`, `po-maze`, …).

| Query | Effect |
|---|---|
| `?tier=hero\|medium\|thumb` | initial quality tier (default `hero`) |
| `&at=N` | start scrubbed to frame N — deterministic stills |
| `&pause=1` | start paused (pair with `at` for headless screenshots) |
| `&debug=1` | start with the ground-truth overlay on |
| `&run=<id>` | drive the scene from a real recorded lane instead of fixtures |
| `&steps=N` | cap the replay length |

Example: `#/harness/mountain-car?tier=hero&at=300&pause=1&debug=1`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OAK_BENCH_DIR` | `<repo>/.oak-benchmarks` | the fleet run store the server reads |
| `OAK_FRONTEND_PORT` | `5174` | API (and, in production, static) port |
| `NODE_ENV=production` | unset | serve `dist/` from the same port as `/api` |

Derived artifacts (run index, columnar fleet series) are cached under
`frontend/.cache/`, keyed by the resolved bench root — pointing `OAK_BENCH_DIR` at
a second store can neither read nor clobber the first store's cache. Deleting
`.cache/` is always safe.

## Troubleshooting

**Cards show `n/a` / `no summary`.** The root has no `summary.json`. Run
`tools/fleet.py report <run-root>` from the repo root; the gallery picks it up on
the next index refresh (60 s TTL, or reload).

**`Could not reach the API` / every card fails to hydrate.** The Vite dev server
on :5173 proxies `/api` to :5174. If the API half died, `bun run dev` again — Vite
alone serves the app but no data.

**Port already in use.** :5173 is Vite, :5174 is the API. Find the holder with
`lsof -nP -iTCP:5174 -sTCP:LISTEN` and kill it, or run the production server on a
different port with `OAK_FRONTEND_PORT=5202 NODE_ENV=production bun server/server.ts`.

**No runs at all.** The server prints its bench root on startup
(`[oak-frontend] bench root: …`) and `GET /api/health` reports it. If it is
pointing somewhere empty, set `OAK_BENCH_DIR`.

**A run says `incomplete`.** It stopped before its declared horizon. That is a
property of the run, not of the app — `fleet.py gate` refuses those roots, and the
dashboard says so in a banner.

**The theater says a run has no recordings.** It was written without
`OAK_RECORD_ACTIONS` / `OAK_RECORD_SEEDS`. `recordVersion` 3 (flat f32, 64 B/step)
and 4 (the compact chunked codec — `src/sim/RECORD_V4_SPEC.md`, default since
2026-08-04, ~20× smaller, all-lane Tier-L ≈ 3 GiB) are servable; older record
formats are ignored on purpose.

**A scene looks wrong.** Turn on the ground-truth overlay (`g`, or `&debug=1` in
the harness). If the art disagrees with the overlay it is a scene bug; if the
overlay itself disagrees with the environment it is a `src/sim` bug, and
`bun test` has a golden for it.

**Production build warns about a chunk > 500 kB.** That is `three` in its own
manual chunk (~554 kB raw, ~138 kB gzipped), loaded only when a 3D view mounts.
The gallery, run and watch views never pull it in.
