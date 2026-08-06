# The OaK Preserve — Style Contract

> **KIT API LOCKED (Phase 2, 2026-08-03).** Both exemplars are approved. From here the
> kit and all frozen contracts are additive-only; scenes adapt to the kit, never the
> reverse. Canon lessons from the exemplars are law — see the section at the end.

All 16 scenes are locations in **one imaginary nature preserve**, across one day and one
water column. Shown side by side, the gallery reads as a full day: dawn → noon → dusk →
night → underwater. One sun logic, one avatar, one celebration grammar. This file is the
law for every artisan scene; the review pass checks against it clause by clause.

Reference implementations (read before building anything):
- `scenes/mountain-car/` — the **visual canon** (terrain, sky, vegetation, particles, celebration, interpolation).
- `scenes/four-rooms/` — the **grid canon** (cellField, wisp, goal respawn) — pattern for all 10 grid scenes.

## Stylization language

**Stylized-PBR with soft toon ramping.** `kit/materials.oakMaterial()` on every lit
surface: soft 2–3-step NdotL ramp (softness ~0.25 — painterly, never hard cel), sky-tinted
fresnel rim, optional vertex wind. Silhouette-first low-poly geometry. Vertex-color /
ramp-driven albedo — **never image textures for surface detail** (procedural
`kit/textures` only, ≤1024², zero network requests). Fog always present (depth gradient
substitutes underwater/cave). A horizon or depth gradient is always visible. HDR
emissives (>1.0) are the only things that bloom. Think *Monument Valley meets Breath of
the Wild*: readable shapes, saturated gradients, atmosphere over detail.

## Hard rules

**MUST use (from `kit/`)**
- `tokens` for every color (scene-local colors may only be blends of token colors).
- `makeGradientSky` (or its underwater/cave parameterization) — exactly one sky.
- `makeLightRig` presets — exactly one shadow-casting DirectionalLight, one
  HemisphereLight, ≤4 PointLights (hero). No other lights.
- `oakMaterial` for all lit surfaces; `emissiveMaterial` for bloom-feeders.
- `kit/grid` (`cellToWorld`, `buildCellField`) + the `Wisp` avatar for every grid env.
- Camera from a kit rig: `TrackRig` (cheetah), `OrbitRig` (grids), `FrameRig` (classic).
- `celebrate` grammar for every reward event; the shared done-sigh for failures.
- `kit/rng` (seeded with `ctx.seed`) for ALL scatter/randomness — scenes are deterministic.
- `ctx.time()` for ALL animation time (pause must freeze the world coherently).

**MAY build bespoke**
- Hero meshes (clam, torii, cairn, cheetah coat, wreck…).
- Scene-specific *parameters* of kit shaders.
- At most ONE bespoke `ShaderMaterial` per scene (aurora, water column…) — it must
  consume token colors and `ctx.time()`.

**MUST NOT**
- Fetch any external asset (textures, fonts, models). Everything procedural.
- Touch the renderer (autoClear, clear color, tone mapping, exposure) or add composer passes.
- Exceed the light budget or omit fog/sky.
- Keep a private clock (`Date.now`, `performance.now`, own Clock) — `ctx.time()` only.
- Allocate in `update()` (no `new Vector3`, no material/geometry churn — preallocate).
- Synthesize events — celebrations fire ONLY from `frame.events`.
- Ease across `scrubbed` or `teleport` — hard snap.
- Path-plan between cells (straight hop only) or extrapolate beyond the given state.

## Accuracy (the whole point)

- Every state-bearing mesh is positioned ONLY from `ctx.geometry` + `FrameState`
  (`cellToWorld`, `fk()`, `heightAt`, anchors). Skins are children of exact bones — if
  the decoration were deleted, what remains would be exactly correct.
- **Honesty rules.** Never show information the agent lacks outside the debug overlay
  (POMaze full map, MemoryChain cue after pos 0, ProceduralMaze carved path). Never hide
  what the env defines (CartPole failure arcs, Acrobot success line, DeepSea mask
  currents — show them, subtly).
- `setDebug(true)` renders the kit `GroundTruthOverlay` (raw grid/walls/agent
  cross/goal/FK stick skeleton/terrain polyline) over the art. Review screenshots use it.
- `probes()` registers every state-bearing object; `fixtures.ts` ships ≥3 canonical
  poses. `bun test` runs `assertAligned` headless: |world pos − truth| < 1e-3.

## World scale & camera

1 grid cell = 1 world unit, grid centered at origin, +x = grid-x (east), +z = grid-y
(south), y up. Wisp height 0.35. Cheetah true scale (torso 1.0 long, hip height ~0.7).
Hero FOV 35–45 (default `tokens.fovHero` = 40). Ground/floor at y = 0.

## Celebration grammar (`kit/celebrate`)

Every +1, in every scene, is the same three beats, skinned per scene:
1. `burst(pos, accent)` — radial particle bloom (petals / spray / sparks / fireflies).
2. `ripple(groundPos)` — expanding ground ring.
3. 300 ms bloom-strength pulse (host post hook — automatic).
Failure / non-reward `done` = the shared cool-blue sigh (brief desaturation + falling
motes). Mid-tier events (door open, key pickup) use `burst` at reduced scale, no ripple.

## Quality tiers

- **hero**: full art. ≤300 draw calls (aim 150), ≤500k tris (aim 250k), 1 shadow
  (2048 PCFSoft, tight frustum on the playfield), DPR ≤2, 60 fps on Apple Silicon.
- **medium**: shadow 1024, DPR ≤1.5, particles ×0.5, vegetation ×0.5, no MSAA,
  bloom quarter-res.
- **thumb**: no post, no shadows, no particles/mist, vegetation ×0.15, fixed camera,
  DPR 1, ≤40 draw calls, ≤50k tris. (This is the future wall-view contract.)
Kit factories take `tier` and scale themselves — prefer kit scaling over hand-rolled.

## The reserved ghost layer (host-owned, additive 2026-08-04)

The host MAY attach one host-owned group named **`oakGhostLayer`** (`kit/ghosts` — every
seed of the env drawn as instanced ghosts around the hero) to the scene AFTER `init()`
resolves, and detaches it before any rebuild. Scenes MUST NOT read, move, re-parent,
traverse into or dispose it, MUST NOT count it in `manifest.budget` or return anything
inside it from `probes()`, and MUST NOT create a group of their own by that name. It is
not part of the scene; it is the population, standing behind it.

## The 16 briefs

Each brief: setting → hero elements → accuracy hooks → celebration skin → idle motion.
Palettes named from `tokens.palettes`; accent from `tokens.accents[slug]`.

1. **HalfCheetah — "Gold Hour Savanna"** (dusk). Endless amber grassland, acacia
   silhouettes, low HDR sun. The one non-wisp hero: capsule bodies skinned exactly on
   `geometry.cheetah.fk` (radius 0.046, head r=0.15), tawny gradient coat. World is a
   deterministic treadmill: terrain/vegetation tiles keyed by `floor(rootx/tileSize)`
   hashed with the lane seed (scrub-reproducible); stone cairns every 10 m make distance
   legible. `teleport` = hard cut + one-frame warm flash + `TrackRig.snap()`. Dust puffs
   when FK toe height < 0.05. Grass wind-lean and dust density scale with rootx velocity.
   Idle: heat shimmer, pollen, distant birds.
2. **FourRooms — "Four Seasons Courtyard"** (noon). 11×11 walled garden; rooms are
   literally seasons — spring blossom TL, summer green TR, autumn gold BL, winter frost
   BR. Hedge walls on exact wall cells; the four hallways are stone moon-gates; (5,5) is
   a sundial plinth (wall cell). Goal = lantern-flower at the exact goal cell; blooms on
   `goal`, petal-burst, respawned lantern grows in elsewhere. Idle: per-quadrant
   particles (petals/pollen/leaves/snow).
3. **DeepSea — "Sunlit Descent"** (underwater). 8×8 dive as terraced turquoise reef
   shelves; the reachable triangle IS the reef — upper-right cells are open blue water.
   God rays, caustics, rising bubbles; wisp rides in a bubble. Kelp on each cell leans
   with the current per the regenerated action mask (truthful, subtle). Treasure at
   (7,7): giant clam opens, pearl flares, spray. `resurface` = bubble rush back to (0,0).
   Rightward steps kick a tiny silt puff (the right-move cost).
4. **KeyCorridor — "The Ivy Gate"** (morning). 9×7 two-room garden; locked wooden door
   at (4,3) wrapped in ivy that unwinds on `door` open; key = glowing amber seed the wisp
   carries once `hasKey`; goal shrine in the far room. On goal: relock — ivy regrows in
   reverse, new key fades in at its exact cell.
5. **Acrobot — "The Willow Swing"** (dawn). Giant willow over a mirror pond; 2-link vine
   trapeze with a lantern at the free end; th1/th2 applied directly, link length 1.0 = 1
   unit. The success height (+1 link above pivot) is a physically placed string of
   hanging lights; tip crossing it: lights flare, petal rain. Elbow torque {−1,0,+1} = a
   faint directional glow curl at the joint. A fading ribbon traces the tip path. Idle:
   pond shimmer, dawn mist.
6. **MountainCar — "Alpine Valley"** (dusk). EXEMPLAR. Terrain IS `sin(3x)·0.45+0.55`
   over [−1.2, 0.6], extruded to a stylized ridge — snow above a fixed line, meadow
   below, conifers off-track. Flag at 0.5 = summit cairn with prayer flags. Wooden cart:
   x = pose[0] exactly, y/tilt from `heightAt` + derivative. Momentum = motion-trail
   ribbon; failed climbs kick scree. On +1: summit flare, flags snap, eagle silhouette.
   Idle: valley mist, snow glitter.
7. **CartPole — "The Boardwalk"** (noon, windy). Boardwalk spanning exactly ±2.4 through
   wind-blown grass over a lake. Cart = planter trolley; pole = young sapling (half-len
   0.5), angle exact. The ±12° failure cones are two faint carved arcs on a backboard.
   Balanced = warm light (the dense +1/step); on fail: sapling drops, light cools,
   done-sigh, upright reset pops with a soft grow. Idle: grass wind synced to pole sway
   direction (cosmetic).
8. **Pendulum — "The Lantern Bell"** (night). Koi pond under a torii arch; hanging
   lantern, θ exact. **This kernel's convention (authority: `src/sim/envs/pendulum.ts`
   L11–21, NOT Gym): θ=0 is the bob hanging straight DOWN and is the shaped-cost
   minimum; fleet init is θ=π (inverted, and it falls).** The shaped reward IS the
   lighting: at rest hanging = the pond glows warm and koi rise; inverted/swinging =
   dim moonlight. Torque (pose[2]) = directional wind swirls at the bob. Never done: a
   meditation piece. Idle: koi ripples, stars, fireflies in the reeds.
9. **Chain — "Sixteen Stones"** (afternoon). Sixteen stepping stones across a creek;
   wisp hops stone to stone; stone 15 under a bare blossom tree that blooms on +1, then
   petals ferry the wisp back to stone 0 (the wrap, literal). Recently-passed stones glow
   faintly (fades in seconds). Minimal, elegant.
10. **LargeDeepSea — "The Abyss"** (abyss). The dive continues past the reef: N=16,
    violet-black trench, light falls off with row index; anglerfish lures and glow-jellies
    replace sunbeams; treasure at (15,15) = a sunken wreck's glowing heart. Same
    triangular-shelf accuracy language as DeepSea — the pair reads as one dive.
11. **MultiKeyCorridor — "Temple of Three Gates"** (morning). 17×7 temple garden, four
    chambers, three sequential gates: bramble gate → river gate (stepping stones surface
    when opened) → mist gate. Keys = seed, pearl, ember at the exact sim-core-regenerated
    cells (never guessed). keyMask/doorMask bits drive carried tokens + gate states. Door
    opens = mid-tier celebration; final goal = full.
12. **MemoryChain — "The Firefly Cave"** (cave). 32-cell cave passage. The cue crystal at
    pos 0 — cyan (0) / magenta (1) from tokens — is lit ONLY while the wisp stands at
    pos 0; leaving, darkness swallows it. NO residual tint anywhere (honesty rule,
    reviewed). Fireflies line the passage. At pos 31: two exit mouths, cyan- and
    magenta-lit; correct choice = matching-color firefly swarm erupts back down the
    corridor; wrong = both mouths dim (done-sigh).
13. **ProceduralMaze — "The Overgrown Ruins"** (noon). 11×11 mossy stone ruin; walls
    placed exactly from the regenerated per-seed wall array — build a wall-SKINNING
    system, not a layout (every seed is a different ruin). The carved solution path is
    debug-overlay only. Goal = a shaft of sunlight breaking into a ruined atrium at the
    goal cell. Idle: dust motes in shafts, ivy tremble.
14. **HighDimMaze — "The Slot Canyons"** (afternoon). Serpentine 11×11 as three slot
    canyons in terracotta stone; gaps at (3,9) and (7,1) are natural arches. `stay` = the
    wisp settles, dust falls; `dash` = gust-sprint with speed lines and sand wake,
    landing exactly on the true cell. pose[4] lastDir drives facing + a windswept scarf.
    Idle: heat shimmer.
15. **POMaze — "The Fogbound Forest"** (dawn). Partial observability IS the scene: dense
    fog; a warm visibility bubble of exactly the 8 adjacent cells; dark spruce ranks
    materialize only entering adjacency and dissolve behind. The compass obs = a glowing
    moth drifting at the fog's edge in the goal's true direction. Camera lower/closer
    than any other grid scene. On goal: fog lifts in a radius for a beat, then closes.
    Full map = debug overlay only.
16. **DelayedChain — "The Long Night Road"** (night). Bare moorland path, 32 unlit
    lantern posts, one per cell; an aurora hangs ONLY over the far end — reward lives
    there and nowhere else. Long, dark, quiet walk (sparse moonlight pools). On +1 at
    pos 31: every lantern ignites in a wave from the end BACK to the start (delayed
    credit assignment, staged). Idle: aurora ribbon (the one bespoke shader), moor grass.

## Canon lessons (measured in the exemplars — follow them, they are law)

From **mountain-car** (visual canon):
1. **Probe roots carry position only.** Tilt/bob/decoration live on child nodes; probe error should be 0, not "within tolerance".
2. **State visuals are pure functions of state, never history.** The momentum trail is the terrain curve traced back over |vel|; wheel roll is −pos/R. Scrub-safe by construction.
3. **Exact-band terrain.** All decorative noise is multiplied by a weight that is identically 0 on the playfield/track band; `makeTerrain` jitter stays 0 there. Bias terrain segments toward the accuracy axis.
4. **The superluminal guard (every continuous env).** The host lerps poses, so a respawn renders as a slide BEFORE the teleport event arrives. Hold the last physically-reachable pose (|Δ| > maxSpeed·Δstep + slack, cap ~4 frames, release on event/scrub) to turn the slide into a hard cut.
5. **Kit particle/mist preset sizes are tuned for ~20-unit fields.** A close diorama (≤2-unit) wants size ÷ ~10; an 11-unit courtyard wants size ≈ 1. Keep particle intensity under the bloom threshold.
6. **Celebration amplitude scales with scene size.** The default grammar at full strength whites out a small frame — halve every weight and keep all three beats. Every flourish envelope multiplies exp-decay by (1 − age/span) so nothing ends on a pop.
7. **All flourish clocks are ctx.time()** — pause freezes celebrations mid-air, coherently.
8. **Shadow traps:** size the shadow focus to the FRAME (a too-small focus paints border-texel amoebas); sub-texel props (poles, cords) must not cast (dashed streaks); prefer a sun elevation ≥ ~25° or long shadows smear across the subject.
9. **Cross-palette blends are the legal off-palette:** white = `palettes.noon.sun`, near-black = `palettes.night.skyTop`.
10. **Per-instance wind needs InstancedMesh** (oakMaterial wind phase comes from object origin — merged meshes sway rigidly). The kit pins wind at local y=0.
11. **Respect the camera for silhouettes** — vFOV is fixed at 40° regardless of aspect; test hero framings at real aspect.

From **four-rooms** (grid canon):
12. **Decoder state IS the art.** Walls from `grid.walls`, hallways from `grid.hallways`, room/quadrant looks from decoder helpers; zero layout re-derivation. Change the env, the art follows.
13. **The cell field is a substrate:** set floor/wall material color to token white and drive the real per-cell hue via `floorTint`/`wallTint` instanceColor — one draw each, arbitrary hues.
14. **A cell wanting a bespoke prop collapses its instance** (`wallInstanceAt` + zero-scale) and a probe-rooted prop owns the cell; the walls array stays untouched so the overlay still draws its box.
15. **Read orientation, never assume it:** a gate faces the axis its flanking wall cells run along (`isWall(h.x, h.y±1)`) — works on regenerated layouts (ProceduralMaze!).
16. **THE WISP PROBE RULE:** `Wisp.face()` yaws the body that `core` hangs from — scenes that call face() (HighDimMaze, POMaze) must probe `wisp.avatar` (exactly cellToWorld, y=0), not `core`.
17. **Hop duration < one control step** (0.11 s at 8 Hz) so the glide lands before the next truth.
18. **The `goal` event blooms where the AGENT is** — the goal respawned in the same step; read the celebration position from the agent's cell (test-lock it).
19. **Noon albedo law:** under the noon key (3.0) keep ground albedo ~140–170 sRGB; a 220-sRGB floor crosses the bloom threshold and smears the grid. Blend toward `night.skyTop` to shade.
20. **Per-quadrant/region particles** = multiple AmbientParticles with bounds from the region scan, sizes per lesson 5.
21. **Camera distance is arithmetic:** a w-unit grid at azimuth a presents w·(|cos a|+|sin a|) across frame; check both corners at your aerial angle before calling it framed.
22. **Emissive gain is palette-relative:** an HDR flame tuned on noon needs more gain in a dark palette (night/cave/abyss) to read and to bloom.
22b. **…and it is palette-relative in BOTH directions — against a bright floor, buy contrast with HUE and VALUE, never more gain.** Measured across the 11 grid scenes the floors run 27.9–189.8 sRGB, and on the bright end an emitter's peak luminance and its background land within ~1.25× of each other: the wisp dissolved into four-rooms' 174-sRGB gold. Gain cannot fix that — the emitter is ADDITIVE, so it can only add light to what is behind it, and past the bloom threshold it just smears (lesson 19). Give it a local background instead: a dark under-silhouette behind the emitter (which also stops its additive colour washing to white — the orb keeps its own hue) plus a soft ground-contact shade. Both are the emitter's OWN colour blended most of the way to `palettes.night.skyTop` (lesson 9's legal near-black), so in a dark palette they are already the floor's value and vanish — background-agnostic by construction, never a per-palette branch. Implement it at KIT level (`kit/wisp`: `silhouette` + `contactShade`, ~2 draws) so all 11 grid scenes inherit one treatment. Measured: four-rooms peak-vs-local contrast 1.43 → 1.58 and orb saturation 0.231 → 0.260; memory-chain (the darkest, and the one that must not be harmed) 5.55 → 9.29. A view-independent SHELL, not a billboard — three's `sizeAttenuation` scales points by `height/2`, not the focal length, so a world-sized sprite renders ~2.75× too small at fov 40.
