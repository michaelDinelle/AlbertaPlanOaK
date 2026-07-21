# Representation Generality Design — geometry from stream shape, generalized G&T

Status: DESIGN (no code changed). Companion prerequisite for
`docs/research/sparse-credit-design.md` (bounded traces + exploration) — this
document supplies the representation that design consumes; see §8 hand-off.

The elephant (user framing): stop treating "reactive linear agent" as OaK's
identity. The linear readout stays; the pinned cheetah-shaped representation is
the bug. The representation must grow to fit the stream — FeatureConstruction
and GenerateAndTest must build the features any stream needs.

All line numbers are `Sources/OakKernel.metal` unless noted. Verified against
the tree on 2026-07-20.

## 0. Verified current geometry (and two corrections to the brief)

- `OAK_CORE = 32`, `OAK_ACT = 64`, `OAK_GATE_BANK = 7` (:339-341). 32 gated
  banks × 7 slots = 224 slots; R = 256, so slots 32..255 are the G&T arena
  (anchors = 32, `Main.swift:261`; replacement eligibility is
  `i ∈ [anchorCount, R)` at :1310, :1355, :1801).
- R = 256 is itself shape-derived: `Arena.input(17,12) = traces(17)·17 + 12 + 2
  = 99`, rank = smallest pow2 ≥ 2·99 = 256 (`Main.swift:137-140, 244-247`).
  traceLevels = 4, `obsTraceMulti` = 68 floats (`Main.swift:265`, arena :249).
- Core layout (:826-847): slots 0..16 = `standardized_obs` (17), 17..28 =
  `actionSignal` (12), 29 = reward, 30 = bias 1.0, **31 = constant 0.0 (dead
  slot today)** (:839-840).
- Gate: `gate_channel(b) = b % (O + D + 1)` (:346) — bank b tiles core channel
  b % 30, cycling obs(17) → action(12) → reward(1); banks 30,31 re-cover
  channels 0,1. `gate_tile(z) = clamp(floor(z+3.5), 0, 6)` (:348-351) — 7
  unit-σ bins over [-3.5, 3.5), exactly one winner per bank (:850-888).
- **Correction 1 — natural widths.** The brief's "FourRooms O≈4, DeepSea O≈2"
  is stale. Per `OAK_ENV_SPECS` (`Sources/Gridworld.metal:51-61`), naturalO /
  actionCount is: cheetah 17/6, FourRooms **17**/4, DeepSea **8**/2,
  KeyCorridor **14**/4, Acrobot 7/3, MountainCar 3/3, CartPole 5/2,
  Pendulum 4/1, Chain **2**/2. The degeneracy is real but differently
  distributed: DeepSea has 9 zero-padding channels (8..16) **plus** a constant
  bias channel (o[7]=1.0, `Gridworld.metal:184`); Chain has 15 padding + o[1]
  constant; KeyCorridor 3 padding + o[13] constant; FourRooms 0 padding but
  o[16] constant.
- **Correction 2 — what "degenerate" means.** Bank slots are not tile-valued
  features; each bank's 7 slots hold G&T-evolved *constructed* features
  (:855-883) and the gate does state-dependent *selection* among them. A bank
  gated by a constant/padding channel sees `gate_tile(0) = 3` forever, so it
  activates slot lo+3's construct every step — an arbitrary, utility-blind
  choice — and 6/7 of the bank is dead capacity. The representation damage is
  real (no state-dependence in those banks) but the mechanism is "fixed
  arbitrary construct," not "zero feature."
- **Even cheetah wastes 6 banks today**: actionSignal coords 6..11 are the
  inert reserved Box scale slots (zeros, AGENTS.md action interface), so banks
  23..28 (gate channels 23..28 = O+6..O+11) see tile 3 forever. Bitwise-cheetah
  means we do not fix this in the same change; it is a follow-up candidate.
- The inexpressibility argument, crisply: DeepSea's hidden per-cell map
  M[r][c] (64 bits/lane, `Gridworld.metal:168-208`) makes optimal behavior an
  arbitrary function h(r,c). Features gated on single channels give a linear
  readout the span of additive functions f(r) + g(c) — at most 8+8−1 = 15
  dimensions of the 64-dimensional cell space. The 49 missing dimensions are
  exactly the joint interactions no current mechanism can express: the gate
  never tiles (r×c), and G&T's only interaction is one rank-1 product (kind 0)
  with parents chosen by utility argmax, which re-picks the same top pair.
  KeyCorridor needs hasKey×(x,y) (door/goal values are multiplicative in the
  key flag); FourRooms needs (x,y)×goal for the hallway distance field.

## 1. Geometry from stream shape — the derivation function

### 1.1 Inputs and the Inv-1 compliance path

The agent may read only stream *shape*, never env identity. Shape delivery:

- Add `uint naturalO, naturalA` to `OakFleetLane` (:1906-1921; 400 B → 408 B,
  bump the `static_assert` at :1921 and the Swift mirror).
- `oak_fleet_init` (:1961) — already a sanctioned `OAK_ENV_SPECS` reader —
  writes `lane->naturalO = OAK_ENV_SPECS[ek].obsDimNatural` and
  `lane->naturalA = OAK_ENV_SPECS[ek].actionCount` next to :1975. The Metal
  table stays the single source of truth; no Swift mirror of the specs.
- Thread the two fields into `Ctx` (new `uint naturalO, naturalA` at :113-118)
  via `make_ctx` (:164-199) at its call sites (:2028, :2118, :2481, :2721,
  :2726, :2731, :2738, :2764), each of which already has the lane. Default
  `(naturalO, naturalA) = (ctx.O, ctx.D/2)` = legacy geometry wherever no lane
  exists.
- `ctx.O` stays 17 for every lane: observation buffers, the obs EMA loop
  (:798-809), `obsTraceMulti` layout (:249), and state strides are all sized by
  the padded width. Padding channels keep tracking zeros — harmless.

This is a shape read, exactly like `D_policy` today; the kernel never switches
on `envKind` outside the four sanctioned sites. The runtime-uint choice (vs a
function constant) is forced by the mixed-env fleet: one compiled pipeline,
per-lane shape. Precedent for shape branching in agent code already exists:
`if (DL == 6u)` shape-specialized emit (:1168, sanctioned Inv-10
shape-specialization).

### 1.2 The legacy predicate (the bitwise firewall)

```
legacy = (naturalO == ctx.O) && (2u * naturalA == ctx.D)   // (17, 6) ≡ cheetah
```

`legacy == true` ⇒ every formula below reduces to today's code path, verbatim.
Only cheetah satisfies it (FourRooms: 17==17 but 2·4 ≠ 12 → new path).
Everything in §1.3-§3 lives behind `!legacy`. Note the predicate is
deliberately shape-based: any future (17,6) stream *should* inherit cheetah's
geometry — that is the Inv-1 reading, not a leak.

### 1.3 The bank/gate layout function

Per lane shape (O_n = naturalO, A_n = naturalA), constants O=17, D=12,
B=32 banks, T=7 slots (all existing):

```
S   = O_n + A_n + 1                    // informative gate sources
src = b % S ; occ = b / S              // bank b ∈ [0, 32)
ch(src) = src < O_n        ? src               // obs core slot
        : src < O_n + A_n  ? O + (src - O_n)   // action core slot
        :                    O + D             // reward core slot
```

The legacy layout is this same family at (O_n, A_n) = (17, 12): S = 30,
`ch(src) = src` — i.e., `gate_channel` (:346) is the degenerate case that
counts inert action coords as sources. Gate rule per bank:

- **occ = 0** — single-channel, zoom 0: `tile = gate_tile(curr[ch(src)])`
  (unchanged formula). Coverage: every informative source gets one bank.
- **occ ≥ 1, b even** — single-channel, zoom `z_l = min(occ, 2)`:
  `tile = clamp(floor(z·2^z_l + 3.5), 0, 6)` — bins of width 2^-z_l σ over
  ±3.5·2^-z_l, edges saturate (extreme detector). occ=0 is the identity, so
  this is one formula family. Zoom cap 2 = ⌊log2(T)⌋: finer than ¼σ bins adds
  no tile capacity.
- **occ ≥ 1, b odd** — JOINT bank over channels
  `ca = ch(src)`, `cb = (src + occ) % O_n` (always an obs channel; occ shifts
  the pairing so successive occurrences cover different pairs):

  ```
  joint_cell(va, vb, occ):
      if (!isfinite(va) || !isfinite(vb) || |va| ≥ 3.5 || |vb| ≥ 3.5) → 6
      va += (occ % 2 == 0) ? 7.0/6.0 : 0.0         // half-bin stagger
      ta = clamp(floor((va + 3.5) · 3/7), 0, 2)    // 3 bins, width 7/3 σ
      tb = clamp(floor((vb + 3.5) · 2/7), 0, 1)    // 2 bins, width 3.5 σ
      cell = ta·2 + tb                              // ∈ [0, 6)
  ```

  6 interior cells + cell 6 = overflow/extreme (mirrors `gate_tile`'s clamp
  and NF conventions). T = 7 is prime, so a 3×2+overflow partition is the
  maximal balanced 2-D tiling of one bank; multi-offset coverage comes from
  the stagger and from later occurrences re-pairing. `winner = lo + cell`
  exactly as today (:853, :885) — **exactly-one-active-per-bank is preserved**
  (Mark III law (b): a stationary function of the current core state; no
  magnitude competition; all raw inputs remain always-visible in the core).

Self-pairs (`cb == ca` when `(src+occ) % O_n == src`) collapse the 2-D grid
onto its diagonal = a 6-bin refined 1-D tiling — useful, not degenerate.
Pairs with a constant channel (o[7] DeepSea, o[13] KC, o[1] Chain, o[16]
FourRooms) collapse to single-channel tilings of the other channel — harmless;
utility sorts the constructs inside.

Gate sources deliberately exclude the derived core channels of §2 (gates stay
functions of the *fast current* state, matching today's `curr[gateCh]`
mechanic with zero new plumbing); traces participate as features and G&T
parents. Trace-gating is a noted extension, not v1.

### 1.4 Bank census under the default fleet

| env | (O_n, A_n) | S | occ-0 coverage | zoom banks (occ≥1, even b) | joint banks (occ≥1, odd b) |
|---|---|---|---|---|---|
| HalfCheetah | legacy | — | unchanged (32 banks, b % 30) | — | — |
| FourRooms | (17,4) | 22 | b0-21 | 5 (occ1) | 5: (1,2) (3,4) (5,6) (7,8) (9,10) |
| DeepSea | (8,2) | 11 | b0-10 | 5 (occ1) + 5 @4× (occ2) | **b11: (r,c)**; occ1: (2,3) (4,5) (6,7), act0×c, rew×obs3; occ2: (1,3) (3,5) (5,7) (7,1), act1×obs3 |
| KeyCorridor | (14,4) | 19 | b0-18 | 6 (occ1) | **b19: (x,y)**, **b21: (hasKey,doorOpen)**, (4,5) (6,7) (8,9) (10,11) (12,13) |
| Acrobot | (7,3) | 11 | b0-10 | 5+5 | 6 + 5 joints (pairs cycle by occ) |
| MountainCar | (3,3) | 7 | b0-6 | occ1-4 evens | joints (pos,vel) family at odd b, occ1-4 |
| CartPole | (5,2) | 8 | b0-7 | occ1-3 evens | joints at odd b |
| Pendulum | (4,1) | 6 | b0-5 | occ1-5 evens | joints at odd b |
| Chain | (2,2) | 5 | b0-4 | occ1-6 evens (multi-σ depth on pos) | odd-b joints collapse to refined 1-D tilings of pos |

DeepSea gets an (r,c) joint gate at bank 11 on step 0; KeyCorridor gets (x,y)
and (hasKey,doorOpen). Chain's 16 positions get occ-0 7 bins plus 2×/4× zooms
— resolving pos 14 vs 15 (the reward boundary), which the single 7-tile bank
aliases today.

## 2. Core re-scope: naturalO obs + derived channels (no per-env constant)

Keep `OAK_CORE = 32`, the padded O=17 layout, and the function constants
(`OAK_SPEC_OBS_DIM` etc., :31-44, set once in `Main.swift:366-380`) — all
buffer strides and the cheetah bitstream depend on them. The re-scope is a
*content* change for the region the padded width leaves unused, driven by the
runtime `naturalO`, not a layout change.

In the core write loop (:826-847), under `!legacy`:

- Slots 0..O_n−1, O..O+D, reward, bias: unchanged writes.
- **Slots [O_n, 17) and slot 31 = derived trace channels.** Free-slot index
  `j` (j = i − O_n for i < 17; j = 17 − O_n for i = 31):

  ```
  if (j / O_n < traceLevels)  curr[i] = obsTraceMulti[(j / O_n)·O + (j % O_n)]
  else                        curr[i] = 0.0f        // capacity exhausted (Chain)
  ```

  i.e., fill order is (channel, level) row-major: every informative channel
  gets its rate-1/2 trace first, then rate-1/4, etc. The traces already exist
  and are maintained every step (:798-809, rates 1/2^m on standardized obs) at
  zero added cost; they are ~unit scale by construction (traces of
  standardized channels), so no new standardization is needed. The write sits
  before the existing `mem_device` barrier (:848), so the gate and all
  downstream readers see them exactly like any core slot; the ring write
  (:1263-1268) and `actC[tid] = i` (:844) ride along unchanged.
- Legacy path: slots [O_n,17) don't exist (O_n = 17) and slot 31 stays 0.0 —
  cheetah bitwise.

Effect per env: DeepSea slots 8..16 = trace-1 of ch0..7, slot 16... wait —
j = 0..8 for i = 8..16 gives trace level 1 of channels 0..7 plus trace level 2
of channel 0; slot 31 = trace level 2 of channel 1. Chain (15 free slots,
capacity O_n·traceLevels = 8): position gets 4 trace levels in-core, 7 slots
fall back to 0.0 (documented capacity limit, still strictly better than 15
dead slots). KeyCorridor: channels 0..2 get trace-1 in-core (x, y, hasKey —
the three channels whose history matters most for door credit). FourRooms:
O_n = 17 ⇒ no free obs slots; slot 31 = trace-1 of channel 0 (its only
change to the core).

Why in-core traces (not more gated slots): core slots are *always active* —
they are the only features the trajectory relay and one-step TD can rely on
for temporal continuity, and they are legal G&T parents under the §3
total-functions rule (parent pool = core). This is the representation the
companion sparse-credit design's bounded traces attach to.

Cost: ≤ 16 extra loads + writes per lane-step on sparse lanes; zero on
cheetah. The dead-slot-31 cleanup is free everywhere except cheetah (which
keeps 0.0 for bitwise).

## 3. Generalized GenerateAndTest

Goal: the constructible space must include joint tiles, temporal terms, and
nonlinear interactions over the stream's *informative* channels, allocated by
utility — without env knowledge and without breaking the legacy path.

### 3.1 Parent selection: total functions of state + systematic enumeration

- **Core-only parents on promotion.** The nursery re-pick at :1432-1450
  currently argmaxes `utilFeat` over all i < R. Gated slots are partial
  functions (0 whenever not the bank winner — `curr[i] = 0` for non-winners,
  :881), so constructs built on them are mostly-zero. Restrict both argmax
  loops (:1435, :1441) to `i < OAK_CORE` under `!legacy`. (Init already seeds
  parents core-only, `Main.swift:311-336` — this change stops the promotion
  drift away from that invariant.) Kind-3 trace parents (flat
  `obsTraceMulti` index, :864/:1745) remain allowed — traces are total.
- **Enumerated exploration slots.** Even with core parents, utility argmax
  re-picks the same top pair; the pair space is never systematically covered.
  Odd nursery slots n (`Rn = 256`, :1739-1770) become *stateless enumerated
  generators*: repurpose `nurKind[n]` as a cursor c (init 0), and derive at
  every use (eval :1740-1745, promotion :1418-1424):

  ```
  e  = (n >> 1) + c                    // pair index, advances per promotion
  pa = e % O_n ; pb = (e / O_n) % O_n  // obs core slots
  kind = (e & 1) ? 4 : 0               // alternate joint-cell / product
  cellCode (kind 4) = (e >> 1) % 7
  c ← c + 1 after each promotion of slot n   // replaces (nurKind+1)%4 at :1450
  ```

  No new state (reuses the `nurKind` uint), no host init change (host seeds at
  `Main.swift:311-317` become don't-care for odd n; keep for even n), fully
  deterministic, and wrap-around falls out of the `% O_n` arithmetic. 128
  enumerated slots × cursor advances sweep the full O_n² pair space over time:
  DeepSea (64 pairs) is covered by the *first* cursor round; KC (196) and
  FourRooms (289) complete within a few rounds, with utility-picked even slots
  exploiting early finds in the meantime. Even n keep the existing
  utility-argmax pick (exploitation), with the kind cycle extended %6.

### 3.2 New construct kinds (new path only)

Evaluated in the bank loop (:858-870) and identically in the nursery
(:1744-1751), from `prev` parents exactly as today (features are functions of
the previous core state — existing semantics, unchanged):

- **kind 4 — joint-cell indicator**: `raw = (joint_cell(prev[pa], prev[pb],
  occ=1) == cellCode) ? 1.0f : 0.0f`, reusing §1.3's `joint_cell` (one shared
  inline). cellCode rides in the high half of `featureParentB`
  (`pb | code << 16` at promotion; use sites mask `& 0xffff` instead of `% R`
  on the new path — pb < R by construction). This is how the agent *builds*
  explicit cell detectors (e.g. DeepSea "(r,c) = cell k") by utility, on top
  of the enumeration guarantee.
- **kind 5 — signed difference**: `raw = (b − a)·0.7071067811865475f`
  (complements the kind-1 sum; goal-direction and velocity-difference terms).

Kinds 0-3 keep their exact legacy definitions. Legacy path keeps `% 4u` at
:860/:1742 and the legacy parent/kind plumbing everywhere — bitwise. Note
kind-0 products over the new in-core traces (§2) already give trace×state
temporal interactions for free; no separate trace-product kind is needed.

Standardization is the existing per-slot EMA (:871-878): a kind-4 indicator
with activation probability p standardizes to amplitude ≤
`(1−p)/√(p(1−p) + ridge)`, bounded by `√O ≈ 4.1` via the ridge floor
(max(1/(age+1), 1/O), :877) — the same Mark III law (b) floor as every other
feature. No new magnitude path is introduced.

### 3.3 Interaction with geometry gating

The two halves compose multiplicatively: a joint *gate* (§1.3) gives
state-localized selection among 7 utility-evolved constructs (a conditional
feature bank), while kind-4 *constructs* give explicit, relocatable cell
detectors that G&T can place anywhere — including inside a gated bank whose
gate further conditions them. Rarely-active slots age only when active
(:886-887), so their EMA rate `max(1/(age+1), 1/R)` is sample-based — correct
for cells visited 1% of the time — and the eligibility gate (featureAge ≥ R,
:1804) protects young tiles for 256 *active* steps before they can be
evicted.

## 4. Head compatibility: R is fixed, slot meaning is per-shape

Every downstream consumer indexes features by opaque slot id in [0, R), reached
only through `actIdx`/`ringIdx` or slot-generic loops:

- value dots and normalizers: :441-456, :478-484 (`act_of(actP, a)`, :49-51);
- psi/hDot/model gathers: :548-565 (slot·K + k indexing);
- plan stash & re-gather: :919-938; learn rows: :1053-1151; value bank:
  :1139-1150; emit gathers: :1176-1208; blend: :1219-1239;
- ring write/read: :1263-1268, :1279-1282; dyna ring dots: :2559-2562,
  :2585-2592; twin copy-2: :2653-2678;
- G&T eligibility/utility/redundancy, metaopt, bonusW, optFeature: loops over
  `i ∈ [anchorCount, R)` or per-slot arrays (:1310-1520, :1737-1814) — all
  slot-generic.

None of them knows what a slot *means*. R=256, K=32, `OAK_ACT=64`,
`OAK_CORE=32`, the ring's 64-float/32-uint format, and every arena stride
(:200-334) are unchanged, so paramTotal/stateTotal/uintTotal
(`Main.swift:255-258`) are unchanged and per-lane parameters (no cross-lane
sharing exists) simply learn whatever their lane's slot geometry means. The
slot→meaning map is fixed within a run (naturalO is constant per lane), so
within-run coherence is trivial; across runs there is nothing to migrate.
`optPolicy` is K·(D/2)·R regardless; emit's tanh means and muRaw logits
(:1194-1213) are untouched. **Conclusion: a per-shape geometry is fully
transparent to every head** — the change is confined to FeatureConstruction,
the gate helpers, and G&T.

## 5. Performance budget and the bitwise/gate split

### 5.1 Added work (sparse lanes only)

Per lane-step: gate function 32 banks × ~12 flops ≈ 400; derived core ≤ 16
loads + writes; kind-4/5 branches ≈ 224 slots × ~6 ≈ 1.3k; nursery enumeration
replaces lookups with ~6 flops of index math. Total ≈ +2k flops against a
≈60-80k fma/lane-step baseline (learn rows ~25k, fcstomp gathers ~20k, emit
~12k, dyna ~8k, encode/normalizers ~5k) → **≤ +3% on sparse lanes, ≈ +0.8%
fleet-wide** at the default 384/1408 sparse share. Warp structure untouched:
one bank per thread (:827, :850), 32-wide simd reductions unchanged, no new
cross-thread communication, no new arenas (cellCode packs into parentB), no
heap (Inv-6), every mechanism every step (Inv-7). All constants derive from
T=7, B=32, R=256, O=17, D=12, traceLevels (Inv-9).

### 5.2 Bit-identical class (cheetah)

The legacy predicate makes cheetah's code path textually identical: same gate
channels, same tiles, same core writes (slot 31 stays 0.0), same kinds,
same nursery plumbing. Acceptance proof (the standing lane-disjointness /
bit-identical bar): candidate build vs keeper build,
`OAK_BENCH_ENVS="HalfCheetah-v4:1024"`, same seeds/steps, `cmp` on
`actions.bin` (recorded seeds) and on `state-hashes.jsonl`
(`OAK_DUMP_STATE_EVERY`) — the hashes cover params/state/uints/scalars only
(`Main.swift:382-387, 923-926`), and the lane-struct growth (400→408 B) does
not touch any of those arenas, so the A/B remains well-defined. `fleet.jsonl`
reward streams are a third byte-compare. Any cheetah delta = hard fail.

### 5.3 FP-reorder / behavior class (sparse envs) — the gate plan

Everything on sparse lanes changes behavior ⇒ the paired population gate:

1. Tier-S smoke (1024×10k, default mixed schedule): hard-fail screen.
2. Tier-M (1024×100k) `tools/fleet.py gate` vs the current keeper
   (`.oak-benchmarks/mk8-tierL-20260705` lineage): all four criteria, paired
   by seed. Cheetah lanes should show Δ≈0 (they are bit-equal in
   distribution); the sparse lanes carry the signal.
3. Tier-L for promotion, plus per-env stratification of the report
   (fleet.jsonl already records per-lane `envId`): headline expectation is
   DeepSea/KeyCorridor/Chain IQM delivered rate lifting from ≈0 to >0 and
   `%>0` per env, with divergence non-inferiority. The gate's IMPROVEMENT
   flag is the SOTA-claim bar as usual.
4. Ablation matrix for the experiment (three new `OAK_ABLATE` compile
   variants, following the named-organ convention): `derived-core` (§2 off),
   `joint-geometry` (§1.3 joints+zooms off, informative-source cycle kept),
   `extended-constructs` (§3 kinds/enumeration off). Run candidate-full vs
   each single-organ ablation at Tier-M on the mixed schedule
   (`tools/ablate.py` pattern), per-env conviction table.
5. Pre-registered kill rules: sparse-env IQM not improved at the 100k common
   milestone; divergence Δp regression beyond δ; cheetah not bitwise (hard
   fail at any point).

## 6. Explosion / degeneracy risks and guards

1. **TOP RISK — utility circularity on rare cells.** A joint-cell slot active
   1% of steps accrues utility/downstream-use slowly; once eligible (256
   active steps ≈ 26k wall steps at 1%), the replacement test
   (`lowDownstreamUse && (lowUtility || highRedundancy)`, :1407-1416) can
   evict a cell feature before its credit role materializes — churn that
   re-imports the "never localizes" failure inside the new geometry.
   Mitigations in place: the age gate is in *active* steps; redundancy
   `|curr[i]·curr[pa]|` = 0 on inactive steps reads as non-redundant; the
   enumeration re-supplies evicted pairs deterministically. Guard to watch:
   per-env feature-reinit rate and `planGap` correlation on sparse lanes
   (`tools/fleet.py lanes` health medians); if churn dominates, the fix is a
   visit-count-weighted utility, designed at the pathology's own depth — not
   a freeze (Mark IV law (d)).
2. **Constant-channel pairs** (o[7]/o[13]/o[1]/o[16] bias channels):
   joints collapse to single tilings — harmless, utility-sorted.
3. **Standardization warm-up**: early obsMean/SM estimates push z outliers
   into clamp bins / overflow cell 6; self-corrects at `track_rate` and no
   warmup gate is permitted (Inv-7). Early-born constructs see skewed
   activity for ~O(O) steps — acceptable, same class as today's trace
   warm-up.
4. **Kind-4 amplitude**: bounded at √O by the ridge floor (:877) — no new
   explosion path; weights, `safe_w`, and the no-caps invariant are untouched
   (feature *values* are bounded by construction: indicators in {0,1},
   products of standardized channels as today's kind 0).
5. **Enumerated coverage vs engineered obs**: FourRooms' 17 channels include
   one-hots/wall sensors, so many enumerated pairs are redundant; the static
   first round covers 128/289 pairs, the rest via cursor rounds and utility
   picks — noted limitation, adaptive fill.
6. **Predicate fragility**: a future (17, 6)-shaped stream inherits cheetah's
   legacy geometry by design (shape, not identity — Inv-1). If that is ever
   wrong, the answer is a better shape-derived rule, not an env branch.
7. **Plumbing default**: any make_ctx caller without lane shape must default
   to `(O, D/2)` = legacy — never to (0,0), which would make S = 1 and gate
   every bank on obs channel 0.

## 7. Implementation touch list (for the parent; no edits made here)

- `Sources/OakKernel.metal`: `OakFleetLane` +2 uints (:1906-1921, assert
  408); `oak_fleet_init` spec write (:1975); `Ctx` fields (:114) + `make_ctx`
  (:164-199) and its 8 call sites; gate helpers — `legacy` predicate,
  `gate_source/ch`, zoom tile, `joint_cell` (new inlines near :346-355);
  bank loop (:850-888); core derived write (:826-847); construct kinds +
  parentB pack (:855-870, :1740-1751); nursery enumerated branch +
  core-restricted picks + cursor (:1418-1450, :1739-1770); kind modulus
  %4→%6 under `!legacy` (:860, :1424, :1450, :1742).
- `Sources/Main.swift`: Swift lane mirror (+8 B, the `static_assert` partner;
  cf. the LaneSnapshot mirror pattern at :593-594). Nothing else — arenas,
  KernelConfig, function constants, record format all unchanged.
- `tools/fleet.py`: per-env IQM/%>0 stratification in `report` (envId is
  already per-lane; add the group-by if missing) — analysis requirement, not
  a behavior change.
- New `OAK_ABLATE` organs: `derived-core`, `joint-geometry`,
  `extended-constructs`.

## 8. Hand-off to sparse-credit-design.md

This representation is the prerequisite; the companion doc should consume,
not reinvent: (i) in-core multi-timescale traces of every informative channel
(§2 — its bounded traces attach here); (ii) state-localized feature activity
via joint gates (§1.3); (iii) utility-evolved explicit cell detectors (§3
kind 4) and reward×state joint gates (reward is source S−1, so odd-b occ≥1
banks pair it with obs channels — reward-site localization for free);
(iv) the rare-cell churn risk (§6.1) which its exploration/utility terms must
be designed against.
