# Design — per-organ reward/GFLOP instrument + utility-driven allocation (2026-07-20)

Architecture dossier. **No code landed here** — every edit is parent-executed after the in-flight fleet run drains. Scope: the measurement layer (per-organ reward/GFLOP) and the allocation framework (K, R, ring M, planning budget, memory-for-compute). Feature-gate geometry and sparse-credit mechanisms belong to the companion agents; this doc only builds the instruments that *surface* degeneracy and the allocation rules that *act* on it.

Ground truth verified against the live tree (all line numbers 2026-07-20):

- **R = 256** — already stream-derived: `inputW = (⌊log2 O⌋+1)·O + D + 2 = 5·17+12+2 = 99`, `R = 2^ceil(log2(2·inputW)) = 256` (`Sources/Main.swift:241-247`, `Arena.input` :137-140).
- **K = 32, pinned** — `let k = 32` (`Sources/Main.swift:253`; tg sizing comment "K = 32, Engine's hardcoded option count" :517-520). **AGENTS.md's "K = R" (Inv-9 text + Step-11 bullet) is stale folklore** — the live engine has K=32 hard-coded since the warp-width consolidation. Fix AGENTS.md when Phase 8 lands.
- **Ring M = R = 256 slots** — `c.ring = ...; s += 64u * c.R` (`Sources/OakKernel.metal:295`), `uint M = ctx.R` (:2539). Per slot: 64 f32 active values + 32 u32 gated indices.
- **Planning budget** — sweep = 8 threads × M/8 priority scan (`tid < 8u`, :2544); 4 candidates `{0, selected, kbBest, (step+tid)%K}` (:2579-2580); copy-2 twin update on all 32 threads. The 32-thread sweep was measured ~3× SPS cost and budgeted down (:2534-2537 comment).
- **Static FLOP model exists** — per-stage, ablation-aware, host-side, written to `meta.json flopsPerLaneStep` (`Sources/Main.swift:1338-1392`); `tools/table.py` already prints reward/GFLOP per env (`table.py:145`).
- **Megakernel stage bits** — 1=encode, 2=plan(+dyna), 4=learn, 8=replace(+window analytics), 16=emit, 32=sample, 64=env (`OakKernel.metal:2720-2786`). `OAK_MEGA_STAGE_MASK` does per-stage timing by successive diffs.
- **The keeper ships `OAK_ABLATE=planning`** — the entire dyna/Planning organ (38.8% of the nominal static model) is already switched off in the promoted configuration.

Per-lane memory (B=1408 fleet): state 27,872 f32 + params 114,944 f32 + uints 10,763 u32 = **0.61 MB/lane → 0.86 GB of 36 GB unified memory (2.4%)** on the M4 Max. The machine is ALU-bound with ~35 GB idle: that asymmetry is the whole argument for trading memory for compute.

---

## 1. Per-organ FLOP attribution (Phase 8a — static model, organ-binned)

Organs (FC-STOMP vocabulary) are not stages; each organ maps to a set of (stage, loop, trip-count) terms. The existing `Main.swift` static model is the right skeleton — it needs re-binning from per-stage to **per-organ**, plus three corrections found by reading the actual loops:

1. **Twin dot undercount**: the encode K-gather computes 11 dots/option-step (psiPrev, psiNext, hDot = 3A; rHat, lRaw, vModel, vModel2 = 4A; pdVs, pdVsA, pdR, pdL stash = 4A — `OakKernel.metal:548-565`), the model counts 10KA (3KA encode + 4KA plan + RR's 3KA is right; the second twin `vModel2` dot is uncounted). Add KA to OptionModels.
2. **FeatureConstruction is absent** from the model: obs stats + trace updates (O + O·traceLevels ≈ 102), core write (32), gated-bank eval (32 banks × 7 slots × ~10 flops ≈ 2,240 — `OakKernel.metal:850-883`). ≈ 2.4K flops (~1% keeper) — small, but it is the organ whose *yield* is under indictment (degenerate banks), so it must appear in the table to carry a 0-yield verdict.
3. **ActionEmission (sample stage) is absent**: Philox + Box-Muller per dim (:2248-2270), credit staging K·D + importance ratio K (:2286-2302), Gumbel-max path for discrete. ≈ 1K flops.

### Keeper-config organ table (static model, `OAK_ABLATE=planning`, flops = 2×fma)

| Organ | Terms (keeper-live) | flops/lane-step | share |
|---|---|---|---|
| OptionPolicies | learn apply KA·DL + emit means KA·DL + polCoef K·DL + consensus DL·K+DL·A | 50,688 | **41.2%** |
| OptionModels | encode 4KA (+twin fix) + plan stash 4KA + learn 2KA + plan_one scalars | ~38,900 | **~31%** |
| RewardRespectingSubtasks | encode psi 2KA + hDot KA + learn 2KA + headScale/psiRho K | 20,480 | 16.7% |
| misc (scalar/serial/stats) | fMisc | 8,000 | 6.5% |
| SearchControl+utility scans | window A·K, prioOpt K, ESS/entropy (¼-step) | ~4,100 | 3.3% |
| FeatureConstruction | (correction 2) | ~2,400 | 1.9% |
| GVFPrediction | value dots 2A + norms 6A + value learn 2A | 1,280 | 1.0% |
| ActionEmission | (correction 3) | ~1,000 | 0.8% |
| OptionKeyboard | softmax refresh K + blend D·K + GPI scans | ~800 | 0.6% |
| GenerateAndTest | nursery Rn/2 + eligScr R + LCB scan | ~800 | 0.6% |
| Planning (dyna) | **ablated in keeper** (nominal: 38.8% of 200,768) | 0 | 0% |
| **Agent total (keeper)** | | **~123–129K** | 100% |
| env (cheetah / gridworld) | | 20,000 / ≤200 | 14% / 0.16% of lane-step |

Two load-bearing readings:

- **The K-bank machinery (policies+models+RR) is ~89% of keeper agent compute.** Every one of those terms is a dense scan over the full option bank and the 64-wide active set — on lanes where most options are inert (e.g. geometry-degenerate sparse-env lanes), nearly all of it multiplies zeros. That is the quantified form of the root-cause complaint.
- **The Planning organ's historical verdict is the proof the instrument works**: ablating 38.8% of nominal FLOPs *raised* IQM (positivity 24.6%→97.8%). Its reward/GFLOP was negative; the organ table would have shown it years of theorizing earlier.

### Calibration (mandatory before any absolute claim)

The static model counts static trip counts (fma=2 flops), not silicon. Sanity: keeper model = 142,944 flops/lane-step incl. env; at the measured ~470K agg steps/s (B=1024) that implies ~67 TFLOPS against a ~14-TFLOPS-nominal 32-core M4 Max — the model over-attributes ~3–5× (dead lanes, idle lanes in masked stages, scalar inflation in `fMisc`). Rules:

- The organ table is consumed as **relative shares** and **per-organ Δ across ablations**, never as absolute silicon FLOPs.
- Calibrate once per keeper promotion: a stage-mask sweep (`OAK_MEGA_STAGE_MASK` successive diffs, the existing SPS-attribution instrument) normalizes static shares to measured wall shares; store the calibration vector next to the model.
- Version the model: `organCostVersion: 1` in `meta.json`. Cross-version absolute comparisons are refused (same discipline as `envSchedule` mismatch in `table.py`).
- AGENTS.md's per-FLOP SOTA rule stands: no external per-FLOP claim until both sides are instrumented; this instrument is the internal half.

### Runtime counters — rejected as the primary mechanism

Per-stage/per-head in-kernel FLOP counters would measure *executed* work, but: loop trip counts are already compile-time-derivable (the megakernel is branch-light on the hot dots), counters cost device RMWs on the hot path (Inv-6-adjacent SPS risk), and they cannot attribute *wall* time anyway (the env-window fan-out overlaps organs across warps). Verdict: static model + stage-mask calibration. An optional counter build in the `OAK_PROBE_*` convention (timing instrument, trajectories garbage) may validate the model once; it is not a standing instrument.

---

## 2. Reward contribution per organ (Phase 8b — three tiers, ascending conviction)

### Tier 0 — online organ-yield signals (zero new kernel cost, every heartbeat)

Per-organ activity/yield proxies already flow through `KernelStats` → `fleet.jsonl` health. Formalize the mapping (no behavior change; reads only):

| Organ | Existing online signals |
|---|---|
| FeatureConstruction | `oakFeatureReinitCountFinal` (churn) |
| GVFPrediction | `mainTd`, `kernelRho` stability, NF-by-phase encode/learn |
| RewardRespectingSubtasks | `terminationRate`, `selectedContinuation`, `optionDurationMean` |
| OptionPolicies | `actorAdvantageL1`, `actorUpdateNormMean`, `actorSupportEntropy`, `actionDeltaL2Mean` |
| OptionModels | `modelPriorityMax` (= max modelErr via prioOpt), `importanceEss` |
| Planning | `planTd` (diagnostic-only since Phase 0) |
| OptionKeyboard | `keyboardL1`, `keyboardEffectiveL1`, `keyboardOrigin`, `selectedKeyboardWeight` |
| SearchControl | `searchControlUseMean` |
| GenerateAndTest | `generateTestEligibleCount`, option/feature reinit counts |
| ActionEmission | `actorSupportEntropy`, `actionDeltaPeak` |

Dead/missing signals to revive or add (Phase 8b kernel delta, **bit-identical class** — new write-only stats fields, never read by behavior): `dynaBackupMean` and `learningMature` are currently zeroed (`OakKernel.metal:1876,1884`) — revive; add five EMA floats computed in the existing `tid==0 && (tPrev&3)==0` stats block (~50 flops/4 steps, negligible): `gatedBankActivityMean` (mean |utilFeat| over gated slots — the FourRooms 28/32-degenerate-bank meter), `optionLiveCountMean` (options with recent continuation/keyboard mass — the K-allocation meter), `planGapMassMean` (mean |planGap| over written ring slots — planning-yield meter), `ringPriorityMassMean` (Σ dynaTd over written slots — search-control health), `nurseryLcbMax` (G&T pressure). Struct growth: `KernelStats` +5 floats ⇒ `OakLaneSnapshot` 280 B static_assert and the Swift mirror (`Main.swift:594`) move in the same commit (keeper-cycle step 2 law); `fleet.py` health readers are additive-safe.

**Standing rule** (organ-bisect law): Tier-0 yields rank hypotheses; they never convict and they are never written into learning targets (Mark IV law (a)).

### Tier 1 — paired ablation matrix at Tier-S, machine-readable (the workhorse)

Upgrade `tools/ablate.py` (it currently prints a text table and keeps nothing):

- Output `ablation-matrix.json` (baseline + one run per organ): per env, IQM delivered rate, divergence rate, positivity, and **paired-by-seed bootstrap CI of ΔIQM vs baseline** (reuse `fleet.py paired_bootstrap`, import not copy).
- `--tier S|M` flag. Tier-S (1024×10k, ~39 s/run) screens the full 13-run matrix in **~8.5 min** at divergence resolution ≳4% and large IQM moves; organs whose Tier-S CI straddles 0 escalate to Tier-M (~6.4 min each, ≤ ~77 min for the remainder). Whole audit < 2 h — inside the protocol wall-clock law. Runs stay sequential (workspace lock serializes anyway).
- The per-organ headline number: **reward forgone per GFLOP = ΔIQM_env(organ) / (organ static share × run flopsPerLaneStep)**. Three verdict classes: Δ ≈ 0 within CI → *inert at this horizon/env* (purge candidate); Δ < 0 → *toxic* (Planning, 2026-07-04); Δ > 0 → *load-bearing at price*.
- Coupling guard (50k focus-matrix lesson): include the named coupling ablations (`model-to-keyboard`, `keyboard-to-policy`, `search-to-generate-test`) in every matrix — single-organ deltas alone do not license purges; the aux-head law (mk6 REJECT: aux learners regularize the representation with consumers dead) means near-zero yield is *necessary, not sufficient* for removal.

### Tier 2 — Tier-M/Tier-L confirmation (conviction)

Any allocation change justified by Tier-0/1 still runs the standard ladder: Tier-M paired screen, Tier-L `fleet.py gate` vs the keeper for anything promoted. The instrument never replaces the promotion law; it decides what is *worth* a Tier-L slot.

---

## 3. The artifact and the consumption loop

Artifacts:

1. **`meta.json` (every run)** — gains `organFlops: {organ: flopsPerLaneStep}` (ablation-aware, exactly the §1 table recomputed from the run's own `OAK_ABLATE` set) + `organCostVersion`. Pure host arithmetic — no kernel change, byte-comparable runs stay byte-comparable.
2. **`<run-root>/organ-report.json`** — written by `tools/fleet.py report` (new "Organ yield × GFLOP" section): per organ, static share, Tier-0 yield trend (heartbeat quartiles), and the last convicted matrix deltas when `.oak-benchmarks/ORGANS.json` is present.
3. **`.oak-benchmarks/ORGANS.json`** — the cross-run ledger: pointer to the latest `ablation-matrix.json` + per-organ verdict class (toxic / inert-at-horizon / load-bearing). The organ-bisect skill's step 4 consumes it; the keeper-cycle's step 1 cites it ("name the FC-STOMP section" now has a price tag attached); promotion commits append the candidate's organ table to the roadmap ledger.

Consumption loop: **Tier-0 yield (every heartbeat, free) → suspects ranked → Tier-1 matrix (paired, ~10–90 min) → verdicts in ORGANS.json → Phase-9 allocation candidates → Tier-M/Tier-L gate → keeper commit records the new organ table.** The instrument (Phase 8) is the prerequisite that makes the purges (Phase 9) measured rather than vibes — and it is also the *acceptance test* for the companion agents' work: geometry fixes must move `gatedBankActivityMean`; sparse-credit fixes must move `planGapMassMean` on sparse envs.

---

## 4. Allocation derivations (Phase 9)

Invariants respected throughout: Inv-6 (all arenas init-sized; runtime sizing = meta-level derivation within arena-max bounds, never per-step malloc), Inv-7 (every assessment runs every step; commits are structural events — the existing age/utility G&T pattern, no step-count gates; age-based maturation is the Inv-9-sanctioned `1/(age+1)` family, not a warmup), Inv-9 (every size below is a pow2-of-stream-shape or an existing structural statistic — no new numeric constants), Inv-10 (each candidate is classified below).

### 4a. Option count K — "arena-max, live-derived"

- **Physical bank K_max = 32 stays** (warp width; the K-wide tg stash is sized `(5K+1)·G·4B` = 20.6 KB at K=32 and would be 41 KB > Apple's 32 KB threadgroup cap at K=64 — growth requires a tg-diet rework first; see §5).
- **Live bank K_live: derived by per-option hibernation.** Predicate, evaluated every step from existing EMAs:
  `park(k) ⟺ optAge[k] ≥ R  ∧  utilOpt[k] < mean(utilOpt over matured options)  ∧  k ∉ {0, selectedOption, vacuumProbe}`.
  All three constants are structural: the maturation horizon R already gates keyboard eligibility (`OakKernel.metal:731`) and nursery LCB (:1788); the threshold is the same eligible-mean statistic `oak_replace_commit` already computes for features (:1403-1410); the exemptions protect the main-task key, the committed option, and the exploration probe. Commit: ≤1 park per step (mirrors the single-reinit `break` :1399). Parked = outputs zeroed (continuation/keyboard/tdPsi), **rows kept** — hibernation, not deletion; parked slots are the first victims of the existing staleStop reinit (:1372-1400), which doubles as the revive path.
- **FLOP effect**: per parked option, skip the A×D-priced loops — encode gather 11A, learn policy/psi/model rows (~A·DL+5A), emit 6A ≈ 3.6K flops/step — while **keeping the K-priced scans full-width** (their parked terms are exact zeros / neutral elements, so utility EMAs and softmax masses are bit-stable). 28 parked options ≈ **100K of 123K keeper agent flops freed (~81%)** on a fully degenerate lane. That is the FourRooms purge, measured by `optionLiveCountMean` and the per-env organ table instead of asserted.
- **Inv-10 class**: behavior change (hibernating options alter learning) → Tier-M paired screen + Tier-L gate. The skip-the-loops mechanics introduce no FP reorder (skipped terms were exact zeros); the A/B bitwise harness (`actions.bin` cmp + state-hash streams) validates the *mechanics* on a forced-park probe build before the paired gate spends a Tier-M.

### 4b. Capacity R — keep the existing derivation

`R = 2^ceil(log2(2·inputW))` already satisfies "derived from stream shape" (Inv-9). Two notes:

- A per-env `naturalO`-derived R would violate Inv-1's universal I/O contract (obsDim=17/D=12 function constants for every env). The sparse-env waste is **gate geometry seeing zero-padded channels** (companion agent's scope), surfaced by `gatedBankActivityMean` — not an R defect.
- R anchors half the agent's structural constants (maturation horizon, staleLimit R·K, metaEta 1/(R(R+D)), track_rate floor 1/R, nursery Rn=R, ring M=R). Any R change is one coherent paired-gate candidate, never a tune.

### 4c. Ring M — the memory-for-compute flagship

- **Derivation: M = 4R = 1024.** Structural anchors: pow2-of-R (Inv-9); ≥ the gym 1000-step episode semantics (one full external-comparability episode of relay horizon); covers the sparse-env credit chains (KeyCorridor key→door→goal, DeepSea exploration runs) that M=256 truncates — the trajectory relay is the *primary* credit mechanism when planning is live (`OakKernel.metal:2610-2616`, Holland-2018 rationale).
- **Memory**: ring values 64·M·4 B = 256 KB + indices 32·M·4 B = 128 KB + side arrays (planU/ringReward/planGap/dynaTd) 16 KB ≈ **400 KB/lane → +420 MB at B=1408** (fleet total ~1.3 GB of 36 GB). M=16R=4096 = +1.7 GB, also affordable — start with 4R; escalate only if the gate promotes and sparse-env curves still read horizon-bound.
- **Compute**: candidate dots unchanged (4×3A per backup); the priority scan grows to M/8=128 streaming reads per sweep thread — memory-bound, and deleted entirely by the priority index (§4d).
- **Sequencing guard**: the ring is fully dead under `OAK_ABLATE=planning` (writes guarded, :1259). M growth pays only after the PLAN-SELECT/TRUST-SWEEP re-promotion (`docs/research/planning-coupling-2026.md`); landing it before then is inert memory. Class: behavior change (longer relay horizon → different U targets) → Tier-M screen, Tier-L gate, per-env verdicts (the hypothesis is sparse-env-specific: expect Δ≈0 on cheetah, Δ>0 on KeyCorridor/DeepSea — a clean instrument test).

### 4d. Planning budget — composition derived, width kept, scan indexed

- **Sweep width: keep 8 threads.** The 32-thread sweep is a measured ~3× SPS regression (:2534-2537); P = warp/4 = 8 stands as the structural budget.
- **Candidate composition (zero-FLOP utility derivation)**: replace the blind round-robin 4th candidate `(stepCount+tid)%K` with `argmax_k prioOpt[k]` — the highest-model-error option is precisely the one the VI backup most needs to re-evaluate (Moore-Atkeson on the option axis, riding the existing K-scan, ~K flops). Class: behavior → paired gate.
- **Priority index (the M-scaling enabler, bit-identical candidate)**: a per-lane top-8 `(prio, slot)` index maintained at the two `dynaTd` write sites (ring write :1288, backup write :2628) with scan-identical tie semantics (strictly-greater, lowest slot wins — :2552). Comparison-only, no FP arithmetic → ship on the bitwise A/B proof + Tier-S smoke, no gate needed. Replaces the M/8-per-thread scan with O(1), which is what makes M=4R/16R compute-free.
- **Backups/step P**: 8 stays; P=16 is a gate candidate only after the index lands and `planGapMassMean` says the sweep is convergence-bound, not scan-bound.

---

## 5. Memory-for-compute trade list, ranked by expected reward/GFLOP

Budget: 0.86 GB used of 36 GB (2.4%) at B=1408; Apple-recommended Metal working-set ceiling ≈ ⅔ unified ≈ 24 GB. Headroom ~27×.

| # | Trade | Memory Δ (fleet) | Compute Δ | Expected payoff | Class |
|---|---|---|---|---|---|
| 1 | **Ring M = 4R** (with §4d index) | +420 MB | ~0 (scan indexed; dots unchanged) | The sparse-env credit lever: relay horizon 256→1024. Conditional on planning re-promotion — land in that candidate, not standalone | paired gate |
| 2 | **K-parking (§4a)** | 0 | **−up to ~80% agent FLOPs on degenerate lanes** | The purge lever itself: same reward at a fifth of the GFLOP on inert-bank lanes; the per-env organ table is the before/after proof | paired gate |
| 3 | **Priority index (§4d)** | +32 B/lane | −M/8 reads/thread-step | Enabler for #1 at 16R; standalone SPS win | bitwise A/B |
| 4 | **Arena-max option bank K_max=64** | +270 MB (params) | 0 at K_live≤32 | Only when `optionLiveCountMean` saturates 32. **Blocked on tg-diet**: K=64 stash = 41 KB > 32 KB threadgroup cap — split the 4K prev-dot stash (2K prev + 2K re-gather) first | paired gate when unblocked |
| 5 | **Nursery Rn = 2R** | +11 MB | +Rn/2 evals/step (~256 flops) | More G&T candidates in flight — Mark III law (a): generate-and-test rate is the ignition engine. Gate decides; `nurseryLcbMax` + reinit stick-rate measure yield | paired gate |
| 6 | **Precomputed constants → KernelConfig** | ~0 | ~0 | **Mined out** — invRank/sampleSigma/actorAlphaDenom already host-precomputed (`Main.swift:18-23`); remaining per-step recomputes are scalar-cheap. Explicit no-op, recorded so it isn't re-proposed | — |
| 7 | **Cross-step model-dot memoization** | — | — | **Dead on analysis**: every model row updates every step, so any cross-step cache of model dots is stale immediately. planU persists precisely because it is planner-owned state, not a cache. Recorded as rejected | — |
| 8 | Fleet B / horizon | — | — | Fixed by the tier-ladder law; not an allocation lever here | — |

Non-obvious ordering rationale: #1 and #4 spend memory; #2 and #3 *refund* compute; the compound sequence is 3→1→2 (index, then ring, then parking) because each makes the next cheaper to gate. #5 is the only genuine both-axes spend and therefore last among kernel changes.

---

## 6. Risks and guards

1. **Static-model drift → wrong purges.** The model is a model (§1 calibration shows a 3–5× overcount vs silicon). Guards: `organCostVersion` gating comparisons; stage-mask calibration at every keeper promotion; organ shares consumed as ratios; absolute per-FLOP claims stay internal-only per the SOTA rules.
2. **Whack-a-mole purges from single-organ deltas.** The 50k focus matrix proved whole-loop dependence; mk6 proved near-zero-yield organs can still regularize. Guards: coupling ablations in every matrix; the two-stage rule (yield ≈ 0 necessary, paired-bisect + Tier-M sufficient); ORGANS.json records REJECTED removals as discoveries, per keeper-cycle law.
3. **K-parking starves exploration.** Guards: probation window (optAge < R unparkable), probe/selection/option-0 exemptions, hibernate-don't-delete (rows intact, staleStop reinit recycles), one-park-per-step commit cap; Tier-M watch metrics are `actorSupportEntropy`, ignition rate, and `optionLiveCountMean` itself.
4. **Ring-M wrap/seed semantics.** planU is seeded from v at ring-write and the relay already guards `nextSlot < written` (:2613) — both scale with M unchanged; the M change rides the planning re-promotion candidate so its gate measures the *compound* (the honest comparison: keeper vs planning-rebuilt-with-long-ring, since planning-alone is already known-toxic at any M).
5. **Stats-struct drift.** KernelStats +5 floats moves `OakLaneSnapshot` past its 280 B static_assert: kernel struct, Swift mirror, and the precondition value change in ONE commit (keeper-cycle step 2); fleet.py/table.py readers are additive-safe (new health keys only).
6. **Inv-7 phrasing.** Every sizing mechanism above is an every-step assessment with a structural commit — the sanctioned G&T pattern. Nothing keys on absolute step counts; maturation uses age (the Inv-9 `1/(age+1)` family). No warmup gates, no frozen post-N sizes.
7. **Fleet-in-flight contamination.** No kernel/tool edits while a run holds the workspace lock (also why this doc is design-only). Phase-8a (meta.json organ block) is host-only and still waits — a dirty tree is a config-mismatch risk for the next table.
8. **Doc drift.** Landing Phase 8 must update AGENTS.md where this dossier found it stale: "K = R" (actual: K=32 pinned), the learnable-heads table (K·R heads are 32×256, not R²), and the invOptionCount/metaEta derivations that cite K=R.

---

## Phase plan (parent-executed)

- **Phase 8a** (host-only, bit-identical): organ-binned `organFlops` + `organCostVersion` in meta.json; `fleet.py report` organ section → `organ-report.json`; `ablate.py --tier` + `ablation-matrix.json` + ORGANS.json ledger. Verify: rebuild, `tools/check.sh` at 100k byte-comparable vs pre-change root (actions.bin cmp), one Tier-S matrix (~9 min) as the smoke.
- **Phase 8b** (kernel instrument, bit-identical class): +5 stats EMAs, revive `dynaBackupMean`/`learningMature`; mirror + precondition in one commit; A/B proof (actions.bin cmp + state-hash streams) + Tier-S smoke.
- **Phase 8c** (calibration): stage-mask sweep on the next idle window; store the calibration vector; first full Tier-S organ matrix → ORGANS.json v1.
- **Phase 9** (allocation, each its own candidate up the ladder, in order): 9.1 priority index (bitwise A/B) → 9.2 ring 4R folded into the planning re-promotion (Tier-M screen, Tier-L gate) → 9.3 K-parking (forced-park probe A/B, then Tier-M/Tier-L) → 9.4 candidate-4 = argmax prioOpt (gate) → 9.5 nursery 2R / K_max=64-with-tg-diet, only on instrument evidence of saturation.
