---
name: keeper-cycle
description: The OaK keeper-promotion loop that produced three PROMOTEd keepers in two days (IQM −0.01 → +0.44). Use whenever building/landing any kernel change — it enforces the check→launch→heartbeat→gate→commit→ledger discipline end to end.
---

# Keeper cycle: one diagnosed change, one visible run, one gate, one commit

## The loop

1. **Name the mechanism** before editing (FC-STOMP section + which measured law
   or forensic conviction motivates it). If you can't cite a conviction, run
   /organ-bisect first.
2. **Edit + build**: `xcrun -sdk macosx metal -fno-fast-math -c
   Sources/OakKernel.metal -o .build/OakKernel.air && xcrun -sdk macosx
   metallib .build/OakKernel.air -o .build/OakKernel.metallib && swift build
   -c release`. Kernel arena and Swift `Arena` MUST stay mirrored (a mismatch
   = instant fleet-wide garbage; check both on any state addition).
3. **Verify at 100k, visibly**: `tools/check.sh <mk-name>-check-100k 100000
   planning`. Bars: 0 deaths, 0 NF, median ≥ keeper curve at 100k, SPS ≥
   keeper. A 30k check cannot clear a horizon pathology.
4. **ONE Tier-L**: `OAK_ABLATE=planning tools/run_fleet.sh 1000000
   <mk-name>-tierL-<date> .oak-benchmarks/<current-keeper>` + a Monitor on the
   log grepping `step .* alive|PROMOTE|REJECT|FAIL\]|PASS\]`.
5. **Heartbeat tables every 100k — ALWAYS via `tools/table.py <cand> <keeper>
   <step>`** (the canonical 10-row table; hand-composed tables drift and drop
   rows). Report objectively — if the candidate is
   behind, say so with the gate arithmetic. Kill mid-run ONLY for a diagnosed
   defect with a causal fix verified at the pathology's own depth (the
   Mark V cycle pattern); never for one bad heartbeat.
6. **Gate rules.** REJECT = the verdict IS data (mk6's REJECT proved the
   aux-task law). PROMOTE = commit everything (kernel + tools + docs + skills)
   as ONE keeper commit, update `docs/roadmap.md` keeper record, update the
   memory ledger, and write the next mk's ledger (what the run's own
   forensics named).

## Watch signatures (from the record)

- Live median < cumulative median = decay onset (earliest warning).
- Death drip accelerating late-run = a head chasing an unbounded target;
  pull `deaths.jsonl` `explosionElem` → head+column immediately.
- Ceiling turnover after a peak = zenith law territory: check per-lane
  paramAbsMax / actionDelta trajectories of the TOP lanes over time
  (`fleet.py lanes` at successive steps), not population aggregates.
- Positivity slipping while ceiling climbs = bimodal split: lane forensics,
  then bisect; do not tune blindly.

## Structural laws (additions since the organ-bisect skill; cite, don't relearn)

- **The zenith law**: unbounded action means past the actuator clamp = zero
  behavioral gradient with unbounded weight growth → living-death explosions,
  ceiling decay, population plateau. Emission geometry must encode actuator
  bounds (tanh means + exact score gradient (a−m)(1−m²)).
- **Every bootstrapped/cumulant head gets per-head differential centering**
  (psiRho / optRho / vsRho). The one family that lacked it was the one that
  died (mk8's vStop/vsAux channel, high-value lanes).
- **Never freeze an organ** (Inv-7 and user doctrine): fix the target/scale
  geometry instead. Freezing measured −IQM and merely moved explosions.
- **REJECTed deletions are discoveries**: mk6 proved aux heads regularize the
  shared representation even with their consumers dead.
