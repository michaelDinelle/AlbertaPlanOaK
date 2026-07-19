# L1 planning rebuild — concrete patch map (2026-07-19, post-slimming kernel 2549 lines)

Architecture rationale: `docs/research/planning-coupling-2026.md`. This file is the
execution spec. Line refs are post-slimming, pre-env-contract (agent-14 shifts them;
anchors are semantic — grep the quoted code).

## Key structural insight (shrinks the candidate)

The keeper (mk8, `OAK_ABLATE=planning`) already nullifies every poison channel:
- `planDelta = 0` under `OAK_ABLATE_PLANNING` → the ρ̄ feed (:798-801) and the
  valueW planTd terms (:1141-1145) are no-ops in the keeper;
- the dyna apply (:1149-1157) is gated `!OAK_ABLATE_PLANNING` → off in the keeper.

So hard-deleting those channels is **behavior-identical to the keeper**, and the
candidate's ONLY functional deltas vs keeper are:
1. optReward/optDuration heads actually learn (they never have — no apply exists);
2. the selection score gains the centering de-bias + twin-min + twin trust gate;
3. planTd (still computed, now nonzero) feeds ring priority + G&T/search signals
   (selection-side only — legal under Mark IV-VIII law (a)).

## Patch A — make optReward/optDuration live (OptionModels)

- Revive the reserved `durationTd[K]` arena region as `optDurationTd[K]`; store
  `dTd` in `oak_plan_one` where `rTd` is stored (:1004 `ctx.optRewardTd[k] = rTd;`
  add `ctx.optDurationTd[k] = dTd;`).
- In `oak_learn_rows` k-loop (the `!OAK_ABLATE_OPTION_MODELS` block, :1109-1115),
  add the two applies over prev-active rows, semi-gradient differential TD with
  the bootstrap-norm scale (the death-channel fix documented at :974-983):
  ```
  ctx.optReward[i*K+k]   = safe_w(w, w + aM * rho * ctx.optRewardTd[k]   * xp * vsc);
  ctx.optDuration[i*K+k] = safe_w(w, w + aM * rho * ctx.optDurationTd[k] * xp * vsc);
  ```
  (dTd is already structurally 0 for k==0 at :997; `vsc = ctx.vsScale[k]` is
  already loaded at :1088.)
- Keep `optRewardTd` feeding the alphaModel MetaOptimize signal as today.

## Patch B — de-biased, twin-min, twin-gated selection score (Planning/OptionKeyboard)

In the encode model-dot pass (:642-677):
- Add the twin dot: `vModel2 = fma(ctx.vsAux[idx], tgCurr[a], vModel2)` alongside
  the vStop dot; cache per-k disagreement `|vModel − vModel2|` into a threadgroup
  array (K=32).
- De-biased score (the (optRho_k − ρ̄)·l̂_k term is MANDATORY — per-head-centered
  r̂ heads are not cross-comparable without it):
  ```
  float q = rHat + (ctx.optRho[tid] - ctx.scalars->rho) * dur + min(vModel, vModel2);
  ```
  Keep the tid==0 main-task floor (`vModel = vCurr`; no de-bias for the
  never-stopping key).
- Twin trust gate (M2AC-style mask): ε = bank-median disagreement (thread 0
  insertion-sorts 32 floats after the barrier — cheap, exact); fold
  `disagreement <= eps` into `queryTrusted` (:672-674).
- Update the comments at :666-669 (score form changed) and :658-660.

## Patch C — hard-delete the poison writes (no-ops vs keeper)

- :798-801: delete the ρ̄ planDelta feed. Rewrite the comment block at :770-801:
  the differential-planning ρ̄ coupling existed only to counteract the value
  writes; both are gone; ρ̄ is real-stream-only (Wan-Naik-Sutton 2006.16318).
- :1141-1145: valueW apply loses the planTd terms:
  `delta = (e < A) ? ctx.td[0]*prev[i]*ctx.valueScale[0] : 0` — and the e≥A half
  becomes a skip (ownership loop can shrink to e<A).
- :1148-1157: delete the dyna apply into valueW. `planScr[0]` is safe — it is the
  real reward written at :1258, not dyna-derived. The `oak_dyna_backups` pass
  itself stays for now (it feeds ring priorities/search signals; TRUST-SWEEP L2
  redesigns it planner-owned) — but its dynaAcc accumulator writes become dead;
  leave the pass, delete only the valueW apply.
- planTd survives untouched as diagnostic/search signal (ring priority :1253,
  G&T signals :1581-1619, stats :437). planScale becomes dead in learn_rows —
  leave the buffer (reserved), stop using it there.

## Verification protocol (learning-dynamics change — keeper-cycle law)

1. Metal + swift rebuild. `tools/check.sh l1-planning-rebuild 100000` (ablate=none).
   Expect: no death wave (keeper at 100k had 2 pre-existing explosion deaths),
   median ≥ ablate=none baseline (`slimming` check P50 was −0.368; the poison
   deletion should materially beat that — the 2026-07-18 planning-live-only
   screen at Tier-M was IQM +0.126).
2. ONE Tier-L via `tools/run_fleet.sh 1000000 l1-planning-rebuild-tierL-YYYYMMDD
   .oak-benchmarks/mk8-tierL-20260705` — candidate runs ablate=**none** vs keeper's
   ablate=planning (table.py exempts ablation as treatment variable).
   Pre-registered kill rules: live median < ½ keeper at any 100k milestone;
   positivity < 90%; death wave > keeper. Canonical `tools/table.py` table pasted
   at every 100k.
3. Gate: all 8 criteria (divergence Δp, IQM, RMST, NF phases + blocking
   igniters/sustainers/bestLaneClass/positiveFrac).
4. Fallback if REJECTed twice: delete optReward/optDuration outright (bloat) and
   ship Patch C alone (behavior-identical to keeper — pure hygiene) as the
   candidate, planning stays ablated pending L2.
