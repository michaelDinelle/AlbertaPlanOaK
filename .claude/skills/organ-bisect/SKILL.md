---
name: organ-bisect
description: The OaK causal-forensics loop that found the planning anti-learner in one day — use for ANY population-level pathology (bleed, deaths, median erosion, SPS regression) instead of theorizing fixes. Invoke before proposing any mechanism change.
---

# Organ bisect: convict before you fix

The 2026-07-04 lesson: eight theorized fixes each moved a symptom; ONE ablation
bisect found the disease (`OAK_ABLATE=planning` → 24.6% → 97.8% positivity).
Estimated cost of theorizing first: ~6 hours. Cost of bisecting first: 10 min.

## The loop (in order — do not skip to step 4)

1. **Trajectory first** (free): read `aggregate.jsonl` heartbeats. Classify the
   pathology by its curve: exponential decay = uniform pressure on learned
   behavior (suspect a biased stream); accelerating erosion = compounding loop
   (suspect a feedback coupling); step change = a maturing trigger (find what
   crosses a threshold at that step count).
2. **Lane forensics** (1 min): split lanes into crashed/near-zero/ignited from
   `fleet.jsonl` at a heartbeat; diff their health medians. The discriminating
   field names the subsystem (`actionDeltaL2Mean` → behavior churn;
   `kernelRho` → centering; `paramAbsMax` → weight loop; `oakFeatureReinit` →
   curation; entropy/ESS → ratios).
3. **Death certificates** (free): `deaths.jsonl` `explosionElem` → param-layout
   head + column (k). A single (head, k) cluster names the exact organ.
4. **ORGAN BISECT** (5 min per organ, decisive): rerun the SAME horizon with
   `OAK_ABLATE=<organ>` for each suspect. The pathology vanishing under one
   ablation = conviction; surviving all = the residual (keyboard/actor
   dynamics). Compare full curves, not endpoints.
5. **Fix at the source, verify at the pathology's own depth**: a 30k check
   cannot clear a 30k→100k bleed — always verify at the depth where the
   pathology lived (100k check = 5 min at B=1024, `OAK_STEPS_PER_CB=1024`).
6. Only then launch Tier-L.

## Standing laws (violations found by this loop — do not relearn them)

- Model estimates may GATE or SELECT; they must never be written into
  `valueW` or ρ̄ (the A4/planTd law, convicted twice). Optimism entering ρ̄
  turns a value offset into a reward-stream poison that makes the agent
  dismantle its own success.
- Suppressing a backup target (÷dur, −modelErr) kills the ceiling; gating its
  magnitude by confidence is fine (gate ≠ shift).
- Every cumulant/advantage consumer needs per-head differential centering
  (psiRho pattern) — including the actor signal.
- Exploration noise scales: dense weight-space perturbations emit √A hotter
  than one-hot; scale by basis/√A.
- The consensus/main key anchors every chord with unit weight (burn-loop fix);
  θ₀'s own gradient needs its meta-learner (actor IDBD), not a fixed α.
- Never-terminating GVFs (β=0 heads) must not have their own off-policy
  learner — alias them to the on-policy critic (frozen psi₀ law).
- Capacity-derived bounds (α∈[1/R²,1], ρ≤K) are structural, not knobs; their
  removal was measured as deaths both times.
- THE ZENITH LAW: bounded actuators need bounded emission geometry (tanh means
  + (a−m)(1−m²) score gradient) — unbounded means past the clamp = living-death
  explosions + ceiling decay + plateau. Diagnose via TOP-lane trajectories over
  time (paramAbsMax/actionDelta), not population aggregates.
- Aux heads regularize the representation even when their consumers are dead
  (mk6 REJECT); delete consumers, keep learners.
- CENTERING LAW, refined (mk9): per-head integrator centering (psiRho/optRho)
  applies ONLY to reward-cumulant heads. A ZERO-cumulant head (stop-value
  GVFs) must never take an integrator — it becomes a marginally-stable double
  integrator (2 full-system detonations by 300k, ρ̄ at −1e37). Its unbounded
  part is the PAYOFF: gauge it (predict v(stop) − valueGauge, add the gauge
  back at consumers).
- Ablation matrix = sequential `OAK_ABLATE=<organ,list> ./fleet.py run
  <steps> ablate-<organ>-<depth>k --role probe` runs (one at a time, probe so
  ROLES.json candidate is untouched); compare `fleet.py report` across runs.
  Never create bisect.py at the repo root — it shadows stdlib `bisect` and hijacks
  argparse repo-wide.

## Instruments

- `OAK_ABLATE=<organ,list>` — the bisect scalpel (function-constant compiled).
- 100k fixed-CB check: `OAK_BENCH_STEPS=100000 OAK_BENCH_SEED_COUNT=1024
  OAK_STEPS_PER_CB=1024` into `.oak-benchmarks/<name>` — ALWAYS visible there.
- Stage masks `OAK_MEGA_STAGE_MASK` for SPS attribution (successive diffs).
- `OAK_PROBE_FTZ` (denormal SPS decay), `OAK_DEBUG_RAW` (weight-level lane
  dumps) for what heartbeats can't resolve.
- One run at a time (workspace lock); every check is a run the user can watch.
