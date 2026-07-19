# Research dossier — planning–learning coupling for a fully online average-reward agent (2026-07-19)

Scout synthesis for rebuilding OaK's ablated planning organ. All IDs verified 2026-07-19. Verdict up front: the literature strongly supports the measured law — model backups must never write into online value weights or ρ̄ — and offers exactly three stable coupling channels: **selection/gating, planner-owned value tables, and search control**. The rebuild uses all three and nothing else.

## 1. Why model→target coupling fails (the measured poison, named)

- **Hallucinated Value Hypothesis (HVH)** — 2006.04363 (JAIR 2024; local): updating values of real states toward values of *simulated* states corrupts the control policy even under small model errors. Their fix (predecessor-model multistep updates that never bootstrap real-state values from hallucinated states) maps onto the trajectory-relay backup below. OaK's planTd→valueW was textbook HVH.
- **Delusional bias** — Lu-Schuurmans-Boutilier (NeurIPS 2018): Q-updates under FA can rest on mutually inconsistent values; off-distribution backups (planning) amplify it. Linear heads reduce, not eliminate.
- **Model exploitation / compounding error** — MBPO (1906.08253): improvement bound degrades with model error ⇒ short branched rollouts only. M2AC (2010.04893): provable value-bound improvement if the model is used ONLY where confident ⇒ mask, don't blend.
- **ρ̄ contamination** — Wan-Naik-Sutton (2006.16318): average-reward estimate updates from the TD error of the REAL stream; Reward Centering (2405.09999, local): center from observed rewards only. Feeding planner TDs into ρ̄ makes the center a functional of the model's own errors — a self-referential estimator. The 13 measured runaways to +539 are the predicted signature.
- Stability without replay/target nets: linear Dyna + prioritized sweeping (Sutton-Szepesvári-Geramifard-Bowling UAI 2008); stream-x (2410.14606) — target nets unnecessary at OaK's scale PROVIDED planner state is memory-separated from learner state.

## 2. Option-model planning theory + the centering hazard

- SPS 1999 (local): intra-option Bellman equations; option model = (reward-to-go, termination-state values).
- **Wan-Naik-Sutton 2110.13855 (NeurIPS 2022; local) — the controlling source.** Differential option decomposition: `q(s,o) = E[Σ(R−ρ̄) + v(S_τ)] = r̂_diff(s,o) + v̂_stop(s,o)`, with convergent intra-option off-policy learners and sample-based planning variants — exactly OaK's head shapes.
- RR-Subtasks (2202.03466, local) eq. 19: `v(s) ← max_o [r̂(s,o) + ṽ(stop)]` — sound as an operator on PLANNER-OWNED values; HVH as a learner target.
- **⚠️ Centering-consistency hazard:** with per-head centering (optRho_k), `r̂_k ≈ E[Σ(R − optRho_k)]`. Cross-option comparison of `q = r̂ + v(stop)` is corrupted by the per-head offsets. The exact de-bias uses the duration head:
  `q(s,k) = r̂_k(s) + (optRho_k − ρ̄)·l̂_k(s) + v̂_stop(s,k)`
  giving the currently diagnostic-only l̂ a REQUIRED planning role. (Alternative: learn reward models against the shared ρ̄, Wan et al.'s setup.) Uncorrected per-head centering inside an argmax is a silent poison vector — same family as the measured one. **This de-bias is mandatory in PLAN-SELECT.**

## 3. Prioritized sweeping & search control under an O(K) budget

- Moore-Atkeson 1993 (priority = |TD-error|); Peng-Williams 1993 (enqueue predecessors); Sutton et al. 2008 (linear-FA compatibility). Budget rule: fixed P backups/step, P structurally derived (warps idle during the env window), priorities from planner-owned TD, staleness-decayed.
- Meta-Gradient Search Control (2406.19561, local): meta-learns per-state planning-query probability online; avoids sampling-inaccurate-transitions and credit-assignment stalls. Constraint: the meta-objective must be improvement on REAL-experience TD, never model TD.
- **Holland et al. 2018 (1806.01825) — decision-relevant:** one-step Dyna gave ~no benefit over more model-free updates EVEN WITH A PERFECT MODEL; benefit came from planning SHAPE (fewer, longer credit chains). Many tiny 1-step backups at the live state is the weakest possible planning organ; the value is in propagating credit along extended chains (ring relay) and in selection. Corollary: HalfCheetah is a planning-null benchmark — sparse-reward envs are the informative test.

## 4. Twin/disagreement-gated trust (model USE, not targets)

- M2AC (2010.04893) — strongest fit: mask model use by ensemble uncertainty; robust improvements even with long rollouts.
- ME-TRPO (1802.10592) — canonical ensemble-disagreement regularization of model use.
- STEVE (1807.01675) — per-example uncertainty-weighted rollout depth.
- TD3 (1802.09477) — twin-min kills overestimation, but TD3 puts the min in the TARGET; OaK uses it in SELECTION scores only — importing the target-side use re-opens the poison channel.

## 5. Benchmarks for long-horizon planning credit (cheap, continuing-portable)

- **Continuing FourRooms** — the actual testbed of 2110.13855: sparse reward at designated states, continuing, trivial physics, directly exercises option-model planning. **First port (Step 3 of the execution plan).**
- **bsuite DeepSea** (1908.03568): N×N grid, O(1)/step, the standard sparse long-horizon credit/exploration probe; continuing port = chain of grids. **Second port.**
- MiniGrid KeyCorridor/DoorKey/MultiRoom — multi-stage sparse credit; continuing port re-generates layout on completion.
- Access-Control Queuing (SB §10.3.1; used in 2006.16318) — canonical average-reward sanity test for ρ̄ hygiene, not horizon.
- ALPS / Laplacian decision-time planning (2602.05031, ICML 2026; local) — subgoal decomposition explicitly to mitigate compounding error over long horizons; validates planning over temporally abstract options, not primitive chains.

## 6. The three approved architectures

Cost model: active width A=64 MACs per head eval; full-bank sweep of r̂, l̂, vStop, vStop2 = 4·K·A ≈ 65K MACs ≈ 0.13 MFLOP/step — noise against the physics chain.

### A. PLAN-SELECT (ships in L1; baseline arm for everything else)
Per step, at the live state only:
```
q̂(s,k)  = r̂_k(s) + (optRho_k − ρ̄)·l̂_k(s) + min(vStop_k(s), vStop2_k(s))
trust_k = 1[ |vStop_k − vStop2_k| ≤ ε ],  ε = bank-median disagreement (structural)
selected = argmax_{k:trust_k} q̂(s,k)   (fallback: RR continuation if none trusted)
```
Writes: option-selection/chord-weight signal ONLY. Nothing to valueW/ψ/ρ̄/policies. Cost: 1 bank sweep + K-argmax.
Formalizes RR eq. 19 as a pure ranking operator. **Requires optReward/optDuration to actually learn (L1 makes them live).**

### B. TRUST-SWEEP (L2; the credit-propagation organ)
Planner-owned value table U over ring entries (init U = v at ring-write; then sealed from the learner). P backups/step, Moore-Atkeson priorities, Peng-Williams predecessor enqueue:
```
(i)  model backup (RR eq.19):  U(x_i) ← max_{k trusted} q̂(x_i,k)      [masked by trust_k]
(ii) trajectory relay (HVH-safe, model-free):  U(x_i) ← R_{i+1} − ρ̄ + U(x_{i+1})   [always allowed]
```
Writes: U only. Consumers: plan–learned gap `d_i = U(x_i) − v(x_i)` as SEARCH-CONTROL priority feeding G&T/feature-replacement attention + diagnostic; selection via A where ring-matched. Optional later: meta-gradient query probabilities (2406.19561) trained on real-experience TD improvement only.
Cost: P × 0.13 MFLOP; P=8 → ~1 MFLOP/step. Only component with genuine horizon-extension evidence.

### C. KEY-COMPOSER (L6; chord-level planning)
Chord λ ∈ R^K from truncated top-m trusted plan scores (softmax over trusted q̂), replacing the heuristic blend; Option Keyboard lineage (2106.13105, 2505.00787, local). Planner ranks, keyboard composes, disagreement gates membership. Writes: chord vector only. Closes the ablation-indicted keyboard gap.

## 7. Ablation matrix (N=1024, Tier-M screen → Tier-L promotion, paired gate)

| Arm | Change | Question |
|---|---|---|
| T0 | keeper (planning writes nothing) | baseline |
| A | +PLAN-SELECT | does model-backed selection help? |
| B | +TRUST-SWEEP on A | does credit propagation help? |
| B−gate | B with twin masking off | is disagreement gating load-bearing? |
| B+SC | B + meta-gradient search control | learned search control vs \|δ\| priorities? |
| C | B + KEY-COMPOSER | does chord-level planning close the keyboard gap? |
| P0 (Tier-S only) | planTd→ρ̄ re-enabled | positive control: gate must REJECT instantly; documents detector sensitivity |
| Port | A/B/C on FourRooms + DeepSea slices | the informative long-horizon test (cheetah null expected, Holland 2018) |

Blocking criteria unchanged (divergence Δp, IQM non-inferiority, RMST, NF prevalence) + ignition/sustain checks — selection changes historically kill the strong mode.

## 8. Poison-risk flags (explicit)

1. NEVER write U, q̂, or planner TD into valueW/ψ/ρ̄ — HVH + delusional bias + internal measurement. ρ̄ stays real-stream-only (TD-error form, Wan et al. 2021).
2. Twin-min and trust masks are SELECTION-side only; never copy TD3's target-side min.
3. The (optRho_k − ρ̄)·l̂_k de-bias is mandatory for any cross-option comparison.
4. Planner scores must not become the actor's TD baseline without a separate gated experiment — that is target-adjacent.
5. Meta-gradient search control must meta-learn on real-experience improvement, never model TD.
6. Beware optimism loops in selection itself: argmax over trusted scores with disagreement masking is the bounded version; unmasked argmax re-imports model exploitation through the policy (ME-TRPO/M2AC).

Sources (verified 2026-07-19): 2202.03466 · 2110.13855 · 2006.16318 · 2405.09999 · 2006.04363 · 1806.01825 · 2406.19561 · 2010.04893 · 1807.01675 · 1906.08253 · 1802.10592 · 1802.09477 · 2410.14606 · 2602.05031 · 1908.03568 · 2106.13105 · Lu-Schuurmans-Boutilier NeurIPS 2018 · SPS 1999 · Moore-Atkeson 1993 · Peng-Williams 1993 · Sutton-Szepesvári-Geramifard-Bowling UAI 2008. Local PDFs for the majority under docs/papers/.
