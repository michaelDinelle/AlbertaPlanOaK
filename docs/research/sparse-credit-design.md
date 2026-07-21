# Sparse-Credit Design: Bounded Eligibility Trace + Count-Gated Exploration

Status: design proposal, implementation-ready. No code changed (read-only analysis of
`Sources/OakKernel.metal` @ mk8-era, 2854 LOC).
Scope: the two binding sparse-reward failure modes on the `-cont-v0` integer/classic-control
lanes — (i) 1-step TD cannot propagate a rare +1 over long horizons, (ii) near-uniform
Gumbel exploration never strings together the directed action runs MountainCar-class envs
require. Both mechanisms are selection- and learning-update changes only: **nothing enters
the reward stream** (Inv-8), **both run every step** (Inv-7), **every constant is
structurally derived** (Inv-9, inventory in §7), **no magnitude caps on weights** (safe_w
only), **arena-allocated, zero heap on the hot path** (Inv-6).

Attribution correction (verified 2026-07-20): the bounded-trace source is **SwiftTD —
Javed, Sharifnassab & Sutton, RLJ 2:840–863 (2024)** (local:
`docs/papers/SwiftTD_FastRobustTD_RLC2024.pdf`). It is *not* by "Kearney, Naphan, Sutton";
arXiv author searches for Naphan/Kearney return no RL eligibility-trace work. SwiftTD has
**no arXiv ID**; canonical PDF: `incompleteideas.net/papers/SwiftTD-RLC.pdf`. The control
extension is **Swift-Sarsa — Javed & Sutton, arXiv:2507.19539** (local).

Why this is the retry the roadmap archive asks for. The archive
(`docs/roadmap.md`, rejected-mechanisms table) says verbatim: "Retry traces only with a
**materially different true-online/off-policy trace formulation and trace-stability
evidence**." All three prior attempts were *unbounded accumulating* traces bolted onto the
1-step path:

- `l3-l7-value-psi-trace-*`: accumulating RR ψ traces improved 10k, then a 50k screen blew
  up (score-gradient 7.6e10, duration 68k). Lesson recorded: ψ traces "smear termination
  credit into the keyboard and starve search control."
- `l3-l7-value-trace-*` / `l3-l7-tdtrace-planstep-*`: value-only accumulating traces →
  score-gradient 1e70, "critic traces cannot be bolted onto the current value/planning path
  without a stronger gradient-TD/actor-critic redesign."
- Mark-III-era commit `6d532226a`: traces stripped as "the canonical Baird-style divergence
  config"; prior failure attributed to correlated Fourier features. The Mark IV
  representation (geometry-gated sparse tile codes, 32 banks × 7 tiles) is exactly the
  decorrelated-feature substrate traces want — the blocking precondition has changed.

SwiftTD's overshoot bound + True Online form + step-size decay is the materially different
formulation; this document also supplies the demanded stability analysis (§1.4, §5).

---

## 1. Mechanism 1 — Bounded dutch trace on the value head (SwiftTD form, average-reward)

### 1.1 Trace family choice: dutch (True Online), not accumulating, not replacing

- **Accumulating** z ← λz + ϕ: the rejected design. Norm grows to |ϕ|/(1−λ); with
  bootstrapping + linear FA this is Baird's divergence configuration. Forbidden by evidence.
- **Replacing** z_i ← 1 on active features: bounded by construction (z ∈ {0,1}∪decayed),
  but breaks the exact λ-return equivalence and interacts poorly with per-feature step
  sizes (the increment must carry α_i for the bound to work — see 1.4).
- **Dutch / True Online** (van Seijen, Mahmood, Pilarski, Machado & Sutton, *True Online
  Temporal-Difference Learning*, JMLR 17:1–40, 2016, **arXiv:1512.04087**): chosen. It is
  exactly equivalent to the online λ-return algorithm (no small-α approximation), which is
  what makes the overshoot bound *applicable at all* (SwiftTD §5.2: the bound needs a
  learner without delayed-target error; TD(λ)+bound diverged on their Atari grid while
  True Online TD(λ)+bound never did — their Fig. 4. The True Online form is load-bearing,
  not cosmetic).

Average-reward form (γ = 1; Tsitsiklis & Van Roy, *Average Cost Temporal-Difference
Learning*, Automatica 35(11):1799–1808, 1999 — no arXiv; the canonical average-cost TD(λ)
analysis. Its convergence conditions — λ < 1 strictly, finite-state irreducible chain,
full-rank features, Robbins–Monro step sizes — do not literally hold under OaK's constant
tracking rates and drifting standardized features; §1.4 item 5 carries that residual risk):

```
δ_t   = (r_t − ρ̄_t) + v(x_t) − v(x_{t−1})                    (differential TD error; OaK: with gauge g, ṽ = v − g)
z_t   = λ z_{t−1} + s_t·α⊙ϕ_{t−1} − s_t·(λ z_{t−1})ᵀϕ_{t−1}·(α⊙ϕ_{t−1})   (dutch trace, increment-scaled)
w_t   = w_{t−1} + δ_t z_t − (increment)_t · vδ_t              (True Online δ-correction)
```

with s_t the overshoot-bound scalar of 1.2 and vδ_t the carried correction accumulator
(§1.3). OaK's value head is linear, so ∇_w v = ϕ and the dutch trace applies verbatim.

### 1.2 SwiftTD's mechanism in full (Javed–Sharifnassab–Sutton 2024, Alg. 1, §5, App. B)

SwiftTD = True Online TD(λ) + three ideas. OaK keeps (2) and (3); (1) is **replaced by
OaK's existing MetaOptimize** per-feature step sizes (`metaopt_update_slot`,
`Sources/OakKernel.metal:369-399`) rather than duplicated:

1. *(theirs → replaced)* Per-feature log-step-sizes β_i adapted by IDBD-style meta-gradient.
   OaK already owns this with `alpha[i] = exp(beta[i])`, β clamped to [2·ln(decay), 0] ⇒
   α ∈ (0,1]. One meta-controller per step size; two would fight.
2. **Overshoot bound.** Correction ratio of an update = fraction of the current prediction
   error removed: for per-feature linear SGD, τ_t = Σ_i α_i ϕ_i(x)² (their eq. 7). τ ≤ 1 ⇔
   the post-update prediction moved toward but *past-never* the target (their App. B.1,
   cases 1–2). Bound enforcement for TD must happen **when incrementing the eligibility
   vector, not when updating weights** (their §5.2): scale the increment by
   s_t = min(1, η/τ_t). Generalized trigger level η (their §6): default η=0.1 is a tuned
   knob; the parameter-free form is **η = 1** (pure never-overshoot, Mahmood et al. 2012's
   original bound — `docs/papers/03-mahmood-sutton-autostep-ICML-2012.pdf`).
3. **Step-size decay on trigger.** When τ_t > η: β_i ← β_i + ϕ_i²·ln ε (their eq. 8;
   geometric decay proportional to each feature's contribution to the violation). Their
   ε=0.99 is a knob; the structural form is **ε = 1 − 1/R** (ln ε ≈ −1/R: an R-step decay
   time constant, the same horizon as everything else — §7).

### 1.3 Exact update, transcribed onto OaK state

Existing 1-step update (to be replaced for `valueW` only): δ at `OakKernel.metal:501-503`,
ρ̄ at `:504-506`, weights at `:1139-1150` as
`valueW[i] += alpha[i]·td0·prev[i]·valueScale[0]` with `valueScale = 1/(1+Σα·x²)` — an
NLMS soft bound that covers only the *current* feature vector; it does nothing about the
trace-accumulated component. τ ≡ denomTd − 1 is already computed every step
(`fcstomp_store_learn_scales`, `:440-461`). New state (arenas, per lane):

| Arena | Size | Role |
|---|---|---|
| `traceZ` | R floats | dutch trace for `valueW` |
| `traceZD` | R floats | z_δ stash (this step's scaled increment, for the vδ correction) |
| `traceVD` | 1 float | vδ carry (Σ_i δw_i·ϕ_i of the previous step) |

Per step (encode computes δ; learn stage applies, replacing the walk at `:1139-1150`).
**The δw payout must be dense over R, not restricted to the active set** — SwiftTD's loop 1
applies `δw_i = δ·z_i − z_δ[i]·v_δ` to every i with nonzero trace, which is precisely the
delayed-credit mechanism: a feature active 100 steps ago is inactive when the reward
arrives, and its decaying z entry is exactly the credit it is owed. (`x` is only valid on
the active set — OaK writes only the 64 active slots of the state buffer — so anything
needing x is membership-gated via the existing `oak_in_prev_active`, `:353`. z_δ is
zeroed on use and re-set only on increment (SwiftTD Alg. 1: `z_δ[i] = 0` after the payout
loop, `z_δ[i] ← min(1,η/τ)·e^{β_i}ϕ_i` in the increment loop), so the correction stash
self-cleans for features inactive more than one step.)

```
# given: δ (= ctx.td[0], already NF-guarded at :502), active-prev set A_p (actP),
#        x = prev features (valid on A_p only), α = ctx.alpha,
#        τ = denomTd − 1 (already computed, :440-461), λ = 1 − invR, η = 1, ε = 1 − invR
s  = min(1, η/τ)                                            # τ = 0 edge: s = 1
if τ > η:  for i ∈ A_p: β[i] ← clamp(β[i] + x_i²·ln ε, 2·ln decay, 0)   # decay = LAST write on β (safety wins over MetaOptimize)
# pass 1 (dense, i ∈ 0..R): pay out + consume correction stash
    δw_i       = δ·z_i − traceZD_i·traceVD
    valueW[i]  ← safe_w(valueW[i], valueW[i] + δw_i)
    if i ∈ A_p: vδ_next += δw_i·x_i                         # membership-gated; x valid only on A_p
    traceZD[i] ← 0
# pass 2 (dense): decay
    z[i] ← λ·z[i]
T  = zᵀx over A_p                                           # dot of the DECAYED trace with x (A-reduce)
# pass 3 (i ∈ A_p): bounded increment
    traceZD[i] ← s·α_i·x_i
    z[i]      += traceZD[i]·(1 − T)
traceVD ← vδ_next
```

Cost: 2 dense R-passes + 2 sparse A-passes + 2 reductions ≈ 2R + 2A FMA-class ops —
noise against the physics chain, and branch-light (GPU-idiomatic).

ρ̄ update (`:504-506`) is **unchanged** and stays 1-step — see §3.1.

### 1.4 Divergence bound and why it holds

1. **Per-update contraction.** s·min-form forces the implied λ-return update's correction
   ratio τ ≤ η = 1 for every feature vector, every step (SwiftTD App. B: with the
   semi-gradient assumption, a τ ≤ 1 update is non-worsening on the current sample and
   never overshoots the target). The deadly-triad feedback loop of the unbounded trace —
   big z → big Δw → big v → big δ → bigger Δw — is broken *at the update level*, not by a
   norm cap: each individual update is target-consistent regardless of trace magnitude.
2. **Exactness.** True Online TD(λ) ≡ online λ-return algorithm (van Seijen et al. 2016,
   Thm. 1), so the bound applies to exact λ-return targets rather than the TD(λ)
   approximation — this is the measured difference between "bound works" and "bound
   diverges" (SwiftTD Fig. 4: TD(λ)+bound diverged on 3+ Atari games; True Online TD(λ)
   +bound on none).
3. **Self-relaxation.** Trigger ⇒ β decays geometrically (ϕ_i²-weighted) ⇒ τ falls below η
   ⇒ bound releases. Step sizes cannot ratchet upward while the bound is active, because
   decay is the last write on β each step (SwiftTD Fig. 5: no divergence anywhere on the
   (α_init, θ) grid *with* decay; poor behavior without it).
4. **Empirical bound.** SwiftTD: no divergence on any Atari Prediction Benchmark game for
   *any* hyperparameter combination tested (their §6.1).
5. **Candid limits.** No full proof exists for constant-step-size, average-reward,
   online-standardized features (TVR-1999's convergence needs Σα=∞, Σα²<∞ and a fixed
   feature map; OaK tracks non-stationarity with constant rates by standing design). The
   residual risk is carried by instruments, not hope: `safe_w` on every write, the existing
   per-phase NF counter (`OAK_CNT_LEARN_NF`) the trace updates route through, the fleet
   death certificate (`paramAbsMax`, `paramExplosion` detector), plus one new stat
   `traceAbsMax` (running max |z_i|) exported to `KernelStats`/`fleet.jsonl` so a trace
   runaway is *dated before* it poisons weights. Worst-case trace norm is
   s·α·|ϕ|/(1−λ) ≤ 256·|ϕ| (α ≤ 1 from the β-clamp) — large only if features are large,
   and each weight move is still overshoot-bounded per (1).

### 1.5 Structural λ (Inv-9)

**λ = 1 − 1/R = 1 − invR** (= 255/256 at the live R=256). Effective trace horizon
1/(1−λ) = R steps — one structural memory horizon shared with the planning ring (R slots,
`OakKernel.metal:295`), the ρ̄ EMA rate (invR, `:504`), and the novelty window (§2). Not a
knob: derived from agent capacity exactly like α₀ = 1/cap and ridge = 1/(age+1). In-repo
precedent: the rolled-back pass-15 trace used λ = 1 − 1/cap (commit `a50eb6837`). TVR-1999
requires λ < 1 for average-cost TD(λ) — satisfied strictly. Per-option duration-derived
λ_k = 1 − 1/l̂_k was considered and rejected: it couples trace geometry to a learned,
non-stationary head (`optDuration` — itself a prior runaway) and buys nothing the
population gate has asked for.

---

## 2. Mechanism 2 — Count-gated exploration: recency-novelty hysteresis + LRU probe

Selection-only by construction: the mechanism changes **which option runs**, never any
reward, target, ρ̄, or weight. Precedent for counts acting at *selection* rather than as
reward bonus: UCT's in-tree rule argmax[Q̂ + c·√(ln N(parent)/N(child))] (Kocsis &
Szepesvári, ECML 2006 — counts drive selection, reward untouched); Go-Explore's cell
selection probability ∝ 1/(count+ε)ᵖ (Ecoffet et al., Nature 590:580, **arXiv:1901.10995** —
counts gate *where to explore from*, no bonus); Strehl–Littman MBIE and UCRL2 (Jaksch,
Ortner & Auer, JMLR 11:1563, 2010) keep optimism inside the planner's model, not the
stream (UCRL2 radii ∝ √(log(2SAt/δ)/(2 max{1,N(s,a)})) — the average-reward optimistic
canonical form). Reward-bonus families are **rejected by Inv-8** and cited only for the
signal shape: Bellemare pseudo-counts β(N̂+0.01)^(−1/2) (**arXiv:1606.01868**), Tang
SimHash counts (**arXiv:1611.04717**), NGU/episodic curiosity (arXiv:2002.06038 /
1810.02274 — we borrow the *episodic-memory scan*, not their intrinsic reward), LPM
(local `2509.25438`, intrinsic learning-progress reward — rejected).

### 2.1 The novelty signal: joint-signature recency over the existing ring

The Mark IV state already computes, per bank b ∈ 0..31, a stationary winning tile
`winner_b = 32 + 7b + tile_b`, `tile_b = clamp(⌊z_b + 3.5⌋, 0, 6)` of the bank's designated
core channel (`gate_tile`, `OakKernel.metal:348-351`), and **already stores all 32 winners
per ring slot** (`ringIdx`, written every step at `:1266-1267`; ring = last R=256 states).
The joint signature σ(x_t) = (winner_0,…,winner_31) is a zero-cost, execution-width-aligned
tile-code hash of state — no SimHash needed, and the gating is a stationary function of
state (Mark III law (b) compliant: geometry, never magnitude competition).

```
M(t)   = max { slot s ≠ slot_t : ringIdx[s] == σ(x_t) }     # most recent exact revisit; ∅ if none
ν(x_t) = min(1, (t − M(t)) / R)  ∈ (0, 1]                    # 1 ⇔ not seen within the window
```

Why joint-exact and windowed, not marginal counts: per-(bank,tile) marginal counts
(224 counters) saturate within hundreds of steps (7 tiles/bank) and go silent exactly
where the frontier signal is needed; the joint windowed scan (i) never saturates, (ii) is
the *episodic* form of visitation whose semantics are precisely "progress vs stagnation":
same position-bank tiles with *different* velocity-bank tiles (a building MountainCar
swing) is novel ⇒ commit; identical full signature (bumping a wall, parked at the bottom)
is a revisit ⇒ switch. Window = R is structural (ring capacity; also the trace horizon).
Failure mode of exact-match brittleness (noise makes every signature unique) is **silence,
not damage** — the gate never fires, behavior ≡ keeper — and is screened by the
`novelGateFire` stat (§4).

### 2.2 The selection rule (exact)

Current logic (`fcstomp_prediction_control_loop`, `OakKernel.metal:688-690` and the vacuum
probe `:748-758`): `continuing = (prevSel≠0) && tgCont[prevSel]>0`; on vacuum
(keyboard mag = 0) probe = argmax_k twinDis[k]·dur[k]. Modified:

```
trusted      = (gpiBestScore > −FLT_MAX)                       # any trust-gated model advantage (existing)
continuingRR = (prevSel ≠ 0) ∧ (tgCont[prevSel] > 0)           # existing RR continuation, untouched
continuing   = continuingRR ∧ (trusted ∨ ν(x_t) == 1)          # NEW: stagnant + no model direction ⇒ break commitment
selected     = continuing ? prevSel : (trusted ? gpiBest : probe)
probe        = argmax_{k ∈ E} twinDis[k]·dur[k],
               E = { k : selectAge[k] ≥ K };  if E = ∅, E = all K   # LRU systematic coverage (new uint arena selectAge[K])
selectAge[k] += 1 ∀k;  selectAge[selected] ← 0                 # mirrors the existing lastStopAge pattern
```

Semantics: **commit while frontier-ward, re-probe systematically when stagnant, trust
models whenever they speak.** Note the `selected = probe` branch: today an untrusted
non-continuing step sets `selected = 0` (argmax over all-(-FLT_MAX) scores collapses to
lane 0) while the vacuum probe only steers the keyboard chord — so `prevSel` is 0 next
step and RR continuation *cannot latch* in vacuum (commitment today is only the
determinism of the twinDis argmax re-picking itself). Aligning `selected` with the probe
makes continuation well-defined exactly where commitment is the scarce resource; the
novelty gate then decides whether the latch holds. Deliberation-hysteresis precedent: Harb, Bacon, Klissarov &
Precup, *When Waiting is not an Option* (AAAI 2018, **arXiv:1709.04571**) put a margin on
the switch decision itself; ours puts a *count-gate* on it. Go-Explore's return-then-explore
is functionally achieved without its archive/restore machinery (archives are episodic
semantics and per-step allocation — Inv-4/Inv-6 violations); full Go-Explore is rejected.

Strong-mode preservation (the standing law): on dense-reward lanes (cheetah) models are
trusted ⇒ gate bypassed; and even untrusted, exact 32-bank signature revisits in a
17-dim continuous stream are rare ⇒ ν ≈ 1 ⇒ commitment unchanged. The gate is silent where
the walking mode lives; it fires only in the sparse vacuum where the failure mode lives.

Not chosen (with reasons): per-step Gumbel logit tilt (reintroduces undirected noise at
the wrong timescale — the failure we are fixing; option commitment is the structural
answer); optimism added to the selection *score* (couples a count scale to a value scale —
an untunable constant by another name; gating is scale-free); marginal rsqrt counts as
primary (saturate; kept as a fallback only if the exact signature proves too brittle).

---

## 3. Interaction with 1-step TD and per-head centering (stability contract)

### 3.1 What the trace touches — and what it must not touch

- **ρ̄ (main), psiRho[k], optRho[k], and every centering scalar stay 1-step EMAs at rate
  invR.** A trace on a centering estimator would low-pass a low-pass and blur the
  fast-weights/slow-rate ordering the average-reward analyses assume (TVR-1999's coupled
  iteration; reward-centering 2405.09999's ρ̄-from-δ estimator). Mark IV law (c) — every
  bootstrapped head carries its own differential centering — is *preserved verbatim*; the
  trace changes only the geometric weighting of feature-direction credit inside `valueW`.
  The same δ_t feeds both the 1-step ρ̄ update and the trace-weighted valueW update — the
  exact coupling the average-reward TD literature analyzes.
- **ψ, optPolicy, optReward, optDuration, vStop/vStop2: no traces in v1.** These heads are
  importance-weighted/off-policy (IS ratio clamped at K); an IS-weighted trace multiplies
  variance by ~1/(1−λ)² — the precise configuration in which optReward's bootstrap, twin
  vStop2, and optDuration ran away before being rate-limited, and the configuration the
  roadmap archive convicted ("ψ traces smear termination credit"). Their current implicit
  normalization (`headScale = 1/(1+ρ·implicitTrace)`, the Implicit-Updates pattern,
  arXiv:2510.06149) stays untouched. Extension criterion (future): per-head trace only
  behind per-head stability evidence of the §1.4 kind.
- **Planner seal intact.** planU/planGap/planTd remain planner-owned; planTd already feeds
  neither valueW nor ρ̄ (Mark IV law (a)). A trace-improved v(x) changes planGap
  (U − v) as a *diagnostic* consequence only — there is no write path back.
- **MetaOptimize co-ownership of β.** Ordering within the step: MetaOptimize update first
  (as today), overshoot-trigger decay second (safety last-write). This mirrors SwiftTD's
  own architecture (meta-gradient raises β, bound-decay lowers it); the clamp
  [2·ln decay, 0] bounds the fight — worst case is a temporarily small α (a performance
  risk, never an explosion risk).

### 3.2 What the novelty gate touches

Only two selection sites (`continuing`, the probe candidate set). No write path to any
weight, reward, target, or centering scalar. The RR stopping comparison itself
(`tgCont`, `:584-592`) is untouched — the gate acts on the *consumer* of the comparison,
which is exactly the "exploration may influence SELECTION" allowance.

---

## 4. Integration sketch (named state/sites; no code written)

New per-lane arenas (all pre-allocated; sizes from existing capacity constants):

| Arena | Type/size | Init | Owner stage |
|---|---|---|---|
| `traceZ` | float × R | 0 | learn (replaces the `valueW` walk at `OakKernel.metal:1139-1150`) |
| `traceZD` | float × R | 0 | learn (z_δ stash) |
| `traceVD` | float × 1 | 0 | learn (vδ carry) |
| `selectAge` | uint × K | saturating K (all eligible at t=0) | prediction/control loop |
| `novelGateFire`, `traceAbsMax`, `nuMean` | KernelStats scalars | 0 | exported to `fleet.jsonl` (wrapper metrics only — metric collection, not agent state) |

- **Trace**: compute in the learn stage where `valueW` is updated today — 2 dense R-passes
  (payout + decay) and 2 sparse A-passes (vδ accumulation, bounded increment) per §1.3.
  Inputs already on device: `ctx.td[0]`, `ctx.alpha`, `prev`/actP, and τ = denomTd − 1
  from `fcstomp_store_learn_scales` (`:440-461`). Cost ≈ 2R + 2A FMA + two O(A)
  reductions per step — noise against the physics chain.
- **Novelty scan**: compare σ(x_t) against the R ring slots (R×32 uint compares,
  early-exit on match), excluding `slot_t = stepCount % R` (must not match the slot about
  to be overwritten by `oak_replace_commit`, `:1261-1268`). Ride the **idle warps of the
  env window** — the same execution pattern prioritized sweeping already uses
  (`:2535-2590`) — so the scan is wall-free. Result ν staged to a scalar for next step's
  selection (same staging pattern as `impStage`/`creditStage`).
- **Selection edits**: `continuing` at `:689`; probe candidate set at `:748-758`;
  `selectAge` updates beside `lastStopAge` aging at `:1349-1352`.
- **Inv-10 classification**: learning-behavior change (not bit-identical) ⇒ ships only via
  Tier-M screen + Tier-L `fleet.py gate` PROMOTE vs the keeper
  (`.oak-benchmarks/mk8-tierL-20260705`), same env schedule both sides (`tools/table.py`
  hard-refuses schedule mismatches).

## 5. Explosion-risk analysis (per mechanism, with the specific guard)

| # | Risk | Mechanism | Guard |
|---|---|---|---|
| 1 | Trace-norm growth (worst case 256·\|ϕ\|) feeding δ→w feedback | trace | Overshoot bound at increment (τ ≤ 1 every step) + β-decay on trigger + `traceAbsMax` cert + existing `OAK_CNT_LEARN_NF`/`paramExplosion` detectors + `safe_w` on every write. The bound breaks the loop at update level; decay relaxes it. |
| 2 | True Online correction term (vδ) amplifying a non-finite δ | trace | δ is NF-guarded to 0 before use (`:502`); vδ is a sum of guarded products; `safe_w` preserves on non-finite. |
| 3 | MetaOptimize vs β-decay oscillation | trace | Bounded by β clamp [2·ln decay, 0]; decay is last write (safety wins within-step); worst outcome is small α, never large. |
| 4 | IS-weighted heads (ψ/optReward/vStop2/optDuration) retracing their known runaways | trace | **Excluded from v1 by design** — their implicit headScale rate-limiting untouched. This was the exact failure of `l3-l7-value-psi-trace-*`. |
| 5 | Commitment thrash (gate fires every step → no action stringing) | novelty | Fires only when untrusted **and** exact-revisit; LRU window K bounds cycling to systematic coverage; when models speak the gate is bypassed (cheetah strong mode untouched by construction). |
| 6 | Permanent lock-in (ν stuck at 1 in noisy continuous lanes) | novelty | Then gate ≡ keeper (silence, not damage); screened by `novelGateFire`/`nuMean` per-env slice before any Tier-L. |
| 7 | Stagnation misread (ν < 1 during a working oscillation) | novelty | Joint signature: growing-amplitude swings differ in velocity-bank tiles ⇒ novel ⇒ commit. Only *identical* full configurations read as stagnation. |
| 8 | Signature brittleness (standardization drift re-labels tiles) | novelty | Drift *helps* recency semantics (window self-refreshes); the scan is windowed so stale matches age out in R steps. |

Top risk: **#1's residual form — no full convergence proof for constant-step
average-reward True Online TD(λ) under drifting standardized features.** Mitigation is the
SwiftTD bound+decay (update-level contraction, empirically divergence-free across their
entire Atari×hyperparameter grid) plus the full existing fleet death-certificate stack
plus `traceAbsMax`. If Tier-S shows any `paramExplosion`/`kernelRho` certificate with
`traceAbsMax` rising before it, kill the trace organ and keep the novelty gate (they are
independently ablatable, §6).

## 6. Ablation gating and experiment protocol

New organs in the `OAK_ABLATE` vocabulary (hyphenated, matching the existing style):
**`bounded-trace`** and **`novelty-gate`**. Canonical runs stay `OAK_ABLATE=none` (both
organs ON); diagnostics use the specialized variants.

1. Tier-S (1024×10k, ~39 s) smoke on the default mixed schedule — hard-fail screen
   (NF, explosion), per-env `novelGateFire` sanity.
2. `tools/ablate.py` organ matrix at Tier-M (1024×100k): keeper baseline, +trace,
   +novelty, +both, and +both-minus-each — conviction table per organ, per env slice.
   Primary readout per `fleet.py report` slices: sparse-lane IQM and %positive
   (MountainCar/DeepSea/KeyCorridor lanes), igniters/sustainers intact on cheetah lanes,
   divergence Δp.
3. Kill rules pre-registered: any sparse-env death wave vs keeper; cheetah-lane zenith or
   sustain damage (Mark III law (c) blocking criteria); `traceAbsMax` monotone climb over
   consecutive heartbeats.
4. Tier-L promotion only via `tools/run_fleet.sh` + `fleet.py gate` vs
   `.oak-benchmarks/mk8-tierL-20260705`, identical `OAK_BENCH_ENVS` schedule, canonical
   `tools/table.py` heartbeat tables at every 100k.
5. Qualitative check: `OAK_RECORD_SEEDS` + `tools/replay_html.py` on MountainCar lanes
   (velocity-amplitude growth visible in the pose streams).

## 7. Inv-9 parameter inventory (every constant is structural)

| Constant | Value | Derivation |
|---|---|---|
| λ (trace decay) | 1 − 1/R | capacity R; horizon = R = ring window = ρ̄ EMA horizon (precedent: commit `a50eb6837`) |
| η (overshoot trigger) | 1 | never-overshoot bound (Mahmood 2012); SwiftTD's 0.1 is a tuned speed knob, rejected |
| ε (β-decay rate) | 1 − 1/R | R-step decay time constant, same horizon |
| Novelty window | R | ring capacity (existing) |
| LRU window | K | option count (existing) |
| α per feature | existing MetaOptimize | unchanged |
| trace/novelty arenas | R, R, 1, K | capacity-derived |

Free choices remaining (none are runtime knobs): dense-vs-sparse trace sweep (dense —
branch-free, GPU-idiomatic); min-vs-EMA for any future soft novelty threshold (binary
window rule needs no threshold at all — chosen). If the exact-signature gate screens
silent at Tier-M, the structural escalation is coarsening the signature (drop banks gated
on near-constant channels), not adding a threshold constant.

## 8. Citations

- Javed, Sharifnassab, Sutton — *SwiftTD: A Fast and Robust Algorithm for Temporal
  Difference Learning*, RLJ 2:840–863 (RLC 2024). Local `docs/papers/SwiftTD_FastRobustTD_RLC2024.pdf`. No arXiv version.
- Javed, Sutton — *Swift-Sarsa: Fast and Robust Linear Control*, arXiv:2507.19539 (RLDM 2025 ext. abs.). Local. (Per-action SwiftTD heads; only the chosen action's trace increments — the control-side precedent.)
- van Seijen, Mahmood, Pilarski, Machado, Sutton — *True Online Temporal-Difference Learning*, JMLR 17:1–40 (2016), arXiv:1512.04087.
- Tsitsiklis, Van Roy — *Average Cost Temporal-Difference Learning*, Automatica 35(11):1799–1808 (1999). No arXiv.
- Mahmood, Sutton, Degris, Pilarski — *Tuning-free step-size adaptation* (AutoStep), ICML 2012. Local `docs/papers/03-mahmood-sutton-autostep-ICML-2012.pdf`.
- Wan, Naik, Sutton — *Learning and Planning in Average-Reward MDPs*, ICML 2021, arXiv:2006.16318 (1-step only; traces explicitly future work — this design fills that gap).
- Kocsis, Szepesvári — *Bandit based Monte-Carlo Planning* (UCT), ECML 2006 (counts at selection).
- Ecoffet, Huizinga, Lehman, Stanley, Clune — *First return, then explore* (Go-Explore), Nature 590:580–586 (2021), arXiv:1901.10995 / 2004.12919 (count-gated cell selection; archive/restore rejected here).
- Strehl, Littman — *An analysis of model-based Interval Estimation for MDPs*, JCSS 74(8):1309–1331 (2008) (known-ness counts; bonus inside the planner's model).
- Jaksch, Ortner, Auer — *Near-optimal Regret Bounds for RL* (UCRL2), JMLR 11:1563–1600 (2010) (average-reward optimism in planning, never in the stream).
- Harb, Bacon, Klissarov, Precup — *When Waiting is not an Option: Learning Options with a Deliberation Cost*, AAAI 2018, arXiv:1709.04571 (selection-time switch margin).
- Rejected-by-Inv-8 reward-bonus family (signal shape only): Bellemare et al. arXiv:1606.01868; Tang et al. arXiv:1611.04717; Badia et al. (NGU) arXiv:2002.06038; Savinov et al. arXiv:1810.02274; Hou, An, Du (LPM) arXiv:2509.25438 (local).
- Sharifnassab, Salehkaleybar, Sutton — MetaOptimize, arXiv:2402.02342 (local; owns β).
- Naik et al. — Reward Centering, arXiv:2405.09999 (local; ρ̄ stays 1-step).
- Caution: Jafarnia-Jahromi et al. EE-QL (arXiv:2006.04354) is **withdrawn** — do not cite as support for model-free average-reward exploration.
