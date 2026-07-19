# Research dossier — plasticity, reward-rate ceilings, post-zenith decay (2026-07-19)

Scout synthesis for the OaK pathologies: population ceiling (~0.4/step vs SOTA 10–17), "never dies, never walks" mode extinction, and post-zenith decay past ~1M steps. All arXiv IDs verified 2026-07-19. Compatibility filter: (a) fully online one-update-per-step, (b) linear-FA-native or cheap per-head, (c) parameter-free (structurally derivable constants), (d) nothing added to the reward stream, (e) no warmup/phase gating.

**Framing caveat.** Liu & Mou (2602.09234) show classic loss-of-plasticity is largely an artifact of *abrupt* task switches, mostly absent in gradually changing worlds. A continuing HalfCheetah stream is gradual — so the ceiling is more plausibly policy-mode/credit-assignment starvation than representation-plasticity loss. The post-zenith *decay* is the genuinely plasticity-flavored symptom. Corroborating: Dohare et al. (Nature 2024, 2306.13812) — persistent random reinitialization is *the* proven fix for continual learning; matches the internal law that high-rate G&T is the ignition engine.

## Ranked candidates

### R1 — Function-space-bounded updates (ObGD / Intentional TD+PG) on every bootstrapped head
Elsayed-Vasan-Mahmood, "Streaming Deep RL Finally Works" (2410.14606, local, ICML 2025); Sharifnassab et al., "Intentional Updates for Streaming RL" (2604.19033, fetch).
Pick the step size in function-output units (bounded fraction κ of TD-error reduction for values; bounded local KL for policy), then solve for the step size achieving it. For a linear head, exact closed form — the post-update change on the current sample is `α·δ·(eᵀx)`; clamping to κ|δ| is one inner product. The Alberta streaming program converged on this (removing ObGD breaks streaming; one κ held across MuJoCo/DMC/Atari).
Compat: (a)✓ batch=1 by design; (b)✓ exact; (c)⚠️ κ constant — structural "never overshoot" κ=1 available, evidence shows insensitivity; (d)✓; (e)✓.
Effect: ceiling ↑ (safely larger effective steps early), decay ↓ (kills late overshoot spirals). Cost: one dot + one divide per head.
Ablation: ObGD clamp on valueW+psi TD applies only, MetaOptimize still nominates α; Tier-M vs keeper → ONE Tier-L; read zenith step/rate + decay-ratio quartiles. **Queue position L3.**

### R2 — SwiftTD-style bounded eligibility traces
Javed-Sharifnassab-Sutton, SwiftTD (RLC 2024, local); Javed-Sutton, Swift-Sarsa (2507.19539, local).
True Online TD(λ) + per-feature step sizes + explicit trace-update bound + step-size decay; validated with linear FA. **INTERNAL CONFLICT: repo's exhausted families recorded catastrophic TD(λ) divergence (score-gradient explosions 1e70) at Plan-structural α/λ; traces are blocked pending feature diversity.** SwiftTD's bound targets exactly that divergence mechanism, but internal evidence outranks external — DEFER until feature-diversity work lands. Structural λ option if revived: per-head λ from optDuration.

### R3 — Weight-level selective reinit with utility protection (G&T granularity upgrade)
Hernandez-Garcia et al., "Reinitializing weights vs units" (2508.00212, local, CoLLAs 2025); Javed et al. UPGD (2302.03281, fetch).
Reinitializing least-useful *weights* beats unit-level Continual Backprop/ReDo precisely when utility is concentrated — OaK's regime. UPGD: protect high-utility weights, perturb (not zero) low-utility ones every step. OaK's G&T commits at whole-feature/bundle granularity; weight-level commits let a strong lane shed stale coordinates without risking the mode — direct post-zenith lever.
Compat: all ✓ (rate derivable from existing age/capacity structure).
Ablation: same replacement budget/rate, commit granularity feature→weight; decisive rows: best-lane sustain + decay ratio (strong mode must be *better* protected, not just not-worse). **Queue position L5.**

### R4 — Implicit (fixed-point) average-reward TD updates
Kim-Cho-Laber 2510.06149 (local); Kim-Laber 2601.18907 (local); Kim-Toulis-Laber 2505.01361 (local).
Update reformulated as fixed point: effective step `α/(1+α·xᵀx…)` — data-adaptive stabilization inversely with feature norm; provably stable over wider step band; same O(A) cost; no new constants. Composes with R1 (different axes: R1 bounds per-update effect, R4 desensitizes nominal α).
Ablation: implicit-form valueW update at same nominal α, Tier-M → Tier-L read at ~400k (historical ceiling-turnover depth) with kernelRho drift diagnostic. **Queue position L9.**

### R5 — n-step differential targets (global-clock divergence risk)
Zhang et al., "On the Divergence of Differential TD Learning without Local Clocks" (2605.06874, fetch); Blaser-Wang-Zhang 2602.16629 (local).
2605.06874 closes the Wan-2021 open problem: differential TD convergent with a local clock CAN diverge with a global clock — and constant-rate ρ̄ EMA + constant α is exactly a global clock (OaK's geometry). Divergence being extinct is empirical margin, not structural guarantee. 2602.16629 proves on-policy n-step differential TD converges a.s. with standard rates, no local clock. Actionable: two-/n-step differential targets on valueW (reuses the existing planning ring; n structural via optDuration).
Ablation: two-step differential valueW at Tier-L; decisive rows: post-zenith decay ratio, kernelRho trajectory, RMST. **Queue position L10.**

### R6 — Gradient-magnitude utility for G&T replacement (GraMa)
Liu et al., 2505.24061 (local, v2 Jan 2026); related 2605.06834, 2605.09044.
Activation-based utility (dormant ratios, contribution) loses statistical power exactly where it matters; per-unit gradient magnitude measures learning capacity and reset decisions guided by it restore plasticity. 2605.09044 proves rank/NTK diagnostics FAIL to predict trainability — don't trigger replacement on them. OaK already tracks gradEnergy for MetaOptimize; pointing the G&T knife at gradient energy instead of age/contribution is a signal swap, not a new organ.
Compat: all ✓, zero new cost.
Ablation: utility-signal swap only, identical rate/budget, Tier-M + organ-bisect conviction table. **Queue position L4.**

### R7 — Parameter-space exploration noise on optPolicy (weak evidence; screen only)
If the ceiling is mode extinction, directed behavioral noise beats plasticity repair. Compliant form: noise in parameter space on the linear actor (α₀-scaled per-weight); wrapper's fixed σ already guarantees action-space support. Explicitly NON-compliant: SAC entropy terms, visitation-entropy intrinsic rewards (Inv-8).
Ablation: fixed-magnitude per-weight noise on optPolicy only, Tier-M; decisive rows: ignition fraction %>0, best-lane class. Run only if R1/R2 don't move ignition.

### R8 — TRAC parameter-free optimizer as MetaOptimize cross-check
Elsayed et al., "Fast TRAC" (2405.16642, fetch, NeurIPS 2024). Coin-betting-style parameter-free optimizer; deep-only evidence; the only optimizer whose pitch is exactly constraint (c). Screening bet, not conviction.

## Do-not-re-tread (literature-verified failures for this setting)

- Pull-to-init / scale-prior regularizers (L2Init 2308.11958, regenerative reg., spectral 2406.06811, Parseval 2412.07224, FIRE 2602.08040, weight clipping 2407.01704): stationary scale prior — collides with Rule 9 (no magnitude caps) and the Big-World pillar.
- Experience replay (2503.20018 shows it eliminates LoP): (a)-forbidden; its success diagnoses data-reuse starvation — compliant substitutes are traces (deferred) and planning backups.
- Target networks (2405.21043): (a)-forbidden, anti-tracking.
- Periodic resets (2205.07802 primacy bias, 2305.15555): phase-gated (e-fail); internal law already measured resets/interruption as a dead end for the strong mode.
- Intrinsic-reward/entropy-objective exploration: Inv-8 violation.
- Adam/RMSProp normalization heuristics in continual settings (Degris et al. 2401.17401, with Sutton): move step-size vectors AWAY from better ones; MetaOptimize is on the right side.
- Rank/NTK/dormancy as replacement triggers (2605.09044): use gradient signals (R6).
- Multi-model/policy-archive plasticity (TeLAPA 2604.15414, KNIFE 2606.25335): violates single-stream semantics.

## Evidence candor

- Only R1, R2, R4, R5 have linear-FA/streaming/continuing evidence. R3/R6/R8 are deep-only; the transfer bet is that fixed sparse linear representations fail by the same utility-concentration mechanism.
- Nothing in 2024–2026 measures a continuing linear actor-critic at 10M steps. OaK's Tier-XL decay constants would be a contribution, not just a check.
- 2603.21921 (local, "Tale of Two TD Errors"): the two TD-error interpretations diverge only under NONLINEAR FA — with linear heads they're identical; that deep-differential-Q pathology doesn't apply.
- Map paper: "Plasticity Loss in Deep RL: A Survey" (2411.04832).

**Suggested order:** R1 → R3 (ceiling/decay), R6 (free), R4/R5 (decay/turnover), R7/R8 only on screen. Each is a single named FC-STOMP section change with a pre-registered kill criterion.
