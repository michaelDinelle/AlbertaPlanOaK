# Strict ϖ-Relative TD: Closing the BWH-Orthodox Bootstrap Stability Gap

**Audience.** OaK Step-11 agent, Pass 16 architectural rework.
**Brief.** Pass 15 implemented a *partial* ϖ-Relative TD form (`td -= δr · ⟨w, featMean⟩` — a scalar centering inside the TD residual). With diagonal encoder and `pEff ≈ 1.75 / R = 17`, it sufficed: feature covariance was so degenerate that bootstrap variance was naturally damped by the rank deficiency. With the new R×O matrix encoder, `pEff` jumped to ≈ 12.89, the L2-Effective-Rank decay (gated on `pEff < R`) became negligible, and `valueW` diverged to fp32-max within ~50K steps via the classical positive-feedback `V grows → td grows → valueW grows`.

This document distils the rigorous fix in Sakha–Kamalapurkar–Meyn ([arXiv:2603.27874v2](https://arxiv.org/abs/2603.27874), IEEE CDC 2026), explains exactly **what we are doing differently from the paper** (and why that difference creates the divergence), and ranks the BWH-orthodox solutions.

---

## 0. TL;DR

**The current code subtracts the centering term *inside the scalar TD residual*; the paper subtracts it *as a separate vector update along ψ̄*.** The two forms have the **same mean flow** but the current form has noise variance that grows quadratically in `‖w‖`, which dominates the Hurwitz damping and produces the observed divergence.

**Concrete fix (one-liner in `oak_learn_value`):**

```
// BEFORE (current — partial ϖ-RTD; same mean, O(‖w‖²) noise variance):
//   td_centered = (r − ρ) + V(s') − V(s) − vMeanBaseline
//   w[i] += α · z[i] · td_centered
//
// AFTER (strict ϖ-RTD per Eq. (24) of arXiv:2603.27874v2; same mean, O(‖w‖) noise variance):
//   td_uncentered = (r − ρ) + V(s') − V(s)        ← REMOVE the −vMeanBaseline subtraction
//   w[i] += α · z[i] · td_uncentered − α · δ_r · vMeanBaseline · featMean[i]
//                                       ↑ NEW separate rank-1 vector update
```

This is not a "tweak." It is the precise update form that makes Theorem 2.4(ii)'s Hurwitz proof go through. With matrix-encoder-richness `Σ(0)` near-singular, the strict form is necessary; the partial form is provably insufficient.

---

## 1. Paper recap (arXiv:2603.27874v2): the full story

### 1.1 Setting

- MDP `Z = (X, U)` with state-action sequence under fixed (oblivious) policy ϕₑ.
- Linear function approximation `Q_θ(z) = θᵀψ(z)`, basis `ψ : Z → ℝᵈ`.
- Discount factor γ ∈ (0, 1); same algorithm in average-cost limit γ ↑ 1.
- Stationary state-action pmf `ϖ`. Mean feature `ψ̄ = E_π[ψ(0)]`. Autocovariance `Σ(0) = E[ψψᵀ] − ψ̄ψ̄ᵀ`.

### 1.2 Standard TD(λ) mean flow (Tsitsiklis–Van Roy 1997)

`A̅(λ; 0) = −R(0) + (1−λ)γ ∑_{k≥0} (λγ)^k R(k+1)`,  with `R(k) = E[ψ(0)ψ(k)ᵀ]`.

The bound that classical proofs use is `θᵀA̅(λ; 0)θ ≤ −(1−γ)θᵀR(0)θ`. **As γ ↑ 1, this damping coefficient `(1−γ) ↓ 0`** and an eigenvalue of `A̅` approaches zero. In the average-cost setting (γ ≡ 1) this damping vanishes entirely along directions in the kernel of `Σ(0)`.

### 1.3 The Relative TD perturbation (Eq. 16)

`A̅(λ; δ_r) = A̅(λ; 0) − (δ_r / (1−λγ)) · ψ̄ · [ψ̄_μ]ᵀ`

with `[ψ̄_μ]_i = ⟨μ, ψ_i⟩` for any chosen baseline pmf μ. **Theorem 2.3** shows that for an arbitrary μ this perturbation can move an eigenvalue *into the right half plane* (instability!) — the choice of μ matters absolutely.

### 1.4 ϖ-RTD: choose μ = empirical state-action distribution (§1.3 + Eq. 18)

Define the running EMA `ψ̄_n` of the feature vector (this is exactly OaK's `featMean[R]`):

```
ψ̄_{n+1} = ψ̄_n + β_{n+1}·(ψ(n+1) − ψ̄_n),   ψ̄_0 = ψ(Z_0)        (Eq. 18b)
```

with two-time-scale rate `β_n / α_n → ∞` (β faster than α; OaK's tracking-EMA `track_rate` ≫ value step satisfies this).

Substituting μ = ϖ collapses the perturbation to a **symmetric rank-1 PSD matrix**:

```
A̅(λ; δ_r) = A̅(λ; 0) − (δ_r / (1−λγ)) · ψ̄ · ψ̄ᵀ                 (Eq. 19)
```

### 1.5 Theorem 2.4(ii) — the stability guarantee we want

> If μ = ϖ then the mean flow associated with the RTD(λ) algorithm is **stable for any δ_r ≥ 0**, and any γ, λ ∈ [0, 1] satisfying γλ < 1. Moreover `sup [cond(A̅(λ; δ_r)) + trace(Σ_Θ*)] < ∞` over all such (γ, λ, δ_r).

**Proof sketch (Appendix A.3, lines 1259–1333 of the paper).** Take any θ:

```
θᵀA̅(λ; δ_r)θ = θᵀA̅(λ; 0)θ − (δ_r / (1−λγ)) (θᵀψ̄)²
              ≤ −γ̄_P · θᵀΣ(0)θ − (δ_r / (1−λγ)) (θᵀψ̄)²    [Prop. A.2(iii) — Dirichlet/Poincaré bound]
              ≤ −γ̄_P · ε_0 · ‖θ‖²                          [from (A1•) Σ(0) > ε_0·I]
```

The crucial line is `−(δ_r/(1−λγ))(θᵀψ̄)² ≤ 0` — the rank-1 PSD perturbation **adds to the contraction** along the ψ̄ direction. Combined with `Σ(0) > 0`, the matrix is uniformly Hurwitz over all `δ_r ≥ 0` and all `γλ < 1`. **No tuning of `δ_r` is required.**

### 1.6 The Algorithm (Eq. 24): "Fixed ϖ-Relative TD(0)"

The paper writes the **per-step update** explicitly:

```
A_{n+1} = ψ(n) [−ψ(n) + γ ψ(n+1)]ᵀ − δ_r · ψ̄ · ψ̄ᵀ
b_{n+1} = c_n · ψ(n)
θ_{n+1} = θ_n + α_{n+1} [A_{n+1} θ_n + b_{n+1}]
```

Multiplying out — this is the form to compare with code:

```
θ_{n+1} = θ_n + α [ ψ(n) · D_n ] − α · δ_r · v̄_n · ψ̄
```

where:
- `D_n = c_n + γ ψ(n+1)ᵀθ_n − ψ(n)ᵀθ_n`        ← the **standard** (uncentered) TD error
- `v̄_n = ψ̄ᵀ θ_n = ⟨w, featMean⟩`                ← the value-mean baseline (OaK's `vMeanBaseline`)

In the average-reward limit `D_n = (r_n − ρ̄) + V(s_{n+1}) − V(s_n)` with no discount.

For TD(λ) the leftmost `ψ(n)` is replaced by the eligibility vector `z_n` (paper's `ζ_n`).

---

## 2. The bug: scalar centering vs vector centering

### 2.1 What the current code does

From `OakKernel.metal:786–800` (in `oak_predict`):

```
ctx.td[gid] = (ctx.reward[gid] − rho) + terminalMask · vNext − vPrev − vMeanBase;
```

Then in `oak_learn_value:1083–1085`:

```
ctx.valueE[feat] = lambdaTrace · ctx.valueE[feat] + xp;
gradStep = α · (1 − λ) · ctx.td[0] · ctx.valueE[feat];
```

Expanding:

```
w[i] += α (1−λ) · z[i] · ( D_n − δ_r · v̄_n )
      = α (1−λ) · z[i] · D_n − α (1−λ) · δ_r · v̄_n · z[i]      ← centering scaled by z, not ψ̄
```

### 2.2 What the paper specifies (Eq. 24)

```
w[i] += α · z[i] · D_n − α · δ_r · v̄_n · ψ̄[i]                 ← centering scaled by ψ̄, not z
```

### 2.3 They have the same mean flow

The expected per-step update matrix in either case is:

```
E[ A_step ]
   current code:  γ R(1) − R(0) − δ_r · E[z]·ψ̄ᵀ = γR(1) − R(0) − δ_r·ψ̄·ψ̄ᵀ = A̅(λ; δ_r)  ✓
   paper Eq.24 :  γ R(1) − R(0) − δ_r · ψ̄ · ψ̄ᵀ                              = A̅(λ; δ_r)  ✓
```

Identical. So in expectation, both are correct.

### 2.4 They have very different noise — and that is the divergence

Decompose `(per-step update) = (mean flow drift) + noise`. The noise contributed by the centering term:

| Form | Per-step centering term | Noise covariance contribution |
|---|---|---|
| Current code (scalar TD) | `−α · δ_r · v̄ · z` | `α² · δ_r² · v̄² · E[zzᵀ]` |
| Paper Eq. 24 (strict)    | `−α · δ_r · v̄ · ψ̄` | `α² · δ_r² · v̄² · ψ̄ψ̄ᵀ`     |

The first is `O(‖w‖²)` in spectrum (because `v̄ = ψ̄ᵀw` scales linearly with `w`, and `E[zzᵀ] ≈ R(0)` is a full-rank matrix that doesn't shrink as `w` grows). The second is also `O(‖w‖²)` in magnitude **but rank-1** — the noise lives entirely along `ψ̄`, exactly the direction that the deterministic damping is contracting.

The deterministic damping from Theorem 2.4(ii) is `−γ̄_P ε_0 ‖w‖²` (linear in `‖w‖²`). For the system to be stable, the second-moment growth rate of the noise must not exceed the contraction rate.

**Why the current form fails:** The noise injects energy into directions *orthogonal* to ψ̄ (any direction in the support of `R(0)` other than ψ̄). The Hurwitz proof relies on **all** of that energy being damped by `−γ̄_P θᵀΣ(0)θ`. In our setup with the matrix encoder, `Σ(0)` is rank-deficient (`pEff = 12.89 < R = 17`); the nullspace of `Σ(0)` is `R − pEff ≈ 4.11` dimensional, and the noise injected by the current form along those directions accumulates without damping.

**Why the strict form succeeds:** The noise is supported entirely along `ψ̄`, which is in the column space of `R(0) ⪰ Σ(0)` by construction. The strict form's noise is rank-1 along the *most-damped* direction. There is no accumulation along under-damped directions.

### 2.5 The same logic applies to ψ (cumulant successor features)

`oak_learn_psi` has the same structure with `tdPsi[k] -= ψ_meanBaseline[k]` inside the scalar. The fix is identical: subtract the cumulant-mean centering as a separate vector update along `featMean` (or `concat[featMean, λ̄]` for the chord-conditional ψ basis), not inside the scalar `tdPsi[k]`.

---

## 3. Why partial centering "worked" in Pass 15 (diagonal encoder)

With the diagonal encoder, `pEff ≈ 1.75 / R = 17`. The features were almost rank-1 — all encoder outputs were tracking essentially one direction in observation space (the dominant Welford-PCA direction). In that regime:
- `Σ(0)` has rank 1–2; nullspace is `R − 1 ≈ 16` dimensional.
- `featMean` is ≈ aligned with the dominant eigenvector of `R(0)`.
- The L2-Effective-Rank decay (`l2er_decay(α₀, R, pEff)` in `oak_learn_value:1072`) was ALWAYS firing because `pEff < R`. That decay pulls `valueW` toward zero with rate `≈ α₀ · (1 − pEff/R)`, providing a *first-order* damping along ALL coordinates (it's a coordinate-wise L2 weight decay, fully isotropic).
- The L2 weight decay was strong enough to drown out the noise from the scalar-centering form.

With the matrix encoder, `pEff ≈ 12.89 / R = 17`:
- The L2-ER decay coefficient drops by factor `(1 − 12.89/17) / (1 − 1.75/17) ≈ 0.27` — about 4× weaker isotropic damping.
- `Σ(0)` has rank ≈ 12.89; nullspace is only ≈ 4.11 dimensional, but it exists.
- The `valueW` directions in that nullspace are no longer damped by the ϖ-RTD perturbation (which only damps along `ψ̄`) AND no longer damped strongly by L2-ER decay.

Net effect: a 4×-richer feature space + 4×-weaker damping = the previously hidden divergence becomes manifest.

---

## 4. Ranked list of fixes (most BWH-orthodox first)

### Tier S+ (apply first; structural; zero asymptotic bias)

#### Fix 1. Strict ϖ-RTD form in `oak_learn_value` and `oak_learn_psi`
**Paper.** Sakha-Kamalapurkar-Meyn, [arXiv:2603.27874v2](https://arxiv.org/abs/2603.27874), Eq. (24), Thm. 2.4(ii).

**Math.**

```
// In oak_predict (replace the centered-TD line):
ctx.td[gid] = (ctx.reward[gid] − rho) + terminalMask · vNext − vPrev;   ← UN-centered standard TD

// In oak_learn_value (the per-feature update):
xp     = prev[feat];
e[feat] = λ_trace · e[feat] + xp;
g_TD   = α · (1−λ_trace) · ctx.td[0] · e[feat];                          ← standard TD term
g_RTD  = α · δ_r · ctx.scalars->vMeanBaseline · ctx.featMean[feat];      ← NEW separate term
w[feat] = w[feat] · (1 − λ_L2) + g_TD − g_RTD;
```

For ψ:

```
// In oak_predict for tdPsi (replace the centered line):
tdPsi[k] = (φ_k − ρ_k) + ψ_k(s')·z' − ψ_k(s);                            ← UN-centered

// In oak_learn_psi (per-feature update; basis dim m ∈ [0, M)):
g_TD   = α · (1−λ_trace) · tdPsi[k] · e_psi[k][m];
g_RTD  = α · δ_r · ψMeanBaseline[k] · meanBasis[m];                      ← NEW; meanBasis = concat[featMean, λ̄]
ψ[k][m] = ψ[k][m] · (1 − λ_L2) + g_TD − g_RTD;
```

For the actor `oak_learn_pi`, the centering does not directly apply (the policy gradient `δ · z · ∇log π` has no `V` term), but the value-head stabilization removes the proximate driver of `δ` divergence.

**Choice of `δ_r`.** Theorem 2.4(ii) says ANY `δ_r ≥ 0` is stable when μ = ϖ. BWH-structural choices:
- `δ_r = 1` (paper uses this in their continuous-state speed-scaling experiments, §3.2).
- `δ_r = 1 − λ_trace = 1/cap` (matches OaK's structural Seijen-Sutton (1−λ) scaling and gives the same coefficient as the per-step update bound).
- `δ_r = 1 − ρ̂` where `ρ̂ ∈ (0, 1)` is the spectral gap proxy from `pEff/R` — adapts to feature richness.

Recommend **`δ_r = 1`** as the simplest BWH-orthodox starting point (paper-validated, no hyperparameter).

**Expected impact.** Removes the `O(‖w‖²)` noise term, restores the rank-1 PSD damping along `ψ̄`. With `Σ(0)` near-singular but rank ≥ 1, the strict form is the *minimum* sufficient condition for Theorem 2.4(ii) to apply.

**Implementation complexity.** ~10 lines per affected kernel (`oak_predict`, `oak_learn_value`, `oak_learn_psi`). One scalar (`δ_r`) added to `KernelScalars` (or hard-code = 1). One additional `R`-vector dot product per step (for the `featMean[feat]` lookup; already in arena).

**BWH-orthodoxy check.**
- Inv-1 (no env branches): ψ̄ is just the existing `featMean` Welford EMA. ✓
- Inv-7 (no cosmetic normalization): the baseline subtraction is **required** for stability; it is the rank-1 PSD perturbation that makes A̅ Hurwitz. ✓
- Inv-9 (parameter-free): δ_r = 1 (paper-validated) or δ_r = 1/cap (structural). ✓
- Inv-8 (objective isolation): centering operates on the linear-FA TD residual; greedy policies are invariant under uniform shifts (Sutton-Barto Ch 10; Devraj-Meyn 2022 for Q-learning). ✓

---

#### Fix 2. Ensure `Σ(0) > 0` strictly: feature decorrelation floor
The Hurwitz proof of Thm. 2.4(ii) requires `Σ(0) > ε_0 I` (assumption A1•). If the matrix encoder has perfectly aligned columns, `Σ(0)` is degenerate and ε_0 = 0 — Theorem 2.4(ii)'s lower bound `−γ̄_P ε_0 ‖θ‖²` becomes vacuous along the kernel direction.

**OaK already has the L2-ER decay** (`l2er_decay(α₀, R, pEff)`) which fires when `pEff < R`. In the matrix-encoder regime where pEff jumped to 12.89, the gate fires only weakly. **The fix is to tighten the gate** so that it always provides at least `ε_0 = ridge` damping along the under-determined directions:

```
// In oak_learn_value (replacing the existing l2er_decay call):
float pEff_floor = max(pEff, R - 1.0f);                  ← always treat as 1-deficient
float lambda_L2 = l2er_decay(α₀, R, pEff_floor);
```

This guarantees a minimum effective `ε_0 ≥ α₀ · ridge` regardless of the actual rank — which is the structural floor Sutton-Barto §9.5 calls for in linear FA.

**Combine with Fix 1** for full Theorem 2.4(ii) coverage.

**BWH-orthodoxy.** PASS — `pEff` is structural (trace/max(featSM)). The `R − 1` floor is "always damp at least one direction", which is the minimum the proof requires.

**Bias.** Zero asymptotic bias (the L2-ER decay is multiplicative shrinkage; in expectation it pulls `w → ε_0/(ε_0 + λ) · w*` which → `w*` as `λ → 0`).

---

### Tier S (apply if Tier S+ is insufficient; require integration work)

#### Fix 3. Bellman Error Centering (BEC) on top of strict ϖ-RTD
**Paper.** Chen-Gong-Yang-Wang, [arXiv:2502.03104](https://arxiv.org/abs/2502.03104), AISTATS 2025.

**Math.** Track a scalar EMA of the TD error itself:
```
bec_ω_{t+1} = bec_ω_t + β·(δ_t − bec_ω_t)
δ_centered = δ_t − bec_ω_t        ← subtract zero-mean shifted residual
```

This is **orthogonal to ϖ-RTD**: ϖ-RTD changes the *mean-flow matrix* to make A̅ Hurwitz; BEC changes the *noise statistics* to be zero-mean. They compose. Theorem 4.1 of Chen et al. proves a.s. convergence for on-policy CTD (Centered TD) with bounded second-moment features.

**OaK addition.** One scalar `bec_omega` in `KernelScalars`. Update in `oak_bk_admit_commit` via `bec_omega += β·(td[0] − bec_omega)`. Subtract in `oak_predict` from `td_uncentered` AFTER the standard TD computation but BEFORE `oak_learn_value`.

**BWH-orthodoxy.** PASS — Welford EMA on TD error; analogous to existing ρ̄ EMA on reward.

---

#### Fix 4. Two-timescale ratio enforcement (Borkar 1997)
**Paper.** Borkar 1997 → arXiv:[2402.01371](https://arxiv.org/abs/2402.01371) (Bhatnagar-Borkar-Reddy 2024 critic-actor for avg-reward FA).

OaK's nominal step-size structure:
- `α_value` (per-feature TIDBD around `α₀ = 1/cap`)
- `α_pi` (REINFORCE around `α₀ / (1 + α₀ · ‖x‖² · ‖scoreGrad‖²)`)
- `β_featMean = track_rate(stepCount, ridge) ≈ 1/min(stepCount, cap)`

Two-timescale convergence (paper Theorem 1) requires `α_pi → 0` *strictly faster* than `α_value`, with `α_pi/α_value → 0`. **Currently both are O(α₀) — a single timescale.** This is the technical reason the actor and critic can co-evolve catastrophically.

**Fix.** Slow the actor: `α_pi := α_value / sqrt(stepCount)` (or any `α_pi = o(α_value)`). Also speed the encoder if possible: `α_enc := α_value / track_rate ≈ α_value · cap` is too aggressive; use `α_enc = α_value` is fine.

The paper's "critic-actor" twist (reverse the usual ordering) gives `Õ(ε^{-2-δ})` sample complexity for avg-reward FA.

**BWH-orthodoxy.** PASS — step-size schedules are structural. No new clamps.

---

#### Fix 5. Streaming LSTD with Sherman-Morrison rank-1 updates
**Paper.** Yu-Bertsekas 2009 (LSPE for avg-cost), Boyan 1999 (LSTD(λ)), iLSTD (Geramifard-Bowling-Sutton 2006).

LSTD computes the closed-form fixpoint of the Bellman equation:
```
A_t = ∑_τ ψ(τ) · (ψ(τ) − γψ(τ+1))ᵀ
b_t = ∑_τ r(τ) · ψ(τ)
w_t = A_t⁻¹ · b_t
```

Sherman-Morrison rank-1 updates maintain `A_t⁻¹` incrementally at cost `O(d²)` per step — affordable at d ≤ 100 (OaK has E ≤ ~50 typically).

**For continuing avg-reward** (γ → 1), the natural form is the **differential LSTD**:
```
A_t = ∑_τ ψ(τ) · (ψ(τ) − ψ(τ+1) + ψ̄)ᵀ        ← centered by feature mean
b_t = ∑_τ (r(τ) − ρ̂) · ψ(τ)
```

This **closes the bootstrap loop in closed form** — there's no `V grows → td grows` feedback because `w_t` is computed exactly each step.

**Cost.** O(E²) per step for the Sherman-Morrison update + O(E) for the final solve. At E = 50, that's 2500 multiplies — comparable to the existing oak_learn_psi cost.

**BWH-orthodoxy.** WEAK PASS. LSTD is a memory-of-statistics method; it stores `(A, b)` running sums, not raw transitions. No replay buffer, but does store more state than pure SGD. Consistent with OaK's existing `featMean`, `featSM`, `wBar` running sums.

**Integration risk.** Replacing `oak_learn_value`'s SGD with LSTD is a substantial architectural change. Probably wait until after Fix 1 is in and verified.

---

### Tier A (use if Fix 1 is insufficient AND Fix 3+4 don't help)

#### Fix 6. Bounded value head with PopArt-style rescaling
**Paper.** van Hasselt-Hessel-Hasselt-Aslanides-Bilal-Silver, *Multi-task Deep RL with PopArt*, [arXiv:1809.04474](https://arxiv.org/abs/1809.04474), AAAI 2019.

PopArt = "Preserving Outputs Precisely while Adaptively Rescaling Targets". Tracks `μ_V, σ_V` of the value targets and reparameterizes:

```
V(s) = σ_V · (w · z) + μ_V        ← physical V
                                    where (w · z) is a "normalized" V
```

When `(μ_V, σ_V)` are updated by Welford EMA, rescale `(w, b_V)` so that the *physical* `V(s)` is preserved exactly:
```
w' := w · σ_V_old / σ_V_new
b_V' := (b_V · σ_V_old + μ_V_old − μ_V_new) / σ_V_new
```

This is the value-head analogue of the encoder's Welford-variance standardization: it tracks the world's value scale instead of imposing one.

**For BWH-orthodox tanh-bounded V:** `V_phys(s) = σ_V · tanh(w · z) + μ_V` — soft saturating, smooth gradient. The gradient becomes `∂V/∂w[i] = σ_V · (1 − tanh²(w·z)) · z[i]`, which **vanishes** as V saturates — providing the missing bounded-output property without imposing a hard cap.

**Important caveat.** PopArt was designed for return scales spanning multiple orders of magnitude across tasks; it is **not** primarily a stability fix. Within a single task, the value-scale tracking may oscillate and inject extra noise that can WORSEN stability. Apply only if Fix 1 + Fix 2 + Fix 3 are insufficient.

**BWH-orthodoxy.** PASS for the rescaling itself (Welford-style world-scale tracking). The tanh squashing is a soft saturator (zero bias at origin, smooth derivative); paper precedent is Iwaki MDAC ([arXiv:2502.03854](https://arxiv.org/abs/2502.03854)).

**Implementation cost.** Substantial: requires a `(μ_V, σ_V)` welford in `KernelScalars`, the rescaling logic in `oak_bk_admit_commit`, and the tanh in `oak_predict` and `oak_meta_action`. Save for Pass 17+.

---

#### Fix 7. Actor `(1 − μ²)` chain rule + matched implicit-α denominator
**Paper.** SAC (Haarnoja et al. 2018), Stream-AC ObGD (Elsayed et al. [arXiv:2410.14606](https://arxiv.org/abs/2410.14606)).

User reported: adding the `(1 − μ²)` tanh chain factor caused valueW to explode. Mechanism:

For a tanh-squashed Gaussian policy, `a = tanh(z)` with `z ~ N(μ_pre, σ²)`, the score function w.r.t. pre-tanh mean is `(z − μ_pre)/σ²`. For w.r.t. **post-tanh action**, the chain rule contributes `(1 − tanh²(z)) = (1 − a²)`:
```
∇_θ log π(a|s) = (a − μ_post)/σ² · (1 − a²) · z          ← post-tanh form
∇_θ log π(z|s) = (z − μ_pre)/σ² · z                      ← pre-tanh form (no chain rule)
```

If `scoreGrad` was using the pre-tanh form (the SAC-paper-correct form per Appendix C), adding `(1 − μ²)` introduces a SPURIOUS factor that:
- Reduces the effective gradient near saturated actions, accelerating policy convergence to the boundary.
- Reduces `‖scoreGrad‖²` in the implicit-α denominator (`α_pi = α₀ / (1 + α₀ · ‖x‖² · ‖scoreGrad‖²)`), which **increases** `α_pi` — net effect: larger actor steps near saturation, the opposite of what was intended.

**Recommendation.** Decide which form scoreGrad uses (pre-tanh or post-tanh) and apply the chain rule consistently:
- If pre-tanh: do NOT add `(1 − μ²)`.
- If post-tanh: DO add `(1 − μ²)` to scoreGrad **and** ensure the implicit-α denominator includes the same factor.

**BWH-orthodoxy.** Score function correctness is a math identity, not a clamp. PASS either form, but be consistent.

---

#### Fix 8. Reject Salimans-Kingma weight normalization (anti-BWH)
**Paper.** Salimans-Kingma, [arXiv:1602.07868](https://arxiv.org/abs/1602.07868), NIPS 2016.

`w = (g/‖v‖)·v` separates direction and magnitude into two parameters. The paper claims acceleration of SGD convergence. **For BWH-orthodox OaK this is anti-pattern**:
- Imposes a magnitude prior `g` (not derived from world statistics).
- The reparameterization is "cosmetic normalization" (Inv-7 violation): the scaling does nothing the world's data couldn't tell us via Welford.
- Weight Normalization's alleged self-stabilization (`‖v‖` grows monotonically) is a *training-dynamics* artifact, not a structural Hurwitz guarantee.

**Reject.** Use PopArt (Fix 6) or strict ϖ-RTD (Fix 1) instead.

---

## 5. Concrete first recommendation

**Implement Fix 1 (strict ϖ-RTD per Eq. 24) on `oak_learn_value` and `oak_learn_psi` only.**

This is:
1. **Theoretically the strongest move.** It is the precise update form for which Theorem 2.4(ii) was proven; the partial form has the same mean but provably bad noise behavior in the matrix-encoder regime.
2. **The minimum change.** ~10–15 lines per kernel, one scalar (`δ_r`) added to `KernelScalars`. Touches no other invariants.
3. **Self-validating.** If Theorem 2.4(ii) is correctly applied, `valueW` should remain bounded over the full 500K-step bench WITHOUT needing Fix 2/3/4. If divergence persists, the bug is somewhere else (encoder, ψ, actor) and we have diagnostic value: Fix 1 isolates the value-head divergence cleanly.
4. **Reversible.** Setting `δ_r = 0` recovers the standard differential TD; the partial-form code path can be kept behind a single flag for ablation.

**Sequence after Fix 1 lands and is verified:**
- (Day +1) Apply Fix 1 to `oak_learn_psi` cumulant heads (same structural reason).
- (Day +2) Add Fix 2 (rank floor on L2-ER decay) — costs nothing, provides guaranteed `ε_0 > 0` damping along all directions.
- (Day +7) If `actor` divergence still surfaces, decide Fix 7 (tanh chain rule consistency).
- (Day +14) Only if all of the above fail, consider Fix 6 (PopArt) or Fix 5 (LSTD) — significant refactors with integration risk.

---

## 6. Pseudocode for the strict ϖ-RTD update

```
// One step of strict ϖ-Relative TD(λ) for linear-FA continuing avg-reward
// Following Sakha-Kamalapurkar-Meyn arXiv:2603.27874v2 Eq. (24) extended to TD(λ).

inputs:  α (step-size), λ (trace decay), δ_r ≥ 0 (ϖ-RTD weight; default = 1 per Thm 2.4(ii))
state:   w[d] (value weights), z[d] (eligibility trace), ψ̄[d] (featMean EMA), ρ̄ (reward EMA)

for each transition (s, a, r, s'):
    ψ_curr[i]  = features(s)[i]
    ψ_next[i]  = features(s')[i]
    V_curr     = ⟨w, ψ_curr⟩
    V_next     = ⟨w, ψ_next⟩
    v̄          = ⟨w, ψ̄⟩                        // value-mean baseline (vMeanBaseline)

    // STANDARD differential TD residual (no centering inside scalar):
    D = (r − ρ̄) + V_next − V_curr

    // Eligibility trace (backward view, λ-trace):
    z[i] = λ · z[i] + ψ_curr[i]                // for all i

    // Weight update — TWO terms:
    for each i:
        g_TD  = α · (1 − λ) · D · z[i]         // Seijen-Sutton (1−λ) scaling
        g_RTD = α · δ_r · v̄ · ψ̄[i]            // NEW: rank-1 PSD damping along ψ̄
        w[i] += g_TD − g_RTD

    // EMA updates (faster timescale per Eq. 18b; β > α):
    ψ̄[i]  += β · (ψ_curr[i] − ψ̄[i])           // featMean (already in OaK)
    ρ̄     += β · (r − ρ̄)                       // reward EMA (already in OaK)

// MEAN FLOW MATRIX (the thing Theorem 2.4(ii) makes Hurwitz):
//     A̅(λ; δ_r) = E[ z · (γψ_next − ψ_curr)ᵀ ] − δ_r/(1−λγ) · ψ̄ · ψ̄ᵀ
// The PSD perturbation −δ_r·ψ̄ψ̄ᵀ adds contraction along ψ̄ to the standard TD(λ) drift.
// In the avg-reward limit γ → 1, the standard term still has Σ(0) damping for any θ
// when (A1•) Σ(0) > 0 holds (i.e., feature covariance is full rank).
```

---

## 7. Diagnostic to verify Fix 1 is working

After implementing Fix 1, log:
1. **`valueWMax` over time.** Should plateau at O(R · max_reward / (γ̄_P · ε_0)) instead of growing exponentially. With δ_r = 1 and Σ(0) ≥ 0.1·I, expect `valueWMax < ~10·max(|r − ρ̄|)`.
2. **Cosine of `valueW` with `featMean`.** The ϖ-RTD perturbation should keep `valueW` close to the column space of `featMean`. If `cos(w, ψ̄) → 0`, then `v̄ → 0` and the centering vanishes — diagnose whether `featMean` is degenerate.
3. **Effective rank of the empirical `A̅`.** Track `cond(A̅_emp)` where `A̅_emp = (1/T) Σ z · (γψ' − ψ − δ_r·ψ̄)ᵀ`. Theorem 2.4(ii) guarantees this is uniformly bounded; if cond → ∞, then either the proof's assumptions fail (Σ(0) singular) or the implementation is wrong.

---

## 8. Cited papers (with arXiv IDs)

| Paper | arXiv ID | Role |
|---|---|---|
| Sakha-Kamalapurkar-Meyn 2026 | [2603.27874v2](https://arxiv.org/abs/2603.27874) | **PRIMARY** — strict ϖ-RTD form, Thm 2.4(ii), Eq. 24 |
| Tsitsiklis-Van Roy 1997 | (TAC; pre-arXiv) | Baseline TD(λ) Hurwitz analysis |
| Devraj-Meyn 2022 | TAC; tabular Q-learning RTD | Original RTD with bounded variance (tabular only) |
| Wan-Naik-Sutton 2021 | wan21a@ICML; [2110.13855](https://arxiv.org/abs/2110.13855) | Differential TD avg-reward — convergence WITHOUT centering |
| Chen-Gong-Yang-Wang 2025 | [2502.03104](https://arxiv.org/abs/2502.03104) | BEC: zero-mean centering on TD residual; composes with RTD |
| Bhatnagar-Borkar-Reddy 2024 | [2402.01371](https://arxiv.org/abs/2402.01371) | Two-timescale critic-actor for avg-reward FA |
| Yu-Bertsekas 2009 | TAC | LSPE(λ) for avg-cost — closed-form alternative |
| Boyan 1999 | (NIPS '99) | Original LSTD(λ) |
| Salimans-Kingma 2016 | [1602.07868](https://arxiv.org/abs/1602.07868) | Weight-norm — REJECTED (anti-BWH, magnitude prior) |
| van Hasselt et al. 2019 | [1809.04474](https://arxiv.org/abs/1809.04474) | PopArt — value-scale tracking (Tier A backup) |
| Iwaki 2026 (v3) | [2502.03854](https://arxiv.org/abs/2502.03854) | MDAC bounded advantage — soft saturator precedent |
| Elsayed-Vasan-Mahmood 2024 | [2410.14606](https://arxiv.org/abs/2410.14606) | Stream-AC ObGD — implicit step for actor (current OaK extension) |
| Kim-Cho-Laber 2025 | [2510.06149](https://arxiv.org/abs/2510.06149) | Implicit avg-reward TD(λ) — already in OaK |
| He et al. 2025 | [2509.22335](https://arxiv.org/abs/2509.22335) | L2-Effective-Rank decay — already in OaK |
| Haarnoja et al. 2018 (SAC) | [1801.01290](https://arxiv.org/abs/1801.01290) | Squashed Gaussian log-prob; (1−μ²) chain rule |

---

## 9. Closing notes

The current pass-15 implementation of "ϖ-RTD" is mathematically the *centered TD residual* form (subtraction inside the scalar `td`). This is what one writes when reading equation (15) of the paper — the **definition** of D_{n+1} — and using the natural update `w += α·z·D`. It was a sensible first reading. But the paper's stability proof relies on the *fixed* form (equation 24), where the centering is a separate vector update. The two forms have the same mean but very different noise — and in the matrix-encoder regime that variance gap becomes the divergence.

The good news: the fix is small, structural, paper-validated, BWH-orthodox, and reversible. We have everything we need (`featMean`, `vMeanBaseline`, `valueE`, `valueW`) — we just need to wire the centering term to its correct place: not inside `td`, but as a separate vector update along `featMean`.
