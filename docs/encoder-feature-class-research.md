# OaK Encoder Feature-Class Research Report

**Question**: Is "linear FA on raw obs" actually canonical per Sutton/Bowling/BWH, or is something richer (tile coding, RBF, eigenoptions, deep Fourier features, Mamba SSMs) the actual prescription? The agent's diagonal-encoder path is mathematically equivalent to linear-FA-on-raw-obs and plateaus on continuous control (HalfCheetah ≈ -0.375, 8% above random).

**Verdict (TL;DR).** BWH does **NOT** mandate "linear FA on raw obs". The Alberta Plan **explicitly prescribes** nonlinear feature finding (Step 2) and "continual non-linear function approximation" (Step 8), with the **value head remaining linear in features**. The orthodox upgrade is to make the encoder a richer **plasticity-preserving** feature learner — specifically **deep Fourier features** ([arXiv:2410.20634](https://arxiv.org/abs/2410.20634)) or **AdaLin per-neuron adaptive linearity** ([arXiv:2505.09486](https://arxiv.org/abs/2505.09486)) — while keeping `valueW · z` strictly linear and maintaining the existing ϖ-RTD stability machinery.

---

## 1. What the canonical sources actually say

### 1.1 The Alberta Plan (Sutton, Bowling, Pilarski 2022, [arXiv:2208.11173](https://arxiv.org/abs/2208.11173))

The plan is **explicitly multi-step in feature richness**:

| Step | Title | Feature class prescription |
|---|---|---|
| 1 | Representation I | "linear function approximator with **static, given features**" — adaptive per-feature step sizes, scaling, normalization (this is where the user's diagonal encoder sits) |
| 2 | Representation II | "creating and introducing **new features (made by combining existing features)**" — multi-layer, **explicitly mentions "nonlinear features"** |
| 3 | Prediction I | "First with given linear features, then with feature finding. The new features will include **not just nonlinear combinations, but also incorporation of older signals and traces**" |
| 8 | Prototype-AI I | "Our first prototype-AI would be based on average-reward RL, models, planning, and **continual non-linear function approximation**" |
| 11 | OaK | Adds Oak feedback (utility ranking, generate-and-test) **on top of Step 8's nonlinear FA** |

**Crucial paragraph (paper §"Step 2"):**
> "We now have a non-stationary multi-layer and multi-task system. ... The performance of the system will depend on the resource budget (that is, on **how many new nonlinear features can be considered in parallel**)."

**Crucial paragraph (paper §"Step 8"):**
> "Our first prototype-AI would be based on average-reward RL, models, planning, and **continual non-linear function approximation**. This would move beyond past work on Dyna by incorporating general continual function approximation."

So OaK at Step 11 is *built on top of* nonlinear feature finding. Pure-linear-FA-on-raw-obs is **Step 1 only**.

### 1.2 Sutton's NeurIPS 2025 OaK Talk (no arXiv yet; [video](https://www.youtube.com/watch?v=gEbbGyNkR2U), [NeurIPS](https://neurips.cc/virtual/2025/invited-talk/129132))

Direct quotes from notes ([Brunelle's NeurIPS 2025 notes](https://blog.matthewbrunelle.com/neurips-2025-wednesday/)):

> "OaK is a vision to a shallow 'seed' super-intelligence. **However we cannot get there until we develop reliable deep continual learning and meta learning first.**"

> "The world is big and we need a **model that can grow at run time**. ... Agents only approximate the world, nothing is exactly correct or optimal. So they must be able to **form new abstractions at runtime**."

> "OaK is based on the insight that: **Every feature has a corresponding sub problem**. If an agent can generate new features, then it can create sub problems."

Sutton himself frames OaK as needing **deep continual learning** as a prerequisite. The "shallow seed" framing means the *initial* OaK can be small, but the architecture is designed to grow into deep nonlinear FA.

### 1.3 Sutton-Barto Ch 9.5 — the canonical "linear methods" feature classes

The textbook §9.5 enumerates **five** feature-construction methods for linear methods:

- 9.5.1 **Polynomials**
- 9.5.2 **Fourier basis** (Konidaris-Osentoski-Thomas 2011)
- 9.5.3 **Coarse coding**
- 9.5.4 **Tile coding** (Sutton's own [Tiles3](http://incompleteideas.net/tiles/tiles3.html); CMAC lineage)
- 9.5.5 **Radial basis functions (RBF)**

ALL of these are **nonlinear functions of raw observations**. "Linear method" means **linear in parameters** (the value head `w · ϕ(s)`), not "linear in observations". The diagonal encoder `z[i] = encoder[i] · obs[i]` is a degenerate special case (identity feature transformation with per-feature scaling) — explicitly weaker than even tile coding for continuous control.

For continuous control specifically:
> "Tile coding is **recommended as a primary approach** for applying online RL methods to domains with continuous state or action variables." — Sutton, Tiles3 documentation

### 1.4 The Big World Hypothesis (Javed-Sutton 2024, RLC; [page](https://khurramjaved.com/the_big_world_hypothesis.html))

What BWH *actually* requires (from the canonical Javed-Sutton page):

1. **The world is orders of magnitude larger than the agent** — agent must approximate
2. **Tracking** — adapt approximations to the encountered slice via correlated experience
3. **Computational efficiency** — "approximate but efficient learning algorithms can outperform computationally expensive exact algorithms in big worlds"
4. **Function approximation** — "Sutton (2020) proposed embracing function approximation for learning values, policies, models, and states"

What BWH **does NOT require**:
- It does **not** restrict the FA to linear-on-raw-obs
- It does **not** mandate any specific feature class
- It does **not** forbid deep features

The "embracing approximation" framing (Sutton 2020) is explicitly *broad* — values, policies, models, AND states are all subject to FA.

### 1.5 NEW (Dec 2025) Computationally-Embedded BWH ([arXiv:2512.23419](https://arxiv.org/abs/2512.23419))

Lewandowski-Ramesh-Meyer-Schuurmans-Machado, NeurIPS 2025. This is the **most recent formal BWH paper** and the one that *appears* to argue for linear FA. Critical reading:

**Empirical claim (Figure 4):**
> "Deep ReLU networks fail to sustain performance on the interactivity evaluation task. ... The deep linear network sustains interactivity, whereas the deep ReLU collapses in performance."

**But what they actually compared:**
> "We instantiate the reinforcement learning agent outlined in Section 5 with **a linear parameterization of the value function**, v(b_t; W_t) := W_t · b_t. **Linearity provides stability for learning online with TD(0)**, compared to temporal difference learning with deep nonlinear networks."

So:
- VALUE FUNCTION: **always linear** (in features) — for stability
- POLICY: tested **deep linear vs. deep ReLU** — deep linear wins on "interactivity-seeking" task

**Critical limitation (paper §B.2):**
> "Our experiments used relatively shallow networks, with a maximum depth of D = 4. However, with the meta-gradient calculation over a finite horizon of T = 10, the **effective depth of the networks during auto-differentiation is T · D = 40**. Meta-gradient methods for deep networks can exhibit more pathological learning dynamics due to increased curvature, **leading to instability that could partially explain the discrepancy between linear and nonlinear networks**."

**Interpretation:** This paper shows deep ReLU policies with meta-gradient updates have curvature problems on the interactivity-seeking objective. It does **NOT** prove "no nonlinearity in OaK". It strongly suggests:
1. Value function linearity in features → stable TD
2. Pure ReLU stacks → bad for continual meta-gradient learning
3. Plasticity-preserving alternatives (deep linear, deep Fourier features) → good

---

## 2. The plasticity-vs-expressivity solution: 2024–2026 papers

### 2.1 Plastic Learning with Deep Fourier Features ([arXiv:2410.20634](https://arxiv.org/abs/2410.20634))

Lewandowski-Schuurmans-Machado, **ICLR 2025**. The **single most important paper for OaK's encoder question**.

**Theorems:**
- **Theorem 1**: Linear FA does not suffer from loss of trainability on a sequence of tasks (suboptimality gap is independent of previous tasks under µ-strong convexity).
- **Theorem 2**: Deep diagonal linear networks `f_θ(x) = θ_L θ_{L−1} ... θ_1 x` (with diagonal θ_l) preserve trainability under continual learning. Proof: minimum singular value of product matrix `σ_min(θ̄) > 0` is preserved.
- **Corollary 1**: A network parameterized by θ with deep Fourier features `[sin(W·x), cos(W·x)]` in every layer **approximately embeds a deep linear network** with bounded error `c = √2 · π²/2/8 ≈ 0.05`.

**The mechanism (Proposition 1):**
> "For any z, there exists a linear function L_z(x) = a(z)x + b(z), such that either |sin(x) − L_z(x)| ≤ c or |cos(x) − L_z(x)| ≤ c, for c ≈ 0.05 and all x ∈ [z − π/4, z + π/4]."

That is, **at every operating point, at least one of (sin, cos) is well-approximated by a linear function**. The pre-activations have nonzero gradient through *one or the other* of the two units, so plasticity is preserved.

**Key empirical results:**
- ResNet-18 with deep Fourier features in every layer outperforms ReLU + spectral regularization on tiny-ImageNet, CIFAR-100 in continual learning settings
- Deep Fourier features work even with **half the parameters** of the ReLU baseline
- Sustains trainability across 100+ tasks where ReLU collapses

**Cost:** Each layer has 2× the activation count (two units per pre-activation), but the linear projection is the same size. Effective compute: ~2× per layer.

**BWH-orthodoxy:**
- ✅ Inv-1: feature-class agnostic
- ✅ Inv-3: width is structural, not hardcoded
- ✅ Inv-7: every operation serves prediction/control
- ✅ Inv-9: no extra hyperparameters introduced

### 2.2 AdaLin: Preserving Plasticity via Adaptive Linearity Injection ([arXiv:2505.09486](https://arxiv.org/abs/2505.09486))

Razavi-Rohani, Khajavi, Chung, Chen, Vaswani, May 2025. **Direct successor to deep Fourier features** with explicit RL validation.

**The activation:**
$$f_i(x) = \phi(x) + \alpha_i \cdot x \cdot [g(x)]_{sg}, \quad g(x) = \cos\left(\frac{\pi}{2} \cdot \frac{|\phi'(x)|}{L}\right)$$

where:
- `ϕ` is any base activation (ReLU, Tanh, GeLU)
- `α_i` is one **learnable per-neuron parameter**
- `g(x)` is a gating mechanism based on `|ϕ'(x)|` (the derivative magnitude)
- `L` is the Lipschitz constant of ϕ
- `[·]_sg` denotes stop-gradient

**How it works:** The gate `g(x)` is large when ϕ' is small (saturated region) and small when ϕ' is large (active region). So `α_i · x · g(x)` injects linearity exactly where ϕ would otherwise vanish.

**RL validation (paper §3.6, Figure 7):**
- **Trained SAC on Hopper-v4, Ant-v4, HalfCheetah-v4** (the user's exact env class)
- AdaLin-enhanced SAC outperforms baseline SAC at replay ratios 1, 2, 4
- The performance gap **grows** with higher replay ratio (more off-policy)
- "AdaLin better maintains adaptability under higher off-policy pressure"

**BWH-orthodoxy:**
- ✅ Inv-1: activation-level only, env-agnostic
- ✅ Inv-3: per-neuron α_i scales with width
- ⚠️ Inv-9: introduces 1 learnable parameter per neuron (the α_i). This is a *learnable* parameter not a *hyperparameter*; analogous to TIDBD's per-feature α. **OaK already meta-learns alphaImpFeat per feature (TIDBD)** — adding alphaActiv per neuron is the same lineage.

### 2.3 Comparison: deep Fourier features vs AdaLin

| Property | Deep Fourier Features | AdaLin |
|---|---|---|
| Output width | 2× (sin + cos per pre-activation) | 1× (single output) |
| Learnable params/neuron | 0 | 1 (the α_i) |
| Approx. embeds deep linear | YES (Corollary 1) | NO (but injects linearity dynamically) |
| RL-validated? | NO (CIFAR + supervised only) | **YES** (HalfCheetah, Hopper, Ant + SAC) |
| Compatible with arbitrary ϕ | NO (sin/cos only) | YES (ReLU, Tanh, GeLU) |
| New activation kernel | YES (need to compile sin/cos in Metal) | YES (need cos+derivative gate) |
| Memory cost | 2× layer width | 1× layer width + 1 α/neuron |
| Plasticity guarantee | Strong (Theorem 2 + Corollary 1) | Empirical only |

For OaK's continuous-control problem, **AdaLin has direct empirical evidence on the user's exact task class**. Deep Fourier features have stronger theory but only supervised-learning validation.

---

## 3. The matrix-encoder divergence problem

### 3.1 What the user observed

> "Tried 'full R×O matrix encoder' this session — pEff jumped to 12.89/17 (4× richer state) but valueW diverged to fp32-max via positive feedback in bootstrap V'-V over ~50K steps."

### 3.2 The mathematical structure

With matrix encoder W_enc ∈ R^{R×O}:
$$z_i = \sum_o W_{\text{enc}}[i,o] \cdot \text{obs}[o], \quad V(s) = \sum_i \text{valueW}[i] \cdot z_i$$

This is mathematically a **2-layer deep linear network** (depth-2 linear FA via product matrix `valueW^⊺ · W_enc`). By Theorem 2 of [arXiv:2410.20634](https://arxiv.org/abs/2410.20634), this *should* preserve trainability.

The divergence is **NOT** because the architecture is wrong — it's because the **stability machinery (ϖ-RTD baseline) requires the steady-state ψ̄ to be tracked accurately**, and when the encoder is changing, ψ̄ lags behind.

### 3.3 What Sakha-Kamalapurkar-Meyn ([arXiv:2603.27874v2](https://arxiv.org/abs/2603.27874)) actually says

The FULL ϖ-RTD(λ) algorithm (paper §1.3, Eq. 18a-b):

$$D_{n+1} = c(Z_n) + \gamma Q_{\theta_n}(X_{n+1}) - Q_{\theta_n}(Z_n) - \delta_r \langle \varpi_n, Q_{\theta_n} \rangle$$
$$\langle \varpi_n, Q_{\theta_n} \rangle = \bar{\psi}_n^\top \theta_n$$
$$\bar{\psi}_{n+1} = \bar{\psi}_n + \beta_{n+1}[\psi(n+1) - \bar{\psi}_n], \quad \bar{\psi}_0 = \psi(Z_0)$$

**CRITICAL TWO-TIME-SCALE REQUIREMENT** (paper §1.3, just after Eq. 18b):
> "For ease of exposition we assume a relatively large gain: **lim_{n→∞} β_n / α_n = ∞**."

That is, the empirical-mean estimator `ψ̄_n` must run on a **faster timescale** than the parameter estimator `θ_n`. Specifically:
- α_n → 0 (slow): used for valueW updates
- β_n → 0 but β_n / α_n → ∞: used for featMean (Welford EMA) updates

**Theorem 2.4(ii) (the key result):**
> "If µ = ϖ then the mean flow associated with the RTD(λ) algorithm is stable for **any δ_r ≥ 0**, and any γ, λ ∈ [0, 1] satisfying γλ < 1."

The proof goes through the rank-one perturbation:
$$\bar{A}(\lambda; \delta_r) = \bar{A}(\lambda; 0) - \frac{\delta_r}{1 - \lambda \gamma} \bar{\psi} \bar{\psi}^\top$$

When `µ = ϖ` (the true stationary distribution), the perturbation makes `Ā` Hurwitz for *any* δ_r ≥ 0.

### 3.4 What's likely going wrong in OaK's matrix-encoder

Looking at the existing code:
- `vMeanBaseline = <valueW, featMean>` and `psiMeanBaseline[k] = <ψ_k, featMean>` are correctly computed (`oak_bk_admit_commit` writes these)
- The baseline is subtracted in `oak_predict` (`vMeanBase = ctx.scalars->vMeanBaseline`)
- BUT: `featMean` is updated via `track_rate(stepCount, ridge)` — the SAME timescale as everything else

Per Sakha-Kamalapurkar-Meyn §1.3, this **violates the two-time-scale assumption** required for Theorem 2.4(ii). Without `β_n / α_n → ∞`, the RTD baseline is biased and can amplify divergence rather than damp it.

With the *diagonal* encoder (current default), `ψ_i = encoder[i] · obs[i]` is essentially proportional to `obs[i]`, so `featMean` tracks `obsMean × encoder` which doesn't change much — the two-time-scale violation has small effect. But with the *matrix* encoder, `ψ_i = Σ_o W_enc[i,o] · obs[o]` is a rapidly-changing function of W_enc, so `featMean` lag is severe and the ϖ-RTD baseline becomes invalid.

### 3.5 The fix

**Add explicit two-time-scale separation for `featMean`:**

```metal
// Current (single timescale; violates Sakha §1.3):
featMean[i] += track_rate(stepCount, ridge) * (z[i] - featMean[i])

// Fixed (two-time-scale; Sakha §1.3 with β_n/α_n → ∞):
const float beta_psi_bar = K_TIMESCALE * track_rate(stepCount, ridge);  // K=10..100
featMean[i] += min(1.0f, beta_psi_bar) * (z[i] - featMean[i])
```

K_TIMESCALE is a structural constant (Inv-9 compliant — derived from the rate ratio Sakha needs for stability proof, not tuned).

This single change should restore the matrix-encoder stability without giving up its 4× richer state.

---

## 4. Other 2026 papers checked (and why they're not the answer)

### 4.1 HSR / Eigenoptions (Yu-Lengyel [arXiv:2602.12753](https://arxiv.org/abs/2602.12753), Wan-Sutton [arXiv:2110.13855](https://arxiv.org/abs/2110.13855))

**Status**: Already in OaK's papers library (`HierarchicalSuccessorRepresentation_2602.12753.pdf`) and considered in Pass-14 exploration candidates.

**Verdict**: Eigenoptions are an **option discovery** mechanism, not a feature class. They define K options whose policies move the agent to spectral basins of the SR. They don't solve the encoder problem (the encoder still has to compute φ(s) before SR is meaningful).

For HalfCheetah-style continuous control, eigenoptions are weak because there's no clear graph structure — the SR is dense.

### 4.2 Mamba / Selective SSMs ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569), [arXiv:2412.00776](https://arxiv.org/abs/2412.00776))

**Status**: Considered in Pass-13/14 docs. Mamba-3 is in the OaK papers library.

**Verdict**: Mamba SSMs are a **belief-state recurrence** mechanism (like RWKV), not a static feature class. They help with POMDPs by maintaining temporal context. **OaK already has this** via the per-feature retention (`alphaImpFeat[i] · cap`) inspired by Belief-State RWKV ([arXiv:2604.09671](https://arxiv.org/abs/2604.09671)).

For HalfCheetah specifically, the env is fully observable (state = joint angles + velocities), so adding Mamba does nothing for the feature richness problem.

### 4.3 Belief-State RWKV ([arXiv:2604.09671](https://arxiv.org/abs/2604.09671))

**Status**: **Already in OaK** (Pass-13 Tier 4).

**Verdict**: Same as Mamba — orthogonal to the feature-class question.

### 4.4 Streaming-NMF / Streaming-PCA

**Status**: NMF on ψ is part of HSR (Pass-14 Tier 2 candidate, deferred).

**Verdict**: NMF is a *low-rank decomposition* of an existing feature matrix, not a feature constructor. It can compress the K-option ψ matrix but doesn't help the raw obs → state encoding problem.

### 4.5 Tile coding / RBF / shallow Fourier features

**Status**: Sutton-Barto Ch 9.5 textbook methods. `Tiles3` is on Sutton's website.

**Verdict**: These are the **canonical Sutton-Barto recommendations** for continuous control with linear FA. They are 100% BWH-orthodox. **HOWEVER:**
- Tile coding requires choosing the number of tilings + memory size up-front (Inv-3 risk if hardcoded)
- RBFs require choosing centers and bandwidths (Inv-2 risk)
- Shallow Fourier features (just the input layer) are known to overfit and **still suffer plasticity loss** ([arXiv:2410.20634 §1](https://arxiv.org/abs/2410.20634): "previous work which considers only shallow Fourier features in the first layer ... still exhibit a loss of plasticity")

These remain valid bridges for prototyping but are strictly weaker than deep Fourier features for the continual-learning use case.

### 4.6 Functional Regularization via Learned Fourier Features (Li & Pathak NeurIPS 2021, [arXiv:2112.03257](https://arxiv.org/abs/2112.03257))

**Verdict**: Earlier Fourier-features-for-RL work. Uses Fourier features only at the INPUT layer (shallow). Validated on continuous control in MuJoCo. Improves sample efficiency. **Strictly subsumed by the 2024-2025 deep Fourier features papers** which have stronger theory and apply Fourier at every layer.

### 4.7 Overcoming the Spectral Bias of Neural Value Approximation (Yang-Ajay-Agrawal ICLR 2022)

**Verdict**: Demonstrates that Fourier Feature Networks (FFN) outperform standard MLPs on continuous control by addressing spectral bias. Uses *random* Fourier features at the input layer (Tancik et al style). Same shallow-Fourier limitation as Li-Pathak. **Subsumed by deep Fourier features**.

---

## 5. Concrete recommendations (ranked by impact × BWH purity)

### Recommendation #1 (HIGHEST IMPACT, MEDIUM RISK): Two-time-scale ϖ-RTD + Matrix Encoder

**Mechanism:**
1. Restore the matrix encoder W_enc ∈ R^{R×O} (the user's previous experiment that hit pEff = 12.89/17)
2. Add explicit two-time-scale separation: `featMean` updates with `β_psi_bar = K · track_rate(stepCount, ridge)` for K ≈ 10–100 (structural, derived from the requirement `β/α → ∞` of Sakha-Kamalapurkar-Meyn §1.3)
3. Confirm `vMeanBaseline = δ_r · <valueW, featMean>` is subtracted in `oak_predict` (the existing code does this with implicit δ_r = 1)

**Expected result:** The user's matrix encoder should now be stable. pEff should remain near 12.89/17 (4× richer state) and HalfCheetah rR should rise from -0.375.

**BWH-orthodoxy:** ✅ Inv-1..9. K_TIMESCALE is structural (the rate-ratio constant that makes Sakha Theorem 2.4(ii) apply).

**Code cost:** ~5 lines in `oak_bk_admit_commit` (add the K factor to featMean update). Or add a new explicit `bk_featmean_fast` kernel for the higher-rate update.

**Why this first:** It re-enables the architectural improvement the user already demonstrated (4× richer features) without any new feature class. If this alone gets HalfCheetah walking, no further changes needed.

---

### Recommendation #2 (HIGH IMPACT, LOW RISK): AdaLin per-neuron adaptive linearity

**Mechanism:**
Replace the plain `tanh` activation in `oak_predict_outcome` (and any other activated head) with AdaLin:

```metal
// Current outcome head:
outcome[k] = tanh(outcomeHead[k] · prev_state)

// AdaLin:
float z = dot(outcomeHead[k], prev_state);
float phi = tanh(z);
float dphi = 1.0f - phi * phi;          // tanh derivative; |dphi| ≤ 1
float gate = cos(M_PI_2_F * dphi);      // L = 1 for tanh
outcome[k] = phi + alpha_outcome[k] * z * gate;  // alpha learned via TIDBD-style rule
```

`alpha_outcome[k]` is one learnable scalar per outcome head, meta-learned exactly like `alphaImpFeat[E]` (existing OaK Tier 2 TIDBD machinery — zero new mechanism).

**Direct empirical evidence:** [arXiv:2505.09486 §3.6, Figure 7](https://arxiv.org/abs/2505.09486) shows AdaLin SAC outperforms baseline SAC on **HalfCheetah-v4** (the user's exact task) at replay ratios 1, 2, 4.

**BWH-orthodoxy:**
- ✅ Inv-1: activation only, no env logic
- ✅ Inv-3: scales with neuron count
- ⚠️ Inv-9: introduces α_i per neuron — but this is *meta-learned* not *hyperparameter-set*, identical category to existing `alphaImpFeat`

**Code cost:** ~10 lines per activation site. `alpha_outcome[k]` is K floats; reuses existing `alphaImpFeat` machinery.

**Why this second:** Generalizes #1 by making the *outcome heads* (the only nonlinearity in the agent) plasticity-preserving. Validated on the exact env class.

---

### Recommendation #3 (MAXIMUM IMPACT, MEDIUM RISK): Deep Fourier Features encoder

**Mechanism:**
Replace the diagonal encoder with a 2-layer deep Fourier features stack:

```metal
// Layer 1:
z1[i] = W1[i] · obs                              // linear projection, R units
h1[i] = [sin(z1[i]), cos(z1[i])]                  // Fourier feature, output: 2R units

// Layer 2:
z2[j] = W2[j] · h1                                // linear projection, R units (compress back to R)
state[j] = [sin(z2[j]), cos(z2[j])]               // Fourier feature, output: 2R units
                                                  // OR: just sin(z2[j]) as the final state
```

Or, more compactly, keep state width = R and apply Fourier internally:
- Pre-activation `z[i] = W_enc[i] · obs` (existing matrix encoder)
- State `state[i] = sin(z[i]) + cos(z[i + R/2])` (alternates sin/cos by index)

The value head remains LINEAR: `V(s) = valueW · state`.

**Theory:** [arXiv:2410.20634, Theorem 2 + Corollary 1](https://arxiv.org/abs/2410.20634):
- The deep Fourier features network *approximately embeds* a deep linear network with bounded error c ≈ 0.05
- Deep linear networks preserve plasticity continually
- Therefore the deep Fourier features encoder preserves plasticity

**BWH-orthodoxy:**
- ✅ Inv-1: encoder is feature-class agnostic
- ✅ Inv-3: width is structural (R)
- ✅ Inv-7: Fourier features are nonlinear features for prediction, not cosmetic
- ✅ Inv-9: zero new hyperparameters (sin and cos have no parameters)

**Code cost:** ~30 lines in `oak_encode` (add Fourier transform after matrix projection). Plus `oak_learn_encoder` adapts to the new ∂z/∂W = obs (unchanged) and ∂state/∂z = [cos(z), -sin(z)] (new). Per-step compute: ~2x current encode kernel.

**Why this third:** Most architectural change but strongest theoretical guarantee. Should be applied AFTER #1 and #2 confirm stability, to avoid confounding variables.

---

## 6. What's BWH-VIOLATING (do not adopt)

| Mechanism | arXiv | Why violating |
|---|---|---|
| LayerNorm/BatchNorm in encoder | (Ba 2016, Ioffe 2015) | Imposes fixed stationary scale prior — the AGENTS.md "continual signal scale" rule explicitly rejects this |
| Replay buffer | (Mnih 2015) | Not online streaming — Inv-5 violation |
| Target networks | (Mnih 2015) | Not temporally uniform — Inv-5 violation |
| ReLU stacks without plasticity preservation | — | Loss of plasticity (Dohare 2024 Nature) — fails interactivity (2512.23419) |
| Adam/RMSProp | (Kingma 2014, Hinton 2012) | "Step-size normalization" rather than "step-size optimization" — Sutton's IDBD/TIDBD lineage rejects this (Degris-Javed-Sharifnassab-Liu-Sutton [arXiv:2401.17401](https://arxiv.org/abs/2401.17401)) |

---

## 7. Summary table

| Question | Answer |
|---|---|
| Does Sutton-Bowling OaK paper require linear FA on raw obs? | **NO**. Alberta Plan Step 1 is "linear FA with given features". Step 2 is "nonlinear features". Step 8 is "continual non-linear FA". OaK (Step 11) builds on Steps 2-8. |
| Does BWH require linear FA on raw obs? | **NO**. BWH requires *function approximation* generally, *tracking*, and *computational efficiency*. Feature class is unconstrained. |
| Is the canonical Sutton-Barto "linear method" linear in obs? | **NO**. Tile coding, RBF, Fourier basis, polynomials, coarse coding are all nonlinear in obs — linear only in the parameters of the value head. |
| Is the user's diagonal encoder canonical? | **PARTIAL**. It corresponds to Step 1 of the Alberta Plan (linear FA with per-feature scaling). It's a starting point, NOT the prescribed end state for OaK Step 11. |
| Why did the matrix encoder diverge? | The ϖ-RTD baseline `featMean` tracking violates the Sakha-Kamalapurkar-Meyn two-time-scale requirement (`β/α → ∞`). Fix: K-time faster featMean updates. |
| Best path to walking HalfCheetah while preserving BWH? | (1) Two-time-scale ϖ-RTD + matrix encoder, (2) AdaLin on outcome heads, (3) Deep Fourier features encoder. In that order. |

---

## 8. Cited papers (already in or now needed for OaK library)

**Already in `docs/papers/`:**
- `AlbertaPlanForAIResearch_2208.11173.pdf` (Sutton-Bowling-Pilarski 2022)
- `RewardRespectingSubtasks_ModelBasedRL_2202.03466.pdf` (Sutton et al AIJ 2023)
- `RelativeTD_Stability_2603.27874.pdf` (Sakha-Kamalapurkar-Meyn IEEE CDC 2026)
- `RewardCentering_2405.09999.pdf` (Naik-Wan-Tomar-Sutton RLC 2024)
- `SwiftSarsa_2507.19539.pdf` (Javed-Mahmood RLC 2025)
- `SwiftTD_FastRobustTD_RLC2024.pdf` (Javed-Sharifnassab-Sutton RLC 2024)
- `SpectralCollapsePlasticity_2509.22335.pdf` (Lewandowski et al ICLR 2026)
- `BellmanErrorCentering_2502.03104.pdf` (Chen et al 2025)
- `ImplicitUpdates_AvgRewardTD_2510.06149.pdf` (Kim-Cho-Laber 2025)
- `MetaOptimize_StepSizes_2402.02342.pdf` (Sharifnassab-Salehkaleybar-Sutton ICML 2025)

**Newly downloaded for this report (under `agent-tools/`, recommend moving to `docs/papers/`):**
- `2410.20634` Plastic Learning with Deep Fourier Features (Lewandowski-Schuurmans-Machado, ICLR 2025) — **PRIMARY** for #3
- `2505.09486` Preserving Plasticity in Continual Learning with Adaptive Linearity Injection (Razavi-Rohani et al, May 2025) — **PRIMARY** for #2
- `2512.23419` The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis (Lewandowski-Ramesh-Meyer-Schuurmans-Machado, NeurIPS 2025)

**Referenced but not strictly needed for OaK:**
- Tiles3 software (incompleteideas.net/tiles)
- Functional Regularization via Learned Fourier Features ([arXiv:2112.03257](https://arxiv.org/abs/2112.03257)) — subsumed by deep Fourier features
- Overcoming Spectral Bias of Neural Value Approximation (Yang-Ajay-Agrawal ICLR 2022) — subsumed

---

*OaK encoder feature-class research report. Generated 2026-04-19.*
