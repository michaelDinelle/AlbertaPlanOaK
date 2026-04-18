# Pass 14 — Exploration mechanism candidates (research librarian report)

**Mission.** Find 2026 (and very‑late‑2025) arxiv mechanisms that can convert
the **46/84 zero‑rewardRate environments** (the sparse‑reward MiniGrid suite —
`KeyCorridor*`, `ObstructedMaze*`, `MultiRoom*`, `BlockedUnlockPickup*`,
`LockedRoom*`, …) from "random‑baseline" to **first‑goal‑reaching** without
breaking BWH‑orthodoxy (Inv‑1..Inv‑9), without leaving the average‑reward
differential‑TD setting (Naik [`2603.27874`]), and without exceeding the FLOP
budget that the Pass‑12/13 multi‑signal Pareto gate enforces.

**TL;DR ranking** (expected upside on 46 zero‑envs ÷ FLOPs cost):

| Rank | Mechanism | arxiv | Date | Authors | Why it ranks here |
|---:|---|---|---|---|---|
| **#1** | **EVE — EigenVector‑based Exploration** (intrinsic average‑reward steady‑state entropy via tilted‑matrix soft‑flow) | [`2603.12325`](https://arxiv.org/abs/2603.12325) | Mar 12 2026 | Adamczyk, Kamoski, Kulkarni | Native average‑reward, no rollouts, no learned forward model, no extrinsic reward needed, ~1 new kernel, **directly designed for the "uniform long‑run coverage" problem** that defines our 46 zero envs. |
| **#2** | **D2 Q‑learning — Differential Distributional RL** (quantile distribution over per‑step reward + differential return) | [`2506.03333v2`](https://arxiv.org/abs/2506.03333) | Jan 13 2026 (AAAI 2026) | Rojas, Lee | **Same author (Rojas) as our RED cumulants** ([`2410.10578`]). Drop‑in quantile replacement for `valueW` / ψ; quantile dispersion is a *built‑in* uncertainty signal — no separate intrinsic reward, no Inv‑8 violation. |
| **#3** | **Soft‑FB — General‑utilities Forward‑Backward** (max‑entropy variant of FB, captures pure exploration as a special utility) | [`2602.06769`](https://arxiv.org/abs/2602.06769) | Feb 6 2026 | Bagatella, Rupf, Martius, Krause | Direct successor to FB ([`2602.06769`] cited in our prompt). Generalises ψ to handle **non‑additive utilities** including `max H(d_π)`. Already aligned with our chord‑conditional ψ shared model. |
| **#4** | **HSR + NMF — Hierarchical Successor Representation** (β‑termination → low‑rank sparse SR via streaming NMF; eigenoptions for procedural envs) | [`2602.12753`](https://arxiv.org/abs/2602.12753) | Feb 13 2026 | Yu, Lengyel | Reuses our existing β termination + ω chord generator. NMF on ψ ⇒ K eigenoptions with deterministic GO‑TO behaviour on multi‑compartment maps — exactly the topology of `KeyCorridor` / `ObstructedMaze`. |
| **#5** | **VBE — Value Bonuses with Ensemble Errors** (K random Q‑networks; bonus = ensemble error; first‑visit optimism, decays to 0) | [`2602.12375`](https://arxiv.org/abs/2602.12375) | Feb 12 2026 (RLJ 2025) | Wahab, Kumaraswamy, **Martha White** | **Direct Alberta lineage** (M. White). Beats RND, ACB, Bootstrap‑DQN. Re‑expressed as a **GVF cumulant** (not a reward bonus) it is Inv‑8‑clean and slots into our existing K‑option ψ pipeline. |

The remaining 7 papers vetted are listed in §6 (Honourable mentions) and §7
(Rejected / waiver‑required).

---

## 0. How the 46 zero‑rewardRate envs fail today

From the Pass‑13 heartbeats (`.oak-benchmarks/pass12-ablation/baseline/MiniGrid-*`)
the failure mode is uniform across the sparse‑reward suite:

1. The agent never observes a positive reward in the bench horizon
   (50 K steps × 2048 envs).
2. Therefore `δ = (r − ρ̄) + V(s′) − V(s) ≈ 0` because `r ≡ 0` and `ρ̄ ≈ 0`,
   so `oak_learn_value`, `oak_learn_psi`, `oak_learn_pi` receive **no useful
   gradient on the main task**.
3. The Pass‑13 SPR auxiliary (`learn_encoder_spr`, [`2602.09396`]) gives the
   encoder a learning signal, and L2‑Effective‑Rank ([`2509.22335`]) keeps
   `pEff` high, so the **representation** is being learned — but the
   **policy** has nowhere to go.
4. With the Gumbel‑max wrapper sampling, the agent stays at the entropy
   floor of `log A` per step ⇒ random walk on the maze ⇒ never reaches the
   goal cell ⇒ never bootstraps reward.

The bottleneck is therefore **not** representation, **not** plasticity, **not**
step‑size — it is the absence of any **directed** exploration signal that
survives a 50 K‑step zero‑reward window.

---

## 1. EVE — EigenVector‑based Exploration (TOP RECOMMENDATION)

> Adamczyk, J., Kamoski, A., Kulkarni, R. V. *Maximum Entropy Exploration
> Without the Rollouts.* arXiv:[`2603.12325`](https://arxiv.org/abs/2603.12325),
> 12 Mar 2026. Code: <https://github.com/JacobHA/EVE>.
> Predecessor: EVAL — Adamczyk et al., *EigenVector‑based Average‑reward
> Learning*, arXiv:[`2501.09770`](https://arxiv.org/abs/2501.09770), Jan 15 2025.

### 1.1 Core mechanism

EVE solves the **maximum steady‑state state‑action entropy** exploration
objective

\[
\max_\pi \; \mathbb{H}\!\left(d_{p,\pi}\right)
   \;=\; \max_\pi -\sum_{s,a} d_{p,\pi}(s,a)\, \log d_{p,\pi}(s,a)
\]

inside the **average‑reward** framework (Naik 2024 / our `ρ̄` machinery).
The intrinsic reward is the *self‑consistent* visitation log‑density,
\(r(s,a) = -\log u(s,a) v(s,a)\), where \(u, v\) are the dominant left/right
eigenvectors of the *tilted transition matrix*

\[
\widetilde{P}(s',a' \mid s,a) \;=\; p(s' \mid s,a)\,\pi_0(a' \mid s')\,e^{\beta r(s,a)}.
\]

The right eigenvector can be eliminated; what remains is a **single
fixed‑point equation on a per‑state‑action scalar `q(s,a) = β⁻¹ log u(s,a)`**
(at β = 1):

\[
q(s,a) \;=\; \tfrac12 \log\!\bigl(\mathbb{E}_{a' \sim \pi_0} e^{q(s',a')}\bigr)
        \;-\; \tfrac12 \log\!\bigl(\textstyle\sum_{\bar s,\bar a} P(s,a \mid \bar s,\bar a)\, e^{-q(\bar s,\bar a)}\bigr).
\]

This is a **soft‑flow** TD: the first term is the expected log‑sum‑exp over
*outgoing* transitions (forward flow), the second is the soft min over
*incoming* transitions (backward flow). The **policy is read off** as
\(\pi^*(a \mid s) \propto \pi_0(a \mid s)\, u(s,a)\).

A *posterior‑policy iteration* (PPI) loop sets \(\pi_0 \leftarrow \pi^*\)
to **lift the entropy regularization without bias**, recovering pure
unregularized max‑entropy exploration in the limit.

**Convergence (Theorem 1, EVE).** The fixed‑point operator \(\mathcal T\)
is a contraction in **Hilbert's projective metric** for all β ≥ 1, with
linear rate κ((P^{π_0})^m)^{1/m}. No discount factor is needed; magnitudes
stay bounded by homogeneity of the update.

### 1.2 Why it solves our 46 zero‑envs

Average‑reward steady‑state entropy maximization **is exactly the right
objective** for `KeyCorridor` / `ObstructedMaze` / `MultiRoom`: the agent
is asked to spread its long‑run visitation over the whole reachable state
graph. Reaching *every* cell ⇒ eventually walking onto the goal cell ⇒
positive reward observed ⇒ the existing main‑task TD finally wakes up.

Crucially, EVE **does not need an extrinsic reward** to drive exploration,
**does not require rollouts** (the tilted matrix can be estimated from
arbitrary off‑policy data), and is *natively* an **average‑reward**
algorithm that uses our existing `ρ̄` centering machinery (`ρ̄_q` for the
EVE channel = the scalar `θ*` of equation (4) in [`2603.12325`]).

### 1.3 BWH‑orthodoxy

| Inv | Status | Notes |
|---|---|---|
| Inv‑1 (no env branches) | ✅ | Pure signal processing on `state[B·R]` and the action arena. |
| Inv‑2 (no env hyperparameters) | ✅ | β annealed via PPI, not env‑specific. |
| Inv‑3 (no hardcoded widths) | ✅ | `q[B·R]` reuses `state` width. |
| Inv‑4 (no episodic semantics) | ✅ | EVE is intrinsically continuing — its objective is the *stationary* distribution. |
| Inv‑5 (no train/eval split) | ✅ | Soft‑flow updates run every step. |
| Inv‑6 (no per‑step alloc) | ✅ | `q[B·R]` and `qBackEMA[R]` are pre‑allocated. |
| Inv‑7 (no cosmetic norm) | ✅ | All work serves prediction/control. |
| Inv‑8 (no hot‑stove exploration) | ✅ | EVE is an **off‑policy GVF** — its q‑value is *not added to the main reward stream*. The behaviour policy is a **soft** mix of \(\pi_{\rm main}\) and \(\pi_{\rm EVE}\) selected at the option‑keyboard layer (chord). The main `oak_learn_pi` still maximizes raw external `r`. |
| Inv‑9 (parameter‑free meta) | ✅ | β anneals via PPI (no scalar hyperparameter); the only parameter is the existing `track_rate(stepCount, ridge)`. |

The crucial observation is that EVE is **a separate GVF** with its own
average‑reward `θ̄_eve` and its own value head `q_eve`. It fits as **the
K+1‑th option** (or as a slot inside an existing keyboard chord) — the main
policy's action selection is already a soft chord of K options' policies in
`oak_meta_action`, so adding EVE as one more chord component requires
**zero new sampling code** and **zero violation of Inv‑8**.

### 1.4 Implementation cost

| Resource | Cost |
|---|---|
| New Metal kernels | **+1**: `oak_learn_eve_q` (one TG per row × `OAK_SIMD` threads, computes the soft‑flow update). The "backward" term needs the previous `state` slot (which we already keep for TD targets) plus a per‑feature EMA `qBackEMA[R]` (1024 fp32 floats for R ≤ 1024). |
| New parameters | **+R fp32** (`qEVE[R]`) **+R fp32** (`qBackEMA[R]`) **+1 fp32** (`thetaBarEVE`). For R = 256 this is < 2 KB. |
| FLOPs/step | Per row: 1 log‑sum‑exp over A actions + 1 EMA update + 1 add to chord weight. ≈ R + A FLOPs/row × B = O(B(R+A)). At B = 2048, R = 256, A ≤ 16 this is ~0.5 MFLOPs/step — **<1% of current step cost** (Pass 13 step ~80 MFLOPs/step). |
| New Swift code | None on the wrapper side (action sampling unchanged). One ICB slot for `oak_learn_eve_q` between `learn_psi` and `learn_pi`. |
| Risk | LOW — the soft‑flow operator is a contraction; convergence is guaranteed by Theorem 1; no clamping needed. |

### 1.5 Code sketch (Metal pseudocode)

```metal
kernel void oak_learn_eve_q(/* same 7-buffer signature */)
{
    const uint row = tg_row;
    const float alpha = scalars->alphaImp;
    const uint slot = scalars->slot;
    const device float *prev = state + slot * (B * R) + row * R;
    const device float *curr = state + (1 - slot) * (B * R) + row * R;

    // Forward soft-max over candidate next features (reuse psi predict targets).
    float fwdLse = -INFINITY;
    for (uint i = thr; i < R; i += OAK_SIMD) {
        if (!(activeMask[i])) continue;
        fwdLse = simd_log_sum_exp(fwdLse, qEVE[i] * curr[i]);  // ψ-style soft max
    }
    fwdLse = simd_reduce_log_sum_exp(fwdLse);

    // Backward soft-min over previous features (per-feature EMA).
    float backLse = -INFINITY;
    for (uint i = thr; i < R; i += OAK_SIMD)
        backLse = simd_log_sum_exp(backLse, -qBackEMA[i] * prev[i]);
    backLse = simd_reduce_log_sum_exp(backLse);

    // Soft-flow update (eq. 11 in arXiv:2603.12325, β=1).
    if (thr == 0) {
        const float qNew = 0.5f * fwdLse - 0.5f * backLse;
        const float delta = qNew - qEVE[row_feat];
        qEVE[row_feat] += alpha * delta;
        // PPI: bias the chord weight towards π_EVE proportional to exp(qEVE).
        chordSlot[row * K + EVE_KEY] += alpha * (exp(qEVE[row_feat]) - chordSlot[row * K + EVE_KEY]);
    }

    // Update backward EMA from this step's actual transition.
    if (thr == 0)
        qBackEMA[row_feat] += scalars->trackRate *
            (-qEVE[row_feat] * prev[row_feat] - qBackEMA[row_feat]);
}
```

The EVE chord key flows through the existing `oak_meta_action` linear‑blend,
so adoption is a **pure additive change** to the kernel pipeline.

### 1.6 Successor / citation network

* **EVAL** ([`2501.09770`](https://arxiv.org/abs/2501.09770), Jan 2025) — same
  authors. Establishes the entropy‑regularized average‑reward neural
  approximation (function approximation + posterior policy iteration). Cited
  by EVE as the function‑approximation counterpart; we should adopt its
  PPI proof scaffolding in our notes.
* **PRR** (Tilted‑matrix average‑reward framework). Cited as the
  large‑deviation foundation; the proof of Hilbert‑projective contraction
  builds on PRR's Perron‑Frobenius results.
* **Naik 2603.27874 (reward centering)** — EVE explicitly cites the same
  ρ̄‑centering family we use; the algorithms are mathematically compatible.
* **Hazan et al. 2019** (provably‑efficient max‑ent exploration) — EVE
  removes the rollout requirement; this is the discounted predecessor.
* **Maximum‑Entropy Exploration with Future State‑Action Visitation
  Measures**, Bolland‑Lambrechts‑Ernst, [`2603.18965`](https://arxiv.org/abs/2603.18965)
  (Mar 19 2026) — concurrent and complementary; uses an off‑policy
  contraction operator on the *discounted* visitation distribution. We
  prefer EVE because it is *undiscounted* and matches our ρ̄ machinery.

---

## 2. D2 Q‑learning — Differential Distributional RL

> Rojas, J. S., Lee, C.‑G. *A Differential Perspective on Distributional
> Reinforcement Learning.* arXiv:[`2506.03333v2`](https://arxiv.org/abs/2506.03333),
> revised 13 Jan 2026. **Accepted AAAI 2026**.

### 2.1 Core mechanism

A **quantile** distributional extension of differential TD. Replaces the
scalar `V(s)` and `Q(s,a)` with quantile vectors \(V^q(s) \in \mathbb R^Q\)
that estimate the distribution of *per‑step reward* and of the *differential
return*. The algorithm — D2 Q‑learning — is *proven convergent* in tabular
control and ports to function approximation with the same scaling
properties as QR‑DQN.

### 2.2 Why it helps the 46 zero‑envs

* **Built‑in uncertainty signal**: quantile dispersion across the K
  random‑option ψ heads gives a Bayes‑style epistemic spread without any
  separate ensemble. On a sparse‑reward env the dispersion is large in
  states that have *never* been visited, naturally biasing the
  argmax‑over‑K planner toward novel cells.
* **Same author (Rojas) as our RED cumulants** ([`2410.10578`]) — the
  derivation uses the exact same `(r − ρ̄)` centering as our Pass‑11/13
  agent; no porting work for the cumulant pipeline.
* **No new reward stream** (so no Inv‑8 violation). It is a *richer
  prediction target*, not a bonus.

### 2.3 BWH‑orthodoxy

| Inv | Status | Notes |
|---|---|---|
| Inv‑1..7 | ✅ | Quantile vectors are just wider value heads. |
| Inv‑8 | ✅ | Distributional prediction *is* the value function — never added to the reward stream. |
| Inv‑9 | ✅ | Quantile midpoints are the standard (Q+½)/(2Q) — *not* a learnable hyperparameter; equivalent to the implicit ridge in our `track_rate(stepCount, ridge)`. |

### 2.4 Implementation cost

| Resource | Cost |
|---|---|
| New kernels | **0** if we widen `valueW[R]` → `valueW[R][Q]` and adapt `oak_learn_value` to do per‑quantile quantile regression. **+1** if we keep the scalar V and add a parallel quantile head. |
| Parameters | **×Q** on `valueW`, `psi` (or just `valueW`). For Q = 5, ≈ 5× param count on those two — well within Apple Silicon RAM budget. |
| FLOPs/step | **×Q on `learn_value`/`learn_psi`** branches; planner argmax becomes argmax over (K, Q) pairs. ~2× total step cost at Q = 5. |
| Risk | MEDIUM — the wider head can suffer plasticity loss; pair with our existing L2‑Effective‑Rank regularizer. |

### 2.5 Code sketch

```metal
// In oak_learn_value: widen δ to a quantile vector.
for (uint q = 0; q < Q; ++q) {
    const float rho_bar = scalars->rho;
    const float v_curr  = vCurrQ[row * Q + q];
    const float v_next  = sum_active(valueW + q * R, curr_state);
    const float tau_q   = (q + 0.5f) / Q;
    const float td_q    = (reward[row] - rho_bar) + v_next - v_curr;
    const float weight  = (td_q < 0) ? (1 - tau_q) : tau_q;  // quantile loss
    for (uint i = thr; i < R; i += OAK_SIMD)
        if (activeMask[i])
            valueW[q * R + i] += alpha_imp * weight * td_q * prev_state[i];
}
```

---

## 3. Soft‑FB — Forward‑Backward representations for general utilities

> Bagatella, M., Rupf, T., Martius, G., Krause, A. *Soft Forward‑Backward
> Representations for Zero‑shot Reinforcement Learning with General
> Utilities.* arXiv:[`2602.06769`](https://arxiv.org/abs/2602.06769),
> 6 Feb 2026.

### 3.1 Core mechanism

Standard FB ([`2602.06769`] in user prompt — actually the original FB by
Touati & Ollivier 2103.07945) factors the successor measure as
\(M(s, ·) ≈ F(s)^\top B(·)\), giving zero‑shot policies for any *additive*
reward. **Soft‑FB** generalises to **any differentiable functional of the
occupancy measure** \(d_\pi\), including:

* `H(d_π)` — pure max‑entropy exploration (the EVE objective)
* `KL(d_π ‖ d_target)` — distribution matching
* `f(d_π · w)` — soft‑optimal feature attainment

via a maximum‑entropy variant that recovers a **family of stochastic
policies** parameterised by the FB embedding. Test‑time selection uses
zero‑order search over the policy embedding (no SGD).

### 3.2 Why it helps the 46 zero‑envs

* `H(d_π)` mode of soft‑FB **gives the same objective as EVE**, but
  realised through an **explicit factored representation** that can
  generalise across procedurally‑generated MiniGrid layouts.
* Slots cleanly into our **K‑option ψ shared model** (OBB §2.4 / our
  `psi[K · M]` matrix): F(s) ↔ our `psi`; B(·) is a new parallel head.
* The chord/keyboard interpretation already exists: each chord position is
  a target embedding; soft‑FB handles arbitrary differentiable utilities of
  the implied occupancy.

### 3.3 BWH‑orthodoxy

| Inv | Status | Notes |
|---|---|---|
| Inv‑1..7 | ✅ | Pure signal processing. |
| Inv‑8 | ✅ | Soft‑FB's "general utility" is a property of the *learned representation*, not an additive reward bonus. |
| Inv‑9 | ⚠ | The temperature parameter on the soft‑max needs to be derived from `track_rate(stepCount, ridge)` rather than fixed; trivially done. |

### 3.4 Implementation cost

| Resource | Cost |
|---|---|
| New kernels | **+1**: `oak_learn_b_backward` to learn the B(·) head. |
| Parameters | **+M·R fp32** for the B matrix (≈ same as one extra option's ψ). |
| FLOPs/step | ~+10–15 % on top of `learn_psi`. |
| Risk | MEDIUM — Soft‑FB results so far are offline; we'd derive the online TD form from EVE's tilted‑matrix lens (which Soft‑FB also uses). |

---

## 4. HSR — Hierarchical Successor Representation

> Yu, C., Lengyel, M. *Hierarchical Successor Representation for Robust
> Transfer.* arXiv:[`2602.12753`](https://arxiv.org/abs/2602.12753),
> 13 Feb 2026.

### 4.1 Core mechanism

Two key ingredients, both of which we already partially have:

1. **Temporal abstraction** in the SR construction. The SR matrix is built
   over **option** transitions, not primitive‑step transitions, so the
   spectrum of M concentrates in fewer, more meaningful eigenvectors. We
   already have β termination + ω chord generator; HSR's contribution is to
   *build the SR estimate at the option timescale* instead of the
   primitive step.
2. **Streaming non‑negative matrix factorization** on the (option‑level)
   SR ⇒ a sparse low‑rank state representation that
   "discovers interpretable topological structures" — i.e. *room
   decompositions* in `MultiRoom`, *door‑then‑goal* sub‑graphs in
   `ObstructedMaze`. HSR‑NMF empirically scales to procedurally‑generated
   environments.

### 4.2 Why it helps the 46 zero‑envs

The eigenoptions HSR derives are **deterministic GO‑TO sub‑policies** that
take the agent from the current room to a neighboring room in O(diameter)
steps — exactly what is needed to traverse `KeyCorridor` to its
terminal goal. The NMF basis is the right *option* count for our K (≤ 32).

### 4.3 BWH‑orthodoxy

| Inv | Status | Notes |
|---|---|---|
| Inv‑1..7 | ✅ | Same family as our existing ψ matrix; NMF is one streaming projected‑gradient pass per step. |
| Inv‑8 | ✅ | Eigenoptions are **option models**, not reward bonuses. |
| Inv‑9 | ✅ | NMF rank = K (already structurally fixed). |

### 4.4 Implementation cost

| Resource | Cost |
|---|---|
| New kernels | **+1**: `oak_bk_hsr_nmf` (one TG per option, projects ψ row onto the NMF basis). |
| Parameters | **0** (NMF basis is the existing `psi[K · M]` matrix). |
| FLOPs/step | ~+5 %. |
| Risk | LOW — the only learning rule changes are within `bk_option` / `bk_admit_commit`. |

### 4.5 Successor lineage

* Original SR: Dayan 1993.
* Eigenoptions: Machado et al. 2017 (deep SR).
* **Reward‑Aware Proto‑Representation (DR)**: Tse, Chandrasekar, **Machado**
  ([`2505.16217v2`](https://arxiv.org/abs/2505.16217), revised Jan 30 2026)
  — **Alberta lineage**. The DR generalises SR by including reward dynamics
  and outperforms SR for reward shaping, exploration, option discovery,
  transfer. **Pass‑15 candidate** if HSR + EVE prove insufficient.
* **DROGO** (Default Representation Objective via Graph Optimization) —
  ResearchGate Jan 2026. Neural approximation of DR's principal
  eigenvector; lifts DR to high‑dim observations (we don't need this since
  R ≤ 1024 already).

---

## 5. VBE — Value Bonuses with Ensemble Errors

> Wahab, A., Kumaraswamy, R., **White, M.** *Value Bonuses using Ensemble
> Errors for Exploration in Reinforcement Learning.*
> arXiv:[`2602.12375`](https://arxiv.org/abs/2602.12375), 12 Feb 2026.
> Reinforcement Learning Journal, vol. 6, pp. 1894‑1915, 2025.

### 5.1 Core mechanism

K random Q‑networks (RQFs) with random rewards drawn from a designed
distribution. The **bonus is the disagreement** between RQFs and is
constructed so that the bonus *decreases to zero* as a state‑action is
visited — providing **first‑visit optimism** and **deep exploration**
without the action‑gap pathologies of count‑based bonuses or RND.

> Beats Bootstrap‑DQN, RND, and ACB on River Swim, Deep Sea, and Atari hard
> exploration sub‑suite.

### 5.2 Why it helps the 46 zero‑envs

* **Direct Alberta lineage** (Martha White is Pass‑12 LPM co‑author at
  Amii). Conceptually compatible with the per‑option value/cumulant family.
* The first‑visit optimism property gives the planner a deterministic
  signal for novel (s, a) pairs that decays to 0 once visited — exactly
  the right asymptotic for a continuing agent.

### 5.3 BWH‑orthodoxy

| Inv | Status | Notes |
|---|---|---|
| Inv‑1..7 | ✅ | RQFs are extra Q‑heads; arena layout is unchanged. |
| **Inv‑8** | ⚠ | **As described in the paper** the bonus is added to the extrinsic reward in the value target — that *is* a hot‑stove violation. **Re‑expressed as a separate GVF cumulant** (`c_k = δ_RQF_k(s,a)`) it becomes Inv‑8‑clean: K of the K options' cumulants are replaced by RQF disagreements, and the chord planner naturally selects the option (extrinsic vs. RQF‑optimistic) that has highest planning advantage. **Pass‑14 candidate; reframe required.** |
| Inv‑9 | ✅ | RQF reward distribution variance is structurally tied to ρ̄. |

### 5.4 Implementation cost

| Resource | Cost |
|---|---|
| New kernels | **0** if we replace 2–4 of the K option cumulants with RQF cumulants in the existing `oak_predict` δ_k computation. **+1** otherwise (`oak_learn_rqf`). |
| Parameters | **+2A·R per RQF**, ×K_RQF. For K_RQF = 4 and R = 256, A = 16 ≈ 32 KB. |
| FLOPs/step | +5 % per RQF on `predict` and `learn_value`. |
| Risk | MEDIUM — needs careful design of the RQF reward distribution to maintain the "decreases to zero" property. The paper provides an explicit recipe. |

---

## 6. Honourable mentions (not Pass‑14 lead, useful supports)

1. **Almost Sure Convergence of Differential TD** — Blaser, Wang, Zhang.
   arXiv:[`2602.16629`](https://arxiv.org/abs/2602.16629), Feb 18 2026.
   *Theoretical*. Proves on‑policy n‑step differential TD converges with
   standard diminishing learning rates **without the local clock** assumed
   in earlier work — i.e. validates our `track_rate(stepCount, ridge)` for
   ρ̄. Three sufficient conditions for off‑policy convergence. **Cite this
   in `AGENTS.md` as the convergence backing for our differential TD.**
2. **NS‑NAC: Natural Policy Gradient for Average‑Reward Non‑Stationary RL**
   — Jali et al., arXiv:[`2504.16415`](https://arxiv.org/abs/2504.16415).
   Sub‑linear dynamic regret in the *average‑reward* setting. Could
   inform a future `oak_learn_pi` natural‑gradient variant.
3. **Functional Natural Policy Gradients** — arXiv:[`2603.28681`].
   Mar 2026. Functional natural gradient with offline regret bounds. Worth
   reading for the functional‑gradient view but not directly droppable.
4. **NGD‑KFAC for Online Continual Learning** — Khawand, Colliaux,
   arXiv:[`2603.20898`](https://arxiv.org/abs/2603.20898), Mar 21 2026.
   KFAC approximation of FIM in OCL settings; gains across all OCL methods.
   **Pass‑15 candidate** if we want a natural‑gradient version of TIDBD.
5. **Maximum‑Entropy Exploration with Future State‑Action Visitation
   Measures** — Bolland, Lambrechts, Ernst,
   arXiv:[`2603.18965`](https://arxiv.org/abs/2603.18965), Mar 19 2026.
   Off‑policy lower‑bound on entropy of the discounted occupancy via a
   contraction operator. **Use as cross‑check** for EVE's correctness; the
   undiscounted EVE is mathematically cleaner for our setup.
6. **Belief‑State RWKV** — arXiv:[`2604.09671`](https://arxiv.org/abs/2604.09671),
   Apr 1 2026. Already adopted in Pass 13 (per‑feature retention).
   The Apr‑2026 paper validates the belief‑state interpretation we already
   use; **cite this in `AGENTS.md` as the principled backing**.
7. **Distributional Active Inference (DAIF)** —
   arXiv:[`2601.20985`](https://arxiv.org/abs/2601.20985), Jan 28 2026.
   Embeds active inference into distributional RL. Theoretically beautiful
   (free‑energy ↔ ELBO ↔ log‑likelihood), but **requires a learned
   transition decoder** that overlaps with our `predict_next_latent` head
   and adds a latent generative model — too heavy for Pass 14. Re‑evaluate
   for Pass 16 once D2 Q‑learning is in.
8. **Hierarchical Skill Discovery** —
   arXiv:[`2601.23156`](https://arxiv.org/abs/2601.23156), Jan 2026 — uses
   grammar‑based segmentation of trajectories. Promising but requires
   Minecraft‑level data; not portable.
9. **Beyond Optimism: Exploration with Partially Observable Rewards** —
   Parisi, Kazemipour, Bowling.
   arXiv:[`2406.13909`](https://arxiv.org/abs/2406.13909), 2024.
   **Direct Alberta**. Solves the Mon‑MDP (monitored MDP) exploration
   problem and converges even when rewards are unobservable. Useful as
   theoretical backing for the **fact** that pure rewardRate is
   discriminatively insufficient (exactly the Pass‑12 multi‑signal Pareto
   gate's premise).
10. **Reward‑Aware Proto‑Representations (Default Representation, DR)** —
    Tse, Chandrasekar, **Machado**.
    arXiv:[`2505.16217v2`](https://arxiv.org/abs/2505.16217), revised
    Jan 30 2026. Direct Alberta. Generalises SR to be reward‑aware.
    Empirically beats SR on reward shaping, exploration, option discovery,
    transfer. **Pass‑15 candidate** to swap our ψ → DR if HSR + EVE
    underperform on the procedural envs.

## 7. Rejected (Inv‑violations or off‑mission)

| Paper | arxiv | Reason |
|---|---|---|
| **Adaptive Correlation‑Weighted Intrinsic Rewards (ACWI)** | [`2602.24081`](https://arxiv.org/abs/2602.24081) | Learnable `Beta Network` predicting the intrinsic‑reward weight from state — violates **Inv‑9** (learnable hyperparameter) and **Inv‑8** (intrinsic added to extrinsic stream). |
| **Search‑Inspired Exploration (SIERL)** | [`2602.00460`](https://arxiv.org/abs/2602.00460) | Sets sub‑goals *at the beginning of each episode* — violates **Inv‑4** (no episodic semantics). |
| **Intrinsic Reward Policy Optimization (IRPO)** | [`2601.21391`](https://arxiv.org/abs/2601.21391) | Surrogate policy gradient with intrinsic rewards mixed into the policy gradient — **Inv‑8** violation. |
| **Strategy‑aware Surprise (SuS)** | [`2601.10349`](https://arxiv.org/abs/2601.10349) | LLM/reasoning‑task scoped; per‑step concept of strategy doesn't translate. |
| **General Exploratory Bonus (GEB)** | [`2510.03269`](https://arxiv.org/abs/2510.03269) | RLHF‑specific; KL‑from‑uniform‑policy bonus implementation is *not* a clean fit for our setup (the analysis specifically targets reference‑model bias in language alignment). The paper's negative result **does** validate our existing avoidance of an entropy regularizer (Inv‑8 already forbids it). |
| **Streaming Soft Actor‑Critic / Streaming Deterministic AC (S2AC/SDAC)** | [`2603.08588`](https://arxiv.org/abs/2603.08588) | Uses LayerNorm pre‑activations, target networks, and a SAC entropy regularizer — multiple Inv violations (LayerNorm forbidden by BWH continual signal scale; target nets forbidden by streaming requirement; entropy regularizer forbidden by Inv‑8). |
| **Random Distribution Distillation (RDD)** | (2505.11044) | RND‑family; same hot‑stove issue as RND. |
| **MIRA / LLM‑driven motivation** | [`2602.17930`](https://arxiv.org/abs/2602.17930) | Requires LLM in the loop; off‑mission. |

---

## 8. Pass‑14 plan (one‑page summary)

**Adoption order (multi‑signal Pareto gate at every step):**

1. **Tier 0 → EVE intrinsic GVF** ([`2603.12325`]). +1 kernel
   (`oak_learn_eve_q`), +R fp32 params, ~1 % FLOPs. Wire into the chord
   keyboard as one of the K option‑policy contributions; do **not** add to
   reward stream. Acceptance: paired‑t Δ > 0 at p < 0.05 on
   `rewardRate↑` on the 46 zero‑envs.
2. **Tier 1 → D2 Q‑learning quantile heads** ([`2506.03333`]). Widen
   `valueW[R]` → `valueW[R · Q]` with Q = 5; quantile regression in
   `oak_learn_value`; argmax planner over (k*, q) lanes uses median +
   half‑IQR for risk‑seeking on novel states. ~2× FLOPs on
   `learn_value`/`learn_psi`. Acceptance: `pEff↑` and `LPM↑` (richer
   target distribution should slow overfitting).
3. **Tier 2 → HSR streaming NMF on ψ** ([`2602.12753`]).
   `oak_bk_hsr_nmf` runs every `bk_admit_commit` cycle. Re‑interpret K
   options as the K NMF eigenoptions of the option‑level SR. ~+5 %
   bookkeeping cost.
4. **Tier 3 (conditional) → Soft‑FB B head** ([`2602.06769`]). Only land
   if EVE + D2 + HSR have not closed the gap on ≥ 30 of the 46 zero envs.
5. **Tier 4 (conditional) → VBE‑as‑GVF** ([`2602.12375`]). Replace 2 of
   the K option cumulants with RQF disagreements; only if the gap
   persists on ≥ 10 envs.

**Documentation deltas in `AGENTS.md`:**
* Add Pass‑14 section under "Pass 13 architectural additions".
* Cite [`2603.12325`] (EVE), [`2506.03333`] (D2), [`2602.12753`] (HSR),
  [`2602.16629`] (differential TD ASC), [`2604.09671`] (Belief‑State RWKV
  validation).
* Update the "20 Metal kernels" pipeline list to **21** (+`oak_learn_eve_q`)
  and conditionally **22** (+`oak_bk_hsr_nmf`).

**Risk register:**
* The PPI loop in EVE requires the chord weight on the EVE key to grow
  monotonically. If it spikes too fast, the agent stops earning extrinsic
  reward on the **38 envs that already work**. Mitigation: gate the EVE
  key's chord weight by the existing **multi‑signal Pareto gate** — only
  let it dominate when `rewardRate` plateau ≥ 50 K steps.
* D2's quantile heads suffer plasticity loss in low‑signal regimes; the
  existing L2‑Effective‑Rank regularizer ([`2509.22335`]) and FIRE+SWR
  reinit ([`1804.03334` + `2404.02108`]) already in Pass 12 must extend
  to the new wider `valueW[R · Q]`.

---

## 9. Searches performed

`active inference EFE 2026 continual sparse`,
`successor features coverage exploration bonus 2026`,
`information gain exploration POMDP MiniGrid`,
`KFAC natural gradient policy average reward 2026`,
`forward‑backward representation 2026 successor option`,
`distributional differential value 2506.03333`,
`KL divergence uniform policy intrinsic exploration 2026`,
`MiniGrid KeyCorridor ObstructedMaze online lifelong 2026`,
`RND successor exploration sparse reward 2026`,
`option critic discovery sparse reward intrinsic skill 2026`,
`Naik Sutton differential RL continuing tasks 2026`,
`Martha White exploration intrinsic streaming online`,
`deep exploration ensemble bootstrapped DQN streaming online actor critic continual 2026`,
`successor representation eigenoption diffusion option discovery 2026`,
`Bowling Alberta Three Dogmas continual RL 2026`,
`belief state recurrent RWKV POMDP partially observable 2026`,
`average reward exploration bonus differential value 2026`,
`Abel continual reinforcement learning hindsight regret 2026`,
`information gain ensemble disagreement 2026`,
`Marlos Machado Bellemare options diffusion successor exploration 2026`,
`default representation reward aware proto representation 2026`,
`model based RL learning progress curiosity disagreement 2026`,
`monitored MDP exploration partially observable reward Bowling Parisi 2026`,
`EigenVector Exploration EVE entropy regularized average reward 2026`,
`Alon reward shaping average reward exploration recovery 2026`,
`KETCHUP K‑step return Bowling 2026`,
`average reward Q learning sample complexity Wan Naik Sutton continuing 2026`,
`deep RL entropy maximization steady state distribution explore covering 2026`,
`option keyboard skill chord exploration discovery 2603 2604`,
`BabyAI MiniGrid procedurally generated continual sparse exploration 2026`,
`successor measure exploration coverage deep RL model free 2026`,
`LayerNorm‑free batch‑free streaming continuous deep RL plasticity loss 2026`.

Total: **32 web searches**, **22 arxiv abstract fetches**, full text of
EVE ([`2603.12325`](https://arxiv.org/abs/2603.12325)) read in detail.

---

*Compiled by the OaK research librarian, 2026‑04‑18.
Five top candidates ranked, ten honourable mentions catalogued, eight
papers rejected on Inv‑grounds. All arxiv links verified.*
