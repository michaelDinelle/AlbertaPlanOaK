# Research dossier — population variance, ignition, mode-locking (2026-07-19)

Scout synthesis for the pathology: ~1–2% of lanes ("rockets") ignite to 0.4–0.7 reward/step while the median lane is slightly negative; the objective is EVERY agent converging to the same high rate. All arXiv IDs verified 2026-07-19. Hard constraints: nothing added to reward/cumulant streams (Inv-8); exploration may only influence action/option/feature SELECTION; no warmup/phase schedules (Inv-7); must not crush the strong mode (measured law).

**Trap flags:** arXiv 2006.04354 (EE-QL, average-reward UCB) is WITHDRAWN (proof bug) — never cite. `docs/roadmap.md:178-187` records dense psiCross/GPI cross-cumulant keyboards as gate-REJECTED — route exploration through the live option-selection path instead.

## 1. Seed sensitivity: measurement and causes

- Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (2108.13264, NeurIPS 2021): per-run outcomes are heavy-tailed, often multimodal; IQM + stratified bootstrap is the prescribed instrument — exactly what fleet.py implements. Patterson et al. (2304.01315, JMLR 2024): most published comparisons lack the seed count to resolve claimed effects. Anchors: Henderson 1709.06560, Colas 1806.08295, Chan 2003.14560.
- **Early path dependence / primacy bias** (Nikishin et al. 2205.07802): agents overfit early experience; final performance substantially decided early — a seed-variance generator. (Their fix is resets — see failures.)
- **Progressive plasticity loss** (Abbas 2303.07507; Dohare Nature 2024; Lyle 2402.18762 decomposition). Juliani & Ash (2405.19153) ran the on-policy mitigation bake-off: **only the regenerative family (shrink-and-perturb, regenerative regularization, continual-backprop replacement) consistently works; several popular mitigations are neutral or worse.** Strongest external corroboration of G&T-as-ignition-engine.
- **Exploration lock-in** — Go-Explore (1901.10995) names detachment (agent leaves the frontier, forgets the way back) and derailment (action noise knocks it off a good mode before consolidation). OaK's "mode extinction" = detachment+derailment in continuing form.
- **Streaming-specific fragility** (2410.14606, 2604.19033, 2605.06764): batch-1 online RL is disproportionately optimizer/step-size sensitive — independent support for per-feature step-size meta-learning as variance control.

## 2. Ignition as candidate-arrival × plasticity × retention

QD lineage is the theory of ignition: novelty search (Lehman-Stanley 2011), MAP-Elites (1504.04909) made "candidate-arrival rate × elite retention" the explicit mode-discovery engine; Dominated Novelty Search (2502.00593); QDHUAC (2604.20381) — high locomotion-mode coverage with 10× fewer samples via distributional critics + dominance selection. Mapping to OaK: G&T replacement rate = arrival rate (matches the measured 64→5 igniter collapse when throttled); keep-what-works = bonusW high-water + utility-gated replacement; residual plasticity is the third term (§1). Per-agent ignition rate = arrival × residual plasticity × frontier retention.

## 3. Ranked mechanisms (constraint-filtered)

### M1 — Regenerative plasticity floor (utility-gated shrink-and-perturb on low-utility slots)
Juliani-Ash 2405.19153; R² 2308.11958; UPGD 2404.00781 (utility-weighted perturbation built for streaming RL — gates perturbation by measured utility, which is what protects the strong mode); Dohare Nature 2024.
Weight-side only, fixed per-step structural rate ✓✓. Effects: ignition ↑ (measured law + external), median ↑, ceiling safe iff utility gate is absolute (ungated drift touches rocket slots). Merged with plasticity-dossier R3 → **queue L5.**

### M2 — Adaptive-temperature entropy floor on behavior sampling (floor, not bonus)
SAC auto-temperature 1812.05905; Average-Reward SAC 2501.09080 (entropy reg. transfers to average-reward objective); entropy-collapse documentation 2606.19771.
Track actorSupportEntropy per step (already emitted); a slow running dual adjusts only the wrapper's sampling temperature to hold H ≥ H_min, H_min anchored structurally to the σ=1/D_policy baseline. Nothing enters reward/cumulants/TD targets ✓; every-step tracker, no schedule ✓.
Effects: ignition ↑ modest, median ↑ (direct anti-mode-extinction, attacks derailment). Ceiling risk: floor binding above a rocket's natural entropy injects gait noise — cap H_min below the observed rocket band; gate must show best-lane class unchanged. **Queue L7.**

### M3 — Posterior-sampled (Thompson-style) option scoring off the twin heads
Osband et al., RVF (1806.03335, JMLR 2019); SUNRISE 2007.05938 (mechanism, not setting).
Twin vStop/vStop2 = a free two-point posterior; today disagreement only gates planning trust. Add: at a fixed structural per-step rate, the option score for SELECTION is drawn from a randomized interpolation of the twins; argmax on the mean remains default. Exploration in option space (coherent behaviors), never raw-action dithering ✓✓.
Effects: ignition ↑ (deep-exploration result: temporally extended exploration beats per-step noise), ceiling safe (twins agree in confident regions — rocket scores barely move). **Queue L8.**

### M4 — Count/visitation-gated option selection (counts as selection gate, never bonus)
Machado-Bellemare-Bowling, SR counts (1807.11622 — take the estimator, drop the bonus); LOOP (2404.12648, ICLR 2024) and Boone et al. (2502.06480, COLT 2025) — optimism pays in average-reward continuing FA; the latter replaces episodic doubling with a stationary rule (Inv-7-matching).
Geometry-gated tile banks are already stationary visit codings — maintain per-bank visit counters; at option-selection, resolve NEAR-TIES in favor of options whose predicted next-bank visitation is low. Lexicographic (tie-break) form is mandatory: any bonus inside the score suppresses a rocket's dominant option.
Cost: 32 counters/lane. Ablation: tie-break on/off, Tier-M → Tier-L; metrics: bank coverage, igniter fraction, best-lane class unchanged.

### M5 — Diversity-aware G&T commit (novelty as construction-selection criterion)
Dominated Novelty Search 2502.00593; QDHUAC 2604.20381; QD-RL 2211.02193.
When the G&T commit picks among candidates, prefer those whose short bank-activity descriptor diverges from the lane's trailing descriptor EMA (elites still protected by utility). Widens the lottery-ticket distribution. Novelty gates what gets BUILT, not reward ✓; every step ✓.

### M6 — Timescale-ordering audit (free)
Wan-Naik-Sutton 2006.16318 (ICML 2021) and 2110.13855 (NeurIPS 2022 — closest published result to OaK's exact setting: options + average reward + inter-option); Bhatnagar TAC 2009; rates 2407.18878; NAC-B 2505.19986.
Every convergence proof needs critic faster than actor, ρ̄ fastest — audit the per-head step-size ordering (MetaOptimize outputs vs track_rate) to guarantee the ordering structurally. Zero new behavior; trims the stuck-lane tail. Do as part of any L-candidate's pre-work.

### M7 — Stationary parameter-space noise on option policies (weak; last)
NoisyNet 1706.10295; PPL 1706.01905; continuations thin. Evidence is episodic/batch-era; wrapper already provides σ-noise; ceiling risk real (derailment). Screen only if M1–M4 stall.

## 4. Known to fail / crush the strong mode online

- Intrinsic motivation in reward (ICM 1705.10935, RND 1810.12894, DIAYN 1802.06070, max-state-entropy 2306.14808): Inv-8 violation.
- Phase-structured exploration (Go-Explore, MAP-Elites archives): Inv-7 violation; diagnosis only.
- Periodic-reset plasticity injection (2305.15555): epoch schedule + endangers the strong mode; M1's per-step utility-gated drift is the compliant substitute.
- Twin-min pessimism as the SELECTED score (TD3 1802.09477, Maxmin 2002.06487): pessimism on the behavior-side score suppresses exploration → deepens lock-in online. Disagreement stays a trust gate only.
- Entropy bonus inside the objective (SAC α·H in TD target): forbidden; M2 keeps only the temperature half.
- Raw-action ε-greedy / high fixed action noise in continuous control: destroys consolidated gaits (derailment).
- Magnitude-competition WTA sparsification: project-measured actor blinding (Mark III law b).

## 5. Population-objective notes

- No 2023–2026 work gives consistency results for populations of independent learners — population variance is an ignition-probability problem, not an averaging problem: with igniter fraction p over N lanes, inter-lane variance is binomial p(1−p)/N. Every mechanism scores by how much it moves p at fixed ceiling.
- The rliable/Patterson line validates the existing paired-seed IQM gate as the correct instrument for this heavy-tailed regime; no protocol change suggested.
