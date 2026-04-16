# Alberta-plan / Oak reading stack

PDFs in this folder are **primary sources** aligned with *The Alberta Plan for AI Research* (arXiv:2208.11173) and the classic continual-learning / options / step-size literature the plan builds on. Filenames are ordered for skimming.

| File | Contents |
|------|----------|
| `01-alberta-plan-2208.11173-arxiv.pdf` | Sutton, Bowling, Pilarski — roadmap, Big World, 12 steps, Oak as Step 11 vision |
| `02-sutton-horde-AAMAS-2011.pdf` | *Horde* (AAMAS 2011) — many parallel GVFs from sensorimotor stream |
| `03-mahmood-sutton-autostep-ICML-2012.pdf` | Autostep (ICML 2012) — tuning-free vector step sizes |
| `04-sutton-IDBD-1992-incremental-dbar-d.pdf` | IDBD — per-weight step sizes (precursor to TIDBD) |
| `05-sutton-precup-options-between-MDPs-1999.pdf` | Options framework — initiation / policy / termination semantics |
| `06-kearney-TIDBD-step-sizes-2018-arxiv.pdf` | TIDBD — per-dimension TD step sizes via meta-descent |

## Papers that **cite** *The Alberta Plan* (arXiv:2208.11173)

Verified in each PDF’s bibliography (`Alberta plan` / `2208.11173`).

| File | Citation |
|------|----------|
| `07-burega-meta-gradient-search-control-Dyna-2406.19561-arxiv.pdf` | Burega et al., *Meta-Gradient Search Control* — learns **search control** for Dyna-style planning (which states to query); ties to **Planning II** / scalable model-based RL |
| `08-orenstein-agents-reason-about-computation-2510.22833-arxiv.pdf` | Orenstein et al. — agents **control compute** via **options** over action × duration; reward includes **compute cost**; cites Alberta Plan |
| `09-alver-precup-minimal-value-equivalent-partial-models-2301.10119-arxiv.pdf` | Alver & Precup — **minimal value-equivalent partial models** for **lifelong** planning (model only what matters for value) |

Oak-specific invited material (e.g. NeurIPS invited talk slides) is not duplicated here; use `incompleteideas.net` and official event pages for the newest Oak “keyboard / FC-STOMP” wording.
