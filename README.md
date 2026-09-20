# OaK (Options and Knowledge)

![Eight OaK environments playing inside a pink butterfly](butterfly.png)

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) — Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski.

<img src="fc-stomp.png" alt="FC-STOMP cycle of discovery: state features define subtasks, learned options define models, and planning and feature use close the feedback loop." width="700">

Cycle of discovery from Richard S. Sutton's [Alberta Plan slides, slide 11](http://incompleteideas.net/Talks/AlbertaPlan.pdf#page=11).

Recorded on a binned M4 Max.

## Base

Geometry: 32 percepts, 480 features (96 active per step); 3,360 parameters (6 action dimensions). 1,374,044 SPS.

Reward per step over the final 100,000 steps of the 1,000,000-step best recording (`e34c68688a85333f69e6913c3f293d45a4812c4c`), all 16 environments, 64 seeds each, with the recorded PPO and Random baselines on the same seeds. Mean over seeds; wins are paired per seed against PPO, ties count half.

| Environment | OaK | PPO | Random | OaK vs PPO wins |
|---|---:|---:|---:|---:|
| Acrobot-cont-v0 | 0.0072 | 0.0092 | 0.000387 | 1.56% |
| Chain-cont-v0 | 0.0664 | 0.0667 | 0.0042 | 0.00% |
| DeepSea-cont-v0 | 0.1422 | 0.0985 | 0.000495 | 96.09% |
| DelayedChain-cont-v0 | 0.0319 | 0.0282 | 0.0010 | 12.50% |
| FourRooms-cont-v0 | 0.0079 | 0.1342 | 0.0027 | 0.00% |
| HalfCheetah-v4 | 1.7559 | 4.0326 | -0.3095 | 6.25% |
| HighDimMaze-cont-v0 | 0.0115 | 0.0697 | 0.0021 | 20.31% |
| JellyBeanWorld-cont-v0 | -3.003e-05 | 4.908e-05 | 2.787e-06 | 41.41% |
| KeyCorridor-cont-v0 | 0.0604 | 0.0846 | 0.0037 | 20.31% |
| LargeDeepSea-cont-v0 | 0.0019 | -8.254e-07 | -0.000310 | 3.12% |
| MemoryChain-cont-v0 | 0.0239 | 0.0128 | 0.000503 | 100.00% |
| MountainCar-cont-v0 | 0.0066 | 0.0048 | 2.214e-05 | 42.19% |
| MultiKeyCorridor-cont-v0 | 0.0045 | 0.0260 | 0.000712 | 40.62% |
| POMaze-cont-v0 | 0.0083 | 0.0445 | 0.0016 | 46.88% |
| Pendulum-cont-v0 | -0.0395 | -0.3516 | -3.8463 | 9.38% |
| ProceduralMaze-cont-v0 | 0.000180 | 0 | 0.000759 | 68.75% |

Against the previous best (`a723de2ff946c0193ba199935a84682471ae9b96`, the same agent with its option subsystem, plan wave, and a learning skip every 32 steps): suite interquartile mean of the paired, IQR-scaled difference +0.059 with a 95% stratified-bootstrap CI of [+0.006, +0.111]; probability of improvement 0.54 [0.51, 0.57]; per-environment regression guard silent at family-wise 0.05; 1,374,044 steps/s against 1,047,478.

## Setup and run

Requires macOS 26+, Xcode with Metal tools, and Bun. Benchmark recordings stay local and are ignored by Git. From the repo root: `./setup.sh --preset base` builds the engine, `./fleet.py` records a candidate and judges it against the best, `./fleet.py adopt` records and sets the best without a comparison (an empty `best` or a changed schedule), and `cd frontend && bun install && bun run dev` serves the recordings and prints the URL to open.

The judge promotes when the candidate is better on reward or throughput and worse on neither. Reward is compared on paired seeds: identical reward streams count as not worse with no uncertainty; otherwise the suite statistic is the interquartile mean over every environment and seed of the difference scaled by the best's per-environment interquartile range, with a stratified bootstrap CI, and a Holm-adjusted one-sided Wilcoxon test per environment guards against a single large regression. Throughput must clear the best by one percent to count as better and stay within one percent to count as not worse. Baselines for `random` and `ppo` are ordinary recordings found by environment source digest, seeds, and horizon, and are recorded only for environments that have none.

Geometry is agent capacity, declared in `Sources/Geometry.metal`; larger presets use more compute and memory. Wide streams increase percept and feature counts above these declared sizes. Total parameters = features × (1 + action dimensions); counts below use 6 action dimensions.

| Preset | Percepts | Features | Active per step | Total parameters |
| --- | ---: | ---: | ---: | ---: |
| base | 32 | 480 | 96 | 3,360 |
| pro | 64 | 960 | 192 | 6,720 |
| max | 128 | 1920 | 384 | 13,440 |
| ultra | 160 | 2400 | 480 | 16,800 |
