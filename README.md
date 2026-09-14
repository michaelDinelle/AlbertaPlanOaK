# OaK (Options and Knowledge)

![Eight OaK environments playing inside a pink butterfly](butterfly.png)

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) — Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski.

<img src="fc-stomp.png" alt="FC-STOMP cycle of discovery: state features define subtasks, learned options define models, and planning and feature use close the feedback loop." width="700">

Cycle of discovery from Richard S. Sutton's [Alberta Plan slides, slide 11](http://incompleteideas.net/Talks/AlbertaPlan.pdf#page=11).

Recorded on a binned M4 Max.

## Base

Geometry: 32 percepts, 480 features (96 active per step), 32 options; 200,160 total parameters (6 action dimensions). 917,550 SPS.


| Environment | OaK mean | OaK IQM | PPO mean | PPO IQM | Random mean | Random IQM | OaK vs PPO wins | OaK vs Random wins | PPO vs Random wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acrobot-cont-v0 | 0.0070594 | 0.00702179 | 0.00923607 | 0.0093897 | 0.000384352 | 0.000383259 | 2.11% | 100.00% | 98.44% |
| Chain-cont-v0 | 0.0664686 | 0.0665351 | 0.066659 | 0.0666592 | 0.00416248 | 0.00417216 | 0.00% | 100.00% | 100.00% |
| DeepSea-cont-v0 | 0.14218 | 0.142184 | 0.0984861 | 0.125892 | 0.000498203 | 0.000500664 | 87.33% | 100.00% | 70.31% |
| DelayedChain-cont-v0 | 0.0318193 | 0.0320618 | 0.0281339 | 0.0322526 | 0.00101144 | 0.00100528 | 14.06% | 100.00% | 87.50% |
| FourRooms-cont-v0 | 0.00694073 | 0.00774509 | 0.133624 | 0.133612 | 0.00260776 | 0.00260058 | 0.00% | 72.66% | 100.00% |
| HalfCheetah-v4 | 1.74358 | 1.68236 | 3.98669 | 4.17957 | -0.309186 | -0.308994 | 6.42% | 100.00% | 100.00% |
| HighDimMaze-cont-v0 | 0.0115278 | 0.0114826 | 0.0691204 | 0.0847164 | 0.00205549 | 0.00205924 | 20.31% | 100.00% | 79.69% |
| JellyBeanWorld-cont-v0 | 7.80251e-07 | -7.80251e-07 | 7.84153e-05 | 7.80251e-07 | -5.77386e-06 | -7.49041e-06 | 42.55% | 53.88% | 60.06% |
| KeyCorridor-cont-v0 | 0.059387 | 0.0645541 | 0.084388 | 0.107874 | 0.00368575 | 0.00369309 | 23.65% | 92.19% | 81.25% |
| LargeDeepSea-cont-v0 | 0.00227206 | -0.000141127 | -8.3009e-07 | -1.13429e-07 | -0.000310005 | -0.000310527 | 4.69% | 100.00% | 100.00% |
| MemoryChain-cont-v0 | 0.0231297 | 0.0237814 | 0.0127726 | 0.0148167 | 0.00050373 | 0.000500765 | 97.61% | 100.00% | 85.94% |
| MountainCar-cont-v0 | 0.00643052 | 0.0068066 | 0.00471311 | 0.00593428 | 2.41878e-05 | 2.38757e-05 | 47.61% | 98.44% | 71.92% |
| MultiKeyCorridor-cont-v0 | 0.00482952 | 0.00482648 | 0.025823 | 0.0293864 | 0.000708234 | 0.000707376 | 40.62% | 100.00% | 59.38% |
| POMaze-cont-v0 | 0.00545419 | 0.00460957 | 0.0438517 | 0.0390553 | 0.00154193 | 0.00153475 | 40.92% | 56.15% | 48.44% |
| Pendulum-cont-v0 | -0.199475 | -0.0165761 | -0.219831 | -4.83485e-05 | -3.88488 | -3.88251 | 10.03% | 98.44% | 97.63% |
| ProceduralMaze-cont-v0 | 0.000115009 | 2.12228e-05 | 0 | 0 | 0.000747793 | 0.000743423 | 75.78% | 4.38% | 0.00% |
| Total | 0.119483 | 0.0120198 | 0.271484 | 0.0264099 | -0.261028 | 0.000678625 | 32.11% | 86.01% | 77.53% |

## Setup and run

Requires macOS 26+, Xcode with Metal tools, Python 3.11, and Bun. Benchmark recordings stay local and are ignored by Git. From the repo root: `./setup.sh --preset base` builds the engine and the page, then `./fleet.py run` records a candidate. Everything is served by one process, `.build/release/OaK`, which `./fleet.py` starts; open http://127.0.0.1:5174. For page development, `cd frontend && bun run dev` and open http://localhost:5173.

Geometry is agent capacity, declared in `Sources/Geometry.metal`; larger presets use more compute and memory. Wide streams increase percept and feature counts above these declared sizes. Total parameters = features × [1 + options × (7 + action dimensions)]; counts below use 6 action dimensions.

| Preset | Percepts | Features | Active per step | Options | Total parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 32 | 480 | 96 | 32 | 200,160 |
| pro | 64 | 960 | 192 | 64 | 799,680 |
| max | 128 | 1920 | 384 | 128 | 3,196,800 |
| ultra | 160 | 2400 | 480 | 160 | 4,994,400 |
