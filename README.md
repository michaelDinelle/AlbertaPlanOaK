# OaK (Options and Knowledge)

![Eight OaK environments playing inside a pink butterfly](butterfly.png)

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) — Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski.

<img src="fc-stomp.png" alt="FC-STOMP cycle of discovery: state features define subtasks, learned options define models, and planning and feature use close the feedback loop." width="700">

Cycle of discovery from Richard S. Sutton's [Alberta Plan slides, slide 11](http://incompleteideas.net/Talks/AlbertaPlan.pdf#page=11).

Recorded on a binned M4 Max.

## Base

Geometry: 32 percepts, 480 features (96 active per step), 32 options; 200,160 total parameters (6 action dimensions). 1,106,635 SPS.

Reward per step over the full 1,000,000-step best recording (`04320edcc1d7309177bcaf94f79ecca46660a94c`), using all 16 environments, 64 seeds each, and matching recorded PPO and Random baselines. Mean reward per step over all 64 seeds; wins are paired per seed, ties count half.

| Environment | OaK | PPO | Random | OaK vs PPO wins |
|---|---:|---:|---:|---:|
| Acrobot-cont-v0 | 0.0059618 | 0.00885534 | 0.000386531 | 3.12% |
| Chain-cont-v0 | 0.0654791 | 0.0664122 | 0.00417067 | 1.56% |
| DeepSea-cont-v0 | 0.140243 | 0.0938155 | 0.000500562 | 86.21% |
| DelayedChain-cont-v0 | 0.0308337 | 0.0274261 | 0.00101069 | 18.55% |
| FourRooms-cont-v0 | 0.0063565 | 0.104155 | 0.0026182 | 0.00% |
| HalfCheetah-v4 | 1.5768 | 3.18892 | -0.309297 | 6.32% |
| HighDimMaze-cont-v0 | 0.00747594 | 0.0462514 | 0.00204973 | 20.31% |
| JellyBeanWorld-cont-v0 | 1.31563e-05 | 0.000107281 | 7.8125e-07 | 27.08% |
| KeyCorridor-cont-v0 | 0.0479658 | 0.0914313 | 0.00369542 | 8.79% |
| LargeDeepSea-cont-v0 | 0.00103251 | -2.21188e-06 | -0.000309996 | 4.69% |
| MemoryChain-cont-v0 | 0.0208138 | 0.0121761 | 0.000503094 | 96.69% |
| MountainCar-cont-v0 | 0.00486516 | 0.00396341 | 2.40469e-05 | 43.62% |
| MultiKeyCorridor-cont-v0 | 0.00434253 | 0.0192767 | 0.000716188 | 40.62% |
| POMaze-cont-v0 | 0.00308941 | 0.0333905 | 0.00153345 | 51.34% |
| Pendulum-cont-v0 | -0.487612 | -0.0671477 | -3.89198 | 23.56% |
| ProceduralMaze-cont-v0 | 0.000277875 | 7.25e-06 | 0.000730344 | 99.78% |
| Total | 0.0892461 | 0.226815 | -0.261478 | 33.27% |

## Setup and run

Requires macOS 26+, Xcode with Metal tools, and Bun. Benchmark recordings stay local and are ignored by Git. From the repo root: `./setup.sh --preset base` builds the engine and the page, `./fleet.py` records a candidate and gates it against the best, and `cd frontend && bun run serve` serves the recordings; open http://127.0.0.1:5174. For page development, `cd frontend && bun run dev` and open http://localhost:5173.

Geometry is agent capacity, declared in `Sources/Geometry.metal`; larger presets use more compute and memory. Wide streams increase percept and feature counts above these declared sizes. Total parameters = features × [1 + options × (7 + action dimensions)]; counts below use 6 action dimensions.

| Preset | Percepts | Features | Active per step | Options | Total parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 32 | 480 | 96 | 32 | 200,160 |
| pro | 64 | 960 | 192 | 64 | 799,680 |
| max | 128 | 1920 | 384 | 128 | 3,196,800 |
| ultra | 160 | 2400 | 480 | 160 | 4,994,400 |
