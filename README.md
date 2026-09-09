# OaK Base

Geometry: 32 percepts, 480 features (96 active per step), 32 options.

Keeper: `ca6221ed5f7d374c3bba4974c310f5e66a91b4df`. Full-run reward per step over 1,000,000 steps, 64 seeds per environment. IQM averages the middle 50% of seeds. Win percentages compare matching seeds; ties count as half a win.

| Environment | OaK mean | OaK IQM | PPO mean | PPO IQM | Random mean | Random IQM | OaK vs PPO wins | OaK vs Random wins | PPO vs Random wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acrobot-cont-v0 | 0.0059618 | 0.00592066 | 0.00885534 | 0.00909559 | 0.000386531 | 0.000385812 | 3.12% | 100.00% | 98.44% |
| Chain-cont-v0 | 0.0654973 | 0.0655481 | 0.0664122 | 0.0664693 | 0.00417067 | 0.00417069 | 1.56% | 100.00% | 100.00% |
| DeepSea-cont-v0 | 0.140243 | 0.140262 | 0.0938155 | 0.117435 | 0.000500562 | 0.000500166 | 93.75% | 100.00% | 70.31% |
| DelayedChain-cont-v0 | 0.0309037 | 0.0310091 | 0.0274261 | 0.0321192 | 0.00101069 | 0.00101266 | 18.75% | 100.00% | 87.50% |
| FourRooms-cont-v0 | 0.0063565 | 0.00665075 | 0.104155 | 0.104583 | 0.0026182 | 0.00262144 | 0.00% | 84.38% | 100.00% |
| HalfCheetah-v4 | 1.5768 | 1.50755 | 3.18892 | 3.28864 | -0.309297 | -0.30943 | 7.81% | 100.00% | 100.00% |
| HighDimMaze-cont-v0 | 0.00718053 | 0.00695137 | 0.0462514 | 0.0562211 | 0.00204973 | 0.00205053 | 20.31% | 100.00% | 79.69% |
| JellyBeanWorld-cont-v0 | 1.30469e-05 | 8.34375e-06 | 0.000107281 | 5.45e-05 | 7.8125e-07 | 9.375e-08 | 27.34% | 57.03% | 82.81% |
| KeyCorridor-cont-v0 | 0.0479658 | 0.0502806 | 0.0914313 | 0.101761 | 0.00369542 | 0.00369712 | 9.38% | 100.00% | 100.00% |
| LargeDeepSea-cont-v0 | 0.000870108 | -0.000208075 | -2.21188e-06 | -1.53514e-06 | -0.000309996 | -0.000310154 | 4.69% | 100.00% | 100.00% |
| MemoryChain-cont-v0 | 0.0203923 | 0.0211873 | 0.0121761 | 0.0144191 | 0.000503094 | 0.0005005 | 96.88% | 100.00% | 87.50% |
| MountainCar-cont-v0 | 0.00486516 | 0.00510931 | 0.00396341 | 0.00466794 | 2.40469e-05 | 2.40625e-05 | 40.62% | 98.44% | 74.22% |
| MultiKeyCorridor-cont-v0 | 0.00426247 | 0.00428634 | 0.0192767 | 0.0215977 | 0.000716188 | 0.000717531 | 40.62% | 100.00% | 60.94% |
| POMaze-cont-v0 | 0.00404928 | 0.00404331 | 0.0333905 | 0.0268092 | 0.00153345 | 0.00153422 | 51.56% | 70.31% | 48.44% |
| Pendulum-cont-v0 | -0.371727 | -0.140029 | -0.0671477 | -0.0109328 | -3.89198 | -3.88951 | 25.00% | 98.44% | 100.00% |
| ProceduralMaze-cont-v0 | 0.000249766 | 0.000171281 | 7.25e-06 | 4.125e-06 | 0.000730344 | 0.000734781 | 100.00% | 3.12% | 0.00% |

## Setup and run

Requires macOS 26+, Xcode with Metal tools, Python 3.11, and Bun for the frontend. From the repo root: `./setup.sh --preset base`, then `./fleet.py run` (requires matching best fleet data and baselines). Frontend: `cd frontend && bun install && bun run dev`, then open http://localhost:5173.

Geometry is agent capacity, declared in `Sources/Geometry.metal`; larger presets use more compute and memory. Wide streams increase percept and feature counts above these declared sizes.

| Preset | Percepts | Features | Active per step | Options |
| --- | ---: | ---: | ---: | ---: |
| base | 32 | 480 | 96 | 32 |
| pro | 64 | 960 | 192 | 64 |
| max | 128 | 1920 | 384 | 128 |
| ultra | 160 | 2400 | 480 | 160 |
