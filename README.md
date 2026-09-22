# OaK (Options and Knowledge)

![Eight OaK environments playing inside a pink butterfly](butterfly.png)

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) — Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski.

<img src="fc-stomp.png" alt="FC-STOMP cycle of discovery: state features define subtasks, learned options define models, and planning and feature use close the feedback loop." width="700">

Cycle of discovery from Richard S. Sutton's [Alberta Plan slides, slide 11](http://incompleteideas.net/Talks/AlbertaPlan.pdf#page=11).

Recorded on a binned M4 Max.

## Base

Geometry starts at 32 percepts, 480 features (96 active per step); 3,360 parameters (6 action dimensions). 4,536,488 SPS on the 123-environment schedule.

Reward per step over the final 100,928 steps of the 1,000,000-step best recording (`4262868f4bce5ea25ac4905e60742f75d11be371`), all 123 environments with their declared seed counts (4,915 lanes), with the Oracle, Random and PPO baselines on the same seeds. Mean over seeds; Score = (OaK − Random) / (Oracle − Random), unclipped.

| Environment | Category | OaK | Oracle | Random | PPO | Score |
|---|---|---:|---:|---:|---:|---:|
| AccessControl-cont-v0 | decision | 2.7283 | 2.7482 | 1.6979 | 2.7417 | 0.9811 |
| AccessControl-drift-cont-v0 | decision | 1.5419 | 1.7143 | 0.9601 | 1.5957 | 0.7714 |
| Acrobot-cont-v0 | control | 0.0073 | 0.0110 | 0.0003926 | 0.0091 | 0.6478 |
| Acrobot-slip-cont-v0 | control | 0.0053 | 0.0077 | 0.0003863 | 0.0081 | 0.6753 |
| Acrobot-position-cont-v0 | control | 0.0016 | 0.0110 | 0.0003884 | 0.0009646 | 0.1156 |
| AliasedRing-8-cont-v0 | memory | 0.1561 | 0.1861 | -0.0056 | 0.0177 | 0.8437 |
| AliasedRing-32-cont-v0 | memory | 0.0273 | 0.0572 | -0.0386 | -3.39e-06 | 0.6877 |
| AliasedRing-128-cont-v0 | memory | 0.0015 | 0.0153 | -0.0471 | -6.038e-06 | 0.7796 |
| Bandit-3-cont-v0 | decision | 0.8299 | 0.8314 | -0.0610 | 0.8312 | 0.9983 |
| Bandit-6-cont-v0 | decision | 1.3209 | 1.3262 | 0.0451 | 1.3254 | 0.9958 |
| Bandit-3-drift-cont-v0 | decision | 0.8791 | 0.9785 | -0.1495 | 0.9619 | 0.9119 |
| Bandit-6-drift-cont-v0 | decision | 1.6858 | 1.8024 | 0.0607 | 1.6964 | 0.9331 |
| BlockingMaze-cont-v0 | navigation | 0.0008577 | 0.0628 | 0.0015 | 0.0009214 | -0.0105 |
| ShortcutMaze-cont-v0 | navigation | 0.0721 | 0.0997 | 0.0029 | 0.0640 | 0.7154 |
| Calibration-30-cont-v0 | decision | -0.000417 | -2.983e-05 | -0.0419 | -2.052e-05 | 0.9908 |
| Calibration-80-cont-v0 | decision | -0.0007859 | -4.131e-05 | -0.0920 | -0.0080 | 0.9919 |
| CartPoleSwingUp-cont-v0 | control | -0.5616 | 1.0000 | -0.8162 | -0.6067 | 0.1402 |
| CartPoleSwingUp-slip-cont-v0 | control | -0.5402 | 0.9998 | -0.8150 | -0.6692 | 0.1514 |
| Catch-5x10-cont-v0 | control | 0.0656 | 0.1111 | -0.0666 | 0.1109 | 0.7440 |
| Catch-10x20-cont-v0 | control | -0.0329 | 0.0526 | -0.0422 | 0.0520 | 0.0985 |
| Catch-noise-cont-v0 | control | 0.0593 | 0.1113 | -0.0665 | 0.1104 | 0.7073 |
| Catch-scale-cont-v0 | control | -2.9920 | 11.1119 | -6.6658 | 11.0875 | 0.2067 |
| Cellular-cont-v0 | prediction | 0.9999 | 1.0000 | 0.5846 | 0.9998 | 0.9998 |
| Chain-cont-v0 | navigation | 0.0665 | 0.0667 | 0.0041 | 0.0667 | 0.9981 |
| CliffWalk-cont-v0 | navigation | -30.4170 | 6.7695 | -10.1237 | -23.1424 | -1.2013 |
| CliffWalk-slip-cont-v0 | navigation | -6.5393 | 4.0658 | -10.1291 | -22.1353 | 0.2529 |
| CoinSwap-slow-cont-v0 | decision | 0.5958 | 0.9001 | 0.5002 | 0.5987 | 0.2390 |
| CoinSwap-fast-cont-v0 | decision | 0.6080 | 0.8998 | 0.5001 | 0.5019 | 0.2701 |
| TraceConditioning-10-cont-v0 | prediction | 0.0209 | 0.0334 | -0.0075 | -0.0001121 | 0.6946 |
| TraceConditioning-100-cont-v0 | prediction | -0.0002174 | 0.0033 | -0.0233 | -0.0002492 | 0.8674 |
| TracePatterning-cont-v0 | prediction | -0.0002131 | 0.0066 | -0.0215 | -0.0011 | 0.7573 |
| Contextual-8-cont-v0 | decision | 0.9724 | 0.9947 | -0.0003318 | 0.9823 | 0.9776 |
| Contextual-16-cont-v0 | decision | 1.0182 | 1.0439 | -0.0013 | 1.0255 | 0.9755 |
| Contextual-8-drift-cont-v0 | decision | 2.7745 | 2.9734 | -0.0015 | 2.8938 | 0.9331 |
| Contextual-16-drift-cont-v0 | decision | 3.7939 | 4.1108 | -0.0031 | 3.9726 | 0.9230 |
| CountRecall-8-cont-v0 | memory | 0.0527 | 0.1250 | 0.0311 | 0.0384 | 0.2309 |
| CountRecall-32-cont-v0 | memory | 0.0078 | 0.0312 | 0.0078 | 0.0078 | 0.0008985 |
| Covariate-fixed-cont-v0 | prediction | 0.9601 | 1.0000 | 0.5002 | 0.9949 | 0.9202 |
| Covariate-shift-cont-v0 | prediction | 0.7305 | 1.0000 | 0.5004 | 0.9954 | 0.4606 |
| DeepSea-8-cont-v0 | navigation | 0.1422 | 0.1423 | 0.0005047 | 0.0973 | 0.9994 |
| DeepSea-16-cont-v0 | navigation | 0.0030 | 0.0663 | -0.0003101 | -8.402e-07 | 0.0490 |
| DeepSea-32-cont-v0 | navigation | -9.213e-05 | 0.0321 | -0.0001561 | -1.453e-07 | 0.0020 |
| DeepSea-16-slip-cont-v0 | navigation | -0.0001574 | 0.0306 | -0.0003111 | -3.261e-05 | 0.0050 |
| Forager-16-1-cont-v0 | world | 0.0182 | 0.0291 | 0.0001413 | 0.0238 | 0.6232 |
| Forager-64-1-cont-v0 | world | 0.0158 | 0.0308 | 5.571e-05 | 0.0226 | 0.5126 |
| Forager-16-2-cont-v0 | world | 0.0147 | 0.0287 | 7.121e-05 | 0.0252 | 0.5126 |
| Forager-16-1-switch-cont-v0 | world | 0.0176 | 0.0291 | -0.0001827 | 0.0239 | 0.6063 |
| FourRoomsHall-cont-v0 | navigation | 0.1139 | 0.1252 | 0.0028 | 0.1220 | 0.9080 |
| FourRoomsHall-moving-cont-v0 | navigation | 0.0110 | 0.1262 | 0.0027 | 0.0168 | 0.0670 |
| Garnet-10-2-2-cont-v0 | decision | 0.8336 | 0.8499 | 0.0207 | 0.7866 | 0.9804 |
| Garnet-50-4-3-cont-v0 | decision | 1.0297 | 1.2132 | 0.0152 | 0.9300 | 0.8468 |
| Garnet-10-2-2-resample-cont-v0 | decision | 0.3047 | 0.6907 | 0.0056 | 0.1316 | 0.4366 |
| Garnet-50-4-3-resample-cont-v0 | decision | 0.2403 | 1.1765 | 0.0057 | 0.1971 | 0.2004 |
| HashedMDP-8-local-cont-v0 | decision | 0.7022 | 0.9457 | -0.0001721 | 0.6657 | 0.7426 |
| HashedMDP-16-local-cont-v0 | decision | 0.0075 | 1.8590 | 6.23e-05 | 0.1209 | 0.0040 |
| HashedMDP-8-global-cont-v0 | decision | 0.5365 | 0.7261 | 1.154e-05 | 0.5185 | 0.7389 |
| HashedMDP-16-global-cont-v0 | decision | 0.0039 | 0.7730 | -0.0001035 | 0.0050 | 0.0051 |
| HiddenMode-slow-cont-v0 | decision | 0.6492 | 1.0000 | 0.5007 | 0.5147 | 0.2974 |
| HiddenMode-fast-cont-v0 | decision | 0.6102 | 1.0000 | 0.5000 | 0.5047 | 0.2204 |
| JellyBeanWorld-cont-v0 | world | -2.462e-05 | 0.0374 | 2.787e-06 | 2.941e-05 | -0.0007334 |
| JellyBeanWorld-8-cont-v0 | world | -8.05e-06 | 0.0382 | 2.632e-06 | 0.0001002 | -0.0002795 |
| KeyDoor-8-cont-v0 | memory | 0.1250 | 0.1250 | 0.0624 | 0.1250 | 1.0000 |
| KeyDoor-64-cont-v0 | memory | 0.0156 | 0.0156 | 0.0078 | 0.0156 | 0.9998 |
| KeyDoor-512-cont-v0 | memory | 0.0015 | 0.0019 | 0.0009722 | 0.0018 | 0.5272 |
| TMaze-4-1-cont-v0 | memory | 0.2371 | 0.2500 | 0.0385 | 0.1105 | 0.9392 |
| TMaze-16-1-cont-v0 | memory | 0.0490 | 0.0625 | 0.0021 | 0.0285 | 0.7758 |
| TMaze-32-1-cont-v0 | memory | 0.0239 | 0.0312 | 0.0005092 | 0.0135 | 0.7620 |
| TMaze-64-1-cont-v0 | memory | 0.0092 | 0.0156 | 0.0001206 | 0.0065 | 0.5829 |
| TMaze-16-2-cont-v0 | memory | 0.0262 | 0.0625 | 0.0010 | 0.0016 | 0.4090 |
| TMaze-64-2-cont-v0 | memory | 0.0034 | 0.0156 | 6.326e-05 | 0.0002942 | 0.2116 |
| MountainCar-cont-v0 | control | 0.0063 | 0.0081 | 2.202e-05 | 0.0043 | 0.7846 |
| MountainCar-slip-cont-v0 | control | 0.0047 | 0.0069 | 2.121e-05 | 0.0048 | 0.6828 |
| MountainCar-position-cont-v0 | control | 0.0049 | 0.0081 | 2.57e-05 | 1.579e-05 | 0.6045 |
| Nexting-1-cont-v0 | prediction | 0.9898 | 1.0000 | 0.5001 | 0.9873 | 0.9797 |
| Nexting-10-cont-v0 | prediction | 0.9732 | 1.0000 | 0.4998 | 0.8957 | 0.9464 |
| Operant-16-cont-v0 | decision | 0.0611 | 0.0624 | -0.0157 | 0.0624 | 0.9832 |
| Operant-64-cont-v0 | decision | 0.0146 | 0.0156 | -0.0414 | 0.0156 | 0.9834 |
| Parity-4-cont-v0 | memory | 0.5126 | 1.0000 | 0.5005 | 0.4997 | 0.0243 |
| Parity-64-cont-v0 | memory | 0.4996 | 1.0000 | 0.5001 | 0.5005 | -0.0009079 |
| Password-2-cont-v0 | memory | 0.4476 | 0.5000 | 0.0499 | 0.4999 | 0.8836 |
| Password-4-cont-v0 | memory | 0.2330 | 0.2500 | 0.0030 | 0.2494 | 0.9310 |
| Password-6-cont-v0 | memory | 0.0750 | 0.1667 | 0.0001838 | 0.0795 | 0.4494 |
| Patience-5-cont-v0 | memory | 0.9999 | 1.0000 | 0.1306 | 0.9982 | 0.9998 |
| Patience-50-cont-v0 | memory | 0.0998 | 1.0002 | 0.0499 | 0.7823 | 0.0525 |
| Patience-500-cont-v0 | memory | 0.0998 | 1.0007 | 0.0500 | 0.1844 | 0.0524 |
| Pendulum-cont-v0 | control | -0.0814 | 0 | -3.9351 | -0.6801 | 0.9793 |
| Pendulum-slip-cont-v0 | control | -0.0450 | -0.0044 | -3.8858 | -0.0020 | 0.9895 |
| Pendulum-position-cont-v0 | control | -0.5242 | 0 | -3.8806 | -2.6680 | 0.8649 |
| Permuted-fixed-cont-v0 | prediction | 0.9997 | 1.0000 | 0.2501 | 0.9998 | 0.9996 |
| Permuted-recode-cont-v0 | prediction | 0.4189 | 1.0000 | 0.2500 | 0.4476 | 0.2253 |
| PuddleWorld-cont-v0 | control | -0.4681 | 3.7153 | -4.6782 | 2.9221 | 0.5016 |
| Recite-4-cont-v0 | memory | 0.2618 | 0.5000 | 0.1253 | 0.1250 | 0.3644 |
| Recite-16-cont-v0 | memory | 0.1674 | 0.5000 | 0.1250 | 0.1252 | 0.1131 |
| Regression-8-cont-v0 | prediction | 0.5962 | 1.0000 | 0.1997 | 0.8196 | 0.4955 |
| Regression-16-cont-v0 | prediction | 0.6796 | 1.0000 | 0.2000 | 0.8357 | 0.5995 |
| Repeat-4-cont-v0 | memory | 0.4511 | 1.0000 | 0.2500 | 0.2504 | 0.2682 |
| Repeat-32-cont-v0 | memory | 0.2648 | 1.0000 | 0.2502 | 0.2508 | 0.0194 |
| Ring-2-cont-v0 | prediction | 0.9375 | 1.0000 | 0.4996 | 0.9687 | 0.8751 |
| Ring-8-cont-v0 | prediction | 0.9858 | 1.0000 | 0.5002 | 0.8750 | 0.9717 |
| RiverSwim-6-cont-v0 | navigation | 0.1994 | 0.1998 | 0.0020 | 0.1330 | 0.9981 |
| RiverSwim-24-cont-v0 | navigation | 0.0047 | 0.1999 | 0.0018 | 0.0578 | 0.0147 |
| RockPaperScissors-10-cont-v0 | decision | 0.8757 | 1.0000 | 0.0002198 | 0.6841 | 0.8757 |
| RockPaperScissors-100-cont-v0 | decision | 0.8916 | 1.0000 | -5.326e-05 | 0.9609 | 0.8916 |
| ScentPuck-cont-v0 | control | -1.4474 | 3.2899 | -1.1858 | -0.5597 | -0.0585 |
| ScentPuck-competing-cont-v0 | control | -1.4944 | 3.2418 | -1.2196 | -0.4810 | -0.0616 |
| Sokoban-cont-v0 | navigation | 0.0011 | 0.1482 | 0.0007612 | 0.0006754 | 0.0020 |
| SubsetParity-8-1-cont-v0 | prediction | 0.9998 | 1.0000 | 0.4997 | 0.9998 | 0.9995 |
| SubsetParity-16-1-cont-v0 | prediction | 0.9997 | 1.0000 | 0.4998 | 0.9999 | 0.9995 |
| SubsetParity-8-2-cont-v0 | prediction | 0.8232 | 1.0000 | 0.5005 | 0.7858 | 0.6460 |
| SubsetParity-16-2-cont-v0 | prediction | 0.7557 | 1.0000 | 0.4998 | 0.7966 | 0.5116 |
| Taxi-cont-v0 | navigation | -1.0007 | 0.5721 | -3.9529 | -1.0011 | 0.6524 |
| Tiger-85-cont-v0 | decision | -0.1216 | 0.1081 | -3.0344 | -0.3726 | 0.9269 |
| Tiger-60-cont-v0 | decision | -0.1358 | -0.0814 | -3.0328 | -1.7254 | 0.9816 |
| Tracking-99-cont-v0 | control | -0.6027 | -0.0201 | -3.9784 | -0.4167 | 0.8528 |
| Tracking-999-cont-v0 | control | -0.3439 | -0.0024 | -4.0023 | -1.7480 | 0.9146 |
| Umbrella-4-cont-v0 | memory | 0.2506 | 0.2508 | 0.0008954 | 0.2504 | 0.9993 |
| Umbrella-16-cont-v0 | memory | 0.0397 | 0.0622 | -0.0002304 | 0.0571 | 0.6405 |
| Umbrella-64-cont-v0 | memory | -0.0005669 | 0.0153 | -0.0003533 | -0.0062 | -0.0137 |
| Umbrella-4-noise-cont-v0 | memory | 0.1613 | 0.2495 | -0.0004954 | 0.2489 | 0.6475 |
| Umbrella-16-noise-cont-v0 | memory | -0.0004706 | 0.0628 | 9.97e-05 | 0.0552 | -0.0091 |
| Umbrella-64-noise-cont-v0 | memory | 0.0003459 | 0.0164 | 0.000766 | -0.0046 | -0.0269 |
| WalkBet-cont-v0 | prediction | 0.8733 | 1.0000 | 0.5004 | 0.9944 | 0.7464 |
| WalkBet-drift-cont-v0 | prediction | 0.7135 | 1.0000 | 0.4998 | 0.9028 | 0.4272 |

Baseline throughput on the same schedule: PPO 4,527,081 steps/s, Random 130,648,945 steps/s, Oracle 24,373,675 steps/s.

Rewards match the reference recording (`019b414811583459f6931b87ecf61724d6e31d68`) bit for bit on all 4,915 lanes across three promotions and one adoption; throughput increased 28.7%, the product of three interleaved paired probes (1.2226, 1.0302, 1.0215). The last two probes timed each candidate against the source snapshot carried by the recording it replaced; the reference recording carries no snapshot, so the first probe's reference binary cannot be rebuilt from this repository. The adoption renamed MemoryChain to TMaze-32-1, DeepSea to DeepSea-8 and LargeDeepSea to DeepSea-16, moved the replay tape to the lane's raw state words, and refiled the baselines under the new environment digests; every refiled baseline matches its predecessor bit for bit. No unhealthy lanes.

![Policy comparison](oak-performance-radar.svg)

Regenerate with `python3 radar.py`.

## Setup and run

The engine is plain C. `./setup.sh --preset base` builds `.build/oak` with clang for Apple silicon, `./fleet.py` needs Python 3, and the theater needs Bun. Benchmark recordings stay local and are ignored by Git. `./fleet.py` records a candidate and judges it against the best, `./fleet.py adopt` makes the candidate the best without judging it, for intended schedule changes, and `cd frontend && bun install && bun run dev` serves the recordings and prints the URL to open. The theater replays any lane on demand through `.build/oak replay` and shows whether the replay reproduces the recorded reward stream bit for bit.

The judge promotes when the candidate is better on reward or throughput and worse on neither. Reward is compared on paired seeds: identical reward streams count as not worse with no uncertainty; otherwise the suite statistic is the interquartile mean over every environment and seed of the difference scaled by the best's per-environment interquartile range, with a stratified bootstrap CI, and a Holm-adjusted one-sided Wilcoxon test per environment guards against a single large regression. Throughput is an interleaved paired probe against the best rebuilt from the source snapshot in its recording, one percent above to count as better and within one percent to count as not worse; a byte-identical binary skips the probe. Random, PPO and Oracle baselines are filed under the environment digests, the baseline policy's source and the horizon, recorded once when missing and reused otherwise.

Geometry is agent capacity, declared in `Sources/geometry.h`; larger presets use more compute and memory. Wide streams increase percept and feature counts above these declared sizes. Total parameters = features × (1 + action dimensions); counts below use 6 action dimensions.

| Preset | Percepts | Features | Active per step | Total parameters |
| --- | ---: | ---: | ---: | ---: |
| base | 32 | 480 | 96 | 3,360 |
| pro | 64 | 960 | 192 | 6,720 |
| max | 128 | 1920 | 384 | 13,440 |
| ultra | 160 | 2400 | 480 | 16,800 |
