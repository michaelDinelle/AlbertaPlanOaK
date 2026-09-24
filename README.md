# OaK (Options and Knowledge)

![Eight OaK environments playing inside a pink butterfly](butterfly.png)

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) — Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski.

<img src="fc-stomp.png" alt="FC-STOMP cycle of discovery: state features define subtasks, learned options define models, and planning and feature use close the feedback loop." width="700">

Cycle of discovery from Richard S. Sutton's [Alberta Plan slides, slide 11](http://incompleteideas.net/Talks/AlbertaPlan.pdf#page=11).

Recorded on a binned M4 Max.

## Base

Geometry starts at 32 percepts, 480 features (96 active per step); 3,360 parameters (6 action dimensions). 4,464,909 SPS on the 245-environment schedule.

Reward per step over the final 100,928 steps of the 1,000,000-step best recording (`24536da670b12ad4dae6d71bebb433f5b0ab56c1`), all 245 environments with 32 seeds each (7,840 lanes), with the Oracle, Random and PPO baselines on the same seeds. Mean over seeds; Score = (OaK − Random) / (Oracle − Random), unclipped.

| Environment | Category | OaK | Oracle | Random | PPO | Score |
|---|---|---:|---:|---:|---:|---:|
| AccessControl-cont-v0 | decision | 2.7284 | 2.7480 | 1.6985 | 2.7397 | 0.9813 |
| AccessControl-drift-cont-v0 | decision | 1.5406 | 1.7138 | 0.9604 | 1.5938 | 0.7701 |
| Acrobot-cont-v0 | control | 0.0073 | 0.0110 | 0.0003926 | 0.0091 | 0.6478 |
| Acrobot-slip-cont-v0 | control | 0.0052 | 0.0076 | 0.0003889 | 0.0084 | 0.6636 |
| Acrobot-position-cont-v0 | control | 0.0022 | 0.0110 | 0.0003892 | 0.0009187 | 0.1675 |
| SymbolicAlchemy-fixed-cont-v0 | puzzles | 0.2640 | 0.9919 | 0.0311 | 0.4521 | 0.2425 |
| SymbolicAlchemy-rotation-cont-v0 | puzzles | 0.2521 | 0.8997 | 0.0314 | 0.2834 | 0.2542 |
| AliasedRing-8-cont-v0 | memory | 0.1537 | 0.1861 | -0.0056 | 0.0160 | 0.8310 |
| AliasedRing-32-cont-v0 | memory | 0.0274 | 0.0573 | -0.0386 | -2.198e-06 | 0.6885 |
| AliasedRing-128-cont-v0 | memory | 0.0026 | 0.0153 | -0.0471 | -6.533e-06 | 0.7961 |
| Ancients-1v1-cont-v0 | games | -0.000153 | 7.803e-05 | -0.000153 | -0.0001533 | 0 |
| Ancients-1v1-fog-cont-v0 | games | -0.0001347 | 5.295e-05 | -0.000135 | -0.0001328 | 0.0016 |
| Ancients-3v3-cont-v0 | games | -2.787e-05 | 1.115e-05 | -2.415e-05 | -2.539e-05 | -0.1053 |
| Ancients-5v5-cont-v0 | games | -9.289e-06 | 1.3e-05 | -9.289e-06 | -5.883e-06 | 0 |
| Ancients-5v5-league-cont-v0 | games | -1.424e-05 | 5.573e-06 | -1.362e-05 | -1.208e-05 | -0.0323 |
| Backgammon-random-cont-v0 | games | 9.134e-05 | 0.0011 | 5.573e-05 | 1.889e-05 | 0.0334 |
| Backgammon-pubeval-cont-v0 | games | -5.233e-05 | 0.0002263 | -0.0009221 | -1.858e-06 | 0.7573 |
| Backgammon-league-cont-v0 | games | -5.14e-05 | 0.0002341 | -0.000936 | -2.787e-06 | 0.7560 |
| Arm2-reach-cont-v0 | control | 0.0001744 | 0.0722 | 7.732e-07 | 0.000626 | 0.0024 |
| Arm2-payload-cont-v0 | control | 0.0001118 | 0.0576 | -1.959e-07 | 0.0007045 | 0.0019 |
| Bandit-3-cont-v0 | decision | 0.8297 | 0.8300 | -0.0862 | 0.8298 | 0.9997 |
| Bandit-6-cont-v0 | decision | 1.1710 | 1.1810 | -0.0162 | 1.1802 | 0.9916 |
| Bandit-3-drift-cont-v0 | decision | 0.9795 | 1.0084 | -0.0575 | 0.9926 | 0.9729 |
| Bandit-6-drift-cont-v0 | decision | 1.8646 | 1.9576 | 0.1414 | 1.8299 | 0.9488 |
| BlockingMaze-cont-v0 | navigation | 0.0008574 | 0.0628 | 0.0015 | 0.0009211 | -0.0103 |
| ShortcutMaze-cont-v0 | navigation | 0.0722 | 0.0997 | 0.0029 | 0.0636 | 0.7161 |
| Calibration-30-cont-v0 | decision | -0.0003502 | -2.983e-05 | -0.0419 | -2.052e-05 | 0.9923 |
| Calibration-80-cont-v0 | decision | -0.0002845 | -4.131e-05 | -0.0920 | -0.0112 | 0.9974 |
| CartPoleSwingUp-cont-v0 | control | -0.5616 | 1.0000 | -0.8162 | -0.6067 | 0.1402 |
| CartPoleSwingUp-slip-cont-v0 | control | -0.5685 | 0.9998 | -0.8160 | -0.6692 | 0.1363 |
| Catch-5x10-cont-v0 | control | 0.0656 | 0.1111 | -0.0666 | 0.1109 | 0.7436 |
| Catch-10x20-cont-v0 | control | -0.0327 | 0.0526 | -0.0422 | 0.0519 | 0.0998 |
| Catch-noise-cont-v0 | control | 0.0589 | 0.1113 | -0.0666 | 0.1103 | 0.7055 |
| Catch-scale-cont-v0 | control | -2.8813 | 11.1119 | -6.6718 | 11.0942 | 0.2131 |
| Cellular-cont-v0 | prediction | 0.9999 | 1.0000 | 0.5851 | 0.9997 | 0.9998 |
| Chess-kqk-cont-v0 | games | 3.096e-07 | 0.0213 | 0 | 0 | 1.451e-05 |
| Chess-krk-cont-v0 | games | 0 | 0.0146 | 3.096e-07 | 0 | -2.121e-05 |
| Chess-random-cont-v0 | games | 0 | 0.0044 | -9.289e-07 | 0 | 0.0002113 |
| Chess-alphabeta-cont-v0 | games | -1.239e-05 | 2.044e-05 | -1.362e-05 | -3.096e-07 | 0.0364 |
| Chain-cont-v0 | navigation | 0.0664 | 0.0667 | 0.0042 | 0.0667 | 0.9962 |
| DelayedChain-cont-v0 | navigation | 0.0320 | 0.0323 | 0.0010 | 0.0312 | 0.9920 |
| CliffWalk-cont-v0 | navigation | -35.0341 | 6.7695 | -10.1181 | -22.1345 | -1.4754 |
| CliffWalk-slip-cont-v0 | navigation | -5.9444 | 4.0655 | -10.1842 | -25.1221 | 0.2975 |
| CoinSwap-slow-cont-v0 | decision | 0.5939 | 0.9001 | 0.5000 | 0.5930 | 0.2348 |
| CoinSwap-fast-cont-v0 | decision | 0.6039 | 0.9000 | 0.5002 | 0.5015 | 0.2594 |
| TraceConditioning-10-cont-v0 | prediction | 0.0214 | 0.0334 | -0.0075 | -0.0002153 | 0.7072 |
| TraceConditioning-100-cont-v0 | prediction | -0.0002162 | 0.0033 | -0.0232 | -0.0002686 | 0.8662 |
| TracePatterning-cont-v0 | prediction | -0.0001918 | 0.0067 | -0.0215 | -0.0009357 | 0.7570 |
| ConnectFour-random-cont-v0 | games | 0.0620 | 0.0686 | 0.0102 | 0.0009314 | 0.8870 |
| ConnectFour-greedy-cont-v0 | games | -0.0001127 | 0.0419 | -0.0226 | -2.849e-05 | 0.3486 |
| ConnectFour-search-cont-v0 | games | -0.000958 | 0.0001477 | -0.0007626 | -7.245e-05 | -0.2146 |
| ConnectFour-league-cont-v0 | games | -0.0008896 | 0.0001932 | -0.0008016 | -0.0001601 | -0.0884 |
| Contextual-8-cont-v0 | decision | 0.9724 | 0.9947 | -0.0003318 | 0.9823 | 0.9776 |
| Contextual-16-cont-v0 | decision | 1.0120 | 1.0374 | -0.0007356 | 1.0194 | 0.9755 |
| Contextual-8-drift-cont-v0 | decision | 2.7575 | 2.9611 | -0.0011 | 2.8811 | 0.9313 |
| Contextual-16-drift-cont-v0 | decision | 3.7453 | 4.0621 | -0.0006166 | 3.9234 | 0.9220 |
| KeyCorridor-1-cont-v0 | navigation | 0.0614 | 0.1116 | 0.0037 | 0.0876 | 0.5355 |
| KeyCorridor-3-cont-v0 | navigation | 0.0045 | 0.0459 | 0.0007001 | 0.0236 | 0.0850 |
| CountRecall-8-cont-v0 | memory | 0.0523 | 0.1250 | 0.0313 | 0.0384 | 0.2236 |
| CountRecall-32-cont-v0 | memory | 0.0078 | 0.0312 | 0.0078 | 0.0078 | 0.0008577 |
| Covariate-fixed-cont-v0 | prediction | 0.9605 | 1.0000 | 0.5002 | 0.9948 | 0.9211 |
| Covariate-shift-cont-v0 | prediction | 0.7455 | 1.0000 | 0.5004 | 0.9955 | 0.4906 |
| Craftax-Classic-cont-v0 | world | 0.0099 | 0.0290 | 0.0089 | 0.0015 | 0.0477 |
| DarkKeyToDoor-9-cont-v0 | memory | 0.0160 | 0.0400 | 0.0051 | 0.0197 | 0.3129 |
| DeepSea-8-cont-v0 | navigation | 0.1422 | 0.1422 | 0.0004853 | 0.0951 | 0.9995 |
| DeepSea-16-cont-v0 | navigation | 0.0019 | 0.0663 | -0.0003098 | -1.964e-07 | 0.0332 |
| DeepSea-32-cont-v0 | navigation | -9.204e-05 | 0.0321 | -0.0001561 | -1.099e-07 | 0.0020 |
| DeepSea-16-slip-cont-v0 | navigation | -0.0001581 | 0.0306 | -0.0003114 | -3.302e-05 | 0.0050 |
| Eleven-drill-cont-v0 | games | 0 | 0.0029 | 3.716e-06 | 2.632e-05 | -0.0013 |
| Eleven-1v1-cont-v0 | games | -0.0002356 | 0.0009245 | -0.0003006 | -0.0002867 | 0.0531 |
| Eleven-3v3-cont-v0 | games | -0.0001508 | 0.0007995 | -0.0001824 | -0.0002378 | 0.0322 |
| Eleven-11v11-hard-cont-v0 | games | -2.291e-05 | 0.0003421 | -5.821e-05 | -4.087e-05 | 0.0882 |
| Eleven-11v11-league-cont-v0 | games | -8.081e-05 | 0.0003409 | -4.737e-05 | -2.88e-05 | -0.0861 |
| Dungeon-6-lit-cont-v0 | world | 0.0056 | 1.3378 | 0.0068 | 0.0041 | -0.0009468 |
| Dungeon-6-dark-cont-v0 | world | 0.0063 | 1.3570 | 0.0072 | 0.0039 | -0.0006955 |
| Dungeon-6-shuffle-cont-v0 | world | 0.0048 | 1.3311 | 0.0073 | 0.0045 | -0.0019 |
| Dungeon-9-shuffle-cont-v0 | world | 0.0060 | 1.7835 | 0.0070 | 0.0044 | -0.000598 |
| Forager-16-1-cont-v0 | world | 0.0180 | 0.0291 | 1.951e-05 | 0.0237 | 0.6178 |
| Forager-64-1-cont-v0 | world | 0.0156 | 0.0306 | 2.013e-05 | 0.0225 | 0.5102 |
| Forager-16-2-cont-v0 | world | 0.0149 | 0.0289 | -8.05e-06 | 0.0253 | 0.5154 |
| Forager-16-1-switch-cont-v0 | world | 0.0175 | 0.0292 | -0.0002124 | 0.0240 | 0.6011 |
| FourRoomsHall-cont-v0 | navigation | 0.1166 | 0.1270 | 0.0027 | 0.1243 | 0.9163 |
| FourRoomsHall-moving-cont-v0 | navigation | 0.0122 | 0.1266 | 0.0027 | 0.0159 | 0.0768 |
| 2048-cont-v0 | puzzles | 2.6843 | 19.6680 | 7.7888 | 7.2635 | -0.4297 |
| FourRooms-cont-v0 | navigation | 0.0069 | 0.1409 | 0.0027 | 0.1300 | 0.0305 |
| FrostHollow-fixed-cont-v0 | prediction | 0 | 0.0490 | 3.096e-07 | 0.0064 | -6.318e-06 |
| FrostHollow-random-cont-v0 | prediction | 6.193e-07 | 0.0421 | 0 | 0.0005143 | 1.472e-05 |
| FrostHollow-drift-cont-v0 | prediction | 3.096e-07 | 0.0420 | 1.239e-06 | 0.0037 | -2.21e-05 |
| Garnet-10-2-2-cont-v0 | decision | 0.7924 | 0.7995 | -0.0068 | 0.7296 | 0.9912 |
| Garnet-50-4-3-cont-v0 | decision | 1.0268 | 1.2191 | 0.0132 | 0.9199 | 0.8405 |
| Garnet-10-2-2-resample-cont-v0 | decision | 0.3541 | 0.7167 | 0.0291 | 0.1733 | 0.4726 |
| Garnet-50-4-3-resample-cont-v0 | decision | 0.2531 | 1.1752 | 0.0099 | 0.2068 | 0.2087 |
| Generals-2-12-cont-v0 | games | -0.0001975 | 0.0004855 | -0.0001948 | -1.641e-05 | -0.0041 |
| Generals-2-23-cont-v0 | games | -2.26e-05 | 0.0003697 | -1.889e-05 | -9.908e-06 | -0.0096 |
| Generals-4-28-cont-v0 | games | -2.384e-05 | 0.0002604 | -2.136e-05 | -1.672e-05 | -0.0088 |
| Generals-8-32-cont-v0 | games | -5.233e-05 | 0.0001471 | -4.087e-05 | -1.889e-05 | -0.0610 |
| Generals-8-32-league-cont-v0 | games | -3.839e-05 | 0.0001502 | -4.49e-05 | -2.105e-05 | 0.0333 |
| Hanabi-2p-literal-cont-v0 | games | 0.0411 | 0.1041 | 0.0030 | 0.0479 | 0.3768 |
| Hanabi-2p-focus-cont-v0 | games | 0.0001022 | 0.0973 | 0.000144 | 1.92e-05 | -0.0004302 |
| Hanabi-4p-focus-cont-v0 | games | 0.0211 | 0.0549 | 0.0079 | 0.0300 | 0.2810 |
| Hanoi-4-cont-v0 | puzzles | 0.000696 | 0.0667 | 0.0006087 | 3.096e-07 | 0.0013 |
| Hanoi-8-cont-v0 | puzzles | 9.289e-07 | 0.0039 | 9.289e-07 | 0 | 0 |
| Hanoi-8-remap-cont-v0 | puzzles | 6.193e-07 | 0.0039 | 1.548e-06 | 0 | -0.0002368 |
| HashedMDP-8-local-cont-v0 | decision | 0.7249 | 0.9549 | -0.0006109 | 0.7405 | 0.7593 |
| HashedMDP-16-local-cont-v0 | decision | 0.0050 | 1.8991 | -4.737e-05 | 0.1416 | 0.0027 |
| HashedMDP-8-global-cont-v0 | decision | 0.5753 | 0.7615 | 0.0016 | 0.5277 | 0.7550 |
| HashedMDP-16-global-cont-v0 | decision | 0.0037 | 0.7772 | 3.3e-05 | -0.0109 | 0.0047 |
| HiddenMode-slow-cont-v0 | decision | 0.6550 | 1.0000 | 0.5006 | 0.5162 | 0.3092 |
| HiddenMode-fast-cont-v0 | decision | 0.6242 | 1.0000 | 0.4999 | 0.5045 | 0.2485 |
| Homeostasis-2-cont-v0 | world | 0.0054 | 0.3000 | 0.0002062 | 0.0697 | 0.0174 |
| Homeostasis-2-hidden-cont-v0 | world | 0.0140 | 0.3000 | 7.617e-05 | 0.0234 | 0.0464 |
| JellyBeanWorld-cont-v0 | world | -5.449e-05 | 0.0386 | 1.92e-05 | -1.61e-05 | -0.0019 |
| JellyBeanWorld-8-cont-v0 | world | -3.189e-05 | 0.0383 | 7.431e-06 | 5.728e-05 | -0.0010 |
| KeyDoor-8-cont-v0 | memory | 0.1250 | 0.1250 | 0.0624 | 0.1250 | 1.0000 |
| KeyDoor-64-cont-v0 | memory | 0.0156 | 0.0156 | 0.0079 | 0.0156 | 0.9997 |
| KeyDoor-512-cont-v0 | memory | 0.0015 | 0.0019 | 0.0009639 | 0.0018 | 0.5906 |
| LightCycles-random-cont-v0 | games | 0.0223 | 0.0358 | -0.0925 | 0.0232 | 0.8945 |
| LightCycles-space-cont-v0 | games | -0.0069 | 0.0221 | -0.0967 | -0.0009704 | 0.7558 |
| LightCycles-minimax-cont-v0 | games | -0.0031 | 0.0013 | -0.0087 | -0.0016 | 0.5563 |
| LBF-8x8-2p-2f-cont-v0 | games | 0.0039 | 0.0460 | 0.0023 | 0.0188 | 0.0366 |
| LBF-8x8-2p-2f-coop-cont-v0 | games | 0.0026 | 0.0350 | 0.0017 | 0.0174 | 0.0266 |
| LBF-2s-8x8-2p-2f-coop-cont-v0 | games | 0.0020 | 0.0129 | 0.0009473 | 0.0063 | 0.0847 |
| LBF-10x10-3p-3f-pen-cont-v0 | games | -0.0001057 | 0.0058 | -0.0002949 | 0.0002257 | 0.0312 |
| ProceduralMaze-5-cont-v0 | navigation | 0.0005056 | 0.0731 | 0.0007815 | 3.096e-07 | -0.0038 |
| ProceduralMaze-15-cont-v0 | navigation | 6.193e-06 | 0.0102 | 9.598e-06 | 0 | -0.000334 |
| POMaze-cont-v0 | navigation | 0.0070 | 0.1066 | 0.0016 | 0.0336 | 0.0518 |
| HighDimMaze-cont-v0 | navigation | 0.0119 | 0.1393 | 0.0021 | 0.0795 | 0.0715 |
| MicroRTS-8-random-cont-v0 | games | -8.329e-05 | 0.0002517 | -8.577e-05 | -7.617e-05 | 0.0073 |
| MicroRTS-8-workerrush-cont-v0 | games | -0.0003551 | 5.945e-05 | -0.000344 | -0.0003387 | -0.0276 |
| MicroRTS-16-lightrush-cont-v0 | games | -0.0001087 | 4.954e-05 | -0.0001074 | -0.0001012 | -0.0079 |
| MicroRTS-16-fog-cont-v0 | games | -4.335e-05 | 4.459e-05 | -4.118e-05 | -3.375e-05 | -0.0253 |
| MicroRTS-16-league-cont-v0 | games | -4.799e-05 | 5.852e-05 | -4.954e-05 | -4.025e-05 | 0.0143 |
| TMaze-4-1-cont-v0 | memory | 0.2403 | 0.2500 | 0.0386 | 0.1120 | 0.9539 |
| TMaze-16-1-cont-v0 | memory | 0.0489 | 0.0625 | 0.0021 | 0.0282 | 0.7744 |
| TMaze-32-1-cont-v0 | memory | 0.0244 | 0.0312 | 0.0005106 | 0.0142 | 0.7771 |
| TMaze-64-1-cont-v0 | memory | 0.0092 | 0.0156 | 0.0001186 | 0.0063 | 0.5834 |
| TMaze-16-2-cont-v0 | memory | 0.0256 | 0.0625 | 0.0010 | 0.0015 | 0.4002 |
| TMaze-64-2-cont-v0 | memory | 0.0033 | 0.0156 | 5.976e-05 | 0.000322 | 0.2090 |
| MinAtar-Breakout-cont-v0 | control | 0.0717 | 0.0784 | 0.0388 | 0.0847 | 0.8287 |
| MinAtar-SpaceInvaders-cont-v0 | control | 0.1525 | 0.1904 | 0.0691 | 0.1393 | 0.6878 |
| MinAtar-Freeway-cont-v0 | control | 0.0109 | 0.0259 | 0.0001455 | 0.0106 | 0.4197 |
| MinAtar-Asterix-cont-v0 | control | 0.0086 | 0.0799 | 0.0076 | 0.0069 | 0.0126 |
| MinAtar-Seaquest-cont-v0 | control | 0.0082 | 0.0445 | 0.0033 | 0.0127 | 0.1174 |
| Minesweeper-9x9-10-cont-v0 | puzzles | -0.0001765 | 0.0181 | -0.0126 | 0 | 0.4048 |
| Minesweeper-16x16-40-cont-v0 | puzzles | -0.0002669 | 0.0045 | -0.0157 | -4.644e-06 | 0.7643 |
| Minesweeper-30x16-99-cont-v0 | puzzles | -0.0002917 | 0.0018 | -0.0196 | -7.431e-06 | 0.9006 |
| MountainCar-cont-v0 | control | 0.0063 | 0.0081 | 2.415e-05 | 0.0047 | 0.7860 |
| MountainCar-slip-cont-v0 | control | 0.0049 | 0.0069 | 1.982e-05 | 0.0046 | 0.7064 |
| MountainCar-position-cont-v0 | control | 0.0052 | 0.0081 | 2.446e-05 | 1.92e-05 | 0.6396 |
| Nexting-1-cont-v0 | prediction | 0.9898 | 1.0000 | 0.5002 | 0.9861 | 0.9797 |
| Nexting-10-cont-v0 | prediction | 0.9729 | 1.0000 | 0.4999 | 0.8955 | 0.9458 |
| Operant-16-cont-v0 | decision | 0.0609 | 0.0624 | -0.0157 | 0.0623 | 0.9810 |
| Operant-64-cont-v0 | decision | 0.0147 | 0.0155 | -0.0415 | 0.0155 | 0.9852 |
| OvercookedAI-cramped-cont-v0 | games | 0.4221 | 0.5336 | 0.3370 | 0.4769 | 0.4326 |
| OvercookedAI-asymmetric-cont-v0 | games | 0.3589 | 0.6884 | 0.3525 | 0.6894 | 0.0191 |
| OvercookedAI-ring-cont-v0 | games | 0.2171 | 0.5478 | 0.1236 | 0.3597 | 0.2203 |
| OvercookedAI-circuit-cont-v0 | games | 0.1504 | 0.4107 | 0.1764 | 0.2448 | -0.1107 |
| Parity-4-cont-v0 | memory | 0.5105 | 1.0000 | 0.5002 | 0.5000 | 0.0207 |
| Parity-64-cont-v0 | memory | 0.4996 | 1.0000 | 0.5003 | 0.5003 | -0.0012 |
| Password-2-cont-v0 | memory | 0.4427 | 0.5000 | 0.0499 | 0.4999 | 0.8726 |
| Password-4-cont-v0 | memory | 0.2417 | 0.2500 | 0.0030 | 0.2493 | 0.9665 |
| Password-6-cont-v0 | memory | 0.0797 | 0.1667 | 0.0001858 | 0.0674 | 0.4775 |
| Patience-5-cont-v0 | memory | 0.9999 | 1.0000 | 0.1306 | 0.9980 | 0.9998 |
| Patience-50-cont-v0 | memory | 0.0998 | 1.0002 | 0.0500 | 0.7982 | 0.0525 |
| Patience-500-cont-v0 | memory | 0.0998 | 1.0007 | 0.0500 | 0.2407 | 0.0524 |
| Pendulum-cont-v0 | control | -0.1298 | 0 | -3.8890 | -0.5678 | 0.9666 |
| Pendulum-slip-cont-v0 | control | -0.0839 | -0.0044 | -3.8906 | -0.0019 | 0.9795 |
| Pendulum-position-cont-v0 | control | -0.4406 | 0 | -3.9155 | -2.7175 | 0.8875 |
| Permuted-fixed-cont-v0 | prediction | 0.9997 | 1.0000 | 0.2500 | 0.9998 | 0.9996 |
| Permuted-recode-cont-v0 | prediction | 0.4080 | 1.0000 | 0.2499 | 0.4462 | 0.2107 |
| HalfCheetah-cont-v0 | control | 1.6990 | 5.7438 | -0.3103 | 4.6826 | 0.3319 |
| HalfCheetah-friction-cont-v0 | control | 1.2107 | 4.1504 | -0.2726 | 4.1929 | 0.3354 |
| Reacher-cont-v0 | control | -0.2149 | -0.0889 | -0.8574 | -0.3129 | 0.8360 |
| InvertedDoublePendulum-cont-v0 | control | 7.7927 | 9.3600 | 7.5127 | 9.2823 | 0.1516 |
| Hopper-cont-v0 | control | 1.7448 | 1.9359 | 0.7865 | 2.8700 | 0.8338 |
| PocketCube-3-cont-v0 | puzzles | 0.0243 | 0.3333 | 0.0007654 | 0.1645 | 0.0709 |
| PocketCube-random-cont-v0 | puzzles | 6.193e-07 | 0.0937 | 3.096e-07 | 0 | 3.303e-06 |
| KuhnPoker-random-cont-v0 | games | 0.4582 | 0.4217 | -0.0014 | 0.4588 | 1.0862 |
| LeducPoker-cfr-cont-v0 | games | -0.0379 | -0.0032 | -0.3451 | -0.0295 | 0.8985 |
| LeducPoker-fictitious-cont-v0 | games | -0.1088 | 0.5672 | -0.7174 | -0.1677 | 0.4738 |
| Pommerman-1v1-cont-v0 | games | -0.0007861 | 0.0040 | -0.0203 | -0.0012 | 0.8026 |
| Pommerman-FFA-cont-v0 | games | -0.0009221 | 0.0006635 | -0.0087 | -0.0007611 | 0.8311 |
| Pommerman-Team-cont-v0 | games | -0.0006546 | 0.0008679 | -0.0006474 | -0.0005858 | -0.0047 |
| Pommerman-FFA-collapse-cont-v0 | games | -0.0009038 | 0.0006688 | -0.0087 | -0.0007443 | 0.8326 |
| Pommerman-FFA-league-cont-v0 | games | -0.0005527 | 0.0002397 | -0.0086 | -0.0005093 | 0.9108 |
| PuddleWorld-cont-v0 | control | -0.3810 | 3.7150 | -4.6625 | 2.5397 | 0.5111 |
| Quadrotor2D-hover-cont-v0 | control | -19.1258 | -0.1508 | -16.5785 | -2.0091 | -0.1551 |
| Quadrotor2D-gust-cont-v0 | control | -19.2402 | -0.1545 | -16.6909 | -3.6021 | -0.1542 |
| Recite-4-cont-v0 | memory | 0.2624 | 0.5000 | 0.1251 | 0.1249 | 0.3663 |
| Recite-16-cont-v0 | memory | 0.1673 | 0.5000 | 0.1249 | 0.1252 | 0.1130 |
| Regression-8-cont-v0 | prediction | 0.5771 | 1.0000 | 0.1994 | 0.8064 | 0.4717 |
| Regression-16-cont-v0 | prediction | 0.7238 | 1.0000 | 0.2001 | 0.8832 | 0.6547 |
| RelicFog-scout-cont-v0 | games | 7.431e-06 | 0.0004555 | 7.121e-06 | 3.096e-06 | 0.0006906 |
| RelicFog-sapper-cont-v0 | games | 1.548e-06 | 0.0003669 | 9.289e-07 | 4.954e-06 | 0.0017 |
| RelicFog-league-cont-v0 | games | 9.908e-06 | 0.0003771 | 8.05e-06 | 7.431e-06 | 0.0050 |
| Repeat-4-cont-v0 | memory | 0.4517 | 1.0000 | 0.2502 | 0.2502 | 0.2687 |
| Repeat-32-cont-v0 | memory | 0.2646 | 1.0000 | 0.2500 | 0.2503 | 0.0195 |
| Reversi-6-greedy-cont-v0 | games | -2.013e-05 | 0.0128 | -0.0002019 | -6.193e-07 | 0.0140 |
| Reversi-8-swh-cont-v0 | games | -1.486e-05 | 0.0064 | -0.000226 | 0 | 0.0317 |
| Reversi-8-alphabeta-cont-v0 | games | -2.57e-05 | 0.0005874 | -0.0001629 | 0 | 0.1828 |
| Reversi-8-league-cont-v0 | games | -2.044e-05 | 0.0061 | -0.0002146 | 0 | 0.0307 |
| Ring-2-cont-v0 | prediction | 0.9383 | 1.0000 | 0.4996 | 0.9687 | 0.8768 |
| Ring-8-cont-v0 | prediction | 0.9893 | 1.0000 | 0.4999 | 0.8750 | 0.9786 |
| RiverSwim-6-cont-v0 | navigation | 0.1999 | 0.2003 | 0.0020 | 0.1149 | 0.9980 |
| RiverSwim-24-cont-v0 | navigation | 0.0047 | 0.2000 | 0.0018 | 0.0540 | 0.0147 |
| RockPaperScissors-10-cont-v0 | decision | 0.8732 | 1.0000 | 0.0003616 | 0.6841 | 0.8731 |
| RockPaperScissors-100-cont-v0 | decision | 0.8908 | 1.0000 | 0.0001718 | 0.9609 | 0.8908 |
| RWARE-tiny-2ag-cont-v0 | games | 0.000414 | 0.0292 | 0.0001365 | 0.0004641 | 0.0095 |
| RWARE-tiny-4ag-cont-v0 | games | 0.0009905 | 0.0335 | 0.0002904 | 0.0022 | 0.0211 |
| RWARE-medium-4ag-hard-cont-v0 | games | 3.654e-05 | 0.0147 | 1.703e-05 | 3.994e-05 | 0.0013 |
| ScentPuck-cont-v0 | control | -1.4715 | 3.2952 | -1.1647 | -0.5906 | -0.0688 |
| ScentPuck-competing-cont-v0 | control | -1.4972 | 3.2448 | -1.2117 | -0.5196 | -0.0641 |
| SlimeVolley-evolved-cont-v0 | games | -0.0098 | 0.0036 | -0.0107 | -0.0049 | 0.0658 |
| SlimeVolley-position-cont-v0 | games | -0.0103 | 0.0036 | -0.0107 | -0.0112 | 0.0311 |
| SlimeVolley-league-cont-v0 | games | -0.0089 | 0.0037 | -0.0096 | -0.0060 | 0.0507 |
| Skirmish-5v5-cont-v0 | games | -0.0009667 | 0.0015 | -0.0016 | -0.0008831 | 0.2015 |
| Skirmish-5v5-random-cont-v0 | games | -0.0007412 | 0.0022 | -0.0015 | -0.0003465 | 0.2154 |
| Skirmish-10v11-cont-v0 | games | -0.0014 | 0.0015 | -0.0021 | -0.0013 | 0.2088 |
| Skirmish-20v23-cont-v0 | games | -0.0010 | 0.0002486 | -0.0013 | -0.0007954 | 0.1518 |
| Skirmish-10v10-league-cont-v0 | games | -0.0004663 | 0.0014 | -0.0007985 | -0.0002951 | 0.1522 |
| Sokoban-cont-v0 | navigation | 0.0011 | 0.1482 | 0.0007614 | 0.0006954 | 0.0020 |
| Sokoban-7x7-2-cont-v0 | navigation | -0.0906 | 0.9871 | -0.0912 | -0.0981 | 0.0004795 |
| Sokoban-10x10-4-cont-v0 | navigation | -0.0952 | 0.3368 | -0.0958 | -0.0982 | 0.0015 |
| SubsetParity-8-1-cont-v0 | prediction | 0.9998 | 1.0000 | 0.5000 | 0.9999 | 0.9995 |
| SubsetParity-16-1-cont-v0 | prediction | 0.9997 | 1.0000 | 0.4999 | 0.9999 | 0.9995 |
| SubsetParity-8-2-cont-v0 | prediction | 0.8183 | 1.0000 | 0.5005 | 0.7814 | 0.6361 |
| SubsetParity-16-2-cont-v0 | prediction | 0.7546 | 1.0000 | 0.4998 | 0.8277 | 0.5094 |
| IDBD-20-5-cont-v0 | prediction | -197.8192 | 0 | -90.3129 | -7.7556 | -1.1904 |
| SlowRegression-20-15-cont-v0 | prediction | -550.7437 | 0 | -195.7283 | -1.7308 | -1.8138 |
| RateTracking-cont-v0 | prediction | -12.4388 | -3.5112 | -12.8362 | -5.4200 | 0.0426 |
| Taxi-cont-v0 | navigation | -1.0006 | 0.5714 | -3.9535 | -1.2143 | 0.6526 |
| Tetris-NES-A0-cont-v0 | puzzles | 0.0156 | 11.2609 | 5.48e-05 | 7.803e-05 | 0.0014 |
| Tetris-NES-A19-cont-v0 | puzzles | 0.0094 | 15.1577 | 0.0025 | 0.0020 | 0.0004545 |
| Tiger-85-cont-v0 | decision | -0.1311 | 0.1078 | -3.0314 | -0.3725 | 0.9239 |
| Tiger-60-cont-v0 | decision | -0.1398 | -0.0815 | -3.0324 | -1.7308 | 0.9802 |
| Tracking-99-cont-v0 | control | -0.5986 | -0.0201 | -3.9838 | -0.4238 | 0.8541 |
| Tracking-999-cont-v0 | control | -0.3393 | -0.0021 | -3.9729 | -1.7777 | 0.9151 |
| Umbrella-4-cont-v0 | memory | 0.2506 | 0.2508 | 0.00067 | 0.2503 | 0.9992 |
| Umbrella-16-cont-v0 | memory | 0.0347 | 0.0622 | -0.0003022 | 0.0572 | 0.5603 |
| Umbrella-64-cont-v0 | memory | -0.000779 | 0.0149 | -0.0007375 | -0.0065 | -0.0027 |
| Umbrella-4-noise-cont-v0 | memory | 0.1746 | 0.2502 | 0.0002756 | 0.2496 | 0.6976 |
| Umbrella-16-noise-cont-v0 | memory | 0.0016 | 0.0630 | 0.0005028 | 0.0553 | 0.0181 |
| Umbrella-64-noise-cont-v0 | memory | 0.0008676 | 0.0167 | 0.0010 | -0.0042 | -0.0109 |
| WalkBet-cont-v0 | prediction | 0.8738 | 1.0000 | 0.5000 | 0.9942 | 0.7475 |
| WalkBet-drift-cont-v0 | prediction | 0.7178 | 1.0000 | 0.4999 | 0.8967 | 0.4356 |

Baseline throughput, lane-weighted over the recordings that hold each environment's baseline: PPO 4,016,650 steps/s, Random 175,973,501 steps/s, Oracle 70,296,194 steps/s.

Every environment runs 32 seeds, from the start of its seed block, so each weighs the same in every suite statistic; 245 environments make 7,840 lanes. The latest adoption added Craftax-Classic and Reacher, each after a speed pass that changed no bit of its behaviour, and made the cost gate ratchet only environments whose sources changed. On all 7,776 lanes it shares with the previous best, rewards match bit for bit, and so does every refiled baseline; its speed gate against the recording that began the environment program measured 1.1227. The adoption before it added Reversi, chess, Kuhn and Leduc poker, backgammon, LightCycles, SlimeVolley, Pommerman, Overcooked-AI, LBF, RWARE and Hanabi; HalfCheetah-friction, InvertedDoublePendulum and Hopper on the planar physics core, Quadrotor2D and Arm2; Eleven, Skirmish, MinAtar, Dungeon and RelicFog; and IDBD, SlowRegression, RateTracking, FrostHollow, SymbolicAlchemy, DarkKeyToDoor and Homeostasis. It also made Contextual, Regression and HalfCheetah faster without changing a bit of their behaviour, and a Tetris game now ends at level 155. On all 5,248 lanes it shares with the previous best, rewards match bit for bit, and so does every refiled baseline. Its speed gate against the recording that began the environment program (`4262868f4bce5ea25ac4905e60742f75d11be371`) measured 1.1297 over three interleaved paired probes. The adoption before that restored KeyCorridor-1, KeyCorridor-3, ProceduralMaze-5, POMaze, HighDimMaze, FourRooms, DelayedChain and HalfCheetah exactly as the suite once ran them, added ProceduralMaze-15, two Sokoban rungs and the Ancients, Generals, Hanoi, PocketCube, 2048, Tetris, Minesweeper and ConnectFour families, and aligned every function to the cache line so each environment's speed no longer depends on the code linked around it; it matched bit for bit on all 4,096 shared lanes and measured 1.1586. The adoption before those moved to the 32-seed schedule and added the MicroRTS family and the engine's league and replay-view hooks, matching bit for bit on all 3,120 shared lanes and measuring 1.1544 against the same recording. The promotion before those rewrote every environment for speed without changing a bit of its behaviour, cutting the lane-weighted environment step from 58.8 ns to 10.7 ns, and moved work only the oracle reads out of environment resets; it measured 1.1955 against the best it replaced, on probes of 20,000 steps per lane, where those resets weigh more than in a full recording. From the reference recording (`019b414811583459f6931b87ecf61724d6e31d68`) to that promotion, rewards matched bit for bit on all 4,915 lanes of the earlier schedule across four promotions and two adoptions, and throughput rose 53.4%, the product of five interleaved paired probes (1.2226, 1.0302, 1.0215, 0.9976, 1.1955). No unhealthy lanes.

![Policy comparison](oak-performance-radar.svg)

Regenerate with `python3 radar.py`.

## Setup and run

The engine is plain C. `./setup.sh --preset base` builds `.build/oak` with clang for Apple silicon, `./fleet.py` needs Python 3, and the theater needs Bun. Benchmark recordings stay local and are ignored by Git. `./fleet.py` records a candidate and judges it against the best, `./fleet.py adopt` makes the candidate the best without judging it, for intended schedule changes, and `cd frontend && bun install && bun run dev` serves the recordings and prints the URL to open. The theater replays any lane on demand through `.build/oak replay` and shows whether the replay reproduces the recorded reward stream bit for bit.

The judge promotes when the candidate is better on reward or throughput and worse on neither. Reward is compared on paired seeds: identical reward streams count as not worse with no uncertainty; otherwise the suite statistic is the interquartile mean over every environment and seed of the difference scaled by the best's per-environment interquartile range, with a stratified bootstrap CI, and a Holm-adjusted one-sided Wilcoxon test per environment guards against a single large regression. Throughput is an interleaved paired probe against the best rebuilt from the source snapshot in its recording, one percent above to count as better and within one percent to count as not worse; a byte-identical binary skips the probe. Every judgment and adoption also times each environment's step alone with `.build/oak cost`, interleaved against the best's rebuilt binary: no environment whose sources changed may get slower than noise allows (an unchanged environment that moves is reported as layout), a new environment above 62 ns is refused unless it is one of the named exceptions in `fleet.py`, and the lane-weighted mean over the environments both binaries share may not rise. An adoption must also stay within one percent of the throughput of the recording that began the environment program (`4262868f4bce5ea25ac4905e60742f75d11be371`). Random, PPO and Oracle baselines are filed per environment under that environment's digest and seeds, the baseline policy's source and the horizon, recorded once when missing and reused otherwise.

Geometry is agent capacity, declared in `Sources/geometry.h`; larger presets use more compute and memory. Wide streams increase percept and feature counts above these declared sizes. Total parameters = features × (1 + action dimensions); counts below use 6 action dimensions.

| Preset | Percepts | Features | Active per step | Total parameters |
| --- | ---: | ---: | ---: | ---: |
| base | 32 | 480 | 96 | 3,360 |
| pro | 64 | 960 | 192 | 6,720 |
| max | 128 | 1920 | 384 | 13,440 |
| ultra | 160 | 2400 | 480 | 16,800 |
