# OaK (Options and Knowledge)

![Eight OaK environments playing inside a pink butterfly](butterfly.png)

![Policy comparison](oak-performance-radar.svg)

[The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173) — Richard S. Sutton, Michael Bowling, and Patrick M. Pilarski.

<img src="fc-stomp.png" alt="FC-STOMP cycle of discovery: state features define subtasks, learned options define models, and planning and feature use close the feedback loop." width="700">

Cycle of discovery from Richard S. Sutton's [Alberta Plan slides, slide 11](http://incompleteideas.net/Talks/AlbertaPlan.pdf#page=11).

Recorded on a binned M4 Max.

## Best recording

Capacity 64, declared once in `Sources/geometry.h`: standardized observations on a seven-level trace ladder, 64 learned units of fan-in 32, and per step 23 hashed tuple cells plus two cells per observation channel on a 16,384-slot table, feeding three actor heads and five table arms that a Thompson-sampling bandit chooses among every 512 steps, plus an exact memory of the last 256 observations that the Q-cell arm consults where its cells leave the choice undecided; 6,197,248 bytes of state per lane. 6,146,915 SPS on the 245-environment schedule.

Reward per step over the final 100,928 steps of the 1,000,000-step best recording (`2d6e2ec1051d0d2d6a1b26d5db3fb9d60cfa0033`), all 245 environments with 32 seeds each (7,840 lanes), with the Oracle, PPO, constant do-nothing and Random baselines on the same seeds. Mean over seeds; Score = (OaK − Floor) / (Oracle − Floor) with Floor = max(Random, Constant), unclipped; a dash where the Oracle does not rise above the floor.

| Environment | Category | OaK | Oracle | PPO | Constant | Random | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| AccessControl-cont-v0 | decision | 2.7367 | 2.7480 | 2.7397 | 0 | 1.6985 | 0.9892 |
| AccessControl-drift-cont-v0 | decision | 1.5821 | 1.7138 | 1.5938 | 0 | 0.9604 | 0.8252 |
| Acrobot-cont-v0 | control | 0.0108 | 0.0137 | 0.0111 | 0 | 0.0004948 | 0.7792 |
| Acrobot-slip-cont-v0 | control | 0.0090 | 0.0093 | 0.0093 | 0.0006001 | 0.0004805 | 0.9590 |
| Acrobot-position-cont-v0 | control | 0.0094 | 0.0080 | 0.0021 | 0 | 0.0004685 | 1.1914 |
| SymbolicAlchemy-fixed-cont-v0 | puzzles | 0.4227 | 0.9919 | 0.4521 | 0 | 0.0311 | 0.4076 |
| SymbolicAlchemy-rotation-cont-v0 | puzzles | 0.3184 | 0.8997 | 0.2834 | 0 | 0.0314 | 0.3305 |
| AliasedRing-8-cont-v0 | memory | 0.1536 | 0.1861 | 0.0160 | 0 | -0.0056 | 0.8251 |
| AliasedRing-32-cont-v0 | memory | 0.0333 | 0.0573 | -2.198e-06 | 0 | -0.0386 | 0.5812 |
| AliasedRing-128-cont-v0 | memory | 0.0051 | 0.0153 | -6.533e-06 | 0 | -0.0471 | 0.3348 |
| Ancients-1v1-cont-v0 | games | -0.0001545 | 0.0001115 | -0.0001508 | -0.0001536 | -0.0001526 | -0.0070 |
| Ancients-1v1-fog-cont-v0 | games | -0.000135 | 8.577e-05 | -0.0001334 | -0.0001359 | -0.0001347 | -0.0014 |
| Ancients-3v3-cont-v0 | games | -3.127e-05 | 4.613e-05 | -2.787e-05 | -2.818e-05 | -2.353e-05 | -0.1111 |
| Ancients-5v5-cont-v0 | games | -1.146e-05 | 2.725e-05 | -1.022e-05 | -9.598e-06 | -9.289e-06 | -0.0593 |
| Ancients-5v5-league-cont-v0 | games | -1.208e-05 | 1.3e-05 | -9.598e-06 | -1.208e-05 | -1.331e-05 | 0 |
| Backgammon-random-cont-v0 | games | -0.0005276 | 0.0114 | 0.0055 | -0.0001232 | -2.818e-05 | -0.0438 |
| Backgammon-pubeval-cont-v0 | games | -0.0108 | 0.0024 | -0.0070 | -0.1050 | -0.0575 | 0.7809 |
| Backgammon-league-cont-v0 | games | -0.0106 | 0.0026 | -0.0067 | -0.1054 | -0.0576 | 0.7806 |
| Arm2-reach-cont-v0 | control | 0.0006854 | 0.0722 | 0.000626 | 0 | 7.732e-07 | 0.0095 |
| Arm2-payload-cont-v0 | control | 0.0006879 | 0.0576 | 0.0007045 | 0 | -1.959e-07 | 0.0119 |
| Bandit-3-cont-v0 | decision | 0.8294 | 0.8300 | 0.8298 | -0.0440 | -0.0862 | 0.9994 |
| Bandit-6-cont-v0 | decision | 1.1803 | 1.1810 | 1.1802 | 0.1068 | -0.0162 | 0.9994 |
| Bandit-3-drift-cont-v0 | decision | 0.9982 | 1.0084 | 0.9926 | 0.2175 | -0.0575 | 0.9871 |
| Bandit-6-drift-cont-v0 | decision | 1.9395 | 1.9576 | 1.8299 | 0.1584 | 0.1414 | 0.9900 |
| BlockingMaze-cont-v0 | navigation | 0.0283 | 0.0628 | 0.0009211 | 0 | 0.0015 | 0.4373 |
| ShortcutMaze-cont-v0 | navigation | 0.0625 | 0.0997 | 0.0636 | 0 | 0.0029 | 0.6159 |
| Calibration-30-cont-v0 | decision | -0.00012 | -2.983e-05 | -2.052e-05 | -0.0900 | -0.0419 | 0.9978 |
| Calibration-80-cont-v0 | decision | -0.0002524 | -4.131e-05 | -0.0112 | -0.6400 | -0.0920 | 0.9977 |
| CartPoleSwingUp-cont-v0 | control | -0.4135 | 1.0000 | -0.6067 | -0.6067 | -0.8162 | 0.1202 |
| CartPoleSwingUp-slip-cont-v0 | control | -0.4788 | 0.9998 | -0.6692 | -0.6706 | -0.8160 | 0.1148 |
| Catch-5x10-cont-v0 | control | 0.1111 | 0.1111 | 0.1109 | -0.0668 | -0.0666 | 1.0000 |
| Catch-10x20-cont-v0 | control | 0.0526 | 0.0526 | 0.0519 | -0.0422 | -0.0422 | 0.9997 |
| Catch-noise-cont-v0 | control | 0.1112 | 0.1113 | 0.1103 | -0.0666 | -0.0666 | 0.9998 |
| Catch-scale-cont-v0 | control | 11.1096 | 11.1119 | 11.0942 | -6.6675 | -6.6718 | 0.9999 |
| Cellular-cont-v0 | prediction | 0.9999 | 1.0000 | 0.9997 | 0.6249 | 0.5851 | 0.9998 |
| Chess-kqk-cont-v0 | games | 0.0001452 | 0.0119 | 5.542e-05 | 0.0001743 | 9.382e-05 | -0.0025 |
| Chess-krk-cont-v0 | games | 6.936e-05 | 0.0099 | 3.406e-05 | 9.041e-05 | 4.211e-05 | -0.0021 |
| Chess-random-cont-v0 | games | 1.208e-05 | 0.0020 | 4.83e-05 | -7.431e-06 | -7.741e-06 | 0.0095 |
| Chess-alphabeta-cont-v0 | games | -0.0030 | 0.0014 | -0.0033 | -0.0515 | -0.0271 | 0.8459 |
| Chain-cont-v0 | navigation | 0.0667 | 0.0667 | 0.0667 | 0 | 0.0042 | 1.0001 |
| DelayedChain-cont-v0 | navigation | 0.0323 | 0.0323 | 0.0312 | 0 | 0.0010 | 0.9998 |
| CliffWalk-cont-v0 | navigation | 6.7648 | 6.7695 | -22.1345 | -1.0000 | -10.1181 | 0.9994 |
| CliffWalk-slip-cont-v0 | navigation | 1.9798 | 4.0655 | -25.1221 | -1.0014 | -10.1842 | 0.5884 |
| CoinSwap-slow-cont-v0 | decision | 0.8947 | 0.9001 | 0.5930 | 0.4913 | 0.5000 | 0.9865 |
| CoinSwap-fast-cont-v0 | decision | 0.7029 | 0.9000 | 0.5015 | 0.4996 | 0.5002 | 0.5070 |
| TraceConditioning-10-cont-v0 | prediction | 0.0306 | 0.0334 | -0.0002153 | 0 | -0.0075 | 0.9169 |
| TraceConditioning-100-cont-v0 | prediction | -1.223e-06 | 0.0033 | -0.0002686 | 0 | -0.0232 | -0.0003662 |
| TracePatterning-cont-v0 | prediction | 0.0029 | 0.0067 | -0.0009357 | 0 | -0.0215 | 0.4299 |
| ConnectFour-random-cont-v0 | games | 0.0216 | 0.0467 | 0.0166 | 5.511e-05 | 0.0001585 | 0.4598 |
| ConnectFour-greedy-cont-v0 | games | -0.0155 | 0.0325 | -0.0161 | -0.1051 | -0.0531 | 0.4386 |
| ConnectFour-search-cont-v0 | games | -0.0284 | 0.0110 | -0.0258 | -0.1970 | -0.0994 | 0.6434 |
| ConnectFour-league-cont-v0 | games | -0.0274 | 0.0138 | -0.0270 | -0.1934 | -0.0977 | 0.6301 |
| Contextual-8-cont-v0 | decision | 0.9790 | 0.9947 | 0.9823 | -0.0009128 | -0.0003318 | 0.9842 |
| Contextual-16-cont-v0 | decision | 1.0167 | 1.0374 | 1.0194 | 0.0004531 | -0.0007356 | 0.9800 |
| Contextual-8-drift-cont-v0 | decision | 2.8852 | 2.9611 | 2.8811 | -0.0013 | -0.0011 | 0.9744 |
| Contextual-16-drift-cont-v0 | decision | 3.9372 | 4.0621 | 3.9234 | -0.0006462 | -0.0006166 | 0.9693 |
| KeyCorridor-1-cont-v0 | navigation | 0.1087 | 0.1116 | 0.0876 | 0 | 0.0037 | 0.9739 |
| KeyCorridor-3-cont-v0 | navigation | 0.0412 | 0.0459 | 0.0236 | 0 | 0.0007001 | 0.8952 |
| CountRecall-8-cont-v0 | memory | 0.1244 | 0.1250 | 0.0384 | 0.0243 | 0.0313 | 0.9941 |
| CountRecall-32-cont-v0 | memory | 0.0304 | 0.0312 | 0.0078 | 0.0079 | 0.0078 | 0.9616 |
| Covariate-fixed-cont-v0 | prediction | 0.9738 | 1.0000 | 0.9948 | 0.5133 | 0.5002 | 0.9462 |
| Covariate-shift-cont-v0 | prediction | 0.9594 | 1.0000 | 0.9955 | 0.4273 | 0.5004 | 0.9188 |
| Craftax-Classic-cont-v0 | world | -0.0060 | 0.0017 | -0.0044 | -0.0155 | -0.0106 | 0.3765 |
| DarkKeyToDoor-9-cont-v0 | memory | 0.0229 | 0.0400 | 0.0197 | 0.0002477 | 0.0051 | 0.5109 |
| DeepSea-8-cont-v0 | navigation | 0.1422 | 0.1422 | 0.0951 | 0 | 0.0004853 | 1.0000 |
| DeepSea-16-cont-v0 | navigation | 0.0145 | 0.0663 | -1.964e-07 | 0 | -0.0003098 | 0.2188 |
| DeepSea-32-cont-v0 | navigation | -8.707e-10 | 0.0321 | -1.099e-07 | 0 | -0.0001561 | -2.712e-08 |
| DeepSea-16-slip-cont-v0 | navigation | -3.132e-05 | 0.0306 | -3.302e-05 | -3.129e-05 | -0.0003114 | -1.035e-06 |
| Eleven-drill-cont-v0 | games | 0 | 0.0022 | 2.632e-05 | 0 | 3.716e-06 | -0.0017 |
| Eleven-1v1-cont-v0 | games | -0.0001681 | 0.0006703 | -0.0002867 | -0.0001065 | -0.0003006 | -0.0793 |
| Eleven-3v3-cont-v0 | games | -0.0001359 | 0.000605 | -0.0002378 | -9.134e-05 | -0.0001824 | -0.0640 |
| Eleven-11v11-hard-cont-v0 | games | -3.375e-05 | 0.0005412 | -4.087e-05 | -2.849e-05 | -5.821e-05 | -0.0092 |
| Eleven-11v11-league-cont-v0 | games | -0.000104 | 0.0005744 | -2.88e-05 | -8.174e-05 | -4.737e-05 | -0.0911 |
| Dungeon-6-lit-cont-v0 | world | 0.0128 | 1.0769 | 0.0053 | 0.0002105 | 0.0068 | 0.0056 |
| Dungeon-6-dark-cont-v0 | world | 0.0087 | 0.9169 | 0.0043 | 0.0001777 | 0.0072 | 0.0016 |
| Dungeon-6-shuffle-cont-v0 | world | 0.0087 | 0.8009 | 0.0032 | 0.000166 | 0.0073 | 0.0018 |
| Dungeon-9-shuffle-cont-v0 | world | 0.0093 | 1.0738 | 0.0038 | 0.0002127 | 0.0070 | 0.0021 |
| Forager-16-1-cont-v0 | world | 0.0240 | 0.0258 | 0.0164 | -1.053e-05 | 1.951e-05 | 0.9299 |
| Forager-64-1-cont-v0 | world | 0.0232 | 0.0254 | 0.0152 | 6.812e-06 | 2.013e-05 | 0.9101 |
| Forager-16-2-cont-v0 | world | 0.0258 | 0.0277 | 0.0175 | -1.3e-05 | -8.05e-06 | 0.9303 |
| Forager-16-1-switch-cont-v0 | world | 0.0086 | 0.0258 | 0.0022 | 3.468e-05 | -0.0002124 | 0.3310 |
| FourRoomsHall-cont-v0 | navigation | 0.1242 | 0.1270 | 0.1243 | 4.613e-05 | 0.0027 | 0.9777 |
| FourRoomsHall-moving-cont-v0 | navigation | 0.0158 | 0.1266 | 0.0159 | 5.666e-05 | 0.0027 | 0.1054 |
| 2048-cont-v0 | puzzles | 9.8727 | 19.6680 | 7.2635 | 0 | 7.7888 | 0.1754 |
| FourRooms-cont-v0 | navigation | 0.0904 | 0.1409 | 0.1300 | 0 | 0.0027 | 0.6350 |
| FrostHollow-fixed-cont-v0 | prediction | 0 | 0.0490 | 0.0064 | 0 | 3.096e-07 | -6.318e-06 |
| FrostHollow-random-cont-v0 | prediction | 9.289e-07 | 0.0421 | 0.0005143 | 0 | 0 | 2.207e-05 |
| FrostHollow-drift-cont-v0 | prediction | 0.0007341 | 0.0420 | 0.0037 | 0 | 1.239e-06 | 0.0174 |
| Garnet-10-2-2-cont-v0 | decision | 0.7978 | 0.7995 | 0.7296 | -0.0789 | -0.0068 | 0.9979 |
| Garnet-50-4-3-cont-v0 | decision | 1.2157 | 1.2191 | 0.9199 | 0.0979 | 0.0132 | 0.9970 |
| Garnet-10-2-2-resample-cont-v0 | decision | 0.6092 | 0.7167 | 0.1733 | 0.0900 | 0.0291 | 0.8285 |
| Garnet-50-4-3-resample-cont-v0 | decision | 0.7034 | 1.1752 | 0.2068 | 0.0469 | 0.0099 | 0.5819 |
| Generals-2-12-cont-v0 | games | -4.056e-05 | 0.0004778 | -1.889e-05 | -8.979e-06 | -0.0001948 | -0.0649 |
| Generals-2-23-cont-v0 | games | -2.044e-05 | 0.0003288 | -6.812e-06 | -1.548e-06 | -1.889e-05 | -0.0572 |
| Generals-4-28-cont-v0 | games | -2.446e-05 | 0.0001133 | -1.517e-05 | -1.084e-05 | -2.136e-05 | -0.1097 |
| Generals-8-32-cont-v0 | games | -3.189e-05 | 5.233e-05 | -2.229e-05 | -7.741e-06 | -4.087e-05 | -0.4021 |
| Generals-8-32-league-cont-v0 | games | -4.087e-05 | 5.14e-05 | -2.167e-05 | -1.672e-05 | -4.49e-05 | -0.3545 |
| Hanabi-2p-literal-cont-v0 | games | 0.1952 | 0.1868 | 0.1792 | 0.0007626 | 0.0077 | 1.0467 |
| Hanabi-2p-focus-cont-v0 | games | 0.0613 | 0.2127 | 0.0516 | 0.0008487 | 0.0005926 | 0.2852 |
| Hanabi-4p-focus-cont-v0 | games | 0.8293 | 0.3336 | 0.6067 | 0.0452 | 0.0831 | 2.9791 |
| Hanoi-4-cont-v0 | puzzles | 0.0667 | 0.0667 | 3.096e-07 | 0 | 0.0006087 | 1.0001 |
| Hanoi-8-cont-v0 | puzzles | 3.096e-07 | 0.0039 | 0 | 0 | 9.289e-07 | -0.0001579 |
| Hanoi-8-remap-cont-v0 | puzzles | 9.289e-07 | 0.0039 | 0 | 0 | 1.548e-06 | -0.0001579 |
| HashedMDP-8-local-cont-v0 | decision | 0.8775 | 0.9549 | 0.7405 | -0.0185 | -0.0006109 | 0.9190 |
| HashedMDP-16-local-cont-v0 | decision | 0.7979 | 1.8991 | 0.1416 | 0 | -4.737e-05 | 0.4202 |
| HashedMDP-8-global-cont-v0 | decision | 0.7242 | 0.7615 | 0.5277 | -0.0140 | 0.0016 | 0.9509 |
| HashedMDP-16-global-cont-v0 | decision | 0.0199 | 0.7772 | -0.0109 | -0.0020 | 3.3e-05 | 0.0255 |
| HiddenMode-slow-cont-v0 | decision | 0.9656 | 1.0000 | 0.5162 | 0.5034 | 0.5006 | 0.9307 |
| HiddenMode-fast-cont-v0 | decision | 0.8761 | 1.0000 | 0.5045 | 0.4988 | 0.4999 | 0.7523 |
| Homeostasis-2-cont-v0 | world | 0.0351 | 0.3000 | 0.0697 | 0 | 0.0002062 | 0.1163 |
| Homeostasis-2-hidden-cont-v0 | world | 0.0086 | 0.3000 | 0.0234 | 0 | 7.617e-05 | 0.0285 |
| JellyBeanWorld-cont-v0 | world | 2.601e-05 | 0.0081 | 4.83e-05 | 0 | -9.908e-06 | 0.0032 |
| JellyBeanWorld-8-cont-v0 | world | 6.193e-06 | 0.0083 | 9.444e-05 | 0 | -1.115e-05 | 0.00075 |
| KeyDoor-8-cont-v0 | memory | 0.1250 | 0.1250 | 0.1250 | 1.347e-06 | 0.0624 | 1.0000 |
| KeyDoor-64-cont-v0 | memory | 0.0156 | 0.0156 | 0.0156 | 8.612e-06 | 0.0079 | 0.9994 |
| KeyDoor-512-cont-v0 | memory | 0.0017 | 0.0019 | 0.0018 | -2.229e-05 | 0.0009639 | 0.7999 |
| LightCycles-random-cont-v0 | games | 0.0232 | 0.0328 | 0.0230 | -0.2500 | -0.0925 | 0.9236 |
| LightCycles-space-cont-v0 | games | -0.0034 | 0.0162 | -0.0010 | -0.2500 | -0.0972 | 0.8278 |
| LightCycles-minimax-cont-v0 | games | -0.0159 | -0.0002951 | -0.0125 | -0.2500 | -0.1025 | 0.8471 |
| LBF-8x8-2p-2f-cont-v0 | games | 0.0153 | 0.0460 | 0.0188 | 0 | 0.0023 | 0.2981 |
| LBF-8x8-2p-2f-coop-cont-v0 | games | 0.0102 | 0.0350 | 0.0174 | 0 | 0.0017 | 0.2559 |
| LBF-2s-8x8-2p-2f-coop-cont-v0 | games | 0.0078 | 0.0129 | 0.0063 | 0 | 0.0009473 | 0.5685 |
| LBF-10x10-3p-3f-pen-cont-v0 | games | 4.157e-05 | 0.0058 | 0.0002257 | 0 | -0.0002949 | 0.0072 |
| ProceduralMaze-5-cont-v0 | navigation | 0.0006149 | 0.0731 | 3.096e-07 | 0 | 0.0007815 | -0.0023 |
| ProceduralMaze-15-cont-v0 | navigation | 4.644e-06 | 0.0102 | 0 | 0 | 9.598e-06 | -0.0004858 |
| POMaze-cont-v0 | navigation | 0.0822 | 0.1066 | 0.0336 | 0 | 0.0016 | 0.7675 |
| HighDimMaze-cont-v0 | navigation | 0.0754 | 0.1393 | 0.0795 | 0 | 0.0021 | 0.5342 |
| MicroRTS-8-random-cont-v0 | games | -9.475e-05 | 0.0095 | -0.0001078 | -0.0006765 | -0.0003499 | 0.0259 |
| MicroRTS-8-workerrush-cont-v0 | games | -0.0004177 | 0.0023 | -0.0004793 | -0.0028 | -0.0033 | 0.4696 |
| MicroRTS-16-lightrush-cont-v0 | games | -0.000131 | 0.000535 | -0.0001437 | -0.0010 | -0.0007892 | 0.4971 |
| MicroRTS-16-fog-cont-v0 | games | -3.808e-05 | 3.065e-05 | -7.772e-05 | -0.0002378 | -0.000444 | 0.7439 |
| MicroRTS-16-league-cont-v0 | games | -5.016e-05 | 0.0003183 | -5.326e-05 | -0.0002861 | -0.0003341 | 0.3904 |
| TMaze-4-1-cont-v0 | memory | 0.2500 | 0.2500 | 0.1120 | 0 | 0.0386 | 0.9998 |
| TMaze-16-1-cont-v0 | memory | 0.0619 | 0.0625 | 0.0282 | 0 | 0.0021 | 0.9905 |
| TMaze-32-1-cont-v0 | memory | 0.0290 | 0.0312 | 0.0142 | 0 | 0.0005106 | 0.9279 |
| TMaze-64-1-cont-v0 | memory | 0.0123 | 0.0156 | 0.0063 | 0 | 0.0001186 | 0.7825 |
| TMaze-16-2-cont-v0 | memory | 0.0595 | 0.0625 | 0.0015 | 0 | 0.0010 | 0.9517 |
| TMaze-64-2-cont-v0 | memory | 0.0077 | 0.0156 | 0.000322 | 0 | 5.976e-05 | 0.4931 |
| MinAtar-Breakout-cont-v0 | control | 0.0815 | 0.0859 | 0.0965 | 0.0455 | 0.0388 | 0.8923 |
| MinAtar-SpaceInvaders-cont-v0 | control | 0.1846 | 0.1906 | 0.1365 | 0 | 0.0691 | 0.9514 |
| MinAtar-Freeway-cont-v0 | control | 0.0137 | 0.0259 | 0.0106 | 0 | 0.0001455 | 0.5277 |
| MinAtar-Asterix-cont-v0 | control | 0.0219 | 0.0459 | 0.0155 | 0.0051 | 0.0076 | 0.3712 |
| MinAtar-Seaquest-cont-v0 | control | 0.0081 | 0.0368 | 0.0126 | 0 | 0.0033 | 0.1414 |
| Minesweeper-9x9-10-cont-v0 | puzzles | -0.0031 | 0.0108 | -0.0052 | -0.0083 | -0.0191 | 0.2754 |
| Minesweeper-16x16-40-cont-v0 | puzzles | -0.0017 | 0.0012 | -0.0042 | -0.0045 | -0.0174 | 0.4866 |
| Minesweeper-30x16-99-cont-v0 | puzzles | -0.0016 | -0.0014 | -0.0036 | -0.0032 | -0.0199 | 0.8964 |
| MountainCar-cont-v0 | control | 0.0068 | 0.0081 | 0.0047 | 0 | 2.415e-05 | 0.8458 |
| MountainCar-slip-cont-v0 | control | 0.0061 | 0.0069 | 0.0046 | 0 | 1.982e-05 | 0.8944 |
| MountainCar-position-cont-v0 | control | 0.0019 | 0.0081 | 1.92e-05 | 0 | 2.446e-05 | 0.2372 |
| Nexting-1-cont-v0 | prediction | 0.9901 | 1.0000 | 0.9861 | 0.9900 | 0.5002 | 0.0098 |
| Nexting-10-cont-v0 | prediction | 0.9886 | 1.0000 | 0.8955 | 0.9000 | 0.4999 | 0.8861 |
| Operant-16-cont-v0 | decision | 0.0623 | 0.0624 | 0.0623 | 0 | -0.0157 | 0.9984 |
| Operant-64-cont-v0 | decision | 0.0155 | 0.0155 | 0.0155 | 0 | -0.0415 | 0.9949 |
| OvercookedAI-cramped-cont-v0 | games | 0.4386 | 0.5079 | 0.4734 | 0.3428 | 0.3370 | 0.5806 |
| OvercookedAI-asymmetric-cont-v0 | games | 0.3754 | 0.6885 | 0.6838 | 0.3582 | 0.3525 | 0.0523 |
| OvercookedAI-ring-cont-v0 | games | 0.3294 | 0.5128 | 0.3645 | 0 | 0.1236 | 0.5289 |
| OvercookedAI-circuit-cont-v0 | games | 0.2384 | 0.4128 | 0.2440 | 0 | 0.1764 | 0.2624 |
| Parity-4-cont-v0 | memory | 0.9977 | 1.0000 | 0.5000 | 0.5003 | 0.5002 | 0.9953 |
| Parity-64-cont-v0 | memory | 0.9985 | 1.0000 | 0.5003 | 0.4996 | 0.5003 | 0.9970 |
| Password-2-cont-v0 | memory | 0.4998 | 0.5000 | 0.4999 | 0.0312 | 0.0499 | 0.9996 |
| Password-4-cont-v0 | memory | 0.2500 | 0.2500 | 0.2493 | 0 | 0.0030 | 1.0000 |
| Password-6-cont-v0 | memory | 0.1667 | 0.1667 | 0.0674 | 0 | 0.0001858 | 1.0000 |
| Patience-5-cont-v0 | memory | 0.9999 | 1.0000 | 0.9980 | 0.1000 | 0.1306 | 0.9998 |
| Patience-50-cont-v0 | memory | 0.1927 | 1.0002 | 0.7982 | 0.1000 | 0.0500 | 0.1030 |
| Patience-500-cont-v0 | memory | 0.0999 | 1.0007 | 0.2407 | 0.1000 | 0.0500 | -0.0001284 |
| Pendulum-cont-v0 | control | -0.0025 | 0 | -0.5678 | -9.8696 | -3.8890 | 0.9994 |
| Pendulum-slip-cont-v0 | control | -0.0031 | -0.0044 | -0.0019 | -3.5988 | -3.8906 | 1.0004 |
| Pendulum-position-cont-v0 | control | -0.0117 | 0 | -2.7175 | -9.8696 | -3.9155 | 0.9970 |
| Permuted-fixed-cont-v0 | prediction | 1.0000 | 1.0000 | 0.9998 | 0.2619 | 0.2500 | 1.0000 |
| Permuted-recode-cont-v0 | prediction | 0.9479 | 1.0000 | 0.4462 | 0.2969 | 0.2499 | 0.9259 |
| HalfCheetah-cont-v0 | control | 4.0240 | 5.7777 | 4.8925 | 8.848e-09 | -0.3119 | 0.6965 |
| HalfCheetah-friction-cont-v0 | control | 3.8240 | 3.9123 | 4.8116 | 1.405e-09 | -0.2740 | 0.9774 |
| Reacher-cont-v0 | control | -0.1964 | -0.0889 | -0.3129 | -0.2344 | -0.8574 | 0.2612 |
| InvertedDoublePendulum-cont-v0 | control | 8.5105 | 9.3600 | 9.2823 | 8.2477 | 7.5127 | 0.2363 |
| Hopper-cont-v0 | control | 1.7177 | 1.9359 | 2.8700 | 0.9806 | 0.7865 | 0.7716 |
| PocketCube-3-cont-v0 | puzzles | 0.3290 | 0.3333 | 0.1645 | 0 | 0.0007654 | 0.9869 |
| PocketCube-random-cont-v0 | puzzles | 3.096e-07 | 0.0937 | 0 | 0 | 3.096e-07 | 0 |
| KuhnPoker-random-cont-v0 | games | 0.4588 | 0.4217 | 0.4588 | -0.4000 | -0.0014 | 1.0877 |
| LeducPoker-cfr-cont-v0 | games | -0.0296 | -0.0032 | -0.0295 | -0.3225 | -0.3451 | 0.9176 |
| LeducPoker-fictitious-cont-v0 | games | -0.0863 | 0.5672 | -0.1677 | -0.6667 | -0.7174 | 0.4704 |
| Pommerman-1v1-cont-v0 | games | -0.0047 | 0.0106 | -0.0069 | -0.0073 | -0.0405 | 0.1453 |
| Pommerman-FFA-cont-v0 | games | -0.0050 | 0.0045 | -0.0050 | -0.0057 | -0.0435 | 0.0760 |
| Pommerman-Team-cont-v0 | games | -0.0031 | 0.0038 | -0.0033 | -0.0035 | -0.0319 | 0.0510 |
| Pommerman-FFA-collapse-cont-v0 | games | -0.0050 | 0.0046 | -0.0051 | -0.0058 | -0.0433 | 0.0746 |
| Pommerman-FFA-league-cont-v0 | games | -0.0026 | 0.0006967 | -0.0027 | -0.0027 | -0.0432 | 0.0221 |
| PuddleWorld-cont-v0 | control | 1.9426 | 3.7150 | 2.5397 | -0.9907 | -4.6625 | 0.6234 |
| Quadrotor2D-hover-cont-v0 | control | -9.4484 | -0.1508 | -2.0091 | -2.2991 | -16.5785 | -3.3278 |
| Quadrotor2D-gust-cont-v0 | control | -10.5174 | -0.1545 | -3.6021 | -9.2524 | -16.6909 | -0.1390 |
| Recite-4-cont-v0 | memory | 0.3576 | 0.5000 | 0.1249 | 0.1252 | 0.1251 | 0.6201 |
| Recite-16-cont-v0 | memory | 0.1694 | 0.5000 | 0.1252 | 0.1251 | 0.1249 | 0.1181 |
| Regression-8-cont-v0 | prediction | 0.7273 | 1.0000 | 0.8064 | 0.3593 | 0.1994 | 0.5744 |
| Regression-16-cont-v0 | prediction | 0.7660 | 1.0000 | 0.8832 | 0.4033 | 0.2001 | 0.6078 |
| RelicFog-scout-cont-v0 | games | 1.796e-05 | 0.0007208 | 2.415e-05 | 5.388e-05 | 1.3e-05 | -0.0539 |
| RelicFog-sapper-cont-v0 | games | 1.548e-05 | 0.0006121 | 1.579e-05 | 4.118e-05 | 4.954e-06 | -0.0450 |
| RelicFog-league-cont-v0 | games | 4.273e-05 | 0.0006669 | 3.406e-05 | 9.289e-05 | 3.839e-05 | -0.0874 |
| Repeat-4-cont-v0 | memory | 0.5813 | 1.0000 | 0.2502 | 0.2503 | 0.2502 | 0.4415 |
| Repeat-32-cont-v0 | memory | 0.4199 | 1.0000 | 0.2503 | 0.2501 | 0.2500 | 0.2264 |
| Reversi-6-greedy-cont-v0 | games | 0.0073 | 0.0170 | 0.0055 | -0.0164 | -0.0089 | 0.6253 |
| Reversi-8-swh-cont-v0 | games | -0.0027 | 0.0050 | -0.0032 | -0.0205 | -0.0107 | 0.5097 |
| Reversi-8-alphabeta-cont-v0 | games | -0.0031 | -0.0002205 | -0.0035 | -0.0235 | -0.0123 | 0.7628 |
| Reversi-8-league-cont-v0 | games | -0.0032 | 0.0049 | -0.0032 | -0.0220 | -0.0113 | 0.5040 |
| Ring-2-cont-v0 | prediction | 0.9990 | 1.0000 | 0.9687 | 0.5000 | 0.4996 | 0.9980 |
| Ring-8-cont-v0 | prediction | 1.0000 | 1.0000 | 0.8750 | 0.5000 | 0.4999 | 1.0000 |
| RiverSwim-6-cont-v0 | navigation | 0.1206 | 0.2003 | 0.1149 | 0.0050 | 0.0020 | 0.5921 |
| RiverSwim-24-cont-v0 | navigation | 0.0050 | 0.2000 | 0.0540 | 0.0050 | 0.0018 | -6.436e-08 |
| RockPaperScissors-10-cont-v0 | decision | 0.9522 | 1.0000 | 0.6841 | -1.0000 | 0.0003616 | 0.9522 |
| RockPaperScissors-100-cont-v0 | decision | 0.9612 | 1.0000 | 0.9609 | -1.0000 | 0.0001718 | 0.9612 |
| RWARE-tiny-2ag-cont-v0 | games | 0.0010 | 0.0253 | 0.0004641 | 0 | 0.0001365 | 0.0345 |
| RWARE-tiny-4ag-cont-v0 | games | 0.0022 | 0.0278 | 0.0022 | 0 | 0.0002904 | 0.0692 |
| RWARE-medium-4ag-hard-cont-v0 | games | 6.131e-05 | 0.0064 | 3.994e-05 | 0 | 1.703e-05 | 0.0070 |
| ScentPuck-cont-v0 | control | -0.8699 | 3.2952 | -0.5906 | -1.1924 | -1.1647 | 0.0661 |
| ScentPuck-competing-cont-v0 | control | -0.8865 | 3.2448 | -0.5196 | -1.0619 | -1.2117 | 0.0407 |
| SlimeVolley-evolved-cont-v0 | games | -0.0098 | 0.0036 | -0.0049 | -0.0110 | -0.0107 | 0.0671 |
| SlimeVolley-position-cont-v0 | games | -0.0103 | 0.0036 | -0.0112 | -0.0110 | -0.0107 | 0.0296 |
| SlimeVolley-league-cont-v0 | games | -0.0089 | 0.0037 | -0.0060 | -0.0103 | -0.0096 | 0.0509 |
| Skirmish-5v5-cont-v0 | games | -0.0009951 | 0.0011 | -0.0011 | -0.0013 | -0.0018 | 0.1323 |
| Skirmish-5v5-random-cont-v0 | games | -0.0012 | 0.0018 | -0.0011 | -0.0014 | -0.0023 | 0.0909 |
| Skirmish-10v11-cont-v0 | games | -0.0017 | 0.0009397 | -0.0017 | -0.0017 | -0.0032 | 0.0131 |
| Skirmish-20v23-cont-v0 | games | -0.0011 | -1.734e-05 | -0.0012 | -0.0010 | -0.0019 | -0.1053 |
| Skirmish-10v10-league-cont-v0 | games | -0.0013 | 2.384e-05 | -0.0014 | -0.0011 | -0.0022 | -0.1534 |
| Sokoban-cont-v0 | navigation | 0.0024 | 0.1482 | 0.0006954 | 2.57e-05 | 0.0007614 | 0.0113 |
| Sokoban-7x7-2-cont-v0 | navigation | -0.0898 | 0.9871 | -0.0991 | -0.1000 | -0.0912 | 0.0012 |
| Sokoban-10x10-4-cont-v0 | navigation | -0.0950 | 0.3368 | -0.0993 | -0.1000 | -0.0958 | 0.0018 |
| SubsetParity-8-1-cont-v0 | prediction | 1.0000 | 1.0000 | 0.9999 | 0.5002 | 0.5000 | 1.0000 |
| SubsetParity-16-1-cont-v0 | prediction | 1.0000 | 1.0000 | 0.9999 | 0.5003 | 0.4999 | 1.0000 |
| SubsetParity-8-2-cont-v0 | prediction | 0.9996 | 1.0000 | 0.7814 | 0.5006 | 0.5005 | 0.9993 |
| SubsetParity-16-2-cont-v0 | prediction | 0.9890 | 1.0000 | 0.8277 | 0.5000 | 0.4998 | 0.9780 |
| IDBD-20-5-cont-v0 | prediction | -7.1664 | 0 | -7.7556 | -4.9979 | -90.3129 | -0.4339 |
| SlowRegression-20-15-cont-v0 | prediction | -5.2660 | 0 | -1.7308 | -3.7391 | -195.7283 | -0.4083 |
| RateTracking-cont-v0 | prediction | -4.5622 | -3.5112 | -5.4200 | -4.5116 | -12.8362 | -0.0506 |
| Taxi-cont-v0 | navigation | -1.0002 | 0.5714 | -1.2143 | -1.0000 | -3.9535 | -0.000149 |
| Tetris-NES-A0-cont-v0 | puzzles | 0.0062 | 11.2609 | 7.803e-05 | 0 | 5.48e-05 | 0.0005477 |
| Tetris-NES-A19-cont-v0 | puzzles | 0.0079 | 15.1577 | 0.0020 | 0 | 0.0025 | 0.0003606 |
| Tiger-85-cont-v0 | decision | -0.0078 | 0.1078 | -0.3725 | -4.4984 | -3.0314 | 0.9632 |
| Tiger-60-cont-v0 | decision | -0.1004 | -0.0815 | -1.7308 | -4.5000 | -3.0324 | 0.9936 |
| Tracking-99-cont-v0 | control | -0.2628 | -0.0201 | -0.4238 | -0.9840 | -3.9838 | 0.7483 |
| Tracking-999-cont-v0 | control | -0.1532 | -0.0021 | -1.7777 | -0.9748 | -3.9729 | 0.8446 |
| Umbrella-4-cont-v0 | memory | 0.2505 | 0.2508 | 0.2503 | 0.000361 | 0.00067 | 0.9988 |
| Umbrella-16-cont-v0 | memory | 0.0591 | 0.0622 | 0.0572 | -0.000309 | -0.0003022 | 0.9509 |
| Umbrella-64-cont-v0 | memory | -0.0006081 | 0.0149 | -0.0065 | -0.0007988 | -0.0007375 | 0.0083 |
| Umbrella-4-noise-cont-v0 | memory | 0.2413 | 0.2502 | 0.2496 | 0.0002632 | 0.0002756 | 0.9644 |
| Umbrella-16-noise-cont-v0 | memory | 0.0078 | 0.0630 | 0.0553 | 0.0005437 | 0.0005028 | 0.1162 |
| Umbrella-64-noise-cont-v0 | memory | 0.0006446 | 0.0167 | -0.0042 | 0.0012 | 0.0010 | -0.0367 |
| WalkBet-cont-v0 | prediction | 1.0000 | 1.0000 | 0.9942 | 0.5508 | 0.5000 | 0.9999 |
| WalkBet-drift-cont-v0 | prediction | 0.9934 | 1.0000 | 0.8967 | 0.9678 | 0.4999 | 0.7948 |

Baseline throughput, lane-weighted over the recordings that hold each environment's baseline: PPO 3,361,389 steps/s, Random 89,834,019 steps/s, Constant 78,065,119 steps/s, Oracle 13,697,803 steps/s.

Every environment runs 32 seeds, from the start of its seed block, so each weighs the same in every suite statistic; 245 environments make 7,840 lanes. The latest promotion changed no bit of behaviour: it draws the Gumbel noise that samples a discrete action for four actions at a time, and computes the trace moments, cell codes, cell hashes and control statistics for four channels at a time. Rewards match the best it replaced bit for bit on all 7,840 lanes, and its paired speed probe measured 1.0183. The promotion before it also changed no bit of behaviour: it standardizes the observation channels four at a time and runs the 23 tuple hash chains four at a time; its paired speed probe measured 1.0557. The promotion before that gave the Q-cell arm an exact memory: pair keys that join the current observation with the observations 2 to 128 steps back, and window keys that count exactly over the last 2 to 128 steps. The arm consults them only where its cells leave the choice undecided, so on seed 0 90 of the 245 environments kept every bit of their behaviour. The promotion also computes the 64 units' activations and learning signals four at a time. Parity-64 and CountRecall-32 went from dead to solved, and the memory category rose from 57.5 to 71.2. Against the best it replaced, its suite IQM difference was +0.005 (95% CI [+0.004, +0.008]), the per-environment regression guard was silent on all 245 environments, and its paired speed probe measured 1.0876. The promotion before those replaced the agent with this portfolio: three actor heads over one shared representation, owner-tagged exact tables keyed on observations and on observation history, a planner, a frontier explorer and a differential Q-learning arm over the tile-coded cells, which include settled trace cells and cells relating the channel the agent controls to every other channel. Against the best it replaced, its suite IQM difference was +1.294 (95% CI [+1.260, +1.328]), the per-environment regression guard was silent on all 245 environments, and its paired speed probe measured 1.0679. The last adoption before it added Craftax-Classic and Reacher, each after a speed pass that changed no bit of its behaviour. On all 7,776 lanes it shares with the previous best, rewards match bit for bit, and so does every refiled baseline; its speed gate against the recording that began the environment program measured 1.1227. The adoption before it added Reversi, chess, Kuhn and Leduc poker, backgammon, LightCycles, SlimeVolley, Pommerman, Overcooked-AI, LBF, RWARE and Hanabi; HalfCheetah-friction, InvertedDoublePendulum and Hopper on the planar physics core, Quadrotor2D and Arm2; Eleven, Skirmish, MinAtar, Dungeon and RelicFog; and IDBD, SlowRegression, RateTracking, FrostHollow, SymbolicAlchemy, DarkKeyToDoor and Homeostasis. It also made Contextual, Regression and HalfCheetah faster without changing a bit of their behaviour, and a Tetris game now ends at level 155. On all 5,248 lanes it shares with the previous best, rewards match bit for bit, and so does every refiled baseline. Its speed gate against the recording that began the environment program (`4262868f4bce5ea25ac4905e60742f75d11be371`) measured 1.1297 over three interleaved paired probes. The adoption before that restored KeyCorridor-1, KeyCorridor-3, ProceduralMaze-5, POMaze, HighDimMaze, FourRooms, DelayedChain and HalfCheetah exactly as the suite once ran them, added ProceduralMaze-15, two Sokoban rungs and the Ancients, Generals, Hanoi, PocketCube, 2048, Tetris, Minesweeper and ConnectFour families, and aligned every function to the cache line so each environment's speed no longer depends on the code linked around it; it matched bit for bit on all 4,096 shared lanes and measured 1.1586. The adoption before those moved to the 32-seed schedule and added the MicroRTS family and the engine's league and replay-view hooks, matching bit for bit on all 3,120 shared lanes and measuring 1.1544 against the same recording. The promotion before those rewrote every environment for speed without changing a bit of its behaviour, cutting the lane-weighted environment step from 58.8 ns to 10.7 ns, and moved work only the oracle reads out of environment resets; it measured 1.1955 against the best it replaced, on probes of 20,000 steps per lane, where those resets weigh more than in a full recording. From the reference recording (`019b414811583459f6931b87ecf61724d6e31d68`) to that promotion, rewards matched bit for bit on all 4,915 lanes of the earlier schedule across four promotions and two adoptions, and throughput rose 53.4%, the product of five interleaved paired probes (1.2226, 1.0302, 1.0215, 0.9976, 1.1955). No unhealthy lanes.

Regenerate with `python3 radar.py`.

## Setup and run

The engine is plain C. `./setup.sh` builds `.build/oak` with clang for Apple silicon, `./fleet.py` needs Python 3, and the theater needs Bun. Benchmark recordings stay local and are ignored by Git. `./fleet.py` queues the working tree for a background runner that builds it, times it against the best, records it when it can still promote, and judges it; `./fleet.py adopt` queues it to become the best without judging reward, for intended schedule changes; `./fleet.py probe` times the working tree against the best without recording and waits for the result. Verdicts land in `.oak-benchmarks/verdicts.log`, and `cd frontend && bun install && bun run dev` serves the recordings and prints the URL to open. The theater replays any lane on demand with the engine its recording carries and shows whether the replay reproduces the recorded reward stream bit for bit.

The judge promotes when the candidate is better on reward or throughput and worse on neither. Reward is compared on paired seeds: identical reward streams count as not worse with no uncertainty; otherwise the suite statistic is the interquartile mean over every environment and seed of the difference scaled by the best's per-environment interquartile range, with a stratified bootstrap CI, and a Holm-adjusted one-sided Wilcoxon test per environment guards against a single large regression. Throughput is an interleaved paired probe against the best rebuilt from the source snapshot in its recording, one percent above to count as better and within one percent to count as not worse. The speed gate runs before the recording: a candidate more than one percent slower is rejected without recording, and a byte-identical binary is rejected as having nothing to judge. An adoption must also stay within one percent of the throughput of the recording that began the environment program (`4262868f4bce5ea25ac4905e60742f75d11be371`). Random, PPO and Oracle baselines are filed per environment under that environment's digest and seeds, the baseline policy's source and the horizon, recorded once when missing and reused otherwise.

Geometry is agent capacity, declared once in `Sources/geometry.h`; every size below derives from it, and growing the agent means changing that one declaration and letting the judge decide.

| Quantity | Value |
| --- | ---: |
| Capacity | 64 |
| Learned units × fan-in | 64 × 32 |
| Trace ladder | 7 levels |
| Feature timescale and bandit block | 512 steps |
| Tuple cells per step | 23 |
| Observation cells per step | 2 per channel |
| Cell table | 16,384 slots |
| Exact tables | 2 × 16,384 rows |
| Actor heads and table arms | 3 and 5 |
| State per lane | 5,670,144 bytes |
