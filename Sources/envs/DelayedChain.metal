#pragma METAL fp math_mode(safe)

inline float delayedChain_step(device uint *words, uint action, thread float *obs,
                            thread uint *doneOut) {
    uint pos;
    float reward = chainWalk_advance(words[0], action, 31u, &pos, doneOut);
    uint nextLastAction = action & 1u;
    words[0] = pos;
    words[1] = nextLastAction;
    obs[0] = float(pos) / 32.0f;
    obs[1] = float(nextLastAction);
    obs[2] = 1.0f;
    return reward;
}
