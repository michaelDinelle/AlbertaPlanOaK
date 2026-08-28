#pragma METAL fp math_mode(safe)

inline uint chainWalk_move(uint pos, uint action) {
    if ((action & 1u) != 0u) return pos + 1u;
    return (pos > 0u) ? pos - 1u : 0u;
}

inline float chainWalk_advance(uint pos, uint action, uint limit,
                               thread uint *posOut, thread uint *doneOut) {
    float reward = 0.0f;
    uint done = 0u;
    pos = chainWalk_move(pos, action);
    if ((action & 1u) != 0u && pos >= limit) { reward = 1.0f; pos = 0u; done = 1u; }
    *posOut = pos;
    *doneOut = done;
    return reward;
}

inline float chain_step(device uint *words, uint action, thread float *obs,
                           thread uint *doneOut) {
    uint pos;
    float reward = chainWalk_advance(words[0], action, 15u, &pos, doneOut);
    words[0] = pos;
    obs[0] = float(pos) / 16.0f;
    obs[1] = 1.0f;
    return reward;
}
