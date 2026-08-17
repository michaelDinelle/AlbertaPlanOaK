#pragma METAL fp math_mode(safe)

inline float chain_step(thread uint *words, uint action, thread float *obs,
                           thread uint *doneOut) {
    uint pos = words[0];
    float reward = 0.0f;
    uint done = 0u;
    if ((action & 1u) != 0u) {
        pos += 1u;
        if (pos >= 15u) { reward = 1.0f; pos = 0u; done = 1u; }
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    words[0] = pos;
    obs[0] = float(pos) / 16.0f;
    obs[1] = 1.0f;
    for (int i = 2; i < 17; i++) obs[i] = 0.0f;
    *doneOut = done;
    return reward;
}
