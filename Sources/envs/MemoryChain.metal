#pragma METAL fp math_mode(safe)

inline void memoryChain_obs(uint pos, uint cue, uint lastAction, thread float *obs) {
    obs[0] = float(pos) / 32.0f;
    obs[1] = (pos == 0u) ? float(cue) : 0.0f;
    obs[2] = (pos == 0u) ? 1.0f : 0.0f;
    obs[3] = (pos == 31u) ? 1.0f : 0.0f;
    obs[4] = float(lastAction);
    obs[5] = 1.0f;
    for (int i = 6; i < 17; i++) obs[i] = 0.0f;
}

inline float memoryChain_step(thread uint *words, uint laneSeed, uint action,
                            thread float *obs, thread uint *doneOut) {
    uint pos = words[0], cue = words[1], resampleIndex = words[3];
    float reward = 0.0f;
    uint done = 0u;
    if (pos >= 31u) {
        reward = ((action & 1u) == cue) ? 1.0f : 0.0f;
        done = 1u;
        uint4 raw = philox4x32(uint2(laneSeed, ENV_RNG_STREAM),
                                  uint4(resampleIndex, 0u, 0u, 0u));
        cue = raw.x & 1u;
        resampleIndex = resampleIndex + 1u;
        pos = 0u;
    } else if ((action & 1u) != 0u) {
        pos += 1u;
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    uint nextLastAction = action & 1u;
    words[0] = pos; words[1] = cue; words[2] = nextLastAction; words[3] = resampleIndex;
    memoryChain_obs(pos, cue, nextLastAction, obs);
    *doneOut = done;
    return reward;
}
