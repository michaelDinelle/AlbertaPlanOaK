#pragma METAL fp math_mode(safe)

inline void mountainCar_obs(float pos, float vel, thread float *obs) {
    obs[0] = pos; obs[1] = vel; obs[2] = 1.0f;
    for (int i = 3; i < 17; i++) obs[i] = 0.0f;
}

inline float mountainCar_step(thread uint *words, uint action, thread float *obs,
                          thread uint *doneOut) {
    float pos = as_type<float>(words[0]), vel = as_type<float>(words[1]);
    float force = float(int(action % 3u)) - 1.0f;
    vel += force * 0.001f - 0.0025f * cos(3.0f * pos);
    vel = clamp(vel, -0.07f, 0.07f);
    pos += vel;
    pos = clamp(pos, -1.2f, 0.6f);
    if (pos == -1.2f && vel < 0.0f) vel = 0.0f;
    float reward = 0.0f;
    uint done = 0u;
    if (pos >= 0.5f) {
        reward = 1.0f;
        done = 1u;
        pos = -0.5f; vel = 0.0f;
    }
    words[0] = as_type<uint>(pos); words[1] = as_type<uint>(vel);
    mountainCar_obs(pos, vel, obs);
    *doneOut = done;
    return reward;
}
