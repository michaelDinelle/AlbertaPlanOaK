#pragma METAL fp math_mode(safe)

inline void pendulum_obs(float theta, float thetaDot, thread float *obs) {
    obs[0] = cos(theta); obs[1] = sin(theta); obs[2] = thetaDot; obs[3] = 1.0f;
}

inline float pendulum_step(thread uint *words, float control, thread float *obs) {
    float theta = as_type<float>(words[0]), thetaDot = as_type<float>(words[1]);
    float torque = clamp(2.0f * control, -2.0f, 2.0f);
    float thetaWrapped = wrapPi(theta);
    float cost = thetaWrapped * thetaWrapped + 0.1f * thetaDot * thetaDot + 0.001f * torque * torque;
    float nextThetaDot = thetaDot + (15.0f * sin(theta + M_PI_F) + 3.0f * torque) * 0.05f;
    theta = theta + nextThetaDot * 0.05f;
    nextThetaDot = clamp(nextThetaDot, -8.0f, 8.0f);
    theta = wrapPi(theta);
    words[0] = as_type<uint>(theta); words[1] = as_type<uint>(nextThetaDot);
    pendulum_obs(theta, nextThetaDot, obs);
    return -cost;
}
