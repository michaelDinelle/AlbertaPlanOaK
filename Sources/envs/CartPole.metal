#pragma METAL fp math_mode(safe)

constant float CARTPOLE_DT = 0.02f;
constant float CARTPOLE_THETA_MAX = 0.2094395f;

inline void cartPole_obs(float cartPosition, float cartVelocity, float poleAngle, float poleAngularVelocity,
                        thread float *obs) {
    obs[0] = cartPosition; obs[1] = cartVelocity; obs[2] = poleAngle; obs[3] = poleAngularVelocity; obs[4] = 1.0f;
    for (int i = 5; i < 17; i++) obs[i] = 0.0f;
}

inline float cartPole_step(thread uint *words, uint action, thread float *obs,
                          thread uint *doneOut) {
    float cartPosition = as_type<float>(words[0]), cartVelocity = as_type<float>(words[1]);
    float poleAngle = as_type<float>(words[2]), poleAngularVelocity = as_type<float>(words[3]);
    float force = ((action & 1u) != 0u) ? 10.0f : -10.0f;
    float sinPole = sin(poleAngle), cosPole = cos(poleAngle);
    float forceOverTotalMass = (force + 0.05f * poleAngularVelocity * poleAngularVelocity * sinPole) / 1.1f;
    float poleAngularAcceleration = (9.8f * sinPole - cosPole * forceOverTotalMass) / (0.5f * (4.0f / 3.0f - 0.1f * cosPole * cosPole / 1.1f));
    float cartAcceleration = forceOverTotalMass - 0.05f * poleAngularAcceleration * cosPole / 1.1f;
    cartPosition += CARTPOLE_DT * cartVelocity;
    cartVelocity += CARTPOLE_DT * cartAcceleration;
    poleAngle += CARTPOLE_DT * poleAngularVelocity;
    poleAngularVelocity += CARTPOLE_DT * poleAngularAcceleration;
    poleAngle = wrapPi(poleAngle);
    uint done = 0u;
    if (fabs(cartPosition) > 2.4f || fabs(poleAngle) > CARTPOLE_THETA_MAX) {
        done = 1u;
        cartPosition = 0.0f; cartVelocity = 0.0f; poleAngle = 0.0f; poleAngularVelocity = 0.0f;
    }
    words[0] = as_type<uint>(cartPosition);  words[1] = as_type<uint>(cartVelocity);
    words[2] = as_type<uint>(poleAngle); words[3] = as_type<uint>(poleAngularVelocity);
    cartPole_obs(cartPosition, cartVelocity, poleAngle, poleAngularVelocity, obs);
    *doneOut = done;
    return 1.0f;
}
