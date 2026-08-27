#pragma METAL fp math_mode(safe)

constant float ACROBOT_DT = 0.2f;
constant float ACROBOT_OMEGA1_MAX = 12.566371f;
constant float ACROBOT_OMEGA2_MAX = 28.274334f;

inline void acrobot_obs(float theta1, float theta2, float omega1, float omega2,
                        thread float *obs) {
    obs[0] = cos(theta1); obs[1] = sin(theta1);
    obs[2] = cos(theta2); obs[3] = sin(theta2);
    obs[4] = omega1; obs[5] = omega2; obs[6] = 1.0f;
}

inline void acrobot_dynamics(float theta1, float theta2, float omega1, float omega2, float tau,
                         thread float *derivative) {
    float cosTheta2 = cos(theta2);
    float inertia1 = 3.5f + cosTheta2;
    float inertia2 = 1.25f + 0.5f * cosTheta2;
    float gravity2 = 4.9f * cos(theta1 + theta2 - M_PI_F / 2.0f);
    float sinTheta2 = sin(theta2);
    float gravity1 = -0.5f * omega2 * omega2 * sinTheta2 - omega2 * omega1 * sinTheta2
               + 14.7f * cos(theta1 - M_PI_F / 2.0f) + gravity2;
    float alpha2 = (tau + inertia2 / inertia1 * gravity1 - 0.5f * omega1 * omega1 * sinTheta2 - gravity2)
             / (inertia2 - inertia2 * inertia2 / inertia1);
    float alpha1 = -(inertia2 * alpha2 + gravity1) / inertia1;
    derivative[0] = omega1; derivative[1] = omega2; derivative[2] = alpha1; derivative[3] = alpha2;
}

inline float acrobot_step(device uint *words, uint action, thread float *obs,
                          thread uint *doneOut) {
    float theta1 = as_type<float>(words[0]), theta2 = as_type<float>(words[1]);
    float omega1 = as_type<float>(words[2]), omega2 = as_type<float>(words[3]);
    float tau = float(int(action % 3u)) - 1.0f;
    float k1[4], k2[4], k3[4], k4[4];
    acrobot_dynamics(theta1, theta2, omega1, omega2, tau, k1);
    float halfDt = ACROBOT_DT / 2.0f;
    acrobot_dynamics(theta1 + halfDt * k1[0], theta2 + halfDt * k1[1], omega1 + halfDt * k1[2], omega2 + halfDt * k1[3], tau, k2);
    acrobot_dynamics(theta1 + halfDt * k2[0], theta2 + halfDt * k2[1], omega1 + halfDt * k2[2], omega2 + halfDt * k2[3], tau, k3);
    acrobot_dynamics(theta1 + ACROBOT_DT * k3[0], theta2 + ACROBOT_DT * k3[1], omega1 + ACROBOT_DT * k3[2], omega2 + ACROBOT_DT * k3[3], tau, k4);
    float sixthDt = ACROBOT_DT / 6.0f;
    theta1 += sixthDt * (k1[0] + 2.0f * k2[0] + 2.0f * k3[0] + k4[0]);
    theta2 += sixthDt * (k1[1] + 2.0f * k2[1] + 2.0f * k3[1] + k4[1]);
    omega1 += sixthDt * (k1[2] + 2.0f * k2[2] + 2.0f * k3[2] + k4[2]);
    omega2 += sixthDt * (k1[3] + 2.0f * k2[3] + 2.0f * k3[3] + k4[3]);
    omega1 = clamp(omega1, -ACROBOT_OMEGA1_MAX, ACROBOT_OMEGA1_MAX);
    omega2 = clamp(omega2, -ACROBOT_OMEGA2_MAX, ACROBOT_OMEGA2_MAX);
    theta1 = wrapPi(theta1);
    theta2 = wrapPi(theta2);
    float reward = 0.0f;
    uint done = 0u;
    if (-cos(theta1) - cos(theta1 + theta2) > 1.0f) {
        reward = 1.0f;
        done = 1u;
        theta1 = 0.0f; theta2 = 0.0f; omega1 = 0.0f; omega2 = 0.0f;
    }
    words[0] = as_type<uint>(theta1); words[1] = as_type<uint>(theta2);
    words[2] = as_type<uint>(omega1);  words[3] = as_type<uint>(omega2);
    acrobot_obs(theta1, theta2, omega1, omega2, obs);
    *doneOut = done;
    return reward;
}
