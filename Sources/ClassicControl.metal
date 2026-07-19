#ifndef OAK_CLASSIC_CONTROL_METAL_INCLUDED
#define OAK_CLASSIC_CONTROL_METAL_INCLUDED

#include <metal_stdlib>
using namespace metal;

// Continuing classic-control envs (Acrobot-cont-v0, MountainCar-cont-v0,
// CartPole-cont-v0, Pendulum-cont-v0) — the Metal fp32 port of the fp64
// Python reference in tools/physics_parity.py, certified by the
// null-calibrated TOST population-equivalence gate (--classic-tost), the
// same standard as the F4 cheetah verdict. All four are CONTINUING
// (Inv-4/Inv-7): completion events respawn the state INSIDE the stream
// (done=1 is a wrapper metric, never a reset signal to the agent).
//
// Universal I/O contract (Inv-1): every env writes its natural obs channels
// and zero-pads to OBS_W=17; discrete envs consume the first N of the
// D_policy=12 policy coords as Gumbel-max logits (N from OAK_ENV_SPECS),
// Pendulum is Box 1 (ctrl[0] in [-1,1] -> torque [-2,2]).
//
// State is float32 stored as bit patterns in the lane's gw[10] uint slots
// (as_type round-trip). Dynamics follow the gym classic-control equations;
// angles are wrapped to [-pi, pi] once per step on both sides (a no-op
// mathematically — every use is 2*pi-periodic — that keeps fp32 argument
// reduction exact over million-step continuing streams).
//
// This file is included from OakKernel.metal (after Gridworld.metal) and
// compiled with -fno-fast-math; the explicit safe-mode pragma keeps the
// dynamics correctly rounded even if the include order moves.
#pragma METAL fp math_mode(safe)

inline float gw_wrap_pi(float x) {
    // floor-mod into [-pi, pi); identical formula on the Python side.
    return x - 2.0f * M_PI_F * floor((x + M_PI_F) / (2.0f * M_PI_F));
}

// ---------------------------------------------------------------- Acrobot
// gym acrobot (Spong 2-link), m1=m2=1, l1=l2=1, lc1=lc2=0.5, I1=I2=1,
// g=9.8, dt=0.2, RK4 (the gym integrator) with constant torque per step,
// velocity clips w1 in [-4pi, 4pi], w2 in [-9pi, 9pi]. D3 torque
// {-1,0,+1}. Sparse +1 on swing-up height (-cos(th1)-cos(th1+th2) > 1),
// reset to hang (all zeros), done=1. gw: 0..3 = th1,th2,w1,w2 bits.
constant float AB_DT = 0.2f;
constant float AB_G = 9.8f;
constant float AB_W1MAX = 12.566371f;   // 4*pi
constant float AB_W2MAX = 28.274334f;   // 9*pi

inline void gw_acro_obs(float th1, float th2, float w1, float w2,
                        thread float *o) {
    o[0] = cos(th1); o[1] = sin(th1);
    o[2] = cos(th2); o[3] = sin(th2);
    o[4] = w1; o[5] = w2; o[6] = 1.0f;
    for (int i = 7; i < 17; i++) o[i] = 0.0f;
}

// ds/dt of (th1, th2, w1, w2) under constant torque (gym _dsdt, nips form).
inline void gw_acro_dsdt(float th1, float th2, float w1, float w2, float tau,
                         thread float *d) {
    float c2 = cos(th2);
    float d1 = 3.5f + c2;            // m1 lc1^2 + m2(l1^2+lc2^2+2 l1 lc2 c2) + I1+I2
    float d2 = 1.25f + 0.5f * c2;    // m2(lc2^2 + l1 lc2 c2) + I2
    float phi2 = 4.9f * cos(th1 + th2 - M_PI_F / 2.0f);
    float s2 = sin(th2);
    float phi1 = -0.5f * w2 * w2 * s2 - w2 * w1 * s2
               + 14.7f * cos(th1 - M_PI_F / 2.0f) + phi2;
    float a2 = (tau + d2 / d1 * phi1 - 0.5f * w1 * w1 * s2 - phi2)
             / (d2 - d2 * d2 / d1);
    float a1 = -(d2 * a2 + phi1) / d1;
    d[0] = w1; d[1] = w2; d[2] = a1; d[3] = a2;
}

inline float gw_acro_step(thread uint *gw, uint action, thread float *obs,
                          thread uint *doneOut) {
    float th1 = as_type<float>(gw[0]), th2 = as_type<float>(gw[1]);
    float w1 = as_type<float>(gw[2]), w2 = as_type<float>(gw[3]);
    float tau = float(int(action % 3u)) - 1.0f;
    // classic RK4, dt = 0.2 (gym acrobot integrator)
    float k1[4], k2[4], k3[4], k4[4];
    gw_acro_dsdt(th1, th2, w1, w2, tau, k1);
    float h = AB_DT / 2.0f;
    gw_acro_dsdt(th1 + h * k1[0], th2 + h * k1[1], w1 + h * k1[2], w2 + h * k1[3], tau, k2);
    gw_acro_dsdt(th1 + h * k2[0], th2 + h * k2[1], w1 + h * k2[2], w2 + h * k2[3], tau, k3);
    gw_acro_dsdt(th1 + AB_DT * k3[0], th2 + AB_DT * k3[1], w1 + AB_DT * k3[2], w2 + AB_DT * k3[3], tau, k4);
    float s = AB_DT / 6.0f;
    th1 += s * (k1[0] + 2.0f * k2[0] + 2.0f * k3[0] + k4[0]);
    th2 += s * (k1[1] + 2.0f * k2[1] + 2.0f * k3[1] + k4[1]);
    w1 += s * (k1[2] + 2.0f * k2[2] + 2.0f * k3[2] + k4[2]);
    w2 += s * (k1[3] + 2.0f * k2[3] + 2.0f * k3[3] + k4[3]);
    w1 = clamp(w1, -AB_W1MAX, AB_W1MAX);
    w2 = clamp(w2, -AB_W2MAX, AB_W2MAX);
    th1 = gw_wrap_pi(th1);
    th2 = gw_wrap_pi(th2);
    float reward = 0.0f;
    uint done = 0u;
    if (-cos(th1) - cos(th1 + th2) > 1.0f) {   // free end one link above base
        reward = 1.0f;
        done = 1u;
        th1 = 0.0f; th2 = 0.0f; w1 = 0.0f; w2 = 0.0f;  // reset to hang
    }
    gw[0] = as_type<uint>(th1); gw[1] = as_type<uint>(th2);
    gw[2] = as_type<uint>(w1);  gw[3] = as_type<uint>(w2);
    gw_acro_obs(th1, th2, w1, w2, obs);
    *doneOut = done;
    return reward;
}

// ------------------------------------------------------------- MountainCar
// gym MountainCar (discrete dynamics): force (a-1)*0.001, gravity 0.0025,
// pos in [-1.2, 0.6], vel in [-0.07, 0.07], left wall zeroes negative vel.
// Sparse +1 at the hilltop (pos >= 0.5), respawn at the valley (-0.5, 0),
// done=1. D3 force {-1,0,+1}. gw: 0=pos bits, 1=vel bits.
inline void gw_mcar_obs(float pos, float vel, thread float *o) {
    o[0] = pos; o[1] = vel; o[2] = 1.0f;
    for (int i = 3; i < 17; i++) o[i] = 0.0f;
}

inline float gw_mcar_step(thread uint *gw, uint action, thread float *obs,
                          thread uint *doneOut) {
    float pos = as_type<float>(gw[0]), vel = as_type<float>(gw[1]);
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
        pos = -0.5f; vel = 0.0f;   // respawn at the valley
    }
    gw[0] = as_type<uint>(pos); gw[1] = as_type<uint>(vel);
    gw_mcar_obs(pos, vel, obs);
    *doneOut = done;
    return reward;
}

// --------------------------------------------------------------- CartPole
// gym CartPole-v1 dynamics: g=9.8, mc=1.0, mp=0.1, length=0.5, tau=0.02,
// force_mag=10, Euler kinematics. Dense +1 per step alive; fall
// (|x| > 2.4 or |theta| > 12 deg) resets to the upright zero state,
// done=1. D2 force {-10,+10}. gw: 0..3 = x, xdot, th, thdot bits.
constant float CP_TAU = 0.02f;
constant float CP_THETAMAX = 0.2094395f;   // 12 deg

inline void gw_pole_obs(float x, float xd, float th, float thd,
                        thread float *o) {
    o[0] = x; o[1] = xd; o[2] = th; o[3] = thd; o[4] = 1.0f;
    for (int i = 5; i < 17; i++) o[i] = 0.0f;
}

inline float gw_pole_step(thread uint *gw, uint action, thread float *obs,
                          thread uint *doneOut) {
    float x = as_type<float>(gw[0]), xd = as_type<float>(gw[1]);
    float th = as_type<float>(gw[2]), thd = as_type<float>(gw[3]);
    float force = ((action & 1u) != 0u) ? 10.0f : -10.0f;
    float st = sin(th), ct = cos(th);
    float temp = (force + 0.05f * thd * thd * st) / 1.1f;
    float thacc = (9.8f * st - ct * temp) / (0.5f * (4.0f / 3.0f - 0.1f * ct * ct / 1.1f));
    float xacc = temp - 0.05f * thacc * ct / 1.1f;
    x += CP_TAU * xd;
    xd += CP_TAU * xacc;
    th += CP_TAU * thd;
    thd += CP_TAU * thacc;
    th = gw_wrap_pi(th);
    uint done = 0u;
    if (fabs(x) > 2.4f || fabs(th) > CP_THETAMAX) {
        done = 1u;
        x = 0.0f; xd = 0.0f; th = 0.0f; thd = 0.0f;   // reset upright
    }
    gw[0] = as_type<uint>(x);  gw[1] = as_type<uint>(xd);
    gw[2] = as_type<uint>(th); gw[3] = as_type<uint>(thd);
    gw_pole_obs(x, xd, th, thd, obs);
    *doneOut = done;
    return 1.0f;   // dense +1 per step alive
}

// --------------------------------------------------------------- Pendulum
// gym Pendulum: g=10, m=1, l=1, dt=0.05, max_speed=8, torque in [-2,2]
// (ctrl[0] in [-1,1] * 2). Dense shaped reward -(th_norm^2 + 0.1 thdot^2 +
// 0.001 u^2) with th_norm = angle from upright; cost uses the PRE-step
// state (gym order). Truly continuing: never done. Box 1.
// gw: 0=th bits, 1=thdot bits.
inline void gw_pend_obs(float th, float thd, thread float *o) {
    o[0] = cos(th); o[1] = sin(th); o[2] = thd; o[3] = 1.0f;
    for (int i = 4; i < 17; i++) o[i] = 0.0f;
}

inline float gw_pend_step(thread uint *gw, float ctrl0, thread float *obs) {
    float th = as_type<float>(gw[0]), thd = as_type<float>(gw[1]);
    float u = clamp(2.0f * ctrl0, -2.0f, 2.0f);
    float thn = gw_wrap_pi(th);   // angle from upright (gym th is 0 = up)
    float cost = thn * thn + 0.1f * thd * thd + 0.001f * u * u;
    float nthd = thd + (15.0f * sin(th + M_PI_F) + 3.0f * u) * 0.05f;
    th = th + nthd * 0.05f;
    nthd = clamp(nthd, -8.0f, 8.0f);
    th = gw_wrap_pi(th);
    gw[0] = as_type<uint>(th); gw[1] = as_type<uint>(nthd);
    gw_pend_obs(th, nthd, obs);
    return -cost;
}

// -------------------------------------------------- case-gate probe kernel
// Single-step probe for the classic-control boundary gate: reads case
// in-records (buffer 0), runs each through the SAME gw_*_step functions the
// fleet env window uses, writes out-records (buffer 1). hdr = {count, inW,
// outW}; inW selects the env (3=MountainCar, 5=Acrobot, 7=CartPole,
// 4=Pendulum). Compared by physics_parity.py --classic-probe-compare at
// fp32-vs-fp64 tolerance (state/reward 1e-4, done flag exact-counted).
kernel void oak_classic_probe(device const float *in  [[buffer(0)]],
                              device float *out       [[buffer(1)]],
                              constant uint *hdr      [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    uint n = hdr[0], inW = hdr[1], outW = hdr[2];
    if (gid >= n) return;
    device const float *p = in + (ulong)gid * inW;
    device float *o = out + (ulong)gid * outW;
    uint gw[10] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    float obs[17];
    uint done = 0u;
    if (inW == 3u) {
        // MountainCar in: [pos, vel, action] -> [reward, npos, nvel, done]
        gw[0] = as_type<uint>(p[0]); gw[1] = as_type<uint>(p[1]);
        float r = gw_mcar_step(gw, uint(p[2]), obs, &done);
        o[0] = r; o[1] = as_type<float>(gw[0]); o[2] = as_type<float>(gw[1]);
        o[3] = float(done);
    } else if (inW == 5u) {
        // Acrobot in: [th1, th2, w1, w2, action] -> [reward, st*4, done]
        for (int i = 0; i < 4; i++) gw[i] = as_type<uint>(p[i]);
        float r = gw_acro_step(gw, uint(p[4]), obs, &done);
        o[0] = r;
        for (int i = 0; i < 4; i++) o[1 + i] = as_type<float>(gw[i]);
        o[5] = float(done);
    } else if (inW == 7u) {
        // CartPole in: [x, xd, th, thd, action, 0, 0] -> [reward, st*4, done]
        for (int i = 0; i < 4; i++) gw[i] = as_type<uint>(p[i]);
        float r = gw_pole_step(gw, uint(p[4]), obs, &done);
        o[0] = r;
        for (int i = 0; i < 4; i++) o[1 + i] = as_type<float>(gw[i]);
        o[5] = float(done);
    } else {
        // Pendulum in: [th, thdot, ctrl, 0] -> [reward, nth, nthdot, 0]
        gw[0] = as_type<uint>(p[0]); gw[1] = as_type<uint>(p[1]);
        float r = gw_pend_step(gw, p[2], obs);
        o[0] = r; o[1] = as_type<float>(gw[0]); o[2] = as_type<float>(gw[1]);
        o[3] = 0.0f;
    }
}

#endif
