#ifndef OAK_GRIDWORLD_METAL_INCLUDED
#define OAK_GRIDWORLD_METAL_INCLUDED

#include <metal_stdlib>
using namespace metal;

// Continuing gridworld envs (FourRooms-cont-v0, DeepSea-cont-v0) — the Metal
// port of tools/gridworld_ref.py, certified BITWISE by the case gate
// (gridworld_ref.py --compare). Both envs are CONTINUING (no done, no reset:
// goal/treasure respawns inside the stream), integer-state, deterministic
// given the per-lane Philox chain.
//
// Universal I/O contract (Inv-1): the AGENT stays env-blind — every env
// emits OBS_W=17 float32 obs channels and consumes the first N of the
// D_policy=12 policy coordinates as categorical logits (N=4 FourRooms,
// N=2 DeepSea). envKind/actionKind are read ONLY in the env window, the
// sample stage, fleet init, and the record path — never by agent stages.
//
// Bitwise rules (both sides obey):
// - state is integer (uint32); obs channels are float32(int) OP
//   float32(literal) with the exact op order of the reference;
// - all randomness is Philox4x32-10, key=(laneSeed, stream), ctr=(idx,
//   block, 0, 0) — the hc_philox4x32 chain (streams 0=RESET, 1=CTRL, 2=ACT
//   are taken by physics/sampler; 3 = gridworld events);
// - u01(x) = float32(x >> 8) * 2^-24 (exact); uniform index = trunc(u01 * 68).
//
// This file is included from OakKernel.metal (after OakPhysics.metal) and
// compiled with -fno-fast-math; the explicit safe-mode pragma keeps the
// obs-formula divisions correctly rounded even if the include order moves.
#pragma METAL fp math_mode(safe)

constant uint OAK_ENV_CHEETAH   = 0u;
constant uint OAK_ENV_FOURROOMS = 1u;
constant uint OAK_ENV_DEEPSEA   = 2u;
constant uint OAK_GW_STREAM     = 3u;  // philox stream: gridworld events

// ---------------------------------------------------------------- FourRooms
// 11x11, border walls; vertical wall col x=5 except hallways y in {2,8};
// horizontal wall row y=5 except hallways x in {2,8}. Center (5,5) is wall.
inline bool gw_four_is_wall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (x == 5 && y != 2 && y != 8) return true;
    if (y == 5 && x != 2 && x != 8) return true;
    return false;
}

// 68 free cells, y-major enumeration (identical to gridworld_ref.FOUR_FREE).
constant uchar GW_FREE_X[68] = {
    1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9,
    2, 8,
    1, 2, 3, 4, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 6, 7, 8, 9 };
constant uchar GW_FREE_Y[68] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 };

inline int gw_four_room(int x, int y) {  // 0=TL 1=TR 2=BL 3=BR ; -1 hallway
    if (x < 5 && y < 5) return 0;
    if (x > 5 && y < 5) return 1;
    if (x < 5 && y > 5) return 2;
    if (x > 5 && y > 5) return 3;
    return -1;
}

inline void gw_four_obs(uint x, uint y, uint gx, uint gy, uint sg,
                        thread float *o) {
    int room = gw_four_room(int(x), int(y));
    int groom = gw_four_room(int(gx), int(gy));
    o[0] = float(x) / 10.0f;
    o[1] = float(y) / 10.0f;
    o[2] = float(int(gx) - int(x)) / 10.0f;
    o[3] = float(int(gy) - int(y)) / 10.0f;
    o[4] = float(abs(int(gx) - int(x)) + abs(int(gy) - int(y))) / 20.0f;
    o[5] = (room == 0) ? 1.0f : 0.0f;
    o[6] = (room == 1) ? 1.0f : 0.0f;
    o[7] = (room == 2) ? 1.0f : 0.0f;
    o[8] = (room == 3) ? 1.0f : 0.0f;
    o[9]  = gw_four_is_wall(int(x) + 0, int(y) - 1) ? 1.0f : 0.0f;
    o[10] = gw_four_is_wall(int(x) + 0, int(y) + 1) ? 1.0f : 0.0f;
    o[11] = gw_four_is_wall(int(x) - 1, int(y) + 0) ? 1.0f : 0.0f;
    o[12] = gw_four_is_wall(int(x) + 1, int(y) + 0) ? 1.0f : 0.0f;
    o[13] = (room == -1) ? 1.0f : 0.0f;
    o[14] = (groom == room && room != -1) ? 1.0f : 0.0f;
    o[15] = min(float(sg) / 256.0f, 1.0f);
    o[16] = 1.0f;
}

// FourRooms lane state in gw[6]: 0=x, 1=y, 2=gx, 3=gy, 4=sinceGoal,
// 5=respawnIdx. action: 0=up 1=down 2=left 3=right. Wall bump = stay,
// reward 0. Goal entry: reward +1, goal respawns to a uniformly drawn free
// cell != agent cell (Philox stream 3, ctr=(respawnIdx, block,0,0), words
// 0..3 then block++). Returns reward; obs written for the NEW state.
inline float gw_four_step(thread uint *gw, uint laneSeed, uint action,
                          thread float *obs) {
    uint x = gw[0], y = gw[1], gx = gw[2], gy = gw[3], sg = gw[4], ri = gw[5];
    int dx = 0, dy = 0;
    switch (action & 3u) {
        case 0u: dy = -1; break;
        case 1u: dy = 1; break;
        case 2u: dx = -1; break;
        default: dx = 1; break;
    }
    int nx = int(x) + dx, ny = int(y) + dy;
    if (gw_four_is_wall(nx, ny)) { nx = int(x); ny = int(y); }
    float reward = 0.0f;
    uint ngx = gx, ngy = gy, nsg = sg + 1u, nri = ri;
    if (uint(nx) == gx && uint(ny) == gy) {
        reward = 1.0f;
        nsg = 0u;
        nri = ri + 1u;
        uint block = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, block, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                int idx = int(hc_u01(words[wi]) * 68.0f);
                if (idx < 68) {
                    uint cx = uint(GW_FREE_X[idx]), cy = uint(GW_FREE_Y[idx]);
                    if (cx != uint(nx) || cy != uint(ny)) {
                        ngx = cx; ngy = cy;
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
            block += 1u;
        }
    }
    gw[0] = uint(nx); gw[1] = uint(ny); gw[2] = ngx; gw[3] = ngy;
    gw[4] = nsg; gw[5] = nri;
    gw_four_obs(uint(nx), uint(ny), ngx, ngy, nsg, obs);
    return reward;
}

// ---------------------------------------------------------------- DeepSea
// N=8. Hidden per-cell action map M[r][c] (bit r*8+c of the 64-bit mask,
// fixed per lane at reset): M==1 -> action 1 moves c+1, action 0 moves c-1
// (normal); M==0 -> inverted. c clamped to [0,N-1]; r += 1 every step.
// Reward: -0.00125 if action==1 (the -0.01/N right-cost at N=8), +1 on
// reaching (r=N-1, c=N-1); on reaching r==N-1 the agent resurfaces to (0,0),
// mask kept. Lane state in gw[6]: 0=r, 1=c, 2=lastA, 3=maskLo, 4=maskHi.
inline void gw_deep_obs(uint r, uint c, uint lastA, thread float *o) {
    float n1 = 7.0f;
    o[0] = float(r) / n1;
    o[1] = float(c) / n1;
    o[2] = float(7 - int(c)) / n1;
    o[3] = float(7 - int(r)) / n1;
    o[4] = (c == 7u) ? 1.0f : 0.0f;
    o[5] = (c == 0u) ? 1.0f : 0.0f;
    o[6] = float(lastA);
    o[7] = 1.0f;
    for (int i = 8; i < 17; i++) o[i] = 0.0f;
}

inline float gw_deep_step(thread uint *gw, uint action, thread float *o) {
    uint r = gw[0], c = gw[1];
    uint bitIdx = r * 8u + c;
    uint bit = (bitIdx < 32u) ? ((gw[3] >> bitIdx) & 1u)
                              : ((gw[4] >> (bitIdx - 32u)) & 1u);
    int dc = ((action & 1u) == bit) ? 1 : -1;
    int nc = clamp(int(c) + dc, 0, 7);
    uint nr = r + 1u;
    float reward = ((action & 1u) != 0u) ? -0.00125f : 0.0f;
    uint nLast = action & 1u;
    if (nr >= 7u) {
        if (nc == 7) reward = reward + 1.0f;
        nr = 0u;
        nc = 0;
    }
    gw[0] = nr;
    gw[1] = uint(nc);
    gw[2] = nLast;
    gw_deep_obs(nr, uint(nc), nLast, o);
    return reward;
}

// -------------------------------------------------- case-gate probe kernel
// Reads a gridworld_ref.py case file's in-records (buffer 0), runs each
// through the SAME env-step functions the fleet env window uses, and writes
// out-records (buffer 1) in the case format. hdr = {count, inW, outW};
// inW selects the env (8=FourRooms, 6=DeepSea).
kernel void oak_gridworld_probe(device const float *in  [[buffer(0)]],
                                device float *out       [[buffer(1)]],
                                constant uint *hdr      [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    uint n = hdr[0], inW = hdr[1], outW = hdr[2];
    if (gid >= n) return;
    device const float *p = in + (ulong)gid * inW;
    device float *o = out + (ulong)gid * outW;
    if (inW == 8u) {
        // FourRooms in: [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal, action]
        uint gw[6] = { uint(p[2]), uint(p[3]), uint(p[4]),
                       uint(p[5]), uint(p[6]), uint(p[1]) };
        float obs[17];
        float r = gw_four_step(gw, uint(p[0]), uint(p[7]), obs);
        o[0] = r;
        for (int i = 0; i < 6; i++) o[1 + i] = float(gw[i]);
        for (int i = 0; i < 17; i++) o[7 + i] = obs[i];
    } else {
        // DeepSea in: [maskLo, maskHi, r, c, lastA, action]; mask words are
        // the uint32 bit pattern reinterpreted as float32.
        uint gw[6] = { uint(p[2]), uint(p[3]), uint(p[4]),
                       as_type<uint>(p[0]), as_type<uint>(p[1]), 0u };
        float obs[17];
        float r = gw_deep_step(gw, uint(p[5]), obs);
        o[0] = r;
        for (int i = 0; i < 3; i++) o[1 + i] = float(gw[i]);
        for (int i = 0; i < 17; i++) o[4 + i] = obs[i];
    }
}

#endif
