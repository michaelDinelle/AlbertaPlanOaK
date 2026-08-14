#ifndef OAK_GRIDWORLD_METAL_INCLUDED
#define OAK_GRIDWORLD_METAL_INCLUDED

#include <metal_stdlib>
using namespace metal;

#pragma METAL fp math_mode(safe)

constant uint OAK_ENV_CHEETAH     = 0u;
constant uint OAK_ENV_FOURROOMS   = 1u;
constant uint OAK_ENV_DEEPSEA     = 2u;
constant uint OAK_ENV_KEYCORRIDOR = 3u;
constant uint OAK_ENV_ACROBOT     = 4u;
constant uint OAK_ENV_MOUNTAINCAR = 5u;
constant uint OAK_ENV_CARTPOLE    = 6u;
constant uint OAK_ENV_PENDULUM    = 7u;
constant uint OAK_ENV_CHAIN       = 8u;
constant uint OAK_ENV_LARGEDEEPSEA = 9u;
constant uint OAK_ENV_MULTIKEYCORRIDOR = 10u;
constant uint OAK_ENV_MEMORYCHAIN = 11u;
constant uint OAK_ENV_PROCEDURALMAZE = 12u;
constant uint OAK_ENV_HIGHDIMMAZE = 13u;
constant uint OAK_ENV_POMAZE      = 14u;
constant uint OAK_ENV_DELAYEDCHAIN = 15u;
constant uint OAK_ENV_COUNT       = 16u;
constant uint OAK_GW_STREAM       = 3u;

struct OakEnvSpec { uint nameHash; uint obsDimNatural; uint actionKind; uint actionCount; };
constant OakEnvSpec OAK_ENV_SPECS[OAK_ENV_COUNT] = {
    { 0x9d7fd44bu, 17u, 0u, 6u },
    { 0x8014a187u, 17u, 1u, 4u },
    { 0x56af5bf2u,  8u, 1u, 2u },
    { 0xdd537ebau, 14u, 1u, 4u },
    { 0x93531833u,  7u, 1u, 3u },
    { 0xf5b3ddc8u,  3u, 1u, 3u },
    { 0x41f9937du,  5u, 1u, 2u },
    { 0x69bbbb73u,  4u, 0u, 1u },
    { 0x7b267422u,  2u, 1u, 2u },
    { 0xb96ddc3du,  8u, 1u, 2u },
    { 0x4c3c2d3fu, 17u, 1u, 4u },
    { 0xcf293fd7u,  6u, 1u, 2u },
    { 0xd75dd5f9u, 17u, 1u, 4u },
    { 0xbfeb2b7au, 16u, 1u, 6u },
    { 0x0582e383u, 15u, 1u, 4u },
    { 0x56184b96u,  3u, 1u, 2u },
};

inline bool gw_four_is_wall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (x == 5 && y != 2 && y != 8) return true;
    if (y == 5 && x != 2 && x != 8) return true;
    return false;
}

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
    8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 };

inline int gw_four_room(int x, int y) {
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

inline bool gw_kc_is_wall(int x, int y) {
    if (x <= 0 || x >= 8 || y <= 0 || y >= 6) return true;
    if (x == 4 && y != 3) return true;
    return false;
}

inline void gw_kc_obs(uint x, uint y, uint kx, uint ky, uint gx, uint gy,
                      uint flags, uint sg, thread float *o) {
    bool hasKey = (flags & 1u) != 0u;
    o[0] = float(x) / 8.0f;
    o[1] = float(y) / 6.0f;
    o[2] = hasKey ? 1.0f : 0.0f;
    o[3] = ((flags & 2u) != 0u) ? 1.0f : 0.0f;
    o[4] = hasKey ? 0.0f : 1.0f;
    o[5] = hasKey ? 0.0f : float(int(kx) - int(x)) / 8.0f;
    o[6] = hasKey ? 0.0f : float(int(ky) - int(y)) / 6.0f;
    o[7] = float(4 - int(x)) / 8.0f;
    o[8] = float(3 - int(y)) / 6.0f;
    o[9] = float(int(gx) - int(x)) / 8.0f;
    o[10] = float(int(gy) - int(y)) / 6.0f;
    o[11] = (x == 4u && y == 3u) ? 1.0f : 0.0f;
    o[12] = min(float(sg) / 256.0f, 1.0f);
    o[13] = 1.0f;
    for (int i = 14; i < 17; i++) o[i] = 0.0f;
}

inline float gw_kc_step(thread uint *gw, uint laneSeed, uint action,
                        thread float *obs, thread uint *doneOut) {
    uint x = gw[0], y = gw[1], kx = gw[2], ky = gw[3], gx = gw[4], gy = gw[5];
    uint flags = gw[6], sg = gw[7], ri = gw[8];
    int dx = 0, dy = 0;
    switch (action & 3u) {
        case 0u: dy = -1; break;
        case 1u: dy = 1; break;
        case 2u: dx = -1; break;
        default: dx = 1; break;
    }
    int nx = int(x) + dx, ny = int(y) + dy;
    if (gw_kc_is_wall(nx, ny)) { nx = int(x); ny = int(y); }
    else if (nx == 4 && ny == 3 && (flags & 2u) == 0u) {
        if ((flags & 1u) != 0u) flags |= 2u;
        else { nx = int(x); ny = int(y); }
    }
    if ((flags & 1u) == 0u && uint(nx) == kx && uint(ny) == ky) flags |= 1u;
    float reward = 0.0f;
    uint done = 0u, nsg = sg + 1u, nri = ri;
    if (uint(nx) == gx && uint(ny) == gy) {
        reward = 1.0f;
        done = 1u;
        nsg = 0u;
        nri = ri + 1u;
        uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                  uint4(ri, 0u, 0u, 0u));
        uint ai = uint(hc_u01(raw.x) * 15.0f);
        uint ki = uint(hc_u01(raw.y) * 15.0f);
        uint gi = uint(hc_u01(raw.z) * 15.0f);
        nx = int(1u + ai % 3u); ny = int(1u + ai / 3u);
        kx = 1u + ki % 3u; ky = 1u + ki / 3u;
        gx = 5u + gi % 3u; gy = 1u + gi / 3u;
        flags = 0u;
    }
    gw[0] = uint(nx); gw[1] = uint(ny); gw[2] = kx; gw[3] = ky;
    gw[4] = gx; gw[5] = gy; gw[6] = flags; gw[7] = nsg; gw[8] = nri;
    gw_kc_obs(uint(nx), uint(ny), kx, ky, gx, gy, flags, nsg, obs);
    *doneOut = done;
    return reward;
}

inline float gw_chain_step(thread uint *gw, uint action, thread float *obs,
                           thread uint *doneOut) {
    uint pos = gw[0];
    float reward = 0.0f;
    uint done = 0u;
    if ((action & 1u) != 0u) {
        pos += 1u;
        if (pos >= 15u) { reward = 1.0f; pos = 0u; done = 1u; }
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    gw[0] = pos;
    obs[0] = float(pos) / 16.0f;
    obs[1] = 1.0f;
    for (int i = 2; i < 17; i++) obs[i] = 0.0f;
    *doneOut = done;
    return reward;
}

inline void gw_ldeep_obs(uint r, uint c, uint lastA, thread float *o) {
    float n1 = 15.0f;
    o[0] = float(r) / n1;
    o[1] = float(c) / n1;
    o[2] = float(15 - int(c)) / n1;
    o[3] = float(15 - int(r)) / n1;
    o[4] = (c == 15u) ? 1.0f : 0.0f;
    o[5] = (c == 0u) ? 1.0f : 0.0f;
    o[6] = float(lastA);
    o[7] = 1.0f;
    for (int i = 8; i < 17; i++) o[i] = 0.0f;
}

inline float gw_ldeep_step(thread uint *gw, uint action, thread float *o) {
    uint r = gw[0] & 15u, c = (gw[0] >> 4u) & 15u;
    uint bitIdx = r * 16u + c;
    uint bit = (gw[1u + (bitIdx >> 5u)] >> (bitIdx & 31u)) & 1u;
    int dc = ((action & 1u) == bit) ? 1 : -1;
    int nc = clamp(int(c) + dc, 0, 15);
    uint nr = r + 1u;
    float reward = ((action & 1u) != 0u) ? -0.000625f : 0.0f;
    uint nLast = action & 1u;
    if (nr >= 15u) {
        if (nc == 15) reward = reward + 1.0f;
        nr = 0u;
        nc = 0;
    }
    gw[0] = nr | (uint(nc) << 4u) | (nLast << 8u);
    gw_ldeep_obs(nr, uint(nc), nLast, o);
    return reward;
}

inline bool gw_mkc_is_wall(int x, int y) {
    if (x <= 0 || x >= 16 || y <= 0 || y >= 6) return true;
    if ((x == 4 || x == 8 || x == 12) && y != 3) return true;
    return false;
}

inline void gw_mkc_obs(uint x, uint y, uint k1p, uint k2p, uint k3p,
                       uint gx, uint gy, uint flags, thread float *o) {
    bool hk1 = (flags & 1u) != 0u;
    bool hk2 = (flags & 2u) != 0u;
    bool hk3 = (flags & 4u) != 0u;
    o[0] = float(x) / 16.0f;
    o[1] = float(y) / 6.0f;
    o[2] = hk1 ? 1.0f : 0.0f;
    o[3] = hk2 ? 1.0f : 0.0f;
    o[4] = hk3 ? 1.0f : 0.0f;
    o[5] = ((flags & 8u) != 0u) ? 1.0f : 0.0f;
    o[6] = ((flags & 16u) != 0u) ? 1.0f : 0.0f;
    o[7] = ((flags & 32u) != 0u) ? 1.0f : 0.0f;
    o[8]  = hk1 ? 0.0f : float(int(k1p & 15u) - int(x)) / 16.0f;
    o[9]  = hk1 ? 0.0f : float(int(k1p >> 4u) - int(y)) / 6.0f;
    o[10] = hk2 ? 0.0f : float(int(k2p & 15u) - int(x)) / 16.0f;
    o[11] = hk2 ? 0.0f : float(int(k2p >> 4u) - int(y)) / 6.0f;
    o[12] = hk3 ? 0.0f : float(int(k3p & 15u) - int(x)) / 16.0f;
    o[13] = hk3 ? 0.0f : float(int(k3p >> 4u) - int(y)) / 6.0f;
    o[14] = float(int(gx) - int(x)) / 16.0f;
    o[15] = float(int(gy) - int(y)) / 6.0f;
    o[16] = 1.0f;
}

inline float gw_mkc_step(thread uint *gw, uint laneSeed, uint action,
                         thread float *obs, thread uint *doneOut) {
    uint x = gw[0], y = gw[1], k1p = gw[2], k2p = gw[3], k3p = gw[4];
    uint gx = gw[5], gy = gw[6], flags = gw[7], sg = gw[8], ri = gw[9];
    int dx = 0, dy = 0;
    switch (action & 3u) {
        case 0u: dy = -1; break;
        case 1u: dy = 1; break;
        case 2u: dx = -1; break;
        default: dx = 1; break;
    }
    int nx = int(x) + dx, ny = int(y) + dy;
    if (gw_mkc_is_wall(nx, ny)) { nx = int(x); ny = int(y); }
    else if (ny == 3 && (nx == 4 || nx == 8 || nx == 12)) {
        uint door = uint(nx >> 2);
        if ((flags & (8u << (door - 1u))) == 0u) {
            if ((flags & (1u << (door - 1u))) != 0u) flags |= (8u << (door - 1u));
            else { nx = int(x); ny = int(y); }
        }
    }
    if ((flags & 1u) == 0u && uint(nx) == (k1p & 15u) && uint(ny) == (k1p >> 4u))
        flags |= 1u;
    if ((flags & 2u) == 0u && uint(nx) == (k2p & 15u) && uint(ny) == (k2p >> 4u))
        flags |= 2u;
    if ((flags & 4u) == 0u && uint(nx) == (k3p & 15u) && uint(ny) == (k3p >> 4u))
        flags |= 4u;
    float reward = 0.0f;
    uint done = 0u, nsg = sg + 1u, nri = ri;
    if (uint(nx) == gx && uint(ny) == gy) {
        reward = 1.0f;
        done = 1u;
        nsg = 0u;
        nri = ri + 1u;
        uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                  uint4(ri, 0u, 0u, 0u));
        uint4 raw2 = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                   uint4(ri, 1u, 0u, 0u));
        uint ai = uint(hc_u01(raw.x) * 15.0f);
        uint k1i = uint(hc_u01(raw.y) * 15.0f);
        uint k2i = uint(hc_u01(raw.z) * 15.0f);
        uint k3i = uint(hc_u01(raw.w) * 15.0f);
        uint gi = uint(hc_u01(raw2.x) * 15.0f);
        nx = int(1u + ai % 3u); ny = int(1u + ai / 3u);
        k1p = (1u + k1i % 3u) | ((1u + k1i / 3u) << 4u);
        k2p = (5u + k2i % 3u) | ((1u + k2i / 3u) << 4u);
        k3p = (9u + k3i % 3u) | ((1u + k3i / 3u) << 4u);
        gx = 13u + gi % 3u; gy = 1u + gi / 3u;
        flags = 0u;
    }
    gw[0] = uint(nx); gw[1] = uint(ny); gw[2] = k1p; gw[3] = k2p; gw[4] = k3p;
    gw[5] = gx; gw[6] = gy; gw[7] = flags; gw[8] = nsg; gw[9] = nri;
    gw_mkc_obs(uint(nx), uint(ny), k1p, k2p, k3p, gx, gy, flags, obs);
    *doneOut = done;
    return reward;
}

inline void gw_mchain_obs(uint pos, uint cue, uint lastA, thread float *o) {
    o[0] = float(pos) / 32.0f;
    o[1] = (pos == 0u) ? float(cue) : 0.0f;
    o[2] = (pos == 0u) ? 1.0f : 0.0f;
    o[3] = (pos == 31u) ? 1.0f : 0.0f;
    o[4] = float(lastA);
    o[5] = 1.0f;
    for (int i = 6; i < 17; i++) o[i] = 0.0f;
}

inline float gw_mchain_step(thread uint *gw, uint laneSeed, uint action,
                            thread float *obs, thread uint *doneOut) {
    uint pos = gw[0], cue = gw[1], ri = gw[3];
    float reward = 0.0f;
    uint done = 0u;
    if (pos >= 31u) {
        reward = ((action & 1u) == cue) ? 1.0f : 0.0f;
        done = 1u;
        uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                  uint4(ri, 0u, 0u, 0u));
        cue = raw.x & 1u;
        ri = ri + 1u;
        pos = 0u;
    } else if ((action & 1u) != 0u) {
        pos += 1u;
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    uint nLast = action & 1u;
    gw[0] = pos; gw[1] = cue; gw[2] = nLast; gw[3] = ri;
    gw_mchain_obs(pos, cue, nLast, obs);
    *doneOut = done;
    return reward;
}

inline void gw_pmaze_gen(uint laneSeed, thread uint *w) {
    w[0] = 0xffffffffu; w[1] = 0xffffffffu; w[2] = 0x1ffffu;
    for (uint j = 0u; j < 5u; j++)
        for (uint i = 0u; i < 5u; i++) {
            uint idx = 2u * i + 9u * (2u * j);
            w[idx >> 5u] &= ~(1u << (idx & 31u));
        }
    uint visited = 1u;
    uint stack[25];
    uint sp = 1u;
    stack[0] = 0u;
    uint ctr = 0u, bi = 0u;
    uint4 buf = hc_philox4x32(uint2(laneSeed, HC_RNG_STREAM_RESET),
                              uint4(0u, 0u, 0u, 0u));
    while (sp > 0u) {
        uint cur = stack[sp - 1u];
        int ci = int(cur % 5u), cj = int(cur / 5u);
        uint cand = 0u, n = 0u;
        const int dxs[4] = { 1, 0, -1, 0 };
        const int dys[4] = { 0, 1, 0, -1 };
        for (uint d = 0u; d < 4u; d++) {
            int ni = ci + dxs[d], nj = cj + dys[d];
            if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5 &&
                ((visited >> uint(ni + 5 * nj)) & 1u) == 0u) {
                cand |= uint(ni + 5 * nj) << (8u * n);
                n++;
            }
        }
        if (n == 0u) { sp--; continue; }
        if (bi == 4u) {
            ctr++;
            buf = hc_philox4x32(uint2(laneSeed, HC_RNG_STREAM_RESET),
                                uint4(ctr, 0u, 0u, 0u));
            bi = 0u;
        }
        uint words[4] = { buf.x, buf.y, buf.z, buf.w };
        uint nxt = (cand >> (8u * (words[bi] % n))) & 0xffu;
        bi++;
        uint ni = nxt % 5u, nj = nxt / 5u;
        uint idx = (uint(ci) + ni) + 9u * (uint(cj) + nj);
        w[idx >> 5u] &= ~(1u << (idx & 31u));
        visited |= 1u << nxt;
        stack[sp] = nxt;
        sp++;
    }
}

inline bool gw_pmaze_is_wall(uint w0, uint w1, uint w2, int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    uint idx = uint(x - 1) + 9u * uint(y - 1);
    uint w = (idx < 32u) ? w0 : ((idx < 64u) ? w1 : w2);
    return ((w >> (idx & 31u)) & 1u) != 0u;
}

inline void gw_pmaze_obs(uint x, uint y, uint gx, uint gy, uint sg,
                         uint w0, uint w1, uint w2, thread float *o) {
    int ix = int(x), iy = int(y);
    o[0] = float(x) / 10.0f;
    o[1] = float(y) / 10.0f;
    o[2] = float(int(gx) - ix) / 10.0f;
    o[3] = float(int(gy) - iy) / 10.0f;
    o[4] = float(abs(int(gx) - ix) + abs(int(gy) - iy)) / 16.0f;
    o[5] = gw_pmaze_is_wall(w0, w1, w2, ix + 0, iy - 1) ? 1.0f : 0.0f;
    o[6] = gw_pmaze_is_wall(w0, w1, w2, ix + 0, iy + 1) ? 1.0f : 0.0f;
    o[7] = gw_pmaze_is_wall(w0, w1, w2, ix - 1, iy + 0) ? 1.0f : 0.0f;
    o[8] = gw_pmaze_is_wall(w0, w1, w2, ix + 1, iy + 0) ? 1.0f : 0.0f;
    o[9] = gw_pmaze_is_wall(w0, w1, w2, ix + 1, iy - 1) ? 1.0f : 0.0f;
    o[10] = gw_pmaze_is_wall(w0, w1, w2, ix - 1, iy - 1) ? 1.0f : 0.0f;
    o[11] = gw_pmaze_is_wall(w0, w1, w2, ix + 1, iy + 1) ? 1.0f : 0.0f;
    o[12] = gw_pmaze_is_wall(w0, w1, w2, ix - 1, iy + 1) ? 1.0f : 0.0f;
    o[13] = (4.0f - o[5] - o[6] - o[7] - o[8]) / 4.0f;
    o[14] = min(float(sg) / 256.0f, 1.0f);
    o[15] = (x == gx || y == gy) ? 1.0f : 0.0f;
    o[16] = 1.0f;
}

inline float gw_pmaze_step(thread uint *gw, uint laneSeed, uint action,
                           thread float *obs, thread uint *doneOut) {
    uint x = gw[0], y = gw[1], gx = gw[2], gy = gw[3], sg = gw[4], ri = gw[5];
    uint w0 = gw[6], w1 = gw[7], w2 = gw[8];
    int dx = 0, dy = 0;
    switch (action & 3u) {
        case 0u: dy = -1; break;
        case 1u: dy = 1; break;
        case 2u: dx = -1; break;
        default: dx = 1; break;
    }
    int nx = int(x) + dx, ny = int(y) + dy;
    if (gw_pmaze_is_wall(w0, w1, w2, nx, ny)) { nx = int(x); ny = int(y); }
    float reward = 0.0f;
    uint done = 0u, nsg = sg + 1u, nri = ri;
    if (uint(nx) == gx && uint(ny) == gy) {
        reward = 1.0f;
        done = 1u;
        nsg = 0u;
        nri = ri + 1u;
        uint b = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, b, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(words[wi]) * 81.0f);
                int cx = 1 + int(idx % 9u), cy = 1 + int(idx / 9u);
                if (!gw_pmaze_is_wall(w0, w1, w2, cx, cy)) {
                    nx = cx; ny = cy; found = true; break;
                }
            }
            if (found) break;
            b += 2u;
        }
        b = 1u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, b, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(words[wi]) * 81.0f);
                int cx = 1 + int(idx % 9u), cy = 1 + int(idx / 9u);
                if (!gw_pmaze_is_wall(w0, w1, w2, cx, cy) &&
                    (cx != nx || cy != ny)) {
                    gx = uint(cx); gy = uint(cy); found = true; break;
                }
            }
            if (found) break;
            b += 2u;
        }
    }
    gw[0] = uint(nx); gw[1] = uint(ny); gw[2] = gx; gw[3] = gy;
    gw[4] = nsg; gw[5] = nri;
    gw_pmaze_obs(uint(nx), uint(ny), gx, gy, nsg, w0, w1, w2, obs);
    *doneOut = done;
    return reward;
}

inline bool gw_hdm_is_wall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (x == 3 && y != 9) return true;
    if (x == 7 && y != 1) return true;
    return false;
}

inline void gw_hdm_obs(uint x, uint y, uint gx, uint gy, thread float *o) {
    int ix = int(x), iy = int(y);
    o[0] = float(x) / 16.0f;
    o[1] = float(y) / 16.0f;
    o[2] = gw_hdm_is_wall(ix + 0, iy - 1) ? 1.0f : 0.0f;
    o[3] = gw_hdm_is_wall(ix + 0, iy + 1) ? 1.0f : 0.0f;
    o[4] = gw_hdm_is_wall(ix - 1, iy + 0) ? 1.0f : 0.0f;
    o[5] = gw_hdm_is_wall(ix + 1, iy + 0) ? 1.0f : 0.0f;
    o[6] = float(int(gx) - ix) / 16.0f;
    o[7] = float(int(gy) - iy) / 16.0f;
    for (uint b = 0u; b < 4u; b++) {
        o[8 + b] = float((x >> b) & 1u);
        o[12 + b] = float((y >> b) & 1u);
    }
    o[16] = 0.0f;
}

inline float gw_hdm_step(thread uint *gw, uint laneSeed, uint action,
                         thread float *obs, thread uint *doneOut) {
    uint x = gw[0], y = gw[1], gx = gw[2], gy = gw[3], sg = gw[4], ri = gw[5];
    uint lastDir = gw[6];
    uint a = action % 6u;
    int nx = int(x), ny = int(y);
    if (a < 4u) {
        int dx = 0, dy = 0;
        switch (a) {
            case 0u: dy = -1; break;
            case 1u: dy = 1; break;
            case 2u: dx = -1; break;
            default: dx = 1; break;
        }
        lastDir = a;
        if (!gw_hdm_is_wall(nx + dx, ny + dy)) { nx += dx; ny += dy; }
    } else if (a == 5u) {
        int dx = 0, dy = 0;
        switch (lastDir & 3u) {
            case 0u: dy = -1; break;
            case 1u: dy = 1; break;
            case 2u: dx = -1; break;
            default: dx = 1; break;
        }
        if (!gw_hdm_is_wall(nx + dx, ny + dy)) {
            nx += dx; ny += dy;
            if (!gw_hdm_is_wall(nx + dx, ny + dy)) { nx += dx; ny += dy; }
        }
    }
    float reward = 0.0f;
    uint done = 0u, nsg = sg + 1u, nri = ri;
    if (uint(nx) == gx && uint(ny) == gy) {
        reward = 1.0f;
        done = 1u;
        nsg = 0u;
        nri = ri + 1u;
        uint b = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, b, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(words[wi]) * 81.0f);
                int cx = 1 + int(idx % 9u), cy = 1 + int(idx / 9u);
                if (!gw_hdm_is_wall(cx, cy)) {
                    nx = cx; ny = cy; found = true; break;
                }
            }
            if (found) break;
            b += 2u;
        }
        b = 1u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, b, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(words[wi]) * 81.0f);
                int cx = 1 + int(idx % 9u), cy = 1 + int(idx / 9u);
                if (!gw_hdm_is_wall(cx, cy) && (cx != nx || cy != ny)) {
                    gx = uint(cx); gy = uint(cy); found = true; break;
                }
            }
            if (found) break;
            b += 2u;
        }
    }
    gw[0] = uint(nx); gw[1] = uint(ny); gw[2] = gx; gw[3] = gy;
    gw[4] = nsg; gw[5] = nri; gw[6] = lastDir;
    gw_hdm_obs(uint(nx), uint(ny), gx, gy, obs);
    *doneOut = done;
    return reward;
}

inline bool gw_po_is_wall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (y == 3 && x != 8) return true;
    if (y == 7 && x != 2) return true;
    return false;
}

inline void gw_po_obs(uint x, uint y, uint gx, uint gy, thread float *o) {
    int ix = int(x), iy = int(y);
    o[0] = gw_po_is_wall(ix + 0, iy - 1) ? 1.0f : 0.0f;
    o[1] = gw_po_is_wall(ix + 1, iy - 1) ? 1.0f : 0.0f;
    o[2] = gw_po_is_wall(ix + 1, iy + 0) ? 1.0f : 0.0f;
    o[3] = gw_po_is_wall(ix + 1, iy + 1) ? 1.0f : 0.0f;
    o[4] = gw_po_is_wall(ix + 0, iy + 1) ? 1.0f : 0.0f;
    o[5] = gw_po_is_wall(ix - 1, iy + 1) ? 1.0f : 0.0f;
    o[6] = gw_po_is_wall(ix - 1, iy + 0) ? 1.0f : 0.0f;
    o[7] = gw_po_is_wall(ix - 1, iy - 1) ? 1.0f : 0.0f;
    o[8] = float(x) / 10.0f;
    o[9] = float(y) / 10.0f;
    o[10] = (gx > x) ? 1.0f : 0.0f;
    o[11] = (gx < x) ? 1.0f : 0.0f;
    o[12] = (gy > y) ? 1.0f : 0.0f;
    o[13] = (gy < y) ? 1.0f : 0.0f;
    o[14] = 1.0f;
    o[15] = 0.0f;
    o[16] = 0.0f;
}

inline float gw_po_step(thread uint *gw, uint laneSeed, uint action,
                        thread float *obs, thread uint *doneOut) {
    uint x = gw[0], y = gw[1], gx = gw[2], gy = gw[3], sg = gw[4], ri = gw[5];
    int dx = 0, dy = 0;
    switch (action & 3u) {
        case 0u: dy = -1; break;
        case 1u: dy = 1; break;
        case 2u: dx = -1; break;
        default: dx = 1; break;
    }
    int nx = int(x) + dx, ny = int(y) + dy;
    if (gw_po_is_wall(nx, ny)) { nx = int(x); ny = int(y); }
    float reward = 0.0f;
    uint done = 0u, nsg = sg + 1u, nri = ri;
    if (uint(nx) == gx && uint(ny) == gy) {
        reward = 1.0f;
        done = 1u;
        nsg = 0u;
        nri = ri + 1u;
        uint b = 0u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, b, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(words[wi]) * 81.0f);
                int cx = 1 + int(idx % 9u), cy = 1 + int(idx / 9u);
                if (!gw_po_is_wall(cx, cy)) {
                    nx = cx; ny = cy; found = true; break;
                }
            }
            if (found) break;
            b += 2u;
        }
        b = 1u;
        for (;;) {
            uint4 raw = hc_philox4x32(uint2(laneSeed, OAK_GW_STREAM),
                                      uint4(ri, b, 0u, 0u));
            uint words[4] = { raw.x, raw.y, raw.z, raw.w };
            bool found = false;
            for (int wi = 0; wi < 4; wi++) {
                uint idx = uint(hc_u01(words[wi]) * 81.0f);
                int cx = 1 + int(idx % 9u), cy = 1 + int(idx / 9u);
                if (!gw_po_is_wall(cx, cy) && (cx != nx || cy != ny)) {
                    gx = uint(cx); gy = uint(cy); found = true; break;
                }
            }
            if (found) break;
            b += 2u;
        }
    }
    gw[0] = uint(nx); gw[1] = uint(ny); gw[2] = gx; gw[3] = gy;
    gw[4] = nsg; gw[5] = nri;
    gw_po_obs(uint(nx), uint(ny), gx, gy, obs);
    *doneOut = done;
    return reward;
}

inline float gw_dchain_step(thread uint *gw, uint action, thread float *obs,
                            thread uint *doneOut) {
    uint pos = gw[0];
    float reward = 0.0f;
    uint done = 0u;
    if ((action & 1u) != 0u) {
        pos += 1u;
        if (pos >= 31u) { reward = 1.0f; pos = 0u; done = 1u; }
    } else {
        pos = (pos > 0u) ? pos - 1u : 0u;
    }
    uint nLast = action & 1u;
    gw[0] = pos;
    gw[1] = nLast;
    obs[0] = float(pos) / 32.0f;
    obs[1] = float(nLast);
    obs[2] = 1.0f;
    for (int i = 3; i < 17; i++) obs[i] = 0.0f;
    *doneOut = done;
    return reward;
}

#endif
