#ifndef OAK_GRIDWORLD_METAL_INCLUDED
#define OAK_GRIDWORLD_METAL_INCLUDED

#include <metal_stdlib>
using namespace metal;

// Continuing gridworld envs (FourRooms-cont-v0, DeepSea-cont-v0,
// KeyCorridor-cont-v0, Chain-cont-v0, LargeDeepSea-cont-v0,
// MultiKeyCorridor-cont-v0, MemoryChain-cont-v0) — the Metal port of
// tools/gridworld_ref.py, certified BITWISE by the case gate
// (gridworld_ref.py --compare). All are CONTINUING (no reset: goal/treasure/
// key/cue respawns inside the stream; done, where emitted, is a wrapper
// metric only), integer-state, deterministic given the per-lane Philox
// chain.
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
constant uint OAK_GW_STREAM       = 3u;  // philox stream: gridworld events

// Phase-3 multi-shape contract: envKind -> natural obs width, action kind
// (0=Box, 1=Discrete), natural action count. The AGENT never reads this
// (Inv-1); only the four sanctioned sites do (fleet init, sample stage, env
// window, record path). Every env writes its natural obs channels and
// zero-pads to OBS_W=17; discrete envs consume the first actionCount policy
// coords as Gumbel-max logits, Box envs the first actionCount as means.
struct OakEnvSpec { uint obsDimNatural; uint actionKind; uint actionCount; };
constant OakEnvSpec OAK_ENV_SPECS[OAK_ENV_COUNT] = {
    { 17u, 0u, 6u },  // OAK_ENV_CHEETAH: Box 6 (6 means + 6 reserved coords)
    { 17u, 1u, 4u },  // OAK_ENV_FOURROOMS
    {  8u, 1u, 2u },  // OAK_ENV_DEEPSEA
    { 14u, 1u, 4u },  // OAK_ENV_KEYCORRIDOR
    {  7u, 1u, 3u },  // OAK_ENV_ACROBOT
    {  3u, 1u, 3u },  // OAK_ENV_MOUNTAINCAR
    {  5u, 1u, 2u },  // OAK_ENV_CARTPOLE
    {  4u, 0u, 1u },  // OAK_ENV_PENDULUM: Box 1
    {  2u, 1u, 2u },  // OAK_ENV_CHAIN
    {  8u, 1u, 2u },  // OAK_ENV_LARGEDEEPSEA (N=16)
    { 17u, 1u, 4u },  // OAK_ENV_MULTIKEYCORRIDOR (3 keys/doors)
    {  6u, 1u, 2u },  // OAK_ENV_MEMORYCHAIN (L=32)
    { 17u, 1u, 4u },  // OAK_ENV_PROCEDURALMAZE (per-lane procedural walls)
    { 16u, 1u, 6u },  // OAK_ENV_HIGHDIMMAZE (6 actions, 16-ch rich obs)
    { 15u, 1u, 4u },  // OAK_ENV_POMAZE (view radius 1)
    {  3u, 1u, 2u },  // OAK_ENV_DELAYEDCHAIN (L=32, reward only at far end)
};

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
    8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9 };

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

// ------------------------------------------------------------ KeyCorridor
// 9x7 (x 0..8, y 0..6), border walls; interior wall x=4 except the door cell
// (4,3). Left room x 1..3, right room x 5..7, y 1..5 (15 cells each). Key
// spawns in the left room, goal in the right room. Auto-pickup when the
// agent is co-located with the key; entering the locked door cell opens it
// iff the key is carried (locked + no key = blocked, stay). Goal entry:
// reward +1, done=1, and the DeepSea-pattern respawn — agent + key redrawn
// in the left room, goal redrawn in the right room, door re-locked (Philox
// stream 3, ctr=(respawnIdx,0,0,0); words x=agent cell, y=key cell,
// z=goal cell; cell idx = trunc(u01(word)*15), room cell (1 or 5) + idx%3,
// 1 + idx/3). CONTINUING: no reset, the stream never breaks.
// Lane state in gw[10]: 0=x, 1=y, 2=kx, 3=ky, 4=gx, 5=gy,
// 6=flags (bit0 hasKey, bit1 doorOpen), 7=sinceGoal, 8=respawnIdx.
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

// action: 0=up 1=down 2=left 3=right (same convention as FourRooms).
// Returns reward; obs written for the NEW state; *doneOut = 1 on completion.
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
        if ((flags & 1u) != 0u) flags |= 2u;       // unlock and enter
        else { nx = int(x); ny = int(y); }          // locked: blocked
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

// ------------------------------------------------------------------ Chain
// bsuite-style chain, L=16: pos 0..15. action 1 moves pos+1, action 0 moves
// pos-1 (clamp 0). Delayed +1 at the chain end (pos 15), wrap to 0, done=1.
// Lane state in gw[10]: 0=pos.
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

// ------------------------------------------------------------ LargeDeepSea
// DeepSea scaled to N=16 (the canonical hard-exploration gridworld,
// exponentially harder with size). Identical mechanics to N=8: a hidden
// per-cell action map (bit r*16+c of the 256-bit mask, fixed per lane at
// reset): M==1 -> action 1 moves c+1, action 0 moves c-1 (normal); M==0 ->
// inverted. c clamped to [0,15]; r += 1 every step. Reward: -0.000625 if
// action==1 (the -0.01/N right-cost at N=16), +1 on reaching (r=15, c=15);
// on reaching r==15 the agent resurfaces to (0,0), mask kept. Lane state in
// gw[10]: gw[0] = r | (c<<4) | (lastA<<8) (packed so the 8 mask words fit in
// gw[1..8]).
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

// ------------------------------------------------------- MultiKeyCorridor
// KeyCorridor extended to 3 sequential keys and doors. 17x7 (x 0..16,
// y 0..6), border walls; interior walls x=4, x=8, x=12 except the door cells
// (4,3), (8,3), (12,3). Four rooms x in {1..3, 5..7, 9..11, 13..15}, y 1..5
// (15 cells each): key1 in room 0, key2 in room 1, key3 in room 2, goal in
// room 3. Auto-pickup on co-location; entering locked door j opens it iff
// key j is carried (locked + no key = blocked, stay). Goal entry: reward +1,
// done=1, and the DeepSea-pattern respawn — agent redrawn in room 0, keys
// redrawn in rooms 0/1/2, goal redrawn in room 3, doors re-locked (Philox
// stream 3, ctr=(respawnIdx,0,0,0): words x=agent cell, y=key1, z=key2,
// w=key3; then ctr=(respawnIdx,1,0,0) word x=goal; cell idx =
// trunc(u01(word)*15), room base (1,5,9,13) + idx%3, 1 + idx/3).
// CONTINUING: no reset, the stream never breaks.
// Lane state in gw[10]: 0=x, 1=y, 2=key1 packed (kx | ky<<4), 3=key2,
// 4=key3, 5=gx, 6=gy, 7=flags (bits0-2 hasKey1..3, bits3-5 doorOpen1..3),
// 8=sinceGoal, 9=respawnIdx.
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

// action: 0=up 1=down 2=left 3=right (same convention as FourRooms).
// Returns reward; obs written for the NEW state; *doneOut = 1 on completion.
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
        uint door = uint(nx >> 2);              // door index 1..3
        if ((flags & (8u << (door - 1u))) == 0u) {      // door still locked
            if ((flags & (1u << (door - 1u))) != 0u) flags |= (8u << (door - 1u));
            else { nx = int(x); ny = int(y); }           // locked: blocked
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

// ------------------------------------------------------------- MemoryChain
// bsuite Memory Chain: corridor of length 32 (pos 0..31). The cue (0=left,
// 1=right) is observable ONLY at pos==0; the agent must carry it across the
// full corridor, and at the far end (pos==31) its action IS the choice:
// reward +1 iff (action&1) == cue (0 otherwise), done=1, and the stream
// respawns at pos 0 with a freshly drawn cue (Philox stream 3,
// ctr=(respawnIdx,0,0,0), cue = raw.x & 1). Away from the end, action 1
// walks pos+1 and action 0 walks pos-1 (clamp 0). CONTINUING: no reset.
// Lane state in gw[10]: 0=pos, 1=cue, 2=lastA, 3=respawnIdx.
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

// --------------------------------------------------------- ProceduralMaze
// 11x11 (border walls) whose 9x9 interior WALL LAYOUT is generated per lane
// at reset from the lane's Philox chain (RESET stream) — deterministic per
// lane, different across lanes, then fixed for the lane's lifetime: the
// agent cannot memorize one maze. Gen runs ONCE at fleet init (amortized;
// ~5 Philox calls): 81 wall bits in 3 words (bit idx = (x-1) + 9*(y-1),
// 1=wall, 50% density; w2 masked to its low 17 bits), then a random monotone
// path from (1,1) to (9,9) carved free (16 moves; move i, when not edge-
// forced, uses Philox RESET ctr=(1+i/4,0,0,0) word i%4, u01 < 0.5 -> right
// else down) — the carved path guarantees the goal is always reachable.
// Completion: +1, done=1, agent and goal rejection-redrawn to free cells
// (stream 3, ctr=(respawnIdx, block,0,0); agent from blocks 0,2,4...,
// goal from blocks 1,3,5... != agent cell; cell idx = trunc(u01(w)*81),
// cell (1 + idx%9, 1 + idx/9)). CONTINUING: the maze itself never resets.
// Lane state in gw[10]: 0=x, 1=y, 2=gx, 3=gy, 4=sinceGoal, 5=respawnIdx,
// 6..8 = wall words w0..w2.
inline void gw_pmaze_gen(uint laneSeed, thread uint *w) {
    uint4 raw = hc_philox4x32(uint2(laneSeed, HC_RNG_STREAM_RESET),
                              uint4(0u, 0u, 0u, 0u));
    w[0] = raw.x; w[1] = raw.y; w[2] = raw.z & 0x1ffffu;
    int x = 1, y = 1;
    for (uint i = 0u; i < 16u; i++) {
        uint idx = uint(x - 1) + 9u * uint(y - 1);
        w[idx >> 5u] &= ~(1u << (idx & 31u));
        if (x == 9) { y++; continue; }
        if (y == 9) { x++; continue; }
        uint4 c = hc_philox4x32(uint2(laneSeed, HC_RNG_STREAM_RESET),
                                uint4(1u + (i >> 2u), 0u, 0u, 0u));
        uint words[4] = { c.x, c.y, c.z, c.w };
        if (hc_u01(words[i & 3u]) < 0.5f) x++; else y++;
    }
    uint idx = 8u + 9u * 8u;   // (9,9)
    w[idx >> 5u] &= ~(1u << (idx & 31u));
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

// action: 0=up 1=down 2=left 3=right (same convention as FourRooms).
// Returns reward; obs written for the NEW state; *doneOut = 1 on completion.
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
        for (;;) {                      // agent redraw: first free cell
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
        for (;;) {                      // goal redraw: free and != agent cell
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

// ------------------------------------------------------------ HighDimMaze
// 11x11 serpentine maze (border walls; interior wall x=3 except the gap
// (3,9), interior wall x=7 except the gap (7,1)): three corridors — down
// x1..2, through (3,9), up x4..6, through (7,1), down x8..9 — a longer
// horizon than FourRooms. SIX discrete actions: 0=up 1=down 2=left 3=right,
// 4=stay, 5=dash (up to two cells in lastDir, stopping at walls; lastDir =
// the last cardinal action, init down). Rich 16-channel observation:
// position, 4 wall sensors, goal relative position, and an 8-bit binary
// position code (the one-hot-ish encoding). +1 on goal entry, done=1,
// agent+goal rejection-redrawn to free cells (same scheme as
// ProceduralMaze). CONTINUING. Lane state in gw[10]: 0=x, 1=y, 2=gx, 3=gy,
// 4=sinceGoal, 5=respawnIdx, 6=lastDir.
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

// action: 0=up 1=down 2=left 3=right 4=stay 5=dash (2 cells in lastDir).
// Returns reward; obs written for the NEW state; *doneOut = 1 on completion.
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
        for (;;) {                      // agent redraw: first free cell
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
        for (;;) {                      // goal redraw: free and != agent cell
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

// ---------------------------------------------------------------- POMaze
// 11x11 PARTIAL-OBSERVABILITY maze (border walls; interior wall y=3 except
// the gap (8,3), interior wall y=7 except the gap (2,7)): three bands, a
// serpentine route. The agent sees ONLY the 8 adjacent cells (wall bits
// N,NE,E,SE,S,SW,W,NW), its own position, and the goal's compass direction
// — never the map; explored structure must be carried in memory. +1 on goal
// entry, done=1, agent+goal rejection-redrawn to free cells (same scheme as
// ProceduralMaze). CONTINUING. Lane state in gw[10]: 0=x, 1=y, 2=gx, 3=gy,
// 4=sinceGoal, 5=respawnIdx.
inline bool gw_po_is_wall(int x, int y) {
    if (x <= 0 || x >= 10 || y <= 0 || y >= 10) return true;
    if (y == 3 && x != 8) return true;
    if (y == 7 && x != 2) return true;
    return false;
}

inline void gw_po_obs(uint x, uint y, uint gx, uint gy, thread float *o) {
    int ix = int(x), iy = int(y);
    o[0] = gw_po_is_wall(ix + 0, iy - 1) ? 1.0f : 0.0f;   // N
    o[1] = gw_po_is_wall(ix + 1, iy - 1) ? 1.0f : 0.0f;   // NE
    o[2] = gw_po_is_wall(ix + 1, iy + 0) ? 1.0f : 0.0f;   // E
    o[3] = gw_po_is_wall(ix + 1, iy + 1) ? 1.0f : 0.0f;   // SE
    o[4] = gw_po_is_wall(ix + 0, iy + 1) ? 1.0f : 0.0f;   // S
    o[5] = gw_po_is_wall(ix - 1, iy + 1) ? 1.0f : 0.0f;   // SW
    o[6] = gw_po_is_wall(ix - 1, iy + 0) ? 1.0f : 0.0f;   // W
    o[7] = gw_po_is_wall(ix - 1, iy - 1) ? 1.0f : 0.0f;   // NW
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

// action: 0=up 1=down 2=left 3=right (same convention as FourRooms).
// Returns reward; obs written for the NEW state; *doneOut = 1 on completion.
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
        for (;;) {                      // agent redraw: first free cell
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
        for (;;) {                      // goal redraw: free and != agent cell
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

// ------------------------------------------------------------ DelayedChain
// Bare corridor, L=32: pos 0..31. action 1 moves pos+1, action 0 moves pos-1
// (clamp 0). Reward +1 ONLY at the far end (reaching pos 31), wrap to 0,
// done=1 — the cleanest long bare credit chain. Lane state in gw[10]:
// 0=pos, 1=lastA.
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

// -------------------------------------------------- case-gate probe kernel
// Reads a gridworld_ref.py case file's in-records (buffer 0), runs each
// through the SAME env-step functions the fleet env window uses, and writes
// out-records (buffer 1) in the case format. hdr = {count, inW, outW};
// inW selects the case type (8=FourRooms, 6=DeepSea, 11=KeyCorridor,
// 2=Chain, 12=LargeDeepSea, 15=MultiKeyCorridor, 7=MemoryChain,
// 13=ProceduralMaze, 1=ProceduralMazeGen, 9=HighDimMaze, 10=POMaze,
// 3=DelayedChain).
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
    } else if (inW == 6u) {
        // DeepSea in: [maskLo, maskHi, r, c, lastA, action]; mask words are
        // the uint32 bit pattern reinterpreted as float32.
        uint gw[6] = { uint(p[2]), uint(p[3]), uint(p[4]),
                       as_type<uint>(p[0]), as_type<uint>(p[1]), 0u };
        float obs[17];
        float r = gw_deep_step(gw, uint(p[5]), obs);
        o[0] = r;
        for (int i = 0; i < 3; i++) o[1 + i] = float(gw[i]);
        for (int i = 0; i < 17; i++) o[4 + i] = obs[i];
    } else if (inW == 11u) {
        // KeyCorridor in: [laneSeed, respawnIdx, x, y, kx, ky, gx, gy, flags,
        // sinceGoal, action] (11)
        uint gw[10] = { uint(p[2]), uint(p[3]), uint(p[4]), uint(p[5]),
                        uint(p[6]), uint(p[7]), uint(p[8]), uint(p[9]),
                        uint(p[1]), 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_kc_step(gw, uint(p[0]), uint(p[10]), obs, &done);
        o[0] = r;
        for (int i = 0; i < 9; i++) o[1 + i] = float(gw[i]);
        o[10] = float(done);
        for (int i = 0; i < 17; i++) o[11 + i] = obs[i];
    } else if (inW == 12u) {
        // LargeDeepSea in: [mask0..mask7, r, c, lastA, action]; mask words
        // are the uint32 bit pattern reinterpreted as float32.
        uint gw[10] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
        gw[0] = uint(p[8]) | (uint(p[9]) << 4u) | (uint(p[10]) << 8u);
        for (int i = 0; i < 8; i++) gw[1 + i] = as_type<uint>(p[i]);
        float obs[17];
        float r = gw_ldeep_step(gw, uint(p[11]), obs);
        o[0] = r;
        o[1] = float(gw[0] & 15u);
        o[2] = float((gw[0] >> 4u) & 15u);
        o[3] = float((gw[0] >> 8u) & 1u);
        for (int i = 0; i < 17; i++) o[4 + i] = obs[i];
    } else if (inW == 15u) {
        // MultiKeyCorridor in: [laneSeed, respawnIdx, x, y, k1x, k1y, k2x,
        // k2y, k3x, k3y, gx, gy, flags, sinceGoal, action] (15)
        uint gw[10] = { uint(p[2]), uint(p[3]),
                        uint(p[4]) | (uint(p[5]) << 4u),
                        uint(p[6]) | (uint(p[7]) << 4u),
                        uint(p[8]) | (uint(p[9]) << 4u),
                        uint(p[10]), uint(p[11]), uint(p[12]), uint(p[13]),
                        uint(p[1]) };
        float obs[17];
        uint done = 0u;
        float r = gw_mkc_step(gw, uint(p[0]), uint(p[14]), obs, &done);
        o[0] = r;
        o[1] = float(gw[0]); o[2] = float(gw[1]);
        o[3] = float(gw[2] & 15u); o[4] = float(gw[2] >> 4u);
        o[5] = float(gw[3] & 15u); o[6] = float(gw[3] >> 4u);
        o[7] = float(gw[4] & 15u); o[8] = float(gw[4] >> 4u);
        o[9] = float(gw[5]); o[10] = float(gw[6]);
        o[11] = float(gw[7]); o[12] = float(gw[8]); o[13] = float(gw[9]);
        o[14] = float(done);
        for (int i = 0; i < 17; i++) o[15 + i] = obs[i];
    } else if (inW == 7u) {
        // MemoryChain in: [pos, cue, lastA, respawnIdx, laneSeed, action,
        // pad] (7; the pad keeps inW distinct from DeepSea's 6)
        uint gw[10] = { uint(p[0]), uint(p[1]), uint(p[2]), uint(p[3]),
                        0u, 0u, 0u, 0u, 0u, 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_mchain_step(gw, uint(p[4]), uint(p[5]), obs, &done);
        o[0] = r;
        o[1] = float(gw[0]); o[2] = float(gw[1]); o[3] = float(gw[2]);
        o[4] = float(gw[3]); o[5] = float(done);
        for (int i = 0; i < 17; i++) o[6 + i] = obs[i];
    } else if (inW == 13u) {
        // ProceduralMaze in: [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal,
        // w0, w1, w2, action, pad, pad] (wall words as f32 bit patterns)
        uint gw[10] = { uint(p[2]), uint(p[3]), uint(p[4]), uint(p[5]),
                        uint(p[6]), uint(p[1]),
                        as_type<uint>(p[7]), as_type<uint>(p[8]),
                        as_type<uint>(p[9]), 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_pmaze_step(gw, uint(p[0]), uint(p[10]), obs, &done);
        o[0] = r;
        for (int i = 0; i < 6; i++) o[1 + i] = float(gw[i]);
        o[7] = float(done);
        for (int i = 0; i < 17; i++) o[8 + i] = obs[i];
    } else if (inW == 1u) {
        // ProceduralMazeGen in: [laneSeed] -> wall words as 16-bit halves
        uint w[3];
        gw_pmaze_gen(uint(p[0]), w);
        for (int i = 0; i < 3; i++) {
            o[2 * i] = float(w[i] & 0xffffu);
            o[2 * i + 1] = float(w[i] >> 16u);
        }
    } else if (inW == 9u) {
        // HighDimMaze in: [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal,
        // lastDir, action] (9)
        uint gw[10] = { uint(p[2]), uint(p[3]), uint(p[4]), uint(p[5]),
                        uint(p[6]), uint(p[1]), uint(p[7]), 0u, 0u, 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_hdm_step(gw, uint(p[0]), uint(p[8]), obs, &done);
        o[0] = r;
        for (int i = 0; i < 7; i++) o[1 + i] = float(gw[i]);
        o[8] = float(done);
        for (int i = 0; i < 17; i++) o[9 + i] = obs[i];
    } else if (inW == 10u) {
        // POMaze in: [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal, action,
        // pad, pad] (10)
        uint gw[10] = { uint(p[2]), uint(p[3]), uint(p[4]), uint(p[5]),
                        uint(p[6]), uint(p[1]), 0u, 0u, 0u, 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_po_step(gw, uint(p[0]), uint(p[7]), obs, &done);
        o[0] = r;
        for (int i = 0; i < 6; i++) o[1 + i] = float(gw[i]);
        o[7] = float(done);
        for (int i = 0; i < 17; i++) o[8 + i] = obs[i];
    } else if (inW == 3u) {
        // DelayedChain in: [pos, lastA, action] (3)
        uint gw[10] = { uint(p[0]), uint(p[1]), 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_dchain_step(gw, uint(p[2]), obs, &done);
        o[0] = r;
        o[1] = float(gw[0]); o[2] = float(gw[1]); o[3] = float(done);
        for (int i = 0; i < 17; i++) o[4 + i] = obs[i];
    } else {
        // Chain in: [pos, action] (2)
        uint gw[10] = { uint(p[0]), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
        float obs[17];
        uint done = 0u;
        float r = gw_chain_step(gw, uint(p[1]), obs, &done);
        o[0] = r;
        o[1] = float(gw[0]);
        o[2] = float(done);
        for (int i = 0; i < 17; i++) o[3 + i] = obs[i];
    }
}

#endif
