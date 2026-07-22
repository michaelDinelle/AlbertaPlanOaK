#!/usr/bin/env python3
"""CPU reference for the continuing Metal gridworlds — the certification contract.

Envs: FourRooms-cont-v0, DeepSea-cont-v0, KeyCorridor-cont-v0, Chain-cont-v0,
LargeDeepSea-cont-v0, MultiKeyCorridor-cont-v0, MemoryChain-cont-v0,
ProceduralMaze-cont-v0, HighDimMaze-cont-v0, POMaze-cont-v0,
DelayedChain-cont-v0.
All are CONTINUING (no done, no reset: goal/treasure/key/cue respawns inside
the stream), deterministic given the per-lane Philox chain, and defined so the
Metal port (Sources/Gridworld.metal) can match BITWISE.

Universal I/O contract (agent kernel is env-blind, Inv-1): every env emits
OBS_W=17 float32 observation channels and consumes the first N of POLICY_W=12
policy coordinates as categorical logits (N=4 FourRooms/KeyCorridor/
MultiKeyCorridor/ProceduralMaze/POMaze, N=2 DeepSea/Chain/LargeDeepSea/
MemoryChain/DelayedChain, N=6 HighDimMaze).

Bitwise rules both sides obey:
- state is integer (uint8/uint32); obs channels are computed as float32(int) OP
  float32(literal) with the exact op order written below;
- all randomness is Philox4x32-10, key=(laneSeed, stream), ctr=(idx, block, 0, 0),
  identical to tools/physics_parity.py:_philox4x32_10 (selftest cross-checks);
- u01(x) = float32(x >> 8) * 2^-24 (exact); uniform index draw = floor(u01 * NFREE).

Streams (0=RESET,1=CTRL,2=ACT taken by the physics/sampler): 3 = gridworld events.

Case format (all little-endian float32; ints transported as exact float32 values):
  FourRooms in : [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal, action]      (8)
  FourRooms out: [reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx, obs*17] (24)
  DeepSea in   : [maskLo, maskHi, r, c, lastA, action]                        (6)
  DeepSea out  : [reward, nr, nc, nLastA, obs*17]                            (21)
  KeyCorr in   : [laneSeed, respawnIdx, x, y, kx, ky, gx, gy, flags,
                  sinceGoal, action]                                         (11)
  KeyCorr out  : [reward, nx, ny, nkx, nky, ngx, ngy, nflags, nSinceGoal,
                  nRespawnIdx, done, obs*17]                                 (28)
  Chain in     : [pos, action]                                                (2)
  Chain out    : [reward, npos, done, obs*17]                                (20)
  LargeDS in   : [mask0..mask7, r, c, lastA, action]                         (12)
  LargeDS out  : [reward, nr, nc, nLastA, obs*17]                            (21)
  MultiKC in   : [laneSeed, respawnIdx, x, y, k1x, k1y, k2x, k2y, k3x, k3y,
                  gx, gy, flags, sinceGoal, action]                          (15)
  MultiKC out  : [reward, nx, ny, k1x, k1y, k2x, k2y, k3x, k3y, ngx, ngy,
                  nflags, nSinceGoal, nRespawnIdx, done, obs*17]             (32)
  MemChain in  : [pos, cue, lastA, respawnIdx, laneSeed, action, pad]         (7)
  MemChain out : [reward, npos, ncue, nlastA, nRespawnIdx, done, obs*17]     (23)
  ProcMaze in  : [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal, w0, w1, w2,
                  action, pad, pad]                                          (13)
  ProcMaze out : [reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx, done,
                  obs*17]                                                    (25)
  ProcGen in   : [laneSeed]                                                   (1)
  ProcGen out  : [w0lo, w0hi, w1lo, w1hi, w2lo, w2hi]  (16-bit halves)        (6)
  HighDim in   : [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal, lastDir,
                  action]                                                     (9)
  HighDim out  : [reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx, nLastDir,
                  done, obs*17]                                              (26)
  POMaze in    : [laneSeed, respawnIdx, x, y, gx, gy, sinceGoal, action,
                  pad, pad]                                                  (10)
  POMaze out   : [reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx, done,
                  obs*17]                                                    (25)
  DChain in    : [pos, lastA, action]                                         (3)
  DChain out   : [reward, npos, nlastA, done, obs*17]                        (21)
Mask/wall words are transported as their uint32 bit pattern reinterpreted as
float32. "ProceduralMaze-gen" is a CASE TYPE (the per-lane wall generator),
not an envId.

CLI:
  --emit-cases --env E --n 2000 --seed S --out cases.bin   (random case gen)
  --expect cases.bin                                        (print golden outputs)
  --compare cases.bin gpu.bin                               (bitwise array_equal)
  --selftest                                                (philox + determinism)
  --map --env E                                             (ASCII sanity print)
"""
import argparse, sys
import numpy as np

OBS_W = 17
FOUR_W, FOUR_H = 11, 11
DEEPSEA_N = 8
STREAM_GW = 3

# ---------------------------------------------------------------- Philox mirror
# Must match tools/physics_parity.py:_philox4x32_10 (selftest asserts this).
def _philox4x32_10(key0: int, key1: int, ctr) -> list[int]:
    M0, M1 = 0xD2511F53, 0xCD9E8D57
    W0, W1 = 0x9E3779B9, 0xBB67AE85
    c = list(ctr)
    k0, k1 = key0 & 0xFFFFFFFF, key1 & 0xFFFFFFFF
    for r in range(10):
        if r:
            k0 = (k0 + W0) & 0xFFFFFFFF
            k1 = (k1 + W1) & 0xFFFFFFFF
        p0 = (M0 * c[0]) & 0xFFFFFFFFFFFFFFFF
        p1 = (M1 * c[2]) & 0xFFFFFFFFFFFFFFFF
        hi0, lo0 = p0 >> 32, p0 & 0xFFFFFFFF
        hi1, lo1 = p1 >> 32, p1 & 0xFFFFFFFF
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
    return c

def _u01(x: int) -> np.float32:
    return np.float32(x >> 8) * np.float32(2.0**-24)

F32 = np.float32

# ---------------------------------------------------------------- FourRooms
# 11x11, border walls; vertical wall col x=5 except hallways y in {2,8};
# horizontal wall row y=5 except hallways x in {2,8}. Center (5,5) is wall.
def four_is_wall(x: int, y: int) -> bool:
    if x <= 0 or x >= 10 or y <= 0 or y >= 10:
        return True
    if x == 5 and y not in (2, 8):
        return True
    if y == 5 and x not in (2, 8):
        return True
    return False

FOUR_FREE = [(x, y) for y in range(FOUR_H) for x in range(FOUR_W) if not four_is_wall(x, y)]
assert len(FOUR_FREE) == 68, len(FOUR_FREE)
FOUR_NFREE = len(FOUR_FREE)  # 68: 121 cells - 40 border - 8 inner wall line cells

def four_room(x: int, y: int) -> int:  # 0=TL 1=TR 2=BL 3=BR ; -1 hallway
    if x < 5 and y < 5: return 0
    if x > 5 and y < 5: return 1
    if x < 5 and y > 5: return 2
    if x > 5 and y > 5: return 3
    return -1

def four_reset() -> tuple[int, int, int, int, int, int]:
    # (x, y, gx, gy, sinceGoal, respawnIdx) — deterministic, matches fleet init.
    return (1, 1, 9, 9, 0, 0)

def four_obs(x, y, gx, gy, since_goal):
    def w(dx, dy): return F32(1.0) if four_is_wall(x + dx, y + dy) else F32(0.0)
    room, groom = four_room(x, y), four_room(gx, gy)
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(x) / F32(10.0);  o[1] = F32(y) / F32(10.0)
    o[2] = F32(gx - x) / F32(10.0); o[3] = F32(gy - y) / F32(10.0)
    o[4] = F32(abs(gx - x) + abs(gy - y)) / F32(20.0)
    o[5] = F32(room == 0); o[6] = F32(room == 1); o[7] = F32(room == 2); o[8] = F32(room == 3)
    o[9], o[10], o[11], o[12] = w(0, -1), w(0, 1), w(-1, 0), w(1, 0)
    o[13] = F32(room == -1)
    o[14] = F32(groom == room and room != -1)
    o[15] = min(F32(since_goal) / F32(256.0), F32(1.0))
    o[16] = F32(1.0)
    return o

def four_step(lane_seed, respawn_idx, x, y, gx, gy, since_goal, action):
    """One step. Returns (reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx, obs).
    action: 0=up 1=down 2=left 3=right. Wall bump = stay, reward 0.
    Goal entry: reward +1, goal respawns to a uniformly drawn free cell != agent
    cell (Philox stream 3, ctr=(respawnIdx, block,0,0), words 0..3 then block++).
    """
    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action & 3]
    nx, ny = x + dx, y + dy
    if four_is_wall(nx, ny):
        nx, ny = x, y
    reward = F32(0.0)
    ngx, ngy, n_since, n_respawn = gx, gy, since_goal + 1, respawn_idx
    if (nx, ny) == (gx, gy):
        reward = F32(1.0)
        n_since = 0
        n_respawn = respawn_idx + 1
        block, word = 0, 0
        while True:
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            for w in raw[word:]:
                idx = int(_u01(w) * F32(FOUR_NFREE))
                if idx < FOUR_NFREE and FOUR_FREE[idx] != (nx, ny):
                    ngx, ngy = FOUR_FREE[idx]
                    return reward, nx, ny, ngx, ngy, n_since, n_respawn, four_obs(nx, ny, ngx, ngy, n_since)
            block, word = block + 1, 0
    return reward, nx, ny, ngx, ngy, n_since, n_respawn, four_obs(nx, ny, ngx, ngy, n_since)

# ---------------------------------------------------------------- DeepSea
# N=8. Hidden per-cell action map M[r][c] (bit r*8+c of the 64-bit mask, fixed
# per lane at reset): M==1 -> action 1 moves c+1, action 0 moves c-1 (normal);
# M==0 -> inverted. c clamped to [0,N-1]; r += 1 every step.
# Reward: -0.00125 if action==1 (the -0.01/N right-cost at N=8), +1 on reaching
# (r=N-1, c=N-1); on reaching r==N-1 the agent resurfaces to (0,0), mask kept.
DEEPSEA_RIGHT_COST = F32(-0.00125)

def deepsea_reset(lane_seed):
    raw = _philox4x32_10(lane_seed, 0, [0, 0, 0, 0])  # RESET stream
    return (0, 0, 0, raw[0], raw[1])  # (r, c, lastA, maskLo, maskHi)

def deepsea_obs(r, c, last_a):
    o = np.zeros(OBS_W, dtype=F32)
    n1 = F32(DEEPSEA_N - 1)
    o[0] = F32(r) / n1; o[1] = F32(c) / n1
    o[2] = F32((DEEPSEA_N - 1) - c) / n1; o[3] = F32((DEEPSEA_N - 1) - r) / n1
    o[4] = F32(c == DEEPSEA_N - 1); o[5] = F32(c == 0)
    o[6] = F32(last_a); o[7] = F32(1.0)
    return o

def deepsea_step(mask_lo, mask_hi, r, c, last_a, action):
    bit = (mask_lo >> (r * 8 + c)) & 1 if r * 8 + c < 32 else (mask_hi >> (r * 8 + c - 32)) & 1
    dc = 1 if (action & 1) == bit else -1
    nc = min(max(c + dc, 0), DEEPSEA_N - 1)
    nr = r + 1
    reward = DEEPSEA_RIGHT_COST if (action & 1) else F32(0.0)
    n_last = action & 1
    if nr >= DEEPSEA_N - 1:
        if nc == DEEPSEA_N - 1:
            reward = F32(reward + F32(1.0))
        nr, nc = 0, 0
    return reward, nr, nc, n_last, deepsea_obs(nr, nc, n_last)

# ---------------------------------------------------------------- KeyCorridor
# 9x7 (x 0..8, y 0..6), border walls; interior wall x=4 except the door cell
# (4,3). Left room x 1..3, right room x 5..7, y 1..5 (15 cells each). Key in
# the left room, goal in the right room. Auto-pickup on co-location with the
# key; the locked door cell opens on entry iff the key is carried (else
# blocked). Goal entry: +1, done=1, DeepSea-pattern respawn — agent+key
# redrawn in the left room, goal redrawn in the right room, door re-locked
# (Philox stream 3, ctr=(respawnIdx,0,0,0); words x=agent, y=key, z=goal;
# cell idx = trunc(u01(word)*15), room cell (1 or 5) + idx%3, 1 + idx//3).
KC_NCELL = 15
KC_LEFT = [(1 + i % 3, 1 + i // 3) for i in range(KC_NCELL)]
KC_RIGHT = [(5 + i % 3, 1 + i // 3) for i in range(KC_NCELL)]
KC_FREE = KC_LEFT + [(4, 3)] + KC_RIGHT

def kc_is_wall(x: int, y: int) -> bool:
    if x <= 0 or x >= 8 or y <= 0 or y >= 6:
        return True
    if x == 4 and y != 3:
        return True
    return False

def kc_reset() -> tuple:
    # (x, y, kx, ky, gx, gy, flags, sinceGoal, respawnIdx) — deterministic,
    # matches fleet init. flags bit0 hasKey, bit1 doorOpen.
    return (1, 1, 3, 5, 7, 1, 0, 0, 0)

def kc_obs(x, y, kx, ky, gx, gy, flags, since_goal):
    has_key = (flags & 1) != 0
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(x) / F32(8.0); o[1] = F32(y) / F32(6.0)
    o[2] = F32(has_key); o[3] = F32((flags & 2) != 0); o[4] = F32(not has_key)
    o[5] = F32(0.0) if has_key else F32(kx - x) / F32(8.0)
    o[6] = F32(0.0) if has_key else F32(ky - y) / F32(6.0)
    o[7] = F32(4 - x) / F32(8.0); o[8] = F32(3 - y) / F32(6.0)
    o[9] = F32(gx - x) / F32(8.0); o[10] = F32(gy - y) / F32(6.0)
    o[11] = F32(x == 4 and y == 3)
    o[12] = min(F32(since_goal) / F32(256.0), F32(1.0))
    o[13] = F32(1.0)
    return o

def kc_step(lane_seed, respawn_idx, x, y, kx, ky, gx, gy, flags, since_goal, action):
    """One step. Returns (reward, nx, ny, nkx, nky, ngx, ngy, nflags,
    nSinceGoal, nRespawnIdx, done, obs). action: 0=up 1=down 2=left 3=right.
    Wall bump = stay, reward 0. Locked door without key = stay; with key it
    opens (flags |= 2) and the agent enters. Key pickup on co-location."""
    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action & 3]
    nx, ny = x + dx, y + dy
    if kc_is_wall(nx, ny):
        nx, ny = x, y
    elif nx == 4 and ny == 3 and not (flags & 2):
        if flags & 1:
            flags |= 2
        else:
            nx, ny = x, y
    if not (flags & 1) and (nx, ny) == (kx, ky):
        flags |= 1
    reward, done, n_since, n_ri = F32(0.0), 0, since_goal + 1, respawn_idx
    if (nx, ny) == (gx, gy):
        reward, done, n_since, n_ri = F32(1.0), 1, 0, respawn_idx + 1
        raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, 0, 0, 0])
        ai = int(_u01(raw[0]) * F32(KC_NCELL))
        ki = int(_u01(raw[1]) * F32(KC_NCELL))
        gi = int(_u01(raw[2]) * F32(KC_NCELL))
        nx, ny = KC_LEFT[ai]
        kx, ky = KC_LEFT[ki]
        gx, gy = KC_RIGHT[gi]
        flags = 0
    return (reward, nx, ny, kx, ky, gx, gy, flags, n_since, n_ri, done,
            kc_obs(nx, ny, kx, ky, gx, gy, flags, n_since))

# ---------------------------------------------------------------- Chain
# bsuite-style chain, L=16: pos 0..15. action 1 moves pos+1, action 0 moves
# pos-1 (clamp 0). Delayed +1 at the chain end (pos 15), wrap to 0, done=1.
CHAIN_L = 16

def chain_obs(pos):
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(pos) / F32(16.0)
    o[1] = F32(1.0)
    return o

def chain_step(pos, action):
    """One step. Returns (reward, npos, done, obs)."""
    reward, done = F32(0.0), 0
    if action & 1:
        pos += 1
        if pos >= CHAIN_L - 1:
            reward, pos, done = F32(1.0), 0, 1
    else:
        pos = max(pos - 1, 0)
    return reward, pos, done, chain_obs(pos)

# ---------------------------------------------------------------- LargeDeepSea
# DeepSea scaled to N=16 (the canonical hard-exploration gridworld,
# exponentially harder with size). Same mechanics as N=8: hidden per-cell
# action map (bit r*16+c of the 256-bit mask, fixed per lane at reset);
# M==1 -> action 1 moves c+1, action 0 moves c-1 (normal); M==0 -> inverted.
# c clamped to [0,15]; r += 1 every step. Reward: -0.000625 if action==1 (the
# -0.01/N right-cost at N=16), +1 on reaching (r=15, c=15); on reaching r==15
# the agent resurfaces to (0,0), mask kept.
LDEEPSEA_N = 16
LDEEPSEA_RIGHT_COST = F32(-0.000625)

def ldeepsea_reset(lane_seed):
    raw0 = _philox4x32_10(lane_seed, 0, [0, 0, 0, 0])  # RESET stream
    raw1 = _philox4x32_10(lane_seed, 0, [1, 0, 0, 0])
    return (0, 0, 0, list(raw0) + list(raw1))  # (r, c, lastA, mask[8])

def ldeepsea_obs(r, c, last_a):
    o = np.zeros(OBS_W, dtype=F32)
    n1 = F32(LDEEPSEA_N - 1)
    o[0] = F32(r) / n1; o[1] = F32(c) / n1
    o[2] = F32((LDEEPSEA_N - 1) - c) / n1; o[3] = F32((LDEEPSEA_N - 1) - r) / n1
    o[4] = F32(c == LDEEPSEA_N - 1); o[5] = F32(c == 0)
    o[6] = F32(last_a); o[7] = F32(1.0)
    return o

def ldeepsea_step(mask, r, c, last_a, action):
    """One step. Returns (reward, nr, nc, nLastA, obs). mask = 8 uint32 words."""
    bit_idx = r * LDEEPSEA_N + c
    bit = (mask[bit_idx >> 5] >> (bit_idx & 31)) & 1
    dc = 1 if (action & 1) == bit else -1
    nc = min(max(c + dc, 0), LDEEPSEA_N - 1)
    nr = r + 1
    reward = LDEEPSEA_RIGHT_COST if (action & 1) else F32(0.0)
    n_last = action & 1
    if nr >= LDEEPSEA_N - 1:
        if nc == LDEEPSEA_N - 1:
            reward = F32(reward + F32(1.0))
        nr, nc = 0, 0
    return reward, nr, nc, n_last, ldeepsea_obs(nr, nc, n_last)

# ------------------------------------------------------- MultiKeyCorridor
# KeyCorridor extended to 3 sequential keys and doors. 17x7 (x 0..16, y 0..6),
# border walls; interior walls x=4, x=8, x=12 except the door cells (4,3),
# (8,3), (12,3). Four rooms x in {1..3, 5..7, 9..11, 13..15}, y 1..5 (15
# cells each): key1 in room 0, key2 in room 1, key3 in room 2, goal in room
# 3. Auto-pickup on co-location; the locked door j cell opens on entry iff
# key j is carried (else blocked). Goal entry: +1, done=1, DeepSea-pattern
# respawn — agent redrawn in room 0, keys redrawn in rooms 0/1/2, goal
# redrawn in room 3, doors re-locked (Philox stream 3,
# ctr=(respawnIdx,0,0,0): x=agent, y=key1, z=key2, w=key3; then
# ctr=(respawnIdx,1,0,0) word x=goal; cell idx = trunc(u01(word)*15), room
# base (1,5,9,13) + idx%3, 1 + idx//3).
MKC_NCELL = 15
MKC_BASE = (1, 5, 9, 13)
MKC_ROOMS = [[(b + i % 3, 1 + i // 3) for i in range(MKC_NCELL)] for b in MKC_BASE]
MKC_FREE = MKC_ROOMS[0] + MKC_ROOMS[1] + MKC_ROOMS[2] + MKC_ROOMS[3] + [
    (4, 3), (8, 3), (12, 3)]

def mkc_is_wall(x: int, y: int) -> bool:
    if x <= 0 or x >= 16 or y <= 0 or y >= 6:
        return True
    if x in (4, 8, 12) and y != 3:
        return True
    return False

def mkc_reset() -> tuple:
    # (x, y, k1, k2, k3, gx, gy, flags, sinceGoal, respawnIdx) —
    # deterministic, matches fleet init. flags bits0-2 hasKey1..3,
    # bits3-5 doorOpen1..3.
    return (1, 1, (3, 5), (7, 5), (11, 5), 15, 1, 0, 0, 0)

def mkc_obs(x, y, k1, k2, k3, gx, gy, flags):
    hk = [(flags & (1 << j)) != 0 for j in range(3)]
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(x) / F32(16.0); o[1] = F32(y) / F32(6.0)
    o[2] = F32(hk[0]); o[3] = F32(hk[1]); o[4] = F32(hk[2])
    o[5] = F32((flags & 8) != 0); o[6] = F32((flags & 16) != 0)
    o[7] = F32((flags & 32) != 0)
    for j, k in enumerate((k1, k2, k3)):
        o[8 + 2 * j] = F32(0.0) if hk[j] else F32(k[0] - x) / F32(16.0)
        o[9 + 2 * j] = F32(0.0) if hk[j] else F32(k[1] - y) / F32(6.0)
    o[14] = F32(gx - x) / F32(16.0); o[15] = F32(gy - y) / F32(6.0)
    o[16] = F32(1.0)
    return o

def mkc_step(lane_seed, respawn_idx, x, y, k1, k2, k3, gx, gy, flags,
             since_goal, action):
    """One step. Returns (reward, nx, ny, nk1, nk2, nk3, ngx, ngy, nflags,
    nSinceGoal, nRespawnIdx, done, obs). action: 0=up 1=down 2=left 3=right.
    Wall bump = stay; locked door without its key = stay, with its key it
    opens and the agent enters. Key pickup on co-location."""
    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action & 3]
    nx, ny = x + dx, y + dy
    if mkc_is_wall(nx, ny):
        nx, ny = x, y
    elif ny == 3 and nx in (4, 8, 12):
        j = (nx >> 2) - 1  # door index 0..2
        if not (flags & (8 << j)):
            if flags & (1 << j):
                flags |= 8 << j
            else:
                nx, ny = x, y
    keys = [k1, k2, k3]
    for j in range(3):
        if not (flags & (1 << j)) and (nx, ny) == keys[j]:
            flags |= 1 << j
    reward, done, n_since, n_ri = F32(0.0), 0, since_goal + 1, respawn_idx
    if (nx, ny) == (gx, gy):
        reward, done, n_since, n_ri = F32(1.0), 1, 0, respawn_idx + 1
        raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, 0, 0, 0])
        raw2 = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, 1, 0, 0])
        ai = int(_u01(raw[0]) * F32(MKC_NCELL))
        ki = [int(_u01(raw[j + 1]) * F32(MKC_NCELL)) for j in range(3)]
        gi = int(_u01(raw2[0]) * F32(MKC_NCELL))
        nx, ny = MKC_ROOMS[0][ai]
        keys = [MKC_ROOMS[j][ki[j]] for j in range(3)]
        gx, gy = MKC_ROOMS[3][gi]
        flags = 0
    k1, k2, k3 = keys
    return (reward, nx, ny, k1, k2, k3, gx, gy, flags, n_since, n_ri, done,
            mkc_obs(nx, ny, k1, k2, k3, gx, gy, flags))

# ------------------------------------------------------------- MemoryChain
# bsuite Memory Chain: corridor of length 32 (pos 0..31). The cue (0=left,
# 1=right) is observable ONLY at pos==0; the agent must carry it across the
# full corridor, and at the far end (pos==31) its action IS the choice: +1
# iff (action&1) == cue (0 otherwise), done=1, respawn at pos 0 with a
# freshly drawn cue (Philox stream 3, ctr=(respawnIdx,0,0,0), cue =
# raw[0]&1). Away from the end, action 1 walks pos+1, action 0 walks pos-1
# (clamp 0).
MCHAIN_L = 32

def mchain_reset() -> tuple:
    # (pos, cue, lastA, respawnIdx) — deterministic, matches fleet init.
    return (0, 0, 0, 0)

def mchain_obs(pos, cue, last_a):
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(pos) / F32(32.0)
    o[1] = F32(cue) if pos == 0 else F32(0.0)
    o[2] = F32(pos == 0)
    o[3] = F32(pos == MCHAIN_L - 1)
    o[4] = F32(last_a)
    o[5] = F32(1.0)
    return o

def mchain_step(lane_seed, pos, cue, last_a, respawn_idx, action):
    """One step. Returns (reward, npos, ncue, nlastA, nRespawnIdx, done, obs)."""
    reward, done = F32(0.0), 0
    if pos >= MCHAIN_L - 1:
        reward = F32(1.0) if (action & 1) == cue else F32(0.0)
        done = 1
        raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, 0, 0, 0])
        cue = raw[0] & 1
        respawn_idx += 1
        pos = 0
    elif action & 1:
        pos += 1
    else:
        pos = max(pos - 1, 0)
    n_last = action & 1
    return reward, pos, cue, n_last, respawn_idx, done, mchain_obs(pos, cue, n_last)

# --------------------------------------------------------- ProceduralMaze
# 11x11 (border walls) whose 9x9 interior WALL LAYOUT is generated per lane
# at reset from the lane's Philox chain (RESET stream) — deterministic per
# lane, different across lanes, then fixed for the lane's lifetime. Gen (once
# at fleet init, amortized; ~5 Philox calls): 81 wall bits in 3 words (bit
# idx = (x-1) + 9*(y-1), 1=wall, 50% density; w2 masked to its low 17 bits),
# then a random monotone path from (1,1) to (9,9) carved free (16 moves;
# move i, when not edge-forced, uses Philox RESET ctr=(1+i//4,0,0,0) word
# i%4, u01 < 0.5 -> right else down) — the carved path guarantees the goal is
# always reachable. Completion: +1, done=1, agent and goal rejection-redrawn
# to free cells (stream 3, ctr=(respawnIdx, block,0,0); agent from blocks
# 0,2,4..., goal from blocks 1,3,5... != agent cell; cell idx =
# trunc(u01(w)*81), cell (1 + idx%9, 1 + idx//9)). CONTINUING: the maze
# itself never resets.
PMAZE_FREE_CACHE = {}

def pmaze_gen(lane_seed):
    """Per-lane wall layout -> [w0, w1, w2] (w2 masked to 17 bits)."""
    raw = _philox4x32_10(lane_seed, 0, [0, 0, 0, 0])  # RESET stream
    w = [raw[0], raw[1], raw[2] & 0x1FFFF]
    x, y = 1, 1
    for i in range(16):
        idx = (x - 1) + 9 * (y - 1)
        w[idx >> 5] &= 0xFFFFFFFF ^ (1 << (idx & 31))
        if x == 9:
            y += 1
            continue
        if y == 9:
            x += 1
            continue
        c = _philox4x32_10(lane_seed, 0, [1 + i // 4, 0, 0, 0])
        if _u01(c[i % 4]) < F32(0.5):
            x += 1
        else:
            y += 1
    idx = 8 + 9 * 8  # (9,9)
    w[idx >> 5] &= 0xFFFFFFFF ^ (1 << (idx & 31))
    return w

def pmaze_is_wall(w, x: int, y: int) -> bool:
    if x <= 0 or x >= 10 or y <= 0 or y >= 10:
        return True
    idx = (x - 1) + 9 * (y - 1)
    return (w[idx >> 5] >> (idx & 31)) & 1 != 0

def pmaze_free(w):
    key = tuple(w)
    if key not in PMAZE_FREE_CACHE:
        PMAZE_FREE_CACHE[key] = [(x, y) for y in range(1, 10) for x in range(1, 10)
                                 if not pmaze_is_wall(w, x, y)]
    return PMAZE_FREE_CACHE[key]

def pmaze_obs(x, y, gx, gy, since_goal, w):
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(x) / F32(10.0); o[1] = F32(y) / F32(10.0)
    o[2] = F32(gx - x) / F32(10.0); o[3] = F32(gy - y) / F32(10.0)
    o[4] = F32(abs(gx - x) + abs(gy - y)) / F32(16.0)
    on = F32(pmaze_is_wall(w, x, y - 1)); os_ = F32(pmaze_is_wall(w, x, y + 1))
    ow = F32(pmaze_is_wall(w, x - 1, y)); oe = F32(pmaze_is_wall(w, x + 1, y))
    o[5], o[6], o[7], o[8] = on, os_, ow, oe
    o[9] = F32(pmaze_is_wall(w, x + 1, y - 1))
    o[10] = F32(pmaze_is_wall(w, x - 1, y - 1))
    o[11] = F32(pmaze_is_wall(w, x + 1, y + 1))
    o[12] = F32(pmaze_is_wall(w, x - 1, y + 1))
    o[13] = ((((F32(4.0) - on) - os_) - ow) - oe) / F32(4.0)
    o[14] = min(F32(since_goal) / F32(256.0), F32(1.0))
    o[15] = F32(x == gx or y == gy)
    o[16] = F32(1.0)
    return o

def pmaze_step(lane_seed, respawn_idx, x, y, gx, gy, since_goal, w, action):
    """One step. Returns (reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx,
    done, obs). action: 0=up 1=down 2=left 3=right. Wall bump = stay."""
    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action & 3]
    nx, ny = x + dx, y + dy
    if pmaze_is_wall(w, nx, ny):
        nx, ny = x, y
    reward, done, n_since, n_ri = F32(0.0), 0, since_goal + 1, respawn_idx
    if (nx, ny) == (gx, gy):
        reward, done, n_since, n_ri = F32(1.0), 1, 0, respawn_idx + 1
        block = 0
        while True:  # agent redraw: first free cell
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            found = False
            for wd in raw:
                idx = int(_u01(wd) * F32(81))
                cx, cy = 1 + idx % 9, 1 + idx // 9
                if not pmaze_is_wall(w, cx, cy):
                    nx, ny = cx, cy
                    found = True
                    break
            if found:
                break
            block += 2
        block = 1
        while True:  # goal redraw: free and != agent cell
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            found = False
            for wd in raw:
                idx = int(_u01(wd) * F32(81))
                cx, cy = 1 + idx % 9, 1 + idx // 9
                if not pmaze_is_wall(w, cx, cy) and (cx, cy) != (nx, ny):
                    gx, gy = cx, cy
                    found = True
                    break
            if found:
                break
            block += 2
    return (reward, nx, ny, gx, gy, n_since, n_ri, done,
            pmaze_obs(nx, ny, gx, gy, n_since, w))

# ------------------------------------------------------------ HighDimMaze
# 11x11 serpentine maze (border walls; interior wall x=3 except the gap
# (3,9), interior wall x=7 except the gap (7,1)): three corridors — down
# x1..2, through (3,9), up x4..6, through (7,1), down x8..9. SIX discrete
# actions: 0=up 1=down 2=left 3=right, 4=stay, 5=dash (up to two cells in
# lastDir, stopping at walls; lastDir = the last cardinal action, init down).
# Rich 16-channel observation: position, 4 wall sensors, goal relative
# position, and an 8-bit binary position code. +1 on goal entry, done=1,
# agent+goal rejection-redrawn to free cells (same scheme as ProceduralMaze).
def hdm_is_wall(x: int, y: int) -> bool:
    if x <= 0 or x >= 10 or y <= 0 or y >= 10:
        return True
    if x == 3 and y != 9:
        return True
    if x == 7 and y != 1:
        return True
    return False

HDM_FREE = [(x, y) for y in range(1, 10) for x in range(1, 10) if not hdm_is_wall(x, y)]

def hdm_obs(x, y, gx, gy):
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(x) / F32(16.0); o[1] = F32(y) / F32(16.0)
    o[2] = F32(hdm_is_wall(x, y - 1)); o[3] = F32(hdm_is_wall(x, y + 1))
    o[4] = F32(hdm_is_wall(x - 1, y)); o[5] = F32(hdm_is_wall(x + 1, y))
    o[6] = F32(gx - x) / F32(16.0); o[7] = F32(gy - y) / F32(16.0)
    for b in range(4):
        o[8 + b] = F32((x >> b) & 1)
        o[12 + b] = F32((y >> b) & 1)
    return o

def hdm_step(lane_seed, respawn_idx, x, y, gx, gy, since_goal, last_dir, action):
    """One step. Returns (reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx,
    nLastDir, done, obs). action: 0=up 1=down 2=left 3=right 4=stay 5=dash."""
    a = action % 6
    nx, ny = x, y
    if a < 4:
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[a]
        last_dir = a
        if not hdm_is_wall(nx + dx, ny + dy):
            nx, ny = nx + dx, ny + dy
    elif a == 5:
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[last_dir & 3]
        if not hdm_is_wall(nx + dx, ny + dy):
            nx, ny = nx + dx, ny + dy
            if not hdm_is_wall(nx + dx, ny + dy):
                nx, ny = nx + dx, ny + dy
    reward, done, n_since, n_ri = F32(0.0), 0, since_goal + 1, respawn_idx
    if (nx, ny) == (gx, gy):
        reward, done, n_since, n_ri = F32(1.0), 1, 0, respawn_idx + 1
        block = 0
        while True:  # agent redraw: first free cell
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            found = False
            for wd in raw:
                idx = int(_u01(wd) * F32(81))
                cx, cy = 1 + idx % 9, 1 + idx // 9
                if not hdm_is_wall(cx, cy):
                    nx, ny = cx, cy
                    found = True
                    break
            if found:
                break
            block += 2
        block = 1
        while True:  # goal redraw: free and != agent cell
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            found = False
            for wd in raw:
                idx = int(_u01(wd) * F32(81))
                cx, cy = 1 + idx % 9, 1 + idx // 9
                if not hdm_is_wall(cx, cy) and (cx, cy) != (nx, ny):
                    gx, gy = cx, cy
                    found = True
                    break
            if found:
                break
            block += 2
    return (reward, nx, ny, gx, gy, n_since, n_ri, last_dir, done,
            hdm_obs(nx, ny, gx, gy))

# ---------------------------------------------------------------- POMaze
# 11x11 PARTIAL-OBSERVABILITY maze (border walls; interior wall y=3 except
# the gap (8,3), interior wall y=7 except the gap (2,7)): three bands, a
# serpentine route. The agent sees ONLY the 8 adjacent cells (wall bits
# N,NE,E,SE,S,SW,W,NW), its own position, and the goal's compass direction —
# never the map. +1 on goal entry, done=1, agent+goal rejection-redrawn to
# free cells (same scheme as ProceduralMaze).
def po_is_wall(x: int, y: int) -> bool:
    if x <= 0 or x >= 10 or y <= 0 or y >= 10:
        return True
    if y == 3 and x != 8:
        return True
    if y == 7 and x != 2:
        return True
    return False

PO_FREE = [(x, y) for y in range(1, 10) for x in range(1, 10) if not po_is_wall(x, y)]

def po_obs(x, y, gx, gy):
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(po_is_wall(x, y - 1))     # N
    o[1] = F32(po_is_wall(x + 1, y - 1))  # NE
    o[2] = F32(po_is_wall(x + 1, y))     # E
    o[3] = F32(po_is_wall(x + 1, y + 1))  # SE
    o[4] = F32(po_is_wall(x, y + 1))     # S
    o[5] = F32(po_is_wall(x - 1, y + 1))  # SW
    o[6] = F32(po_is_wall(x - 1, y))     # W
    o[7] = F32(po_is_wall(x - 1, y - 1))  # NW
    o[8] = F32(x) / F32(10.0); o[9] = F32(y) / F32(10.0)
    o[10] = F32(gx > x); o[11] = F32(gx < x)
    o[12] = F32(gy > y); o[13] = F32(gy < y)
    o[14] = F32(1.0)
    return o

def po_step(lane_seed, respawn_idx, x, y, gx, gy, since_goal, action):
    """One step. Returns (reward, nx, ny, ngx, ngy, nSinceGoal, nRespawnIdx,
    done, obs). action: 0=up 1=down 2=left 3=right. Wall bump = stay."""
    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action & 3]
    nx, ny = x + dx, y + dy
    if po_is_wall(nx, ny):
        nx, ny = x, y
    reward, done, n_since, n_ri = F32(0.0), 0, since_goal + 1, respawn_idx
    if (nx, ny) == (gx, gy):
        reward, done, n_since, n_ri = F32(1.0), 1, 0, respawn_idx + 1
        block = 0
        while True:  # agent redraw: first free cell
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            found = False
            for wd in raw:
                idx = int(_u01(wd) * F32(81))
                cx, cy = 1 + idx % 9, 1 + idx // 9
                if not po_is_wall(cx, cy):
                    nx, ny = cx, cy
                    found = True
                    break
            if found:
                break
            block += 2
        block = 1
        while True:  # goal redraw: free and != agent cell
            raw = _philox4x32_10(lane_seed, STREAM_GW, [respawn_idx, block, 0, 0])
            found = False
            for wd in raw:
                idx = int(_u01(wd) * F32(81))
                cx, cy = 1 + idx % 9, 1 + idx // 9
                if not po_is_wall(cx, cy) and (cx, cy) != (nx, ny):
                    gx, gy = cx, cy
                    found = True
                    break
            if found:
                break
            block += 2
    return (reward, nx, ny, gx, gy, n_since, n_ri, done,
            po_obs(nx, ny, gx, gy))

# ------------------------------------------------------------ DelayedChain
# Bare corridor, L=32: pos 0..31. action 1 moves pos+1, action 0 moves pos-1
# (clamp 0). Reward +1 ONLY at the far end (reaching pos 31), wrap to 0,
# done=1 — the cleanest long bare credit chain.
DCHAIN_L = 32

def dchain_reset() -> tuple:
    return (0, 0)  # (pos, lastA) — deterministic, matches fleet init

def dchain_obs(pos, last_a):
    o = np.zeros(OBS_W, dtype=F32)
    o[0] = F32(pos) / F32(32.0)
    o[1] = F32(last_a)
    o[2] = F32(1.0)
    return o

def dchain_step(pos, last_a, action):
    """One step. Returns (reward, npos, nlastA, done, obs)."""
    reward, done = F32(0.0), 0
    if action & 1:
        pos += 1
        if pos >= DCHAIN_L - 1:
            reward, pos, done = F32(1.0), 0, 1
    else:
        pos = max(pos - 1, 0)
    n_last = action & 1
    return reward, pos, n_last, done, dchain_obs(pos, n_last)

# ---------------------------------------------------------------- case I/O
def _u32f(x: int) -> F32:
    return np.array([x & 0xFFFFFFFF], dtype=np.uint32).view(F32)[0]

def gen_cases(env: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    ins, outs = [], []
    for _ in range(n):
        if env == "FourRooms-cont-v0":
            ls = int(rng.integers(0, 2**31))
            # The case format transports ints as exact float32 values; f32
            # rounds above 2**24, so the TRANSPORTED value is the contract:
            # round-trip the seed through f32 and use that integer for both
            # the record and the golden output (uint(F32(ls)) == ls exactly).
            ls = int(F32(ls))
            ri = int(rng.integers(0, 64))
            x, y = FOUR_FREE[int(rng.integers(0, FOUR_NFREE))]
            gx, gy = FOUR_FREE[int(rng.integers(0, FOUR_NFREE))]
            sg = int(rng.integers(0, 1024))
            a = int(rng.integers(0, 4))
            ins.append([F32(ls), F32(ri), F32(x), F32(y), F32(gx), F32(gy), F32(sg), F32(a)])
            r, nx, ny, ngx, ngy, nsg, nri, obs = four_step(ls, ri, x, y, gx, gy, sg, a)
            outs.append(np.concatenate([[r, F32(nx), F32(ny), F32(ngx), F32(ngy), F32(nsg), F32(nri)], obs]))
        elif env == "DeepSea-cont-v0":
            ml, mh = int(rng.integers(0, 2**32)), int(rng.integers(0, 2**32))
            r_, c_ = int(rng.integers(0, 8)), int(rng.integers(0, 8))
            la, a = int(rng.integers(0, 2)), int(rng.integers(0, 2))
            ins.append([_u32f(ml), _u32f(mh), F32(r_), F32(c_), F32(la), F32(a)])
            rw, nr, nc, nla, obs = deepsea_step(ml, mh, r_, c_, la, a)
            outs.append(np.concatenate([[rw, F32(nr), F32(nc), F32(nla)], obs]))
        elif env == "KeyCorridor-cont-v0":
            ls = int(F32(int(rng.integers(0, 2**31))))
            ri = int(rng.integers(0, 64))
            x, y = KC_FREE[int(rng.integers(0, len(KC_FREE)))]
            kx, ky = KC_LEFT[int(rng.integers(0, KC_NCELL))]
            gx, gy = KC_RIGHT[int(rng.integers(0, KC_NCELL))]
            flags = int(rng.integers(0, 4))
            sg = int(rng.integers(0, 1024))
            a = int(rng.integers(0, 4))
            u = rng.random()
            if u < 0.25:
                # force the completion/respawn path: step from a free neighbor
                # cell straight into the goal
                cands = []
                for ddx, ddy, aa in ((1, 0, 2), (-1, 0, 3), (0, 1, 0), (0, -1, 1)):
                    cx, cy = gx + ddx, gy + ddy
                    if not kc_is_wall(cx, cy) and not (cx == 4 and cy == 3):
                        cands.append((cx, cy, aa))
                if cands:
                    x, y, a = cands[int(rng.integers(0, len(cands)))]
            elif u < 0.5:
                # force the door path: step into the door cell with random flags
                x, y, a = (3, 3, 3) if rng.random() < 0.5 else (5, 3, 2)
            ins.append([F32(ls), F32(ri), F32(x), F32(y), F32(kx), F32(ky),
                        F32(gx), F32(gy), F32(flags), F32(sg), F32(a)])
            (r, nx, ny, nkx, nky, ngx, ngy, nfl, nsg, nri, dn, obs) = kc_step(
                ls, ri, x, y, kx, ky, gx, gy, flags, sg, a)
            outs.append(np.concatenate([
                [r, F32(nx), F32(ny), F32(nkx), F32(nky), F32(ngx), F32(ngy),
                 F32(nfl), F32(nsg), F32(nri), F32(dn)], obs]))
        elif env == "LargeDeepSea-cont-v0":
            mask = [int(rng.integers(0, 2**32)) for _ in range(8)]
            r_, c_ = int(rng.integers(0, 16)), int(rng.integers(0, 16))
            la, a = int(rng.integers(0, 2)), int(rng.integers(0, 2))
            if rng.random() < 0.25:
                # force the resurface path (and sometimes the treasure)
                r_ = 14
            ins.append([_u32f(m) for m in mask] + [F32(r_), F32(c_), F32(la), F32(a)])
            rw, nr, nc, nla, obs = ldeepsea_step(mask, r_, c_, la, a)
            outs.append(np.concatenate([[rw, F32(nr), F32(nc), F32(nla)], obs]))
        elif env == "MultiKeyCorridor-cont-v0":
            ls = int(F32(int(rng.integers(0, 2**31))))
            ri = int(rng.integers(0, 64))
            x, y = MKC_FREE[int(rng.integers(0, len(MKC_FREE)))]
            ks = [MKC_ROOMS[j][int(rng.integers(0, MKC_NCELL))] for j in range(3)]
            gx, gy = MKC_ROOMS[3][int(rng.integers(0, MKC_NCELL))]
            flags = int(rng.integers(0, 64))
            sg = int(rng.integers(0, 1024))
            a = int(rng.integers(0, 4))
            u = rng.random()
            if u < 0.25:
                # force the completion/respawn path: step from a free neighbor
                # cell straight into the goal
                cands = []
                for ddx, ddy, aa in ((1, 0, 2), (-1, 0, 3), (0, 1, 0), (0, -1, 1)):
                    cx, cy = gx + ddx, gy + ddy
                    if not mkc_is_wall(cx, cy) and not (cx in (4, 8, 12) and cy == 3):
                        cands.append((cx, cy, aa))
                if cands:
                    x, y, a = cands[int(rng.integers(0, len(cands)))]
            elif u < 0.5:
                # force a door path: step into a random door cell (random flags)
                d0 = 4 * int(rng.integers(1, 4))
                x, y, a = (d0 - 1, 3, 3) if rng.random() < 0.5 else (d0 + 1, 3, 2)
            ins.append([F32(ls), F32(ri), F32(x), F32(y),
                        F32(ks[0][0]), F32(ks[0][1]), F32(ks[1][0]), F32(ks[1][1]),
                        F32(ks[2][0]), F32(ks[2][1]), F32(gx), F32(gy),
                        F32(flags), F32(sg), F32(a)])
            (r, nx, ny, nk1, nk2, nk3, ngx, ngy, nfl, nsg, nri, dn, obs) = mkc_step(
                ls, ri, x, y, ks[0], ks[1], ks[2], gx, gy, flags, sg, a)
            outs.append(np.concatenate([
                [r, F32(nx), F32(ny), F32(nk1[0]), F32(nk1[1]), F32(nk2[0]),
                 F32(nk2[1]), F32(nk3[0]), F32(nk3[1]), F32(ngx), F32(ngy),
                 F32(nfl), F32(nsg), F32(nri), F32(dn)], obs]))
        elif env == "MemoryChain-cont-v0":
            ls = int(F32(int(rng.integers(0, 2**31))))
            pos = int(rng.integers(0, MCHAIN_L))
            cue, la = int(rng.integers(0, 2)), int(rng.integers(0, 2))
            ri = int(rng.integers(0, 64))
            a = int(rng.integers(0, 2))
            if rng.random() < 0.3:
                pos = MCHAIN_L - 1   # force the choice/respawn path
            ins.append([F32(pos), F32(cue), F32(la), F32(ri), F32(ls), F32(a),
                        F32(0.0)])
            r, npos, ncue, nla, nri, dn, obs = mchain_step(ls, pos, cue, la, ri, a)
            outs.append(np.concatenate([
                [r, F32(npos), F32(ncue), F32(nla), F32(nri), F32(dn)], obs]))
        elif env == "ProceduralMaze-cont-v0":
            ls = int(F32(int(rng.integers(0, 2**31))))
            w = pmaze_gen(ls)
            free = pmaze_free(w)
            ri = int(rng.integers(0, 64))
            x, y = free[int(rng.integers(0, len(free)))]
            gx, gy = free[int(rng.integers(0, len(free)))]
            sg = int(rng.integers(0, 1024))
            a = int(rng.integers(0, 4))
            if rng.random() < 0.25:
                # force the completion/respawn path: step from a free neighbor
                # cell straight into the goal
                cands = []
                for ddx, ddy, aa in ((1, 0, 2), (-1, 0, 3), (0, 1, 0), (0, -1, 1)):
                    cx, cy = gx + ddx, gy + ddy
                    if not pmaze_is_wall(w, cx, cy):
                        cands.append((cx, cy, aa))
                if cands:
                    x, y, a = cands[int(rng.integers(0, len(cands)))]
            ins.append([F32(ls), F32(ri), F32(x), F32(y), F32(gx), F32(gy),
                        F32(sg), _u32f(w[0]), _u32f(w[1]), _u32f(w[2]), F32(a),
                        F32(0.0), F32(0.0)])
            (r, nx, ny, ngx, ngy, nsg, nri, dn, obs) = pmaze_step(
                ls, ri, x, y, gx, gy, sg, w, a)
            outs.append(np.concatenate([
                [r, F32(nx), F32(ny), F32(ngx), F32(ngy), F32(nsg), F32(nri),
                 F32(dn)], obs]))
        elif env == "ProceduralMaze-gen":
            ls = int(F32(int(rng.integers(0, 2**31))))
            w = pmaze_gen(ls)
            ins.append([F32(ls)])
            outs.append(np.array(
                [F32(w[0] & 0xFFFF), F32(w[0] >> 16), F32(w[1] & 0xFFFF),
                 F32(w[1] >> 16), F32(w[2] & 0xFFFF), F32(w[2] >> 16)], dtype=F32))
        elif env == "HighDimMaze-cont-v0":
            ls = int(F32(int(rng.integers(0, 2**31))))
            ri = int(rng.integers(0, 64))
            x, y = HDM_FREE[int(rng.integers(0, len(HDM_FREE)))]
            gx, gy = HDM_FREE[int(rng.integers(0, len(HDM_FREE)))]
            ld = int(rng.integers(0, 4))
            sg = int(rng.integers(0, 1024))
            a = int(rng.integers(0, 6))
            u = rng.random()
            if (u < 0.125 and gx >= 3 and not hdm_is_wall(gx - 1, gy)
                    and not hdm_is_wall(gx - 2, gy)):
                # force completion via a dash: two cells left of the goal
                x, y, ld, a = gx - 2, gy, 3, 5
            elif u < 0.25:
                # force completion via a walk-in from a free neighbor cell
                cands = []
                for ddx, ddy, aa in ((1, 0, 2), (-1, 0, 3), (0, 1, 0), (0, -1, 1)):
                    cx, cy = gx + ddx, gy + ddy
                    if not hdm_is_wall(cx, cy):
                        cands.append((cx, cy, aa))
                if cands:
                    x, y, a = cands[int(rng.integers(0, len(cands)))]
            ins.append([F32(ls), F32(ri), F32(x), F32(y), F32(gx), F32(gy),
                        F32(sg), F32(ld), F32(a)])
            (r, nx, ny, ngx, ngy, nsg, nri, nld, dn, obs) = hdm_step(
                ls, ri, x, y, gx, gy, sg, ld, a)
            outs.append(np.concatenate([
                [r, F32(nx), F32(ny), F32(ngx), F32(ngy), F32(nsg), F32(nri),
                 F32(nld), F32(dn)], obs]))
        elif env == "POMaze-cont-v0":
            ls = int(F32(int(rng.integers(0, 2**31))))
            ri = int(rng.integers(0, 64))
            x, y = PO_FREE[int(rng.integers(0, len(PO_FREE)))]
            gx, gy = PO_FREE[int(rng.integers(0, len(PO_FREE)))]
            sg = int(rng.integers(0, 1024))
            a = int(rng.integers(0, 4))
            if rng.random() < 0.25:
                cands = []
                for ddx, ddy, aa in ((1, 0, 2), (-1, 0, 3), (0, 1, 0), (0, -1, 1)):
                    cx, cy = gx + ddx, gy + ddy
                    if not po_is_wall(cx, cy):
                        cands.append((cx, cy, aa))
                if cands:
                    x, y, a = cands[int(rng.integers(0, len(cands)))]
            ins.append([F32(ls), F32(ri), F32(x), F32(y), F32(gx), F32(gy),
                        F32(sg), F32(a), F32(0.0), F32(0.0)])
            (r, nx, ny, ngx, ngy, nsg, nri, dn, obs) = po_step(
                ls, ri, x, y, gx, gy, sg, a)
            outs.append(np.concatenate([
                [r, F32(nx), F32(ny), F32(ngx), F32(ngy), F32(nsg), F32(nri),
                 F32(dn)], obs]))
        elif env == "DelayedChain-cont-v0":
            pos = int(rng.integers(0, DCHAIN_L))
            la, a = int(rng.integers(0, 2)), int(rng.integers(0, 2))
            if rng.random() < 0.3:
                pos, a = DCHAIN_L - 2, 1   # force the far-end reward/wrap path
            ins.append([F32(pos), F32(la), F32(a)])
            r, npos, nla, dn, obs = dchain_step(pos, la, a)
            outs.append(np.concatenate([[r, F32(npos), F32(nla), F32(dn)], obs]))
        else:  # Chain-cont-v0
            pos = int(rng.integers(0, CHAIN_L))
            a = int(rng.integers(0, 2))
            if rng.random() < 0.25:
                pos, a = CHAIN_L - 2, 1   # force the chain-end wrap path
            ins.append([F32(pos), F32(a)])
            r, npos, dn, obs = chain_step(pos, a)
            outs.append(np.concatenate([[r, F32(npos), F32(dn)], obs]))
    return (np.array(ins, dtype=F32), np.array(outs, dtype=F32))

def ascii_map(env: str) -> str:
    if env == "FourRooms-cont-v0":
        rows = []
        for y in range(FOUR_H):
            rows.append("".join("#" if four_is_wall(x, y) else "." for x in range(FOUR_W)))
        return "\n".join(rows)
    if env == "KeyCorridor-cont-v0":
        rows = []
        for y in range(7):
            rows.append("".join(
                "D" if (x, y) == (4, 3) else ("#" if kc_is_wall(x, y) else ".")
                for x in range(9)))
        return "\n".join(rows)
    if env == "Chain-cont-v0":
        return f"Chain L={CHAIN_L}: pos 0..{CHAIN_L - 1}, +1 at pos {CHAIN_L - 1}, wrap to 0"
    if env == "MultiKeyCorridor-cont-v0":
        rows = []
        for y in range(7):
            rows.append("".join(
                "D" if (x in (4, 8, 12) and y == 3) else ("#" if mkc_is_wall(x, y) else ".")
                for x in range(17)))
        return "\n".join(rows)
    if env == "MemoryChain-cont-v0":
        return (f"MemoryChain L={MCHAIN_L}: pos 0..{MCHAIN_L - 1}, cue visible at pos 0, "
                f"choice at pos {MCHAIN_L - 1} (+1 iff action == cue)")
    if env == "LargeDeepSea-cont-v0":
        return f"DeepSea {LDEEPSEA_N}x{LDEEPSEA_N}: mask-dependent columns; treasure at (15,15)"
    if env == "ProceduralMaze-cont-v0":
        w = pmaze_gen(0)
        rows = []
        for y in range(11):
            rows.append("".join("#" if pmaze_is_wall(w, x, y) else "." for x in range(11)))
        return "\n".join(rows) + "\n(ProceduralMaze layout for lane seed 0)"
    if env == "HighDimMaze-cont-v0":
        rows = []
        for y in range(11):
            rows.append("".join("#" if hdm_is_wall(x, y) else "." for x in range(11)))
        return "\n".join(rows)
    if env == "POMaze-cont-v0":
        rows = []
        for y in range(11):
            rows.append("".join("#" if po_is_wall(x, y) else "." for x in range(11)))
        return "\n".join(rows)
    if env == "DelayedChain-cont-v0":
        return (f"DelayedChain L={DCHAIN_L}: pos 0..{DCHAIN_L - 1}, +1 ONLY at pos "
                f"{DCHAIN_L - 1}, wrap to 0")
    return f"DeepSea {DEEPSEA_N}x{DEEPSEA_N}: mask-dependent columns; treasure at (7,7)"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=["FourRooms-cont-v0", "DeepSea-cont-v0",
                                      "KeyCorridor-cont-v0", "Chain-cont-v0",
                                      "LargeDeepSea-cont-v0", "MultiKeyCorridor-cont-v0",
                                      "MemoryChain-cont-v0", "ProceduralMaze-cont-v0",
                                      "ProceduralMaze-gen", "HighDimMaze-cont-v0",
                                      "POMaze-cont-v0", "DelayedChain-cont-v0"],
                    default="FourRooms-cont-v0")
    ap.add_argument("--emit-cases", metavar="OUT")
    ap.add_argument("--expect", metavar="CASES")
    ap.add_argument("--compare", nargs=2, metavar=("CASES", "GPU"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if args.selftest:
        try:
            from physics_parity import _philox4x32_10 as ref
            for k0, k1, ctr in [(0, 3, [0, 0, 0, 0]), (12345, 3, [7, 1, 0, 0]), (2**31 - 1, 3, [63, 0, 0, 0])]:
                assert _philox4x32_10(k0, k1, ctr) == ref(k0, k1, ctr), (k0, k1, ctr)
            print("philox mirror: MATCHES physics_parity")
        except ImportError:
            print("philox mirror: physics_parity not importable (mujoco absent); local copy in force")
        for env in ("FourRooms-cont-v0", "DeepSea-cont-v0", "KeyCorridor-cont-v0",
                    "Chain-cont-v0", "LargeDeepSea-cont-v0", "MultiKeyCorridor-cont-v0",
                    "MemoryChain-cont-v0", "ProceduralMaze-cont-v0", "ProceduralMaze-gen",
                    "HighDimMaze-cont-v0", "POMaze-cont-v0", "DelayedChain-cont-v0"):
            a_i, a_o = gen_cases(env, 64, 999)
            b_i, b_o = gen_cases(env, 64, 999)
            # Compare the uint32 bit view: mask words transported as f32 bit
            # patterns can BE NaN values (array_equal would fail NaN != NaN
            # even on bit-identical arrays).
            assert (np.array_equal(a_o.view(np.uint32), b_o.view(np.uint32))
                    and np.array_equal(a_i.view(np.uint32), b_i.view(np.uint32)))
            print(f"{env}: determinism OK (64 cases)")
        print(ascii_map("FourRooms-cont-v0"))
        print(ascii_map("KeyCorridor-cont-v0"))
        print(ascii_map("MultiKeyCorridor-cont-v0"))
        print(ascii_map("ProceduralMaze-cont-v0"))
        print(ascii_map("HighDimMaze-cont-v0"))
        print(ascii_map("POMaze-cont-v0"))
        return 0
    if args.map:
        print(ascii_map(args.env)); return 0
    if args.emit_cases:
        ins, outs = gen_cases(args.env, args.n, args.seed)
        with open(args.emit_cases, "wb") as f:
            np.array([ins.shape[0], ins.shape[1], outs.shape[1]], dtype=np.uint32).tofile(f)
            ins.tofile(f); outs.tofile(f)
        print(f"wrote {args.n} {args.env} cases -> {args.emit_cases} (in {ins.shape[1]} f32, out {outs.shape[1]} f32)")
        return 0
    def load(p):
        d = np.fromfile(p, dtype=F32)
        n, wi, wo = np.fromfile(p, dtype=np.uint32, count=3)
        off = 3
        ins = d[off:off + n * wi].reshape(n, wi)
        outs = d[off + n * wi:off + n * wi + n * wo].reshape(n, wo)
        return ins, outs
    if args.expect:
        ins, outs = load(args.expect)
        print(f"{len(ins)} cases; first out: {outs[0]}"); return 0
    if args.compare:
        ins, exp = load(args.compare[0])
        gpu = np.fromfile(args.compare[1], dtype=F32).reshape(exp.shape)
        eq = np.array_equal(gpu, exp)
        if not eq:
            bad = np.argwhere(~np.isclose(gpu, exp, rtol=0, atol=0))
            print(f"BITWISE FAIL: {len(bad)} mismatched fields; first: case {bad[0][0]} field {bad[0][1]} gpu={gpu[tuple(bad[0])]} ref={exp[tuple(bad[0])]}")
            return 1
        print(f"BITWISE PASS: {len(exp)} cases x {exp.shape[1]} fields identical")
        return 0
    ap.print_help(); return 0

if __name__ == "__main__":
    sys.exit(main())
