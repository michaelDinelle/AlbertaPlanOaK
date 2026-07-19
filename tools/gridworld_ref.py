#!/usr/bin/env python3
"""CPU reference for the continuing Metal gridworlds — the certification contract.

Envs: FourRooms-cont-v0, DeepSea-cont-v0. Both are CONTINUING (no done, no reset:
goal/treasure respawns inside the stream), deterministic given the per-lane Philox
chain, and defined so the Metal port (Sources/Gridworld.metal) can match BITWISE.

Universal I/O contract (agent kernel is env-blind, Inv-1): every env emits
OBS_W=17 float32 observation channels and consumes the first N of POLICY_W=12
policy coordinates as categorical logits (N=4 FourRooms, N=2 DeepSea).

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
Mask words are transported as their uint32 bit pattern reinterpreted as float32.

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
        else:
            ml, mh = int(rng.integers(0, 2**32)), int(rng.integers(0, 2**32))
            r_, c_ = int(rng.integers(0, 8)), int(rng.integers(0, 8))
            la, a = int(rng.integers(0, 2)), int(rng.integers(0, 2))
            ins.append([_u32f(ml), _u32f(mh), F32(r_), F32(c_), F32(la), F32(a)])
            rw, nr, nc, nla, obs = deepsea_step(ml, mh, r_, c_, la, a)
            outs.append(np.concatenate([[rw, F32(nr), F32(nc), F32(nla)], obs]))
    return (np.array(ins, dtype=F32), np.array(outs, dtype=F32))

def ascii_map(env: str) -> str:
    if env == "FourRooms-cont-v0":
        rows = []
        for y in range(FOUR_H):
            rows.append("".join("#" if four_is_wall(x, y) else "." for x in range(FOUR_W)))
        return "\n".join(rows)
    return f"DeepSea {DEEPSEA_N}x{DEEPSEA_N}: mask-dependent columns; treasure at (7,7)"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=["FourRooms-cont-v0", "DeepSea-cont-v0"], default="FourRooms-cont-v0")
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
        for env in ("FourRooms-cont-v0", "DeepSea-cont-v0"):
            a_i, a_o = gen_cases(env, 64, 999)
            b_i, b_o = gen_cases(env, 64, 999)
            assert np.array_equal(a_o, b_o) and np.array_equal(a_i, b_i)
            print(f"{env}: determinism OK (64 cases)")
        print(ascii_map("FourRooms-cont-v0"))
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
