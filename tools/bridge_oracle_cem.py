#!/usr/bin/env python3
import argparse, ctypes, json, math, os, statistics, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_ID = b"HalfCheetah-v4"
INFO_KEYS = ("info:reward_run", "info:reward_ctrl", "info:x_position", "info:x_velocity")
SEED_THETA = np.array([
    0.7946, 0.4094, 0.2879, -0.6115, 0.0105, 0.1478,
    2.8648, 2.9840, 1.0458, 3.5620, 4.6253, 5.0093,
    -0.0577, -0.4530, -0.2659, 0.0431, 0.0491, 0.0872, -0.7985,
], dtype=np.float64)


class View(ctypes.Structure):
    _fields_ = [
        ("obs_dim", ctypes.c_int32), ("act_dim", ctypes.c_int32),
        ("d_policy", ctypes.c_int32), ("done", ctypes.c_int32),
        ("reward", ctypes.c_float), ("action_delta_mean", ctypes.c_double),
        ("info_count", ctypes.c_int32), ("info_names", ctypes.POINTER(ctypes.c_char_p)),
        ("info_values", ctypes.POINTER(ctypes.c_double)),
        ("obs", ctypes.POINTER(ctypes.c_float)), ("action", ctypes.POINTER(ctypes.c_float)),
        ("credit", ctypes.POINTER(ctypes.c_float)), ("imp", ctypes.POINTER(ctypes.c_float)),
    ]


def load_lib():
    lib = ctypes.CDLL(os.path.join(ROOT, ".build/oak-native/liboakenv.dylib"))
    lib.oak_env_open.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p]
    lib.oak_env_open.restype = ctypes.c_void_p
    lib.oak_env_close.argtypes = [ctypes.c_void_p]
    lib.oak_env_error.argtypes = [ctypes.c_void_p]
    lib.oak_env_error.restype = ctypes.c_char_p
    lib.oak_env_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.oak_env_prepare.restype = ctypes.c_int32
    lib.oak_env_reset.argtypes = [ctypes.c_void_p]
    lib.oak_env_reset.restype = ctypes.c_int32
    lib.oak_env_step_oak.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.oak_env_step_oak.restype = ctypes.c_int32
    lib.oak_env_view.argtypes = [ctypes.c_void_p, ctypes.POINTER(View)]
    lib.oak_env_view.restype = ctypes.c_int32
    return lib


LIB = load_lib()


def check(ok, h):
    if ok != 1:
        msg = LIB.oak_env_error(h)
        raise RuntimeError(msg.decode() if msg else "native bridge error")


def run(theta, horizon, seed):
    base = os.path.join(ROOT, ".build/oak-native/env-assets").encode()
    h = LIB.oak_env_open(seed, ENV_ID, base)
    if not h:
        raise RuntimeError((LIB.oak_env_error(None) or b"open failed").decode())
    try:
        v = View()
        check(LIB.oak_env_view(h, ctypes.byref(v)), h)
        check(LIB.oak_env_prepare(h, 0), h)
        check(LIB.oak_env_reset(h), h)
        check(LIB.oak_env_view(h, ctypes.byref(v)), h)
        p = (ctypes.c_float * v.d_policy)()
        total, info = 0.0, {k: {"sum": 0.0, "last": None} for k in INFO_KEYS}
        amps, phases, biases, freq = theta[:v.act_dim], theta[v.act_dim:2*v.act_dim], theta[2*v.act_dim:3*v.act_dim], math.exp(theta[-1])
        for t in range(horizon):
            for i in range(v.d_policy):
                p[i] = 0.0
            for i in range(v.act_dim):
                p[i] = float(np.clip(biases[i] + amps[i] * math.sin(freq * t + phases[i]), -1.0, 1.0))
            check(LIB.oak_env_step_oak(h, p, None, 0), h)
            check(LIB.oak_env_view(h, ctypes.byref(v)), h)
            total += float(v.reward)
            for i in range(max(v.info_count, 0)):
                name = v.info_names[i].decode()
                if name in info:
                    x = float(v.info_values[i])
                    info[name]["sum"] += x
                    info[name]["last"] = x
        for x in info.values():
            x["mean"] = x["sum"] / horizon
        return {"rewardSum": total, "envInfo": info}
    finally:
        LIB.oak_env_close(h)


def clipped(theta):
    out = theta.copy()
    out[:6] = np.clip(out[:6], -1.0, 1.0)
    out[12:18] = np.clip(out[12:18], -0.7, 0.7)
    out[18] = np.clip(out[18], math.log(0.02), math.log(0.45))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cem-iters", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--eval-horizon", type=int, default=5000)
    ap.add_argument("--seed-count", type=int, default=5)
    args = ap.parse_args()
    rng, mean = np.random.default_rng(11), SEED_THETA.copy()
    std = np.r_[np.full(6, 0.18), np.full(6, 0.35), np.full(6, 0.12), 0.12]
    best = (-float("inf"), mean.copy())
    t0 = time.time()
    for it in range(args.cem_iters):
        pop = []
        for j in range(32):
            th = clipped(mean + std * rng.standard_normal(mean.size))
            scores = [run(th, args.horizon, it * 10000 + j * 10 + s)["rewardSum"] for s in range(3)]
            pop.append((statistics.mean(scores), th, scores))
        pop.sort(key=lambda x: x[0], reverse=True)
        elite = np.stack([x[1] for x in pop[:6]])
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.025)
        best = max(best, (pop[0][0], pop[0][1]), key=lambda x: x[0])
        print(json.dumps({"iter": it, "bestMean": pop[0][0], "bestScores": pop[0][2]}, sort_keys=True), flush=True)
    theta = best[1] if args.cem_iters else mean
    evals = [run(theta, args.eval_horizon, 9200 + s) for s in range(args.seed_count)]
    print(json.dumps({
        "envId": ENV_ID.decode(), "controller": "open_loop_sinusoid_cem",
        "cemIters": args.cem_iters, "evalHorizon": args.eval_horizon,
        "rewardSums": [x["rewardSum"] for x in evals],
        "rewardMean": statistics.mean(x["rewardSum"] for x in evals),
        "envInfo": {k: {m: statistics.mean(x["envInfo"][k][m] for x in evals) for m in ("sum", "mean", "last")} for k in INFO_KEYS},
        "theta": [float(x) for x in theta], "elapsedSeconds": time.time() - t0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
