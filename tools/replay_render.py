#!/usr/bin/env python3
"""Replay a recorded OaK action stream through gymnasium with on-screen rendering.

Recordings are produced by setting OAK_RECORD_ACTIONS=1 during a benchmark run.
Each run-dir contains meta.json (envId, seed, actDim) and actions.bin (a flat
stream of `actDim x float32` records, each followed by one uint8 done flag).

Frames are pulled with render_mode="rgb_array" and displayed via pygame to
sidestep gymnasium's human-mode overlay (broken under newer mujoco).
"""
import argparse, json, os, sys, time
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="Directory containing meta.json and actions.bin")
    ap.add_argument("--steps", type=int, default=0, help="Max steps to replay (0 = all)")
    ap.add_argument("--fps", type=float, default=30.0, help="Target render frames/sec (0 = unlimited)")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "meta.json")) as f:
        meta = json.load(f)
    env_id, seed, act_dim = meta["envId"], int(meta["seed"]), int(meta["actDim"])

    import gymnasium as gym
    import pygame
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    frame = env.render()
    if frame is None:
        sys.exit(f"env {env_id} returned no frame; rgb_array unsupported")
    h, w = frame.shape[:2]
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"OaK replay: {env_id} seed={seed}")

    def show(frame):
        pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return False
        return True

    show(frame)
    record_size = act_dim * 4 + 1
    delay = 1.0 / args.fps if args.fps > 0 else 0.0
    step = 0
    with open(os.path.join(args.run_dir, "actions.bin"), "rb") as f:
        while not args.steps or step < args.steps:
            chunk = f.read(record_size)
            if len(chunk) < record_size:
                break
            action = np.frombuffer(chunk[:act_dim * 4], dtype=np.float32).copy()
            recorded_done = chunk[-1]
            _, _, terminated, truncated, _ = env.step(action)
            if recorded_done or terminated or truncated:
                env.reset()
            if not show(env.render()):
                break
            if delay:
                time.sleep(delay)
            step += 1
    env.close()
    pygame.quit()
    print(f"replayed {step} steps from {args.run_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
