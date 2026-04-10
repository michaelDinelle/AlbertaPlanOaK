import os
import sys
import numpy as np
import multiprocessing as mp
import warnings

# Suppress ALL Python warnings globally
warnings.filterwarnings("ignore")

import contextlib

@contextlib.contextmanager
def SuppressOutput():
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()
    
    def _redirect_stdout(to_fd):
        sys.stdout.flush()
        os.dup2(to_fd, fd_out)
        
    def _redirect_stderr(to_fd):
        sys.stderr.flush()
        os.dup2(to_fd, fd_err)
        
    saved_out = os.dup(fd_out)
    saved_err = os.dup(fd_err)
    
    try:
        with open(os.devnull, 'w') as devnull:
            _redirect_stdout(devnull.fileno())
            _redirect_stderr(devnull.fileno())
            yield
    finally:
        _redirect_stdout(saved_out)
        _redirect_stderr(saved_err)
        os.close(saved_out)
        os.close(saved_err)

class DMControlGymWrapper:
    def __init__(self, domain_name, task_name, seed=None):
        from dm_control import suite
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed} if seed is not None else {})
        self.action_spec = self.env.action_spec()
        self.act_dim = int(self.action_spec.shape[0])
        self.action_low = np.array(self.action_spec.minimum, dtype=np.float32)
        self.action_high = np.array(self.action_spec.maximum, dtype=np.float32)
        
    def reset(self):
        time_step = self.env.reset()
        return self._flatten_obs(time_step.observation)
        
    def step(self, action):
        safe_action = np.clip(action, self.action_low, self.action_high)
        time_step = self.env.step(safe_action)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        done = time_step.last()
        return self._flatten_obs(time_step.observation), reward, done
        
    def _flatten_obs(self, observation):
        result = []
        for key in sorted(observation.keys()):
            val = observation[key]
            if np.isscalar(val):
                result.append([val])
            else:
                result.append(np.asarray(val).flatten())
        return np.concatenate(result).astype(np.float32)
        
    def close(self):
        self.env.close()

def _worker_chunk(env_type, env_args, chunk_size, worker_idx, cmd_pipe, result_pipe):
    import os, sys
    # Suppress verbose C-level stdout/stderr spam from physics engines like PyBullet
    with SuppressOutput():
        envs = []
        try:
            for i in range(chunk_size):
                # Seed offset logic
                worker_args = list(env_args)
                if worker_args[-1] is not None:
                    worker_args[-1] = worker_args[-1] + (worker_idx * 1000) + i * 100
                    
                if env_type == "dm_control":
                    domain, task, seed = worker_args
                    env = DMControlGymWrapper(domain, task, seed)
                elif env_type == "minigrid":
                    import gymnasium as gym
                    import minigrid
                    env_id, seed = worker_args
                    env = gym.make(env_id)
                    if seed is not None:
                        env.reset(seed=seed)
                elif env_type == "pyflyt":
                    import gymnasium as gym
                    import PyFlyt.gym_envs
                    env_id, seed = worker_args
                    env = gym.make(env_id)
                    if seed is not None:
                        env.reset(seed=seed)
                else:
                    raise ValueError(f"Unknown env_type {env_type}")
                envs.append(env)
                
            def _reset_single(e):
                if env_type == "dm_control":
                    return e.reset()
                else:
                    obs, _ = e.reset()
                    if env_type == "minigrid":
                        return obs["image"].flatten() / 10.0
                    return obs

            # Get initial observations
            initial_obs = [_reset_single(e) for e in envs]
            # Send back observation shape/info initially
            result_pipe.send(np.concatenate(initial_obs).flatten().astype(np.float32))
            
            while True:
                cmd, payload = cmd_pipe.recv()
                
                if cmd == "step":
                    # payload is a list/array of actions for this chunk
                    next_obs_list = []
                    rewards_list = []
                    dones_list = []
                    
                    for i, action in enumerate(payload):
                        e = envs[i]
                        if env_type == "dm_control":
                            next_obs, reward, done = e.step(action)
                        elif env_type == "minigrid":
                            # Map continuous action [-1, 1] to discrete [0, 6]
                            act_idx = int(np.floor((action[0] + 1.0) / 2.0 * 7.0))
                            act_idx = max(0, min(act_idx, 6))
                            res = e.step(act_idx)
                            next_obs, reward, done = res[0], float(res[1]), (res[2] or res[3])
                            next_obs = next_obs["image"].flatten() / 10.0
                        elif env_type == "pyflyt":
                            res = e.step(action)
                            next_obs, reward, done = res[0], float(res[1]), (res[2] or res[3])
                            
                        if done:
                            next_obs = _reset_single(e)
                            reward = 0.0 # continual abstraction
                            
                        next_obs_list.append(next_obs)
                        rewards_list.append(reward)
                        dones_list.append(done)
                        
                    next_obs_arr = np.concatenate(next_obs_list).flatten().astype(np.float32)
                    rewards_arr = np.array(rewards_list, dtype=np.float32)
                    dones_arr = np.array(dones_list, dtype=bool)
                    result_pipe.send((next_obs_arr, rewards_arr, dones_arr))
                    
                elif cmd == "close":
                    for e in envs:
                        e.close()
                    break
        except EOFError:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_pipe.send(e)

class BatchedEnvBridge:
    def __init__(self, env_type, env_args, num_envs):
        self.num_envs = num_envs
        self.env_type = env_type
        
        # Cap workers to cpu count to avoid excessive context switching
        # 4 environments per process chunk is extremely efficient
        chunk_target = 4 
        self.num_workers = max(1, min(num_envs // chunk_target, os.cpu_count() or 16))
        if num_envs < chunk_target:
            self.num_workers = num_envs
            
        self.cmd_pipes = []
        self.result_pipes = []
        self.processes = []
        self.chunks = []
        
        # Use spawn to ensure AppKit/OpenGL initializes safely on main threads in workers
        try:
            ctx = mp.get_context('spawn')
            ctx.set_executable(sys.executable if 'python' in sys.executable.lower() else os.environ.get("PYTHON_BIN", sys.executable))
        except Exception:
            ctx = mp.get_context('spawn')
        
        assigned = 0
        for i in range(self.num_workers):
            chunk_size = (num_envs - assigned) // (self.num_workers - i)
            self.chunks.append((assigned, assigned + chunk_size))
            assigned += chunk_size
            
            parent_cmd, child_cmd = ctx.Pipe()
            parent_res, child_res = ctx.Pipe()
            
            p = ctx.Process(target=_worker_chunk, args=(env_type, env_args, chunk_size, i, child_cmd, child_res))
            p.daemon = True
            p.start()
            
            self.cmd_pipes.append(parent_cmd)
            self.result_pipes.append(parent_res)
            self.processes.append(p)
            
        # Collect initial observations to get dimensions
        initial_obs_flat = []
        for pipe in self.result_pipes:
            res = pipe.recv()
            if isinstance(res, Exception):
                raise res
            initial_obs_flat.append(res)
            
        self.initial_obs_array = np.concatenate(initial_obs_flat).astype(np.float32)
        
        # Determine act_dim
        if env_type == "dm_control":
            dummy = DMControlGymWrapper(*env_args)
            self.act_dim = dummy.act_dim
            dummy.close()
        elif env_type == "minigrid":
            self.act_dim = 1
        elif env_type == "pyflyt":
            import gymnasium as gym
            import PyFlyt.gym_envs
            dummy = gym.make(env_args[0])
            self.act_dim = dummy.action_space.shape[0]
            dummy.close()
            
        # obs_dim = total elements / num_envs
        self.obs_dim = len(self.initial_obs_array) // num_envs
        
    def get_obs_dim(self):
        return self.obs_dim
        
    def get_act_dim(self):
        return self.act_dim
        
    def initial_reset(self):
        return self.initial_obs_array
        
    def step(self, batched_actions):
        actions = batched_actions.reshape(self.num_envs, self.act_dim)
        
        # Dispatch
        for i, (start, end) in enumerate(self.chunks):
            self.cmd_pipes[i].send(("step", actions[start:end]))
            
        next_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        
        # Collect with a larger timeout to avoid hangs on slow environments like PyFlyt
        for i, (start, end) in enumerate(self.chunks):
            if not self.result_pipes[i].poll(120.0):
                print(f"Worker {i} timed out during step!", file=sys.stderr)
                raise RuntimeError(f"Worker {i} timed out")
            res = self.result_pipes[i].recv()
            if isinstance(res, Exception):
                raise res
            obs_arr, r_arr, d_arr = res
            
            chunk_size = end - start
            next_obs[start:end] = obs_arr.reshape(chunk_size, self.obs_dim)
            rewards[start:end] = r_arr
            dones[start:end] = d_arr
            
        return next_obs.flatten(), rewards.flatten(), dones.flatten()
        
    def close(self):
        for pipe in self.cmd_pipes:
            pipe.send(("close", None))
        for p in self.processes:
            p.join()
