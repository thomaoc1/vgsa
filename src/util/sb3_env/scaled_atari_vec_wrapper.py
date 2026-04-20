import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class ScaledAtariVecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        obs_space = venv.observation_space
        self.observation_space = type(obs_space)(low=0.0, high=1.0, shape=obs_space.shape, dtype=np.float32)

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32) / 255.0
        raise ValueError("Only np.ndarray observation type supported.")

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32) / 255.0, rewards, dones, infos
        raise ValueError("Only np.ndarray observation type supported.")
