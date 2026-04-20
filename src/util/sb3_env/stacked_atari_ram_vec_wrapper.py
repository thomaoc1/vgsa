from collections import deque
from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper


class StackedAtariRamVecWrapper(VecEnvWrapper):
    """
    A VecEnv wrapper that stacks Atari RAM observations per environment.
    It preserves a deque of last N RAM states for each parallel env.
    """

    def __init__(self, venv, ram_annotations: Optional[dict] = None, stack_size: int = 4):
        super().__init__(venv)
        self.num_stack = stack_size

        self.indices = list(ram_annotations.values()) if ram_annotations else None

        example_ram = self._get_ram_obs_from_venv()
        ram_dim = example_ram.shape[1] if example_ram.ndim == 2 else example_ram.shape[0]
        if self.indices:
            ram_dim = len(self.indices)

        self.ram_buffers = [deque(maxlen=self.num_stack) for _ in range(self.num_envs)]

        low = np.zeros((self.num_stack, ram_dim), dtype=np.uint8)
        high = np.full((self.num_stack, ram_dim), 255, dtype=np.uint8)
        self.ram_space = spaces.Box(low=low, high=high, dtype=np.uint8)

    def _get_ram_obs_from_venv(self):
        ram_list = []
        ale_envs = self.venv.get_attr("ale")
        for ale_env in ale_envs:
            ram = ale_env.getRAM()
            if self.indices:
                ram = ram[self.indices]
            ram_list.append(ram)
        return np.stack(ram_list, axis=0)

    def reset(self):
        obs = self.venv.reset()
        ram_obs = self._get_ram_obs_from_venv()

        for i in range(self.num_envs):
            self.ram_buffers[i].clear()
            for _ in range(self.num_stack):
                self.ram_buffers[i].append(ram_obs[i])

        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        ram_obs = self._get_ram_obs_from_venv()

        for i in range(self.num_envs):
            if dones[i]:
                # Clear RAM stack on reset
                self.ram_buffers[i].clear()
                for _ in range(self.num_stack):
                    self.ram_buffers[i].append(ram_obs[i])
            else:
                self.ram_buffers[i].append(ram_obs[i])

        return obs, rewards, dones, infos

    def get_stacked_ram_obs(self) -> np.ndarray:
        """Return the current stacked RAM observations for all envs."""
        stacked_rams = np.array([np.stack(buf, axis=0) for buf in self.ram_buffers])
        return stacked_rams
