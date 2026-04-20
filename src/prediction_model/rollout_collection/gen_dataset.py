import random
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import ale_py
import gymnasium as gym
import hydra
import numpy as np
import torch
import tqdm
from attacks_on_drl.victim import ActorCriticVictim, DQNVictim
from omegaconf import DictConfig
from stable_baselines3.common.vec_env import VecEnv

from src.util.agent import init_agent
from src.util.config.definitions import EnvConfig, PolicyConfig
from src.util.config.paths import CONFIG_PATH
from src.util.mem_usage import get_memory_usage
from src.util.path_builder import DatasetPaths, PolicyPaths
from src.util.sb3_env import StackedAtariRamVecWrapper, init_env
from src.util.set_global_seed import set_global_seed

from .transition_dataset import TransitionDataset

gym.register_envs(ale_py)

SEED = 101


@dataclass
class GenDataConfig:
    gym_env: EnvConfig
    policy: PolicyConfig
    n_frames: int
    dataset_save_name: str
    eps: float = 0.0
    encode: bool = False
    is_ram_env: bool = False
    max_episode_steps: int | float = float("+inf")


class GenDataVictimAgent(Protocol):
    def choose_action(self, obs, deterministic: bool) -> np.ndarray: ...
    def enc_obs(self, obs) -> torch.Tensor: ...


def log_progress(episode_rewards: deque, frames_collected: int, pbar):
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    cpu_mem, _, _ = get_memory_usage()
    pbar.set_postfix({"Ep Reward": avg_reward, "Frames Collected": frames_collected, "CPU": f"{cpu_mem:.1f}GB"})


def check_obs(observation):
    if observation.dtype in [np.float32, np.float64] and observation.max() <= 1.0:
        observation = (observation * 255).round().astype(np.uint8)
    return observation


@torch.no_grad()
def run_episode(
    gen_data_cfg: GenDataConfig,
    env: VecEnv | StackedAtariRamVecWrapper,
    agent: GenDataVictimAgent,
    pbar,
    encode_obs: bool = False,
):
    observation = env.reset()

    is_done = False
    ep_reward = 0
    frames_this_episode = 0

    def is_max_frames_exceeded():
        return frames_this_episode > gen_data_cfg.max_episode_steps

    states = []
    actions = []

    while not is_done and not is_max_frames_exceeded():
        if isinstance(observation, np.ndarray):
            scaled_observation = observation / 255.0
        else:
            raise ValueError("Only np.ndarray observations are supported.")

        if random.random() < gen_data_cfg.eps:
            action = env.action_space.sample()
        else:
            action = agent.choose_action(scaled_observation, deterministic=True)

        if isinstance(env, StackedAtariRamVecWrapper):
            desired_state = env.get_stacked_ram_obs().squeeze(0)
        elif encode_obs:
            desired_state = agent.enc_obs(scaled_observation).squeeze(0).numpy()
        else:
            desired_state = check_obs(observation.squeeze(0))

        states.append(desired_state)
        actions.append(action.item())

        frames_this_episode += 1
        pbar.update(1)

        observation, reward, is_done, _ = env.step(action)
        ep_reward += reward

    return states, actions, ep_reward, frames_this_episode


def generate_dataset(
    env: VecEnv,
    agent,
    gen_data_cfg: GenDataConfig,
):
    dataset = TransitionDataset(env.action_space.n)  # pyright: ignore[reportAttributeAccessIssue]

    frames_collected = 0
    episode_rewards = deque(maxlen=100)

    with tqdm.tqdm(total=gen_data_cfg.n_frames, leave=False) as pbar:
        while frames_collected < gen_data_cfg.n_frames:
            episode = run_episode(gen_data_cfg, env, agent, pbar)
            states, actions, ep_reward, frames_during_episode = episode

            dataset.add_episode(states, actions)

            frames_collected += frames_during_episode
            episode_rewards.append(ep_reward)

            log_progress(episode_rewards, frames_collected, pbar)

            if frames_collected >= gen_data_cfg.n_frames:
                break

    pbar.n = gen_data_cfg.n_frames
    pbar.refresh()

    return dataset


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="generate_data")
def main(cfg: DictConfig):
    gen_data_cfg = GenDataConfig(**cfg)  # pyright: ignore[reportCallIssue]
    gen_data_cfg.gym_env = EnvConfig(**cfg.gym_env)
    gen_data_cfg.policy = PolicyConfig(**cfg.policy)

    policy_path_builder = PolicyPaths(
        gen_data_cfg.policy.name,
        gen_data_cfg.gym_env.name,
        seed=gen_data_cfg.policy.seed,
    )
    dataset_path_builder = DatasetPaths(
        gen_data_cfg.policy.name,
        gen_data_cfg.gym_env.name,
        encoded=gen_data_cfg.encode,
        agent_seed=gen_data_cfg.policy.seed,
    )

    assert 1.0 >= gen_data_cfg.eps >= 0

    set_global_seed(SEED)

    agent = init_agent(gen_data_cfg.policy, policy_path_builder)
    if gen_data_cfg.gym_env.name.lower() == "dqn":
        agent = DQNVictim(agent)
    else:
        agent = ActorCriticVictim(agent)

    atari_wrapper_args = {"terminal_on_life_loss": False, "clip_reward": False}
    env = init_env(
        env_config=gen_data_cfg.gym_env,
        scale_obs=False,
        atari_wrapper_args=atari_wrapper_args,
        is_ram_env=gen_data_cfg.is_ram_env,
        seed=SEED,
    )

    dataset = generate_dataset(env, agent, gen_data_cfg)
    print(f"Saving to {dataset_path_builder.train_file}")

    if gen_data_cfg.is_ram_env:
        dataset.save(dataset_path_builder.ram_train_file)
    else:
        dataset.save(dataset_path_builder.train_file)


if __name__ == "__main__":
    main()
