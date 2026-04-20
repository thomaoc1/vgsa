from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env

from src.util.config.definitions import EnvConfig
from .scaled_atari_vec_wrapper import ScaledAtariVecWrapper
from .stacked_atari_ram_vec_wrapper import StackedAtariRamVecWrapper


def init_env(
    env_config: EnvConfig,
    seed: int | None = None,
    scale_obs: bool = True,
    atari_wrapper_args: dict | None = None,
    is_ram_env: bool = False,
) -> VecEnv:
    if not atari_wrapper_args:
        atari_wrapper_args = dict()

    env = make_atari_env(
        env_config.name,
        wrapper_kwargs=atari_wrapper_args,
        env_kwargs={"render_mode": env_config.render_mode},
        n_envs=env_config.n_envs,
    )
    if seed:
        env.seed(seed)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    if scale_obs:
        env = ScaledAtariVecWrapper(env)

    if is_ram_env:
        env = StackedAtariRamVecWrapper(env, stack_size=4)

    return env
