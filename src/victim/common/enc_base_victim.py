from abc import abstractmethod, ABC
from stable_baselines3.common.vec_env import VecEnv
import torch


class EncBaseVictim(ABC):
    @abstractmethod
    def enc_obs(self, obs: VecEnv | torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_enc_obs(self, obs: VecEnv | torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_action_logits_from_encoded(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    def choose_action_from_enc_obs(self, obs: VecEnv | torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)
        action_logits = self.get_action_logits_from_encoded(obs)
        return action_logits.argmax(dim=-1)
