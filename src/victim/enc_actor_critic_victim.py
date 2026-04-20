from attacks_on_drl.victim import ActorCriticVictim
from attacks_on_drl.victim.actor_critic_victim import VecEnvObs
import torch

from src.victim.common.enc_base_victim import EncBaseVictim


class EncActorCriticVictim(ActorCriticVictim, EncBaseVictim):
    def enc_obs(self, obs: VecEnvObs | torch.Tensor):
        obs = torch.as_tensor(obs)
        obs = self._ensure_batch(obs)
        return self.model.policy.extract_features(obs)

    def eval_enc_obs(self, enc_obs: torch.Tensor):
        latent_vf = self.model.policy.mlp_extractor.forward_critic(enc_obs)
        return self.model.policy.value_net(latent_vf)

    def get_action_logits_from_encoded(self, enc_obs: torch.Tensor) -> torch.Tensor:
        latent_pi = self.model.policy.mlp_extractor.forward_actor(enc_obs)
        return self.model.policy.action_net(latent_pi)
