from attacks_on_drl.victim.dqn_victim import DQNVictim
import torch

from src.victim.common.enc_base_victim import EncBaseVictim


class EncDQNVictim(DQNVictim, EncBaseVictim):
    def enc_obs(self, obs: torch.Tensor):
        obs = torch.as_tensor(obs)
        obs = self._ensure_batch(obs)

        features = self.model.policy.q_net.features_extractor(obs)
        return features

    def eval_enc_obs(self, enc_obs: torch.Tensor):
        q_values = self.model.q_net.q_net(enc_obs)
        return q_values.max(dim=-1).values.unsqueeze(1)

    def get_action_logits_from_encoded(self, enc_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.model.q_net.q_net(enc_obs)
        return q_values
