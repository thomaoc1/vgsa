from typing import Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F
from attacks_on_drl.victim.actor_critic_victim import BaseVictim

from src.attacker.rollout_helper.common.base_rollout_helper import BaseRolloutHelper
from src.prediction_model.model.policy_enc_prediction_model import PolicyEncodingPredictionModel
from src.victim.common import EncBaseVictim


class EncVictim(Protocol):
    def choose_action(self, obs, deterministic: bool) -> np.ndarray: ...
    def enc_obs(self, obs) -> torch.Tensor: ...


class EncObsRolloutHelper(BaseRolloutHelper):
    def __init__(
        self,
        enc_obs_prediction_model: PolicyEncodingPredictionModel,
        victim: BaseVictim,
        n_actions: int,
        action_enum_len: int,
        baseline_obs_len: int,
    ) -> None:
        super().__init__(
            victim=victim,
            n_actions=n_actions,
            action_enum_len=action_enum_len,
            baseline_obs_len=baseline_obs_len,
        )
        self.enc_obs_prediction_model = enc_obs_prediction_model
        self.victim = cast(EncBaseVictim, self.victim)

    def _compute_agent_trajectory(self, initial_state: torch.Tensor, steps: int, is_encoded=False):
        if not is_encoded:
            current_state = self.victim.enc_obs(initial_state)
        else:
            current_state = initial_state

        for _ in range(steps):
            agent_action = self.victim.choose_action_from_enc_obs(current_state)
            one_hot_action = F.one_hot(agent_action.long(), num_classes=self.n_actions)
            if one_hot_action.dim() < 2:
                one_hot_action = one_hot_action.unsqueeze(0)

            current_state = self.enc_obs_prediction_model(current_state, one_hot_action.float())

        return current_state

    def compute_baseline_rollout(self, initial_state: torch.Tensor):
        return self._compute_agent_trajectory(initial_state, self.baseline_obs_dist)

    def compute_full_rollout(self, initial_state: torch.Tensor):
        current_actions = self.onehot_action.float()
        current_states = self.victim.enc_obs(initial_state)

        for step in range(self.action_enum_len):
            current_states = current_states.repeat_interleave(self.n_actions, dim=0)

            if step > 0:
                current_actions = current_actions.repeat(self.n_actions, 1)

            current_states = self.enc_obs_prediction_model(current_states, current_actions)

        if self.baseline_obs_dist - self.action_enum_len > 0:
            current_states = self._compute_agent_trajectory(
                current_states, self.baseline_obs_dist - self.action_enum_len, is_encoded=True
            )

        return current_states
