import torch
import numpy as np
import torch.nn.functional as F
from attacks_on_drl.victim.common import BaseVictim
import matplotlib.pyplot as plt

from src.attacker.rollout_helper.common.base_rollout_helper import BaseRolloutHelper
from src.prediction_model.model.obs_prediction_model import ObsPredictionModel


class ObsRolloutHelper(BaseRolloutHelper):
    def __init__(
        self,
        obs_prediction_model: ObsPredictionModel,
        victim: BaseVictim,
        n_actions: int,
        action_enum_len: int,
        baseline_obs_dist: int,
    ) -> None:
        super().__init__(
            victim=victim,
            n_actions=n_actions,
            action_enum_len=action_enum_len,
            baseline_obs_len=baseline_obs_dist,
        )
        self.obs_prediction_model = obs_prediction_model

    @torch.no_grad()
    def _compute_agent_trajectory(self, initial_states: torch.Tensor, steps: int):
        current_state = initial_states.float()

        for _ in range(steps):
            agent_action = torch.from_numpy(self.victim.choose_action(current_state, deterministic=True))
            one_hot_action = F.one_hot(agent_action.long(), num_classes=self.n_actions)
            if one_hot_action.dim() < 2:
                one_hot_action = one_hot_action.unsqueeze(0)

            self.frame_cycler.save_current_state(current_state)
            predicted_next_pm_states = self.obs_prediction_model(current_state, one_hot_action.float())
            current_state = self.frame_cycler.cycle_frames(predicted_next_pm_states)

        return current_state

    def get_action_sequence(self, idx: int) -> tuple[int, ...]:
        return self.action_enumeration[idx]

    def collect_baseline_observation(self, obs: torch.Tensor | np.ndarray):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        return self._compute_agent_trajectory(obs, self.baseline_obs_dist)

    @torch.no_grad()
    def collect_all_rollout_observations(self, obs: torch.Tensor | np.ndarray):
        
        current_actions = self.onehot_action.float()
        
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
            
        current_states = obs

        for step in range(self.action_enum_len):
            current_states = current_states.repeat_interleave(self.n_actions, dim=0)
            self.frame_cycler.save_current_state(current_states)

            if step > 0:
                current_actions = current_actions.repeat(self.n_actions, 1)
                
            predicted_next_states = self.obs_prediction_model(current_states, current_actions)
            current_states = self.frame_cycler.cycle_frames(predicted_next_states)

        if self.baseline_obs_dist - self.action_enum_len > 0:
            current_states = self._compute_agent_trajectory(
                current_states, self.baseline_obs_dist - self.action_enum_len
            )

        return current_states
