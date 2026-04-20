import logging
from collections import deque

import torch
import torch.nn.functional as F
from attacks_on_drl.victim.common.base_victim import BaseVictim

from src.attacker.rollout_helper.common.base_rollout_helper import BaseRolloutHelper
from src.prediction_model.model.obs_prediction_model import ObsPredictionModel
from src.prediction_model.model.ram_prediction_model import RamPredictionModel
from src.util.sb3_env import StackedAtariRamVecWrapper


class RamRolloutHelper(BaseRolloutHelper):
    def __init__(
        self,
        env: StackedAtariRamVecWrapper,
        obs_prediction_model: ObsPredictionModel,
        ram_prediction_model: RamPredictionModel,
        victim: BaseVictim,
        n_actions: int,
        action_enum_len: int,
        baseline_obs_len: int,
    ):
        super().__init__(
            victim=victim,
            n_actions=n_actions,
            action_enum_len=action_enum_len,
            baseline_obs_len=baseline_obs_len,
        )
        assert baseline_obs_len == action_enum_len, "Baseline len greater than action enum len is not implemented."  
        self.env = env
        self.obs_prediction_model = obs_prediction_model
        self.ram_prediction_model = ram_prediction_model

    def _compute_agent_trajectory(self, ram_initial_state: torch.Tensor, agent_initial_state: torch.Tensor, steps: int):
        current_ram_state = ram_initial_state.float()
        current_agent_state = agent_initial_state
        predicted_ram_state_queue = deque([current_ram_state[:, i] for i in range(4)], maxlen=4)

        if current_agent_state.max() > 1:
            logging.warning("Non normalised state passed to ram rollout helper!")

        hidden = None
        for step in range(steps):
            agent_action = self.victim.choose_action(current_agent_state.numpy(), deterministic=True)
            agent_action = torch.from_numpy(agent_action)
            one_hot_action = F.one_hot(agent_action.long(), num_classes=self.n_actions).float()

            predicted_next_ram_state, hidden = self.ram_prediction_model(
                current_ram_state / 255.0, one_hot_action.unsqueeze(0), hidden
            )
            current_ram_state = predicted_next_ram_state.argmax(dim=-1).float().unsqueeze(1)
            predicted_ram_state_queue.append(current_ram_state.squeeze(0))

            if step < steps - 1:
                self.frame_cycler.save_current_state(current_agent_state)
                predicted_next_state = self.obs_prediction_model(current_agent_state, one_hot_action).unsqueeze(1)
                current_agent_state = self.frame_cycler.cycle_frames(predicted_next_state)

        return torch.stack(list(predicted_ram_state_queue), dim=1)
    
    def get_action_sequence(self, idx: int) -> tuple[int, ...]:
        return self.action_enumeration[idx]

    def collect_baseline_observation(self, obs: torch.Tensor):
        ram_state = torch.from_numpy(self.env.get_stacked_ram_obs())
        return self._compute_agent_trajectory(ram_state, obs, self.baseline_obs_dist)

    def collect_all_rollout_observations(self, obs: torch.Tensor):
        current_ram_state = torch.from_numpy(self.env.get_stacked_ram_obs())
        current_actions = self.onehot_action.float()

        hidden = None
        for step in range(self.action_enum_len):
            if hidden is not None:
                h, c = hidden
                h = h.repeat_interleave(self.n_actions, dim=1)
                c = c.repeat_interleave(self.n_actions, dim=1)
                hidden = (h, c)

            current_ram_state = current_ram_state.repeat_interleave(self.n_actions, dim=0) / 255.0

            if step > 0:
                current_actions = current_actions.repeat(self.n_actions, 1)

            predicted_next_ram_state, hidden = self.ram_prediction_model(
                current_ram_state, current_actions, hidden
            )
            current_ram_state = predicted_next_ram_state.unsqueeze(dim=1).argmax(dim=-1)

        return current_ram_state
