from itertools import product

import torch
import torch.nn.functional as F
from attacks_on_drl.attacker.critical_point_attack.rollout_helper import RolloutHelper
from attacks_on_drl.victim.common import BaseVictim

from src.prediction_model.training.common import FrameCycler


class BaseRolloutHelper(RolloutHelper):
    def __init__(
        self,
        victim: BaseVictim,
        n_actions: int,
        action_enum_len: int,
        baseline_obs_len: int,
    ) -> None:

        self.victim = victim
        self.n_actions = n_actions
        self.action_enum_len = action_enum_len
        self.baseline_obs_dist = baseline_obs_len
        
        actions = [action for action in range(n_actions)]
        self.onehot_action = F.one_hot(torch.tensor(actions), num_classes=self.n_actions)
        self.action_enumeration = list(product(actions, repeat=self.action_enum_len))
        
        self.frame_cycler = FrameCycler()