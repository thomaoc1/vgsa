import torch
import torchattacks
from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.attacker.critical_point_attack.critical_point_attack import CriticalPointAttack
from attacks_on_drl.attacker.critical_point_attack.rollout_helper import RolloutHelper
from attacks_on_drl.victim.common import BaseVictim
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


class VGSAAttacker(CriticalPointAttack):
    def __init__(
        self,
        victim: BaseVictim,
        rollout_helper: RolloutHelper,
        attack_threshold: float,
        cw_kwargs: dict | None = None,
    ) -> None:
        self.rollout_helper = rollout_helper
        self.attack_threshold = attack_threshold
        wrapped_victim = VictimModuleWrapper(self.victim)

        if not cw_kwargs:
            cw_kwargs = dict()

        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label(quiet=True)

        self.current_attack_action_seq = None
        self.current_attack_action_seq_idx = 0

    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        if self.current_attack_action_seq is not None:
            return self._attack(observation), True

        baseline_value = self.victim.eval_state(self.rollout_helper.collect_baseline_observation(observation))

        all_final_observations = self.rollout_helper.collect_all_rollout_observations(observation)
        best_attack_value, best_attack_value_idx = torch.max(
            self.victim.eval_state(all_final_observations),
            dim=0,
        )

        if abs(best_attack_value.item() - baseline_value.item()) > self.attack_threshold:
            self.current_attack_action_seq = self.rollout_helper.get_action_sequence(int(best_attack_value_idx.item()))
            return self._attack(observation), True

        return observation, False
