from attacks_on_drl.runner.attack_runner import BaseAttacker
import torch
import torchattacks
from attacks_on_drl.attacker.common import VictimModuleWrapper
from attacks_on_drl.attacker.critical_point_attack.rollout_helper import RolloutHelper
from attacks_on_drl.victim.common import BaseVictim
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

from src.victim.common.enc_base_victim import EncBaseVictim


class VGSAAttacker(BaseAttacker):
    def __init__(
        self,
        victim: BaseVictim,
        rollout_helper: RolloutHelper,
        attack_threshold: float,
        cw_kwargs: dict | None = None,
        is_encoded: bool = False,
    ) -> None:
        super().__init__(victim=victim)

        self.rollout_helper = rollout_helper
        self.attack_threshold = attack_threshold
        self.is_encoded = is_encoded
        wrapped_victim = VictimModuleWrapper(self.victim)

        if not cw_kwargs:
            cw_kwargs = dict()

        self._perturbation_method = torchattacks.CW(wrapped_victim, **cw_kwargs)
        self._perturbation_method.set_mode_targeted_by_label(quiet=True)

        self.current_attack_action_seq = None
        self.current_attack_action_seq_idx = 0

    def _attack(self, obs: VecEnvObs) -> VecEnvObs:
        assert self.current_attack_action_seq is not None, "Current action sequence is None."

        target_action = torch.tensor(self.current_attack_action_seq[self.current_attack_action_seq_idx]).unsqueeze(0)
        tens_obs = torch.from_numpy(obs)
        adversarial_obs = self._perturbation_method(tens_obs, target_action).numpy()

        self.current_attack_action_seq_idx += 1
        if self.current_attack_action_seq_idx == len(self.current_attack_action_seq):
            self.current_attack_action_seq = None
            self.current_attack_action_seq_idx = 0

        return adversarial_obs

    def step(self, observation: VecEnvObs) -> tuple[VecEnvObs, bool]:
        if self.current_attack_action_seq is not None:
            return self._attack(observation), True

        with torch.no_grad():
            baseline_obs = self.rollout_helper.collect_baseline_obs(observation)
            all_final_obs = self.rollout_helper.collect_all_rollout_obs(observation)

            if not self.is_encoded:
                baseline_value = self.victim.eval_state(baseline_obs)
                all_final_obs_values = self.victim.eval_state(all_final_obs)
            elif self.is_encoded and isinstance(self.victim, EncBaseVictim):
                baseline_value = self.victim.eval_enc_obs(baseline_obs)
                all_final_obs_values = self.victim.eval_enc_obs(all_final_obs)

            best_attack_value, best_attack_value_idx = torch.min(
                all_final_obs_values,
                dim=0,
            )

        if abs(best_attack_value.item() - baseline_value.item()) > self.attack_threshold:
            self.current_attack_action_seq = self.rollout_helper.get_action_sequence(int(best_attack_value_idx.item()))
            return self._attack(observation), True

        return observation, False
