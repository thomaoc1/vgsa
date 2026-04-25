from typing import Any, Mapping


def run_attack_summary(cfg: Mapping[str, Any]) -> dict:
    """The fields that actually matter for comparing attack runs.

    Drops training-time hyperparameters from the policy (irrelevant at inference),
    redundant n_actions (duplicated across gym_env and attacker), and env fields
    that are fixed by the attack workflow (n_envs=1, render_mode).
    """
    return {
        "env": cfg["gym_env"]["name"],
        "policy": cfg["policy"]["name"],
        "policy_seed": cfg["policy"]["seed"],
        "attacker": cfg["attacker"]["name"],
        "attack_threshold": cfg["attacker"]["attack_threshold"],
        "action_enum_len": cfg["attacker"]["rollout_helper"]["action_enum_len"],
        "baseline_obs_dist": cfg["attacker"]["rollout_helper"]["baseline_obs_dist"],
        "is_encoded": cfg["attacker"]["is_encoded"],
        "uses_ram": cfg["attacker"]["uses_ram"],
        "perturbation": cfg["attacker"]["perturbation_type"],
        "n_episodes": cfg["n_episodes"],
    }
