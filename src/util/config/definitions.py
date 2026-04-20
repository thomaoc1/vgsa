from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class EnvConfig:
    name: str
    n_actions: int
    render_mode: Optional[str] = None
    n_envs: int = 1


@dataclass
class PolicyConfig:
    name: str
    agent_parameters: Dict[str, Any] = field(default_factory=dict)
    rms_prop_tf_like: bool = False
    agent_normalise_obs: bool = True
    seed: int = 0


@dataclass
class PredictionModelConfig:
    model_type: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    unique_name: str | None = None
    seed: int = 42


@dataclass
class PredictionModelTrainerConfig:
    env: EnvConfig
    next_state_pm: PredictionModelConfig
    sb3_algo: PolicyConfig
    trainer: str
    state_processor: Optional[str]
    dataset_name: str
    victim_policy: bool = True
    lr: Optional[float] = None
    curriculum: Optional[List[Tuple[int, int, float]]] = None
    k: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = None
    kl_loss: bool = True
    value_loss: bool = True
    load: bool = False
    suffix: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutHelperConfig:
    action_enum_len: int
    baseline_obs_dist: int


@dataclass
class AttackerConfig:
    name: str
    attack_threshold: float | int
    n_actions: int
    rollout_helper: RolloutHelperConfig
    is_encoded: bool = False
    uses_ram: bool = False
    perturbation_type: str = "CW"
