from dataclasses import dataclass

import hydra
import torch
from attacks_on_drl.victim import ActorCriticVictim
from attacks_on_drl.victim.dqn_victim import DQNVictim
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from src.attacker.global_perturbation.obs_dataset import ObservationDataset
from src.prediction_model.rollout_collection.transition_dataset import TransitionDataset
from src.util.agent import init_agent
from src.util.config.definitions import EnvConfig, PolicyConfig
from src.util.config.paths import CONFIG_PATH
from src.util.path_builder import DatasetPaths, PolicyPaths

from .global_perturbation_generator import GlobalPerturbationGenerator


@dataclass
class GlobalPerturbationGeneratorConfig:
    env: EnvConfig
    policy_cfg: PolicyConfig
    dataset_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="gen_perturbation")
def main(cfg: DictConfig):
    gpg_config = GlobalPerturbationGeneratorConfig(**cfg)  # pyright: ignore[reportCallIssue]
    gpg_config.env = EnvConfig(**cfg.env)
    gpg_config.policy_cfg = PolicyConfig(**cfg.policy_cfg)

    policy_paths = PolicyPaths(
        algo=gpg_config.policy_cfg.name,
        env=gpg_config.env.name,
        seed=gpg_config.policy_cfg.seed,
    )

    dataset_paths = DatasetPaths(
        algo=gpg_config.policy_cfg.name,
        env=gpg_config.env.name,
        agent_seed=gpg_config.policy_cfg.seed,
        encoded=False,
    )

    policy = init_agent(gpg_config.policy_cfg, device=gpg_config.device, path_builder=policy_paths)

    if gpg_config.policy_cfg.name.upper() == "DQN":
        victim_agent = DQNVictim(policy)
    else:
        victim_agent = ActorCriticVictim(policy)

    gpg = GlobalPerturbationGenerator(policy=victim_agent, n_actions=gpg_config.env.n_actions, device=gpg_config.device)

    sas_dataset = TransitionDataset.load(gpg.n_actions, dataset_paths.train_file)
    so_dataset = ObservationDataset(sas_dataset)

    train_size = int(0.9 * len(so_dataset))
    val_size = len(so_dataset) - train_size
    train_ds, val_ds = random_split(so_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=True)

    masks = gpg.generate(train_loader, val_loader)

    torch.save(masks, dataset_paths.perturbation_masks)
    print(f"Saved perturbation masks to: {dataset_paths.perturbation_masks}")


if __name__ == "__main__":
    main()
