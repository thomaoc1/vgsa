import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


class BasePathConfig:
    def __init__(self, root: str, algo: str, env: str):
        self.root = root
        self.algo = algo
        self.env = env

    @property
    def base(self):
        return ensure_dir(os.path.join(self.root, self.algo, self.env))

    def join(self, *parts):
        return os.path.join(self.base, *parts)


class ModelsPaths(BasePathConfig):
    def __init__(self, algo: str, env: str, seed: int | None, run_name: str | None):
        super().__init__("models", algo, env)
        self.seed = seed
        self.run_name = run_name


class PredictionModelPaths(ModelsPaths):
    def __init__(
        self,
        algo: str,
        env: str,
        encoded: bool,
        agent_seed: int,
        seed: int | None = None,
        run_name: str | None = None,
    ):
        seed = seed if seed else 42
        super().__init__(algo, env, seed, run_name)
        self.agent_seed = agent_seed
        self.encoded = encoded

    @property
    def prediction_model_dir(self):
        if self.encoded:
            return ensure_dir(self.join("prediction_model", "encoded"))
        else:
            return ensure_dir(self.join("prediction_model"))

    @property
    def prediction_model_ram_dir(self):
        return ensure_dir(self.join("prediction_model", "ram"))

    @property
    def prediction_model_weights(self):
        filename = f"model_weights_agent_seed_{self.agent_seed}_seed_{self.seed}.pt"
        return os.path.join(self.prediction_model_dir, filename)

    @property
    def ram_prediction_model_weights(self):
        filename = f"model_weights_agent_seed_{self.agent_seed}_seed_{self.seed}.pt"
        return os.path.join(self.prediction_model_ram_dir, filename)

    def ram_model_weights(self):
        base = self.ram_prediction_model_weights
        base_no_ext = os.path.splitext(base)[0]

        suffix = ""

        if self.seed is not None:
            suffix += f"_seed_{self.seed}"

        return f"{base_no_ext}{suffix}.pt"

    def model_weights(self):
        base = self.prediction_model_weights
        base_no_ext = os.path.splitext(base)[0]

        suffix = ""

        if self.seed is not None:
            suffix += f"_seed_{self.seed}"

        return f"{base_no_ext}{suffix}.pt"


class PolicyPaths(ModelsPaths):
    def __init__(self, algo: str, env: str, seed: int | None = None, run_name: str | None = None):
        super().__init__(algo, env, seed, run_name)

    # --- Directories ---
    @property
    def policy_dir(self):
        return ensure_dir(self.join("policy"))

    @property
    def checkpoint_dir(self):
        return ensure_dir(os.path.join(self.policy_dir, "checkpoint"))

    # --- Files ---
    @property
    def policy_file(self):
        return self.join("policy", "policy.zip")

    def sb3_run_path(self):
        base = self.policy_file
        base_no_ext = os.path.splitext(base)[0]

        suffix = ""

        if self.seed is not None:
            suffix += f"_seed_{self.seed}"

        return f"{base_no_ext}{suffix}.zip"


class DatasetPaths(BasePathConfig):
    def __init__(self, algo, env, encoded: bool, agent_seed: int):
        super().__init__("datasets", algo, env)
        self.encoded = encoded
        self.agent_seed = agent_seed

    # --- Directories ---
    @property
    def dataset_dir(self):
        if self.encoded:
            return ensure_dir(self.join("dataset", "encoded"))
        else:
            return ensure_dir(self.join("dataset"))

    @property
    def ram_dataset_dir(self):
        return ensure_dir(self.join("dataset", "ram"))

    @property
    def perturbation_mask_dir(self):
        return ensure_dir(self.join("perturbation_mask"))

    # --- Files ---
    @property
    def train_file(self):
        filename = f"train_agent_seed_{self.agent_seed}.pt"
        return os.path.join(self.dataset_dir, filename)

    @property
    def ram_train_file(self):
        filename = f"train_agent_seed_{self.agent_seed}.pt"
        return os.path.join(self.ram_dataset_dir, filename)

    @property
    def perturbation_masks(self):
        filename = f"perturbation_masks_agent_seed_{self.agent_seed}.pt"
        return self.join("perturbation_mask", filename)


def main():
    # ----- Create policy + checkpoint dirs -----
    ext = "NoFrameskip-v4"
    for algo in ["A2C", "PPO", "DQN"]:
        for env in [f"Pong{ext}", f"Breakout{ext}", f"SpaceInvaders{ext}", f"Qbert{ext}"]:
            policy_paths = PolicyPaths(algo=algo, env=env, seed=42, run_name="test_run")
            _ = policy_paths.policy_dir
            _ = policy_paths.checkpoint_dir

            # ----- Create prediction model dirs (encoded + non-encoded) -----
            pred_paths = PredictionModelPaths(algo=algo, env=env, agent_seed=123, encoded=False)
            _ = pred_paths.prediction_model_dir

            pred_paths_encoded = PredictionModelPaths(algo=algo, env=env, agent_seed=123, encoded=True)
            _ = pred_paths_encoded.prediction_model_dir

            # ----- Create dataset dirs -----
            dataset_paths = DatasetPaths(algo=algo, env=env, encoded=False, agent_seed=123)
            _ = dataset_paths.dataset_dir
            _ = dataset_paths.perturbation_mask_dir

            dataset_paths_encoded = DatasetPaths(algo=algo, env=env, encoded=True, agent_seed=123)
            _ = dataset_paths_encoded.dataset_dir
    print("All example directories created successfully.")


if __name__ == "__main__":
    main()
