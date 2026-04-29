# Value-Guided Strategic Attack (VGSA)
Attack on DRL policies which uses a learned environment transition function and leverages the victim policy's value estimate to dictate the direction of the attack. 

Uses Hydra-based pipelines for:
1. rollout collection,
2. next-state prediction model training,
3. perturbation mask generation,
4. attack evaluation.

## Setup (uv)

```bash
# from repo root
uv sync
source .venv/bin/activate  # optional
```

Python 3.12+ is required (`pyproject.toml`).

## Project workflow

The repo uses Hydra configs in `conf/` and script entrypoints under `src/`:

- Collect rollouts: `src/prediction_model/rollout_collection/collect.py`
- Train prediction model: `src/prediction_model/training/run.py`
- Generate perturbation masks: `src/attacker/global_perturbation/gen_perturbation.py`
- Run attacks: `src/run_attack/run.py`

## Example commands

```bash
# 1) Collect dataset (saved under datasets/<POLICY>/<ENV>/dataset/)
uv run python -m src.prediction_model.rollout_collection.collect \
  policy=dqn gym_env=pong n_frames=500000

# 2) Train next-state predictor (saved under models/<POLICY>/<ENV>/prediction_model/)
uv run python -m src.prediction_model.training.run \
  policy=dqn gym_env=pong next_state_pm=atari_obs

# 3) Run VGSA attack
uv run python -m src.run_attack.run \
  policy=dqn gym_env=pong attacker=vgsa
```

## Notes

- Policies are loaded from `models/<POLICY>/<ENV>/policy/policy_seed_<SEED>.zip`.
- `run_attack` and prediction-model training use Weights & Biases (`wandb`); configure login/environment before running.
- Atari env presets are in `conf/gym_env/` (e.g. `pong`, `breakout`, `space_invaders`, `qbert`, `mspacman`).
