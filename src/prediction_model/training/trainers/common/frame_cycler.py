import torch
from typing import Optional


class FrameCycler:
    def __init__(self):
        self._current_state: Optional[torch.Tensor] = None

    def save_current_state(self, current_state: torch.Tensor):
        self._current_state = current_state

    def cycle_frames(self, predicted_states: torch.Tensor):
        assert self._current_state is not None, "Need to pass current state first"

        if predicted_states.dim() < self._current_state.dim():
            predicted_states = predicted_states.unsqueeze(1)

        return torch.concat(
            [
                self._current_state[:, 1:],
                predicted_states,
            ],
            dim=1,
        )
