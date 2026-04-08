# ---------------------------
# 2) MuJoCo wrapper:
#    returns Dict obs = {"image", "text"}
# ---------------------------
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from gymnasium import spaces
import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym

from Text_tokenizer import tokenize
from Text_tokenizer import VOCAB, TASKS, MAX_TEXT_LEN, IMAGE_SIZE


class InstructionalHalfCheetah(gym.Env):
    """
    Dict observation:
      image: uint8, shape (3, 84, 84)
      text : int64, shape (MAX_TEXT_LEN,)
    Action: same as HalfCheetah-v4 continuous action space.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_size: Tuple[int, int] = IMAGE_SIZE, state_indices_to_keep: Optional[List[int]] = None, frame_skip: int = 5):
        super().__init__()
        self.render_size = render_size
        # Pass frame_skip directly to gym.make
        self.env = gym.make("HalfCheetah-v5", render_mode="rgb_array", frame_skip=frame_skip)
        self.render_mode = "rgb_array" # Explicitly set render_mode for the wrapper

        self.action_space = self.env.action_space

        original_state_dim = self.env.observation_space.shape[0]
        self._state_selection_indices = state_indices_to_keep
        if self._state_selection_indices is not None:
            current_state_dim = len(self._state_selection_indices)
        else:
            current_state_dim = original_state_dim


        self.observation_space = spaces.Dict(
            {
                # Channel-first to keep the SB3 pipeline simple.
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, render_size[0], render_size[1]),
                    dtype=np.uint8,
                ),
                "text": spaces.Box(
                    low=0,
                    high=len(VOCAB) - 1,
                    shape=(MAX_TEXT_LEN,),
                    dtype=np.int64,
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(current_state_dim,), dtype=np.float32
                ), # Include raw state observation
            }
        )

        self.current_task_id: int = 0
        self.current_instruction: str = TASKS[self.current_task_id]
        self._rng = np.random.default_rng()
        self.step_count = 0 # Initialize step_count

    def _sample_task(self, options: Optional[dict]) -> int:
        if options is not None and "task_id" in options:
            return int(options["task_id"])
        return int(self._rng.integers(0, len(TASKS)))

    def _render_image(self) -> np.ndarray:
        # Render every step to ensure fresh observations
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None. Check render_mode='rgb_array'.")

        frame = cv2.resize(frame, (self.render_size[1], self.render_size[0]))
        #img = Image.fromarray(frame).convert("L")
        #img = img.resize((self.render_size[1], self.render_size[0]), Image.BILINEAR)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        arr = np.asarray(img, dtype=np.uint8)   #HWC
        arr = np.expand_dims(arr, axis=0)
        #arr = np.transpose(arr, (2, 0, 1))     # CHW
        return arr

    def _get_obs(self) -> Dict[str, np.ndarray]:
        full_state = self.env.unwrapped._get_obs() # Access state from the unwrapped environment
        if self._state_selection_indices is not None:
            selected_state = full_state[self._state_selection_indices]
        else:
            selected_state = full_state

        return {
            "image": self._render_image(),
            "text": tokenize(self.current_instruction),
            "state": selected_state, # Return raw state, possibly subsetted
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        _, info = self.env.reset(seed=seed, options=options)
        self.step_count = 0 # Reset step count on environment reset

        self.current_task_id = self._sample_task(options)
        self.current_instruction = TASKS[self.current_task_id]

        obs = self._get_obs()
        info = dict(info)
        info["instruction"] = self.current_instruction
        info["task_id"] = self.current_task_id
        print("RESET TASK:", self.current_task_id)
        return obs, info

    def step(self, action):
        # Compute reward from the *instruction-conditioned* objective.
        xpos_before = float(self.env.unwrapped.data.qpos[0])

        # Removed the internal frame_skip loop, as frame_skip is now handled by gym.make
        _, _, terminated, truncated, info = self.env.step(action)

        xpos_after = float(self.env.unwrapped.data.qpos[0])
        # Corrected: Access frame_skip from the unwrapped environment
        x_velocity = (xpos_after - xpos_before) / float(self.env.unwrapped.dt * self.env.unwrapped.frame_skip)
        ctrl_cost = 0.1 * float(np.sum(np.square(action)))

        if self.current_task_id == 0:      # run forward
            reward = x_velocity - ctrl_cost
        elif self.current_task_id == 1:    # run backward
            reward = -x_velocity - ctrl_cost
        else:                              # run slowly
            reward = -abs(x_velocity) - ctrl_cost

        obs = self._get_obs()

        info = dict(info)
        info["instruction"] = self.current_instruction
        info["task_id"] = self.current_task_id
        info["x_velocity"] = x_velocity
        info["ctrl_cost"] = ctrl_cost
        info["shaped_reward"] = reward

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
