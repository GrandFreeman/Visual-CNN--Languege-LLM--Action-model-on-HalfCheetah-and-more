# ---------------------------
# 4) Train PPO
# ---------------------------

from MuJoCo_wrapper import InstructionalHalfCheetah
from Feature_extractor import ImageTextExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from typing import Dict, Tuple, Optional, List
from Text_tokenizer import VOCAB

import re
import os

def make_env(state_indices_to_keep: Optional[List[int]] = None, frame_skip: int = 5):
    return InstructionalHalfCheetah(state_indices_to_keep=state_indices_to_keep, frame_skip=frame_skip)


if __name__ == "__main__":

    dummy_env = gym.make("HalfCheetah-v5")
    original_state_dim = dummy_env.observation_space.shape[0]
    dummy_env.close()

    selected_state_indices = list(range(8))
    # Pass the desired frame_skip to make_env
    env = DummyVecEnv([lambda: make_env(state_indices_to_keep=selected_state_indices, frame_skip=5)])
    #env = VecFrameStack(env, n_stack=4)

    policy_kwargs = dict(
        features_extractor_class=ImageTextExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            vocab_size=len(VOCAB),
        ),
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128],
        ),
        normalize_images=False,  # we normalize manually in the extractor
    )


    # Define save path for the main model file and TensorBoard logs
    # This assumes `checkpoint_dir` is defined in a previous cell (after Google Drive is mounted)
    # If you run this cell independently, ensure `checkpoint_dir` is set appropriately.
    global checkpoint_dir # Access global variable defined after drive mount
    if 'checkpoint_dir' not in locals() and 'checkpoint_dir' not in globals():
        print("Warning: checkpoint_dir not found. Setting a default local path.")
        checkpoint_dir = './colab_checkpoints/halfcheetah_ppo_state_image_text/'
        os.makedirs(checkpoint_dir, exist_ok=True)

    model_save_path = os.path.join(checkpoint_dir, "latest_ppo_hyb_model.zip")
    tensorboard_log_path = os.path.join(checkpoint_dir, "tb_logs")
    total_timesteps_to_train = 100_000 # Consider increasing this significantly, e.g., to 1_000_000 or more

    model = None
    load_path = None
    reset_num_timesteps_flag = True

    max_timesteps_found = -1
    best_load_path = None

    # First, consider the explicitly saved final model (latest_ppo_hyb_model.zip)
    if os.path.exists(model_save_path):
        try:
            # Temporarily load to get num_timesteps for comparison
            # We pass env=None here to avoid potentially recreating the environment twice with different settings
            temp_model = PPO.load(model_save_path, env=None)
            if temp_model.num_timesteps > max_timesteps_found:
                max_timesteps_found = temp_model.num_timesteps
                best_load_path = model_save_path
            del temp_model # Release memory
        except Exception as e:
            print(f"Warning: Could not load {model_save_path} to check timesteps: {e}")

    # Now, scan for checkpoint files in the directory
    for fname in os.listdir(checkpoint_dir):
        # Check if the file is a checkpoint saved by CheckpointCallback
        match = re.match(r"ppo_halfcheetah_checkpoint_(\d+)_steps\.zip", fname)
        if match:
            steps = int(match.group(1))
            if steps > max_timesteps_found:
                max_timesteps_found = steps
                best_load_path = os.path.join(checkpoint_dir, fname)

    if best_load_path:
        load_path = best_load_path
        print(f"Loading model with highest timesteps from: {load_path} (timesteps: {max_timesteps_found}) to resume training.")
        model = PPO.load(load_path, env=env, custom_objects={
            "lr_schedule": lambda progress_remaining: 3e-4 # Re-add learning rate schedule
        })
        print(f"Resuming training from timestep {model.num_timesteps}.")
        reset_num_timesteps_flag = False
    else:
        print("No existing model or checkpoint found. Starting training from scratch.")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2048,
            batch_size=128,
            n_epochs=3,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            tensorboard_log=tensorboard_log_path, # Updated TensorBoard log path
        )

    # Setup checkpoint callback to save the model periodically
    # Save a checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="ppo_halfcheetah_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Train the model with the callback
    if best_load_path:
      remaining_steps = total_timesteps_to_train - model.num_timesteps

      if remaining_steps > 0:
          model.learn(
              total_timesteps=remaining_steps,
              callback=checkpoint_callback,
              reset_num_timesteps=False,
          )
      else:
          print("Already trained enough.")
    else:
      model.learn(
          total_timesteps=total_timesteps_to_train,
          callback=checkpoint_callback,
      )
    
    # Save the final model to the designated path
    model.save(model_save_path)
    print(f"Training finished. Final model saved to {model_save_path}")


    # ---------------------------
    # 5) Quick rollout demo
    # ---------------------------
    env.envs[0].current_task_id = 1
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        print(obs.keys())
        print(infos[0]["task_id"], infos[0]["x_velocity"])

        if dones[0]:
            obs = env.reset()
