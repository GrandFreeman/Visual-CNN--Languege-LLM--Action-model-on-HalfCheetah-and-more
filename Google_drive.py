from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define a base directory for checkpoints and logs on Drive
COLAB_CHECKPOINT_BASE_DIR = '/content/drive/MyDrive/colab_checkpoints/'
os.makedirs(COLAB_CHECKPOINT_BASE_DIR, exist_ok=True)

# Specific directory for this HalfCheetah PPO_VLA experiment
checkpoint_dir = os.path.join(COLAB_CHECKPOINT_BASE_DIR, 'halfcheetah_ppo_state_image_text')
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Checkpoints and TensorBoard logs will be saved to: {checkpoint_dir}")
