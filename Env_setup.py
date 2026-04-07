!pip install mujoco
!pip install stable_baselines3

# Install libosmesa6-dev for headless rendering with OSMesa
!apt-get update && apt-get install -y libosmesa6-dev

%env MUJOCO_GL=osmesa

print('Skipping GPU-specific setup as requested, using OSMesa for rendering.')

# Check if installation was succesful.
try:
  print('Checking that the installation succeeded:')
  import mujoco
  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  # Modified error message to reflect removal of GPU-specific setup
  raise e from RuntimeError(
      'Something went wrong during installation or MuJoCo initialization. '
      'Check the shell output above for more information. '
      'The GPU-specific setup has been removed as per the request. '
      'If you encounter rendering issues, consider configuring a CPU-based renderer (e.g., using MUJOCO_GL=osmesa) '
      'or ensure a properly configured GPU runtime if GPU rendering is intended.')

print('Installation successful.')

# Graphics and plotting.
print('Installing mediapy:')
!command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
!pip install -q mediapy


from IPython.display import clear_output
clear_output()
