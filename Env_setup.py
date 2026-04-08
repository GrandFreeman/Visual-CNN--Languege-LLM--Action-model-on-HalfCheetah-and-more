import os
import subprocess
import sys

def run(cmd):
    print(f"[Running] {cmd}")
    subprocess.check_call(cmd, shell=True)

# Install Python packages
run(f"{sys.executable} -m pip install mujoco")
run(f"{sys.executable} -m pip install stable_baselines3")

# Install system package (Linux only)
run("sudo apt-get update")
run("sudo apt-get install -y libosmesa6-dev")

# Set environment variable
os.environ["MUJOCO_GL"] = "osmesa"

print("Using OSMesa for headless rendering.")

# Check MuJoCo
try:
    print("Checking that the installation succeeded:")
    import mujoco
    mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
    raise RuntimeError(
        "Something went wrong during installation or MuJoCo initialization.\n"
        "Check the shell output above for more information.\n"
        "Ensure OSMesa is correctly installed for CPU rendering."
    ) from e

print("Installation successful.")

# Install ffmpeg if not exists
try:
    subprocess.check_call("command -v ffmpeg", shell=True)
except subprocess.CalledProcessError:
    run("sudo apt-get install -y ffmpeg")

# Install mediapy
run(f"{sys.executable} -m pip install mediapy")

print("All setup complete.")
