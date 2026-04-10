# Visual-CNN--Languege-LLM--Action-model-on-HalfCheetah-and-more

In this repository, suppose we are the robotic engineer try to design a robot and see how it interacts with our physical world, we need to determine which environment we are working in and how our model processes physical information, such as numerical state data, visual observations, or even human commands. 

We utilize MuJoCo as our simulation environment, and select the HalfCheetah-v5 task as our primary testbed, as it provides a relatively simple yet representative control problem. Then we integrate multiple sources of information—such as physical state data, visual inputs, and human commands—into the system, enabling the robot to perceive and respond to its environment.

## About this "Ver.0.9" branch:

This is demo version, only contain simplified text input with strict reward options, raw CNN ingestion, and simple features concatenate.

## Features in this project

Featuring as HalfCheetah-V.5:
- Import PPO from stable_baselines3 package 
- Customized featuresExtractor with:
  - Part of (8 of 17, which contain positions & angles) numerical state data inputs.
  - Deep-CNN image input.
  - Simple text ingestion which demostrate human commands that instruction reward functions.

Note that due to the excessive time of the simulation, we utilized Google drive checkpoints to preventing unintentional abrupt in our learning process. This saving method can be replace with others depending on which serve you best.

### The following features will be add in new version(branch):
- Introduce Vision Transformer (ViT) layers after the CNN module to capture global image structure.
- Apply transformer-based encoders after feature fusion for better global representation learning.
- Replace simple instruction input with LLM to understand more general text inputs.
- Redesign reward function to be more flexible and continuous. (For example, an LLM could map instructions into a latent representation encoding both direction and magnitude (e.g., velocity scaling from -1 to 1), which can then be used to parameterize the reward.)

### Potential improvements:

- Improve parallelization, for example by moving CNN computations to CUDA while avoiding bottlenecks from rendering, or by replacing the rendering pipeline with a more parallelizable alternative.
- Explore the use of vision-language models (VLMs) to jointly process image and text inputs, potentially reducing the need for training CNN modules from scratch and improving efficiency.
- Apply additional reinforcement learning algorithms beyond PPO algorithm.
- Extend our gym from HalfCheetah to other environments to see how our current model was learned or can learn.
- Note that when using DummyVecEnv, the CNN processes only single frames and cannot capture temporal dependencies. So in the future we might have to utilize the "VecFrameStack" to let CNN framework learn more properly.



## Contents of code

VLA\
├── Google_drive.py      # Assignig files location on drive.

├── Env_setup.py                # python method of environment setup. 

├── Text_tokenizer.py            # demo method to text embeding. \
&emsp;&emsp;└── image input size  # we take input image resolution as 36*36.

├── Mujpco_wrapper.py                # use InstructionalHalfCheetah to configure Mujoco.\
&emsp;&emsp;├── gym.make  # gym environment setting, currently "HalfCheetahV5". 

&emsp;&emsp;├── observation_space  # set the data that will be observed in our env.\
&emsp;&emsp;&emsp;&emsp;├── states	 \
&emsp;&emsp;&emsp;&emsp;├── text 		\
&emsp;&emsp;&emsp;&emsp;└── image   

&emsp;&emsp;├── _get_obs   #  observing current env to get input.\
&emsp;&emsp;&emsp;&emsp;├── states	# get numerical states from env. \
&emsp;&emsp;&emsp;&emsp;├── text 		# get the text instruction\
&emsp;&emsp;&emsp;&emsp;└── image   # get reder input as image.

&emsp;&emsp;├── steps   # how our model will learn in each iteration.\
&emsp;&emsp;&emsp;&emsp;└── reward fuction.  # ingeste simple text instruction to choose desire reward.

├── Feature extractor.py                # use class ImageTextExtractor to extracte features.
&emsp;&emsp;├── CNN  # Send render image trough CNN layer to get 32*batch_size features map. 

&emsp;&emsp;├── text_emb  # project text onto 32 *batch_size features map.

&emsp;&emsp;├── state_mlp   #  project states onto 32 *batch_size features map.

&emsp;&emsp;└── fusion.  # fuse all three input to a 32 *3 *batch_size features map.


├── Training and Rollout  # train our PPO model with custom max iterate steps, number of steps, number of epoch, and batch_size. \

└── README.md

## How to operate

### Mount Drive
- First, you might need to mount the model on the Google drive.

Note that since we want to leveraging the checkpoints saving from the Google drive, we will need to operate following code in Google Colab notebook environment very quick, then we can on our way to model preparation via python files downloading and operating in terminal.

```bash
from google.colab import drive
drive.mount('/content/drive')

```

After the drive is mounted with Colab notebook, we now can execute following command to assigning checkpoints' file location in our Google drive.
```bash
python3 Google_drive.py
```

### Setting up a environment

- Second, you'll need to install necessary packages. You can either use bash command as Section 1, or download "Env_setup.py" to setup environment with Section 2 command, or even just copy "Env_setup_colab" code to Colab notebook.

#### Section 1
```bash

pip install mujoco stable_baselines3 mediapy
apt-get update
apt-get install -y libosmesa6-dev ffmpeg

export MUJOCO_GL=osmesa
```

#### Section 2
```bash
python3 Env_setup.py
```


### Start the training
- Third, after our environment is installed, we can now excute "Train&Rollout.py" to start training and proceed several rollout prediction.

The input image resolution of CNN is declared in "Text_tokenizer.py", and the max training steps and PPO variables, such as number of steps(ex: 2048), number of epoch(ex: 3), batch_size(ex: 128) and neural numbers of policy "pi" and state value function "vf" layers, are with "Train&Rollout.py".

```bash
python3 Train&Rollout.py
```

### Render the Mujoco video
- Finally, after we complete the training and rollout, we can render the vedio of model's learning to see how our cheetah will move.

```bash
python Rendering.py
```
The expected result is shown in the vedio below. 

![demo](Adobe_Express_VLAback.gif)
