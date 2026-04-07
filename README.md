# Visual-CNN--Languege-LLM--Action-model-on-HalfCheetah-and-more

In this repository, suppose we are the robotic engineer try to design a robot and see how it interacts with our physical world, we need to determine which environment we are working in and how our model processes physical information, such as numerical state data, visual observations, or even human commands. 

We utilize MuJoCo as our simulation environment, and select the HalfCheetah-v5 task as our primary testbed, as it provides a relatively simple yet representative control problem. Then we integrate multiple sources of information—such as physical state data, visual inputs, and human commands—into the system, enabling the robot to perceive and respond to its environment.

### About this "Ver.0.9" branch:

This is demo version, only contain simplified text input with strict reward options, raw CNN ingestion, and simple features concatenate.

The following features will be add in new version(branch):
- Introduce Vision Transformer (ViT) layers after the CNN module to capture global image structure.
- Apply transformer-based encoders after feature fusion for better global representation learning.
- Replace simple instruction input with LLM to understand more general text inputs.
- Redesign reward function to be more flexible and continuous. (For example, an LLM could map instructions into a latent representation encoding both direction and magnitude (e.g., velocity scaling from -1 to 1), which can then be used to parameterize the reward.)
- 

## Features in this project

Featuring as HalfCheetah-V.5:
- Import PPO from stable_baselines3 package 
- Customized featuresExtractor with:
  - Part of (8 of 17, which contain positions & angles) numerical state data inputs.
  - Deep-CNN image input.
  - Simple text ingestion which demostrate human commands that instruction reward functions.
- 

Note: This project requires access to IBM watsonx.ai.
 If you do not have credentials, 
 please refer to the code structure and architecture sections for implementation details.


## Contents of code

qabot\
├── doc_load_embbed.py      # documents loading func. + embbeding func.\
&emsp;&emsp;└── document_loader     # PDF / CSV / TXT / json loader

├── qabot.py                # Gradio entry point \
&emsp;&emsp;├── llm \
&emsp;&emsp;&emsp;&emsp;├── watsonx_llm()    # get_llm() \
&emsp;&emsp;&emsp;&emsp;├── model_id      			# model_id \
&emsp;&emsp;&emsp;&emsp;└── project_id       # project_id \

&emsp;&emsp;├── retrievers\
&emsp;&emsp;&emsp;&emsp;├── parent_retriever  # ParentDocumentRetriever \
&emsp;&emsp;&emsp;&emsp;├── embedding									# watsonx_embedding() \
&emsp;&emsp;&emsp;&emsp;├── vectorstore 						# Chroma()\
&emsp;&emsp;&emsp;&emsp;└── document_loader\

&emsp;&emsp;├── qa_chains + globle chatmemory\
&emsp;&emsp;&emsp;&emsp;├── llm\
&emsp;&emsp;&emsp;&emsp;└── LCEL structure # prompt, chain, chat_memory\ 

&emsp;&emsp;├── gradio\
&emsp;&emsp;&emsp;&emsp;└── gr.Interface\

├── Setting up a virtual environment + requirements.txt   # Necessary libs\

└── README.md

## How to operate

- First, you might need to create a virtual environment to operate it.

### Setting up a virtual environment

```bash
python3.11 -m venv my_env
source my_env/bin/activate

```
You should see "(my_env)" before your machine as the env had set up successfully.

- Second, you'll need to install necessary packages.
```bash
pip install -r requirements.txt
```

- Third, after libs installed, you can now compile and excute the code.

```bash
python3.11 qabot.py
```

- Finally, through the following http site at your local terminal, you now can access the Langchain-Chatbot-with-document-loading, and ask it any question that you see fit.

```bash
http://---.-.-.-:7860
```
The expected format is shown in the figure below. 
![image](chat_demo.jpg)
