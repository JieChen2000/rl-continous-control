[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Continuous Control RL Example

## Introduction

For this project, we work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, the Unity environment contains 20 identical agents, each with its own copy of the environment.  
It is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores.

- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Enviroment Setup

The code is written in PyTorch and Python 3.

1.Clone this repo by
* git clone https://github.com/JieChen2000/rl-continous-control.git

2.Current code is tested with the Unity Machine Learning Agents (ML-Agents) environment on windows 64-bit operating system, download the enviroment
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3.Place the file in the this GitHub repository, and unzip (or decompress) the file.

4.Install openai gym, which is required by ML-Agents.

* `git clone https://github.com/openai/gym.git`

* `cd gym`

* `pip install -e .`

5.Install udacity git repo with unityAgent enviroment.

* `git clone https://github.com/udacity/deep-reinforcement-learning.git`
* `cd deep-reinforcement-learning/python`
* `conda install pytorch=0.4.0 -c pytorch`  
* `pip install .`  

## Running Instructions

* `python train_agent.py`
* `python run_agent.py`  //this will call the trained model and run the smart agent to interact with enviroment.

## Demo and Algorithm Description

The report `report.ipynb` describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures. A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward of at least +30 over 100 episodes.
