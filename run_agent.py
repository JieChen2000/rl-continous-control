
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from ddpg_agent import Agent

env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe', seed = 0)  ##multi agent env ## mutlip-agent env, seed = 0

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain_name, brain)

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])



agent = Agent(state_size=state_size, action_size=action_size,random_seed=0)
# load the weights from file
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))


env_info = env.reset(train_mode=False)[brain_name]     # reset the environment  
states = env_info.vector_observations         # get the current state (for each agent)
scores = np.zeros(num_agents)
for t in range(1000):
    actions = agent.act(states, add_noise=False)
    env_info = env.step(actions)[brain_name]     # send the action to the environment                rewards = env_info.rewards                   # get the reward
    dones = env_info.local_done                  # see if episode has finished
    rewards = env_info.rewards                   # get the reward
    states = env_info.vector_observations   # get the next state
    scores += rewards
    if np.any(dones):
        break 
print("average score for the episode is", np.mean(scores))
env.close()