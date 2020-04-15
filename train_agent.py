
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt
import torch

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

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
def ddpg(n_episodes=1000, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=100)
    scores_mean = []
    for i_episode in range(1, n_episodes+1):
        # state = env.reset()
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment  
        states = env_info.vector_observations         # get the current state (for each agent)
        scores = np.zeros(num_agents)
        agent.reset()

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]     # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards
            if np.any(dones):
                break 
        scores_deque.append(np.mean(scores))
        scores_mean.append(np.mean(scores))

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverag Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if np.mean(scores_deque) > 30 and len(scores_deque) >= 100:
            print('\nEn)ironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores_mean

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(r'local_test\train_score_vs_episode.png')

env.close()