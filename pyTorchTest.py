#This file is to learn the pytorch implementation of DQN deu to depricationa and version issues with keras and tensorflow

from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random


gamma = .99
batchSize = 32
bufferSize = 50000
minReplaySize = 1000
epsilonStart = 1.0
epsilonEnd = 0.02
epsilonDecay = 10000
targetUpdateFreq = 1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        inFeatures = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(inFeatures, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        qValues = self(obs_t.unsqueeze(0))

        maxQIndex = torch.argmax(qValues, dim=1)[0]
        action = maxQIndex.detach().item()
        return action

env = gym.make('CartPole-v0')

replayBuffer = deque(maxlen=bufferSize)
rewBuffer = deque([0, 0], maxlen = 100)

episodeReward = 0.0

onlineNet = Network(env)
targetNet = Network(env)

targetNet.load_state_dict(onlineNet.state_dict())

optimizer = torch.optim.Adam(onlineNet.parameters(), lr = 5e-4)

#initalize replay buffer
obs = env.reset()

for _ in range(minReplaySize):
    action = env.action_space.sample()

    newObs, rew, done, _, _ = env.step(action)
    if len(obs) == 2:
        obs = obs[0]
    transition = (obs, action, rew, done, newObs)
    
    replayBuffer.append(transition)
    obs = newObs

    if done:
        obs = env.reset()

obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, epsilonDecay], [epsilonStart, epsilonEnd])

    randSample = random.random()

    if len(obs) == 2:
        obs = obs[0]

    if randSample <= epsilon:
        action = env.action_space.sample()
    else:
        action = onlineNet.act(obs)

    newObs, rew, done, _, _ = env.step(action)
    
    transition = (obs, action, rew, done, newObs)
    replayBuffer.append(transition)
    obs = newObs

    episodeReward += rew

    if done:
        obs = env.reset()

        rewBuffer.append(episodeReward)
        episodeReward = 0

    if len(rewBuffer) >= 100:
        if np.mean(rewBuffer) >= 195:
            while True:
                action = onlineNet.act(obs)

                obs, _, done, _, _ = env.step(action)
                env.render()
                if done:
                    env.reset()

    #gradient steps
    transitions = random.sample(replayBuffer, batchSize)

    obses = np.asarray([t[0] for  t in transitions])
    actions = np.asarray([t[1] for  t in transitions])
    rews = np.asarray([t[2] for  t in transitions])
    dones = np.asarray([t[3] for  t in transitions])
    newObses = np.asarray([t[4] for  t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    newObses_t = torch.as_tensor(newObses, dtype=torch.float32)

    #targets
    targetQValues = targetNet(newObses_t)
    maxTargetQValues = targetQValues.max(dim=1, keepdim=True)[0]

    targets = rews_t + gamma * (1-dones_t) * maxTargetQValues

    #loss
    qValues = onlineNet(obses_t)

    actionQValues = torch.gather(input=qValues, dim=1, index=actions_t)

    loss = nn.functional.smooth_l1_loss(actionQValues, targets)

    #gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #update target network
    if step % targetUpdateFreq == 0:
        targetNet.load_state_dict(onlineNet.state_dict())

    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Rew', np.mean(rewBuffer))