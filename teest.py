from keras.layers import Dense, Flatten
import gym
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10

def buildModel(statez, actiones):
    model = Sequential()
    model.add(Flatten(input_shape=(1, statez)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actiones, activation='linear'))
    return model

model = buildModel(states, actions)

def buildAgent(modell, actionz):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=modell, memory=memory, policy=policy, nb_actions=actionz, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

DQN = buildAgent(model, actions)
DQN.compile(Adam(lr=1e-3), metrics=['mae'])
DQN.fit(env, nb_steps=50000, visualize=False, verbose=1)