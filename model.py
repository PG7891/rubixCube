from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy


def solvedCube(size):
    def initalizeFace(color):
            result = np.zeros((size, size))
            result.fill(color)
            return result
        
    build = []

    for i in range(6):
        build.append(initalizeFace(i))

    return np.array(build)

def mixCube(cube, amount):
    for i in range(amount):
        rand = random.randint(0, 1)
        if rand == 0:
            rand = random.randint(0, 1)
            if rand == 0:
                cube.turnCube(True)
            else:
                cube.turnCube(False)
        else:
            rand = random.randint(0, cube.size-1)
            cube.makeMove(rand)

def checkSolved(cube):
    def allEqual(face):
        res = np.all(face == face[0, 0])
        return res
    
    solved = True
    for i in range(6):
        if not allEqual(cube.cube[i]):
            solved = False
    return solved

class cube:
    def __init__(self, size):
        self.size = size
        self.cube = solvedCube(size)

        self.colors = {
                0: "white",
                1: "blue",
                2: "orange",
                3: "green",
                4: "red",
                5: "yellow"
            }
        print(self.cube.shape)

    def drawCube(self):
        #define Matplotlib figure and axis
        fig, ax = plt.subplots()
        
        #function to draw one of the faces
        def fillRect(x, y, val, face):
            ax.add_patch(Rectangle((x, y), .25, .25, facecolor = "white", edgecolor = "black"))
            for i in range(self.size):
                xVal = x + (.25 / self.size * i)
                for j in range(self.size):
                    yVal = y + (.25 / self.size * (self.size-j-1))
                    ax.add_patch(Rectangle((xVal, yVal), .25/self.size, .25/self.size, 
                                           facecolor = self.colors[face[j][i]]))

        #draw all the faces
        fillRect(.25, .25, 0, self.cube[0])
        fillRect(.25, 0, 1, self.cube[1])
        fillRect(.5, .25, 2, np.rot90(self.cube[2], k=1))
        fillRect(.25, .5, 3, np.rot90(self.cube[3], k=2))
        fillRect(0, .25, 4, np.rot90(self.cube[4], k=3))
        fillRect(.75, .25, 5, self.cube[5])

        plt.show()
    
    def makeMove(self, index):
        #rotates on the index couting from the top
        if index == 0:
            self.cube[0] = np.rot90(self.cube[0])
        elif index == self.size-1:
            self.cube[5] = np.rot90(self.cube[5], k=3)
        #in all cases make rotation at specified index
        temp = copy.copy(self.cube[4][index])
        self.cube[4][index] = self.cube[3][index]
        self.cube[3][index] = self.cube[2][index]
        self.cube[2][index] = self.cube[1][index]
        self.cube[1][index] = temp

    def turnCube(self, up):
        if up:
            #side rotations
            self.cube[0] = np.rot90(self.cube[0], k=2)
            self.cube[2] = np.rot90(self.cube[2], k=3)
            self.cube[4] = np.rot90(self.cube[4], k=1)
            self.cube[5] = np.rot90(self.cube[5], k=2)   
            #big rotate
            self.cube = np.array([self.cube[1], self.cube[5], self.cube[2], self.cube[0], self.cube[4], self.cube[3]])
        else:
            #turn left
            #side rotations
            self.cube[0] = np.rot90(self.cube[0], k=1)
            self.cube[1] = np.rot90(self.cube[1], k=1)
            self.cube[2] = np.rot90(self.cube[2], k=1)
            self.cube[3] = np.rot90(self.cube[3], k=3) 
            self.cube[4] = np.rot90(self.cube[4], k=3)
            self.cube[5] = np.rot90(self.cube[5], k=3)
            #big rotate
            self.cube = np.array([self.cube[2], self.cube[1], self.cube[5], self.cube[3], self.cube[0], self.cube[4]])

class RubixCubeEnv(Env):
    def __init__(self, size):
        self.action_space = Discrete(size+2)
        self.cubeSize = size
        self.cube = cube(size)
        self.observation_space = np.zeros((self.cube.cube.shape))

        mixCube(self.cube, 100)

        self.maxLength = 200
        self.length = 200

    def step(self, action):
        #perform action
        if action == 0:
            self.cube.turnCube(True)
        elif action == 1:
            self.cube.turnCube(False)
        else:
            self.cube.makeMove(action-2)

        done = False

        #check if cube is solved for rewards
        if checkSolved(self.cube):
            reward = 100
            done = True
        else:
            reward = -1

        #check if episode is over
        if self.length <= 0:
            done = True
        
        self.length -= 1

        info = {}

        return self.cube, reward, done, info

    def render(self):
        pass
    def reset(self):
        self.cube = cube(self.cubeSize)
        mixCube(self.cube, 100)
        self.length = self.maxLength
        print("Here", self.length)
        return self.cube
    

modelSize = 3
env = RubixCubeEnv(modelSize)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def buildModel(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(6, modelSize, modelSize)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

states = env.observation_space.shape
print(states)
actions = env.action_space.n
model = buildModel(states, actions)
model.summary()

from keras import __version__
tf.keras.__version__ = __version__

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae'])
#dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)