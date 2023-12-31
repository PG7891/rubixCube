import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy

colors = {
    0: "white",
    1: "blue",
    2: "orange",
    3: "green",
    4: "red",
    5: "yellow"
}

faces = {
    0: ([1, 2, 3, 4], ),
    1: [5, 2, 0, 5],
    2: [],
}

class cube:
    def __init__(self, size):
        self.size = size

        def initalizeFace(color):
            result = np.zeros((size, size))
            result.fill(color)
            return result
        
        build = []

        for i in range(6):
            build.append(initalizeFace(i))

        self.cube = np.array(build)

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
                                           facecolor = colors[face[j][i]]))

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
        print(self.cube[2])
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
        



    
        
a = cube(4)
a.makeMove(0)
a.turnCube(True)
a.makeMove(1)
a.turnCube(False)
a.makeMove(2)
a.drawCube()
a.makeMove(0)
a.drawCube()
a.turnCube(True)
a.drawCube()
a.makeMove(1)
a.drawCube()
a.turnCube(False)
a.drawCube()
a.makeMove(2)
a.drawCube()
a.makeMove(0)
a.drawCube()
a.turnCube(True)
a.drawCube()
a.makeMove(1)
a.drawCube()
a.turnCube(False)
a.drawCube()
a.makeMove(2)
a.drawCube()