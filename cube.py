import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

colors = {
    0: "pink",
    1: "white",
    2: "blue",
    3: "orange",
    4: "green",
    5: "red",
    6: "yellow"
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
        def fillRect(x, y, val):
            ax.add_patch(Rectangle((x, y), .25, .25, facecolor = "white", edgecolor = "black"))
            for i in range(3):
                xVal = x + (.25 / self.size * i)
                for j in range(3):
                    yVal = y + (.25 / self.size * j)
                    ax.add_patch(Rectangle((xVal, yVal), .25/self.size, .25/self.size, 
                                           facecolor = colors[val]))

        #draw all the faces
        fillRect(0, .25, 4)
        fillRect(.25, .25, 0)
        fillRect(.5, .25, 2)
        fillRect(.25, .5, 3)
        fillRect(.25, 0, 1)
        fillRect(.75, .25, 5)

        plt.show()



    
        
a = cube(3)
a.drawCube()