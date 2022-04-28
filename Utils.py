#Class for storing solution values from various models
class Solution():
    def __init__(self, sizeDict):
        self.x={(t,j,k):0 for t in range(sizeDict['T']) for j in range(sizeDict['M']) for k in range(sizeDict['Mj'][j])}
        self.y={(t,j):0 for t in range(sizeDict['T']) for j in range(sizeDict['M'])}