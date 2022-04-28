import json
from math import atan, exp
import numpy as np
import pandas as pd
from time import time

from Utils import Solution


class GreedyOne():
    def __init__(self, Data, params = None):
        tic=time()
        self.Data=Data
        self.printDetails = False
        self.searchMode = 'm'

        if params is not None:
            self.__dict__.update(params)
        
        #Remaining budget for incumbent solution
        self.B_star = list(self.Data.B)
        #Max outlets at each station
        self.Mj_star = list(self.Data.Mj)
        #Incumbent solution
        self.x_star = np.zeros((self.Data.T, self.Data.M))
        for t in range(self.Data.T):
            for j in range(self.Data.M):
                self.x_star[t][j] = self.Data.x0[j]
        #Objective value of incumbent solution
        self.TotalEVs_star=np.zeros(self.Data.T)

        self.Covering = [pd.DataFrame() for _ in range(self.Data.T)]
        for t in range(self.Data.T):
            self.Covering[t]['Home'] = self.Data.Covering[t]['Home']
            for j in range(self.Data.M):
                if self.x_star[t][j] > 0:
                    self.Covering[t][str(j)] = self.Data.Covering[t][(j, self.x_star[t][j])]
            self.Covering[t] = pd.DataFrame({'Covering':self.Covering[t].any(axis=1)})
            self.TotalEVs_star[t] = (self.Covering[t]['Covering']).dot(self.Data.Populations[t])
        if self.searchMode == 'h':
            for t in range(1, self.Data.T, -1):
                self.TotalEVs_star[t-1] += self.TotalEVs_star[t]  
 
        for t in range(self.Data.T):
            if self.printDetails:
                print("t=", t)
                print('\n')
                
            self.Cost=np.ones(self.Data.M)
            if self.printDetails:
                print(self.Covering.head(3))
            
            i = 0
            #Iteratively find best outlet to place until no budget remains in the year
            #or no improvements can be found
            while True:
                TotalNewEVs = self.FindStationTotals(t)
                if self.printDetails:
                    i += 1
                    print("Iteration:", i)
                    print("EV total:", self.TotalEVs_star[t])
                    print(TotalNewEVs)
                    print('\n')
                new = max(TotalNewEVs)
                if new > self.TotalEVs_star[t]:
                    j = TotalNewEVs.argmax()
                    self.B_star[t] -= self.Cost[j]
                    for t1 in range(t, self.Data.T):                        
                        self.x_star[t1][j] += 1
                        self.Covering[t1][str(j)] = self.Data.Covering[t1][(j, self.x_star[t1][j])]
                        self.Covering[t1] = pd.DataFrame({'Covering':self.Covering[t1].any(axis=1)})
                    self.TotalEVs_star[t] = new
                else:
                    break




        toc=time()
        self.HeuristicSolvingTime = toc-tic
        self.HeuristicStats = {'EVs':self.TotalEVs_star, 'Budget Remaining': self.B_star, 'Runtime (seconds)':toc-tic}
        self.Solution = self.ConvertSolution(self.x_star)
        print("Heuristic solving time (seconds): ", toc-tic)
    
    #Calculate maximum number of new outlets that can be installed at
    #station j in year t (given budget and max sizing)
    def MaxNewOutlets(self,t,j):
        if self.x_star[t][j] >= self.Mj_star[j] -1:
            return 0
        
        elif self.x_star[t][j] > 0:
            nStations = int(self.B_star[t]/self.Data.c[t][j])
            self.Cost[j] = self.Data.c[t][j]
            return min(self.Mj_star[j]-1 - self.x_star[t][j], nStations)
            
        else:
            nStations = int((self.B_star[t]-self.Data.f[t][j])/self.Data.c[t][j])
            self.Cost[j] = self.Data.f[t][j]+self.Data.c[t][j]
            return min(self.Mj_star[j]-1 - self.x_star[t][j], nStations)

    #Calculate the objective function after adding one outlet
    #to all possible stations. 
    #Returns an array of size M
    def FindStationTotals(self,t):
        TotalNewEVs = np.zeros(self.Data.M)
        for j in range(self.Data.M):
            mj = self.MaxNewOutlets(t,j)
            if mj <= 0:
                TotalNewEVs[j]=0
                continue
            
            #In myopic search mode, we only take into account increases to
            #EV adoption in the current year
            if self.searchMode == 'm':
                self.Covering[t]['Station'] = self.Data.Covering[t][(j, self.x_star[t][j]+1)]
                TotalNewEVs[j] += (self.Covering[t].any(axis=1)).dot(self.Data.Populations[t])
                del self.Covering[t]['Station']

            #In hyperoptic search mode, we take into account increases to 
            #EV adoption across the entire planning horizon
            elif self.searchMode == 'h':
                for t1 in range(t, self.Data.T):
                    self.Covering[t1]['Station'] = self.Data.Covering[t1][(j, self.x_star[t1][j]+1)]
                    TotalNewEVs[j] += (self.Covering[t1].any(axis=1)).dot(self.Data.Populations[t1])  
                    del self.Covering[t1]['Station']     
        return TotalNewEVs

    #Convert solution to use as warmstart in maximum covering model
    def ConvertSolution(self,sol):
        Sol =Solution({'T': self.Data.T, 'M': self.Data.M, 'Mj':self.Data.Mj})
        for t in range(self.Data.T):
            for j in range(self.Data.M):
                Sol.y[(t,j)] = 1 if sol[t][j] >0 else 0
                for k in range(self.Data.Mj[j]):
                    Sol.x[(t,j,k)] = 1 if sol[t][j] == k else 0
        return Sol


if __name__ == "__main__":
    from MaximumCover_Data import *
    Data = Test()
    G = GreedyOne(Data)
    G = GreedyOne(Data, {'searchMode':'h'})