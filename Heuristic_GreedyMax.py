import json
from math import atan, exp
import numpy as np
import pandas as pd
from time import time

from Utils import Solution


class GreedyMax():
    def __init__(self, Data, printDetails=False):
        tic=time()
        self.Data=Data
        
        #Remaining budget for incumbent solution
        self.B_star = list(self.Data.B)
        #Max outlets at each station
        self.Mj_star = list(self.Data.Mj)
         #Incumbent solution
        self.x_star = {(t,j): self.Data.x0[j] for j in range(self.Data.M) for t in range(self.Data.T)}
        #Objective value of incumbent solution
        self.TotalEVs_star=np.zeros(self.Data.T)
        


        for t in range(self.Data.T):
            if printDetails:
                print("t=", t)
                print('\n')
                
            self.Cost=np.ones(self.Data.M)
            
            self.Covering = pd.DataFrame()
            self.Covering['Home'] = self.Data.Covering[t]['Home']


            for j in range(self.Data.M):
                if self.x_star[(t,j)] > 0:
                    self.Covering[str(j)] = self.Data.Covering[t][(j, self.x_star[(t,j)])]

            if printDetails:
                print(self.Covering.head(3))
            
            self.TotalEVs_star[t] = (self.Covering.any(axis=1)).dot(self.Data.Populations[t])
            i=0
            #Iteratively find best station at which to maximise outlets until no budget remains in the year
            #or no improvements can be found
            while True:
                TotalNewEVs = self.FindStationTotals(t)
                if printDetails:
                    i += 1
                    print("Iteration:", i)
                    print("EV total:", self.TotalEVs_star[t])
                    print(TotalNewEVs)
                    print('\n')
                new=max(TotalNewEVs)

                if new > self.TotalEVs_star[t]:
                    j = ((TotalNewEVs-self.TotalEVs_star[t])/self.Cost).argmax()
                    mj = self.MaxNewOutlets(t,j)
                    mj = self.BestNewOutlets(t,j,mj, new)
                    if self.x_star[(t,j)] >0:
                        self.B_star[t] -= self.Data.c[t][j] * mj
                    else:
                        self.B_star[t] -= self.Data.c[t][j] * mj + self.Data.f[t][j]
                        
                    self.x_star[(t,j)] +=  mj
                    self.Covering[str(j)] = self.Data.Covering[t][(j, mj)]
                    self.TotalEVs_star[t]=new
                else:
                    break
                
            for j in range(self.Data.M):
                self.x_star[(t+1,j)] = self.x_star[(t,j)]



        toc=time()
        self.HeuristicSolvingTime=toc-tic
        self.HeuristicStats = {'EVs':self.TotalEVs_star, 'Budget Remaining': self.B_star, 'Runtime (seconds)':toc-tic}
        self.Solution = self.ConvertSolution(self.x_star)
        print("Heuristic solving time (seconds): ", toc-tic)
    
    def MaxNewOutlets(self,t,j):
        if self.x_star[(t,j)] >= self.Mj_star[j] -1:
            return 0
        
        elif self.x_star[(t,j)] > 0:
            nStations=int(self.B_star[t]/self.Data.c[t][j])
            self.Cost[j]=nStations*self.Data.c[t][j]
            return min(self.Mj_star[j]-1 - self.x_star[(t,j)], nStations)
            
        else:
            nStations=int((self.B_star[t]-self.Data.f[t][j])/self.Data.c[t][j])
            self.Cost[j]=self.Data.f[t][j]+nStations*self.Data.c[t][j]
            return min(self.Mj_star[j]-1 - self.x_star[(t,j)], nStations)

    #Calculate smallest number of new outlets which has maximum impact
    #(i.e. if 6 outlets has same objective value as 7 outlets, but more than 5 outlets, we return 6)
    def BestNewOutlets(self,t,j,mj,new):
        for mj_new in range(mj-1, 0, -1):
            self.Covering['Station'] = self.Data.Covering[t][(j, mj_new)]
            EVs = (self.Covering.any(axis=1)).dot(self.Data.Populations[t])
            del self.Covering['Station']
            if EVs < new:
                return mj
            else:
                mj = mj_new
        return mj

    #Calculate the objective function after maximising outlets at each station
    #Returns an array of size M
    def FindStationTotals(self,t):
        TotalNewEVs = np.zeros(self.Data.M)
        for j in range(self.Data.M):
            mj = self.MaxNewOutlets(t,j)
            if mj <= 0:
                TotalNewEVs[j]=0
                continue
            
            self.Covering['Station'] = self.Data.Covering[t][(j, mj)]
            TotalNewEVs[j] += (self.Covering.any(axis=1)).dot(self.Data.Populations[t])
            del self.Covering['Station']

        return TotalNewEVs

    #Convert solution to use as warmstart in maximum covering model
    def ConvertSolution(self,sol):
        Sol = Solution({'T': self.Data.T, 'M': self.Data.M, 'Mj':self.Data.Mj})
        for t in range(self.Data.T):
            for j in range(self.Data.M):
                Sol.y[(t,j)] = 1 if sol[(t,j)] >0 else 0
                for k in range(self.Data.Mj[j]):
                    Sol.x[(t,j,k)] = 1 if sol[(t,j)] == k else 0
        return Sol


