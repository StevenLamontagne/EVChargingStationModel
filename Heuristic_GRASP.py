import json
import numpy as np
import pandas as pd
from time import time
from random import shuffle, choice



class GRASP():
    def __init__(self, Data, params):
        tic=time()
        self.Data = Data

        #Set to True to have new solutions printed as they are found
        self.printDetails = False
        #Time limit (in seconds), after which algorithm will no longer
        #search for new solutions
        self.timelimit = 7200
        #Whether search for new solutions should be done in myopic ('m') or hyperoptic ('h') mode
        self.searchMode = 'm'
        #Decimal in [0,1], determines how close to optimal greedy
        #choice an option must be to be considered
        self.alpha = 0.85
        #Maximum number of solutions in solution pools
        #(before terminating algorithm)
        self.poolsize = 300
        #Maximum number of solutions to remove at build phase
        #(before terminating algorithm)
        self.maxremoved = 500
        #Threshold value for terminating local search iterations
        self.cutoff = 0.001

        if params:
            self.__dict__.update(params)

        self.SolutionsRemoved = {option:0 for option in ['Build','Search','RelativeIncrease']}
        self.SolvingTimes = {
            'Pool':0,
            'Filter':0,
            'Search':0}
        
        x_init = np.zeros((self.Data.T, self.Data.M), dtype = int)
        for t in range(self.Data.T):
            for j in range(self.Data.M):
                x_init[t][j] = self.Data.x0[j]
        #Candidate pools keep solution and budget info, to avoid repeatedly recalculating
        self.CandidatePool_init = list([ [x_init, self.CalcBudget(x_init)] ])      
        self.CandidatePool_final = []
        
        self.BestSolutionValue = 0
        self.BestSolution = None
        self.MaximumRelativeImprovement = 1
        self.CandidatePool_inter = []
        counter = 0
        

        while (len(self.CandidatePool_init) > 0) and (len(self.CandidatePool_final) < self.poolsize) and (self.SolutionsRemoved['Build'] < self.maxremoved) and (time() - tic < self.timelimit):
            Sol = choice(self.CandidatePool_init)   
            Sol = self.BuildSolutionPool(Sol)
            if not self.FilterSolution(self.CandidatePool_final, Sol[0], phase = 'Build'):
                continue
            else:
                self.CandidatePool_final.append(Sol[0])
            Sol = Sol[0]
            quality = self.Data.SolutionQuality(Sol)
            tic_Search = time()
            #If 10% of max poolsize have been examined, filter any solution that is unlikely to be better than incumbent
            if (self.MaximumRelativeImprovement*quality < self.BestSolutionValue) and (counter >= 0.1*self.poolsize):
                self.SolutionsRemoved['RelativeIncrease'] += 1
                continue

            newSolution = self.Search(Sol, 'first', False)
            newquality = self.Data.SolutionQuality(newSolution)
            toc_Search = time()
            self.SolvingTimes['Search'] += toc_Search - tic_Search
            
            relImprovement = (newquality - quality)/quality
            self.MaximumRelativeImprovement = max(self.MaximumRelativeImprovement, relImprovement)
            if newquality > self.BestSolutionValue:
                self.BestSolution = newSolution
                self.BestSolutionValue = newquality
                if self.printDetails:
                    print("Local Search found solution of quality {}".format(quality))
            counter += 1

            
        toc = time()
        self.HeuristicSolvingTime = toc-tic



    #First phase of GRASP algorithm: Find promising solutions via  greedy method
    def BuildSolutionPool(self, Sol):
        if (len(self.CandidatePool_final) >= self.poolsize) or (self.SolutionsRemoved['Build'] >= self.maxremoved):
            return
        tic_Pool = time()
        x_star = np.copy(Sol[0])
        B_star = np.copy(Sol[1])
    
        TotalEVs_star = np.zeros(self.Data.T)

        Covering = [pd.DataFrame() for _ in range(self.Data.T)]
        for t in range(self.Data.T):
            Covering[t]['Home'] = self.Data.Covering[t]['Home']
            for j in range(self.Data.M):
                if x_star[t][j] > 0:
                    Covering[t][str(j)] = self.Data.Covering[t][(j, x_star[t][j])]
            #Covering[t] = pd.DataFrame({'Covering':Covering[t].any(axis=1)})
            #TotalEVs_star[t] = (Covering[t]['Covering']).dot(self.Data.Populations[t])
            TotalEVs_star[t] = (Covering[t].any(axis=1)).dot(self.Data.Populations[t])
        if self.searchMode == 'h':
            for t in range(self.Data.T-1, 0, -1):
                TotalEVs_star[t-1] += TotalEVs_star[t]     
        
        for t in range(self.Data.T):
            while True:
                Cost = np.ones(self.Data.M)  
                TotalNewEVs, Cost = self.FindStationTotals(t, x_star, B_star, Cost, Covering)
                    
                new = max(TotalNewEVs)
                if new > TotalEVs_star[t]:
                    threshold = TotalEVs_star[t] + (new - TotalEVs_star[t])*self.alpha
                    jPool = [j for j in range(self.Data.M) if (TotalNewEVs[j] >= threshold and TotalNewEVs[j]>0)]
                    j = choice(jPool)
                    B_star[t] -= Cost[j]
                    for t1 in range(t, self.Data.T):                       
                        Covering[t1][str(j)] = self.Data.Covering[t1][(j, x_star[t1][j]+1)]
                        #Covering[t1] = pd.DataFrame({'Covering':Covering[t1].any(axis=1)})
                        x_star[t1][j] += 1
                    TotalEVs_star[t] = new

                else:
                    break    
        toc_Pool = time()
        self.SolvingTimes['Pool'] += toc_Pool - tic_Pool
        return [x_star,B_star]

    def FindStationTotals(self, t, x_star, B_star, Cost, Covering):
        TotalNewEVs = np.zeros(self.Data.M)
        for j in range(self.Data.M):
            mj, Cost = self.MaxNewOutlets(t, x_star, B_star, Cost, j)
            if mj <= 0:
                TotalNewEVs[j] = 0
                continue
            if self.searchMode == 'm':
                Covering[t]['Station'] = self.Data.Covering[t][(j, x_star[t][j]+1)]
                TotalNewEVs[j] += (Covering[t].any(axis=1)).dot(self.Data.Populations[t])
                del Covering[t]['Station']
            elif self.searchMode == 'h':
                for t1 in range(t, self.Data.T):
                    Covering[t1]['Station'] = self.Data.Covering[t1][(j, x_star[t1][j]+1)]
                    TotalNewEVs[j] += (Covering[t1].any(axis=1)).dot(self.Data.Populations[t1])  
                    del Covering[t1]['Station']


        return TotalNewEVs, Cost

    def MaxNewOutlets(self, t, x_star, B_star, Cost, j):
        if x_star[t][j] >= self.Data.Mj[j]-1:
            mj = 0
            
        elif x_star[t][j] > 0:
            nStations = int(B_star[t]/self.Data.c[t][j])
            Cost[j] = self.Data.c[t][j]
            mj = min(self.Data.Mj[j]-1 - x_star[t][j], nStations)
                
        else:
            nStations = int((B_star[t]-self.Data.f[t][j])/self.Data.c[t][j])
            Cost[j] = self.Data.f[t][j]+self.Data.c[t][j]
            mj = min(self.Data.Mj[j]-1 - x_star[t][j], nStations)
        return mj, Cost

    #Second phase of GRASP algorithm: local search to improve given solution
    def Search(self, Solution,  mode = 'best', printDetails = False):
        if type(Solution) == dict:
            self.Solution=np.zeros((self.Data.T, self.Data.M))
            for t in range(self.Data.T):
                for j in range(self.Data.M):
                    self.Solution[t][j] = Solution[(t,j)]
        else:
            self.Solution = Solution
            
        if not self.FilterSolution(self.CandidatePool_inter,self.Solution, phase = 'Search'):
            return self.Solution
        else:
            self.CandidatePool_inter.append(self.Solution)
            
        self.TotalEVs = self.Data.SolutionQuality(self.Solution)
        self.mode = mode   
        self.Budget = self.CalcBudget(self.Solution)
        
        if printDetails:
            print("Initial solution quality: ", self.TotalEVs)
        if mode == 'best':
            while True:
                foundImprovementTotal=False
                self.IterStart = float(self.TotalEVs)
                for t in range(self.Data.T):
                    while True:
                        foundImprovementYear=False
                        Solutions=[]
                        Values=[]
                        options=[]                
                        for j in range(self.Data.M):
                            
                            if self.Budget[t] >0:
                                newSolution = self.Add(t,j)
                                quality = self.Data.SolutionQuality(newSolution)
                                budget = self.CalcBudget(newSolution)
                                if min(budget)>=0 and quality > self.TotalEVs:
                                    Solutions.append(newSolution)
                                    Values.append(quality)
                                    options.append("add ({}, year {})".format(j,t))                               
                            if self.Solution[t][j] <= 0:
                                continue
                            
                            for j1 in range(j+1, self.Data.M):                        
                                newSolution = self.Split(t, j, j1)
                                quality = self.Data.SolutionQuality(newSolution)
                                if quality > self.TotalEVs:
                                    Solutions.append(newSolution)
                                    Values.append(quality)
                                    options.append("split ({},{}, year {})".format(j,j1,t)) 
                            for j1 in range(self.Data.M):
                                if j1 != j:
                                    newSolution = self.Transfer(t, j, j1)
                                    quality = self.Data.SolutionQuality(newSolution)
                                    if quality > self.TotalEVs:
                                        Solutions.append(newSolution)
                                        Values.append(quality)
                                        options.append("transfer ({},{}, year {})".format(j,j1,t))                                         
                        if len(Solutions) > 0:
                            jMax = np.argmax(Values)
                            if printDetails:
                                print("New solution found of quality: ",Values[jMax])
                                print("Solution found by performing {} operation".format(options[jMax]))
                                print('\n')
                            self.Moves[options[jMax].split()[0]] += 1
                            self.TotalEVs = Values[jMax]
                            self.Improvements_Values.append(Values[jMax])
                            self.Solution = Solutions[jMax]
                            self.Budget = self.CalcBudget(Solutions[jMax])
                            if min(self.Budget) < 0:
                                print(self.Budget)
                                raise Exception("Budget is negative, solution infeasible")
                            foundImprovementYear = True
                            foundImprovementTotal = True
                        if not foundImprovementYear or self.Cutoff_Best():
                            break
                if not foundImprovementTotal or self.Cutoff_Year():
                    break

        elif mode == 'first':
            while True:
                foundImprovementTotal=False
                self.IterStart = float(self.TotalEVs)
                for t in range(self.Data.T):
                    while True:
                        foundImprovementYear=False
                        self.initialValue = float(self.TotalEVs)
                        Solutions=[]
                        Values=[]
                        options=[]                
                        for j in range(self.Data.M):
                            
                            if self.Budget[t] >0:
                                newSolution = self.Add(t,j)
                                quality = self.Data.SolutionQuality(newSolution)
                                budget = self.CalcBudget(newSolution)
                                if min(budget)>=0 and quality > self.TotalEVs:
                                    self.Solution = newSolution
                                    if not self.FilterSolution(self.CandidatePool_inter,self.Solution, phase = 'Search'):
                                        return self.Solution
                                    else:
                                        self.CandidatePool_inter.append(self.Solution)
                                    self.TotalEVs = quality
                                    self.Budget = budget
                                    foundImprovementYear=True
                                    if printDetails:
                                        print("New solution found of quality: ",quality)
                                        print("Solution found by performing {} operation".format("add ({}, year {})".format(j,t)))                                
                            if self.Solution[t][j] <= 0:
                                continue
                            
                            for j1 in range(j+1, self.Data.M):                        
                                newSolution = self.Split(t, j, j1)
                                quality = self.Data.SolutionQuality(newSolution)
                                if quality > self.TotalEVs:
                                    self.Solution = newSolution
                                    if not self.FilterSolution(self.CandidatePool_inter,self.Solution, phase = 'Search'):
                                        return self.Solution
                                    else:
                                        self.CandidatePool_inter.append(self.Solution)
                                    self.TotalEVs = quality
                                    self.Budget = self.CalcBudget(newSolution)
                                    foundImprovementYear=True
                                    if printDetails:
                                        print("New solution found of quality: ",quality)
                                        print("Solution found by performing {} operation".format("split ({},{}, year {})".format(j,j1,t)))  
                            for j1 in range(self.Data.M):
                                if j1 != j:
                                    newSolution = self.Transfer(t, j, j1)
                                    quality = self.Data.SolutionQuality(newSolution)
                                    if quality > self.TotalEVs:
                                        self.Solution = newSolution
                                        if not self.FilterSolution(self.CandidatePool_inter,self.Solution, phase = 'Search'):
                                            return self.Solution
                                        else:
                                            self.CandidatePool_inter.append(self.Solution)
                                        self.TotalEVs = quality
                                        self.Budget = self.CalcBudget(newSolution)
                                        foundImprovementYear=True                           
                                        if printDetails:
                                            print("New solution found of quality: ",quality)
                                            print("Solution found by performing {} operation".format("transfer ({},{}, year {})".format(j,j1,t)))                                          
                        if not foundImprovementYear or self.Cutoff_First():
                            break
                if not foundImprovementTotal or self.Cutoff_Year():
                    break
        return self.Solution

    #Checks if a given solution x_star has already been found
    def FilterSolution(self, Pool, x_star, phase = 'Build'):
        tic_Filter = time()
        for x in Pool:
            if (x == x_star).all():
                self.SolutionsRemoved[phase] += 1
                toc_Filter = time()
                self.SolvingTimes['Filter'] += toc_Filter - tic_Filter
                return False
        toc_Filter = time()
        self.SolvingTimes['Filter'] += toc_Filter - tic_Filter
        return True

    #Checks if best improvement in each iteration through stations are
    #better than a threshold value
    def Cutoff_Best(self):
        if self.cutoff is None:
            return False
        frac = (self.Improvements_Values[-1]-self.Improvements_Values[-2])/self.Improvements_Values[-2]
        return frac <= self.cutoff

    #Checks if improvements in each iterations through stations are better
    #than a threshold value
    def Cutoff_First(self):
        if self.cutoff is None:
            return False
        frac = (self.TotalEVs - self.initialValue)/self.initialValue
        return frac <= self.cutoff 

    #Checks if the total improvements throughout the entire year are
    #better than a threshold value
    def Cutoff_Year(self):
        if self.cutoff is None:
            return False
        frac = (self.TotalEVs - self.IterStart)/self.IterStart
        return frac <= self.cutoff  


    #Calculates remaining budget for each year given a solution
    def CalcBudget(self, Solution):
        Budget = list(self.Data.B) 
        for j in range(self.Data.M):
            nOutlets = Solution[0][j]
            if nOutlets > 0:
                Budget[0] -= (nOutlets - self.Data.x0[j]) * self.Data.c[0][j]
                if self.Data.x0[j] == 0:
                    Budget[0] -= self.Data.f[0][j]
            for t in range(1,self.Data.T):
                nOutlets = Solution[t][j]
                if nOutlets > 0:
                    Budget[t] -= (nOutlets - Solution[t-1][j]) * self.Data.c[t][j]
                    if Solution[t-1][j] == 0:
                        Budget[t] -= self.Data.f[t][j]
        return Budget

    #Calculates the amount spent on station jBar starting years tBar, ...,  T
    def CalcSpending(self, Solution, tBar, jBar):
        money = [0 for _ in range(self.Data.T)]
        if tBar == 0:
            previous = self.Data.x0[jBar]
        else:
            previous = Solution[tBar-1][jBar]
        if (previous == 0) and (Solution[tBar][jBar] > 0):
            money[tBar] += self.Data.f[tBar][jBar]
        money[tBar] += (Solution[tBar][jBar] - previous) * self.Data.c[tBar][jBar]
        for t1 in range(tBar+1, self.Data.T):
            money[t1] += (Solution[t1][jBar] - Solution[t1-1][jBar])* self.Data.c[t1][jBar]
        return np.array(money)
            
    #Returns new solution from transfering amount spent on station j1 to station j2
    #(for all years tBar, ...., T)
    def Transfer(self, tBar, j1, j2):
        newSolution = np.copy(self.Solution)
        money = self.CalcSpending(self.Solution,tBar,j1)
        if tBar == 0:
            previous_j1 = self.Data.x0[j1]
            previous_j2 = self.Data.x0[j2]
        else:
            previous_j1 = self.Solution[tBar-1][j1]
            previous_j2 = self.Solution[tBar-1][j2]
        #Check if we need to open station j2, if so pay the cost
        if (previous_j2 == 0) and (self.Solution[tBar][j2] ==0):
            money[tBar] -= self.Data.f[tBar][j2]
        if money[tBar] <= 0:
            return None
        
        for t1 in range(tBar, self.Data.T):
            newOutlets_j2 = int( money[t1]/self.Data.c[t1][j2] )
            if (t1 == tBar) and (newOutlets_j2 == 0):
                return None        
            new_j2 = min(previous_j2 + newOutlets_j2, self.Data.Mj[j2]-1)
            newOutlets_j2 = new_j2 - previous_j2
            newSolution[t1][j2] = new_j2
            previous_j2 = new_j2

            money[t1] -= newOutlets_j2 * self.Data.c[t1][j2] 
            if money[t1] > 0 :
                #Check if we need to open station j1, if so pay the cost
                if (previous_j1 ==0) and (money[t1] >= self.Data.c[t1][j1] +  self.Data.f[t1][j1]):
                    money[t1] -= self.Data.f[t1][j1]
                    newOutlets_j1 = int( money[t1]/self.Data.c[t1][j1] )
                    new_j1 = min(previous_j1 + newOutlets_j1, self.Data.Mj[j1]-1)
                    newOutlets_j1 = new_j1 - previous_j1
                    newSolution[t1][j1] = new_j1
                elif (previous_j1 > 0) and (money[t1] >= self.Data.c[t1][j1]):
                    newOutlets_j1 = int( money[t1]/self.Data.c[t1][j1] )
                    new_j1 = min(previous_j1 + newOutlets_j1, self.Data.Mj[j1]-1)
                    newOutlets_j1 = new_j1 - previous_j1
                    newSolution[t1][j1] = new_j1
                else:
                    newSolution[t1][j1] = previous_j1           
            else:        
                newSolution[t1][j1] = previous_j1

        return newSolution

    #Returns solution from adding an outlet at station j1
    #(for all years tBar, ...., T)
    def Add(self, tBar, j1):
        newSolution = np.copy(self.Solution)
        for t1 in range(tBar, self.Data.T):
            newSolution[t1][j1] = min(newSolution[t1][j1] +1, self.Data.Mj[j1]-1)
        return newSolution


    #Returns new solution from evenly splitting amount spent on stations j1 and j2
    #(for all years tBar, ...., T)
    def Split(self, t, j1, j2):
        newSolution = np.copy(self.Solution)
        money = self.CalcSpending(self.Solution,t,j1)
        money += self.CalcSpending(self.Solution,t,j2)
        if t == 0:
            previous_j1 = self.Data.x0[j1]
            previous_j2 = self.Data.x0[j2]
        else:
            previous_j1 = self.Solution[t-1][j1]
            previous_j2 = self.Solution[t-1][j2]
        #Check if we need to open station j1, if so pay the money
        if (previous_j1 == 0):
            money[t] -= self.Data.f[t][j1]
        #Check if we need to open station j2, if so pay the money
        if (previous_j2 == 0):
            money[t] -= self.Data.f[t][j2]
        if money[t] <= 0:
            return None
        money = 0.5*money
        for t1 in range(t, self.Data.T):
            if money[t1] <= 0:
                newSolution[t1][j1] = previous_j1
                newSolution[t1][j2] = previous_j2               
                continue
            newOutlets_j1 = int( money[t1]/self.Data.c[t1][j1] )
            newOutlets_j2 = int( money[t1]/self.Data.c[t1][j2] )
            new_j1 = min( int(previous_j1 + newOutlets_j1) , self.Data.Mj[j1]-1)
            new_j2 = min( int(previous_j2 + newOutlets_j2) , self.Data.Mj[j2]-1)
            if new_j1 == 0 or new_j2 == 0:
                return None
            newSolution[t1][j1] = int(new_j1)
            newSolution[t1][j2] = int(new_j2)
            previous_j1 = int(new_j1)
            previous_j2 = int(new_j2)

        return newSolution

    #Summarises information about the algorithm and transfers to json file
    def to_json(self):
        temp = {
            'alpha': self.alpha,
            'poolsize':self.poolsize,
            'maxremoved': self.maxremoved,
            'SolutionsRemoved': self.SolutionsRemoved,
            'MaxRelativeIncrease': self.MaximumRelativeImprovement,
            'SolvingTimes': self.SolvingTimes,
            'TotalSolvingTime': self.HeuristicSolvingTime,
            'SolutionValue': self.BestSolutionValue,
            'Solution': self.BestSolution.tolist()}
        return temp


if __name__ == "__main__":
    from MaximumCover_Data import *
    Data = Test()
    G = GRASP(Data, {'poolsize':100, 'searchMode':'m'})
    G = GRASP(Data, {'poolsize':100, 'searchMode':'m'})