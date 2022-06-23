import json
import numpy as np
import pandas as pd
from time import time
from random import shuffle, choice


from copy import deepcopy

#Class for storing information about a GRASP candidate solution
class GRASPSolution():
    def __init__(self, Data):
        self.Data = Data
        self.Budget = np.copy(self.Data.B)
        self.Solution = np.zeros(shape = (self.Data.T, self.Data.M), dtype = int)
        self.Covering = [pd.DataFrame() for _ in range(self.Data.T)]
        self.SolutionQuality = np.zeros(self.Data.T)
    
    #Setting the solution to new, and update information accordingly
    def SetSolution(self, new, mode):
        self.Solution = new[0]
        cols = [[(j, new[0][t][j]) for j in range(self.Data.M) if new[0][t][j] > 0] for t in range(self.Data.T)]
        self.Covering = [self.Data.ReducedCovering[t][cols[t]].copy() for t in range(self.Data.T)]
        self.Budget = new[1]
        self.UpdateSolutionQuality(mode)
    
    #Recalculate the solution quality when the solution changes
    def UpdateSolutionQuality(self, mode = 'm'):
        self.SolutionQuality = [self.Covering[t].any(axis=1).dot(self.Data.ReducedPopulations[t])for t in range(self.Data.T)]
        if mode == 'h':
            for t in range(self.Data.T-1, 0, -1):
                self.SolutionQuality[t-1] += self.SolutionQuality[t]              

    #Ensuring comparable formats for final solution quality
    ##May be different to sum(self.SolutionQuality) depending on search mode
    def FinalSolutionQuality(self):
        return self.Data.SolutionQuality(self.Solution)

    #Create a copy of the GRASP solution object (for local search manipulation)
    def copy(self):
        new = GRASPSolution(self.Data)
        new.Budget = np.copy(self.Budget)
        new.Solution = deepcopy(self.Solution)
        new.Covering = [self.Covering[t].copy() for t in range(self.Data.T)]
        new.SolutionQuality = np.copy(self.SolutionQuality)
        return new
    
    #Reconstructs the covering attribute from solution
    def RebuildCovering(self):
        cols = [[(j, self.Solution[t][j]) for j in range(self.Data.M) if self.Solution[t][j] > 0] for t in range(self.Data.T)]
        self.Covering = [self.Data.ReducedCovering[t][cols[t]].copy() for t in range(self.Data.T)]



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
        self.InitialSolution = GRASPSolution(Data)
        self.InitialSolution.SetSolution([x_init, self.CalcBudget(x_init)], mode = self.searchMode)     
        self.CandidatePool_final = []
        
        self.BestSolutionValue = 0
        self.BestSolution = None
        self.MaximumRelativeImprovement = 1
        self.CandidatePool_inter = []
        counter = 0
        
        while (len(self.CandidatePool_final) < self.poolsize) and (self.SolutionsRemoved['Build'] < self.maxremoved) and (time() - tic < self.timelimit):
            Sol = self.BuildSolutionPool(self.InitialSolution.copy())
            if not self.FilterSolution(self.CandidatePool_final, Sol.Solution, phase = 'Build'):
                continue
            else:
                self.CandidatePool_final.append(Sol.Solution)

            quality = np.sum(Sol.SolutionQuality)
            tic_Search = time()
            #If 10% of max poolsize have been examined, filter any solution that is unlikely to be better than incumbent
            if (self.MaximumRelativeImprovement*quality < self.BestSolutionValue) and (counter >= 0.1*self.poolsize):
                self.SolutionsRemoved['RelativeIncrease'] += 1
                continue

            newSolution = self.Search(Sol, 'first', False)
            newquality = np.sum(Sol.SolutionQuality)(newSolution)
            toc_Search = time()
            self.SolvingTimes['Search'] += toc_Search - tic_Search
            
            relImprovement = (newquality - quality)/quality
            self.MaximumRelativeImprovement = max(self.MaximumRelativeImprovement, relImprovement)
            if newquality > self.BestSolutionValue:
                self.BestSolution = Sol
                self.BestSolutionValue = newquality
                if self.printDetails:
                    print("Local Search found solution of quality {}".format(quality))
            counter += 1

            
        toc = time()
        self.HeuristicSolvingTime = toc-tic
        if self.printDetails:
            print("Final solution quality:", np.sum(self.BestSolution.FinalSolutionQuality()))


    #First phase of GRASP algorithm: Find promising solutions via  greedy method
    def BuildSolutionPool(self, Sol):
        if (len(self.CandidatePool_final) >= self.poolsize) or (self.SolutionsRemoved['Build'] >= self.maxremoved):
            return
        tic_Pool = time()
    
        TotalEVs_star = np.copy(Sol.SolutionQuality)

        if self.searchMode == 'h':
            for t in range(self.Data.T-1, 0, -1):
                TotalEVs_star[t-1] += TotalEVs_star[t]     
        
        for t in range(self.Data.T):
            while True: 
                TotalNewEVs, Cost = self.FindStationTotals(t, Sol)
                    
                new = max(TotalNewEVs)
                if new > TotalEVs_star[t]:
                    threshold = TotalEVs_star[t] + (new - TotalEVs_star[t])*self.alpha
                    jPool = [j for j in range(self.Data.M) if (TotalNewEVs[j] >= threshold and TotalNewEVs[j]>0)]
                    j = choice(jPool)
                    Sol.Budget[t] -= Cost[j]
                    for t1 in range(t, self.Data.T):  
                        new = Sol.Solution[t1][j]+1 
                        Sol.Covering[t1][(j, new)] = self.Data.Covering[t1][(j, new)]
                        ###
                        # Not necessary, but I found it faster for large instances
                        Sol.Covering[t1] = pd.DataFrame({'Covering':Sol.Covering[t1].any(axis=1)})
                        ###
                        Sol.Solution[t1][j] += 1
                    Sol.UpdateSolutionQuality(mode = self.searchMode)
                    TotalEVs_star[t] = Sol.SolutionQuality[t]

                else:
                    break    
        toc_Pool = time()
        self.SolvingTimes['Pool'] += toc_Pool - tic_Pool
        Sol.RebuildCovering()
        return Sol

    #Calculate the improvements to the solution quality when adding one outlet to each station
    def FindStationTotals(self, t, Sol):
        Cost = np.zeros(self.Data.M) 
        TotalNewEVs = np.zeros(self.Data.M)
        for j in range(self.Data.M):
            mj, cost = self.MaxNewOutlets(t, j, Sol)
            if mj <= 0:
                TotalNewEVs[j] = 0
                continue
            Cost[j] = cost
            if self.searchMode == 'm':              
                Sol.Covering[t]['Station'] = self.Data.ReducedCovering[t][(j, Sol.Solution[t][j]+1)]
                TotalNewEVs[j] += (Sol.Covering[t].any(axis=1)).dot(self.Data.ReducedPopulations[t])
                del Sol.Covering[t]['Station']
            elif self.searchMode == 'h':
                for t1 in range(t, self.Data.T):
                    Sol.Covering[t1]['Station'] = self.Data.ReducedCovering[t1][(j, Sol.Solution[t1][j]+1)]
                    TotalNewEVs[j] += (Sol.Covering[t1].any(axis=1)).dot(self.Data.ReducedPopulations[t1])  
                    del Sol.Covering[t1]['Station']
        return TotalNewEVs, Cost

    #Calculate the maximum number of *new* outlets that can be installed at outlet j in solution Sol in year t
    def MaxNewOutlets(self, t, j, Sol):
        if Sol.Solution[t][j] >= self.Data.Mj[j]-1:
            cost = 0
            mj = 0
               
        elif Sol.Solution[t][j] > 0:
            nStations = int(Sol.Budget[t]/self.Data.c[t][j])
            cost = self.Data.c[t][j]
            mj = min(self.Data.Mj[j]-1 - Sol.Solution[t][j], nStations)
                
        else:
            nStations = int((Sol.Budget[t]-self.Data.f[t][j])/self.Data.c[t][j])
            cost = self.Data.f[t][j]+self.Data.c[t][j]
            mj = min(self.Data.Mj[j]-1 - Sol.Solution[t][j], nStations)
        return mj, cost

    #Second phase of GRASP algorithm: local search to improve given solution
    def Search(self, Sol,  mode = 'first', printDetails = False):
        if not self.FilterSolution(self.CandidatePool_inter, Sol.Solution, phase = 'Search'):
            return Sol
        else:
            self.CandidatePool_inter.append(Sol.Solution)
            
        TotalEVs = np.sum(Sol.SolutionQuality) 
        
        if printDetails:
            print("Initial solution quality: ", self.TotalEVs)
        ###########################################
        ##Note: Not updated to use GRASPSolution. Previous method is left here for reference.
        ###########################################
        # if mode == 'best':
        #     while True:
        #         foundImprovementTotal=False
        #         self.IterStart = float(self.TotalEVs)
        #         for t in range(self.Data.T):
        #             while True:
        #                 foundImprovementYear=False
        #                 Solutions=[]
        #                 Values=[]
        #                 options=[]                
        #                 for j in range(self.Data.M):
                            
        #                     if self.Budget[t] >0:
        #                         newSolution = self.Add(t,j)
        #                         quality = self.Data.SolutionQuality(newSolution)
        #                         budget = self.CalcBudget(newSolution)
        #                         if min(budget)>=0 and quality > self.TotalEVs:
        #                             Solutions.append(newSolution)
        #                             Values.append(quality)
        #                             options.append("add ({}, year {})".format(j,t))                               
        #                     if self.Solution[t][j] <= 0:
        #                         continue
                            
        #                     for j1 in range(j+1, self.Data.M):                        
        #                         newSolution = self.Split(t, j, j1)
        #                         quality = self.Data.SolutionQuality(newSolution)
        #                         if quality > self.TotalEVs:
        #                             Solutions.append(newSolution)
        #                             Values.append(quality)
        #                             options.append("split ({},{}, year {})".format(j,j1,t)) 
        #                     for j1 in range(self.Data.M):
        #                         if j1 != j:
        #                             newSolution = self.Transfer(t, j, j1)
        #                             quality = self.Data.SolutionQuality(newSolution)
        #                             if quality > self.TotalEVs:
        #                                 Solutions.append(newSolution)
        #                                 Values.append(quality)
        #                                 options.append("transfer ({},{}, year {})".format(j,j1,t))                                         
        #                 if len(Solutions) > 0:
        #                     jMax = np.argmax(Values)
        #                     if printDetails:
        #                         print("New solution found of quality: ",Values[jMax])
        #                         print("Solution found by performing {} operation".format(options[jMax]))
        #                         print('\n')
        #                     self.Moves[options[jMax].split()[0]] += 1
        #                     self.TotalEVs = Values[jMax]
        #                     self.Improvements_Values.append(Values[jMax])
        #                     self.Solution = Solutions[jMax]
        #                     self.Budget = self.CalcBudget(Solutions[jMax])
        #                     if min(self.Budget) < 0:
        #                         print(self.Budget)
        #                         raise Exception("Budget is negative, solution infeasible")
        #                     foundImprovementYear = True
        #                     foundImprovementTotal = True
        #                 if not foundImprovementYear or self.Cutoff_Best():
        #                     break
        #         if not foundImprovementTotal or self.Cutoff_Year():
        #             break
        if mode == 'first':
            while True:
                foundImprovementTotal=False
                IterStart = float(TotalEVs)
                for t in range(self.Data.T):
                    while True:
                        foundImprovementYear=False
                        initialValue = float(TotalEVs)
                        Solutions=[]
                        Values=[]
                        options=[]                
                        for j in range(self.Data.M):
                            if (Sol.Budget[t] > 0) and (self.CheckCost(t, j, Sol) <= Sol.Budget[t]):
                                newSolution = self.Add(t,j, Sol)
                                quality = np.sum(newSolution.SolutionQuality)
                                if quality > TotalEVs:
                                    Sol = newSolution.copy()
                                    if not self.FilterSolution(self.CandidatePool_inter, Sol.Solution, phase = 'Search'):
                                        return Sol
                                    else:
                                        self.CandidatePool_inter.append(Sol.Solution)
                                    TotalEVs = quality
                                    foundImprovementYear=True
                                    if printDetails:
                                        print("New solution found of quality: ",quality)
                                        print("Solution found by performing {} operation".format("add ({}, year {})".format(j,t)))                                                     
                            for j1 in range(self.Data.M):
                                #Since Split is symmetric, only necessary to do either Split(j1,j2) or Split(j2,j1) 
                                if ((Sol.Solution[t][j] > 0 or Sol.Solution[t][j1] > 0) and (j1 > j)):  
                                    newSolution = self.Split(t, j, j1, Sol)
                                    if newSolution is not None:
                                        quality = np.sum(newSolution.SolutionQuality)
                                        if quality > TotalEVs:
                                            Sol = newSolution.copy()
                                            if not self.FilterSolution(self.CandidatePool_inter, Sol.Solution, phase = 'Search'):
                                                return Sol
                                            else:
                                                self.CandidatePool_inter.append(Sol.Solution)
                                            TotalEVs = quality
                                            foundImprovementYear=True
                                            if printDetails:
                                                print("New solution found of quality: ",quality)
                                                print("Solution found by performing {} operation".format("split ({},{}, year {})".format(j,j1,t)))  
                                #Only necessary to try Transfer if any money was spent on station j in year t
                                if (j1 != j) and ((t == 0 and Sol.Solution[t][j] - self.Data.x0[j] > 0) or (t > 0 and Sol.Solution[t][j] - Sol.Solution[t-1][j] > 0)):
                                    newSolution = self.Transfer(t, j, j1, Sol)
                                    if newSolution is not None:
                                        quality = np.sum(newSolution.SolutionQuality)
                                        if quality > TotalEVs:
                                            Sol = newSolution.copy()
                                            if not self.FilterSolution(self.CandidatePool_inter, Sol.Solution, phase = 'Search'):
                                                return Sol
                                            else:
                                                self.CandidatePool_inter.append(Sol.Solution)
                                            TotalEVs = quality
                                            foundImprovementYear=True                          
                                            if printDetails:
                                                print("New solution found of quality: ",quality)
                                                print("Solution found by performing {} operation".format("transfer ({},{}, year {})".format(j,j1,t)))                                          
                        if not foundImprovementYear or self.Cutoff_First(TotalEVs, initialValue):
                            break
                if not foundImprovementTotal or self.Cutoff_Year(TotalEVs, IterStart):
                    break
        return Sol

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
    def Cutoff_First(self, TotalEVs, initialValue):
        if self.cutoff is None:
            return False
        frac = (TotalEVs - initialValue)/initialValue
        return frac <= self.cutoff 

    #Checks if the total improvements throughout the entire year are
    #better than a threshold value
    def Cutoff_Year(self, TotalEVs, IterStart):
        if self.cutoff is None:
            return False
        frac = (TotalEVs - IterStart)/IterStart
        return frac <= self.cutoff  

    #Calculates the cost to add an outlet at outlet j to Sol in year t
    def CheckCost(self, t, j, Sol):
        if Sol.Solution[t][j] > 0:
            cost = self.Data.c[t][j]
        else:
            cost = self.Data.c[t][j] + self.Data.f[t][j]
        return cost


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
    def Transfer(self, tBar, j1, j2, Sol):
        newSolution = Sol.copy()
        money = self.CalcSpending(newSolution.Solution,tBar,j1)
        if tBar == 0:
            previous_j1 = self.Data.x0[j1]
            previous_j2 = self.Data.x0[j2]
        else:
            previous_j1 = newSolution.Solution[tBar-1][j1]
            previous_j2 = newSolution.Solution[tBar-1][j2]

        #Check if we need to open station j2, if so pay the cost
        if (previous_j2 == 0) and (newSolution.Solution[tBar][j2] ==0):
            money[tBar] -= self.Data.f[tBar][j2]
        if money[tBar] <= 0:
            return None
        
        for t1 in range(tBar, self.Data.T):
            current_j1 = newSolution.Solution[t1][j1]
            current_j2 = newSolution.Solution[t1][j2]

            newOutlets_j2 = int( money[t1]/self.Data.c[t1][j2] )
            if (t1 == tBar) and (newOutlets_j2 == 0):
                return None        
            new_j2 = min(previous_j2 + newOutlets_j2, self.Data.Mj[j2]-1)
            newOutlets_j2 = new_j2 - previous_j2
            newSolution.Solution[t1][j2] = new_j2
            if current_j2 > 0:
                del newSolution.Covering[t1][(j2, current_j2)]
            newSolution.Covering[t1][(j2, new_j2)] = self.Data.ReducedCovering[t1][(j2, new_j2)]
            previous_j2 = int(new_j2)
            money[t1] -= newOutlets_j2 * self.Data.c[t1][j2] 

            #If there's leftover money, we spend it on station j1
            if money[t1] > 0 :
                #Check if we need to open station j1, if so pay the cost
                if (previous_j1 ==0) and (money[t1] >= self.Data.c[t1][j1] +  self.Data.f[t1][j1]):
                    money[t1] -= self.Data.f[t1][j1]
                    newOutlets_j1 = int( money[t1]/self.Data.c[t1][j1] )
                    new_j1 = min(previous_j1 + newOutlets_j1, self.Data.Mj[j1]-1)
                    newOutlets_j1 = new_j1 - previous_j1
                    money[t1] -= newOutlets_j1*self.Data.c[t1][j1]
                    newSolution.Solution[t1][j1] = new_j1
                    del newSolution.Covering[t1][(j1, current_j1)]
                    newSolution.Covering[t1][(j1, new_j1)] = self.Data.ReducedCovering[t1][(j1, new_j1)]
                elif (previous_j1 > 0) and (money[t1] >= self.Data.c[t1][j1]):
                    newOutlets_j1 = int( money[t1]/self.Data.c[t1][j1] )
                    new_j1 = min(previous_j1 + newOutlets_j1, self.Data.Mj[j1]-1)
                    newOutlets_j1 = new_j1 - previous_j1
                    money[t1] -= newOutlets_j1*self.Data.c[t1][j1]
                    newSolution.Solution[t1][j1] = int(new_j1)
                    del newSolution.Covering[t1][(j1, current_j1)]
                    newSolution.Covering[t1][(j1, new_j1)] = self.Data.ReducedCovering[t1][(j1, new_j1)]
                else:
                    new_j1 = int(previous_j1)
                    newSolution.Solution[t1][j1] = previous_j1
                    del newSolution.Covering[t1][(j1, current_j1)]
                    if previous_j1 > 0:
                        newSolution.Covering[t1][(j1, new_j1)] = self.Data.ReducedCovering[t1][(j1, previous_j1)]  
                previous_j1 = int(new_j1)        
            else:        
                newSolution.Solution[t1][j1] = previous_j1
                del newSolution.Covering[t1][(j1, current_j1)]
                if previous_j1 > 0:
                    newSolution.Covering[t1][(j1, previous_j1)] = self.Data.ReducedCovering[t1][(j1, previous_j1)]        
        newSolution.Budget += np.array(money)
        newSolution.UpdateSolutionQuality(mode = self.searchMode)

        return newSolution

    #Returns solution from adding an outlet at station j1
    #(for all years tBar, ...., T)
    def Add(self, tBar, j1, Sol):
        newSolution = Sol.copy()
        for t1 in range(tBar, self.Data.T):
            current = int(newSolution.Solution[t1][j1])
            new = min(current +1, self.Data.Mj[j1]-1)
            if current > 0:
                del newSolution.Covering[t1][(j1, current)]
            newSolution.Solution[t1][j1] = new
            newSolution.Covering[t1][(j1, new)] = self.Data.ReducedCovering[t1][(j1, new)]
        cost = self.CheckCost(tBar, j1, Sol)
        newSolution.Budget[tBar] -= cost
        newSolution.UpdateSolutionQuality(mode = self.searchMode)
        return newSolution


    #Returns new solution from evenly splitting amount spent on stations j1 and j2
    #(for all years tBar, ...., T)
    def Split(self, tBar, j1, j2, Sol):
        newSolution = Sol.copy()
        money = self.CalcSpending(newSolution.Solution,tBar,j1)
        money += self.CalcSpending(newSolution.Solution,tBar,j2)
        if tBar == 0:
            previous_j1 = self.Data.x0[j1]
            previous_j2 = self.Data.x0[j2]
        else:
            previous_j1 = newSolution.Solution[tBar-1][j1]
            previous_j2 = newSolution.Solution[tBar-1][j2]
        #Check if we need to open station j1, if so pay the money
        if (previous_j1 == 0):
            money[tBar] -= self.Data.f[tBar][j1]
        #Check if we need to open station j2, if so pay the money
        if (previous_j2 == 0):
            money[tBar] -= self.Data.f[tBar][j2]
        if money[tBar] <= 0:
            return None

        for t1 in range(tBar, self.Data.T):
            current_j1 = int(newSolution.Solution[t1][j1])
            current_j2 = int(newSolution.Solution[t1][j2])

            newOutlets_j1 = int( money[t1]/(2*self.Data.c[t1][j1]) )
            newOutlets_j2 = int( money[t1]/(2*self.Data.c[t1][j2]) )
            new_j1 = min( int(previous_j1 + newOutlets_j1) , self.Data.Mj[j1]-1)
            new_j2 = min( int(previous_j2 + newOutlets_j2) , self.Data.Mj[j2]-1)
            if new_j1 == 0 or new_j2 == 0:
                return None

            money[t1] -= (new_j1 - previous_j1) *  self.Data.c[t1][j1]
            money[t1] -= (new_j2 - previous_j2) *  self.Data.c[t1][j2]

            if current_j1 > 0:
                del newSolution.Covering[t1][(j1, current_j1)]
            newSolution.Covering[t1][(j1, new_j1)] = self.Data.ReducedCovering[t1][(j1, new_j1)]
            newSolution.Solution[t1][j1] = int(new_j1)

            if current_j2 > 0:
                del newSolution.Covering[t1][(j2, current_j2)]
            newSolution.Covering[t1][(j2, new_j2)]  = self.Data.ReducedCovering[t1][(j2, new_j2)]
            newSolution.Solution[t1][j2] = int(new_j2)

            previous_j1 = int(new_j1)
            previous_j2 = int(new_j2)

        newSolution.Budget += np.array(money)
        newSolution.UpdateSolutionQuality(mode = self.searchMode)
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
            'Solution': self.BestSolution.Solution.tolist()}
        return temp
