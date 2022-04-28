from docplex.mp.model import Model
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import mkdir
from time import time
import json

from MaximumCover_Data import Data_MaximumCover
from MaximumCover_Model import ChargingStationModel_MaximumCover
from Utils import Solution

from collections import OrderedDict


#################################################################################################################################
class RollingHorizon():

    def __init__(self, mdl, Data, params= None):
        self.mdl=mdl
        self.Data=Data
        #Default parameters
        #Filepath and name for saving log and results
        self.filepath = "Results"
        self.filename = "Test"
        #Solution components when using a fixed solution
        #(forces exact solution)
        self.SolutionX = None
        self.SolutionY = None
        #Index for appending to solutions
        self.testNumber= '-1'

        #Actual step size (in usual sense) is 1+ self.stepSize
        #due to Python ranges
        self.stepSize= 0

        #CPLEX parameters
        self.threads = 8
        self.timelimit = 900
        self.nodelimit = 9223372036800000000
        self.logdisplay = 4
        self.compressMemory = False


        self.Solution=Solution({'T': self.Data.T, 'M': self.Data.M, 'Mj':self.Data.Mj})
 
        #Solving time is stored as an array, to track each
        #year separately
        self.HeuristicTime=[]

        if params:
            self.__dict__.update(params)
 
        #Set CPLEX parameters
        self.mdl.context.cplex_parameters.threads = self.threads
        self.mdl.set_time_limit(self.timelimit)
        self.mdl.parameters.mip.display=self.logdisplay
        self.mdl.parameters.mip.limits.nodes= self.nodelimit
        self.mdl.parameters.emphasis.memory = int(self.compressMemory)
        self.Solve()

    #Iterate over each time step and find optimal
    #We only fix one year, regardless of time step
    def Solve(self):
        for t in range(self.Data.T):
            tic = time()
            tMin = t
            tMax = min(t + self.stepSize, self.Data.T - 1)
            self.mdl.clear()
            print("Starting Rolling Horizon for t=", t)
            self.RollingHorizon(tMin, tMax)
            print("Rolling Horizon complete for t=", t)
            print('\n')
            toc = time()
            self.HeuristicTime.append(toc-tic)

    #Restriction of maximum covering model for subset of years [tMin, tMax]
    def RollingHorizon(self, tMin, tMax):
        tic = time()
        tMax = tMax + 1
        #Decision variables
        ##x,u, and w variables have too many indices to use built-in Cplex variables, so tuples must be created to use as dictionary keys.
        id_x=[(t,j,k) for j in range(self.Data.M) for k in range(self.Data.Mj[j]) for t in range(tMin, tMax)]
        id_w=[(t,i,r) for i in range(self.Data.N) for r in range(self.Data.R[i])for t in range(tMin, tMax)]


        print("Creating variables")
        #Binary indicating if station j has k charging outlets in year t   
        self.mdl.x_vars=self.mdl.binary_var_dict(id_x,name='x')
        #Binary indicating if station j is open in year t
        self.mdl.y_vars=self.mdl.binary_var_matrix(self.Data.T,self.Data.M,name='y')
        #Continuous variables for covering  
        self.mdl.w_vars=self.mdl.continuous_var_dict(id_w, ub = 1,name='w')        
        print("Variables created")


        #Constraints
        print("Beginning constraints")
        #Covering constraints
        self.mdl.add_constraints(self.mdl.sum([self.Data.a[t][j][i][k][r]*self.mdl.x_vars[(t,j,k)] for j in range(self.Data.M) for k in range(self.Data.Mj[j])]) >= self.mdl.w_vars[(t,i,r)]
                            for i in range(self.Data.N)  for r in range(self.Data.R[i]) for t in range(tMin, tMax))
        if tMin == 0:
            #Budget, year 0
            self.mdl.add_constraint(
                self.mdl.sum(self.Data.c[0][j]*self.mdl.sum(k*self.mdl.x_vars[(0,j,k)] for k in range(self.Data.Mj[j])) for j in range(self.Data.M))
                -self.mdl.sum(self.Data.c[0][j]*self.Data.x0[j] for j in range(self.Data.M))
                +self.mdl.sum(self.Data.f[0][j]*self.mdl.y_vars[(0,j)] for j in range(self.Data.M))
                <=self.Data.B[0]+self.mdl.sum(self.Data.f[0][j]*self.Data.y0[j] for j in range(self.Data.M))
                )

            #Pay one-time cost
            self.mdl.add_constraints([self.mdl.sum(self.mdl.x_vars[(0,j,k)] for k in range(1,self.Data.Mj[j])) == self.mdl.y_vars[(0,j)] for j in range(self.Data.M)])
            
            ##Can't remove charging outlets, year 0
            self.mdl.add_constraints([self.mdl.sum(k*self.mdl.x_vars[(0,j,k)] for k in range(self.Data.Mj[j]))>=self.Data.x0[j] for j in range(self.Data.M)])

            ##Stations stay open, year 0
            self.mdl.add_constraints([self.mdl.y_vars[(0,j)]>=self.Data.y0[j] for j in range(self.Data.M)])
        else:
            t = tMin
            #Pay one-time cost
            self.mdl.add_constraints([self.mdl.sum(self.mdl.x_vars[(t,j,k)] for k in range(1,self.Data.Mj[j])) == self.mdl.y_vars[(t,j)] for j in range(self.Data.M)])
            
            #Budget, year 1+
            self.mdl.add_constraint(self.mdl.sum(self.Data.c[t][j]*self.mdl.sum(k*self.mdl.x_vars[(t,j,k)] for k in range(self.Data.Mj[j])) for j in range(self.Data.M))
                                                -self.mdl.sum(self.Data.c[t][j]*self.mdl.sum(k*self.Solution.x[(t-1,j,k)]for k in range(self.Data.Mj[j])) for j in range(self.Data.M))
                                                +self.mdl.sum(self.Data.f[t][j]*self.mdl.y_vars[(t,j)] for j in range(self.Data.M))
                                                <=self.Data.B[t]+self.mdl.sum(self.Data.f[t][j]*self.Solution.y[(t-1,j)] for j in range(self.Data.M)))

            #Can't remove charging outlets, year 1+
            self.mdl.add_constraints([self.mdl.sum(k*self.mdl.x_vars[(t,j,k)] for k in range(self.Data.Mj[j]))>=self.mdl.sum(k*self.Solution.x[(t-1,j,k)] for k in range(self.Data.Mj[j])) for j in range(self.Data.M)])

            ##Stations stay open, year 1+
            self.mdl.add_constraints([self.mdl.y_vars[(t,j)]>=self.Solution.y[(t-1,j)] for t in range(1,self.Data.T) for j in range(self.Data.M)])
        
        for t in range(tMin+1, tMax):
            #Pay one-time cost
            self.mdl.add_constraints([self.mdl.sum(self.mdl.x_vars[(t,j,k)] for k in range(1,self.Data.Mj[j])) == self.mdl.y_vars[(t,j)] for j in range(self.Data.M)])
            
            #Budget, year 1+
            self.mdl.add_constraint(self.mdl.sum(self.Data.c[t][j]*self.mdl.sum(k*self.mdl.x_vars[(t,j,k)] for k in range(self.Data.Mj[j])) for j in range(self.Data.M))
                                                -self.mdl.sum(self.Data.c[t][j]*self.mdl.sum(k*self.mdl.x_vars[(t-1,j,k)] for k in range(self.Data.Mj[j])) for j in range(self.Data.M))
                                                +self.mdl.sum(self.Data.f[t][j]*self.mdl.y_vars[(t,j)] for j in range(self.Data.M))
                                                <=self.Data.B[t]+self.mdl.sum(self.Data.f[t][j]*self.mdl.y_vars[(t-1,j)] for j in range(self.Data.M)))

            #Can't remove charging outlets, year 1+
            self.mdl.add_constraints([self.mdl.sum(k*self.mdl.x_vars[(t,j,k)] for k in range(self.Data.Mj[j]))>=self.mdl.sum(k*self.mdl.x_vars[(t-1,j,k)] for k in range(self.Data.Mj[j])) for j in range(self.Data.M)])

            ##Stations stay open, year 1+
            self.mdl.add_constraints([self.mdl.y_vars[(t,j)]>=self.mdl.y_vars[(t-1,j)] for t in range(1,self.Data.T) for j in range(self.Data.M)])

        #########################################################################################################################
        #Force given solution if provided
        if self.SolutionX is not None:
            for x in self.SolutionX:
                self.mdl.add_constraint(self.mdl.x_vars[x] == self.SolutionX[x])

        if self.SolutionY is not None:
            for y in self.SolutionY:
                self.mdl.add_constraint(self.mdl.y_vars[y] == self.SolutionY[y])
                
        ######################################################################################################################
        
        print("Constraints added")
        #Objective
        print("Adding objective")
        self.mdl.maximize(self.mdl.sum( (self.Data.Ni[t][i]/self.Data.R[i]) * self.mdl.w_vars[(t,i,r)] for i in range(self.Data.N) for r in range(self.Data.R[i]) for t in range(tMin, tMax)))

        toc=time()
        self.ModelCreationTime=toc-tic
        print("Model creation time (seconds):",self.ModelCreationTime)
        print('Model created')


        #Solve
        print('Begining solving process')
        self.mdl.solve(agent='local',log_output=False)
        print('Solving process complete!')
        print(self.mdl.solve_details)

        self.mdl=self.mdl
        self.Data=self.Data

        print('Recovering solution')
        self.RecoverSolution(tMin, tMax)
        print('Solution recovered successfully!')

    #Recreate the solution from the Cplex model. Number of charging outlets at each station is calculated.
    def RecoverSolution(self, tMin, tMax):
        #Not saving names will not crash normally, but will
        #set future years to the zero-solution. So we catch
        #that case here
        if len(self.mdl.find_matching_vars('x_')) == 0:
            raise Exception("Solution cannot be recovered successfully; names have not been saved")
        for t in range(tMin,tMax):
            for x in self.mdl.find_matching_vars('x_'+str(t)+'_'):
                xCoord=tuple(x.get_key())
                #xCoord: [0]=year, [1]=station, [2]=number of stations
                self.Solution.x[(t, xCoord[1], xCoord[2])] = round(x.solution_value)


            for y in self.mdl.find_matching_vars('y_'+str(t)+'_'):
                yCoord=tuple(y.get_key())
                #yCoord: [0]=year, [1]=station
                self.Solution.y[(t, yCoord[1])] = round(y.solution_value)
        



