from docplex.mp.model import Model
import numpy as np
from os import mkdir
from time import time
import json

from MaximumCover_Data import Data_MaximumCover, Test

from collections import OrderedDict





#################################################################################################################################
class ChargingStationModel_MaximumCover():

    def __init__(self, mdl, Data, params = None):
        tic=time()
    #Default parameters
        self.filepath = None
        self.filename = None
        self.SolutionX = None
        self.SolutionY = None
        self.HeuristicSolution = None
        self.Display = False
        self.ReturnSolution =False
        self.testNumber= '-1'

        self.mdl=mdl
        self.Data=Data


        self.threads = 8
        self.timelimit = 7200
        self.nodelimit = 9223372036800000000
        self.logdisplay = 4
        self.checkheuristic=False
        self.compressMemory = False
        self.BendersStrategy = -1

        if params:
            self.__dict__.update(params)
            if 'Solution' in params:
                self.SolutionX, self.SolutionY = self.ConvertSolutionMC(params['Solution'])

    #Model parameters
        mdl.context.cplex_parameters.threads = self.threads
        mdl.set_time_limit(self.timelimit)
        mdl.parameters.mip.display=self.logdisplay
        mdl.parameters.mip.limits.nodes= self.nodelimit
        mdl.parameters.emphasis.memory = int(self.compressMemory)
        mdl.parameters.benders.strategy = self.BendersStrategy

        if self.checkheuristic:
            mdl.parameters.mip.tolerances.absmipgap=1000000000
            mdl.parameters.mip.strategy.probe=-1
            mdl.parameters.preprocessing.relax=0
            mdl.parameters.mip.limits.cutpasses=-1
            mdl.parameters.preprocessing.repeatpresolve=0
            



    #Decision variables
        ##x,u, and w variables have too many indices to use built-in Cplex variables, so tuples must be created to use as dictionary keys.
        id_x=[(t,j,k) for j in range(Data.M) for k in range(Data.Mj[j]) for t in range(Data.T)]
        id_w=[(t,i,r) for i in range(Data.N) for r in range(Data.R[i]) for t in range(Data.T)]


        print("Creating variables")
        #Binary indicating if station j has *at least* k charging outlets in year t   
        mdl.x_vars=mdl.binary_var_dict(id_x,name='x')
        #Continuous variables for covering  
        mdl.w_vars=mdl.continuous_var_dict(id_w, ub = 1,name='w')
                
        print("Variables created")


    #Warmstart (if applicable)
        if self.HeuristicSolution is not None:
            print("Heuristic solution detected")
            warmstart=mdl.new_solution()
            for t in range(Data.T):
                for j in range(Data.M):
                    warmstart.add_var_value(mdl.y_vars[(t,j)], self.HeuristicSolution.y[(t,j)])
                    for k in range(Data.Mj[j]):
                        warmstart.add_var_value(mdl.x_vars[(t,j,k)], self.HeuristicSolution.x[(t,j,k)])
            
            mdl.add_mip_start(warmstart)
            print("Heuristic solution added")

    #Constraints
        print("Beginning constraints")
        #Budget, year 0
        mdl.add_constraint(
            mdl.sum(Data.c[0][j][k]*(mdl.x_vars[(0,j,k)]-Data.x0[j][k]) for j in range(Data.M) for k in range(1, Data.Mj[j])) 
            <=Data.B[0]
            )
        ##Can't remove charging outlets, year 0      
        mdl.add_constraints(mdl.x_vars[(0,j,k)] >= Data.x0[j][k] for j in range(Data.M) for k in range(Data.Mj[j]))
        

        #Covering constraints
        mdl.add_constraints(mdl.sum([Data.a[t][j][i][k][r]*mdl.x_vars[(t,j,k)] for j in range(Data.M) for k in range(1, Data.Mj[j])]) 
                            + Data.HC_covering[t][i][r]
                            >= mdl.w_vars[(t,i,r)]
                            for i in range(Data.N)  for r in range(Data.R[i]) for t in range(Data.T))
        
        for t in range(Data.T):
            #At least k outlets
            mdl.add_constraints(mdl.x_vars[(t,j,k)] <= mdl.x_vars[(t,j,k-1)]  for j in range(Data.M) for k in range(1, Data.Mj[j]))
     
            if t >0:
                #Budget, year 1+
                mdl.add_constraint(mdl.sum(Data.c[t][j][k]*(mdl.x_vars[(t,j,k)] - mdl.x_vars[(t-1,j,k)]) for j in range(Data.M) for k in range(1, Data.Mj[j])) 
                                                  <=Data.B[t])
                #Can't remove charging outlets, year 1+
                mdl.add_constraints(mdl.x_vars[(t-1,j,k)] <= mdl.x_vars[(t,j,k)]  for j in range(Data.M) for k in range(1, Data.Mj[j]))
 
        #########################################################################################################################
        #########################################################################################################################
        #Force given solution if provided
        if self.SolutionX is not None:
            mdl.add_constraints([mdl.x_vars[x] == self.SolutionX[x] for x in self.SolutionX])
        
        ######################################################################################################################
        print("Constraints added")
    #Objective
        print("Adding objective")
        mdl.maximize(mdl.sum( (Data.Ni[t][i]/Data.R[i]) * mdl.w_vars[(t,i,r)] for i in range(Data.N) for r in range(Data.R[i]) for t in range(Data.T)))

        toc=time()
        self.ModelCreationTime=toc-tic
        print("Model creation time (seconds):",self.ModelCreationTime)
        print('Model created')


    #Solve
        print('Begining solving process')
        if self.filename is not None:
            with open(self.filepath+'/'+self.filename+"_log.log", "a+") as out:
                out.write("\n")
                out.write("\n")
                out.write("\n")
                out.write("Test number "+str(self.testNumber)+"\n")
                mdl.solve(agent='local',log_output=out)
            print('Solving process complete!')
            print(mdl.solve_details)

            print('Recovering solution')
            try:
                self.Solution, self.EVs = self.RecoverSolution()
                print('Solution recovered successfully!')
                FullSolution=OrderedDict()
                FullSolution['Solution'] = self.Solution.tolist()
                FullSolution['EVs'] = self.EVs.tolist()
                json.dump(FullSolution,open(self.filepath+'/'+self.filename+".json", "w+"), indent=3)
            except Exception as e:
                print('Solution could not be recovered:')
                print(e)
            print('\n')
            print('\n')


        else:
            mdl.solve(agent='local',log_output=False)
            print('Solving process complete!')
            print(mdl.solve_details)

            print('Recovering solution')
            try:
                self.Solution, self.EVs = self.RecoverSolution()
                print('Solution recovered successfully!')
            except Exception as e:
                print('Solution could not be recovered:')
                print(e)
            print('\n')
            print('\n')
        
        if self.ReturnSolution:
            return FullSolution

    ###Function to recreate the solution from the Cplex model. Number of charging outlets at each station is calculated.
    def RecoverSolution(self):
        T=self.Data.T
        Solution = np.zeros( shape = (T, self.Data.M), dtype = int)
        EVs = np.zeros(shape = (T, self.Data.N))
        
        for x in self.mdl.find_matching_vars('x_'):
            #xCoord: [0]=year, [1]=station, [2]=number of stations
            xCoord=tuple(x.get_key())
            #location = int(self.Data.stations[str(xCoord[1])]["Location"])
            current = Solution[xCoord[0]][int(xCoord[1])]
            new = xCoord[2]*round(x.solution_value)
            if (new > current):
                Solution[xCoord[0]][int(xCoord[1])] = new

        for w in self.mdl.find_matching_vars('w_'):
            #wCoord: [0]=year, [1]=user class, [2]=scenario
            wCoord = tuple(w.get_key())
            t = wCoord[0]
            i = wCoord[1]
            EVs[t][i] += (self.Data.Ni[t][i]/self.Data.R[i])*w.solution_value

            
        return Solution, EVs
 