from docplex.mp.model import Model
from docplex.mp.constants import WriteLevel
import numpy as np
from os import mkdir
from time import time
import json

from MaximumCover_Data import Data_MaximumCover
from MaximumCover_Model import ChargingStationModel_MaximumCover


#Class for storing solution details
class Solution():

    def __init__(self, sizeDict):
        self.x={(t,j,k):0 for t in range(sizeDict['T']) for j in range(sizeDict['M']) for k in range(sizeDict['Mj'][j])}
        self.obj = np.zeros(sizeDict['T'])

    def Convert(self, T, M, Mj):
        temp = np.zeros((T,M), dtype = int)
        for t in range(T):
            for j in range(M):
                for k in range(Mj[j]):
                    if self.x[(t,j,k)] > 1 - 1e-06: 
                        temp[t][j] =k
        self.x = temp

#################################################################################################################################
class RollingHorizon():

    def __init__(self, mdl, Data, params= None):
    #Default parameters
        self.filepath = "Results"
        self.filename = "Test"
        self.SolutionX = None
        self.SolutionY = None
        self.Display = False
        self.ReturnSolution =False
        self.testNumber= '-1'
        self.stepSize= 0

        self.threads = 8
        self.timelimit = 900
        self.nodelimit = 9223372036800000000
        self.logdisplay = 4
        self.compressMemory = False
        self.mode = "Single" 

        self.mdl=mdl
        self.Data=Data
        
        self.Solution = Solution({'T': self.Data.T, 'M': self.Data.M, 'Mj':self.Data.Mj})
 
        self.HeuristicTime=[]

        if params:
            self.__dict__.update(params)
        if self.mode not in ["Single", "Multi"]:
            raise ValueError("Unrecognised run mode. Must be among 'Single', 'Multi'. ")
        #Convert the timelimit to list form (for backwards compatibility)
        if type(self.timelimit) in [int, float]:
            self.timelimit = [self.timelimit for _ in range(self.Data.T)]
 
        #Model parameters
        self.mdl.context.cplex_parameters.threads = self.threads
        self.mdl.parameters.mip.display=self.logdisplay
        self.mdl.parameters.mip.limits.nodes= self.nodelimit
        self.mdl.parameters.emphasis.memory = int(self.compressMemory)

    def Solve(self):
        for t in range(self.Data.T):
            tic = time()
            self.mdl.clear()
            self.mdl.set_time_limit(self.timelimit[t])
            print("Starting Rolling Horizon for t=", t)
            self.RollingHorizon(self.mdl, t)
            print("Rolling Horizon complete for t=", t)
            print('\n')
            toc = time()
            self.HeuristicTime.append(round(self.mdl.solve_details.time,2))
        self.Solution.Convert(self.Data.T, self.Data.M, self.Data.Mj)

    def RollingHorizon(self, mdl, tMin):
        tic = time()
        #Decision variables
        ##x,u, and w variables have too many indices to use built-in Cplex variables, so tuples must be created to use as dictionary keys.
        id_x_discrete=[(tMin,j,k) for j in range(self.Data.M) for k in range(self.Data.Mj[j])]
        id_x_continuous=[(t,j,k) for j in range(self.Data.M) for k in range(self.Data.Mj[j]) for t in range(tMin+1, self.Data.T)]
        id_w=[(t,i,r) for i in range(self.Data.N) for r in range(self.Data.R[i])for t in range(tMin, self.Data.T)]

        #Binary indicating if station j has at least k charging outlets in year t   
        mdl.x_vars_d = mdl.binary_var_dict(id_x_discrete,name='x')
        mdl.x_vars_c = mdl.continuous_var_dict(id_x_continuous, ub =1 , name='xc')

        #Continuous variables for covering  
        mdl.w_vars = mdl.continuous_var_dict(id_w, ub = 1,name='w')        


        #Constraints
        ####################################
        ####Year tMin
        ####################################
        #Must have at least k-1 outlets in order to have at least k outlets
        mdl.add_constraints([mdl.x_vars_d[(tMin,j,k)] <= mdl.x_vars_d[(tMin,j,k-1)]  for j in range(self.Data.M) for k in range(1, self.Data.Mj[j])])

        #Covering constraints
        mdl.add_constraints(mdl.sum([self.Data.a[tMin][j][i][k][r]*mdl.x_vars_d[(tMin,j,k)] for j in range(self.Data.M) for k in range(1, self.Data.Mj[j])]) 
                            + self.Data.HC_covering[tMin][i][r]
                            >= mdl.w_vars[(tMin,i,r)]
                            for i in range(self.Data.N)  for r in range(self.Data.R[i]))

        if tMin == 0:
            #Budget
            mdl.add_constraint(mdl.sum(self.Data.c[0][j][k]*(mdl.x_vars_d[(0,j,k)] - self.Data.x0[j][k]) for j in range(self.Data.M) for k in range(1, self.Data.Mj[j]))
                                    <= self.Data.B[0])

            ##Can't remove charging outlets       
            mdl.add_constraints(mdl.x_vars_d[(0,j,k)] >= self.Data.x0[j][k] for j in range(self.Data.M) for k in range(self.Data.Mj[j]))


        else:
            #Budget
            mdl.add_constraint(mdl.sum(self.Data.c[tMin][j][k]*(mdl.x_vars_d[(tMin,j,k)] - self.Solution.x[(tMin-1,j,k)]) for j in range(self.Data.M) for k in range(1, self.Data.Mj[j]))
                                    <= self.Data.B[tMin])

            ##Can't remove charging outlets
            mdl.add_constraints([mdl.x_vars_d[(tMin,j,k)] >= self.Solution.x[(tMin-1,j,k)] for j in range(self.Data.M) for k in range(1, self.Data.Mj[j])])


        ####################################
        ####Years tMin +1, ...,  T
        ####################################
        if self.mode == "Multi":
            for t in range(tMin + 1, self.Data.T):
                  #Must have at least k-1 outlets in order to have at least k outlets
                mdl.add_constraints([mdl.x_vars_c[(t,j,k)] <= mdl.x_vars_c[(t,j,k-1)]  for j in range(self.Data.M) for k in range(1, self.Data.Mj[j]) ])

                #Covering constraints
                mdl.add_constraints(mdl.sum([self.Data.a[t][j][i][k][r]*mdl.x_vars_c[(t,j,k)] for j in range(self.Data.M) for k in range(1, self.Data.Mj[j])]) 
                                    + self.Data.HC_covering[t][i][r]
                                    >= mdl.w_vars[(t,i,r)]
                                    for i in range(self.Data.N)  for r in range(self.Data.R[i]))

                if t == tMin + 1:
                    #Budget
                    mdl.add_constraint(mdl.sum(self.Data.c[t][j][k]*(mdl.x_vars_c[(t,j,k)] - mdl.x_vars_d[(t-1,j,k)]) for j in range(self.Data.M) for k in range(1, self.Data.Mj[j]))
                                                        <=self.Data.B[t])

                    #Can't remove charging outlets, year 1+
                    mdl.add_constraints([mdl.x_vars_d[(t-1,j,k)] <= mdl.x_vars_c[(t,j,k)]  for j in range(self.Data.M) for k in range(1, self.Data.Mj[j])])

                else:
                    #Budget
                    mdl.add_constraint(mdl.sum(self.Data.c[t][j][k]*(mdl.x_vars_c[(t,j,k)] - mdl.x_vars_c[(t-1,j,k)]) for j in range(self.Data.M) for k in range(1, self.Data.Mj[j]))
                                                        <=self.Data.B[t])

                    #Can't remove charging outlets, year 1+
                    mdl.add_constraints([mdl.x_vars_c[(t-1,j,k)] <= mdl.x_vars_c[(t,j,k)]  for j in range(self.Data.M) for k in range(1, self.Data.Mj[j]) ])



        if self.mode == "Multi":
            #Objective                      
            mdl.maximize(self.mdl.sum( (self.Data.Ni[t][i]/self.Data.R[i]) * self.mdl.w_vars[(t,i,r)] for i in range(self.Data.N) for r in range(self.Data.R[i]) for t in range(tMin, self.Data.T)))
        elif self.mode == "Single":
            #Objective                      
            mdl.maximize(self.mdl.sum( (self.Data.Ni[tMin][i]/self.Data.R[i]) * self.mdl.w_vars[(tMin,i,r)] for i in range(self.Data.N) for r in range(self.Data.R[i])))

        mdl.solve(agent='local', log_output=False)
        self.RecoverSolution(mdl, tMin)

    def RecoverSolution(self, mdl, tMin):
        if len(mdl.find_matching_vars('x_')) == 0:
            raise Exception("Solution cannot be recovered successfully; names have not been saved")
        for x in self.mdl.find_matching_vars('x_'+str(tMin)+'_'):
            xCoord=tuple(x.get_key())
            #xCoord: [0]=year, [1]=station, [2]=number of stations
            self.Solution.x[(tMin, xCoord[1], xCoord[2])] = x.solution_value        
        for w in self.mdl.find_matching_vars('w_'+str(tMin)+'_'):
            wCoord=tuple(w.get_key())
            #wCoord: [0]=year, [1]=user class, [2]=scenario
            i = wCoord[1]
            if w.solution_value > 1 - 1e-06:
                self.Solution.obj[tMin] += (self.Data.Ni[tMin][i]/self.Data.R[i])


if __name__ == "__main__":
        fp = "Data/Precomputed/Simple/MaximumCover/MC0.pickle"
        print("Reading data file")
        try:
            Data_MC = Data_MaximumCover( unpickle = fp)
        except:
            # Data_MC = Data_MaximumCover( load = "/local_2/outer/lamste/Data/Precomputed/{}/MaximumCover/MC{}.json".format(set,testNumber))
            # Data_MC.pickle(fp)
            print("Could not load file: "+fp)
        print("Creating covering")
        Data_MC.CreateCovering()
        Data_MC.Process_NonBenders()
        print("Covering Created")

        mdl = Model(name='charging station', ignore_names=False)

        timelimit = [7200*(2**(-t-1)) for t in range(Data_MC.T)]
        timelimit[-1] += 7200 - sum(timelimit)
        params_RH = {'filepath':None, 'filename':None, 'timelimit':timelimit, 'compressMemory':True, 'mode':"Single"}
        solver = RollingHorizon(mdl, Data_MC, params_RH)
        solver.Solve()
        print("Single: ", Data_MC.SolutionQuality(solver.Solution.x))
        mdl.clear()   

        params_RH = {'filepath':None, 'filename':None, 'timelimit':timelimit, 'compressMemory':True, 'mode':"Multi"}
        solver = RollingHorizon(mdl, Data_MC, params_RH)
        solver.Solve()
        print("Multi: ", Data_MC.SolutionQuality(solver.Solution.x))
        mdl.clear()   