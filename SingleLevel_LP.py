from docplex.mp.model import Model
import numpy as np
from os import mkdir
from time import time
import json

from numpy.core.numeric import Inf

from SingleLevel_Data import Data_SingleLevel






#Find the solution of the Linear Programming (LP) relaxation of the Single-Level model
class ChargingStationModel_SingleLevel():

    def __init__(self, mdl, Data, params = None):
        tic=time()
        self.mdl=mdl
        self.Data=Data
    #Default parameters
        self.filepath = None
        self.filename = None
        self.SolutionX = None
        self.SolutionY = None
        self.HeuristicSolution = None
        self.Display = False
        self.ReturnSolution =False
        self.testNumber= '-1'
        self.UseCuts = True
        self.verbose = False

        self.threads = 8
        self.timelimit = 7200
        self.nodelimit = 9223372036800000000
        self.logdisplay = 4
        self.checkheuristic=False
        self.compressMemory = False

        if params:
            self.__dict__.update(params)

    #Model parameters
        mdl.context.cplex_parameters.threads = self.threads
        mdl.set_time_limit(self.timelimit)
        mdl.parameters.mip.display=self.logdisplay
        mdl.parameters.mip.limits.nodes= self.nodelimit
        mdl.parameters.emphasis.memory = int(self.compressMemory)
        #mdl.parameters.emphasis.mip = 1
        mdl.parameters.mip.strategy.file = 2

        if self.checkheuristic:
            mdl.parameters.mip.tolerances.absmipgap=1000000000
            mdl.parameters.mip.strategy.probe=-1
            mdl.parameters.preprocessing.relax=0
            mdl.parameters.mip.limits.cutpasses=-1
            mdl.parameters.preprocessing.repeatpresolve=0

        else:
            mdl.parameters.mip.tolerances.absmipgap=1e-06
            mdl.parameters.mip.strategy.probe=0
            mdl.parameters.preprocessing.relax=-1
            mdl.parameters.mip.limits.cutpasses=0
            mdl.parameters.preprocessing.repeatpresolve=-1            
            



    #Decision variables
        ##x,u,w, and alpha variables have too many indices to use built-in Cplex variables, so tuples must be created to use as dictionary keys.
        id_x=[(t,j,k) for j in range(Data.M) for k in range(Data.Mj[j]) for t in range(Data.T)]
        id_alpha=[(t,i,r)for i in range(Data.N) for r in range(Data.R[i])for t in range(Data.T)]
        id_u0=[]
        id_u=[]
        for t in range(Data.T):
            for i in range(Data.N):
                for r in range(Data.R[i]):
                    for j in Data.C0i[t][i]:
                        id_u0.append((t,j,i,r))
                    for j in Data.C1i[t][i]:
                        id_u.append((t,j,i,r))


        print("Creating variables")
        #Binary indicating if station j has k charging outlets in year t   
        mdl.x_vars=mdl.continuous_var_dict(id_x, ub = 1, name='x')
        #Binary indicating if station j is open in year t
        #Continuous data variable with utility for opt-out option
        mdl.u0_vars=mdl.continuous_var_dict(id_u0,name='u0', lb=-mdl.infinity)
        #Discounted utility vars
        mdl.uBar_vars=mdl.continuous_var_dict(id_u, name='uBar', lb=-mdl.infinity) 
        #Binary indicating if opt-out was selected (i.e. had highest utility)
        mdl.w0_vars=mdl.continuous_var_dict(id_u0,ub=1,name='w0')
        #Binary indicating if choice j was selected (i.e. had highest utility)
        mdl.w1_vars=mdl.continuous_var_dict(id_u,ub=1,name='w')
        #Continuous variables for dual objective value
        mdl.alpha_vars=mdl.continuous_var_dict(id_alpha, name='alpha',lb=-mdl.infinity)
            

        print("Variables created")


    #Warmstart (if applicable)
        if self.HeuristicSolution is not None:
            print("Heuristic solution detected")
            mdl.parameters.conflict.display = 2
            warmstart=mdl.new_solution()
            for t in range(Data.T):
                for j in range(Data.M):
                    for k in range(Data.Mj[j]):
                        warmstart.add_var_value(mdl.x_vars[(t,j,k)],self.HeuristicSolution.x[(t,j,k)])

            if 'w0' in self.HeuristicSolution.__dict__:
                for t in range(Data.T):
                    for i in range(Data.N):
                        for r in range(Data.R[i]):
                            for j in Data.C0i[t][i]:
                                warmstart.add_var_value(mdl.w0_vars[(t,j,i,r)],self.HeuristicSolution.w0[(t,j,i,r)])
                            for j in Data.C1i[t][i]:
                                warmstart.add_var_value(mdl.w1_vars[(t,j,i,r)],self.HeuristicSolution.w1[(t,j,i,r)])
            
            mdl.add_mip_start(warmstart)
            print("Heuristic solution added")

    # #Preprocessing (if applicable)
    #     if Data.Preprocess_w1 is not None:
    #         print("Preprocessing detected")
    #         mdl.change_var_upper_bounds([mdl.w1_vars[key] for key in Data.Preprocess_w1],0)
    #         print("Preprocessing added")

    #Constraints
        print("Beginning constraints")
        #Budget, year 0
        mdl.add_constraint(
            mdl.sum(Data.c[0][j][k]*(mdl.x_vars[(0,j,k)]-Data.x0[j][k]) for j in range(Data.M) for k in range(1, Data.Mj[j])) 
            <=Data.B[0]
            )

        ##Can't remove charging outlets, year 0      
        mdl.add_constraints(mdl.x_vars[(0,j,k)] >= Data.x0[j][k] for j in range(Data.M) for k in range(Data.Mj[j]))

        
        for t in range(Data.T):
            #At least k outlets
            mdl.add_constraints(mdl.x_vars[(t,j,k)] <= mdl.x_vars[(t,j,k-1)]  for j in range(Data.M) for k in range(1, Data.Mj[j]))
            if t >0:
                #Budget, year 1+
                mdl.add_constraint(mdl.sum(Data.c[t][j][k]*(mdl.x_vars[(t,j,k)] - mdl.x_vars[(t-1,j,k)]) for j in range(Data.M) for k in range(1, Data.Mj[j])) 
                                                  <=Data.B[t])
                #Can't remove charging outlets, year 1+
                mdl.add_constraints(mdl.x_vars[(t-1,j,k)] <= mdl.x_vars[(t,j,k)]  for j in range(Data.M) for k in range(1, Data.Mj[j]))

            for i in range(Data.N):
                if len(Data.C1i[t][i]) > 0:
                    #Can only select one option in choice set
                    mdl.add_constraints([mdl.sum( mdl.w1_vars[(t,int(j),i,r)] for j in Data.C1i[t][i] )+ mdl.sum( mdl.w0_vars[(t,int(j),i,r)] for j in Data.C0i[t][i])\
                                    ==1 for r in range(Data.R[i])], 'SelectOneOption' )
                    ##Set u0
                    mdl.add_constraints([mdl.u0_vars[(t,j,i,r)]==Data.d0[t][j][i][r] for j in Data.C0i[t][i] for r in range(Data.R[i])])
                        
                    ##Discounted utility constraints
                
                    mdl.add_constraints([mdl.uBar_vars[(t,j,i,r)] >= Data.aBar[t][i] for j in Data.C1i[t][i] for r in range(Data.R[i])
                                        if mdl.w1_vars[(t,j,i,r)].ub >0], 'Discounted1')
                    mdl.add_constraints([mdl.uBar_vars[(t,j,i,r)] <= Data.aBar[t][i] + Data.MBar[t][j][i][r]*mdl.x_vars[(t,j,1)] for j in Data.C1i[t][i] for r in range(Data.R[i])
                                        if mdl.w1_vars[(t,j,i,r)].ub >0] , 'Discounted2')
                    mdl.add_constraints([mdl.uBar_vars[(t,j,i,r)] >= mdl.sum( Data.beta[t][j][i][k]*mdl.x_vars[(t,j,k)] for k in range(Data.Mj[j]) ) +Data.d1[t][j][i][r]
                                        - Data.MBar[t][j][i][r] * (1-mdl.x_vars[(t,j,1)]) for j in Data.C1i[t][i] for r in range(Data.R[i])
                                        if mdl.w1_vars[(t,j,i,r)].ub >0] , 'Discounted3')
                    mdl.add_constraints([mdl.uBar_vars[(t,j,i,r)] <= mdl.sum( Data.beta[t][j][i][k]*mdl.x_vars[(t,j,k)] for k in range(Data.Mj[j]) )
                                        +Data.d1[t][j][i][r] for j in Data.C1i[t][i] for r in range(Data.R[i])
                                        if mdl.w1_vars[(t,j,i,r)].ub >0] , 'Discounted4')
                    mdl.add_constraints([mdl.uBar_vars[(t,j,i,r)] == Data.aBar[t][i] for j in Data.C1i[t][i] for r in range(Data.R[i])
                                        if mdl.w1_vars[(t,j,i,r)].ub == 0] , 'Discounted5')


                    ##Set dual variable to max utility, opt-out
                    mdl.add_constraints([mdl.alpha_vars[(t,i,r)]>=mdl.u0_vars[(t,j,i,r)] for j in Data.C0i[t][i] for r in range(Data.R[i])])
                    ##Set dual variable to max utility, public charging
                    mdl.add_constraints([mdl.alpha_vars[(t,i,r)]>=mdl.uBar_vars[(t,j,i,r)] for j in Data.C1i[t][i] for r in range(Data.R[i]) if mdl.w1_vars[(t,j,i,r)].ub >0], 'Dual2')

                    ##Set correct w value to 1, opt-out
                    mdl.add_constraints([mdl.u0_vars[(t,j,i,r)] - mdl.alpha_vars[(t,i,r)] + (1 - mdl.w0_vars[(t,j,i,r)] ) * Data.mu0[t][j][i][r]
                                        >= 0 for j in Data.C0i[t][i] for r in range(Data.R[i])])
                    ##Set correct w value to 1, public charging
                    mdl.add_constraints([mdl.uBar_vars[(t,j,i,r)]- mdl.alpha_vars[(t,i,r)] + (1 - mdl.w1_vars[(t,j,i,r)] ) * Data.mu[t][i][r]
                                        >= 0 for j in Data.C1i[t][i] for r in range(Data.R[i])  if mdl.w1_vars[(t,j,i,r)].ub >0], 'ComplementarySlackness2')
                else:
                    #Can only select one option in choice set
                    mdl.add_constraints([mdl.sum( mdl.w0_vars[(t,int(j),i,r)] for j in Data.C0i[t][i])\
                                    ==1 for r in range(Data.R[i])], 'SelectOneOption' )
                    ##Set u0
                    mdl.add_constraints([mdl.u0_vars[(t,j,i,r)]==Data.d0[t][j][i][r] for j in Data.C0i[t][i] for r in range(Data.R[i])])
                        

                    ##Set dual variable to max utility, opt-out
                    mdl.add_constraints([mdl.alpha_vars[(t,i,r)]>=mdl.u0_vars[(t,j,i,r)] for j in Data.C0i[t][i] for r in range(Data.R[i])])

                    ##Set correct w value to 1, opt-out
                    mdl.add_constraints([mdl.u0_vars[(t,j,i,r)] - mdl.alpha_vars[(t,i,r)] + (1 - mdl.w0_vars[(t,j,i,r)] ) * Data.mu0[t][j][i][r]
                                        >= 0 for j in Data.C0i[t][i] for r in range(Data.R[i])])

            
        #########################################################################################################################
        #########################################################################################################################
        #Force given solution if provided
        if self.SolutionX is not None:
            for x in self.SolutionX:
                mdl.add_constraint(mdl.x_vars[x] == self.SolutionX[x])
    
        ######################################################################################################################
        print("Constraints added")
    #Objective
        print("Adding objective")
        mdl.minimize(mdl.sum(mdl.sum((Data.Ni[i]/Data.R[i])*mdl.sum(mdl.w0_vars[(t,0,i,r)] for r in range(Data.R[i])) for i in range(Data.N)) for t in range(Data.T)) )

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
                self.Solution = self.RecoverSolution()
                print('Solution recovered successfully!')
                FullSolution = {}
                # #FullSolution['StationTotals'] = StationTotals.tolist()
                FullSolution['Solution'] = self.Solution.tolist()
                # FullSolution['TotalUsers(yearly)']=sum(Data.Ni)
                # FullSolution['ObjectiveValue']=mdl.objective_value
                # FullSolution['DataCreationTime']=Data.DataCreationTime
                # FullSolution['ModelCreationTime']=self.ModelCreationTime
                # FullSolution['SolveTime']=round(mdl.solve_details.time,2)
                json.dump(FullSolution,open(self.filepath+'/'+self.filename+".txt", "w+"), indent=3)
            except Exception as e:
                print('Solution could not be recovered:')
                print(e)
            print('\n')
            print('\n')
        else:
            mdl.solve(agent='local', log_output = self.verbose)
            print('Solving process complete!')
            print(mdl.solve_details)







    ###Function to recreate the solution from the Cplex model. Number of charging outlets at each station is calculated.
    def RecoverSolution(self):
        T = self.Data.T
        #StationTotals={t:{j: 0 for j in range(self.Data.network["NumberOfNodes"])} for t in range(T)}
        Solution = np.zeros( shape = (T, self.Data.M), dtype = int)    
        for x in self.mdl.find_matching_vars('x_'):
            xCoord=tuple(x.get_key())
            #xCoord: [0]=year, [1]=station, [2]=number of stations
            #location=int(self.Data.stations[str(xCoord[1])]["Location"])
            #StationTotals[xCoord[0]][location]+=xCoord[2]*int(x.solution_value)
            Solution[xCoord[0]][int(xCoord[1])] += xCoord[2]*int(x.solution_value)


        return Solution
        


if __name__ == "__main__":
        fp = "Data/Precomputed/Simple/SingleLevel/SL0.json"
        print("Reading data file")
        try:
            Data_MC = Data_SingleLevel(load = fp)
        except:
            # Data_MC = Data_MaximumCover( load = "/local_2/outer/lamste/Data/Precomputed/{}/MaximumCover/MC{}.json".format(set,testNumber))
            # Data_MC.pickle(fp)
            print("Could not load file: "+fp)
        Data_MC.Process_NonBenders()

        mdl = Model(name='charging station', ignore_names=False)
        solver = ChargingStationModel_SingleLevel(mdl, Data_MC, {'verbose':True, 'threads':4, 'timelimit':300000, 'compressMemory':True})
        print(round(mdl.solve_details.time,2))

