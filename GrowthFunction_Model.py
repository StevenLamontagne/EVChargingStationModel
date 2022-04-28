from docplex.mp.model import Model
import json

from GrowthFunction_Data import Data_GrowthFunction


class ChargingStationModel_GrowthFunction():
   def __init__(self, mdl, Data, params = None):
      self.Data = Data

      #Default parameters
      #Filepath and name for saving log and results
      self.filepath = None 
      self.filename = None 
      #Solution components when using a fixed solution
      #(forces exact solution)
      self.SolutionX = None 
      self.SolutionY = None
      

      #Default CPLEX parameters
      self.threads = 8
      self.timelimit = 7200
      self.nodelimit = 9223372036800000000
      self.logdisplay = 4
      self.compressMemory = False

      if params:
         self.__dict__.update(params)
         if 'Solution' in params:
            self.SolutionX,self.SolutionY = self.ConvertSolutionGF(params['Solution'])

      #Set CPLEX model parameters
      mdl.context.cplex_parameters.threads = self.threads
      mdl.set_time_limit(self.timelimit)
      mdl.parameters.mip.display=self.logdisplay
      mdl.parameters.mip.limits.nodes= self.nodelimit
      mdl.parameters.emphasis.memory = int(self.compressMemory)

      ################################################################################################################
      #Decision variables
      ##x,u, and w variables have too many indices to use built-in Cplex variables, so tuples must be created to use as dictionary keys.
      id_h = [(t,i,j) for t in range(Data.T) for i in range(Data.N) for j in Data.Ni[i]]
      id_x = [(t,j) for j in range(Data.M) for t in range(Data.T)]
      id_w = [(t,s) for s in range(Data.s) for t in range(Data.T)]

      #Continuous variable for number of EVs based in location i and charging at station j at the end of year t
      mdl.h_vars = mdl.continuous_var_dict(id_h, name = "h")
      #Integer variable for number of charging outlets at station j in year t
      mdl.x_vars = mdl.integer_var_dict(id_x, name = 'x')
      #Binary variable indicating if station j is open in year t
      mdl.y_vars = mdl.binary_var_dict(id_x, name = 'y')
      #Binary variable indicating if EV total is in segment s in year t
      mdl.w_vars = mdl.binary_var_dict(id_w, ub = 1, name = 'w')
      #Continuous variable for number of EVs in each segment s at the beginning of year t
      mdl.z_vars = mdl.continuous_var_dict(id_w, name = "z")

      #############################################################################################################
      #Constraints
      ##Budget, adjusted for each year
      mdl.ct_budget0 = mdl.add_constraint(mdl.sum(Data.c[0][j]*(mdl.x_vars[(0,j)]-Data.x0[j])+Data.f[0][j]*mdl.y_vars[(0,j)] for j in range(Data.M))
                                       <=Data.B[0]+mdl.sum(Data.f[0][j]*Data.y0[j] for j in range(Data.M)))
      for t in range(1,Data.T):
         mdl.ct_budget = mdl.add_constraint(mdl.sum(Data.c[t][j]*(mdl.x_vars[(t,j)]-mdl.x_vars[(t-1,j)] )+Data.f[t][j]*mdl.y_vars[(t,j)] for j in range(Data.M))
                                       <=Data.B[t]+mdl.sum([Data.f[t][j]*mdl.y_vars[(t-1,j)] for j in range(Data.M)]))
                           
      ##Forced to pay one-time cost
      for t in range(Data.T):
         for j in range(Data.M):
               mdl.add_constraint(mdl.x_vars[(t,j)] <= Data.Mj[j]*mdl.y_vars[(t,j)])

      ##Can't remove charging stations
      mdl.add_constraints([mdl.x_vars[(0,j)] >= Data.x0[j] for j in range(Data.M)])
      for t in range(1,Data.T):
         for j in range(Data.M):
               mdl.add_constraint(mdl.x_vars[(t,j)] >= mdl.x_vars[(t-1,j)])

      ##One-time costs already paid
      mdl.add_constraints([mdl.y_vars[(0,j)] >= Data.y0[j] for j in range(Data.M)])

      ##Stations stay open
      mdl.add_constraints([mdl.y_vars[(t,j)] >= mdl.y_vars[(t-1,j)] for t in range(1,Data.T) for j in range(Data.M)])


      ##Assign new EV users for each year
      mdl.add_constraint(mdl.sum([mdl.z_vars[(0,s)] for s in range(Data.s)]) == mdl.sum(Data.initialEVs[i] for i in range(Data.N)))
      for t in range(1,Data.T):
         mdl.add_constraint(mdl.sum([mdl.z_vars[(t,s)] for s in range(Data.s)]) == mdl.sum(mdl.sum(mdl.h_vars[(t-1,i,j)] for j in Data.Ni[i]) for i in range(Data.N)))

      ##Find segment in piecewise linear function for current EV totals
      for t in  range(Data.T):
         for s in range(Data.s):
               mdl.add_constraint(Data.S[s].startpoint*mdl.w_vars[(t,s)] <= mdl.z_vars[(t,s)])
               mdl.add_constraint(Data.S[s].endpoint*mdl.w_vars[(t,s)] >= mdl.z_vars[(t,s)])

      ##Can only select one segment
      for t in range(Data.T):
         mdl.add_constraint(mdl.sum([mdl.w_vars[(t,s)] for s in range(Data.s)]) <= 1)

      ##Bound new EV users by piecewise linear function
      for i in range(Data.N):
               mdl.add_constraint(mdl.sum([mdl.h_vars[(0,i,j)] for j in Data.Ni[i]]) <=
                                 Data.initialEVs[i]
                                 +(Data.Ri[i]/Data.R)*mdl.sum([Data.S[s].intercept*mdl.w_vars[(t,s)]+(Data.S[s].slope-1)*mdl.z_vars[(0,s)] for s in range(Data.s)])
                                 ) 
      for t in range(1,Data.T):
         for i in range(Data.N):
               mdl.add_constraint(mdl.sum([mdl.h_vars[(t,i,j)] for j in Data.Ni[i]]) <=
                                 mdl.sum([mdl.h_vars[(t-1,i,j)] for j in Data.Ni[i]])
                                 +(Data.Ri[i]/Data.R)*mdl.sum([Data.S[s].intercept*mdl.w_vars[(t,s)]+(Data.S[s].slope-1)*mdl.z_vars[(t,s)] for s in range(Data.s)])
                                 )

      ##Bound new EV users by previous EV users
      for i in range(Data.N):
         mdl.add_constraint(Data.initialEVs[i] <= mdl.sum([mdl.h_vars[(0,i,j)] for j in Data.Ni[i]]))
         for t in range(1,Data.T):
               mdl.add_constraint(mdl.sum([mdl.h_vars[(t-1,i,j)] for j in Data.Ni[i]]) <= mdl.sum([mdl.h_vars[(t,i,j)] for j in Data.Ni[i]]))
               
      ##Bound new EV users by capacity of stations
      for t in range(Data.T):
         for j in range(Data.M):
               mdl.add_constraint(Data.alpha*mdl.sum([mdl.h_vars[(t,i,j)] for i in range(Data.N) if j in Data.Ni[i]]) <= Data.a[t]*mdl.x_vars[(t,j)])
      
      ###################################################################################################################### 
      #Force given solution if provided
      if self.SolutionX is not None:
         for x in self.SolutionX:
               mdl.add_constraint(mdl.x_vars[x] == self.SolutionX[x])

      if self.SolutionY is not None:
         for y in self.SolutionY:
               mdl.add_constraint(mdl.y_vars[y] == self.SolutionY[y])
               

      ######################################################################################################################                
      #Objective
      mdl.maximize(mdl.sum([mdl.sum([mdl.h_vars[(Data.T-1,i,j)] for j in Data.Ni[i]]) for i in range(Data.N)]))

      print('Model created')


      #Solve
      print('Begining solving process')
      if self.filepath is not None and self.filename is not None:
         with open(self.filepath+'/'+self.filename+"_log.txt", "a+") as out:
            out.write("\n")
            out.write("\n")
            out.write("\n")
            mdl.solve(agent='local',log_output = out)
         print('Solving process complete!')
         print(mdl.solve_details)
         self.Solution, self.TotalEVs, self.EVStationChoice = self.RecoverSolution(mdl,Data)

         FullSolution = {}
         FullSolution['Solution'] = self.Solution
         FullSolution['TotalEVs'] = self.TotalEVs
         FullSolution['StationChoice'] = self.EVStationChoice
         json.dump(FullSolution, open(self.filepath+"/"+self.filename+".json", "w+"), indent=3)
         print('Solution recovered successfully!')
      else:
         mdl.solve(agent='local',log_output = False)
         self.Solution, self.TotalEVs, self.EVStationChoice = self.RecoverSolution(mdl,Data)


   #Recreate the solution from the Cplex model. Number of charging outlets at each station is calculated.
   def RecoverSolution(self, mdl, Data):
      T = Data.T
      Solution={t:{} for t in range(T)}
      totalEVs={t:{i:0 for i in range(Data.N)} for t in range(T)}
      EVStationChoice={t:{i:{j:0 for j in Data.Ni[i]} for i in range(Data.N)} for t in range(T)}
      
      for x in mdl.find_matching_vars('x_'):
         xCoord = tuple(x.get_key())
         #xCoord: [0] = year, [1] = station
         if xCoord[1] not in Solution[xCoord[0]]:
               Solution[xCoord[0]][xCoord[1]]={'value':0}
         Solution[xCoord[0]][xCoord[1]]['value'] += int(x.solution_value)
      for h in mdl.find_matching_vars('h_'):
         if h.solution_value > 0:
            hCoord = tuple(h.get_key())
            #hCoord: [0] = year, [1] = origin, [2] = charging destination
            totalEVs[hCoord[0]][hCoord[1]] += h.solution_value
            EVStationChoice[hCoord[0]][hCoord[1]][hCoord[2]] = int(h.solution_value)
      return Solution, totalEVs, EVStationChoice

   #Converts a predefined solution in array or dict format for warmstart
   def ConvertSolutionGF(self,solution):
    T = self.Data.T
    M = self.Data.M
    id_x=[(t,j) for j in range(M) for t in range(T)]
    SolutionX={x:0 for x in id_x}
    SolutionY={y:0 for y in id_x}
    for t in range(T):
        for j in range(M):
            SolutionX[(int(t),int(j))] = solution[t][j]
            if solution[t][j]>0:
                SolutionY[(int(t),int(j))]=1
    return SolutionX, SolutionY   




      
    

