import json
from math import atan, exp
import numpy as np
import pandas as pd
from time import time
import pickle



class Data_MaximumCover():
    #Parameters can either be supplied via params, initialised to their default, loaded from a json file (via load), or loaded from a pickle file (via unpickle)
    def __init__(self, userFilepath = None, stationFilepath = None, params=None, load = None, unpickle = None):
        if (userFilepath is None or stationFilepath is None) and (load is None) and (unpickle is None):
            raise Exception("Unable to load or create data. Ensure a file is provided to load or that a user filepath and station filepath is provided.")
        if (load is None and unpickle is None):
            start=time()
            self.network=json.load(open("Data/Network/TroisRivieresNetwork.json", "r"))
            self.stations=json.load(open(stationFilepath, "r"))
            self.userClasses=json.load(open(userFilepath, "r"))
            self.userFilepath = userFilepath
            self.stationFilepath = stationFilepath

            self.Preprocess_w1 = None
            
            #Number of years for the simulation
            self.T = 4

            #Percentage of population that is considering purchasing a vehicle in each year
            self.n_v = 0.1
            
            #Number of charging stations
            self.M = len(self.stations)

            #Number of user classes
            self.N = len(self.userClasses)

            #Population of each class
            #To support an older style of user class file (when population was constant across years)
            #we check and convert float to an array
            if type(self.userClasses[0]["Population"]) == float:
                self.Ni = [[int(self.n_v*i["Population"]) for i in self.userClasses] for t in range(self.T)]
            else:
                self.Ni = [[int(self.n_v*i["Population"][t]) for i in self.userClasses] for t in range(self.T)]

            #Scale parameter for Gumbel distribution for error terms
            self.gumbelScale = 3
            #Location parameter for Gumbel distribution for error terms
            self.gumbelLocation = 0
            #Scale parameter for Normal distribution for error terms
            self.normalScale = 1
            #Location parameter for Normal distribution for error terms
            self.normalLocation = 0
            #Factor for controlling inclusion of correlation terms
            self.delta = 1

            
            #Initial number of charging outlets at each station
            self.x0 = [j["StartingOutlets"] for j in self.stations]
            #Binary indicating whether each station was originally open or not
            self.y0 = [1 if j>0 else 0 for j in self.x0]

            #Number of simulations per option for the user class
            self.R = 15

            #Budget for each year
            self.B = [400, 400, 400, 400]

            #Maximum number of outlets at each station
            self.Mj = [j["MaxOutlets"]+1 for j in self.stations]

            #Cost for opening each charging station
            self.f = 100

            #Cost for installing each charging outlet
            self.c = 50

            #If given specific parameters, update the defaults
            if params:
                self.__dict__.update(params)

            #Updating defaults can cause errors, depending on what values are included and how,
            #so we update certain other values accordingly
            #As before, we ensure choice sets are set as arrays that depend on the year
            if len(np.shape(self.userClasses[0]["OptOutChoiceSet"])) == 1:
                self.C0i = [[list(set(i["OptOutChoiceSet"])) for i in self.userClasses] for t in range(self.T)]
                self.C1i = [[list(set([int(j) for j in i["StationChoiceSet"]])) for i in self.userClasses] for t in range(self.T)]
            else:
                self.C0i = [[list(set(i["OptOutChoiceSet"][t])) for i in self.userClasses] for t in range(self.T)]
                self.C1i = [[list(set([int(j) for j in i["StationChoiceSet"][t]])) for i in self.userClasses] for t in range(self.T)]               
            self.R = [self.R*(len(self.C0i[0][i]) + len(self.C1i[0][i])) for i in range(len(self.C0i[0]))]
            self.f = [[self.f for j in range(self.M)] for t in range(self.T)]
            self.c = [[self.c for j in range(self.M)] for t in range(self.T)]


            self.CalculateCoefficients()
            self.CalculateErrorTerms()
            self.CreateCovering()

            end=time()
            self.DataCreationTime = end-start
        else:
            if load is not None:
                self.load(load)
            if unpickle is not None:
                self.unpickle(unpickle)
    
    #Calculate the utility coefficients beta
    def CalculateCoefficients(self):
        self.beta = [ [ [ [np.nan for k in range(self.Mj[j])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        for t in range(self.T):
            for i in range(self.N):
                for j in self.C1i[t][i]:
                    self.beta[t][j][i] = [self.userClasses[i]["StationCoefficients"][str(j)][k] for k in range(self.Mj[j])]

    #Draw the error terms epsilon
    #These are combined with the alternative specific constants  to create the d0 and d1 variables
    def CalculateErrorTerms(self):
        self.d0 = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(2)] for t in range(self.T)]
        self.d1 = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        gen=np.random.default_rng()
        try:
            for t in range(self.T):
                for i in range(self.N):
                    nOptions = len(self.userClasses[i]["FactorOrder"])
                    nFactors = len(self.userClasses[i]["FactorScale"])
                    factormatrix = self.delta*np.matmul(np.array(self.userClasses[i]["FactorLoading"]),np.diag(self.userClasses[i]["FactorScale"]))
                    errorsNormal = gen.normal(self.normalLocation, self.normalScale, size=(self.R[i], nFactors))
                    errorsGumbel = gen.gumbel(self.gumbelLocation, self.gumbelScale, self.R[i]*nOptions)
                    k=0
                    for r in range(self.R[i]):
                        for j in range(nOptions):
                            jBar = self.userClasses[i]["FactorOrder"][j]
                            #Factor Order contains an array with the choice set for that class, but we need to separate out
                            #the exogenous options before checking for stations
                            if jBar == "OptOut":                
                                self.d0[t][0][i][r] = self.userClasses[i]["OptOutConstants"][str(0)]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                            elif jBar == "Home":                
                                self.d0[t][1][i][r] = self.userClasses[i]["OptOutConstants"][str(1)]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                            else:
                                jBar = int(jBar)
                                if len(np.shape(self.userClasses[i]["StationConstants"])) == 1:
                                    self.d1[t][jBar][i][r] = self.userClasses[i]["StationConstants"][jBar]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                                else:
                                    self.d1[t][jBar][i][r] = self.userClasses[i]["StationConstants"][t][jBar]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]                                

                            k+=1
        except:
            print('t:',t)
            print('j:',j)
            print('i:',i)
            print('r:',r)
            print('jBar:',jBar)
            raise Exception

    #Calculate the coverings for each station given the error terms
    #This calculates the vector 'a' for the model, but also stores self.Covering
    #and self.Populations for faster computation of solution values in SolutionQuality()
    def CreateCovering(self):
        self.a = [ [ [ [ [ 0 for r in range(self.R[i])] for k in range(self.Mj[j])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        self.HC_covering = [ [ [ 0 for r in range(self.R[i])] for i in range(self.N)]  for t in range(self.T)]
        self.Covering = {t:pd.DataFrame() for t in range(self.T)}
        Utilities = {t:pd.DataFrame() for t in range(self.T)}
        self.Populations = pd.DataFrame()
        for t in range(self.T):
            Utilities[t]['UserClass'] = [i for i in range(self.N) for r in range(self.R[i])]
            Utilities[t]['Scenario'] = [r for i in range(self.N) for r in range(self.R[i])]
            Utilities[t]['OptOut'] = [self.d0[t][0][i][r] for i in range(self.N) for r in range(self.R[i])]
            #Sloppy way to check this, but currently only time they'll be more than one option is 
            #if home charging is available
            if len(self.d0[t]) > 1:
                Utilities[t]['Home'] = [self.d0[t][1][i][r] for i in range(self.N) for r in range(self.R[i])]
                self.Covering[t]['Home'] = Utilities[t]['Home'] >= Utilities[t]['OptOut']
            else:
                self.Covering[t]['Home'] = [0 for i in range(self.N) for r in range(self.R[i])]
            for row in zip(Utilities[t]['UserClass'], Utilities[t]['Scenario'], self.Covering[t]['Home']):
                i = row[0]
                r = row[1]
                if row[2]:
                    self.HC_covering[t][i][r] = 1             
            for j in range(self.M):
                    Station=np.array([self.d1[t][j][i][r] for i in range(self.N) for r in range(self.R[i])])
                    for k in range(1,self.Mj[j]):
                            Utilities[t][(j,k)] = Station + np.array([self.beta[t][j][i][k] for i in range(self.N) for r in range(self.R[i])])
                            Utilities[t][(j,k)] = Utilities[t][(j,k)] >= Utilities[t]['OptOut']
                            self.Covering[t][(j,k)] = Utilities[t][(j,k)]
                            for row in zip(Utilities[t]['UserClass'], Utilities[t]['Scenario'], Utilities[t][(j,k)]):
                                i = row[0]
                                r = row[1]
                                if row[2]:
                                    self.a[t][j][i][k][r] = 1
                            
            self.Populations[t] = np.array([self.Ni[t][i]/self.R[i] for i in range(self.N) for r in range(self.R[i])])

        

    #Calculate objective value (in maximum covering model) for a given solution
    def SolutionQuality(self, Solution):
        if Solution is None:
            return 0
        TotalEVs=0
        if type(Solution) == dict:
            Solution = self.ConvertToArray(Solution)
        for t in range(self.T):
            cols = []
            for j in range(self.M):
                if Solution[t][j] > 0:
                    cols.append( (j, Solution[t][j]) )
            subCovering = self.Covering[t][cols]
            subCovering['Home'] = self.Covering[t]['Home']
            TotalEVs += subCovering.any(axis=1).dot(self.Populations[t])
        return TotalEVs

    #Shortcut to compare objective value with single-level model
    def SolutionQuality_Minimise(self, Solution):
        return sum(self.Ni[t] for t in range(self.T)) - self.SolutionQuality(Solution)

    #Convert given solution from dict form to array form for use in SolutionQuality()
    def ConvertToArray(self, Solution):
            new = np.zeros((self.T, self.M))
            if (0,0) in Solution:
                for t in range(self.T):
                        for j in range(self.M):
                                new[t][j] = Solution[(t,j)]
            elif (0,0,0) in Solution:
                for t in range(self.T):
                        for j in range(self.M):
                                new[t][j] = sum([k*Solution[(t,j,k)] for k in range(self.Mj[j])])              
            return new
    
    #Keep parameter values, but remove error terms
    def ClearErrorTerms(self):
        print("Clearing error terms")
        del self.d0
        del self.d1
        del self.a

    #Dump parameter values to json file
    def dump(self,f):
        info = {
            'userFilepath':self.userFilepath,
            'stationFilepath':self.stationFilepath,
            'T':self.T,  
            'n_v':self.n_v,
            'M':self.M,
            'N':self.N,
            'Ni':self.Ni,
            'gumbelScale':self.gumbelScale,
            'gumbelLocation':self.gumbelLocation,
            'normalScale':self.normalScale,
            'normalLocation':self.normalLocation,
            'delta':self.delta,
            'x0':self.x0,
            'y0':self.y0,
            'R':self.R,
            'B':self.B,
            'Mj':self.Mj,
            'f':self.f,
            'c':self.c,
            'd0':self.d0,
            'd1':self.d1,
            'C0i':self.C0i,
            'C1i':self.C1i,
            'beta':self.beta,
            'a':self.a,
            'HC_covering': self.HC_covering
            }

        json.dump(info, open(f, "w"))

    #Load parameter values from json file
    def load(self,f):
        data = json.load(open(f,"r"))
        self.__dict__.update(data)
        if len(np.shape(data["Ni"])) == 1:
            self.Ni = [data["Ni"] for t in range(self.T)]

    #Dump parameter values to pickle file
    def pickle(self,f):
        info = {
            'userFilepath':self.userFilepath,
            'stationFilepath':self.stationFilepath,
            'T':self.T,  
            'n_v':self.n_v,
            'M':self.M,
            'N':self.N,
            'Ni':self.Ni,
            'gumbelScale':self.gumbelScale,
            'gumbelLocation':self.gumbelLocation,
            'normalScale':self.normalScale,
            'normalLocation':self.normalLocation,
            'delta':self.delta,
            'x0':self.x0,
            'y0':self.y0,
            'R':self.R,
            'B':self.B,
            'Mj':self.Mj,
            'f':self.f,
            'c':self.c,
            'd0':self.d0,
            'd1':self.d1,
            'C0i':self.C0i,
            'C1i':self.C1i,
            'beta':self.beta
            }

        pickle.dump(info, open(f, "wb"))

    #Load parameter values from pickle file
    def unpickle(self,f):
        data = pickle.load(open(f, "rb"))
        self.__dict__.update(data)
        if len(np.shape(data["Ni"])) == 1:
            self.Ni = [data["Ni"] for t in range(self.T)]


