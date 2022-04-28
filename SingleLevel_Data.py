import json
from math import atan, exp
import numpy as np
import pandas as pd
from time import time


class Data_SingleLevel():
    def __init__(self, userFilepath = None, stationFilepath = None, params=None, load = None):
        if (userFilepath is None or stationFilepath is None) and (load is None):
            raise Exception("Unable to load or create data. Ensure a file is provided to load or that a user filepath and station filepath is provided.")
        if load is None:
            start=time()
            self.network=json.load(open("Data/Network/TroisRivieresNetwork.json", "r"))
            self.stations=json.load(open(stationFilepath, "r"))
            self.userClasses=json.load(open(userFilepath, "r"))
            self.userFilepath = userFilepath
            self.stationFilepath = stationFilepath

            self.Preprocess_w1= None
            
            #Number of years for the simulation
            self.T=4

            #Percentage of population that is considering purchasing a vehicle in each year
            self.n_v=0.1
            
            #Number of charging stations
            self.M=len(self.stations)

            #Number of user classes
            self.N=len(self.userClasses)

            #Population of each class
            self.Ni=[int(self.n_v*i["Population"]) for i in self.userClasses]

            #Scale parameter for Gumbel distribution for error terms
            self.gumbelScale=3
            #Location parameter for Gumbel distribution for error terms
            self.gumbelLocation=0
            #Scale parameter for Normal distribution for error terms
            self.normalScale=1
            #Location parameter for Normal distribution for error terms
            self.normalLocation=0
            #Factor for controlling inclusion of correlation terms
            self.delta=1

            #Method for single-level reduction of the model
            self.lowerLevel='cs'
            
            #Initial number of charging outlets at each station
            self.x0=[j["StartingOutlets"] for j in self.stations]
            #Binary indicating whether each station was originally open or not
            self.y0=[1 if j>0 else 0 for j in self.x0]

            #Number of simulations per option for the user class
            #R_i = R*(|C_i^0|+ |C_i^1|)
            self.R = 15

            #Budget for each year
            self.B=[550,300,250,500]

            #Maximum number of outlets at each station
            self.Mj=[j["MaxOutlets"]+1 for j in self.stations]


            #Cost for opening each charging station
            self.f=100

            #Cost for installing each charging outlet
            self.c=50





            #If given specific parameters, update the defaults
            if params:
                self.__dict__.update(params)

            #Calculate full values for the parameters after updating defaults
            self.C0i=[[list(set(i["OptOutChoiceSet"])) for i in self.userClasses] for t in range(self.T)]
            self.C1i=[[list(set([int(j) for j in i["StationChoiceSet"]])) for i in self.userClasses] for t in range(self.T)]
            self.R=[self.R*(len(self.C0i[0][i]) + len(self.C1i[0][i])) for i in range(len(self.C0i[0]))]
            self.f=[[self.f for j in range(self.M)] for t in range(self.T)]
            self.c=[[self.c for j in range(self.M)] for t in range(self.T)]



            self.CalculateCoefficients()
            self.CalculateErrorTerms()
            self.CalculateBounds()
            self.CreateCovering()

            end=time()
            self.DataCreationTime=end-start
        else:
            self.load(load)
            
    #Calculate the utility coefficients beta  
    def CalculateCoefficients(self):
        self.beta = [ [ [ [np.nan for k in range(self.Mj[j])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        for t in range(self.T):
            for i in range(self.N):
                for j in self.C1i[t][i]:
                    for k in range(self.Mj[j]):
                        #Current calculation for utility uses a diminishing utility formulation in terms of the number of charging outlets
                        self.beta[t][j][i][k]= self.userClasses[i]["StationCoefficients"][str(j)][k]

    #Draw the error terms epsilon
    #These are combined with the alternative specific constants  to create the d0 and d1 variables
    def CalculateErrorTerms(self):
        self.d0 = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(2)] for t in range(self.T)]
        self.d1 = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        gen=np.random.default_rng()
        try:
            for t in range(self.T):
                for i in range(self.N):
                    nOptions=len(self.userClasses[i]["FactorOrder"])
                    nFactors=len(self.userClasses[i]["FactorScale"])
                    factormatrix=self.delta*np.matmul(np.array(self.userClasses[i]["FactorLoading"]),np.diag(self.userClasses[i]["FactorScale"]))
                    errorsNormal=gen.normal(self.normalLocation, self.normalScale, size=(self.R[i], nFactors))
                    errorsGumbel=gen.gumbel(self.gumbelLocation, self.gumbelScale, self.R[i]*nOptions)
                    k=0
                    for r in range(self.R[i]):
                        for j in range(nOptions):
                            jBar=self.userClasses[i]["FactorOrder"][j]
                            if jBar == "OptOut":                
                                self.d0[t][0][i][r]=self.userClasses[i]["OptOutConstants"][str(0)]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                            elif jBar == "Home":                
                                self.d0[t][1][i][r]=self.userClasses[i]["OptOutConstants"][str(1)]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                            else:
                                jBar=int(jBar)
                                self.d1[t][jBar][i][r]=self.userClasses[i]["StationConstants"][jBar]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                            k+=1

        except Exception as e:
            print("t: ", t)
            print("i: ", i)
            print("r: ", r)
            print("j: ", j)
            print("jBar: ", jBar)
            print(e)
            raise Exception
                
    #Calculate the upper and lower bounds for the utility
    #(using the error terms)
    def CalculateBounds(self):
        #Calculate bounds for constraints
        self.aBar = [ [np.nan for i in range(self.N)] for t in range(self.T)]
        self.b = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        self.MBar = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        self.mu0 = [ [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for j in range(2)] for t in range(self.T)]
        self.mu =  [ [ [np.nan for r in range(self.R[i])] for i in range(self.N)] for t in range(self.T)]
        for t in range(self.T):
            for i in range(self.N):
                if len([self.d1[t][int(j)][i][0] for j in self.C1i[t][i]]) == 0:
                    self.aBar[t][i] = -np.inf
                else:
                    self.aBar[t][i] = min(min(self.d1[t][int(j)][i][r] for j in self.C1i[t][i]) for r in range(self.R[i]))
                for r in range(self.R[i]):
                    for j in self.C1i[t][i]:
                        j = int(j)
                        self.b[t][j][i][r] = self.beta[t][j][i][self.Mj[j]-1]+self.d1[t][j][i][r]
                        self.MBar[t][j][i][r] = self.b[t][j][i][r]-self.aBar[t][i]
                    if len([self.b[t][int(j)][i][0] for j in self.C1i[t][i]]) == 0:
                        alpha = max(self.d0[t][int(j)][i][r] for j in self.C0i[t][i])
                    else:
                        alpha=max(max(self.b[t][int(j)][i][r] for j in self.C1i[t][i]), max(self.d0[t][int(j)][i][r] for j in self.C0i[t][i]))
                    for j in self.C0i[t][i]:
                        j = int(j)
                        self.mu0[t][j][i][r] = alpha-self.d0[t][j][i][r]
                    self.mu[t][i][r] = alpha-self.aBar[t][i]

    #Clear all error terms and derived bounds
    # For generating new instance without recreating all parameter  
    def ClearErrorTerms(self):
        print("Clearing error terms")
        del self.d0
        del self.d1
        del self.a0
        del self.a1
        del self.b0
        del self.b1
        del self.U

    #Dump parameter values to json file
    #Since instance sizes are small, we include bounds in the json file as well
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
            'beta':self.beta,
            'd0':self.d0,
            'd1':self.d1,
            'C0i':self.C0i,
            'C1i':self.C1i,
            'aBar':self.aBar,
            'b':self.b,
            'MBar':self.MBar,
            'mu0':self.mu0,
            'mu':self.mu
            }

        if self.Preprocess_w1 is not None:
            info['Preprocess_stats']=self.Preprocess_stats
        json.dump(info, open(f, "w+"), indent=3)
    
    #Read parameter values from json file
    def load(self, f):
        data = json.load(open(f,"r"))
        self.__dict__.update(data)

    #Function to determine if any w1 variables can be fixed to 0, 
    # based on if the highest possible utility (with maximum outlets)
    # is still less than the opt-out
    def Preprocess(self):
        tic1=time()
        self.Preprocess_w1=[]
        self.Preprocess_stats={'Fixed w1':0, 'Option Removed':0, 'Time':0}
        for t in range(self.T):
            Utilities = pd.DataFrame()
            Utilities['OptOut'] = [self.d0.get((t,0,i,r), np.nan) for i in range(self.N) for r in range(self.R[i])]
            Utilities['Home'] = [self.d0.get((t,1,i,r), np.nan) for i in range(self.N) for r in range(self.R[i])]
            Utilities['Exo'] = Utilities.max(axis=1)

            Utilities['UserClass'] = [i for i in range(self.N) for r in range(self.R[i])]
            Utilities['Scenario'] = [r for i in range(self.N) for r in range(self.R[i])]

            del Utilities['OptOut']
            del Utilities['Home']

            for j in range(self.M):
                Station = np.array([self.d1.get((t,j,i,r), np.nan) for i in range(self.N) for r in range(self.R[i])])
                Station += np.array([self.beta.get((t,j, i, self.Mj[j]-1), np.nan) for i in range(self.N) for r in range(self.R[i])])
                Utilities['Station'] = Station

                toRemove = Utilities[Utilities['Exo'] > Utilities['Station']]

                for i in range(self.N):
                    subUtilities=toRemove[toRemove['UserClass'] == i]
                    if j in self.C1i[t][i] and len(subUtilities) == self.R[i]:
                        self.C1i[t][i].remove(j)
                        self.Preprocess_stats['Option Removed'] += 1
                    elif j in self.C1i[t][i]:
                        self.Preprocess_w1 += [(t,j,i,r) for r in subUtilities['Scenario']]
                        self.Preprocess_stats['Fixed w1'] += len(subUtilities)
        toc1=time()
        self.Preprocess_stats['Time']=toc1-tic1
        print("Preprocessing time (seconds): ", self.Preprocess_stats['Time'])
                        
     
                
def Test():
    userFilename="Data/TroisRivieres/Default/UserClasses_Rand28_Stations10.json"
    stationFilename="Data/TroisRivieres/Default/Stations_Rand10.json"
    Data = Data_SingleLevel(userFilename, stationFilename)
    return Data

def Test_Adv():
    userFilename = "Data/TroisRivieres/Heuristic/UserClassesNested_Rand28_Stations30.json"
    stationFilename = "Data/TroisRivieres/Heuristic/Stations_Rand30.json"
    Data = Data_SingleLevel(userFilename,
                          stationFilename,
                          {'B':[900, 900, 900, 900, 900, 900, 900, 900, 900, 900], 'T':10, 'Mj':[6 for _ in range(30)], 'R':5})
    return Data
