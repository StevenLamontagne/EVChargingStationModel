import json
from math import atan, exp
import numpy as np
import pandas as pd
from time import time
import pickle
from os import path



class Data_MaximumCover():
    def __init__(self, userFilepath = None, stationFilepath = None, params=None, load = None, unpickle = None, load_compressed = None):
        if (userFilepath is None or stationFilepath is None) and (load is None) and (unpickle is None) and (load_compressed is None):
            raise Exception("Unable to load or create data. Ensure a file is provided to load or that a user filepath and station filepath is provided.")
        if (load is None and unpickle is None and load_compressed is None):
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
            #self.n_v=0.053
            self.n_v=0.1
            
            #Number of charging stations
            self.M=len(self.stations)

            #Number of user classes
            self.N=len(self.userClasses)

            #Population of each class
            if type(self.userClasses[0]["Population"]) == float:
                self.Ni = [[int(self.n_v*i["Population"]) for i in self.userClasses] for t in range(self.T)]
            else:
                self.Ni = [[int(self.n_v*i["Population"][t]) for i in self.userClasses] for t in range(self.T)]

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

            
            #Initial number of charging outlets at each station
            self.x0=[j["StartingOutlets"] for j in self.stations]
            #Binary indicating whether each station was originally open or not
            self.y0=[1 if j>0 else 0 for j in self.x0]

            #Number of simulations per option for the user class
            self.R=15

            #Budget for each year
            self.B=[400, 400, 400, 400]

            #Maximum number of outlets at each station
            self.Mj=[j["MaxOutlets"]+1 for j in self.stations]
            #self.Mj=[2+1 for j in self.stations]

            #Cost for opening each charging station
            self.f=100

            #Cost for installing each charging outlet
            self.c=50

            #If given specific parameters, update the defaults
            if params:
                self.__dict__.update(params)

            #Calculate full values for the parameters after updating defaults
            if len(np.shape(self.userClasses[0]["OptOutChoiceSet"])) == 1:
                self.C0i=[[list(set(i["OptOutChoiceSet"])) for i in self.userClasses] for t in range(self.T)]
                self.C1i=[[list(set([int(j) for j in i["StationChoiceSet"]])) for i in self.userClasses] for t in range(self.T)]
            else:
                self.C0i=[[list(set(i["OptOutChoiceSet"][t])) for i in self.userClasses] for t in range(self.T)]
                self.C1i=[[list(set([int(j) for j in i["StationChoiceSet"][t]])) for i in self.userClasses] for t in range(self.T)]               
            self.R=[self.R*(len(self.C0i[0][i]) + len(self.C1i[0][i])) for i in range(len(self.C0i[0]))]
            self.f=[[self.f for j in range(self.M)] for t in range(self.T)]
            self.c=[[self.c for j in range(self.M)] for t in range(self.T)]


            self.CalculateCoefficients()
            self.CalculateErrorTerms()
            self.CreateCovering()

            end=time()
            self.DataCreationTime=end-start
        else:
            if load is not None:
                self.load(load)
            if unpickle is not None:
                self.unpickle(unpickle)
            if load_compressed is not None:
                self.load_compressed(load_compressed[0], load_compressed[1])
    
    #Calculate appropriate beta coefficients for each station and outlet configuration
    def CalculateCoefficients(self):
        self.beta=[ [ [ [np.nan for k in range(self.Mj[j])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        for t in range(self.T):
            for i in range(self.N):
                for j in self.C1i[t][i]:
                    self.beta[t][j][i] = [self.userClasses[i]["StationCoefficients"][str(j)][k] for k in range(self.Mj[j])]

    #Generate the error terms for exogenous and endogenous alternatives
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
                                if len(np.shape(self.userClasses[i]["StationConstants"])) == 1:
                                    self.d1[t][jBar][i][r]=self.userClasses[i]["StationConstants"][jBar]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]
                                else:
                                    self.d1[t][jBar][i][r]=self.userClasses[i]["StationConstants"][t][jBar]+np.dot(factormatrix[j], errorsNormal[r])+errorsGumbel[k]                                

                            k+=1
        except:
            print('t:',t)
            print('j:',j)
            print('i:',i)
            print('r:',r)
            print('jBar:',jBar)
            raise Exception

    #Calculate the coverage based on the error terms. Stored as Pandas dataframes for faster computing while solving
    def CreateCovering(self):
        #a[t][j][i][k][r]
        self.a = [ [ [ [ [ 0 for r in range(self.R[i])] for k in range(self.Mj[j])] for i in range(self.N)] for j in range(self.M)] for t in range(self.T)]
        self.HC_covering = [ [ [ 0 for r in range(self.R[i])] for i in range(self.N)]  for t in range(self.T)]
        self.Covering = {t:pd.DataFrame() for t in range(self.T)}
        Utilities = {t:pd.DataFrame() for t in range(self.T)}
        self.Populations = pd.DataFrame()
        for t in range(self.T):
            Utilities[t]['UserClass'] = [i for i in range(self.N) for r in range(self.R[i])]
            Utilities[t]['Scenario'] = [r for i in range(self.N) for r in range(self.R[i])]
            Utilities[t]['OptOut'] = [self.d0[t][0][i][r] for i in range(self.N) for r in range(self.R[i])]
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

    #Helper function for identifying user class and scenario for triplets
    def CreateCoveringIndices(self):
        df = pd.DataFrame()
        df['UserClass'] = [i for i in range(self.N) for r in range(self.R[i])]
        df['Scenario'] = [r for i in range(self.N) for r in range(self.R[i])]
        self.CoveringIndices = df
        

    #Calculates the coverage of a given solution
    def SolutionQuality(self, Solution):
        if Solution is None:
            return 0
        TotalEVs = 0
        if type(Solution) == dict:
            Solution = self.ConvertToArray(Solution)
        for t in range(self.T):
            cols = []
            for j in range(self.M):
                if Solution[t][j] > 0:
                    for k in range(1,Solution[t][j] + 1):
                        cols.append( (j, k) )
            subCovering = self.Covering[t][cols].copy()
            subCovering['Home'] = self.Covering[t]['Home']
            TotalEVs += subCovering.any(axis=1).dot(self.Populations[t])
        return TotalEVs

    #Helper function for comparing with Single-Level formulation
    def SolutionQuality_Minimise(self, Solution):
        return sum(self.Ni[t] for t in range(self.T)) - self.SolutionQuality(Solution)

    #Converting from dict-style to list-style solution representations
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
    
    #Generates a single error term (deprecated)
    def GenerateError(self):
        return float(np.random.gumbel(self.gumbelLocation,self.gumbelScale,1))
    
    #Used for deleting error terms when running multiple tests that differ only in error terms
    def ClearErrorTerms(self):
        print("Clearing error terms")
        del self.d0
        del self.d1
        del self.a

    #Stores the data in a json format. Includes additional metadata unnecessary for solving, but used for data creation
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

    #Load data stored in json format
    def load(self,f):
        data = json.load(open(f,"r"))
        self.__dict__.update(data)
        if len(np.shape(data["Ni"])) == 1:
            self.Ni = [data["Ni"] for t in range(self.T)]

    #Compress the data in a pickle format. Includes additional metadata unnecessary for solving, but used for data creation
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

    #Loads uncompressed data stored in pickle format
    def unpickle(self,f):
        data = pickle.load(open(f, "rb"))
        self.__dict__.update(data)
        if len(np.shape(data["Ni"])) == 1:
            self.Ni = [data["Ni"] for t in range(self.T)]

    #Compress the data in a minimal pickle format (not compressed). Data can be loaded via unpickle function
    def pickle_minimal(self,f):
        info = {
            'T':self.T,  
            'n_v':self.n_v,
            'M':self.M,
            'N':self.N,
            'Ni':self.Ni,
            'x0':self.x0,
            'y0':self.y0,
            'R':self.R,
            'B':self.B,
            'Mj':self.Mj,
            'f':self.f,
            'c':self.c,
            'a':self.a
            }
        pickle.dump(info, open(f, "wb"))


    #Modify variables (in place) for use with the reformulated models
    def Process_Reformulation(self):
        c = [[[0 for _ in range(self.Mj[j])] for j in range(self.M)] for _ in range(self.T)]
        for t in range(self.T):
            for j in range(self.M):
                c[t][j][1] = self.c[t][j] + self.f[t][j]
                for k in range(2, self.Mj[j]):
                   c[t][j][k] = self.c[t][j] 
        self.c = c
        
        x0 = [[0 for _ in range(self.Mj[j])] for j in range(self.M)]
        for j in range(self.M):
            for k in range(self.Mj[j]):
                if self.x0[j] >= k:
                    x0[j][k] = 1
                else:
                    x0[j][k] = 0
        self.x0 = x0

        #Transform coverages for new variables (only count coverage as *first* time a triplet is covered)
        for t in range(self.T):
            for j in range(self.M):
                for k in range(2,self.Mj[j]):
                    self.Covering[t][(j,k)] = self.Covering[t][(j,k)] ^ self.Covering[t][(j,k-1)]                     

    #Dump the data in a compressed format. Can be loaded using the load_compressed function
    def dump_compressed(self, fp, test):
        user_coord = [(i,r) for i in range(self.N) for r in range(self.R[i])]
        station_coord = [(j,k) for j in range(self.M) for k in range(1, self.Mj[j])]
        convert_user = {user_coord[k]:k for k in range(len(user_coord))}

        if not path.exists(fp+'/Shared.json'):
            info = {
                'T':self.T,  
                'M':self.M,
                'N':self.N,
                'Ni':self.Ni,
                'x0':self.x0,
                'R':self.R,
                'B':self.B,
                'Mj':self.Mj,
                'c':self.c,
                'user_coord': user_coord,
                'station_coord': station_coord
                }
            json.dump(info, open(fp+'/Shared.json', "w"))

        #Convert coverage to a coordinate sparse format
        a = [[] for _ in range(self.T)]
        for t in range(self.T):
            for id_user in range(len(user_coord)):
                i,r = user_coord[id_user]
                for id_station in range(len(station_coord)):
                    j,k = station_coord[id_station]
                    if self.a[t][j][i][k][r] and (k == 1 or self.a[t][j][i][k-1][r] == 0):
                        a[t].append((id_user,id_station))

        home = [[] for _ in range(self.T)]
        for t in range(self.T):
            df = pd.DataFrame({"Home":self.Covering[t]["Home"]})
            df['UserClass'] = [i for i in range(self.N) for r in range(self.R[i])]
            df['Scenario'] = [r for i in range(self.N) for r in range(self.R[i])]
            for i, r, cover in zip(df["UserClass"], df["Scenario"], df["Home"]):
                if cover:
                    id = convert_user[(i,r)]
                    home[t].append(id)
        
        info = {
            'Home':home,
            'a':a
        }
        json.dump(info, open(fp+'/MC{}_compressed.json'.format(test), "w"))

    #Load data from the compressed files. Requires the appropriate Shared.json for the dataset, as well as the JSON file for the instance
    def load_compressed(self, fp, test):
        shared = json.load(open(fp+'/Shared.json'))
        self.__dict__.update(shared)

        #a[t][j][i][k][r]
        self.a = [ [ [ [ [ 0 for r in range(self.R[i])] 
        for k in range(self.Mj[j])]
        for i in range(self.N)] 
        for j in range(self.M)] 
        for t in range(self.T)]

        info = json.load(open(fp+'/MC{}_compressed.json'.format(test)))
        for t in range(self.T):
            for id_user,id_station in info['a'][t]:
                i,r = self.user_coord[id_user]
                j,k = self.station_coord[id_station]
                self.a[t][j][i][k][r] = 1

        self.HC_covering = [ [ [ 0 for r in range(self.R[i])] for i in range(self.N)]  for t in range(self.T)]
        for t in range(self.T):
            for id_user in info['Home'][t]:
                i,r = self.user_coord[id_user]
                self.HC_covering[t][i][r] = 1

        self.Covering = [pd.DataFrame() for _ in range(self.T)]
        self.Populations = [pd.Series(dtype = 'float64') for _ in range(self.T)]
        for t in range(self.T):
            self.Covering[t]["Home"] = np.array([self.HC_covering[t][i][r] for i in range(self.N) for r in range(self.R[i])])
            for j in range(self.M):
                for k in range(1,self.Mj[j]):       
                    self.Covering[t][(j,k)] = np.array([self.a[t][j][i][k][r] for i in range(self.N) for r in range(self.R[i])])
                    if k > 1:
                        self.Covering[t][(j,k)] = self.Covering[t][(j,k)] | self.Covering[t][(j,k-1)]
            self.Populations[t] = pd.Series(np.array([self.Ni[t][i]/self.R[i] for i in range(self.N) for r in range(self.R[i])]))
                



