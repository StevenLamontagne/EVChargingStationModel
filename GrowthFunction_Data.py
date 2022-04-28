import json
import numpy as np


#Class for storing segment information
class Segment():
    def __init__(self, parameters):
        self.startpoint=parameters['Startpoint']
        self.endpoint=parameters['Endpoint']
        self.intercept=parameters['Intercept']
        self.slope=parameters['Slope']
    
    def to_dict(self):
        return {
            'Startpoint':self.startpoint,
            'Endpoint':self.endpoint,
            'Intercept':self.intercept,
            'Slope':self.slope
        }



                 
#Class for storing the data for the model. Data can be set internally or via external json file
class Data_GrowthFunction():
    def __init__(self, GrowthFunction = None, stationFilepath = None, params = None, load = None):
        if (GrowthFunction is None or stationFilepath is None) and (load is None):
            raise Exception("Unable to load or create data. Ensure a file is provided to load or that a user filepath and station filepath is provided.")

        if load is None:
            self.Network = json.load(open("Data/TroisRivieres/TroisRivieresNetwork.json", "r"))
            self.stationFilepath = stationFilepath
            self.stations=json.load(open(stationFilepath, "r"))
            #Populations at each node
            self.Ri = list(json.load(open("Data/TroisRivieres/ListNodes.json","r")).values()) #Equivalent to SimpleClass.json
            with open("Data/GrowthFunctions/"+GrowthFunction, "r") as f:
                piecewiseSegments=json.load(f)
                piecewiseSegments=[piecewiseSegments[str(i)] for i in range(len(piecewiseSegments))]

            
            #Number of years for the simulation
            self.T=4

            #Number of nodes
            self.N=self.Network["NumberOfNodes"]

            #Number of stations
            self.M = len(self.stations)

            #Maximum number of outlets at each station
            self.Mj=[j["MaxOutlets"] for j in self.stations]
            
            #Maximum acceptable travel distance for users
            self.maxDistance = 10000
            
            #Total population
            self.R = sum(self.Ri)

            #Fraction of EV owners (versus CV owners)
            self.ev = 0.05

            #Fraction of users who own EV chargers
            self.alpha=0.566365
    
            #Maximum number of users each outlet can support
            self.a=[45*(1.1**t) for t in range(self.T)] 

            #Initial number of charging outlets at each station
            self.x0=[j["StartingOutlets"] for j in self.stations]
            #Binary indicating whether each station was originally open or not
            self.y0=[1 if j>0 else 0 for j in self.x0]
            
            #Cost for adding another charging outlet at each station
            self.c=[[50 for j in range(self.M)] for t in range(self.T)]
            #Cost to open up each station
            self.f=[[100 for j in range(self.M)] for t in range(self.T)]
            #Budget for each year
            self.B=[400,400,400,400]

            #Initial demand for EVs
            self.initialEVs=[0.0137*self.Ri[i]  for i in range(self.N)]
            
            #If given specific parameters, update the defaults
            if params:
                self.__dict__.update(params)

            keys = list(self.Network["NodePositions"].keys())
            ID = {keys[k]:k for k in range(len(keys))}
            del keys
            #Nodes within travel range for each node
            self.Ni = [[j for j in range(self.M) if self.Network["ShortestPaths"][i][ID[self.stations[j]["Location"]]]<=self.maxDistance] for i in range(self.N)]

            #Set of segments
            for line in piecewiseSegments:
                line['Startpoint']*=self.R
                line['Endpoint']*=self.R
                line['Intercept']*=self.R
            self.S=[Segment(line) for line in piecewiseSegments]

            #Number of segments in the piecewise linear function
            self.s=len(self.S)
        else:
            self.load(load)

    #Dump settings to json file
    def dump(self,f):
        info = {
            'stationFilepath':self.stationFilepath,
            'T':self.T,  
            'M':self.M,
            'N':self.N,
            'Ri':self.Ri,
            'maxDistance':self.maxDistance,
            'alpha':self.alpha,
            'x0':self.x0,
            'y0':self.y0,
            'B':self.B,
            'Mj':self.Mj,
            'f':self.f,
            'c':self.c,
            'a':self.a,
            'R':self.R,
            'ev':self.ev,
            'Ni':self.Ni,
            'initialEVs':self.initialEVs,
            'S':[s.to_dict() for s in self.S],
            's':self.s
            }

        json.dump(info, open(f, "w"))

    #Read settings from json file
    def load(self,f):
        data = json.load(open(f,"r"))
        self.S = [Segment(s) for s in data['S']]
        del data['S']
        self.__dict__.update(data)