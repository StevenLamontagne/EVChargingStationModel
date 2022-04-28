import json
from random import sample
import pandas as pd
import numpy as np

nStations = 30
stationsFilename = "Stations_Price2.json"
simpleFilename = "UserClasses_Simple.json"
distanceFilename = "UserClasses_Distance.json"
homechargingFilename = "UserClasses_HomeCharging.json"

maxDistance = 10000
beta_L3   = 1.463601        
beta_dist = -0.062626       
beta_cc   = 0.173505        
beta_chargers= 0.281135     
def OutletUtility(beta, mj):
    return [beta*k for k in range(mj+1)]

'''
#Defaults
beta_L3   = 1.463601        
beta_dist = -0.062626       
beta_cc   = 0.173505        
beta_chargers= 0.281135     
def OutletUtility(beta, mj):
    return [beta*k for k in range(mj+1)]
'''


def read_json(fp):
    return json.load(open(fp,"r"))


nodes = read_json('ListNodes.json')
network = read_json('TroisRivieresNetwork.json')
try:
    stations = read_json(stationsFilename)
    dumpStations = False
except:   
    stations = []
    nodelist= list(nodes.keys())
    #stationLocations = sample(list(nodes.keys()), k = nStations)
    stationLocations = set(["24370042","24370074","24370115","24370163","24370233","24370275","24370298","24380045","24380064","24510065"])
    while len(stationLocations) < nStations:
        stationLocations.update(sample(nodelist, k = nStations-len(stationLocations)))
    stationLocations = list(stationLocations)
    for j in range(nStations):
        stations.append({"Location":stationLocations[j],
                        "MaxOutlets": 6,
                        "isLevel3": True,
                        "StartingOutlets": 0
                        })
    dumpStations=True

locations = [network["IDs"][j["Location"]] for j in stations]


distance = []
homecharging = []
# for maxDistance, beta_dist, fp in zip(
#     [10000,20000,30000,np.inf,10000,20000,30000,np.inf ], 
#     [-0.062626,-0.062626,-0.062626,-0.062626,5*-0.062626,5*-0.062626,5*-0.062626,5*-0.062626],
#     ["UserClasses_Distance10km.json","UserClasses_Distance20km.json","UserClasses_Distance30km.json","UserClasses_DistanceInfkm.json","UserClasses_Distance10kmHighDecay.json","UserClasses_Distance20kmHighDecay.json","UserClasses_Distance30kmHighDecay.json","UserClasses_DistanceInfkmHighDecay.json"]
#   ):
simple = []
fp = "UserClasses_LongSpan.json"
const = [-2*0.443,-1*0.443, 0*0.443, 1*0.443, 2*0.443]
growthrate = [1*0.443, 0.75*0.443, 0.5*0.443, 0.25*0.443, 0]
maxDistance = np.inf
for node in nodes:
    index = network["IDs"][node]
    data = pd.read_excel('DisseminationAreas.xlsx', sheet_name = node)
    data["Total"] = data["Total"].fillna(0)
    # pop_1 = data.loc[179]["Total"] + data.loc[180]["Total"] + 0.5*data.loc[181]["Total"]
    # pop_2 = data.loc[182]["Total"] + data.loc[183]["Total"] + 0.5*data.loc[181]["Total"]
    # pop_3 = data.loc[184]["Total"] + data.loc[185]["Total"] + 0.5*data.loc[186]["Total"]
    # pop_4 = data.loc[188]["Total"] + data.loc[189]["Total"] + 0.5*data.loc[186]["Total"]
    # pop_5 = data.loc[190]["Total"]
    pop = data.loc[0]["Total"]
    # detached = data.loc[42]["Total"]
    # attached = data.loc[45]["Total"]+data.loc[46]["Total"]+data.loc[47]["Total"]+data.loc[49]["Total"]
    # app = data.loc[43]["Total"]+data.loc[48]["Total"]
    # can_hc = 0.9*detached + 0.75*attached + 0.4*app
    # cant_hc = 0.1*detached + 0.25*attached + 0.6*app

    del data

    # for i, pop in zip(range(5),[pop_1,pop_2,pop_3,pop_4,pop_5]):
    #     if pop < 1:
    #         continue
    s = {}
    s = {"Home":node,
        "Population": pop,
        "StationChoiceSet": [j for j in range(len(stations)) if network["ShortestPaths"][index][locations[j]] <= maxDistance],
        "StationCoefficients": {j:OutletUtility(beta_chargers, stations[j]["MaxOutlets"]) for j in range(len(stations))},
        "OptOutChoiceSet": [0],
        "OptOutConstants": {"0":4.5},
        "FactorOrder": ["OptOut"]+[i for i in range(len(stations))],
        "FactorScale": [1, 1],
        "FactorLoading": [[1,0]]+[[0,1] for _ in range(len(stations))],                    
        }
    constants = [[] for _ in range(10)]
    for t in range(10):
        for j in range(len(stations)):
            total = (beta_dist) * (network["ShortestPaths"][index][locations[j]]) / 1000 #Unit conversion for meters to kilometers
            if stations[j]["isLevel3"]:
                total += beta_L3
            #total += const[i] + growthrate[i]*t
            #if int(stations[j]["Location"]) in [1,3,4,5,6,7,8,12,13,17,21]:
            #    total+=beta_cc
            constants[t].append(total)
    s["StationConstants"] = constants
    simple.append(s)
    print("Completed node: "+str(node))
json.dump(simple,open(fp,"w+"))
    # hc_true = {}
    # hc_true = {"Home":node,
    #     "Population": can_hc,
    #     "StationChoiceSet": [j for j in range(len(stations)) if network["ShortestPaths"][index][locations[j]] <= maxDistance],
    #     "StationCoefficients": {j:OutletUtility(0.75*beta_chargers, stations[j]["MaxOutlets"]) for j in range(len(stations))},
    #     "OptOutChoiceSet": [0,1],
    #     "OptOutConstants": {"0":4.5, "1":2.5},
    #     "FactorOrder": ["OptOut", "Home"]+[i for i in range(len(stations))],
    #     "FactorScale": [1, 1, 1],
    #     "FactorLoading": [[1,0,0],[0,1,0]]+[[0,0,1] for _ in range(len(stations))],                    
    #     }
    # constants = []
    # for j in range(len(stations)):
    #     total = 0*beta_dist * (network["ShortestPaths"][index][locations[j]]) / 1000 #Unit conversion for meters to kilometers
    #     if stations[j]["isLevel3"]:
    #         total += 1*beta_L3
    #     #if int(stations[j]["Location"]) in [1,3,4,5,6,7,8,12,13,17,21]:
    #     #    total+=beta_cc
    #     constants.append(total)
    # hc_true["StationConstants"] = constants
    # homecharging.append(hc_true)

    # hc_false = {}
    # hc_false = {"Home":node,
    #     "Population": cant_hc,
    #     "StationChoiceSet": [j for j in range(len(stations)) if network["ShortestPaths"][index][locations[j]] <= maxDistance],
    #     "StationCoefficients": {j:OutletUtility(1.25*beta_chargers, stations[j]["MaxOutlets"]) for j in range(len(stations))},
    #     "OptOutChoiceSet": [0],
    #     "OptOutConstants": {"0":4.5},
    #     "FactorOrder": ["OptOut"]+[i for i in range(len(stations))],
    #     "FactorScale": [1, 1],
    #     "FactorLoading": [[1,0]]+[[0,1] for _ in range(len(stations))],                    
    #     }
    # constants = []
    # for j in range(len(stations)):
    #     total = 0*beta_dist * (network["ShortestPaths"][index][locations[j]]) / 1000 #Unit conversion for meters to kilometers
    #     if stations[j]["isLevel3"]:
    #         total += 1*beta_L3
    #     #if int(stations[j]["Location"]) in [1,3,4,5,6,7,8,12,13,17,21]:
    #     #    total+=beta_cc
    #     constants.append(total)
    # hc_false["StationConstants"] = constants
    # homecharging.append(hc_false)

 
# if dumpStations:        
#     json.dump(stations, open(stationsFilename, "w+"), indent = 3)
# json.dump(simple,open(simpleFilename,"w+"), indent = 3)
#json.dump(homecharging,open(homechargingFilename,"w+"), indent = 3)
