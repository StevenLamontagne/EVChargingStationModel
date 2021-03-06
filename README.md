# EVChargingStationModel
 Optimisation model for determining optimal location of electric vehicle charging stations as to maximise electric vehicle adoption

The paper associated with this work is now available on arXiv: https://arxiv.org/abs/2206.11165 .

In order to run any of the solving methods, you must first instantiate the data using the appropriate data file and class. This can be done, as an example, by the following code (in Python shell):
    from MaximumCover_Data import Data_MaximumCover
    from MaximumCover_Model import ChargingStationModel_MaximumCover
    from docplex.mp.model import Model

    Data = Data_MaximumCover(userFilepath = "Data/StationsAndClasses/UserClasses_Simple.json", stationFilepath = "Data/StationsAndClasses/Stations_Simple.json")
    mdl = Model()
    solver = ChargingStationModel_MaximumCover(mdl, Data)

 This code will generate new error terms, which can be saved by using the dump() or pickle() functions in the Data_MaximumCover class. Due to size limitations, the data files used for testing in the paper are not contained in this repository. However, they are available upon request. We also note that the heuristic methods use the data class from the Maximum Covering model, and so data should be instantiated using the Data_MaximumCover class found in MaximumCover_Data.py.

Network information, as well as ressources to create user class and station files, can be found within the Data folder.

