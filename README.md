# EVChargingStationModel
 Optimisation model for determining optimal location of electric vehicle charging stations as to maximise electric vehicle adoption

The paper associated with this work is accepted to INFORMS Journal of Computing, and is now available online: https://pubsonline.informs.org/doi/10.1287/ijoc.2022.0185.
The instances used in that paper can be found here: https://datashare.ed.ac.uk/handle/10283/4856 . For the Simple, Distance, and HomeCharging datasets, this includes:
- `SL{instance}.json` file (one per instance), designed to be loaded with `Data_SingleLevel` (using the `load` argument during initialisation).
- `GF_{dataset}.json` file (one per dataset), designed to be loaded with `Data_GrowthFunction` (using the `load` argument during initialisation).
- `MC{instance}.pickle` file (one per instance), designed to be loaded with `Data_MaximumCover` (using the `load` argument during initialisation). This file also contains the raw error terms, which can be accessed after unpickling the files. Once unpickled, these files are in JSON format, with the keys `d0` and `d1` corresponding to the error terms + alternative-specific constants for the exogenous and endogenous alternatives (respectively).
- `Shared.json` (one per dataset), used while loading the compressed files. This file includes the parameter values which do not vary by instance.
- `MC{instance}_compressed.json` file (one per instance), designed to be loaded with `Data_MaximumCover` (using the `load_compressed` argument during initialisation. Make sure to include the Shared file and instance file in the same path. See example below). These files have the coverage "baked in" for the maximum covering formulation, and so cannot be used with the single-level formulation. However, they load significantly faster for models based on the maximum covering formulation, which includes the heuristic methods.  
For the Price and LongSpan datasets, only the last three are included.

In order to run any of the solving methods, you must first instantiate the data using the appropriate data file and class. This can be done, as an example, by the following code (in Python shell):
```
    from MaximumCover_Data import Data_MaximumCover
    from MaximumCover_Model import ChargingStationModel_MaximumCover
    from docplex.mp.model import Model

    Data = Data_MaximumCover(userFilepath = "Data/StationsAndClasses/UserClasses_Simple.json", stationFilepath = "Data/StationsAndClasses/Stations_Simple.json")
    mdl = Model()
    solver = ChargingStationModel_MaximumCover(mdl, Data)
```
 This code will generate new error terms, which can be saved by using the `dump()` or `pickle()` functions in the `Data_MaximumCover` class. We also note that the heuristic methods use the data class from the Maximum Covering model, and so data should be instantiated using the `Data_MaximumCover` class found in `MaximumCover_Data.py`.

If loading compressed data, ensure that the `Shared.json` file is included within the same folder as the instances (or modify the load_compressed function accordingly). Additionally, note that running the `Process_Reformulation()` method within the `Data_MaximumCover` class (or the equivalent version in the `Data_SingleLevel` class) is necessary to run the reformulated models. This can be done, as an example, by the following code (in Python shell):
```
    from MaximumCover_Data import Data_MaximumCover
    from MaximumCover_Reformulation import ChargingStationModel_MaximumCover
    from docplex.mp.model import Model

    Data = Data_MaximumCover(load_compressed = (<path_to_folder_containing_shared_file_and_instance_file>, <name_of_instance_file>))
    Data.Process_Reformulation()
    mdl = Model()
    solver = ChargingStationModel_MaximumCover(mdl, Data)
```
The Greedy and GRASP methods have not been modified, as they are not significantly impacted by the reformulation. However, minor variable adjustments (notably for the cost parameters `c` and the initial station state `x0`) may be necessary if using compressed data.

Network information, as well as resources to create user class and station files, can be found within the Data folder. 

