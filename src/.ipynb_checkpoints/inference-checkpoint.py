from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore

import pandas as pd
import numpy as np

import joblib
from pathlib import Path

import os
import sys
sys.path.append("../src/")
sys.path.append("../")
import config

def GetHopsworksProject() -> hopsworks.project.Project:
    
    return hopsworks.login(project = config.HopsworksProjectName, api_key_value = config.HOPSWORKSAPIKEY)

def GetFeatureStore() -> FeatureStore:
    
    Project = GetHopsworksProject()
    return Project.get_feature_store()

def GetModelPredictions(model, features:pd.DataFrame) -> pd.DataFrame:
    
    Predictions = model.predict(features)
    
    results = pd.DataFrame()
    results["pickup_location_id"] = features["pickup_location_id"].values
    results["predicted_demand"] = Predictions.round(0)
    
    return results

def LoadBatchOfFeaturesFromStore(currentdate:datetime) -> pd.DataFrame:
    
    feature_store = GetFeatureStore()
    
    nFeatures = config.N_Features
    
    #Read TimeSeries Data from the Feature Store
    
    fetch_data_from = currentdate - timedelta(hours=1)
    fetch_data_to = currentdate - timedelta(days=28)
    print(f"Fetching data backwards from {fetch_data_from} to {fetch_data_to}")
    
    FeatureView = feature_store.get_feature_view(name=config.FeatureViewName, version= config.FeatureViewVersion)
    
    TS_Data = FeatureView.get_batch_data(start_time=(fetch_data_from - timedelta (days=1)), end_time=(fetch_data_to + timedelta (days=1)))
    TS_Data = TS_Data[TS_Data["pickup_hour"].between(fetch_data_from, fetch_data_to)]
    
    #Validate we are not Missing any Data in the Feature Store
    LocationIDs = TS_Data["pickup_location_id"].unique()
    assert len(TS_Data) == nFeatures*len(LocationIDs), "Time-Series Data is Incomplete"
    
    #Sort Data by Location and Time
    x = np.ndarray(shape=(len(LocationIDs), nFeatures), dtype=np.float32)
    
    for i, location in enumerate(LocationIDs):
        TS_Data_i = TS_Data.loc[TS_Data["pickup_location_id"] == location, :]
        TS_Data_i = TS_Data_i.sort_values(by = ["pickup_hour"])
        x[i,:] = TS_Data_i["num_rides"].values
        
    Features = pd.DataFrame(x, columns = [f"Rides {i+1} Hours Before" for i in reversed(range(nFeatures))])
    Features["pickup_hour"] = currentdate
    Features["pickup_location_id"] = LocationIDs
    
    return Features

def LoadModelFromRegistry():
    
    Project = GetHopsworksProject() 
    
    ModelRegistry = Project.get_model_registry()
    
    Model = ModelRegistry.get_model(name=config.FeatureViewName, version= config.FeatureViewVersion)
    
    ModelDir = Model.download()
    Model = joblib.load(Path(ModelDir) / "Model.pkl")
    
    return Model
