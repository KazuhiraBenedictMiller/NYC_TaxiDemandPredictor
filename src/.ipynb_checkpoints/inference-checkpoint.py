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
import featurestoreapi
    
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
    
    fetch_data_to = pd.to_datetime(currentdate - timedelta(hours=1), utc= True)
    fetch_data_from = pd.to_datetime(currentdate - timedelta(days=28), utc= True)
    print(f"Fetching data backwards from {fetch_data_from} to {fetch_data_to}")
    
    #Transforming TimeStamp Data to Datetime
    #fetch_data_from = pd.to_datetime(fetch_data_from, utc=True)
    #fetch_data_to = pd.to_datetime(fetch_data_to, utc=True)
    
    FeatureView = feature_store.get_feature_view(name=config.FeatureViewName, version= config.FeatureViewVersion)
    
    TS_Data = FeatureView.get_batch_data(start_time=pd.to_datetime(fetch_data_from - timedelta (days=1), utc=True), end_time=pd.to_datetime(fetch_data_to + timedelta(days = 1), utc=True))
    TS_Data["pickup_hour"] = pd.to_datetime(TS_Data["pickup_hour"], utc=True)
    TS_Data = TS_Data[TS_Data["pickup_hour"].between(fetch_data_from, fetch_data_to)]
    
    #Validate we are not Missing any Data in the Feature Store
    LocationIDs = TS_Data["pickup_location_id"].unique()
    assert len(TS_Data) == nFeatures*len(LocationIDs), "Time-Series Data is Incomplete, make sure your Feature Pipeline is up and running"
    
    #Sort Data by Location and Time
    TS_Data.sort_values(by=["pickup_location_id", "pickup_hour",], inplace = True)
    
    #Transpose Time-Series Data as a Feature Vector for each "picup_location_id"
    x = np.ndarray(shape=(len(LocationIDs), nFeatures), dtype=np.float32)
    
    for i, location in enumerate(LocationIDs):
        TS_Data_i = TS_Data.loc[TS_Data["pickup_location_id"] == location, :]
        TS_Data_i = TS_Data_i.sort_values(by = ["pickup_hour"])
        x[i,:] = TS_Data_i["numrides"].values
        
    Features = pd.DataFrame(x, columns = [f"rides_{i+1}_hours_before" for i in reversed(range(nFeatures))])
    Features["pickup_hour"] = currentdate
    Features["pickup_location_id"] = LocationIDs
    Features.sort_values(by=["pickup_location_id"], inplace = True)
    
    return Features
    
def LoadModelFromRegistry():
    
    Project = GetHopsworksProject() 
    
    ModelRegistry = Project.get_model_registry()
    
    Model = ModelRegistry.get_model(name=config.ModelName, version= config.ModelVersion)
    
    ModelDir = Model.download()
    Model = joblib.load(Path(ModelDir) / "Model.pkl")
    
    return Model

def LoadPredictionsFromStore(from_pickup_hour: datetime, to_pickup_hour: datetime) -> pd.DataFrame:

    #Transforming TimeStamp Data to Datetime
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)
    
    #Connects to the Feature Store and retrieves Model Predictions for all
    #"pickup_location_id" and for the time period from "from_pickup_hour" to "to_pickup_hour"

    FeatureStore = featurestoreapi.GetFeatureStore()

    PredictionsFeatureGroup = FeatureStore.get_feature_group(
        name = config.FeatureGroupModelPredictions,
        version = config.FeatureGroupModelPredictionsVersion,
    )

    try:
        #create feature view since it does not exist yet
        FeatureStore.create_feature_view(
            name = config.FeatureViewModelPredictions,
            version = config.FeatureViewModelPredictionsVersion,
            query = PredictionsFeatureGroup.select_all()
        )
        
    except:
        print(f'Feature view {config.FeatureViewModelPredictions} already exist. Skipped creation.')
        
    PredictionsFeatureView = FeatureStore.get_feature_view(
        name = config.FeatureViewModelPredictions,
        version = 1
    )
    
    print(f'Fetching Predictions for "pickup_hours" between {from_pickup_hour} and {to_pickup_hour}')
    
    Predictions = PredictionsFeatureView.get_batch_data(
        start_time = from_pickup_hour - timedelta(days=1),
        end_time = to_pickup_hour + timedelta(days=1)
    )
    
    #Ensure UTC-Awareness
    Predictions["pickup_hour"] = pd.to_datetime(Predictions["pickup_hour"], utc=True)
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)
    
    Predictions = Predictions[Predictions["pickup_hour"].between(from_pickup_hour, to_pickup_hour)]

    # sort by `pick_up_hour` and `pickup_location_id`
    Predictions.sort_values(by = ["pickup_hour", "pickup_location_id"], inplace = True)

    return Predictions