from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import os
import sys
sys.path.append("../src/")
sys.path.append("../")

import config
import featurestoreapi

def LoadPredictionsAndActualValuesFromStore(fromdate, todate) -> pd.DataFrame:
    
    #Transforming Datetime Data with UTC Awareness
    #fromdate = pd.to_datetime(fromdate, utc=True)
    #todate = pd.to_datetime(fromdate, utc=True)
    
    #Feature Groups we need to Merge
    PredictionsFeatureGroup = featurestoreapi.GetFeatureGroup(name = config.FeatureGroupModelPredictions)
    ActualsFeatureGroup = featurestoreapi.GetFeatureGroup(name = config.FeatureGroupName)
    
    #Query to Join the 2 Feature Groups by "pickup_hour" and "location_id"
    query = PredictionsFeatureGroup.select_all().join(ActualsFeatureGroup.select_all(), on = ["pickup_hour", "pickup_location_id"]).filter(PredictionsFeatureGroup["pickup_hour"] >= pd.to_datetime(fromdate, utc=True)).filter(PredictionsFeatureGroup["pickup_hour"] <= pd.to_datetime(todate, utc=True))
    
    #Create the Feature View "Monitoring" if it doesn't exist yet
    FeatureStore = featurestoreapi.GetFeatureStore()
    
    try:
        #Create Feature View since it doesn't exist
        FeatureStore.create_feature_view(name = config.FeatureViewMonitoring, version = config.FeatureViewMonitoringVersion, query = query)
        
    except:
        print("Feature View already exist")
        
    #Feature View
    MonitoringFeatureView = FeatureStore.get_feature_view(name = config.FeatureViewMonitoring, version = config.FeatureViewMonitoringVersion)
    
    #Fetching Data from the Feature View
    #Fetch Predictions and Actual Values for the last 30 days
    new_startdate = pd.to_datetime(fromdate - timedelta(days=7))
    new_todate = pd.to_datetime(todate + timedelta(days=7))
    
    MonitoringDF = MonitoringFeatureView.get_batch_data(start_time = new_startdate, end_time = new_todate)
    MonitoringDF = MonitoringDF[MonitoringDF["pickup_hour"].between(pd.to_datetime(fromdate, utc=True), pd.to_datetime(todate, utc=True))]
    
    return MonitoringDF