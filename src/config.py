'''
import os
import sys
sys.path.append("../")

import APIKey

HOPSWORKSAPIKEY = APIKey.HOPSWORKSAPIKEY

HopsworksProjectName = "TaxiDemandPrediction"

FeatureGroupName = "ts_hourly_featuregroup"
FeatureGroupVersion = 1
FeatureViewName = "ts_hourly_featureview"
FeatureViewVersion = 1
'''

import os
from dotenv import load_dotenv
import sys
sys.path.append("../src/")
sys.path.append("../")

import paths

load_dotenv(paths.PARENT_DIR / ".env")

HopsworksProjectName = "TaxiDemandPrediction"

try:
    HOPSWORKSAPIKEY = os.environ["HOPSWORKSAPIKEY"]

except:
    raise Exception("Create a .env File in the Parent Dir of the Project")

FeatureGroupName = "ts_hourly_featuregroup"
FeatureGroupVersion = 1

FeatureViewName = "ts_hourly_featureview"
FeatureViewVersion = 1

N_Features = 24*7*4

ModelName = "taxi_demand_predictor_next_hour"
ModelVersion = 1

FeatureGroupModelPredictions = "model_predictions_feature_group"
FeatureViewModelPredictions = "model_predictions_feature_view"

FeatureViewMonitoring = "model_monitoring_feature_view"