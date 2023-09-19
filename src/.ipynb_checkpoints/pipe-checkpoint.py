import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline as PPLN
from sklearn.pipeline import Pipeline

def AverageRidesLast4Weeks(x:pd.DataFrame) -> pd.DataFrame:
    
    x["Avg Rides Last 4 Weeks"] = (x[f"Rides {7*24*1} Hours Before"] + x[f"Rides {7*24*2} Hours Before"] + x[f"Rides {7*24*3} Hours Before"] + x[f"Rides {7*24*4} Hours Before"])/4
    
    return x

class TemporalFeatureEngineering(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        X = x.copy()
        X["Hour"] = X["PickupHour"].dt.hour
        X["DoW"] = X["PickupHour"].dt.dayofweek
        
        return X.drop(columns = ["PickupHour"], inplace=False)
    
def MakePipeline(**Hyperparameters) -> Pipeline:
    
    add_feature_avgrideslast4weeks = FunctionTransformer(AverageRidesLast4Weeks, validate = False)
    
    add_temporalfeatures = TemporalFeatureEngineering()
    
    return PPLN(add_feature_avgrideslast4weeks, add_temporalfeatures, lgb.LGBMRegressor(**Hyperparameters))
