import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline as PPLN
from sklearn.pipeline import Pipeline

import lightgbm as lgb

def AverageRidesLast4Weeks(x:pd.DataFrame) -> pd.DataFrame:
    
    x["avg_rides_last_4_weeks"] = (x[f"rides_{7*24*1}_hours_before"] + x[f"rides_{7*24*2}_hours_before"] + x[f"rides_{7*24*3}_hours_before"] + x[f"rides_{7*24*4}_hours_before"])/4
    
    return x

class TemporalFeatureEngineering(BaseEstimator, TransformerMixin):
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        X = x.copy()
        X["hour"] = X["pickup_hour"].dt.hour
        X["dow"] = X["pickup_hour"].dt.dayofweek
        
        return X.drop(columns = ["pickup_hour"], inplace=False)
    
def MakePipeline(**Hyperparameters) -> Pipeline:
    
    add_feature_avgrideslast4weeks = FunctionTransformer(AverageRidesLast4Weeks, validate = False)
    
    add_temporalfeatures = TemporalFeatureEngineering()
    
    return PPLN(add_feature_avgrideslast4weeks, add_temporalfeatures, lgb.LGBMRegressor(**Hyperparameters))
