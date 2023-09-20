from typing import Optional, List
from datetime import timedelta
import pandas as pd

import plotly.express as px

def PlotOneRidesSample(features:pd.DataFrame, targets:pd.Series, exampleID: int, predictions:Optional[pd.Series] = None):
    
    features_ = features.iloc[exampleID]
    targets_ = targets.iloc[exampleID]
    
    TS_Columns = [c for c in features.columns if c.endswith("hours_before")]
    TS_Values = [features_[c] for c in TS_Columns] + [targets_]
    TS_Dates = pd.date_range(features_["pickup_hour"] - timedelta(hours = len(TS_Columns)), features_["pickup_hour"], freq = "H")
    
    title = f'Pickup Hours = {features_["pickup_hour"]}, Location ID = {features_["pickup_location_id"]}'
    
    fig = px.line(x = TS_Dates, y = TS_Values, template = "plotly_dark", markers = True, title = title)
    
    fig.add_scatter(x = TS_Dates[-1:], y = [targets_], line_color = "green", mode = "markers", marker_size = 10, name = "Actual Value")
    
    if predictions is not None:
        #Big Red X for the Predicted Value if Passed
        prediction_ = predictions.iloc[exampleID]
        
        fig.add_scatter(x = TS_Dates[-1:], y = [prediction_], line_color = "red", mode = "markers", marker_symbol = "x", marker_size = 15, name = "Prediction")
        
    return fig