import zipfile
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd

import streamlit as st
import geopandas as gpd
import pydeck as pdk

import os
import sys
sys.path.append("../src/")
sys.path.append("../")

import inference
import paths
import plot

#Setting Layout as Wide
st.set_page_config(layout="wide")

currentdate = pd.to_datetime(datetime.utcnow() - timedelta(weeks=52)).floor("H")

#Title
st.title(f"Taxi Demand Prediction")
st.header(f"{currentdate} UTC")

#Plotting a Progress Bar to improve UI while Loading Time
ProgressBar = st.sidebar.header("Working Progress")
ProgressBar = st.sidebar.progress(0)
N_Steps = 6

def LoadShapeDataFile() -> gpd.geodataframe.GeoDataFrame:
    
    #Download File
    URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    Response = requests.get(URL)
    path = paths.DATA_DIR / "taxi_zones.zip"
    
    if Response.status_code == 200:
        open(path, "wb").write(Response.content)
        
    else:
        raise Exception(f"{URL} is not available")
        
    #Unzip File
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(paths.DATA_DIR / "taxi_zones")
        
    #Load and Return Shape File
    return gpd.read_file(paths.DATA_DIR / "taxi_zones/taxi_zones.shp").to_crs("epsg:4326")

@st.cache_data
def LoadFeatures(currentdate:datetime) -> pd.DataFrame:
    return inference.LoadBatchOfFeaturesFromStore(currentdate)

@st.cache_data
def LoadPredictions(from_pickup_hour:datetime, to_pickup_hour:datetime) -> pd.DataFrame:
    return inference.LoadPredictionsFromStore(from_pickup_hour, to_pickup_hour)

with st.spinner(text = "Downloading Shape File to Plot Taxi Zones"):
    geo_df = LoadShapeDataFile()
    st.sidebar.write("Shape File Was Downloaded")
    ProgressBar.progress(1/N_Steps)
    
with st.spinner(text = "Loading Model Predictions from the Store"):
    PredictionsDF = LoadPredictions(from_pickup_hour = currentdate - timedelta(hours = 1), to_pickup_hour = currentdate)
    st.sidebar.write("Model Predictions Arrived")
    ProgressBar.progress(2/N_Steps)
    
#Here we are implementing a Logic to check if the Predictions for the Current Hour have already been computed and are Available
NextHourPredictionsReady = False if PredictionsDF[PredictionsDF["pickup_hour"] == pd.to_datetime(currentdate, utc=True)].empty else True
PrevHourPredictionsReady = False if PredictionsDF[PredictionsDF["pickup_hour"] == pd.to_datetime(currentdate - timedelta(hours=1), utc=True)].empty else True

if NextHourPredictionsReady:
    PredictionsDF = PredictionsDF[PredictionsDF["pickup_hour"] == pd.to_datetime(currentdate, utc=True)]
    
elif PrevHourPredictionsReady:
    #If Next Predictions didn't arrive we'll use the Previous Ones
    PredictionsDF[PredictionsDF["pickup_hour"] == pd.to_datetime(currentdate - timedelta(hours=1), utc=True)]
    currentdate = pd.to_datetime(currentdate - timedelta(hours=1))
    st.warning("The most recent Data is currently not available, using Data from the Previous Hour.")
    
else:
    raise Exception("Features are not available for the last 2 Hours, please check if the Pipeline is up and Running.")

with st.spinner(text = "Perparing Data to Plot"):
    
    def Pseudocolor(val, minval, maxval, startcolor, endcolor):
        #Convert Value in the Range of minval-maxval to a Color in the Range startcolor-endcolor.
        #The Colors passed and the one returned are composed by a sequence of N component values
    
        f = float(val-minval)/(maxval-minval)
    
        return tuple(f*(b-a)+a for (a,b) in zip(startcolor, endcolor))
    
    df = pd.merge(geo_df, PredictionsDF, right_on="pickup_location_id", left_on="LocationID", how="inner")
    
    BLACK, GREEN = (0,0,0), (0, 255, 0)
    
    df["color_scaling"] = df["predicted_demand"]
    max_pred, min_pred = df["color_scaling"] .max(), df["color_scaling"] .min()
    df["fill_color"] = df["color_scaling"].apply(lambda x:Pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    ProgressBar.progress(3/N_Steps)
    
with st.spinner(text="Generating NYC Map"):
    
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=12,
        max_zoom=16,
        pitch=45,
        bearing=0
    )
    
    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation = 10,
        get_fill_color="fill_color",
        get_line_color=[255,255,255],
        auto_highlight=True,
        pickable=True,
    )
    
    tooltip = {"html":"<b>Zone:</b> [{LocationID}]{zone} <br/> <b>Predicted Rides:</b> {predicted_demand}"}
    
    r = pdk.Deck(        
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )
    
    st.pydeck_chart(r)
    ProgressBar.progress(4/N_Steps)
    
with st.spinner(text="Fetching Batch of Features used in the last run"):
    
    FeaturesDF = LoadFeatures(currentdate)
    st.sidebar.write("Inference Features Fetched from the Store")
    ProgressBar.progress(5/N_Steps)
    
with st.spinner(text="Plotting TimeSeries Data"):
    
    row_indices = np.argsort(PredictionsDF["predicted_demand"].values)[::-1]
    nToPlot = 10
    
    #Plot Each Time-Series with the Prediction
    for row_id in row_indices[:nToPlot]:
        fig = plot.PlotOneRidesSample(
            features=FeaturesDF,
            targets=PredictionsDF["predicted_demand"],
            exampleID=row_id,
            predictions=pd.Series(PredictionsDF["predicted_demand"])
        )
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
    
    ProgressBar.progress(6/N_Steps)