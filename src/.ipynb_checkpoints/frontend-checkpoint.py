import zipfile
from datetime import datetime

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

#Title
currentdate = pd.to_datetime(datetime.utcnow()).floor("H")
st.title(f"Taxi Demand Prediction")
st.header(f"{currentdate}")

#Plotting a Progress Bar to improve UI while Loading Time
ProgressBar = st.sidebar.header("Working Progress")
ProgressBar = st.sidebar.progress(0)
N_Steps = 7

def LoadShapeDataFile():
    
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

with st.spinner(text = "Downloading Shape File to Plot Taxi Zones"):
    geo_df = LoadShapeDataFile()
    st.sidebar.write("Shape File Was Downloaded")
    ProgressBar.progress(1/N_Steps)
    
with st.spinner(text = "Fetching Batch of Inference Data"):
    Features = inference.LoadBatchOfFeaturesFromStore(currentdate)
    st.sidebar.write("Inference Features Fetched from the Store")
    ProgressBar.progress(2/N_Steps)
    print(f"{Features}")
    
with st.spinner(text = "Loading ML Model From the Registry"):
    Model = inference.LoadModelFromRegistry()
    st.sidebar.write("ML Model was Loaded from the Registry")
    ProgressBar.progress(3/N_Steps)

with st.spinner(text = "Computing Model Prediction"):
    Results = inference.GetModelPredictions(Model, Features)
    st.sidebar.write("Model Predictions Arrived")
    ProgressBar.progress(4/N_Steps)
    
with st.spinner(text = "Perparing Data to Plot"):
    
    def Pseudocolor(val, minval, maxval, startcolor, endcolor):
        #Convert Value in the Range of minval-maxval to a Color in the Range startcolor-endcolor.
        #The Colors passed and the one returned are composed by a sequence of N component values
    
        f = float(val-minval)/(maxval-minval)
    
        return tuple(f*(b-a)+a for (a,b) in zip(startcolor, endcolor))
    
    df = pd.merge(geo_df, Results, right_on="pickup_location_id", left_on="LocationIDs", how="inner")
    
    BLACK, GREEN = (0,0,0), (0, 255, 0)
    
    df["color_scaling"] = df["predicted_demand"]
    max_pred, min_pred = df["color_scaling"] .max(), df["color_scaling"] .min()
    df["fill_color"] = df["color_scaling"].apply(lambda x:Pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    ProgressBar.progress(5/N_Steps)
    
with st.spinner(text="Generating NYC Map"):
    
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=48.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=8
    )
    
    geojson = pdk.layer(
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
    ProgressBar.progress(6/N_Steps)
    
with st.spinner(text="Plotting TimeSeries Data"):
    
    row_indices = np.argsort(Results["predicted_demand"].values)[::-1]
    nToPlot = 10
    
    #Plot Each Time-Series with the Prediction
    for row_id in row_indices[:nToPlot]:
        fig = plot.PlotOneRidesSample(
            features=Features,
            targets=Results["predicted_demand"],
            exampleID=row_id,
            predictions=pd.Series(Results["predicted_demand"])
        )
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)
    
    ProgressBar.progress(7/N_Steps)