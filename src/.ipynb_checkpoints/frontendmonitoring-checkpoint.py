from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_absolute_error as MAE
import os
import sys
sys.path.append("../src/")
sys.path.append("../")

import monitoring

st.set_page_config(layout = "wide")

#Title
currentdate = pd.to_datetime(datetime.utcnow()).floor("H") - timedelta(weeks=52)
st.title(f"Taxi Demand Predictions Model Monitoring Dashboard")
st.header(f"{currentdate} UTC")

#Plotting a Progress Bar to improve UI while Loading Time
ProgressBar = st.sidebar.header("Working Progress")
ProgressBar = st.sidebar.progress(0)
N_Steps = 3

@st.cache_data
def LoadPredictionsActuals(fromdate:datetime, todate:datetime) -> pd.DataFrame:
    return monitoring.LoadPredictionsAndActualValuesFromStore(fromdate, todate)

with st.spinner("Fetching Model Predictions and Actual Values from the Store"):
    
    MonitoringDF = LoadPredictionsActuals(fromdate = currentdate - timedelta(days=14), todate = currentdate)
    st.sidebar.write("Model Predictions and Actual Values Arrived")
    ProgressBar.progress(1/N_Steps)
    
with st.spinner("Plotting Aggregate MAE Hour-by-Hour"):
    
    st.header("Mean Absolute Error (MAE) Hour-by-Hour")
    
    #MAE per Hour
    HourlyMAE = (
        MonitoringDF
        .groupby("pickup_hour")
        .apply(lambda df: MAE(df["numrides"], df["predicted_demand"]))
        .reset_index()
        .rename(columns={0: "MAE"})
        .sort_values(by="pickup_hour")
    )

    fig = px.bar(
        HourlyMAE,
        x="pickup_hour", y=MAE,
        template='plotly_dark',
    )
    
    st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    ProgressBar.progress(2/N_STEPS)
    
with st.spinner("Plotting Hourly MAE for Top Locations"):
    
    st.header('Mean Absolute Error (MAE) per Location and Hour')

    TopLocationsByDemand = (
        MonitoringDF
        .groupby("pickup_location_id")["numrides"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .head(10)["pickup_location_id"]
    )

    for locationid in TopLocationsByDemand:
        
        HourlyMAE = (
            MonitoringDF[MonitoringDF["pickup_location_id"] == locationid]
            .groupby("pickup_hour")
            .apply(lambda df: MAE(df["numrides"], df["predicted_demand"]))
            .reset_index()
            .rename(columns={0: "MAE"})
            .sort_values(by="pickup_hour")
        )

        fig = px.bar(
            HourlyMAE,
            x="pickup_hour", y="MAE",
            template='plotly_dark',
        )
        
        st.subheader(f'{locationid=}')
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    ProgressBar.progress(3/N_STEPS)