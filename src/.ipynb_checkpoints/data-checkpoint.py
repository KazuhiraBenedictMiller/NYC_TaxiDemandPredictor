import requests
from pathlib import Path
import os

from typing import Optional, List
from datetime import datetime, timedelta
from tqdm import tqdm

import pandas as pd
import numpy as np

from paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def DownloadOneFileRawData(year:int, month:int) -> Path:
    URL = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f"{URL} is not available")
        
def ValidateRawData(rides:pd.DataFrame, year:int, month:int) -> pd.DataFrame:

    #Keep only Rides for the Month
    ThisMonthStart = f"{year}-{month:02d}-01"
    NextMonthStart = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"
    rides = rides[rides["PickupDatetime"] >= ThisMonthStart]
    rides = rides[rides["PickupDatetime"] <= NextMonthStart]
    
    return rides

def LoadRawData(year:int, months:Optional[List[int]] = None) -> pd.DataFrame:
    
    rides = pd.DataFrame()
    
    if months is None:
        #Download Data only for months Specified in Months
        months = list(range(1,13))
        
    elif isinstance(months, int):
        #Download Data for ALL Year
        months = [months]
        
    for month in months:
        local_file = RAW_DATA_DIR / f"rides_{year}-{month:02d}.parquet"
        
        if not local_file.exists():
            try:
                #Download File from the NYC Cab Rides Website
                print(f"Downloading file {year}-{month:02d}")
                DownloadOneFileRawData(year, month)
            except:
                print(f"File {year}-{month:02d} is not available")
                continue
        else:
            print(f"File {year}-{month:02d} was already in local storage")
            
        #Load the File into Pandas
        rides_one_month = pd.read_parquet(local_file)
        
        #Rename Columns
        rides_one_month = rides_one_month[["tpep_pickup_datetime", "PULocationID"]]
        rides_one_month.rename(columns={"tpep_pickup_datetime":"PickupDatetime", "PULocationID":"PickupLocationID"}, inplace=True)
        
        #Validate the File
        rides_one_month = ValidateRawData(rides_one_month, year, month)
        
        #Append to Existing Data
        rides = pd.concat([rides, rides_one_month])
        
    #Keep only Time and Origin of the Ride
    rides = rides[["PickupDatetime","PickupLocationID"]]
    
    return rides

def AddMissingSlots(aggrides:pd.DataFrame) -> pd.DataFrame:
    
    locations = aggrides["PickupLocationID"].unique()
    full_range = pd.date_range(aggrides["PickupHour"].min(), aggrides["PickupHour"].max(), freq = "H")
    output = pd.DataFrame()
    
    for locid in tqdm(locations):
    
        #Keep only Rides for this Location ID
        aggrides_i = aggrides.loc[aggrides["PickupLocationID"] == locid, ["PickupHour", "NumOfRides"]]
        
        #Adding Missing Dates with 0 in a Series
        aggrides_i.set_index("PickupHour", inplace = True)
        aggrides_i.index = pd.DatetimeIndex(aggrides_i.index)
        aggrides_i = aggrides_i.reindex(full_range, fill_value = 0)
        
        #Add Back Location ID Columns
        aggrides_i["PickupLocationID"] = locid
        
        output = pd.concat([output, aggrides_i])
        
    #Move the PickupHour from Index to Column
    output = output.reset_index().rename(columns = {"index":"PickupHour"})
    
    return output

def TransformRawDataIntoTSData(rides:pd.DataFrame) -> pd.DataFrame:
    
    #Sum Rides per Location and per Pickup Hour
    rides["PickupHour"] = rides["PickupDatetime"].dt.floor("H")
    aggrides = rides.groupby(["PickupHour", "PickupLocationID"]).size().reset_index()
    aggrides.rename(columns = {0 : "NumOfRides"}, inplace = True)
    
    #Add Rows for Locations, Pickup Hour with 0 Rides
    aggrides_allslots = AddMissingSlots(aggrides)
    
    return aggrides_allslots

def GetCutoffIndeces(data:pd.DataFrame, nFeatures:int, SlidingFactor:int) -> list:
    
    StopPosition = len(data)-1
    
    #Start the First SubSequence at Index Position 0
    SubseqFirstIdx = 0
    SubseqStepIdx = nFeatures
    SubseqLastIdx = nFeatures +1
    
    #[FirstIdx, StepIdx, LastIdx]
    
    Indeces = []
    
    while SubseqLastIdx <= StopPosition:
        Indeces.append([SubseqFirstIdx, SubseqStepIdx, SubseqLastIdx])
        
        #StepSize is used as Sliding Factor
        SubseqFirstIdx += SlidingFactor
        SubseqStepIdx += SlidingFactor
        SubseqLastIdx += SlidingFactor
        
    return Indeces

def TransformALL(tsData:pd.DataFrame, nFeatures:int, SlidingFactor:int) -> pd.DataFrame:
    
    #assert set(tsData.columns) == {"PickupHour", "NumOfRides", "PickupLocationID"}
    
    locationIDs = tsData["PickupLocationID"].unique()
    Features = pd.DataFrame()
    Targets = pd.DataFrame()
    
    for locid in tqdm(locationIDs):
        #Keep only Time-Series Data for this Location
        tsDataOneLocation = tsData.loc[tsData["PickupLocationID"] == locid, ["PickupHour", "NumOfRides"]].sort_values(by = ["PickupHour"])
        
        #Pre-Compute Cutoff Indeces to Split DataFrame Rows
        indeces = GetCutoffIndeces(tsDataOneLocation, nFeatures, SlidingFactor)

        #Slice and Transpose Data into NumPy Arrays for Features and Target
        nSamples = len(indeces)
        X = np.ndarray(shape=(nSamples, nFeatures), dtype=np.float32)
        Y = np.ndarray(shape=(nSamples), dtype=np.float32)
        PickupHours = []
        
        for i, idx in enumerate(indeces):
            X[i,:] = tsDataOneLocation.iloc[idx[0]:idx[1]]["NumOfRides"].values
            Y[i] = tsDataOneLocation.iloc[idx[1]:idx[2]]["NumOfRides"].values
            PickupHours.append(tsDataOneLocation.iloc[idx[1]]["PickupHour"])
            
        #NumPy -> Pandas
        FeaturesOneLocationDF = pd.DataFrame(X, columns = [f"Rides {i+1} Hours Before" for i in reversed(range(nFeatures))])
        FeaturesOneLocationDF["PickupHour"] = PickupHours
        FeaturesOneLocationDF["PickupLocationID"] = locid
        
        TargetsOneLocationDF = pd.DataFrame(Y, columns = ["Target Rides Next Hour"])

        #Concatenate Results
        Features = pd.concat([Features, FeaturesOneLocationDF])
        Targets = pd.concat([Targets, TargetsOneLocationDF])
        
    Features.reset_index(inplace = True, drop = True)
    Targets.reset_index(inplace = True, drop = True)
    
    return Features, Targets["Target Rides Next Hour"]