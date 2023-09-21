
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

currentdate = pd.to_datetime(datetime.utcnow()).floor("H")

features = inference.LoadBatchOfFeaturesFromStore(currentdate)