
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

df = pd.read_parquet(paths.RAW_DATA_DIR / "rides_2023-06.parquet")

print(df)