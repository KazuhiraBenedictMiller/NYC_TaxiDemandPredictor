from typing import Optional
import hsfs
import hopsworks

import os
import sys
sys.path.append("../src/")
sys.path.append("../")

import config

def GetFeatureStore() -> hsfs.feature_store.FeatureStore:
    
    #Connects to Hopsworks and returns a Pointer to the Feature Store
    
    Project = hopsworks.login(project = config.HopsworksProjectName, api_key_value = config.HOPSWORKSAPIKEY)
    
    return Project.get_feature_store()

def GetFeatureGroup(name:str, version:Optional[int] = 1) -> hsfs.feature_group.FeatureGroup:
    
    #Connects to the Feature Store and Returns a Pointer to the Given Feature Group with "name"
    
    return GetFeatureStore().get_feature_group(name = name, version = version)