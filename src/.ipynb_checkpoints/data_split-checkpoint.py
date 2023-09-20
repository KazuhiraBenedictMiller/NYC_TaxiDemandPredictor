from datetime import datetime
from typing import Tuple

import pandas as pd

def TrainTestSplit(df:pd.DataFrame, cutoff_date:datetime, target_column_name:str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    TrainData = df[df["pickup_hour"] < cutoff_date].reset_index(drop=True)
    TestData = df[df["pickup_hour"] >= cutoff_date].reset_index(drop=True)

    xTrain = TrainData.drop(columns = [target_column_name])
    yTrain = TrainData[target_column_name]
    xTest = TestData.drop(columns = [target_column_name])
    yTest = TestData[target_column_name]

    return xTrain, yTrain, xTest, yTest