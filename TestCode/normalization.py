import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
import logging
import polars as pl

class normalize:
    def __init__(self, inputDf):
        self.inputDf = inputDf
    def normalization(self, colName, normFunct, valueInput= None):
        if isinstance(colName, str):
            colName = [colName]
        numpyArray = self.inputDf.loc[:, colName].to_numpy()
        if normFunct == "MinMaxScaler":
            if not isinstance(valueInput, list) or len(valueInput) != 2:
                return False
            minValue, maxValue = valueInput

            denom = maxValue - minValue
            # ✅ Avoid division by zero or very small values causing inf/nan
            if denom == 0:
                print(f"⚠️ WARNING: {colName} constant (min=max={minValue}), setting to 0.5")
                self.inputDf.loc[:, colName] = 0.5
            elif denom < 1e-10:
                print(f"⚠️ WARNING: {colName} range too small ({denom}), may cause numerical issues")
                self.inputDf.loc[:, colName] = (numpyArray - minValue) / (denom + 1e-10)
            else:
                self.inputDf.loc[:, colName] = (numpyArray - minValue) / denom
        elif normFunct == "RobustScaler":
            if not isinstance(valueInput, list) or len(valueInput) != 3:
                return False
            Q1, Q2, Q3 = valueInput
            self.inputDf.loc[:, colName] = (numpyArray - Q2) / (Q3 - Q1)
        elif normFunct == "StandardScaler":
            if not isinstance(valueInput, list) or len(valueInput) != 2:
                return False
            mean, std = valueInput
            self.inputDf.loc[:, colName] = (numpyArray - mean) / std
        elif normFunct == "LogScaler":
            self.inputDf.loc[:, colName] = np.log1p(numpyArray)
            return True
        elif normFunct == "AverageScaler":
            if not isinstance(valueInput, list) or len(valueInput) != 1:
                return False
            valueInput = valueInput[0]
            if valueInput != 0:
                self.inputDf.loc[:, colName] = (numpyArray) / valueInput
                return True
            else:
                return "Divide 0 error, None error"
        elif normFunct == "MaxScaler":
            if not isinstance(valueInput, list) or len(valueInput) != 1:
                return False
            valueInput = valueInput[0]
            if valueInput != 0:
                self.inputDf.loc[:, colName] = (numpyArray) / valueInput
                return True
            else:
                return "Divide 0 error, None error"
        else:
            return "Not exists scaler"