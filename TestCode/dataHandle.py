import torch
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/TestCode")
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import logging
import pandas as pd
import numpy as np
from configData import createDataForAnalyst
import polars as pl
from tqdm import tqdm
import json

def returnTestDataset(testDataFrame, device, batch_size, timestamps, outputLabel, exceptBand, fullBand):
    bandType = [
        f"{band}_t{i:+d}" if band not in exceptBand else band
        for i in range(-timestamps, timestamps)
        for band in fullBand
    ]

    XTestInput, yTestLabel = testDataFrame.loc[:,bandType].values.astype('float32'), testDataFrame.loc[:,outputLabel].values.astype('float32')
    del testDataFrame

    X_tensor = torch.tensor(XTestInput.reshape(-1, timestamps * 2, len(bandType) // (timestamps * 2)), dtype= torch.float32, device= device)
    y_tensor = torch.tensor(yTestLabel, dtype= torch.float32, device= device)

    del XTestInput, yTestLabel
    TensorDSTest = TensorDataset(X_tensor, y_tensor)

    del X_tensor, y_tensor
    testDataset = DataLoader(
        TensorDSTest,
        batch_size=batch_size,
        shuffle=False
    )

    return testDataset

# Tạo band diff, normalize band, tạo dataset cho tập train và tập test
def loadedFullDataset(fullDataSet, diffBand, exceptBand, timeStamps, inputInfo, fullBand):
    # Load vào để lấy các key có thể được dùng để chuẩn hóa.
    listOfBandInfo = {}
    with open(inputInfo, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            listOfBandInfo.update(data)
    
    dataObject = createDataForAnalyst(fullDataSet, timeStamps, inputInfo)
    if diffBand is not None:
        dataObject.createDiffBand(diffBand)
    for band in fullBand:
        # Hàm lấy min max value
        if band in listOfBandInfo:
            if band not in exceptBand:
                dataObject.normalBand(band, exceptBand, "MinMaxScaler", keyValue= ["min", "max"])
            else:
                dataObject.normalBand(band, exceptBand, "MaxScaler", keyValue= ["max"])
    print("Create diff band completed")
    return dataObject.inputDf

