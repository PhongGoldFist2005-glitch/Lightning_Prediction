import torch
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
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
def loadedFullDataset(inputFileList, diffBand, exceptBand, timeStamps, inputInfo, fullBand):
    dataFrames = []
    for file in tqdm(inputFileList, total= len(inputFileList)):
        df = pl.read_parquet(file).to_pandas()
        dataFrames.append(df)
    
    # Load vào để lấy các key có thể được dùng để chuẩn hóa.
    listOfBandInfo = {}
    with open(inputInfo, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            listOfBandInfo.update(data)
    
    fullDataSet = pd.concat(dataFrames, ignore_index= True)
    del dataFrames
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

# Tạo dataset 6 thời gian cho tập train.
def handleTrainData(trainDataset, fullBand, exceptBand, timeStamps, outputLabel):
    features = [
        f"{band}_t{i:+d}" if band not in exceptBand else band
        for i in range(-timeStamps, 0)
        for band in fullBand
    ]

    features_unique = list(dict.fromkeys(features))
    outputs = [outputLabel[0]]  # Chỉ lấy label ở thời điểm t0
    newCols = features_unique + outputs

    chunks = []  # Giữ list of DataFrames thay vì list of numpy rows

    for i in range(timeStamps):
        period = [
            f"{band}_t{j:+d}" if band not in exceptBand else band
            for j in range(i - timeStamps, i)
            for band in fullBand
        ]

        period_unique = list(dict.fromkeys(period))
        output = [outputLabel[i]]

        # Chỉ lấy đúng cột cần, rename để khớp newCols
        chunk = trainDataset[period_unique + output].copy()
        chunk.columns = newCols
        chunks.append(chunk)

        # Drop in-place để không giữ bản copy
        columns = []
        for band in fullBand:
            if band not in exceptBand:
                columns.append(f"{band}_t{i - timeStamps}")
            else:
                continue
        columns += output

        trainDataset.drop(
            columns= columns,
            inplace=True,
        )
    del trainDataset  # Giải phóng ngay sau khi xử lý xong
    fullResult = pd.concat(chunks).sample(frac=1, random_state=42).reset_index(drop=True)
    del chunks  # Giải phóng ngay sau khi concat xong

    print("Config train data completed")
    return features, outputs, fullResult

def returnDataset(trainValDataset, exceptBand, device, batch_size, timestamps, outputLabel, fullBand):
    bandType = [
        f"{band}_t{i:+d}" if band not in exceptBand else band
        for i in range(-timestamps, timestamps)
        for band in fullBand
    ]
    trainDataset, valDataset = train_test_split(trainValDataset ,test_size= 0.2, random_state= 42)
    # Choose the columns that we want it to be features and labels
    features, outputs, trainDataset = handleTrainData(trainDataset, fullBand, exceptBand, timestamps, outputLabel)
    XTrainInput, yTrainLabel = trainDataset.loc[:,features], trainDataset.loc[:, outputs[0]] # features and label first
    XValInput, yValLabel = valDataset.loc[:,bandType], valDataset.loc[:,outputLabel] #  features and label

    del trainDataset, valDataset

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    XTrainInput = XTrainInput.values.astype('float32')
    yTrainLabel = yTrainLabel.values.astype('float32')
    XValInput = XValInput.values.astype('float32')
    yValLabel = yValLabel.values.astype('float32')

    XValInput = torch.tensor(XValInput.reshape(-1, timestamps * 2, len(bandType) // (timestamps * 2)), dtype= torch.float32, device= device)
    yValLabel = torch.tensor(yValLabel, dtype= torch.float32, device= device)
    
    # 6 khoảng thời gian, 10 bands -> 33 bands
    XTrainInput = torch.tensor(XTrainInput.reshape(-1, timestamps, len(bandType) // (timestamps * 2)), dtype=torch.float32, device=device)
    yTrainLabel = torch.tensor(yTrainLabel, dtype=torch.float32, device=device)
    

    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTrain = TensorDataset(XTrainInput, yTrainLabel)
    TensorDSVal = TensorDataset(XValInput, yValLabel)
    del XTrainInput, yTrainLabel, XValInput, yValLabel

    train_dataset = DataLoader(
        TensorDSTrain,
        batch_size= batch_size,
        shuffle=True
    )

    val_dataset = DataLoader(
        TensorDSVal,
        batch_size= batch_size,
        shuffle=False
    )

    del TensorDSTrain, TensorDSVal
    return train_dataset, val_dataset

