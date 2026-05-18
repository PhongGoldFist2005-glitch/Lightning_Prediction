import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import logging
import pandas as pd
import numpy as np


bandName = [
    'B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB','NDVI','CAPE','EWSS','IE','ISOR','KX',
    'PEV','R250','R500','R850','SLHF','SLOR','SSHF','TCLW','TCW','TCWV','U250','U500','U850','V250','V500','V850','Dem_value'
]

startPeriod = -6
endPeriod = 6


demLabel = [
    f'Dem_value_t{("-" + str(-i)) if i < 0 else ("+" + str(i))}'
    for i in range(startPeriod, endPeriod)
]
# print(demLabel)

bandType = [
    f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}" if band != "Dem_value" else "Dem_value"
    for i in range(startPeriod, endPeriod)
    for band in bandName
]

startPeriodTrain = -6
endPeriodTrain = 0
bandTypeTrain = [
    f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}" if band != "Dem_value" else "Dem_value"
    for i in range(startPeriodTrain, endPeriodTrain)
    for band in bandName
]


outputLabel = [
    "output_0",
    "output_1",
    "output_2",
    "output_3",
    "output_4",
    "output_5"
]

# Dem 1, ERA5 21, Hima: 10, NDVI: 1.
# 1 lựa chọn sẽ là tạo thêm 5 cột Dem trong dữ liệu

def returnDataset(trainValDataset, logFile, device, batch_size, timestamps, outputLabel= outputLabel, bandType= bandType, bandTypeTrain= bandTypeTrain):
    # Lọc dữ các dòng dữ liệu không hợp lệ
    # Rename columns
    trainValDataset = trainValDataset.dropna(subset=bandTypeTrain)

    # Train test split
    # Không shuffle để đảm bảo mỗi chunk khi lặp lại không bị trộn dữ liệu train và test
    # Chỉ shuffle batch train
    trainDataset, valDataset = train_test_split(trainValDataset ,test_size= 0.2, random_state= 42)
    print(len(outputLabel), "output labels chosen for training.")
    print(len(bandTypeTrain), "band types chosen for training.")

    # Choose the columns that we want it to be features and labels
    XTrainInput, yTrainLabel = trainDataset.loc[:,bandTypeTrain], trainDataset.loc[:,outputLabel] # features and label first
    XValInput, yValLabel = valDataset.loc[:,bandTypeTrain], valDataset.loc[:,outputLabel] #  features and label

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    XTrainInput = XTrainInput.to_numpy().astype(float)
    XValInput = XValInput.to_numpy().astype(float)

    yTrainLabel = yTrainLabel.to_numpy().astype(float)
    yValLabel = yValLabel.to_numpy().astype(float)

    yTrainLabel = torch.tensor(yTrainLabel, dtype=torch.float32, device=device)
    yValLabel = torch.tensor(yValLabel, dtype=torch.float32, device=device)

    # 6 khoảng thời gian, 10 bands -> 33 bands
    XTrainInput = torch.tensor(XTrainInput.reshape(-1, timestamps, len(bandTypeTrain) // timestamps), dtype=torch.float32, device=device)
    # 6 khoảng thời gian, 10 bands -> 33 bands
    XValInput = torch.tensor(XValInput.reshape(-1, timestamps, len(bandTypeTrain) // (timestamps)), dtype=torch.float32, device=device)

    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTrain = TensorDataset(XTrainInput, yTrainLabel)
    TensorDSVal = TensorDataset(XValInput, yValLabel)

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

    return train_dataset, val_dataset
    
def returnTestDataset(testDataFrame, device, batch_size, timestamps, outputLabel= outputLabel, bandType= bandType):
    # Lọc dữ các dòng dữ liệu không hợp lệ
    # Truyền vào test để ý band type là bandTypeTrain
    testDataFrame = testDataFrame.dropna(subset=bandType)

    # Choose the columns that we want it to be features and labels
    # 12 thời gian,10 bands
    # 6 output
    XTestInput, yTestLabel = testDataFrame.loc[:,bandType], testDataFrame.loc[:,outputLabel] # features and label

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    # Test 6 khoảng thời gian thôi 
    XTestInput = XTestInput.to_numpy().astype(float)
    XTestInput = torch.tensor(XTestInput.reshape(-1, timestamps, len(bandType) // timestamps), dtype=torch.float32, device=device)

    yTestLabel = yTestLabel.to_numpy().astype(float)
    yTestLabel = torch.tensor(yTestLabel, dtype=torch.float32, device=device)


    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTest = TensorDataset(XTestInput, yTestLabel)

    testDataset = DataLoader(
        TensorDSTest,
        batch_size= batch_size,
        shuffle= False
    )

    return testDataset