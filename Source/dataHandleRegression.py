import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def returnDatasetOutput1(trainValDataset, logFile, 
    device, batch_size, timestamps, outputLabel, bandType, 
    bandTypeTrain,
    scaler_X=None, scaler_y=None
    ):
    # Rename columns
    trainValDataset = trainValDataset.dropna(subset=bandType)

    # Train test split
    # Không shuffle để đảm bảo mỗi chunk khi lặp lại không bị trộn dữ liệu train và test
    # Chỉ shuffle batch train
    trainDataset, valDataset = train_test_split(trainValDataset ,test_size= 0.2, random_state= 42)

    # Choose the columns that we want it to be features and labels
    XTrainInput, yTrainLabel = trainDataset.loc[:,bandTypeTrain], trainDataset.loc[:,outputLabel[0]] # features and label first
    XValInput, yValLabel = valDataset.loc[:,bandType], valDataset.loc[:,outputLabel] #  features and label

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    XTrainInput = XTrainInput.to_numpy().astype(float)
    XValInput = XValInput.to_numpy().astype(float)

    XTrainInput = XTrainInput.reshape(-1, timestamps, len(bandTypeTrain) // timestamps)
    XValInput = XValInput.reshape(-1, timestamps * 2, len(bandType) // (timestamps * 2))

    yTrainLabel = yTrainLabel.to_numpy().astype(float)
    yValLabel = yValLabel.to_numpy().astype(float)

    # Chuẩn hóa X
    if scaler_X is not None:
        # print("Scale X")
        for t in range(XTrainInput.shape[1]):
            XTrainInput[:, t, :] = scaler_X.transform(XTrainInput[:, t, :])
        for t in range(XValInput.shape[1]):
            XValInput[:, t, :] = scaler_X.transform(XValInput[:, t, :])

    # Chuẩn hóa y sẽ áp dụng bên dùng, cho dễ tùy biến
    if scaler_y is not None:
        yTrainLabel = scaler_y.transform(yTrainLabel.reshape(-1, 1))
        # yValLabel   = scaler_y.transform(yValLabel.reshape(-1, len(outputLabel)))

    yTrainLabel = torch.tensor(yTrainLabel, dtype=torch.float32, device=device)
    yValLabel = torch.tensor(yValLabel, dtype=torch.float32, device=device)

    # 6 khoảng thời gian, 10 bands -> 33 bands
    XTrainInput = torch.tensor(XTrainInput, dtype=torch.float32, device=device)
    # 12 khoảng thời gian, 10 bands -> 33 bands
    XValInput = torch.tensor(XValInput, dtype=torch.float32, device=device)

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

# 6 đầu ra
def returnDatasetOutput6(trainValDataset, logFile, 
    device, batch_size, timestamps, outputLabel, bandType, 
    bandTypeTrain,
    scaler_X=None, scaler_y=None
    ):
    # Rename columns
    trainValDataset = trainValDataset.dropna(subset=bandTypeTrain)

    # Train test split
    # Không shuffle để đảm bảo mỗi chunk khi lặp lại không bị trộn dữ liệu train và test
    # Chỉ shuffle batch train
    trainDataset, valDataset = train_test_split(trainValDataset ,test_size= 0.2, random_state= 42)

    # Choose the columns that we want it to be features and labels
    XTrainInput, yTrainLabel = trainDataset.loc[:,bandTypeTrain], trainDataset.loc[:,outputLabel] # features and label 
    XValInput, yValLabel = valDataset.loc[:,bandTypeTrain], valDataset.loc[:,outputLabel] #  features and label

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    XTrainInput = XTrainInput.to_numpy().astype(float)
    XValInput = XValInput.to_numpy().astype(float)

    yTrainLabel = yTrainLabel.to_numpy().astype(float)
    yValLabel = yValLabel.to_numpy().astype(float)

    # Chuẩn hóa X
    if scaler_X is not None:
        # print("Scale X")
        XTrainInput = scaler_X.transform(XTrainInput)
        XValInput   = scaler_X.transform(XValInput)

    # Chuẩn hóa y
    if scaler_y is not None:
        yTrainLabel = scaler_y.transform(yTrainLabel.reshape(-1, len(outputLabel))).ravel()
        yValLabel   = scaler_y.transform(yValLabel.reshape(-1, len(outputLabel)))

    yTrainLabel = torch.tensor(yTrainLabel, dtype=torch.float32, device=device)
    yValLabel = torch.tensor(yValLabel, dtype=torch.float32, device=device)

    # 6 khoảng thời gian, 10 bands -> 33 bands
    XTrainInput = torch.tensor(XTrainInput.reshape(-1, timestamps, len(bandTypeTrain) // timestamps), dtype=torch.float32, device=device)
    # 12 khoảng thời gian, 10 bands -> 33 bands
    XValInput = torch.tensor(XValInput.reshape(-1, timestamps, len(bandType) // (timestamps)), dtype=torch.float32, device=device)

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

# Cho 1 đầu ra thì load 12 để dịch còn 6 đầu ra thì load 6 thôi không cần dịch
# BandType
def returnTestDataset1(testDataFrame, device, batch_size, timestamps, outputLabel, bandType, scaler_X=None, scaler_y=None):
    # Lọc dữ các dòng dữ liệu không hợp lệ
    testDataFrame = testDataFrame.dropna(subset=bandType)

    # Choose the columns that we want it to be features and labels
    # 12 thời gian,10 bands
    # 6 output
    XTestInput, yTestLabel = testDataFrame.loc[:,bandType], testDataFrame.loc[:,outputLabel] # features and label

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    XTestInput = XTestInput.to_numpy().astype(float)

    # Chuẩn hóa X
    if scaler_X is not None:
        # print("Scale X")
        XTestInput = scaler_X.transform(XTestInput)

    XTestInput = torch.tensor(XTestInput.reshape(-1, timestamps * 2, len(bandType) // (timestamps * 2)), dtype=torch.float32, device=device)

    yTestLabel = yTestLabel.to_numpy().astype(float)
    # Chuẩn hóa y
    if scaler_y is not None:
        yTestLabel = scaler_y.transform(yTestLabel.reshape(-1, len(outputLabel))).ravel()

    yTestLabel = torch.tensor(yTestLabel, dtype=torch.float32, device=device)


    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTest = TensorDataset(XTestInput, yTestLabel)

    testDataset = DataLoader(
        TensorDSTest,
        batch_size= batch_size,
        shuffle= False
    )

    return testDataset

# TRuyền vào bandTrainType thôi vì chỉ cần 6 đầu vào đầu thôi không cần dịch
def returnTestDataset6(testDataFrame, device, batch_size, timestamps, outputLabel, bandType, scaler_X=None, scaler_y=None):
    # Lọc dữ các dòng dữ liệu không hợp lệ
    testDataFrame = testDataFrame.dropna(subset=bandType)

    # Choose the columns that we want it to be features and labels
    # 12 thời gian,10 bands
    # 6 output
    XTestInput, yTestLabel = testDataFrame.loc[:,bandType], testDataFrame.loc[:,outputLabel] # features and label

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    XTestInput = XTestInput.to_numpy().astype(float)

    # Chuẩn hóa X
    if scaler_X is not None:
        # print("Scale X")
        XTestInput = scaler_X.transform(XTestInput)

    XTestInput = torch.tensor(XTestInput.reshape(-1, timestamps, len(bandType) // (timestamps)), dtype=torch.float32, device=device)

    yTestLabel = yTestLabel.to_numpy().astype(float)
    # Chuẩn hóa y
    if scaler_y is not None:
        yTestLabel = scaler_y.transform(yTestLabel.reshape(-1, len(outputLabel))).ravel()

    yTestLabel = torch.tensor(yTestLabel, dtype=torch.float32, device=device)


    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTest = TensorDataset(XTestInput, yTestLabel)

    testDataset = DataLoader(
        TensorDSTest,
        batch_size= batch_size,
        shuffle= False
    )

    return testDataset