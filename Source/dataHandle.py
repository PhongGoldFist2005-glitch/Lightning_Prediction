import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import logging
import pandas as pd
import numpy as np
from .configData import createDataForAnalyst
import polars as pl
from tqdm import tqdm

bandName = [
    'B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB'
]

startPeriod = -6
endPeriod = 6


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

timeStamps = [i for i in range(startPeriod, endPeriod)]
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
    # Train test split
    # Không shuffle để đảm bảo mỗi chunk khi lặp lại không bị trộn dữ liệu train và test
    # Chỉ shuffle batch train
    trainDataset, valDataset = train_test_split(trainValDataset ,test_size= 0.2, random_state= 42)

    # Choose the columns that we want it to be features and labels
    XTrainInput, yTrainLabel = trainDataset.loc[:,bandTypeTrain], trainDataset.loc[:,outputLabel[0]] # features and label first
    XValInput, yValLabel = valDataset.loc[:,bandType], valDataset.loc[:,outputLabel] #  features and label
    del trainDataset, valDataset

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
    # 12 khoảng thời gian, 10 bands -> 33 bands
    XValInput = torch.tensor(XValInput.reshape(-1, timestamps * 2, len(bandType) // (timestamps * 2)), dtype=torch.float32, device=device)

    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTrain = TensorDataset(XTrainInput, yTrainLabel)
    TensorDSVal = TensorDataset(XValInput, yValLabel)

    del XTrainInput, XValInput, yTrainLabel, yValLabel

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
    
def returnTestDataset(testDataFrame, device, batch_size, timestamps, outputLabel= outputLabel, bandType= bandType):
    
    # 2. Lấy dữ liệu và ép kiểu float32 NGAY LẬP TỨC để tiết kiệm 50% RAM
    # Sử dụng .values thay vì .to_numpy() để lấy thẳng mảng gốc
    # testDataFrame = testDataFrame.dropna(subset=bandType)
    XTestInput, yTestLabel = testDataFrame.loc[:,bandType], testDataFrame.loc[:,outputLabel]

    # print(len(XTestInput.columns), len(yTestLabel.columns))

    X_np = XTestInput.values.astype('float32')
    y_np = yTestLabel.values.astype('float32')

    # 3. Reshape và tạo Tensor trên CPU (KHÔNG đưa vào device ở đây)
    # Dùng torch.from_numpy để không tốn thêm RAM copy dữ liệu
    X_tensor = torch.from_numpy(X_np).reshape(-1, timestamps * 2, len(bandType) // (timestamps * 2))
    y_tensor = torch.from_numpy(y_np)

    # 4. Tạo Dataset
    TensorDSTest = TensorDataset(X_tensor, y_tensor)

    # 5. DataLoader quan trọng: dùng pin_memory để đẩy lên GPU nhanh nhất
    testDataset = DataLoader(
        TensorDSTest,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True, # Cực kỳ quan trọng để tăng tốc truyền dữ liệu sang GPU
        num_workers=2    # Tận dụng đa nhân xử lý batch
    )

    return testDataset
    

def imageHandle(bandImage, labelImage, device, batch_size):
    # Keys: images, labels, lightning_value
    # Shape: Bands, timeStamp, height, width
    lenTrain = 0

    if len(bandImage) == len(labelImage):
        lenData = len(bandImage)
        lenTrain = int(lenData * 0.8) # 80%
    else:
        print("Mismatch data size")
        return None
    
    bandImageTrain, bandImageVal = bandImage[: lenData], bandImage[lenData : ]
    labelImageTrain, labelImageVal = labelImage[: lenData], labelImage[lenData : ]

    bandTrainTensor = torch.tensor(bandImageTrain, device= device, dtype= torch.float32)
    bandValTensor = torch.tensor(bandImageVal, device= device, dtype= torch.float32)

    labelTrainTensor = torch.tensor(labelImageTrain, device= device, dtype= torch.float32)
    labelValTensor = torch.tensor(labelImageVal, device= device, dtype= torch.float32)

    tensorTrainData = TensorDataset(
        bandTrainTensor, labelTrainTensor
    )

    tensorValData = TensorDataset(
        bandValTensor, labelValTensor
    )

    trainData = DataLoader(
        tensorTrainData,
        batch_size= batch_size,
        shuffle= True
    )

    testData = DataLoader(
        tensorValData,
        batch_size = batch_size,
        shuffle= False
    )

    return trainData, testData

def imageTestHandle(bandImage, labelImage, device, batch_size):
    # Keys: images, labels, lightning_value
    # Shape: Bands, timeStamp, height, width
    bandTestTensor = torch.tensor(bandImage, device= device, dtype= torch.float32)
    labelTestTensor = torch.tensor(labelImage, device= device, dtype= torch.float32)

    tensorTestData = TensorDataset(
        bandTestTensor, labelTestTensor
    )

    testData = DataLoader(
        tensorTestData,
        batch_size = batch_size,
        shuffle= False
    )

    return testData

def MLHandleData(inputData, bandTypeTrain, bandName, labelList, startTime = -6, endTime = 0, step = 6):
    trainSet, valSet = train_test_split(inputData ,test_size=0.2, random_state=42)
    X_train, y_train = trainSet.loc[:, bandTypeTrain].copy(), trainSet.loc[:, labelList[0]].copy()
    valList = []

    for i in range(startTime, endTime):
        startPeriod = i
        endPeriod = i + step

        bandType = [
            f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}"
            for i in range(startPeriod, endPeriod)
            for band in bandName
        ]

        X_val, y_val = valSet.loc[:, bandType].copy(), valSet.loc[:, labelList[endPeriod]].copy()
        X_val.columns = bandTypeTrain
        valList.append((X_val, y_val))
    valList.reverse()
    return ((X_train, y_train), valList)

def MLHandleTestData(inputData, bandTypeTrain, bandName, labelList, startTime = -6, endTime = 0, step = 6):
    testList = []

    for i in range(startTime, endTime):
        startPeriod = i
        endPeriod = i + step

        bandType = [
            f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}"
            for i in range(startPeriod, endPeriod)
            for band in bandName
        ]

        X_test, y_test = inputData.loc[:, bandType].copy(), inputData.loc[:, labelList[endPeriod]].copy()
        X_test.columns = bandTypeTrain
        testList.append((X_test, y_test))
    testList.reverse()
    return testList

def loadedFullDataset(inputFileList, diffBand, timeStamps, inputInfo, fullBand):
    dataFrames = []
    for file in tqdm(inputFileList, total= len(inputFileList)):
        df = pl.read_parquet(file).to_pandas()
        dataFrames.append(df)
    
    fullDataSet = pd.concat(dataFrames, ignore_index= True)
    dataObject = createDataForAnalyst(fullDataSet, timeStamps, inputInfo)
    if diffBand is not None:
        dataObject.createDiffBand(diffBand)
    for band in fullBand:
        # Hàm lấy min max value
        dataObject.normalBand(band, "MinMaxScaler", keyValue= ["min", "max"])
    print("Create diff band completed")
    return dataObject.inputDf

def loadDiffDataset(singleBandDf, diffBandDf, singleBandName, diffBandName, outputLabel, batch_size, timeStamps, device):
    if not isinstance(singleBandName, list):
        singleBandName = [singleBandName]
    if not isinstance(diffBandName, list):
        diffBandName = [diffBandName]

    fullData = pd.concat([singleBandDf, diffBandDf], axis= 1)
    del singleBandDf, diffBandDf

    fullBandName = singleBandName + diffBandName
    bandTime = [
        f"{band}_t{i:+d}" if band != "Dem_value" else "Dem_value"
        for i in timeStamps
        for band in fullBandTime
    ]

    # Choose the columns that we want it to be features and labels
    XTestInput, yTestLabel = fullData[bandTime].to_numpy().astype(float), fullData[outputLabel].to_numpy().astype(float)
    del fullData

    # We change input format into (Batch, num of units, features) which has datatype is tensor
    # Split to format(num of units,features) first
    # numpy -> format(num of units,features) -> tensor
    yTestLabel = torch.tensor(yTestLabel, dtype=torch.float32, device=device)

    # timeStamps thuộc [-6:6]
    XTestInput = torch.tensor(XTestInput.reshape(-1, timeStamps, len(bandTime) // timeStamps), dtype=torch.float32, device=device)

    # Combine to make a tensor-dataset and split it into small batch
    TensorDSTest = TensorDataset(XTestInput, yTestLabel)

    del XTestInput, yTestLabel

    test_dataset = DataLoader(
        TensorDSTest,
        batch_size= batch_size,
        shuffle=False
    )

    del TensorDSTest
    return test_dataset



