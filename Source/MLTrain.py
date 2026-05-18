import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("/sdd/Dubaoset")
from src.Phong.Model.Source.dataHandle import MLHandleData
from sklearn.metrics import recall_score, precision_score, f1_score
from contextlib import redirect_stdout

def trainMLModel(inputFileList, model, bandTypeTrain, bandName, outputLabel, startTime, endTime, step, logFile, outputResult, modelPath):
    # Concat tất cả dữ liệu
    trainList = []
    for file in tqdm(inputFileList, total= len(inputFileList)):
        data = pd.read_parquet(file)
        trainList.append(data)
    fullData = pd.concat(trainList, ignore_index= True)
    # Giải phong memory khi không còn dùng nữa
    del trainList
    
    # Split dữ liệu
    (X_train, y_train), valList = MLHandleData(
        inputData = fullData, 
        bandTypeTrain = bandTypeTrain, 
        bandName = bandName, 
        labelList = outputLabel, 
        startTime = startTime, 
        endTime = endTime, 
        step = step
    )
    eval_set = [(X_train, y_train)] + valList
    # Giải phong memory khi không còn dùng nữa
    del fullData
    
    # Giúp ghi log vào file khác
    # fit model vào dữ liệu
    with open(logFile, "a") as f:
        with redirect_stdout(f):
            model.fit(
                X_train, y_train,
                eval_set= eval_set,
                verbose=True
            )
    
    # Sau khi train xong thì tính eval metrics
    # Lưu vào file
    with open(outputResult, "a") as f:
        best_iter = model.best_iteration
        best_score = model.best_score
        f.write(f"Early stop at :{best_iter}\n")
        f.write(f"Best score: {best_score}\n")
        for idx, (X, y) in enumerate(valList):
            pred = model.predict(X)
            recallValue = recall_score(y, pred, zero_division=0)
            precisionValue = precision_score(y, pred, zero_division=0)
            f1Value = f1_score(y, pred, zero_division=0)
            f.write(f"time stamps t + {idx}, recall: {recallValue}, precision: {precisionValue}, f1: {f1Value}\n")
    
    # Save model
    # file json
    model.save_model(modelPath)
    print("Train complete")