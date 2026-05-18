import pandas as pd
import polars as pl
import sys
sys.path.append("/sdd/Dubaoset/src/Thang/TrainProcessMB/code")
from normalization import normalize
import os
from tqdm import tqdm
import logging
import json
import gc

class createDataForAnalyst:
    def __init__(self, inputDf, timeStamps, inputInfo):
        self.inputDf = inputDf
        self.timeStamps = timeStamps
        self.inputInfo = inputInfo
        with open(self.inputInfo, "r", encoding="utf-8") as f:
            self.listOfBandInfo = {}
            for line in f:
                data = json.loads(line)
                self.listOfBandInfo.update(data)

    def takeRandomSample(self):
        self.inputDf = self.inputDf.sample(frac=1).reset_index(drop=True)
        return True

    # timeStamps chỉ từ -6 đến 0 cho tập train
    # format band1-band2
    def createDiffBand(self, diffBands):
        new_cols = {}
        listOfCol = list(self.inputDf.columns)

        for diffBand in diffBands:
            arrayBand = diffBand.strip().split("-")
            if len(arrayBand) != 2:
                continue

            bandT1, bandT2 = arrayBand

            for time in range(-self.timeStamps, self.timeStamps):
                timeName = f"+{time}" if time >= 0 else f"{time}"
                bandTimeT1 = f"{bandT1}_t{timeName}"
                bandTimeT2 = f"{bandT2}_t{timeName}"
                if bandTimeT1 not in listOfCol or bandTimeT2 not in listOfCol:
                    continue

                new_col_name = f"{bandT1}-{bandT2}_t{timeName}"

                new_cols[new_col_name] = (
                    self.inputDf[bandTimeT1].values
                    - self.inputDf[bandTimeT2].values
                )
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=self.inputDf.index)
            self.inputDf = pd.concat([self.inputDf, new_df], axis=1)
            del new_df
        return True

    def listCol(self, listOfBand):
        bandList = []
        if isinstance(listOfBand, str):
            listOfBand = [listOfBand]
        for time in range(-self.timeStamps, self.timeStamps):
            for band in listOfBand:
                timeName = f"+{time}" if time >= 0 else f"{time}"
                bandTime = f"{band}_t{timeName}"
                bandList.append(bandTime)
        return bandList

    def selectedCol(self, listOfBand, listOfOutput):
        bandList = self.listCol(listOfBand)
        fullList = bandList + listOfOutput
        self.inputDf = self.inputDf.loc[:, fullList]

    def takeInfoTrainData(self, bandInput, keyValue):
        if isinstance(keyValue, (float, int)):
            keyValue = [keyValue]
        outputInfo = []
        for key in keyValue:
            dataInfo = self.listOfBandInfo[bandInput][key]
            outputInfo.append(dataInfo)
        return outputInfo

    def normalBand(self, band, exceptBand, typeOfNormal, keyValue):
        if band not in exceptBand:
            bandList = self.listCol(band)
        else:
            bandList = [band]
        dataInfo = self.takeInfoTrainData(band, keyValue)
        
        # ✅ DEBUG: Check trước khi normalize
        nan_before = self.inputDf[bandList].isna().sum().sum()
        if nan_before > 0:
            print(f"⚠️ WARNING: {band} có {nan_before} NaN trước normalize")
        
        # ✅ DEBUG: Check min/max thực tế vs info file
        actual_min = self.inputDf[bandList].min().min()
        actual_max = self.inputDf[bandList].max().max()
        info_min, info_max = dataInfo if typeOfNormal == "MinMaxScaler" else (None, None)
        
        if info_min is not None and info_max is not None:
            print(f"Band {band}: Info=[{info_min}, {info_max}], Actual=[{actual_min}, {actual_max}]")
            if actual_min < info_min or actual_max > info_max:
                print(f"⚠️ WARNING: Dữ liệu thực tế vượt ra ngoài info range!")
        
        # normalObject cũ không có các cột mới
        normalObject = normalize(self.inputDf)
        normalObject.normalization(bandList, typeOfNormal, dataInfo)
        self.inputDf = normalObject.inputDf
        del normalObject
        
        # ✅ DEBUG: Check sau khi normalize
        nan_after = self.inputDf[bandList].isna().sum().sum()
        if nan_after > 0:
            print(f"❌ ERROR: {band} tạo ra {nan_after} NaN sau normalize!")

    def getXandyValue(self, listOfBand, listOfOutput, outputXFile, outputYFile):
        bandList = self.listCol(listOfBand)
        X = self.inputDf.loc[:, bandList].copy()
        y = self.inputDf.loc[:, listOfOutput].copy()

        if os.path.exists(outputXFile) and os.path.exists(outputYFile):
            return False
        X.to_parquet(outputXFile)
        y.to_parquet(outputYFile)
        return True