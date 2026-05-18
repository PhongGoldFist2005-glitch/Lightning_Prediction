import pandas as pd
import polars as pl
import sys
sys.path.append("/sdd/Dubaoset/src/Phong/Model/Tuning")
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

            for time in self.timeStamps:
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
        for time in self.timeStamps:
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
        # normalObject cũ không có các cột mới
        normalObject = normalize(self.inputDf)
        normalObject.normalization(bandList, typeOfNormal, dataInfo)
        self.inputDf = normalObject.inputDf
        del normalObject

    def getXandyValue(self, listOfBand, listOfOutput, outputXFile, outputYFile):
        bandList = self.listCol(listOfBand)
        X = self.inputDf.loc[:, bandList].copy()
        y = self.inputDf.loc[:, listOfOutput].copy()

        if os.path.exists(outputXFile) and os.path.exists(outputYFile):
            return False
        X.to_parquet(outputXFile)
        y.to_parquet(outputYFile)
        return True