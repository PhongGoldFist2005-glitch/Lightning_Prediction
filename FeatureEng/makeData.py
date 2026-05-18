# lấy 20.000 mẫu random
# Tạo band kép
# Normalize
# Lấy X và y tương ứng
import pandas as pd
import polars as pl
from CleanData import normalize
import os
from tqdm import tqdm
import logging
import json
import gc

file_log = '/sdd/Dubaoset/src/Phong/Model/logs/makeData.log'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(file_log)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

    def normalBand(self, band, typeOfNormal, keyValue):
        bandList = self.listCol(band)
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

if __name__ == "__main__":
    timeStamps = [i for i in range(-6, 0)]
    fileList = [
        "/sdd/Dubaoset/src/Phong/Model/data/testClean/merged_data_part_0.parquet"
    ]
    diffBand = [
        'B11B-B12B','WVB-B14B','B11B-IRB','WVB-IRB','IRB-I2B'
    ]
    singleBand = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB']
    listOfOutput = [
        "output_0"
    ]
    jsonInfo = "/sdd/Dubaoset/src/Phong/Model/data/describe.jsonl"
    fullCol = singleBand + diffBand
    for file in tqdm(fileList, total= len(fileList)):
        df = pl.read_parquet(file).sample(n=20000).to_pandas()
        logging.info(f"Reading {os.path.basename(file)} completed")
        analystObject = createDataForAnalyst(df, timeStamps, jsonInfo)
        logging.info(f"Created Object completed")
        analystObject.takeRandomSample()
        analystObject.createDiffBand(diffBand)
        analystObject.selectedCol(fullCol, listOfOutput)
        logging.info(f"Create collumns completed")
        # Single Band First
        for band in fullCol:
            # Hàm lấy min max value
            analystObject.normalBand(band, "MinMaxScaler", keyValue= ["min", "max"])
        logging.info(f"Normalize completed")
        booleanValue = analystObject.getXandyValue(fullCol, listOfOutput, f"/sdd/Dubaoset/src/Phong/Source/addInput/XTest/X_{os.path.basename(file)}", f"/sdd/Dubaoset/src/Phong/Source/addInput/yTest/y_{os.path.basename(file)}")
        if booleanValue == False:
            logging.info("file existed")
        logging.info(f"Processing {os.path.basename(file)} completed")
        del analystObject, df
        gc.collect()


