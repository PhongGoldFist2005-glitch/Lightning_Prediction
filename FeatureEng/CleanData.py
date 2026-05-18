import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import numpy as np
import logging
import polars as pl

# *
file_log = '/sdd/Dubaoset/src/Phong/Model/data/testClean/clean_data.log'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(file_log)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class cleanData:
    def __init__(self, inputDf):
        self.inputDf = inputDf
    def handleDuplicate(self):
        oldLen = len(self.inputDf)
        self.inputDf = self.inputDf.drop_duplicates().reset_index(drop=True)
        return f"length of old df: {oldLen}, length of removed duplicated dataframe: {len(self.inputDf)}"
    def handleMissingValue(self, band, filledValue, newValue, timeStamps):
        if not isinstance(filledValue, float):
            return
        # Band with many timestamp
        if timeStamps is not None:
            bandList = []
            for time in timeStamps:
                bandTime = f"{band}_t+{time}" if time >= 0 else f"{band}_t{time}"
                bandList.append(bandTime)
            if len(bandList) == 0:
                return
            self.inputDf.loc[:, bandList] = self.inputDf.loc[:, bandList].replace(filledValue, newValue)
        else:
            self.inputDf.loc[:, band] = self.inputDf.loc[:, band].replace(filledValue, newValue)
            return True
    def handleMissingValueWithMask(self, band, filledValue, newValue, timeStamps, additionCol):
        # Band with many timestamp
        if timeStamps is not None:
            bandList = []
            addCol = []
            for time in timeStamps:
                bandTime = f"{band}_t+{time}" if time >= 0 else f"{band}_t{time}"
                colTime = f"{additionCol}_t+{time}" if time >= 0 else f"{additionCol}_t{time}"
                bandList.append(bandTime)
                addCol.append(colTime)
            
            if len(bandList) == 0 or len(addCol) == 0:
                return
            # Default value của những mask là 1
            self.inputDf[addCol] = np.int8(1)
            # Thay những vị trí trong addCol mà bandList cùng t có giá trị -9999.0
            # Duyệt từng time, loc ra từng band tại t mà có giá trị -9999.0 thì cho col tại đó bằng 0
            for bandTime, colTime in zip(bandList, addCol):
                mask = self.inputDf[bandTime] == -9999.0
                self.inputDf.loc[mask, colTime] = np.int8(0)
            self.inputDf.loc[:, bandList] = self.inputDf.loc[:, bandList].replace(filledValue, newValue)
        else:
            self.inputDf[additionCol] = 1
            mask = self.inputDf[band] == -9999.0
            self.inputDf.loc[mask, additionCol] = 0
            self.inputDf.loc[:, band] = self.inputDf.loc[:, band].replace(filledValue, newValue)

    def handleDataType(self, inputCol, inputDataType):
        if inputDataType is not None:
            # inputCol is a list
            dictType = {item:inputDataType for item in inputCol}
            self.inputDf = self.inputDf.astype(dictType)
        else:
            # inputCol is already a dict
            self.inputDf = self.inputDf.astype(inputCol)
        return
    def handleOutlinear(self, band):
        plt.figure(figsize=(12, 4))

        plt.subplot(1,2,1)
        plt.boxplot(self.inputDf[band])
        plt.title(f"Boxplot of {band}")

        plt.subplot(1,2,2)
        plt.hist(self.inputDf[band])
        plt.title(f"Histogram of {band}")

        plt.show()
    
    def handleOutputFile(self, outputPath):
        if os.path.exists(outputPath):
            logger.info("File has already existed")
            return
        self.inputDf.to_parquet(outputPath)

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

            self.inputDf.loc[:, colName] = (numpyArray - minValue) / (maxValue - minValue)
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

class findInfoOfBand:
    def __init__(self,inputDf, outputFile, timeStamps):
        self.inputDf = inputDf
        self.outputFile = outputFile
        self.timeStamps = timeStamps
    # Not for DEM
    def writeInfoBand(self, bandInput):
        with open(self.outputFile, "a", encoding='utf-8') as f:
            for band in bandInput:
                listOfBand = []
                for i in self.timeStamps:
                    bandName = f"{band}_t{i:+d}"
                    listOfBand.append(bandName)
                df_long = self.inputDf[listOfBand].values.flatten()
                df_long = pd.Series(df_long).describe().to_dict()
                inputData = {band: df_long}
                json_record = json.dumps(inputData, ensure_ascii=False)
                f.write(json_record + '\n')
        return True


if __name__ == "__main__":
    # Define bandType and outputLabel
    bandName = [
        'B04B','B05B','B06B','VSB','B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB','NDVI'
    ]
    # Choose time bands (*)
    startPeriod = -6
    endPeriod = 6

    seaDEMValue = -2603
    bandType = [
        f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}"
        for i in range(startPeriod, endPeriod)
        for band in bandName
    ] + ['Dem_value']

    additionDataType = { 
        "year": "int16","month": "int8","day": "int8", "hour": "int8","minute": "int8","row": "int32","col": "int32",
        "output_0": "int8", "output_1": "int8", "output_2": "int8", "output_3": "int8", "output_4": "int8", "output_5": "int8"
    }

    bandType += ["lightning_0","lightning_1","lightning_2","lightning_3","lightning_4","lightning_5"]
    # *
    fullFile = [
        os.path.join("/sdd/Dubaoset/src/Phong/Model/data/Test_NDVI_DEM", f)
        for f in os.listdir("/sdd/Dubaoset/src/Phong/Model/data/Test_NDVI_DEM")
        if f.endswith(".parquet")
    ]
    # Setup Logger
    logging.info(f"Found {len(fullFile)} files to process.")

    for filepath in tqdm(fullFile, total= len(fullFile)):
        file_name = os.path.basename(filepath)
        logging.info(f"Processing file: {file_name}")
        result = pl.read_parquet(filepath)
        result = result.to_pandas()
       
        cleanObject = cleanData(result)
        logger.info("Create cleaner completed")
        checker = cleanObject.handleDuplicate()
        logger.info("Remove duplicate completed")

        timeStamps = [i for i in range(-6, 6)]
        cleanObject.handleMissingValueWithMask("NDVI", -9999.0, -1, timeStamps,"NDVIIsLand")
        cleanObject.handleMissingValueWithMask("Dem_value", -9999.0, seaDEMValue, None,"DEMIsLand")
        logger.info("Handle missing value completed")
        cleanObject.handleDataType(bandType, "float32")
        cleanObject.handleDataType(additionDataType, None)
        logger.info("Handle data type completed")
        # *
        outputFile = f"/sdd/Dubaoset/src/Phong/Model/data/testClean/{file_name}"
        cleanObject.handleOutputFile(outputFile)
        logger.info(f"Finished processing file: {file_name}")
    