from evalScore import (
    eval_stepwise_forward, eval_stepwise_backward,
    eval_rfe, eval_point_biserial, eval_mutual_information, takeTimeBand
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import polars as pl
import pandas as pd
import logging

logger = logging.getLogger("feature_selection")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("/sdd/Dubaoset/src/Phong/Source/addInput/Result/logging.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

if __name__ == "__main__":
    startPos = 0
    endPos = 45
    stepPos = 5

    fullBand = [
        "IRB-I2B", "B09B-B14B", "WVB-B14B","B11B-B14B","I2B-B14B",
        "WVB-IRB","B11B-IRB","B14B-I2B","B11B-B12B","WVB-B10B",
        "B09B","B10B","B11B","B12B","B14B","B16B","I2B","I4B","IRB","WVB"
    ]
    timeStamps = [i for i in range(-6 , 0)]
    bandList = []
    for band in fullBand:
        bandList += takeTimeBand(band, timeStamps)
    

    for i in range(startPos, endPos, stepPos):
        XFileName = f"/sdd/Dubaoset/src/Phong/Source/addInput/X/X_merged_data_part_{i}.parquet"
        yFileName = f"/sdd/Dubaoset/src/Phong/Source/addInput/y/y_merged_data_part_{i}.parquet"
        X = pl.read_parquet(XFileName).to_pandas()
        X = X.loc[:, bandList]
        y = pl.read_parquet(yFileName).to_pandas().squeeze()
        result = eval_mutual_information(X, y, fullBand, timeStamps)

        logger.info(
            f"[MutualInfo] Top bands: "
            f"{[(x['feature'], round(x['MI_score'], 4)) for x in result]}"
        )
