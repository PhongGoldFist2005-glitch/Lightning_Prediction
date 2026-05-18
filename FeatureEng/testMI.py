import numpy as np
from scipy.special import digamma
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
    file_handler = logging.FileHandler("/sdd/Dubaoset/src/Phong/Source/addInput/logging.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# ================================
# 1. Compute epsilon_i (joint space đúng chuẩn)
# ================================
def compute_eps_joint(x, y, k):
    """
    Tính ε_i = khoảng cách đến k-th neighbor trong JOINT SPACE (X, y)

    Metric:
        d((x_i,y_i),(x_j,y_j)) =
            max(||x_i - x_j||, 0 nếu cùng class, inf nếu khác class)
    """
    n = len(x)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    eps = np.zeros(n)

    for i in range(n):
        # khoảng cách trong X space
        dists = np.max(np.abs(x - x[i]), axis=1)

        # áp điều kiện joint space (khác class → vô hạn)
        dists[y != y[i]] = np.inf

        # bỏ chính nó
        dists[i] = np.inf

        # lấy k-th nearest neighbor (không cần sort full)
        eps[i] = np.partition(dists, k - 1)[k - 1]

    return eps


# ================================
# 2. Count m_i (marginal space X)
# ================================
def count_neighbors(x, eps):
    """
    m_i = số điểm trong toàn dataset thỏa:
          ||x_j - x_i|| < ε_i  (không tính bản thân)
    """
    n = len(x)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    m = np.zeros(n, dtype=int)

    for i in range(n):
        dists = np.max(np.abs(x - x[i]), axis=1)
        m[i] = np.sum(dists < eps[i]) - 1  # bỏ chính nó

    return m


# ================================
# 3. Mutual Information (KSG estimator)
# ================================
def mutual_info_single_feature(x, y, k=3, random_state=42):
    """
    MI ≈ ψ(k) - <ψ(m_i)> + <ψ(n_{y_i})> - ψ(n)
    """
    rng = np.random.default_rng(random_state)

    # jitter tránh trùng khoảng cách
    x = x + rng.normal(0, 1e-10, size=x.shape)

    n = len(x)

    # đếm số lượng mỗi class
    classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(classes, counts))

    # --- Bước 1: ε_i trong joint space ---
    eps = compute_eps_joint(x, y, k)

    # --- Bước 2: m_i trong marginal space ---
    m = count_neighbors(x, eps)

    # --- Bước 3: các thành phần digamma ---
    psi_k = digamma(k)
    psi_n = digamma(n)

    psi_m = digamma(np.maximum(m, 1))  # tránh digamma(0)
    psi_cls = np.array([digamma(class_counts[yi]) for yi in y])

    # --- Bước 4: MI ---
    mi = psi_k - np.mean(psi_m) + np.mean(psi_cls) - psi_n

    return max(0.0, mi)


# ================================
# 4. Evaluate multiple features
# ================================
def eval_mutual_information(X, y, bandName, timeStamps, k=3, random_state=42):
    scores = []

    for band in bandName:
        timeBands = takeTimeBand(band, timeStamps)
        X_candidate = X.loc[:, timeBands]

        mi = mutual_info_single_feature(
            X_candidate.values,
            y.values,
            k=k,
            random_state=random_state
        )

        scores.append({
            "feature": band,
            "MI_score": mi
        })

    # sort giảm dần (feature tốt nhất ở đầu)
    sortedScores = sorted(scores, key=lambda x: x["MI_score"], reverse=True)

    return sortedScores

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
        result = eval_mutual_information(X= X, y=y, bandName= fullBand, timeStamps= timeStamps,k = 5)

        logger.info(
            f"[MutualInfo] Top bands: "
            f"{[(x['feature'], round(x['MI_score'], 4)) for x in result]}"
        )