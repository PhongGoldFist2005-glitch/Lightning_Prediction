from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_classif
from libpysal.weights import KNN
from esda.moran import Moran

def takeTimeBand(band, timeStamps):
    listOfBand = []
    for i in timeStamps:
        bandName = f"{band}_t{i:+d}"
        listOfBand.append(bandName)
    return listOfBand

def eval_stepwise_forward(
    model, X, y, bandList,
    scoring="average_precision", cv=5,
    random_state=42, timeStamps=None
):
    if timeStamps is None:
        timeStamps = list(range(-6, 0))

    maximumBand = len(bandList)
    y_arr = np.array(y)

    best_combo = []
    bestTime_combo = []

    # lưu toàn bộ quá trình
    history = []

    for i in range(0, maximumBand):
        best_score = float("-inf")
        current_combo = best_combo[:]
        time_combo = bestTime_combo[:]

        for j in range(0, maximumBand):
            addBand = bandList[j]
            if addBand in current_combo:
                continue

            addTimeList = takeTimeBand(addBand, timeStamps)

            candidate_combo = current_combo + [addBand]
            candidate_time_combo = time_combo + addTimeList

            X_candidate = X.loc[:, candidate_time_combo].values

            scores = cross_val_score(
                model, X_candidate, y_arr,
                scoring=scoring,
                cv=cv,
                n_jobs=-1
            )

            mean_score = scores.mean()

            if best_score < mean_score:
                best_score = mean_score
                best_combo = candidate_combo[:]
                bestTime_combo = candidate_time_combo[:]

        # log mỗi step
        history.append({
            "step": i + 1,
            "bands": best_combo[:],
            "n_bands": len(best_combo),
            "n_features": len(bestTime_combo),
            "score": best_score
        })

    # tìm step tốt nhất
    best_step_info = max(history, key=lambda x: x["score"])

    return best_combo, history, best_step_info

def eval_stepwise_backward(
    model, X, y, bandList,
    scoring="average_precision", cv=5,
    random_state=42, timeStamps=None
):

    if timeStamps is None:
        timeStamps = list(range(-6, 0))

    y_arr = np.array(y)

    # bắt đầu từ full band
    current_combo = bandList[:]
    current_time_combo = []
    for band in current_combo:
        current_time_combo.extend(takeTimeBand(band, timeStamps))

    # history log
    history = []

    # số bước = số band (loại dần về 1)
    for step in range(len(bandList), 0, -1):
        best_score = float("-inf")
        best_combo = None
        best_time_combo = None

        # thử loại từng band
        for band in current_combo:
            candidate_combo = [b for b in current_combo if b != band]

            candidate_time_combo = []
            for b in candidate_combo:
                candidate_time_combo.extend(takeTimeBand(b, timeStamps))

            if len(candidate_time_combo) == 0:
                continue

            X_candidate = X.loc[:, candidate_time_combo].values

            scores = cross_val_score(
                model, X_candidate, y_arr,
                scoring=scoring,
                cv=cv,
                n_jobs=-1
            )

            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_combo = candidate_combo
                best_time_combo = candidate_time_combo
        
        # cập nhật combo sau khi loại tốt nhất
        if best_combo is None:  # ← không còn gì để loại
            break    
        current_combo = best_combo
        current_time_combo = best_time_combo
        # cập nhật combo sau khi loại tốt nhất
        current_combo = best_combo
        current_time_combo = best_time_combo

        # log
        history.append({
            "step": len(bandList) - step + 1,
            "bands": current_combo[:],
            "n_bands": len(current_combo),
            "n_features": len(current_time_combo),
            "score": best_score
        })

    # tìm step tốt nhất
    best_step_info = max(history, key=lambda x: x["score"])

    return current_combo, history, best_step_info


def eval_rfe(
    X, y,
    feature_names,
    C,
    timeStamps,
    n_features=60,
    max_iter=2000
):
    # Yêu cầu X phải có các cột đầu vào đúng thứ tự
    # band_t-6 -> band_t-1
    n_total       = len(feature_names)
    # số feature cần loại bỏ
    n_to_remove   = n_total - n_features
    lenTime = len(timeStamps)

    lr = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs')
    bandSorted = []

    # Theo dõi thứ tự bị loại: ranking[i] sẽ được gán sau
    # Thứ tự ranking cho các band
    ranking = np.ones(n_total, dtype=int)

    # Một cái mask cho biết các cột nào đang được sử dụng
    active_mask = np.ones(n_total * lenTime, dtype=bool)

    # Index gốc để map về lại feature_names sau này
    original_idx = np.arange(n_total * lenTime)

    scaler = StandardScaler()

    # ranking bắt đầu từ n_total, đếm ngược về 1
    current_rank = n_total  

    print(f"Tổng features: {n_total} | Cần giữ: {n_features} | Cần loại: {n_to_remove}")

    for step in range(n_to_remove):

        # 1. Lấy các feature đang còn active
        active_cols = original_idx[active_mask]           # index gốc
        X_active    = X.iloc[:, active_cols].values       # subset data

        # 2. Scale
        X_scaled = scaler.fit_transform(X_active)

        # 3. Fit Logistic Regression
        lr.fit(X_scaled, y)

        # 4. Lấy coef_ → tính importance
        # coef_ shape: (n_classes-1, n_active_features) với binary
        # shape (C, F)
        coef = lr.coef_                    
        # trung bình qua classes → shape (F,)               
        importance = np.mean(np.abs(coef), axis=0)
        # Trung bình qua các timeStamps của từng features (6 timeStamp cho 1 band)
        arrrayBand = np.squeeze([[np.mean(np.abs(importance[i : i + lenTime])) for i in range(0, len(importance), lenTime)]])

        # 5. Tìm feature YẾU NHẤT trong nhóm active
        worst_local_idx  = np.argmin(arrrayBand)          # index trong active_cols
        start = worst_local_idx * lenTime
        end   = start + lenTime
        worst_global_idx = active_cols[start : end]   # index gốc trong feature_names

        # 6. Gán ranking và loại khỏi active
        worst_band_original_idx = worst_global_idx[0] // lenTime
        ranking[worst_band_original_idx] = current_rank
        bandSorted.append(feature_names[worst_band_original_idx])
        active_mask[worst_global_idx] = False
        current_rank -= 1

        print(
            f"Step {step+1}/{n_to_remove} | "
            f"Loại: '{feature_names[worst_band_original_idx]}' | "
            f"importance={arrrayBand[worst_local_idx]:.6f}"
        )

    return bandSorted

# Đo độ tương quan tuyến tính giữa từng biến đến đầu ra y
def eval_point_biserial(X, y, timeStamps, bandName, random_state=42):
    dictOfResult = {}
    y_arr = np.array(y)
    for band in bandName:
        avg_corr = []
        avg_p_value = []
        bandTime = takeTimeBand(band, timeStamps)
        for time in bandTime:
            X_candidate = X.loc[:, time].values
            corr, p_value = pointbiserialr(y_arr, X_candidate)
            avg_corr.append(np.abs(corr))
            avg_p_value.append(p_value)
        corrA = np.mean(avg_corr)
        p_valueA = np.mean(avg_p_value)
        dictOfResult.update({band:{"corr": corrA, "p_value": p_valueA}})
    
    dictOfResult = dict(
        sorted(dictOfResult.items(), key=lambda x: np.abs(x[1]["corr"]), reverse= True)
    )
    return dictOfResult

# Từng band Xi trong X, nó sẽ đánh giá xem kNN của các điểm cùng band đó của từng
# class xa bao nhiêu và có bao nhiêu điểm quanh đó từ đó quyết định chúng đóng góp trong việc phân biệt các classs tốt bao nhiêu
def eval_mutual_information(X, y, bandName, timeStamps):
    scores = []

    for band in bandName:
        timeBands = takeTimeBand(band, timeStamps)

        mi_list = []
        for t in timeBands:
            mi = mutual_info_classif(
                X[[t]], y,
                discrete_features=False,
                random_state=42
            )[0]
            mi_list.append(mi)

        scores.append({
            "feature": band,
            "MI_score": np.max(mi_list)
        })

    return sorted(scores, key=lambda x: x["MI_score"], reverse=True)

# Lấy từng timeStamps, tọa độ mới
def morans_I(X, bandList, timeStamps, coords, k):
    w = KNN.from_array(coords, k=k)
    result = {}
    for time in timeStamps:
        for band in bandList:
            bandName = f"{band}_t{time:+d}"
            X_candidate = X.loc[:, bandName].values
            moran = Moran(X_candidate, w)
            if band not in result:
                result[band] = []
            result[band].append(moran.I)
    finalResult = []
    for key, point in result.items():
        item = (key, np.max(point))
        finalResult.append(item)
    finalResult = sorted(finalResult, key= lambda x:x[1], reverse=True)

    return finalResult
    