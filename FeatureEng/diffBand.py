import pandas as pd
import numpy as np
import polars as pl

from sklearn.feature_selection import (
    mutual_info_classif,
    SequentialFeatureSelector,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import pointbiserialr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Custom lại cơ chế
# Mỗi lần train bỏ 1 band và 6 timestamp của nó cùng lúc thay vì chỉ bỏ 1 band
def eval_stepwise_forward(
    X, y,
    n_features=60,
    scoring="average_precision",
    cv=5,
    random_state=42
):
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        max_samples=0.8,
        n_jobs=1,
        class_weight="balanced",
        random_state=42
    )

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction="forward",
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    sfs.fit(X, y)

    return pd.DataFrame({
        "feature": X.columns,
        "selected": sfs.get_support()
    }).sort_values("selected", ascending=False)

def eval_stepwise_backward(
    X, y,
    n_features=60,
    scoring="average_precision",
    cv=5,
    random_state=42
):
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        max_samples=0.8,
        n_jobs=1,
        class_weight="balanced",
        random_state=42
    )

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction="backward",
        scoring=scoring,
        cv=cv
    )

    sfs.fit(X, y)

    return pd.DataFrame({
        "feature": X.columns,
        "selected": sfs.get_support()
    }).sort_values("selected", ascending=False)


def eval_rfe(
    X, y,
    n_features=60,
    max_iter=2000
):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1
        ))
    ])

    rfe = RFE(
        estimator=model,
        n_features_to_select=n_features,
        importance_getter=lambda est: est.named_steps["lr"].coef_
    )

    rfe.fit(X, y)

    return (
        pd.DataFrame({
            "feature": X.columns,
            "ranking": rfe.ranking_,
            "selected": rfe.support_
        })
        .sort_values("ranking")
        .reset_index(drop=True)
    )

def eval_mutual_information(X, y, random_state=42):
    mi = mutual_info_classif(X, y, random_state=random_state)
    return (
        pd.DataFrame({
            "feature": X.columns,
            "MI_score": mi
        })
        .sort_values("MI_score", ascending=False)
        .reset_index(drop=True)
    )

def eval_point_biserial(X, y):
    scores = []

    for col in X.columns:
        corr, pval = pointbiserialr(y, X[col])
        scores.append({
            "feature": col,
            "PBC_corr": abs(corr),
            "p_value": pval
        })

    return (
        pd.DataFrame(scores)
        .sort_values("PBC_corr", ascending=False)
        .reset_index(drop=True)
    )



# MI + PBC → lọc 20 feature
# → RFE / SF → chọn 10 feature
# → Train model
# → SHAP giải thích vật lý
if __name__ == "__main__":
    for i in range(0, 45, 5):
        XFileName = f"/sdd/Dubaoset/src/Phong/Source/addInput/X/X_merged_data_part_{i}.parquet"
        yFileName = f"/sdd/Dubaoset/src/Phong/Source/addInput/y/y_merged_data_part_{i}.parquet"
        X = pl.read_parquet(XFileName).to_pandas()
        y = pl.read_parquet(yFileName).to_pandas()
        result = eval_stepwise_forward(X, y)
        result.to_csv(f"/sdd/Dubaoset/src/Phong/Source/addInput/Result/result_ESF{i}.csv")