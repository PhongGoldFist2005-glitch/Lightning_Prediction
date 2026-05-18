import os, pandas as pd
import torch
import numpy as np
import pandas as pd
import os, json
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import pyarrow.parquet as pq
import duckdb
import matplotlib.pyplot as plt
from collections import defaultdict
import re


# ── 1. Thiết lập kết nối ──
con = duckdb.connect()
con.execute("SET memory_limit='30GB'")
con.execute("SET threads=8")

outputLabel = ["output_0","output_1","output_2","output_3","output_4","output_5"]
outputRegressLabel = ["lightning_0", "lightning_1", "lightning_2", "lightning_3", "lightning_4", "lightning_5"]
startPeriod      = -6
endPeriod        =  6
startPeriodTrain = -6
endPeriodTrain   =  0


TRAIN_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/trainClean/"
TEST_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/testClean"
VAL_PATH = "/sdd/Dubaoset/src/Phong/Model/data/unknown/clean_eval_data_6.parquet"

# listFile = ['/sdd/Dubaoset/src/Phong/Model/data/trainClean/merged_data_part_0.parquet']
listFile = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.endswith('.parquet')]
print(len(listFile))

# Đường dẫn file
train_stats_path = '/sdd/Dubaoset/src/Thang/dataStatistic/Label/detailed_stats_by_label_2.json'
val_json_output = '/sdd/Dubaoset/src/Thang/dataStatistic/Label/test_detailed_stats.json'
val_plot_output = '/sdd/Dubaoset/src/Thang/dataStatistic/Label/test_comparison_report.png'

# Load thông số Train để lấy thang đo chuẩn
with open(train_stats_path, 'r') as f:
    train_stats_data = json.load(f)

# Giả định listFile đã được định nghĩa
file_list_str = ", ".join([f"'{f}'" for f in listFile])
bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']

detailed_stats_json = {}
hist_results = {}

# ── 2. Tính toán Stats & Histogram ──
for band in bandName:
    print(f"Đang phân tích Validation cho Band: {band}...")
    
    # 2.1 Tạo logic trượt cửa sổ (Sliding Window)
    all_window_queries = []
    for i in range(6):
        label_col = f"output_{i}"
        window_start = -6 + i
        window_end = window_start + 6 
        for t_val in range(window_start, window_end):
            if band in ("Dem_value", "DEMIsLand"):
                col_name = band
            else:
                t_str = f"t{('-' + str(-t_val)) if t_val < 0 else ('+' + str(t_val))}"
                col_name = f"{band}_{t_str}"
            
            all_window_queries.append(
                f'SELECT "{col_name}" AS val, "{label_col}" AS label FROM read_parquet([{file_list_str}])'
            )

    union_parts = "\n    UNION ALL\n    ".join(all_window_queries)
    
    # 2.2 Tính Stats thực tế của tập Validation (Để ghi vào JSON Val)
    stats_label_query = f"""
    SELECT
        label,
        COUNT(*) AS total_rows,
        MIN(val) AS min_val,
        MAX(val) AS max_val,
        AVG(val) AS mean_val,
        APPROX_QUANTILE(val, 0.5) AS median_val,
        APPROX_QUANTILE(val, 0.02) AS p2,
        APPROX_QUANTILE(val, 0.98) AS p98
    FROM ({union_parts})
    WHERE val IS NOT NULL
    GROUP BY label
    """
    s_label_df = con.execute(stats_label_query).df().set_index('label')
    
    # Lưu stats thực tế của Validation vào JSON
    val_band_stats = {}
    for lbl in [0, 1]:
        if lbl in s_label_df.index:
            row = s_label_df.loc[lbl]
            val_band_stats[f"label_{lbl}"] = {
                'total_rows': int(row['total_rows']),
                'min': float(row['min_val']),
                'max': float(row['max_val']),
                'p2': float(row['p2']),
                'p98': float(row['p98']),
                'mean': float(row['mean_val']),
                'median': float(row['median_val'])
            }
    detailed_stats_json[band] = val_band_stats

    # 2.3 Lấy khung đo (Scale) từ tập TRAIN đã load
    # Ta sử dụng P2 của label_1 và P98 của label_0 (hoặc ngược lại) để bao phủ như logic Global cũ
    train_p2_0 = train_stats_data[band]['label_0']['p2']
    train_p2_1 = train_stats_data[band]['label_1']['p2']
    train_p98_0 = train_stats_data[band]['label_0']['p98']
    train_p98_1 = train_stats_data[band]['label_1']['p98']
    
    # Thang đo chuẩn từ Train
    scale_p2 = min(train_p2_0, train_p2_1)
    scale_p98 = max(train_p98_0, train_p98_1)
    val_range = scale_p98 - scale_p2
    
    # Lưu scale vào JSON Val để sau này đối chiếu
    detailed_stats_json[band]['train_scale_used'] = {'p2': scale_p2, 'p98': scale_p98}

    # 2.4 Tính Histogram cho Validation dựa trên khung đo của Train
    n_bins = 100 if band == 'NDVI' else (150 if band == 'Dem_value' else 80)
        
    hist_sql = f"""
    SELECT 
        label,
        floor((val - {scale_p2}) / (NULLIF({val_range}, 0)) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) as cnt
    FROM ({union_parts})
    WHERE val BETWEEN {scale_p2} AND {scale_p98}
    GROUP BY 1, 2 ORDER BY 2
    """
    h_df = con.execute(hist_sql).df()

    # Tách và tính % mật độ của Validation
    res_band = {'n_bins': n_bins, 'scale_p2': scale_p2, 'scale_p98': scale_p98}
    for lbl in [0, 1]:
        df_lbl = h_df[h_df['label'] == lbl].copy()
        # Dùng tổng rows của chính Validation để tính mật độ
        total_lbl = val_band_stats[f"label_{lbl}"]['total_rows']
        df_lbl['density_pct'] = df_lbl['cnt'] * 100.0 / total_lbl if total_lbl > 0 else 0
        res_band[f'df{lbl}'] = df_lbl
        
    hist_results[band] = res_band

# ── 3. Lưu file JSON & Vẽ đồ thị ──

# Lưu JSON Test (Stats thật của Test)
with open(val_json_output, 'w') as f:
    json.dump(detailed_stats_json, f, indent=4)

# Vẽ đồ thị (Dùng khung X của Train)
fig, axes = plt.subplots(len(bandName), 1, figsize=(14, 5 * len(bandName)))
if len(bandName) == 1: axes = [axes]

for ax, band in zip(axes, bandName):
    res = hist_results[band]
    p2, p98 = res['scale_p2'], res['scale_p98']
    bin_width = (p98 - p2) / res['n_bins']
    
    # Vẽ Nhãn 0 (Val data)
    if not res['df0'].empty:
        c0 = p2 + (res['df0']['bin_idx'] + 0.5) * bin_width
        ax.bar(c0, res['df0']['density_pct'], width=bin_width*0.8, color="royalblue", alpha=0.5, label="Test Label 0")
    
    # Vẽ Nhãn 1 (Val data)
    if not res['df1'].empty:
        c1 = p2 + (res['df1']['bin_idx'] + 0.5) * bin_width
        ax.bar(c1, res['df1']['density_pct'], width=bin_width*0.8, color="crimson", alpha=0.5, label="Test Label 1")
    
    # Vẽ Mean thực của Validation
    ax.axvline(detailed_stats_json[band]['label_0']['mean'], color="blue", linestyle="--", label="Test Mean L0")
    ax.axvline(detailed_stats_json[band]['label_1']['mean'], color="red", linestyle="--", label="Test Mean L1")
    
    # Cố định trục X theo Train
    ax.set_xlim(p2, p98)
    ax.set_title(f"Band: {band} | Test Distribution (Scaled by Train)", fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig(val_plot_output)
print(f"Hoàn thành! JSON lưu tại: {val_json_output}")