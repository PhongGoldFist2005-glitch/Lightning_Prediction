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
con.execute("SET memory_limit='50GB'")
con.execute("SET threads=8")

outputLabel = ["output_0","output_1","output_2","output_3","output_4","output_5"]
outputRegressLabel = ["lightning_0", "lightning_1", "lightning_2", "lightning_3", "lightning_4", "lightning_5"]
startPeriod      = -6
endPeriod        =  6
startPeriodTrain = -6
endPeriodTrain   =  0

bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']
# bandKep = ['IRB-I2B', 'B09B-B14B', 'WVB-B14B', 'B11B-B14B', 'I2B-B14B', 'WVB-IRB', 'B11B-IRB', 'B14B-I2B', 'B11B-B12B', 'WVB-B10B']

# bandName = bandName + bandKep

bandType = [
    f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}" if (band != "Dem_value" and band != "DEMIsLand") else band
    for i in range(startPeriod, endPeriod)
    for band in bandName
]

bandTypeTrain = [
    f"{band}_t{('-' + str(-i)) if i < 0 else ('+' + str(i))}" if (band != "Dem_value" and band != "DEMIsLand") else band
    for i in range(startPeriodTrain, endPeriodTrain)
    for band in bandName
]

TRAIN_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/trainClean/"
TEST_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/testClean"
VAL_PATH = "/sdd/Dubaoset/src/Phong/Model/data/unknown/clean_eval_data_6.parquet"

# listFile = ['/sdd/Dubaoset/src/Phong/Model/data/trainClean/merged_data_part_0.parquet']
listFile = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.endswith('.parquet')]

# band_list = ", ".join([f'MIN("{b}") AS "min_{b}", AVG("{b}") AS "mean_{b}", MEDIAN("{b}") AS "median_{b}", MAX("{b}") AS "max_{b}", PERCENTILE_CONT(0.02) WITHIN GROUP (ORDER BY "{b}") AS "p2_{b}", PERCENTILE_CONT(0.98) WITHIN GROUP (ORDER BY "{b}") AS "p98_{b}"' for b in bandType])


# Val_non_normalize
# Test_non_normalize cùng thang đo với Train khi chưa chuẩn hóa nha
try:
    with open('train_scales.json', 'r') as f:
        train_standard_scales = json.load(f)
    print("--- Đã load thành công thước đo chuẩn từ Train ---")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file train_scales.json. Hãy chạy bước Train trước!")
    exit()

file_list_str = ", ".join([f"'{f}'" for f in listFile])

band_groups = defaultdict(list)
for col in bandType:
    if col in ("Dem_value", "DEMIsLand"):
        band_groups[col].append(col)
        continue
    m = re.match(r"^(.+)_t[+\-]\d+$", col)
    if m:
        band_groups[m.group(1)].append(col)

test_hist_results = {}

# ── 2. Tính toán Histogram cho Test dựa trên thước đo Train ──
for band_name, cols in band_groups.items():
    if band_name not in train_standard_scales:
        continue
        
    print(f"Đang xử lý tập Test cho Band: {band_name}...")
    
    p2 = train_standard_scales[band_name]['p2']
    p98 = train_standard_scales[band_name]['p98']
    mean_tr = train_standard_scales[band_name]['mean']
    val_range = p98 - p2
    
    union_parts = "\n    UNION ALL\n    ".join([
        f'SELECT "{c}" AS val FROM read_parquet([{file_list_str}]) WHERE "{c}" IS NOT NULL'
        for c in cols
    ])
    
    if band_name == 'NDVI': n_bins = 100
    elif band_name == 'Dem_value': n_bins = 150
    else: n_bins = 80 

    # SỬA LẠI SQL:
    # 1. Dùng GREATEST và LEAST để "clip" dữ liệu vào khoảng P2-P98 (Thấy được Drift)
    # 2. Không scale về 0-1, giữ nguyên giá trị thật để tính bin
    # 3. Mẫu số chia cho TỔNG SỐ DÒNG (raw_data)
    hist_query = f"""
    WITH raw_data AS (
        SELECT val FROM ({union_parts})
    ),
    clipped_data AS (
        SELECT 
            least(greatest(val, {p2}), {p98}) AS val_clipped 
        FROM raw_data
    )
    SELECT
        floor((val_clipped - {p2}) / NULLIF({val_range}, 0) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM raw_data) as density_pct
    FROM clipped_data
    GROUP BY bin_idx 
    ORDER BY bin_idx
    """
    
    h_df = con.execute(hist_query).df()
    test_actual_stats = con.execute(f"SELECT AVG(val) as m FROM ({union_parts})").df().iloc[0]
    
    test_hist_results[band_name] = {
        'df': h_df, 
        'test_mean': test_actual_stats['m'],
        'n_bins': n_bins,
        'p2': p2,
        'p98': p98,
        'mean_tr': mean_tr
    }

# ── 3. Vẽ đồ thị so sánh (SỬA LẠI TRỤC X) ──
fig, axes = plt.subplots(len(test_hist_results), 1, figsize=(12, 5 * len(test_hist_results)))
if len(test_hist_results) == 1: axes = [axes]

for ax, band_name in zip(axes, test_hist_results.keys()):
    res = test_hist_results[band_name]
    h_df = res['df']
    
    p2_tr = res['p2']
    p98_tr = res['p98']
    
    # Tính lại bin_centers theo GIÁ TRỊ THẬT (không phải 0-1)
    bin_width = (p98_tr - p2_tr) / res['n_bins']
    bin_centers = p2_tr + (h_df['bin_idx'] + 0.5) * bin_width
    
    # Vẽ cột
    ax.bar(bin_centers, h_df['density_pct'], width=bin_width * 0.9, 
           color="orange", alpha=0.7, label="Val/Test Distribution")
    
    # Vẽ đường Mean gốc (Giá trị thật, không scale)
    ax.axvline(res['mean_tr'], color="blue", lw=2, label=f"Train Mean: {res['mean_tr']:.2f}")
    ax.axvline(res['test_mean'], color="red", lw=2, linestyle="--", label=f"Test Mean: {res['test_mean']:.2f}")
    
    # Giới hạn trục X đúng bằng P2-P98 của Train
    ax.set_xlim(p2_tr, p98_tr)
    ax.set_title(f"Comparison Band: {band_name} (Val vs Train Scale)", fontweight='bold')
    ax.set_xlabel("Giá trị (Original Scale)")
    ax.set_ylabel("Mật độ (%)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/test_non_normalize.png", dpi=150)
print("Đã vẽ xong ảnh test_non_normalize.png")