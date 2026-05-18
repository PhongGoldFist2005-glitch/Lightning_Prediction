import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import json

# Code DEM max
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
from collections import defaultdict

startPeriod      = -6
endPeriod        =  6
startPeriodTrain = -6
endPeriodTrain   =  0
# DEM cho max
# NDVI không cần
bandName = ['Dem_value']

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

# ── 1. Thiết lập kết nối ──
con = duckdb.connect()
con.execute("SET memory_limit='30GB'")
con.execute("SET threads=8")

TRAIN_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/trainClean/"
TEST_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/testClean"
VAL_PATH = "/sdd/Dubaoset/src/Phong/Model/data/unknown/clean_eval_data_6.parquet"

# listFile = ['/sdd/Dubaoset/src/Phong/Model/data/trainClean/merged_data_part_0.parquet']
listFile = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.endswith('.parquet')]
print(len(listFile))
# Giả định listFile chứa danh sách các file Parquet của tập VAL (hoặc TEST)
file_list_str = ", ".join([f"'{f}'" for f in listFile])

# Nhóm các cột theo band gốc
band_groups = defaultdict(list)
for col in bandType: 
    if col in ("Dem_value", "DEMIsLand"):
        band_groups[col].append(col)
        continue
    m = re.match(r"^(.+)_t[+\-]\d+$", col)
    if m:
        band_groups[m.group(1)].append(col)

# ── 2. LOAD THƯỚC ĐO CỦA TẬP TRAIN ──
TRAIN_STATS_PATH = "/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/train_max_normalization_stats_dem.json"
with open(TRAIN_STATS_PATH, "r") as f:
    train_stats = json.load(f)

val_max_tracking_stats = {} 
hist_results = {}

# ── 3. TÍNH TOÁN, LƯU FILE VÀ CHUẨN HÓA MẬT ĐỘ ──
for band_name, cols in band_groups.items():
    print(f"Đang xử lý {band_name} trên tập Validation (Max Normalization)...")
    
    if band_name not in train_stats:
        print(f"⚠️ Bỏ qua {band_name} vì không có thông số Train!")
        continue

    union_parts = "\n    UNION ALL\n    ".join([
        f'SELECT "{c}" AS val FROM read_parquet([{file_list_str}]) WHERE "{c}" IS NOT NULL'
        for c in cols
    ])
    
    # --- BƯỚC A: Tính thông số thực tế của tập Val ---
    stats_query = f"""
    SELECT
        COUNT(*) AS total_rows,
        MIN(val) AS min_val,
        AVG(val) AS mean_val,
        APPROX_QUANTILE(val, 0.5) AS median_val,
        MAX(val) AS max_val,
        MAX(ABS(val)) AS max_abs_val
    FROM ({union_parts})
    """
    v_df = con.execute(stats_query).df()
    v_stats = v_df.iloc[0]
    
    # Rút thước đo MAX ABS của Train ra để chuẩn bị tính toán
    max_abs_train = train_stats[band_name]['original']['max_abs']

    # Hàm tính giá trị Scaled (chưa bị clip) để xem Val lệch bao nhiêu so với Train
    def scale_val_by_train_max(x):
        if max_abs_train == 0: return 0.0
        return float(x) / max_abs_train

    # LƯU DICTIONARY ĐẦY ĐỦ (Gốc và Đã Scale Trước Khi Clip)
    val_max_tracking_stats[band_name] = {
        'total_rows': int(v_stats['total_rows']),
        'used_train_scale': {
            'max_abs_train': max_abs_train
        },
        'val_original_stats': {
            'min': float(v_stats['min_val']),
            'max': float(v_stats['max_val']),
            'max_abs': float(v_stats['max_abs_val']),
            'mean': float(v_stats['mean_val']),
            'median': float(v_stats['median_val'])
        },
        'val_scaled_stats_before_clip': {
            'min': scale_val_by_train_max(v_stats['min_val']),
            'max': scale_val_by_train_max(v_stats['max_val']),
            'max_abs': scale_val_by_train_max(v_stats['max_abs_val']),
            'mean': scale_val_by_train_max(v_stats['mean_val']),
            'median': scale_val_by_train_max(v_stats['median_val'])
        }
    }

    # --- BƯỚC B: CHUẨN HÓA, ÉP TRÀN VÀ TÍNH HISTOGRAM TRÊN DẢI [-1, 1] ---
    n_bins = 150 if band_name == 'Dem_value' else 100
        
    hist_query = f"""
    WITH scaled_data AS (
        SELECT 
            -- Ép tràn: Gò cứng dữ liệu trong khoảng -1.0 đến 1.0 an toàn cho Model
            GREATEST(-1.0, LEAST(1.0, val / NULLIF({max_abs_train}, 0))) AS val_scaled
        FROM ({union_parts})
    )
    SELECT
        -- Đưa dải [-1, 1] về [0, 1] bằng cách cộng 1 rồi chia 2, sau đó nhân với n_bins
        floor(((val_scaled + 1.0) / 2.0) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM scaled_data) as density_pct
    FROM scaled_data
    GROUP BY bin_idx 
    ORDER BY bin_idx
    """
    h_df = con.execute(hist_query).df()
    
    hist_results[band_name] = {
        'df': h_df, 
        'val_scaled_mean_before_clip': scale_val_by_train_max(v_stats['mean_val']), 
        'max_abs_train': max_abs_train,
        'n_bins': n_bins
    }

# ── 4. GHI LOG RA FILE JSON ──
TEST_STATS_OUTPUT = "/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/test_max_tracking_stats_dem.json"
with open(TEST_STATS_OUTPUT, "w") as f:
    json.dump(val_max_tracking_stats, f, indent=4)
print(f"✅ Đã ghi thông số thống kê của tập Val ra file: {TEST_STATS_OUTPUT}")

# ── 5. VẼ ĐỒ THỊ VÀ LƯU ẢNH ──
fig, axes = plt.subplots(len(band_groups), 1, figsize=(12, 5 * len(band_groups)))
if len(band_groups) == 1: axes = [axes]

for ax, band_name in zip(axes, band_groups.keys()):
    if band_name not in hist_results: continue
        
    data = hist_results[band_name]
    h_df = data['df']
    n_bins = data['n_bins']
    max_abs_t = data['max_abs_train']
    scaled_val_mean = data['val_scaled_mean_before_clip']
    
    # Tính toán tọa độ X cho các cột Bin (trên dải -1 đến 1)
    bin_width = 2.0 / n_bins
    h_df['bin_center'] = (h_df['bin_idx'] / (n_bins - 1)) * 2.0 - 1.0 + (bin_width / 2.0)

    # Đổi màu thành darkorange để phân biệt với tập Train
    ax.bar(h_df['bin_center'], h_df['density_pct'], width=bin_width * 0.9, 
           color="darkorange", alpha=0.7, label="Val Data (Max Scaled & Clipped)")
    
    # Vẽ đường Mean của tập Val (đã scale)
    ax.axvline(scaled_val_mean, color="red", lw=2, linestyle="--", 
               label=f"Val Mean (Scaled): {scaled_val_mean:.2f}")
    
    # THIẾT LẬP TRỤC X CHO DẢI TỪ -1 ĐẾN 1 (Thêm lề 0.05)
    ax.set_xlim(-1.05, 1.05) 
    ax.set_ylabel("Mật độ (%)")
    ax.set_xlabel("Giá trị chuẩn hóa (Chia cho Train Max_Abs)")
    ax.set_title(f"Band: {band_name} | Val Data Scaled by Train (Train Max_Abs: {max_abs_t:.1f})", 
                 fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')

plt.tight_layout()
PLOT_OUTPUT = "/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/test_max_normalized_dem.png"
plt.savefig(PLOT_OUTPUT, dpi=150)
print(f"✅ Đã lưu ảnh plot đồ thị ra file: {PLOT_OUTPUT}")
plt.close()