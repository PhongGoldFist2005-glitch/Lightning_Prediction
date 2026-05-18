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

bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB']
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
TRAIN_STATS_PATH = "/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/train_normalization_stats.json"
with open(TRAIN_STATS_PATH, "r") as f:
    train_stats = json.load(f)

val_tracking_stats = {} # Biến mới để lưu log thông số của tập Val/Test
hist_results = {}

# ── 3. TÍNH TOÁN, LƯU FILE VÀ CHUẨN HÓA MẬT ĐỘ ──
for band_name, cols in band_groups.items():
    print(f"Đang xử lý {band_name} trên tập Validation...")
    
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
        APPROX_QUANTILE(val, 0.02) AS p2,
        APPROX_QUANTILE(val, 0.98) AS p98
    FROM ({union_parts})
    """
    v_df = con.execute(stats_query).df()
    v_stats = v_df.iloc[0]
    
    # Rút thước đo của Train ra để chuẩn bị tính toán
    p2_train = train_stats[band_name]['original']['p2']
    p98_train = train_stats[band_name]['original']['p98']
    val_range_train = p98_train - p2_train

    # Hàm tính giá trị Scaled (chưa bị clip) để xem Val lệch bao nhiêu so với Train
    def scale_val_value(x):
        if val_range_train == 0: return 0.0
        return (float(x) - p2_train) / val_range_train

    # LƯU DICTIONARY ĐẦY ĐỦ (Gốc và Đã Scale)
    val_tracking_stats[band_name] = {
        'total_rows': int(v_stats['total_rows']),
        'used_train_scale': {
            'p2_train': p2_train,
            'p98_train': p98_train
        },
        'val_original_stats': {
            'min': float(v_stats['min_val']),
            'max': float(v_stats['max_val']),
            'p2': float(v_stats['p2']),
            'p98': float(v_stats['p98']),
            'mean': float(v_stats['mean_val']),
            'median': float(v_stats['median_val'])
        },
        'val_scaled_stats_before_clip': {
            'min': scale_val_value(v_stats['min_val']),
            'max': scale_val_value(v_stats['max_val']),
            'p2': scale_val_value(v_stats['p2']),
            'p98': scale_val_value(v_stats['p98']),
            'mean': scale_val_value(v_stats['mean_val']),
            'median': scale_val_value(v_stats['median_val'])
        }
    }

    # --- BƯỚC B: CHUẨN HÓA BẰNG THÔNG SỐ TRAIN VÀ TÍNH HISTOGRAM ---
    if band_name == 'NDVI': n_bins = 100
    elif band_name == 'Dem_value': n_bins = 150
    else: n_bins = 80 
        
    hist_query = f"""
    WITH scaled_data AS (
        SELECT 
            -- Ép tràn: Dùng P2 và P98 của Train
            GREATEST(0.0, LEAST(1.0, (val - {p2_train}) / NULLIF({val_range_train}, 0))) AS val_scaled
        FROM ({union_parts})
    )
    SELECT
        floor(val_scaled * ({n_bins} - 1)) as bin_idx,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM scaled_data) as density_pct
    FROM scaled_data
    GROUP BY bin_idx 
    ORDER BY bin_idx
    """
    h_df = con.execute(hist_query).df()
    
    hist_results[band_name] = {
        'df': h_df, 
        'val_mean': float(v_stats['mean_val']), 
        'p2_train': p2_train, 
        'p98_train': p98_train, 
        'val_range_train': val_range_train,
        'n_bins': n_bins
    }

# ── 4. GHI LOG RA FILE JSON ──
Test_STATS_OUTPUT = "/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/test_tracking_stats.json"
with open(Test_STATS_OUTPUT, "w") as f:
    json.dump(val_tracking_stats, f, indent=4)
print(f"✅ Đã ghi thông số thống kê của tập Test ra file: {Test_STATS_OUTPUT}")

# ── 5. VẼ ĐỒ THỊ VÀ LƯU ẢNH ──
fig, axes = plt.subplots(len(band_groups), 1, figsize=(12, 5 * len(band_groups)))
if len(band_groups) == 1: axes = [axes]

for ax, band_name in zip(axes, band_groups.keys()):
    if band_name not in hist_results:
        continue
        
    data = hist_results[band_name]
    h_df = data['df']
    n_bins = data['n_bins']
    p2_t, p98_t, range_t = data['p2_train'], data['p98_train'], data['val_range_train']
    
    bin_width_scaled = 1.0 / n_bins
    bin_centers_scaled = (h_df['bin_idx'] + 0.5) * bin_width_scaled

    ax.bar(bin_centers_scaled, h_df['density_pct'], width=bin_width_scaled * 0.9, 
           color="darkorange", alpha=0.7, label="Validation Data")
    
    # Vẽ đường Mean của chính tập Val (nhưng đã bị scale theo thước đo Train)
    scaled_val_mean = (data['val_mean'] - p2_t) / range_t if range_t != 0 else 0
    ax.axvline(scaled_val_mean, color="red", lw=2, linestyle="--", 
               label=f"Val Mean (Scaled): {scaled_val_mean:.2f}")
    
    ax.set_xlim(-0.05, 1.05) 
    ax.set_ylabel("Mật độ (%)")
    ax.set_xlabel("Giá trị chuẩn hóa")
    ax.set_title(f"Band: {band_name} | Val Data Scaled by Train (Train P2:{p2_t:.1f}, Train P98:{p98_t:.1f})", 
                 fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')

plt.tight_layout()
PLOT_OUTPUT = "/sdd/Dubaoset/src/Thang/dataStatistic/Teacher/test_normalized_by_train.png"
plt.savefig(PLOT_OUTPUT, dpi=150)
print(f"✅ Đã lưu ảnh plot đồ thị ra file: {PLOT_OUTPUT}")
plt.close()