import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os

# ── 1. Thiết lập kết nối ──
con = duckdb.connect()
con.execute("SET memory_limit='32GB'")
con.execute("SET threads=8") # Tận dụng tối đa CPU

# Đường dẫn file
train_stats_path = '/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_456_scales.json'
val_json_output = '/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/456Data/unnorm/train_detailed_stats_0_1.json'
val_plot_output = '/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/456Data/unnorm/train_comparison_report_0_1.png'
# TRAIN_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/trainClean/"
# listFile = [os.path.join(TRAIN_DATA_DIR, file) for file in os.listdir(TRAIN_DATA_DIR) if file.endswith('.parquet')]
TRAIN_DATA_PATH = '/sdd/Dubaoset/src/Phong/Model/data/trainDistributed/Choosen/total_5_clean.parquet'
VAL_PATH = '/sdd/Dubaoset/src/Phong/Model/data/validation/validationCleaned/clean_eval_data_6.parquet'
# Tập tháng 5
listFile = [TRAIN_DATA_PATH]

# Load thông số Train (Lấy Min/Max làm mốc chuẩn)
with open(train_stats_path, 'r') as f:
    train_stats_data = json.load(f)

# Giả định listFile đã được lấy từ thư mục Training
file_list_str = ", ".join([f"'{f}'" for f in listFile])
bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']

detailed_stats_json = {}
hist_results = {}

# ── 2. Tính toán Stats & Histogram ──
for band in bandName:
    print(f">>> Đang phân tích Training cho Band: {band}...")
    
    # 2.1 Tạo logic trượt cửa sổ (Sliding Window) 
    # Kết hợp 6 output với các window tương ứng
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
            
            # Gộp dữ liệu: mỗi dòng kèm theo nhãn thời điểm tương ứng
            all_window_queries.append(
                f'SELECT "{col_name}" AS val, "{label_col}" AS label FROM read_parquet([{file_list_str}])'
            )

    union_parts = "\n    UNION ALL\n    ".join(all_window_queries)
    
    # 2.2 Tính Stats thực tế của tập Training (Sử dụng SAMPLE 10% nếu dữ liệu quá khổng lồ)
    # Ở đây tôi tính trực tiếp để đảm bảo độ chính xác cho JSON
    stats_label_query = f"""
    SELECT
        label,
        COUNT(*) AS total_rows,
        MIN(val) AS min_val,
        MAX(val) AS max_val,
        AVG(val) AS mean_val
    FROM ({union_parts})
    WHERE val IS NOT NULL
    GROUP BY label
    """
    try:
        s_label_df = con.execute(stats_label_query).df().set_index('label')
    except Exception as e:
        print(f"Lỗi khi đọc band {band}: {e}")
        continue
    
    # Lưu stats thực tế của Training vào dict
    val_band_stats = {}
    for lbl in [0, 1]:
        if lbl in s_label_df.index:
            row = s_label_df.loc[lbl]
            val_band_stats[f"label_{lbl}"] = {
                'total_rows': int(row['total_rows']),
                'min': float(row['min_val']),
                'max': float(row['max_val']),
                'mean': float(row['mean_val'])
            }
        else:
            val_band_stats[f"label_{lbl}"] = {'total_rows': 0, 'min': 0, 'max': 0, 'mean': 0}
    
    detailed_stats_json[band] = val_band_stats

    # 2.3 Lấy khung đo (Scale) từ tập TRAIN (Sử dụng Min/Max làm mốc)
    # Logic: Lấy Min nhỏ nhất và Max lớn nhất giữa 2 nhãn của Train để làm trục X
    t_min = train_stats_data[band]['min']
    t_max = train_stats_data[band]['max']

    val_range = t_max - t_min
    
    detailed_stats_json[band]['train_scale_used'] = {'min': t_min, 'max': t_max}

    # 2.4 Tính Histogram cho Validation dựa trên khung đo của Train
    n_bins = 100 if band == 'NDVI' else (150 if band == 'Dem_value' else 200)
        
    hist_sql = f"""
    SELECT 
        label,
        floor((val - {t_min}) / (NULLIF({val_range}, 0)) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) as cnt
    FROM ({union_parts})
    WHERE val BETWEEN {t_min} AND {t_max}
    GROUP BY 1, 2 ORDER BY 2
    """
    h_df = con.execute(hist_sql).df()

    # Tính mật độ % dựa trên tổng số dòng của từng nhãn trong Training
    res_band = {'n_bins': n_bins, 'scale_min': t_min, 'scale_max': t_max}
    for lbl in [0, 1]:
        df_lbl = h_df[h_df['label'] == lbl].copy()
        total_lbl = val_band_stats[f"label_{lbl}"]['total_rows']
        df_lbl['density_pct'] = df_lbl['cnt'] * 100.0 / total_lbl if total_lbl > 0 else 0
        res_band[f'df{lbl}'] = df_lbl
        
    hist_results[band] = res_band

# ── 3. Lưu và Vẽ ──
with open(val_json_output, 'w') as f:
    json.dump(detailed_stats_json, f, indent=4)

fig, axes = plt.subplots(len(bandName), 1, figsize=(14, 5 * len(bandName)))
if len(bandName) == 1: axes = [axes]

for ax, band in zip(axes, bandName):
    res = hist_results[band]
    s_min, s_max = res['scale_min'], res['scale_max']
    bin_width = (s_max - s_min) / res['n_bins']
    
    # Vẽ Nhãn 0
    if not res['df0'].empty:
        c0 = s_min + (res['df0']['bin_idx'] + 0.5) * bin_width
        ax.bar(c0, res['df0']['density_pct'], width=bin_width*0.8, color="royalblue", alpha=0.5, label="Val Label 0")
    
    # Vẽ Nhãn 1
    if not res['df1'].empty:
        c1 = s_min + (res['df1']['bin_idx'] + 0.5) * bin_width
        ax.bar(c1, res['df1']['density_pct'], width=bin_width*0.8, color="crimson", alpha=0.5, label="Val Label 1")
    
    # Kẻ Mean của Validation (đường nét đứt)
    ax.axvline(detailed_stats_json[band]['label_0']['mean'], color="blue", linestyle="--", alpha=0.8, label="Mean L0 (Val)")
    ax.axvline(detailed_stats_json[band]['label_1']['mean'], color="red", linestyle="--", alpha=0.8, label="Mean L1 (Val)")
    
    ax.set_xlim(s_min, s_max)
    ax.set_title(f"Band: {band} | Val Distribution (Fixed Train Scale: {s_min:.2f} to {s_max:.2f})", fontsize=13, fontweight='bold')
    ax.set_ylabel("Density (%)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(val_plot_output, dpi=150)
print(f"Hoàn thành! Biểu đồ lưu tại: {val_plot_output}")