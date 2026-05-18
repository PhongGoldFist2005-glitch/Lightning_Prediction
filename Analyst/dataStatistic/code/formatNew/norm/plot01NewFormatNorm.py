import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Đường dẫn
# ── 1. Thiết lập kết nối ──
con = duckdb.connect()
con.execute("SET memory_limit='32GB'")
con.execute("SET threads=16")

# Đường dẫn
train_stats_path = '/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_456_scales.json'
val_json_output = '/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/456Data/norm/train_detailed_stats_0_1_norm.json'
val_plot_output = '/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/456Data/norm/train_comparison_report_0_1_norm.png'
# TRAIN_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/trainClean/"
TRAIN_DATA_PATH = '/sdd/Dubaoset/src/Phong/Model/data/trainDistributed/Choosen/total_5_clean.parquet'
VAL_DATA_PATH = '/sdd/Dubaoset/src/Phong/Model/data/validation/validationCleaned/clean_eval_data_6.parquet'

with open(train_stats_path, 'r') as f:
    train_stats_data = json.load(f)

# listFile = [os.path.join(TRAIN_DATA_DIR, f) for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.parquet')]
listFile = [TRAIN_DATA_PATH]
file_list_str = ", ".join([f"'{f}'" for f in listFile])
bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']

detailed_stats_json = {}
hist_results = {}

# ── 2. Tính toán Stats & Histogram ──
for band in bandName:
    print(f">>> Đang phân tích Training (Chuẩn hóa) cho Band: {band}...")
    
    t_min = train_stats_data[band]['min']
    t_max = train_stats_data[band]['max']
    t_range = t_max - t_min

    # Xác định dải hiển thị
    if band == 'NDVI' or band == 'Dem_value':
        display_min, display_max = -1.0, 1.0
    else:
        display_min, display_max = 0.0, 1.0
    display_range = display_max - display_min

    # 2.1 Tạo Union query cho Sliding Window
    all_window_queries = []
    for i in range(6):
        label_col = f"output_{i}"
        window_start, window_end = -6 + i, i
        for t_val in range(window_start, window_end):
            if band in ("Dem_value", "DEMIsLand"):
                col_name = band
            else:
                t_str = f"t{('-' + str(-t_val)) if t_val < 0 else ('+' + str(t_val))}"
                col_name = f"{band}_{t_str}"
            
            abs_max_val = max(abs(t_max), abs(t_min))
            denom = abs_max_val if abs_max_val != 0 else 1

            # 2. Thiết lập công thức dựa trên loại Band
            if band == 'NDVI':
                display_min, display_max = -1.0, 1.0
                current_norm_expr = f'"{col_name}"'
            elif band == 'Dem_value':
                display_min, display_max = -1.0, 1.0
                # Sử dụng biến denom đã tính ở trên
                current_norm_expr = f'"{col_name}" / {denom}'
            else:
                display_min, display_max = 0.0, 1.0
                current_norm_expr = f'("{col_name}" - {t_min}) / NULLIF({t_range}, 0)'
            
            all_window_queries.append(
                f'SELECT {current_norm_expr} AS norm_val, "{label_col}" AS label, "{col_name}" AS raw_val FROM read_parquet([{file_list_str}])'
            )

    union_parts = "\n    UNION ALL\n    ".join(all_window_queries)
    
    # 2.2 Tính Stats (Dùng TRY-EXCEPT để tránh crash nếu band thiếu data)
    stats_query = f"""
    SELECT
        label,
        COUNT(*) AS total_rows,
        MIN(norm_val) AS min_norm,
        MAX(norm_val) AS max_norm,
        AVG(norm_val) AS mean_norm
    FROM ({union_parts})
    WHERE raw_val BETWEEN {t_min} AND {t_max}
    GROUP BY label
    """
    try:
        s_label_df = con.execute(stats_query).df().set_index('label')
    except Exception as e:
        print(f"Bỏ qua Band {band} do lỗi: {e}")
        continue
    
    # Lưu kết quả vào dict
    val_band_stats = {}
    for lbl in [0, 1]:
        if lbl in s_label_df.index:
            row = s_label_df.loc[lbl]
            val_band_stats[f"label_{lbl}"] = {
                'total_rows': int(row['total_rows']),
                'min_norm': float(row['min_norm']),
                'max_norm': float(row['max_norm']),
                'mean_norm': float(row['mean_norm'])
            }
        else:
            val_band_stats[f"label_{lbl}"] = {'total_rows': 0, 'min_norm': 0, 'max_norm': 0, 'mean_norm': 0}
    
    detailed_stats_json[band] = val_band_stats

    # 2.3 Tính Histogram
    n_bins = 100 if band == 'NDVI' else (150 if band == 'Dem_value' else 200)
    hist_sql = f"""
    SELECT 
        label,
        floor((norm_val - {display_min}) / (NULLIF({display_range}, 0)) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) as cnt
    FROM ({union_parts})
    WHERE raw_val BETWEEN {t_min} AND {t_max}
    GROUP BY 1, 2 ORDER BY 2
    """
    h_df = con.execute(hist_sql).df()

    res_band = {'n_bins': n_bins, 'd_min': display_min, 'd_max': display_max}
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
    if band not in hist_results: continue
    res = hist_results[band]
    d_min, d_max = res['d_min'], res['d_max']
    bin_width = (d_max - d_min) / res['n_bins']
    
    for lbl, color, name in [(0, "royalblue", "L0"), (1, "crimson", "L1")]:
        df_lbl = res[f'df{lbl}']
        if not df_lbl.empty:
            centers = d_min + (df_lbl['bin_idx'] + 0.5) * bin_width
            ax.bar(centers, df_lbl['density_pct'], width=bin_width*0.8, color=color, alpha=0.5, label=f"Label {lbl}")
            m_val = detailed_stats_json[band][f'label_{lbl}']['mean_norm']
            ax.axvline(m_val, color=color, linestyle="--", alpha=0.8)
    
    ax.set_xlim(d_min, d_max)
    ax.set_title(f"Band: {band} | Normalized Distribution", fontsize=13, fontweight='bold')
    ax.set_ylabel("Density (%)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(val_plot_output, dpi=150)
print(f"Xong! File ảnh tại: {val_plot_output}")