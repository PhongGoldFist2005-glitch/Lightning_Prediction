import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
# ── 1. Thiết lập kết nối ──
con = duckdb.connect()
con.execute("SET memory_limit='30GB'")
con.execute("SET threads=8")


settings ="old"
value_set = {
    "type": "val"  # Hoặc "val",
}

train_stats_path = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_{settings}_scales.json'
val_plot_output = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/{settings}Data/norm/{value_set["type"]}_comparison_report.png'
# TEST_DATA_DIR = "/sdd/Dubaoset/src/Phong/Model/data/testClean"
TRAIN_DATA_PATH = f'/sdd/Dubaoset/src/Phong/Model/data/trainDistributed/Cleaned/total_{settings}_clean.parquet'
VAL_DATA_PATH = '/sdd/Dubaoset/src/Phong/Model/data/validation/validationCleaned/clean_eval_data_6.parquet'

# listFile = [os.path.join(TEST_DATA_DIR, f) for f in os.listdir(TEST_DATA_DIR) if f.endswith('.parquet')]
if value_set["type"] == "val":
    listFile = [VAL_DATA_PATH]
else:
    listFile = [TRAIN_DATA_PATH]

with open(train_stats_path, 'r') as f:
    train_stats_data = json.load(f)

# Giả định listFile đã được định nghĩa
file_list_str = ", ".join([f"'{f}'" for f in listFile])
bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']

hist_results = {}

# ── 2. Tính toán Stats & Histogram ──
for band in bandName:
    print(f"Đang xử lý Band: {band}...")
    
    # Lấy thông số gốc từ tập Train
    raw_min = train_stats_data[band]['min']
    raw_max = train_stats_data[band]['max']
    raw_range = raw_max - raw_min

    # Xác định công thức chuẩn hóa và thang đo (X-axis)
    if band == 'NDVI':
        # Giữ nguyên bản
        display_min, display_max = -1.0, 1.0
        sql_val_expr = 'val'
    elif band == 'Dem_value':
        # Chuẩn hóa Max (về dải -1 đến 1)
        display_min, display_max = -1.0, 1.0
        # Chia cho trị tuyệt đối của Max để đưa về dải [-1, 1]
        sql_val_expr = f"val / {max(abs(raw_max), abs(raw_min))}" 
    else:
        # 10 Bands đầu: Min-Max chuẩn hóa về 0-1
        display_min, display_max = 0.0, 1.0
        sql_val_expr = f"(val - {raw_min}) / NULLIF({raw_range}, 0)"

    display_range = display_max - display_min

    # 2.1 Tạo Union query cho Sliding Window
    all_window_queries = []
    for i in range(6):
        window_start, window_end = -6 + i, i
        for t_val in range(window_start, window_end):
            col_name = band if band in ("Dem_value", "DEMIsLand") else f"{band}_t{('-' + str(-t_val)) if t_val < 0 else ('+' + str(t_val))}"
            all_window_queries.append(f'SELECT "{col_name}" AS val FROM read_parquet([{file_list_str}])')

    union_parts = "\n    UNION ALL\n    ".join(all_window_queries)

    # 2.2 Tính Mean và Total Rows trên giá trị ĐÃ CHUẨN HÓA
    stats_query = f"""
        SELECT COUNT(*), AVG({sql_val_expr}) 
        FROM ({union_parts}) 
        WHERE val IS NOT NULL 
          AND val BETWEEN {raw_min} AND {raw_max}
    """
    stats_res = con.execute(stats_query).fetchone()
    total_rows, scaled_mean = stats_res[0], stats_res[1]

    # 2.3 Tính Histogram trên giá trị ĐÃ CHUẨN HÓA
    n_bins = 100 if band == 'NDVI' else (150 if band == 'Dem_value' else 200)
    
    hist_sql = f"""
    SELECT 
        floor(({sql_val_expr} - {display_min}) / (NULLIF({display_range}, 0)) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) as cnt
    FROM ({union_parts})
    WHERE val BETWEEN {raw_min} AND {raw_max}
    GROUP BY 1 ORDER BY 1
    """
    h_df = con.execute(hist_sql).df()
    h_df['density_pct'] = (h_df['cnt'] * 100.0 / total_rows) if total_rows > 0 else 0

    hist_results[band] = {
        'df': h_df, 
        'min': display_min, 
        'max': display_max, 
        'mean': scaled_mean, 
        'n_bins': n_bins
    }

# ── 3. Vẽ đồ thị ──
fig, axes = plt.subplots(len(bandName), 1, figsize=(14, 5 * len(bandName)))
if len(bandName) == 1: axes = [axes]

for ax, band in zip(axes, bandName):
    res = hist_results[band]
    b_min, b_max = res['min'], res['max']
    bin_width = (b_max - b_min) / res['n_bins']
    
    centers = b_min + (res['df']['bin_idx'] + 0.5) * bin_width
    ax.bar(centers, res['df']['density_pct'], width=bin_width * 0.8, color="steelblue", alpha=0.7, label="Phân phối chuẩn hóa")
    
    if res['mean'] is not None:
        ax.axvline(res['mean'], color="crimson", linestyle="--", linewidth=2.5, 
                   label=f"Mean: {res['mean']:.4f}")

    ax.set_xlim(b_min, b_max)
    ax.set_title(f"Band: {band} | Thang đo: {b_min} đến {b_max}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mật độ (%)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(val_plot_output, dpi=150)
plt.show()
print(f"Xong! Ảnh đã lưu tại: {val_plot_output}")