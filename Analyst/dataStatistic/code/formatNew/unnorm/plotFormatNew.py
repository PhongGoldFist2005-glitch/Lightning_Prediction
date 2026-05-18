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
    "type": "val",
}

train_stats_path = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_{settings}_scales.json'
val_plot_output = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/{settings}Data/unnorm/{value_set["type"]}_comparison_report.png'
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

file_list_str = ", ".join([f"'{f}'" for f in listFile])
bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']

hist_results = {}

# ── 2. Tính toán Stats & Histogram ──
for band in bandName:
    print(f"Đang xử lý Band: {band}...")
    
    # 2.1 Logic trượt cửa sổ
    all_window_queries = []
    for i in range(6):
        window_start, window_end = -6 + i, i
        for t_val in range(window_start, window_end):
            col_name = band if band in ("Dem_value", "DEMIsLand") else f"{band}_t{('-' + str(-t_val)) if t_val < 0 else ('+' + str(t_val))}"
            all_window_queries.append(f'SELECT "{col_name}" AS val FROM read_parquet([{file_list_str}])')

    union_parts = "\n    UNION ALL\n    ".join(all_window_queries)

    # 2.2 Lấy Min/Max chuẩn và tính Mean thực tế
    min_val = train_stats_data[band]['min']
    max_val = train_stats_data[band]['max']
    val_range = max_val - min_val

    # Tính Mean và Total Rows trong 1 lần quét
    stats_res = con.execute(f"SELECT COUNT(*), AVG(val) FROM ({union_parts}) WHERE val IS NOT NULL").fetchone()
    total_rows, avg_val = stats_res[0], stats_res[1]

    # 2.3 Tính Histogram
    n_bins = 100 if band == 'NDVI' else (150 if band == 'Dem_value' else 200)
    hist_sql = f"""
    SELECT 
        floor((val - {min_val}) / (NULLIF({val_range}, 0)) * ({n_bins} - 1)) as bin_idx,
        COUNT(*) as cnt
    FROM ({union_parts})
    WHERE val BETWEEN {min_val} AND {max_val}
    GROUP BY 1 ORDER BY 1
    """
    h_df = con.execute(hist_sql).df()
    h_df['density_pct'] = (h_df['cnt'] * 100.0 / total_rows) if total_rows > 0 else 0

    hist_results[band] = {'df': h_df, 'min': min_val, 'max': max_val, 'mean': avg_val, 'n_bins': n_bins}

# ── 3. Vẽ đồ thị ──
fig, axes = plt.subplots(len(bandName), 1, figsize=(14, 5 * len(bandName)))
if len(bandName) == 1: axes = [axes]

for ax, band in zip(axes, bandName):
    res = hist_results[band]
    b_min, b_max = res['min'], res['max']
    bin_width = (b_max - b_min) / res['n_bins']
    
    # Vẽ Histogram
    centers = b_min + (res['df']['bin_idx'] + 0.5) * bin_width
    ax.bar(centers, res['df']['density_pct'], width=bin_width * 0.8, color="steelblue", alpha=0.7, label="Phân phối thực tế")
    
    # Kẻ đường Mean
    if res['mean'] is not None:
        ax.axvline(res['mean'], color="crimson", linestyle="--", linewidth=2.5, 
                   label=f"Mean: {res['mean']:.2f}")

    # Định dạng trục
    ax.set_xlim(b_min, b_max)
    ax.set_title(f"Band: {band} (Scale chuẩn: {b_min:.1f} - {b_max:.1f})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mật độ (%)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig(val_plot_output, dpi=150)
plt.show()

print(f"Xong! Ảnh đã lưu tại: {val_plot_output}")