import duckdb
import numpy as np
import json
import os

# ── 1. Thiết lập kết nối ──
con = duckdb.connect()
con.execute("SET memory_limit='48GB'")
con.execute("SET threads=8")

# Chọn tập dữ liệu bạn muốn tạo Cache (Ví dụ: Tập TEST)
PATH = '/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/train_dataset.parquet'

listFileOk = [PATH]

file_list_str = ", ".join([f"'{f}'" for f in listFileOk])

bandName = ['B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','WVB', 'NDVI','Dem_value']

# File xuất Cache
cache_json_output = '/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/plot/raw/train_phong_data_master_cache.json'

N_MASTER_BINS = 50000
master_cache = {}

for band in bandName:
    print(f">>> Đang tạo Master Cache cho Band: {band}...")
    
    if band == 'NDVI':
        g_min, g_max = -1.0, 1.0
    elif band in ('Dem_value', 'DEMIsLand'):
        g_min, g_max = -3000.0, 3000.0 
    else:
        g_min, g_max = 0.0, 1000.0

    # --- Giữ nguyên logic trượt cửa sổ của bạn ---
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
    
    # --- Tính Master Bins bằng SQL ---
    cache_sql = f"""
    SELECT 
        label,
        floor((val - {g_min}) / ({g_max} - {g_min}) * {N_MASTER_BINS}) as master_bin_idx,
        COUNT(*) as cnt
    FROM ({union_parts})
    WHERE val IS NOT NULL AND val >= {g_min} AND val < {g_max}
    GROUP BY 1, 2
    """
    
    try:
        df_cache = con.execute(cache_sql).df()
    except Exception as e:
        print(f"Lỗi khi đọc band {band}: {e}")
        continue

    # --- Đổ dữ liệu vào mảng 10.000 phần tử ---
    counts_l0 = np.zeros(N_MASTER_BINS, dtype=int)
    counts_l1 = np.zeros(N_MASTER_BINS, dtype=int)
    total_l0 = 0
    total_l1 = 0
    
    for _, row in df_cache.iterrows():
        idx = int(row['master_bin_idx'])
        cnt = int(row['cnt'])
        lbl = int(row['label'])
        
        if 0 <= idx < N_MASTER_BINS:
            if lbl == 0:
                counts_l0[idx] = cnt
                total_l0 += cnt
            else:
                counts_l1[idx] = cnt
                total_l1 += cnt
                
    # --- Lưu vào Dictionary ---
    master_cache[band] = {
        "g_min": float(g_min),
        "g_max": float(g_max),
        "n_master_bins": N_MASTER_BINS,
        "total_rows_l0": int(total_l0),
        "total_rows_l1": int(total_l1),
        "counts_l0": counts_l0.tolist(),
        "counts_l1": counts_l1.tolist()
    }

# ── 3. Lưu Cache ra file ──
with open(cache_json_output, 'w') as f:
    json.dump(master_cache, f)

print(f"Hoàn thành! Master Cache đã được lưu tại: {cache_json_output}")