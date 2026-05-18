"""
mergeEra5.py — Parallel ERA5 merge với 20 process
==================================================
Flow:
  1. Load parquet → lọc theo year
  2. Build ERA5 index 1 lần duy nhất (tránh 20 process cùng build)
  3. Chia data thành N_WORKERS chunk (mặc định 20)
  4. Mỗi process gọi merge_ERA5_df_ultrafast_return → trả DataFrame
  5. Sau khi tất cả xong → pd.concat → xuất 1 file parquet duy nhất
"""

from mergeData import (
    merge_ERA5_df_ultrafast_return,
    build_era5_index,
)

import os
import math
import polars as pl
import pandas as pd
from multiprocessing import Pool


# ── Config ─────────────────────────────────────────────────────────────────────
YEAR       = 2020
INPUT_PARQUET   = "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/train_dataset_d79eaf16.parquet"
ERA5_FOLDER     = "/sdd/Dubaoset/DATA/ERA5"
OUTPUT_FILEPATH = f"/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/train/train_{YEAR}.parquet"
LOG_DIR         = "/sdd/Dubaoset/src/Phong/Model/data/trainNorthSummer/log"


TIMESTAMPS = 6
N_WORKERS  = 4          # số process song song
MAX_THREADS_PER_WORKER = 2   # thread pool BÊN TRONG mỗi process (band processing)
                              # 20 process × 2 thread = 40 thread tổng
                              # Điều chỉnh tuỳ CPU và RAM

REMOVE_COLS = [
    'lightning_value', 'version', 'year', 'month', 'day', 'hour', 'minute',
    'second', 'second_n', 'lat', 'lon', 'lightning_number', 'sensor', 'DOF',
    'ellip angle', 'ellip long', 'ellip short', 'chi square', 'rise time',
    'peak time', 'speed max', 'lightning type', 'sensor angle', 'idx signal',
    'idx time',
    'B04B_t-1', 'B05B_t-1', 'B06B_t-1', 'VSB_t-1',
    'B04B_t-2', 'B05B_t-2', 'B06B_t-2', 'VSB_t-2',
    'B04B_t-3', 'B05B_t-3', 'B06B_t-3', 'VSB_t-3',
    'B04B_t-4', 'B05B_t-4', 'B06B_t-4', 'VSB_t-4',
    'B04B_t-5', 'B05B_t-5', 'B06B_t-5', 'VSB_t-5',
    'B04B_t-6', 'B05B_t-6', 'B06B_t-6', 'VSB_t-6',
    'label lightning', 'Location', 'min distance',
    'B04B_t+0', 'B05B_t+0', 'B06B_t+0', 'VSB_t+0',
    'B04B_t+1', 'B05B_t+1', 'B06B_t+1', 'VSB_t+1',
    'B04B_t+2', 'B05B_t+2', 'B06B_t+2', 'VSB_t+2',
    'B04B_t+3', 'B05B_t+3', 'B06B_t+3', 'VSB_t+3',
    'B04B_t+4', 'B05B_t+4', 'B06B_t+4', 'VSB_t+4',
    'B04B_t+5', 'B05B_t+5', 'B06B_t+5', 'VSB_t+5',
    'lightning_0', 'lightning_1', 'lightning_2',
    'lightning_3', 'lightning_4', 'lightning_5',
]


# ── Worker function ────────────────────────────────────────────────────────────

def worker_merge(args):
    """
    Chạy trong một process riêng biệt.

    args = (worker_id, chunk_df, remove_cols, era5_folder,
            start_time, end_time, max_threads, era5_index)

    Trả về: pd.DataFrame đã được merge ERA5
    """
    (worker_id, chunk_df, remove_cols, era5_folder,
     start_time, end_time, max_threads, era5_index) = args

    print(f"[Worker {worker_id:02d}] Start — {len(chunk_df):,} rows")

    try:
        result_df = merge_ERA5_df_ultrafast_return(
            data            = chunk_df,
            remove_cols     = remove_cols,
            eras5InfoFolder = era5_folder,
            startTime       = start_time,
            endTime         = end_time,
            max_workers     = max_threads,
            chunk_size_mb   = 300,          # mỗi process dùng ít RAM hơn
            use_processor_class = True,
            era5_index      = era5_index,   # dùng index build sẵn — không scan lại
        )
        print(f"[Worker {worker_id:02d}] Done  — shape: {result_df.shape}")
        return result_df

    except Exception as e:
        import traceback
        print(f"[Worker {worker_id:02d}] ERROR: {e}")
        traceback.print_exc()
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILEPATH), exist_ok=True)

    # 1. Load & lọc theo year ──────────────────────────────────────────────────
    print(f"Loading parquet: {INPUT_PARQUET}")
    data = pl.read_parquet(INPUT_PARQUET).to_pandas()
    data = data.loc[data["rounded_dt_up"].dt.year == YEAR].reset_index(drop=True)
    print(f"Filtered to year {YEAR}: {len(data):,} rows")

    # 2. Build ERA5 index 1 lần ────────────────────────────────────────────────
    # Truyền era5_index vào từng worker để tránh 20 process đồng thời scan disk
    print("\nBuilding ERA5 index (once for all workers)...")
    era5_index = build_era5_index(ERA5_FOLDER, max_workers=8, use_cache=True)
    print(f"ERA5 index ready: {len(era5_index):,} entries\n")

    # 3. Chia data thành N_WORKERS chunk ───────────────────────────────────────
    chunk_size = math.ceil(len(data) / N_WORKERS)
    chunks = [
        data.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
        for i in range(N_WORKERS)
        if i * chunk_size < len(data)   # guard: tránh chunk rỗng cuối
    ]
    actual_workers = len(chunks)
    print(f"Split into {actual_workers} chunks (~{chunk_size:,} rows each)")

    # 4. Đóng gói args cho mỗi worker ──────────────────────────────────────────
    args_list = [
        (
            worker_id,
            chunk,
            REMOVE_COLS,
            ERA5_FOLDER,
            -TIMESTAMPS,        # startTime
            TIMESTAMPS,         # endTime
            MAX_THREADS_PER_WORKER,
            era5_index,
        )
        for worker_id, chunk in enumerate(chunks)
    ]

    # 5. Chạy song song ────────────────────────────────────────────────────────
    print(f"\nSpawning {actual_workers} worker processes...\n")
    with Pool(processes=actual_workers) as pool:
        results = pool.map(worker_merge, args_list)

    # 6. Lọc bỏ worker bị lỗi ─────────────────────────────────────────────────
    valid_results = [df for df in results if df is not None]
    failed_count  = actual_workers - len(valid_results)

    if failed_count > 0:
        print(f"\n⚠️  {failed_count} worker(s) failed — kết quả có thể thiếu rows!")

    if not valid_results:
        raise RuntimeError("Tất cả worker đều thất bại — không có dữ liệu để xuất!")

    # 7. Ghép tất cả kết quả ───────────────────────────────────────────────────
    print(f"\nMerging {len(valid_results)} DataFrames...")
    final_df = pd.concat(valid_results, ignore_index=True)
    print(f"Final shape: {final_df.shape}")

    # 8. Xuất file ─────────────────────────────────────────────────────────────
    final_df.to_parquet(OUTPUT_FILEPATH, compression="snappy", index=False)
    print(f"\n✅ Saved → {OUTPUT_FILEPATH}")
    print(f"   Rows : {len(final_df):,}")
    print(f"   Cols : {len(final_df.columns):,}")