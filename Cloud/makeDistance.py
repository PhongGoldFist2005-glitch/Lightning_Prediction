import numpy as np
import rasterio
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import math

# ── helpers ────────────────────────────────────────────────────────────────────

def find_nearest_one(data, r, c, max_radius=200):
    rows, cols = data.shape
    if data[r, c] == 1:
        return (r, c, 0)

    best_dist2 = None
    best_r = best_c = None

    for radius in range(1, max_radius + 1):
        r_min, r_max = max(0, r - radius), min(rows - 1, r + radius)
        c_min, c_max = max(0, c - radius), min(cols - 1, c + radius)
        found = False

        for rr in range(r_min, r_max + 1):
            for cc in range(c_min, c_max + 1):
                if rr in (r_min, r_max) or cc in (c_min, c_max):
                    if data[rr, cc] == 1:
                        found = True
                        dist2 = (rr - r) ** 2 + (cc - c) ** 2
                        if best_dist2 is None or dist2 < best_dist2:
                            best_dist2, best_r, best_c = dist2, rr, cc
        if found:
            break

    if best_dist2 is None:
        return None
    return (best_r, best_c, 4 * np.sqrt(best_dist2))   # pixel = 4 km


def process_chunk(args):
    """
    Chạy trong một worker process.
    Trả về list of (original_index, dist_km) để ghép lại sau.
    """
    chunk_df, input_folder, list_of_tif, worker_id = args

    # Xây unique_time chỉ cho chunk này (dùng vị trí trong chunk_df)
    unique_time = {}
    for local_idx, (orig_idx, row) in enumerate(chunk_df.iterrows()):
        key = row["datetime_str"]
        if key not in unique_time:
            unique_time[key] = []
        unique_time[key].append(orig_idx)   # giữ index gốc để map lại

    results = {}   # orig_idx -> dist_km

    for key, orig_indices in tqdm(
        unique_time.items(),
        desc=f"Worker {worker_id}",
        position=worker_id,
        leave=False,
    ):
        tc = key.split("-")
        year, month, day, hour, minute = tc[0], tc[1], tc[2], tc[3], tc[4]
        file_path = f"{input_folder}/AHI_{year}{month}{day}_Z{hour}{minute}.tif"

        if file_path not in list_of_tif:
            continue

        with rasterio.open(file_path) as src:
            data = src.read(1)

        for orig_idx in orig_indices:
            row_data = chunk_df.loc[orig_idx]
            r, c = int(row_data["row"]), int(row_data["col"])
            result = find_nearest_one(data, r, c)
            if result is not None:
                results[orig_idx] = result[2]   # dist_km

    return results


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_CSV    = "/sdd/Dubaoset/src/Phong/Source/Cloud/bucket_time.csv"
    INPUT_FOLDER = "/sdd/Dubaoset/src/Phong/Source/Cloud/cloud_out"
    OUTPUT_CSV   = "/sdd/Dubaoset/src/Phong/Source/Cloud/analyst_buffer.csv"
    N_WORKERS    = 8

    # Load data
    input_LN = pd.read_csv(INPUT_CSV)
    input_LN["nearest_cloud_distance_km"] = np.nan

    list_of_tif = set(
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".tif")
    )

    # Chia đều thành N_WORKERS khối theo index gốc
    chunk_size = math.ceil(len(input_LN) / N_WORKERS)
    chunks = [
        input_LN.iloc[i * chunk_size : (i + 1) * chunk_size].copy()
        for i in range(N_WORKERS)
    ]

    # Đóng gói args cho mỗi worker
    args_list = [
        (chunk, INPUT_FOLDER, list_of_tif, worker_id)
        for worker_id, chunk in enumerate(chunks)
    ]

    # Chạy song song
    print(f"Spawning {N_WORKERS} workers …")
    with Pool(processes=N_WORKERS) as pool:
        all_results = pool.map(process_chunk, args_list)

    # Ghép kết quả từ tất cả worker vào dataframe gốc
    for worker_results in all_results:
        for orig_idx, dist_km in worker_results.items():
            input_LN.loc[orig_idx, "nearest_cloud_distance_km"] = dist_km

    # Xuất file
    input_LN.to_csv(OUTPUT_CSV, index=False)
    print(f"Done → {OUTPUT_CSV}")