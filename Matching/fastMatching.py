import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
import rasterio
from collections import defaultdict
from functools import lru_cache
import rasterio

@lru_cache(maxsize=3000)  # tùy RAM
def open_tif_cached(filepath):
    try:
        return rasterio.open(filepath)
    except:
        return None

# Cấu hình
path = "/sdd/Dubaoset/DATA/MERGE"
list_features = ['B04B', 'B05B', 'B06B', 'B09B','B10B','B11B','B12B','B14B','B16B','I2B','I4B','IRB','VSB','WVB']
himaPath = "/sdd/Dubaoset/DATA/HimaVN_UET"
headers = ['version', 'year', 'month', 'day', 'hour', 'minute', 'second', 'ms', 'lat', 'lon', 'value_điện', 'Số cú sét', 'số sensor', 'DOF', 'Góc elip đo', 'chiều dài elip', 'chiều rộng elip', 'chi-square', 'time_tăng dữ liệu', 'thời gian của đỉnh peak', 'tốc độ tăng tối đa của dạng sóng', 'Loại sét', 'dl_góc cảm biến', 'chỉ báo tín hiệu', 'chỉ báo thời gian']

# --- Helpers thời gian & filename ---
def twoNum(x):
    x = int(x)
    return f"{x:02d}"

def roundMinute(date: datetime):
    out = int(date.minute) % 10
    if out != 0:
        return date + timedelta(minutes= 10 - out)
    return date

def timeInThePast(year: int, month: int, day: int, hour: int, minute: int, step: int):
    unRoundedTime = datetime(year= year, month= month, day= day, hour= hour, minute= minute)
    start_time = roundMinute(unRoundedTime)
    # giữ như logic cũ: từ 10 tới step-1 mỗi 10 phút
    step_minutes = range(10, step, 10)
    watcher = []
    for i in step_minutes:
        past_time = start_time - timedelta(minutes=i)
        watcher.append([
            past_time.year,
            past_time.month,
            past_time.day,
            past_time.hour,
            past_time.minute
        ])
    return watcher

def build_band_filepath(band, year, month, day, hour, minute):
    return os.path.join(
        himaPath,
        band,
        str(year),
        twoNum(month),
        twoNum(day),
        f"{band}_{year}{twoNum(month)}{twoNum(day)}.Z{twoNum(hour)}{twoNum(minute)}.tif"
    )

# --- Đọc CSV bằng pandas (như bạn) ---
def Label_data_Pandas(headers=headers):
    lightning_list = []
    for item in os.listdir(path):
        if item == "set_2020.csv":   # giữ logic cũ
            lightning_list.append(os.path.join(path, item))
    data_label = []
    for file_path in lightning_list:
        df = pd.read_csv(file_path, header=None, names=headers)
        data_label.extend(df.to_dict('records'))  # list of dicts dễ dùng hơn
    return data_label

# --- Hàm core: xây dựng mapping file -> list of requests trong 1 chunk ---
def build_requests_for_chunk(records_chunk, step=70):
    """
    records_chunk: list of dict (một chunk bản ghi)
    Trả về:
      - file_requests: dict { filepath: [ (local_row_idx, dest_index, lon, lat), ... ] }
      - per_row_meta: list metadata cho mỗi bản ghi trong chunk (giữ giá trị gốc để nối kết)
      - max_len_future: số cột band cho mỗi row (len(watcher) * len(list_features))
    """
    file_requests = defaultdict(list)
    per_row_meta = []  # lưu row gốc (dạng list) để ghép output
    max_len_future = 0

    # với mỗi row, tạo watcher và mapping
    for local_idx, rec in enumerate(records_chunk):
        year = int(rec['year'])
        month = int(rec['month'])
        day = int(rec['day'])
        hour = int(rec['hour'])
        minute = int(rec['minute'])
        lat = float(rec['lat'])
        lon = float(rec['lon'])
        watcher = timeInThePast(year, month, day, hour, minute, step)
        # số cột band cho row này
        row_len = len(watcher) * len(list_features)
        if row_len > max_len_future:
            max_len_future = row_len

        # lưu các info gốc (dạng list) - giữ ordering giống code gốc
        row_list = [rec.get(h, "") for h in headers]
        per_row_meta.append(row_list)

        # Với mỗi (i, time) và mỗi band j -> compute filepath và dest_index
        for i, (py, pm, pd, ph, pmin) in enumerate(watcher):
            for j, band in enumerate(list_features):
                dest_index = i * len(list_features) + j
                filepath = build_band_filepath(band, py, pm, pd, ph, pmin)
                file_requests[filepath].append((local_idx, dest_index, lon, lat))

    return file_requests, per_row_meta, max_len_future

# --- Xử lý 1 file: mở file 1 lần, trả về list cập nhật cho các row trong chunk ---
def process_file(filepath, requests_for_file):
    outputs = []
    src = open_tif_cached(filepath)

    if src is None:
        for local_idx, dest_index, lon, lat in requests_for_file:
            outputs.append((local_idx, dest_index, None))
        return outputs

    try:
        band = src.read(1)
        for local_idx, dest_index, lon, lat in requests_for_file:
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = band[row, col]
                else:
                    val = None
            except:
                val = None
            outputs.append((local_idx, dest_index, val))
    except:
        for local_idx, dest_index, lon, lat in requests_for_file:
            outputs.append((local_idx, dest_index, None))

    return outputs


# --- Hàm chính: xử lý chunk bằng cách xây mapping và mở file song song ---
def process_chunk(records_chunk, out_handle, step=70, max_workers=8):
    """
    records_chunk: list of dict records
    out_handle: opened file handle để append kết quả
    step: khoảng thời gian lùi (giữ 70 như cũ)
    Trả về: None (ghi trực tiếp vào out_handle)
    """
    file_requests, per_row_meta, max_len_future = build_requests_for_chunk(records_chunk, step=step)
    n_rows = len(records_chunk)
    # khởi tạo band_results cho chunk
    band_results = [ [None] * max_len_future for _ in range(n_rows) ]

    # xử lý từng file (song song theo file)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for filepath, reqs in file_requests.items():
            futures[executor.submit(process_file, filepath, reqs)] = filepath

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files", leave=False):
            res = fut.result()
            # res là list (local_row_idx, dest_index, val)
            for local_idx, dest_index, val in res:
                # chỉ cập nhật trong giới hạn max_len_future
                if dest_index < len(band_results[local_idx]):
                    band_results[local_idx][dest_index] = val

    # Sau khi xử lý xong tất cả file cho chunk -> ghi ra file
    lines_to_write = []
    for i in range(n_rows):
        row_base = per_row_meta[i]
        row_bands = band_results[i]
        # convert None -> empty string để tương thích csv cũ
        row_bands_str = ["" if v is None else str(v) for v in row_bands]
        line = ",".join(map(str, row_base + row_bands_str))
        lines_to_write.append(line)

    out_handle.write("\n".join(lines_to_write) + "\n")
    out_handle.flush()

# --- Hàm compareHima mới: chia chunk & gọi process_chunk ---
def compareHima_fast(records, output_file, chunk_size=1000, step=70, max_workers=8):
    """
    records: list of dict (như Label_data_Pandas trả về)
    chunk_size: số bản ghi xử lý 1 lần (giữ 1000 như code gốc)
    """
    total = len(records)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a") as f:
        for start in tqdm(range(0, total, chunk_size), desc="Chunks"):
            end = min(start + chunk_size, total)
            chunk = records[start:end]
            process_chunk(chunk, f, step=step, max_workers=max_workers)
    return True

# --- main ---
def main():
    lightning_content = Label_data_Pandas()
    compareHima_fast(lightning_content, "/sdd/Dubaoset/src/Phong/matchingPos2020.csv", chunk_size=1000, step=70, max_workers=12)

if __name__ == "__main__":
    main()
