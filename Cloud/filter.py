import pandas as pd
from tqdm import tqdm
from readHima import ReadHimawari
import os
import uuid


headers = ['version', 'year', 'month', 'day', 'hour', 'minute', 'second', 'ms', 'lat', 'lon', 'value_điện', 'Số cú sét', 'số sensor', 'DOF', 'Góc elip đo', 'chiều dài elip', 'chiều rộng elip', 'chi-square', 'time_tăng dữ liệu', 'thời gian của đỉnh peak', 'tốc độ tăng tối đa của dạng sóng', 'Loại sét', 'dl_góc cảm biến', 'chỉ báo tín hiệu', 'chỉ báo thời gian']
list_csv = [
    "/sdd/Dubaoset/DATA/MERGE/set_2020.csv",
    "/sdd/Dubaoset/DATA/MERGE/set_2021.csv",
    "/sdd/Dubaoset/DATA/MERGE/set_2022.csv",
    "/sdd/Dubaoset/DATA/MERGE/set_2023.csv",
    "/sdd/Dubaoset/DATA/MERGE/set_2024.csv"
]

# Tạo thêm 2 cột row và col
# Lọc điều kiện
# Lọc hết các file
# Tạo ra 1 file csv mới
row_min = 8
row_max = 119
col_min = 13
col_max = 187
# Lọc theo tháng
month_filter = [5,6,7]

def filter_csv(list_csv, sample_file, output_csv):
    transform = ReadHimawari(sample_file).transform

    df_list = []
    for csv_file in tqdm(list_csv, total=len(list_csv), desc="Processing CSV files"):
        df = pd.read_csv(csv_file, names=headers, header=None)
        cols, rows = (~transform) * (df['lon'].values, df['lat'].values)
        df['col'] = cols.astype(int)
        df['row'] = rows.astype(int)

        # Lọc theo điều kiện lat, lon, và tháng
        df_filtered = df[
            (df['row'] >= row_min) & (df['row'] <= row_max) &
            (df['col'] >= col_min) & (df['col'] <= col_max) &
            (df['month'].isin(month_filter))
        ]
        df_list.append(df_filtered)

    # Gộp tất cả DataFrame lại thành một DataFrame duy nhất
    combined_df = pd.concat(df_list, ignore_index=True)

    # Lưu DataFrame đã lọc vào file CSV mới
    if not os.path.exists(output_csv):
        combined_df.to_csv(output_csv, index=False)
    else:
        new_output_name = uuid.uuid4() + ".csv"
        combined_df.to_csv(new_output_name, index=False)
        print(f"File {output_csv} already exists. Saved filtered data to {new_output_name} instead.")


if __name__ == "__main__":
    filter_csv(
        list_csv= list_csv, 
        sample_file= "/sdd/Dubaoset/DATA/HimaVN_UET/B04B/2021/01/01/B04B_20210101.Z0000.tif", 
        output_csv= "/sdd/Dubaoset/src/Phong/Source/Cloud/summer_northVN_LN_record.csv"
    )