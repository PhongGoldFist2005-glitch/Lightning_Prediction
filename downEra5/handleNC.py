import os
import rioxarray
import pandas as pd
import zipfile
import re
from tqdm import tqdm
import xarray as xr


def handZipFile(inputZipFile, outputFolder):
    fileNameSplit = re.split(r"[_.]",os.path.basename(inputZipFile))
    if fileNameSplit[len(fileNameSplit) - 1] != "zip":
        print("Not zip file")
        return None
    if len(fileNameSplit) == 5:
        finalOutputFolder = os.path.join(outputFolder, f"{fileNameSplit[0]}_{fileNameSplit[1]}_{fileNameSplit[2]}_{fileNameSplit[3]}")
    elif len(fileNameSplit) == 4:
        finalOutputFolder = os.path.join(outputFolder, f"{fileNameSplit[0]}_{fileNameSplit[1]}_{fileNameSplit[2]}")
    else:
        print("File name error")
        return None
    os.makedirs(finalOutputFolder, exist_ok= True)
    with zipfile.ZipFile(inputZipFile) as zip_ref:
        zip_ref.extractall(finalOutputFolder)


# def handleNCFolder(inputPath, root_dir):
#     # Có 2 loại band cần phải mở 2 file ra và xử lý
#     ds = xr.open_dataset(inputPath)
#     # đảm bảo có CRS
#     ds = ds.rio.write_crs("EPSG:4326")
#     # load trước để tăng tốc
#     ds.load()

#     for var in ds.data_vars:
#         for t in ds.valid_time.values:
#             time = pd.to_datetime(t)
#             year  = time.strftime("%Y")
#             month = time.strftime("%m")
#             day   = time.strftime("%d")
#             timestr = time.strftime("%Y%m%d%H%M%S")
        
#             # tạo folder theo cấu trúc
#             out_dir = os.path.join(root_dir, var.upper(), year, month, day)
#             os.makedirs(out_dir, exist_ok=True)
        
#             # chọn dữ liệu
#             single = ds[var].sel(valid_time=t)
        
#             filename = f"{var.upper()}_{timestr}.tif"
#             filepath = os.path.join(out_dir, filename)
        
#             single.rio.to_raster(filepath)
#     print(f"{os.path.basename(inputPath)} handled")

import os
import xarray as xr
import rioxarray as rxr
import pandas as pd
import numpy as np
from rasterio.enums import Resampling


def handleNCFolder(inputPath, root_dir, template_path):

    # 🔹 Mở raster mẫu (chỉ mở 1 lần)
    template = rxr.open_rasterio(template_path)

    # 🔹 Mở file NetCDF
    ds = xr.open_dataset(inputPath)
    ds.load()

    for var in ds.data_vars:
        for t in ds.valid_time.values:

            time = pd.to_datetime(t)
            year  = time.strftime("%Y")
            month = time.strftime("%m")
            day   = time.strftime("%d")
            timestr = time.strftime("%Y%m%d%H%M%S")

            out_dir = os.path.join(root_dir, var.upper(), year, month, day)
            os.makedirs(out_dir, exist_ok=True)

            # 🔹 Lấy 1 time step
            single = ds[var].sel(valid_time=t)

            # Nếu có expver (ERA5 thường có)
            if "expver" in single.dims:
                single = single.isel(expver=0)

            # Đảm bảo đúng thứ tự dimension
            single = single.transpose("latitude", "longitude")

            # Đảm bảo latitude giảm dần (north ở trên)
            if single.latitude[0] < single.latitude[-1]:
                single = single.sortby("latitude", ascending=False)

            # Gán CRS
            single = single.rio.write_crs("EPSG:4326")

            # 🔥 Quan trọng nhất: match với raster mẫu
            single_resampled = single.rio.reproject_match(
                template,
                resampling=Resampling.average
            )

            # Chuẩn hóa nodata (tránh -inf)
            single_resampled = single_resampled.where(
                np.isfinite(single_resampled)
            )
            single_resampled.rio.write_nodata(np.nan, inplace=True)

            filename = f"{var.upper()}_{timestr}.tif"
            filepath = os.path.join(out_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                print("Đã xóa file bị trùng lặp")

            single_resampled.rio.to_raster(filepath)

    print(f"{os.path.basename(inputPath)} handled")

if __name__ == "__main__":
    root_dir = "/sdd/Dubaoset/DATA/ERA5"
    template_path = "/sdd/Dubaoset/src/Phong/Model/data/Template/original/CAPE_20200101000000.tif"
    inputFolderPath = "/sdd/Dubaoset/src/Phong/Model/data/ERA5"

    # Check tránh trùng file đỡ phải làm lại
    # listZipFile = [os.path.join(inputFolderPath, i) for i in os.listdir(inputFolderPath)]
    # result = ["/sdd/Dubaoset/src/Phong/Model/data/tempERA/era5_2022_07_08.zip"]
    # for item in listZipFile:
    #     if item.endswith(".zip"):
    #         check = item[:-4]
    #         if check not in listZipFile:
    #             result.append(item)
    
    # Unzip file
    # for i in tqdm(range(len(result)), desc= f"Unzipping file"):
    #     zip_path = result[i]
    #     tqdm.write(f"Processing {os.path.basename(zip_path)}")
    #     handZipFile(result[i], "/sdd/Dubaoset/src/Phong/Model/data/ERA5")
        
    newListUnZipFolder = [os.path.join(inputFolderPath, i) for i in os.listdir(inputFolderPath) if not i.endswith(".zip")]
    for folder in tqdm(newListUnZipFolder, total= len(newListUnZipFolder), desc= f"Handle folder"):
        tqdm.write(f"Processing {os.path.basename(folder)}")
        listOfFileNC = [os.path.join(folder, i) for i in os.listdir(folder)]
        for file in listOfFileNC:
            handleNCFolder(file, root_dir, template_path)
