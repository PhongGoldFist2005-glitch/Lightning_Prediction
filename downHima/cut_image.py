import xarray as xr
import rasterio
import os
import glob
import rioxarray

def cutImage(input_dir, sample_file, output_root):
    os.makedirs(output_root, exist_ok=True)
    # ==== 2. Đọc ảnh mẫu để làm chuẩn grid ====
    with rasterio.open(sample_file) as src:
        dst_crs = src.crs
        dst_transform = src.transform
        width, height = src.width, src.height
        sample_bounds = src.bounds

    print("Ảnh mẫu bounds:", sample_bounds)
    print("Ảnh mẫu size:", width, "x", height)

    # ==== 3. Đổi tên band ====
    band_map = {
        "albedo_03": "VSB",
        "albedo_04": "B04B",
        "albedo_05": "B05B",
        "albedo_06": "B06B",
        "tbb_07": "I4B",
        "tbb_08": "WVB",
        "tbb_09": "B09B",
        "tbb_10": "B10B",
        "tbb_11": "B11B",
        "tbb_12": "B12B",
        "tbb_13": "IRB",
        "tbb_14": "B14B",
        "tbb_15": "I2B",
        "tbb_16": "B16B"
    }

    # ==== 4. Lặp qua các file NC ====
    nc_file = input_dir if input_dir[-3:] == ".nc" else None
    if not os.path.isfile(input_dir) or not input_dir.endswith(".nc"):
        print("Không phải file nc hoặc không tồn tại:", input_dir)
        return None

    # for nc_file in nc_files:
    print(f"\nĐang xử lý file: {nc_file}")
    
    # Lấy ngày giờ từ tên file
    # dạng: NC_H08_YYYYMMDD_hhmm_R21_FLDK.06001_06001.nc
    filename = os.path.basename(nc_file)
    parts = filename.split("_")
    date_str = parts[2]   # YYYYMMDD
    time_str = parts[3]   # hhmm
    year = date_str[0:4]
    month = date_str[4:6]
    day = date_str[6:8]
    # Mở dataset
    with xr.open_dataset(nc_file, decode_timedelta=True) as ds:    
        for band_name, band_short in band_map.items():
            if band_name not in ds.data_vars:
                print(f"Band {band_name} không có trong {nc_file}")
                continue
            
            print(f"Xử lý {band_name} ({band_short}) ...")
            da = ds[band_name]

            # Gán CRS
            da.rio.write_crs("EPSG:4326", inplace=True)

            # Ép kiểu float32
            da = da.astype("float32")
            da.rio.write_nodata(-9999, inplace=True)

            # Resample về grid mẫu
            da_resampled = da.rio.reproject(
                dst_crs=dst_crs,
                transform=dst_transform,
                shape=(height, width),
                resampling=rasterio.enums.Resampling.average
            )

            # Tạo thư mục band riêng
            band_dir = output_root + "/" + band_short + "/" + year + "/" + month + "/" + day
            os.makedirs(band_dir, exist_ok=True)

            # Tạo tên file output
            out_name = f"{band_short}_{date_str}_Z{time_str}_VN.tif"
            out_file = os.path.join(band_dir, out_name)
            if os.path.exists(out_file):
                os.remove(out_file)

            # Xuất raster
            da_resampled.rio.to_raster(out_file, compress="lzw")
            print(f"Xuất: {out_file}")
    try:
        os.remove(nc_file)
        print("Đã xóa file:", nc_file)
    except Exception as e:
        print("Không thể xóa file:", e)
    print("\nHoàn thành toàn bộ.")

# cutImage("/sdd/Dubaoset/src/Phong/Source/handleDown/B04B_20200101.Z0000.nc","/sdd/Dubaoset/DATA/HimaVN_UET/B04B/2021/01/01/B04B_20210101.Z0000.tif","/sdd/Dubaoset/src/Phong/Source/handleDown")