import os
import numpy as np
import rasterio
from datetime import datetime
from scipy.ndimage import binary_dilation

# Thay đổi tập band, thay đổi cơ chế lưu chỉ lưu các ô là class 1 mây trong ô vuông khi có mây
# Dilation: pixel (i,j) được coi là "vùng có mây" nếu trong ô vuông 11×11 tâm (i,j) tồn tại ÍT NHẤT 1 pixel có nhãn 1
# Đổi kích thước ô vuông từ 9 * 9 thành 11 * 11.
def classification(folder_cloud_path):
    sea_cloud = []
    land_cloud = []
    sea_non_cloud = []
    land_non_cloud = []

    band_need = ["B04B", "B05B", "B06B" ,"I4B",  "IRB", "B16B"]

    # Root Himawari data
    himaPath = "/sdd/Dubaoset/DATA/HimaVN_UET"

    # Load DEM (chỉ 1 lần)
    with rasterio.open("/sdd/Dubaoset/DATA/DEM/DEM_4km.tif") as src:
        dem_data = src.read(1)

    sea_mask = dem_data < 0  # True = biển

    # ===== Kernel dilation: ô vuông 11×11 (bán kính 5 ô mỗi chiều) =====
    RADIUS = 4
    kernel = np.ones((2 * RADIUS + 1, 2 * RADIUS + 1), dtype=bool)

    for file in os.listdir(folder_cloud_path):
        if not file.endswith(".tif"):
            continue

        try:
            # ===== Parse time =====
            date_str = file.split('_')[1].split('.')[0]
            dt_obj = datetime.strptime(date_str, "%Y%m%d%H%M")
            year, month, day = dt_obj.year, dt_obj.month, dt_obj.day
            hour, minute = dt_obj.hour, dt_obj.minute

            # ===== Load bands =====
            band_data = {}

            for band in band_need:
                filename = f"{band}_{year}{str(month).zfill(2)}{str(day).zfill(2)}.Z{str(hour).zfill(2)}{str(minute).zfill(2)}.tif"
                filepath = os.path.join(
                    himaPath,
                    band,
                    str(year),
                    str(month).zfill(2),
                    str(day).zfill(2),
                    filename
                )

                if not os.path.exists(filepath):
                    print(f"[WARNING] Missing file: {filepath}")
                    raise FileNotFoundError

                with rasterio.open(filepath) as src:
                    band_data[band] = src.read(1)

            # ===== Load cloud mask =====
            cloud_path = os.path.join(folder_cloud_path, file)
            with rasterio.open(cloud_path) as src:
                cloud_mask = src.read(1)

            # ===== Check shape =====
            h, w = cloud_mask.shape
            if dem_data.shape != (h, w):
                print(f"[WARNING] Shape mismatch DEM vs cloud: {file}")
                continue

            for band in band_need:
                if band_data[band].shape != (h, w):
                    print(f"[WARNING] Shape mismatch band {band}: {file}")
                    continue

            # ===== Extract bands =====
            b04b = band_data["B04B"]
            b05b = band_data["B05B"]
            b06b = band_data["B06B"]
            irb  = band_data["IRB"]
            b16b = band_data["B16B"]
            i4b  = band_data["I4B"]

            # ===== Masks =====
            raw_cloud = cloud_mask != 0   # nhãn gốc

            # Dilation: pixel (i,j) được coi là "vùng có mây" nếu trong ô vuông
            # 11×11 tâm (i,j) tồn tại ÍT NHẤT 1 pixel có nhãn 1
            cloud_dilated = binary_dilation(raw_cloud, structure=kernel)
            non_cloud_dilated = ~cloud_dilated

            sea_cloud_mask      = sea_mask  & cloud_dilated
            land_cloud_mask     = (~sea_mask) & cloud_dilated
            sea_non_cloud_mask  = sea_mask  & non_cloud_dilated
            land_non_cloud_mask = (~sea_mask) & non_cloud_dilated

            # ===== Append data (vectorized) =====
            sea_cloud.extend(zip(
                b04b[sea_cloud_mask],
                b05b[sea_cloud_mask],
                b06b[sea_cloud_mask],
                irb[sea_cloud_mask],
                b16b[sea_cloud_mask],
                i4b[sea_cloud_mask]
            ))

            land_cloud.extend(zip(
                b04b[land_cloud_mask],
                b05b[land_cloud_mask],
                b06b[land_cloud_mask],
                irb[land_cloud_mask],
                b16b[land_cloud_mask],
                i4b[land_cloud_mask]
            ))

            sea_non_cloud.extend(zip(
                b04b[sea_non_cloud_mask],
                b05b[sea_non_cloud_mask],
                b06b[sea_non_cloud_mask],
                irb[sea_non_cloud_mask],
                b16b[sea_non_cloud_mask],
                i4b[sea_non_cloud_mask]
            ))

            land_non_cloud.extend(zip(
                b04b[land_non_cloud_mask],
                b05b[land_non_cloud_mask],
                b06b[land_non_cloud_mask],
                irb[land_non_cloud_mask],
                b16b[land_non_cloud_mask],
                i4b[land_non_cloud_mask]
            ))

            print(f"[OK] Processed: {file}")

        except Exception as e:
            print(f"[ERROR] Skip file {file}: {e}")
            continue

    return sea_cloud, land_cloud, sea_non_cloud, land_non_cloud