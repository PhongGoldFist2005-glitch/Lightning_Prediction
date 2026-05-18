from pathlib import Path
from cut_image import cutImage

sample_file = "/sdd/Dubaoset/DATA/HimaVN_UET/B04B/2021/01/01/B04B_20210101.Z0000.tif"
output_root = "/sdd/Dubaoset/DATA/Result1-6"
folderPath = Path("/sdd/Dubaoset/DATA/HimaVN_UET/2024")

array = sorted(folderPath.glob("*/*/"))

print(f"Tổng thư mục: {len(array)}\n")

# print(array[:10])
for idx, day_folder in enumerate(array, start=1):
    if not day_folder.is_dir():
        continue
    print(f"[{idx:03}] {day_folder}")
    nc_files = list(day_folder.glob("*.nc"))
    if not nc_files:
        print(f"Không có file .nc trong {day_folder}\n")
        continue
    try:
        cutImage(day_folder, sample_file, output_root)
        print(f"Done: {day_folder}\n")
    except Exception as e:
        print(f"Lỗi {day_folder}: {e}\n")

print("Hoàn tất!")


# thieu 2024/01/01 & 2024/01/02
# thieu NC_H09_20240106_1620_R21_FLDK.06001_06001.nc