from getDataFServer_1 import download_nc
from datetime import datetime, timedelta
from TwoWord import TwoNum

# --- THÔNG SỐ KẾT NỐI ---
HOST = "ftp.ptree.jaxa.jp"
USER = "phat14hy_gmail.com"
PASS = "SP+wari8"

# --- Đường dẫn ---
sample_file = "/sdd/Dubaoset/DATA/HimaVN_UET/B04B/2021/01/01/B04B_20210101.Z0000.tif"
output_root = "/sdd/Dubaoset/DATA/Continue"

startTime = datetime(year=2024, month= 5, day= 29, hour= 0, minute= 0)
endTime = datetime(year=2024, month= 5, day= 31, hour= 23, minute= 50)

try:
    while startTime <= endTime:
        year = str(startTime.year)
        month = TwoNum(startTime.month)
        day = TwoNum(startTime.day)
        hour = TwoNum(startTime.hour)
        minute = TwoNum(startTime.minute)

        download = download_nc(
            HOST= HOST,
            USER= USER,
            PASS= PASS,
            year= year,
            month= month,
            day= day,
            hour= hour,
            minute= minute,
            timeRange = 50,
            limitTime = endTime,
            save_dir= "/sdd/Dubaoset/src/Phong/Source/handleDown/Temp")
        
        startTime = download
        if startTime == None:
            print("Lỗi lấy đata")
            break
        else:
            startTime += timedelta(minutes= 10)
except KeyboardInterrupt:
    print("Stop Code")

# Thang 5
# NC_H09_20240101_0000_R21_FLDK.06001_06001.nc
# NC_H09_20240101_0000_R21_FLDK.06001_06001.nc
# 12h10 03/04