from datetime import datetime, timedelta
import random
import os
import pandas as pd

def randomLN(inputCSV, month, year, outputPath):
    northVN = pd.read_csv(inputCSV)
    # Làm tròn phút theo kiểu làm tròn lên để đưa hết về dạng bội của 10
    northVN['datetime'] = pd.to_datetime(northVN[['year', 'month', 'day', 'hour', 'minute']])
    northVN["datetime_rounded"] = northVN["datetime"].dt.ceil('10min')
    check = northVN.loc[(northVN["month"] == month) & (northVN["year"] == year), :].copy()
    # Muốn ghi tất cả dữ liệu thời gian năm-tháng-ngày-giờ-phút vào 1 file txt cho các dòng của tháng 5
    # format lai thời gian thành năm-tháng-ngày-giờ-phút
    check["datetime_str"] = check["datetime_rounded"].dt.strftime('%Y-%m-%d-%H-%M')
    # Ghi các thời gian unique vào file txt
    unique_times = check["datetime_str"].unique()
    print(len(unique_times))
    with open(outputPath, 'w') as f:
        for time in unique_times:
            f.write(f"{time}\n")
    check.drop(columns=["datetime", "datetime_rounded"], inplace=True)
    check.to_csv(outputPath.replace(".txt", ".csv"), index=False)
    return

def randomTime(outputPath, num):
    startTime = datetime(year= 2020, month=1, day= 1, hour=0, minute=0)
    endTime = datetime(year= 2022, month= 12, day= 31, hour= 23, minute= 50)
    bucket = []
    while startTime <= endTime:
        if startTime.month == 7 or startTime.month == 5 or startTime.month == 6:
            bucket.append(startTime.strftime("%Y-%m-%d-%H-%M"))
        startTime += timedelta(minutes=10)
    random.shuffle(bucket)
    bucket = bucket[:num]
    if os.path.exists(outputPath):
        print("Already exists file")
        with open(outputPath, "r") as f:
            bucket = f.read().splitlines()
    else:
        with open(outputPath, "w") as f:
            for item in bucket:
                f.write("%s\n" % item)
    return bucket

if __name__ == "__main__":
    inputCSV = "/sdd/Dubaoset/src/Phong/Source/Cloud/summer_northVN_LN_record.csv"
    month = 5
    year = 2021
    outputPath = "/sdd/Dubaoset/src/Phong/Source/Cloud/bucket_time.txt"
    randomLN(inputCSV, month, year, outputPath)