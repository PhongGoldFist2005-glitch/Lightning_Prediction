import pandas as pd
from datetime import datetime,timedelta

# Take time in the past
def roundMinute(Minute):
    out = Minute % 10
    return Minute - out


missing_path = "/sdd/Dubaoset/Cuong/matching_2021/file_error/matching_2021_file_error.csv" # 
missing_txt_path = "/sdd/Dubaoset/src/Phong/Log_a_Cuong/missing_2021.txt" #
band_type = ['B04B_10p', 'B05B_10p', 'B06B_10p', 'B09B_10p', 'B10B_10p', 'B11B_10p', 'B12B_10p', 'B14B_10p', 'B16B_10p', 'I2B_10p', 'I4B_10p', 'IRB_10p', 'VSB_10p', 'WVB_10p', 'B04B_20p', 'B05B_20p', 'B06B_20p', 'B09B_20p', 'B10B_20p', 'B11B_20p', 'B12B_20p', 'B14B_20p', 'B16B_20p', 'I2B_20p', 'I4B_20p', 'IRB_20p', 'VSB_20p', 'WVB_20p', 'B04B_30p', 'B05B_30p', 'B06B_30p', 'B09B_30p', 'B10B_30p', 'B11B_30p', 'B12B_30p', 'B14B_30p', 'B16B_30p', 'I2B_30p', 'I4B_30p', 'IRB_30p', 'VSB_30p', 'WVB_30p', 'B04B_40p', 'B05B_40p', 'B06B_40p', 'B09B_40p', 'B10B_40p', 'B11B_40p', 'B12B_40p', 'B14B_40p', 'B16B_40p', 'I2B_40p', 'I4B_40p', 'IRB_40p', 'VSB_40p', 'WVB_40p', 'B04B_50p', 'B05B_50p', 'B06B_50p', 'B09B_50p', 'B10B_50p', 'B11B_50p', 'B12B_50p', 'B14B_50p', 'B16B_50p', 'I2B_50p', 'I4B_50p', 'IRB_50p', 'VSB_50p', 'WVB_50p', 'B04B_60p', 'B05B_60p', 'B06B_60p', 'B09B_60p', 'B10B_60p', 'B11B_60p', 'B12B_60p', 'B14B_60p', 'B16B_60p', 'I2B_60p', 'I4B_60p', 'IRB_60p', 'VSB_60p', 'WVB_60p']

def takedata():
    miss_data = pd.read_csv(missing_path)
    with open(missing_txt_path,"a") as f:
        for idx, row in miss_data.iterrows():
            for band in band_type:
                if row[band] == "File error":
                    time = int(band[-3:-1])
                    band_m = str(band[:-4])
                    year = int(row["year"])
                    month = int(row["month"])
                    day = int(row["day"])
                    hour = int(row["hour"])
                    minute = roundMinute(int(row["minute"]))
                    start_time = datetime(year, month, day, hour, minute)
                    past_time = start_time - timedelta(minutes=time)
                    f.writelines(f"{past_time.year}/{past_time.month}/{past_time.day}:{past_time.hour}h{past_time.minute}; Band Missing: {band_m}\n")
    return True
def check_duplicate(output_file):
    result = set()
    with open(missing_txt_path,"r") as f:
        for item in f:
            data = str(item).strip()
            result.add(data)
    with open(output_file,"w") as f:
        f.writelines('\n'.join(list(result)))
    return True

# a = 'I2B_10p'
# print(a[:-4])
check_duplicate("/sdd/Dubaoset/src/Phong/Log_a_Cuong/final_2021.txt")



