import datetime

YEAR = [2020, 2021, 2022, 2023, 2024]
for year in YEAR:
    countRange = 0
    startTime = datetime.datetime(year= year,month= 1,day= 1,hour= 0,minute= 0)
    endTime = datetime.datetime(year= year,month= 12,day= 31,hour= 23,minute= 50)
    while startTime <= endTime:
        countRange += 1
        startTime += datetime.timedelta(minutes = 10)
    print(f"{year}: {countRange}")

# 2020: 52704
# 2021: 52560
# 2022: 52560
# 2023: 52560
# 2024: 52704
# NC_H09_20221031_0000_R21_FLDK.06001_06001.nc
# NC_H08_20221031_0000_R21_FLDK.06001_06001.nc