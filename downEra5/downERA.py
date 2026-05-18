import cdsapi
dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_dewpoint_temperature",
        "2m_temperature",
        "instantaneous_surface_sensible_heat_flux",
        "high_cloud_cover",
        "medium_cloud_cover",
        "total_column_cloud_ice_water",
        "vertically_integrated_moisture_divergence",
        "convective_available_potential_energy",
        "k_index",
        "total_column_supercooled_liquid_water",
        "total_totals_index"
    ],
    "year": ["2022"],
    "month": ["07","08"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [24, 101, 6.48, 111]
}
print(request["month"], request["year"])
client = cdsapi.Client() # 
client.retrieve(dataset, request).download(f"/sdd/Dubaoset/src/Phong/Model/data/tempERA/era5_{request["year"][0]}_{request["month"][0]}_{request["month"][1]}.zip")

# Download về tất cả (File 3_4 va 4_5 bị trùng tháng 4)
# Chuyển hết về file zip
# Mở ra đưa về định dạng chuẩn mỗi cái gồm 2 loại file