import pandas as pd
import math
import tqdm

locations = {
    "Lao Cai": (22.5007, 103.968872),
    "Viet Tri": (21.325395, 105.403193),
    "Son La": (21.332236, 103.905399),
    "Cao Bang": (22.666667, 106.25),
    "Phu Lien": (20.794468, 106.61402),
    "Ha Dong": (20.955764, 105.753945),
    "Thanh Hoa": (19.761764, 105.778221),
    "Tuong Duong": (19.265, 104.47),
    "Vinh": (18.675, 105.691),
    "Dong Hoi": (17.483333, 106.6),
    "Da Nang": (16.04316, 108.206642),
    "Quy Nhon": (13.765258, 109.226201),
    "Nha Trang": (12.217986, 109.204996),
    "Phan Thiet": (10.933333, 108.1),
    "Pleiku": (13.966667, 108.016667),
    "Chau Doc": (10.7, 105.133333),
    "Ca Mau": (9.183333, 105.150008),
    "Nha Be": (10.638, 106.734)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def nearest_location(row):
    lat1, lon1 = float(row['lat']), float(row['lon'])
    distances = {pos: haversine(lat1, lon1, lat2, lon2) for pos, (lat2, lon2) in locations.items()}
    loc = min(distances, key=distances.get)
    return pd.Series([loc, distances[loc]])

def addLatLon(inputFile, outputFile):
    first = True
    for chunk in pd.read_csv(inputFile, chunksize=50000):
        chunk[['Location','min distance']] = chunk.apply(nearest_location, axis=1)
        chunk.to_csv(outputFile, mode='a', index=False, header=first)
        first = False

# This part of programe is run when it is directly called
if __name__ == "__main__":
    addLatLon("/sdd/Dubaoset/src/Phong/matchingPos2024.csv", "/sdd/Dubaoset/src/Phong/matchingPos2024addLatLon.csv")
