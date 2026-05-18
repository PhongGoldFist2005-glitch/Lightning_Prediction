import rasterio

class ReadHimawari():
    def __init__(self, path):
        self.path = path
        with rasterio.open(self.path) as src:
            self.get_band = src.read(1)
            #self.src = src # !
            self.num_features = src.count
            self.height = src.height
            self.width = src.width
            self.transform = src.transform
            self.shape = src.shape

    def get_bands(self):
        return self.num_features
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_col_row(self, lon: float, lat: float):
        col, row = (~self.transform) * (lon, lat)
        return int(col), int(row)
    
    def get_value_feature(self, row: int, col: int):
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.get_band[row, col]
        return "Out of bound"