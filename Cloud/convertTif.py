import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import warnings
import os
from tqdm import tqdm
from multiprocessing import Pool, Manager
import traceback
warnings.filterwarnings('ignore')


# ─── Cấu hình đường dẫn ──────────────────────────────────────────────────────
TIF_FILE = r"/sdd/Dubaoset/DATA/HimaVN_UET/B11B/2021/01/01/B11B_20210101.Z0000.tif"

# Buffer (degree) để tránh thiếu điểm sát biên khi lọc
BBOX_MARGIN = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def read_nc(nc_path: str) -> dict:
    """
    Đọc file NC của AHI-CMSK.

    Trả về dict:
        lat2d     : np.ndarray (rows, cols) – vĩ độ 2D
        lon2d     : np.ndarray (rows, cols) – kinh độ 2D
        cloud_bin : np.ndarray (rows, cols) – CloudMaskBinary (0/1)
    """
    print(f"[1/5] Đọc NC: {nc_path}")
    ds = nc.Dataset(nc_path)

    lat2d     = ds.variables['Latitude'][:].data.astype(np.float64)
    lon2d     = ds.variables['Longitude'][:].data.astype(np.float64)
    cloud_bin = ds.variables['CloudMaskBinary'][:].data.astype(np.float32)

    ds.close()

    print(f"   Kích thước lưới gốc : {lat2d.shape}")
    print(f"   Lat range            : [{lat2d.min():.2f}, {lat2d.max():.2f}]")
    print(f"   Lon range            : [{lon2d.min():.2f}, {lon2d.max():.2f}]")

    return dict(lat2d=lat2d, lon2d=lon2d, cloud_bin=cloud_bin)


def read_ref_tif(tif_path: str) -> dict:
    """
    Đọc file TIF tham chiếu để lấy bbox, kích thước lưới đích.

    Trả về dict:
        lat_min, lat_max, lon_min, lon_max : float         – giới hạn khu vực
        height, width                       : int           – kích thước lưới đích
        transform                           : rasterio.Affine
        crs                                 : rasterio.CRS  – dùng thẳng, tránh lỗi PROJ/EPSG
    """
    print(f"[2/5] Đọc TIF tham chiếu: {tif_path}")
    with rasterio.open(tif_path) as src:
        b = src.bounds
        info = dict(
            lat_min=b.bottom, lat_max=b.top,
            lon_min=b.left,   lon_max=b.right,
            height=src.height, width=src.width,
            transform=src.transform,
            crs=src.crs,
        )
    print(f"   Bbox Vietnam: lon=[{info['lon_min']}, {info['lon_max']}], "
          f"lat=[{info['lat_min']}, {info['lat_max']}]")
    print(f"   Kích thước lưới đích: {info['height']} x {info['width']}")
    print(f"   CRS (từ TIF): {info['crs']}")
    return info


def filter_bbox(nc_data: dict, ref: dict, margin: float = 0.5) -> dict:
    """
    Bước quan trọng: chuyển lưới cong 2D → tập điểm 1D nằm trong bbox.

    Cụ thể:
      - Từ mảng 2D (rows×cols) ta tìm tất cả pixel có tọa độ nằm trong bbox
      - Flatten chúng thành mảng 1D: lat_pts, lon_pts, bin_pts
      - Đây là bước "2D → 1D" trước khi nội suy ra lưới đều
    """
    print(f"[3/5] Lọc điểm 2D nằm trong bbox (margin={margin}°)...")

    lat2d = nc_data['lat2d']
    lon2d = nc_data['lon2d']
    inside = (
        (lon2d >= ref['lon_min'] - margin) & (lon2d <= ref['lon_max'] + margin) &
        (lat2d >= ref['lat_min'] - margin) & (lat2d <= ref['lat_max'] + margin)
    )

    lat_pts = lat2d[inside]
    lon_pts = lon2d[inside]
    bin_pts = nc_data['cloud_bin'][inside]

    print(f"   Số điểm 1D sau lọc: {lat_pts.shape[0]:,} / {inside.size:,}")

    return dict(lat=lat_pts, lon=lon_pts, cloud_bin=bin_pts)


def regrid(pts: dict, ref: dict, method: str = 'nearest') -> dict:
    """
    Nội suy từ tập điểm 1D lên lưới phẳng đều (regular grid).

    Phương pháp:
      'nearest'  – gán nhãn từ pixel gần nhất, phù hợp nhất cho mask/label (discrete)
      'linear'   – nội suy tuyến tính, phù hợp cho dữ liệu liên tục (probability)

    Trả về:
        cloud_bin_grid : np.ndarray (height, width) int8
        lat_1d         : np.ndarray (height,)  – trục vĩ độ 1D (top→bottom)
        lon_1d         : np.ndarray (width,)   – trục kinh độ 1D (left→right)
    """
    print(f"[4/5] Nội suy lên lưới đều ({method})...")

    # Tạo trục tọa độ 1D (đây chính là kết quả "1D lat, lon")
    lat_1d = np.linspace(ref['lat_max'], ref['lat_min'], ref['height'])
    lon_1d = np.linspace(ref['lon_min'], ref['lon_max'], ref['width'])

    # Tạo meshgrid để nội suy
    lon_g, lat_g = np.meshgrid(lon_1d, lat_1d)   # (height, width)
    target_pts   = np.column_stack([lon_g.ravel(), lat_g.ravel()])
    source_pts   = np.column_stack([pts['lon'], pts['lat']])

    # Nội suy
    bin_out = griddata(source_pts, pts['cloud_bin'], target_pts, method=method)

    # Reshape về 2D
    bin_out = bin_out.reshape(ref['height'], ref['width']).astype(np.int8)

    print(f"   lat_1d: shape={lat_1d.shape}, range=[{lat_1d[-1]:.3f}, {lat_1d[0]:.3f}]")
    print(f"   lon_1d: shape={lon_1d.shape}, range=[{lon_1d[0]:.3f}, {lon_1d[-1]:.3f}]")
    print(f"   CloudMaskBin unique: {np.unique(bin_out)}")

    return dict(
        cloud_bin=bin_out,
        lat_1d=lat_1d,
        lon_1d=lon_1d,
    )


def write_geotif(grids: dict, ref: dict, out_path: str) -> None:
    """
    Ghi kết quả ra GeoTIF với 1 band:
      Band 1: CloudMaskBinary (0=Clear, 1=Cloud)
    """
    print(f"[5/5] Ghi GeoTIF: {out_path}")

    transform = from_bounds(
        ref['lon_min'], ref['lat_min'],
        ref['lon_max'], ref['lat_max'],
        ref['width'],   ref['height'],
    )

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=ref['height'],
        width=ref['width'],
        count=1,
        dtype='int8',
        crs=ref['crs'],
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(grids['cloud_bin'], 1)
        dst.update_tags(1,
            name="CloudMaskBinary",
            description="Binary Cloud Mask: 0=Clear, 1=Cloud"
        )
        dst.update_tags(
            source="AHI Himawari-8 NOAA CMSK",
            crs=str(ref['crs']),
            region="Vietnam",
        )

    print(f"   ✅ Xong! Output: {out_path}")
    with rasterio.open(out_path) as r:
        print(f"   Shape : {r.height} x {r.width}")
        print(f"   Bounds: {r.bounds}")
        print(f"   CRS   : {r.crs}")



# ─── Worker function for multiprocessing ─────────────────────────────────────
def process_file_worker(args):
    """
    Worker function để xử lý một file NC.
    
    Args:
        args: tuple (file_nc, folderPath, outputFolder, TIF_FILE, shared_status_dict)
    
    Returns:
        dict: {'file': file_nc, 'status': 'success'|'failed', 'error': error_msg or None}
    """
    file_nc, folderPath, outputFolder, TIF_FILE_PATH, shared_status = args
    try:
        # Đánh dấu file đang xử lý
        shared_status[file_nc] = 'processing'
        
        filepath = os.path.join(folderPath, file_nc)
        chunk = file_nc.split("_")[-2]
        year, month, day, hour, minute = chunk[1:5], chunk[5:7], chunk[7:9], chunk[9:11], chunk[11:13]
        
        OUT_FILE = os.path.join(outputFolder, f"AHI_{year}{month}{day}_Z{hour}{minute}.tif")
        main(nc_file=filepath, tif_file=TIF_FILE_PATH, out_file=OUT_FILE)
        
        # Đánh dấu file đã xong
        shared_status[file_nc] = 'completed'
        return {'file': file_nc, 'status': 'success', 'error': None}
        
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        shared_status[file_nc] = 'failed'
        return {'file': file_nc, 'status': 'failed', 'error': error_msg}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(nc_file, out_file, tif_file=TIF_FILE):
    print("=" * 60)
    print("AHI CMSK | 2D curvilinear → 1D regular grid | Vietnam clip")
    print("=" * 60)

    nc_data = read_nc(nc_file)
    ref     = read_ref_tif(tif_file)
    pts     = filter_bbox(nc_data, ref, margin=BBOX_MARGIN)
    grids   = regrid(pts, ref, method='nearest')
    write_geotif(grids, ref, out_file)


# ─── Main Parallel Processing ────────────────────────────────────────────────
def main_parallel(folderPath, outputFolder, TIF_FILE, num_processes=8):
    """
    Xử lý tất cả files bằng multiprocessing với số process được chỉ định.
    
    Args:
        folderPath: đường dẫn folder chứa files NC
        outputFolder: đường dẫn folder output
        TIF_FILE: đường dẫn file TIF tham chiếu
        num_processes: số process (mặc định 8)
    """
    os.makedirs(outputFolder, exist_ok=True)
    
    # Lọc danh sách files NC chưa xử lý
    nc_files = [f for f in os.listdir(folderPath) if f.endswith(".nc")]
    tif_files = os.listdir(outputFolder)
    
    not_finished = []
    for file_nc in nc_files:
        chunk = file_nc.split("_")[-2]
        year, month, day, hour, minute = chunk[1:5], chunk[5:7], chunk[7:9], chunk[9:11], chunk[11:13]
        filepath = f"AHI_{year}{month}{day}_Z{hour}{minute}.tif"
        if filepath not in tif_files:
            not_finished.append(file_nc)
    
    print(f"📊 Tổng file cần xử lý: {len(not_finished)}")
    print(f"🔄 Sử dụng {num_processes} processes...\n")
    
    if not not_finished:
        print("✅ Không có file nào cần xử lý!")
        return
    
    # Tạo shared dictionary để track trạng thái files
    with Manager() as manager:
        shared_status = manager.dict()
        
        # Khởi tạo trạng thái cho tất cả files
        for file_nc in not_finished:
            shared_status[file_nc] = 'pending'
        
        # Chuẩn bị arguments cho worker function
        worker_args = [
            (file_nc, folderPath, outputFolder, TIF_FILE, shared_status) 
            for file_nc in not_finished
        ]
        
        # Xử lý song song
        success_count = 0
        failed_count = 0
        failed_files = []
        
        with Pool(processes=num_processes) as pool:
            results = tqdm(
                pool.imap_unordered(process_file_worker, worker_args),
                total=len(worker_args),
                desc="Processing NC files"
            )
            
            for result in results:
                if result['status'] == 'success':
                    success_count += 1
                else:
                    failed_count += 1
                    failed_files.append((result['file'], result['error']))
                    print(f"\n   ❌ LỖI: {result['file']}")
                    print(f"      {result['error']}\n")
        
        # In báo cáo trạng thái cuối cùng
        print(f"\n{'═'*70}")
        print(f"✅ Thành công: {success_count} files")
        print(f"❌ Thất bại:   {failed_count} files")
        print(f"📊 Tổng cộng:  {success_count + failed_count} files")
        print(f"{'═'*70}\n")
        
        # In chi tiết trạng thái từng file
        print("📋 Chi tiết trạng thái từng file:")
        for file_nc in not_finished:
            status = shared_status[file_nc]
            icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
            print(f"   {icon} {file_nc}: {status}")
        
        if failed_files:
            print("\n📋 Danh sách files thất bại:")
            for file_nc, error in failed_files:
                print(f"   - {file_nc}")


if __name__ == "__main__":
    folderPath = "/sdd/Dubaoset/src/Phong/Source/Cloud/survey_05_2023"
    outputFolder = "/sdd/Dubaoset/src/Phong/Source/Cloud/cloud_out"
    
    # Chạy với 8 processes
    main_parallel(folderPath, outputFolder, TIF_FILE, num_processes=10)

        