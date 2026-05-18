import ftplib, os, logging
from datetime import datetime, timedelta
from cut_image import cutImage
import time
from checkLost import retr_with_timeout

def download_nc(HOST: str, USER: str, PASS: str, listOfTime , save_dir="/tmp"):
    sample_file = "/sdd/Dubaoset/DATA/HimaVN_UET/B04B/2021/01/01/B04B_20210101.Z0000.tif"
    output_root = "/sdd/Dubaoset/DATA/Continue_1"
    
    ftps = None
    try:
        ftps = ftplib.FTP(HOST)
        ftps.login(USER, PASS)
        ftps.set_pasv(True)
        ftps.voidcmd('TYPE I')

        for startTime in listOfTime:
            year = str(startTime.year)
            month = f"{int(startTime.month):02d}"
            day = f"{int(startTime.day):02d}"
            hour = f"{int(startTime.hour):02d}"
            minute = f"{int(startTime.minute):02d}"

            remote_dir = f"/jma/netcdf/{year}{month}/{day}"
            filename = f"NC_H08_{year}{month}{day}_{hour}{minute}_R21_FLDK.06001_06001.nc"
            remote_path = f"{remote_dir}/{filename}"

            local_dir = os.path.join(save_dir, year, month, day)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)

            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info(f"Kiểm tra file: {remote_path}")

            ftps.cwd(remote_dir)
            files = ftps.nlst()
            if filename not in files:
                logging.warning(f"File {filename} không tồn tại trong {remote_dir}")
                continue
            
            try:
                fileSize = retr_with_timeout(ftps, filename, local_path, timeout=300)
            except TimeoutError:
                logging.error(f"Tải {filename} quá lâu — bỏ qua.")
                if os.path.exists(local_path):
                    os.remove(local_path)
                return startTime


            # Kiểm tra dung lượng file
            if os.path.getsize(local_path) < 1000:  # <1KB coi như hỏng
                logging.error(f"File {local_path} quá nhỏ, có thể bị lỗi. Xóa file.")
                os.remove(local_path)
                continue

            logging.info(f"Tải thành công: {local_path}")

            # Cut image
            cutImage(local_path, sample_file, output_root)
    
    except ftplib.all_errors as e:
        logging.error(f"Lỗi FTP: {e}")
        return startTime
    except Exception as e:
        logging.error(f"Lỗi khác: {e}")
        return startTime
    finally:
        if ftps:
            try:
                ftps.quit()
                logging.info("Đã đóng kết nối FTP.")
            except:
                pass
    return None