import ftplib, os, logging
from datetime import datetime, timedelta
from cut_image import cutImage

def download_nc(HOST: str, USER: str, PASS: str, year: str, month: str, day: str, hour: str, minute: str,timeRange: int, limitTime: datetime, save_dir="/tmp"):
    sample_file = "/sdd/Dubaoset/DATA/HimaVN_UET/B04B/2021/01/01/B04B_20210101.Z0000.tif"
    output_root = "/sdd/Dubaoset/DATA/Continue"
    startTime = datetime(year= int(year), month= int(month), day= int(day), hour= int(hour), minute= int(minute))
    endTime = startTime + timedelta(minutes= timeRange * 10)
    
    if endTime > limitTime:
        endTime = limitTime
    
    ftps = None
    try:
        ftps = ftplib.FTP(HOST)
        ftps.login(USER, PASS)
        ftps.set_pasv(True)
        ftps.voidcmd('TYPE I')

        while startTime <= endTime:
            year = str(startTime.year)
            month = f"{int(startTime.month):02d}"
            day = f"{int(startTime.day):02d}"
            hour = f"{int(startTime.hour):02d}"
            minute = f"{int(startTime.minute):02d}"

            remote_dir = f"/jma/netcdf/{year}{month}/{day}"
            filename = f"NC_H09_{year}{month}{day}_{hour}{minute}_R21_FLDK.06001_06001.nc"
            remote_path = f"{remote_dir}/{filename}"

            local_dir = os.path.join(save_dir, year, month, day)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)
            # tempDir = f"/sdd/Dubaoset/src/Phong/Source/handleDown/Temp/{year}/{month}/{day}"
            # input_dir = os.path.join(tempDir, filename)

            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info(f"Kiểm tra file: {remote_path}")

            # Update time
            startTime += timedelta(minutes= 10)

            ftps.cwd(remote_dir)
            files = ftps.nlst()
            if filename not in files:
                logging.warning(f"File {filename} không tồn tại trong {remote_dir}")
                continue
            
            with open(local_path, "wb") as f:
                ftps.retrbinary(f"RETR {filename}", f.write)
            
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
        return None
    except Exception as e:
        logging.error(f"Lỗi khác: {e}")
        return None
    finally:
        if ftps:
            try:
                ftps.quit()
                logging.info("Đã đóng kết nối FTP.")
            except:
                pass
        return startTime