import ftplib, os, time, logging

def retr_with_timeout(ftps, filename, local_path, timeout=30):
    """Tải file FTP với giám sát tiến độ — nếu bị treo thì dừng."""
    last_progress = time.time()
    downloaded = 0
    block_size = 65536  # 8KB mỗi lần đọc

    with open(local_path, "wb") as f:
        def callback(data):
            nonlocal last_progress, downloaded
            f.write(data)
            downloaded += len(data)
            last_progress = time.time()

        # Bắt đầu tải trong 1 luồng kiểm soát thời gian
        from threading import Thread

        done = False
        error = None

        def download():
            nonlocal done, error
            try:
                ftps.retrbinary(f"RETR {filename}", callback, blocksize=block_size)
                done = True
            except Exception as e:
                error = e

        t = Thread(target=download)
        t.start()

        # Giám sát quá trình
        while not done:
            time.sleep(1)
            if time.time() - last_progress > timeout:
                logging.error(f"File {filename} bị treo {timeout}s — hủy tải.")
                try:
                    ftps.abort()  # gửi lệnh ABOR tới server
                except:
                    pass
                t.join(timeout=2)
                raise TimeoutError(f"Tải {filename} bị treo.")
            if error:
                raise error

        t.join()

    size = os.path.getsize(local_path)
    logging.info(f"Tải xong {filename}, kích thước: {size} bytes")
    return size
