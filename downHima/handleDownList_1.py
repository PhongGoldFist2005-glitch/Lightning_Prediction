from getDataFServerList import download_nc
from datetime import datetime
from TwoWord import TwoNum

HOST = "ftp.ptree.jaxa.jp"
USER = "phat14hy_gmail.com"
PASS = "SP+wari8"

List_file = "/sdd/Dubaoset/src/Phong/Source/Analyst/2020Phong_1.txt"
ListOfTime = []

# Đọc danh sách thời gian
with open(List_file, "r") as f:
    for line in f:
        data = line.strip().split("/")
        ListOfTime.append(datetime(
            year=int(data[0]), month=int(data[1]), day=int(data[2]), hour=int(data[3]), minute=int(data[4])
        ))

try:
    for i in range(0, len(ListOfTime), 50):
        batch = ListOfTime[i:i+50]

        # Bắt đầu từ phần tử đầu tiên trong batch
        start_index = 0
        
        while start_index < len(batch):
            # Gọi download bắt đầu từ phần tử idx trở đi
            result = download_nc(
                HOST=HOST,
                USER=USER,
                PASS=PASS,
                listOfTime=batch[start_index:],
                save_dir="/sdd/Dubaoset/src/Phong/Source/handleDown/Temp_1"
            )

            # Nếu trả về None tức batch xong
            if result is None:
                break

            # Nếu trả về 1 datetime bị lỗi → tính lại index để chạy tiếp
            try:
                start_index = batch.index(result)
            except ValueError:
                # Lỡ server trả về timestamp không thuộc batch → bỏ batch
                break

except KeyboardInterrupt:
    print("Stop Code")

# 3240 ảnh có thể tải được
# Thống kê cho 2024 trên server bổ sung được những cái gì down được đưa vào addition
# Thống kê cho 2020 trên server có những cái gì down được đưa vào addition
# Đưa vào datahave, bản cuối và thống kê hima sẽ có bao nhiêu mẫu cho từng năm
# Lùi theo từng mốc cho mẫu dương
# Lùi theo từng mốc cho mẫu âm
# Mẫu âm lấy bao nhiêu vào không hoàn toàn và hoàn toàn a Thắng
