# Dự Báo Sét Cho Việt Nam Sử Dụng Dữ Liệu Đa Nguồn và Deep Learning (2020–2024)

## Tổng Quan

Dự án tập trung xây dựng hệ thống dự báo sét cho toàn lãnh thổ Việt Nam sử dụng dữ liệu đa nguồn bao gồm dữ liệu vệ tinh, khí tượng tái phân tích, địa hình, thực vật và dữ liệu trạm sét trong giai đoạn từ năm 2020 đến 2024.

Mục tiêu chính của dự án là xây dựng bộ dữ liệu quy mô lớn phục vụ huấn luyện các mô hình Deep Learning nhằm dự báo hiện tượng sét dựa trên đặc trưng khí quyển, sự phát triển mây đối lưu và điều kiện môi trường.

Hệ thống tích hợp các nguồn dữ liệu:

- Himawari-8/9
- ERA5 Reanalysis
- DEM (Digital Elevation Model)
- NDVI
- Dữ liệu trạm sét

Pipeline bao gồm:
- Xác định cụ thể bài toán
- Thu thập dữ liệu đa nguồn
- Đồng bộ dữ liệu không gian và thời gian
- Làm sạch dữ liệu
- Feature Engineering
- Chuẩn hóa dữ liệu
- Lọc dữ liệu sét dựa trên đặc tính mây
- Sinh bộ dữ liệu Train / Validation / Test
- Huấn luyện mô hình Deep Learning
- Tuning siêu tham số
- Đánh giá mô hình

Các mô hình chính được triển khai:

- LSTM
- Attention-LSTM

---

# Mục Tiêu Dự Án

Dự án hướng tới:

- Xây dựng bộ dữ liệu dự báo sét quy mô lớn cho toàn Việt Nam
- Tích hợp dữ liệu khí tượng và viễn thám đa nguồn
- Học đặc trưng không gian - thời gian của mây dông và sét
- Cải thiện khả năng dự báo sét bằng Deep Learning
- Đánh giá hiệu quả của cơ chế Attention trong bài toán chuỗi thời gian
- Xây dựng pipeline có khả năng mở rộng cho hệ thống dự báo thực tế

---

# Nguồn Dữ Liệu

## 1. Dữ Liệu Himawari-8/9

Dữ liệu vệ tinh Himawari cung cấp ảnh vệ tinh địa tĩnh với tần suất cao cho khu vực Việt Nam.

### Thông Tin Khai Thác

- Nhiệt độ đỉnh mây
- Brightness Temperature
- Kênh hồng ngoại
- Kênh hơi nước
- Sự phát triển của mây đối lưu
- Diễn biến mây theo thời gian

### Vai Trò

Đây là nguồn dữ liệu chính dùng để phân tích sự hình thành và phát triển của mây đối lưu liên quan đến hiện tượng sét.

---

## 2. Dữ Liệu ERA5

ERA5 cung cấp dữ liệu tái phân tích khí tượng với độ phân giải thời gian theo giờ.

### Các Biến ERA5 Sử Dụng

| Biến | Ý nghĩa |
|---|---|
| CAPE | Convective Available Potential Energy |
| D2M | Nhiệt độ điểm sương tại 2m |
| HCC | High Cloud Cover |
| ISHF | Instantaneous Surface Heat Flux |
| KX | K Index |
| MCC | Medium Cloud Cover |
| T2M | Nhiệt độ tại 2m |
| TCIW | Total Column Ice Water |
| TCSLW | Total Column Supercooled Liquid Water |
| TOTALX | Total Totals Index |
| VIMD | Vertically Integrated Moisture Divergence |

### Vai Trò

Các biến ERA5 giúp mô hình học được:

- Độ bất ổn khí quyển
- Điều kiện hình thành đối lưu
- Độ ẩm khí quyển
- Đặc trưng mây
- Điều kiện nhiệt bề mặt
- Khả năng phát triển dông sét

---

## 3. Dữ Liệu DEM

DEM được sử dụng để bổ sung thông tin địa hình.

### Đặc Trưng

- Độ cao địa hình

### Vai Trò

Địa hình ảnh hưởng mạnh đến quá trình hình thành dông sét, đặc biệt ở khu vực miền núi Việt Nam.

---

## 4. Dữ Liệu NDVI

NDVI được sử dụng để mô tả đặc trưng bề mặt thực vật.

### Vai Trò

- Mô tả điều kiện bề mặt
- Đặc trưng môi trường
- Ảnh hưởng theo mùa

---

## 5. Dữ Liệu Trạm Sét

Dữ liệu trạm sét được sử dụng làm nhãn cho bài toán học có giám sát.

### Thông Tin

- Thời gian xuất hiện sét
- Tọa độ
- Cường độ sét

### Vai Trò

Dữ liệu sét được đồng bộ với Himawari và ERA5 để tạo nhãn dự báo.

---

# Pipeline Xử Lý Dữ Liệu

## 1. Đồng Bộ Dữ Liệu Đa Nguồn

Toàn bộ dữ liệu được đồng bộ theo:

- Không gian
- Thời gian
- Hệ tọa độ
- Grid dữ liệu

### Các Bài Toán Xử Lý

- Khác biệt độ phân giải
- Thiếu timestamp
- Sai lệch cảm biến
- Nội suy không gian

---

## 2. Làm Sạch Dữ Liệu

Nhiều kỹ thuật tiền xử lý được áp dụng:

- Xác định các khoảng dữ liệu hiện có
- Thông tin của từng loại dữ liệu
- Loại bỏ trùng lặp
- Đưa dữ liệu về cùng độ phân giải về không gian và thời gian
- Missing value handle
- Đưa về cùng format datatype
- Sửa dữ liệu bị sai
- Thống nhất đơn vị
- Làm sạch text
- Đánh giá các thông tin phân bố của dữ liệu (Min, Max, Mean, Std, Histogram, Median, Histogram).
- Dựa vào khoảng và thông tin phân bố của dữ liệu để chọn ra function chuẩn hóa dữ liệu hợp lý nhất  (Normalization).
- feature encoding

---

## 3. Lọc Dữ Liệu Sét Dựa Trên Đặc Tính Mây

Dữ liệu sét được lọc dựa trên đặc trưng vật lý của mây đối lưu.

### Điều Kiện Lọc

- Nhiệt độ đỉnh mây
- Đặc điểm mây đối lưu
- Quá trình phát triển mây
- Tính liên tục theo thời gian
- Điều kiện không gian

### Mục Tiêu

Loại bỏ các điểm sét nhiễu hoặc không phù hợp về mặt vật lý nhằm tăng chất lượng dữ liệu huấn luyện.

---

# Feature Engineering

Nhiều kỹ thuật Feature Engineering được triển khai nhằm tăng hiệu quả mô hình.

## Phương Pháp Feature Engineering

Các phương pháp lựa chọn và đánh giá đặc trưng được sử dụng:

- BC
- MI (Mutual Information)
- RFE (Recursive Feature Elimination)
- SB
- SF
- SHAP

## Đặc Trưng Band Kép

Ngoài các biến gốc, dự án còn xây dựng các đặc trưng band kép dựa trên:

- Chênh lệch nhiệt độ giữa các band
- Quan hệ giữa các kênh hồng ngoại
- Đặc trưng vật lý của mây đối lưu
- Kiến thức domain về quá trình hình thành sét

Các đặc trưng này được thống kê và lựa chọn dựa trên hiểu biết khí tượng và đặc tính phát triển của mây dông.


---

# Chuẩn Hóa Dữ Liệu

Các kỹ thuật chuẩn hóa được sử dụng:

- Min-Max Scaling
- Max Normalization

### Mục Tiêu

- Tăng độ ổn định khi huấn luyện
- Giúp mô hình hội tụ nhanh hơn
- Đồng nhất dữ liệu đa nguồn

---

# Sinh Bộ Dữ Liệu

Dữ liệu được chia thành:

- Train Set
- Validation Set
- Test Set

## Giai Đoạn Dữ Liệu

| Dataset | Thời Gian |
|---|---|
| Train | 2020–2023 |
| Validation | Tháng 6/2024 (theo tỷ lệ thực tế) |
| Test | Tháng 5/2024 và Tháng 7/2024 (theo tỷ lệ thực tế) |

## Phạm Vi Không Gian

- Toàn lãnh thổ Việt Nam
- Dữ liệu dạng grid


---

# Mô Hình Deep Learning

## 1. LSTM

Mô hình LSTM được sử dụng để học đặc trưng chuỗi thời gian của khí quyển và mây đối lưu.


---

## 2. Attention-LSTM

Attention được tích hợp vào LSTM nhằm cải thiện khả năng tập trung vào các thời điểm quan trọng.

---

# Pipeline Huấn Luyện

## Thành Phần Pipeline

- Data Loader
- Sequence Generator
- Batch Processing
- Model Training
- Validation Monitoring
- Testing Pipeline

## Kỹ Thuật Huấn Luyện

- Early Stopping
- Learning Rate Scheduler
- Regularization

---

# Hyperparameter Tuning

Các siêu tham số được tuning:

- lr
- weight_decay
- batch_size
- input_size
- hidden_size
- dropout
- num_layer
- output_features
- alpha
- gamma


## Phương Pháp

- Grid Search
- Random Search

---

# Metrics Đánh Giá

Các metrics sử dụng:

- Precision
- Recall
- F1-score
- PR AUC


### Trọng Tâm

- Giảm false alarm
- Tăng khả năng phát hiện sét
- Tăng độ nhạy với dông mạnh

---

# Kiến Trúc Hệ Thống

```text
Nguồn Dữ Liệu
│
├── Himawari-8/9
├── ERA5
├── DEM
├── NDVI
└── Trạm Sét
        │
        ▼
Đồng Bộ Dữ Liệu
        │
        ▼
Làm Sạch & Lọc Dữ Liệu Sét
        │
        ▼
Feature Engineering
        │
        ▼
Chuẩn Hóa Dữ Liệu
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Huấn Luyện LSTM / Attention-LSTM
        │
        ▼
Đánh Giá & Tuning
        │
        ▼
Hệ Thống Dự Báo Sét