import json
import numpy as np
import os
import matplotlib.pyplot as plt

# ── 1. Cấu hình File ──
value_set = {
    "type": "test",
    "settings": "5",
    "norm_type": "unnorm"
}

train_stats_path = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_{value_set["settings"]}_scales.json'
cache_json_input = '/sdd/Dubaoset/src/Thang/dataStatistic/code/cache/test_master_cache.json'

# Đảm bảo thư mục Output tồn tại
out_dir = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/{value_set["settings"]}Data/{value_set["norm_type"]}'
os.makedirs(out_dir, exist_ok=True)

test_mapped_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_report_combined.json')
test_plot_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_plot_combined.png')

# Load dữ liệu
with open(train_stats_path, 'r') as f:
    train_stats = json.load(f)

with open(cache_json_input, 'r') as f:
    master_cache = json.load(f)

bandName = list(master_cache.keys())
mapped_metrics = {}

# ── Chuẩn bị khung vẽ (Canvas) ──
fig, axes = plt.subplots(len(bandName), 1, figsize=(14, 5 * len(bandName)))
if len(bandName) == 1: axes = [axes]

# ── 2. Logic Ánh Xạ (Mapping) & Vẽ ──
for ax, band in zip(axes, bandName):
    # 2.1 Lấy thông số thước đo từ Train Stats
    if band == 'NDVI':
        t_min_v1, t_max_v1 = -1.0, 1.0
        n_target_bins = 100
    elif band in ('Dem_value', 'DEMIsLand'):
        t_min_v1 = train_stats.get(band, {}).get('min', -3000.0)
        t_max_v1 = train_stats.get(band, {}).get('max', 3000.0)
        n_target_bins = 150
    else:
        t_min_v1 = train_stats[band]['min']
        t_max_v1 = train_stats[band]['max']
        n_target_bins = 200
        
    range_v1 = t_max_v1 - t_min_v1
    if range_v1 <= 0: continue

    # 2.2 Xử lý Master Cache 
    cache_data = master_cache[band]
    g_min = cache_data['g_min']
    g_max = cache_data['g_max']
    n_master = cache_data['n_master_bins']
    
    # GỘP NHÃN: Cộng gộp L0 và L1 thành 1 mảng tổng thể
    counts_total = np.array(cache_data['counts_l0']) + np.array(cache_data['counts_l1'])
    total_rows = cache_data['total_rows_l0'] + cache_data['total_rows_l1']

    # --- Áp dụng công thức Mapping ---
    master_bin_idx = np.arange(n_master)
    V_raw = g_min + (master_bin_idx + 0.5) * ((g_max - g_min) / n_master)
    
    # Lọc giá trị hợp lệ theo thang đo Train
    valid_mask = (V_raw >= t_min_v1) & (V_raw <= t_max_v1)
    V_raw_valid = V_raw[valid_mask]
    counts_total_valid = counts_total[valid_mask]

    # ── 3. TÍNH CHỈ SỐ THỐNG KÊ RÚT GỌN (Lưu JSON) ──
    def get_stats(v_arr, counts_arr):
        valid_bins = counts_arr > 0
        if not np.any(valid_bins):
            return 0.0, 0.0, 0.0
        c_min = np.min(v_arr[valid_bins])
        c_max = np.max(v_arr[valid_bins])
        c_mean = np.sum(v_arr * counts_arr) / np.sum(counts_arr)
        return float(c_min), float(c_max), float(c_mean)

    comb_min, comb_max, comb_mean_exact = get_stats(V_raw_valid, counts_total_valid)

    # Lưu Metrics rút gọn thẳng vào key của Band (Không chia label_0, label_1)
    mapped_metrics[band] = {
        "total_rows": total_rows,
        "min": comb_min,
        "max": comb_max,
        "mean": comb_mean_exact,
        "train_scale_used": {
            "min": float(t_min_v1),
            "max": float(t_max_v1)
        }
    }

    # ── 4. CHUẨN BỊ DỮ LIỆU VÀ VẼ BIỂU ĐỒ ──
    # Ép về số lượng bin mục tiêu (target bins) để vẽ Bar Chart
    bin_idx_v1 = np.floor((V_raw_valid - t_min_v1) / range_v1 * n_target_bins).astype(int)
    bin_idx_v1 = np.clip(bin_idx_v1, 0, n_target_bins - 1)
    
    target_counts = np.zeros(n_target_bins)
    np.add.at(target_counts, bin_idx_v1, counts_total_valid)
    
    # Tính mật độ phần trăm (%) tổng thể
    density_pct = (target_counts / total_rows * 100) if total_rows > 0 else target_counts
    
    # Tính tọa độ trục X (centers)
    target_bin_width = range_v1 / n_target_bins
    centers = t_min_v1 + (np.arange(n_target_bins) + 0.5) * target_bin_width

    # Vẽ Bar Chart gộp chung (Màu steelblue)
    ax.bar(centers, density_pct, width=target_bin_width * 0.8, color="steelblue", alpha=0.7, label="Phân phối gốc (Gộp nhãn)")
    
    # Vẽ đường Mean tổng (Màu crimson)
    if comb_mean_exact != 0.0:
        ax.axvline(comb_mean_exact, color="crimson", linestyle="--", linewidth=2.5, label=f"Mean: {comb_mean_exact:.2f}")

    ax.set_xlim(t_min_v1, t_max_v1)
    ax.set_title(f"Band: {band} | Thang đo gốc: {t_min_v1:.2f} đến {t_max_v1:.2f}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mật độ (%)")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right")

# ── 5. LƯU KẾT QUẢ ──
with open(test_mapped_output, 'w') as f:
    json.dump(mapped_metrics, f, indent=4)

plt.tight_layout()
plt.savefig(test_plot_output, dpi=150)
print(f"Hoàn thành! File JSON rút gọn (Gộp nhãn) lưu tại: {test_mapped_output}")
print(f"Xong! Ảnh đã lưu tại: {test_plot_output}")