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

# Đảm bảo thư mục tồn tại
out_dir = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/{value_set["settings"]}Data/{value_set["norm_type"]}'
os.makedirs(out_dir, exist_ok=True)

test_mapped_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_report_0_1.json')
test_plot_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_plot_0_1.png')

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
        # Lấy từ JSON của bạn
        t_min_v1 = train_stats[band]['min']
        t_max_v1 = train_stats[band]['max']
        n_target_bins = 200
        
    range_v1 = t_max_v1 - t_min_v1
    if range_v1 <= 0: continue

    # 2.2 Xử lý Master Cache (Giả sử bạn đã tăng N_MASTER_BINS = 100000)
    cache_data = master_cache[band]
    g_min = cache_data['g_min']
    g_max = cache_data['g_max']
    n_master = cache_data['n_master_bins']
    
    counts_l0 = np.array(cache_data['counts_l0'])
    counts_l1 = np.array(cache_data['counts_l1'])
    total_rows_l0 = cache_data['total_rows_l0']
    total_rows_l1 = cache_data['total_rows_l1']

    # --- Áp dụng công thức Mapping ---
    master_bin_idx = np.arange(n_master)
    V_raw = g_min + (master_bin_idx + 0.5) * ((g_max - g_min) / n_master)
    
    valid_mask = (V_raw >= t_min_v1) & (V_raw <= t_max_v1)
    V_raw_valid = V_raw[valid_mask]
    counts_l0_valid = counts_l0[valid_mask]
    counts_l1_valid = counts_l1[valid_mask]
    
    bin_idx_v1 = np.floor((V_raw_valid - t_min_v1) / range_v1 * n_target_bins).astype(int)
    bin_idx_v1 = np.clip(bin_idx_v1, 0, n_target_bins - 1)
    
    target_counts_l0 = np.zeros(n_target_bins)
    target_counts_l1 = np.zeros(n_target_bins)
    
    np.add.at(target_counts_l0, bin_idx_v1, counts_l0_valid)
    np.add.at(target_counts_l1, bin_idx_v1, counts_l1_valid)
    
    # Tính mật độ phần trăm (%)
    dens_l0 = (target_counts_l0 / total_rows_l0 * 100) if total_rows_l0 > 0 else target_counts_l0
    dens_l1 = (target_counts_l1 / total_rows_l1 * 100) if total_rows_l1 > 0 else target_counts_l1
    
    # Tính tọa độ trục X (centers)
    target_bin_width = range_v1 / n_target_bins
    centers = t_min_v1 + (np.arange(n_target_bins) + 0.5) * target_bin_width

    # ── 3. TÍNH TOÁN MIN, MAX, MEAN ĐỂ LƯU JSON ──
    # Hàm trích xuất Min, Max, Mean thực tế dựa trên mảng V_raw_valid
    def get_stats(v_arr, counts_arr):
        valid_bins = counts_arr > 0
        if not np.any(valid_bins):
            return 0.0, 0.0, 0.0
        c_min = np.min(v_arr[valid_bins])
        c_max = np.max(v_arr[valid_bins])
        c_mean = np.sum(v_arr * counts_arr) / np.sum(counts_arr)
        return float(c_min), float(c_max), float(c_mean)

    min_l0, max_l0, val_mean_l0_exact = get_stats(V_raw_valid, counts_l0_valid)
    min_l1, max_l1, val_mean_l1_exact = get_stats(V_raw_valid, counts_l1_valid)

    # 2.3 Lưu Metrics gọn nhẹ theo đúng format bạn yêu cầu
    mapped_metrics[band] = {
        "label_0": {
            "total_rows": total_rows_l0,
            "min": min_l0,
            "max": max_l0,
            "mean": val_mean_l0_exact
        },
        "label_1": {
            "total_rows": total_rows_l1,
            "min": min_l1,
            "max": max_l1,
            "mean": val_mean_l1_exact
        },
        "train_scale_used": {
            "min": float(t_min_v1),
            "max": float(t_max_v1)
        }
    }

    # Tính Mean trên Bar Chart để hiển thị trực quan (theo code cũ của bạn)
    val_sum_c0 = np.sum(target_counts_l0)
    val_sum_c1 = np.sum(target_counts_l1)
    val_mean_l0 = np.sum(centers * target_counts_l0) / val_sum_c0 if val_sum_c0 > 0 else 0
    val_mean_l1 = np.sum(centers * target_counts_l1) / val_sum_c1 if val_sum_c1 > 0 else 0

    # ── 4. VẼ BIỂU ĐỒ ──
    # Vẽ Histogram
    ax.bar(centers, dens_l0, width=target_bin_width*0.8, color="royalblue", alpha=0.5, label="Val Label 0")
    ax.bar(centers, dens_l1, width=target_bin_width*0.8, color="crimson", alpha=0.5, label="Val Label 1")
    
    # Vẽ đường Mean của Validation (Nét đứt)
    if val_mean_l0 != 0:
        ax.axvline(val_mean_l0, color="blue", linestyle="--", linewidth=2.5, label=f"Mean L0 (Val): {val_mean_l0:.2f}")
    if val_mean_l1 != 0:
        ax.axvline(val_mean_l1, color="red", linestyle="--", linewidth=2.5, label=f"Mean L1 (Val): {val_mean_l1:.2f}")

    ax.set_xlim(t_min_v1, t_max_v1)
    ax.set_title(f"Band: {band} | Validation Distribution & Means", fontsize=14, fontweight='bold')
    ax.set_ylabel("Density (%)")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)

# Lưu kết quả
with open(test_mapped_output, 'w') as f:
    json.dump(mapped_metrics, f, indent=4)

plt.tight_layout()
plt.savefig(test_plot_output, dpi=150)
print(f"Hoàn thành! File JSON (Tóm tắt) đã lưu tại: {test_mapped_output}")
print(f"Hoàn thành! Biểu đồ đã lưu tại: {test_plot_output}")