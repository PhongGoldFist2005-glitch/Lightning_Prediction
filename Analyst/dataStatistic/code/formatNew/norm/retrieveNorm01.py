import json
import numpy as np
import os
import matplotlib.pyplot as plt

# ── 1. Cấu hình File ──
value_set = {
    "type": "test",
    "settings": "5",
    "norm_type": "norm"
}

# train_stats_path = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_{value_set["settings"]}_scales.json'
# cache_json_input = '/sdd/Dubaoset/src/Thang/dataStatistic/code/cache/test_master_cache.json'
# print(train_stats_path)
# out_dir = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/{value_set["settings"]}Data/{value_set["norm_type"]}'
# os.makedirs(out_dir, exist_ok=True)

# test_mapped_output = os.path.join(out_dir, f'{value_set["type"]}_stats_summary.json') # Đổi tên file cho hợp lý
# test_plot_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_plot.png')

train_stats_path = f'/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/Phong/total_all_phong_scales.json'
cache_json_input = '/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/plot/raw/train_phong_data_master_cache.json'
print(train_stats_path)
out_dir = f'/sdd/Dubaoset/src/Thang/DataMB/TrainProcess/plot/phong'
os.makedirs(out_dir, exist_ok=True)

test_mapped_output = os.path.join(out_dir, f'total_train_stats_summary.json')
test_plot_output = os.path.join(out_dir, f'total_train_comparison_plot.png')

# Load dữ liệu
with open(train_stats_path, 'r') as f:
    train_stats = json.load(f)

with open(cache_json_input, 'r') as f:
    master_cache = json.load(f)

bandName = list(master_cache.keys())
summary_metrics = {} # Đổi tên biến cho đúng mục đích

# ── Chuẩn bị khung vẽ ──
fig, axes = plt.subplots(len(bandName), 1, figsize=(14, 5 * len(bandName)))
if len(bandName) == 1: axes = [axes]

# ── 2. Logic Tính toán & Chuẩn hóa ──
for ax, band in zip(axes, bandName):
    # Lấy thông số gốc
    raw_min = train_stats.get(band, {}).get('min', 0.0)
    raw_max = train_stats.get(band, {}).get('max', 1.0)
    raw_range = raw_max - raw_min
    
    if raw_range <= 0 and band != 'NDVI': continue

    # Thang đo hiển thị chuẩn hóa
    if band == 'NDVI':
        display_min, display_max = -1.0, 1.0
        n_target_bins = 100
    elif band in ('Dem_value', 'DEMIsLand'):
        display_min, display_max = -1.0, 1.0
        n_target_bins = 150
    else:
        display_min, display_max = 0.0, 1.0
        n_target_bins = 200
        
    display_range = display_max - display_min

    # Dữ liệu từ Cache
    cache_data = master_cache[band]
    g_min, g_max, n_master = cache_data['g_min'], cache_data['g_max'], cache_data['n_master_bins']
    
    # Giữ nguyên L0 và L1 để tính thống kê riêng biệt
    counts_l0 = np.array(cache_data['counts_l0'])
    counts_l1 = np.array(cache_data['counts_l1'])
    total_rows_l0 = cache_data['total_rows_l0']
    total_rows_l1 = cache_data['total_rows_l1']

    # 1. Khôi phục V_raw
    master_bin_idx = np.arange(n_master)
    V_raw = g_min + (master_bin_idx + 0.5) * ((g_max - g_min) / n_master)
    
    # 2. Lọc mảng
    if band == 'B10B':
        print(raw_min, raw_max)
    valid_mask = (V_raw >= raw_min) & (V_raw <= raw_max)
    V_raw_valid = V_raw[valid_mask]
    counts_l0_valid = counts_l0[valid_mask]
    counts_l1_valid = counts_l1[valid_mask]
    
    # 3. Chuẩn hóa V_raw thành V_norm
    if band == 'NDVI':
        V_norm = V_raw_valid
    elif band in ('Dem_value', 'DEMIsLand'):
        max_abs_val = max(abs(raw_max), abs(raw_min))
        V_norm = V_raw_valid / max_abs_val if max_abs_val > 0 else V_raw_valid
    else:
        V_norm = (V_raw_valid - raw_min) / raw_range

    # ── 3. TÍNH CHỈ SỐ THỐNG KÊ (JSON Format) ──
    # Hàm tính min/max/mean an toàn bỏ qua các bin có count = 0
    def get_stats(counts_arr):
        valid_bins = counts_arr > 0
        if not np.any(valid_bins):
            return 0.0, 0.0, 0.0
            
        c_min = np.min(V_norm[valid_bins])
        c_max = np.max(V_norm[valid_bins])
        c_mean = np.sum(V_norm * counts_arr) / np.sum(counts_arr)
        return c_min, c_max, c_mean

    min_l0, max_l0, mean_l0 = get_stats(counts_l0_valid)
    min_l1, max_l1, mean_l1 = get_stats(counts_l1_valid)

    # Lưu theo đúng Format của bạn
    summary_metrics[band] = {
        "label_0": {
            "total_rows": total_rows_l0,
            "min_norm": min_l0,
            "max_norm": max_l0,
            "mean_norm": mean_l0
        },
        "label_1": {
            "total_rows": total_rows_l1,
            "min_norm": min_l1,
            "max_norm": max_l1,
            "mean_norm": mean_l1
        }
    }

    # ── 4. PHẦN VẼ (Gộp lại vẽ bar chart như code cũ của bạn) ──
    target_bin_width = display_range / n_target_bins
    centers = display_min + (np.arange(n_target_bins) + 0.5) * target_bin_width

    def get_density(counts_arr, total_r):
        t_counts = np.zeros(n_target_bins)
        # Re-binning vào target bins để vẽ
        bin_idx = np.floor((V_norm - display_min) / display_range * n_target_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_target_bins - 1)
        np.add.at(t_counts, bin_idx, counts_arr)
        return (t_counts / total_r * 100) if total_r > 0 else t_counts

    density_l0 = get_density(counts_l0_valid, total_rows_l0)
    density_l1 = get_density(counts_l1_valid, total_rows_l1)

    # Vẽ Bar cho Label 0
    ax.bar(centers, density_l0, width=target_bin_width * 0.4, color="steelblue", 
           alpha=0.6, label=f"Label 0 (N={total_rows_l0})", align='edge')
    
    # Vẽ Bar cho Label 1 (lệch sang một chút để không bị che khuất hoàn toàn)
    ax.bar(centers, density_l1, width=-target_bin_width * 0.4, color="darkorange", 
           alpha=0.6, label=f"Label 1 (N={total_rows_l1})", align='edge')

    # Vẽ 2 đường Mean riêng biệt
    if mean_l0 != 0:
        ax.axvline(mean_l0, color="blue", linestyle="--", linewidth=2, 
                   label=f"Mean L0: {mean_l0:.4f}")
    
    if mean_l1 != 0:
        ax.axvline(mean_l1, color="red", linestyle="-.", linewidth=2, 
                   label=f"Mean L1: {mean_l1:.4f}")

    ax.set_xlim(display_min, display_max)
    ax.set_title(f"Band: {band} | Comparison Label 0 vs Label 1", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mật độ (%)")
    ax.set_xlabel("Normalized Value")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right", fontsize='small', ncol=2)

# ── 5. LƯU KẾT QUẢ ──
with open(test_mapped_output, 'w') as f:
    json.dump(summary_metrics, f, indent=4)

plt.tight_layout()
plt.savefig(test_plot_output, dpi=150)
print(f"Hoàn thành! File JSON (Chỉ chứa Stats Summary) lưu tại: {test_mapped_output}")