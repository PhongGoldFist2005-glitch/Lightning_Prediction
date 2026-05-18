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

train_stats_path = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/total_{value_set["settings"]}_scales.json'
cache_json_input = '/sdd/Dubaoset/src/Thang/dataStatistic/code/cache/test_master_cache.json'

out_dir = f'/sdd/Dubaoset/src/Thang/dataStatistic/NewTeacher/Output/{value_set["settings"]}Data/{value_set["norm_type"]}'
os.makedirs(out_dir, exist_ok=True)

test_mapped_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_report_total.json')
test_plot_output = os.path.join(out_dir, f'{value_set["type"]}_comparison_plot_total.png')

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

# ── 2. Logic Ánh Xạ, Chuẩn hóa và Vẽ ──
for ax, band in zip(axes, bandName):
    
    # 2.1 Lấy thông số từ Train Stats để thực hiện Norm
    raw_min = train_stats.get(band, {}).get('min', 0.0)
    raw_max = train_stats.get(band, {}).get('max', 1.0)
    raw_range = raw_max - raw_min
    
    if raw_range <= 0 and band != 'NDVI': 
        continue

    # Xác định thang đo hiển thị (Target Scale) và số lượng Bins để vẽ
    if band == 'NDVI':
        display_min, display_max, n_target_bins = -1.0, 1.0, 100
    elif band in ('Dem_value', 'DEMIsLand'):
        display_min, display_max, n_target_bins = -1.0, 1.0, 150
    else:
        display_min, display_max, n_target_bins = 0.0, 1.0, 200
        
    display_range = display_max - display_min

    # 2.2 Xử lý Master Cache (Gộp nhãn)
    cache_data = master_cache[band]
    counts_total = np.array(cache_data['counts_l0']) + np.array(cache_data['counts_l1'])
    total_rows = cache_data['total_rows_l0'] + cache_data['total_rows_l1']

    # Khôi phục giá trị vật lý gốc từ Master Bins
    master_bin_idx = np.arange(cache_data['n_master_bins'])
    V_raw = cache_data['g_min'] + (master_bin_idx + 0.5) * ((cache_data['g_max'] - cache_data['g_min']) / cache_data['n_master_bins'])
    
    # Lọc mask theo dải Train (Bỏ outliers)
    valid_mask = (V_raw >= raw_min) & (V_raw <= raw_max)
    V_raw_valid = V_raw[valid_mask]
    counts_valid = counts_total[valid_mask]

    if len(V_raw_valid) == 0: continue

    # 2.3 Thực hiện CHUẨN HÓA (Normalization)
    if band == 'NDVI':
        V_norm = V_raw_valid
    elif band in ('Dem_value', 'DEMIsLand'):
        max_abs_val = max(abs(raw_max), abs(raw_min))
        V_norm = V_raw_valid / max_abs_val if max_abs_val > 0 else V_raw_valid
    else:
        V_norm = (V_raw_valid - raw_min) / raw_range

    # 2.4 Tính toán thống kê cho JSON
    actual_data_mask = counts_valid > 0
    v_actual = V_norm[actual_data_mask]
    
    c_min = float(np.min(v_actual)) if len(v_actual) > 0 else 0.0
    c_max = float(np.max(v_actual)) if len(v_actual) > 0 else 0.0
    c_mean = float(np.sum(V_norm * counts_valid) / np.sum(counts_valid)) if np.sum(counts_valid) > 0 else 0.0

    mapped_metrics[band] = {
        "total_rows": int(total_rows),
        "min_norm": c_min,
        "max_norm": c_max,
        "mean_norm": c_mean
    }

    # ── 3. VẼ BIỂU ĐỒ ──
    # Map V_norm vào các Target Bins để vẽ bar chart
    bin_idx_norm = np.floor((V_norm - display_min) / display_range * n_target_bins).astype(int)
    bin_idx_norm = np.clip(bin_idx_norm, 0, n_target_bins - 1)
    
    target_counts = np.zeros(n_target_bins)
    np.add.at(target_counts, bin_idx_norm, counts_valid)
    
    # Tính mật độ % và tọa độ X
    density_pct = (target_counts / total_rows * 100) if total_rows > 0 else target_counts
    target_bin_width = display_range / n_target_bins
    centers = display_min + (np.arange(n_target_bins) + 0.5) * target_bin_width

    # Vẽ cột phân phối
    ax.bar(centers, density_pct, width=target_bin_width * 0.8, color="steelblue", alpha=0.7, label="Phân phối chuẩn hóa (Gộp nhãn)")
    
    # Vẽ đường Mean (Crimson)
    ax.axvline(c_mean, color="crimson", linestyle="--", linewidth=2.5, label=f"Mean Norm: {c_mean:.4f}")

    ax.set_xlim(display_min, display_max)
    ax.set_title(f"Band: {band} | Normalized Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mật độ (%)")
    ax.set_xlabel("Giá trị chuẩn hóa")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right")

# ── 4. LƯU KẾT QUẢ ──
with open(test_mapped_output, 'w') as f:
    json.dump(mapped_metrics, f, indent=4)

plt.tight_layout()
plt.savefig(test_plot_output, dpi=150)

print(f"--- Hoàn thành ---")
print(f"JSON Stats lưu tại: {test_mapped_output}")
print(f"Ảnh biểu đồ lưu tại: {test_plot_output}")