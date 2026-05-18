import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_distributions_1(sea_cloud, land_cloud, bins=200, save_path=None):

    # ── Unpack ──────────────────────────────────────────────────────────────
    COLS = ["b04b", "b05b", "b06b", "i4b", "irb", "b16b"]

    def to_arrays(data):
        if len(data) == 0:
            return {k: np.array([]) for k in COLS}
        arr = np.array(data, dtype=np.float32)
        return {k: arr[:, i] for i, k in enumerate(COLS)}

    sc = to_arrays(sea_cloud)
    lc = to_arrays(land_cloud)

    # Band kép (hiệu)
    for d in (sc, lc):
        d["b04b_b05b"] = d["b04b"] - d["b05b"]
        d["b05b_b06b"] = d["b05b"] - d["b06b"]
        d["irb_b16b"]  = d["irb"]  - d["b16b"]
        d["i4b_b16b"]  = d["i4b"]  - d["b16b"]

    # ── Style ────────────────────────────────────────────────────────────────
    COLOR_SEA  = "#E65100"   # cam đậm  → Biển
    COLOR_LAND = "#1565C0"   # xanh dương → Đất liền
    ALPHA_HIST = 0.35
    LW_KDE     = 2.2

    # ── Helper: ranh giới KDE ────────────────────────────────────────────────
    def kde_zero_boundaries(xs, kde_fn, threshold_frac=0.015):
        ys        = kde_fn(xs)
        peak      = ys.max()
        thr       = threshold_frac * peak
        left_idx  = np.argmax(ys > thr)
        right_idx = len(ys) - 1 - np.argmax((ys > thr)[::-1])
        x_left  = xs[left_idx]  if left_idx  > 0           else None
        x_right = xs[right_idx] if right_idx < len(xs) - 1 else None
        return x_left, x_right

    # ── Vẽ một dataset lên ax ────────────────────────────────────────────────
    def _draw_single(ax, values, color, label):
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            return None, None          # không có dữ liệu

        lo, hi = np.nanpercentile(valid, 0.5), np.nanpercentile(valid, 99.5)
        edges  = np.linspace(lo, hi, bins + 1)
        xs     = np.linspace(lo, hi, 1024)

        ax.hist(valid, bins=edges, density=True,
                color=color, alpha=ALPHA_HIST, label=label)

        x_left = x_right = None
        if valid.size > 10:
            kde_fn = gaussian_kde(valid, bw_method="scott")
            ax.plot(xs, kde_fn(xs), color=color, lw=LW_KDE)
            x_left, x_right = kde_zero_boundaries(xs, kde_fn)

        return lo, hi

    # ── Vẽ ngưỡng KDE (dashed line + label) ─────────────────────────────────
    def _draw_thresholds(ax, values, color, lo, hi):
        valid = values[np.isfinite(values)]
        if valid.size <= 10 or lo is None or hi is None:
            return
        xs     = np.linspace(lo, hi, 1024)
        kde_fn = gaussian_kde(valid, bw_method="scott")
        x_left, x_right = kde_zero_boundaries(xs, kde_fn)

        for x_thresh, side in [(x_left, "L"), (x_right, "R")]:
            if x_thresh is None:
                continue
            ax.axvline(x=x_thresh, color=color,
                       lw=1.4, linestyle="--", alpha=0.80)
            ha     = "left"  if side == "L" else "right"
            offset = (hi - lo) * 0.012 * (1 if side == "L" else -1)
            ax.text(x_thresh + offset, 0.97,
                    f"{x_thresh:.2f}",
                    transform=ax.get_xaxis_transform(),
                    color=color, fontsize=8, fontweight="bold",
                    ha=ha, va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec=color, alpha=0.8))

    # ── Hàm tổng hợp: vẽ cả hai dataset lên một ax ──────────────────────────
    def _draw(ax, key, xlabel, title):
        sea_vals  = sc[key]
        land_vals = lc[key]

        lo_s, hi_s = _draw_single(ax, sea_vals,  COLOR_SEA,  "Mây – Biển")
        lo_l, hi_l = _draw_single(ax, land_vals, COLOR_LAND, "Mây – Đất liền")

        # Tính xlim chung
        lo_vals = [v for v in (lo_s, lo_l) if v is not None]
        hi_vals = [v for v in (hi_s, hi_l) if v is not None]
        if not lo_vals:
            ax.set_title(title + "\n(no data)"); return

        lo_all, hi_all = min(lo_vals), max(hi_vals)
        ax.set_xlim(lo_all, hi_all)

        # Vẽ ngưỡng sau khi biết xlim chung
        if lo_s is not None and hi_s is not None:
            _draw_thresholds(ax, sea_vals,  COLOR_SEA,  lo_all, hi_all)
        if lo_l is not None and hi_l is not None:
            _draw_thresholds(ax, land_vals, COLOR_LAND, lo_all, hi_all)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Mật độ xác suất", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)

    # ── Layout 5 × 2 ─────────────────────────────────────────────────────────
    #   Hàng 0-2 : band đơn  (6 ô)
    #   Hàng 3-4 : band kép  (4 ô)
    fig, axes = plt.subplots(5, 2, figsize=(16, 26))
    fig.suptitle("Phân bố phổ MÂY – Biển & Đất liền",
                 fontsize=16, fontweight="bold", y=1.005)

    # ── Band đơn ─────────────────────────────────────────────────────────────
    SINGLE = [
        ("b04b", "Giá trị band B04B", "Band đơn – B04B"),
        ("b05b", "Giá trị band B05B", "Band đơn – B05B"),
        ("b06b", "Giá trị band B06B", "Band đơn – B06B"),
        ("i4b",  "Giá trị band I4B",  "Band đơn – I4B"),
        ("irb",  "Giá trị band IRB",  "Band đơn – IRB"),
        ("b16b", "Giá trị band B16B", "Band đơn – B16B"),
    ]
    for idx, (key, xlabel, title) in enumerate(SINGLE):
        row, col = divmod(idx, 2)
        _draw(axes[row, col], key, xlabel, title)

    # Thêm đường phân cách giữa band đơn và kép
    line_y = axes[2, 0].get_position().y0 - 0.012
    fig.add_artist(
        plt.Line2D([0.05, 0.95], [line_y, line_y],
                   transform=fig.transFigure,
                   color="gray", linewidth=1.2, linestyle=":")
    )
    fig.text(0.5, line_y - 0.006, "▼  Band kép  ▼",
             ha="center", va="top", fontsize=11,
             color="gray", style="italic")

    # ── Band kép ─────────────────────────────────────────────────────────────
    DUAL = [
        ("b04b_b05b", "B04B − B05B", "Band kép – B04B − B05B"),
        ("b05b_b06b", "B05B − B06B", "Band kép – B05B − B06B"),
        ("irb_b16b",  "IRB − B16B",  "Band kép – IRB − B16B"),
        ("i4b_b16b",  "I4B − B16B",  "Band kép – I4B − B16B"),
    ]
    for idx, (key, xlabel, title) in enumerate(DUAL):
        row, col = divmod(idx, 2)
        _draw(axes[3 + row, col], key, xlabel, title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVED] {save_path}")

    plt.show()