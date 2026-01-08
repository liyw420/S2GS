import argparse
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
from matplotlib.lines import Line2D
import numpy as np


def make_plot(save_path: Optional[str] = None, show: bool = True):
    # -----------------------
    # 数据（可按需要再修改）
    # -----------------------
    methods = [
        "4DGC\n(CVPR25)",
        "ReCon-GS\n(NeurIPS25)",
        "HiCoM\n(NeurIPS24)",
        "QUEEN-1\n(NeurIPS24)",
        "3DGStream\n(CVPR24)",
        "StreamSTGS\n(AAAI26)",
        "S²GS-fast\n(Ours)",
        "S²GS-full\n(Ours)",
    ]

    # x：Training Time (sec) ，采用对数坐标
    training = np.array([41, 6.4, 6.25, 7.43, 8.14, 25, 2.26, 4.12])
    # y：PSNR(dB)
    psnr = np.array([31.76, 32.66, 31.85, 32.18, 31.69, 32.35, 32.41, 32.76])

    storage = np.array([ 0.49, 0.44, 0.63, 0.79, 7.8, 0.17, 0.12, 0.11])  # MB
    
    Rendering_FPS = np.array([92, 250, 372, 426, 245, 199, 484, 482])  # FPS
    # -----------------------
    # 样式与配色（自动生成，长度与 methods 对齐）
    # -----------------------
    plt.style.use("seaborn-whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"

    n = len(methods)
    # 颜色映射将基于 Rendering_FPS（连续色标），而非离散 colors 列表
    cmap = plt.get_cmap("viridis")

    # 气泡大小：与 storage (MB) 成比例，使用对数缩放以平衡数量级差异
    # 使用对数缩放能让 0.1MB 与 20MB 之间的差异既可见又不过分夸张
    s_log = np.log10(storage + 1e-6)
    s_norm = (s_log - s_log.min()) / (s_log.ptp() + 1e-8)
    min_size = 30
    max_size = 300
    sizes = min_size + s_norm * (max_size - min_size)

    # -----------------------
    # 绘图
    # -----------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # 使用 Rendering_FPS 为颜色值，添加 colormap（颜色条将显示 FPS）
    vmin, vmax = Rendering_FPS.min(), Rendering_FPS.max()
    sc = ax.scatter(training, psnr, s=sizes, c=Rendering_FPS, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.6, alpha=0.95, zorder=3)

    # # 注释每个点，先创建文本对象，稍后使用 adjustText 自动避让（若可用）
    # texts = []
    # for i, (m, x, y) in enumerate(zip(methods, training, psnr)):
    #     # 取消初始位置的乘性偏移：将文本放在点的坐标附近（紧贴点），
    #     # adjustText 会在需要时微调位置。若想完全禁止自动移动，可把 adjustText 的调用注释掉。
    #     ha = "left" if x < np.median(training) else "right"
    #     va = "center"
    #     # 把文本先放在点的位置（或略微垂直居中），以减少初始偏移造成的大移动
    #     txt = ax.text(x, y, m, fontsize=9, ha=ha, va=va)
    #     txt.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground="white"), patheffects.Normal()])
    #     texts.append(txt)

    # # # 高亮标注我们的方法（假设最后一项是主要的 'Ours' 变体）
    # # try:
    # #     ours_idx = methods.index("S²GS-full\n(Ours)")
    # # except ValueError:
    # #     ours_idx = n - 1
    # # ax.scatter([training[ours_idx]], [psnr[ours_idx]], s=sizes[ours_idx] * 1.4,
    # #            facecolors="none", edgecolors="red", linewidths=1.4, zorder=4)

    # # 尝试使用 adjustText 自动调整文本位置，避免文本与点或其他文本重叠
    # try:
    #     from adjustText import adjust_text

    #     # 调低 adjustText 的移动强度以避免文字被移动得过远
    #     adjust_text(
    #         texts,
    #         x=training,
    #         y=psnr,
    #         expand_text=(1.01, 1.05),
    #         expand_points=(1.05, 1.1),
    #         force_text=0.1,
    #         force_points=0.1,
    #         # 避让点（scatter artist）以保证文字不与圆形重叠
    #         add_objects=[sc],
    #         only_move={'points': 'y', 'text': 'xy'},
    #         autoalign=True,
    #         ax=ax,
    #     )
    # except Exception:
    #     # adjustText 是可选依赖；没有时保持原始位置并给出提示
    #     print("Tip: install 'adjustText' (pip install adjustText) to auto-avoid label overlaps.")

    # # 拟合一条平滑趋势线（在对数 x 上拟合）用于引导视线
    # coeff = np.polyfit(np.log10(training), psnr, deg=1)
    # xs = np.logspace(np.log10(training.min() * 0.9), np.log10(training.max() * 1.1), 200)
    # ys = np.polyval(coeff, np.log10(xs))
    # ax.plot(xs, ys, color="#ff4136", linewidth=1.2, linestyle="--", zorder=2)

    # -----------------------
    # 坐标轴与外观
    # -----------------------
    ax.set_xscale("log")
    xmin = training.min() * 0.8
    # 右端至少覆盖到 51，以保证 x 轴范围超过 50
    xmax = max(50.0, training.max() * 1.2)
    ax.set_xlim(xmin, xmax)

    # y 轴范围：留出少量边距
    ypad = 0.12
    ax.set_ylim(psnr.min() - ypad, psnr.max() + ypad)

    ax.set_xlabel("Per-frame Training (sec)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)

    # 对数刻度选择：以 1, 2, 5 的步长显示
    ticks = np.array([1e-1, 2e-1, 5e-1, 1, 2, 5, 10, 20, 50, 100])
    ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]
    if len(ticks) >= 3:
        ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())

    # 网格优化
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)

    plt.tight_layout()
    # Storage 图例：为每个 storage 值生成一个圆圈及其数值标签，
    # 按 storage 大小排序，颜色与主图中对应点（按 Rendering_FPS 通过 cmap 映射）完全一致
    legend_handles = []
    legend_labels = []
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    order = np.argsort(storage)  # 从小到大排序
    for idx in order:
        sz = sizes[idx]
        st = storage[idx]
        fps = Rendering_FPS[idx]
        col = cmap(norm(fps))
        # 使用 scatter 作为句柄，边框风格与主图中的圆完全一致
        h = ax.scatter(
            [], [], s=sz, c=[col], edgecolors="k", linewidths=0.6, alpha=0.95
        )
        legend_handles.append(h)
        legend_labels.append(f"{st:.2f}")

    ax.legend(
        legend_handles,
        legend_labels,
        title="Storage (MB)",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        labelspacing=0.9,
        handletextpad=1.0,
        handlelength=1.0,
        frameon=True,
        fontsize=9,
        title_fontsize=9,
        borderpad=0.6,
    )

    # 在主图右侧添加颜色条，表示 Rendering FPS
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Rendering Speed (FPS)', fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw publication-quality scatter comparing methods")
    parser.add_argument("--save", type=str, default=None, help="Path to save the figure (PNG/PDF)")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    args = parser.parse_args()

    make_plot(save_path=args.save, show=not args.no_show)
