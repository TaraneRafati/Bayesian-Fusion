import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.font_manager as fm


def configure_paper_font():
    preferred = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "Times"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)

    if chosen is None:
        print(
            "[WARN] 'Times New Roman' (or a metric-compatible substitute) was "
            "not found by matplotlib's font manager. Legends will fall back "
            "to the default sans-serif font. Install a Times New Roman .ttf "
            "(or msttcorefonts on Linux) and clear the matplotlib font cache "
            "(~/.cache/matplotlib or %USERPROFILE%\\.matplotlib) to fix this."
        )
        chosen = "serif"

    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    return chosen


PAPER_FONT = configure_paper_font()

PALETTE = {
    "precision": "#2a78d6",
    "recall": "#eb6834",
    "f1": "#1baf7a",
}

CAM_CMAP = "inferno"
POSTERIOR_CMAP = "viridis"
LLR_CMAP = "RdBu_r"
UNCERTAINTY_CMAP = "cividis"
LLR_CLIP = 20.0

MAP_WIDTH_PT = 216.0
MAP_HEIGHT_PT = 216.0
PT_PER_IN = 72.0

CBAR_HEIGHT_IN = MAP_HEIGHT_PT / PT_PER_IN
CBAR_WIDTH_IN = (MAP_WIDTH_PT / 3.0) / PT_PER_IN

BAR_WIDTH_FRAC = 0.30
BAR_LEFT_FRAC = 0.06
BAR_BOTTOM_FRAC = 0.025
BAR_HEIGHT_FRAC = 0.95


def generate_colorbar_legends(legends_dir):
    colorbars = [
        ("attention_colorbar.pdf", CAM_CMAP, 0.0, 1.0),
        ("posterior_colorbar.pdf", POSTERIOR_CMAP, 0.0, 1.0),
        ("uncertainty_colorbar.pdf", UNCERTAINTY_CMAP, 0.0, 1.0),
        ("llr_colorbar.pdf", LLR_CMAP, -LLR_CLIP, LLR_CLIP),
    ]
    for filename, cmap_name, vmin, vmax in colorbars:
        path = os.path.join(legends_dir, filename)

        fig = plt.figure(figsize=(CBAR_WIDTH_IN, CBAR_HEIGHT_IN))
        ax = fig.add_axes([BAR_LEFT_FRAC, BAR_BOTTOM_FRAC, BAR_WIDTH_FRAC, BAR_HEIGHT_FRAC])

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cb = mcolorbar.ColorbarBase(ax, cmap=matplotlib.colormaps[cmap_name], norm=norm, orientation="vertical")

        cb.ax.tick_params(labelsize=7, length=2.5, pad=2)
        for tick_label in cb.ax.get_yticklabels():
            tick_label.set_fontname(PAPER_FONT)

        fig.savefig(path, format="pdf")
        plt.close(fig)
        print(f"[INFO] Wrote {path}  ({CBAR_WIDTH_IN * PT_PER_IN:.2f} x {CBAR_HEIGHT_IN * PT_PER_IN:.2f} pt)")


def generate_category_key_legend(legends_dir):
    path = os.path.join(legends_dir, "category_key.pdf")
    entries = [
        ("Ground truth / True Positive", PALETTE["f1"]),
        ("Attention (CAM) mask / False Negative", PALETTE["recall"]),
        ("Posterior-refined mask / False Positive", PALETTE["precision"]),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 0.35 * len(entries) + 0.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(entries))
    ax.set_axis_off()
    for i, (label, hex_color) in enumerate(entries):
        y = len(entries) - i - 1
        ax.add_patch(plt.Rectangle((0.0, y + 0.15), 0.08, 0.7, color=hex_color))
        ax.text(0.12, y + 0.5, label, va="center", fontsize=9, fontname=PAPER_FONT)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[INFO] Wrote {path}")


if __name__ == "__main__":
    output_dir = "legends_out"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Font selected for legends: {PAPER_FONT}")
    if PAPER_FONT == "serif":
        print("[WARN] Generic fallback in use, not literally 'Times New Roman' — see warning above.")

    generate_colorbar_legends(output_dir)
    generate_category_key_legend(output_dir)
