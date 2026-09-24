import matplotlib as mpl

PALETTE = {
    "Recall": "#eb6834",
    "F1 Score": "#1baf7a",
    "grid": "#c3c2b7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
}

COLOR_DECREASE = PALETTE["Recall"]
COLOR_INCREASE = PALETTE["F1 Score"]

FONT_SERIF_STACK = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"]

BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 9
SUPTITLE_FONT_SIZE = 12

SAVE_DPI = 300


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF_STACK,
        "mathtext.fontset": "stix",
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": SUPTITLE_FONT_SIZE,
        "axes.edgecolor": PALETTE["text_secondary"],
        "axes.labelcolor": PALETTE["text_primary"],
        "text.color": PALETTE["text_primary"],
        "xtick.color": PALETTE["text_secondary"],
        "ytick.color": PALETTE["text_secondary"],
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "axes.grid": False,
        "savefig.dpi": SAVE_DPI,
        "figure.dpi": SAVE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def style_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, alpha=0.6, color=PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["text_secondary"])
    ax.title.set_color(PALETTE["text_primary"])
    ax.xaxis.label.set_color(PALETTE["text_primary"])
    ax.yaxis.label.set_color(PALETTE["text_primary"])


def save_pdf(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"[INFO] Saved PDF figure to {path}")
