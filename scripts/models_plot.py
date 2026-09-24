import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "hemorica_metrics_plot.pdf")

MODEL_DATA = {
    "ResNet-18 (original)": {
        "source": "literature",
        "params_millions": 11.7,
        "Precision": (0.858, 0.0),
        "Recall": (0.807, 0.0),
        "F1 Score": (0.830, 0.0),
        "AUC": (0.959, 0.0),
    },
    "EfficientNetV2-Large (original)": {
        "source": "literature",
        "params_millions": 118.5,
        "Precision": (0.908, 0.0),
        "Recall": (0.846, 0.0),
        "F1 Score": (0.875, 0.0),
        "AUC": (0.976, 0.0),
    },
    "MobileViT-S (original)": {
        "source": "literature",
        "params_millions": 5.6,
        "Precision": (0.914, 0.0),
        "Recall": (0.867, 0.0),
        "F1 Score": (0.889, 0.0),
        "AUC": (0.976, 0.0),
    },
    "ResNet-18 (ours)": {
        "source": "ours",
        "params_millions": 11.69,
        "Precision": (0.9499, 0.0064),
        "Recall": (0.9048, 0.0064),
        "F1 Score": (0.9268, 0.0036),
        "AUC": (0.9877, 0.0017),
    },
    "DINOv2-S (ours)": {
        "source": "ours",
        "params_millions": 21.0,
        "Precision": (0.9486, 0.0151),
        "Recall": (0.9105, 0.0125),
        "F1 Score": (0.9290, 0.0033),
        "AUC": (0.9882, 0.0028),
    },
    "EfficientNetV2-B3 (ours)": {
        "source": "ours",
        "params_millions": 14.36,
        "Precision": (0.9588, 0.0028),
        "Recall": (0.9258, 0.0009),
        "F1 Score": (0.9420, 0.0010),
        "AUC": (0.9912, 0.0001),
    },
}

METRICS_TO_PLOT = [
    "Precision",
    "Recall",
    "F1 Score",
    "AUC",
]

SORT_BY_METRIC = "Precision"
SORT_ASCENDING = True

COLOR_PALETTE = {
    "Precision": "#2a78d6",
    "Recall": "#eb6834",
    "F1 Score": "#1baf7a",
    "AUC": "#e87ba4",
    "Parameters": "#4a3aa7",
    "baseline_bg": "#e6e5e0",
    "grid": "#c3c2b7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "separator": "#8a8a82",
}

MARKERS_BY_METRIC = {
    "Precision": "o",
    "Recall": "s",
    "F1 Score": "^",
    "AUC": "D",
}

FONT_PREFERENCE = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]

FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9.5
FONT_SIZE_GROUP_LABEL = 10.5

FIG_WIDTH_IN = 9.0
FIG_HEIGHT_IN = 5.4
FIG_DPI = 600

PARAMS_AXIS_STYLE = "line"
PARAMS_AXIS_LOG_SCALE = True
PARAMS_BAR_WIDTH = 0.55
PARAMS_BAR_ALPHA = 0.35
PARAMS_MARKER = "P"

ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_MARKERSIZE = 7

GROUP_ARROW_Y = 1.025
GROUP_LABEL_Y = 1.055
GROUP_ARROW_LEN = 1.15
GROUP_ARROW_GAP = 0.12


def resolve_font_family(preferred_fonts):
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available:
            return name
    return "serif"


def strip_provenance_suffix(name):
    return name.replace(" (original)", "").replace(" (ours)", "")


def sorted_models(model_data, sort_metric, ascending):
    names = list(model_data.keys())
    baseline_names = [n for n in names if model_data[n]["source"] != "ours"]
    ours_names = [n for n in names if model_data[n]["source"] == "ours"]

    def sort_group(group):
        values = [model_data[n][sort_metric][0] for n in group]
        order = np.argsort(values)
        if not ascending:
            order = order[::-1]
        return [group[i] for i in order]

    return sort_group(baseline_names) + sort_group(ours_names)


def build_plot():
    resolved_font = resolve_font_family(FONT_PREFERENCE)
    print(f"Requested font preference order: {FONT_PREFERENCE}", flush=True)
    print(f"Font actually used for rendering: {resolved_font}", flush=True)
    if resolved_font != "Times New Roman":
        print(
            "Times New Roman was not found on this system; using a "
            f"metric-compatible / fallback serif ('{resolved_font}') instead. "
            "Install Times New Roman and re-run to use it directly.",
            flush=True,
        )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [resolved_font]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = True

    model_order = sorted_models(MODEL_DATA, SORT_BY_METRIC, SORT_ASCENDING)
    print(
        f"Models sorted by {SORT_BY_METRIC} "
        f"({'ascending' if SORT_ASCENDING else 'descending'}), "
        f"baseline group first: {model_order}",
        flush=True,
    )

    n_models = len(model_order)
    x = np.arange(n_models)

    baseline_mask = [MODEL_DATA[name]["source"] != "ours" for name in model_order]
    n_baseline = sum(baseline_mask)
    n_ours = n_models - n_baseline
    if n_baseline == 0 or n_ours == 0:
        raise ValueError("Expected at least one baseline and one 'ours' model.")
    separator_x = n_baseline - 0.5

    display_labels = [strip_provenance_suffix(name) for name in model_order]

    fig, ax_metrics = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
    ax_params = ax_metrics.twinx()

    ax_metrics.axvspan(
        -0.5, separator_x,
        color=COLOR_PALETTE["baseline_bg"], zorder=0, linewidth=0,
    )

    params_values = [MODEL_DATA[name]["params_millions"] for name in model_order]

    if PARAMS_AXIS_STYLE == "bar":
        ax_params.bar(
            x,
            params_values,
            width=PARAMS_BAR_WIDTH,
            color=COLOR_PALETTE["Parameters"],
            alpha=PARAMS_BAR_ALPHA,
            zorder=1,
            label="Parameters (M)",
            edgecolor="none",
        )
    else:
        ax_params.plot(
            x,
            params_values,
            color=COLOR_PALETTE["Parameters"],
            marker=PARAMS_MARKER,
            markersize=ERRORBAR_MARKERSIZE,
            markerfacecolor=COLOR_PALETTE["Parameters"],
            markeredgecolor=COLOR_PALETTE["Parameters"],
            markeredgewidth=1.3,
            linewidth=ERRORBAR_LINEWIDTH,
            linestyle="-",
            zorder=2,
            label="Parameters (M)",
        )

    if PARAMS_AXIS_LOG_SCALE:
        ax_params.set_yscale("log")
        ax_params.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{v:g}"))
        ax_params.yaxis.set_minor_formatter(ticker.NullFormatter())

    for metric_name in METRICS_TO_PLOT:
        means = [MODEL_DATA[name][metric_name][0] for name in model_order]
        stds = [MODEL_DATA[name][metric_name][1] for name in model_order]
        face_colors = [
            COLOR_PALETTE[metric_name] if MODEL_DATA[name]["source"] == "ours" else "none"
            for name in model_order
        ]
        color = COLOR_PALETTE[metric_name]
        marker = MARKERS_BY_METRIC.get(metric_name, "o")

        ax_metrics.plot(
            x,
            means,
            color=color,
            linewidth=ERRORBAR_LINEWIDTH,
            zorder=2,
            label=metric_name,
        )
        for xi, yi, fc in zip(x, means, face_colors):
            ax_metrics.plot(
                xi,
                yi,
                marker=marker,
                markersize=ERRORBAR_MARKERSIZE,
                markerfacecolor=fc,
                markeredgecolor=color,
                markeredgewidth=1.3,
                linestyle="none",
                zorder=3,
            )

        ours_x = [xi for xi, name in zip(x, model_order) if MODEL_DATA[name]["source"] == "ours"]
        ours_means = [
            MODEL_DATA[name][metric_name][0]
            for name in model_order
            if MODEL_DATA[name]["source"] == "ours"
        ]
        ours_stds = [
            MODEL_DATA[name][metric_name][1]
            for name in model_order
            if MODEL_DATA[name]["source"] == "ours"
        ]
        ax_metrics.errorbar(
            ours_x,
            ours_means,
            yerr=ours_stds,
            fmt="none",
            ecolor=color,
            capsize=ERRORBAR_CAPSIZE,
            elinewidth=1.1,
            zorder=4,
        )

        print(
            f"Plotted metric '{metric_name}': "
            f"{list(zip(model_order, means, stds))}",
            flush=True,
        )

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(display_labels, fontsize=FONT_SIZE_TICK, rotation=12, ha="right")
    ax_metrics.set_xlim(-0.5, n_models - 0.5)

    ax_metrics.axvline(
        separator_x,
        color=COLOR_PALETTE["separator"],
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    xaxis_transform = ax_metrics.get_xaxis_transform()
    baseline_arrow_start = separator_x - GROUP_ARROW_GAP
    baseline_arrow_end = baseline_arrow_start - GROUP_ARROW_LEN
    ours_arrow_start = separator_x + GROUP_ARROW_GAP
    ours_arrow_end = ours_arrow_start + GROUP_ARROW_LEN

    ax_metrics.annotate(
        "",
        xy=(baseline_arrow_end, GROUP_ARROW_Y),
        xytext=(baseline_arrow_start, GROUP_ARROW_Y),
        xycoords=xaxis_transform,
        textcoords=xaxis_transform,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_PALETTE["text_secondary"], linewidth=1.2),
        annotation_clip=False,
    )
    ax_metrics.text(
        (baseline_arrow_start + baseline_arrow_end) / 2.0,
        GROUP_LABEL_Y,
        "Baseline",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE_GROUP_LABEL,
        color=COLOR_PALETTE["text_primary"],
        clip_on=False,
    )

    ax_metrics.annotate(
        "",
        xy=(ours_arrow_end, GROUP_ARROW_Y),
        xytext=(ours_arrow_start, GROUP_ARROW_Y),
        xycoords=xaxis_transform,
        textcoords=xaxis_transform,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_PALETTE["text_secondary"], linewidth=1.2),
        annotation_clip=False,
    )
    ax_metrics.text(
        (ours_arrow_start + ours_arrow_end) / 2.0,
        GROUP_LABEL_Y,
        "Ours",
        transform=xaxis_transform,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE_GROUP_LABEL,
        color=COLOR_PALETTE["text_primary"],
        clip_on=False,
    )

    ax_metrics.set_xlabel("Model", fontsize=FONT_SIZE_AXIS_LABEL, color=COLOR_PALETTE["text_primary"])
    ax_metrics.set_ylabel("Metric Value", fontsize=FONT_SIZE_AXIS_LABEL, color=COLOR_PALETTE["text_primary"])
    ax_params.set_ylabel(
        "Number of Parameters (Millions, log scale)" if PARAMS_AXIS_LOG_SCALE else "Number of Parameters (Millions)",
        fontsize=FONT_SIZE_AXIS_LABEL,
        color=COLOR_PALETTE["text_primary"],
    )

    metric_min = min(
        MODEL_DATA[name][m][0] - MODEL_DATA[name][m][1]
        for name in model_order
        for m in METRICS_TO_PLOT
    )
    metric_max = max(
        MODEL_DATA[name][m][0] + MODEL_DATA[name][m][1]
        for name in model_order
        for m in METRICS_TO_PLOT
    )
    margin = (metric_max - metric_min) * 0.15
    ax_metrics.set_ylim(metric_min - margin, min(1.0, metric_max + margin))

    if PARAMS_AXIS_LOG_SCALE:
        ax_params.set_ylim(min(params_values) * 0.6, max(params_values) * 1.8)
    else:
        param_margin = max(params_values) * 0.25
        ax_params.set_ylim(0, max(params_values) + param_margin)

    ax_metrics.tick_params(axis="both", labelsize=FONT_SIZE_TICK, colors=COLOR_PALETTE["text_primary"])
    ax_params.tick_params(axis="y", labelsize=FONT_SIZE_TICK, colors=COLOR_PALETTE["text_primary"])

    ax_metrics.grid(
        True,
        axis="y",
        color=COLOR_PALETTE["grid"],
        linewidth=0.6,
        linestyle="-",
        alpha=0.6,
        zorder=0.5,
    )
    ax_metrics.set_axisbelow(True)

    for spine in ["top"]:
        ax_metrics.spines[spine].set_visible(False)
        ax_params.spines[spine].set_visible(False)
    ax_metrics.spines["left"].set_color(COLOR_PALETTE["text_secondary"])
    ax_metrics.spines["bottom"].set_color(COLOR_PALETTE["text_secondary"])
    ax_params.spines["right"].set_color(COLOR_PALETTE["text_secondary"])

    handles_metrics, labels_metrics = ax_metrics.get_legend_handles_labels()
    handles_params, labels_params = ax_params.get_legend_handles_labels()

    all_handles = handles_metrics + handles_params
    all_labels = labels_metrics + labels_params

    legend = ax_metrics.legend(
        all_handles,
        all_labels,
        loc="lower right",
        ncol=len(all_labels),
        fontsize=FONT_SIZE_LEGEND,
        frameon=True,
        framealpha=0.92,
        edgecolor=COLOR_PALETTE["grid"],
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.set_zorder(5)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {OUTPUT_PDF}", flush=True)


if __name__ == "__main__":
    build_plot()
