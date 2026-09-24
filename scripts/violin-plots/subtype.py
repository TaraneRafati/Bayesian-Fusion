import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_style as ps

ps.apply()
DIFF_COLUMN = "dice_diff"

SUBTYPE_COLUMNS = {
    "IVH": "IVH",
    "ICH": "ICH",
    "SDH": "SDH",
    "EDH": "EPH",
    "SAH": "SAH",
}

SAVE_PLOT_PATH = "fig_subtype_dice_violin.pdf"

DIRECTION_PALETTE = {
    "Decrease (fusion hurt)": ps.COLOR_DECREASE,
    "Increase (fusion helped)": ps.COLOR_INCREASE,
}


def _to_bool_series(s):
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes"])


def load_and_merge(dice_diff_csv_path, subtype_csv_paths):
    dice_df = pd.read_csv(dice_diff_csv_path)
    for col in ("patient_id", "slice_idx"):
        if col not in dice_df.columns:
            raise ValueError(f"'{col}' not found in {dice_diff_csv_path}. "
                              f"Available columns: {list(dice_df.columns)}")

    subtype_dfs = [pd.read_csv(p) for p in subtype_csv_paths]
    subtype_df = pd.concat(subtype_dfs, ignore_index=True)
    print(f"[INFO] subtype csv columns: {list(subtype_df.columns)}")

    for col in ("patient_name", "slice_number"):
        if col not in subtype_df.columns:
            raise ValueError(f"'{col}' not found in subtype CSV(s). "
                              f"Available columns: {list(subtype_df.columns)}")

    dice_df["patient_id"] = dice_df["patient_id"].astype(str).str.strip()
    dice_df["slice_idx"] = pd.to_numeric(dice_df["slice_idx"], errors="coerce")
    subtype_df["patient_name"] = subtype_df["patient_name"].astype(str).str.strip()
    subtype_df["slice_number"] = pd.to_numeric(subtype_df["slice_number"], errors="coerce")

    merged = dice_df.merge(
        subtype_df,
        left_on=["patient_id", "slice_idx"],
        right_on=["patient_name", "slice_number"],
        how="left",
        suffixes=("", "_subtype"),
    )

    probe_col = "ICH" if "ICH" in merged.columns else None
    n_matched = merged[probe_col].notna().sum() if probe_col else len(merged)
    print(f"[INFO] Successfully merged {n_matched} / {len(merged)} slices")
    if n_matched < len(merged):
        unmatched = merged[merged[probe_col].isna()] if probe_col else merged
        print(f"[WARN] {len(merged) - n_matched} slices had no matching subtype info. Sample:")
        print(unmatched[["patient_id", "slice_idx", "filename"]].head(5).to_string(index=False))

    return merged


def build_subtype_violin_dataframe(merged, diff_column, subtype_columns):
    records = []
    counts = {}

    for display_name, col in subtype_columns.items():
        if col not in merged.columns:
            print(f"[WARN] Column '{col}' (for subtype '{display_name}') not found — skipping this subtype.")
            counts[display_name] = (0, 0)
            continue

        is_positive = _to_bool_series(merged[col])
        subset = merged.loc[is_positive]

        decrease_mask = subset[diff_column] > 0
        increase_mask = subset[diff_column] < 0

        for v in subset.loc[decrease_mask, diff_column]:
            records.append({"subtype": display_name, "magnitude": v, "direction": "Decrease (fusion hurt)"})
        for v in subset.loc[increase_mask, diff_column].abs():
            records.append({"subtype": display_name, "magnitude": v, "direction": "Increase (fusion helped)"})

        counts[display_name] = (int(decrease_mask.sum()), int(increase_mask.sum()))

    plot_df = pd.DataFrame(records)
    return plot_df, counts


def plot_subtype_violins(plot_df, counts, subtype_order, diff_label, save_plot_path):
    if plot_df.empty:
        print("[WARN] No data to plot — check SUBTYPE_COLUMNS / FILENAME_COLUMN mapping.")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    sns.violinplot(
        data=plot_df,
        x="subtype",
        y="magnitude",
        hue="direction",
        split=True,
        inner="quartile",
        order=subtype_order,
        palette=DIRECTION_PALETTE,
        cut=0,
        ax=ax,
    )

    xtick_labels = []
    for name in subtype_order:
        n_dec, n_inc = counts.get(name, (0, 0))
        xtick_labels.append(f"{name}\n(n_dec={n_dec}, n_inc={n_inc})")
    ax.set_xticks(range(len(subtype_order)))
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel("")
    ax.set_ylabel(f"|{diff_label} change|")
    ax.set_title(f"{diff_label} Change by Hemorrhage Subtype: Decrease vs. Increase")
    ax.legend(title=None, loc="upper right", frameon=False)
    ps.style_axis(ax, grid_axis="y")
    plt.tight_layout()

    if save_plot_path:
        ps.save_pdf(fig, save_plot_path)
        plt.close(fig)
    else:
        plt.show()


def print_summary(counts):
    print("\n" + "=" * 55)
    print("[DICE CHANGE BY SUBTYPE — SUMMARY]")
    print("=" * 55)
    for name, (n_dec, n_inc) in counts.items():
        print(f"{name:6s}  decrease={n_dec:5d}   increase={n_inc:5d}")
    print("=" * 55)


def run(dice_diff_csv_path, subtype_csv_paths, diff_column=DIFF_COLUMN,
        subtype_columns=SUBTYPE_COLUMNS, save_plot_path=SAVE_PLOT_PATH):

    merged = load_and_merge(dice_diff_csv_path, subtype_csv_paths)
    plot_df, counts = build_subtype_violin_dataframe(merged, diff_column, subtype_columns)
    print_summary(counts)
    plot_subtype_violins(plot_df, counts, list(subtype_columns.keys()), "Dice", save_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dice_csv", required=True)
    parser.add_argument("--subtype_csv", nargs="+", required=True)
    parser.add_argument("--output", default=SAVE_PLOT_PATH)
    cli = parser.parse_args()
    run(dice_diff_csv_path=cli.dice_csv, subtype_csv_paths=cli.subtype_csv, save_plot_path=cli.output)
