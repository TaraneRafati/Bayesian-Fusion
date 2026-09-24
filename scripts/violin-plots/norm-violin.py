import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_style as ps

ps.apply()
DIFF_COLUMN = "dice_diff"
PATIENT_COLUMN = "patient_id"
SLICE_COLUMN = "slice_idx"
SAVE_PLOT_PATH = "fig_normalized_slice_violin.pdf"

DIRECTION_PALETTE = {
    "Decrease (fusion hurt)": ps.COLOR_DECREASE,
    "Increase (fusion helped)": ps.COLOR_INCREASE,
}


def add_normalized_slice_column(df, patient_col, slice_col):
    df = df.copy()
    df[slice_col] = pd.to_numeric(df[slice_col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=[slice_col])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"[WARN] Dropped {n_dropped} rows with non-numeric '{slice_col}'.")

    def _normalize(group):
        lo, hi = group.min(), group.max()
        if hi > lo:
            return (group - lo) / (hi - lo)
        return pd.Series(0.5, index=group.index)

    df["normalized_slice"] = df.groupby(patient_col)[slice_col].transform(_normalize)
    return df


def build_violin_dataframe(df, diff_column):
    decrease_mask = df[diff_column] > 0
    increase_mask = df[diff_column] < 0

    records = []
    records += [{"normalized_slice": v, "direction": "Decrease (fusion hurt)"}
                for v in df.loc[decrease_mask, "normalized_slice"]]
    records += [{"normalized_slice": v, "direction": "Increase (fusion helped)"}
                for v in df.loc[increase_mask, "normalized_slice"]]

    plot_df = pd.DataFrame(records)
    plot_df["group"] = "All slices"
    n_unchanged = int((df[diff_column] == 0).sum())
    return plot_df, int(decrease_mask.sum()), int(increase_mask.sum()), n_unchanged


def plot_split_violin(plot_df, n_decrease, n_increase, n_unchanged, save_plot_path):
    if plot_df.empty:
        print("[WARN] No non-zero differences found — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sns.violinplot(
        data=plot_df,
        x="group",
        y="normalized_slice",
        hue="direction",
        split=True,
        inner="quartile",
        palette=DIRECTION_PALETTE,
        cut=0,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Normalized slice position (0 = first, 1 = last)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Normalized Slice Position: Decrease vs. Increase\n"
        f"(n_decrease={n_decrease}, n_increase={n_increase}, n_unchanged={n_unchanged})"
    )
    ax.legend(title=None, loc="upper right", frameon=False)
    ps.style_axis(ax, grid_axis="y")
    plt.tight_layout()

    if save_plot_path:
        ps.save_pdf(fig, save_plot_path)
        plt.close(fig)
    else:
        plt.show()


def print_summary(plot_df, n_decrease, n_increase, n_unchanged):
    print("\n" + "=" * 50)
    print("[NORMALIZED SLICE POSITION SUMMARY]")
    print("=" * 50)
    print(f"Slices with decrease (fusion hurt):   {n_decrease}")
    print(f"Slices with increase (fusion helped): {n_increase}")
    print(f"Slices unchanged:                     {n_unchanged}")

    for direction in ["Decrease (fusion hurt)", "Increase (fusion helped)"]:
        vals = plot_df.loc[plot_df["direction"] == direction, "normalized_slice"]
        if len(vals) == 0:
            continue
        print(f"\n{direction}:")
        print(f"  mean={vals.mean():.4f}  median={vals.median():.4f}  std={vals.std():.4f}")
    print("=" * 50)


def run(csv_path, diff_column=DIFF_COLUMN, patient_col=PATIENT_COLUMN,
        slice_col=SLICE_COLUMN, save_plot_path=SAVE_PLOT_PATH):

    df = pd.read_csv(csv_path)
    for col in (diff_column, patient_col, slice_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}. "
                              f"Available columns: {list(df.columns)}")

    df = add_normalized_slice_column(df, patient_col, slice_col)
    plot_df, n_decrease, n_increase, n_unchanged = build_violin_dataframe(df, diff_column)
    print_summary(plot_df, n_decrease, n_increase, n_unchanged)
    plot_split_violin(plot_df, n_decrease, n_increase, n_unchanged, save_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default=SAVE_PLOT_PATH)
    cli = parser.parse_args()
    run(csv_path=cli.csv, save_plot_path=cli.output)
