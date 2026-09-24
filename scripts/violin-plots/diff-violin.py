import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plot_style as ps

ps.apply()
DIFF_COLUMN = "dice_diff"
LABEL = "Dice"
SAVE_PLOT_PATH = "fig_dice_diff_violin.pdf"

DIRECTION_PALETTE = {
    "Decrease (fusion hurt)": ps.COLOR_DECREASE,
    "Increase (fusion helped)": ps.COLOR_INCREASE,
}


def build_violin_dataframe(df, diff_column):
    diffs = df[diff_column].astype(float)

    decreased = diffs[diffs > 0]
    increased = diffs[diffs < 0].abs()
    unchanged_n = int((diffs == 0).sum())

    records = []
    records += [{"magnitude": v, "direction": "Decrease (fusion hurt)"} for v in decreased]
    records += [{"magnitude": v, "direction": "Increase (fusion helped)"} for v in increased]

    plot_df = pd.DataFrame(records)
    plot_df["group"] = "All slices"

    return plot_df, len(decreased), len(increased), unchanged_n


def plot_split_violin(plot_df, n_decrease, n_increase, n_unchanged, label, save_plot_path):
    if plot_df.empty:
        print("[WARN] No non-zero differences found — nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sns.violinplot(
        data=plot_df,
        x="group",
        y="magnitude",
        hue="direction",
        split=True,
        inner="quartile",
        palette=DIRECTION_PALETTE,
        cut=0,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(f"|{label} change|")
    ax.set_title(
        f"{label} Change Distribution: Decrease vs. Increase\n"
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


def print_summary(plot_df, n_decrease, n_increase, n_unchanged, label):
    print("\n" + "=" * 50)
    print(f"[{label.upper()} CHANGE SUMMARY]")
    print("=" * 50)
    print(f"Slices with decrease (fusion hurt):   {n_decrease}")
    print(f"Slices with increase (fusion helped): {n_increase}")
    print(f"Slices unchanged:                     {n_unchanged}")

    for direction in ["Decrease (fusion hurt)", "Increase (fusion helped)"]:
        vals = plot_df.loc[plot_df["direction"] == direction, "magnitude"]
        if len(vals) == 0:
            continue
        print(f"\n{direction}:")
        print(f"  mean={vals.mean():.4f}  median={vals.median():.4f}  "
              f"std={vals.std():.4f}  max={vals.max():.4f}")
    print("=" * 50)


def run(csv_path, diff_column=DIFF_COLUMN, label=LABEL,
        save_plot_path=SAVE_PLOT_PATH):

    df = pd.read_csv(csv_path)
    if diff_column not in df.columns:
        raise ValueError(f"Column '{diff_column}' not found in {csv_path}. "
                          f"Available columns: {list(df.columns)}")

    plot_df, n_decrease, n_increase, n_unchanged = build_violin_dataframe(df, diff_column)
    print_summary(plot_df, n_decrease, n_increase, n_unchanged, label)
    plot_split_violin(plot_df, n_decrease, n_increase, n_unchanged, label, save_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default=SAVE_PLOT_PATH)
    cli = parser.parse_args()
    run(csv_path=cli.csv, save_plot_path=cli.output)
