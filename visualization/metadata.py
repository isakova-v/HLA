import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

# very light pastel colormaps
PASTEL_BLUE = LinearSegmentedColormap.from_list(
    "pastel_blue",
    ["#ffffff", "#eef6fb", "#dbeaf5", "#c7deef", "#b3d2e9"]
)

PASTEL_GREEN = LinearSegmentedColormap.from_list(
    "pastel_green",
    ["#ffffff", "#eef7f3", "#d7efe6", "#bfe7d9", "#a7dfcc"]
)
# --------------------------
# Light colors / colormaps
# --------------------------
LIGHT_BAR_1 = "#A8DADC"   # light teal
LIGHT_BAR_2 = "#F1A7C0"   # light pink
LIGHT_HIST  = "#BFD7EA"   # light blue

HEATMAP_CMAP_INPUT = "YlGnBu"   # light sequential
HEATMAP_CMAP_TOTAL = "YlGnBu"
HEATMAP_CMAP_FILT  = "PuBuGn"


def detect_vaccines(columns):
    """Vaccines are inferred from *_vaccine_info columns."""
    vinfo_cols = [c for c in columns if c.endswith("_vaccine_info")]
    prefixes = [c[:-len("_vaccine_info")] for c in vinfo_cols]
    return prefixes, vinfo_cols


def detect_regions(columns):
    """Regions/cities inferred from is_from_* columns."""
    return [c for c in columns if c.startswith("is_from_")]


def find_me_cols(prefix, columns):
    """
    Preference:
    1) exact '{prefix}_ME_ml'
    2) any '{prefix}_*ME_ml' (e.g., HBV_antiHBsAg_ME_ml)
    """
    exact = f"{prefix}_ME_ml"
    if exact in columns:
        return [exact]
    alt = [c for c in columns if c.startswith(prefix + "_") and c.endswith("ME_ml")]
    return alt


def find_noanswer_col(prefix, columns):
    c = f"{prefix}_NoAnswer_coef"
    return [c] if c in columns else []


def find_vinfo_col(prefix, columns):
    c = f"{prefix}_vaccine_info"
    return [c] if c in columns else []


def find_pc_cols(columns, n=20):
    pcs = [f"PC{i}" for i in range(1, n + 1)]
    return [c for c in pcs if c in columns]


# --------------------------
# Visualization helpers
# --------------------------
def plot_input_heatmap(input_summary: pd.DataFrame, vaccines, region_cols, plots_dir, show_plots):
    if input_summary.empty or not region_cols:
        print("[WARN] Skipping heatmap: empty input_summary or no regions detected.")
        return

    mat = input_summary.set_index("vaccine")[region_cols].reindex(vaccines).fillna(0).values

    plt.figure(figsize=(max(6, 0.8 * len(region_cols)), max(4, 0.5 * len(vaccines))))
    plt.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP_INPUT)
    plt.colorbar(label="N samples")
    plt.xticks(range(len(region_cols)), region_cols, rotation=45, ha="right")
    plt.yticks(range(len(vaccines)), vaccines)
    plt.title("Input data: samples per vaccine x region")
    plt.tight_layout()
    out = plots_dir / "input_vaccine_x_region_heatmap.png"
    plt.savefig(out, dpi=200)
    if show_plots:
        plt.show()
    plt.close()
    print(f"[OK] Saved plot: {out}")


def plot_filtered_bar(filtered_summary: pd.DataFrame, plots_dir, show_plots):
    if filtered_summary.empty:
        print("[WARN] Skipping filtered barplot: empty filtered_summary.")
        return

    plt.figure(figsize=(max(5, 0.8 * len(filtered_summary)), 4))
    plt.bar(filtered_summary["vaccine"], filtered_summary["filtered_N"], color=LIGHT_BAR_1)
    plt.ylabel("Filtered N")
    plt.title("Filtered data size per vaccine")
    plt.tight_layout()
    out = plots_dir / "filtered_sizes_bar.png"
    plt.savefig(out, dpi=200)
    if show_plots:
        plt.show()
    plt.close()
    print(f"[OK] Saved plot: {out}")


def plot_age_histograms(filtered_dfs: dict, plots_dir, show_plots, bins=20):
    # filtered_dfs: vaccine -> subset dataframe
    vaccines = list(filtered_dfs.keys())
    if not vaccines:
        return

    # only plot if age exists
    if all("age" not in df.columns for df in filtered_dfs.values()):
        print("[WARN] Skipping age histograms: no 'age' column.")
        return

    n = len(vaccines)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(5 * ncols, 3.5 * nrows))
    for i, v in enumerate(vaccines, 1):
        df = filtered_dfs[v]
        if "age" not in df.columns:
            continue
        plt.subplot(nrows, ncols, i)
        ages = df["age"].dropna()
        plt.hist(ages, bins=bins, color=LIGHT_HIST, edgecolor="white")
        plt.title(f"{v}: age distribution")
        plt.xlabel("age")
        plt.ylabel("count")

    plt.tight_layout()
    out = plots_dir / "filtered_age_histograms.png"
    plt.savefig(out, dpi=200)
    if show_plots:
        plt.show()
    plt.close()
    print(f"[OK] Saved plot: {out}")


def plot_sex_bars(filtered_dfs: dict, plots_dir, show_plots):
    vaccines = list(filtered_dfs.keys())
    if not vaccines:
        return

    if all("sex" not in df.columns for df in filtered_dfs.values()):
        print("[WARN] Skipping sex plots: no 'sex' column.")
        return

    sex0 = []
    sex1 = []
    for v in vaccines:
        s = filtered_dfs[v]["sex"]
        sex0.append(int((s == 0).sum()))
        sex1.append(int((s == 1).sum()))

    x = np.arange(len(vaccines))
    width = 0.4

    plt.figure(figsize=(max(6, 0.9 * len(vaccines)), 4))
    plt.bar(x - width/2, sex0, width, label="sex=0", color=LIGHT_BAR_1)
    plt.bar(x + width/2, sex1, width, label="sex=1", color=LIGHT_BAR_2)
    plt.xticks(x, vaccines, rotation=30, ha="right")
    plt.ylabel("count")
    plt.title("Filtered data: sex distribution per vaccine")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / "filtered_sex_distribution.png"
    plt.savefig(out, dpi=200)
    if show_plots:
        plt.show()
    plt.close()
    print(f"[OK] Saved plot: {out}")


def plot_filtered_vs_total(input_summary: pd.DataFrame, filtered_summary: pd.DataFrame, plots_dir, show_plots):
    """
    График: для каждой вакцины два столбца —
    total (из input_summary, тот же фильтр, что и для XLSX)
    и filtered_N (из filtered_summary),
    плюс подпись процента filtered от total.
    """
    if input_summary.empty or filtered_summary.empty:
        print("[WARN] Skipping filtered_vs_total plot: empty summaries.")
        return

    total_map = dict(zip(input_summary["vaccine"], input_summary["total_filtered_samples"]))
    filt_map = dict(zip(filtered_summary["vaccine"], filtered_summary["filtered_N"]))

    vaccines = sorted(set(total_map) | set(filt_map))
    total = np.array([total_map.get(v, 0) for v in vaccines])
    filt = np.array([filt_map.get(v, 0) for v in vaccines])

    x = np.arange(len(vaccines))
    width = 0.4

    plt.figure(figsize=(max(7, 1.0 * len(vaccines)), 4.5))
    plt.bar(x - width/2, total, width, label="total (summary filter)", color=LIGHT_BAR_1)
    plt.bar(x + width/2, filt,  width, label="filtered (XLSX)", color=LIGHT_BAR_2)

    for i, (t, f) in enumerate(zip(total, filt)):
        pct = 0 if t == 0 else 100 * f / t
        plt.text(i + width/2, f, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, vaccines, rotation=30, ha="right")
    plt.ylabel("N samples")
    plt.title("Filtered vs total samples per vaccine")
    plt.legend()
    plt.tight_layout()

    out = plots_dir / "filtered_vs_total.png"
    plt.savefig(out, dpi=200)
    if show_plots:
        plt.show()
    plt.close()
    print(f"[OK] Saved plot: {out}")


def plot_total_vs_filtered_by_region(df, filtered_dfs, vaccines, region_cols, plots_dir, show_plots):

    total_rows = []
    for v in vaccines:
        row = {"vaccine": v}
        for r in region_cols:
            row[r] = int((df[r] == 1).sum())
        total_rows.append(row)

    total_mat = pd.DataFrame(total_rows).set_index("vaccine").reindex(vaccines).fillna(0)

    filtered_rows = []
    for v in vaccines:
        sub = filtered_dfs.get(v)
        row = {"vaccine": v}
        for r in region_cols:
            row[r] = int((sub[r] == 1).sum())
        filtered_rows.append(row)

    filtered_mat = pd.DataFrame(filtered_rows).set_index("vaccine").reindex(vaccines).fillna(0)

    def annotate(ax, mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.patch.set_facecolor("white")

    # TOTAL – pastel blue
    im1 = axes[0].imshow(total_mat.values, cmap=PASTEL_BLUE)
    axes[0].set_title("TOTAL (raw data)")
    axes[0].set_xticks(range(len(region_cols)))
    axes[0].set_xticklabels(region_cols, rotation=45, ha="right")
    axes[0].set_yticks(range(len(vaccines)))
    axes[0].set_yticklabels(vaccines)
    annotate(axes[0], total_mat.values)
    fig.colorbar(im1, ax=axes[0], fraction=0.046)

    # FILTERED – pastel green
    im2 = axes[1].imshow(filtered_mat.values, cmap=PASTEL_GREEN)
    axes[1].set_title("FILTERED (after mask)")
    axes[1].set_xticks(range(len(region_cols)))
    axes[1].set_xticklabels(region_cols, rotation=45, ha="right")
    annotate(axes[1], filtered_mat.values)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.suptitle("Samples per vaccine × region (Total vs Filtered)", fontsize=14)
    plt.tight_layout()

    out = plots_dir / "total_vs_filtered_by_region.png"
    plt.savefig(out, dpi=200)
    if show_plots:
        plt.show()
    plt.close()

def main(
    input_path: str,
    out_dir: str = "./filtered_vaccines",
    summary_path: str = "./summary.xlsx",
    show_plots: bool = False,
):
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="\t")
    cols = df.columns.tolist()

    vaccines, vinfo_cols = detect_vaccines(cols)
    region_cols = detect_regions(cols)
    pc_cols = find_pc_cols(cols, n=20)

    vaccines = sorted(vaccines)

    if region_cols and "region" not in df.columns:
        region_flags = df[region_cols]
        has_region = (region_flags == 1).any(axis=1)
        region_name = region_flags.idxmax(axis=1)

        df.loc[has_region, "region"] = region_name[has_region].str.replace("is_from_", "", regex=False)
        df.loc[~has_region, "region"] = np.nan
        cols = df.columns.tolist()

    base_cols = [c for c in ["ZLIMS ID", "age", "sex", "region"] if c in cols]

    input_rows = []
    for v in vaccines:
        vinfo = f"{v}_vaccine_info"
        if vinfo not in cols:
            continue

        mask = df[vinfo].isin([1, -1])

        if region_cols:
            has_region = (df[region_cols] == 1).any(axis=1)
            mask &= has_region
        elif "region" in df.columns:
            mask &= df["region"].notna()

        total_v = int(mask.sum())
        row = {"vaccine": v, "total_filtered_samples": total_v}

        for r in region_cols:
            row[r] = int((mask & (df[r] == 1)).sum())

        input_rows.append(row)

    input_summary = pd.DataFrame(input_rows).sort_values("vaccine")

    print("\n=== DISTRIBUTION OF vaccine_info FOR EACH VACCINE ===")
    vaccine_info_stats = []

    for v in vaccines:
        vinfo_col = f"{v}_vaccine_info"
        if vinfo_col not in df.columns:
            print(f"[WARN] {vinfo_col} not found")
            continue

        s = df[vinfo_col]
        counts = s.value_counts(dropna=False).sort_index()

        stats_row = {"vaccine": v}
        for key, val in counts.items():
            if pd.isna(key):
                stats_row["NaN"] = int(val)
            else:
                stats_row[str(int(key))] = int(val)

        vaccine_info_stats.append(stats_row)

    vaccine_info_summary = pd.DataFrame(vaccine_info_stats).fillna(0)
    print(vaccine_info_summary.to_string(index=False))

    filtered_stats_rows = []
    filtered_dfs = {}

    for v in vaccines:
        vinfo_col = f"{v}_vaccine_info"
        if vinfo_col not in cols:
            print(f"[WARN] Missing {vinfo_col}, skipping vaccine {v}")
            continue

        mask = df[vinfo_col].isin([1, -1])

        if region_cols:
            has_region = (df[region_cols] == 1).any(axis=1)
            mask &= has_region
        elif "region" in df.columns:
            mask &= df["region"].notna()

        sub = df[mask].copy()

        me_cols = find_me_cols(v, cols)
        noanswer_cols = find_noanswer_col(v, cols)
        vinfo_cols_ = find_vinfo_col(v, cols)

        keep_cols = base_cols + me_cols + noanswer_cols + vinfo_cols_ + pc_cols
        keep_cols = [c for c in keep_cols if c in cols]

        sub_out = sub[keep_cols]

        out_file = out_dir / f"{v}.xlsx"
        sub_out.to_excel(out_file, index=False)
        print(f"[OK] Saved {out_file} with {len(sub_out)} rows")

        filtered_dfs[v] = sub

        stat = {"vaccine": v, "filtered_N": len(sub_out)}

        for r in region_cols:
            stat[r] = int((sub[r] == 1).sum())

        if "age" in sub.columns:
            stat["age_mean"] = float(sub["age"].mean())
            stat["age_std"] = float(sub["age"].std())

        if "sex" in sub.columns:
            stat["sex_0_count"] = int((sub["sex"] == 0).sum())
            stat["sex_1_count"] = int((sub["sex"] == 1).sum())

        num_cols = me_cols + noanswer_cols + pc_cols
        for c in num_cols:
            if c in sub.columns:
                stat[f"missing_{c}"] = int(sub[c].isna().sum())

        filtered_stats_rows.append(stat)

    filtered_summary = pd.DataFrame(filtered_stats_rows).sort_values("vaccine")

    with pd.ExcelWriter(summary_path) as writer:
        input_summary.to_excel(writer, sheet_name="input_counts_by_region", index=False)
        filtered_summary.to_excel(writer, sheet_name="filtered_stats", index=False)

    print(f"\n[OK] Summary saved to {summary_path}")

    print("\n=== INPUT DATA SUMMARY (vaccine x region counts) ===")
    print(input_summary.to_string(index=False))

    print("\n=== FILTERED DATA SUMMARY ===")
    print(filtered_summary.to_string(index=False))

    plot_input_heatmap(input_summary, vaccines, region_cols, plots_dir, show_plots)
    plot_filtered_bar(filtered_summary, plots_dir, show_plots)
    plot_age_histograms(filtered_dfs, plots_dir, show_plots, bins=20)
    plot_sex_bars(filtered_dfs, plots_dir, show_plots)
    plot_filtered_vs_total(input_summary, filtered_summary, plots_dir, show_plots)
    plot_total_vs_filtered_by_region(df, filtered_dfs, vaccines, region_cols, plots_dir, show_plots)


if __name__ == "__main__":
    main("all_pheno_unrel.tsv", show_plots=False)