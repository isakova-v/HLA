# --- plot_hla_pc_regression_figures.py ---
import os, math, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "hla_pc_reg_figs"
IN_DEFAULT = "hla_pc_reg_out/combined_coeffs.csv"  # <- change if needed

os.makedirs(OUT_DIR, exist_ok=True)

def infer_se_from_p(coef, p_two_sided):
    """
    Infer SE from coef & two-sided p via normal approx:
      z = |coef|/SE  => SE = |coef| / z,  where z = Phi^{-1}(1 - p/2).
    """
    try:
        p = float(p_two_sided); b = float(coef)
    except Exception:
        return float("nan")
    if not (0 < p < 1) or not math.isfinite(b) or b == 0:
        return float("nan")
    p = max(min(p, 1 - 1e-12), 1e-12)
    # z = norm.isf(p/2) = sqrt(2) * erfcinv(p)
    from math import erfcinv, sqrt
    z = sqrt(2.0) * erfcinv(p)
    if z <= 0 or not math.isfinite(z):
        return float("nan")
    return abs(b) / z

def make_plots(csv_path, out_dir=OUT_DIR, min_terms=2, max_labels=25):
    df = pd.read_csv(csv_path)
    need = {"Vaccine","Gene","Allele","coef","p_value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df["se_inferred"] = [infer_se_from_p(b, p) for b, p in zip(df["coef"], df["p_value"])]
    df["ci_low"]  = df["coef"] - 1.96 * df["se_inferred"]
    df["ci_high"] = df["coef"] + 1.96 * df["se_inferred"]
    df["sig"] = df["p_value"] < 0.05

    vaccines = sorted(df["Vaccine"].dropna().unique())
    for vacc in vaccines:
        sub_v = df[df["Vaccine"] == vacc].copy()
        v_dir = os.path.join(out_dir, str(vacc).replace(" ", "_"))
        os.makedirs(v_dir, exist_ok=True)

        # Overall per-vaccine (Top by p-value)
        overall = sub_v.sort_values(["p_value","Gene","Allele"])
        overall = overall.head(min(len(overall), 40))
        if not overall.empty:
            y = np.arange(len(overall))[::-1]
            labels = [f"{g}-{a}" for g,a in zip(overall["Gene"], overall["Allele"])]
            fig, ax = plt.subplots(figsize=(8, max(3, 0.3*len(overall)+1)))
            ax.hlines(y, overall["ci_low"], overall["ci_high"])
            ax.plot(overall["coef"], y, "o")
            ax.axvline(0, ls="--", lw=1)
            ax.set_yticks(y); ax.set_yticklabels(labels)
            ax.set_xlabel("Coefficient (β) — baseline: most frequent allele per gene")
            ax.set_title(f"{vacc} — Top coefficients (approx. 95% CI)")
            fig.tight_layout()
            fig.savefig(os.path.join(v_dir, f"{str(vacc).replace(' ','_')}_overall.png"), dpi=200)
            plt.close(fig)

        # Per-gene plots
        for g, gdf in sub_v.groupby("Gene"):
            gdf = gdf.sort_values("p_value")
            if gdf.shape[0] < min_terms:
                continue
            gdf = gdf.head(min(gdf.shape[0], max_labels))
            y = np.arange(len(gdf))[::-1]
            labels = list(gdf["Allele"])
            fig, ax = plt.subplots(figsize=(7, max(3, 0.3*len(gdf)+1)))
            ax.hlines(y, gdf["ci_low"], gdf["ci_high"])
            ax.plot(gdf["coef"], y, "o")
            ax.axvline(0, ls="--", lw=1)
            ax.set_yticks(y); ax.set_yticklabels(labels)
            ax.set_xlabel("Coefficient (β) — baseline: most frequent allele")
            ax.set_title(f"{vacc} — {g}")
            fig.tight_layout()
            fig.savefig(os.path.join(v_dir, f"{str(vacc).replace(' ','_')}_{g}_coeffs.png"), dpi=200)
            plt.close(fig)

if __name__ == "__main__":
    in_csv = IN_DEFAULT if os.path.exists(IN_DEFAULT) else None
    if in_csv is None:
        # try to locate automatically
        matches = glob.glob("**/combined_coeffs.csv", recursive=True)
        if matches:
            in_csv = matches[0]
    if in_csv is None:
        raise SystemExit("Could not find combined_coeffs.csv — set IN_DEFAULT to your file path.")
    os.makedirs(OUT_DIR, exist_ok=True)
    make_plots(in_csv, OUT_DIR)
    print(f"Saved figures under: {OUT_DIR}")