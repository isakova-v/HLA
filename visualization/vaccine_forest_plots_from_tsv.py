
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vaccine forest plots from TSV
--------------------------------
- Reads a TSV/CSV with vaccination phenotype columns
- Computes Hedges' g for each vaccine by region:
    Group = NoAnswer (1) vs Answer (0), among vaccinated only
    Metric = normalized quantitative titer (log1p or z-score)
- Draws per-vaccine forest plots (horizontal CIs) across regions
- Saves CSV summaries and PNG figures

Assumptions based on the user's schema:
Columns may include (examples):
    ID:                 'ZLIMS ID' (string-like)
    Demography:         'age', 'sex' (optional)
    Regions (one-hot):  is_from_Irkutsk, is_from_Amur, is_from_NiNo, is_from_Kaliningrad
    Vaccines:
        measles_ME_ml, measles_NoAnswer_coef, measles_vaccine_info
        rubella_ME_ml, rubella_NoAnswer_coef, rubella_vaccine_info
        diphtheria_ME_ml, diphtheria_NoAnswer_coef, diphtheria_vaccine_info
        mumps_vaccine_info, mumps_NoAnswer_coef (no quantitative column given — skipped)
        HBV_antiHBsAg_ME_ml, HBV_NoAnswer_coef, HBV_vaccine_info

Usage:
    python vaccine_forest_plots_from_tsv.py \
        --vacc /path/to/all_pheno_unrel.tsv \
        --out  /path/to/outdir \
        --sep '\t' \
        --norm log1p \
        --min-per-group 3
"""

from __future__ import annotations
import argparse
import math
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def normalize_antibody(s: pd.Series, method: str) -> pd.Series:
    """Normalize raw titers.
    - 'log1p'  -> log1p(x), negatives -> NaN
    - 'zscore' -> (x - mean) / sd (global on the provided series)
    """
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x >= 0)  # negatives -> NaN
    if method == "log1p":
        return np.log1p(x)
    elif method == "zscore":
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=1)
        if pd.isna(sd) or sd == 0:
            return x * 0  # avoid division by zero
        return (x - mu) / sd
    return x


def hedges_g_and_var(mean1, sd1, n1, mean0, sd0, n0) -> Tuple[Optional[float], Optional[float]]:
    """Hedges g and variance (two-group)."""
    if n1 < 2 or n0 < 2 or sd1 <= 0 or sd0 <= 0:
        return None, None
    sp2 = ((n1 - 1) * (sd1 ** 2) + (n0 - 1) * (sd0 ** 2)) / (n1 + n0 - 2)
    if sp2 <= 0:
        return None, None
    d = (mean1 - mean0) / math.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)  # small-sample correction
    g = J * d
    var_g = (n1 + n0) / (n1 * n0) + (g ** 2) / (2.0 * (n1 + n0 - 2))
    return g, var_g


def p_from_z(z: float) -> float:
    """Two-sided p-value from Z (no SciPy)."""
    # 2 * (1 - Phi(|z|)), Phi via erf
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def first_present(d: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in d.columns}
    for cand in candidates:
        k = str(cand).strip().lower()
        if k in low:
            return low[k]
    return None


def infer_region_from_flags(row: pd.Series, flag_cols: List[str]) -> str:
    """Map one-hot columns to a single Region string. If multiple are 1, mark as 'Multi'."""
    active = [c for c in flag_cols if pd.to_numeric(row.get(c, 0), errors="coerce") == 1]
    if len(active) == 0:
        return "Unknown"
    if len(active) == 1:
        # prettier name without "is_from_"
        return active[0].replace("is_from_", "")
    return "Multi"


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Core computation
# -----------------------------

VACCINES: Dict[str, Dict[str, Optional[str]]] = {
    "measles": {
        "title": "Measles",
        "q_col": "measles_ME_ml",
        "noans_col": "measles_NoAnswer_coef",
        "info_col": "measles_vaccine_info",
    },
    "rubella": {
        "title": "Rubella",
        "q_col": "rubella_ME_ml",
        "noans_col": "rubella_NoAnswer_coef",
        "info_col": "rubella_vaccine_info",
    },
    "diphtheria": {
        "title": "Diphtheria",
        "q_col": "diphtheria_ME_ml",
        "noans_col": "diphtheria_NoAnswer_coef",
        "info_col": "diphtheria_vaccine_info",
    },
    # mumps has no quantitative titer column in user's schema -> skip
    "HBV": {
        "title": "HBV (anti-HBsAg)",
        "q_col": "HBV_antiHBsAg_ME_ml",
        "noans_col": "HBV_NoAnswer_coef",
        "info_col": "HBV_vaccine_info",
    },
}


def compute_effects_by_region(
    df: pd.DataFrame,
    region_cols: List[str],
    norm: str = "log1p",
    min_per_group: int = 3,
) -> pd.DataFrame:
    """
    Returns long table with columns:
        vaccine, region, n1, n0, g, g_se, g_ci_low, g_ci_high, p
    where group1 = NoAnswer=1, group0 = NoAnswer=0, within vaccinated only.
    """
    # region column
    region_cols = [c for c in region_cols if c in df.columns]
    if region_cols:
        df = df.copy()
        df["Region"] = df.apply(lambda r: infer_region_from_flags(r, region_cols), axis=1)
    else:
        df = df.copy()
        df["Region"] = "All"

    rows = []
    for key, meta in VACCINES.items():
        q_col = meta["q_col"]
        no_col = meta["noans_col"]
        info_col = meta["info_col"]
        if (q_col not in df.columns) or (no_col not in df.columns) or (info_col not in df.columns):
            continue

        # vaccinated only
        sub = df[pd.to_numeric(df[info_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        # quantitative titer
        sub["TiterRaw"] = pd.to_numeric(sub[q_col], errors="coerce")
        # drop negative -> NaN handled in normalize
        sub["TiterNorm"] = normalize_antibody(sub["TiterRaw"], norm)
        # group by NoAnswer (0/1)
        sub["NoAnswer"] = pd.to_numeric(sub[no_col], errors="coerce").fillna(0).astype(int)

        # Per region
        for reg, gdf in sub.groupby("Region"):
            gdf = gdf.dropna(subset=["TiterNorm"])
            vals1 = gdf.loc[gdf["NoAnswer"] == 1, "TiterNorm"].astype(float)
            vals0 = gdf.loc[gdf["NoAnswer"] == 0, "TiterNorm"].astype(float)
            if len(vals1) < min_per_group or len(vals0) < min_per_group:
                continue
            m1, s1, n1 = float(vals1.mean()), float(vals1.std(ddof=1)), int(len(vals1))
            m0, s0, n0 = float(vals0.mean()), float(vals0.std(ddof=1)), int(len(vals0))
            g, var_g = hedges_g_and_var(m1, s1, n1, m0, s0, n0)
            if g is None or var_g is None or var_g <= 0:
                continue
            se = math.sqrt(var_g)
            z = g / se if se > 0 else 0.0
            p = p_from_z(z)
            rows.append({
                "vaccine": meta["title"],
                "vaccine_key": key,
                "region": reg,
                "n1": n1, "n0": n0,
                "g": g,
                "g_se": se,
                "g_ci_low": g - 1.96 * se,
                "g_ci_high": g + 1.96 * se,
                "p": p,
            })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------

def forest_plot(df_vac: pd.DataFrame, title: str, out_png: str) -> None:
    """
    Simple forest plot: horizontal CIs per region with a point at g.
    """
    if df_vac.empty:
        return

    # Sort regions: place "Overall" last, others alphabetically
    df_vac = df_vac.copy()
    df_vac = df_vac.sort_values(["region"])

    y = np.arange(len(df_vac))
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(df_vac)))

    # CI lines
    for i, (_, r) in enumerate(df_vac.iterrows()):
        ax.hlines(y=i, xmin=r["g_ci_low"], xmax=r["g_ci_high"])
    # Points
    ax.plot(df_vac["g"].values, y, 'o')

    # Vertical zero line
    ax.axvline(0.0, linestyle='--')

    # y-ticks as region labels (with n1/n0)
    labels = [f"{reg} (n1={int(n1)}, n0={int(n0)})"
              for reg, n1, n0 in zip(df_vac["region"], df_vac["n1"], df_vac["n0"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Hedges g (NoAnswer − Answer)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute vaccine effect sizes by region and draw forest plots.")
    ap.add_argument("--vacc", required=True, help="Path to TSV/CSV with vaccination phenotypes")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--sep", default="\t", help=r"Field separator (default: '\\t')")
    ap.add_argument("--norm", choices=["log1p", "zscore"], default="log1p", help="Normalization for titers")
    ap.add_argument("--min-per-group", type=int, default=3, help="Minimal N in each group (NoAnswer=0/1) to include a region")
    ap.add_argument("--region-flag-prefix", default="is_from_", help="Prefix to auto-detect region one-hot columns")
    args = ap.parse_args()

    ensure_outdir(args.out)

    # Read
    # Use engine='python' for regex separators too
    df = pd.read_csv(args.vacc, sep=args.sep, encoding="utf-8", engine="python")

    # Clean up columns (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # Identify region columns
    region_cols = [c for c in df.columns if c.startswith(args.region_flag_prefix)]

    # Compute effects
    eff = compute_effects_by_region(
        df=df,
        region_cols=region_cols,
        norm=args.norm,
        min_per_group=args.min_per_group,
    )

    # Save CSV summary
    out_csv = os.path.join(args.out, "vaccine_effects_by_region.csv")
    eff_sorted = eff.sort_values(["vaccine", "p", "region"])
    eff_sorted.to_csv(out_csv, index=False)

    if eff_sorted.empty:
        print("No quantitative vaccines with sufficient data to compute effects.")
        return

    # One forest plot per vaccine
    for vac, sub in eff_sorted.groupby("vaccine"):
        out_png = os.path.join(args.out, f"forest_{vac.replace(' ', '_')}.png")
        forest_plot(sub, title=f"{vac} — effect by region", out_png=out_png)

    print(f"Saved: {out_csv}")
    print(f"Saved {len(list(sub for _, sub in eff_sorted.groupby('vaccine')))} figures to {args.out}")


if __name__ == "__main__":
    main()
