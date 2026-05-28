#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vaccine forest plots by HLA alleles — per-region AND all-regions
----------------------------------------------------------------
- Reads a TSV/CSV with vaccination phenotype columns and HLA allele columns.
- Regions are encoded by one-hot flags: e.g., is_from_Irkutsk, is_from_Amur, ...
- For each *group* (either each region, or ALL regions combined), and for each *vaccine*,
  compute effect size (Hedges' g) comparing HLA allele carriers vs non-carriers on normalized titer.
- Draw a forest plot per (group, vaccine). Save PNG and CSV summaries.

Light fixes vs the user's original:
- Added missing imports (e.g., os) and cleaned type hints.
- Safer inference of separator (CSV/TSV) and column detection.
- New: "--no-region" flag to also output "All regions" plots (no division by region).
"""

import os
import re
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def auto_read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    # try to guess by first line
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    sep = "\t" if ("\t" in head and "," not in head) else ","
    return pd.read_csv(path, sep=sep)


def normalize_antibody(s: pd.Series) -> pd.Series:
    """Robust z-score (median & MAD)."""
    x = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        std = np.nanstd(x)
        if std == 0 or np.isnan(std):
            return (x - np.nanmean(x))
        return (x - np.nanmean(x)) / std
    return (x - med) / (1.4826 * mad)


def hedges_g(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (g, ci_low, ci_high) using Hedges' g with 95% CI."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return (np.nan, np.nan, np.nan)

    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    # pooled std
    s_p = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if s_p == 0 or np.isnan(s_p):
        return (np.nan, np.nan, np.nan)

    d = (mx - my) / s_p  # Cohen's d
    # small sample correction J
    J = 1 - (3 / (4 * (nx + ny) - 9))
    g = J * d

    # SE for g (Hedges & Olkin approx.)
    se_g = np.sqrt((nx + ny) / (nx * ny) + (g**2) / (2 * (nx + ny - 2)))
    # 95% CI
    ci_low = g - 1.96 * se_g
    ci_high = g + 1.96 * se_g
    return (g, ci_low, ci_high)


def p_value_from_g(g: float, nx: int, ny: int) -> float:
    """Two-sided p-value from standardized effect (approx using z)."""
    if np.isnan(g) or nx < 2 or ny < 2:
        return np.nan
    se = np.sqrt(1/nx + 1/ny)
    if se == 0 or np.isnan(se):
        return np.nan
    z = g / se
    # normal approx
    from math import erf, sqrt
    # two-sided
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return p


def detect_regions(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("is_from_")]


def detect_allele_cols(df: pd.DataFrame) -> List[str]:
    # naive rule: HLA* or columns that look like allele names (e.g., HLA_A_01_01)
    candidates = [c for c in df.columns if re.search(r"(HLA|allele)", c, flags=re.I)]
    # Keep binary-ish columns (0/1/True/False) as carriers
    keep = []
    for c in candidates:
        vc = df[c].dropna().unique()
        if len(vc) == 0:
            continue
        # treat as binary if only {0,1} or booleans or small set
        if set(np.unique(df[c].dropna().astype(str))).issubset({"0", "1", "True", "False", "true", "false"}):
            keep.append(c)
        elif df[c].dropna().isin([0,1]).all():
            keep.append(c)
    return keep


def detect_vaccines(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Return mapping: key -> {"titer": titer_col, "flag": vaccinated_flag_col, "label": label}
    Heuristics:
      - titer columns contain e.g. 'measles', 'rubella', 'tetanus', etc.
      - vaccinated flag: same prefix + something like 'vaccine_info', 'vaccinated', 'got_*'
    """
    vaccines = {}
    patterns = ["measles", "rubella", "tetanus", "diphtheria", "pertussis", "hepb", "hepatitis", "varicella", "mumps"]
    for p in patterns:
        titer_candidates = [c for c in df.columns if p in c.lower() and re.search(r"(ME|titer|ab|igg|ml)", c, re.I)]
        if not titer_candidates:
            continue
        titer = titer_candidates[0]
        # find a flag
        flag_candidates = [c for c in df.columns if p in c.lower() and re.search(r"(vaccine|vaccinated|got|shot|immun)", c, re.I)]
        flag = flag_candidates[0] if flag_candidates else None
        key = p
        vaccines[key] = {"titer": titer, "flag": flag, "label": p.capitalize()}
    return vaccines


def forest_plot_by_allele(df_effects: pd.DataFrame, title: str, out_png: str, max_labels: int = 30) -> None:
    """df_effects: columns 'allele', 'g', 'ci_low', 'ci_high', 'p'"""
    d = df_effects.dropna(subset=["g"]).copy()
    if len(d) == 0:
        return
    # order by |g| desc then p asc
    d["abs_g"] = d["g"].abs()
    d = d.sort_values(["p", "abs_g"], ascending=[True, False])
    if max_labels is not None:
        d = d.head(max_labels)

    y = np.arange(len(d))
    fig = plt.figure(figsize=(8, max(4, 0.35 * len(d) + 2)))
    ax = plt.gca()

    ax.errorbar(d["g"], y, xerr=[d["g"] - d["ci_low"], d["ci_high"] - d["g"]], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(d["allele"])
    ax.set_xlabel("Hedges' g (carriers vs non-carriers)")
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def compute_effects(df: pd.DataFrame, allele_cols: List[str], vaccine: Dict[str, str], where_mask: np.ndarray) -> pd.DataFrame:
    """Compute per-allele Hedges' g for a given boolean mask (group)."""
    titer_col = vaccine["titer"]
    flag_col = vaccine["flag"]

    d = df.copy()
    # filter group
    d = d.loc[where_mask].copy()

    # require vaccinated==1 if flag present and titer not null
    if flag_col and flag_col in d.columns:
        d = d[(d[flag_col] == 1) | (d[flag_col] == True)]

    # normalize titer
    d["_normalized_titer"] = normalize_antibody(d[titer_col])

    rows = []
    for allele in allele_cols:
        carriers = d[d[allele] == 1]["_normalized_titer"].to_numpy(dtype=float)
        noncarriers = d[d[allele] != 1]["_normalized_titer"].to_numpy(dtype=float)
        g, lo, hi = hedges_g(carriers, noncarriers)
        p = p_value_from_g(g, len(carriers), len(noncarriers))
        rows.append({"allele": allele, "g": g, "ci_low": lo, "ci_high": hi, "p": p})
    out = pd.DataFrame(rows)
    out["vaccine"] = vaccine.get("label", vaccine.get("titer", "vaccine"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV/TSV path")
    ap.add_argument("--out", dest="out", required=True, help="Output directory")
    ap.add_argument("--max-labels", type=int, default=30, help="Max alleles to label on forest plot")
    ap.add_argument("--no-region", action="store_true", help="Also produce 'All regions' plots (no division by region)")
    args = ap.parse_args()

    ensure_outdir(args.out)
    df = auto_read_table(args.inp)

    region_cols = detect_regions(df)
    allele_cols = detect_allele_cols(df)
    vaccines = detect_vaccines(df)

    if not vaccines:
        raise SystemExit("No vaccine columns detected. Pass a file with titer & vaccination info columns.")

    # Index of outputs
    index_rows = []

    # 1) Per-region (if regions exist)
    if region_cols:
        for region in region_cols:
            mask = df[region] == 1
            if mask.sum() == 0:
                continue
            region_dir = os.path.join(args.out, region)
            ensure_outdir(region_dir)

            for key, vac in vaccines.items():
                eff = compute_effects(df, allele_cols, vac, where_mask=mask)
                eff["group"] = region
                eff["group_type"] = "region"

                csv_path = os.path.join(region_dir, f"effects_{key}.csv")
                eff.sort_values(["p", "allele"]).to_csv(csv_path, index=False)

                title = f"{eff['vaccine'].iloc[0]} — {region} — HLA allele effects"
                png_path = os.path.join(region_dir, f"forest_{key}.png")
                forest_plot_by_allele(eff, title=title, out_png=png_path, max_labels=args.max_labels)

                index_rows.append({"group_type": "region", "group": region, "vaccine": eff["vaccine"].iloc[0],
                                   "csv": csv_path, "png": png_path})

    # 2) All regions combined (no division)
    if args.no_region:
        group_dir = os.path.join(args.out, "ALL_REGIONS")
        ensure_outdir(group_dir)
        mask_all = np.ones(len(df), dtype=bool)
        for key, vac in vaccines.items():
            eff = compute_effects(df, allele_cols, vac, where_mask=mask_all)
            eff["group"] = "ALL_REGIONS"
            eff["group_type"] = "all"

            csv_path = os.path.join(group_dir, f"effects_{key}.csv")
            eff.sort_values(["p", "allele"]).to_csv(csv_path, index=False)

            title = f"{eff['vaccine'].iloc[0]} — ALL REGIONS — HLA allele effects"
            png_path = os.path.join(group_dir, f"forest_{key}.png")
            forest_plot_by_allele(eff, title=title, out_png=png_path, max_labels=args.max_labels)

            index_rows.append({"group_type": "all", "group": "ALL_REGIONS", "vaccine": eff["vaccine"].iloc[0],
                               "csv": csv_path, "png": png_path})

    # Write index
    if index_rows:
        idx = pd.DataFrame(index_rows)
        idx_path = os.path.join(args.out, "index.csv")
        idx.to_csv(idx_path, index=False)
        print(f"Saved index CSV: {idx_path}")
    else:
        print("Nothing produced: check that your file has region flags / vaccine columns / allele columns.")


if __name__ == "__main__":
    main()
