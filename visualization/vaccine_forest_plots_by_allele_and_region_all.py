#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vaccine forest plots by HLA alleles, split by regions
-----------------------------------------------------
- Reads a TSV/CSV with vaccination phenotype columns and HLA allele columns.
- Regions are encoded by one-hot flags: e.g., is_from_Irkutsk, is_from_Amur, is_from_NiNo, is_from_Kaliningrad.
- For each *region*, and for each *vaccine* (quantitative titer present and vaccinated = 1),
  compute effect size (Hedges' g) comparing HLA allele *carriers* vs *non-carriers* on normalized titer.
- Draw a forest plot (horizontal CIs) per (region, vaccine). Save PNG and CSV summaries.

Assumptions (based on user's schema):
    Regions: columns starting with 'is_from_'
    Vaccines:
        measles_ME_ml, measles_vaccine_info
        rubella_ME_ml, rubella_vaccine_info
        diphtheria_ME_ml, diphtheria_vaccine_info
        HBV_antiHBsAg_ME_ml, HBV_vaccine_info
    (NoAnswer flags are not used here; analysis is among vaccinated only.)

HLA allele columns:
    - Detected via --allele-regex (default: r'^(HLA|hla)[A-Z]*[\\*_]')
    - Values should be 0/1/2 or 0/1; we treat carriers as value > 0.

Usage:
    python vaccine_forest_plots_by_allele_and_region.py \
        --vacc /path/to/all_pheno_unrel.tsv \
        --out  /path/to/outdir \
        --sep '\\t' \
        --norm log1p \
        --min-per-group 3 \
        --allele-regex '^(HLA|hla)[A-Z]*[\\*_]'

Outputs:
    out/
      RegionNameA/
         effects_{VACCINE_KEY}.csv
         forest_{VACCINE_KEY}.png
      RegionNameB/
         ...
    And a combined index: out/effects_index.csv
"""
from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    """Hedges g and its large-sample variance for two independent groups."""
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
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def infer_region_from_flags(df: pd.DataFrame, region_flag_prefix: str) -> pd.Series:
    """Infer a single 'Region' label from one-hot region flags.
       If multiple region flags are 1 -> 'Multi'. If none -> 'Unknown'.
    """
    region_cols = [c for c in df.columns if str(c).startswith(region_flag_prefix)]
    if not region_cols:
        return pd.Series(["All"] * len(df), index=df.index)

    def _row_region(row) -> str:
        active = [c for c in region_cols if pd.to_numeric(row.get(c, 0), errors="coerce") == 1]
        if len(active) == 0:
            return "Unknown"
        if len(active) == 1:
            return active[0].replace(region_flag_prefix, "")
        return "Multi"

    return df.apply(_row_region, axis=1)


# -----------------------------
# Vaccines of interest
# -----------------------------

VACCINES: Dict[str, Dict[str, Optional[str]]] = {
    "measles": {
        "title": "Measles",
        "q_col": "measles_ME_ml",
        "info_col": "measles_vaccine_info",
    },
    "rubella": {
        "title": "Rubella",
        "q_col": "rubella_ME_ml",
        "info_col": "rubella_vaccine_info",
    },
    "diphtheria": {
        "title": "Diphtheria",
        "q_col": "diphtheria_ME_ml",
        "info_col": "diphtheria_vaccine_info",
    },
    "HBV": {
        "title": "HBV (anti-HBsAg)",
        "q_col": "HBV_antiHBsAg_ME_ml",
        "info_col": "HBV_vaccine_info",
    },
}


# -----------------------------
# Core computation
# -----------------------------

def find_allele_columns(df: pd.DataFrame, allele_regex: str) -> List[str]:
    pat = re.compile(allele_regex)
    cols = [c for c in df.columns if pat.search(str(c))]
    return cols


def compute_allele_effects_by_region(
    df: pd.DataFrame,
    allele_cols: List[str],
    norm: str = "log1p",
    min_per_group: int = 3,
    region_flag_prefix: str = "is_from_",
) -> pd.DataFrame:
    """
    Returns a long table with columns:
        region, vaccine_key, vaccine, allele, n_carrier, n_noncarrier, g, se, ci_low, ci_high, p
    Computed among vaccinated (info_col == 1) with non-missing titers.
    Carrier is defined as allele value > 0.
    """
    # Create Region column
    df = df.copy()
    df["Region"] = infer_region_from_flags(df, region_flag_prefix)

    rows = []
    for vac_key, meta in VACCINES.items():
        q_col = meta["q_col"]
        info_col = meta["info_col"]
        if (q_col not in df.columns) or (info_col not in df.columns):
            continue

        sub_v = df[pd.to_numeric(df[info_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        sub_v["TiterRaw"] = pd.to_numeric(sub_v[q_col], errors="coerce")
        sub_v["TiterNorm"] = normalize_antibody(sub_v["TiterRaw"], norm)

        for region, gdf in sub_v.groupby("Region"):
            gdf = gdf.dropna(subset=["TiterNorm"])
            if gdf.empty:
                continue

            for allele in allele_cols:
                a = pd.to_numeric(gdf[allele], errors="coerce")
                # carriers (value > 0), non-carriers (==0)
                carriers = gdf.loc[a > 0, "TiterNorm"].astype(float)
                noncar = gdf.loc[(a == 0) | a.isna(), "TiterNorm"].astype(float)

                if len(carriers) < min_per_group or len(noncar) < min_per_group:
                    continue

                m1, s1, n1 = float(carriers.mean()), float(carriers.std(ddof=1)), int(len(carriers))
                m0, s0, n0 = float(noncar.mean()), float(noncar.std(ddof=1)), int(len(noncar))
                g, var_g = hedges_g_and_var(m1, s1, n1, m0, s0, n0)
                if g is None or var_g is None or var_g <= 0:
                    continue
                se = math.sqrt(var_g)
                z = g / se if se > 0 else 0.0
                p = p_from_z(z)

                rows.append({
                    "region": region,
                    "vaccine_key": vac_key,
                    "vaccine": meta["title"],
                    "allele": str(allele),
                    "n_carrier": n1,
                    "n_noncarrier": n0,
                    "g": g,
                    "se": se,
                    "ci_low": g - 1.96 * se,
                    "ci_high": g + 1.96 * se,
                    "p": p,
                })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------

def forest_plot_by_allele(df_sub: pd.DataFrame, title: str, out_png: str, max_labels: Optional[int] = None) -> None:
    """
    Draw a forest plot: each row is an allele with its CI and point estimate.
    Optionally limit number of rows with max_labels (e.g. top N by |g|).
    """
    if df_sub.empty:
        return

    df = df_sub.copy()
    # Order by effect magnitude descending
    df["abs_g"] = df["g"].abs()
    df = df.sort_values(["abs_g", "allele"], ascending=[False, True])
    if max_labels is not None and len(df) > max_labels:
        df = df.head(max_labels)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(df)))

    # CIs
    for i, (_, r) in enumerate(df.iterrows()):
        ax.hlines(y=i, xmin=r["ci_low"], xmax=r["ci_high"])
    # Points
    ax.plot(df["g"].values, y, 'o')

    ax.axvline(0.0, linestyle='--')
    labels = [f"{allele} (n1={int(n1)}, n0={int(n0)})"
              for allele, n1, n0 in zip(df["allele"], df["n_carrier"], df["n_noncarrier"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hedges g (carriers − non-carriers)")
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------


def compute_allele_effects_all(
    df: pd.DataFrame,
    allele_cols: List[str],
    norm: str = "log1p",
    min_per_group: int = 3,
) -> pd.DataFrame:
    """
    Like compute_allele_effects_by_region, but aggregates ALL samples (no regional split).
    Returns columns: region='ALL', vaccine_key, vaccine, allele, n_carrier, n_noncarrier, g, se, ci_low, ci_high, p
    """
    rows = []
    for vac_key, meta in VACCINES.items():
        q_col = meta["q_col"]
        info_col = meta["info_col"]
        if (q_col not in df.columns) or (info_col not in df.columns):
            continue

        d = df.copy()
        # vaccinated and with non-missing quantitative titer
        d = d[(pd.to_numeric(d[info_col], errors="coerce") == 1)]
        x = pd.to_numeric(d[q_col], errors="coerce")
        d["_titer_norm"] = normalize_antibody(x, norm)

        for allele in allele_cols:
            a = pd.to_numeric(d[allele], errors="coerce")
            grp1 = d.loc[a > 0, "_titer_norm"]
            grp0 = d.loc[~(a > 0), "_titer_norm"]

            grp1 = grp1.dropna()
            grp0 = grp0.dropna()

            n1, n0 = len(grp1), len(grp0)
            if n1 < min_per_group or n0 < min_per_group:
                continue

            g, var_g = hedges_g_and_var(grp1.mean(), grp1.std(ddof=1), n1,
                                        grp0.mean(), grp0.std(ddof=1), n0)
            if g is None or var_g is None:
                continue
            se = math.sqrt(var_g)
            z = g / se if se > 0 else 0.0
            p = p_from_z(z)

            rows.append({
                "region": "ALL",
                "vaccine_key": vac_key,
                "vaccine": meta["title"],
                "allele": str(allele),
                "n_carrier": n1,
                "n_noncarrier": n0,
                "g": g,
                "se": se,
                "ci_low": g - 1.96 * se,
                "ci_high": g + 1.96 * se,
                "p": p,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Compute HLA allele effect sizes by region and vaccine; draw forest plots.")
    ap.add_argument("--vacc", required=True, help="Path to TSV/CSV with vaccination + HLA data")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--sep", default="\\t", help=r"Field separator (default: '\\t')")
    ap.add_argument("--norm", choices=["log1p", "zscore"], default="log1p", help="Normalization for titers")
    ap.add_argument("--min-per-group", type=int, default=3, help="Minimal N in each group (carriers / non-carriers)")
    ap.add_argument("--region-flag-prefix", default="is_from_", help="Prefix to auto-detect region one-hot columns")
    ap.add_argument("--allele-regex", default=r"^(HLA|hla)[A-Z]*[\*_]", help="Regex to detect HLA allele columns")
    ap.add_argument("--max-labels", type=int, default=None, help="Optional: cap number of alleles per plot (top-|g|)")
    ap.add_argument("--all-samples", action="store_true",
                    help="Additionally produce plots/CSVs for ALL samples combined (no regional split).")
    args = ap.parse_args()

    ensure_outdir(args.out)

    # Read (engine='python' to support regex sep if provided)
    df = pd.read_csv(args.vacc, sep=args.sep, encoding="utf-8", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Discover allele columns
    allele_cols = find_allele_columns(df, args.allele_regex)

    if not allele_cols:
        print("No HLA allele columns were detected. Adjust --allele-regex.")
        return

    # Compute effects per region (original behavior)
    effects = compute_allele_effects_by_region(
        df=df,
        allele_cols=allele_cols,
        norm=args.norm,
        min_per_group=args.min_per_group,
        region_flag_prefix=args.region_flag_prefix,
    )

    # Save global per-region index (unchanged behavior)
    index_csv = os.path.join(args.out, "effects_index.csv")
    effects.sort_values(["region", "vaccine_key", "p", "allele"]).to_csv(index_csv, index=False)

    if effects.empty:
        print("No effects computed (check data, vaccinated flags, and min-per-group threshold).")
        # Note: we still may want ALL samples output; continue.

    # Make per-region directories; save per-(region, vaccine) CSV + figure
    for region, df_r in effects.groupby("region"):
        region_dir = os.path.join(args.out, region.replace("/", "_"))
        ensure_outdir(region_dir)

        for vac_key, df_rv in df_r.groupby("vaccine_key"):
            # CSV
            out_csv = os.path.join(region_dir, f"effects_{vac_key}.csv")
            sort_cols = [c for c in ["p", "allele"] if c in df_rv.columns]
            if not sort_cols:
                sort_cols = df_rv.columns.tolist()
            df_rv.sort_values(sort_cols).to_csv(out_csv, index=False)

            # Forest plot
            title = f"{df_rv['vaccine'].iloc[0]} — {region} — HLA allele effects"
            out_png = os.path.join(region_dir, f"forest_{vac_key}.png")
            forest_plot_by_allele(df_rv, title=title, out_png=out_png, max_labels=args.max_labels)

    # NEW: ALL samples combined (optional; does not change originals)
    if args.all_samples:
        all_dir = os.path.join(args.out, "ALL_SAMPLES")
        ensure_outdir(all_dir)
        eff_all = compute_allele_effects_all(df=df, allele_cols=allele_cols, norm=args.norm, min_per_group=args.min_per_group)
        if eff_all.empty:
            print("ALL_SAMPLES: no effects computed (check data & thresholds).")
        else:
            # Write per-vaccine CSV + plot
            for vac_key, df_v in eff_all.groupby("vaccine_key"):
                out_csv = os.path.join(all_dir, f"effects_{vac_key}.csv")
                sort_cols = [c for c in ["p", "allele"] if c in df_v.columns]
                if not sort_cols:
                    sort_cols = df_v.columns.tolist()
                df_v.sort_values(sort_cols).to_csv(out_csv, index=False)

                title = f"{df_v['vaccine'].iloc[0]} — ALL SAMPLES — HLA allele effects"
                out_png = os.path.join(all_dir, f"forest_{vac_key}.png")
                forest_plot_by_allele(df_v, title=title, out_png=out_png, max_labels=args.max_labels)

            # local index for ALL_SAMPLES
            eff_all.sort_values(["vaccine_key", "p", "allele"]).to_csv(os.path.join(all_dir, "effects_index.csv"), index=False)

    print(f"Saved index CSV: {index_csv}")
    print(f"Created {len(effects['region'].unique()) if not effects.empty else 0} region folders in: {args.out}")


if __name__ == "__main__":
    main()
