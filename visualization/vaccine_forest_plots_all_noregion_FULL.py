#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vaccine forest plots by HLA alleles — ALL SAMPLES (no region split)
-------------------------------------------------------------------
Эта версия основана на вашем исходном скрипте (две таблицы), но полностью
убирает деление по регионам `is_from_*`. Все эффекты считаются по всей
выборке целиком. Интерфейс CLI сохранён.

Вывод в директорию --out:
  effects_index.csv
  effects_<vaccine>.csv
  forest_<vaccine>.png
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
    """В этой версии всегда одна группа ALL (никакого сплита по is_from_*)."""
    return pd.Series(["ALL"] * len(df), index=df.index)


# -----------------------------
# Config (vaccines)
# -----------------------------

VACCINES: Dict[str, Dict[str, str]] = {
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


def compute_allele_effects_all(
    df: pd.DataFrame,
    allele_cols: List[str],
    norm: str = "log1p",
    min_per_group: int = 3,
) -> pd.DataFrame:
    """
    Возвращает таблицу:
        vaccine_key, vaccine, allele, n_carrier, n_noncarrier, g, se, ci_low, ci_high, p
    Считает эффекты по ВСЕЙ выборке (без регионов), только среди вакцинированных (info_col == 1).
    Носитель: allele value > 0.
    """
    rows = []
    for vac_key, meta in VACCINES.items():
        q_col = meta["q_col"]
        info_col = meta["info_col"]
        if (q_col not in df.columns) or (info_col not in df.columns):
            continue

        d = df.copy()
        # Только вакцинированные и с не-missing количественным титром
        d = d[(pd.to_numeric(d[info_col], errors="coerce") == 1)]
        x = pd.to_numeric(d[q_col], errors="coerce")
        d["_titer_norm"] = normalize_antibody(x, norm)

        for allele in allele_cols:
            a = pd.to_numeric(d[allele], errors="coerce")
            grp1 = d.loc[a > 0, "_titer_norm"].dropna()
            grp0 = d.loc[~(a > 0), "_titer_norm"].dropna()

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

def main():
    ap = argparse.ArgumentParser(description="Compute HLA allele effect sizes for ALL samples (no region split).")
    ap.add_argument("--vacc", required=True, help="Path to TSV/CSV with vaccination + HLA data")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--sep", default="\\t", help=r"Field separator (default: '\t')")
    ap.add_argument("--norm", choices=["log1p", "zscore"], default="log1p", help="Normalization for titers")
    ap.add_argument("--min-per-group", type=int, default=3, help="Minimal N in each group (carriers / non-carriers)")
    ap.add_argument("--region-flag-prefix", default="is_from_", help="(ignored here)")
    ap.add_argument("--allele-regex", default=r"^(HLA|hla)[A-Z]*[\*_]", help="Regex to detect HLA allele columns")
    ap.add_argument("--max-labels", type=int, default=None, help="Optional: cap number of alleles per plot (top-|g|)")
    args = ap.parse_args()

    ensure_outdir(args.out)

    # Read (engine='python' to support regex sep if provided)
    df = pd.read_csv(args.vacc, sep=args.sep, encoding="utf-8", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Detect HLA allele columns
    allele_cols = find_allele_columns(df, args.allele_regex)
    if not allele_cols:
        print("No HLA allele columns were detected. Adjust --allele-regex.")
        return

    # Compute effects across ALL samples
    effects = compute_allele_effects_all(
        df=df,
        allele_cols=allele_cols,
        norm=args.norm,
        min_per_group=args.min_per_group,
    )

    if effects.empty:
        print("No effects computed (check data, vaccinated flags, and min-per-group threshold).")
        return

    # Global index
    index_csv = os.path.join(args.out, "effects_index.csv")
    sort_cols_idx = [c for c in ["vaccine_key", "p", "allele"] if c in effects.columns]
    if not sort_cols_idx:
        sort_cols_idx = effects.columns.tolist()
    effects.sort_values(sort_cols_idx).to_csv(index_csv, index=False)

    # Per-vaccine CSV + forest plot
    for vac_key, df_v in effects.groupby("vaccine_key"):
        out_csv = os.path.join(args.out, f"effects_{vac_key}.csv")
        sort_cols = [c for c in ["p", "allele"] if c in df_v.columns]
        if not sort_cols:
            sort_cols = df_v.columns.tolist()
        df_v.sort_values(sort_cols).to_csv(out_csv, index=False)

        title = f"{df_v['vaccine'].iloc[0]} — ALL SAMPLES — HLA allele effects"
        out_png = os.path.join(args.out, f"forest_{vac_key}.png")
        forest_plot_by_allele(df_v, title=title, out_png=out_png, max_labels=args.max_labels)

    print(f"Saved index CSV: {index_csv}")
    print(f"Wrote per-vaccine CSV/PNG files into: {args.out}")


if __name__ == "__main__":
    main()
