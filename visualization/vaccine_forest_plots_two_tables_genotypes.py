#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forest plots of HLA allele effects — TWO TABLES (vacc + HLA genotypes), NO REGIONS
-----------------------------------------------------------------------------------
- Vaccines table: titers + vaccinated flags
- HLA table: genotype-style columns like 'HLA-A_1', 'HLA-A_2', 'HLA-B_1', ... with values 'HLA-A*01:01:01'
- We expand genotypes into binary carrier indicators per allele (at chosen resolution), merge, and compute effects.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_any_table(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    elif lower.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    elif lower.endswith(".csv"):
        return pd.read_csv(path, sep=sep or ",")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.readline()
        guess = "\t" if ("\t" in head and "," not in head) else ","
        return pd.read_csv(path, sep=sep or guess)


def normalize_antibody(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x >= 0)
    if method == "log1p":
        return np.log1p(x)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd == 0:
        return x * 0
    return (x - mu) / sd


def hedges_g_and_var(mean1, sd1, n1, mean0, sd0, n0) -> Tuple[Optional[float], Optional[float]]:
    if n1 < 2 or n0 < 2 or sd1 <= 0 or sd0 <= 0:
        return None, None
    sp2 = ((n1 - 1) * (sd1 ** 2) + (n0 - 1) * (sd0 ** 2)) / (n1 + n0 - 2)
    if sp2 <= 0:
        return None, None
    d = (mean1 - mean0) / math.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
    g = J * d
    var_g = (n1 + n0) / (n1 * n0) + (g**2) / (2.0 * (n1 + n0 - 2))
    return g, var_g


def p_from_z(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


# -----------------------------
# Vaccine config
# -----------------------------

VACCINES: Dict[str, Dict[str, str]] = {
    "measles":     {"title": "Measles",              "q_col": "measles_ME_ml",         "info_col": "measles_vaccine_info"},
    "rubella":     {"title": "Rubella",              "q_col": "rubella_ME_ml",         "info_col": "rubella_vaccine_info"},
    "diphtheria":  {"title": "Diphtheria",           "q_col": "diphtheria_ME_ml",      "info_col": "diphtheria_vaccine_info"},
    "HBV":         {"title": "HBV (anti-HBsAg)",     "q_col": "HBV_antiHBsAg_ME_ml",   "info_col": "HBV_vaccine_info"},
}


# -----------------------------
# HLA genotype -> carriers expansion
# -----------------------------

LOCUS_ORDER = ["HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"]

def hla_normalize_allele(a: str, resolution: int = 2) -> Optional[str]:
    """Normalize genotype string like 'HLA-A*01:01:01' -> 'HLA-A*01:01' (resolution=2).
       Returns None for missing ('-', NaN). Resolution can be 1,2,3 (fields)."""
    if pd.isna(a):
        return None
    a = str(a).strip()
    if a == "" or a == "-" or a.lower() == "nan":
        return None
    # Expect 'HLA-<LOCUS>*<xx>:<yy>[:<zz>...]'
    if "*" not in a:
        return None
    head, tail = a.split("*", 1)
    fields = tail.split(":")
    fields = [re.sub(r"[^0-9]", "", f) for f in fields]  # keep digits
    fields = [f for f in fields if f != ""]
    if len(fields) == 0:
        return None
    res_fields = fields[:max(1, min(resolution, len(fields)))]
    locus = head.replace(" ", "").upper()  # 'HLA-A'
    return f"{locus}*{':'.join(res_fields)}"


def detect_genotype_columns(df_hla: pd.DataFrame, loci: Iterable[str]) -> List[str]:
    """Return list of genotype columns like 'HLA-A_1','HLA-A_2' for chosen loci."""
    cols = []
    setcols = set(df_hla.columns)
    for locus in loci:
        for suffix in ("_1", "_2"):
            cand = f"{locus}{suffix}"
            if cand in setcols:
                cols.append(cand)
    return cols


def build_carrier_matrix(df_hla: pd.DataFrame, id_col: str, loci: List[str], resolution: int, min_carriers: int) -> pd.DataFrame:
    """Expand genotype columns into binary carrier columns per allele (by resolution)."""
    geno_cols = detect_genotype_columns(df_hla, loci)
    if not geno_cols:
        raise SystemExit(f"No genotype columns like 'HLA-A_1'/'HLA-A_2' found for loci {loci}.")

    # Normalize alleles per cell
    norm = df_hla[geno_cols].applymap(lambda x: hla_normalize_allele(x, resolution))
    # Gather unique alleles and frequencies
    vals = pd.unique(norm.values.ravel())
    alleles = [a for a in vals if isinstance(a, str)]
    # Frequency filter
    freq = pd.Series(0, index=alleles, dtype=int)
    for a in alleles:
        freq[a] = ((norm == a).any(axis=1)).sum()
    keep = [a for a in alleles if freq[a] >= min_carriers]
    if not keep:
        raise SystemExit("After filtering by min_carriers, no HLA alleles remained. Lower --min-carriers.")

    # Build carriers 0/1 matrix
    carriers = pd.DataFrame(index=df_hla.index)
    for a in keep:
        carriers[a] = ((norm == a).any(axis=1)).astype(int)

    carriers.insert(0, id_col, df_hla[id_col].values)
    return carriers


# -----------------------------
# Effects computation & plotting
# -----------------------------

def compute_effects_all(df: pd.DataFrame, allele_cols: List[str], norm: str, min_per_group: int) -> pd.DataFrame:
    rows = []
    for vac_key, meta in VACCINES.items():
        q_col, info_col, title = meta["q_col"], meta["info_col"], meta["title"]
        if (q_col not in df.columns) or (info_col not in df.columns):
            continue
        d = df.copy()
        d = d[(pd.to_numeric(d[info_col], errors="coerce") == 1)]
        x = pd.to_numeric(d[q_col], errors="coerce")
        d["_titer_norm"] = normalize_antibody(x, norm)

        for allele in allele_cols:
            grp1 = d.loc[d[allele] == 1, "_titer_norm"].dropna()
            grp0 = d.loc[d[allele] != 1, "_titer_norm"].dropna()
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
                "vaccine_key": vac_key, "vaccine": title, "allele": allele,
                "n_carrier": n1, "n_noncarrier": n0,
                "g": g, "se": se, "ci_low": g - 1.96*se, "ci_high": g + 1.96*se, "p": p,
            })
    return pd.DataFrame(rows)


def forest_plot_by_allele(df_sub: pd.DataFrame, title: str, out_png: str, max_labels: Optional[int] = None) -> None:
    if df_sub.empty:
        return
    df = df_sub.copy()
    df["abs_g"] = df["g"].abs()
    df = df.sort_values(["abs_g", "allele"], ascending=[False, True])
    if max_labels is not None and len(df) > max_labels:
        df = df.head(max_labels)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(df)))
    for i, (_, r) in enumerate(df.iterrows()):
        ax.hlines(y=i, xmin=r["ci_low"], xmax=r["ci_high"])
    ax.plot(df["g"].values, y, 'o')
    ax.axvline(0.0, linestyle='--')
    labels = [f"{allele} (n1={int(n1)}, n0={int(n0)})"
              for allele, n1, n0 in zip(df["allele"], df["n_carrier"], df["n_noncarrier"])]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Hedges g (carriers − non-carriers)"); ax.set_title(title)
    plt.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="HLA effects from TWO tables (vacc + HLA genotypes), no region split")
    ap.add_argument("--vacc", required=True, help="Vaccine table (CSV/TSV)")
    ap.add_argument("--hla",  required=True, help="HLA table (XLSX/XLS/CSV/TSV) with genotype columns like 'HLA-A_1'")
    ap.add_argument("--out",  required=True, help="Output directory")
    ap.add_argument("--sep", default="\\t", help="Separator for --vacc if CSV/TSV (default: '\\t')")
    ap.add_argument("--hla-sep", default=None, help="Separator for --hla if CSV/TSV")
    ap.add_argument("--vacc-id-col", required=True, help="Join key column in vaccine table (e.g., 'ZLIMS ID')")
    ap.add_argument("--hla-id-col", required=True, help="Join key column in HLA table (e.g., 'sample_id')")
    ap.add_argument("--loci", default="HLA-A,HLA-B,HLA-C,HLA-DRB1,HLA-DQB1,HLA-DPB1",
                    help="Comma-separated loci to use")
    ap.add_argument("--resolution", type=int, default=2, choices=[1,2,3],
                    help="Allele resolution (fields) to collapse to: 1, 2, or 3")
    ap.add_argument("--min-carriers", type=int, default=10, help="Min #carriers to include an allele as a feature")
    ap.add_argument("--norm", choices=["log1p","zscore"], default="log1p", help="Normalization for titers")
    ap.add_argument("--min-per-group", type=int, default=3, help="Min N per group (carriers / non-carriers) for effect")
    ap.add_argument("--max-labels", type=int, default=None, help="Cap number of alleles on forest plots")
    args = ap.parse_args()

    ensure_outdir(args.out)

    vacc = read_any_table(args.vacc, sep=args.sep)
    hla  = read_any_table(args.hla,  sep=args.hla_sep)
    vacc.columns = [str(c).strip() for c in vacc.columns]
    hla.columns  = [str(c).strip() for c in hla.columns]

    if args.vacc_id_col not in vacc.columns:
        raise SystemExit(f"--vacc-id-col '{args.vacc_id_col}' not in vaccine columns: {list(vacc.columns)[:10]} ...")
    if args.hla_id_col not in hla.columns:
        raise SystemExit(f"--hla-id-col '{args.hla_id_col}' not in HLA columns: {list(hla.columns)[:10]} ...")

    loci = [x.strip() for x in args.loci.split(",") if x.strip()]
    carriers = build_carrier_matrix(hla, id_col=args.hla_id_col, loci=loci,
                                    resolution=args.resolution, min_carriers=args.min_carriers)

    # Merge: rename keys to common name
    vacc_ren = vacc.rename(columns={args.vacc_id_col: "_JOIN_ID_"})
    carr_ren = carriers.rename(columns={args.hla_id_col: "_JOIN_ID_"}) if args.hla_id_col in carriers.columns else carriers
    if "_JOIN_ID_" not in carr_ren.columns:
        carr_ren = carriers.rename(columns={args.hla_id_col: "_JOIN_ID_"})

    df = pd.merge(vacc_ren, carr_ren, on="_JOIN_ID_", how="inner")

    # Allele columns are all carrier columns (exclude join & original columns)
    allele_cols = [c for c in df.columns if c.startswith("HLA-") and "*" in c]
    if not allele_cols:
        raise SystemExit("No allele carrier columns after expansion. Check loci, resolution, and min-carriers.")

    effects = compute_effects_all(df, allele_cols=allele_cols, norm=args.norm, min_per_group=args.min_per_group)
    if effects.empty:
        print("No effects computed. Try lowering --min-carriers or --min-per-group.")
        return

    # Save outputs
    index_csv = os.path.join(args.out, "effects_index.csv")
    effects.sort_values(["vaccine_key", "p", "allele"]).to_csv(index_csv, index=False)

    for vac_key, df_v in effects.groupby("vaccine_key"):
        out_csv = os.path.join(args.out, f"effects_{vac_key}.csv")
        df_v.sort_values(["p", "allele"]).to_csv(out_csv, index=False)
        title = f"{df_v['vaccine'].iloc[0]} — ALL SAMPLES — HLA allele effects"
        out_png = os.path.join(args.out, f"forest_{vac_key}.png")
        forest_plot_by_allele(df_v, title=title, out_png=out_png, max_labels=args.max_labels)

    print(f"Saved index CSV: {index_csv}")
    print(f"Wrote per-vaccine CSV/PNG files into: {args.out}")


if __name__ == "__main__":
    main()
