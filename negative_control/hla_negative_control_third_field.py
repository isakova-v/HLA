#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Negative-control HLA effect-size analysis for vaccine antibody titers.

Idea:
- Original analysis worked at second-field resolution, e.g. HLA-A*02:01.
- This script keeps the SAME second-field allele fixed, but compares DIFFERENT
  third-field variants inside it, e.g. HLA-A*02:01:01 vs other HLA-A*02:01:* carriers.

For each vaccine file:
1. find exact third-field alleles,
2. group them by second-field parent,
3. for every third-field allele with enough carriers, compute Hedges' g between
   carriers of this exact third-field variant and carriers of the same
   second-field allele but with another third-field variant,
4. draw per-vaccine barplots.

This is intended as a negative control: if third-field differences mostly do not
change the phenotype, the effects should be centered near zero.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Normalization helpers
# ============================================================

def normalize_allele_second_field(allele: str) -> str | None:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None
    if "*" not in allele:
        return None
    prefix, rest = allele.split("*", 1)
    parts = rest.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return f"{prefix}*{parts[0]}:{parts[1]}"


def normalize_allele_third_field(allele: str) -> str | None:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None
    if "*" not in allele:
        return None
    prefix, rest = allele.split("*", 1)
    parts = rest.split(":")
    if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
        return None
    return f"{prefix}*{parts[0]}:{parts[1]}:{parts[2]}"


def get_hla_genes_from_columns(df: pd.DataFrame) -> List[str]:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c.startswith("HLA-")})


# ============================================================
# 2. Input handling
# ============================================================

def detect_titer_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.endswith("_ME_ml")]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one titer column ending with '_ME_ml', got: {candidates}"
        )
    return candidates[0]


def load_vaccine_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path).copy()

    required_any = ["sample_id"]
    for col in required_any:
        if col not in df.columns:
            raise ValueError(f"{path} must contain '{col}'.")

    tcol = detect_titer_column(df)
    df["sample_id"] = df["sample_id"].astype(str)
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[tcol] > 0].copy()
    df["log_titer"] = np.log10(df[tcol])

    # one row per sample
    df = df.drop_duplicates(subset=["sample_id"]).copy()
    return df


# ============================================================
# 3. Hedges' g
# ============================================================

def hedges_g(t1: np.ndarray, t0: np.ndarray) -> Tuple[float, float, float, float, float]:
    from scipy.stats import norm

    t1 = np.asarray(t1, dtype=float)
    t0 = np.asarray(t0, dtype=float)

    n1, n0 = len(t1), len(t0)
    if n1 < 2 or n0 < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    m1, m0 = float(t1.mean()), float(t0.mean())
    s1, s0 = float(t1.std(ddof=1)), float(t0.std(ddof=1))
    if (s1 == 0 and s0 == 0) or (n1 + n0 - 2) <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if sp == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    d = (m1 - m0) / sp
    J = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
    g = J * d

    se = np.sqrt((n1 + n0) / (n1 * n0) + (g**2) / (2 * (n1 + n0 - 2)))
    ci_lo, ci_hi = g - 1.96 * se, g + 1.96 * se

    if se > 0:
        z = g / se
        p = float(2 * (1 - norm.cdf(abs(z))))
    else:
        p = np.nan

    return (float(g), float(se), float(ci_lo), float(ci_hi), float(p))


# ============================================================
# 4. Negative-control comparisons
# ============================================================

def collect_exact_allele_rows(df: pd.DataFrame) -> pd.DataFrame:
    genes = get_hla_genes_from_columns(df)
    rows = []

    for _, r in df.iterrows():
        sid = r["sample_id"]
        lt = r["log_titer"]

        for gene in genes:
            c1, c2 = f"{gene}_1", f"{gene}_2"
            if c1 not in df.columns or c2 not in df.columns:
                continue

            for col in (c1, c2):
                third = normalize_allele_third_field(r[col])
                if third is None:
                    continue
                second = normalize_allele_second_field(r[col])
                if second is None:
                    continue
                rows.append(
                    {
                        "sample_id": sid,
                        "gene": gene,
                        "allele_second": second,
                        "allele_third": third,
                        "log_titer": lt,
                    }
                )

    return pd.DataFrame(rows)


def compute_negative_control_effects(
    df: pd.DataFrame,
    min_variant_n: int = 5,
    min_other_n: int = 5,
    min_total_second_field_n: int = 10,
) -> pd.DataFrame:
    long = collect_exact_allele_rows(df)
    if long.empty:
        return pd.DataFrame()

    sample_titers = df[["sample_id", "log_titer"]].drop_duplicates().set_index("sample_id")

    res = []
    grouped = long.groupby(["gene", "allele_second"])

    for (gene, allele_second), sub in grouped:
        variants = sub.groupby("allele_third")["sample_id"].unique()
        variant_sets = {k: set(v) for k, v in variants.items()}
        all_second_ids = set().union(*variant_sets.values()) if variant_sets else set()

        if len(all_second_ids) < min_total_second_field_n:
            continue
        if len(variant_sets) < 2:
            continue

        for allele_third, ids_this in variant_sets.items():
            ids_other = all_second_ids - ids_this

            if len(ids_this) < min_variant_n or len(ids_other) < min_other_n:
                continue

            t1 = sample_titers.loc[list(ids_this), "log_titer"].values
            t0 = sample_titers.loc[list(ids_other), "log_titer"].values
            g, se, lo, hi, p = hedges_g(t1, t0)
            if not np.isfinite(g):
                continue

            other_variants = sorted([v for v in variant_sets if v != allele_third])
            res.append(
                {
                    "gene": gene,
                    "allele_second": allele_second,
                    "allele_third": allele_third,
                    "n_variant": len(ids_this),
                    "n_other_same_second": len(ids_other),
                    "n_total_second": len(all_second_ids),
                    "n_other_variants": len(other_variants),
                    "other_variants": ", ".join(other_variants),
                    "g": g,
                    "se": se,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "p_val": p,
                }
            )

    return pd.DataFrame(res)


# ============================================================
# 5. BH-FDR
# ============================================================

def add_fdr(df: pd.DataFrame, p: str = "p_val", q: str = "q") -> pd.DataFrame:
    df = df.copy()
    df[q] = np.nan

    m = df[p].notna()
    pv = df.loc[m, p].values
    if len(pv) == 0:
        return df

    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=float)

    qv = ranked * len(ranked) / ranks
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0.0, 1.0)

    out = np.full(len(pv), np.nan, dtype=float)
    out[order] = qv
    df.loc[m, q] = out
    return df


# ============================================================
# 6. Plotting
# ============================================================

def plot_negative_control(
    eff: pd.DataFrame,
    vaccine_name: str,
    outdir: str,
    top_n: int = 40,
    sort_by: str = "abs_g",
):
    if eff.empty:
        return

    os.makedirs(outdir, exist_ok=True)
    plot_df = eff.copy()
    plot_df["abs_g"] = plot_df["g"].abs()

    if sort_by not in plot_df.columns:
        sort_by = "abs_g"

    plot_df = plot_df.sort_values(sort_by, ascending=False).head(top_n).copy()
    plot_df["label"] = [
        f"{a3}\nvs other {a2}:*"
        for a2, a3 in zip(plot_df["allele_second"], plot_df["allele_third"])
    ]

    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(plot_df)), 6))

    ax.bar(
        x,
        plot_df["g"].values,
        yerr=plot_df["se"].fillna(0).values,
        capsize=3,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"].tolist(), rotation=60, ha="right")
    ax.set_ylabel("Hedges' g (log10 titer)")
    ax.set_title(
        f"{vaccine_name}: negative control\n"
        "same second-field allele, different third-field variant"
    )

    ymax = np.max(np.abs(plot_df["g"].values) + 1.5 * plot_df["se"].fillna(0).values)
    ymax = max(1.0, float(ymax))
    ax.set_ylim(-1.2 * ymax, 1.2 * ymax)

    plt.tight_layout()
    out = os.path.join(outdir, f"negative_control_{vaccine_name}.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out}")


# ============================================================
# 7. CLI
# ============================================================

def parse_vaccine_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Invalid --vaccine '{spec}'. Use NAME:PATH.xlsx")
    name, path = spec.split(":", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Negative-control effect sizes within the same second-field HLA allele using third-field variants."
    )
    ap.add_argument(
        "--vaccine",
        action="append",
        required=True,
        help="Vaccine spec NAME:PATH.xlsx (repeatable)",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min_variant_n", type=int, default=5)
    ap.add_argument("--min_other_n", type=int, default=5)
    ap.add_argument("--min_total_second", type=int, default=10)
    ap.add_argument("--top_n", type=int, default=40)
    ap.add_argument("--fdr", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_eff = []
    for spec in args.vaccine:
        vaccine_name, path = parse_vaccine_spec(spec)
        print(f"\n=== {vaccine_name} ===")
        df = load_vaccine_table(path)
        print(f"Valid samples: {len(df)}")

        eff = compute_negative_control_effects(
            df,
            min_variant_n=args.min_variant_n,
            min_other_n=args.min_other_n,
            min_total_second_field_n=args.min_total_second,
        )
        if eff.empty:
            print("No valid negative-control comparisons.")
            continue

        eff.insert(0, "vaccine", vaccine_name)
        eff = add_fdr(eff, "p_val", "q")

        out_tsv = os.path.join(args.outdir, f"negative_control_effects_{vaccine_name}.tsv")
        eff.to_csv(out_tsv, sep="\t", index=False)
        print(f"Saved table: {out_tsv} (n={len(eff)})")

        sig = eff[eff["q"] < args.fdr].copy()
        out_sig = os.path.join(args.outdir, f"negative_control_effects_{vaccine_name}_significant.tsv")
        sig.to_csv(out_sig, sep="\t", index=False)
        print(f"Saved significant table: {out_sig} (n={len(sig)})")

        plot_negative_control(eff, vaccine_name, args.outdir, top_n=args.top_n)
        all_eff.append(eff)

    if all_eff:
        comb = pd.concat(all_eff, ignore_index=True)
        comb = add_fdr(comb, "p_val", "q_global")
        out_all = os.path.join(args.outdir, "negative_control_effects_all.tsv")
        comb.to_csv(out_all, sep="\t", index=False)
        print(f"\nSaved combined table: {out_all}")

    print("Done.")


if __name__ == "__main__":
    main()
