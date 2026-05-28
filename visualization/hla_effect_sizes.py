#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HLA effect-size analysis for vaccine antibody titers.

All effects are computed on log10-transformed titers.

Dose–response plots:
- Y axis: Hedges' g (effect size on log10 titer), consistent with main effect definition
- For each FDR-significant (gene, allele):
    g(dose=1 vs dose=0) and g(dose=2 vs dose=0)
  Baseline dose=0 is shown as g=0 but:
    - NOT included in legend
    - No p-value annotations on top
"""

import argparse
import os
import re
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Allele normalization
# ============================================================

def normalize_allele_first_field(allele: str) -> str | None:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None
    if "*" not in allele:
        return None
    prefix, rest = allele.split("*", 1)
    a = rest.split(":")[0]
    return f"{prefix}*{a}" if a else None


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
    if not parts or not parts[0]:
        return None
    if len(parts) == 1:
        return f"{prefix}*{parts[0]}"
    return f"{prefix}*{parts[0]}:{parts[1]}"


def get_hla_genes_from_columns(df: pd.DataFrame) -> List[str]:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c.startswith("HLA-")})


# ============================================================
# 2. Rare filters
# ============================================================

def load_excluded_genes(path: str) -> Set[str]:
    """
    Read hla_rare_alleles.txt and return set of gene names that should be excluded.
    Expected lines like: "HLA-G (10 unique alleles):"
    """
    pat = re.compile(r"^(HLA-[A-Z0-9]+)\s+\(")
    out: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out.add(m.group(1))
    return out


def load_excluded_alleles(path: str | None) -> Set[Tuple[str, str]]:
    """
    Read rare_alleles_<field>.tsv with columns gene, allele and return a set of pairs.
    """
    if path is None:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rare alleles file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    if not {"gene", "allele"}.issubset(df.columns):
        raise ValueError(f"Rare alleles file {path} must have columns: gene, allele")
    return {(r["gene"], r["allele"]) for _, r in df.iterrows()}


# ============================================================
# 3. Load data
# ============================================================

def load_hla(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "sample_id" not in df.columns:
        raise ValueError("combined_hla_out.xlsx must contain column 'sample_id'.")
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_vaccine_pheno(vaccine: str, path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    if "ZLIMS ID" not in df.columns:
        raise ValueError(f"{path} must contain column 'ZLIMS ID'.")
    if "region" not in df.columns:
        raise ValueError(f"{path} must contain column 'region'.")

    tcol = f"{vaccine}_ME_ml"
    if tcol not in df.columns:
        raise ValueError(f"{path} must contain column '{tcol}'.")

    df["ZLIMS ID"] = df["ZLIMS ID"].astype(str)
    df["region"] = df["region"].astype(str)

    t = df[tcol].astype(float)
    t = t.replace([np.inf, -np.inf], np.nan)
    df = df[t > 0].copy()
    df["log_titer"] = np.log10(df[tcol].astype(float))

    return df[["ZLIMS ID", "region", "log_titer"]]


def merge_pheno_hla(pheno: pd.DataFrame, hla: pd.DataFrame) -> pd.DataFrame:
    merged = pheno.merge(hla, left_on="ZLIMS ID", right_on="sample_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples between phenotype and HLA tables.")
    # keep one row per sample_id
    merged = merged.drop_duplicates(subset=["sample_id"]).copy()
    return merged


# ============================================================
# 4. Long table
# ============================================================

def build_long_table(
    merged: pd.DataFrame,
    field: str,
    allowed_genes: List[str],
    excluded_alleles: Set[Tuple[str, str]],
) -> pd.DataFrame:
    norm = normalize_allele_first_field if field == "first" else normalize_allele_second_field
    rows = []

    for _, r in merged.iterrows():
        sid = r["sample_id"]
        reg = r["region"]
        lt = r["log_titer"]

        for g in allowed_genes:
            c1, c2 = f"{g}_1", f"{g}_2"
            if c1 not in merged.columns or c2 not in merged.columns:
                continue

            a1 = norm(r[c1])
            a2 = a1 if str(r[c2]).strip() == "-" else norm(r[c2])

            for a in (a1, a2):
                if a is None:
                    continue
                if (g, a) in excluded_alleles:
                    continue
                rows.append(
                    {"sample_id": sid, "region": reg, "gene": g, "allele": a, "log_titer": lt}
                )

    return pd.DataFrame(rows)


# ============================================================
# 5. Effect size core (Hedges' g)
# ============================================================

def hedges_g(t1: np.ndarray, t0: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Hedges' g for two independent groups on already-prepared measurements (here: log10 titers).
    Returns: (g, se_g, ci_low, ci_high, p_val)
    """
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


def compute_effects(long: pd.DataFrame, min_c: int = 5, min_n: int = 5) -> pd.DataFrame:
    """
    Overall effect sizes for each (gene, allele) across all regions.
    """
    if long.empty:
        return pd.DataFrame(
            columns=["gene", "allele", "n_carriers", "n_noncarriers", "g", "se", "ci_lower", "ci_upper", "p_val"]
        )

    res = []
    titers = long[["sample_id", "log_titer"]].drop_duplicates().set_index("sample_id")

    all_ids = set(titers.index)

    for (gene, allele), sub in long.groupby(["gene", "allele"]):
        carriers = set(sub["sample_id"].unique())
        non = all_ids - carriers

        if len(carriers) < min_c or len(non) < min_n:
            continue

        t1 = titers.loc[list(carriers), "log_titer"].values
        t0 = titers.loc[list(non), "log_titer"].values
        g, se, lo, hi, p = hedges_g(t1, t0)
        if np.isnan(g):
            continue

        res.append(
            dict(
                gene=gene,
                allele=allele,
                n_carriers=len(carriers),
                n_noncarriers=len(non),
                g=g,
                se=se,
                ci_lower=lo,
                ci_upper=hi,
                p_val=p,
            )
        )

    return pd.DataFrame(res)


# ============================================================
# 6. Dosage table
# ============================================================

def dosage_table(merged: pd.DataFrame, gene: str, allele: str, field: str) -> pd.DataFrame:
    norm = normalize_allele_first_field if field == "first" else normalize_allele_second_field
    c1, c2 = f"{gene}_1", f"{gene}_2"
    if c1 not in merged.columns or c2 not in merged.columns:
        return pd.DataFrame(columns=["sample_id", "dosage", "log_titer"])

    rows = []
    for _, r in merged.iterrows():
        a1 = norm(r[c1])
        a2 = norm(r[c2])
        dose = int(a1 == allele) + int(a2 == allele)
        rows.append({"sample_id": r["sample_id"], "dosage": dose, "log_titer": r["log_titer"]})
    return pd.DataFrame(rows)


# ============================================================
# 7. Dose–response plot (g on Y axis), no "0 copies" in legend, no p-value text
# ============================================================

def plot_dose_response(
    sig: pd.DataFrame,
    vaccine: str,
    merged: pd.DataFrame,
    field: str,
    outdir: str,
    min_c: int,
    min_n: int,
):
    sub = sig[sig["vaccine"] == vaccine].copy()
    if sub.empty:
        return

    os.makedirs(outdir, exist_ok=True)

    # keep order, remove duplicates
    alleles = list(dict.fromkeys(zip(sub["gene"].tolist(), sub["allele"].tolist())))
    if not alleles:
        return

    gvals = {0: [], 1: [], 2: []}
    sevals = {0: [], 1: [], 2: []}
    labels: List[str] = []

    for gene, allele in alleles:
        df = dosage_table(merged, gene, allele, field)

        t0 = df.loc[df["dosage"] == 0, "log_titer"].values
        n0 = len(t0)

        # baseline (shown but not in legend)
        gvals[0].append(0.0)
        sevals[0].append(0.0)

        for d in (1, 2):
            td = df.loc[df["dosage"] == d, "log_titer"].values
            if len(td) < min_c or n0 < min_n:
                g, se = np.nan, 0.0
            else:
                g, se, _, _, _ = hedges_g(td, t0)

            gvals[d].append(g)
            sevals[d].append(se if np.isfinite(se) else 0.0)

        labels.append(allele)

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 5))

    # dose=0 (no legend)
    ax.bar(
        x - w,
        gvals[0],
        w,
        yerr=sevals[0],
        capsize=3,
        label="_nolegend_",
    )
    # dose=1 and dose=2 in legend
    ax.bar(
        x,
        gvals[1],
        w,
        yerr=sevals[1],
        capsize=3,
        label="1 copy",
    )
    ax.bar(
        x + w,
        gvals[2],
        w,
        yerr=sevals[2],
        capsize=3,
        label="2 copies",
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Hedges' g (log10 titer)")
    ax.set_title(f"{vaccine}: dose–response (effect size g vs dose=0)")
    ax.legend()

    # y-limits
    ymax = 0.0
    for d in (1, 2):
        arr_g = np.asarray(gvals[d], dtype=float)
        arr_se = np.asarray(sevals[d], dtype=float)
        mask = np.isfinite(arr_g) & np.isfinite(arr_se)
        if mask.any():
            ymax = max(ymax, float(np.max(np.abs(arr_g[mask]) + 1.5 * arr_se[mask])))

    if ymax == 0.0:
        ymax = 1.0
    ax.set_ylim(-1.2 * ymax, 1.2 * ymax)

    plt.tight_layout()
    out_path = os.path.join(outdir, f"dose_response_{vaccine}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Dose–response plot saved to: {out_path}")


# ============================================================
# 8. FDR (Benjamini–Hochberg)
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
# 9. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Compute HLA allele effect sizes on vaccine antibody titers (log10), with dose–response plots in Hedges' g."
    )
    ap.add_argument("--hla", required=True, help="Path to combined_hla_out.xlsx")
    ap.add_argument("--rare_genes", required=True, help="Path to hla_rare_alleles.txt (genes to exclude)")
    ap.add_argument("--rare_first", default=None, help="Path to rare_alleles_first_field.tsv (optional)")
    ap.add_argument("--rare_second", default=None, help="Path to rare_alleles_second_field.tsv (optional)")
    ap.add_argument("--field", choices=["first", "second"], default="first", help="Allele resolution")
    ap.add_argument("--vaccine", action="append", required=True, help="Vaccine spec NAME:PATH.xlsx (repeatable)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min_carriers", type=int, default=5, help="Min carriers")
    ap.add_argument("--min_noncarriers", type=int, default=5, help="Min non-carriers")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR threshold")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading HLA data...")
    hla = load_hla(args.hla)

    all_genes = get_hla_genes_from_columns(hla)
    excl_genes = load_excluded_genes(args.rare_genes)
    genes = [g for g in all_genes if g not in excl_genes]
    print(f"Detected genes: {len(all_genes)}; excluded genes: {len(excl_genes)}; kept: {len(genes)}")

    excl_alleles = load_excluded_alleles(args.rare_first if args.field == "first" else args.rare_second)
    print(f"Excluded alleles for field '{args.field}': {len(excl_alleles)}")

    merged_cache: Dict[str, pd.DataFrame] = {}
    all_eff: List[pd.DataFrame] = []

    # parse vaccine specs
    vaccine_specs: Dict[str, str] = {}
    for spec in args.vaccine:
        if ":" not in spec:
            raise ValueError(f"Invalid --vaccine '{spec}'. Use NAME:PATH.xlsx")
        name, path = spec.split(":", 1)
        vaccine_specs[name.strip()] = path.strip()

    for name, path in vaccine_specs.items():
        print(f"\n=== Vaccine: {name} ===")
        ph = load_vaccine_pheno(name, path)
        print(f"Phenotype valid samples: {len(ph)}")

        merged = merge_pheno_hla(ph, hla)
        print(f"Merged samples: {len(merged)}")
        merged_cache[name] = merged

        long = build_long_table(merged, args.field, genes, excl_alleles)
        print(f"Long table rows: {len(long)}")

        eff = compute_effects(long, args.min_carriers, args.min_noncarriers)
        if eff.empty:
            print("No alleles passed thresholds for this vaccine.")
            continue

        eff.insert(0, "vaccine", name)
        all_eff.append(eff)

    if not all_eff:
        print("\nNo effects computed for any vaccine (check inputs / thresholds).")
        return

    comb = pd.concat(all_eff, ignore_index=True)
    comb = add_fdr(comb, "p_val", "q")

    # save all
    out_all = os.path.join(args.outdir, f"effects_all_{args.field}_field.tsv")
    comb.to_csv(out_all, sep="\t", index=False)
    print(f"\nSaved: {out_all}")

    # significant
    sig = comb[comb["q"] < args.fdr].copy()
    out_sig = os.path.join(args.outdir, f"effects_significant_{args.field}_field.tsv")
    sig.to_csv(out_sig, sep="\t", index=False)
    print(f"Saved: {out_sig} (n={len(sig)})")

    # dose–response plots for significant alleles
    if not sig.empty:
        dose_dir = os.path.join(args.outdir, f"dose_plots_{args.field}_field")
        for v in sig["vaccine"].unique():
            if v not in merged_cache:
                continue
            plot_dose_response(
                sig=sig,
                vaccine=v,
                merged=merged_cache[v],
                field=args.field,
                outdir=dose_dir,
                min_c=args.min_carriers,
                min_n=args.min_noncarriers,
            )

    print("Done.")


if __name__ == "__main__":
    main()