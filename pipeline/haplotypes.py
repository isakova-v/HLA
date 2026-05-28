#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Haplotype heatmaps (unphased inter-gene pairs) for vaccine datasets.

For each vaccine:
- load phenotype: columns ["ZLIMS ID", "region", f"{VACC}_ME_ml"]
- keep positive titers, compute log10
- merge with HLA table on sample_id
- for a list of genes, build ALL inter-gene pairs (combinations)
- for each gene pair:
    - build unphased pairs via product(allelesA, allelesB)
    - count pairs -> pivot (geneA_allele x geneB_allele) -> heatmap
    - print top haplotypes in console (per vaccine & gene pair)
- optional filtering by min allele count (marginal frequency threshold)
- optional per-region heatmaps + per-region top haplotypes

Additionally (GLOBAL report):
- Print TOP-15 haplotypes ranked by *individual vaccine subset counts* (NOT by total sum).
  i.e. rank all (vaccine, haplotype) entries by count_in_that_vaccine and show top-15.
- For each of those top entries, also show in which vaccines this haplotype is "significant":
  significant := count_in_that_vaccine >= signif_min_count.

Notes:
- counts reflect "product" expansion (a sample can contribute up to 4 pairs).
- Global keys are (geneA, alleleA, geneB, alleleB) to avoid collisions between gene pairs.
"""

import argparse
import os
from collections import Counter, defaultdict
from itertools import product, combinations
from typing import Optional, Tuple, List, Dict, DefaultDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False


# ------------------------------------------------------------
# Allele normalization
# ------------------------------------------------------------

def normalize_allele_first_field(allele: str) -> Optional[str]:
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
    return f"{prefix}*{parts[0]}"


def normalize_allele_second_field(allele: str) -> Optional[str]:
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


def get_gene_prefixes(df: pd.DataFrame) -> List[str]:
    """Prefixes like 'HLA-A' from columns 'HLA-A_1', 'HLA-A_2' or 'A_1','A_2'."""
    cols = [c for c in df.columns if c.endswith("_1") or c.endswith("_2")]
    return sorted({c.rsplit("_", 1)[0] for c in cols})


def resolve_gene_prefix(df: pd.DataFrame, gene: str) -> str:
    """
    Resolve gene prefix as it appears in HLA table columns.
    Accepts:
      - exact prefix (e.g., 'HLA-B' or 'B')
      - short name without 'HLA-' (e.g., 'DRB1' -> 'HLA-DRB1' if present)
    """
    prefixes = set(get_gene_prefixes(df))
    gene = gene.strip()

    if gene in prefixes:
        return gene

    if not gene.startswith("HLA-"):
        g2 = "HLA-" + gene
        if g2 in prefixes:
            return g2

    if gene.startswith("HLA-"):
        short = gene.replace("HLA-", "")
        if short in prefixes:
            return short

    raise ValueError(
        f"Cannot resolve gene '{gene}' to HLA-table prefixes. "
        f"Available examples: {sorted(list(prefixes))[:12]} ..."
    )


def resolve_genes(df: pd.DataFrame, genes: List[str]) -> List[str]:
    """Resolve all user-provided genes to actual prefixes present in the HLA table."""
    resolved = [resolve_gene_prefix(df, g) for g in genes]

    # keep order but unique
    out: List[str] = []
    seen = set()
    for g in resolved:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_hla(hla_path: str) -> pd.DataFrame:
    hla = pd.read_excel(hla_path)
    if "sample_id" not in hla.columns:
        raise ValueError("HLA file must contain column 'sample_id'.")
    hla["sample_id"] = hla["sample_id"].astype(str)
    return hla


def load_vaccine_pheno(vaccine_name: str, path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    if "ZLIMS ID" not in df.columns:
        raise ValueError(f"{path} must contain 'ZLIMS ID' column.")
    if "region" not in df.columns:
        raise ValueError(f"{path} must contain 'region' column.")

    tcol = f"{vaccine_name}_ME_ml"
    if tcol not in df.columns:
        raise ValueError(f"{path}: column '{tcol}' not found.")

    df["ZLIMS ID"] = df["ZLIMS ID"].astype(str)
    df["region"] = df["region"].astype(str)

    titers = df[tcol].astype(float).replace([np.inf, -np.inf], np.nan)
    df = df[titers > 0].copy()
    df["log_titer"] = np.log10(df[tcol].astype(float))

    return df[["ZLIMS ID", "region", "log_titer"]].copy()


def merge_pheno_hla(pheno: pd.DataFrame, hla: pd.DataFrame) -> pd.DataFrame:
    merged = pheno.merge(hla, left_on="ZLIMS ID", right_on="sample_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples between phenotype and HLA tables.")
    merged = merged.drop_duplicates(subset=["sample_id"]).copy()
    return merged


# ------------------------------------------------------------
# Haplotype pairs (unphased) and heatmaps
# ------------------------------------------------------------

def extract_two_alleles(row: pd.Series, gene_prefix: str, normalizer) -> Tuple[Optional[str], Optional[str]]:
    c1, c2 = f"{gene_prefix}_1", f"{gene_prefix}_2"
    if c1 not in row.index or c2 not in row.index:
        return (None, None)

    a1 = normalizer(row[c1])
    raw2 = row[c2]
    # convention: "-" in _2 means homozygous as _1
    if isinstance(raw2, str) and raw2.strip() == "-":
        a2 = a1
    else:
        a2 = normalizer(raw2)

    return (a1, a2)


def build_unphased_pairs(
    merged: pd.DataFrame,
    gene_a: str,
    gene_b: str,
    field: str,
) -> List[Tuple[str, str]]:
    """
    Build all unphased inter-gene pairs (gene_a_allele, gene_b_allele)
    using product of two alleles each.
    """
    normalizer = normalize_allele_first_field if field == "first" else normalize_allele_second_field

    pairs: List[Tuple[str, str]] = []
    for _, row in merged.iterrows():
        a1, a2 = extract_two_alleles(row, gene_a, normalizer)
        b1, b2 = extract_two_alleles(row, gene_b, normalizer)

        alleles_a = sorted([x for x in (a1, a2) if x is not None])
        alleles_b = sorted([x for x in (b1, b2) if x is not None])

        if not alleles_a or not alleles_b:
            continue

        pairs.extend(list(product(alleles_a, alleles_b)))

    return pairs


def pairs_to_pivot(pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    ctr = Counter(pairs)
    if not ctr:
        return pd.DataFrame()

    df = pd.DataFrame([(a, b, c) for (a, b), c in ctr.items()], columns=["A_allele", "B_allele", "Count"])
    pivot = df.pivot_table(index="A_allele", columns="B_allele", values="Count", fill_value=0, aggfunc="sum")
    return pivot


def filter_by_min_allele_count(pivot: pd.DataFrame, min_allele_count: int) -> pd.DataFrame:
    """Filter alleles by marginal count >= threshold."""
    if pivot.empty or min_allele_count <= 0:
        return pivot

    row_sum = pivot.sum(axis=1)
    col_sum = pivot.sum(axis=0)

    keep_rows = row_sum[row_sum >= min_allele_count].index
    keep_cols = col_sum[col_sum >= min_allele_count].index

    return pivot.loc[keep_rows, keep_cols]


def plot_heatmap(pivot: pd.DataFrame, title: str, xlabel: str, ylabel: str, out_png: str, figsize=(14, 9)):
    if pivot.empty:
        print(f"[skip] empty pivot: {title}")
        return

    out_dir = os.path.dirname(out_png) or "."
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=figsize)

    if _HAS_SEABORN:
        sns.heatmap(pivot, linewidths=0.5, cmap="PuBuGn")
    else:
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar()
        plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns, rotation=90)
        plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved heatmap: {out_png}")


def print_top_haplotypes(
    pairs: List[Tuple[str, str]],
    gene_a: str,
    gene_b: str,
    top_k: int = 20,
    min_pair_count: int = 1,
):
    """
    Print most frequent unphased inter-gene allele pairs (A_allele, B_allele).
    Note: counts reflect "product" expansion, so a sample contributes up to 4 pairs.
    """
    ctr = Counter(pairs)
    if not ctr:
        print(f"Top haplotypes {gene_a}~{gene_b}: (none)")
        return

    items = [(ab, c) for ab, c in ctr.most_common() if c >= min_pair_count]
    items = items[:top_k]

    print(f"Top haplotypes {gene_a} ~ {gene_b} (top {top_k}, min_count={min_pair_count}):")
    for (a, b), c in items:
        print(f"  {a}  ×  {b} : {c}")


def print_global_top15_by_individual_vaccine_counts(
    by_vaccine: Dict[Tuple[str, str, str, str], Counter],
    top_k: int = 15,
    signif_min_count: int = 1,
):
    """
    GLOBAL top-K ranked by count within a vaccine, across all vaccines and all gene pairs.
    Rank all (vaccine, haplotype) entries by count_in_that_vaccine and show top-K.

    by_vaccine[key] is Counter({vaccine_name: count_in_that_vaccine, ...})
    key = (geneA, alleleA, geneB, alleleB)
    """
    rows = []
    for key, vac_ctr in by_vaccine.items():
        gA, a, gB, b = key
        for vacc, cnt in vac_ctr.items():
            rows.append((cnt, vacc, gA, a, gB, b))

    if not rows:
        print("\n=== GLOBAL TOP HAPLOTYPES (by individual vaccine counts) ===\n(none)")
        return

    rows.sort(key=lambda x: x[0], reverse=True)

    print("\n=== GLOBAL TOP HAPLOTYPES (ranked by count within a vaccine) ===")
    print(f"Top {top_k}. 'Significant in vaccine' := count_in_that_vaccine >= {signif_min_count}")

    shown = 0
    for cnt, vacc, gA, a, gB, b in rows:
        key = (gA, a, gB, b)
        vac_ctr = by_vaccine.get(key, Counter())
        signif_vaccines = [(v, c) for v, c in vac_ctr.most_common() if c >= signif_min_count]
        signif_str = ", ".join([f"{v}({c})" for v, c in signif_vaccines]) if signif_vaccines else "—"

        print(f"  {vacc}: {gA}:{a}  ×  {gB}:{b} : {cnt} | significant in: {signif_str}")

        shown += 1
        if shown >= top_k:
            break


def safe_name(s: str) -> str:
    return s.replace("-", "").replace("*", "_").replace(":", "_")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build unphased inter-gene HLA haplotype heatmaps for vaccine datasets (ALL gene pairs)."
    )
    p.add_argument("--hla", required=True, help="Path to combined_hla_out.xlsx (with sample_id and gene_1/gene_2 columns)")
    p.add_argument("--field", choices=["first", "second"], default="first", help="Allele resolution (default: first)")
    p.add_argument(
        "--vaccine",
        action="append",
        required=True,
        help="Vaccine spec NAME:PATH.xlsx (repeatable), e.g. --vaccine HBV:HBV.xlsx",
    )
    p.add_argument("--outdir", required=True, help="Output directory for heatmaps")

    p.add_argument(
        "--genes",
        required=True,
        help="Comma-separated genes to consider; all pairwise combinations will be plotted. "
             "Example: --genes HLA-A,HLA-B,HLA-C,HLA-DRB1 (or A,B,C,DRB1)",
    )

    p.add_argument(
        "--min_allele_count",
        type=int,
        default=0,
        help="If >0, filter alleles by marginal count >= threshold.",
    )
    p.add_argument(
        "--by_region",
        action="store_true",
        help="Also build heatmaps separately for each region in phenotype.",
    )

    p.add_argument("--top_k", type=int, default=20, help="How many most frequent haplotypes to print per gene pair.")
    p.add_argument("--min_pair_count", type=int, default=1, help="Print only haplotypes with count >= this (per-pair printing).")

    p.add_argument(
        "--signif_min_count",
        type=int,
        default=0,
        help="For GLOBAL top output: a haplotype is 'significant' in a vaccine if count_in_that_vaccine >= this. "
             "If 0, defaults to --min_pair_count.",
    )

    p.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Safety: if >0, limit the number of gene pairs processed (useful for big gene lists).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    signif_min = args.signif_min_count if args.signif_min_count > 0 else args.min_pair_count

    hla = load_hla(args.hla)

    genes_raw = [x.strip() for x in args.genes.split(",") if x.strip()]
    if len(genes_raw) < 2:
        raise ValueError("--genes must contain at least 2 genes.")
    genes = resolve_genes(hla, genes_raw)

    gene_pairs = list(combinations(genes, 2))
    if args.max_pairs and args.max_pairs > 0:
        gene_pairs = gene_pairs[:args.max_pairs]

    print(f"Resolved genes ({len(genes)}): {genes}")
    print(f"Gene pairs to process ({len(gene_pairs)}): {gene_pairs}")

    # Per-vaccine breakdown for each haplotype key:
    # key -> Counter({vaccine_name: count})
    global_by_vaccine: DefaultDict[Tuple[str, str, str, str], Counter] = defaultdict(Counter)

    # parse vaccine specs
    vaccine_specs: Dict[str, str] = {}
    for spec in args.vaccine:
        if ":" not in spec:
            raise ValueError(f"Invalid --vaccine '{spec}'. Use NAME:PATH.xlsx")
        name, path = spec.split(":", 1)
        vaccine_specs[name.strip()] = path.strip()

    for vacc, path in vaccine_specs.items():
        print(f"\n=== Vaccine: {vacc} ===")
        pheno = load_vaccine_pheno(vacc, path)
        merged = merge_pheno_hla(pheno, hla)

        # overall per gene-pair heatmaps + top haplotypes
        for gA, gB in gene_pairs:
            pairs = build_unphased_pairs(merged, gA, gB, args.field)

            # accumulate counts for GLOBAL "by individual vaccine" ranking
            local_ctr = Counter(pairs)
            for (a, b), c in local_ctr.items():
                key = (gA, a, gB, b)
                global_by_vaccine[key][vacc] += c

            # per-vaccine per-pair printing
            print_top_haplotypes(
                pairs=pairs,
                gene_a=gA,
                gene_b=gB,
                top_k=args.top_k,
                min_pair_count=args.min_pair_count,
            )

            pivot = pairs_to_pivot(pairs)
            pivot = filter_by_min_allele_count(pivot, args.min_allele_count)

            title = f"{vacc}: {gA} ~ {gB} haplotypes (unphased), field={args.field}"
            if args.min_allele_count > 0:
                title += f", min_allele_count={args.min_allele_count}"

            out_png = os.path.join(
                args.outdir,
                f"heatmap_{vacc}_{safe_name(gA)}_{safe_name(gB)}_{args.field}.png",
            )
            plot_heatmap(
                pivot=pivot,
                title=title,
                xlabel=f"{gB} allele",
                ylabel=f"{gA} allele",
                out_png=out_png,
                figsize=(14, 9),
            )

        # per-region heatmaps + per-region top haplotypes (NOT accumulated globally to avoid double counting)
        if args.by_region:
            for region, subm in merged.groupby("region"):
                print(f"\n--- Region: {region} ---")
                for gA, gB in gene_pairs:
                    pairs_r = build_unphased_pairs(subm, gA, gB, args.field)

                    print_top_haplotypes(
                        pairs=pairs_r,
                        gene_a=gA,
                        gene_b=gB,
                        top_k=args.top_k,
                        min_pair_count=args.min_pair_count,
                    )

                    pivot_r = pairs_to_pivot(pairs_r)
                    pivot_r = filter_by_min_allele_count(pivot_r, args.min_allele_count)

                    title_r = f"{vacc} ({region}): {gA} ~ {gB} haplotypes (unphased), field={args.field}"
                    if args.min_allele_count > 0:
                        title_r += f", min_allele_count={args.min_allele_count}"

                    out_png_r = os.path.join(
                        args.outdir,
                        f"heatmap_{vacc}_{region}_{safe_name(gA)}_{safe_name(gB)}_{args.field}.png",
                    )
                    plot_heatmap(
                        pivot=pivot_r,
                        title=title_r,
                        xlabel=f"{gB} allele",
                        ylabel=f"{gA} allele",
                        out_png=out_png_r,
                        figsize=(14, 9),
                    )

    # Final: GLOBAL top-15 ranked by individual vaccine counts
    print_global_top15_by_individual_vaccine_counts(
        by_vaccine=global_by_vaccine,
        top_k=15,
        signif_min_count=signif_min,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()