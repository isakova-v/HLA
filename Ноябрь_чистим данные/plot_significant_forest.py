#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forest plots for significant HLA allele effects:
X axis: effect size g (log10 titer)
Y axis: region + overall.

Inputs:
- effects_significant_FDR_first_field.tsv  (or ...second_field.tsv)
  (overall effects per vaccine, gene, allele)
- combined_hla_out.xlsx
- hla_rare_alleles.txt
- rare_alleles_first_field.tsv / rare_alleles_second_field.tsv
- filtered_vaccines/<vaccine>.xlsx (phenotype with region & titers)

For each (vaccine, gene, allele) from the significant file:
- recompute region-specific effect sizes,
- plot region points + overall point.

Usage example:

python plot_effects_by_region.py \
    --effects effects_significant_FDR_first_field.tsv \
    --filtered_dir filtered_vaccines \
    --hla combined_hla_out.xlsx \
    --rare_genes hla_rare_alleles.txt \
    --rare_first rare_alleles_first_field.tsv \
    --rare_second rare_alleles_second_field.tsv \
    --field first \
    --outdir plots_effects_by_region
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# ------------------------------
# Basic HLA helpers (same logic as in main script)
# ------------------------------

def normalize_allele_first_field(allele: str) -> str | None:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None
    if "*" not in allele:
        return None
    prefix, rest = allele.split("*", 1)
    parts = rest.split(":")
    if len(parts) == 0 or parts[0] == "":
        return None
    return f"{prefix}*{parts[0]}"


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
    if len(parts) == 0 or parts[0] == "":
        return None
    if len(parts) == 1:
        return f"{prefix}*{parts[0]}"
    return f"{prefix}*{parts[0]}:{parts[1]}"


def get_hla_genes_from_columns(df: pd.DataFrame) -> List[str]:
    hla_cols = [c for c in df.columns if c.startswith("HLA-")]
    genes = sorted({c.rsplit("_", 1)[0] for c in hla_cols})
    return genes


def load_excluded_genes(rare_genes_path: str) -> Set[str]:
    excluded: Set[str] = set()
    import re

    pattern = re.compile(r"^(HLA-[A-Z0-9]+)\s+\(")
    with open(rare_genes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                excluded.add(m.group(1))
    return excluded


def load_excluded_alleles(rare_alleles_path: str | None) -> Set[Tuple[str, str]]:
    if rare_alleles_path is None:
        return set()
    if not os.path.exists(rare_alleles_path):
        raise FileNotFoundError(f"Rare alleles file not found: {rare_alleles_path}")
    df = pd.read_csv(rare_alleles_path, sep="\t")
    if not {"gene", "allele"}.issubset(df.columns):
        raise ValueError(
            f"Rare alleles file {rare_alleles_path} must have 'gene' and 'allele' columns."
        )
    return {(row["gene"], row["allele"]) for _, row in df.iterrows()}


def load_hla(hla_path: str) -> pd.DataFrame:
    hla = pd.read_excel(hla_path)
    if "sample_id" not in hla.columns:
        raise ValueError("combined_hla_out.xlsx must have column 'sample_id'.")
    hla["sample_id"] = hla["sample_id"].astype(str)
    return hla


def load_vaccine_pheno_from_filtered(vaccine_name: str, path: str) -> pd.DataFrame:
    """
    Load filtered_vaccines/<vaccine>.xlsx and construct:
      ZLIMS ID, region, log_titer
    """
    df = pd.read_excel(path)

    if "ZLIMS ID" not in df.columns:
        raise ValueError(f"{path} must contain 'ZLIMS ID' column.")
    if "region" not in df.columns:
        raise ValueError(f"{path} must contain 'region' column.")

    titer_col = f"{vaccine_name}_ME_ml"
    if titer_col not in df.columns:
        raise ValueError(f"{path}: column '{titer_col}' not found.")

    df["ZLIMS ID"] = df["ZLIMS ID"].astype(str)
    df["region"] = df["region"].astype(str)

    titers = df[titer_col].astype(float)
    titers = titers.replace([np.inf, -np.inf], np.nan)
    valid = titers > 0
    df = df[valid].copy()
    titers = titers[valid]

    df["log_titer"] = np.log10(titers)

    return df[["ZLIMS ID", "region", "log_titer"]].copy()


def merge_pheno_hla(pheno: pd.DataFrame, hla: pd.DataFrame) -> pd.DataFrame:
    merged = pheno.merge(hla, left_on="ZLIMS ID", right_on="sample_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples between phenotype and HLA tables.")
    return merged


def build_long_allele_table_for_vaccine(
    merged: pd.DataFrame,
    field: str,
    allowed_genes: List[str],
    excluded_alleles: Set[Tuple[str, str]],
) -> pd.DataFrame:
    """
    ['sample_id', 'region', 'gene', 'allele', 'log_titer']
    """
    if field == "first":
        normalizer = normalize_allele_first_field
    elif field == "second":
        normalizer = normalize_allele_second_field
    else:
        raise ValueError("field must be 'first' or 'second'")

    records = []
    hla_cols = set(merged.columns)

    for _, row in merged.iterrows():
        sample_id = row["sample_id"]
        region = row["region"]
        log_titer = row["log_titer"]

        for gene in allowed_genes:
            col1 = f"{gene}_1"
            col2 = f"{gene}_2"
            if col1 not in hla_cols or col2 not in hla_cols:
                continue

            raw1 = row[col1]
            raw2 = row[col2]

            a1 = normalizer(raw1)
            if isinstance(raw2, str) and raw2.strip() == "-":
                a2 = a1
            else:
                a2 = normalizer(raw2)

            for a in (a1, a2):
                if a is None:
                    continue
                if (gene, a) in excluded_alleles:
                    continue
                records.append(
                    {
                        "sample_id": sample_id,
                        "region": region,
                        "gene": gene,
                        "allele": a,
                        "log_titer": log_titer,
                    }
                )

    return pd.DataFrame.from_records(records)


def compute_effect_sizes_by_region(
    long_df: pd.DataFrame,
    min_carriers: int = 5,
    min_noncarriers: int = 5,
) -> pd.DataFrame:
    """
    Hedges' g for each (gene, allele, region).
    """
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "allele",
                "region",
                "n_carriers",
                "n_noncarriers",
                "mean_carriers",
                "mean_noncarriers",
                "g",
                "ci_lower",
                "ci_upper",
                "se_g",
                "p_val",
            ]
        )

    sample_info = (
        long_df[["sample_id", "region", "log_titer"]]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")
    )
    all_regions: np.ndarray = sample_info["region"].unique()

    results = []
    grouped = long_df.groupby(["gene", "allele"])

    for (gene, allele), sub in grouped:
        for region in all_regions:
            region_ids: Set[str] = set(
                sample_info.index[sample_info["region"] == region]
            )
            if not region_ids:
                continue

            carriers_ids: Set[str] = set(
                sub.loc[sub["region"] == region, "sample_id"].unique()
            )
            if not carriers_ids:
                continue

            non_ids: Set[str] = region_ids - carriers_ids

            n1 = len(carriers_ids)
            n0 = len(non_ids)
            if n1 < min_carriers or n0 < min_noncarriers:
                continue

            t1 = sample_info.loc[list(carriers_ids), "log_titer"].values
            t0 = sample_info.loc[list(non_ids), "log_titer"].values

            m1 = t1.mean()
            m0 = t0.mean()
            s1 = t1.std(ddof=1)
            s0 = t0.std(ddof=1)

            if (s1 == 0 and s0 == 0) or (n1 + n0 - 2) <= 0:
                continue

            sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2))
            if sp == 0:
                continue

            d = (m1 - m0) / sp
            J = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
            g = J * d

            se_g = np.sqrt((n1 + n0) / (n1 * n0) + (g ** 2) / (2 * (n1 + n0 - 2)))
            ci_low = g - 1.96 * se_g
            ci_high = g + 1.96 * se_g

            if se_g > 0:
                z = g / se_g
                p_val = float(2 * (1 - norm.cdf(abs(z))))
            else:
                p_val = np.nan

            results.append(
                {
                    "gene": gene,
                    "allele": allele,
                    "region": region,
                    "n_carriers": n1,
                    "n_noncarriers": n0,
                    "mean_carriers": m1,
                    "mean_noncarriers": m0,
                    "g": g,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "se_g": se_g,
                    "p_val": p_val,
                }
            )

    return pd.DataFrame(results)


# ------------------------------
# Plotting
# ------------------------------

def safe_filename(*parts: str) -> str:
    raw = "_".join(str(p) for p in parts)
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_" for c in raw
    )


def plot_forest_overall_plus_regions(
    overall_row: pd.Series,
    region_df: pd.DataFrame,
    outdir: Path,
):
    """
    overall_row: row from significant overall table (vaccine, gene, allele, g, g_low, g_high, ...)
    region_df: rows with columns ['region', 'g', 'ci_lower'/'g_low', 'ci_upper'/'g_high']
    """
    vaccine = overall_row["vaccine"]
    gene = overall_row["gene"]
    allele = overall_row["allele"]

    rows = []

    # overall row
    rows.append(
        {
            "region": "overall",
            "g": overall_row["g"],
            "g_low": overall_row.get("g_low", overall_row.get("ci_lower")),
            "g_high": overall_row.get("g_high", overall_row.get("ci_upper")),
        }
    )

    # region-specific rows
    if not region_df.empty:
        # normalize CI names
        if "g_low" not in region_df.columns and "ci_lower" in region_df.columns:
            region_df = region_df.rename(columns={"ci_lower": "g_low"})
        if "g_high" not in region_df.columns and "ci_upper" in region_df.columns:
            region_df = region_df.rename(columns={"ci_upper": "g_high"})

        for _, r in region_df.iterrows():
            rows.append(
                {
                    "region": r["region"],
                    "g": r["g"],
                    "g_low": r["g_low"],
                    "g_high": r["g_high"],
                }
            )

    plot_df = pd.DataFrame(rows)
    # order: overall first, then sorted regions
    plot_df = pd.concat(
        [
            plot_df[plot_df["region"] == "overall"],
            plot_df[plot_df["region"] != "overall"].sort_values("region"),
        ],
        ignore_index=True,
    )

    n = len(plot_df)
    y = np.arange(n)

    fig_height = max(3.0, 0.6 * n)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    gs = plot_df["g"].values
    low = plot_df["g_low"].values
    high = plot_df["g_high"].values

    ax.hlines(y, low, high, linewidth=1.5)
    ax.scatter(gs, y, zorder=3)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["region"].values)
    ax.invert_yaxis()

    ax.set_xlabel("Effect size g (log10 titer)")
    ax.set_title(f"{vaccine}: {gene} {allele}")

    plt.tight_layout()
    fname = safe_filename(vaccine, gene, allele) + "_forest_regions.png"
    out_path = outdir / fname
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved forest plot: {out_path}")


# ------------------------------
# Main
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forest plots for significant HLA allele effects: overall + per region."
        )
    )
    parser.add_argument(
        "--effects",
        required=True,
        help="Path to effects_significant_FDR_*.tsv (overall significant effects).",
    )
    parser.add_argument(
        "--filtered_dir",
        required=True,
        help="Directory with filtered_vaccines/<vaccine>.xlsx files.",
    )
    parser.add_argument(
        "--hla",
        required=True,
        help="Path to combined_hla_out.xlsx.",
    )
    parser.add_argument(
        "--rare_genes",
        required=True,
        help="Path to hla_rare_alleles.txt.",
    )
    parser.add_argument(
        "--rare_first",
        required=False,
        default=None,
        help="Path to rare_alleles_first_field.tsv (optional, used if --field first).",
    )
    parser.add_argument(
        "--rare_second",
        required=False,
        default=None,
        help="Path to rare_alleles_second_field.tsv (optional, used if --field second).",
    )
    parser.add_argument(
        "--field",
        choices=["first", "second"],
        default="first",
        help="Allele resolution: first or second field (default: first).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for forest plots.",
    )
    parser.add_argument(
        "--min_carriers",
        type=int,
        default=5,
        help="Minimal number of carriers per region (default: 5).",
    )
    parser.add_argument(
        "--min_noncarriers",
        type=int,
        default=5,
        help="Minimal number of non-carriers per region (default: 5).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    effects_path = Path(args.effects)
    filtered_dir = Path(args.filtered_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading significant effects from: {effects_path}")
    eff = pd.read_csv(effects_path, sep="\t")

    required_cols = {"vaccine", "gene", "allele", "g"}
    missing = required_cols - set(eff.columns)
    if missing:
        raise ValueError(
            f"Significant effects file must contain columns: {required_cols}. Missing: {missing}"
        )

    # combined script earlier renamed ci_lower/ci_upper to g_low/g_high
    if "g_low" not in eff.columns and "ci_lower" in eff.columns:
        eff = eff.rename(columns={"ci_lower": "g_low"})
    if "g_high" not in eff.columns and "ci_upper" in eff.columns:
        eff = eff.rename(columns={"ci_upper": "g_high"})

    print("Loading HLA data...")
    hla = load_hla(args.hla)
    all_genes = get_hla_genes_from_columns(hla)

    print("Loading gene exclusion list (hla_rare_alleles.txt)...")
    excluded_genes = load_excluded_genes(args.rare_genes)
    allowed_genes = [g for g in all_genes if g not in excluded_genes]
    print(f"Genes kept in analysis: {len(allowed_genes)}")

    # rare alleles per field
    rare_alleles_path = args.rare_first if args.field == "first" else args.rare_second
    print(f"Loading rare alleles for field '{args.field}'...")
    excluded_alleles = load_excluded_alleles(rare_alleles_path)
    print(f"Alleles to exclude for this field: {len(excluded_alleles)}")

    # map vaccine -> set of (gene, allele) we need
    vacc2alleles: Dict[str, Set[Tuple[str, str]]] = {}
    for _, row in eff.iterrows():
        v = row["vaccine"]
        key = (row["gene"], row["allele"])
        vacc2alleles.setdefault(v, set()).add(key)

    all_region_effects = []

    for vacc_name, allele_set in vacc2alleles.items():
        vacc_file = filtered_dir / f"{vacc_name}.xlsx"
        if not vacc_file.exists():
            print(f"[WARN] Filtered vaccine file not found: {vacc_file}, skipping vaccine.")
            continue

        print(f"\n=== Vaccine: {vacc_name} ===")
        print(f"Loading phenotype from: {vacc_file}")
        pheno = load_vaccine_pheno_from_filtered(vacc_name, str(vacc_file))
        print(f"Phenotype samples with valid titers: {len(pheno)}")

        print("Merging phenotype with HLA...")
        merged = merge_pheno_hla(pheno, hla)
        print(f"Merged samples: {len(merged)}")

        print(f"Building long table ({args.field}-field)...")
        long_df = build_long_allele_table_for_vaccine(
            merged,
            field=args.field,
            allowed_genes=allowed_genes,
            excluded_alleles=excluded_alleles,
        )
        print(f"Haplotypes in long table: {len(long_df)}")

        # оставляем только нужные (gene, allele)
        long_df = long_df[
            long_df[["gene", "allele"]].apply(tuple, axis=1).isin(allele_set)
        ].copy()
        print(f"Haplotypes for significant alleles: {len(long_df)}")

        if long_df.empty:
            print("No haplotypes for significant alleles in this vaccine, skipping.")
            continue

        print("Computing region-specific effect sizes...")
        reg_eff = compute_effect_sizes_by_region(
            long_df,
            min_carriers=args.min_carriers,
            min_noncarriers=args.min_noncarriers,
        )
        if reg_eff.empty:
            print("No region-specific effects computed, skipping.")
            continue

        reg_eff.insert(0, "vaccine", vacc_name)
        all_region_effects.append(reg_eff)

    if not all_region_effects:
        print("\nNo region-specific effects were computed for any vaccine.")
        return

    region_effects = pd.concat(all_region_effects, ignore_index=True)

    # ----------------- Plotting per (vaccine, gene, allele) -----------------
    group_cols = ["vaccine", "gene", "allele"]
    eff_groups = eff.groupby(group_cols)

    print("\nPlotting forest plots (overall + regions)...")
    for (vaccine, gene, allele), overall_group in eff_groups:
        overall_row = overall_group.iloc[0]

        sub_reg = region_effects[
            (region_effects["vaccine"] == vaccine)
            & (region_effects["gene"] == gene)
            & (region_effects["allele"] == allele)
        ].copy()

        if sub_reg.empty:
            print(f"[WARN] No region-specific effects for {vaccine} {gene} {allele}, plotting overall only.")
        plot_forest_overall_plus_regions(
            overall_row=overall_row,
            region_df=sub_reg,
            outdir=outdir,
        )

    print("Done.")


if __name__ == "__main__":
    main()