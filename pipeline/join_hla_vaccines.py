#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Join HLA alleles from combined_hla_out.xlsx into vaccine*.xlsx files.

What it does (per vaccine file):
1) Reads vaccine file and keeps:
   - cohort_region, age, sex
   - <vaccine>*ME_ml
   - <vaccine>*_result
   - Years_after_<vaccine>_vaccine
   - <vaccine>*_result_multy
2) Inner-joins with HLA table on: vaccine.zlims_id == hla.sample_id
3) Drops HLA genes (both _1/_2 columns) that have < 10 distinct alleles in the merged dataset
4) Drops patients (rows) that contain any "rare" allele:
   allele frequency < 10 occurrences within that allele column in the merged dataset

Usage example:
  python join_hla_vaccines.py \
    --hla combined_hla_out.xlsx \
    --vaccines "HBV.xlsx" "diphtheria.xlsx" \
    --outdir merged

Or with glob:
  python join_hla_vaccines.py --hla combined_hla_out.xlsx --glob "*vaccine*.xlsx" --outdir merged
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)  # Excel float IDs like 12345.0
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return s


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of columns found: {candidates}")


def detect_vaccine_cols(df: pd.DataFrame, vaccine: str) -> dict:
    v = vaccine.lower()

    def find_first(pred):
        for c in df.columns:
            if pred(c):
                return c
        return None

    years_col = None
    for c in df.columns:
        if c.lower() == f"years_after_{v}_vaccine":
            years_col = c
            break

    me_col = find_first(lambda c: v in c.lower() and c.lower().endswith("me_ml"))
    result_col = find_first(lambda c: v in c.lower() and c.lower().endswith("_result") and "multy" not in c.lower())
    result_multy_col = find_first(lambda c: v in c.lower() and c.lower().endswith("_result_multy"))

    missing = [k for k, val in {
        "ME_ml": me_col,
        "result": result_col,
        "Years_after": years_col,
        "result_multy": result_multy_col,
    }.items() if val is None]

    return {
        "me": me_col,
        "result": result_col,
        "years_after": years_col,
        "result_multy": result_multy_col,
        "missing": missing,
    }


def hla_allele_columns(hla_df: pd.DataFrame) -> list[str]:
    return [c for c in hla_df.columns if c != "sample_id"]


def gene_key(col: str) -> str:
    # HLA-A_1 -> HLA-A ; HLA-DPA1_2 -> HLA-DPA1
    return col.rsplit("_", 1)[0]


def filter_genes_by_diversity(df: pd.DataFrame, allele_cols: list[str], min_unique_alleles: int = 10) -> tuple[pd.DataFrame, list[str]]:
    genes: dict[str, list[str]] = {}
    for c in allele_cols:
        genes.setdefault(gene_key(c), []).append(c)

    keep_cols: list[str] = []
    dropped_genes: list[str] = []

    for g, cols in genes.items():
        uniq = pd.unique(pd.concat([df[c] for c in cols], axis=0).dropna().astype(str))
        if len(uniq) >= min_unique_alleles:
            keep_cols.extend(cols)
        else:
            dropped_genes.append(g)

    base_cols = [c for c in df.columns if c not in allele_cols]
    return df[base_cols + keep_cols].copy(), dropped_genes


def filter_rows_with_rare_alleles(df: pd.DataFrame, allele_cols: list[str], min_count: int = 10) -> tuple[pd.DataFrame, int]:
    # Remove rows where ANY allele in allele_cols occurs < min_count times in the dataset (per column).
    rare_mask = pd.Series(False, index=df.index)
    for c in allele_cols:
        counts = df[c].astype(str).replace("nan", np.nan).value_counts(dropna=True)
        rare_alleles = set(counts[counts < min_count].index)
        rare_mask |= df[c].astype(str).isin(rare_alleles)
    removed = int(rare_mask.sum())
    return df.loc[~rare_mask].copy(), removed


def process_one(vaccine_path: str, hla: pd.DataFrame, outdir: str, min_unique_alleles: int, min_allele_count: int) -> dict:
    vp = Path(vaccine_path)
    vaccine_name = vp.stem  # e.g. HBV, diphtheria

    vdf = pd.read_excel(vp)

    id_col = pick_col(vdf, ["zlims_id", "ZLIMS ID", "ZLIMS_ID", "zlims id"])
    vdf[id_col] = normalize_id(vdf[id_col])

    base_cols = [c for c in ["cohort_region", "age", "sex"] if c in vdf.columns]
    vcols = detect_vaccine_cols(vdf, vaccine_name)

    keep_vaccine_cols = [vcols["me"], vcols["result"], vcols["years_after"], vcols["result_multy"]]
    keep_vaccine_cols = [c for c in keep_vaccine_cols if c is not None]

    keep_cols = [id_col] + base_cols + keep_vaccine_cols
    out = vdf[keep_cols].copy()

    hla_alleles = hla_allele_columns(hla)

    merged = out.merge(hla, left_on=id_col, right_on="sample_id", how="inner")

    merged2, dropped_genes = filter_genes_by_diversity(merged, hla_alleles, min_unique_alleles=min_unique_alleles)
    kept_allele_cols = [c for c in merged2.columns if c in hla_alleles]

    merged3, removed_rows = filter_rows_with_rare_alleles(merged2, kept_allele_cols, min_count=min_allele_count)

    out_path = Path(outdir) / f"{vaccine_name}_with_HLA_filtered.xlsx"
    merged3.to_excel(out_path, index=False)

    return {
        "vaccine_file": vp.name,
        "vaccine_name": vaccine_name,
        "input_rows": len(vdf),
        "merged_rows": len(merged),
        "rows_after_filters": len(merged3),
        "dropped_genes_count": len(dropped_genes),
        "rare_allele_rows_removed": removed_rows,
        "missing_expected_vaccine_cols": ", ".join(vcols["missing"]) if vcols["missing"] else "",
        "saved_to": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hla", required=True, help="Path to combined_hla_out.xlsx")
    ap.add_argument("--vaccines", nargs="*", default=[], help="Explicit list of vaccine xlsx files")
    ap.add_argument("--glob", default="", help='Glob pattern for vaccine files, e.g. "*vaccine*.xlsx"')
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--min_unique_alleles", type=int, default=10, help="Min distinct alleles per gene to keep the gene (default: 10)")
    ap.add_argument("--min_allele_count", type=int, default=10, help="Min allele frequency to keep a row (default: 10)")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    hla = pd.read_excel(args.hla)
    if "sample_id" not in hla.columns:
        raise KeyError("HLA file must contain column 'sample_id'")
    hla["sample_id"] = normalize_id(hla["sample_id"])

    vaccine_paths = list(args.vaccines)
    if args.glob:
        vaccine_paths += [str(p) for p in Path(".").glob(args.glob)]
    vaccine_paths = list(dict.fromkeys(vaccine_paths))  # dedupe while preserving order

    if not vaccine_paths:
        raise SystemExit("No vaccine files provided. Use --vaccines ... or --glob ...")

    rows = []
    for vp in vaccine_paths:
        rows.append(process_one(vp, hla, args.outdir, args.min_unique_alleles, args.min_allele_count))

    summary = pd.DataFrame(rows)
    summary_path = Path(args.outdir) / "summary.xlsx"
    summary.to_excel(summary_path, index=False)
    print("Done. Summary:", summary_path)


if __name__ == "__main__":
    main()
