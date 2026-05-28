#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def normalize_iedb_columns(df: pd.DataFrame, mhc_class: str) -> pd.DataFrame:
    """Support both old and new IEDB export formats."""
    df = df.copy()

    rename_map = {}

    # rank / percentile
    if "rank" not in df.columns:
        for cand in [
            "median binding percentile",
            "percentile_rank",
            "Rank",
            "rank percentile",
        ]:
            if cand in df.columns:
                rename_map[cand] = "rank"
                break

    # optional metadata used later or useful for debugging
    aliases = {
        "seq #": "seq_num",
        "peptide length": "length",
        "peptide index": "peptide_index",
    }
    if mhc_class == "I":
        aliases.update(
            {
                "netmhcpan_el score": "affinity_metric",
                "netmhcpan_el core": "core",
                "netmhcpan_el icore": "icore",
                "netmhcpan_el percentile": "model_percentile",
                "ic50": "affinity_metric",
            }
        )
    else:
        aliases.update(
            {
                "netmhciipan_el score": "affinity_metric",
                "netmhciipan_el core": "core",
                "netmhciipan_el percentile": "model_percentile",
                "score": "affinity_metric",
            }
        )

    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["allele", "rank"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"MHC-{mhc_class}: не найдены обязательные колонки {missing}. "
            f"Доступные колонки: {list(df.columns)}"
        )

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df



def build_summary(
    effects_path: Path,
    freq_path: Path,
    mhc_i_path: Path,
    mhc_ii_path: Path,
    out_path: Path,
    alpha: float = 0.05,
    use_fdr: bool = False,
    mhc_i_rank_threshold: float = 1.0,
    mhc_ii_rank_threshold: float = 10.0,
) -> pd.DataFrame:
    effects = pd.read_csv(effects_path, sep="\t")
    freq = pd.read_csv(freq_path, sep="\t")
    mhc_i = normalize_iedb_columns(pd.read_csv(mhc_i_path), "I")
    mhc_ii = normalize_iedb_columns(pd.read_csv(mhc_ii_path), "II")

    p_col = "p_val_fdr" if use_fdr else "p_val"
    if p_col not in effects.columns:
        raise ValueError(
            f"В effects не найдена колонка {p_col}. Есть: {list(effects.columns)}"
        )

    significant_alleles = effects.loc[effects[p_col] < alpha].copy()

    freq_wide = (
        freq.pivot_table(
            index=["gene", "allele"],
            columns="region",
            values="freq",
            aggfunc="first",
        )
        .reset_index()
    )
    freq_wide.columns.name = None

    overall_freq = (
        freq.groupby(["gene", "allele"], as_index=False)
        .agg(count_total=("count", "sum"), total_sample=("total_region", "sum"))
    )
    overall_freq["freq_overall"] = (
        overall_freq["count_total"] / overall_freq["total_sample"]
    )

    mhc_i_sig = mhc_i.loc[mhc_i["rank"] <= mhc_i_rank_threshold].copy()
    mhc_ii_sig = mhc_ii.loc[mhc_ii["rank"] <= mhc_ii_rank_threshold].copy()

    mhc_i_counts = (
        mhc_i_sig.groupby("allele", as_index=False)
        .size()
        .rename(columns={"size": "iedb_mhc_i_epitope_count"})
    )
    mhc_ii_counts = (
        mhc_ii_sig.groupby("allele", as_index=False)
        .size()
        .rename(columns={"size": "iedb_mhc_ii_epitope_count"})
    )

    iedb_counts = pd.merge(mhc_i_counts, mhc_ii_counts, on="allele", how="outer").fillna(0)
    iedb_counts["iedb_mhc_i_epitope_count"] = iedb_counts[
        "iedb_mhc_i_epitope_count"
    ].astype(int)
    iedb_counts["iedb_mhc_ii_epitope_count"] = iedb_counts[
        "iedb_mhc_ii_epitope_count"
    ].astype(int)
    iedb_counts["iedb_total_epitope_count"] = (
        iedb_counts["iedb_mhc_i_epitope_count"]
        + iedb_counts["iedb_mhc_ii_epitope_count"]
    )

    result = (
        significant_alleles.merge(overall_freq, on=["gene", "allele"], how="left")
        .merge(freq_wide, on=["gene", "allele"], how="left")
        .merge(iedb_counts, on="allele", how="left")
    )

    for col in [
        "iedb_mhc_i_epitope_count",
        "iedb_mhc_ii_epitope_count",
        "iedb_total_epitope_count",
    ]:
        result[col] = result[col].fillna(0).astype(int)

    preferred_base_cols = [
        "vaccine",
        "gene",
        "allele",
        "g",
        "g_low",
        "g_high",
        "se",
        "p_val",
        "p_val_fdr",
        "count_total",
        "total_sample",
        "freq_overall",
        "iedb_mhc_i_epitope_count",
        "iedb_mhc_ii_epitope_count",
        "iedb_total_epitope_count",
    ]
    base_cols = [c for c in preferred_base_cols if c in result.columns]
    region_cols = [c for c in result.columns if c not in base_cols]
    result = result[base_cols + region_cols]

    if "p_val_fdr" in result.columns:
        sort_cols = ["p_val_fdr", "p_val"]
    elif "p_val" in result.columns:
        sort_cols = ["p_val"]
    else:
        sort_cols = ["allele"]
    result = result.sort_values(sort_cols, kind="stable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(out_path, index=False)
    return result



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Сводная таблица по значимым HLA-аллелям с учетом нового формата IEDB."
    )
    p.add_argument("--effects", required=True, help="TSV с эффектами аллелей")
    p.add_argument("--freq", required=True, help="TSV с частотами аллелей по регионам")
    p.add_argument("--mhc1", required=True, help="CSV IEDB для MHC I")
    p.add_argument("--mhc2", required=True, help="CSV IEDB для MHC II")
    p.add_argument("--out", required=True, help="Выходной xlsx")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--use-fdr", action="store_true")
    p.add_argument("--mhc1-rank-threshold", type=float, default=1.0)
    p.add_argument("--mhc2-rank-threshold", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_summary(
        effects_path=Path(args.effects),
        freq_path=Path(args.freq),
        mhc_i_path=Path(args.mhc1),
        mhc_ii_path=Path(args.mhc2),
        out_path=Path(args.out),
        alpha=args.alpha,
        use_fdr=args.use_fdr,
        mhc_i_rank_threshold=args.mhc1_rank_threshold,
        mhc_ii_rank_threshold=args.mhc2_rank_threshold,
    )
    print("Готово.")
    print(f"Сохранен файл: {args.out}")
    print(f"Строк в таблице: {len(result)}")
