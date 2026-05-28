from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MHC_I_RANK_MAX = 1.0
MHC_II_RANK_MAX = 10.0


def split_mhc2_allele(a: str):
    if pd.isna(a):
        return []
    a = str(a).strip()
    return [x.strip() for x in a.split("/")] if "/" in a else [a]


def effect_for_allele_string(allele: str, eff_map: dict[str, float]):
    if pd.isna(allele):
        return np.nan
    allele = str(allele).strip()
    if allele in eff_map:
        return eff_map[allele]
    parts = split_mhc2_allele(allele)
    vals = [eff_map[p] for p in parts if p in eff_map]
    return float(np.mean(vals)) if vals else np.nan


def pick_first_existing(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Не нашёл ни одну из колонок {candidates}. Колонки файла: {list(df.columns)}")
    return None


def normalize_iedb_columns(df: pd.DataFrame, mhc_class: str, eff_g_map: dict[str, float]) -> pd.DataFrame:
    """Приводит старый и новый IEDB output к общему формату.

    Поддерживает новый IEDB CSV, где есть, например:
    - peptide
n    - allele
    - median binding percentile
    - netmhcpan_el score / percentile
    - netmhciipan_el score / percentile
    """
    df = df.copy()

    allele_col = pick_first_existing(df, ["allele"])
    pep_col = pick_first_existing(df, ["peptide", "epitope", "peptideseq", "peptide_seq"])

    rank_candidates = [
        "rank",
        "percentile_rank",
        "median binding percentile",
        "netmhcpan_el percentile",
        "netmhciipan_el percentile",
    ]
    rank_col = pick_first_existing(df, rank_candidates)

    # affinity / score columns are optional, but useful in hover/export.
    score_col = pick_first_existing(
        df,
        [
            "score",
            "ic50",
            "netmhcpan_el score",
            "netmhciipan_el score",
            "predicted score",
        ],
        required=False,
    )

    core_col = pick_first_existing(
        df,
        ["core", "netmhcpan_el core", "netmhciipan_el core"],
        required=False,
    )
    icore_col = pick_first_existing(df, ["icore", "netmhcpan_el icore"], required=False)
    start_col = pick_first_existing(df, ["start"], required=False)
    end_col = pick_first_existing(df, ["end"], required=False)
    length_col = pick_first_existing(df, ["length", "peptide length"], required=False)
    seq_col = pick_first_existing(df, ["seq_num", "seq #"], required=False)
    peptide_index_col = pick_first_existing(df, ["peptide index", "peptide_index"], required=False)

    out = pd.DataFrame({
        "allele": df[allele_col].astype(str).str.strip(),
        "peptide": df[pep_col].astype(str).str.strip(),
        "rank": pd.to_numeric(df[rank_col], errors="coerce"),
    })

    if start_col is not None:
        out["start"] = pd.to_numeric(df[start_col], errors="coerce")
    if end_col is not None:
        out["end"] = pd.to_numeric(df[end_col], errors="coerce")
    if length_col is not None:
        out["length"] = pd.to_numeric(df[length_col], errors="coerce")
    if seq_col is not None:
        out["seq_num"] = pd.to_numeric(df[seq_col], errors="coerce")
    if peptide_index_col is not None:
        out["peptide_index"] = pd.to_numeric(df[peptide_index_col], errors="coerce")
    if core_col is not None:
        out["core"] = df[core_col].astype(str)
    if icore_col is not None:
        out["icore"] = df[icore_col].astype(str)

    if score_col is not None:
        out["affinity_metric"] = pd.to_numeric(df[score_col], errors="coerce")
        out["affinity_name"] = score_col
    else:
        out["affinity_metric"] = np.nan
        out["affinity_name"] = "score"

    if mhc_class == "I":
        out["g_effect"] = out["allele"].map(eff_g_map)
        rank_max = MHC_I_RANK_MAX
    else:
        out["g_effect"] = out["allele"].apply(lambda a: effect_for_allele_string(a, eff_g_map))
        rank_max = MHC_II_RANK_MAX

    out["mhc_class"] = mhc_class

    eps = 1e-12
    out["bind_strength"] = np.log10(rank_max / out["rank"].clip(lower=eps))
    return out


def prepare_viz_df(
    edges: pd.DataFrame,
    mhc_class: str | None = None,
    rank_max: float | None = None,
    min_effect_known: int = 1,
    top_epitopes: int = 30,
    top_alleles: int = 40,
) -> pd.DataFrame:
    df = edges.copy()
    if mhc_class is not None:
        df = df[df["mhc_class"] == mhc_class].copy()

    if rank_max is not None:
        df = df[df["rank"] < rank_max].copy()
    else:
        df = df[
            ((df["mhc_class"] == "I") & (df["rank"] < MHC_I_RANK_MAX)) |
            ((df["mhc_class"] == "II") & (df["rank"] < MHC_II_RANK_MAX))
        ].copy()

    df = df[df["g_effect"].notna()].copy()

    epi_counts = df.groupby("peptide")["allele"].nunique().sort_values(ascending=False)
    epi_keep = epi_counts[epi_counts >= min_effect_known].head(top_epitopes).index
    df = df[df["peptide"].isin(epi_keep)].copy()

    allele_counts = df.groupby("allele")["peptide"].nunique().sort_values(ascending=False)
    allele_keep = allele_counts.head(top_alleles).index
    df = df[df["allele"].isin(allele_keep)].copy()

    allele_effect = df.groupby("allele")["g_effect"].median().sort_values()
    df["allele"] = pd.Categorical(df["allele"], categories=allele_effect.index.tolist(), ordered=True)

    epi_order = df.groupby("peptide")["allele"].nunique().sort_values()
    df["peptide"] = pd.Categorical(df["peptide"], categories=epi_order.index.tolist(), ordered=True)

    return df


def epitope_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["mhc_class", "peptide"])
        .agg(
            n_alleles=("allele", "nunique"),
            g_median=("g_effect", "median"),
            g_mean=("g_effect", "mean"),
            bind_median=("bind_strength", "median"),
        )
        .reset_index()
        .sort_values(["mhc_class", "g_median"])
    )


def save_bubble_plot(df: pd.DataFrame, out_png: Path, title: str):
    import matplotlib.pyplot as plt

    if df.empty:
        return
    x_codes = df["allele"].cat.codes
    y_codes = df["peptide"].cat.codes
    plt.figure(figsize=(16, 10))
    sc = plt.scatter(x_codes, y_codes, s=20 + 80 * df["bind_strength"], c=df["g_effect"])
    plt.colorbar(sc, label="g_effect")
    plt.xticks(np.arange(df["allele"].cat.categories.size), df["allele"].cat.categories, rotation=90)
    plt.yticks(np.arange(df["peptide"].cat.categories.size), df["peptide"].cat.categories)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Adapted epitope↔allele visualization for new IEDB CSV output")
    p.add_argument("--mhc1", required=True, help="Path to IEDB MHC-I CSV")
    p.add_argument("--mhc2", required=True, help="Path to IEDB MHC-II CSV")
    p.add_argument("--effects", required=True, help="TSV with columns allele and g")
    p.add_argument("--outdir", default="out_iedb_viz", help="Output directory")
    p.add_argument("--top-epitopes", type=int, default=35)
    p.add_argument("--top-alleles", type=int, default=45)
    args = p.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_mhc1 = pd.read_csv(args.mhc1)
    df_mhc2 = pd.read_csv(args.mhc2)
    df_eff = pd.read_csv(args.effects, sep="\t")

    if "allele" not in df_eff.columns or "g" not in df_eff.columns:
        raise ValueError("В effects-файле ожидаю колонки allele и g")
    eff_g_map = dict(zip(df_eff["allele"].astype(str), pd.to_numeric(df_eff["g"], errors="coerce")))

    edges_i = normalize_iedb_columns(df_mhc1, "I", eff_g_map)
    edges_ii = normalize_iedb_columns(df_mhc2, "II", eff_g_map)
    edges = pd.concat([edges_i, edges_ii], ignore_index=True)

    viz_all = prepare_viz_df(edges, mhc_class=None, rank_max=None, top_epitopes=args.top_epitopes, top_alleles=args.top_alleles)
    viz_i = prepare_viz_df(edges, mhc_class="I", rank_max=MHC_I_RANK_MAX, top_epitopes=min(args.top_epitopes, 30), top_alleles=min(args.top_alleles, 35))
    viz_ii = prepare_viz_df(edges, mhc_class="II", rank_max=MHC_II_RANK_MAX, top_epitopes=min(args.top_epitopes, 30), top_alleles=min(args.top_alleles, 35))

    summ_all = epitope_effect_summary(viz_all)
    summ_i = epitope_effect_summary(viz_i)
    summ_ii = epitope_effect_summary(viz_ii)

    edges.to_csv(out_dir / "edges_all_normalized.tsv", sep="\t", index=False)
    viz_all.to_csv(out_dir / "edges_filtered_all.tsv", sep="\t", index=False)
    viz_i.to_csv(out_dir / "edges_filtered_mhcI.tsv", sep="\t", index=False)
    viz_ii.to_csv(out_dir / "edges_filtered_mhcII.tsv", sep="\t", index=False)
    summ_all.to_csv(out_dir / "epitope_summary_all.tsv", sep="\t", index=False)
    summ_i.to_csv(out_dir / "epitope_summary_mhcI.tsv", sep="\t", index=False)
    summ_ii.to_csv(out_dir / "epitope_summary_mhcII.tsv", sep="\t", index=False)

    save_bubble_plot(viz_all, out_dir / "bubble_all.png", "Epitope ↔ allele links (MHC I+II)")
    save_bubble_plot(viz_i, out_dir / "bubble_mhcI.png", "Epitope ↔ allele links (MHC I)")
    save_bubble_plot(viz_ii, out_dir / "bubble_mhcII.png", "Epitope ↔ allele links (MHC II)")

    with open(out_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"MHC-I raw: {df_mhc1.shape}\n")
        f.write(f"MHC-II raw: {df_mhc2.shape}\n")
        f.write(f"Effects: {df_eff.shape}\n")
        f.write(f"Edges normalized: {edges.shape}\n")
        f.write(f"Filtered all: {viz_all.shape}\n")
        f.write(f"Filtered MHC-I: {viz_i.shape}\n")
        f.write(f"Filtered MHC-II: {viz_ii.shape}\n")

    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
