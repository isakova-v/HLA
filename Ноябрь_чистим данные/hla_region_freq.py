#!/usr/bin/env python
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not found, p-values for significance will not be computed.")


def filter_rare_alleles(
    long_df: pd.DataFrame,
    min_freq: float,
    rare_out_path: str | None = None,
) -> pd.DataFrame:
    """
    Убираем из long_df аллели, у которых глобальная частота (по гену,
    по всей выборке) < min_freq.
    
    long_df: ожидаются колонки ['gene', 'allele', ...]
    Возвращает отфильтрованный long_df (только "частые" аллели).
    Если rare_out_path не None — сохраняет таблицу редких аллелей.
    """
    # Сколько гаплотипов данного (gene, allele) во всей выборке
    counts = (
        long_df
        .groupby(["gene", "allele"])
        .size()
        .reset_index(name="count_global")
    )

    # Сколько всего гаплотипов данного гена во всей выборке
    totals = (
        long_df
        .groupby("gene")
        .size()
        .reset_index(name="total_gene")
    )

    stats = counts.merge(totals, on="gene", how="left")
    stats["freq_global"] = stats["count_global"] / stats["total_gene"]

    rare_mask = stats["freq_global"] < min_freq
    rare = stats[rare_mask].copy()
    common = stats[~rare_mask].copy()

    # Опционально выпишем редкие аллели
    if rare_out_path is not None:
        os.makedirs(os.path.dirname(rare_out_path), exist_ok=True)
        rare.to_csv(rare_out_path, sep="\t", index=False)

    # Фильтруем long_df, оставляя только "частые" аллели
    filtered = long_df.merge(
        common[["gene", "allele"]],
        on=["gene", "allele"],
        how="inner",
    )
    return filtered

# ------------------------------
# 1. Allele normalisation utils
# ------------------------------

def normalize_allele_first_field(allele: str) -> str | None:
    """
    Convert raw allele string to first-field resolution.
    Example: 'HLA-A*25:01:01' -> 'HLA-A*25'
    Returns None for missing / invalid alleles.
    """
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None

    # Ensure has '*'
    if "*" not in allele:
        return None

    # Keep prefix (e.g. 'HLA-A')
    prefix, rest = allele.split("*", 1)
    # Handle things like '25:01:01'
    parts = rest.split(":")
    if len(parts) == 0 or parts[0] == "":
        return None

    return f"{prefix}*{parts[0]}"


def normalize_allele_second_field(allele: str) -> str | None:
    """
    Convert raw allele string to second-field resolution.
    Example: 'HLA-A*25:01:01' -> 'HLA-A*25:01'
    Returns None for missing / invalid alleles.
    """
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
        # only one field, second field == first field
        return f"{prefix}*{parts[0]}"
    return f"{prefix}*{parts[0]}:{parts[1]}"


# ------------------------------
# 2. Data preparation
# ------------------------------

def load_and_filter_pheno(pheno_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all_pheno_unrel.tsv and keep only rows with:
      - at least one 1 in *vaccine*_vaccine_info columns
      - at least one 1 in is_from_*region* columns
    Also derive a 'region' column from is_from_* columns.

    Возвращает:
        filtered, vaccine_cols
    """
    pheno = pd.read_csv(pheno_path, sep="\t")

    # Detect vaccine and region columns
    vaccine_cols = [c for c in pheno.columns if c.endswith("_vaccine_info")]
    region_cols = [c for c in pheno.columns if c.startswith("is_from_")]

    if not vaccine_cols:
        raise ValueError("No *_vaccine_info columns found in phenotype file.")
    if not region_cols:
        raise ValueError("No is_from_* region columns found in phenotype file.")

    has_vaccine = (pheno[vaccine_cols] == 1).any(axis=1)
    has_region = (pheno[region_cols] == 1).any(axis=1)

    filtered = pheno[has_vaccine & has_region].copy()

    region_col = filtered[region_cols].idxmax(axis=1)
    filtered["region"] = region_col.str.replace("is_from_", "", regex=False)

    return filtered, vaccine_cols


def load_and_merge_hla(
    pheno: pd.DataFrame,
    hla_path: str,
    vaccine_cols: list[str],
) -> pd.DataFrame:
    """
    Load combined_hla_out.xlsx, merge with filtered phenotype by:
        pheno['ZLIMS ID'] <-> hla['sample_id']
    Returns merged DataFrame with 'region', vaccine columns and all HLA-* columns.
    """
    hla = pd.read_excel(hla_path)

    if "sample_id" not in hla.columns:
        raise ValueError("combined_hla_out.xlsx must have column 'sample_id'.")

    if "ZLIMS ID" not in pheno.columns:
        raise ValueError("phenotype table must have column 'ZLIMS ID'.")

    pheno["ZLIMS ID"] = pheno["ZLIMS ID"].astype(str)
    hla["sample_id"] = hla["sample_id"].astype(str)

    merge_cols = ["ZLIMS ID", "region"] + vaccine_cols

    merged = pheno[merge_cols].merge(
        hla,
        left_on="ZLIMS ID",
        right_on="sample_id",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlapping samples between phenotype and HLA tables.")

    return merged


def get_hla_genes(df: pd.DataFrame) -> list[str]:
    """
    From columns like 'HLA-A_1', 'HLA-A_2', derive list of gene prefixes 'HLA-A', 'HLA-B', etc.
    """
    hla_cols = [c for c in df.columns if c.startswith("HLA-")]
    genes = sorted({c.rsplit("_", 1)[0] for c in hla_cols})
    return genes


# ------------------------------
# 3. Expand diploids to haplotypes
# ------------------------------

def build_long_allele_table(
    df: pd.DataFrame,
    field: str,
    vaccine_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert wide HLA columns to long table of haplotypes.

    long_df columns:
        ['sample_id', 'region', 'gene', 'allele', ...вакцинные_колонки...]
    """
    if field == "first":
        normalizer = normalize_allele_first_field
    elif field == "second":
        normalizer = normalize_allele_second_field
    else:
        raise ValueError("field must be 'first' or 'second'")

    if vaccine_cols is None:
        vaccine_cols = []

    genes = get_hla_genes(df)

    records = []
    for gene in genes:
        col1 = f"{gene}_1"
        col2 = f"{gene}_2"
        if col1 not in df.columns or col2 not in df.columns:
            continue

        base_cols = ["sample_id", "region"] + vaccine_cols
        sub = df[base_cols + [col1, col2]]

        for _, row in sub.iterrows():
            sample_id = row["sample_id"]
            region = row["region"]
            vacc_info = {c: row[c] for c in vaccine_cols}

            raw1 = row[col1]
            raw2 = row[col2]

            a1 = normalizer(raw1)
            if isinstance(raw2, str) and raw2.strip() == "-":
                a2 = a1
            else:
                a2 = normalizer(raw2)

            for a in (a1, a2):
                if a is not None:
                    rec = {
                        "sample_id": sample_id,
                        "region": region,
                        "gene": gene,
                        "allele": a,
                    }
                    rec.update(vacc_info)
                    records.append(rec)

    long_df = pd.DataFrame.from_records(records)
    return long_df


def compute_gene_region_infection_counts(
    long_df: pd.DataFrame,
    vaccine_cols: list[str],
) -> pd.DataFrame:
    """
    Считает для каждого (gene, region, infection) число уникальных аллелей,
    которые вошли в анализ (т.е. присутствуют в long_df).

    long_df должен содержать колонки:
        'gene', 'allele', 'region', а также vaccine_cols (0/1).
    """
    if not vaccine_cols:
        raise ValueError("vaccine_cols is empty – nothing to use as infections.")

    # Берём только нужное
    df = long_df[["gene", "allele", "region"] + vaccine_cols].copy()

    # Маппинг infection_col -> infection_name (убираем суффикс)
    infection_name = {
        c: c.replace("_vaccine_info", "") for c in vaccine_cols
    }

    # Переводим широкую таблицу вакцин в длинную: одна строка = один (gene, allele, region, infection)
    m = df.melt(
        id_vars=["gene", "allele", "region"],
        value_vars=vaccine_cols,
        var_name="infection_col",
        value_name="vaccinated",
    )

    m = m[m["vaccinated"] == 1].copy()
    if m.empty:
        # нет ни одного вакцинированного образца (после фильтра) – возвращаем пустую таблицу
        return pd.DataFrame(columns=["gene", "region", "infection", "n_alleles"])

    m["infection"] = m["infection_col"].map(infection_name)

    # Считаем число уникальных аллелей
    counts = (
        m.drop_duplicates(["gene", "allele", "region", "infection"])
        .groupby(["gene", "region", "infection"])
        .size()
        .reset_index(name="n_alleles")
    )

    return counts

# ------------------------------
# 4. Frequency & significance
# ------------------------------

def compute_freq_and_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute allele frequencies per region and chi-square p-values
    + FDR (Benjamini–Hochberg) correction across all (gene, allele).

    Returns:
      freq: DataFrame with columns
        ['gene', 'allele', 'region', 'count', 'freq', 'total_region',
         'p_value', 'p_fdr']
    """
    # Count haplotypes per (gene, allele, region)
    counts = (
        long_df
        .groupby(["gene", "allele", "region"])
        .size()
        .reset_index(name="count")
    )

    # Total haplotypes per (gene, region)
    totals = (
        long_df
        .groupby(["gene", "region"])
        .size()
        .reset_index(name="total_region")
    )

    # Merge frequencies
    freq = counts.merge(totals, on=["gene", "region"], how="left")
    freq["freq"] = freq["count"] / freq["total_region"]

    # Если scipy нет — только частоты, без статистики
    if not SCIPY_AVAILABLE:
        freq["p_value"] = np.nan
        freq["p_fdr"] = np.nan
        return freq

    p_values = {}
    regions = sorted(long_df["region"].unique())

    # Быстрый доступ к total per (gene, region)
    total_dict = (
        totals
        .set_index(["gene", "region"])["total_region"]
        .to_dict()
    )

    # 1) посчитать сырые p-values для каждого (gene, allele)
    for (gene, allele), sub in freq.groupby(["gene", "allele"]):
        allele_counts = []
        total_counts = []

        for r in regions:
            total = total_dict.get((gene, r), 0)
            total_counts.append(total)
            c = int(sub.loc[sub["region"] == r, "count"].sum())
            allele_counts.append(c)

        # Аллель только в одном регионе или нигде — тест бессмысленен
        if sum(allele_counts) == 0 or sum(x > 0 for x in allele_counts) <= 1:
            p_values[(gene, allele)] = np.nan
            continue

        # "Другие" гаплотипы
        others = [t - c for t, c in zip(total_counts, allele_counts)]

        # Если нет "others" (аллель фиксирован) — χ² не работает
        if sum(others) == 0:
            p_values[(gene, allele)] = np.nan
            continue

        # Убираем регионы без гаплотипов вообще
        valid_idx = [i for i, t in enumerate(total_counts) if t > 0]
        if len(valid_idx) < 2:
            p_values[(gene, allele)] = np.nan
            continue

        ac = [allele_counts[i] for i in valid_idx]
        ot = [others[i] for i in valid_idx]

        # Дегенератный случай — пропускаем
        if sum(ac) == 0 or sum(ot) == 0:
            p_values[(gene, allele)] = np.nan
            continue

        table = np.vstack([ac, ot])

        # Ещё одна защита: ни один столбец не должен иметь суммарно 0
        if np.any(table.sum(axis=0) == 0):
            p_values[(gene, allele)] = np.nan
            continue

        try:
            chi2, p, dof, exp = stats.chi2_contingency(table)
        except ValueError:
            p = np.nan

        p_values[(gene, allele)] = p

    # 2) Переложить сырые p-values обратно в freq
    freq["p_value"] = freq.apply(
        lambda row: p_values.get((row["gene"], row["allele"]), np.nan),
        axis=1,
    )

    # 3) FDR (Benjamini–Hochberg) по всем (gene, allele) сразу
    # Собираем уникальные p (по комбинациям gene+allele)
    pa_rows = (
        freq[["gene", "allele", "p_value"]]
        .drop_duplicates()
        .dropna(subset=["p_value"])
    )

    if pa_rows.empty:
        freq["p_fdr"] = np.nan
        return freq

    # Список (индекс строки в pa_rows, p)
    p_list = list(enumerate(pa_rows["p_value"].values))
    # Сортируем по p
    p_list.sort(key=lambda x: x[1])

    m = len(p_list)  # число тестов
    q_vals = np.zeros(m, dtype=float)

    # Шаг 1: прямой BH q_i = p_i * m / rank
    for rank, (idx, p) in enumerate(p_list, start=1):
        q_vals[rank - 1] = p * m / rank

    # Шаг 2: делаем q монотонно неубывающими при движении снизу вверх
    prev = 1.0
    for i in range(m - 1, -1, -1):
        if q_vals[i] > prev:
            q_vals[i] = prev
        prev = q_vals[i]
    q_vals = np.clip(q_vals, 0, 1)

    # Возвращаем обратно к (gene, allele)
    # pa_rows имеет свой индекс, нам нужно сопоставить
    pa_rows = pa_rows.reset_index(drop=True)
    pa_rows["p_fdr"] = q_vals

    # Словарь (gene, allele) -> p_fdr
    fdr_dict = {
        (row["gene"], row["allele"]): row["p_fdr"]
        for _, row in pa_rows.iterrows()
    }

    freq["p_fdr"] = freq.apply(
        lambda row: fdr_dict.get((row["gene"], row["allele"]), np.nan),
        axis=1,
    )

    return freq
# ------------------------------
# 5. Plotting
# ------------------------------

def plot_top_alleles(
    freq_df: pd.DataFrame,
    field_label: str,
    outdir: str,
    top_n: int = 10,
    p_threshold: float = 0.05,
):
    """
    For each gene, plot barplots of the top_n alleles showing strongest
    evidence of regional differences (smallest p-value).

    One figure per gene per resolution (field_label).
    """
    os.makedirs(outdir, exist_ok=True)

    genes = sorted(freq_df["gene"].unique())
    regions = sorted(freq_df["region"].unique())

    for gene in genes:
        sub = freq_df[freq_df["gene"] == gene].copy()

        # Select alleles
        if SCIPY_AVAILABLE:
            # Keep only alleles with a defined p-value
            sub_nonan = sub.dropna(subset=["p_value"])
            if sub_nonan.empty:
                continue
            # Take best alleles by p-value
            best = (
                sub_nonan
                .sort_values("p_value")
                .groupby(["gene", "allele"])
                .head(1)  # one row per allele after sorting
            )
            best = best[best["p_value"] <= p_threshold]
            if best.empty:
                # If nothing passes threshold, take top_n smallest p-values anyway
                best = (
                    sub_nonan
                    .sort_values("p_value")
                    .groupby(["gene", "allele"])
                    .head(1)[:top_n]
                )
        else:
            # If no scipy, just pick alleles with largest global variance in freq
            allele_stats = []
            for allele, sub_a in sub.groupby("allele"):
                # get frequencies across regions, fill missing with 0
                freqs = []
                for r in regions:
                    val = sub_a.loc[sub_a["region"] == r, "freq"]
                    freqs.append(val.iloc[0] if not val.empty else 0.0)
                allele_stats.append((allele, np.var(freqs)))
            allele_stats.sort(key=lambda x: x[1], reverse=True)
            selected_alleles = [a for a, _ in allele_stats[:top_n]]
            best = sub[sub["allele"].isin(selected_alleles)].copy()

        if best.empty:
            continue

        selected_alleles = best["allele"].unique()
        n_alleles = len(selected_alleles)

        # Plot
        fig, ax = plt.subplots(figsize=(max(8, n_alleles * 0.6), 5))

        x = np.arange(n_alleles)
        width = 0.8 / max(1, len(regions))

        for i, region in enumerate(regions):
            freqs = []
            for allele in selected_alleles:
                row = sub[(sub["allele"] == allele) & (sub["region"] == region)]
                if not row.empty:
                    freqs.append(row["freq"].iloc[0])
                else:
                    freqs.append(0.0)

            ax.bar(
                x + i * width,
                freqs,
                width=width,
                label=region,
            )

        ax.set_xticks(x + width * (len(regions) - 1) / 2)
        ax.set_xticklabels(selected_alleles, rotation=45, ha="right")
        ax.set_ylabel("Allele frequency (haplotypes)")
        title = f"{gene} – allele frequencies by region ({field_label})"
        if SCIPY_AVAILABLE:
            title += " (top by chi-square p-value)"
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        fname = os.path.join(outdir, f"{gene}_{field_label}_allele_freq.png")
        plt.savefig(fname, dpi=300)
        plt.close(fig)


# ------------------------------
# 6. Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare HLA allele frequencies between regions at "
            "first- and second-field resolution."
        )
    )
    parser.add_argument(
        "--pheno",
        required=True,
        help="Path to all_pheno_unrel.tsv",
    )
    parser.add_argument(
        "--hla",
        required=True,
        help="Path to combined_hla_out.xlsx",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of alleles per gene to plot (default: 10)",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.05,
        help="p-value threshold for significance (used if scipy is available, default: 0.05)",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading and filtering phenotype data...")
    pheno, vaccine_cols = load_and_filter_pheno(args.pheno)
    print(f"Phenotype samples after vaccine+region filter: {len(pheno)}")

    print("Loading and merging HLA data...")
    merged = load_and_merge_hla(pheno, args.hla, vaccine_cols=vaccine_cols)
    print(f"Merged samples: {len(merged)}")

    print("Building long table (first-field)...")
    long_first = build_long_allele_table(
        merged,
        field="first",
        vaccine_cols=vaccine_cols,
    )
    print(f"Haplotypes (first-field, before rare filter): {len(long_first)}")

    print("Filtering rare alleles (first-field)...")
    long_first = filter_rare_alleles(
        long_first,
        min_freq=0.01,
        rare_out_path=os.path.join(args.outdir, "rare_alleles_first_field.tsv"),
    )
    print(f"Haplotypes (first-field, after rare filter): {len(long_first)}")

    print("Computing frequencies and statistics (first-field)...")
    freq_first = compute_freq_and_stats(long_first)
    freq_first.to_csv(
        os.path.join(args.outdir, "allele_freq_first_field.tsv"),
        sep="\t",
        index=False,
    )

    print("Saving significant alleles (first-field)...")
    signif_first = freq_first[
        freq_first["p_value"].notna()
        & (freq_first["p_value"] < args.p_threshold)
    ].copy()
    signif_first.to_csv(
        os.path.join(args.outdir, "significant_alleles_first_field.tsv"),
        sep="\t",
        index=False,
    )

    print("Computing per-gene/region/infection allele counts (first-field)...")
    gene_reg_inf_first = compute_gene_region_infection_counts(
        long_first,
        vaccine_cols=vaccine_cols,
    )
    gene_reg_inf_first.to_csv(
        os.path.join(
            args.outdir,
            "allele_counts_gene_region_infection_first_field.tsv",
        ),
        sep="\t",
        index=False,
    )

    print("Plotting (first-field)...")
    plot_top_alleles(
        freq_first,
        field_label="first-field",
        outdir=os.path.join(args.outdir, "first_field"),
        top_n=args.top_n,
        p_threshold=args.p_threshold,
    )

    # ---------- SECOND FIELD ----------
    print("Building long table (second-field)...")
    long_second = build_long_allele_table(
        merged,
        field="second",
        vaccine_cols=vaccine_cols,
    )
    print(f"Haplotypes (second-field, before rare filter): {len(long_second)}")

    print("Filtering rare alleles (second-field)...")
    long_second = filter_rare_alleles(
        long_second,
        min_freq=0.01,
        rare_out_path=os.path.join(args.outdir, "rare_alleles_second_field.tsv"),
    )
    print(f"Haplotypes (second-field, after rare filter): {len(long_second)}")

    print("Computing frequencies and statistics (second-field)...")
    freq_second = compute_freq_and_stats(long_second)
    freq_second.to_csv(
        os.path.join(args.outdir, "allele_freq_second_field.tsv"),
        sep="\t",
        index=False,
    )

    print("Saving significant alleles (second-field)...")
    signif_second = freq_second[
        freq_second["p_value"].notna()
        & (freq_second["p_value"] < args.p_threshold)
    ].copy()
    signif_second.to_csv(
        os.path.join(args.outdir, "significant_alleles_second_field.tsv"),
        sep="\t",
        index=False,
    )

    print("Plotting (second-field)...")
    plot_top_alleles(
        freq_second,
        field_label="second-field",
        outdir=os.path.join(args.outdir, "second_field"),
        top_n=args.top_n,
        p_threshold=args.p_threshold,
    )

    print("Done.")


if __name__ == "__main__":
    main()