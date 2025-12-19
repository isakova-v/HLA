#!/usr/bin/env python3
import argparse
import re
import pandas as pd
from collections import Counter, defaultdict

PLACEHOLDER_VALUES = {"-", "", "nan", "NaN", None}


def is_real_allele(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    return s not in PLACEHOLDER_VALUES


def normalize_to_field2(allele: str) -> str:
    """
    Reduce allele to 2-field resolution.
    Examples:
        HLA-A*66:01:01 -> HLA-A*66:01
        HLA-B*44:02 -> HLA-B*44:02
    """
    allele = allele.strip()

    if "*" in allele:
        prefix, rest = allele.split("*", 1)
    else:
        prefix, rest = None, allele

    fields = rest.split(":")
    rest2 = ":".join(fields[:2]) if len(fields) >= 2 else rest

    return f"{prefix}*{rest2}" if prefix else rest2


def detect_id_column(df: pd.DataFrame, preferred: str, candidates) -> str:
    """
    Pick preferred column if exists, else first match from candidates.
    """
    if preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Не нашёл колонку с id. Пробовал '{preferred}' и кандидаты {candidates}. "
        f"Доступные колонки: {list(df.columns)[:50]} ..."
    )


def main(
    path_in: str,
    path_pheno: str,
    out_counts: str,
    out_rare: str,
    out_by_region: str,
    id_col_excel: str,
    id_col_pheno: str,
):
    # =======================
    # LOAD FILES
    # =======================
    df = pd.read_excel(path_in)
    pheno = pd.read_csv(path_pheno, sep="\t")

    # =======================
    # DETECT / FIX ID COLS
    # =======================
    excel_id_candidates = ["sample_id", "ZLIMS ID", "zlms_id", "SampleID", "sampleid", "id"]
    pheno_id_candidates = ["sample_id", "ZLIMS ID", "zlms_id", "SampleID", "sampleid", "id"]

    id_excel = detect_id_column(df, id_col_excel, excel_id_candidates)
    id_pheno = detect_id_column(pheno, id_col_pheno, pheno_id_candidates)

    # унифицируем название
    df = df.rename(columns={id_excel: "sample_id"})
    pheno = pheno.rename(columns={id_pheno: "sample_id"})

    # =======================
    # FIND REGION COLUMNS
    # =======================
    region_cols = [c for c in pheno.columns if c.startswith("is_from_")]
    if not region_cols:
        raise ValueError("В TSV не найдено колонок is_from_*city*.")

    # список регионов в строке
    pheno["region_list"] = pheno[region_cols].apply(
        lambda row: [col.replace("is_from_", "") for col in region_cols if row[col] == 1],
        axis=1,
    )

    # фильтр: только если есть хотя бы один регион
    pheno = pheno[pheno["region_list"].map(len) > 0]

    # =======================
    # MERGE BY sample_id
    # =======================
    df = df.merge(pheno[["sample_id", "region_list"]], on="sample_id", how="inner")

    if df.empty:
        raise ValueError(
            "После merge по sample_id не осталось строк. "
            "Проверь совпадение id между Excel и TSV."
        )

    # =======================
    # EXTRACT HLA COLUMNS
    # =======================
    hla_cols = [c for c in df.columns if re.match(r"^HLA-[A-Za-z0-9]+_[12]$", c)]
    if not hla_cols:
        raise ValueError("Не найдено колонок HLA-*_1 / HLA-*_2.")

    # группировка по генам
    genes = {}
    for col in hla_cols:
        gene = col[:-2]
        genes.setdefault(gene, []).append(col)

    # =======================
    # OUTPUT STRUCTURES
    # =======================
    results = []
    rare_report_lines = []

    region_stats = defaultdict(lambda: defaultdict(Counter))
    total_stats = defaultdict(Counter)

    # =======================
    # PROCESS EACH GENE
    # =======================
    for gene, cols in sorted(genes.items()):
        alleles_all = []

        # --- общая выборка ---
        for col in cols:
            s = df[col]
            vals = [str(x).strip() for x in s if is_real_allele(x)]
            vals = [normalize_to_field2(v) for v in vals]
            alleles_all.extend(vals)

        unique_alleles = sorted(set(alleles_all))
        n_unique = len(unique_alleles)

        results.append({
            "gene": gene,
            "n_unique_alleles": n_unique,
            "columns_used": ",".join(cols)
        })

        if n_unique <= 10:
            rare_report_lines.append(f"{gene} ({n_unique} unique alleles):")
            for a in unique_alleles:
                rare_report_lines.append(f"  {a}")
            rare_report_lines.append("")

        # --- региональный подсчёт ---
        for _, row in df.iterrows():
            sample_regions = row["region_list"]
            if not sample_regions:
                continue

            sample_alleles = []
            for col in cols:
                x = row[col]
                if is_real_allele(x):
                    sample_alleles.append(normalize_to_field2(str(x).strip()))

            for ra in sample_alleles:
                total_stats[gene][ra] += 1
                for reg in sample_regions:
                    region_stats[reg][gene][ra] += 1

    # =======================
    # SAVE main counts
    # =======================
    res_df = pd.DataFrame(results).sort_values("n_unique_alleles", ascending=False)
    res_df.to_csv(out_counts, index=False)

    # =======================
    # SAVE rare report
    # =======================
    with open(out_rare, "w", encoding="utf-8") as f:
        if rare_report_lines:
            f.write("\n".join(rare_report_lines))
        else:
            f.write("Нет генов с 10 или менее уникальными аллелями.\n")

    # =======================
    # SAVE region allele counts
    # =======================
    rows = []
    for reg in sorted(region_stats.keys()):
        for gene in sorted(region_stats[reg].keys()):
            for allele, cnt in region_stats[reg][gene].items():
                rows.append({
                    "region": reg,
                    "gene": gene,
                    "allele_field2": allele,
                    "count": cnt
                })

    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(out_by_region, index=False)

    print(f"[OK] Gene counts → {out_counts}")
    print(f"[OK] Rare alleles → {out_rare}")
    print(f"[OK] Region allele counts → {out_by_region}")
    print(f"[INFO] Used id columns: Excel='{id_excel}', Pheno='{id_pheno}'")
    print(f"[INFO] Samples after merge: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count HLA alleles (2-field) overall and by region."
    )
    parser.add_argument("input", help="Path to combined_hla_out.xlsx")
    parser.add_argument("pheno", help="Path to all_pheno_unrel.tsv")
    parser.add_argument("--out-counts", default="hla_allele_counts.csv")
    parser.add_argument("--out-rare", default="hla_rare_alleles.txt")
    parser.add_argument("--out-by-region", default="hla_alleles_by_region.csv")

    # NEW: id columns
    parser.add_argument("--id-col-excel", default="sample_id",
                        help="ID column name in Excel (default: sample_id)")
    parser.add_argument("--id-col-pheno", default="sample_id",
                        help="ID column name in pheno TSV (default: sample_id)")

    args = parser.parse_args()

    main(
        args.input,
        args.pheno,
        args.out_counts,
        args.out_rare,
        args.out_by_region,
        args.id_col_excel,
        args.id_col_pheno,
    )