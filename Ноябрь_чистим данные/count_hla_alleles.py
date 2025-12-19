#!/usr/bin/env python3
import argparse
import re
import pandas as pd

PLACEHOLDER_VALUES = {"-", "", "nan", "NaN", None}


def is_real_allele(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    return s not in PLACEHOLDER_VALUES


def normalize_to_field2(allele: str) -> str:
    """
    Приводим формат к уровню 2 поля.
    Примеры:
        HLA-A*01:01:03 -> HLA-A*01:01
        HLA-B*44:02 -> HLA-B*44:02
        B*44:02:01 -> B*44:02
        44:02:01 -> 44:02
    """
    allele = allele.strip()

    # вычленяем часть после “*”
    if "*" in allele:
        prefix, rest = allele.split("*", 1)
    else:
        # нет префикса, просто режем двоеточия
        prefix, rest = None, allele

    fields = rest.split(":")
    if len(fields) >= 2:
        rest2 = ":".join(fields[:2])
    else:
        rest2 = rest  # если вдруг только одно поле

    return f"{prefix}*{rest2}" if prefix else rest2


def main(path_in: str, path_pheno: str, out_counts: str, out_rare: str, out_by_region: str):
    # =======================
    # LOAD FILES
    # =======================
    df = pd.read_excel(path_in)

    pheno = pd.read_csv(path_pheno, sep="\t")

    # -----------------------
    # Ищем городские колонки
    # -----------------------
    region_cols = [c for c in pheno.columns if c.startswith("is_from_")]
    if not region_cols:
        raise ValueError("В TSV не найдено колонок is_from_*city*.")

    # -----------------------
    # Фильтруем только образцы,
    # которые принадлежат хотя бы 1 региону
    # -----------------------
    pheno["region_list"] = pheno[region_cols].apply(
        lambda row: [col.replace("is_from_", "") for col in region_cols if row[col] == 1],
        axis=1,
    )
    pheno = pheno[pheno["region_list"].map(len) > 0]

    # оставляем ключевой идентификатор
    if "ZLIMS ID" not in df.columns:
        raise ValueError("Ожидается колонка 'ZLIMS ID' в Excel.")

    df = df.merge(pheno[["ZLIMS ID", "region_list"]], on="ZLIMS ID", how="inner")

    # =======================
    # EXTRACT HLA COLUMNS
    # =======================
    hla_cols = [c for c in df.columns if re.match(r"^HLA-[A-Za-z0-9]+_[12]$", c)]
    if not hla_cols:
        raise ValueError("Не найдено колонок HLA-*_1 / HLA-*_2.")

    # группировка по генам
    genes = {}
    for col in hla_cols:
        gene = col[:-2]  # обрезаем _1/_2
        genes.setdefault(gene, []).append(col)

    # =======================
    # OUTPUTS
    # =======================
    results = []
    rare_report_lines = []

    # структура для региональной статистики:
    # region → gene → Counter(allele → count)
    from collections import Counter, defaultdict

    region_stats = defaultdict(lambda: defaultdict(Counter))
    total_stats = defaultdict(Counter)

    # =======================
    # PROCESS EACH GENE
    # =======================
    for gene, cols in sorted(genes.items()):
        alleles_all = []

        # --- Общая выборка по этому гену ---
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

        # отчёт по редким
        if n_unique <= 10:
            rare_report_lines.append(f"{gene} ({n_unique} unique alleles):")
            for a in unique_alleles:
                rare_report_lines.append(f"  {a}")
            rare_report_lines.append("")

        # ------------------------------
        # РЕГИОНАЛЬНЫЙ ПОДСЧЁТ
        # ------------------------------
        for idx, row in df.iterrows():
            sample_regions = row["region_list"]
            if not sample_regions:
                continue

            # собираем аллели этого образца по гену
            sample_alleles = []
            for col in cols:
                x = row[col]
                if is_real_allele(x):
                    sample_alleles.append(normalize_to_field2(str(x).strip()))

            # фиксируем только валидные
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
    # SAVE rare allele report
    # =======================
    with open(out_rare, "w", encoding="utf-8") as f:
        if rare_report_lines:
            f.write("\n".join(rare_report_lines))
        else:
            f.write("Нет генов с 10 или менее уникальными аллелями.\n")

    # =======================
    # SAVE region counts
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count HLA alleles aggregated to field-2 and optionally by region."
    )
    parser.add_argument("input", help="Path to combined_hla_out.xlsx")
    parser.add_argument("pheno", help="Path to all_pheno_unrel.tsv")
    parser.add_argument("--out-counts", default="hla_allele_counts.csv")
    parser.add_argument("--out-rare", default="hla_rare_alleles.txt")
    parser.add_argument("--out-by-region", default="hla_alleles_by_region.csv")

    args = parser.parse_args()
    main(args.input, args.pheno, args.out_counts, args.out_rare, args.out_by_region)