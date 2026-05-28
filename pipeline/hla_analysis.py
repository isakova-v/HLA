import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from collections import Counter
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


import re

def normalize_allele(allele: str, level: int = 2) -> str:
    """
    Приводит аллель к формату GENE*XX или GENE*XX:YY в зависимости от level.
    Примеры:
        HLA-A*01:01:01 → A*01 или A*01:01
        '-' → '-'
    """
    if not isinstance(allele, str) or allele == "-":
        return allele
    pattern = r"HLA-([A-Z0-9]+)\*(\d{2})(?::(\d{2}))?"
    match = re.match(pattern, allele)
    if match:
        gene, group, protein = match.groups()
        if level == 1:
            return f"{gene}*{group}"
        elif level == 2 and protein:
            return f"{gene}*{group}:{protein}"
    return allele


def normalize_and_fill(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Нормализует все аллели и заменяет '-' значением из соседней колонки.
    """
    df_clean = df.copy()
    gene_columns = [col for col in df.columns if "_" in col]

    # Группируем по генам: A_1 и A_2, B_1 и B_2 и т.д.
    gene_prefixes = sorted(set(col.split("_")[0] for col in gene_columns))

    for gene in gene_prefixes:
        col1, col2 = f"{gene}_1", f"{gene}_2"
        if col1 in df_clean and col2 in df_clean:
            df_clean[col1] = df_clean[col1].apply(lambda x: normalize_allele(x, level))
            df_clean[col2] = df_clean[col2].apply(lambda x: normalize_allele(x, level))
            # Заполняем прочерки значением из соседней колонки
            df_clean[col1] = df_clean.apply(
                lambda row: row[col2] if row[col1] == "-" else row[col1], axis=1)
            df_clean[col2] = df_clean.apply(
                lambda row: row[col1] if row[col2] == "-" else row[col2], axis=1)

    return df_clean


def process_hla_dataframe(df: pd.DataFrame, id_col: str, level: int, group_label: str) -> pd.DataFrame:
    """
    Возвращает DataFrame вида (Gene, Allele, Group) с учётом гомозиготности.
    """
    df = normalize_and_fill(df, level)
    records = []

    for gene in sorted(set(col.split("_")[0] for col in df.columns if "_" in col)):
        col1, col2 = f"{gene}_1", f"{gene}_2"
        if col1 not in df.columns or col2 not in df.columns:
            continue
        for _, row in df.iterrows():
            a1, a2 = row[col1], row[col2]
            if a1 == a2:
                # гомозигота
                records.append((gene, a1, group_label))
                records.append((gene, a1, group_label))
            else:
                records.append((gene, a1, group_label))
                records.append((gene, a2, group_label))

    return pd.DataFrame(records, columns=["Gene", "Allele", "Group"])


def compute_and_plot_frequencies(combined_df, output_folder, control_label, test_label):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    freq_summary = []

    for gene in sorted(combined_df["Gene"].unique()):
        df_gene = combined_df[combined_df["Gene"] == gene]
        counts = df_gene.groupby(["Allele", "Group"]).size().unstack(fill_value=0)

        total_test = df_gene[df_gene["Group"] == test_label].shape[0]
        total_control = df_gene[df_gene["Group"] == control_label].shape[0]

        # χ²-тест для каждого аллеля отдельно
        for allele in counts.index:
            test_count = counts.loc[allele].get(test_label, 0)
            control_count = counts.loc[allele].get(control_label, 0)
            contingency = [
                [test_count, total_test - test_count],
                [control_count, total_control - control_count]
            ]
            try:
                chi2, p, _, _ = chi2_contingency(contingency)
                results.append({
                    "Gene": gene,
                    "Allele": allele,
                    "p_value": p
                })
            except Exception:
                continue

        # Частоты аллелей
        freqs = counts.div(counts.sum(axis=0), axis=1).fillna(0)

        # Гистограмма
        freqs_plot = freqs.loc[freqs[control_label].sort_values(ascending=False).index]
        freqs_plot.plot(kind="bar", figsize=(10, 6))
        plt.title(f"{gene} allele frequencies")
        plt.xlabel("Allele")
        plt.ylabel("Frequency")
        plt.legend(title="Group")
        plt.tight_layout()
        plt.annotate(f"n(control)={total_control}, n(test)={total_test}", xy=(0.95, 0.95),
                     xycoords='axes fraction', ha='right', va='top', fontsize=10)
        plt.savefig(os.path.join(output_folder, f"{gene}_frequencies.png"))
        plt.close()

        # Таблица частот
        temp_df = freqs_plot.reset_index().melt(id_vars="Allele", var_name="Group", value_name="Frequency")
        temp_df["Gene"] = gene
        freq_summary.append(temp_df)

    # Сохраняем таблицу частот
    freq_summary_df = pd.concat(freq_summary, ignore_index=True)
    freq_summary_df.to_csv(os.path.join(output_folder, "allele_frequencies.csv"), index=False)

    # χ²-тест с поправкой
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        reject, pvals_corrected, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["p_corrected"] = pvals_corrected
        results_df["Significant_corrected"] = reject
        results_df = results_df.sort_values("p_corrected")
        results_df.to_csv(os.path.join(output_folder, "significant_alleles.csv"), index=False)
    else:
        print("Нет аллелей с достаточными данными для χ²-теста.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, help="Файл с результатами опроса пациентов")
    parser.add_argument("--survey_id_col", required=True, help="Название колонки с ID в опросе")
    parser.add_argument("--hla", required=True, help="Файл с HLA-генами пациентов")
    parser.add_argument("--hla_id_col", required=True, help="Название колонки с ID пациента в HLA-файле")
    parser.add_argument("--control", required=True, help="Контрольный CSV файл с HLA-типированием")
    parser.add_argument("--output", required=True, help="Папка для сохранения графиков и результатов")
    parser.add_argument("--allele_field", type=int, choices=[1, 2], default=1, help="Уровень агрегации аллеля: 1 или 2 поле")

    args = parser.parse_args()

    survey_df = pd.read_excel(args.survey)
    hla_df = pd.read_excel(args.hla)
    control_df = pd.read_csv(args.control, sep="\t")

    test_ids = set(survey_df[args.survey_id_col].dropna())
    hla_df = hla_df[hla_df[args.hla_id_col].isin(test_ids)]
    control_df = control_df[control_df[args.hla_id_col].notna()]

    test_alleles = process_hla_dataframe(hla_df, args.hla_id_col, args.allele_field, "test")
    control_alleles = process_hla_dataframe(control_df, args.hla_id_col, args.allele_field, "control")

    combined_df = pd.concat([test_alleles, control_alleles], ignore_index=True)
    compute_and_plot_frequencies(combined_df, args.output, "control", "test")
