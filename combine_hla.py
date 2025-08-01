import pandas as pd
import os
import glob
import argparse

def combine_hla_tables(input_folder: str, output_file: str = "combined_hla_table.xlsx"):
    # Ищем все подходящие файлы
    file_paths = glob.glob(os.path.join(input_folder, "T1K_*_genotype.tsv"))

    all_rows = []
    all_genes = set()

    for file_path in file_paths:
        sample_id = os.path.basename(file_path).split("_")[1]
        df = pd.read_csv(file_path, sep="\t", header = None)

        sample_data = {"sample_id": sample_id}
        for _, row in df.iterrows():
            gene = row[0]
            allele1 = row[2]
            allele2 = row[5] if row[5] != "." else "-"
            sample_data[f"{gene}_1"] = allele1
            sample_data[f"{gene}_2"] = allele2
            all_genes.add(gene)

        all_rows.append(sample_data)

    # Собираем итоговую таблицу
    final_df = pd.DataFrame(all_rows)

    final_columns = ["sample_id"]
    for gene in sorted(all_genes):
        final_columns.append(f"{gene}_1")
        final_columns.append(f"{gene}_2")

    final_df = final_df.reindex(columns=final_columns)

    # Сохраняем результат
    final_df.to_excel(output_file, index=False)
    print(f"Готово! Таблица сохранена в {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Объединение HLA-таблиц в одну Excel-таблицу.")
    parser.add_argument("input_folder", type=str, help="Путь к папке с файлами T1K_*_genotype.tsv")
    parser.add_argument("--output", type=str, default="combined_hla_table.xlsx", help="Имя выходного файла")
    args = parser.parse_args()

    combine_hla_tables(args.input_folder, args.output)
