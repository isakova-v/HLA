#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

SICK_COLS = ["measles_sick", "rubella_sick", "mumps_sick", "HBV_sick"]


def main(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    # Проверим, что все нужные колонки есть
    missing_cols = [c for c in SICK_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"В TSV отсутствуют колонки: {missing_cols}")

    print("=== Проверка переболевших ===")
    print(f"Файл: {tsv_path}\n")

    # Проверка, что значения только 0 или 1
    for c in SICK_COLS:
        bad_values = df[~df[c].isin([0, 1])]
        if not bad_values.empty:
            print(f"❌ В колонке {c} есть значения, отличные от 0/1:")
            print(bad_values[[c]].head().to_string(index=False))
            return

    print("✔ Все значения в *_sick — только 0 или 1")

    # Ищем переболевших (любой sick == 1)
    sick_mask = (df[SICK_COLS].sum(axis=1) > 0)
    sick_df = df[sick_mask]

    if sick_df.empty:
        print("✅ В выборке нет переболевших — всё корректно.")
    else:
        print(f"❌ Найдены переболевшие: {len(sick_df)} образцов.")

        cols_to_show = []
        if "ZLIMS ID" in df.columns:
            cols_to_show.append("ZLIMS ID")
        cols_to_show += SICK_COLS

        print("\nПримеры переболевших:")
        print(sick_df[cols_to_show].head(50).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Проверка отсутствия переболевших в TSV")
    p.add_argument("--tsv", required=True, help="Путь к TSV")
    args = p.parse_args()

    main(args.tsv)