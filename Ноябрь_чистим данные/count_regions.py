#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

REGION_COLS = [
    "is_from_Irkutsk",
    "is_from_Amur",
    "is_from_NiNo",
    "is_from_Kaliningrad",
]

def main(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    # Проверим, что колонки есть
    missing = [c for c in REGION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"В TSV отсутствуют колонки регионов: {missing}")

    print("=== Подсчёт образцов по регионам ===")
    print(f"Файл: {tsv_path}\n")

    # Проверка значений 0/1
    for c in REGION_COLS:
        bad = df[~df[c].isin([0, 1])]
        if not bad.empty:
            print(f"❌ В колонке {c} есть значения, отличные от 0/1.")
            print(bad[[c]].head(10).to_string(index=False))
            return

    # Сколько по каждому региону
    counts = df[REGION_COLS].sum(axis=0).astype(int)
    total = len(df)

    print("Кол-во образцов по регионам:")
    for col, cnt in counts.items():
        region_name = col.replace("is_from_", "")
        pct = 100.0 * cnt / total if total else 0.0
        print(f"  {region_name:12s}: {cnt:5d}  ({pct:5.1f}%)")

    # Доп. контроль качества разметки
    flags_per_row = df[REGION_COLS].sum(axis=1)

    n_multi = int((flags_per_row > 1).sum())
    n_none  = int((flags_per_row == 0).sum())

    print("\n--- Контроль разметки ---")
    print(f"Всего строк: {total}")
    print(f"Без региона (все 0): {n_none}")
    print(f"Сразу несколько регионов (сумма >1): {n_multi}")

    if n_multi > 0:
        cols_to_show = ["ZLIMS ID"] if "ZLIMS ID" in df.columns else []
        cols_to_show += REGION_COLS
        print("\nПримеры строк с несколькими регионами:")
        print(df.loc[flags_per_row > 1, cols_to_show].head(20).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Подсчёт числа образцов из каждого региона в TSV")
    p.add_argument("--tsv", required=True, help="Путь к TSV")
    args = p.parse_args()

    main(args.tsv)