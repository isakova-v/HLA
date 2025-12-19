#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np


REGION_COLS = [
    "is_from_Irkutsk",
    "is_from_Amur",
    "is_from_NiNo",
    "is_from_Kaliningrad",
]

# Строго под допустимые значения cohort_region
REGION_MAP = {
    "irkutsk": "is_from_Irkutsk",
    "amur": "is_from_Amur",
    "nizhniy novgorod": "is_from_NiNo",
    "kaliningrad": "is_from_Kaliningrad",
    # Crimea намеренно НЕ мапим на флаг, т.к. флага нет
}

ALLOWED_REGIONS = {"irkutsk", "kaliningrad", "crimea", "amur", "nizhniy novgorod"}


def norm_region(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def main(tsv_path, xlsx_path):
    tsv = pd.read_csv(tsv_path, sep="\t")
    xlsx = pd.read_excel(xlsx_path)

    # --- проверки колонок ---
    for c in ["ZLIMS ID"] + REGION_COLS:
        if c not in tsv.columns:
            raise ValueError(f'В TSV нет колонки "{c}"')

    for c in ["ZLIMS ID", "cohort_region"]:
        if c not in xlsx.columns:
            raise ValueError(f'В XLSX нет колонки "{c}"')

    # --- нормализация ID ---
    tsv["ZLIMS ID"] = tsv["ZLIMS ID"].astype(str).str.strip()
    xlsx["ZLIMS ID"] = xlsx["ZLIMS ID"].astype(str).str.strip()

    # --- нормализация флагов ---
    for c in REGION_COLS:
        tsv[c] = pd.to_numeric(tsv[c], errors="coerce").fillna(0).astype(int)
        bad = tsv[~tsv[c].isin([0, 1])]
        if not bad.empty:
            raise ValueError(f'В колонке {c} есть значения не 0/1')

    # --- merge cohort_region ---
    x_map = xlsx[["ZLIMS ID", "cohort_region"]].copy()
    x_map["cohort_region_norm"] = x_map["cohort_region"].apply(norm_region)

    df = tsv.merge(
        x_map,
        on="ZLIMS ID",
        how="left",
        validate="m:1"
    )

    if df["cohort_region"].isna().any():
        missing_ids = df.loc[df["cohort_region"].isna(), "ZLIMS ID"].tolist()
        print("❌ Для некоторых ZLIMS ID нет cohort_region в XLSX:")
        print(missing_ids[:50], "..." if len(missing_ids) > 50 else "")
        return

    # --- подсчет регионов по TSV ---
    total = len(df)
    print("=== Подсчёт регионов по TSV-флагам ===")
    for col in REGION_COLS:
        cnt = int(df[col].sum())
        name = col.replace("is_from_", "")
        pct = 100.0 * cnt / total if total else 0.0
        print(f"  {name:12s}: {cnt:5d} ({pct:5.1f}%)")

    df["_flags_sum"] = df[REGION_COLS].sum(axis=1)

    n_none = int((df["_flags_sum"] == 0).sum())
    n_multi = int((df["_flags_sum"] > 1).sum())
    print("\n--- Контроль TSV-разметки ---")
    print(f"Всего строк: {total}")
    print(f"Без региона по TSV (все 0): {n_none}")
    print(f"Сразу несколько регионов по TSV (сумма >1): {n_multi}")

    # --- ожидаемый флаг ---
    df["_expected_flag"] = df["cohort_region_norm"].map(REGION_MAP)

    # --- неожиданные значения cohort_region ---
    unexpected = df.loc[~df["cohort_region_norm"].isin(ALLOWED_REGIONS), "cohort_region_norm"]
    if not unexpected.empty:
        print("\n❌ Найдены неожиданные cohort_region (не из списка 5):")
        print(unexpected.value_counts().to_string())

    # --- проверка для 4 регионов ---
    in4 = df["_expected_flag"].notna()

    # правильный флаг = 1
    good_flag = pd.Series(False, index=df.index)
    for col in REGION_COLS:
        good_flag |= (df["_expected_flag"] == col) & (df[col] == 1)

    mismatch_in4 = df[in4 & ((df["_flags_sum"] != 1) | (~good_flag))]

    # --- проверка для Crimea ---
    is_crimea = df["cohort_region_norm"] == "crimea"
    mismatch_crimea = df[is_crimea & (df["_flags_sum"] != 0)]

    print("\n=== Проверка соответствия TSV ↔ XLSX ===")
    if mismatch_in4.empty:
        print("✅ Для Irkutsk/Amur/Nizhniy Novgorod/Kaliningrad соответствие идеальное.")
    else:
        print(f"❌ Несоответствия в 4 регионах: {len(mismatch_in4)}")
        print(mismatch_in4[["ZLIMS ID", "cohort_region"] + REGION_COLS]
              .head(50).to_string(index=False))

    if mismatch_crimea.empty:
        print("✅ Для Crimea все TSV-флаги корректно = 0.")
    else:
        print(f"❌ Для Crimea найдены строки с ненулевыми TSV-флагами: {len(mismatch_crimea)}")
        print(mismatch_crimea[["ZLIMS ID", "cohort_region"] + REGION_COLS]
              .head(50).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Счёт регионов и проверка TSV ↔ XLSX cohort_region")
    p.add_argument("--tsv", required=True, help="Путь к TSV")
    p.add_argument("--xlsx", required=True, help="Путь к XLSX")
    args = p.parse_args()

    main(args.tsv, args.xlsx)