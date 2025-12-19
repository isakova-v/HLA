#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def parse_dups_kinship(val: str):
    """
    Парсит строку вида "0 / 0", "1 / 0", "0 / 2" -> (dup, kin).
    Если формат неожиданный, возвращает (None, None).
    """
    if pd.isna(val):
        return (None, None)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
    if not m:
        return (None, None)
    return int(m.group(1)), int(m.group(2))


def main(tsv_path, xlsx_path, min_coverage, group_col, mode):
    # --- load ---
    tsv = pd.read_csv(tsv_path, sep="\t")
    xlsx = pd.read_excel(xlsx_path)

    # --- normalize IDs ---
    for df in (tsv, xlsx):
        if "ZLIMS ID" not in df.columns:
            raise ValueError(f'В файле нет колонки "ZLIMS ID": {df.columns}')
        df["ZLIMS ID"] = df["ZLIMS ID"].astype(str).str.strip()

    filtered_ids = set(tsv["ZLIMS ID"])
    xlsx_map = xlsx.set_index("ZLIMS ID", drop=False)

    # --- sanity: all filtered IDs exist in xlsx ---
    missing_in_xlsx = sorted(filtered_ids - set(xlsx_map.index))
    if missing_in_xlsx:
        print("❌ Есть ID из TSV, которых нет в XLSX:")
        print(missing_in_xlsx[:50], "..." if len(missing_in_xlsx) > 50 else "")
    else:
        print("✅ Все ZLIMS ID из TSV найдены в XLSX")

    # select rows for filtered IDs
    filt_xlsx = xlsx_map.loc[list(filtered_ids)].copy()

    # ------------------------------------------------------------------
    # 1) Coverage check
    # ------------------------------------------------------------------
    if "Покрытие" not in filt_xlsx.columns:
        raise ValueError('В XLSX нет колонки "Покрытие"')

    # coverage might be numeric or string; coerce to numeric if possible
    cov = pd.to_numeric(filt_xlsx["Покрытие"], errors="coerce")
    filt_xlsx["_coverage_num"] = cov

    bad_cov = filt_xlsx[(cov.isna()) | (cov < min_coverage)]
    print("\n=== Проверка покрытия ===")
    print(f"Порог покрытия: {min_coverage}")

    if bad_cov.empty:
        print("✅ Все отфильтрованные образцы имеют хорошее покрытие")
    else:
        print(f"❌ Найдены отфильтрованные образцы с плохим покрытием: {len(bad_cov)}")
        print(bad_cov[["ZLIMS ID", "Покрытие"]].head(50).to_string(index=False))

    # ------------------------------------------------------------------
    # 2) Duplicates / kinship check
    # ------------------------------------------------------------------
    if "Дубли/ Родство" not in filt_xlsx.columns:
        raise ValueError('В XLSX нет колонки "Дубли/ Родство"')

    parsed = filt_xlsx["Дубли/ Родство"].apply(parse_dups_kinship)
    filt_xlsx["_dup_flag"] = parsed.apply(lambda x: x[0])
    filt_xlsx["_kin_flag"] = parsed.apply(lambda x: x[1])

    print("\n=== Проверка дубли/родство ===")
    print(f"Режим проверки: {mode}")

    if mode == "strict":
        # В строгом режиме считаем, что в отфильтрованных НЕ должно остаться
        # ни одного образца с ненулевым флагом дубликатов или родства.
        viol = filt_xlsx[
            (filt_xlsx["_dup_flag"].fillna(0) > 0) |
            (filt_xlsx["_kin_flag"].fillna(0) > 0)
        ]
        if viol.empty:
            print("✅ В отфильтрованном наборе нет образцов с ненулевым флагом дубли/родство")
        else:
            print(f"❌ В отфильтрованном наборе есть образцы с дубли/родством: {len(viol)}")
            print(viol[["ZLIMS ID", "Дубли/ Родство"]].head(50).to_string(index=False))

    elif mode == "by_group":
        # Этот режим полезен, если у вас есть ЯВНЫЙ столбец,
        # который задаёт ID группы родственников/дубликатов.
        # Тогда требование: в TSV ≤ 1 образец на группу.
        if group_col is None:
            raise ValueError("Для режима by_group нужно указать --group-col")

        if group_col not in filt_xlsx.columns:
            raise ValueError(f'В XLSX нет колонки "{group_col}"')

        grp_counts = (
            filt_xlsx
            .groupby(group_col)["ZLIMS ID"]
            .nunique()
            .sort_values(ascending=False)
        )
        viol_groups = grp_counts[grp_counts > 1]

        if viol_groups.empty:
            print(f"✅ В каждой группе '{group_col}' оставлен один образец")
        else:
            print(f"❌ В некоторых группах '{group_col}' осталось >1 образца: {len(viol_groups)}")
            print(viol_groups.head(50).to_string())

            # показываем сами ID для первых нарушенных групп
            bad_groups = viol_groups.index[:10]
            print("\nПримеры нарушенных групп и их ZLIMS ID:")
            for g in bad_groups:
                ids = filt_xlsx.loc[filt_xlsx[group_col] == g, "ZLIMS ID"].tolist()
                print(f"  {g}: {ids}")

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Проверка покрытия и фильтрации родственников/дубликатов по ZLIMS ID"
    )
    p.add_argument("--tsv", required=True, help="Путь к отфильтрованному TSV")
    p.add_argument("--xlsx", required=True, help="Путь к неотфильтрованному XLSX")
    p.add_argument("--min-coverage", type=float, default=30,
                   help="Минимально допустимое покрытие (по умолчанию 30)")
    p.add_argument("--mode", choices=["strict", "by_group"], default="strict",
                   help=("strict: в TSV не должно остаться флагов дубли/родство; "
                         "by_group: проверка 1 образец на группу по явному ID группы"))
    p.add_argument("--group-col", default=None,
                   help=("Название колонки в XLSX, которая задаёт ID группы родственников/дубликатов "
                         "(нужно для mode=by_group)"))

    args = p.parse_args()
    main(args.tsv, args.xlsx, args.min_coverage, args.group_col, args.mode)