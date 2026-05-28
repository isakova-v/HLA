#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


VACCINE_FILTER_COLS = {
    "measles": "measles_MASK_filter",
    "rubella": "rubella_MASK_filter",
    "diphtheria": "diphtheria_MASK_filter",
    "HBV": "HBV_MASK_filter",
}


def read_unrelated_ids(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        ids.add(s)
    return ids


def find_zlims_col(df: pd.DataFrame) -> str:
    # ожидаем строго "ZLIMS ID", но на всякий случай — мягкий поиск
    if "ZLIMS ID" in df.columns:
        return "ZLIMS ID"
    norm_map = {c.strip().lower().replace("_", " ").replace("  ", " "): c for c in df.columns}
    key = "zlims id"
    if key in norm_map:
        return norm_map[key]
    raise KeyError(
        "Не нашёл колонку 'ZLIMS ID' в pheno_clean. "
        f"Первые колонки: {list(df.columns)[:20]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pheno", required=True, help="pheno_clean.tsv path")
    ap.add_argument("--unrelated", required=True, help="list_unrelated.txt path")
    ap.add_argument("--outdir", required=True, help="Output folder for XLSX files")
    args = ap.parse_args()

    pheno_path = Path(args.pheno).expanduser().resolve()
    unrelated_path = Path(args.unrelated).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    unrelated_ids = read_unrelated_ids(unrelated_path)

    df = pd.read_csv(pheno_path, sep="\t", dtype=str)
    zlims_col = find_zlims_col(df)

    # 1) фильтрация по list_unrelated.txt
    df[zlims_col] = df[zlims_col].astype(str).str.strip()
    df_filt = df[df[zlims_col].isin(unrelated_ids)].copy()

    # проверим, что нужные колонки есть
    missing = [c for c in VACCINE_FILTER_COLS.values() if c not in df_filt.columns]
    if missing:
        raise KeyError(f"В pheno_clean нет колонок: {missing}")

    # 2) создаём файлы по вакцинам
    for vaccine, col in VACCINE_FILTER_COLS.items():
        mask = df_filt[col].astype(str).str.strip().str.upper().eq("PASS")
        out_df = df_filt[mask].copy()

        out_path = outdir / f"{vaccine}.xlsx"
        out_df.to_excel(out_path, index=False)
        print(f"Wrote {out_path}  (rows={len(out_df)})")

    print(f"Done. Base filtered rows (by unrelated): {len(df_filt)}")


if __name__ == "__main__":
    main()