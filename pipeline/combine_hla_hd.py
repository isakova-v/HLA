#!/usr/bin/env python3
"""
Convert HLA-HD *_final.result.txt batch into a combined XLSX table
with the same schema as a given template (e.g., combined_hla_table.xlsx).

Expected filenames:
    000026034170_final.result.txt  (12 digits prefix)

Usage:
    python combine_hla_hd.py \
      --in ./hla-hd_026_jan26_results \
      --template ./combined_hla_table.xlsx \
      --out ./combined_hla_out.xlsx
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd


FNAME_RE = re.compile(r"^(?P<sid>\d{12})_final\.result\.txt$")


def parse_hla_result_file(path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parses *_final.result.txt lines. Supports tab/comma separators.

    Expected line formats:
      LOCUS<TAB>ALLELE1<TAB>ALLELE2
      LOCUS,ALLELE1,ALLELE2
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()

    def norm(v: str) -> str:
        v = v.strip()
        if v in {"", "-", ".", "Not typed", "not typed", "NOT TYPED", "NA", "NaN", "nan"}:
            return "-"
        return v

    for line in lines:
        parts = re.split(r"[,\t]+", line.strip())
        if len(parts) < 2:
            continue
        locus = parts[0].strip()
        a1 = norm(parts[1]) if len(parts) > 1 else "-"
        a2 = norm(parts[2]) if len(parts) > 2 else "-"
        mapping[locus] = (a1, a2)

    return mapping


def build_table_from_folder(input_dir: Path, template_xlsx: Path) -> pd.DataFrame:
    input_dir = input_dir.expanduser().resolve()
    template_xlsx = template_xlsx.expanduser().resolve()

    # Template schema
    template = pd.read_excel(template_xlsx)
    template_cols = list(template.columns)
    if "sample_id" not in template_cols:
        raise ValueError("Template XLSX must contain 'sample_id' column.")

    # Prefixes from template like HLA-A from HLA-A_1/HLA-A_2
    col_prefixes = [c[:-2] for c in template_cols if c.endswith("_1") or c.endswith("_2")]
    col_prefixes = list(dict.fromkeys(col_prefixes))

    def choose_prefix(locus: str) -> str:
        if f"HLA-{locus}" in col_prefixes:
            return f"HLA-{locus}"
        if locus in col_prefixes:
            return locus
        if locus.upper() in col_prefixes:
            return locus.upper()
        return f"HLA-{locus}"

    # Collect files: first non-recursive, then recursive fallback
    candidates = sorted(input_dir.glob("*_final.result.txt"))
    if not candidates:
        candidates = sorted(input_dir.rglob("*_final.result.txt"))

    # Keep only strictly matching names: 12digits_final.result.txt
    files: List[Path] = [p for p in candidates if FNAME_RE.match(p.name)]

    if not files:
        raise FileNotFoundError(
            f"No files matching '\\d{{12}}_final.result.txt' found in: {input_dir}"
        )

    rows: List[Dict[str, str]] = []

    for f in files:
        sid = FNAME_RE.match(f.name).group("sid")
        data = parse_hla_result_file(f)

        row = {c: "-" for c in template_cols}
        row["sample_id"] = sid

        for locus, (a1, a2) in data.items():
            pref = choose_prefix(locus)
            c1 = f"{pref}_1"
            c2 = f"{pref}_2"
            if c1 in row and c2 in row:
                row[c1] = a1
                row[c2] = a2

        rows.append(row)

    print(
        f"[INFO] input_dir={input_dir}\n"
        f"[INFO] result files found={len(files)} (strict name match)\n"
        f"[INFO] rows={len(rows)}"
    )

    return pd.DataFrame(rows)[template_cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_dir", required=True, help="Folder with *_final.result.txt files")
    ap.add_argument("--template", dest="template_xlsx", required=True, help="Template XLSX to copy schema/columns")
    ap.add_argument("--out", dest="out_xlsx", required=True, help="Output XLSX path")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    template_xlsx = Path(args.template_xlsx)
    out_xlsx = Path(args.out_xlsx)

    df = build_table_from_folder(input_dir, template_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"Wrote: {out_xlsx}  (rows={len(df)}, cols={df.shape[1]})")


if __name__ == "__main__":
    main()