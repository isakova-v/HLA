#!/usr/bin/env python3
"""
Convert HLA-HD *_final.result.txt batch into a combined XLSX table
with the same schema as a given template (e.g., combined_hla_table.xlsx).

Usage:
    python hla_results_to_xlsx.py --in ./hla-hd_026_oct25_results --template ./combined_hla_table.xlsx --out ./combined_hla_out.xlsx
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd

def parse_hla_result_file(path: Path) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    for line in text:
        parts = re.split(r"[,\t]+", line.strip())
        if len(parts) < 2:
            continue
        locus = parts[0].strip()
        a1 = parts[1].strip() if len(parts) > 1 else "-"
        a2 = parts[2].strip() if len(parts) > 2 else "-"
        def norm(v: str) -> str:
            v = v.strip()
            if v in {"", "-", "Not typed", "not typed", "NOT TYPED"}:
                return "-"
            return v
        a1 = norm(a1)
        a2 = norm(a2)
        mapping[locus] = (a1, a2)
    return mapping

def build_table_from_folder(input_dir: Path, template_xlsx: Path) -> pd.DataFrame:
    template = pd.read_excel(template_xlsx)
    template_cols = list(template.columns)

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

    files = sorted(input_dir.glob("*_final.result.txt"))
    if not files:
        raise FileNotFoundError(f"No *_final.result.txt files found in: {input_dir}")

    rows: List[Dict[str, str]] = []
    for f in files:
        data = parse_hla_result_file(f)
        row = {c: "-" for c in template_cols}
        m = re.match(r"^(\d+)_final\.result\.txt$", f.name)
        sample_id = m.group(1) if m else f.stem.replace("_final.result", "")
        row["sample_id"] = sample_id

        for locus, (a1, a2) in data.items():
            pref = choose_prefix(locus)
            c1 = f"{pref}_1"
            c2 = f"{pref}_2"
            if c1 in row and c2 in row:
                row[c1] = a1
                row[c2] = a2
        rows.append(row)

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
