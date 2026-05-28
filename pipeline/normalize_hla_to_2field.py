#!/usr/bin/env python3
"""
Normalize HLA allele strings to 2-field resolution (e.g. HLA-A*03:01:01 -> HLA-A*03:01).

- Treats "-", "", "NA", "NULL", etc. as missing.
- By default normalizes all columns whose name starts with "HLA-".
- Writes a new XLSX next to the input, with suffix "_2field.xlsx".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MISSING = {"", "-", "NA", "N/A", "NONE", "NULL", "NAN"}


def normalize_hla_2field(val: Any) -> Any:
    """Return allele normalized to second field, or NaN for missing."""
    if val is None:
        return np.nan
    # pandas may give float NaN
    if isinstance(val, float) and np.isnan(val):
        return np.nan

    s = str(val).strip()
    if s == "" or s.strip().upper() in _MISSING:
        return np.nan

    # Typical format: HLA-A*03:01:01, HLA-DPB1*04:01:01G, etc.
    m = re.match(r"^(HLA-[A-Za-z0-9]+)\*([0-9]{2,3})(?::([0-9]{2,3}))?", s)
    if m:
        gene, f1, f2 = m.group(1), m.group(2), m.group(3)
        return f"{gene}*{f1}:{f2}" if f2 else f"{gene}*{f1}"

    # Fallback: keep only first two ':'-separated parts after '*'
    if "*" in s and ":" in s:
        pref, rest = s.split("*", 1)
        parts = rest.split(":")
        if len(parts) >= 2:
            return f"{pref}*{parts[0]}:{parts[1]}"

    # If something unexpected, return as-is (better than corrupting data)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input .xlsx file(s) or glob(s), e.g. data/*_with_HLA_filtered.xlsx",
    )
    ap.add_argument(
        "--hla-prefix",
        default="HLA-",
        help='Normalize columns starting with this prefix (default: "HLA-")',
    )
    args = ap.parse_args()

    # Expand globs
    files: list[Path] = []
    for item in args.inputs:
        expanded = list(Path().glob(item))
        if expanded:
            files.extend(expanded)
        else:
            files.append(Path(item))

    for in_path in files:
        in_path = in_path.resolve()
        if not in_path.exists():
            raise FileNotFoundError(in_path)

        df = pd.read_excel(in_path)

        hla_cols = [c for c in df.columns if str(c).startswith(args.hla_prefix)]
        if not hla_cols:
            print(f"[WARN] No HLA columns found in {in_path.name} (prefix={args.hla_prefix!r})")

        for c in hla_cols:
            df[c] = df[c].map(normalize_hla_2field)

        out_path = in_path.with_name(in_path.stem + "_2field.xlsx")
        df.to_excel(out_path, index=False)
        print(f"[OK] {in_path.name} -> {out_path.name}  (normalized cols: {len(hla_cols)})")


if __name__ == "__main__":
    main()
