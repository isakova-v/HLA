
import pandas as pd
import sys
from pathlib import Path

MIN_ALLELES = 10

def filter_genes_by_allele_count(df: pd.DataFrame, min_alleles: int = 10):
    hla_cols = [c for c in df.columns if c.startswith("HLA-")]
    keep_cols = []

    for col in hla_cols:
        n_alleles = df[col].dropna().nunique()
        if n_alleles >= min_alleles:
            keep_cols.append(col)

    base_cols = [c for c in df.columns if not c.startswith("HLA-")]
    return df[base_cols + keep_cols], keep_cols


def process_file(path: Path):
    df = pd.read_excel(path)
    df_filt, kept_genes = filter_genes_by_allele_count(df, MIN_ALLELES)

    out_path = path.with_name(path.stem + "_gene_filtered.xlsx")
    df_filt.to_excel(out_path, index=False)

    print(f"{path.name}: kept {len(kept_genes)} HLA genes → {out_path.name}")


def main(files):
    for f in files:
        process_file(Path(f))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_hla_genes_by_alleles.py <file1.xlsx> <file2.xlsx> ...")
        sys.exit(1)

    main(sys.argv[1:])
