#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ----------------------------
# Allele normalization
# ----------------------------

def normalize_allele_first_field(allele: str) -> str | None:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None
    if "*" not in allele:
        return None
    prefix, rest = allele.split("*", 1)
    a = rest.split(":")[0]
    return f"{prefix}*{a}" if a else None


def normalize_allele_second_field(allele: str) -> str | None:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN"):
        return None
    if "*" not in allele:
        return None
    prefix, rest = allele.split("*", 1)
    parts = rest.split(":")
    if not parts or not parts[0]:
        return None
    if len(parts) == 1:
        return f"{prefix}*{parts[0]}"
    return f"{prefix}*{parts[0]}:{parts[1]}"


def get_hla_genes_from_columns(df: pd.DataFrame) -> List[str]:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c.startswith("HLA-")})


# ----------------------------
# Rare genes (exclude whole genes)
# ----------------------------

def load_excluded_genes(path: str) -> Set[str]:
    """
    Parse hla_rare_alleles.txt and return set of gene names to exclude.
    Expected blocks like:
      HLA-DMA (4 unique alleles):
        ...
    """
    pat = re.compile(r"^(HLA-[A-Z0-9]+)\s+\(")
    out: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                out.add(m.group(1))
    return out


# ----------------------------
# Utilities
# ----------------------------

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def standardize_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def pick_id_column(df: pd.DataFrame) -> str:
    if "sample_id" in df.columns:
        return "sample_id"
    if "ZLIMS ID" in df.columns:
        return "ZLIMS ID"
    for c in df.columns:
        if c.strip().lower().replace("_", " ") == "zlims id":
            return c
    raise KeyError("No id column found: neither 'sample_id' nor 'ZLIMS ID'.")


def vaccine_years_after_column(vaccine: str) -> str:
    m = {
        "measles": "Years_after_measles_vaccine",
        "rubella": "Years_after_rubella_vaccine",
        "diphtheria": "Years_after_diphtheria_vaccine",
        "HBV": "Years_after_HBV_vaccine",
    }
    if vaccine not in m:
        raise ValueError(f"Unknown vaccine '{vaccine}'. Expected one of: {list(m)}")
    return m[vaccine]


def pick_hbv_target_column(df: pd.DataFrame, prefix: str) -> str:
    """
    Choose the first column that starts with HBV_antiHBsAg (or provided prefix).
    """
    matches = [c for c in df.columns if str(c).startswith(prefix)]
    if not matches:
        raise KeyError(f"No HBV titer column starts with prefix '{prefix}'.")
    # если несколько — берём первую (обычно достаточно, но можно усложнить при желании)
    return matches[0]


# ----------------------------
# HLA dosage features
# ----------------------------

def build_hla_dosage_features(
    hla_df: pd.DataFrame,
    resolution: str = "second",
    min_carriers: int = 10,
    excluded_genes: Set[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if resolution not in ("first", "second"):
        raise ValueError("resolution must be 'first' or 'second'")
    norm = normalize_allele_second_field if resolution == "second" else normalize_allele_first_field

    excluded_genes = excluded_genes or set()

    genes = get_hla_genes_from_columns(hla_df)
    genes = [g for g in genes if g not in excluded_genes]  # <-- exclude rare genes

    cols_dict: Dict[str, pd.Series] = {}
    ref_alleles: Dict[str, str] = {}

    for gene in genes:
        c1, c2 = f"{gene}_1", f"{gene}_2"
        if c1 not in hla_df.columns or c2 not in hla_df.columns:
            continue

        a1 = hla_df[c1].map(norm)
        a2 = hla_df[c2].map(norm)

        counts = pd.concat([a1, a2], axis=0).dropna().value_counts()
        if counts.empty:
            continue

        ref = counts.index[0]
        ref_alleles[gene] = ref

        for allele in counts.index.tolist()[1:]:
            dosage = (a1.eq(allele).astype(np.int8) + a2.eq(allele).astype(np.int8))
            carriers = int((dosage > 0).sum())
            if carriers < min_carriers:
                continue
            cols_dict[f"{gene}__{allele}"] = dosage.astype(np.int8)

    return pd.DataFrame(cols_dict, index=hla_df.index), ref_alleles


# ----------------------------
# Design matrix
# ----------------------------

def build_design_matrix(
    df: pd.DataFrame,
    vaccine: str,
    target_col: str,
    resolution: str,
    min_carriers: int,
    use_pc: bool,
    n_pcs: int,
    excluded_genes: Set[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in merged table.")

    years_col = vaccine_years_after_column(vaccine)

    cov_cols = [
        "age",
        "sex",
        "cohort_region",
        f"{vaccine}_vaccine_totalnum",
        years_col,
    ]
    if use_pc:
        cov_cols += [f"PC{i}" for i in range(1, n_pcs + 1)]

    missing = [c for c in cov_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing covariate columns for {vaccine}: {missing}")

    y = coerce_numeric(df[target_col])

    X_cov = df[cov_cols].copy()
    X_cov["age"] = coerce_numeric(X_cov["age"])
    X_cov[f"{vaccine}_vaccine_totalnum"] = coerce_numeric(X_cov[f"{vaccine}_vaccine_totalnum"])
    X_cov[years_col] = coerce_numeric(X_cov[years_col])
    if use_pc:
        for i in range(1, n_pcs + 1):
            X_cov[f"PC{i}"] = coerce_numeric(X_cov[f"PC{i}"])

    X_cov["sex"] = X_cov["sex"].astype(str).str.strip()
    X_cov["cohort_region"] = X_cov["cohort_region"].astype(str).str.strip()
    X_cov = pd.get_dummies(X_cov, columns=["sex", "cohort_region"], drop_first=True)

    hla_cols = ["sample_id"] + [c for c in df.columns if c.startswith("HLA-")]
    hla_part = df[hla_cols].copy()

    X_hla, ref_alleles = build_hla_dosage_features(
        hla_part,
        resolution=resolution,
        min_carriers=min_carriers,
        excluded_genes=excluded_genes,
    )

    X = pd.concat([X_cov, X_hla], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")

    X_arr = X.to_numpy(dtype=float, copy=False)
    y_arr = y.to_numpy(dtype=float, copy=False)

    valid = np.isfinite(y_arr) & np.isfinite(X_arr).all(axis=1)
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    X = sm.add_constant(X, has_constant="add")
    return X, y, ref_alleles


# ----------------------------
# Fit + save + plot
# ----------------------------

def fit_ols(X: pd.DataFrame, y: pd.Series):
    model = sm.OLS(y.to_numpy(dtype=float, copy=False), X.to_numpy(dtype=float, copy=False))
    return model.fit()


def save_betas(res, X_cols: List[str], out_path: Path, meta: Dict[str, str]) -> pd.DataFrame:
    params = pd.Series(res.params, index=X_cols, name="beta")
    se = pd.Series(res.bse, index=X_cols, name="se")
    pval = pd.Series(res.pvalues, index=X_cols, name="pvalue")
    ci = pd.DataFrame(res.conf_int(), index=X_cols, columns=["ci_low", "ci_high"])

    out = pd.concat([params, se, pval, ci], axis=1)
    out.insert(0, "term", out.index)
    out.insert(1, "N", int(res.nobs))
    out.insert(2, "R2", float(res.rsquared))
    for k, v in meta.items():
        out[k] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_path, index=False)
    print(f"Wrote {out_path} (N={int(res.nobs)}, terms={out.shape[0]})")
    return out


def plot_significant_betas(
    betas_df: pd.DataFrame,
    vaccine: str,
    target: str,
    resolution: str,
    outdir: Path,
    p_thr: float = 0.05,
    top_n: int = 40,
    show: bool = False,
):
    df = betas_df.copy()
    df = df[pd.to_numeric(df["pvalue"], errors="coerce") < p_thr].copy()
    df = df[df["term"].astype(str).str.startswith("HLA-")].copy()

    if df.empty:
        print(f"[plot] No significant HLA terms for {vaccine} at p<{p_thr}")
        return

    df = df.sort_values("beta", key=lambda s: s.abs(), ascending=False)
    if top_n is not None and len(df) > top_n:
        df = df.head(top_n)

    plt.figure(figsize=(10, max(4, 0.33 * len(df))))
    plt.barh(df["term"], df["beta"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("beta")
    plt.ylabel("term")
    plt.title(f"{vaccine} | {target} | {resolution} | p<{p_thr} (top {len(df)})")
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{vaccine}.{target}.{resolution}.p{p_thr}.betas.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot: {out_path}")

    if show:
        plt.show()
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaccine-xlsx", nargs="+", required=True)
    ap.add_argument("--hla-xlsx", required=True)
    ap.add_argument("--target-col", required=True, help="Target column name for non-HBV; for HBV can be 'AUTO'.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resolution", choices=["first", "second"], default="second")
    ap.add_argument("--min-carriers", type=int, default=10)

    # PC control
    ap.add_argument("--use-pc", action="store_true", help="Include PC covariates (default: OFF).")
    ap.add_argument("--n-pcs", type=int, default=20)

    # HBV titer column prefix
    ap.add_argument("--hbv-prefix", default="HBV_antiHBsAg", help="Prefix for HBV titer column selection.")

    # Rare genes file
    ap.add_argument("--rare-genes", required=True, help="Path to hla_rare_alleles.txt")

    # Immediate visualization
    ap.add_argument("--p-thr", type=float, default=0.05)
    ap.add_argument("--top-n-plot", type=int, default=40)
    ap.add_argument("--show-plots", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"

    excluded_genes = load_excluded_genes(str(Path(args.rare_genes).expanduser().resolve()))
    print(f"Excluded (rare) genes loaded: {len(excluded_genes)}")

    hla = pd.read_excel(Path(args.hla_xlsx).expanduser().resolve(), dtype=str)
    if "sample_id" not in hla.columns:
        raise KeyError("combined_hla_out.xlsx must contain 'sample_id' column.")
    hla["sample_id"] = standardize_id(hla["sample_id"])

    for vx in args.vaccine_xlsx:
        vx_path = Path(vx).expanduser().resolve()
        vaccine = vx_path.stem  # measles/rubella/diphtheria/HBV

        ph = pd.read_excel(vx_path, dtype=str)
        id_col = pick_id_column(ph)
        ph = ph.rename(columns={id_col: "sample_id"})
        ph["sample_id"] = standardize_id(ph["sample_id"])

        # (1) HBV: auto-pick target column by prefix
        if vaccine == "HBV" and (args.target_col.upper() == "AUTO" or args.target_col not in ph.columns):
            target_col = pick_hbv_target_column(ph, args.hbv_prefix)
            print(f"[HBV] Using target column: {target_col}")
        else:
            target_col = args.target_col

        df = ph.merge(hla, on="sample_id", how="inner")
        if len(df) == 0:
            raise RuntimeError(f"After merge 0 rows for {vaccine}. Check IDs match between tables.")

        X, y, ref_alleles = build_design_matrix(
            df=df,
            vaccine=vaccine,
            target_col=target_col,
            resolution=args.resolution,
            min_carriers=args.min_carriers,
            use_pc=args.use_pc,
            n_pcs=args.n_pcs,
            excluded_genes=excluded_genes,
        )

        res = fit_ols(X, y)

        ref_path = outdir / f"{vaccine}.reference_alleles.tsv"
        pd.DataFrame([{"gene": g, "reference_allele": a} for g, a in sorted(ref_alleles.items())]) \
            .to_csv(ref_path, sep="\t", index=False)

        out_path = outdir / f"{vaccine}.betas.xlsx"
        meta = {"vaccine": vaccine, "target": target_col, "resolution": args.resolution}
        betas_df = save_betas(res, list(X.columns), out_path, meta)
        print(f"Wrote {ref_path}")

        plot_significant_betas(
            betas_df=betas_df,
            vaccine=vaccine,
            target=target_col,
            resolution=args.resolution,
            outdir=plots_dir,
            p_thr=args.p_thr,
            top_n=args.top_n_plot,
            show=args.show_plots,
        )

    print("Done.")


if __name__ == "__main__":
    main()