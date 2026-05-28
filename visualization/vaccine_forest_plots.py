#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Vaccine forest plots per region, with HLA normalization helpers.
#
# - Reads vaccination phenotypes (TSV/CSV/Excel) and HLA genotypes (Excel).
# - Detects regions either from a categorical `cohort_region` column
#   OR from boolean flag columns like: is_from_Irkutsk, is_from_Amur, is_from_NiNo, is_from_Kaliningrad.
#   If flag-columns are used, a record is *expanded* to multiple region-rows for every True flag.
# - For each vaccine present in the file, among *vaccinated* patients:
#     * Uses quantitative titer column(s) for that vaccine (auto-detected) and
#       compares responders (result==1) vs non-responders (result==0).
#     * Computes Hedges' g with 95% CI per region and a fixed-effect meta estimate (Overall).
#     * Saves per-vaccine CSVs and a forest plot (PNG) with horizontal CIs.
#
# Example:
#     python vaccine_forest_plots.py \
#         --vacc /mnt/data/all_pheno_unrel.tsv \
#         --hla  /mnt/data/combined_hla_out.xlsx \
#         --out  ./outputs \
#         --allele-level 2 \
#         --norm log1p \
#         --region-flag-prefix is_from_
#
# Note: HLA file is optional here (plots use phenotype-only metrics).

import argparse
import os
import math
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers from the dashboard code (lightweight, no Streamlit)
# -----------------------------

def normalize_allele(allele: str, level: int = 2) -> str:
    ALLELE_PATTERN = re.compile(
        r"^(?:HLA-)?([A-Z0-9]+)\*"
        r"(\d{2})"
        r"(?::([0-9A-Za-z]{1,}))?"
        r"(?::[0-9A-Za-z]{1,})*"
        r".*$", flags=re.IGNORECASE
    )
    if not isinstance(allele, str) or allele.strip() == "" or allele == "-":
        return allele
    m = ALLELE_PATTERN.match(allele.strip())
    if not m:
        return allele
    gene, group, protein = m.groups()
    gene = gene.upper()
    if level == 1:
        return f"{gene}*{group}"
    if protein:
        return f"{gene}*{group}:{protein}"
    return f"{gene}*{group}"


def _gene_prefixes(df: pd.DataFrame) -> List[str]:
    gene_cols = [c for c in df.columns if "_" in str(c)]
    return sorted({c.split("_")[0] for c in gene_cols})


def normalize_and_fill(hla_df: pd.DataFrame, level: int) -> pd.DataFrame:
    df = hla_df.copy()
    for g in _gene_prefixes(df):
        c1, c2 = f"{g}_1", f"{g}_2"
        if c1 in df.columns and c2 in df.columns:
            df[c1] = df[c1].apply(lambda x: normalize_allele(x, level) if isinstance(x, str) else x)
            df[c2] = df[c2].apply(lambda x: normalize_allele(x, level) if isinstance(x, str) else x)
            m1 = df[c1] == "-"
            m2 = df[c2] == "-"
            df.loc[m1, c1] = df.loc[m1, c2]
            df.loc[m2, c2] = df.loc[m2, c1]
    return df


def process_hla_long(df: pd.DataFrame, level: int) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise ValueError("Expected 'ID' in HLA table.")
    gene_list = _gene_prefixes(df)
    dot_masks = {}
    for g in gene_list:
        c1, c2 = f"{g}_1", f"{g}_2"
        if c1 in df.columns and c2 in df.columns:
            s1 = df[c1].astype(str)
            s2 = df[c2].astype(str)
            dot_masks[g] = (s1 == ".") | (s2 == ".")
    df_norm = normalize_and_fill(df, level)
    rows = []
    for g in gene_list:
        c1, c2 = f"{g}_1", f"{g}_2"
        if c1 not in df_norm.columns or c2 not in df_norm.columns:
            continue
        sub = df_norm[["ID", c1, c2]].copy()
        m_dot = dot_masks.get(g)
        if m_dot is not None:
            sub = sub.loc[~m_dot.values]
        homo = sub[c1] == sub[c2]
        left = sub.loc[~homo, ["ID", c1]].rename(columns={c1: "Allele"})
        right = sub.loc[~homo, ["ID", c2]].rename(columns={c2: "Allele"})
        both = pd.concat([left, right], ignore_index=True)
        homo_df = sub.loc[homo, ["ID", c1]].rename(columns={c1: "Allele"})
        homo_df2 = pd.concat([homo_df, homo_df], ignore_index=True)
        all_ = pd.concat([both, homo_df2], ignore_index=True)
        all_["Gene"] = g
        rows.append(all_)
    if not rows:
        return pd.DataFrame(columns=["ID", "Gene", "Allele"])
    out = pd.concat(rows, ignore_index=True)
    out = out[out["Allele"].notna()]
    out = out[~out["Allele"].astype(str).isin(["-", "."])]
    return out[["ID", "Gene", "Allele"]]


def build_allele_dosage_matrix(hla_long: pd.DataFrame) -> pd.DataFrame:
    if hla_long.empty:
        return pd.DataFrame()
    dose = hla_long.groupby(["ID", "Allele"]).size().unstack(fill_value=0)
    dose = dose.clip(lower=0, upper=2)
    return dose


def normalize_antibody(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x >= 0)
    if method == "log1p":
        return np.log1p(x)
    elif method == "zscore":
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True, ddof=1)
        if pd.isna(sd) or sd == 0:
            return x * 0
        return (x - mu) / sd
    return x


def hedges_g_and_var(mean1, sd1, n1, mean0, sd0, n0):
    if n1 < 2 or n0 < 2 or sd1 <= 0 or sd0 <= 0:
        return None, None
    sp2 = ((n1 - 1) * (sd1 ** 2) + (n0 - 1) * (sd0 ** 2)) / (n1 + n0 - 2)
    if sp2 <= 0:
        return None, None
    d = (mean1 - mean0) / math.sqrt(sp2)
    J = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
    g = J * d
    var_g = (n1 + n0) / (n1 * n0) + (g ** 2) / (2.0 * (n1 + n0 - 2))
    return g, var_g


def p_from_z(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


# -----------------------------
# Vaccine column discovery
# -----------------------------

TITER_HINTS = ["me_ml", "igg", "anti", "titer", "титр", "ме", "iu", "ml"]


def detect_vaccines(df: pd.DataFrame) -> List[Tuple[str, Optional[str], str, str]]:
    """
    Returns list of tuples: (prefix, titer_col, result_col, info_col).
    """
    cols = [str(c) for c in df.columns]
    prefixes = []
    for c in cols:
        if c.endswith("_vaccine_info"):
            prefixes.append(c[:-13])
    found = []
    for pref in sorted(set(prefixes)):
        info_col = f"{pref}_vaccine_info"
        result_col = f"{pref}_result"
        if result_col not in df.columns:
            continue
        # candidate titer columns
        cand_titers = [c for c in cols if c.startswith(pref) and any(h in c.lower() for h in TITER_HINTS)]
        titer_col = None
        for pattern in [f"{pref}_me_ml", f"{pref}_ME_ml", f"{pref}_ME_mL"]:
            if pattern in df.columns:
                titer_col = pattern
                break
        if titer_col is None and cand_titers:
            # choose with most numeric coverage
            best, best_non_nan = None, -1
            for c in cand_titers:
                v = pd.to_numeric(df[c], errors="coerce")
                cnt = v.notna().sum()
                if cnt > best_non_nan:
                    best_non_nan, best = cnt, c
            titer_col = best
        found.append((pref, titer_col, result_col, info_col))
    return found


# -----------------------------
# Region handling
# -----------------------------

def expand_regions(df: pd.DataFrame,
                   region_col: Optional[str] = None,
                   region_flag_prefix: str = "is_from_") -> pd.DataFrame:
    """
    Returns a copy of df with an explicit 'Region' column.
    If `region_col` exists, use it. Otherwise, scan boolean flag columns
    starting with `region_flag_prefix` and expand rows per True flag.
    """
    if region_col and region_col in df.columns:
        out = df.copy()
        out["Region"] = out[region_col].astype(str)
        return out

    flag_cols = [c for c in df.columns if str(c).startswith(region_flag_prefix)]
    if not flag_cols:
        out = df.copy()
        out["Region"] = "All"
        return out

    rows = []
    for _, row in df.iterrows():
        regions_true = []
        for c in flag_cols:
            val = row[c]
            is_true = False
            try:
                # accept bools and truthy strings/numbers
                if isinstance(val, (bool, np.bool_)):
                    is_true = bool(val)
                else:
                    s = str(val).strip().lower()
                    is_true = s in ["1", "true", "yes", "да", "y", "истина"]
            except Exception:
                is_true = False
            if is_true:
                regions_true.append(c[len(region_flag_prefix):])
        if not regions_true:
            regions_true = ["Unknown"]
        for r in regions_true:
            nr = row.copy()
            nr["Region"] = r
            rows.append(nr)
    return pd.DataFrame(rows)


# -----------------------------
# Effect size per region and forest-plot
# -----------------------------

def effect_rows_for_vaccine(df: pd.DataFrame,
                            vaccine_name: str,
                            titer_col: str,
                            result_col: str,
                            info_col: str,
                            norm: str = "log1p") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Hedges g per region for a given vaccine. Returns (per_region_df, meta_df).
    """
    dfw = df.copy()
    # only vaccinated
    info_s = dfw[info_col]
    if pd.api.types.is_bool_dtype(info_s):
        mask_vacc = info_s.fillna(False)
    else:
        mask_vacc = info_s.astype(str).str.lower().isin(["1", "true", "yes", "да", "y", "истина"])
    dfw = dfw.loc[mask_vacc].copy()

    # numeric titer + normalization
    dfw["AntibodyRaw"] = pd.to_numeric(dfw[titer_col], errors="coerce")
    if norm == "log1p":
        dfw["Antibody"] = normalize_antibody(dfw["AntibodyRaw"], "log1p")
    else:
        dfw["Antibody"] = (
            dfw.groupby("Region", group_keys=False)["AntibodyRaw"]
               .apply(lambda s: normalize_antibody(s, "zscore"))
        )
    dfw = dfw.dropna(subset=["Antibody", "Region"])

    # groups by result (1 vs 0)
    res = pd.to_numeric(dfw[result_col], errors="coerce")
    mask = res.isin([0, 1])
    dfw = dfw.loc[mask].copy()
    dfw["Group"] = res.loc[dfw.index].astype(int)

    per_rows = []
    for reg, sub in dfw.groupby("Region"):
        vals1 = sub.loc[sub["Group"] == 1, "Antibody"].astype(float)
        vals0 = sub.loc[sub["Group"] == 0, "Antibody"].astype(float)
        if len(vals1) < 3 or len(vals0) < 3:
            continue
        m1, s1, n1 = float(vals1.mean()), float(vals1.std(ddof=1)), int(len(vals1))
        m0, s0, n0 = float(vals0.mean()), float(vals0.std(ddof=1)), int(len(vals0))
        g_r, var_r = hedges_g_and_var(m1, s1, n1, m0, s0, n0)
        if g_r is None or var_r is None or var_r <= 0:
            continue
        se_r = math.sqrt(var_r)
        per_rows.append({
            "Vaccine": vaccine_name,
            "Region": str(reg),
            "g": g_r,
            "se": se_r,
            "ci_low": g_r - 1.96 * se_r,
            "ci_high": g_r + 1.96 * se_r,
            "n1": n1,
            "n0": n0
        })
    per_df = pd.DataFrame(per_rows)

    # fixed-effect meta
    if per_df.empty:
        meta_df = pd.DataFrame([{
            "Vaccine": vaccine_name, "g_fixed": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "p_meta": np.nan, "k_regions": 0
        }])
    else:
        weights = 1.0 / (per_df["se"].values ** 2)
        ests = per_df["g"].values
        sum_w = weights.sum()
        g_fixed = float(np.sum(weights * ests) / sum_w) if sum_w > 0 else np.nan
        se_fixed = float(math.sqrt(1.0 / sum_w)) if sum_w > 0 else np.nan
        z = g_fixed / se_fixed if (se_fixed and se_fixed > 0) else np.nan
        p_meta = p_from_z(z) if (z is not None and not np.isnan(z)) else np.nan
        meta_df = pd.DataFrame([{
            "Vaccine": vaccine_name, "g_fixed": g_fixed, "se": se_fixed,
            "ci_low": g_fixed - 1.96 * se_fixed if not np.isnan(se_fixed) else np.nan,
            "ci_high": g_fixed + 1.96 * se_fixed if not np.isnan(se_fixed) else np.nan,
            "p_meta": p_meta, "k_regions": int(per_df.shape[0])
        }])

    return per_df, meta_df


def forest_plot(per_df: pd.DataFrame, meta_df: pd.DataFrame, title: str, out_png: str):
    """
    Draw a forest plot with horizontal CIs per region and an 'Overall' row.
    """
    if per_df.empty:
        plt.figure(figsize=(6, 2))
        plt.title(title + " (no data)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        return

    overall = meta_df.iloc[0]
    overall_row = pd.DataFrame([{
        "Vaccine": per_df["Vaccine"].iloc[0],
        "Region": "Overall",
        "g": overall["g_fixed"],
        "se": overall["se"],
        "ci_low": overall["ci_low"],
        "ci_high": overall["ci_high"],
        "n1": None, "n0": None
    }])
    plot_df = pd.concat([per_df, overall_row], ignore_index=True)

    regions = sorted([r for r in plot_df["Region"].unique() if r != "Overall"]) + ["Overall"]
    plot_df["Region"] = pd.Categorical(plot_df["Region"], categories=regions, ordered=True)
    plot_df = plot_df.sort_values("Region", ascending=True)

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.35 * len(plot_df)))

    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.hlines(y, plot_df["ci_low"], plot_df["ci_high"])
    ax.plot(plot_df["g"], y, "o")

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Region"].astype(str).tolist())
    ax.set_xlabel("Hedges g (Responders − Non‑responders)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# IO utils
# -----------------------------

def read_table_auto(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tsv", ".tab", ".txt"]:
        return pd.read_csv(path, sep="\t")
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Forest plots per vaccine and region (Hedges g).")
    ap.add_argument("--vacc", required=True, help="Path to vaccination phenotypes (TSV/CSV/XLSX).")
    ap.add_argument("--hla", required=False, default="", help="Path to HLA genotypes (XLSX) — optional.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--allele-level", type=int, default=2, help="HLA allele aggregation level (1 or 2).")
    ap.add_argument("--norm", choices=["log1p", "zscore"], default="log1p", help="Normalization for antibody titers.")
    ap.add_argument("--region-col", default="", help="Explicit region column name if present (e.g., cohort_region).")
    ap.add_argument("--region-flag-prefix", default="is_from_", help="Prefix for region flag columns.")
    args = ap.parse_args()

    ensure_dir(args.out)
    df = read_table_auto(args.vacc)

    # Region preparation (handles is_from_* flags automatically)
    region_col = args.region_col if args.region_col.strip() else None
    df = expand_regions(df, region_col=region_col, region_flag_prefix=args.region_flag_prefix)

    # Vaccine discovery
    vaccines = detect_vaccines(df)
    if not vaccines:
        print("No vaccines discovered: expecting columns like <name>_vaccine_info and <name>_result.")
        return

    # Optional HLA read to validate normalization pipeline (not used in plots)
    if args.hla and os.path.exists(args.hla):
        try:
            hla_df = pd.read_excel(args.hla)
            id_cols = [c for c in ["sample_id", "ID", "Sample_ID", "ZLIMS ID", "ID_x"] if c in hla_df.columns]
            if id_cols:
                hla_df["ID"] = hla_df[id_cols[0]].astype(str)
            else:
                hla_df["ID"] = hla_df.index.astype(str)
            _ = process_hla_long(hla_df[["ID"] + [c for c in hla_df.columns if "_" in str(c)]].copy(), args.allele_level)
        except Exception as e:
            print(f"Warning: HLA file parsed with issues: {e}")

    # Per vaccine effects
    all_meta_rows = []
    for pref, titer_col, result_col, info_col in vaccines:
        vaccine_name = pref
        out_dir_v = os.path.join(args.out, vaccine_name)
        ensure_dir(out_dir_v)

        if titer_col is None or titer_col not in df.columns:
            # skip vaccines without quantitative titers
            continue
        if result_col not in df.columns or info_col not in df.columns:
            continue

        per_df, meta_df = effect_rows_for_vaccine(
            df=df, vaccine_name=vaccine_name, titer_col=titer_col,
            result_col=result_col, info_col=info_col, norm=args.norm
        )

        # Save tables
        per_csv = os.path.join(out_dir_v, "per_region_effects.csv")
        meta_csv = os.path.join(out_dir_v, "meta_summary.csv")
        per_df.to_csv(per_csv, index=False)
        meta_df.to_csv(meta_csv, index=False)

        # Plot
        title = f"{vaccine_name}: Hedges g by Region"
        out_png = os.path.join(out_dir_v, f"{vaccine_name}_forest.png")
        forest_plot(per_df, meta_df, title, out_png)

        if not meta_df.empty:
            r = meta_df.iloc[0].to_dict()
            r["Vaccine"] = vaccine_name
            all_meta_rows.append(r)

    if all_meta_rows:
        meta_all = pd.DataFrame(all_meta_rows)
        meta_all.to_csv(os.path.join(args.out, "meta_all_vaccines.csv"), index=False)


if __name__ == "__main__":
    main()
