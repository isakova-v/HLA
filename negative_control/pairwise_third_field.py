#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Negative-control pairwise comparison of HLA alleles:
same second field, different third field.

For each vaccine:
- detect HLA columns directly in the vaccine xlsx
- find fully specified alleles with at least 3 fields, e.g. HLA-A*02:01:01
- group them by second-field family, e.g. HLA-A*02:01
- compare third-field variants pairwise within the same second-field family:
    HLA-A*02:01:01 vs HLA-A*02:01:02,
    HLA-A*02:01:01 vs HLA-A*02:01:03, etc.
- compute Hedges' g on log10 titer
- adjust p-values by BH-FDR within each vaccine
- draw barplots of the top pairwise effects

Interpretation:
This is a negative control: variants that are identical up to the second field
should usually not show large systematic differences against each other.
"""

import argparse
import itertools
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_third_field(allele: str) -> Optional[str]:
    if pd.isna(allele):
        return None
    allele = str(allele).strip()
    if allele in ("", "-", "NA", "NaN", "nan"):
        return None
    if "*" not in allele:
        return None
    gene, rest = allele.split("*", 1)
    parts = [p.strip() for p in rest.split(":") if p.strip()]
    if len(parts) < 3:
        return None
    return f"{gene}*{parts[0]}:{parts[1]}:{parts[2]}"


def second_field_from_third(allele3: str) -> str:
    gene, rest = allele3.split("*", 1)
    p1, p2, _ = rest.split(":")[:3]
    return f"{gene}*{p1}:{p2}"


def get_hla_genes_from_columns(df: pd.DataFrame) -> List[str]:
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c.startswith("HLA-")})


def detect_titer_column(df: pd.DataFrame, vaccine_name: str) -> str:
    exact = f"{vaccine_name}_ME_ml"
    if exact in df.columns:
        return exact

    if vaccine_name.upper() == "HBV":
        for cand in ["HBV_antiHBsAg_ME_ml", "HBV_ME_ml"]:
            if cand in df.columns:
                return cand

    candidates = [c for c in df.columns if str(c).endswith("_ME_ml")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Не удалось однозначно определить колонку титра для {vaccine_name}. "
        f"Кандидаты: {candidates}"
    )


def load_vaccine_table(vaccine_name: str, path: str) -> pd.DataFrame:
    df = pd.read_excel(path).copy()

    if "sample_id" not in df.columns:
        raise ValueError(f"{path} must contain 'sample_id'")
    df["sample_id"] = df["sample_id"].astype(str)

    tcol = detect_titer_column(df, vaccine_name)
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[tcol] > 0].copy()
    df["log_titer"] = np.log10(df[tcol])

    return df


def hedges_g(t1: np.ndarray, t0: np.ndarray) -> Tuple[float, float, float, float, float]:
    from scipy.stats import norm

    t1 = np.asarray(t1, dtype=float)
    t0 = np.asarray(t0, dtype=float)

    n1, n0 = len(t1), len(t0)
    if n1 < 2 or n0 < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    m1, m0 = float(t1.mean()), float(t0.mean())
    s1, s0 = float(t1.std(ddof=1)), float(t0.std(ddof=1))

    if (s1 == 0 and s0 == 0) or (n1 + n0 - 2) <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    if sp == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    d = (m1 - m0) / sp
    J = 1.0 - 3.0 / (4.0 * (n1 + n0) - 9.0)
    g = J * d

    se = np.sqrt((n1 + n0) / (n1 * n0) + (g**2) / (2 * (n1 + n0 - 2)))
    ci_lo, ci_hi = g - 1.96 * se, g + 1.96 * se

    if se > 0:
        z = g / se
        p = float(2 * (1 - norm.cdf(abs(z))))
    else:
        p = np.nan

    return (float(g), float(se), float(ci_lo), float(ci_hi), float(p))


def add_fdr(df: pd.DataFrame, p: str = "p_val", q: str = "q") -> pd.DataFrame:
    df = df.copy()
    df[q] = np.nan

    mask = df[p].notna()
    pv = df.loc[mask, p].values
    if len(pv) == 0:
        return df

    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=float)

    qv = ranked * len(ranked) / ranks
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0.0, 1.0)

    out = np.full(len(pv), np.nan, dtype=float)
    out[order] = qv
    df.loc[mask, q] = out
    return df


def build_sample_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    genes = get_hla_genes_from_columns(df)
    rows = []

    for _, r in df.iterrows():
        sid = r["sample_id"]
        lt = r["log_titer"]

        for gene in genes:
            c1, c2 = f"{gene}_1", f"{gene}_2"
            if c1 not in df.columns or c2 not in df.columns:
                continue

            for allele_raw in (r.get(c1), r.get(c2)):
                allele3 = normalize_third_field(allele_raw)
                if allele3 is None:
                    continue
                rows.append(
                    {
                        "sample_id": sid,
                        "gene": gene,
                        "allele3": allele3,
                        "allele2": second_field_from_third(allele3),
                        "log_titer": lt,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.drop_duplicates(subset=["sample_id", "gene", "allele3"])




def summarize_second_field_groups(sv: pd.DataFrame) -> pd.DataFrame:
    if sv.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "allele2",
                "n_third_field_variants",
                "third_field_variants",
                "n_unique_samples",
            ]
        )

    rows = []
    for (gene, allele2), fam in sv.groupby(["gene", "allele2"]):
        variants = sorted(fam["allele3"].dropna().unique())
        rows.append(
            {
                "gene": gene,
                "allele2": allele2,
                "n_third_field_variants": len(variants),
                "third_field_variants": "; ".join(variants),
                "n_unique_samples": int(fam["sample_id"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["n_third_field_variants", "gene", "allele2"],
        ascending=[False, True, True],
    )


def compute_pairwise_effects(
    sv: pd.DataFrame,
    min_n_per_variant: int = 3,
) -> pd.DataFrame:
    if sv.empty:
        return pd.DataFrame()

    results = []

    for (gene, allele2), fam in sv.groupby(["gene", "allele2"]):
        variants = sorted(fam["allele3"].unique())
        if len(variants) < 2:
            continue

        variant_to_ids = {
            var: set(fam.loc[fam["allele3"] == var, "sample_id"].unique())
            for var in variants
        }
        titers = fam[["sample_id", "log_titer"]].drop_duplicates().set_index("sample_id")["log_titer"]

        for a, b in itertools.combinations(variants, 2):
            ids_a = variant_to_ids[a]
            ids_b = variant_to_ids[b]

            # direct pairwise comparison of carriers of allele a vs carriers of allele b
            # to avoid dependence, restrict to samples unique to one variant
            only_a = sorted(ids_a - ids_b)
            only_b = sorted(ids_b - ids_a)

            if len(only_a) < min_n_per_variant or len(only_b) < min_n_per_variant:
                continue

            t1 = titers.loc[only_a].values
            t0 = titers.loc[only_b].values
            g, se, lo, hi, p = hedges_g(t1, t0)
            if np.isnan(g):
                continue

            results.append(
                {
                    "gene": gene,
                    "allele2": allele2,
                    "allele_a": a,
                    "allele_b": b,
                    "n_a": len(only_a),
                    "n_b": len(only_b),
                    "g": g,
                    "se": se,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "p_val": p,
                    "comparison": f"{a} vs {b}",
                }
            )

    return pd.DataFrame(results)


def plot_pairwise_effects(
    df: pd.DataFrame,
    vaccine_name: str,
    outdir: str,
    top_n: int = 40,
    fdr: float = 0.05,
) -> None:
    if df.empty:
        return

    os.makedirs(outdir, exist_ok=True)

    plot_df = df.copy()
    plot_df["abs_g"] = plot_df["g"].abs()
    plot_df = plot_df.sort_values(["q", "abs_g"], ascending=[True, False])

    sig = plot_df[plot_df["q"] < fdr].copy()
    to_plot = sig if not sig.empty else plot_df.head(top_n).copy()
    to_plot = to_plot.head(top_n).copy()

    if to_plot.empty:
        return

    labels = [
        f"{row['allele2']}\n{row['allele_a'].split('*',1)[1]} vs {row['allele_b'].split('*',1)[1]}"
        for _, row in to_plot.iterrows()
    ]

    fig_h = max(5, 0.38 * len(to_plot) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y = np.arange(len(to_plot))
    ax.barh(
        y,
        to_plot["g"].values,
        xerr=1.96 * to_plot["se"].fillna(0).values,
        capsize=3,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hedges' g on log10 titer")
    ax.set_title(
        f"{vaccine_name}: pairwise third-field comparisons within the same second-field family"
    )

    for i, (_, row) in enumerate(to_plot.iterrows()):
        txt = f"n={int(row['n_a'])} vs {int(row['n_b'])}, q={row['q']:.3g}" if pd.notna(row["q"]) else f"n={int(row['n_a'])} vs {int(row['n_b'])}"
        x = row["g"]
        ax.text(
            x + (0.03 if x >= 0 else -0.03),
            i,
            txt,
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    out = os.path.join(outdir, f"pairwise_third_field_negative_control_{vaccine_name}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out}")


def parse_vaccine_specs(specs: List[str]) -> Dict[str, str]:
    out = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --vaccine '{spec}'. Use NAME:PATH.xlsx")
        name, path = spec.split(":", 1)
        out[name.strip()] = path.strip()
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Pairwise negative-control plots: same second field, different third field."
    )
    ap.add_argument("--vaccine", action="append", required=True, help="NAME:path.xlsx")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_n_per_variant", type=int, default=3)
    ap.add_argument("--top_n", type=int, default=40)
    ap.add_argument("--fdr", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    vaccine_specs = parse_vaccine_specs(args.vaccine)
    all_results = []

    for vaccine_name, path in vaccine_specs.items():
        print(f"\n=== {vaccine_name} ===")
        df = load_vaccine_table(vaccine_name, path)
        sv = build_sample_variant_table(df)
        print(f"Rows in sample-variant table: {len(sv)}")

        summary = summarize_second_field_groups(sv)
        summary_path = os.path.join(args.outdir, f"second_field_group_summary_{vaccine_name}.tsv")
        summary.to_csv(summary_path, sep="	", index=False)
        print(f"Saved group summary: {summary_path}")

        res = compute_pairwise_effects(
            sv,
            min_n_per_variant=args.min_n_per_variant,
        )
        if res.empty:
            print("No valid pairwise third-field comparisons.")
            continue

        res.insert(0, "vaccine", vaccine_name)
        res = add_fdr(res, "p_val", "q")
        res = res.sort_values(["q", "p_val", "gene", "allele2", "allele_a", "allele_b"])

        tsv_path = os.path.join(args.outdir, f"pairwise_third_field_effects_{vaccine_name}.tsv")
        res.to_csv(tsv_path, sep="\t", index=False)
        print(f"Saved table: {tsv_path}")

        plot_pairwise_effects(
            res,
            vaccine_name=vaccine_name,
            outdir=args.outdir,
            top_n=args.top_n,
            fdr=args.fdr,
        )

        all_results.append(res)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.outdir, "pairwise_third_field_effects_all.tsv")
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"\nSaved combined table: {combined_path}")
    else:
        print("\nNo results for any vaccine.")


if __name__ == "__main__":
    main()
