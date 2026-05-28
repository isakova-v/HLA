#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build forest plots for top-10 significant HLA alleles (first-field) per vaccine and gene,
and save per-(vaccine,gene) significance tables with BH-FDR.

Inputs:
  --pheno  : TSV с фенотипами (должны быть столбцы титров и флаги регионов)
  --hla    : XLSX с HLA-типированием (широкий формат, как в текущем коде)
Outputs:
  outdir/<vaccine>/<gene>/*.png   — forest-плоты (или пусто, если данных нет)
  outdir/allele_significance_tables.csv  — сводная таблица по всем вакцинам/генам

Авторские допущения:
  • ID ищется среди колонок: ["ZLIMS ID","ID","ID_x","sample_id","Sample_ID","sampleid","sample id"]
  • Регион определяется по ровно одному флагу из:
      is_from_Irkutsk, is_from_Amur, is_from_NiNo, is_from_Kaliningrad
    (несколько флагов → "Mixed", без флага → наблюдение исключается)
  • Нормализация титров: log1p (по умолчанию) или z-score.
  • Агрегация аллелей до 1-го поля: A*01:01:01G → A*01 и т.п.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Конфигурация колонок ----

REGION_FLAGS = {
    "is_from_Irkutsk": "Irkutsk",
    "is_from_Amur": "Amur",
    "is_from_NiNo": "Nizhny Novgorod",
    "is_from_Kaliningrad": "Kaliningrad",
}

VACCINES = {
    "measles": "measles_ME_ml",
    "rubella": "rubella_ME_ml",
    "diphtheria": "diphtheria_ME_ml",
    "HBV": "HBV_antiHBsAg_ME_ml",
}

ID_CANDS = ["ZLIMS ID", "ID", "ID_x", "sample_id", "Sample_ID", "sampleid", "sample id"]

HLA_ALLELE_RE = re.compile(r'^(?:[A-Z]{1,4}[0-9]?[A-Z]?)\*[0-9]{2}(?::[0-9]{2}){0,3}[A-Z]*$')
GENE_HEADER_RE = re.compile(r'^HLA-', re.IGNORECASE)


# ---------- Утилиты ----------

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def first_existing_col(df: pd.DataFrame, cands: Iterable[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def normalize_gene_header(s: str) -> str:
    return re.sub(GENE_HEADER_RE, '', str(s)).strip()

def _clean_allele_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = s.replace("HLA-", "").replace("hla-", "")
    s = s.replace("~", "").replace("?", "").replace(" ", "")
    return s

def _allele_to_level(allele: str, level: int) -> str:
    # "A*01:01:01G" -> "A*01" (level=1)
    if "*" in allele:
        gene, rest = allele.split("*", 1)
    else:
        gene, rest = "", allele
    rest = rest.split("+")[0]
    rest = rest.split("(")[0]
    rest = re.sub(r"[A-Za-z]$", "", rest)  # убрать финальные буквы (G,N и т.п.)
    fields = rest.split(":")
    if level == 1:
        rest_norm = fields[0]
    elif level == 2:
        rest_norm = ":".join(fields[:2])
    else:
        rest_norm = rest
    return (gene + "*" if gene else "") + rest_norm

def pick_hla_cols_by_values(df: pd.DataFrame, sample_rows: int = 200) -> List[str]:
    keep = []
    head = df.head(sample_rows)
    for c in df.columns:
        cs = str(c)
        if cs.startswith(("HLA-", "hla-")) and "_" in cs:
            keep.append(c)
            continue
        s = head[c].astype(str)
        if s.str.contains(r'[A-Z]{1,4}\*\d{2}', regex=True, na=False).any():
            keep.append(c)
    return keep

def process_hla_long(hla_wide: pd.DataFrame, allele_level: int) -> pd.DataFrame:
    id_col = first_existing_col(hla_wide, ID_CANDS)
    if id_col is None:
        raise RuntimeError("ID column not found in HLA table")
    keep_cols = [c for c in hla_wide.columns if c != id_col]
    rows = []
    for _, row in hla_wide[[id_col] + keep_cols].iterrows():
        pid = str(row[id_col]).strip()
        if not pid or pid.lower() in ("nan", "none"):
            continue
        for c in keep_cols:
            val = _clean_allele_text(row[c])
            if not val or val in (".", "-", "0", "nan", "None"):
                continue
            col = str(c)
            gene = normalize_gene_header(col.split("_")[0]).upper()
            if "*" in val:
                full = _allele_to_level(val.upper(), allele_level)
            else:
                full = _allele_to_level(f"{gene}*{val.upper()}", allele_level)
            rows.append((pid, gene, full))
    long = pd.DataFrame(rows, columns=["ID", "Gene", "Allele"]).dropna()
    bad = {"", ".", "-", "0", "NAN", "NONE"}
    long = long[~long["Allele"].astype(str).str.upper().isin(bad)]
    long["ID"] = long["ID"].astype(str)
    long["Gene"] = long["Gene"].astype(str)
    long["Allele"] = long["Allele"].astype(str)
    return long

def sanitize_hla_long(hla_long: pd.DataFrame) -> pd.DataFrame:
    if hla_long.empty:
        return hla_long
    out = hla_long.copy()
    out["Allele"] = out["Allele"].astype(str).str.strip().str.replace(r'\s+', '', regex=True)
    mask_valid = out["Allele"].str.match(HLA_ALLELE_RE)
    out = out[mask_valid].copy()
    out["Gene"] = out["Allele"].str.split("*").str[0]
    return out

def normalize_antibody(s: pd.Series, method: str = "log1p") -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s >= 0)
    if method == "zscore":
        mu, sd = s.mean(skipna=True), s.std(ddof=1, skipna=True)
        return (s - mu) / sd if (sd and sd > 0) else pd.Series(np.nan, index=s.index)
    return np.log1p(s)

def derive_region_from_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def pick_region(row):
        hits = [name for flag, name in REGION_FLAGS.items()
                if flag in row.index and bool(row[flag])]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return "Mixed"
        return None
    df["cohort_region"] = df.apply(pick_region, axis=1)
    return df

# ---------- Статистика ----------

def hedges_g_and_var(m1, s1, n1, m0, s0, n0) -> Tuple[Optional[float], Optional[float]]:
    if n1 < 2 or n0 < 2 or s1 <= 0 or s0 <= 0:
        return None, None
    sp2 = (((n1 - 1) * (s1 ** 2)) + ((n0 - 1) * (s0 ** 2))) / (n1 + n0 - 2)
    sp = math.sqrt(sp2)
    if sp <= 0:
        return None, None
    d = (m1 - m0) / sp
    J = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0))
    g = J * d
    var_g = ((n1 + n0) / (n1 * n0)) + ((g ** 2) / (2 * (n1 + n0 - 2)))
    return g, var_g

def p_from_z(z: float) -> float:
    # двусторонний p из нормального CDF
    return float(2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adj = p * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out

# ---------- Готовим группы и считаем эффекты ----------

def build_groups_for_vaccine(pheno: pd.DataFrame, vacc_key: str, norm: str) -> pd.DataFrame:
    titer_col = VACCINES[vacc_key]
    df = pheno.copy()
    if titer_col not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=[titer_col])
    df[titer_col] = pd.to_numeric(df[titer_col], errors="coerce")
    df = df.dropna(subset=[titer_col])
    df = derive_region_from_flags(df)
    df = df.dropna(subset=["cohort_region"])
    df["Antibody"] = normalize_antibody(df[titer_col], norm)
    id_col = first_existing_col(df, ID_CANDS)
    if id_col is None:
        return pd.DataFrame()
    groups = df[[id_col, "Antibody", "cohort_region"]].rename(
        columns={id_col: "ID", "cohort_region": "Region"}
    )
    groups["ID"] = groups["ID"].astype(str)
    groups["Region"] = groups["Region"].astype(str)
    return groups

def compute_meta_for_vaccine_gene(groups: pd.DataFrame, hla_long: pd.DataFrame,
                                  gene: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub_gene = hla_long[hla_long["Gene"].str.upper() == gene.upper()]
    if sub_gene.empty or groups.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = sub_gene.merge(groups, on="ID", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta_rows, per_reg_rows = [], []
    for allele in sorted(merged["Allele"].unique()):
        region_stats = []
        for reg, sub_r in merged.groupby("Region"):
            pres = (sub_r.assign(is_a=(sub_r["Allele"] == allele).astype(int))
                          .groupby("ID")["is_a"].max())
            ant = sub_r.groupby("ID")["Antibody"].first()
            df_r = pd.concat([pres, ant], axis=1).dropna()
            vals1 = df_r.loc[df_r["is_a"] == 1, "Antibody"].astype(float)
            vals0 = df_r.loc[df_r["is_a"] == 0, "Antibody"].astype(float)
            if len(vals1) < 3 or len(vals0) < 3:
                continue
            m1, s1, n1 = float(vals1.mean()), float(vals1.std(ddof=1)), int(len(vals1))
            m0, s0, n0 = float(vals0.mean()), float(vals0.std(ddof=1)), int(len(vals0))
            g_r, var_r = hedges_g_and_var(m1, s1, n1, m0, s0, n0)
            if g_r is None or var_r is None or var_r <= 0:
                continue
            se_r = math.sqrt(var_r)
            region_stats.append((reg, g_r, se_r, n1, n0))

        if not region_stats:
            continue

        weights = [1.0 / (se ** 2) for (_, _, se, _, _) in region_stats]
        ests = [g for (_, g, _, _, _) in region_stats]
        sum_w = sum(weights)
        if sum_w <= 0:
            continue

        g_fixed = sum(w * e for w, e in zip(weights, ests)) / sum_w
        se_fixed = math.sqrt(1.0 / sum_w)
        z = g_fixed / se_fixed if se_fixed > 0 else 0.0
        p_meta = p_from_z(z)
        ci_lo = g_fixed - 1.96 * se_fixed
        ci_hi = g_fixed + 1.96 * se_fixed

        meta_rows.append({
            "Gene": gene, "Allele": allele,
            "g_fixed": g_fixed, "g_se": se_fixed,
            "g_ci_low": ci_lo, "g_ci_high": ci_hi,
            "p_meta": p_meta, "k_regions": len(region_stats)
        })

        for (reg, g_r, se_r, n1, n0) in region_stats:
            per_reg_rows.append({
                "Gene": gene, "Allele": allele, "Region": reg,
                "g": g_r, "se": se_r,
                "ci_low": g_r - 1.96 * se_r,
                "ci_high": g_r + 1.96 * se_r,
                "n_carriers": n1, "n_noncarriers": n0
            })

    return pd.DataFrame(meta_rows), pd.DataFrame(per_reg_rows)

# ---------- Визуализация (matplotlib) ----------

def forest_plot_png(save_path: Path, title: str, per_region: pd.DataFrame, overall_row: pd.Series):
    # per_region: columns Region, g, ci_low, ci_high
    plot_df = per_region.copy().sort_values("Region")
    overall = overall_row.copy()
    overall["Region"] = "Overall"
    plot_df = pd.concat([plot_df, overall.to_frame().T], ignore_index=True)

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(plot_df) + 2.8), dpi=150)

    # Вертикальная линия x=0
    ax.axvline(0.0, linestyle="--", linewidth=1)

    # Отрезки доверительных интервалов
    ax.hlines(y, plot_df["ci_low"], plot_df["ci_high"], lw=2)

    # Точки-оценки
    ax.plot(plot_df["g"], y, 'o', ms=6)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Region"])
    ax.set_xlabel("Hedges g (Carrier – Other)")
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

# ---------- CLI ----------

def load_hla_long(hla_path: str, allele_level: int) -> pd.DataFrame:
    hla_df = pd.read_excel(hla_path)
    hla_df = _normalize_cols(hla_df)
    id_col = first_existing_col(hla_df, ID_CANDS)
    hla_cols = pick_hla_cols_by_values(hla_df)
    keep = ([id_col] if id_col else []) + hla_cols
    hla_sub = hla_df[keep].copy()
    if id_col is None:
        for c in ID_CANDS:
            if c in hla_df.columns:
                id_col = c
                break
    if id_col is None:
        raise RuntimeError("No ID column in HLA file.")
    hla_sub["ID"] = hla_sub[id_col].astype("string").str.strip()
    hla_sub = hla_sub[hla_sub["ID"].notna() & (hla_sub["ID"] != "")]
    long = process_hla_long(hla_sub, allele_level)
    long = sanitize_hla_long(long)
    long["ID"] = long["ID"].astype(str)
    return long

def main():
    ap = argparse.ArgumentParser(description="HLA forest plots per vaccine/gene (top-10, BH-FDR).")
    ap.add_argument("--pheno", default="all_pheno_unrel.tsv",
                    help="TSV с фенотипами (по умолчанию all_pheno_unrel.tsv)")
    ap.add_argument("--hla", default="combined_hla_out.xlsx",
                    help="XLSX с HLA-типированием (по умолчанию combined_hla_out.xlsx)")
    ap.add_argument("--outdir", default="hla_forests", help="Папка для результатов")
    ap.add_argument("--allele-level", type=int, default=1, choices=[1, 2],
                    help="Уровень агрегации аллелей (1 или 2; по умолчанию 1)")
    ap.add_argument("--norm", choices=["log1p", "zscore"], default="log1p",
                    help="Нормализация титров (log1p|zscore)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Порог значимости FDR(BH), по умолчанию 0.05")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pheno = pd.read_csv(args.pheno, sep="\t")
    pheno = _normalize_cols(pheno)

    hla_long = load_hla_long(args.hla, allele_level=args.allele_level)

    all_tables = []

    for vacc in VACCINES.keys():
        groups = build_groups_for_vaccine(pheno, vacc, args.norm)
        if groups.empty:
            continue

        for gene in sorted(hla_long["Gene"].unique()):
            meta_df, per_region_df = compute_meta_for_vaccine_gene(groups, hla_long, gene)
            if meta_df.empty:
                continue

            meta_df = meta_df.copy()
            meta_df["p_fdr_bh"] = fdr_bh(meta_df["p_meta"].values)
            meta_df["signif_fdr"] = meta_df["p_fdr_bh"] <= args.alpha
            meta_df["vaccine"] = vacc

            sub_out = meta_df.sort_values(["p_fdr_bh", "p_meta", "Allele"])
            all_tables.append(sub_out)

            # выбрать 10 лучших
            sel = sub_out[sub_out["signif_fdr"]].copy()
            if sel.empty:
                sel = sub_out.nsmallest(10, ["p_meta"])
            else:
                sel = sel.nsmallest(10, ["p_fdr_bh", "p_meta"])

            # рисуем forest plots
            per_r = per_region_df.copy()
            for _, r in sel.iterrows():
                a = r["Allele"]
                rows = per_r[(per_r["Gene"] == gene) & (per_r["Allele"] == a)]
                if rows.empty:
                    continue
                overall = pd.Series({
                    "g": r["g_fixed"],
                    "ci_low": r["g_ci_low"],
                    "ci_high": r["g_ci_high"],
                    "p_meta": r["p_meta"],
                })
                title = f"{gene}-{a} | g={r['g_fixed']:.3f} [{r['g_ci_low']:.3f}; {r['g_ci_high']:.3f}], p={r['p_meta']:.3g}"
                plot_dir = outdir / f"{vacc}" / f"{gene}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                safe_a = a.replace("*", "-").replace(":", "_").replace("/", "_")
                save_path = plot_dir / f"{gene}-{safe_a}.png"
                forest_plot_png(save_path, title, rows[["Region","g","ci_low","ci_high"]], overall)

    if all_tables:
        master = pd.concat(all_tables, ignore_index=True)
        master = master[[
            "vaccine","Gene","Allele","k_regions",
            "g_fixed","g_ci_low","g_ci_high","p_meta","p_fdr_bh","signif_fdr"
        ]].sort_values(["vaccine","Gene","p_fdr_bh","p_meta","Allele"])
        master.to_csv(outdir / "allele_significance_tables.csv", index=False, encoding="utf-8")
    else:
        print("No results produced (check inputs)")

if __name__ == "__main__":
    main()