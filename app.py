"""PID/AID Dashboard — refactored module.

Колонки жёстко зафиксированы под этот файл:
- ID:            'ZLIMS ID'
- Возраст/пол:   'age', 'sex'
- Кори (колич.): 'measles_ME_ml'
- Флаги кори:    'measles_vaccine_info', 'measles_NoAnswer_coef'
- Краснуха:      'rubella_ME_ml', 'rubella_vaccine_info', 'rubella_NoAnswer_coef'
- Дифтерия:      'diphtheria_ME_ml', 'diphtheria_vaccine_info', 'diphtheria_NoAnswer_coef'
- Паротит:       'mumps_vaccine_info', 'mumps_NoAnswer_coef'
- HBV:           'HBV_vaccine_info', 'HBV_NoAnswer_coef', 'HBV_antiHBsAg_ME_ml'
- PCA:           'PC1'..'PC20'
- Регионы (флаги): 'is_from_Irkutsk','is_from_Amur','is_from_NiNo','is_from_Kaliningrad'
  → сводим в 'cohort_region'.

Внимание: в TSV нет 'measles_result' (0/1), поэтому разделы, требующие бинарного ответа, корректно
покажут предупреждение и отключатся.
"""
# app.py
from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import plotly.express as px
import plotly.graph_objects as go
import math

GENE_HEADER_RE = re.compile(r'^HLA-', re.IGNORECASE)
def normalize_gene_header(s: str) -> str:
    """'HLA-A' -> 'A', 'hla-DRB1' -> 'DRB1'."""
    return re.sub(GENE_HEADER_RE, '', str(s)).strip()

# ======== Minimal helpers to unblock HLA pipeline ========

def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def _clean_allele_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = s.replace("HLA-", "").replace("hla-", "")
    s = s.replace("~", "").replace("?", "").replace(" ", "")
    return s

def _allele_to_level(allele: str, level: int) -> str:
    # Expect like "A*01:01:01G" or "01:01" (gene will be prepended later)
    if "*" in allele:
        gene, rest = allele.split("*", 1)
    else:
        # when only digits "01:01", leave rest only
        parts = allele.split("*")
        gene = parts[0] if len(parts) > 1 else ""
        rest = allele if ":" in allele else allele

    # strip trailing letter suffixes (G, N, L, S, Q, etc.)
    rest = rest.split("+")[0]
    rest = rest.split("(")[0]
    rest = re.sub(r"[A-Za-z]$", "", rest)

    fields = rest.split(":")
    fields = fields[:level] if level in (1, 2) else fields
    rest_norm = ":".join(fields)
    return (gene + "*" if gene else "") + rest_norm

def process_hla_long(hla_wide: pd.DataFrame, allele_level: int) -> pd.DataFrame:
    """
    Convert wide HLA table (columns like 'A_1','A_2' or 'A_01:01','B_07:02', etc.)
    into long with columns: ID, Gene, Allele.
    """
    df = hla_wide.copy()
    # choose ID column
    id_col = None
    for c in ["ID", "Sample_ID", "sample_id", "ZLIMS ID", "ID_x"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise RuntimeError("Не найден ID столбец в HLA-таблице")

    id_series = df[id_col].astype("string").str.strip()
    keep_cols = [c for c in df.columns if c != id_col]

    rows = []
    for _, row in df[ [id_col] + keep_cols ].iterrows():
        pid = str(row[id_col]).strip()
        if not pid or pid.lower() in ("nan", "none"):
            continue
        for c in keep_cols:
            val = _clean_allele_text(row[c])
            if not val or val in (".", "-", "0", "nan", "None"):
                continue
            # gene from column name: take left part before underscore if present, else before first digit/asterisk
            col = str(c)
            gene = normalize_gene_header(col.split("_")[0]).upper()
            if "*" in val:
                # значение уже содержит ген — _clean_allele_text убирает префикс 'HLA-'
                full = _allele_to_level(val.upper(), allele_level)
            else:
                # без гена в значении: подставляем нормализованный gene
                full = _allele_to_level(f"{gene}*{val.upper()}", allele_level)
            rows.append((pid, gene, full))

    long = pd.DataFrame(rows, columns=["ID", "Gene", "Allele"]).dropna()
    # remove placeholders
    bad = set(["", ".", "-", "0", "NAN", "NONE"])
    long = long[~long["Allele"].astype(str).str.upper().isin(bad)]
    long["ID"] = long["ID"].astype(str)
    long["Gene"] = long["Gene"].astype(str)
    long["Allele"] = long["Allele"].astype(str)
    return long

def build_allele_dosage_matrix(hla_long: pd.DataFrame) -> pd.DataFrame:
    """
    Returns ID x Allele matrix with 0/1/2 dosage per allele.
    """
    if hla_long.empty:
        return pd.DataFrame()
    counts = (
        hla_long.groupby(["ID", "Allele"]).size().rename("dose").reset_index()
    )
    counts["dose"] = counts["dose"].clip(upper=2)
    mat = counts.pivot(index="ID", columns="Allele", values="dose").fillna(0).astype(int)
    mat.index = mat.index.astype(str)
    return mat

def normalize_and_fill(hla_wide: pd.DataFrame, allele_level: int) -> pd.DataFrame:
    """
    Produces normalized long table (ID, Gene, Allele) for zygosity calculation.
    """
    return process_hla_long(hla_wide, allele_level)

def gene_zygosity_table(hla_long_norm: pd.DataFrame) -> pd.DataFrame:
    """
    From long(ID,Gene,Allele) → zygosity per gene:
    homozygote if a single unique allele for (ID,Gene), else heterozygote.
    """
    if {"ID","Gene","Allele"}.issubset(set(hla_long_norm.columns)):
        g = (
            hla_long_norm.groupby(["ID","Gene"])["Allele"]
            .nunique()
            .reset_index(name="n_alleles")
        )
        g["Zygosity"] = np.where(g["n_alleles"] <= 1, "homozygote", "heterozygote")
        return g[["ID","Gene","Zygosity"]]
    # fallback (wide not expected here)
    return pd.DataFrame(columns=["ID","Gene","Zygosity"])

def normalize_antibody(s: pd.Series, method: str = "log1p") -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if method == "zscore":
        mu, sd = s.mean(skipna=True), s.std(ddof=1, skipna=True)
        return (s - mu) / sd if sd and sd > 0 else pd.Series(index=s.index, dtype=float)
    # default log1p on non-negative values; drop negatives
    s = s.where(s >= 0)
    return np.log1p(s)

def st_bar_chart_safe(df_counts: pd.DataFrame) -> None:
    try:
        st.bar_chart(df_counts)
    except Exception:
        try:
            import plotly.express as px
            cc = df_counts.reset_index()
            if cc.shape[1] == 2:
                x, y = cc.columns[0], cc.columns[1]
                st.plotly_chart(px.bar(cc, x=x, y=y), use_container_width=True)
        except Exception:
            pass

# ---- Stats helpers for meta-analysis (Hedges' g, p from z) ----
def _hedges_g_and_var(m1, s1, n1, m0, s0, n0):
    if n1 < 2 or n0 < 2 or s1 <= 0 or s0 <= 0:
        return None, None
    sp2 = (((n1 - 1) * (s1 ** 2)) + ((n0 - 1) * (s0 ** 2))) / (n1 + n0 - 2)
    sp = math.sqrt(sp2)
    d = (m1 - m0) / sp if sp > 0 else 0.0
    # small sample correction J
    J = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0))
    g = J * d
    var_g = ( (n1 + n0) / (n1 * n0) ) + ( (g ** 2) / (2 * (n1 + n0 - 2)) )
    return g, var_g

def _p_from_z(z):
    try:
        import mpmath as mp
        return float(2.0 * (1.0 - 0.5*(1+math.erf(abs(z)/math.sqrt(2)))))
    except Exception:
        # fallback using SciPy if present, else normal CDF approximation above already used
        return float(2.0 * (1.0 - 0.5*(1+math.erf(abs(z)/math.sqrt(2)))))

def _region_summary_for_allele(groups_all: pd.DataFrame, hla_long_gene: pd.DataFrame, allele: str) -> pd.DataFrame:
    # groups_all: ID, Antibody, Region ; hla_long_gene: ID,Gene,Allele
    carriers = (
        hla_long_gene.assign(is_car=lambda d: (d["Allele"].astype(str) == str(allele)).astype(int))
        .groupby("ID")["is_car"].sum().clip(upper=2)
    )
    df = groups_all[["ID","Region"]].drop_duplicates().copy()
    df["dose"] = df["ID"].map(carriers).fillna(0).astype(int)
    # per region counts
    agg = (
        df.groupby("Region")["dose"]
          .agg(N="count",
               hom=lambda s: int((s == 2).sum()),
               het=lambda s: int((s == 1).sum()),
               car=lambda s: int((s >= 1).sum()))
          .reset_index()
    )
    for col in ["hom","het","car"]:
        agg[f"{col}_freq"] = agg[col] / agg["N"].replace(0, np.nan)
    return agg

def _plot_region_freq_bars(freq_df: pd.DataFrame, allele_label: str = ""):
    try:
        import plotly.express as px
        long = freq_df.melt(id_vars=["Region","N"], value_vars=["hom_freq","het_freq","car_freq"],
                            var_name="type", value_name="freq")
        fig = px.bar(long, x="Region", y="freq", color="type",
                     barmode="group", title=f"Частоты носителей по регионам · {allele_label}",
                     labels={"freq":"Доля","Region":"Регион","type":"Тип"})
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass
# ======== End helpers ========

# --- HLA column/allele sanitizers ---
HLA_ALLELE_RE = re.compile(r'^(?:[A-Z]{1,4}[0-9]?[A-Z]?)\*[0-9]{2}(?::[0-9]{2}){0,3}[A-Z]*$')

def pick_hla_cols_by_values(df: pd.DataFrame, sample_rows: int = 200) -> list[str]:
    """Вернёт колонки, где встречаются строки формата HLA-аллелей (A*02:01 …).
    Смотрим только первые sample_rows строк ради скорости."""
    keep = []
    head = df.head(sample_rows)
    for c in df.columns:
        cs = str(c)
        # Если шапка формата "HLA-A_1" / "HLA-DRB1_2" — принимаем сразу
        if cs.startswith(("HLA-","hla-")) and "_" in cs:
            keep.append(c)
            continue
        s = head[c].astype(str)
        if s.str.contains(r'[A-Z]{1,4}\*\d{2}', regex=True, na=False).any():
            keep.append(c)
            keep.append(c)
    return keep

def sanitize_hla_long(hla_long: pd.DataFrame) -> pd.DataFrame:
    """Убираем мусорные «аллели», оставляем только валидные HLA по регексу."""
    if hla_long.empty:
        return hla_long
    out = hla_long.copy()
    out["Allele"] = (
        out["Allele"].astype(str)
        .str.strip()
        .str.replace(r'\s+', '', regex=True)
    )
    # отсекаем явные не-аллели
    bad_prefixes = ("SAMPLE", "SAMPLES", "ID", "NAME", "COMMENT")
    mask_valid = out["Allele"].str.match(HLA_ALLELE_RE)
    mask_bad_prefix = out["Allele"].str.startswith(bad_prefixes)
    out = out[mask_valid & ~mask_bad_prefix].copy()
    # пересчитаем Gene из аллеля (до звёздочки), чтобы не тянуть мусор
    out["Gene"] = out["Allele"].str.split("*").str[0]
    return out

groups_all: pd.DataFrame = pd.DataFrame()
hla_long_q: pd.DataFrame = pd.DataFrame()

# ======== Extra helpers to unblock "NoAnswer logit" and related code ========

# Константы-списки колонок, которые код использует
VACC_ID_COLS = ["ZLIMS ID", "ID", "sample_id", "ID_x"]
VACC_DATE_COLS = ["date_vaccination", "vacc_date", "date"]

def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coalesce_first_nonnull(df: pd.DataFrame, cols: List[str]) -> Optional[pd.Series]:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    s = df[cols[0]].copy()
    for c in cols[1:]:
        s = s.fillna(df[c])
    return s

# Паттерны для текстовой разметки (используются в табе "Метаданные")
PATTERNS = {
    "disregulation": r"\bдисрегуляц|dysregul",
    "cvid_ovin": r"\bсвич|cvid|овин",
    "avz_autoinfl": r"авз|autoinflamm",
    "neutropenia": r"нейтропен",
    "urticaria_skin": r"крапивниц|urticaria|кожн",
    "atopic_derm": r"атопическ.*дермат",
    "ige_mentioned": r"\bige\b|иммуноглобулин *e",
    "ige_elevated": r"ige.*(выс|повыш)",
    "asthma": r"астма|asthma",
    "edema_angio": r"ангиоот|квинке|edema",
    "rash": r"сып|rash|высып",
    "arthritis_joint": r"артрит|артроз|сустав",
    "anemia_thrombocytopenia": r"анеми|тромбоцитопен",
    "allergy_block": r"аллерг|sensiti|сенсиб",
    "sle": r"\bскв\b|lupus|sle",
    "behcet": r"бехчет|behcet",
    "autoimmune": r"аутоиммун",
}

# --- тесты частот для таба с бинарной корью (если появится measles_result) ---
def counts_and_tests(hla_long: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """
    groups: columns ['ID','Group'] где Group ∈ {0,1}
    hla_long: ['ID','Gene','Allele']
    """
    if hla_long.empty or groups.empty:
        return pd.DataFrame()

    g = (hla_long.assign(has_allele=1)
                  .drop_duplicates(["ID","Gene","Allele"])
                  .merge(groups, on="ID", how="inner"))

    rows = []
    for (gene, allele), sub in g.groupby(["Gene","Allele"]):
        # таблица 2x2: allele present vs result group
        pres = sub.groupby(["Group"])["has_allele"].sum()
        n_by_group = sub.groupby(["Group"])["ID"].nunique()
        a = int(pres.get(1, 0))                # carriers with result=1
        b = int(n_by_group.get(1, 0) - a)      # non-carriers with result=1
        c = int(pres.get(0, 0))                # carriers with result=0
        d = int(n_by_group.get(0, 0) - c)      # non-carriers with result=0

        n1 = int(n_by_group.get(1, 0))
        n0 = int(n_by_group.get(0, 0))
        if n1 + n0 == 0:
            continue

        # odds ratio + fisher/chi2
        test_name = "fisher"
        pval = np.nan
        or_est, ci_lo, ci_hi = np.nan, np.nan, np.nan
        try:
            # Fisher exact
            from math import isfinite
            odds, p = fisher_exact([[a, b], [c, d]])
            or_est, pval = float(odds), float(p)
            # простая Wald CI на log(OR), если все ячейки >0
            if min(a, b, c, d) > 0 and isfinite(or_est) and or_est > 0:
                se = np.sqrt(1/a + 1/b + 1/c + 1/d)
                lo = np.exp(np.log(or_est) - 1.96*se)
                hi = np.exp(np.log(or_est) + 1.96*se)
                ci_lo, ci_hi = float(lo), float(hi)
            else:
                ci_lo = ci_hi = np.nan
        except Exception:
            try:
                from scipy.stats import chi2_contingency
                chi2, p, _, _ = chi2_contingency([[a, b], [c, d]], correction=False)
                test_name, pval = "chi2", float(p)
            except Exception:
                pass

        rows.append(dict(
            Gene=str(gene), Allele=str(allele),
            count_res1=int(a), count_res0=int(c),
            n_res1=int(n1), n_res0=int(n0),
            freq_res1=(a / n1) if n1 else np.nan,
            freq_res0=(c / n0) if n0 else np.nan,
            delta_freq=((a / n1) - (c / n0)) if (n1 and n0) else np.nan,
            odds_ratio=or_est, or_ci_low=ci_lo, or_ci_high=ci_hi,
            test=test_name, p_value=pval,
        ))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    try:
        rej, pcor, _, _ = multipletests(out["p_value"], method="fdr_bh")
        out["p_fdr_bh"] = pcor
        out["signif_fdr"] = rej
    except Exception:
        out["p_fdr_bh"] = np.nan
        out["signif_fdr"] = False
    return out.sort_values(["p_fdr_bh","p_value","Allele"], na_position="last")

def plot_grouped_freq_with_sig(df_g: pd.DataFrame, gene_label: str, fdr_threshold: float, raw_p_fallback: float):
    """Простой столбчатый график частот для топ-аллелей гена."""
    try:
        import plotly.express as px
        show = df_g.copy()
        show["sig"] = np.where(
            (show["p_fdr_bh"].notna() & (show["p_fdr_bh"] <= fdr_threshold)) |
            (show["p_fdr_bh"].isna() & (show["p_value"] <= raw_p_fallback)),
            "*", ""
        )
        long = show.melt(
            id_vars=["Allele","sig"],
            value_vars=["freq_res1","freq_res0"],
            var_name="Group", value_name="freq"
        )
        long["Group"] = long["Group"].map({"freq_res1":"Result=1","freq_res0":"Result=0"})
        fig = px.bar(long, x="Allele", y="freq", color="Group", barmode="group",
                     title=f"{gene_label}: частоты в группах (звёздочка — значимо)",
                     labels={"freq":"Доля","Allele":"Аллель"})
        fig.update_layout(xaxis_tickangle=-25, margin=dict(l=10,r=10,t=40,b=10))
        return st.plotly_chart(fig, use_container_width=True)
    except Exception:
        return

def compute_vaccine_effect_sizes_by_region(vacc_df: pd.DataFrame, diseases: Dict[str,dict], norm_key: str) -> pd.DataFrame:
    """Hedges' g Responders vs Non-responders по регионам; если нет result/me_ml — возвращаем пусто."""
    rows = []
    if vacc_df.empty or "cohort_region" not in vacc_df.columns:
        return pd.DataFrame()
    for dis, meta in diseases.items():
        me_col = meta.get("me_ml")
        res_col = meta.get("result")
        if not res_col or not me_col or res_col not in vacc_df.columns or me_col not in vacc_df.columns:
            continue
        tmp = vacc_df[[me_col, res_col, "cohort_region"]].copy()
        tmp[me_col] = pd.to_numeric(tmp[me_col], errors="coerce")
        tmp = tmp.dropna(subset=[me_col, res_col, "cohort_region"])
        if tmp.empty: 
            continue
        # нормализация
        if norm_key == "zscore":
            tmp["val"] = tmp.groupby("cohort_region", group_keys=False)[me_col].apply(lambda s: normalize_antibody(s, "zscore"))
        else:
            tmp["val"] = normalize_antibody(tmp[me_col], "log1p")
        for reg, sub in tmp.groupby("cohort_region"):
            vals1 = sub.loc[sub[res_col] == 1, "val"].astype(float)
            vals0 = sub.loc[sub[res_col] == 0, "val"].astype(float)
            if len(vals1) < 3 or len(vals0) < 3:
                continue
            g, var_g = _hedges_g_and_var(vals1.mean(), vals1.std(ddof=1), len(vals1),
                                         vals0.mean(), vals0.std(ddof=1), len(vals0))
            if g is None or var_g is None:
                continue
            se = math.sqrt(var_g)
            rows.append(dict(
                disease=dis, region=str(reg),
                g=g, g_ci_low=g-1.96*se, g_ci_high=g+1.96*se
            ))
    return pd.DataFrame(rows)

def render_noanswer_logit_section(
    section_title: str,
    vacc_df: pd.DataFrame,
    hla_long: pd.DataFrame,
    selected_region: Optional[str],
    vacc_id_col: Optional[str],
    alpha_fdr: float,
    alpha_raw_fb: float,
    top_n_show: int,
    freq_thr_reg_pct: float,
):
    """
    Упрощённый вариант «логистической регрессии по NoAnswer».
    Делает сводку по *_NoAnswer_coef по выбранному региону и вакцинированным.
    Если бинарных outcome нет — рисует метрики и пропускает регрессию.
    """
    st.markdown("### Отсутствие ответов (NoAnswer) — сводка")
    if vacc_df.empty:
        st.info("Нет данных вакцинаций.")
        return

    # выбор региона и вакцинированных, если поле известно
    df = vacc_df.copy()
    if selected_region and "cohort_region" in df.columns:
        df = df[df["cohort_region"].astype(str) == str(selected_region)]
    # Пытаемся выделить «вакцинированных» по колонке measles_vaccine_info (если она есть)
    if "measles_vaccine_info" in df.columns:
        df = df[str_bool(df["measles_vaccine_info"])]

    # соберём все *_NoAnswer_coef
    noans_cols = [c for c in df.columns if c.endswith("_NoAnswer_coef")]
    if not noans_cols:
        st.info("В таблице нет колонок *_NoAnswer_coef — раздел пропущен.")
        return

    # Доли True по каждому такому флагу
    stats = []
    for c in noans_cols:
        s = str_bool(df[c])
        stats.append((c, float(s.mean()), int(s.sum()), int(s.notna().sum())))
    out = pd.DataFrame(stats, columns=["flag","share","count_true","N"]).sort_values("share", ascending=False)
    st.dataframe(out.assign(share=lambda d: (d["share"]*100).round(1)).rename(columns={"share":"% True"}), use_container_width=True)

    # Простой bar-chart
    try:
        st_bar_chart_safe(out.set_index("flag")[["share"]].rename(columns={"share":"share_true"}))
    except Exception:
        pass

    st.caption("Примечание: бинарного исхода (например, measles_result) в файле нет, поэтому модель логистической регрессии отключена в этой сборке.")
# ======== End extra helpers ========

# ========== Streamlit config ==========
st.set_page_config(page_title="PID/AID Dashboard", layout="wide")

# ========== Constants / Config ==========

# Жёстко заданные колонки под all_pheno_unrel.tsv
ID_COL_FIXED = "ZLIMS ID"
AGE_COL_FIXED = "age"
SEX_COL_FIXED = "sex"

# Флаги регионов в файле
REGION_FLAGS = {
    "is_from_Irkutsk": "Irkutsk",
    "is_from_Amur": "Amur",
    "is_from_NiNo": "Nizhny Novgorod",
    "is_from_Kaliningrad": "Kaliningrad",
}
REGION_COL = "cohort_region"  # создадим из флагов

# Вакцинные колонки (как в TSV)
MEAS_Q_COL_FIXED = "measles_ME_ml"
MEAS_NOANS_COL_FIXED = "measles_NoAnswer_coef"
MEAS_VACCINFO_COL_FIXED = "measles_vaccine_info"
MEAS_RES_COL_FIXED = None 

DISEASES = {
    "measles": {
        "title": "Корь",
        "me_ml": "measles_ME_ml",
        "result": None,  # В TSV нет бинарного ответа
        "sick": "measles_sick",  # присутствует
        "vacc_info": "measles_vaccine_info",
        "vacc_total": None,
        "noanswer": "measles_NoAnswer_coef",
    },
    "rubella": {
        "title": "Краснуха",
        "me_ml": "rubella_ME_ml",
        "result": None,
        "sick": "rubella_sick",
        "vacc_info": "rubella_vaccine_info",
        "vacc_total": None,
        "noanswer": "rubella_NoAnswer_coef",
    },
    "diphtheria": {
        "title": "Дифтерия",
        "me_ml": "diphtheria_ME_ml",
        "result": None,
        "sick": "diphtheria_sick",
        "vacc_info": "diphtheria_vaccine_info",
        "vacc_total": None,
        "noanswer": "diphtheria_NoAnswer_coef",
    },
    "mumps": {
        "title": "Паротит",
        "me_ml": None,
        "result": None,
        "sick": "mumps_sick",
        "vacc_info": "mumps_vaccine_info",
        "vacc_total": None,
        "noanswer": "mumps_NoAnswer_coef",
    },
    "HBV": {
        "title": "Гепатит B (HBV)",
        "me_ml": "HBV_antiHBsAg_ME_ml",
        "result": None,
        "sick": "HBV_sick",
        "vacc_info": "HBV_vaccine_info",
        "vacc_total": None,
        "noanswer": "HBV_NoAnswer_coef",
    },
}

# Остальные константы из прежнего кода (оставлены без изменений)
TEXT_COLS_CANDIDATES = [
    "clinical_discuss", "clinical_comments", "comments_interesting_vars",
    "INTERPRET_VAR_table_user", "INTERPRET_VAR_table",
    "IUIS_classification", "combi_diganosis_mcch52", "clin_manifestations",
    "autoiimunity", "markers",
]
HLA_ID_COLS = ["ID", "Sample_ID", "sample_id", "ZLIMS ID", "ID_x"]
SEQ_COLS_CANDIDATES = ["status_coverage", "VAR_IEI_panel_table_status", "SANGER"]
COMBI_COLS = ["Combi_NoAnswer_coef", "Combi_NoAnswer_TYPE", "fail_merge_tubes"]

# Для экспорта и HLA-файла по умолчанию
HLA_DEFAULT_PATH = "combined_hla_out.xlsx"

# ===== Утилиты (без изменений, кроме мелких зависимостей) =====
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def str_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["1", "true", "yes", "да", "y", "истина"])

def to_datetime_inplace(df: pd.DataFrame, col: str) -> None:
    if col in df.columns and pd.api.types.is_string_dtype(df[col]):
        with pd.option_context("mode.chained_assignment", None):
            df[col] = pd.to_datetime(df[col], errors="coerce")

def concat_text(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if not cols:
        return pd.Series(index=df.index, dtype="string")
    return df[cols].astype(str).agg(" | ".join, axis=1).str.lower()

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    bio.seek(0)
    return bio.read()

# ========== Жёсткая загрузка данных из TSV ==========
@st.cache_data(show_spinner=False)
def read_tsv_fixed(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

def derive_region_from_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Строим колонку cohort_region из фиксированного набора булевых флагов REGION_FLAGS."""
    df = df.copy()
    def pick_region(row) -> Optional[str]:
        hits = [name for flag, name in REGION_FLAGS.items()
                if flag in row.index and bool(row[flag])]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return "Mixed"
        return None
    df[REGION_COL] = df.apply(pick_region, axis=1)
    return df

# ===== Ниже остаётся вся аналитическая логика как была, с минимальными вкраплениями =====

# Сайдбар: данные (файла мета и вакцинаций больше не спрашиваем)
st.sidebar.header("Данные")
st.sidebar.info("Метаданные и вакцинации загружаются из одного файла TSV:\n"
                "`all_pheno_unrel.tsv`")

# ЖЁСТКАЯ загрузка и нормализация
try:
    df = read_tsv_fixed("all_pheno_unrel.tsv")
    vacc_df = df.copy()  # мета и вакцинации — это одна таблица
    st.sidebar.success("Файл загружен: all_pheno_unrel.tsv")
except Exception as e:
    st.error(f"Не удалось загрузить TSV: {e}")
    st.stop()

df = _normalize_cols(df)
vacc_df = _normalize_cols(vacc_df)

# Добавляем колонку региона из флагов
df = derive_region_from_flags(df)
vacc_df = derive_region_from_flags(vacc_df)

# Явно укажем фиксированные ID/поля для дальнейшего кода
vacc_id_col = ID_COL_FIXED if ID_COL_FIXED in vacc_df.columns else None
meas_vacc_col = MEAS_VACCINFO_COL_FIXED if MEAS_VACCINFO_COL_FIXED in vacc_df.columns else None
meas_q_col = MEAS_Q_COL_FIXED if MEAS_Q_COL_FIXED in vacc_df.columns else None

# --- region selection ---
selected_region: Optional[str] = None
if REGION_COL in vacc_df.columns:
    regions = sorted(vacc_df[REGION_COL].dropna().astype(str).unique())
    if regions:
        selected_region = st.sidebar.selectbox("Регион для анализа (обязательно)", regions, index=0)
    else:
        st.sidebar.warning("Колонка региона есть, но значения отсутствуют.")
else:
    st.sidebar.warning(f"Нет колонки '{REGION_COL}' — региональный анализ будет ограничен.")

# ======== Ниже — исходный код приложения, БЕЗ автопоиска имён для этих полей ========
# Нормализация (выбор) метода
norm_method = st.sidebar.selectbox(
    "Нормализация титров (количественный анализ)",
    ["log1p (по умолчанию)", "z-score"],
    index=0
)
norm_method_key = "log1p" if norm_method.startswith("log1p") else "zscore"

# --------------- HLA загрузка как раньше ----------------
hla_uploaded = st.sidebar.file_uploader("Загрузите HLA-таблицу (.xlsx)", type=["xlsx"], key="hla_upl")
from typing import Tuple
@st.cache_data(show_spinner=False)
def read_xlsx(path_or_file):
    return pd.read_excel(path_or_file)

try:
    if hla_uploaded is not None:
        hla_df = read_xlsx(hla_uploaded)
        st.sidebar.info("Используется загруженный HLA-файл.")
    else:
        hla_df = read_xlsx(HLA_DEFAULT_PATH)
        st.sidebar.info(f"Используется HLA-файл по умолчанию: {HLA_DEFAULT_PATH}")
except Exception as e:
    st.sidebar.error(f"Не удалось загрузить HLA: {e}")
    hla_df = pd.DataFrame()

hla_df = _normalize_cols(hla_df) if not hla_df.empty else hla_df

# --- HLA analysis params ---
allele_level = st.sidebar.selectbox("HLA: уровень агрегации аллеля", [1, 2], index=0, help="1 → A*01; 2 → A*01:01")
min_carriers = int(st.sidebar.number_input("Мин. число копий аллеля для показа", min_value=1, value=1, step=1))

alpha_fdr = float(st.sidebar.number_input("Порог FDR (BH) для значимости", min_value=0.0, max_value=0.25, value=0.05, step=0.01))
alpha_raw_fb = float(st.sidebar.number_input("Fallback порог raw p (если FDR значимых нет)", min_value=0.0, max_value=0.25, value=0.10, step=0.01))
top_n_show = int(st.sidebar.number_input("Top-N для показа при отсутствии FDR-значимых", min_value=5, value=20, step=1))

# — Порог частоты аллеля (для мульти-ген модели), % по дозе
freq_thr_reg_pct = float(st.sidebar.number_input("Порог f для мульти-ген модели, % (по дозе)", min_value=0.0, max_value=5.0, value=1.0, step=0.1))

top_n_plot = int(st.sidebar.number_input("Top-N аллелей для графика", min_value=5, value=10, step=1))
alpha_fdr_plot = float(st.sidebar.number_input("Порог FDR для отметки *", min_value=0.0, max_value=0.25, value=0.05, step=0.01))

genes_all = sorted({
    normalize_gene_header(str(c).split("_")[0]).upper()
    for c in hla_df.columns if "_" in str(c)
}) if not hla_df.empty else []
genes_pick = st.sidebar.multiselect("Гены HLA (для анализа)", genes_all, default=genes_all)

# --- ID coalescing ---
def coalesce_id(df_: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    return coalesce_first_nonnull(df_, list(candidates))

hla_id_series: Optional[pd.Series] = None
if not hla_df.empty:
    if "sample_id" in hla_df.columns:
        hla_id_series = hla_df["sample_id"].astype("string").str.strip().replace({"": pd.NA})
    else:
        hla_id_series = coalesce_id(hla_df, HLA_ID_COLS)


# --- Prepare HLA long (filtered by selected genes) ---
hla_long = pd.DataFrame()
if not hla_df.empty and hla_id_series is not None and hla_id_series.notna().sum() > 0 and genes_pick:
    keep_cols = [c for c in HLA_ID_COLS if c in hla_df.columns]
    hla_cols  = pick_hla_cols_by_values(hla_df)               # <-- только реальные HLA-поля
    if genes_pick:
        # сузим по выбранным генам уже после melt через sanitize_hla_long
        hla_sub = hla_df[keep_cols + hla_cols].copy()
    else:
        hla_sub = hla_df[keep_cols + hla_cols].copy()

    hla_sub["ID"] = hla_id_series.astype("string").str.strip()
    hla_sub = hla_sub[hla_sub["ID"].notna() & (hla_sub["ID"] != "")]

    hla_long = process_hla_long(hla_sub, allele_level)
    hla_long = sanitize_hla_long(hla_long)                    # <-- фильтрация аллелей

if genes_pick:
    _pick = {g.upper() for g in genes_pick}
    hla_long = hla_long[hla_long["Gene"].str.upper().isin(_pick)]

with st.expander("Проверка агрегации HLA-аллелей", expanded=False):
    st.write("Уровень:", allele_level)
    st.write("Примеры аллелей (первые 10):", list(hla_long["Allele"].dropna().astype(str).unique())[:10])

# ========== Title ==========
st.title("PID/AID — Метаданные")

st.markdown("## Верхняя сводка: логистическая регрессия по NoAnswer")
render_noanswer_logit_section(
    section_title="",
    vacc_df=vacc_df,
    hla_long=hla_long,
    selected_region=selected_region,
    vacc_id_col=vacc_id_col,
    alpha_fdr=alpha_fdr,
    alpha_raw_fb=alpha_raw_fb,
    top_n_show=top_n_show,
    freq_thr_reg_pct=freq_thr_reg_pct,
)

cA, cB = st.columns(2)
# ========== Text-marking & summary ==========
text_cols = pick_existing(df, TEXT_COLS_CANDIDATES)

df_marked = df.copy()
if text_cols:
    texts = concat_text(df_marked, text_cols)
    for key, pat in PATTERNS.items():
        df_marked[key] = texts.str.contains(pat, regex=True, na=False, case=False)
else:
    for key in PATTERNS:
        df_marked[key] = False

presence_cols = list(PATTERNS.keys())
binary_table = df_marked[presence_cols].copy().astype(bool)
id_show_cols = pick_existing(df_marked, ["ID", "uin2_number", "uin2_fulltxt", "zlims_id", "ZLIMS ID", "ID_x"])
binary_table = pd.concat([df_marked[id_show_cols], binary_table], axis=1) if id_show_cols else binary_table

# ========== Tabs ==========
tab1, tab2, tab3, tab_reg, tab_freq = st.tabs(
    ["🧬 Метаданные", "💉 Вакцины", "🧬⇄💉 HLA × Measles (бинарный)", "📈 Регрессия (HLA+пол → титр)", "Частоты"]
)

with tab1:
    st.subheader("Сводка по меткам (в текущей выборке)")
    counts = df_marked[presence_cols].sum().sort_values(ascending=False).rename("count").to_frame()
    st.dataframe(counts)

    st.markdown("### Гистограммы по метаданным")
    try:
        st_bar_chart_safe(counts)
    except Exception:
        st.write("Не удалось построить bar chart для сводки.")

    try:
        matches_per_row = df_marked[presence_cols].astype(bool).sum(axis=1)
        st.markdown("**Распределение количества упоминаний на пациента**")
        hist_df = matches_per_row.value_counts().sort_index().rename_axis("num_flags").to_frame("patients")
        st_bar_chart_safe(hist_df)
    except Exception:
        st.write("Не удалось построить гистограмму распределения по пациентам.")

    st.markdown("### Таблица «есть/нет» по пациентам")
    st.dataframe(binary_table)

    st.markdown("### Пациенты без упоминаний (все метки = False)")
    try:
        no_flags_mask = ~df_marked[presence_cols].any(axis=1)
        df_no_flags = df.loc[no_flags_mask]
        if not df_no_flags.empty:
            st.dataframe(df_no_flags)
            st.caption(f"Показано {len(df_no_flags)} строк без упоминаний.")
        else:
            st.info("Все пациенты имеют хотя бы одну метку.")
    except Exception as e:
        st.warning(f"Не удалось вычислить пациентов без упоминаний: {e}")

    st.markdown("### Экспорт таблиц")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.download_button("⬇️ 'есть/нет' (CSV)", data=to_csv_bytes(binary_table), file_name="binary_table.csv", mime="text/csv")
    with col_b:
        st.download_button("⬇️ 'есть/нет' (XLSX)", data=to_xlsx_bytes(binary_table), file_name="binary_table.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col_c:
        st.download_button("⬇️ Сводка (CSV)", data=to_csv_bytes(counts.reset_index()), file_name="summary_counts.csv", mime="text/csv")
    with col_d:
        st.download_button("⬇️ Сводка (XLSX)", data=to_xlsx_bytes(counts.reset_index()),
                           file_name="summary_counts.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("### Описание полей")
    st.markdown(
        """
- **disregulation** — упоминания дисрегуляции иммунитета
- **cvid_ovin** — ОВИН / CVID
- **avz_autoinfl** — АВЗ / аутовоспаление
- **neutropenia** — нейтропения
- **urticaria_skin** — крапивница / кожный синдром
- **atopic_derm** — атопический дерматит / дерматит
- **ige_mentioned** — упоминание IgE
- **ige_elevated** — явное упоминание повышения IgE
- **asthma** — астма
- **edema_angio** — отёки/ангиоотёк/Квинке/НАО
- **rash** — высыпания
- **arthritis_joint** — артрит/артроз/суставы
- **anemia_thrombocytopenia** — анемии/тромбоцитопении
- **allergy_block** — аллергия/сенсибилизация/пыльца
- **sle** — СКВ
- **behcet** — болезнь Бехчета
- **autoimmune** — «аутоиммун-»
"""
    )

with tab2:
    st.subheader("Метаданные по вакцинациям")

    if vacc_df.empty:
        st.error("Файл вакцинаций не загружен — см. сайдбар.")
        st.stop()

    col_id = ID_COL_FIXED  # "ZLIMS ID"
    col_date = None  
    if col_date:
        to_datetime_inplace(vacc_df, col_date)

    st.markdown("### Сводка")
    m = [("Всего записей (доз)", len(vacc_df))]
    if col_id:
        m.append(("Уникальных пациентов", vacc_df[col_id].nunique()))
    metrics_df = pd.DataFrame(m, columns=["metric", "value"])
    st.dataframe(metrics_df, hide_index=True)

    st.markdown("### Демография")
    for col in ["age", "sex", "cohort_region", "test_tube", "date_birth"]:
        if col in vacc_df.columns:
            s = vacc_df[col]
            if pd.api.types.is_numeric_dtype(s):
                med = pd.to_numeric(s, errors="coerce").median()
                st.write(f"**{col}** — медиана: {med:.1f}")
            else:
                vc = s.astype(str).str.strip().replace({"": pd.NA}).dropna().value_counts().head(15)
                if not vc.empty:
                    st.dataframe(vc.rename_axis(col).to_frame("count"))
                    try:
                        st_bar_chart_safe(vc.rename_axis(col).to_frame("count"))
                    except Exception:
                        pass

    st.markdown("### Секвенирование / техметаданные")
    for col in [
        "Проект", "Платформы", "Чтения", "Cov", "Ген. Портрет", "Втор. Находки", "Рец. Носит.",
        "Мед. Находки", "Мед. Стат.", "Пол/ УАП", "Дубли/ Родство", "chr CNV", "Гаплогруппы",
        "DNB", "Статус", "IGV", "Дата обсчета", "Комментарий",
    ]:
        if col in vacc_df.columns:
            s = vacc_df[col]
            if pd.api.types.is_numeric_dtype(s):
                try:
                    st.metric(col, f"{pd.to_numeric(s, errors='coerce').median():.0f}")
                except Exception:
                    st.metric(col, "—")
            else:
                vc = s.astype(str).str.strip().replace({"": pd.NA}).dropna().value_counts().head(15)
                if not vc.empty:
                    st.dataframe(vc.rename_axis(col).to_frame("count"))
                    try:
                        st_bar_chart_safe(vc.rename_axis(col).to_frame("count"))
                    except Exception:
                        pass

    st.markdown("### Комбинированные флаги")
    for c in COMBI_COLS:
        if c in vacc_df.columns:
            s = str_bool(vacc_df[c])
            st.write(f"**{c}** — доля: {float(s.mean()):.1%}")

    st.markdown("### Нозологии")
    def _clean(s): return s.astype(str).str.strip().replace({"": pd.NA})
    def _num(s): return pd.to_numeric(s, errors="coerce")

    for key, meta in DISEASES.items():
        st.markdown(f"#### {meta['title']}")
        me_col = meta.get("me_ml")
        res_col = meta.get("result")
        sick_col = meta.get("sick")
        info_col = meta.get("vacc_info")
        total_col = meta.get("vacc_total")
        na_col = meta.get("noanswer")
        extra = [c for c in meta.get("result_extra", []) if c in vacc_df.columns]

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Строк", f"{len(vacc_df):,}".replace(",", " "))
        with c2:
            if col_id:
                st.metric("Пациентов", f"{vacc_df[col_id].nunique():,}".replace(",", " "))
            else:
                st.metric("Пациентов", "—")
        with c3:
            if sick_col in vacc_df.columns:
                st.metric("Болел", f"{float(str_bool(vacc_df[sick_col]).mean()):.1%}")
            else:
                st.metric("Болел", "—")
        with c4:
            if na_col in vacc_df.columns:
                st.metric("Нет ответа", f"{float(str_bool(vacc_df[na_col]).mean()):.1%}")
            else:
                st.metric("Нет ответа", "—")

        if res_col in vacc_df.columns:
            vc = _clean(vacc_df[res_col]).dropna().value_counts()
            if not vc.empty:
                st.dataframe(vc.rename_axis(res_col).to_frame("count"))
                try:
                    st_bar_chart_safe(vc.rename_axis(res_col).to_frame("count"))
                except Exception:
                    pass

        for c in extra:
            vc = _clean(vacc_df[c]).dropna().value_counts()
            if not vc.empty:
                st.dataframe(vc.rename_axis(c).to_frame("count"))
                try:
                    st_bar_chart_safe(vc.rename_axis(c).to_frame("count"))
                except Exception:
                    pass

        if me_col and me_col in vacc_df.columns:
            vals = _num(vacc_df[me_col]).dropna()
            if not vals.empty:
                try:
                    bins = pd.qcut(vals[vals >= 0], q=min(20, max(5, int(len(vals) ** 0.5))), duplicates="drop")
                    hist = bins.value_counts().sort_index().rename_axis("bin").to_frame("count")
                except Exception:
                    hist = pd.cut(vals, bins=20).value_counts().sort_index().rename_axis("bin").to_frame("count")
                st_bar_chart_safe(hist)
            else:
                st.info("Нет численных значений титров для гистограммы.")

        if info_col in vacc_df.columns:
            vc = _clean(vacc_df[info_col]).dropna().value_counts()
            if not vc.empty:
                st.dataframe(vc.rename_axis(info_col).to_frame("count"))
                try:
                    st_bar_chart_safe(vc.rename_axis(info_col).to_frame("count"))
                except Exception:
                    pass

        if total_col in vacc_df.columns:
            doses = _num(vacc_df[total_col]).fillna(0).astype(int)
            vc = doses.value_counts().sort_index().rename_axis("doses").to_frame("patients")
            st.dataframe(vc)
            try:
                st_bar_chart_safe(vc)
            except Exception:
                pass

        with st.expander("Сырые поля по нозологии"):
            show = [c for c in [col_id, me_col, res_col, sick_col, info_col, total_col, na_col] if c]
            show += extra
            show = [c for c in show if c in vacc_df.columns]
            if show:
                st.dataframe(vacc_df[show].head(200))
            else:
                st.write("Нет колонок для отображения.")

    st.markdown("### Распределение числа доз на пациента")
    if col_id:
        doses_per_patient = vacc_df.groupby(col_id).size().rename("doses").reset_index()
        st.dataframe(doses_per_patient.head(100))
        try:
            hist = doses_per_patient["doses"].value_counts().sort_index().rename_axis("doses").to_frame("patients")
            st_bar_chart_safe(hist)
        except Exception:
            pass
    else:
        st.warning("Не найдена колонка ID пациента — пропускаем блок с дозами на пациента.")

    st.markdown("### Экспорт данных по вакцинациям")
    col_va, col_vb = st.columns(2)
    with col_va:
        st.download_button("⬇️ Сырые данные (CSV)", data=to_csv_bytes(vacc_df), file_name="vaccinations_raw.csv", mime="text/csv")
    with col_vb:
        st.download_button("⬇️ Сырые данные (XLSX)", data=to_xlsx_bytes(vacc_df), file_name="vaccinations_raw.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


    st.markdown("### Размер эффекта по регионам (Hedges’ g, по титрам)")
    eff_df = compute_vaccine_effect_sizes_by_region(
        vacc_df=vacc_df,
        diseases=DISEASES,
        norm_key=norm_method_key
    )

    if eff_df.empty:
        st.info("Недостаточно данных для расчёта Hedges’ g по регионам.")
    else:
        for disease, meta in DISEASES.items():
            title = meta.get("title", disease)
            df_sub = eff_df[eff_df["disease"] == disease]
            if df_sub.empty:
                continue

            st.markdown(f"#### 💉 {title}")

            fig = px.line(
                df_sub,
                x="region",
                y="g",
                error_y=df_sub["g_ci_high"] - df_sub["g"],
                error_y_minus=df_sub["g"] - df_sub["g_ci_low"],
                markers=True,
                title=f"{title}: Hedges’ g по регионам (Responders vs Non-responders)",
                labels={"region": "Регион", "g": "Hedges’ g"},
            )
            fig.update_traces(line=dict(width=2))
            fig.update_layout(
                yaxis_title="Hedges’ g",
                xaxis_title="Регион",
                xaxis_tickangle=-20,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if MEAS_RES_COL_FIXED is None:
        st.info("В TSV нет бинарного результата по кори (measles_result). Раздел отключён.")
        st.stop()
    st.subheader("Ассоциации HLA ↔︎ антитела к кори (measles)")
    if not selected_region:
        st.warning("Выберите регион в сайдбаре — анализ проводится только внутри одного региона.")
        st.stop()
    if vacc_df.empty or hla_df.empty or hla_long.empty or not vacc_id_col:
        st.error("Нет необходимых данных (HLA/вакцины/ID).")
        st.stop()

    with st.expander("Обнаруженные ID-колонки", expanded=False):
        st.write({
            "HLA_ID_candidates_present": [c for c in HLA_ID_COLS if c in hla_df.columns],
            "HLA_ID_used_nonnull": int(hla_id_series.notna().sum()) if hla_id_series is not None else 0,
            "VACC_ID_col_used": vacc_id_col,
        })

    st.caption(f"Выбранный регион: **{selected_region}**")

    vacc_work = vacc_df.copy()
    vacc_work = vacc_work[
        (vacc_work[REGION_COL].astype(str) == str(selected_region)) & str_bool(vacc_work[meas_vacc_col])
    ]
    meas_res = pd.to_numeric(vacc_work[meas_res_col], errors="coerce")
    vacc_work = vacc_work.loc[meas_res.isin([0, 1])].copy()
    vacc_work[vacc_id_col] = vacc_work[vacc_id_col].astype(str)
    vacc_work["Group"] = meas_res.loc[vacc_work.index].astype(int)

    groups = vacc_work[[vacc_id_col, "Group"]].rename(columns={vacc_id_col: "ID"}).dropna()
    groups["ID"] = groups["ID"].astype(str)

    # фильтр по выбранным генам
    if genes_pick:
        keep_cols = [c for c in HLA_ID_COLS if c in hla_df.columns]
        _pick = {g.upper() for g in (genes_pick or [])}
        gene_cols = [
            c for c in hla_df.columns
            if "_" in str(c)
            and normalize_gene_header(str(c).split("_")[0]).upper() in _pick
        ]
        hla_sub = hla_df[keep_cols + gene_cols].copy()
    else:
        hla_sub = hla_df.copy()

    hla_sub["ID"] = hla_id_series.astype("string").str.strip()
    hla_sub = hla_sub[hla_sub["ID"].notna() & (hla_sub["ID"] != "")]

    try:
        hla_long_local = process_hla_long(hla_sub, allele_level)
        hla_long_local["ID"] = hla_long_local["ID"].astype(str)
    except Exception as e:
        st.error(f"HLA нормализация не удалась: {e}")
        st.stop()

    res = counts_and_tests(hla_long_local, groups)
    if res.empty:
        st.info("Недостаточно данных для тестов. Проверьте пересечение ID и фильтры.")
        st.stop()

    res = res[(res["count_res1"] + res["count_res0"]) >= int(min_carriers)]

    st.markdown("### Результаты по каждому гену (χ² / Fisher, FDR BH)")
    show_cols = [
        "Gene", "Allele", "count_res1", "count_res0", "n_res1", "n_res0",
        "freq_res1", "freq_res0", "delta_freq",
        "odds_ratio", "or_ci_low", "or_ci_high",   # NEW
        "test", "p_value", "p_fdr_bh", "signif_fdr",
    ]
    for g in sorted(res["Gene"].dropna().unique()):
        res_g = res[res["Gene"] == g].sort_values(["p_fdr_bh", "p_value", "Allele"])
        st.subheader(f"Ген: {g}")
        if res_g.empty:
            st.info("Нет данных для этого гена после фильтров.")
            continue
        st.dataframe(res_g[show_cols], hide_index=True, use_container_width=True)
        top_g = res_g.nsmallest(top_n_plot, ["p_fdr_bh", "p_value"])
        if not top_g.empty:
            plot_grouped_freq_with_sig(top_g, gene_label=g, fdr_threshold=alpha_fdr, raw_p_fallback=alpha_raw_fb)


    st.markdown("### Экспорт результатов")
    st.download_button(
        "⬇️ Скачать результаты (CSV)",
        data=to_csv_bytes(res),
        file_name="hla_measles_associations.csv",
        mime="text/csv",
    )

# ========== Quantitative tab (expander) ==========
with st.expander("🧬⇄💉 HLA × Measles (количественный титр)", expanded=False):
    st.subheader("Ассоциации HLA ↔ количество антител (measles_ME_ml)")
    if meas_q_col is None:
        st.error(
            "Не удалось найти колонку с количественным титром кори. "
            "Переименуйте поле (например, 'measles_ME_ml') или добавьте его имя в MEASLES_Q_COLS."
        )
        st.stop()
    if meas_q_col not in vacc_df.columns or meas_vacc_col not in vacc_df.columns:
        st.warning("В таблице вакцинаций нет measles_ME_ml / measles_vaccine_info.")
        st.stop()
    if selected_region is None:
        st.warning("Выберите регион в сайдбаре.")
        st.stop()

    vacc_df[vacc_id_col] = vacc_df[vacc_id_col].astype(str)
    vacc_q = vacc_df[(vacc_df[REGION_COL].astype(str) == str(selected_region)) & str_bool(vacc_df[meas_vacc_col])].copy()
    vacc_q[meas_q_col] = pd.to_numeric(vacc_q[meas_q_col], errors="coerce")
    vacc_q["Antibody"] = normalize_antibody(vacc_q[meas_q_col], norm_method_key)
    vacc_q = vacc_q.dropna(subset=["Antibody"])

    groups_q = vacc_q[[vacc_id_col, "Antibody"]].rename(columns={vacc_id_col: "ID"})
    groups_q["ID"] = groups_q["ID"].astype(str)


    # HLA long (те же гены/уровень)
    if genes_pick:
        keep_cols = [c for c in HLA_ID_COLS if c in hla_df.columns]
        gene_cols = [c for c in hla_df.columns if "_" in str(c) and c.split("_")[0] in genes_pick]
        hla_sub_q = hla_df[keep_cols + gene_cols].copy()
    else:
        hla_sub_q = hla_df.copy()

    hla_sub_q["ID"] = hla_id_series.astype("string").str.strip()
    hla_sub_q = hla_sub_q[hla_sub_q["ID"].notna() & (hla_sub_q["ID"] != "")]
    hla_long_q = process_hla_long(hla_sub_q, allele_level)
    hla_long_q = sanitize_hla_long(hla_long_q)
    hla_long_q["ID"] = hla_long_q["ID"].astype(str)

    merged_q = hla_long_q.merge(groups_q, on="ID", how="inner")
    merged_q = merged_q[~merged_q["Allele"].astype(str).isin([".", "-"])]

    # NEW: таблица дозировок по аллелям (0/1/2) для варианта "по аллелю"
    hla_dose_q = build_allele_dosage_matrix(hla_long_q)  # индекс=ID, колонки=Allele (0/1/2)
    hla_dose_q.index = hla_dose_q.index.astype(str)

    # NEW: зиготность по гену
    # Берём wide-форму hla_sub_q (та, из которой делали hla_long_q), нормализуем и считаем зиготность
    try:
        hla_wide_norm = normalize_and_fill(hla_sub_q, allele_level)
        hla_wide_norm["ID"] = hla_wide_norm["ID"].astype(str)
        gene_zygo = gene_zygosity_table(hla_wide_norm)  # ID, Gene, Zygosity
    except Exception as e:
        gene_zygo = pd.DataFrame(columns=["ID","Gene","Zygosity"])

    res_q = []
    for g in sorted(merged_q["Gene"].unique()):
        sub = merged_q[merged_q["Gene"] == g]
        for allele in sub["Allele"].unique():
            pres = (sub.assign(is_a=sub["Allele"].eq(allele).astype(int)).groupby("ID")["is_a"].max())
            ant = sub.groupby("ID")["Antibody"].first()
            df_a = pd.concat([pres, ant], axis=1).dropna()
            vals_yes = df_a.loc[df_a["is_a"] == 1, "Antibody"]
            vals_no = df_a.loc[df_a["is_a"] == 0, "Antibody"]
            if len(vals_yes) < 3 or len(vals_no) < 3:
                continue
            try:
                _, p = mannwhitneyu(vals_yes, vals_no, alternative="two-sided")
                res_q.append(
                    dict(
                        Gene=g,
                        Allele=allele,
                        n_carriers=int(len(vals_yes)),
                        n_noncarriers=int(len(vals_no)),
                        mean_carriers=float(vals_yes.mean()),
                        mean_noncarriers=float(vals_no.mean()),
                        delta_mean=float(vals_yes.mean() - vals_no.mean()),
                        p_value=float(p),
                    )
                )
            except Exception:
                continue

    res_q = pd.DataFrame(res_q)
    if res_q.empty:
        st.info("Недостаточно данных для количественного анализа.")
        st.stop()

    reject, pcor, _, _ = multipletests(res_q["p_value"], method="fdr_bh")
    res_q["p_fdr_bh"] = pcor
    res_q["signif_fdr"] = reject
    res_q = res_q.sort_values("p_fdr_bh")

    norm_label = "log1p" if norm_method_key == "log1p" else "z-score"

    st.markdown("### Графики значимых аллелей")
    max_plots = int(st.number_input("Максимум графиков для показа", min_value=1, value=12, step=1))
    alpha_raw = float(st.number_input("Порог p-value без поправки (фолбэк)", min_value=1e-6, max_value=0.1, value=0.05, step=0.01, format="%.6f"))

    to_plot = None
    sig_fdr = res_q[res_q["signif_fdr"]]
    if not sig_fdr.empty:
        to_plot = sig_fdr.sort_values(["p_fdr_bh", "p_value"]).head(max_plots)
        st.caption("Показаны аллели, значимые по FDR (BH).")
    else:
        sig_raw = res_q[res_q["p_value"] < alpha_raw]
        if not sig_raw.empty:
            to_plot = sig_raw.sort_values("p_value").head(max_plots)
            st.caption(f"По FDR значимых нет. Показаны аллели со значимостью по сырому p < {alpha_raw:g}.")
        else:
            st.info("Значимых аллелей не найдено при текущих порогах.")

    # лёгкая отрисовка без seaborn (здесь сохраним логику: st.bar_chart для средних)
    if to_plot is not None and not to_plot.empty:
        for _, row in to_plot.iterrows():
            g, a = row["Gene"], row["Allele"]
            sub_g = merged_q[merged_q["Gene"] == g].copy()
            if sub_g.empty:
                continue

            # готовим таблицу: у кого есть аллель (1/0) и их титры
            pres = (sub_g.assign(is_a=sub_g["Allele"].eq(a).astype(int))
                        .groupby("ID")["is_a"].max())
            ant  = sub_g.groupby("ID")["Antibody"].first()
            df_a = pd.concat([pres, ant], axis=1).dropna().reset_index()

            # подписи групп
            df_a["Group"] = np.where(df_a["is_a"] == 1, "Carrier", "Other")

            # boxplot с точками
            p_fdr = row["p_fdr_bh"]
            p_fdr_txt = f"{p_fdr:.3g}" if pd.notna(p_fdr) else "—"

            fig = px.box(
                df_a,
                x="Group", y="Antibody", color="Group",
                points="all",               # показываем индивидуальные точки
                labels={"Antibody": f"Титр антител (норм.: {norm_label})", "Group": "Группа"},
                title=f"**{g}-{a}** | p={row['p_value']:.3g}, p(FDR)={p_fdr_txt}"
            )

            # отображать среднее (горизонтальная линия внутри бокса)
            fig.update_traces(boxmean=True)

            # немного косметики
            fig.update_layout(
                showlegend=False,
                yaxis_title="Распределение титров (boxplot)",
                xaxis_title="",
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Результаты U-теста (Манна–Уитни)")
    for g in sorted(res_q["Gene"].dropna().unique()):
        sub = res_q[res_q["Gene"] == g].sort_values("p_fdr_bh")
        st.subheader(f"Ген: {g}")
        if sub.empty:
            st.info("Нет результатов для этого гена.")
            continue
        st.dataframe(sub, hide_index=True, use_container_width=True)

    st.download_button(
        "⬇️ Количественный анализ (CSV)",
        data=to_csv_bytes(res_q),
        file_name="hla_measles_antibody_levels.csv",
        mime="text/csv",
    )

    st.markdown("## Зиготность ↔ титр антител")

    zygo_mode = st.radio(
        "Что сравниваем?",
        ["По гену: homozygote vs heterozygote", "По аллелю: dose=2 vs dose=1 среди носителей"],
        index=0,
        help="По гену — сравнение по паре *_1/*_2. По аллелю — только среди носителей конкретного аллеля (2 копии vs 1 копия)."
    )

    if zygo_mode.startswith("По гену"):
        # --- По гену ---
        if gene_zygo.empty:
            st.info("Нет данных о зиготности по генам.")
        else:
            # доступные гены (пересечение с выбранными)
            genes_for_zygo = sorted(set(gene_zygo["Gene"]).intersection(set(genes_pick or [])))
            gene_pick = st.multiselect("Гены для отображения (зиготность)", genes_for_zygo, default=genes_for_zygo[:3])
            if not gene_pick:
                st.warning("Выберите хотя бы один ген.")
            else:
                # подготовим Antibody по ID
                ant_by_id = groups_q.set_index("ID")["Antibody"]
                for g in gene_pick:
                    sub = gene_zygo[gene_zygo["Gene"] == g].copy()
                    sub = sub.merge(ant_by_id, left_on="ID", right_index=True, how="inner")
                    if sub.empty or sub["Zygosity"].nunique() < 2:
                        st.info(f"{g}: недостаточно данных для сравнения.")
                        continue

                    # U-тест homo vs hetero
                    vals_h = sub.loc[sub["Zygosity"] == "homozygote", "Antibody"].astype(float)
                    vals_t = sub.loc[sub["Zygosity"] == "heterozygote", "Antibody"].astype(float)
                    if len(vals_h) >= 3 and len(vals_t) >= 3:
                        try:
                            _, p_u = mannwhitneyu(vals_h, vals_t, alternative="two-sided")
                        except Exception:
                            p_u = np.nan
                    else:
                        p_u = np.nan

                    fig = px.box(
                        sub.rename(columns={"Zygosity": "Group"}),
                        x="Group", y="Antibody", color="Group", points="all",
                        labels={"Antibody": f"Титр (норм.: {norm_label})", "Group": "Зиготность"},
                        title=f"{g}: homozygote vs heterozygote | U-test p={p_u:.3g}" if pd.notna(p_u) else f"{g}: homozygote vs heterozygote"
                    )
                    fig.update_traces(boxmean=True)
                    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

    elif zygo_mode.startswith("По аллелю"):
        # --- По аллелю (доза 2 vs 1 среди носителей) ---
        if hla_dose_q.empty:
            st.info("Нет матрицы дозировок аллелей.")
        else:
            # Список аллелей ограничим часто встречающимися в регионе
            Nq = len(hla_dose_q)
            allele_freq = hla_dose_q.sum() / (2.0 * Nq)  # частота по дозе
            # оставим аллели с freq >= 1% и присутствующие в merged_q (чтобы был Antibody)
            keep = [a for a in allele_freq.index if allele_freq[a] >= 0.01 and (merged_q["Allele"].eq(a)).any()]
            keep = sorted(keep)[:50]  # не раздувать UI
            alleles_pick = st.multiselect("Аллели для сравнения (доза=2 vs доза=1)", keep, default=keep[:5])
            if not alleles_pick:
                st.warning("Выберите хотя бы один аллель.")
            else:
                # Антитела по ID
                ant_by_id = groups_q.set_index("ID")["Antibody"]
                for a in alleles_pick:
                    if a not in hla_dose_q.columns:
                        continue
                    # берём только носителей (доза 1 или 2)
                    sub = hla_dose_q[[a]].copy()
                    sub = sub.rename(columns={a: "dose"}).query("dose >= 1")
                    if sub.empty:
                        continue
                    sub = sub.join(ant_by_id, how="inner")
                    sub["Group"] = sub["dose"].map({1: "het (dose=1)", 2: "hom (dose=2)"}).astype("category")
                    if sub["Group"].nunique() < 2:
                        st.info(f"{a}: нет обеих групп (1 и 2 копии).")
                        continue

                    vals_hom = sub.loc[sub["Group"] == "hom (dose=2)", "Antibody"].astype(float)
                    vals_het = sub.loc[sub["Group"] == "het (dose=1)", "Antibody"].astype(float)
                    if len(vals_hom) >= 3 and len(vals_het) >= 3:
                        try:
                            _, p_u = mannwhitneyu(vals_hom, vals_het, alternative="two-sided")
                        except Exception:
                            p_u = np.nan
                    else:
                        p_u = np.nan

                    title = f"{a}: hom (n={len(vals_hom)}) vs het (n={len(vals_het)})"
                    if pd.notna(p_u):
                        title += f" | U-test p={p_u:.3g}"

                    fig = px.box(
                        sub, x="Group", y="Antibody", color="Group", points="all",
                        labels={"Antibody": f"Титр (норм.: {norm_label})", "Group": "Группа"},
                        title=title
                    )
                    fig.update_traces(boxmean=True)
                    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

    # ================== Мета-анализ по всем регионам ==================
    st.markdown("## Мета-анализ по всем регионам (фикс-эффект, Hedges g)")

    # Готовим все регионы сразу
    if REGION_COL not in vacc_df.columns:
        st.info("В данных нет колонки региона — мета-анализ по регионам недоступен.")
    else:
        vacc_df[vacc_id_col] = vacc_df[vacc_id_col].astype(str)
        vacc_all = vacc_df[str_bool(vacc_df[meas_vacc_col])].copy()
        vacc_all[meas_q_col] = pd.to_numeric(vacc_all[meas_q_col], errors="coerce")
        vacc_all = vacc_all.dropna(subset=[meas_q_col, REGION_COL])

        # NEW: нормализация для мета-анализа
        if norm_method_key == "log1p":
            vacc_all["Antibody"] = normalize_antibody(vacc_all[meas_q_col], "log1p")
        else:  # z-score — отдельно в каждом регионе
            vacc_all["Antibody"] = (
                vacc_all
                .groupby(REGION_COL, group_keys=False)[meas_q_col]
                .apply(lambda s: normalize_antibody(s, "zscore"))
            )

        groups_all = vacc_all[[vacc_id_col, "Antibody", REGION_COL]].rename(
            columns={vacc_id_col: "ID", REGION_COL: "Region"}
        )
        groups_all["ID"] = groups_all["ID"].astype(str)
        groups_all["Region"] = groups_all["Region"].astype(str)

        # Соединяем с HLA long (те же hla_long_q уже посчитаны выше)
        merged_all = hla_long_q.merge(groups_all, on="ID", how="inner")
        merged_all = merged_all[~merged_all["Allele"].astype(str).isin([".", "-"])]

        meta_rows = []   # по (Gene, Allele): общий эффект и p
        per_region_rows = []  # отдельные строки для лес-плота

        # Возьмем те же аллели, что и в res_q (они уже удовлетворяют базовым фильтрам по численности)
        alleles_set = set(zip(res_q["Gene"], res_q["Allele"]))

        for (g, a) in sorted(alleles_set):
            sub = merged_all[merged_all["Gene"].eq(g)].copy()
            if sub.empty:
                continue
            # для каждого региона считаем Hedges g (Carrier vs Other)
            region_stats = []
            for reg, sub_r in sub.groupby("Region"):
                # признак наличия аллеля по ID
                pres = (sub_r.assign(is_a=sub_r["Allele"].eq(a).astype(int))
                              .groupby("ID")["is_a"].max())
                ant = sub_r.groupby("ID")["Antibody"].first()
                df_r = pd.concat([pres, ant], axis=1).dropna()
                vals1 = df_r.loc[df_r["is_a"] == 1, "Antibody"].astype(float)
                vals0 = df_r.loc[df_r["is_a"] == 0, "Antibody"].astype(float)
                if len(vals1) < 3 or len(vals0) < 3:
                    continue
                m1, s1, n1 = float(vals1.mean()), float(vals1.std(ddof=1)), int(len(vals1))
                m0, s0, n0 = float(vals0.mean()), float(vals0.std(ddof=1)), int(len(vals0))
                g_r, var_r = _hedges_g_and_var(m1, s1, n1, m0, s0, n0)
                if g_r is None or var_r is None or var_r <= 0:
                    continue
                se_r = math.sqrt(var_r)
                region_stats.append((reg, g_r, se_r, n1, n0))

            if len(region_stats) < 1:
                continue

            # фикс-эффект мета-анализ
            weights = [1.0/(se**2) for (_, _, se, _, _) in region_stats]
            ests    = [g_r for (_, g_r, _, _, _) in region_stats]
            sum_w   = sum(weights)
            if sum_w <= 0:
                continue
            g_fixed = sum(w*e for w, e in zip(weights, ests)) / sum_w
            se_fixed = math.sqrt(1.0 / sum_w)
            z = g_fixed / se_fixed if se_fixed > 0 else 0.0
            p_meta = _p_from_z(z)
            ci_lo = g_fixed - 1.96 * se_fixed
            ci_hi = g_fixed + 1.96 * se_fixed

            meta_rows.append({
                "Gene": g,
                "Allele": a,
                "g_fixed": g_fixed,
                "g_se": se_fixed,
                "g_ci_low": ci_lo,
                "g_ci_high": ci_hi,
                "p_meta": p_meta,
                "k_regions": len(region_stats),
            })

            # запомним покомпонентно для лес-плота
            for (reg, g_r, se_r, n1, n0) in region_stats:
                per_region_rows.append({
                    "Gene": g, "Allele": a, "Region": reg,
                    "g": g_r, "se": se_r,
                    "ci_low": g_r - 1.96*se_r,
                    "ci_high": g_r + 1.96*se_r,
                    "n_carriers": n1, "n_noncarriers": n0
                })

        meta_df = pd.DataFrame(meta_rows)
        if meta_df.empty:
            st.info("Недостаточно данных для мета-анализа по регионам.")
        else:
            # отсортируем по значимости и оставим топ-10
            meta_top = meta_df.sort_values(["p_meta", "g_fixed"], ascending=[True, False]).head(10)
            st.caption("Показаны 10 аллелей с наименьшим p по фикс-эффекту (Hedges g).")
            st.dataframe(
                meta_top[["Gene", "Allele", "k_regions", "g_fixed", "g_ci_low", "g_ci_high", "p_meta"]]
                    .assign(g_fixed=lambda d: d["g_fixed"].round(3),
                            g_ci_low=lambda d: d["g_ci_low"].round(3),
                            g_ci_high=lambda d: d["g_ci_high"].round(3),
                            p_meta=lambda d: d["p_meta"].map(lambda x: f"{x:.3g}")),
                hide_index=True, use_container_width=True
            )

            per_reg_df = pd.DataFrame(per_region_rows)

            # Лес-плот для каждого из топ-10
            for _, row in meta_top.iterrows():
                gname, aname = row["Gene"], row["Allele"]
                sub_r = per_reg_df[(per_reg_df["Gene"] == gname) & (per_reg_df["Allele"] == aname)].copy()
                if sub_r.empty:
                    continue

                # добавим "Overall" строку
                overall = pd.DataFrame([{
                    "Region": "Overall",
                    "g": row["g_fixed"],
                    "se": row["g_se"],
                    "ci_low": row["g_ci_low"],
                    "ci_high": row["g_ci_high"],
                    "n_carriers": None, "n_noncarriers": None,
                    "Gene": gname, "Allele": aname
                }])
                sub_r_plot = pd.concat([sub_r, overall], ignore_index=True)
                sub_r_plot["Region"] = pd.Categorical(sub_r_plot["Region"],
                                                      categories=list(sub_r["Region"].sort_values().unique()) + ["Overall"],
                                                      ordered=True)

                # Forest-plot через Plotly
                fig = go.Figure()
                fig.add_vline(x=0, line_dash="dot")  # линия нулевого эффекта

                # отрезки доверительных интервалов
                for i, r in sub_r_plot.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[r["ci_low"], r["ci_high"]],
                        y=[r["Region"], r["Region"]],
                        mode="lines",
                        showlegend=False
                    ))
                # точки-оценки
                fig.add_trace(go.Scatter(
                    x=sub_r_plot["g"],
                    y=sub_r_plot["Region"],
                    mode="markers",
                    marker=dict(size=10),
                    showlegend=False,
                    text=[None if pd.isna(nc) else f"n1={n1}, n0={n0}"
                          for n1, n0, nc in zip(sub_r_plot["n_carriers"],
                                                sub_r_plot["n_noncarriers"],
                                                sub_r_plot["n_carriers"])],
                    hovertemplate="Region=%{y}<br>g=%{x:.3f}<extra></extra>"
                ))

                fig.update_layout(
                    title=f"{gname}-{aname} | g={row['g_fixed']:.3f} "
                          f"[{row['g_ci_low']:.3f}; {row['g_ci_high']:.3f}], p={row['p_meta']:.3g}",
                    xaxis_title="Hedges g (Carrier − Other)",
                    yaxis_title="Регион",
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=350 + 18 * (sub_r_plot.shape[0])
                )
                st.plotly_chart(fig, use_container_width=True)

        # === NEW (regions meta): сводная таблица по всем регионам ===
        st.markdown("## Мета-информация по регионам")
        # Базовая сводка: численность и описательные по титру в каждом регионе (среди вакцинированных с валидным титром)
        base_reg = (
            groups_all
            .groupby("Region")
            .agg(N=("ID", "nunique"),
                 Antibody_mean=("Antibody", "mean"),
                 Antibody_median=("Antibody", "median"),
                 Antibody_sd=("Antibody", "std"))
            .reset_index()
        )
        st.dataframe(
            base_reg.assign(
                Antibody_mean=lambda d: d["Antibody_mean"].round(3),
                Antibody_median=lambda d: d["Antibody_median"].round(3),
                Antibody_sd=lambda d: d["Antibody_sd"].round(3),
            ),
            hide_index=True, use_container_width=True
        )

        # === NEW (regions meta): топ-аллели по эффекту и их частоты (гомо/гетеро) по регионам ===
        st.markdown("## Топ-аллели по эффекту: частоты гомо/гетеро по регионам")
        col_et1, col_et2 = st.columns(2)
        with col_et1:
            alpha_meta = float(st.number_input("Порог значимости p_meta", min_value=1e-6, max_value=0.2, value=0.05, step=0.01, format="%.6f"))
        with col_et2:
            top_k_effect = int(st.number_input("Сколько аллелей показать (по |эффекту|)", min_value=1, value=5, step=1))

        if not meta_df.empty:
            meta_sig = meta_df[meta_df["p_meta"] <= alpha_meta].copy()
            if meta_sig.empty:
                st.info("Нет значимых по p_meta аллелей при текущем пороге.")
            else:
                meta_sig["abs_g"] = meta_sig["g_fixed"].abs()
                meta_eff_top = (meta_sig.sort_values(["abs_g", "p_meta"], ascending=[False, True])
                                       .head(top_k_effect))
                # Матрица доз по всем регионам (будет использована в сводках)
                # (hla_long_q и groups_all уже определены выше)
                for _, r in meta_eff_top.iterrows():
                    gname, aname = r["Gene"], r["Allele"]
                    st.markdown(f"### {gname}-{aname}  |  g={r['g_fixed']:.3f}  p={r['p_meta']:.3g}")
                    # Пер-регион сводка: N, counts/freq для hom/het/car
                    freq_df = _region_summary_for_allele(groups_all, hla_long_q[hla_long_q["Gene"] == gname], aname)
                    if freq_df.empty:
                        st.info("Недостаточно данных для расчёта частот.")
                        continue
                    # Таблица
                    st.dataframe(
                        freq_df.assign(
                            hom_freq=lambda d: d["hom_freq"].round(4),
                            het_freq=lambda d: d["het_freq"].round(4),
                            car_freq=lambda d: d["car_freq"].round(4),
                        ),
                        hide_index=True, use_container_width=True
                    )
                    # Гистограммы частот по регионам: гомозиготы vs гетерозиготы
                    _plot_region_freq_bars(freq_df, allele_label=f"{gname}-{aname}")


with tab_reg:
    st.subheader("Регрессия (HLA+пол → лог-титр) — OLS без формул, Top-N по raw p")

    # Базовые проверки
    if vacc_df.empty or hla_df.empty:
        st.warning("Нужны файлы вакцинаций и HLA.")
        st.stop()
    if selected_region is None:
        st.warning("Выберите регион в сайдбаре.")
        st.stop()
    if meas_q_col is None or meas_q_col not in vacc_df.columns:
        st.error("Нет количественного титра кори (например, 'measles_ME_ml').")
        st.stop()
    if not vacc_id_col:
        st.error("Не найдена ID-колонка в данных вакцинаций.")
        st.stop()

    # Выборка по региону и вакцинации, лог-нормализация титра (ВАЖНО: используем вашу функцию)
    vacc_loc = vacc_df.copy()
    vacc_loc[vacc_id_col] = vacc_loc[vacc_id_col].astype(str)
    vacc_loc = vacc_loc[
        (vacc_loc[REGION_COL].astype(str) == str(selected_region)) &
        str_bool(vacc_loc[meas_vacc_col])
    ].copy()
    vacc_loc["AntibodyRaw"] = pd.to_numeric(vacc_loc[meas_q_col], errors="coerce")
    vacc_loc["AntibodyLog"] = normalize_antibody(vacc_loc["AntibodyRaw"], "log1p")
    vacc_loc = vacc_loc.dropna(subset=["AntibodyLog"])

    covariates = []
    if "age" in vacc_loc.columns:
        vacc_loc["age"] = pd.to_numeric(vacc_loc["age"], errors="coerce")
        covariates.append("age")
    if "sex" in vacc_loc.columns:
        vacc_loc["sex"] = (
            vacc_loc["sex"].astype(str).str.lower()
            .map({"female": 1, "жен": 1, "f": 1, "male": 0, "муж": 0, "m": 0})
            .fillna(0)
        )
        covariates.append("sex")

    # HLA long под выбранные гены и уровень агрегации
    if genes_pick:
        keep_cols = [c for c in HLA_ID_COLS if c in hla_df.columns]
        gene_cols = [c for c in hla_df.columns if "_" in str(c) and c.split("_")[0] in genes_pick]
        hla_sub_reg = hla_df[keep_cols + gene_cols].copy()
    else:
        hla_sub_reg = hla_df.copy()

    if hla_id_series is None or hla_id_series.fillna("").eq("").all():
        st.error("Не удалось определить ID в HLA-таблице.")
        st.stop()
    hla_sub_reg["ID"] = hla_id_series.astype("string").str.strip()
    hla_sub_reg = hla_sub_reg[hla_sub_reg["ID"].notna() & (hla_sub_reg["ID"] != "")]

    try:
        hla_long_reg = process_hla_long(hla_sub_reg, allele_level)
        hla_long_reg["ID"] = hla_long_reg["ID"].astype(str)
    except Exception as e:
        st.error(f"HLA нормализация не удалась: {e}")
        st.stop()

    # Матрица дозировок (0/1/2)
    hla_dose_reg = build_allele_dosage_matrix(hla_long_reg)
    if hla_dose_reg.empty:
        st.info("Матрица дозировок HLA пуста после фильтров.")
        st.stop()
    hla_dose_reg.index = hla_dose_reg.index.astype(str)

    # Датафрейм для регрессии
    meta_reg = (
        vacc_loc[[vacc_id_col, "AntibodyLog"] + covariates]
        .rename(columns={vacc_id_col: "ID"})
        .dropna(subset=["AntibodyLog"])
        .copy()
    )
    meta_reg["ID"] = meta_reg["ID"].astype(str)

    data = meta_reg.merge(hla_dose_reg, left_on="ID", right_index=True, how="inner")
    if data.empty:
        st.info("Нет пересечения ID между вакцинациями и HLA для выбранного региона.")
        st.stop()

    allele_cols = [c for c in data.columns if c not in (["ID", "AntibodyLog"] + covariates)]

    # Оставляем аллели с вариацией дозы: >=1 носитель и >=1 неноситель
    allele_variation = {a: ((data[a] > 0).sum(), (data[a] == 0).sum()) for a in allele_cols}
    keep_alleles = [a for a, (n_car, n_non) in allele_variation.items() if n_car > 0 and n_non > 0]

    if not keep_alleles:
        st.warning("Нет аллелей с вариацией дозы (все 0 или все >0).")
        st.write("Число носителей по аллелям (carrier, non-carrier):", allele_variation)
        st.stop()

    # ==== МНОГОФАКТОРНАЯ OLS ПО ГЕНУ С РЕФЕРЕНС-АЛЛЕЛЕМ ====
    import statsmodels.api as sm

    N = len(data)
    allele_cols = [c for c in data.columns if c not in (["ID", "AntibodyLog"] + covariates)]

    # Частота по дозе (0/1/2) — доля копий аллеля: sum(dose)/(2N)
    freq = data[allele_cols].sum() / (2.0 * N)
    freq_thr_reg_pct = st.sidebar.slider(
        "Минимальная частота аллеля для включения в модель (%)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="freq_thr_reg_pct"
    )
    freq_thr = freq_thr_reg_pct / 100.0

    # Разбивка аллелей по генам
    genes_in_data = sorted({a.split("*")[0] for a in allele_cols if "*" in a})

    rows = []
    for g in genes_in_data:
        g_alleles = [a for a in allele_cols if a.startswith(g + "*")]

        # оставляем только «частые» аллели
        g_keep = [a for a in g_alleles if float(freq.get(a, 0.0)) >= freq_thr]
        if len(g_keep) < 1:
            continue

        # референс = самый частый среди оставшихся
        ref = pd.Series({a: float(freq.get(a, 0.0)) for a in g_keep}).sort_values(ascending=False).index[0]
        g_terms = [a for a in g_keep if a != ref]

        # конструируем дизайн: AntibodyLog ~ (все g_terms) + ковариаты
        cols = ["AntibodyLog"] + g_terms + covariates
        df_g = data[cols].dropna()

        # выкидываем константные предикторы, если вдруг нет вариации
        g_terms_var = [a for a in g_terms if df_g[a].nunique() > 1]
        if len(g_terms_var) == 0 or df_g["AntibodyLog"].nunique() < 2:
            continue

        y = df_g["AntibodyLog"]
        X = df_g[g_terms_var + covariates].copy()
        X = sm.add_constant(X, has_constant="add")

        try:
            model = sm.OLS(y, X).fit(cov_type="HC3")  # робастные SE
        except Exception:
            continue

        # Берём коэффициенты и p-value напрямую из модели
        terms = [t for t in g_terms_var if t in model.params.index]
        if not terms:
            continue

        for t in terms:
            coef_t = model.params.get(t, np.nan)
            pval_t = model.pvalues.get(t, np.nan)
            if pd.isna(coef_t) or pd.isna(pval_t):
                continue
            rows.append({
                "Gene": g,
                "Allele": t,                      # эффект vs референс-аллель
                "coef": float(coef_t),
                "p_value": float(pval_t),
                "n": int(df_g.shape[0]),
                "ref_allele_dropped": ref,
                "covs": "+".join(covariates) if covariates else "(none)"
            })

    res_lin = pd.DataFrame(rows)

    if res_lin.empty:
        st.warning("Не удалось подогнать модели по генам при текущем пороге частоты/фильтрах.")
        st.stop()

    # FDR справочно
    rej, pcor, _, _ = multipletests(res_lin["p_value"], method="fdr_bh")
    res_lin["p_fdr_bh"] = pcor

    # Показ Top-N по raw p (как и раньше)
    top_n_lin = int(st.number_input("Top-N по raw p", min_value=5, value=10, step=1, key="top_n_lin_ols_raw"))
    to_show = res_lin.sort_values(["p_value", "Gene", "Allele"]).head(top_n_lin)

    st.caption(
        f"Показаны Top-{top_n_lin} аллелей по raw p-value. "
        f"В каждой OLS-модели по гену референсом взят самый частый аллель (freq ≥ {freq_thr_reg_pct:.1f}%)."
    )
    st.dataframe(
        to_show[["Gene", "Allele", "coef", "p_value", "p_fdr_bh", "n", "ref_allele_dropped", "covs"]],
        hide_index=True, use_container_width=True
    )

    # Небольшой график
    try:
        fig = px.bar(
            to_show.assign(coef_round=lambda d: d["coef"].astype(float)),
            x="Allele", y="coef_round", color="Gene",
            labels={"coef_round": "Коэффициент (β)", "Allele": "Аллель"},
            title="Коэффициенты (vs референс-аллель в своём гене)"
        )
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    st.download_button(
        "⬇️ Результаты OLS (геновые модели с референсом) — CSV",
        data=to_csv_bytes(res_lin),
        file_name="hla_measles_log_titer_ols_gene_models.csv",
        mime="text/csv",
    )

with tab_freq:
    st.subheader("Частоты носителей аллелей по регионам")

    if hla_long_q.empty and not hla_df.empty and hla_id_series is not None:
        try:
            keep_cols = [c for c in HLA_ID_COLS if c in hla_df.columns]
            hla_cols  = pick_hla_cols_by_values(hla_df)
            hla_sub_f = hla_df[keep_cols + hla_cols].copy()
            hla_sub_f["ID"] = hla_id_series.astype("string").str.strip()
            hla_sub_f = hla_sub_f[hla_sub_f["ID"].notna() & (hla_sub_f["ID"] != "")]
            _tmp = process_hla_long(hla_sub_f, allele_level)
            hla_long_q = sanitize_hla_long(_tmp)
            hla_long_q["ID"] = hla_long_q["ID"].astype(str)
        except Exception:
            pass

    if groups_all.empty and (REGION_COL in vacc_df.columns) and vacc_id_col:
        try:
            _vacc_ok = vacc_df.copy()
            _vacc_ok[vacc_id_col] = _vacc_ok[vacc_id_col].astype(str)
            groups_all = _vacc_ok[[vacc_id_col, REGION_COL]].rename(
                columns={vacc_id_col: "ID", REGION_COL: "Region"}
            )
            groups_all["ID"] = groups_all["ID"].astype(str)
            groups_all["Region"] = groups_all["Region"].astype(str)
        except Exception:
            pass

    if hla_long_q.empty or groups_all.empty:
        st.warning("Нет данных HLA или регионов — сравнение невозможно.")
    else:
        # Соберём DataFrame: ID, Gene, Allele, Region
        hla_with_reg = hla_long_q.merge(groups_all[["ID", "Region"]], on="ID", how="inner")
        if hla_with_reg.empty:
            st.info("Нет пересечения ID между HLA и метаданными по регионам.")
        else:
            genes_avail = sorted(hla_with_reg["Gene"].dropna().unique())
            gene_pick_freq = st.multiselect("Гены для отображения", genes_avail, default=genes_avail[:3])
            min_freq = st.slider("Мин. частота для показа", 0.0, 0.5, 0.01, 0.01)

            for g in gene_pick_freq:
                sub = hla_with_reg[hla_with_reg["Gene"] == g].copy()
                # Частоты: хотя бы одна копия аллеля
                freq_df = (
                    sub.groupby(["Region", "Allele"])["ID"]
                       .nunique()
                       .reset_index(name="n_carriers")
                )
                # общее число пациентов по регионам
                n_per_reg = groups_all.groupby("Region")["ID"].nunique().rename("N")
                freq_df = freq_df.merge(n_per_reg, on="Region", how="left")
                freq_df["freq"] = freq_df["n_carriers"] / freq_df["N"]
                freq_df = freq_df[freq_df["freq"] >= min_freq]

                if freq_df.empty:
                    st.info(f"Ген {g}: нет аллелей с freq ≥ {min_freq:.2f}")
                    continue

                st.markdown(f"### Ген {g}")
                fig = px.bar(
                    freq_df, x="Allele", y="freq", color="Region",
                    barmode="group",
                    title=f"Частоты носителей аллелей гена {g} по регионам",
                    labels={"freq": "Доля носителей", "Allele": "Аллель"}
                )
                fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)