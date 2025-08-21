import re
from typing import List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import io
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------
# Config & helpers
# ----------------------

st.set_page_config(page_title="PID/AID Dashboard", layout="wide")

TEXT_COLS_CANDIDATES = [
    "clinical_discuss", "clinical_comments", "comments_interesting_vars",
    "INTERPRET_VAR_table_user", "INTERPRET_VAR_table",
    "IUIS_classification", "combi_diganosis_mcch52", "clin_manifestations",
    "autoiimunity", "markers"
]

ID_COLS = {
    "uin2_fulltxt": ["uin2_fulltxt", "uin2"],
    "uin2_number": ["uin2_number"],
    "zlims_id": ["zlims_id", "zlims", "ZLIMS ID"],
}

HLA_ID_COLS = ["ID", "Sample_ID", "sample_id", "ZLIMS ID", "ID_x"]

SEQ_COLS_CANDIDATES = [
    "status_coverage", "VAR_IEI_panel_table_status", "SANGER"
]

COHORT_COL = "combi_type_mcch52"

# -------- Вакцины: конфиг авто-обнаружения колонок --------
# Все списки — «кандидаты», берём те, которые реально есть в данных
VACC_DEFAULT_PATH = "RPB_metavac_wizard_ZLIMS_GOOD_04.08.25.xlsx"
HLA_DEFAULT_PATH = "combined_hla_table.xlsx"

# Единая схема по нозологиям
DISEASES = {
    "measles": {
        "title": "Корь",
        "me_ml": "measles_ME_ml",
        "result": "measles_result",
        "sick": "measles_sick",
        "vacc_info": "measles_vaccine_info",
        "vacc_total": "measles_vaccine_totalnum",
        "noanswer": "measles_NoAnswer_coef",
    },
    "rubella": {
        "title": "Краснуха",
        "me_ml": "rubella_ME_ml",
        "result": "rubella_result",
        "sick": "rubella_sick",
        "vacc_info": "rubella_vaccine_info",
        "vacc_total": "rubella_vaccine_totalnum",
        "noanswer": "rubella_NoAnswer_coef",
    },
    "diphtheria": {
        "title": "Дифтерия",
        "me_ml": "diphtheria_ME_ml",
        "result": "diphtheria_result",
        "sick": "diphtheria_sick",
        "vacc_info": "diphtheria_vaccine_info",
        "vacc_total": "diphtheria_vaccine_totalnum",
        "noanswer": "diphtheria_NoAnswer_coef",
    },
    "mumps": {
        "title": "Паротит",
        "me_ml": None,
        "result": "mumps_result",
        "sick": "mumps_sick",
        "vacc_info": "mumps_vaccine_info",
        "vacc_total": "mumps_vaccine_totalnum",
        "noanswer": "mumps_NoAnswer_coef",
    },
    "HAV": {
        "title": "Гепатит A (HAV)",
        "me_ml": None,
        "result": "HAV_result",
        "sick": "HAV_sick",
        "vacc_info": "HAV_vaccine_info",
        "vacc_total": "HAV_vaccine_totalnum",
        "noanswer": "HAV_NoAnswer_coef",
    },
    "HBV": {
        "title": "Гепатит B (HBV)",
        "me_ml": "HBV_antiHBsAg_ME_ml",
        "result": "HBV_antiHBsAg_result",
        "result_extra": ["HBV_HBsAg_result", "HBV_antiHBcAg_result"],  # доп. маркеры
        "sick": "HBV_sick",
        "vacc_info": "HBV_vaccine_info",
        "vacc_total": "HBV_vaccine_totalnum",
        "noanswer": "HBV_NoAnswer_coef",
    },
}


COMBI_COLS = [
    "Combi_NoAnswer_coef", "Combi_NoAnswer_TYPE", "fail_merge_tubes"
]

VACC_ID_COLS = [
    "ZLIMS ID", "ID_x", "zlims_id", "zlims", "patient_id", "uin2", "uin2_number", "uin2_fulltxt", "ID"
]

VACC_DATE_COLS = [
    "date", "vaccination_date", "vacc_date", "dt", "Дата вакцинации"
]
VACC_NAME_COLS = [
    "vaccine", "vaccine_name", "vacc_name", "vaccine_product", "vaccine_brand", "Вакцина"
]
VACC_MANUF_COLS = [
    "manufacturer", "producer", "brand", "Производитель"
]
VACC_DOSE_NUM_COLS = [
    "dose", "dose_number", "dose_no", "Номер дозы"
]
VACC_SERIES_COLS = [
    "batch", "series", "lot", "Серия"
]
VACC_AE_FLAG_COLS = [
    "ae", "adverse_event", "aefi", "НППИ", "Нежелательное_явление", "adverse"
]
VACC_AE_TYPE_COLS = [
    "ae_type", "aefi_type", "adverse_type", "Тип_НППИ"
]
VACC_AE_SEV_COLS = [
    "ae_severity", "severity", "Степень_тяжести"
]


# patterns to search (case-insensitive)
PATTERNS: Dict[str, str] = {
    "disregulation": r"дисрегул",  # дисрегуляци
    "cvid_ovin": r"(овин|общ(ая|ий)\s+вариабел\w*\s+иммун\w*\s+недостат\w*|cvid)",
    "avz_autoinfl": r"(авз|аутовоспалени\w*|аутовоспалит\w*)",
    "neutropenia": r"нейтропен\w*",
    "urticaria_skin": r"(крапивниц\w*|кожн\w*\s*синдром\w*)",
    "atopic_derm": r"(атопическ\w*.*дерматит|атопическ\w+|дерматит)",
    "ige_mentioned": r"\bige\b",
    "ige_elevated": r"\bige\b.*(повыш|elevat)",
    "asthma": r"астм\w*",
    "edema_angio": r"(отек\w*|ангиоотек\w*|квинке|нао|наследственн\w*\s*ангио)",
    "rash": r"высыпа\w*",
    "arthritis_joint": r"(артрит\w*|сустав\w*|артроз\w*)",
    "anemia_thrombocytopenia": r"(анеми\w*|тромбоцитопен\w*)",
    "allergy_block": r"(аллерг\w*|сенсибилиз\w*|пыльц\w*)",
    "sle": r"(волчан\w*|\bскв\b)",
    "behcet": r"бехчет\w*",
    "autoimmune": r"аутоиммун\w*",
}


def pick_existing(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    low = {c.lower(): c for c in df.columns}
    res = []
    for cand in candidates:
        if cand.lower() in low:
            res.append(low[cand.lower()])
    return res


def first_existing(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = pick_existing(df, candidates)
    return cols[0] if cols else None


def pick_first_or_none(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Синоним для читаемости кода по вакцинам."""
    return first_existing(df, candidates)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ======== HLA helpers ========

def normalize_allele(allele: str, level: int = 2) -> str:
    """
    'HLA-A*01:01:01' -> 'A*01' (level=1) или 'A*01:01' (level=2).
    '-' оставляем как есть.
    """
    if not isinstance(allele, str) or allele == "-":
        return allele
    m = _re.match(r"HLA-([A-Z0-9]+)\*(\d{2})(?::(\d{2}))?", allele)
    if m:
        gene, group, protein = m.groups()
        if level == 1:
            return f"{gene}*{group}"
        elif level == 2 and protein:
            return f"{gene}*{group}:{protein}"
    return allele

def normalize_and_fill(hla_df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Нормализация всех HLA-аллелей + замена '-' на соседний аллель (если есть)."""
    df = hla_df.copy()
    gene_cols = [c for c in df.columns if "_" in str(c)]
    genes = sorted(set(c.split("_")[0] for c in gene_cols))
    for g in genes:
        c1, c2 = f"{g}_1", f"{g}_2"
        if c1 in df.columns and c2 in df.columns:
            df[c1] = df[c1].apply(lambda x: normalize_allele(x, level))
            df[c2] = df[c2].apply(lambda x: normalize_allele(x, level))
            df[c1] = df.apply(lambda r: r[c2] if r[c1] == "-" else r[c1], axis=1)
            df[c2] = df.apply(lambda r: r[c1] if r[c2] == "-" else r[c2], axis=1)
    return df

def process_hla_long(hla_df: pd.DataFrame, allele_level: int) -> pd.DataFrame:
    """
    Wide -> long: колонки ID, Gene, Allele.
    По человеку создаётся по 2 записи на ген (учитываем гомозиготу).
    """
    df = normalize_and_fill(hla_df, allele_level)
    # гарантируем наличие ID
    if "ID" not in df.columns:
        raise ValueError("В HLA-таблице отсутствует колонка ID.")
    rec = []
    gene_cols = [c for c in df.columns if "_" in str(c)]
    genes = sorted(set(c.split("_")[0] for c in gene_cols))
    for g in genes:
        c1, c2 = f"{g}_1", f"{g}_2"
        if c1 in df.columns and c2 in df.columns:
            for _, r in df[[ "ID", c1, c2 ]].iterrows():
                a1, a2 = r[c1], r[c2]
                if pd.isna(r["ID"]):
                    continue
                if a1 == a2:
                    rec.append((str(r["ID"]), g, a1))
                    rec.append((str(r["ID"]), g, a1))
                else:
                    rec.append((str(r["ID"]), g, a1))
                    rec.append((str(r["ID"]), g, a2))
    return pd.DataFrame(rec, columns=["ID","Gene","Allele"])

def counts_and_tests(hla_long: pd.DataFrame, groups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Бинарный исход (Group 0/1): χ²/Fisher на каждом аллеле внутри каждого гена.
    hla_long: ID, Gene, Allele
    groups_df: ID, Group (0/1)
    """
    df = hla_long.merge(groups_df[["ID","Group"]], on="ID", how="inner")
    out = []
    for g in sorted(df["Gene"].dropna().unique()):
        sub = df[df["Gene"] == g]
        # частоты носительства по ID
        carr = (sub.groupby(["ID","Allele"]).size().reset_index()[["ID","Allele"]])
        carr["present"] = 1
        carr = carr.merge(groups_df[["ID","Group"]], on="ID", how="left").dropna(subset=["Group"])
        # всего по группам
        n_by_group = groups_df.drop_duplicates("ID")["Group"].value_counts().to_dict()
        for a in sorted(carr["Allele"].dropna().unique()):
            tab = (
                carr.assign(is_a=(carr["Allele"] == a).astype(int))
                    .groupby("Group")["is_a"].sum()
                    .reindex([0,1], fill_value=0)
            )
            a0, a1 = int(tab.get(0,0)), int(tab.get(1,0))
            n0, n1 = int(n_by_group.get(0,0)), int(n_by_group.get(1,0))
            if (a0 + a1) == 0 or n0 == 0 or n1 == 0:
                continue
            cont = [[a1, n1 - a1],[a0, n0 - a0]]
            try:
                if min(min(cont)) < 5:
                    _, p = fisher_exact(cont)
                    test = "Fisher"
                else:
                    chi2, p, _, _ = chi2_contingency(cont)
                    test = "Chi2"
            except Exception:
                continue
            freq1 = a1 / n1 if n1 else 0.0
            freq0 = a0 / n0 if n0 else 0.0
            out.append({
                "Gene": g, "Allele": a,
                "count_res1": a1, "count_res0": a0,
                "n_res1": n1, "n_res0": n0,
                "freq_res1": freq1, "freq_res0": freq0,
                "delta_freq": freq1 - freq0,
                "test": test, "p_value": p,
            })
    res = pd.DataFrame(out)
    if res.empty:
        return res
    reject, pcor, _, _ = multipletests(res["p_value"], method="fdr_bh")
    res["p_fdr_bh"] = pcor
    res["signif_fdr"] = reject
    return res.sort_values(["p_fdr_bh","p_value","Gene","Allele"])


def st_bar_chart_safe(obj):
    """
    Безопасная обёртка над st.bar_chart:
    - конвертирует IntervalIndex (и интервальные категории) в строковые метки,
      чтобы Altair/Streamlit не падали с ошибкой схемы.
    """
    # Приводим к DataFrame
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name or "value")
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = pd.DataFrame(obj)

    try:
        # Случай: индекс — IntervalIndex
        if isinstance(df.index, pd.IntervalIndex):
            df = df.reset_index().rename(columns={"index": "bin"})
            df["bin"] = df["bin"].astype(str)
            df = df.set_index("bin")
        else:
            # Случай: индекс — MultiIndex (кортежи) → склеиваем в строку
            if isinstance(df.index, pd.MultiIndex):
                df.index = df.index.map(lambda x: " | ".join(map(str, x)))
                df.index.name = df.index.name or "bin"

            if len(df.index) and isinstance(df.index[0], pd.Interval):
                df.index = df.index.map(str)
    except Exception:
        # На всякий случай: если что-то пошло не так, просто строкизуем индекс
        try:
            df.index = df.index.map(str)
        except Exception:
            pass

    st.bar_chart(df)


############################################
# HLA нормализация — как в эталонном коде  #
############################################
def normalize_allele(allele: str, level: int = 2) -> str:
    """
    Ровно как в эталоне:
      - строка должна соответствовать шаблону 'HLA-GENE*XX(:YY)?'
      - level=1 -> 'GENE*XX'
      - level=2 -> 'GENE*XX:YY' (только если есть YY)
      - '-' и всё нераспознанное возвращаем как есть
    """
    if not isinstance(allele, str) or allele == "-":
        return allele
    pattern = r"HLA-([A-Z0-9]+)\*(\d{2})(?::(\d{2}))?"
    match = re.match(pattern, allele)
    if match:
        gene, group, protein = match.groups()
        if level == 1:
            return f"{gene}*{group}"
        elif level == 2 and protein:
            return f"{gene}*{group}:{protein}"
    return allele

def normalize_and_fill(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    Ровно как в эталоне:
      - применяем normalize_allele к каждой аллельной колонке *_1 / *_2
      - прочерки '-' заменяем значением из соседней колонки пары
    """
    df_clean = df.copy()
    gene_columns = [col for col in df.columns if "_" in str(col)]
    gene_prefixes = sorted(set(col.split("_")[0] for col in gene_columns))

    for gene in gene_prefixes:
        col1, col2 = f"{gene}_1", f"{gene}_2"
        if col1 in df_clean and col2 in df_clean:
            df_clean[col1] = df_clean[col1].apply(lambda x: normalize_allele(x, level))
            df_clean[col2] = df_clean[col2].apply(lambda x: normalize_allele(x, level))
            # Заполняем прочерки значением из соседней колонки
            df_clean[col1] = df_clean.apply(
                lambda row: row[col2] if row[col1] == "-" else row[col1], axis=1)
            df_clean[col2] = df_clean.apply(
                lambda row: row[col1] if row[col2] == "-" else row[col2], axis=1)
    return df_clean

def process_hla_long(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """
    long-таблица для дашборда: (ID, Gene, Allele)
    Поведение по аллелям совпадает с эталоном:
      - normalize_and_fill
      - дублируем аллель при гомозиготности
      - затем выбрасываем '-' из результата
    Ожидается, что в df уже есть колонка 'ID' (строка).
    """
    if "ID" not in df.columns:
        raise ValueError("Ожидается колонка 'ID' в HLA-таблице.")

    # нормализуем пары *_1/*_2
    df_norm = normalize_and_fill(df, level)

    records = []
    gene_columns = [col for col in df_norm.columns if "_" in str(col)]
    gene_prefixes = sorted(set(col.split("_")[0] for col in gene_columns))

    for _, row in df_norm.iterrows():
        pid = str(row["ID"])
        for gene in gene_prefixes:
            col1, col2 = f"{gene}_1", f"{gene}_2"
            if col1 not in df_norm.columns or col2 not in df_norm.columns:
                continue
            a1, a2 = row[col1], row[col2]
            if a1 == a2:
                # гомозигота — две копии одного и того же аллеля
                records.append((pid, gene, a1))
                records.append((pid, gene, a1))
            else:
                records.append((pid, gene, a1))
                records.append((pid, gene, a2))

    out = pd.DataFrame(records, columns=["ID", "Gene", "Allele"])
    # Чтобы '-' не попадали в тесты
    out = out[out["Allele"] != "-"]
    return out

def coalesce_first_nonnull(df: pd.DataFrame, cols: list[str]) -> pd.Series | None:
    """
    Возвращает серию — первый непустой ID по списку колонок-кандидатов.
    Полезно, если ID размазан по нескольким полям.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    s = pd.Series(index=df.index, dtype="object")
    for c in present:
        v = df[c].astype("string").str.strip()
        s = s.fillna(v.where(v.notna() & (v != "")))
    # подчистим полностью пустые строки
    s = s.where(s.notna() & (s != ""))
    return s


def concat_text_row(row: pd.Series, text_cols: List[str]) -> str:
    parts = []
    for c in text_cols:
        val = row.get(c, None)
        if pd.notna(val):
            parts.append(str(val))
    return " | ".join(parts).lower()

def has_nonempty(row: pd.Series, cols: List[str]) -> bool:
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return True
    return False

def mark_patterns(df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
    if not text_cols:
        for key in PATTERNS:
            df[key] = False
        return df
    texts = df[text_cols].astype(str).agg(" | ".join, axis=1).str.lower()
    for key, pat in PATTERNS.items():
        df[key] = texts.str.contains(pat, regex=True, na=False, case=False)
    return df

def compute_completeness(df: pd.DataFrame) -> pd.Series:
    # IDs: require (zlims_id) AND (uin2_number OR uin2_fulltxt)
    id_zlims = first_existing(df, ID_COLS["zlims_id"])
    id_uin_num = first_existing(df, ID_COLS["uin2_number"])
    id_uin_txt = first_existing(df, ID_COLS["uin2_fulltxt"])

    has_zlims = df[id_zlims].notna() & (df[id_zlims].astype(str).str.strip() != "") if id_zlims else False
    has_uin_any = False
    if id_uin_num:
        has_uin_any = df[id_uin_num].notna() & (df[id_uin_num].astype(str).str.strip() != "")
    if id_uin_txt:
        u = df[id_uin_txt].notna() & (df[id_uin_txt].astype(str).str.strip() != "")
        has_uin_any = (has_uin_any | u) if isinstance(has_uin_any, pd.Series) else u

    # Diagnosis info present if any of these non-empty
    diag_cols = pick_existing(df, ["IUIS_classification", "combi_diganosis_mcch52", "clinical_discuss"])
    has_diag = df[diag_cols].astype(str).apply(lambda s: s.str.strip() != "", axis=0).any(axis=1) if diag_cols else False

    # Sequencing info present if any of these non-empty
    seq_cols = pick_existing(df, SEQ_COLS_CANDIDATES)
    has_seq = df[seq_cols].astype(str).apply(lambda s: s.str.strip() != "", axis=0).any(axis=1) if seq_cols else False

    complete = has_zlims & has_uin_any & has_diag & has_seq
    return complete.fillna(False) if isinstance(complete, pd.Series) else pd.Series([False] * len(df))

@st.cache_data(show_spinner=False)
def read_xlsx(src):
    return pd.read_excel(src)

# ----------------------
# Load data
# ----------------------

st.sidebar.header("Данные")
uploaded = st.sidebar.file_uploader("Загрузите Excel-файл (.xlsx) c таблицей", type=["xlsx"])
default_path = "PID_AID_phenotypes_13.08.25.xlsx"

# Вакцины: отдельный загрузчик (по желанию), иначе используем путь по умолчанию
vacc_uploaded = st.sidebar.file_uploader("Загрузите файл с вакцинами (.xlsx)", type=["xlsx"], key="vacc_upl")
vacc_default_path = VACC_DEFAULT_PATH


if uploaded is not None:
    df = pd.read_excel(uploaded)
elif default_path:
    try:
        df = pd.read_excel(default_path)
        st.sidebar.info("Используется файл по умолчанию: PID_AID_phenotypes_13.08.25.xlsx")
    except Exception as e:
        st.error(f"Не удалось загрузить файл по умолчанию: {e}")
        st.stop()
else:
    st.stop()

st.title("PID/AID — Метаданные")

# normalize columns (strip)
df = normalize_cols(df)

# cohort selector
if COHORT_COL in df.columns:
    cohorts = ["(все)"] + sorted([str(x) for x in df[COHORT_COL].dropna().unique()])
    picked = st.sidebar.selectbox("Когорта (combi_type_mcch52)", cohorts, index=0)
    if picked != "(все)":
        df = df[df[COHORT_COL].astype(str) == picked]
else:
    st.sidebar.warning("Колонка 'combi_type_mcch52' не найдена — фильтр по когорте отключён.")

# completeness
complete_mask = compute_completeness(df)
mode = st.sidebar.radio("Полнота данных", ["Все", "Только полные (ID + диагноз + секвенирование)", "Только неполные"], index=0)
if mode != "Все":
    if mode.startswith("Только полные"):
        df = df.loc[complete_mask]
    else:
        df = df.loc[~complete_mask]

# build text columns to search
text_cols = pick_existing(df, TEXT_COLS_CANDIDATES)

# mark presence columns
df_marked = df.copy()
df_marked = mark_patterns(df_marked, text_cols)

presence_cols = list(PATTERNS.keys())
binary_table = df_marked[presence_cols].copy().astype(bool)
# attach IDs for reference (best-effort)
id_show_cols = pick_existing(df_marked, ["ID", "uin2_number", "uin2_fulltxt", "zlims_id", "ZLIMS ID", "ID_x"])
binary_table = pd.concat([df_marked[id_show_cols], binary_table], axis=1) if id_show_cols else binary_table

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3 = st.tabs([
    "🧬 Метаданные",
    "💉 Вакцины",
    "🧬⇄💉 HLA × Measles (бинарный)",
])

# отдельная вкладка для количественного анализа антител
tab4 = st.container()

with tab1:
    st.subheader("Сводка по меткам (в текущей выборке)")
    counts = df_marked[presence_cols].sum().sort_values(ascending=False).rename("count").to_frame()
    st.dataframe(counts)

    # === Визуализации метаданных ===
    st.markdown("### Гистограммы по метаданным")
    # 1) Столбчатая диаграмма по суммарным упоминаниям
    try:
        st_bar_chart_safe(counts)
    except Exception:
        st.write("Не удалось построить bar chart для сводки.")

    # 2) Распределение числа меток на пациента (сколько совпадений паттернов у каждой строки)
    try:
        matches_per_row = df_marked[presence_cols].astype(bool).sum(axis=1)
        st.markdown("**Распределение количества упоминаний на пациента**")
        # превратим в частоты
        hist_df = matches_per_row.value_counts().sort_index().rename_axis("num_flags").to_frame("patients")
        st_bar_chart_safe(hist_df)
    except Exception:
        st.write("Не удалось построить гистограмму распределения по пациентам.")

    st.markdown("### Таблица «есть/нет» по пациентам")
    st.dataframe(binary_table)

    # === Отладка: пациенты без упоминаний ===
    st.markdown("### Пациенты без упоминаний (все метки = False)")
    try:
        no_flags_mask = ~df_marked[presence_cols].any(axis=1)
        df_no_flags = df.loc[no_flags_mask]
        if not df_no_flags.empty:
            st.dataframe(df_no_flags)
            st.caption(f"Показано {len(df_no_flags)} строк из исходной таблицы без упоминаний.")
        else:
            st.info("Все пациенты имеют хотя бы одну метку.")
    except Exception as e:
        st.warning(f"Не удалось вычислить пациентов без упоминаний: {e}")

    # === Экспорт таблиц ===
    st.markdown("### Экспорт таблиц")
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        bio.seek(0)
        return bio.read()

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.download_button(
            label="⬇️ Скачать 'есть/нет' (CSV)",
            data=to_csv_bytes(binary_table),
            file_name="binary_table.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            label="⬇️ Скачать 'есть/нет' (XLSX)",
            data=to_xlsx_bytes(binary_table),
            file_name="binary_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col_c:
        st.download_button(
            label="⬇️ Скачать сводку (CSV)",
            data=to_csv_bytes(counts.reset_index()),
            file_name="summary_counts.csv",
            mime="text/csv",
        )
    with col_d:
        st.download_button(
            label="⬇️ Скачать сводку (XLSX)",
            data=to_xlsx_bytes(counts.reset_index()),
            file_name="summary_counts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("### Описание полей")

    # === Круговые диаграммы по диагнозам ===
    st.markdown("### Распределение по диагнозам и классификациям")
    import matplotlib.pyplot as plt

    def plot_pie(series: pd.Series, title: str):
        vc = series.dropna().astype(str).value_counts()
        if vc.empty:
            st.info(f"Нет данных для {title}")
            return
        # ограничим топ-10, остальное суммируем в 'Other'
        top10 = vc.head(10)
        if len(vc) > 10:
            top10.loc["Other"] = vc.iloc[10:].sum()
        fig, ax = plt.subplots()
        ax.pie(top10.values, labels=top10.index, autopct="%1.1f%%", startangle=90, counterclock=False)
        ax.axis("equal")
        ax.set_title(title)
        st.pyplot(fig)

    if "combi_diganosis_mcch52" in df.columns:
        plot_pie(df["combi_diganosis_mcch52"], "combi_diganosis_mcch52")
    else:
        st.warning("Колонка combi_diganosis_mcch52 не найдена")

    if "IUIS_classification" in df.columns:
        plot_pie(df["IUIS_classification"], "IUIS_classification")
    else:
        st.warning("Колонка IUIS_classification не найдена")

    st.markdown(
        """
- **disregulation** — упоминания дисрегуляции иммунитета
- **cvid_ovin** — ОВИН / CVID
- **avz_autoinfl** — АВЗ / аутовоспаление
- **neutropenia** — нейтропения
- **urticaria_skin** — крапивница / кожный синдром
- **atopic_derm** — атопический дерматит / атопические проявления / дерматит
- **ige_mentioned** — упоминание IgE
- **ige_elevated** — явное упоминание повышения IgE
- **asthma** — астма
- **edema_angio** — отеки, ангиоотеки, Квинке, НАО
- **rash** — высыпания
- **arthritis_joint** — артрит/артроз/суставные
- **anemia_thrombocytopenia** — анемии и тромбоцитопении
- **allergy_block** — аллергия, сенсибилизация, ринит, пыльцевая
- **sle** — СКВ / системная красная волчанка
- **behcet** — болезнь Бехчета
- **autoimmune** — упоминания «аутоиммун-»
"""
    )

    st.caption("Подсчёт выполнен по свободному тексту из клинических и диагностических колонок. Регистронезависимый поиск по регулярным выражениям.")



# ----------------------
# Вкладка: Вакцины
# ----------------------
with tab2:
    st.subheader("Метаданные по вакцинациям")

    # === Загрузка данных о вакцинациях ===
    try:
        if vacc_uploaded is not None:
            vacc_df = pd.read_excel(vacc_uploaded)
            st.info("Используется загруженный файл с вакцинами.")
        else:
            vacc_df = pd.read_excel(vacc_default_path)
            st.info(f"Используется файл по умолчанию: {vacc_default_path}")
    except Exception as e:
        st.error(f"Не удалось загрузить файл с вакцинами: {e}")
        st.stop()
    vacc_df = normalize_cols(vacc_df.copy())

    # === Авто-детекция ключевых колонок ===
    col_id   = pick_first_or_none(vacc_df, VACC_ID_COLS)
    col_date = pick_first_or_none(vacc_df, VACC_DATE_COLS)


    # === Приведение даты ===
    if col_date and pd.api.types.is_string_dtype(vacc_df[col_date]):
        with pd.option_context("mode.chained_assignment", None):
            vacc_df[col_date] = pd.to_datetime(vacc_df[col_date], errors="coerce")

    # === Сводные показатели ===
    st.markdown("### Сводка")
    total_rows = len(vacc_df)
    unique_patients = vacc_df[col_id].nunique() if col_id else None


    m = []
    m.append(("Всего записей (доз)", total_rows))
    if unique_patients is not None:
        m.append(("Уникальных пациентов", unique_patients))
    metrics_df = pd.DataFrame(m, columns=["metric", "value"])
    st.dataframe(metrics_df, hide_index=True)

    # === Демография ===
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

    # === Секвенирование / техметаданные ===
    st.markdown("### Секвенирование / техметаданные")
    for col in ["Проект","Платформы","Чтения","Cov","Ген. Портрет","Втор. Находки","Рец. Носит.",
                "Мед. Находки","Мед. Стат.","Пол/ УАП","Дубли/ Родство","chr CNV","Гаплогруппы",
                "DNB","Статус","IGV","Дата обсчета","Комментарий"]:
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

    # === Комбинированные флаги ===
    st.markdown("### Комбинированные флаги")
    for c in ["Combi_NoAnswer_coef", "Combi_NoAnswer_TYPE", "fail_merge_tubes"]:
        if c in vacc_df.columns:
            s = vacc_df[c]
            if not pd.api.types.is_bool_dtype(s):
                s = s.astype(str).str.lower().isin(["1","true","yes","да","y","истина"])
            st.write(f"**{c}** — доля: {float(s.mean()):.1%}")

    # === Нозологии ===
    st.markdown("### Нозологии")
    def _clean(s): return s.astype(str).str.strip().replace({"": pd.NA})
    def _num(s): return pd.to_numeric(s, errors="coerce")
    def _bool(s):
        if pd.api.types.is_bool_dtype(s): return s.fillna(False)
        return _clean(s).str.lower().isin(["1","true","yes","да","y","истина"])

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
                st.metric("Болел", f"{float(_bool(vacc_df[sick_col]).mean()):.1%}")
            else:
                st.metric("Болел","—")
        with c4:
            if na_col in vacc_df.columns:
                st.metric("Нет ответа", f"{float(_bool(vacc_df[na_col]).mean()):.1%}")
            else:
                st.metric("Нет ответа","—")

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
                    bins = pd.qcut(vals[vals >= 0], q=min(20, max(5, int(len(vals)**0.5))), duplicates="drop")
                    hist = bins.value_counts().sort_index().rename_axis("bin").to_frame("count")
                except Exception:
                    hist = pd.cut(vals, bins=20).value_counts().sort_index().rename_axis("bin").to_frame("count")
                st_bar_chart_safe(hist)
            else:
                st.info("Нет численных значений титров для построения гистограммы.")

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
            show = [col_id] if col_id else []
            show += [x for x in [me_col, res_col, sick_col, info_col, total_col, na_col] if x]
            show += extra
            show = [c for c in show if c in vacc_df.columns]
            if show:
                st.dataframe(vacc_df[show].head(200))
            else:
                st.write("Нет колонок для отображения.")


    # === Дозы на пациента ===
    st.markdown("### Распределение числа доз на пациента")
    if col_id:
        doses_per_patient = (
            vacc_df.groupby(col_id)
                   .size()
                   .rename("doses")
                   .reset_index()
        )
        st.dataframe(doses_per_patient.head(100))
        try:
            # строим гистограмму по количеству доз
            hist = doses_per_patient["doses"].value_counts().sort_index().rename_axis("doses").to_frame("patients")
            st_bar_chart_safe(hist)
        except Exception:
            pass
    else:
        st.warning("Не найдена колонка ID пациента — пропускаем блок с дозами на пациента.")


    # === Экспорт ===
    st.markdown("### Экспорт данных по вакцинациям")
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")
    def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        bio.seek(0)
        return bio.read()

    col_va, col_vb = st.columns(2)
    with col_va:
        st.download_button("⬇️ Скачать сырые данные (CSV)", data=to_csv_bytes(vacc_df), file_name="vaccinations_raw.csv", mime="text/csv")
    with col_vb:
        st.download_button("⬇️ Скачать сырые данные (XLSX)", data=to_xlsx_bytes(vacc_df), file_name="vaccinations_raw.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


#####################
# Tab3: HLA × Measles
#####################
with tab3:
    st.subheader("Ассоциации HLA ↔︎ антитела к кори (measles)")
    # Загрузчик HLA
    hla_uploaded = st.sidebar.file_uploader("Загрузите HLA-таблицу (.xlsx)", type=["xlsx"], key="hla_upl")
    try:
        if hla_uploaded is not None:
            hla_df = read_xlsx(hla_uploaded)
            st.info("Используется загруженный HLA-файл.")
        else:
            hla_df = read_xlsx(HLA_DEFAULT_PATH)
            st.info(f"Используется HLA-файл по умолчанию: {HLA_DEFAULT_PATH}")
    except Exception as e:
        st.error(f"Не удалось загрузить HLA: {e}")
        st.stop()
    hla_df = normalize_cols(hla_df.copy())

    # ===== ЯВНО задаём ID-колонки, если они есть =====
    # HLA: sample_id
    if "sample_id" in hla_df.columns:
        hla_id_series = hla_df["sample_id"].astype("string").str.strip().replace({"": pd.NA})
    else:
        hla_id_series = coalesce_first_nonnull(hla_df, HLA_ID_COLS)
    # Вакцины: ZLIMS ID
    if "ZLIMS ID" in vacc_df.columns:
        vacc_id_col = "ZLIMS ID"
    else:
        vacc_id_col = pick_first_or_none(vacc_df if "vacc_df" in locals() else df, VACC_ID_COLS)

    with st.expander("Обнаруженные ID-колонки", expanded=False):
        st.write({
            "HLA_ID_candidates_present": [c for c in HLA_ID_COLS if c in hla_df.columns],
            "HLA_ID_used_nonnull": int(hla_id_series.notna().sum()) if hla_id_series is not None else 0,
            "VACC_ID_col_used": vacc_id_col,
        })
    if hla_id_series is None or hla_id_series.notna().sum() == 0 or not vacc_id_col:
        st.error("Не удалось определить ID в одном из файлов (HLA или вакцины). Проверьте названия колонок.")
        st.stop()

    # Настройки анализа
    c1, c2, c3 = st.columns(3)
    with c1:
        allele_level = st.selectbox("HLA: уровень агрегации аллеля", [1, 2], index=0, help="1 → A*01; 2 → A*01:01")
    with c2:
        min_carriers = st.number_input("Мин. число копий аллеля для показа", min_value=1, value=5, step=1)
    with c3:
        genes_all = sorted({c.split("_")[0] for c in hla_df.columns if "_" in str(c)})
        genes_pick = st.multiselect("Гены HLA", genes_all, default=genes_all)

    # Вакцинированные + валидный measles_result
    def _as_bool1(s):
        return s.astype(str).str.lower().isin(["1", "true", "yes", "да", "y", "истина"])
    meas_res_col = "measles_result"
    meas_vacc_col = "measles_vaccine_info"
    if meas_res_col not in vacc_df.columns or meas_vacc_col not in vacc_df.columns:
        st.warning("В таблице вакцинаций нет нужных колонок measles_result / measles_vaccine_info.")
        st.stop()
    vacc_work = vacc_df.copy()
    # только вакцинированные
    vacc_work = vacc_work[_as_bool1(vacc_work[meas_vacc_col])]
    # нормализуем результат: числовой, исключаем -1
    meas_res = pd.to_numeric(vacc_work[meas_res_col], errors="coerce")
    vacc_work = vacc_work.loc[meas_res.isin([0, 1])].copy()
    vacc_work[vacc_id_col] = vacc_work[vacc_id_col].astype(str)
    vacc_work["Group"] = meas_res.loc[vacc_work.index].astype(int)
    # Группа по ID (для merge)
    groups = vacc_work[[vacc_id_col, "Group"]].rename(columns={vacc_id_col: "ID"}).dropna()
    groups["ID"] = groups["ID"].astype(str)

    # Приводим HLA и фильтруем по выбранным генам
    if genes_pick:
        # оставим только выбранные гены + все id-кандидаты (для коалесценции)
        id_cols_present = [c for c in HLA_ID_COLS if c in hla_df.columns]
        gene_cols = [c for c in hla_df.columns if "_" in str(c) and c.split("_")[0] in genes_pick]
        keep_cols = id_cols_present + gene_cols
        hla_sub = hla_df[keep_cols].copy()
    else:
        hla_sub = hla_df.copy()
    # формируем единую колонку ID из коалесцированных кандидатов
    hla_sub["ID"] = hla_id_series.astype("string").str.strip()
    hla_sub = hla_sub[hla_sub["ID"].notna() & (hla_sub["ID"] != "")]

    # Long-форма HLA
    try:
        hla_long = process_hla_long(hla_sub, allele_level)
        hla_long["ID"] = hla_long["ID"].astype(str)
    except Exception as e:
        st.error(f"HLA нормализация не удалась: {e}")
        st.stop()

    # Аналитика
    res = counts_and_tests(hla_long, groups)
    if res.empty:
        st.info("Недостаточно данных для тестов. Проверьте пересечение ID и фильтры.")
        st.stop()

    # Фильтр по минимальному числу копий
    res = res[(res["count_res1"] + res["count_res0"]) >= int(min_carriers)]

    st.markdown("### Результаты ассоциаций (χ² / Fisher, FDR BH)")
    show_cols = ["Gene","Allele","count_res1","count_res0","n_res1","n_res0","freq_res1","freq_res0","delta_freq","test","p_value","p_fdr_bh","signif_fdr"]
    st.dataframe(res[show_cols], hide_index=True, use_container_width=True)

    # Топ-аллели по значимости
    st.markdown("### Топ-аллелей по значимости (≤ 20)")
    top = res.nsmallest(20, ["p_fdr_bh", "p_value"])
    if not top.empty:
        # построим частоты для топ-аллелей (плоская метка вместо MultiIndex)
        freq_df = top[["Gene","Allele","freq_res0","freq_res1"]].copy()
        freq_df["label"] = (freq_df["Gene"].astype(str) + ":" +
                            freq_df["Allele"].astype(str))
        freq_df = freq_df.set_index("label")[["freq_res0","freq_res1"]]
        st_bar_chart_safe(freq_df)
    else:
        st.info("Нет аллелей, удовлетворяющих фильтрам.")

    # Экспорт
    st.markdown("### Экспорт результатов")
    st.download_button(
        "⬇️ Скачать результаты (CSV)",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="hla_measles_associations.csv",
        mime="text/csv",
    )

#############################
# Tab4: HLA × measles_ME_ml
#############################
with st.expander("🧬⇄💉 HLA × Measles (количественный титр)", expanded=False):
    st.subheader("Ассоциации HLA ↔ количество антител (measles_ME_ml)")

    # проверка колонки с антителами
    meas_q_col = "measles_ME_ml"
    if meas_q_col not in vacc_df.columns or meas_vacc_col not in vacc_df.columns:
        st.warning("В таблице вакцинаций нет measles_ME_ml / measles_vaccine_info.")
        st.stop()

    # только вакцинированные с валидным числом антител
    vacc_q = vacc_df[_as_bool1(vacc_df[meas_vacc_col])].copy()
    vacc_q[meas_q_col] = pd.to_numeric(vacc_q[meas_q_col], errors="coerce")
    vacc_q = vacc_q.dropna(subset=[meas_q_col])

    groups_q = vacc_q[[vacc_id_col, meas_q_col]].rename(
        columns={vacc_id_col: "ID", meas_q_col: "Antibody"}
    )
    groups_q["ID"] = groups_q["ID"].astype(str)

    # long HLA для количественного анализа + строковые ID
    hla_sub_q = hla_sub.copy()
    hla_long_q = process_hla_long(hla_sub_q, allele_level)
    hla_long_q["ID"] = hla_long_q["ID"].astype(str)
    # приводим ID в таблице вакцинаций к строке перед merge
    vacc_df[vacc_id_col] = vacc_df[vacc_id_col].astype(str)
    merged_q = pd.merge(
        hla_long_q,
        vacc_df[[vacc_id_col, "measles_ME_ml", "cohort_region"]].rename(columns={vacc_id_col: "ID"}),
        on="ID",
        how="inner"
    )
    st.write("Совмещено строк (аллельные записи):", merged_q.shape)

    if merged_q.empty:
        st.warning("Нет пересечений ID между HLA и вакцинацией.")
    else:
        # лог-трансформация для титров
        merged_q["measles_ME_log10"] = np.log10(1 + merged_q["measles_ME_ml"].clip(lower=0))
        merged_q["measles_ME_int"] = merged_q["measles_ME_ml"].fillna(0).astype(int)

        st.write("Пример данных:", merged_q.head())

        import statsmodels.formula.api as smf

        results_by_region = {}
        for region, df_region in merged_q.groupby("cohort_region"):
            if df_region["measles_ME_log10"].nunique() < 2:
                continue
            try:
                # линейная регрессия log10 титра ~ наличие аллеля (пример: A*01:01)
                # для упрощения: проверим каждый аллель отдельно
                region_results = []
                # подготовим titer per ID
                titers = (df_region.groupby("ID")["measles_ME_log10"]
                                   .first()
                                   .to_frame("measles_ME_log10"))
                for allele in sorted(df_region["Allele"].dropna().unique()):
                    pres = (df_region.assign(allele_present=df_region["Allele"].eq(allele).astype(int))
                                     .groupby("ID")["allele_present"].max()
                                     .to_frame("allele_present"))
                    df_tmp = titers.join(pres, how="left").fillna({"allele_present":0})
                    if df_tmp["allele_present"].nunique() < 2:
                        continue
                    model = smf.ols("measles_ME_log10 ~ allele_present", data=df_tmp).fit()
                    region_results.append({
                        "region": region,
                        "allele": allele,
                        "coef": model.params.get("allele_present", np.nan),
                        "pval": model.pvalues.get("allele_present", np.nan),
                        "n": int(df_tmp.shape[0]),
                    })
                results_by_region[region] = pd.DataFrame(region_results)
            except Exception as e:
                st.write(f"Ошибка в регионе {region}: {e}")
        if results_by_region:
            for region, df_r in results_by_region.items():
                st.subheader(f"Регион: {region}")
                st.dataframe(df_r.sort_values("pval").head(20))
        else:
            st.warning("Не удалось построить регрессии по регионам.")

    merged_q = hla_long_q.merge(groups_q, on="ID", how="inner")
    res_q = []
    for g in sorted(merged_q["Gene"].unique()):
        sub = merged_q[merged_q["Gene"] == g]
        for allele in sub["Allele"].unique():
            # агрегируем до уровня пациента (чтобы не дублировать наблюдения)
            pres = (sub.assign(is_a=sub["Allele"].eq(allele).astype(int))
                      .groupby("ID")["is_a"].max())
            ant = sub.groupby("ID")["Antibody"].first()
            df_a = pd.concat([pres, ant], axis=1).dropna()
            vals_yes = df_a.loc[df_a["is_a"]==1, "Antibody"]
            vals_no  = df_a.loc[df_a["is_a"]==0, "Antibody"]
            if len(vals_yes) < 3 or len(vals_no) < 3:
                continue
            try:
                stat, p = mannwhitneyu(vals_yes, vals_no, alternative="two-sided")
                res_q.append({
                    "Gene": g,
                    "Allele": allele,
                    "n_carriers": int(len(vals_yes)),
                    "n_noncarriers": int(len(vals_no)),
                    "mean_carriers": vals_yes.mean(),
                    "mean_noncarriers": vals_no.mean(),
                    "delta_mean": vals_yes.mean() - vals_no.mean(),
                    "p_value": p
                })
            except Exception:
                continue
    res_q = pd.DataFrame(res_q)
    if res_q.empty:
        st.info("Недостаточно данных для количественного анализа.")
        st.stop()

    # FDR коррекция
    reject, pcor, _, _ = multipletests(res_q["p_value"], method="fdr_bh")
    res_q["p_fdr_bh"] = pcor
    res_q["signif_fdr"] = reject
    res_q = res_q.sort_values("p_fdr_bh")

    st.markdown("### Результаты U-теста (Манна–Уитни)")
    st.dataframe(res_q, hide_index=True, use_container_width=True)

    # Топ-аллели
    st.markdown("### Боксплоты для топ-аллелей")
    top_q = res_q.nsmallest(6, "p_fdr_bh")
    for _, row in top_q.iterrows():
        g, a = row["Gene"], row["Allele"]
        sub = merged_q.assign(
            Carrier=merged_q["Allele"].eq(a).map({True: f"{g}-{a}", False: "Other"})
        )
        fig, ax = plt.subplots(figsize=(5,4))
        sns.boxplot(data=sub, x="Carrier", y="Antibody", ax=ax)
        sns.stripplot(data=sub, x="Carrier", y="Antibody", color="black", size=3, alpha=0.6, ax=ax)
        ax.set_title(f"{g}-{a} (p={row['p_fdr_bh']:.3g})")
        st.pyplot(fig)
    # Экспорт
    st.download_button(
        "⬇️ Скачать количественный анализ (CSV)",
        data=res_q.to_csv(index=False).encode("utf-8"),
        file_name="hla_measles_antibody_levels.csv",
        mime="text/csv",
    )
