
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
import io

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
    "zlims_id": ["zlims_id", "zlims"]
}

SEQ_COLS_CANDIDATES = [
    "status_coverage", "VAR_IEI_panel_table_status", "SANGER"
]

COHORT_COL = "combi_type_mcch52"

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

# ----------------------
# Load data
# ----------------------

st.sidebar.header("Данные")
uploaded = st.sidebar.file_uploader("Загрузите Excel-файл (.xlsx) c таблицей", type=["xlsx"])
default_path = "PID_AID_phenotypes_13.08.25.xlsx"

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
df.columns = [str(c).strip() for c in df.columns]

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
id_show_cols = pick_existing(df_marked, ["ID", "uin2_number", "uin2_fulltxt", "zlims_id"])
binary_table = pd.concat([df_marked[id_show_cols], binary_table], axis=1) if id_show_cols else binary_table

# ----------------------
# Tabs
# ----------------------
tab1, = st.tabs(["🧬 Метаданные"])

with tab1:
    st.subheader("Сводка по меткам (в текущей выборке)")
    counts = df_marked[presence_cols].sum().sort_values(ascending=False).rename("count").to_frame()
    st.dataframe(counts)

    # === Визуализации метаданных ===
    st.markdown("### Гистограммы по метаданным")
    # 1) Столбчатая диаграмма по суммарным упоминаниям
    try:
        st.bar_chart(counts)
    except Exception:
        st.write("Не удалось построить bar chart для сводки.")

    # 2) Распределение числа меток на пациента (сколько совпадений паттернов у каждой строки)
    try:
        matches_per_row = df_marked[presence_cols].astype(bool).sum(axis=1)
        st.markdown("**Распределение количества упоминаний на пациента**")
        # превратим в частоты
        hist_df = matches_per_row.value_counts().sort_index().rename_axis("num_flags").to_frame("patients")
        st.bar_chart(hist_df)
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
