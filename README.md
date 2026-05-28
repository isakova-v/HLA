# HLA & Vaccine Immune Response Analysis

Анализ ассоциаций аллелей HLA с гуморальным иммунным ответом на вакцинацию (корь, краснуха, дифтерия, гепатит В) в российских когортах (Иркутск, Амур, Нижний Новгород, Калининград).

## Установка зависимостей

```bash
pip install pandas matplotlib seaborn statsmodels openpyxl scipy numpy plotly catboost xgboost shap streamlit scikit-learn
```

---

## Структура репозитория

```
.
├── app.py                          # Streamlit-дашборд PID/AID
├── pipeline/                       # Подготовка данных и статистический анализ
├── visualization/                  # Форест-плоты и визуализация
├── epitope_analysis/               # Интеграция с IEDB (эпитопный анализ)
├── negative_control/               # Негативный контроль (3-е поле аллелей)
└── ml/                             # Машинное обучение (классификация ответа)
```

---

## Пайплайн анализа

Типичный порядок запуска:

```
1. combine_hla_hd.py     Конвертация HLA-HD → XLSX
2. split_pheno_by_vaccine.py  Разделение фенотипа по вакцинам
3. normalize_hla_to_2field.py Нормализация аллелей до 2-го поля
4. join_hla_vaccines.py   Объединение HLA + вакцинные данные
5. filter.py              Исключение пациентов
6. hla_analysis.py        Частотный анализ + chi² + FDR
7. betas.py / hla_pc_regression.py   Размеры эффекта / регрессия
8. Визуализация (форест-плоты, SHAP)
9. Эпитопный анализ (IEDB)
```

---

## pipeline/ — Подготовка данных и анализ

### combine_hla_hd.py

Конвертирует пакет результатов HLA-HD (`*_final.result.txt`) в единую таблицу XLSX.

| | Описание |
|---|---|
| **Вход** | `--in` — директория с `*_final.result.txt` файлами HLA-HD |
| | `--template` — XLSX-шаблон со схемой колонок |
| **Выход** | `--out` — XLSX-таблица: строки = образцы, колонки = `HLA-{Gene}_1`, `HLA-{Gene}_2` |

```bash
python pipeline/combine_hla_hd.py --in hla-hd_results/ --template combined_hla_table.xlsx --out combined_hla_out.xlsx
```

### hla_results_to_xlsx.py

Альтернативный конвертер HLA-HD результатов (аналогичный интерфейс).

### split_pheno_by_vaccine.py

Разделяет единый фенотипический файл на отдельные файлы по вакцинам.

| | Описание |
|---|---|
| **Вход** | `--pheno` — TSV с фенотипом (титры, вакцинация, демография) |
| | `--unrelated` — TXT со списком неродственных sample ID |
| **Выход** | `{vaccine}.xlsx` в `--outdir` для каждой вакцины (measles, rubella, diphtheria, HBV) |

```bash
python pipeline/split_pheno_by_vaccine.py --pheno pheno_clean.tsv --unrelated list_unrelated.txt --outdir pheno_by_vaccine/
```

### normalize_hla_to_2field.py

Приводит аллели к разрешению 2-го поля (например, `HLA-A*03:01:01G` → `HLA-A*03:01`).

| | Описание |
|---|---|
| **Вход** | Позиционные аргументы — XLSX-файлы (или glob-паттерн) |
| **Выход** | `{имя}_2field.xlsx` рядом с каждым входным файлом |

```bash
python pipeline/normalize_hla_to_2field.py data/*.xlsx
```

### join_hla_vaccines.py

Объединяет HLA-таблицу с вакцинными файлами, отфильтровывая редкие аллели и низковариабельные гены.

| | Описание |
|---|---|
| **Вход** | `--hla` — XLSX с HLA-генотипами (колонки `HLA-{Gene}_1`, `HLA-{Gene}_2`) |
| | `--vaccines` — список XLSX с вакцинными фенотипами |
| **Параметры** | `--min_unique_alleles` (default=10) — мин. число уникальных аллелей для гена |
| | `--min_allele_count` (default=10) — мин. число носителей аллеля |
| **Выход** | `{vaccine}_with_HLA_filtered.xlsx` + `summary.xlsx` в `--outdir` |

```bash
python pipeline/join_hla_vaccines.py --hla combined_hla_out.xlsx \
  --vaccines measles.xlsx rubella.xlsx diphtheria.xlsx HBV.xlsx --outdir data/
```

### filter_hla_genes_by_alleles.py

Фильтрует HLA-колонки, оставляя только гены с >= 10 уникальных аллелей.

| | Описание |
|---|---|
| **Вход** | Позиционные аргументы — XLSX-файлы |
| **Выход** | `{имя}_gene_filtered.xlsx` рядом с каждым входным файлом |

### filter.py

Фильтрует пациентов по списку исключений.

| | Описание |
|---|---|
| **Вход** | `--survey` — XLSX с данными пациентов |
| | `--exclude_ids` — XLSX со списком ID для исключения |
| **Параметры** | `--survey_id_col`, `--exclude_id_col` — названия колонок с ID |
| **Выход** | `--output` — отфильтрованный XLSX |

```bash
python pipeline/filter.py --survey data/survey.xlsx --exclude_ids data/exclude_ids.xlsx \
  --survey_id_col patient_id --exclude_id_col patient_id --output data_filtered/survey_filtered.xlsx
```

### hla_analysis.py

Частотный анализ аллелей HLA между группами (тест/контроль) с chi²-тестами и FDR-коррекцией.

| | Описание |
|---|---|
| **Вход** | `--survey` — XLSX с фенотипом, `--hla` — XLSX с HLA-типами |
| | `--control` — CSV/TSV с HLA контрольной группы |
| **Параметры** | `--allele_field` (1 или 2) — уровень разрешения аллеля |
| **Выход** | В `--output`: `allele_frequencies.csv`, `significant_alleles.csv`, PNG-гистограммы частот |

```bash
python pipeline/hla_analysis.py --survey data_filtered/survey.xlsx --survey_id_col patient_id \
  --hla data/HLA.xlsx --hla_id_col sample_id --control data/control.csv --output results/ --allele_field 2
```

### betas.py

Анализ размеров эффекта (Hedges' g) аллелей HLA на уровень антител.

| | Описание |
|---|---|
| **Вход** | `--vaccine-xlsx` — список XLSX с вакцинными данными |
| | `--hla-xlsx` — XLSX с HLA-генотипами |
| | `--rare-genes` — TXT со списком редких аллелей |
| **Параметры** | `--target-col` — колонка с титрами (например `measles_ME_ml`) |
| | `--resolution` (first/second) — уровень разрешения аллеля |
| | `--min-carriers` (default=10) — мин. число носителей |
| | `--use-pc` — включить PC-коррекцию, `--n-pcs` (default=20) |
| | `--p-thr` (default=0.05) — порог значимости |
| **Выход** | В `--outdir`: `{vaccine}.betas.xlsx` (коэффициенты, SE, p-value, 95% CI), `{vaccine}.reference_alleles.tsv`, PNG-плоты в `outdir/plots/` |

```bash
python pipeline/betas.py --vaccine-xlsx measles.xlsx rubella.xlsx \
  --hla-xlsx combined_hla_out.xlsx --rare-genes hla_rare_alleles.txt \
  --target-col measles_ME_ml --resolution second --outdir out_betas/
```

### hla_pc_regression.py

OLS-регрессия: `log1p(titer) ~ allele_dosage + PC1..PC20 + covariates`.

| | Описание |
|---|---|
| **Вход** | `--vacc` — TSV/CSV с фенотипом (титры, демография, PC1-PC20) |
| | `--hla` — XLSX с HLA-генотипами |
| **Параметры** | `--allele-level` (1/2) — разрешение аллеля |
| | `--min-freq-percent` (default=1.0) — мин. частота аллеля (%) |
| **Выход** | В `--out`: `combined_coeffs.csv`, `design_stats.csv`, `{vaccine}_coeffs.csv` (бета-коэффициенты, SE, p-value, 95% CI для каждого аллеля) |

```bash
python pipeline/hla_pc_regression.py --vacc pheno.tsv --hla combined_hla_out.xlsx --out hla_pc_reg_out/
```

### haplotypes.py

Анализ нефазированных межгенных гаплотипов HLA.

| | Описание |
|---|---|
| **Вход** | `--hla` — XLSX с HLA-генотипами |
| | `--vaccine` (многократный) — `NAME:PATH.xlsx` |
| **Параметры** | `--genes` — гены через запятую (например `A,B,DRB1`) |
| | `--field` (first/second), `--top_k` (default=20), `--by_region` (флаг) |
| **Выход** | PNG-тепловые карты гаплотипов в `--outdir` |

```bash
python pipeline/haplotypes.py --hla combined_hla_out.xlsx \
  --vaccine measles:measles.xlsx --vaccine rubella:rubella.xlsx \
  --genes A,B,DRB1 --field first --outdir haplo_out/
```

---

## visualization/ — Визуализация результатов

### hla_pc_reg_figs.py

Строит форест-плоты по результатам `hla_pc_regression.py`.

| | Описание |
|---|---|
| **Вход** | CSV с колонками: Vaccine, Gene, Allele, coef, p_value (default: `hla_pc_reg_out/combined_coeffs.csv`) |
| **Выход** | PNG-графики коэффициентов в `hla_pc_reg_figs/` |

### make_hla_forest.py

Форест-плоты для топ-10 значимых аллелей по каждой вакцине и гену с BH-FDR.

| | Описание |
|---|---|
| **Вход** | `--pheno` — TSV с титрами и флагами регионов |
| | `--hla` — XLSX с HLA-генотипами |
| **Параметры** | `--allele-level` (1/2), `--norm` (log1p/zscore), `--alpha` (default=0.05) |
| **Выход** | PNG форест-плоты + `allele_significance_tables.csv` в `--outdir` |

### hla_effect_sizes.py

Дозо-зависимый анализ эффектов: сравнение титров между носителями 0, 1 и 2 копий аллеля.

| | Описание |
|---|---|
| **Вход** | `--hla` — XLSX, `--vaccine` (многократный) — `NAME:PATH.xlsx` |
| | `--rare_genes` — TXT (опционально: `--rare_first`, `--rare_second` — TSV) |
| **Параметры** | `--field` (first/second), `--min_carriers` (default=5), `--fdr` (default=0.05) |
| **Выход** | `effects_all_{field}_field.tsv`, `effects_significant_{field}_field.tsv`, PNG-плоты в `--outdir` |

### vaccine_forest_plots_*.py (7 вариантов)

Каждый вариант обрабатывает свой формат входных данных:

| Скрипт | Вход | Особенность |
|---|---|---|
| `vaccine_forest_plots.py` | TSV с титрами + регион-флаги + HLA | По регионам |
| `vaccine_forest_plots_by_allele.py` | CSV/TSV с аллелями + титрами | По аллелям, опция `--no-region` |
| `vaccine_forest_plots_by_allele_and_region_all.py` | CSV/TSV с аллелями + регион-флаги | По регионам + «Все регионы» |
| `vaccine_forest_plots_from_tsv.py` | TSV/CSV с фенотипом | Hedges' g (NoAnswer vs Answer) |
| `vaccine_forest_plots_all_noregion_FULL.py` | TSV + HLA | Все образцы без разбиения по регионам |
| `vaccine_forest_plots_two_tables_noregion_joinfix.py` | Два файла: фенотип + HLA | Авто-объединение по ID |
| `vaccine_forest_plots_two_tables_genotypes.py` | Два файла с генотипами (HLA-A_1, HLA-A_2) | Раскрытие генотипов в бинарные индикаторы |

**Общий выход:** PNG форест-плоты + CSV с размерами эффекта (`effects_index.csv`) в `--out`.

### metadata.py

Визуализация метаданных когорты с пастельными колормапами.

---

## epitope_analysis/ — Интеграция с IEDB

### epitope_allele_effect_viz_iedb_adapted.py

Визуализация связей «эпитоп–аллель» с окраской по размеру генетического эффекта.

| | Описание |
|---|---|
| **Вход** | `--mhc1` — CSV (IEDB MHC-I предсказания), `--mhc2` — CSV (IEDB MHC-II) |
| | `--effects` — TSV с колонками `allele`, `g` (размер эффекта) |
| **Параметры** | `--top-epitopes` (default=35), `--top-alleles` (default=45) |
| **Выход** | В `--outdir`: PNG bubble-плоты, TSV (`edges_mhc1.tsv`, `edges_mhc2.tsv`, `epitope_summary.tsv`) |

### tables_epitopes_freq_iedb_adapted.py

Сводная таблица: частоты аллелей + размеры эффекта + IEDB-предсказания.

| | Описание |
|---|---|
| **Вход** | `--effects` — TSV (эффекты), `--freq` — TSV (частоты аллелей) |
| | `--mhc1` — CSV, `--mhc2` — CSV (IEDB) |
| **Параметры** | `--alpha` (default=0.05), `--use-fdr` (флаг) |
| **Выход** | `--out` — XLSX сводная таблица |

---

## negative_control/ — Негативный контроль

Валидация: различия между вариантами 3-го поля внутри одного аллеля 2-го поля не должны влиять на титры.

### hla_negative_control_third_field.py

| | Описание |
|---|---|
| **Вход** | `--vaccine` (многократный) — `NAME:PATH.xlsx` (файлы с 3-полевыми аллелями + титрами) |
| **Параметры** | `--min_variant_n` (default=5), `--fdr` (default=0.05) |
| **Выход** | TSV + PNG в `--outdir`: `negative_control_effects_all.tsv`, плоты |

### pairwise_third_field.py

Попарные сравнения вариантов 3-го поля (например, `HLA-A*02:01:01` vs `HLA-A*02:01:02`).

| | Описание |
|---|---|
| **Вход** | `--vaccine` (многократный) — `NAME:PATH.xlsx` |
| **Параметры** | `--min_n_per_variant` (default=3), `--fdr` (default=0.05) |
| **Выход** | TSV + PNG в `--outdir` |

---

## ml/ — Машинное обучение

### vaccine_prediction_app.py

Streamlit-приложение: предсказание иммунного ответа **только по HLA-генотипу** (без возраста, пола, региона).

| | Описание |
|---|---|
| **Вход** | Предобученные модели в `./artifacts/`: `{VACCINE}_classifier.cbm`, `{VACCINE}_shap_classif.csv`, `meta.json` |
| **Выход** | Интерактивный веб-интерфейс с предсказаниями и SHAP-визуализацией |

```bash
streamlit run ml/vaccine_prediction_app.py
```

### app_ML.py

Полный ML-пайплайн: обучение CatBoost/XGBoost/sklearn моделей с кросс-валидацией и SHAP-анализом.

| | Описание |
|---|---|
| **Вход** | Загрузка через UI: XLSX с HLA-генотипами + вакцинными фенотипами |
| **Параметры (UI)** | use_age, folds, seed, cb_iters, shap_rows |
| **Выход** | Метрики CV, SHAP-плоты важности признаков (в интерфейсе) |

```bash
streamlit run ml/app_ML.py
```

---

## app.py — Дашборд PID/AID

Интерактивный Streamlit-дашборд для исследования когорт с первичными иммунодефицитами (PID) и аутоиммунными заболеваниями (AID).

```bash
streamlit run app.py
```

---

## Форматы данных

### Входные файлы

| Файл | Формат | Ключевые колонки |
|---|---|---|
| HLA-генотипы | XLSX | `sample_id`, `HLA-A_1`, `HLA-A_2`, `HLA-B_1`, `HLA-B_2`, ... |
| Фенотип (вакцины) | XLSX/TSV | `{vaccine}_ME_ml` (титр), `{vaccine}_vaccine_info` (флаг), `age`, `sex` |
| Регион-флаги | в фенотипе | `is_from_Irkutsk`, `is_from_Amur`, `is_from_NiNo`, `is_from_Kaliningrad` |
| Контрольная группа | CSV/TSV | Аналогично HLA-генотипам |
| IEDB предсказания | CSV | Экспорт из IEDB (MHC-I: `%Rank`, `Allele`, `Peptide`; MHC-II аналогично) |
| Редкие аллели | TXT | Один аллель на строку |

### Выходные файлы

| Тип | Формат | Содержание |
|---|---|---|
| Размеры эффекта | XLSX/TSV | `allele`, `g` (Hedges' g), `p_value`, `p_fdr`, `CI_low`, `CI_high`, `n_carriers` |
| Регрессия | CSV | `Vaccine`, `Gene`, `Allele`, `coef` (beta), `se`, `p_value`, `CI_0.025`, `CI_0.975` |
| Частоты | CSV | `gene`, `allele`, `freq_test`, `freq_control`, `chi2`, `p_value`, `fdr` |
| Форест-плоты | PNG | Горизонтальные barplot/errorbar с CI, отсортированные по |beta| |
| SHAP-плоты | PNG | Beeswarm/bar SHAP-визуализация важности признаков |
