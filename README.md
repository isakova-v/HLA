Анализ HLA-данных

Репозиторий содержит два основных скрипта для работы с данными HLA-типирования пациентов:
	1.	filter.py — фильтрация пациентов по списку исключений.
	2.	hla_analysis.py — анализ частот аллелей HLA, построение гистограмм и проведение статистических тестов (χ² с FDR-поправкой).

 1. Скрипт filter.py

Назначение

Фильтрует таблицу результатов опроса пациентов, исключая записи с ID, указанные во втором файле.

Параметры
	•	--survey — Excel-файл с результатами опроса пациентов.
	•	--exclude_ids — Excel-файл со списком ID, подлежащих исключению.
	•	--survey_id_col — название колонки с ID в файле опроса.
	•	--exclude_id_col — название колонки с ID в списке исключений.
	•	--output — имя выходного Excel-файла.

Пример запуска
python filter.py \
  --survey data/survey.xlsx \
  --exclude_ids data/exclude_ids.xlsx \
  --survey_id_col patient_id \
  --exclude_id_col patient_id \
  --output data_filtered/survey_filtered.xlsx

2. Скрипт hla_analysis.py

Назначение

Выполняет анализ частот аллелей HLA между группами (контроль/тест), строит гистограммы для каждого гена и определяет статистически значимые различия по аллелям с поправкой на множественные сравнения (FDR).

Параметры
	•	--survey — Excel-файл с результатами опроса пациентов.
	•	--survey_id_col — название колонки с ID в файле опроса.
	•	--hla — Excel-файл с HLA-типами пациентов.
	•	--hla_id_col — название колонки с ID пациента в HLA-файле.
	•	--control — файл (CSV/TSV) с HLA-типами контрольной группы.
	•	--output — папка для сохранения результатов.
	•	--allele_field — уровень агрегации аллеля (1 — A02, 2 — A02:01).

Результаты работы

В указанной папке --output создаются:
	•	allele_frequencies.csv — частоты аллелей по каждому гену (тест/контроль).
	•	significant_alleles.csv — таблица с p-value, FDR и флагами значимости по каждому аллелю.
	•	PNG-гистограммы частот для каждого гена (A_frequencies.png, B_frequencies.png и т.д.).

Пример запуска
python hla_analysis.py \
  --survey data_filtered/survey_filtered.xlsx \
  --survey_id_col patient_id \
  --hla data/HLA_patients.xlsx \
  --hla_id_col sample_id \
  --control data/control_group.csv \
  --output results_hla \
  --allele_field 2

  Установка зависимостей
  pip install pandas matplotlib seaborn statsmodels openpyxl

  Структура репозитория
  .
├── filter.py
├── hla_analysis.py
└── README.md
