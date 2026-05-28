import pandas as pd
import argparse
import os

def filter_patients(survey_path, exclude_ids_path, output_filename, survey_id_col, exclude_id_col):
    # Загрузка данных
    df = pd.read_excel(survey_path)
    exclude_ids = pd.read_excel(exclude_ids_path)

    # Приводим названия колонок к строковому типу
    df.columns = df.columns.astype(str)
    exclude_ids.columns = exclude_ids.columns.astype(str)

    # Проверка наличия нужных колонок
    if survey_id_col not in df.columns:
        raise ValueError(f"Колонка '{survey_id_col}' не найдена в файле опроса.")
    if exclude_id_col not in exclude_ids.columns:
        raise ValueError(f"Колонка '{exclude_id_col}' не найдена в файле исключений.")

    # Получение списка ID для исключения
    exclude_list = exclude_ids[exclude_id_col].astype(str).tolist()

    # Фильтрация
    df_filtered = df[~df[survey_id_col].astype(str).isin(exclude_list)]

    # Создание папки для сохранения, если её нет
    output_dir = "data_filtered"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Сохраняем результат
    df_filtered.to_excel(output_path, index=False)
    print(f"Сохранено: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, help="Файл с результатами опроса (.xlsx)")
    parser.add_argument("--exclude", required=True, help="Файл с ID для исключения (.xlsx)")
    parser.add_argument("--output", required=True, help="Имя выходного файла (без пути)")
    parser.add_argument("--survey-id-col", required=True, help="Название колонки с ID в таблице опроса")
    parser.add_argument("--exclude-id-col", required=True, help="Название колонки с ID для исключения")
    args = parser.parse_args()

    filter_patients(
        args.survey,
        args.exclude,
        args.output,
        args.survey_id_col,
        args.exclude_id_col
    )
