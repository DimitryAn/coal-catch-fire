import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sqlalchemy import create_engine
import os
from pathlib import Path

# Пути
MODEL_DIR = Path("src/models")
MODELS_DICT_PATH = MODEL_DIR / "models_dict.pkl"
BASE_HISTORICAL_PATH = MODEL_DIR / "base_historical.pkl"

# Загрузка при старте
models_dict = joblib.load(MODELS_DICT_PATH)
base_historical = joblib.load(BASE_HISTORICAL_PATH)

# БД
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dev:chakaton@db:5432/coal_fire_db")
engine = create_engine(DATABASE_URL)

# Функции из ноутбука — копируйте сюда (или импортируйте, если вынесете)
# Для краткости — приведу только вызовы
from all_functions import (
    preprocess_weather,
    preprocess_supplies,
    preprocess_temperature,
    predict_from_raw_without_fires,
    evaluate_predictions_with_new_fires,
    build_labeled_base_for_new_period,
    finetune_with_new_labels
)

def load_raw_data_from_db() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает свежие данные из БД
    """
    temp_df = pd.read_sql("SELECT * FROM temperatures_data", engine)
    supplies_df = pd.read_sql("SELECT * FROM supplies_data", engine)
    weather_df = pd.read_sql("SELECT * FROM weather_data", engine)
    
    return temp_df, supplies_df, weather_df

def get_predictions() -> Dict:
    """
    Основной вызов: получить прогноз на 3 дня вперёд
    """
    temp_df, supplies_df, weather_df = load_raw_data_from_db()

    # Прогноз
    base_new, preds_df = predict_from_raw_without_fires(
        models_dict=models_dict,
        supplies_raw=supplies_df,
        temp_raw=temp_df,
        weather_raw=weather_df
    )

    # Преобразуем в JSON-дружественный формат
    preds_list = preds_df.to_dict(orient="records")
    for row in preds_list:
        row["target_date"] = row["target_date"].strftime("%Y-%m-%d")

    return {
        "predictions": preds_list,
        "base_new": base_new.to_dict(orient="records")  # для дообучения
    }

def update_model_with_new_fires(fires_csv_content: str):
    """
    Дообучение: пришёл fires.csv → обновляем модель
    """
    # Читаем fires
    fires_df = pd.read_csv(pd.io.StringIO(fires_csv_content))

    # Загружаем свежие данные
    temp_df, supplies_df, weather_df = load_raw_data_from_db()

    # Перестраиваем base_new
    base_new = build_base_from_raw_without_fires(
        supplies_raw=supplies_df,
        temp_raw=temp_df,
        weather_raw=weather_df
    )

    # Склеиваем с новыми метками
    base_new_labeled = build_labeled_base_for_new_period(base_new, fires_df)

    # Дообучаем
    combined_base = pd.concat([base_historical, base_new_labeled], ignore_index=True)
    models_dict_updated = finetune_with_new_labels(
        models_dict=models_dict,
        df_historical=combined_base,
        df_new_labeled=base_new_labeled
    )

    # Сохраняем
    joblib.dump(models_dict_updated, MODELS_DICT_PATH)
    joblib.dump(combined_base, BASE_HISTORICAL_PATH)

    # Перезагружаем глобальные переменные
    global models_dict, base_historical
    models_dict = models_dict_updated
    base_historical = combined_base

    # Считаем метрики
    metrics = evaluate_predictions_with_new_fires(
        preds_df=pd.DataFrame(get_predictions()["predictions"]),
        fires_new_raw=fires_df
    )

    return {"status": "success", "metrics": metrics}