import joblib
import pandas as pd
from typing import Dict
from sqlalchemy import create_engine
import os
from pathlib import Path
from io import StringIO

# === Добавляем путь к модулям ===
import sys
sys.path.append(str(Path(__file__).parent / "models"))

# === Импортируем функции ===
from models.functions import (
    predict_from_raw_without_fires,
    evaluate_predictions_with_new_fires,
    build_labeled_base_for_new_period,
    finetune_with_new_labels,
    build_base_from_raw_without_fires
)

# === Пути ===
MODEL_DIR = Path("src/models")
MODELS_DICT_PATH = MODEL_DIR / "models_dict.pkl"
BASE_HISTORICAL_PATH = MODEL_DIR / "base_historical.pkl"

PREDS_DF = []

# === Загрузка моделей ===
try:
    models_dict = joblib.load(MODELS_DICT_PATH)
    base_historical = joblib.load(BASE_HISTORICAL_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

# === БД ===
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dev:chakaton@db:5432/coal_fire_db")
engine = create_engine(DATABASE_URL)

# === Функции ===
def load_raw_data_from_db():
    temp_df = pd.read_sql("SELECT * FROM temperatures_data", engine)
    supplies_df = pd.read_sql("SELECT * FROM supplies_data", engine)
    weather_df = pd.read_sql("SELECT * FROM weather_data", engine)
    return temp_df, supplies_df, weather_df

def get_predictions() -> Dict:
    try:
        temp_df, supplies_df, weather_df = load_raw_data_from_db()
        

        if temp_df.empty or supplies_df.empty or weather_df.empty:
            return {"predictions": []}

        preds_df = predict_from_raw_without_fires(
            models_dict=models_dict,
            supplies_raw=supplies_df,
            temp_raw=temp_df,
            weather_raw=weather_df
        )

        if preds_df.empty:
            return {"predictions": []}


        preds_list = preds_df.to_dict(orient="records")
        PREDS_DF = preds_df
        for row in preds_list:
            row["target_date"] = row["target_date"].strftime("%Y-%m-%d")
        return {"predictions": preds_list}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Ошибка прогноза: {str(e)}")

def update_model_with_new_fires(fires_csv_content: str):
    """
    Дообучение модели после загрузки fires.csv
    """
    try:
        global models_dict, base_historical

        fires_df = pd.read_csv(StringIO(fires_csv_content))  

        temp_df, supplies_df, weather_df = load_raw_data_from_db()

        base_new = build_base_from_raw_without_fires(supplies_df, temp_df, weather_df) 

        base_new_labeled = build_labeled_base_for_new_period(base_new=base_new, fires_new_raw=fires_df)

        combined_base = pd.concat([base_historical, base_new_labeled], ignore_index=True)
        models_dict_updated = finetune_with_new_labels(
            models_dict=models_dict,
            df_historical=combined_base,
            df_new_labeled=base_new_labeled
        )

        joblib.dump(models_dict_updated, MODELS_DICT_PATH)
        joblib.dump(combined_base, BASE_HISTORICAL_PATH)

        models_dict = models_dict_updated
        base_historical = combined_base

        metrics = evaluate_predictions_with_new_fires(
            preds_df=PREDS_DF,
            fires_new_raw=fires_df
        )

        return {
            "status": "success",
            "metrics": metrics,
            "rows_processed": len(fires_df),
            "message": "Model retrained and evaluated"
        }

    except Exception as e:
        raise RuntimeError(f"Ошибка дообучения: {str(e)}")
