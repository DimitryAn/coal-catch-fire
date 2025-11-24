from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query 
import pandas as pd
import numpy as np 
import io
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from database import Base 
from prediction import get_predictions, update_model_with_new_fires
import time



DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dev:chakaton@db:5432/coal_fire_db")

engine = create_engine(DATABASE_URL)

while True:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        break
    except OperationalError as e:
        time.sleep(2)
    except Exception as e:
        time.sleep(2)

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPECTED_COLUMNS = {
    "temperatures": [
       "Склад", "Штабель", "Марка", "Максимальная температура", "Пикет", "Дата акта", "Смена"
    ],
    "supplies": [
        "ВыгрузкаНаСклад", "Наим. ЕТСНГ", "Штабель", "ПогрузкаНаСудно",
        "На склад, тн", "На судно, тн", "Склад"
    ],
    "weather": [
        "date", "t", "p", "humidity", "precipitation", "wind_dir", "v_avg", "v_max",
        "cloudcover", "visibility", "weather_code"
    ],
    "fires": [
        "Дата составления", "Груз", "Вес по акту, тн", "Склад",
        "Дата начала", "Дата оконч.", "Нач.форм.штабеля", "Штабель"
    ]
}

TABLE_NAMES = {
    "temperatures": "temperatures_data",
    "supplies": "supplies_data",
    "weather": "weather_data",
    "fires": "fires_data"
}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/upload-multiple-csv/")
async def upload_multiple_csv(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не загружены")

    uploaded = {}
    missing = []
    invalid_files = []
    results = []

    for file in files:
        original_name = file.filename
        name_lower = original_name.lower()

        if "temperature" in name_lower and "weather" not in name_lower:
            file_type = "temperatures"
        elif "suppl" in name_lower:
            file_type = "supplies"
        elif "weather" in name_lower:
            file_type = "weather"
        else:
            invalid_files.append(original_name)
            continue

        try:
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
            df = df.where(pd.notnull(df), None)
            # Валидация столбцов
            expected_cols = EXPECTED_COLUMNS[file_type]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют столбцы: {missing_cols}")

            table_name = TABLE_NAMES[file_type]
            df.to_sql(table_name, engine, if_exists='append', index=False)

            results.append({
                "file": original_name,
                "type": file_type,
                "rows": len(df),
                "table": table_name
            })
            uploaded[file_type] = True

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка при обработке файла {original_name}: {str(e)}"
            )

    # Проверка
    for key in ["temperatures", "supplies", "weather"]:
        if not uploaded.get(key):
            missing.append(key)

    if missing:
        return JSONResponse(
            content={
                "warning": f"Не все файлы загружены. Отсутствуют: {', '.join(missing)}",
                "uploaded": results,
                "missing": missing
            },
            status_code=206
        )

    return JSONResponse(
        content={
            "message": "Все файлы успешно загружены",
            "files": results
        }
    )

@app.get("/get_data")
async def get_data(
    table: str = Query(..., description="Название таблицы: temperatures, supplies, weather fires"),
    limit: int = Query(100, le=1000) 
):

    table_mapping = {
        "temperatures": "temperatures_data",
        "supplies": "supplies_data",
        "weather": "weather_data",
        "fires": "fires_data"
    }

    if table not in table_mapping:
        raise HTTPException(
            status_code=400,
            detail="Допустимые значения: temperatures, supplies, weather"
        )

    actual_table = table_mapping[table]

    try:

        query = f"SELECT * FROM {actual_table} ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql(query, engine)
        df = df.replace({np.nan: None})
        return JSONResponse(
            content={
                "table": actual_table,
                "count": len(df),
                "data": df.to_dict(orient="records")
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения из БД: {str(e)}")
@app.post("/upload-fires-csv/")
async def upload_fires_csv(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        df = df.where(pd.notnull(df), None)

        # Валидация столбцов
        expected_cols = EXPECTED_COLUMNS["fires"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют столбцы: {missing_cols}")

        # Загрузка
        table_name = TABLE_NAMES["fires"]
        df.to_sql(table_name, engine, if_exists='append', index=False)

        return JSONResponse(
            content={
                "message": "Файл fires.csv успешно загружен",
                "rows": len(df),
                "table": table_name
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при обработке fires.csv: {str(e)}"
        )
        
@app.get("/predict/")
async def predict_endpoint():
    """
    Получить прогноз риска самовозгорания
    """
    try:
        result = get_predictions()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прогноза: {str(e)}")

@app.post("/update-model/")
async def update_model_endpoint(file: UploadFile = File(...)):
    """
    Дообучение модели на новых данных из fires.csv
    """
    try:
        content = await file.read()
        content_str = content.decode("utf-8")
        result = update_model_with_new_fires(content_str)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка дообучения: {str(e)}")

# Раздача фронтенда
app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("src/frontend/index.html")
