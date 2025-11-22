from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware


from database import engine, Base

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /ping с методом GET
@app.get("/ping")
async def ping():
    return {"status": "ok"}


t 
Base.metadata.create_all(bind=engine)

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), table_name: str = "data-from-web"):
    if file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
        raise HTTPException(status_code=400, detail="File must be CSV")

    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        df.to_sql(table_name, engine, if_exists='append', index=False)

        return JSONResponse(
            content={
                "message": f"Файл загружен в таблицу '{table_name}'",
                "rows": len(df),
                "columns": list(df.columns)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки csv: {str(e)}")


# Раздача frontend
app.mount("/static", StaticFiles(directory="src/frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("src/frontend/index.html")