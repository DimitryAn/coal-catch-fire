from sqlalchemy import Column, Integer, String, Float, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()

DATABASE_URL = "postgresql://dev:chakaton@localhost:5432/coal_fire_db"
engine = create_engine(DATABASE_URL)

# Модели
class Temperature(Base):
    __tablename__ = "temperatures_data"
    id = Column(Integer, primary_key=True, index=True)
    stabel = Column("Штабель", String, nullable=False)
    marka = Column("Марка", String)
    max_temperature = Column("Максимальная температура", Float)
    piket = Column("Пикет", String)
    date_act = Column("Дата акта", String)
    smena = Column("Смена", String)

class Supply(Base):
    __tablename__ = "supplies_data"
    id = Column(Integer, primary_key=True, index=True)
    unload_time = Column("ВыгрузкаНаСклад", String)
    coal_mark = Column("Наим ЕТСНГ", String)  # ← без точки — как в вашем CSV
    stabel = Column("Штабель", String, nullable=False)
    load_time = Column("ПогрузкаНаСудно", String)
    to_sklad_ton = Column("На склад тн", Numeric)
    to_ship_ton = Column("На судно тн", Numeric)
    sklad = Column("Склад", String)

class Weather(Base):
    __tablename__ = "weather_data"
    id = Column(Integer, primary_key=True, index=True)
    temp_c = Column("t", Float)
    pressure = Column("p", Float)
    humidity = Column("humidity", Float)
    precipitation = Column("precipitation", Float)
    wind_dir = Column("wind_dir", Float)
    wind_avg = Column("v_avg", Float)
    wind_max = Column("v_max", Float)
    cloudcover = Column("cloudcover", Float)
    visibility = Column("visibility", Float)
    weather_code = Column("weather_code", Integer)

# Создаём таблицы
def create_tables():
    Base.metadata.create_all(bind=engine)


DATABASE_URL = "postgresql://dev:chakaton@localhost:5432/coal_fire_db"
engine = create_engine(DATABASE_URL)
def create_tables():
    Base.metadata.create_all(bind=engine)
