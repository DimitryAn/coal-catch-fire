from sqlalchemy import Column, Integer, String, Float, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()

# Модели
class Temperature(Base):
    __tablename__ = "temperatures_data"
    id = Column(Integer, primary_key=True, index=True)
    sklad = Column("Склад", String)
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
    coal_mark = Column("Наим. ЕТСНГ", String)  # ← без точки — как в вашем CSV
    stabel = Column("Штабель", String, nullable=False)
    load_time = Column("ПогрузкаНаСудно", String)
    to_sklad_ton = Column("На склад, тн", Numeric)
    to_ship_ton = Column("На судно, тн", Numeric)
    sklad = Column("Склад", String)

class Weather(Base):
    __tablename__ = "weather_data"
    id = Column(Integer, primary_key=True, index=True)
    date = Column("date", String)  # ← новое поле
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

class Fire(Base):
    __tablename__ = "fires_data"
    id = Column(Integer, primary_key=True, index=True)
    date_created = Column("Дата составления", String)
    cargo = Column("Груз", String)
    weight_act = Column("Вес по акту, тн", Numeric)
    sklad = Column("Склад", String)
    date_start = Column("Дата начала", String)
    date_end = Column("Дата оконч.", String)
    date_formed = Column("Нач.форм.штабеля", String)
    stabel = Column("Штабель", String, nullable=False)



