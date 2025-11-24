import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ===
RANDOM_STATE = 42
HORIZON_DAYS = 3  # горизонт прогноза

FEATURE_COLS_TO_FORECAST = [
    'max_temp_stack', 'in_tons', 'out_tons', 'net_tons',
    't_mean', 'p_mean', 'humidity_mean', 'precip_sum',
    'wind_avg', 'wind_max', 'cloudcover_mean'
]

CLASS_FEATURES = ['Склад', 'Штабель'] + FEATURE_COLS_TO_FORECAST

# === ПРЕДОБРАБОТКА ДАННЫХ ===
def preprocess_weather(df):
    """Дневная агрегация погодных данных"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    agg = df.groupby('date').agg({
        't': 'mean', 'p': 'mean', 'humidity': 'mean', 'precipitation': 'sum',
        'v_avg': 'mean', 'v_max': 'mean', 'cloudcover': 'mean'
    }).reset_index()
    agg.rename(columns={
        't': 't_mean', 'p': 'p_mean', 'humidity': 'humidity_mean',
        'precipitation': 'precip_sum', 'v_avg': 'wind_avg',
        'v_max': 'wind_max', 'cloudcover': 'cloudcover_mean'
    }, inplace=True)
    num_cols = agg.select_dtypes(include=[np.number]).columns
    agg[num_cols] = agg[num_cols].fillna(agg[num_cols].median())
    agg['date'] = pd.to_datetime(agg['date'])
    return agg

def preprocess_supplies(df):
    """Дневные погрузки и выгрузки по складу/штабелю."""
    df = df.copy()
    df['ВыгрузкаНаСклад'] = pd.to_datetime(df['ВыгрузкаНаСклад'])
    df['ПогрузкаНаСудно'] = pd.to_datetime(df['ПогрузкаНаСудно'])
    in_df = df.groupby(['Склад', 'Штабель', df['ВыгрузкаНаСклад'].dt.date])['На склад, тн'].sum().reset_index()
    in_df.rename(columns={'ВыгрузкаНаСклад': 'date', 'На склад, тн': 'in_tons'}, inplace=True)
    out_df = df.groupby(['Склад', 'Штабель', df['ПогрузкаНаСудно'].dt.date])['На судно, тн'].sum().reset_index()
    out_df.rename(columns={'ПогрузкаНаСудно': 'date', 'На судно, тн': 'out_tons'}, inplace=True)
    base = pd.merge(in_df, out_df, on=['Склад', 'Штабель', 'date'], how='outer')
    base['in_tons'] = base['in_tons'].fillna(0)
    base['out_tons'] = base['out_tons'].fillna(0)
    base['net_tons'] = base['in_tons'] - base['out_tons']
    base['date'] = pd.to_datetime(base['date'])
    return base

def preprocess_temperature(df):
    """Максимум температуры по дню и штабелю."""
    df = df.copy()
    df['Дата акта'] = pd.to_datetime(df['Дата акта'])
    df['date'] = df['Дата акта'].dt.date
    tmp = df.groupby(['Склад', 'Штабель', 'date'])['Максимальная температура'].max().reset_index()
    tmp.rename(columns={'Максимальная температура': 'max_temp_stack'}, inplace=True)
    tmp['date'] = pd.to_datetime(tmp['date'])
    return tmp

def preprocess_fires(df):
    """Дата возгорания (метка) и дата формирования штабеля."""
    df = df.copy()
    df['Дата начала'] = pd.to_datetime(df['Дата начала'])
    df['Дата оконч.'] = pd.to_datetime(df['Дата оконч.'])
    df['Нач.форм.штабеля'] = pd.to_datetime(df['Нач.форм.штабеля'])
    fires_daily = df.groupby(['Склад', 'Штабель', df['Дата начала'].dt.date]).size().reset_index(name='fire')
    fires_daily['date'] = pd.to_datetime(fires_daily['Дата начала'])
    fires_daily.drop(columns=['Дата начала'], inplace=True)
    fires_daily['fire'] = 1
    formation = df.groupby(['Склад', 'Штабель'])['Нач.форм.штабеля'].min().reset_index()
    formation.rename(columns={'Нач.форм.штабеля': 'formation_date'}, inplace=True)
    return fires_daily, formation

# === ОБУЧЕНИЕ ПРИЗНАКОВЫХ МОДЕЛЕЙ ===
def train_feature_models(df, feature_cols=FEATURE_COLS_TO_FORECAST, horizon_days=HORIZON_DAYS):
    """Обучаем модели для предсказания признаки(t + d) для d = 1...horizon_days."""
    models = {}
    df = df.copy()
    df = df.sort_values(['Склад', 'Штабель', 'date'])
    group = df.groupby(['Склад', 'Штабель'])
    for d in range(1, horizon_days + 1):
        shifted = df.copy()
        for col in feature_cols:
            shifted[f'{col}_target'] = group[col].shift(-d)
        target_cols = [f'{c}_target' for c in feature_cols]
        train_rows = shifted.dropna(subset=target_cols)
        if train_rows.empty:
            continue
        X = train_rows[feature_cols]
        Y = train_rows[target_cols]
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))
        model.fit(X, Y)
        models[d] = model
        print(f'Обучена модель прогнозирования признаков для горизонта d={d}')
    return models

def forecast_features_for_horizon(feature_models, base_features, feature_cols=FEATURE_COLS_TO_FORECAST, horizon_days=HORIZON_DAYS):
    """На вход: признаки на день запроса прогноза (по штабелям), на выход: предсказанные признаки на каждый день горизонта."""
    results = []
    for d in range(1, horizon_days + 1):
        model = feature_models.get(d)
        X = base_features[feature_cols]
        Y_pred = model.predict(X)
        pred_df = base_features[['Склад', 'Штабель', 'date']].copy()
        for i, col in enumerate(feature_cols):
            pred_df[col] = Y_pred[:, i]
        pred_df['delta_day'] = d
        pred_df['target_date'] = pred_df['date'] + pd.to_timedelta(d, unit='D')
        results.append(pred_df)
    if results:
        all_preds = pd.concat(results, ignore_index=True)
    else:
        all_preds = pd.DataFrame()
    return all_preds

# === МЕТРИКИ ===
def compute_fire_event_metrics(df, window_days=2):
    """Precision/Recall/F1 с допуском ±window_days по дате."""
    true_events = df[df['fire_true'] == 1][['Склад', 'Штабель', 'date']].drop_duplicates().reset_index(drop=True)
    pred_events = df[df['fire_pred'] == 1][['Склад', 'Штабель', 'date']].drop_duplicates().reset_index(drop=True)
    true_list = list(true_events.itertuples(index=False, name=None))
    pred_list = list(pred_events.itertuples(index=False, name=None))
    matched_true = set()
    matched_pred = set()
    for i, (sk_t, stack_t, date_t) in enumerate(true_list):
        for j, (sk_p, stack_p, date_p) in enumerate(pred_list):
            if j in matched_pred:
                continue
            if (sk_t == sk_p) and (stack_t == stack_p):
                if abs((date_p - date_t).days) <= window_days:
                    matched_true.add(i)
                    matched_pred.add(j)
                    break
    TP_fire = len(matched_true)
    FN_fire = len(true_list) - TP_fire
    FP_fire = len(pred_list) - TP_fire
    precision_fire = TP_fire / (TP_fire + FP_fire) if (TP_fire + FP_fire) > 0 else 0.0
    recall_fire = TP_fire / (TP_fire + FN_fire) if (TP_fire + FN_fire) > 0 else 0.0
    f1_fire = 2 * precision_fire * recall_fire / (precision_fire + recall_fire) if (precision_fire + recall_fire) > 0 else 0.0
    return {
        'TP_fire': TP_fire, 'FP_fire': FP_fire, 'FN_fire': FN_fire,
        'precision_fire': precision_fire, 'recall_fire': recall_fire, 'f1_fire': f1_fire,
        'true_events': len(true_list), 'pred_events': len(pred_list)
    }

# === ПОСТРОЕНИЕ base НОВЫХ ДАННЫХ ===
def build_base_from_raw_without_fires(supplies_raw, temp_raw, weather_raw):
    """Строим дневной датафрейм base для нового периода."""
    weather_daily = preprocess_weather(weather_raw)
    supplies_daily = preprocess_supplies(supplies_raw)
    temp_daily = preprocess_temperature(temp_raw)
    formation_src = pd.concat([
        supplies_daily[['Склад', 'Штабель', 'date']],
        temp_daily[['Склад', 'Штабель', 'date']]
    ], ignore_index=True)
    formation_df = (formation_src.dropna(subset=['Склад', 'Штабель', 'date'])
                    .groupby(['Склад', 'Штабель'])['date'].min()
                    .reset_index().rename(columns={'date': 'formation_date'}))
    base_new = pd.merge(supplies_daily, temp_daily, on=['Склад', 'Штабель', 'date'], how='outer')
    base_new = pd.merge(base_new, weather_daily, on='date', how='left')
    base_new = pd.merge(base_new, formation_df, on=['Склад', 'Штабель'], how='left')
    base_new['formation_date'] = pd.to_datetime(base_new['formation_date'])
    base_new['date'] = pd.to_datetime(base_new['date'])
    base_new['age_days'] = (base_new['date'] - base_new['formation_date']).dt.days
    base_new['age_days'] = base_new['age_days'].fillna(0)
    base_new['fire'] = 0
    num_cols = ['in_tons', 'out_tons', 'net_tons', 'max_temp_stack', 't_mean', 'p_mean',
                'humidity_mean', 'precip_sum', 'wind_avg', 'wind_max', 'cloudcover_mean', 'age_days']
    for col in num_cols:
        if col in base_new.columns:
            base_new[col] = base_new[col].fillna(0)
    base_new = base_new.dropna(subset=['Склад', 'Штабель', 'date'])
    base_new['Склад'] = base_new['Склад'].astype(int)
    base_new['Штабель'] = base_new['Штабель'].astype(int)
    return base_new

def predict_from_raw_without_fires(models_dict, supplies_raw, temp_raw, weather_raw, horizon_days=HORIZON_DAYS, forecast_date=""):
    """Функция для web: возвращает прогнозы."""
    base_new = build_base_from_raw_without_fires(supplies_raw, temp_raw, weather_raw)
    if forecast_date is None:
        forecast_date = base_new['date'].max()
    forecast_date = pd.to_datetime(forecast_date)
    snapshot = base_new[base_new['date'] == forecast_date].copy()
    needed_cols = ['Склад', 'Штабель', 'date'] + FEATURE_COLS_TO_FORECAST
    snapshot = snapshot[needed_cols]
    feature_models = models_dict['feature_models']
    feature_preds = forecast_features_for_horizon(feature_models, snapshot, models_dict['feature_cols'], horizon_days)
    cls_df = feature_preds.copy()
    cls_df['Склад'] = cls_df['Склад'].astype(int)
    cls_df['Штабель'] = cls_df['Штабель'].astype(int)
    X_cls = cls_df[models_dict['class_features']]
    clf = models_dict['clf']
    proba = clf.predict_proba(X_cls)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cls_df['fire_pred'] = pred
    return cls_df[['Склад', 'Штабель', 'target_date', 'fire_pred']]

def evaluate_predictions_with_new_fires(preds_df, fires_new_raw, window_days=2):
    fires_daily_new, _ = preprocess_fires(fires_new_raw)
    fires_daily_new = fires_daily_new[['Склад', 'Штабель', 'date', 'fire']].copy()
    fires_daily_new.rename(columns={'fire': 'fire_true'}, inplace=True)
    day_df = preds_df.copy()
    day_df = pd.merge(day_df, fires_daily_new, left_on=['Склад', 'Штабель', 'target_date'], right_on=['Склад', 'Штабель', 'date'], how='left')
    day_df['fire_true'] = day_df['fire_true'].fillna(0).astype(int)
    day_df.drop(columns=['date'], inplace=True)
    fire_df = day_df[['Склад', 'Штабель', 'target_date', 'fire_true', 'fire_pred']].copy()
    fire_df.rename(columns={'target_date': 'date'}, inplace=True)
    return compute_fire_event_metrics(fire_df, window_days)

def build_labeled_base_for_new_period(base_new, fires_new_raw):
    fires_daily_new, _ = preprocess_fires(fires_new_raw)
    fires_daily_new = fires_daily_new[['Склад', 'Штабель', 'date', 'fire']].copy()
    labeled = pd.merge(base_new, fires_daily_new, on=['Склад', 'Штабель', 'date'], how='left', suffixes=('', '_new'))
    if 'fire_new' in labeled.columns:
        labeled['fire'] = labeled['fire_new'].fillna(labeled['fire']).fillna(0)
        labeled.drop(columns=['fire_new'], inplace=True)
    else:
        labeled['fire'] = labeled['fire'].fillna(0)
    labeled['fire'] = labeled['fire'].astype(int)
    return labeled

def finetune_with_new_labels(models_dict, df_historical, df_new_labeled, horizon_days=HORIZON_DAYS):
    df_all = pd.concat([df_historical, df_new_labeled], ignore_index=True)
    feature_models = train_feature_models(df_all, horizon_days=horizon_days)
    df_model = df_all.dropna(subset=CLASS_FEATURES + ['fire']).copy()
    X = df_model[CLASS_FEATURES]
    y = df_model['fire']
    clf = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    clf.fit(X, y)
    return {
        'feature_models': feature_models,
        'clf': clf,
        'feature_cols': FEATURE_COLS_TO_FORECAST,
        'class_features': CLASS_FEATURES
    }