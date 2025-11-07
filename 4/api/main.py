from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from feast import FeatureStore
import os
from datetime import datetime
import random
import logging

# --- Настройка логирования для Shadow Mode ---
# Будем записывать сравнения моделей в файл
logging.basicConfig(
    filename='shadow_mode_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# --- Настройка ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "../2/nyc_taxi_demand/feature_repo")

# --- Параметры безопасного деплоя ---
CANARY_TRAFFIC_PERCENT = 5  # 5% трафика на новую модель

app = FastAPI(title="Taxi Demand Forecasting API")

# --- Глобальные объекты ---
try:
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1. Загружаем Production модель
    prod_model_info = client.get_latest_versions("LGBM-Demand-Forecaster", stages=["Production"])[0]
    prod_model_uri = f"models:/LGBM-Demand-Forecaster/{prod_model_info.version}"
    prod_model = mlflow.pyfunc.load_model(prod_model_uri)
    PRODUCTION_MODEL_VERSION = prod_model_info.version
    
    # 2. Загружаем Staging (Canary) модель
    staging_model = None
    STAGING_MODEL_VERSION = None
    try:
        staging_model_info = client.get_latest_versions("LGBM-Demand-Forecaster", stages=["Staging"])[0]
        staging_model_uri = f"models:/LGBM-Demand-Forecaster/{staging_model_info.version}"
        staging_model = mlflow.pyfunc.load_model(staging_model_uri)
        STAGING_MODEL_VERSION = staging_model_info.version
        print(f"Successfully loaded Production model v{PRODUCTION_MODEL_VERSION} and Staging model v{STAGING_MODEL_VERSION}")
    except IndexError:
        print(f"Successfully loaded Production model v{PRODUCTION_MODEL_VERSION}. No model found in Staging.")

    MODEL_INPUT_COLUMNS = prod_model.metadata.get_input_schema().input_names()

except Exception as e:
    raise RuntimeError(f"Failed to initialize models or feature store: {e}")

class DemandRequest(BaseModel):
    pulocationid: int

class DemandResponse(BaseModel):
    pulocationid: int
    predicted_demand: float
    model_version: str

@app.post("/predict", response_model=DemandResponse)
def predict(request: DemandRequest):
    try:
        # --- 1. Получение фичей (один раз для обеих моделей) ---
        prediction_timestamp = datetime(2019, 2, 3, 19, 0, 0)
        entity_df = pd.DataFrame.from_records([{"PULocationID": request.pulocationid, "event_timestamp": prediction_timestamp}])
        features_df = store.get_historical_features(entity_df=entity_df, features=[f"taxi_stats:{col}" for col in MODEL_INPUT_COLUMNS]).to_df()
        
        if features_df.empty:
            raise HTTPException(status_code=404, detail="Features not found.")
            
        features_df = features_df[MODEL_INPUT_COLUMNS]

        # --- 2. Логика Canary Release и Shadow Mode ---
        
        # Определяем, какую модель использовать для ответа пользователю
        use_canary = (staging_model is not None) and (random.random() < CANARY_TRAFFIC_PERCENT / 100)

        if use_canary:
            # --- CANARY PATH (5% трафика) ---
            # Новая модель обрабатывает запрос и ее результат возвращается пользователю
            prediction = staging_model.predict(features_df)[0]
            model_version_for_response = STAGING_MODEL_VERSION
        else:
            # --- PRODUCTION PATH (95% трафика) ---
            # Старая модель обрабатывает запрос
            prediction = prod_model.predict(features_df)[0]
            model_version_for_response = PRODUCTION_MODEL_VERSION

        # --- SHADOW MODE LOGIC ---
        # Если есть Staging модель, втихую прогоняем данные и через нее, чтобы сравнить результаты
        if staging_model is not None:
            # Получаем предсказания от обеих моделей (одно уже есть)
            prod_prediction = prod_model.predict(features_df)[0] if use_canary else prediction
            staging_prediction = prediction if use_canary else staging_model.predict(features_df)[0]
            
            # Логируем разницу для последующего анализа
            prediction_diff = prod_prediction - staging_prediction
            log_message = (
                f"PULocationID: {request.pulocationid}, "
                f"Prod_v{PRODUCTION_MODEL_VERSION}: {prod_prediction:.2f}, "
                f"Staging_v{STAGING_MODEL_VERSION}: {staging_prediction:.2f}, "
                f"Difference: {prediction_diff:.2f}"
            )
            logging.info(log_message)

        return {
            "pulocationid": request.pulocationid,
            "predicted_demand": round(float(prediction), 2),
            "model_version": model_version_for_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")