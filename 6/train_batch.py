import pandas as pd
import lightgbm as lgb
import mlflow
from mlflow.tracking import MlflowClient
from feast import FeatureStore
from datetime import datetime, timedelta
import pytz

# --- Настройки ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Taxi Demand Strategy Comparison")

FEAST_REPO_PATH = "../2/nyc_taxi_demand/feature_repo"
MODEL_NAME = "Demand-Forecaster-Batch"

def train_batch_model():
    """
    Выполняет ежедневное пакетное обучение.
    1. Загружает фичи из Feast за последний месяц.
    2. Обучает модель LightGBM.
    3. Логирует модель в MLflow.
    4. Переводит новую модель в 'Staging' для Canary-тестирования.
    """
    print("--- Запуск пакетного обучения (Batch Training) ---")
    
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    client = MlflowClient()

    # --- 1. Получение данных для обучения ---
    utc_tz = pytz.UTC
    end_date = datetime.now(utc_tz)
    start_date = end_date - timedelta(days=30)
    

    try:
        locations_df = pd.read_parquet('../2/nyc_taxi_demand/feature_repo/data/demand_agg_with_ts.parquet')
    except FileNotFoundError:
        print("Ошибка: Файл 'demand_agg_with_ts.parquet' не найден.")
        return

    entity_df = locations_df[
        (locations_df['pickup_hour'] >= start_date) & 
        (locations_df['pickup_hour'] < end_date)
    ][['pickup_hour', 'PULocationID']]

    # Проверка, что после фильтрации остались данные
    if entity_df.empty:
        print("Предупреждение: Не найдено данных за последний 30-дневный период для обучения.")
        return

    print(f"Загрузка обучающих данных с {start_date} по {end_date}...")
    
    features_to_get = [
        "taxi_stats:trip_count", "taxi_stats:lag_1h", "taxi_stats:lag_24h",
        "taxi_stats:lag_168h", "taxi_stats:rolling_mean_24h", "taxi_stats:hour", "taxi_stats:dayofweek"
    ]
    
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_get
    ).to_df().dropna()
    
    if training_df.empty:
        print("Предупреждение: После загрузки из Feast и очистки NaN не осталось данных для обучения.")
        return

    X_train = training_df.drop(columns=['pickup_hour', 'PULocationID', 'trip_count'])
    y_train = training_df['trip_count']

    # --- 2. Обучение модели ---
    print("Обучение модели LightGBM...")
    params = {'objective': 'regression_l1', 'n_estimators': 200, 'learning_rate': 0.05, 'random_state': 42}
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    print("Модель обучена.")

    # --- 3. Логирование в MLflow ---
    with mlflow.start_run(run_name="Batch Training Run") as run:
        mlflow.log_params(params)
        
        # Логируем модель
        input_example = X_train.head(5)
        mlflow.lightgbm.log_model(
            lgbm_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Модель залогирована: {model_uri}")
        
        # --- 4. "Деплой" в Staging (для Canary) ---
        print("Регистрация новой версии и перевод в 'Staging'...")
        try:
            # Ищем последнюю созданную версию
            latest_version_info = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version_info.version,
                stage="Staging",
                archive_existing_versions=True # Архивирует предыдущую модель в Staging
            )
            print(f"Модель версии {latest_version_info.version} переведена в 'Staging'.")
        except IndexError:
            print(f"Не найдено новых версий модели '{MODEL_NAME}' для перевода в Staging.")
        except Exception as e:
            print(f"Ошибка при переводе модели в Staging: {e}")


if __name__ == "__main__":
    train_batch_model()
