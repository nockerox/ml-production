import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import numpy as np

# --- Настройки ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Taxi Demand Strategy Comparison")

BATCH_MODEL_NAME = "Demand-Forecaster-Batch"
ONLINE_MODEL_NAME = "Demand-Forecaster-Online"

def compare_models():
    """
    Сравнивает производительность и задержку пакетной и онлайн-моделей.
    1. Загружает последнюю Production-модель (Batch) и Staging-модель (Online).
    2. Готовит тестовые данные, которые модели не видели.
    3. Оценивает точность (MAE, RMSE).
    4. Оценивает задержку инференса.
    5. Логирует результаты сравнения в MLflow.
    """
    print("--- Сравнение пакетной и онлайн-стратегий ---")
    client = MlflowClient()

    # --- 1. Загрузка моделей ---
    try:
        # Пакетная модель обычно самая стабильная, берем из Production
        batch_model_info = client.get_latest_versions(BATCH_MODEL_NAME, stages=["Production"])[0]
        batch_model_uri = f"models:/{batch_model_info.name}/{batch_model_info.version}"
        batch_model = mlflow.pyfunc.load_model(batch_model_uri)
        print(f"Загружена Batch-модель: версия {batch_model_info.version}")

        # Онлайн-модель постоянно обновляется, берем последнюю из Staging
        online_model_info = client.get_latest_versions(ONLINE_MODEL_NAME, stages=["Staging"])[0]
        online_model_uri = f"models:/{online_model_info.name}/{online_model_info.version}"
        online_model = mlflow.pyfunc.load_model(online_model_uri)
        print(f"Загружена Online-модель: версия {online_model_info.version}")
    except IndexError as e:
        print(f"Ошибка: Не удалось загрузить одну из моделей. Убедитесь, что скрипты обучения были запущены. {e}")
        return

    # --- 2. Подготовка тестовых данных ---
    # Используем данные, которые не участвовали в обучении ни одной из моделей
    test_df = pd.read_parquet('../3/monitoring/data/current_data.parquet')
    
    # Фичи для пакетной модели
    X_test_batch = test_df.drop(columns=['pickup_hour', 'PULocationID', 'trip_count'])
    
    # Фичи для онлайн-модели (только простые)
    X_test_online = test_df[['hour', 'dayofweek']]
    
    y_test = test_df['trip_count']

    # --- 3. Оценка точности ---
    print("\nОценка точности моделей...")
    batch_preds = batch_model.predict(X_test_batch)
    online_preds = online_model.predict(X_test_online)

    batch_metrics = {
        "mae": mean_absolute_error(y_test, batch_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, batch_preds))
    }
    online_metrics = {
        "mae": mean_absolute_error(y_test, online_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, online_preds))
    }
    
    print(f"Batch Model -> MAE: {batch_metrics['mae']:.2f}, RMSE: {batch_metrics['rmse']:.2f}")
    print(f"Online Model -> MAE: {online_metrics['mae']:.2f}, RMSE: {online_metrics['rmse']:.2f}")

    # --- 4. Оценка задержки (Latency) ---
    print("\nОценка задержки инференса...")
    n_samples = 1000
    
    start_time = time.time()
    for i in range(n_samples):
        batch_model.predict(X_test_batch.head(1))
    batch_latency = (time.time() - start_time) * 1000 / n_samples # в миллисекундах на 1 предсказание
    
    start_time = time.time()
    for i in range(n_samples):
        online_model.predict(X_test_online.head(1))
    online_latency = (time.time() - start_time) * 1000 / n_samples
    
    print(f"Batch Model -> Latency: {batch_latency:.4f} ms/предсказание")
    print(f"Online Model -> Latency: {online_latency:.4f} ms/предсказание")
    
    # --- 5. Логирование результатов ---
    with mlflow.start_run(run_name="Strategy Comparison"):
        mlflow.log_metrics({f"batch_{k}": v for k, v in batch_metrics.items()})
        mlflow.log_metrics({f"online_{k}": v for k, v in online_metrics.items()})
        mlflow.log_metric("batch_latency_ms", batch_latency)
        mlflow.log_metric("online_latency_ms", online_latency)
        print("\nРезультаты сравнения залогированы в MLflow.")

if __name__ == "__main__":
    compare_models()