import pandas as pd
from river import compose, linear_model, preprocessing, metrics, stream
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import time

# --- Настройки ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Taxi Demand Strategy Comparison")

MODEL_NAME = "Demand-Forecaster-Online"
DEPLOY_EVERY_N_EVENTS = 5000 # Деплоим новую модель каждые 5000 событий

def train_online_model():
    """
    Симулирует онлайн-обучение.
    1. Создает "легкую" онлайн-модель с помощью River.
    2. Симулирует поток данных, итерируясь по датасету.
    3. На каждом шаге делает предсказание, оценивает ошибку и обновляет модель.
    4. Периодически сохраняет текущее состояние модели в MLflow и переводит в Staging.
    """
    print("--- Запуск онлайн-обучения (Online Training) ---")
    client = MlflowClient()

    # --- 1. Создание онлайн-модели ---
    # Простая модель: стандартизация фичей -> линейная регрессия с оптимизатором SGD
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LinearRegression(intercept_lr=0.1)
    )
    mae_tracker = metrics.MAE()

    # --- 2. Симуляция потока данных ---
    # Загружаем данные и сортируем их по времени, чтобы симулировать реальный поток
    data_stream_df = pd.read_parquet('../3/monitoring/data/reference_data.parquet').sort_values('pickup_hour')
    
    # Для онлайн-обучения используем только простые фичи, доступные в момент события
    # В реальной системе более сложные фичи (лаги) приходили бы из онлайн-feature store
    features_to_use = ['hour', 'dayofweek']
    target_to_use = 'trip_count'

    print(f"Симуляция потока из {len(data_stream_df)} событий...")

    with mlflow.start_run(run_name="Online Training Run") as run:
        for i, row in data_stream_df.iterrows():
            features = {k: row[k] for k in features_to_use}
            true_value = row[target_to_use]
            
            # --- 3. Предсказание, оценка и обучение ---
            # Предсказываем ДО обучения
            prediction = model.predict_one(features)
            
            # Обновляем метрику ошибки
            mae_tracker.update(true_value, prediction)
            
            # Обучаем модель на новом примере
            model.learn_one(features, true_value)
            
            # Логируем метрику в MLflow (можно делать реже, чтобы не спамить)
            if (i + 1) % 1000 == 0:
                mlflow.log_metric("rolling_mae", mae_tracker.get(), step=i+1)

            # --- 4. Периодический деплой ---
            if (i + 1) % DEPLOY_EVERY_N_EVENTS == 0:
                print(f"\nСобытие #{i+1}: Деплой новой версии онлайн-модели...")
                
                # River-модели можно сохранить как pyfunc
                # Важно сохранить не только модель, но и трекер метрик
                model_artifact = {"model": model, "mae_tracker": mae_tracker}

                mlflow.pyfunc.log_model(
                    artifact_path=f"model_step_{i+1}",
                    python_model=RiverWrapper(model_artifact),
                    registered_model_name=MODEL_NAME
                )
                
                # Переводим в Staging
                latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=latest_version.version,
                    stage="Staging",
                    archive_existing_versions=True
                )
                print(f"Модель версии {latest_version.version} переведена в 'Staging'. Текущая MAE: {mae_tracker.get():.2f}")

# Обертка для сохранения River-модели в формате MLflow PyFunc
class RiverWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_artifact):
        self.model = model_artifact["model"]

    def predict(self, context, model_input):
        dict_records = model_input.to_dict(orient='records')
        return [self.model.predict_one(features) for features in dict_records]


if __name__ == "__main__":
    train_online_model()