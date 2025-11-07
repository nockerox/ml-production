import os
import requests
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import datetime
import json
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

# --- 1. Настройки ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Taxi Demand Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Пороги
MODEL_PERFORMANCE_DEGRADATION_MAE_THRESHOLD = 1.2 
RETRAIN_PERFORMANCE_DEGRADATION_MAE_THRESHOLD = 1.3

# --- 2. Функции для алертинга ---
def send_alert(message: str, is_critical: bool = False):
    prefix = "🚨 *Критический алерт* 🚨" if is_critical else "⚠️ *Предупреждение* ⚠️"
    full_message = f"{prefix}\n{message}"
    
    print(full_message)
    
    if not SLACK_WEBHOOK_URL:
        print("Переменная SLACK_WEBHOOK_URL не установлена. Алерт не отправлен в Slack.")
        return
        
    try:
        payload = {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": full_message}}]}
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        print("Алерт успешно отправлен в Slack.")
    except Exception as e:
        print(f"Ошибка при отправке алерта в Slack: {e}")

# --- 3. Функции мониторинга (адаптированные под API с Snapshot) ---
def monitor_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """Генерирует отчет о дрифте данных, используя точную структуру JSON."""
    print("\n--- Запуск мониторинга дрифта данных ---")
    
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    
    report_dict = json.loads(snapshot.json())

    try:
        drift_metric_value = report_dict['metrics'][0]['value']
        num_drifted_columns = int(drift_metric_value['count'])
        dataset_drift_detected = num_drifted_columns > 0
        
    except (KeyError, IndexError, TypeError) as e:
        print(f"Ошибка при извлечении результатов дрифта из отчета: {e}")
        return {"dataset_drift": False, "drifted_columns": 0}

    with mlflow.start_run(run_name="Data Drift Report"):
        snapshot.save_html("data_drift_report.html")
        mlflow.log_artifact("data_drift_report.html", "reports")
        mlflow.log_dict(report_dict, "data_drift_report.json")
        
        mlflow.log_metric("num_drifted_columns", num_drifted_columns)
        mlflow.log_metric("dataset_drift", int(dataset_drift_detected))

    print(f"Обнаружен дрифт в {num_drifted_columns} колонках.")
    if dataset_drift_detected:
        send_alert(f"Обнаружен общий дрифт данных! Количество смещенных колонок: {num_drifted_columns}.")

    return {"dataset_drift": dataset_drift_detected, "drifted_columns": num_drifted_columns}


def monitor_model_performance(model, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """Генерирует отчет о производительности модели (с явным указанием задачи)."""
    print("\n--- Запуск мониторинга производительности модели ---")

    ref_data_copy = reference_data.copy()
    curr_data_copy = current_data.copy()

    ref_predictions = model.predict(ref_data_copy.drop('trip_count', axis=1))
    curr_predictions = model.predict(curr_data_copy.drop('trip_count', axis=1))
    
    ref_data_copy['target'] = ref_data_copy['trip_count']
    curr_data_copy['target'] = curr_data_copy['trip_count']
    ref_data_copy.drop('trip_count', axis=1, inplace=True)
    curr_data_copy.drop('trip_count', axis=1, inplace=True)

    ref_data_copy['prediction'] = ref_predictions
    curr_data_copy['prediction'] = curr_predictions
    
    from evidently.legacy.pipeline.column_mapping import TaskType

    column_mapping = ColumnMapping(
        target='target', 
        prediction='prediction',
        task=TaskType.REGRESSION_TASK
    )
    
    report = Report(metrics=[RegressionPreset()])
    
    try:
        snapshot = report.run(reference_data=ref_data_copy, 
                              current_data=curr_data_copy,
                              column_mapping=column_mapping)
    except TypeError as e:
         print(f"Не удалось передать column_mapping в run. Ошибка: {e}")
         return {"reference_mae": -1, "current_mae": -1}

    report_dict = json.loads(snapshot.json())

    try:
        quality_widget_results = report_dict['widgets'][1]['results']
        ref_mae = quality_widget_results['reference']['mean_abs_error']
        curr_mae = quality_widget_results['current']['mean_abs_error']
    except (KeyError, IndexError) as e:
        print(f"Ошибка при извлечении метрик качества из отчета: {e}")
        print("--- СТРУКТУРА JSON ОТЧЕТА О КАЧЕСТВЕ ---")
        print(json.dumps(report_dict, indent=4))
        return {"reference_mae": -1, "current_mae": -1}

    with mlflow.start_run(run_name="Model Performance Report"):
        snapshot.save_html("model_performance_report.html")
        mlflow.log_artifact("model_performance_report.html", "reports")
        mlflow.log_dict(report_dict, "model_performance_report.json")
        mlflow.log_metrics({"reference_mae": ref_mae, "current_mae": curr_mae})

    print(f"Reference MAE: {ref_mae:.2f}, Current MAE: {curr_mae:.2f}")

    if ref_mae > 0 and curr_mae > ref_mae * MODEL_PERFORMANCE_DEGRADATION_MAE_THRESHOLD:
        degradation = ((curr_mae / ref_mae) - 1) * 100
        send_alert(f"Обнаружена деградация модели! MAE увеличился на {degradation:.2f}% (с {ref_mae:.2f} до {curr_mae:.2f}).")
        
    return {"reference_mae": ref_mae, "current_mae": curr_mae}

# --- 4. Логика ретрейна ---
def retrain_model():
    send_alert("Запущена процедура автоматического переобучения модели.", is_critical=True)

def check_and_run_retrain(data_drift_info: dict, model_performance_info: dict):
    print("\n--- Проверка необходимости переобучения ---")
    retrain_needed = False
    reason = ""

    if datetime.date.today().weekday() == 0:
        retrain_needed = True
        reason = "Плановое еженедельное переобучение."

    if data_drift_info.get('dataset_drift', False) and not retrain_needed:
        retrain_needed = True
        reason = "Критический дрифт данных."

    ref_mae = model_performance_info.get('reference_mae', -1)
    curr_mae = model_performance_info.get('current_mae', -1)
    if ref_mae > 0 and curr_mae > ref_mae * RETRAIN_PERFORMANCE_DEGRADATION_MAE_THRESHOLD and not retrain_needed:
        retrain_needed = True
        reason = "Критическая деградация производительности модели."

    if retrain_needed:
        print(f"Принято решение о переобучении. Причина: {reason}")
        retrain_model()
    else:
        print("Переобучение не требуется.")

# --- 5. Основной пайплайн ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_data_path = os.path.join(base_dir, '..', '3', 'monitoring', 'data', 'reference_data.parquet')
    curr_data_path = os.path.join(base_dir, '..', '3', 'monitoring', 'data', 'current_data.parquet')

    try:
        ref_data = pd.read_parquet(ref_data_path)
        curr_data = pd.read_parquet(curr_data_path)
    except FileNotFoundError:
        print(f"Ошибка: Файлы для мониторинга не найдены по путям:\n{ref_data_path}\n{curr_data_path}")
        exit()

    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions("LGBM-Demand-Forecaster", stages=["Production"])
        if not latest_versions:
            raise IndexError("No model versions found in Production stage.")
        prod_model_info = latest_versions[0]
        model_uri = f"models:/{prod_model_info.name}/{prod_model_info.version}"
        production_model = mlflow.pyfunc.load_model(model_uri)
    except IndexError as e:
        print(f"Ошибка: {e}. Пожалуйста, убедитесь, что хотя бы одна версия модели имеет стейдж 'Production'.")
        exit()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        exit()

    data_drift_results = monitor_data_drift(ref_data.drop('trip_count', axis=1), curr_data.drop('trip_count', axis=1))
    model_performance_results = monitor_model_performance(production_model, ref_data, curr_data)
    
    check_and_run_retrain(data_drift_results, model_performance_results)
