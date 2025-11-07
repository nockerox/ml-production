import requests
import time
import numpy as np
import asyncio
import aiohttp
from tqdm import tqdm

# --- Настройки ---
API_URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 1000  # Общее количество запросов
CONCURRENCY = 50     # Количество одновременных запросов

# --- Глобальные переменные для сбора статистики ---
latencies = []
errors = 0

async def send_request(session, pbar):
    """Асинхронно отправляет один запрос и записывает результат."""
    global errors
    
    # Генерируем случайный ID локации для каждого запроса
    random_location_id = np.random.randint(1, 260)
    payload = {"pulocationid": random_location_id}
    
    start_time = time.perf_counter()
    try:
        async with session.post(API_URL, json=payload, timeout=10) as response:
            if response.status == 200:
                # Успешный запрос, добавляем задержку в список
                latency = (time.perf_counter() - start_time) * 1000  # в миллисекундах
                latencies.append(latency)
            else:
                # Ошибка на стороне сервера (500, 404 и т.д.)
                errors += 1
    except (aiohttp.ClientError, asyncio.TimeoutError):
        # Ошибка сети или таймаут
        errors += 1
    finally:
        # Обновляем прогресс-бар после каждого запроса
        pbar.update(1)

async def main():
    """Основная функция для запуска асинхронного нагрузочного теста."""
    print(f"Starting load test with {NUM_REQUESTS} requests and concurrency of {CONCURRENCY}...")
    
    # Используем tqdm для визуализации прогресса
    with tqdm(total=NUM_REQUESTS) as pbar:
        async with aiohttp.ClientSession() as session:
            # Создаем задачи для всех запросов
            tasks = [send_request(session, pbar) for _ in range(NUM_REQUESTS)]
            # Запускаем задачи с определенным уровнем параллелизма
            semaphore = asyncio.Semaphore(CONCURRENCY)

            async def run_with_semaphore(task):
                async with semaphore:
                    await task
            
            await asyncio.gather(*[run_with_semaphore(task) for task in tasks])

def print_results(total_time):
    """Выводит результаты тестирования."""
    print("\n--- Load Test Results ---")
    if not latencies:
        print("No successful requests were made.")
        return

    total_requests_made = len(latencies) + errors
    error_rate_percent = (errors / total_requests_made) * 100 if total_requests_made > 0 else 0
    throughput = len(latencies) / total_time if total_time > 0 else 0

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total requests: {total_requests_made}")
    print(f"Successful requests: {len(latencies)}")
    print(f"Failed requests: {errors}")
    print(f"Error Rate: {error_rate_percent:.2f}%")
    print(f"Throughput: {throughput:.2f} req/s")
    
    print("\n--- Latency Percentiles ---")
    print(f"p50 (Median): {np.percentile(latencies, 50):.2f} ms")
    print(f"p90: {np.percentile(latencies, 90):.2f} ms")
    print(f"p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"p99: {np.percentile(latencies, 99):.2f} ms")
    print(f"Max Latency: {np.max(latencies):.2f} ms")

if __name__ == "__main__":
    # Установка библиотек, если они не установлены
    try:
        import aiohttp
        from tqdm import tqdm
    except ImportError:
        print("Required libraries not found. Installing aiohttp and tqdm...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "tqdm"])
        print("Installation complete. Please run the script again.")
        sys.exit(1)

    # Запускаем тест
    start_test_time = time.time()
    asyncio.run(main())
    end_test_time = time.time()
    
    # Выводим результаты
    print_results(total_time=end_test_time - start_test_time)