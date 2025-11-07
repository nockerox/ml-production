import pandas as pd

# Загружаем и агрегируем данные
df = pd.read_parquet('../../assets/taxi/fhvhv_tripdata_2019-02.parquet')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['pickup_hour'] = df['pickup_datetime'].dt.floor('H')
demand_df = df.groupby(['PULocationID', 'pickup_hour']).size().reset_index(name='trip_count')

demand_df['hour'] = demand_df['pickup_hour'].dt.hour
demand_df['dayofweek'] = demand_df['pickup_hour'].dt.dayofweek

# # Создаем дополнительные фичис
demand_df['lag_1h'] = demand_df.groupby('PULocationID')['trip_count'].shift(1)
demand_df['lag_24h'] = demand_df.groupby('PULocationID')['trip_count'].shift(24)
demand_df['lag_168h'] = demand_df.groupby('PULocationID')['trip_count'].shift(168)
demand_df['rolling_mean_24h'] = demand_df.groupby('PULocationID')['trip_count'].shift(1).rolling(window=24).mean()

# # Заполняем пропуски нулями
demand_df.fillna(0, inplace=True)

# Сохраняем в папку data внутри репозитория Feast
demand_df.to_parquet('./nyc_taxi_demand/feature_repo/data/demand_agg_with_ts.parquet')

print("Файл с исходными данными для Feast создан.")