# This is an example feature definition file

from datetime import timedelta

import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    Project,
    PushSource,
    RequestSource,
    ValueType
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64


project = Project(name="nyc_taxi_demand", description="A project for driver statistics")

location = Entity(name="PULocationID", value_type=ValueType.INT64, description="Pickup Location ID")

taxi_demand_source = FileSource(
    name="taxi_demand_source",
    path="data/demand_agg_with_ts.parquet",
    timestamp_field="pickup_hour"
)

demand_features_fv = FeatureView(
    name="taxi_stats",
    entities=[location],
    schema=[
        Field(name="trip_count", dtype=Int64),
        Field(name="lag_1h", dtype=Float32),
        Field(name="lag_24h", dtype=Float32),
        Field(name="lag_168h", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="dayofweek", dtype=Int64),
        Field(name="rolling_mean_24h", dtype=Float32),
    ],
    online=True,
    source=taxi_demand_source,
    tags={"team": "demand_forecasting"},
)