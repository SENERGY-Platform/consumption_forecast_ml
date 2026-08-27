import typing


import pandas as pd
import ray

from mlflow.pyfunc import PythonModel

from util import extract_timestamp_and_value, convert_dataset_to_timeseries
from operator_lib.util import logger
from operator_lib.util.helpers import TrainMlflowLogger

import darts
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.prophet_model import Prophet

from build_models import NHitsForecastingModel, ProphetForecastingModel


NHITS_INPUT_CHUNK_LENGTH = 7
NHITS_OUTPUT_CHUNK_LENGTH = 1


def build_model(model_type: str) -> typing.Any:
    if model_type == "nhits":
        return NHiTSModel(
            num_stacks=3,
            num_blocks=2,
            num_layers=1,
            input_chunk_length=NHITS_INPUT_CHUNK_LENGTH,
            output_chunk_length=NHITS_OUTPUT_CHUNK_LENGTH,
        )
    else:
        return Prophet(country_holidays="DE")


def minimum_training_points(model_type: str) -> int:
    if model_type == "nhits":
        return NHITS_INPUT_CHUNK_LENGTH + NHITS_OUTPUT_CHUNK_LENGTH
    return 3
    
def parse_row(row):
    parsed = extract_timestamp_and_value(row)

    if parsed is None:
        return []

    return [{
        "ts": parsed[0],
        "value": parsed[1],
    }]
    
@ray.remote
def train_forecasting_model(ds: typing.List[ray.ObjectRef[ray.data.Dataset]], 
                            model_type: str,
                            time_series_frequency: str, 
                            mlflow_logger: TrainMlflowLogger
                            ) -> PythonModel:
    with mlflow_logger.trace("operator_training_pipeline"):
        parsed_datasets: typing.List[ray.data.Dataset] = []

        with mlflow_logger.trace("parse_input_datasets"):
            for ds_ref in ds:
                dataset = ray.get(ds_ref) if isinstance(
                    ds_ref, ray.ObjectRef) else ds_ref

                parsed_dataset = dataset.flat_map(parse_row)
                parsed_datasets.append(parsed_dataset)

        if len(parsed_datasets) == 0:
            raise RuntimeError("Need at least one dataset to train the model.")

        with mlflow_logger.trace("merge_datasets"):
            merged_dataset = parsed_datasets[0]
            for additional_dataset in parsed_datasets[1:]:
                merged_dataset = merged_dataset.union(additional_dataset)

        with mlflow_logger.trace("sort_materialize"):
            sorted_dataset = merged_dataset.sort("ts").materialize()

        with mlflow_logger.trace("count_points"):
            num_points = sorted_dataset.count()
        if num_points < 2:
            raise RuntimeError(
                "Need at least two timestamp/value points to train the model.")


        training_timeseries: darts.TimeSeries = convert_dataset_to_timeseries(sorted_dataset, time_series_frequency)
        last_ts = training_timeseries.time_index[-1]
        logger.debug(f"Last timestamp in training data: {last_ts}")



        with mlflow_logger.trace("count_resampled_timeseries_points"):
            num_training_points = len(training_timeseries)
        required_training_points = minimum_training_points(model_type)
        if num_training_points < required_training_points:
            raise RuntimeError(
                f"Need at least {required_training_points} points in resampled "
                "historic time series to fit the model."
            )

        # Log training metadata.
        mlflow_logger.set_tags({
            "training.framework": "ray-train-consumption_forecast",
            "training.task": "predict-next-consumption-value",
            "training.trace": "enabled",
        })
        mlflow_logger.log_params({
            "num_points": num_points,
            "num_points_resampled_timeseries": num_training_points,
            "num_input_datasets": len(parsed_datasets),
        })
        mlflow_logger.log_dict({
            "counts": {
                "num_points": num_points,
                "num_points_resampled_timeseries": num_training_points,
                "num_input_datasets": len(parsed_datasets),
            },
        }, "training_dataset_summary.json")

        model = build_model(model_type)

        model.fit(training_timeseries)

        if model_type == "nhits":
            return NHitsForecastingModel(model, pd.Timestamp(last_ts))
        else:
            return ProphetForecastingModel(model, pd.Timestamp(last_ts))
