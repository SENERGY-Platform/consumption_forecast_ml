import typing


import pandas as pd
import ray

from util import convert_dataset_to_timeseries, parse_row
from operator_lib.util import logger
from operator_lib.util.helpers import TrainMlflowLogger

import darts
from darts.dataprocessing.transformers.scaler import Scaler
from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.prophet_model import Prophet

from build_models import Forecaster


MODEL_TYPES = ("prophet", "nhits")

NHITS_INPUT_CHUNK_LENGTH = 7
NHITS_OUTPUT_CHUNK_LENGTH = 1


def time_covariate_encoders(timeseries_frequency: str, covariates: str) -> typing.Dict[str, typing.Any]:
    # The calendar attributes the raster resolves. NHiTS takes them as past,
    # Prophet as future covariates.
    attributes = ["month"]
    if timeseries_frequency != "W":
        attributes.append("dayofweek")
    if timeseries_frequency in ("H", "4H"):
        attributes.append("hour")
    return {
        "datetime_attribute": {covariates: attributes},
        "transformer": Scaler(),
    }


def build_model(model_type: str, timeseries_frequency: str, add_time_covariates: bool) -> typing.Any:
    model_kwargs = {}
    if model_type == "nhits":
        if add_time_covariates:
            model_kwargs["add_encoders"] = time_covariate_encoders(
                timeseries_frequency, "past")
        return NHiTSModel(
            num_stacks=3,
            num_blocks=2,
            num_layers=1,
            input_chunk_length=NHITS_INPUT_CHUNK_LENGTH,
            output_chunk_length=NHITS_OUTPUT_CHUNK_LENGTH,
            **model_kwargs,
        )
    else:
        if add_time_covariates:
            model_kwargs["add_encoders"] = time_covariate_encoders(
                timeseries_frequency, "future")
        return Prophet(country_holidays="DE", **model_kwargs)


def minimum_training_points(model_type: str) -> int:
    if model_type == "nhits":
        return NHITS_INPUT_CHUNK_LENGTH + NHITS_OUTPUT_CHUNK_LENGTH
    return 3


def fit_forecaster(training_timeseries: darts.TimeSeries,
                   model_type: str,
                   timeseries_frequency: str,
                   add_time_covariates: bool,
                   ) -> Forecaster:
    required_training_points = minimum_training_points(model_type)
    if len(training_timeseries) < required_training_points:
        raise RuntimeError(
            f"Need at least {required_training_points} points in resampled "
            "historic time series to fit the model."
        )

    model = build_model(model_type, timeseries_frequency, add_time_covariates)
    model.fit(training_timeseries)

    last_ts = pd.Timestamp(training_timeseries.time_index[-1])
    logger.debug(
        f"Last timestamp in {timeseries_frequency} training data: {last_ts}")
    return Forecaster(model, model_type, add_time_covariates, last_ts)


@ray.remote
def train_forecasters(ds: typing.List[ray.ObjectRef[ray.data.Dataset]],
                      model_type: str,
                      timeseries_frequencies: typing.List[str],
                      add_time_covariates: bool,
                      mlflow_logger: TrainMlflowLogger
                      ) -> typing.Dict[str, Forecaster]:
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

        # Log training metadata.
        mlflow_logger.set_tags({
            "training.framework": "ray-train-consumption_forecast",
            "training.task": "predict-next-consumption-value",
            "training.trace": "enabled",
        })

        forecasters = {}
        for timeseries_frequency in timeseries_frequencies:
            with mlflow_logger.trace(f"fit_{timeseries_frequency}"):
                training_timeseries: darts.TimeSeries = convert_dataset_to_timeseries(
                    sorted_dataset, timeseries_frequency)
                counts = {
                    "num_points": num_points,
                    "num_points_resampled_timeseries": len(training_timeseries),
                    "num_input_datasets": len(parsed_datasets),
                }
                # Prefixed, since one run trains several frequencies and MLflow
                # rejects a param logged again with another value.
                mlflow_logger.log_params({
                    f"{timeseries_frequency}.{name}": value
                    for name, value in counts.items()
                })
                mlflow_logger.log_dict(
                    {"counts": counts},
                    f"training_dataset_summary_{timeseries_frequency}.json",
                )
                forecasters[timeseries_frequency] = fit_forecaster(
                    training_timeseries,
                    model_type,
                    timeseries_frequency,
                    add_time_covariates,
                )
        return forecasters
