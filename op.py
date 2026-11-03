from operator_lib.util import Config, Inputs, MLOperator, Outputs, logger
from operator_lib.util.helpers import (
    HistoryNeedsRayError,
    provide_historic_data,
    provide_historic_data_local,
    provide_historic_frames_local,
    TrainMlflowLogger,
)


import ray
import time
import typing
import datetime
import mlflow
import pandas as pd
from mlflow.pyfunc import PyFuncModel, PythonModel
from build_models import ForecastingModels
from process_inference_data import (
    convert_inference_frames_to_ts,
    format_period_end,
    get_period_start,
    get_raster_start,
)
from inference import (
    compute_single_output_total,
    interval_duration,
    model_forecasters,
    predict_multi_output,
    predict_single_output,
)

from training import MODEL_TYPES, train_forecasters
from util import VALUE_COLUMN, to_utc_naive_timestamp


def parse_time_periods(value: typing.Any) -> typing.List[str]:
    periods = []
    for period in str(value).split(","):
        period = period.strip()
        if not period or period in periods:
            continue
        if period not in Operator.period_single_or_multi_output:
            raise ValueError(f"Unsupported time period: {period}")
        periods.append(period)
    if not periods:
        raise ValueError(f"time_periods names no time period: {value!r}")
    return periods


def parse_bool(value: typing.Any) -> bool:
    # Config values arrive as strings.
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    raise ValueError(f"Not a boolean: {value!r}")


class CustomConfig(Config):
    time_periods: str = "D"
    model: str = "prophet"
    add_time_covariates: bool = False
    # A field of the library, declared again so that flows can set it.
    logger_level: str = "warning"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        # Validated here because operator-lib may start a training before
        # Operator.init gets to the configuration.
        parse_time_periods(self.time_periods)
        parse_bool(self.add_time_covariates)
        if self.model not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {self.model}")


class ConsumptionInputs(Inputs):
    Consumption: float


class ForecastOutputs(Outputs):
    HourPrediction: float
    HourPredictionTotal: float
    HourTimestamp: str
    FourHourPrediction: float
    FourHourPredictionTotal: float
    FourHourTimestamp: str
    DayPrediction: float
    DayPredictionTotal: float
    DayTimestamp: str
    WeekPrediction: float
    WeekPredictionTotal: float
    WeekTimestamp: str
    MonthPrediction: float
    MonthPredictionTotal: float
    MonthTimestamp: str
    YearPrediction: float
    YearPredictionTotal: float
    YearTimestamp: str


class Operator(MLOperator):
    configType = CustomConfig
    inputs = ConsumptionInputs
    outputs = ForecastOutputs
    period_single_or_multi_output = {
        'H': 'H',
        # Hourly, so that the Berlin 4H blocks consist of whole intervals.
        '4H': 'H',
        'D': '4H',
        'W': 'D',
        'M': 'D',
        'Y': 'W',
    }
    period_translation_dict = {
        'H': 'Hour',
        '4H': 'FourHour',
        'D': 'Day',
        'W': 'Week',
        'M': 'Month',
        'Y': 'Year',
    }
    inference_history_duration = {
        'H': datetime.timedelta(hours=2),
        '4H': datetime.timedelta(hours=8),
        'D': datetime.timedelta(hours=8),
        'W': datetime.timedelta(days=2),
        'M': datetime.timedelta(days=2),
        'Y': datetime.timedelta(days=14),
    }
    # Per timeseries frequency; periods sharing one share these as well.
    training_history_duration = {
        'H': datetime.timedelta(days=14),
        '4H': datetime.timedelta(days=14),
        'D': datetime.timedelta(days=14),
        'W': datetime.timedelta(days=365),
    }
    retraining_intervals = {
        'H': datetime.timedelta(days=1),
        '4H': datetime.timedelta(days=1),
        'D': datetime.timedelta(days=7),
        'W': datetime.timedelta(days=28),
    }
    training_retry_delay = datetime.timedelta(hours=1)

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)

        self.periods = parse_time_periods(self.config.time_periods)

        logger.info(self.periods)

        self.last_timestamp = None
        # Per period: (period start, raster interval start) of the latest
        # forecast, and that forecast's output fields.
        self.forecast_keys = {}
        self.latest_outputs = {}

        self.current_message_ignored = False
        self.history_needs_ray = False


    def infer(self, model: typing.Optional[PyFuncModel], data: typing.Dict[str, typing.Any], selector: str, device_id: str, timestamp: datetime.datetime) -> typing.Tuple[typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]]:
        self.current_message_ignored = False
        current_value = data.get(VALUE_COLUMN)
        if current_value is None:
            self.current_message_ignored = True
            return None, None, None

        current_timestamp = to_utc_naive_timestamp(timestamp)
        if (
            self.last_timestamp is not None
            and current_timestamp <= self.last_timestamp
        ):
            self.current_message_ignored = True
            logger.warning(
                "Ignoring out-of-order or duplicate message at %s; "
                "last accepted timestamp is %s.",
                current_timestamp,
                self.last_timestamp,
            )
            return None, None, None

        self.timestamp = current_timestamp
        self.last_timestamp = current_timestamp

        logger.debug(f"Time: {self.timestamp}, Value: {current_value}")

        payload = {
            "timestamp": self.timestamp,
            "value": float(current_value),
        }

        due_periods = self.periods_due_for_forecast(model)
        multi_output_periods = [
            period for period in due_periods
            if self.period_single_or_multi_output[period] != period
        ]
        inference_frames = None
        if multi_output_periods:
            try:
                inference_frames = self.read_inference_history(max(
                    self.get_inference_history_duration(period)
                    for period in multi_output_periods
                ))
            except Exception:
                logger.exception("Loading the inference history failed!")
                due_periods = [
                    period for period in due_periods
                    if period not in multi_output_periods
                ]

        inference_timeseries = {}
        updated = False
        for period in due_periods:
            try:
                self.latest_outputs[period] = self.forecast_period(
                    model,
                    payload,
                    period,
                    inference_frames,
                    inference_timeseries,
                )
            except Exception:
                logger.exception(f"Prediction for {period} failed!")
                continue
            self.forecast_keys[period] = self.forecast_key(period)
            updated = True

        if not updated:
            return None, None, None

        output = self.current_outputs()
        logger.debug(f"Operator output at {self.timestamp}: {output}")
        return self.timestamp.to_pydatetime(), output, None

    def forecast_key(self, period: str):
        timeseries_frequency = self.period_single_or_multi_output[period]
        return (
            get_period_start(self.timestamp, period),
            get_raster_start(self.timestamp, timeseries_frequency),
        )

    def periods_due_for_forecast(self, model: typing.Optional[PyFuncModel]) -> typing.List[str]:
        # Like the old operator, a period is forecast again whenever its raster
        # moves on; here also as soon as the period itself starts.
        forecasters = self.usable_forecasters(model)
        return [
            period for period in self.periods
            if self.period_single_or_multi_output[period] in forecasters
            and self.forecast_keys.get(period) != self.forecast_key(period)
        ]

    def read_inference_history(self, duration: datetime.timedelta) -> typing.List[pd.DataFrame]:
        # infer() runs without a cluster connection, where
        # provide_historic_data_local starts a local Ray instance of about 2 GB.
        # Only the history of an input fed by another operator, which is in
        # Kafka, still needs it.
        if not self.history_needs_ray:
            try:
                return provide_historic_frames_local(duration)
            except HistoryNeedsRayError:
                logger.warning(
                    "The history of the input is only in Kafka; reading it "
                    "with Ray, which starts a local Ray instance."
                )
                self.history_needs_ray = True
        return [dataset.to_pandas() for dataset in provide_historic_data_local(duration)]

    def forecast_period(self, model, payload, period, inference_frames, inference_timeseries) -> typing.Dict[str, typing.Any]:
        timeseries_frequency = self.period_single_or_multi_output[period]
        if timeseries_frequency == period:
            forecast = predict_single_output(
                model,
                payload,
                timeseries_frequency,
                period,
            )
        else:
            if timeseries_frequency not in inference_timeseries:
                inference_timeseries[timeseries_frequency] = convert_inference_frames_to_ts(
                    inference_frames,
                    timeseries_frequency,
                )
            forecast = predict_multi_output(
                model,
                payload,
                timeseries_frequency,
                period,
                inference_timeseries[timeseries_frequency],
            )

        name = self.period_translation_dict[period]
        return {
            f'{name}Prediction': forecast.period_consumption,
            f'{name}PredictionTotal': compute_single_output_total(
                forecast.remaining_consumption,
                payload["value"],
            ),
            f'{name}Timestamp': format_period_end(self.timestamp, period),
        }

    def current_outputs(self) -> typing.Dict[str, typing.Any]:
        # Every period with a forecast for the period it is in, like the old
        # operator sent them.
        output = {}
        for period in self.periods:
            key = self.forecast_keys.get(period)
            if key is not None and key[0] == get_period_start(self.timestamp, period):
                output.update(self.latest_outputs[period])
        return output

    def get_inference_history_duration(self, period: str) -> datetime.timedelta:
        elapsed_in_period = (
            self.timestamp - get_period_start(self.timestamp, period)
        ).to_pytimedelta()
        timeseries_frequency = self.period_single_or_multi_output[period]
        return max(
            self.inference_history_duration[period],
            elapsed_in_period
            + interval_duration(timeseries_frequency).to_pytimedelta(),
        )

    def train(self, model: typing.Optional[PyFuncModel], mlflow_logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
        self.last_training_attempt = time.monotonic()
        frequencies_to_train = self.frequencies_to_train(model)
        needed_frequencies = self.timeseries_frequencies()
        forecasters = {
            timeseries_frequency: forecaster
            for timeseries_frequency, forecaster in self.usable_forecasters(model).items()
            if timeseries_frequency in needed_frequencies
            and timeseries_frequency not in frequencies_to_train
        }

        # One history read per duration, so a yearly forecast does not make the
        # daily retraining read a year.
        groups = {}
        for timeseries_frequency in frequencies_to_train:
            history_duration = self.training_history_duration[timeseries_frequency]
            groups.setdefault(history_duration, []).append(timeseries_frequency)

        try:
            for history_duration, timeseries_frequencies in groups.items():
                datasets = provide_historic_data(history_duration)
                if len(datasets) == 0:
                    raise RuntimeError("Expected at least one ray Dataset!")
                forecasters.update(ray.get(train_forecasters.remote(
                    datasets,
                    self.config.model,
                    timeseries_frequencies,
                    parse_bool(self.config.add_time_covariates),
                    mlflow_logger,
                )))
        except Exception:
            # Raising would stop the operator. Keep the current model and
            # retry later instead.
            logger.exception(
                "Training failed! Retrying in %s at the earliest.",
                self.training_retry_delay,
            )
            return None
        return ForecastingModels(forecasters)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if getattr(self, "current_message_ignored", False):
            return False

        if not hasattr(self, "timestamp"):
            return False

        if not self.frequencies_to_train(model):
            return False

        if self.training_attempted_recently():
            return False

        # operator-lib leaves its MLflow run open when a training returns no
        # model, which would make the next mlflow.start_run() fail.
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        return True

    def timeseries_frequencies(self) -> typing.List[str]:
        # From the config rather than self.periods: operator-lib trains in its
        # init, before Operator.init has set them.
        periods = parse_time_periods(self.config.time_periods)
        return list(dict.fromkeys(
            self.period_single_or_multi_output[period] for period in periods
        ))

    def usable_forecasters(self, model: typing.Optional[PyFuncModel]) -> typing.Dict[str, typing.Any]:
        if model is None:
            return {}
        add_time_covariates = parse_bool(self.config.add_time_covariates)
        return {
            timeseries_frequency: forecaster
            for timeseries_frequency, forecaster in model_forecasters(model).items()
            if forecaster.model_type == self.config.model
            and forecaster.add_time_covariates == add_time_covariates
        }

    def frequencies_to_train(self, model: typing.Optional[PyFuncModel]) -> typing.List[str]:
        forecasters = self.usable_forecasters(model)
        timestamp = getattr(self, "timestamp", None)
        frequencies = []
        for timeseries_frequency in self.timeseries_frequencies():
            forecaster = forecasters.get(timeseries_frequency)
            if forecaster is None or (
                timestamp is not None
                and timestamp - to_utc_naive_timestamp(forecaster.last_ts)
                >= self.retraining_intervals[timeseries_frequency]
            ):
                frequencies.append(timeseries_frequency)
        return frequencies

    def training_attempted_recently(self) -> bool:
        last_attempt = getattr(self, "last_training_attempt", None)
        return (
            last_attempt is not None
            and time.monotonic() - last_attempt
            < self.training_retry_delay.total_seconds()
        )
