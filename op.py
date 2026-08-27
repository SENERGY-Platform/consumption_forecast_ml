from operator_lib.util import Config, MLOperator, logger
from operator_lib.util.helpers import provide_historic_data, provide_historic_data_local, TrainMlflowLogger


import ray
import typing
import datetime
from mlflow.pyfunc import PyFuncModel, PythonModel
from process_inference_data import (
    check_for_period_change,
    convert_inference_ds_to_ts,
    get_period_end,
    get_period_start,
)
from inference import (
    compute_single_output_total,
    model_last_timestamp,
    predict_multi_output,
    predict_single_output,
)

from training import train_forecasting_model
from util import to_utc_naive_timestamp

class CustomConfig(Config):
    time_period = "H"
    model_type = "prophet"
    
    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
    
class Operator(MLOperator):
    configType = CustomConfig
    period_single_or_multi_output = {
        'H': 'H',
        '4H': '4H',
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
    training_history_duration = {
        'H': datetime.timedelta(days=14),
        '4H': datetime.timedelta(days=14),
        'D': datetime.timedelta(days=14),
        'W': datetime.timedelta(days=14),
        'M': datetime.timedelta(days=14),
        'Y': datetime.timedelta(days=365),
    }
    forecast_interval_duration = {
        'H': datetime.timedelta(hours=1),
        '4H': datetime.timedelta(hours=4),
        'D': datetime.timedelta(days=1),
        'W': datetime.timedelta(days=7),
    }
    retraining_intervals = {
        'H': datetime.timedelta(days=1),
        '4H': datetime.timedelta(days=1),
        'D': datetime.timedelta(days=1),
        'W': datetime.timedelta(days=7),
        'M': datetime.timedelta(days=7),
        'Y': datetime.timedelta(days=28),
    }

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        
        self.period = self.config.time_period

        logger.info(self.period)

        self.all_possible_periods = set(self.period_single_or_multi_output)
        if self.period not in self.all_possible_periods:
            raise ValueError(f"Unsupported time period: {self.period}")

        self.timeseries_frequency = self.period_single_or_multi_output[self.period]

        self.last_timestamp = None
        self.last_predicted_period_start = None
        self.timedelta_for_inference_data = self.inference_history_duration[self.period]

        self.period_changed = False
        self.current_message_ignored = False


    def infer(self, model: typing.Optional[PyFuncModel], data: typing.Dict[str, typing.Any], selector: str, device_id: str, timestamp: datetime.datetime) -> typing.Tuple[typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]]:
        self.current_message_ignored = False
        current_value = data.get("value")
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

        previous_timestamp = self.last_timestamp
        self.timestamp = current_timestamp
        self.last_timestamp = current_timestamp

        if previous_timestamp is None:
            self.period_changed = False
        else:
            self.period_changed = check_for_period_change(
                current_timestamp,
                previous_timestamp,
                self.period,
            )

        logger.debug(f"Time: {self.timestamp}, Value: {current_value}")

        payload = {
            "timestamp": self.timestamp,
            "value": float(current_value),
        }

        current_period_start = get_period_start(self.timestamp, self.period)
        should_predict = (
            model is not None
            and current_period_start
            != getattr(self, "last_predicted_period_start", None)
        )

        if self.period_changed:
            logger.debug(
                f"Period changed! New {self.period_translation_dict[self.period]} "
                f"started at {self.timestamp}"
            )

        if should_predict:
            try:
                if self.timeseries_frequency == self.period:
                    forecast = predict_single_output(
                        model,
                        payload,
                        self.timeseries_frequency,
                        self.period,
                    )
                else:
                    inference_history_duration = (
                        self.get_inference_history_duration(current_period_start)
                    )
                    inference_datasets = provide_historic_data_local(
                        inference_history_duration
                    )
                    inference_timeseries = convert_inference_ds_to_ts(
                        inference_datasets,
                        self.timeseries_frequency,
                    )
                    logger.debug(
                        "Last timestamp of inference timeseries for multi "
                        f"output prediction: {inference_timeseries.time_index[-1]}"
                    )
                    forecast = predict_multi_output(
                        model,
                        payload,
                        self.timeseries_frequency,
                        self.period,
                        inference_timeseries,
                    )

                predicted_value = forecast.period_consumption
                predicted_total = compute_single_output_total(
                    forecast.remaining_consumption,
                    payload["value"],
                )
                output_timestamp = (
                    get_period_end(self.timestamp, self.period)
                    .floor("us")
                    .to_pydatetime()
                )
                logger.debug(
                    f"Predicted consumption: {predicted_value} for next "
                    f"{self.period} at timestamp: {output_timestamp}"
                )
                self.last_predicted_period_start = current_period_start
                return (
                    output_timestamp,
                    {
                        f'{self.period_translation_dict[self.period]}Prediction': predicted_value,
                        f'{self.period_translation_dict[self.period]}PredictionTotal': predicted_total,
                    },
                    None,
                )
            except Exception:
                logger.exception("Prediction failed!")
        return None, None, None

    def get_inference_history_duration(self, current_period_start):
        elapsed_in_period = (
            self.timestamp - current_period_start
        ).to_pytimedelta()
        interval_duration = self.forecast_interval_duration[
            self.timeseries_frequency
        ]
        return max(
            self.timedelta_for_inference_data,
            elapsed_in_period + interval_duration,
        )

    def train(self, _: typing.Optional[PyFuncModel], logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
        period = self.config.time_period
        try:
            history_duration = self.training_history_duration[period]
            timeseries_frequency = self.period_single_or_multi_output[period]
        except KeyError as exc:
            raise ValueError(f"Unsupported time period: {period}") from exc

        datasets = provide_historic_data(history_duration)
        if len(datasets) == 0:
            raise RuntimeError("Expected at least one ray Dataset!")
        return ray.get(train_forecasting_model.remote(
            datasets,
            self.config.model_type,
            timeseries_frequency,
            logger,
        ))

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if getattr(self, "current_message_ignored", False):
            return False

        if not hasattr(self, "timestamp"):
            return False

        if model is None:
            return True

        last_ts_model = model_last_timestamp(model)
        return (
            self.timestamp - last_ts_model
            >= self.retraining_intervals[self.period]
        )
