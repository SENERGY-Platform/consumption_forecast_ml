from operator_lib.util import Config, MLOperator, logger
from operator_lib.util.helpers import provide_historic_data, provide_historic_data_local, TrainMlflowLogger


import ray
import typing
import datetime
from mlflow.pyfunc import PyFuncModel, PythonModel
from process_inference_data import check_for_period_change, convert_inference_ds_to_ts, get_period_end
from inference import compute_single_output_total, predict_single_output, predict_multi_output

from training import train_forecasting_model

import pandas as pd

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

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        
        self.period = self.config.time_period

        logger.info(self.period)

        self.all_possible_periods = set(self.period_single_or_multi_output)
        if self.period not in self.all_possible_periods:
            raise ValueError(f"Unsupported time period: {self.period}")

        self.timeseries_frequency = self.period_single_or_multi_output[self.period]

        self.last_timestamp = None
        self.timedelta_for_inference_data = self.inference_history_duration[self.period]

        self.period_changed = False


    def infer(self, model: typing.Optional[PyFuncModel], data: typing.Dict[str, typing.Any], selector: str, device_id: str, timestamp: datetime.datetime) -> typing.Tuple[typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]]:
        current_value = data.get("value")
        if current_value is None:
            return None, None, None
        
        self.timestamp = pd.Timestamp(timestamp)

        logger.debug(f"Time: {self.timestamp}, Value: {current_value}")

        payload = {
            "timestamp": timestamp,
            "value": float(current_value),
        }
        

        if self.last_timestamp is None:
            self.last_timestamp = self.timestamp
            return None, None, None
        else:
            self.period_changed = check_for_period_change(self.timestamp, self.last_timestamp, self.period)
            self.last_timestamp = self.timestamp
            if self.period_changed:
                logger.debug(f"Period changed! New {self.period_translation_dict[self.period]} started at {self.timestamp}")
                if model is not None:
                    try:
                        if self.timeseries_frequency == self.period:
                            predicted_value = predict_single_output(model, payload)
                            predicted_total = compute_single_output_total(
                                predicted_value,
                                payload["value"],
                            )
                        else:
                            inference_datasets = provide_historic_data_local(self.timedelta_for_inference_data)
                            inference_timeseries = convert_inference_ds_to_ts(inference_datasets, self.timeseries_frequency)
                            logger.debug(f"Last timestamp of inference timeseries for multi output prediction: {inference_timeseries.time_index[-1]}")
                            predicted_value = predict_multi_output(model, payload, self.timeseries_frequency, self.period, inference_timeseries)
                            predicted_total = predicted_value + inference_timeseries.last_value()
                        output_timestamp = (
                            get_period_end(self.timestamp, self.period)
                            .floor("us")
                            .to_pydatetime()
                        )
                        logger.debug(f"Predicted consumption: {predicted_value} for next {self.period} at timestamp: {output_timestamp}")
                        return (output_timestamp,
                                {f'{self.period_translation_dict[self.period]}Prediction': predicted_value,
                                f'{self.period_translation_dict[self.period]}PredictionTotal': predicted_total},
                                None)
                    except Exception as e:
                        logger.exception("Prediction failed!")
        return None, None, None

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
        if model is not None:
            last_ts_model = model._model_impl.python_model.last_ts
            if self.timestamp - last_ts_model > pd.Timedelta(1,"D"):
                return True # This means the model was trained on data which is older than one week. This mostly happens when an old model exists and for whatever reason there was no training for the last week.
        if self.period_changed:
            return True 
        else:
            return False
