from operator_lib.util import Config, MLOperator, logger
from operator_lib.util.helpers import provide_historic_data, provide_historic_data_local, TrainMlflowLogger


import ray
import typing
import datetime
from mlflow.pyfunc import PyFuncModel, PythonModel
from process_inference_data import check_for_period_change, convert_inference_ds_to_ts
from inference import predict_single_output, predict_multi_output

from training import train_forecasting_model

import pandas as pd

class CustomConfig(Config):
    time_period = "H"
    model_type = "prophet"
    
    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
    
class Operator(MLOperator):
    configType = CustomConfig
    period_single_or_multi_output = {'H': 'H', 'D': '4H', 'W': 'D', 'M': 'D', 'Y': 'W'}

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        
        self.period = self.config.time_period

        logger.info(self.period)

        self.all_possible_periods = {'H', '4H', 'D', 'W', 'M', 'Y'}
        self.period_translation_dict = {'H': 'Hour', 'D': 'Day', 'W': 'Week', 'M': 'Month', 'Y': 'Year'}
        
        self.timeseries_frequency = self.period_single_or_multi_output[self.period]

        self.last_timestamp = None

        if self.period == 'H':
            self.timedelta_for_inference_data = datetime.timedelta(hours=2)
        elif self.period == 'D':
            self.timedelta_for_inference_data = datetime.timedelta(hours=8)
        elif self.period == 'W':
            self.timedelta_for_inference_data = datetime.timedelta(days=2)
        elif self.period == 'M':
            self.timedelta_for_inference_data = datetime.timedelta(days=2)
        elif self.period == 'Y':
            self.timedelta_for_inference_data = datetime.timedelta(days=14)

        self.period_changed = False


    def infer(self, model: typing.Optional[PyFuncModel], data: typing.Dict[str, typing.Any], selector: str, device_id: str, timestamp: datetime.datetime) -> typing.Tuple[typing.Optional[typing.Any], typing.Optional[PythonModel]]:
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
            self.period_changed = check_for_period_change(timestamp, self.last_timestamp, self.period)
            self.last_timestamp = self.timestamp
            if self.period_changed:
                logger.debug(f"Period changed! New {self.period_translation_dict[self.period]} started at {self.timestamp}")
                if model is not None:
                    try:
                        if self.timeseries_frequency == self.period:
                            predicted_value = predict_single_output(model, payload)
                        else:
                            inference_datasets = provide_historic_data_local(self.timedelta_for_inference_data)
                            inference_timeseries = convert_inference_ds_to_ts(inference_datasets, self.timeseries_frequency)
                            logger.debug(f"Last timestamp of inference timeseries for multi output prediction: {inference_timeseries.time_index[-1]}")
                            predicted_value = predict_multi_output(model, payload, self.timeseries_frequency, self.period, inference_timeseries)
                        output_timestamp = self.timestamp.to_period(self.period).to_timestamp(how="end").to_pydatetime()
                        logger.debug(f"Predicted consumption: {predicted_value} for next {self.period} at timestamp: {output_timestamp}")
                        return (output_timestamp, 
                                {f'{self.period_translation_dict[self.period]}Prediction': predicted_value,
                                f'{self.period_translation_dict[self.period]}PredictionTotal': predicted_value + inference_timeseries.last_value()},
                                None)
                    except Exception as e:
                        logger.exception("Prediction failed!")
        return None, None, None

    def train(self, _: typing.Optional[PyFuncModel], logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
        datasets = provide_historic_data(
            datetime.timedelta(days=14))
        if len(datasets) == 0:
            raise RuntimeError("Expected at least one ray Dataset!")
        return ray.get(train_forecasting_model.remote(datasets, self.config.model_type, self.period_single_or_multi_output[self.config.time_period], logger))

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is not None:
            last_ts_model = model._model_impl.python_model.last_ts
            if self.timestamp - last_ts_model > pd.Timedelta(1,"D"):
                return True # This means the model was trained on data which is older than one week. This mostly happens when an old model exists and for whatever reason there was no training for the last week.
        if self.period_changed:
            return True 
        else:
            return False
