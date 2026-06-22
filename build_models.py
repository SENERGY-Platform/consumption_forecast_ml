import typing

import pandas as pd

from mlflow.pyfunc import PythonModel


from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.prophet_model import Prophet



class NHitsForecastingModel(PythonModel):
    def __init__(
        self,
        model: NHiTSModel,
        last_ts: pd.Timestamp
    ) -> None:
        self._model = model
        self.last_ts = last_ts

    def predict(self, data: typing.Any) -> typing.Any:
        if isinstance(data, dict):
            missing_steps = data.get("missing_steps")
            ts = data.get("timestamp")
            value = data.get("value")
            if missing_steps is None or ts is None or value is None:
                raise ValueError(
                    "data must include 'missing_steps', 'timestamp', and 'value'.")
            prediction = self._model.predict(missing_steps)
            return prediction
        raise ValueError("Model input must be a dict.")
        
class ProphetForecastingModel(PythonModel):
    def __init__(
        self,
        model: Prophet,
        last_ts: pd.Timestamp
    ) -> None:
        self._model = model
        self.last_ts = last_ts

    def predict(self, data: typing.Any) -> typing.Any:
        if isinstance(data, dict):
            missing_steps = data.get("missing_steps")
            ts = data.get("timestamp")
            value = data.get("value")
            if missing_steps is None or ts is None or value is None:
                raise ValueError(
                    "data must include 'missing_steps', 'timestamp', and 'value'.")
            prediction = self._model.predict(missing_steps)
            return prediction
        raise ValueError("Model input must be a dict.")

