import numbers
import typing

import pandas as pd

from mlflow.pyfunc import PythonModel


from darts.models.forecasting.nhits import NHiTSModel
from darts.models.forecasting.prophet_model import Prophet


def get_missing_steps(data: typing.Any) -> int:
    if not isinstance(data, dict):
        raise ValueError("Model input must be a dict.")

    missing_steps = data.get("missing_steps")
    if (
        isinstance(missing_steps, bool)
        or not isinstance(missing_steps, numbers.Integral)
        or missing_steps < 1
    ):
        raise ValueError("data must include a positive integer 'missing_steps'.")
    return int(missing_steps)



class NHitsForecastingModel(PythonModel):
    def __init__(
        self,
        model: NHiTSModel,
        last_ts: pd.Timestamp
    ) -> None:
        self.model = model
        self.last_ts = last_ts

    def predict(self, data: typing.Any) -> typing.Any:
        return self.model.predict(get_missing_steps(data))
        
class ProphetForecastingModel(PythonModel):
    def __init__(
        self,
        model: Prophet,
        last_ts: pd.Timestamp
    ) -> None:
        self.model = model
        self.last_ts = last_ts

    def predict(self, data: typing.Any) -> typing.Any:
        return self.model.predict(get_missing_steps(data))

