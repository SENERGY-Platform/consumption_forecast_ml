import dataclasses
import numbers
import typing

import pandas as pd

from mlflow.pyfunc import PythonModel


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



@dataclasses.dataclass(frozen=True)
class Forecaster:
    """A fitted darts model for one timeseries frequency."""

    model: typing.Any
    model_type: str
    add_time_covariates: bool
    last_ts: pd.Timestamp


class ForecastingModels(PythonModel):
    """
    The operator's MLflow model: one forecaster per timeseries frequency, so
    that one operator forecasts several periods.
    """

    def __init__(self, forecasters: typing.Dict[str, Forecaster]) -> None:
        self.forecasters = forecasters

    def predict(self, model_input):
        missing_steps = get_missing_steps(model_input)
        timeseries_frequency = model_input.get("timeseries_frequency")
        forecaster = self.forecasters.get(timeseries_frequency)
        if forecaster is None:
            raise ValueError(
                f"No forecaster for timeseries frequency {timeseries_frequency!r}."
            )
        return forecaster.model.predict(missing_steps)
