import darts
import numpy as np
import pandas as pd

from inference import INTERVAL_FREQUENCIES


class FakeDataset:
    """Stands in for a ray Dataset, with the operations the operator uses."""

    def __init__(self, rows):
        self.rows = list(rows)

    def flat_map(self, fn):
        return FakeDataset(out for row in self.rows for out in fn(row))

    def union(self, other):
        return FakeDataset(self.rows + other.rows)

    def sort(self, key):
        return FakeDataset(sorted(self.rows, key=lambda row: row[key]))

    def materialize(self):
        return self

    def count(self):
        return len(self.rows)

    def to_pandas(self):
        return pd.DataFrame(self.rows)


def meter_readings(start, end, kwh_per_hour=1.0):
    """Rows of historic data as operator-lib reads them: a reading every 15 minutes."""
    times = pd.date_range(start, end, freq="15min")
    hours = (times - times[0]) / pd.Timedelta(hours=1)
    return [
        {"time": time, "Consumption": 1000.0 + kwh_per_hour * hour}
        for time, hour in zip(times, hours)
    ]


class FakeForecaster:
    def __init__(self, last_ts, value=1.0, values=None, model_type="prophet", add_time_covariates=False):
        self.last_ts = pd.Timestamp(last_ts)
        self.value = value
        self.values = values
        self.model_type = model_type
        self.add_time_covariates = add_time_covariates


class FakeModel:
    """
    Stands in for the MLflow model: each forecaster forecasts its values for
    the intervals after its last_ts.
    """

    def __init__(self, forecasters):
        self.forecasters = forecasters

    def unwrap_python_model(self):
        return self

    def predict(self, data):
        forecaster = self.forecasters[data["timeseries_frequency"]]
        steps = data["missing_steps"]
        offset = pd.tseries.frequencies.to_offset(
            INTERVAL_FREQUENCIES[data["timeseries_frequency"]])
        index = pd.date_range(forecaster.last_ts + offset, periods=steps, freq=offset)
        if forecaster.values is None:
            values = np.full(steps, forecaster.value)
        else:
            values = np.asarray(forecaster.values[:steps], dtype=float)
        return darts.TimeSeries.from_times_and_values(index, values)
