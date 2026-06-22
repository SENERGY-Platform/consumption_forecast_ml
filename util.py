import datetime
import typing
import ray

import darts
from darts.dataprocessing.transformers import MissingValuesFiller

import pandas as pd


def to_epoch_seconds(value: typing.Any) -> float:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return float(value.timestamp())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parsed = datetime.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return float(parsed.timestamp())
    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def extract_timestamp_and_value(row: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Tuple[float, float]]:
    ts = row.get("timestamp", row.get("time", row.get("ts")))
    value = row.get("value")
    if ts is None or value is None:
        return None
    return to_epoch_seconds(ts), float(value)


def convert_dataset_to_timeseries(ds: ray.data.Dataset, time_series_frequency: str) -> darts.TimeSeries:
    df = ds.to_pandas().set_index("ts")
    df.index = pd.to_datetime(df.index, unit="s")
    df.index = df.index.map(lambda x: x.replace(microsecond=0)) # Cut microseconds of timestamps in order to being able to resample to seconds

    df = df[~df.index.duplicated(keep='first')] # Delete duplicate timestamps if there are any. This can happen if the input data has a higher frequency than seconds and we cut microseconds of the timestamps. In this case, we keep only the first value for each timestamp.

    df_resampled = df.resample('s').interpolate().resample(time_series_frequency).asfreq().dropna() # First resample to seconds and interpolate to lose only few information in a first step. Then resample to timeseries frequency.

    time_series = darts.TimeSeries.from_dataframe(df_resampled, freq=time_series_frequency) # Convert to darts timeseries.
    
    transformer = MissingValuesFiller()
    filled_time_series = transformer.transform(time_series) # Again fill missing values to be sure.

    timeseries_of_differences = filled_time_series.shift(-1).diff() # To get consumption per timeslot (e.g. hour, 4 hours, day,...) one needs to take the difference of two consecutive total consumption values. 
                                                                    # We want the corresponding index timestamp of the final series such that the value of the series for that index corresponds to the consumption of the period starting at that timestamp. To achieve this, we need to shift by -1.
    return timeseries_of_differences
