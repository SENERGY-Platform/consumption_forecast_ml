import datetime
import typing
import ray

import darts

import pandas as pd


FREQUENCY_ALIASES = {
    "H": "h",
    "4H": "4h",
}


def to_utc_timestamp(value: typing.Any) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        return pd.Timestamp(value, unit="s", tz="UTC")

    if not isinstance(value, (datetime.datetime, str, pd.Timestamp)):
        raise TypeError(f"Unsupported timestamp type: {type(value)}")

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def to_utc_naive_timestamp(value: typing.Any) -> pd.Timestamp:
    return to_utc_timestamp(value).tz_localize(None)


def to_epoch_seconds(value: typing.Any) -> float:
    return float(to_utc_timestamp(value).timestamp())


def extract_timestamp_and_value(row: typing.Dict[str, typing.Any]) -> typing.Optional[typing.Tuple[float, float]]:
    ts = row.get("timestamp", row.get("time", row.get("ts")))
    value = row.get("value")
    if ts is None or value is None:
        return None
    return to_epoch_seconds(ts), float(value)


def convert_dataset_to_timeseries(ds: ray.data.Dataset, time_series_frequency: str) -> darts.TimeSeries:
    frequency = FREQUENCY_ALIASES.get(
        time_series_frequency,
        time_series_frequency,
    )
    df = ds.to_pandas().set_index("ts").sort_index()
    df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_localize(None)
    # Align timestamps to whole seconds.
    df.index = df.index.map(lambda x: x.replace(microsecond=0))

    # Keep the first value for duplicate seconds.
    df = df[~df.index.duplicated(keep='first')]

    target_index = df.resample(frequency).asfreq().index
    interpolation_index = df.index.union(target_index).sort_values()
    df_resampled = (
        df.reindex(interpolation_index)
        .interpolate(method="time")
        .reindex(target_index)
        .dropna()
    )
    if len(df_resampled) < 2:
        raise RuntimeError("Need at least two aligned meter values.")

    differences = df_resampled.diff().shift(-1).iloc[:-1]
    # Ignore counter resets.
    differences = differences.mask(differences < 0)
    differences = (
        differences
        .interpolate(method="time", limit_direction="both")
        .dropna()
    )
    if len(differences) == 0:
        raise RuntimeError("Need at least one valid consumption difference.")

    return darts.TimeSeries.from_dataframe(
        differences,
        freq=frequency,
    )
