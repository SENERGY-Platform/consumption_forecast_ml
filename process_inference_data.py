import typing
import ray
from util import (
    convert_dataset_to_timeseries,
    parse_row,
    to_utc_naive_timestamp,
    to_utc_timestamp,
)
import darts
import pandas as pd


PERIOD_TIMEZONE = "Europe/Berlin"

PERIOD_FREQUENCIES = {
    "D": "D",
    "W": "W-SUN",
    "M": "M",
    "Y": "Y-DEC",
}

RASTER_FREQUENCIES = {
    "H": "h",
    "4H": "4h",
    "D": "D",
}


def local_to_utc_naive(timestamp: pd.Timestamp) -> pd.Timestamp:
    return (
        timestamp.tz_localize(PERIOD_TIMEZONE)
        .tz_convert("UTC")
        .tz_localize(None)
    )


def get_period_bounds(timestamp, period):
    current_timestamp = to_utc_timestamp(timestamp)

    if period == "H":
        # Berlin hours always coincide with UTC hours.
        period_start = current_timestamp.floor("h").tz_localize(None)
        return period_start, period_start + pd.Timedelta(hours=1)

    # All other periods start at midnight or a multiple of four hours local
    # time, which never falls into the hour skipped or repeated on DST changes.
    local_timestamp = current_timestamp.tz_convert(PERIOD_TIMEZONE).tz_localize(None)

    if period == "4H":
        period_start = local_timestamp.floor("4h")
        period_end = period_start + pd.Timedelta(hours=4)
    else:
        frequency = PERIOD_FREQUENCIES.get(period)
        if frequency is None:
            raise ValueError(f"Unsupported period: {period}")
        current_period = local_timestamp.to_period(frequency)
        period_start = current_period.start_time
        period_end = (current_period + 1).start_time

    return local_to_utc_naive(period_start), local_to_utc_naive(period_end)


def get_period_start(timestamp, period) -> pd.Timestamp:
    period_start, _ = get_period_bounds(timestamp, period)
    return period_start


def get_period_end(timestamp, period) -> pd.Timestamp:
    _, period_end_exclusive = get_period_bounds(timestamp, period)
    return period_end_exclusive - pd.Timedelta(nanoseconds=1)


def format_period_end(timestamp, period) -> str:
    # Timestamps in operator outputs are UTC with Z, as the platform's
    # checklist for shared content asks.
    return get_period_end(timestamp, period).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_raster_start(timestamp, timeseries_frequency) -> pd.Timestamp:
    """Start of the UTC interval of the forecasting raster containing timestamp."""
    current_timestamp = to_utc_naive_timestamp(timestamp)
    if timeseries_frequency == "W":
        # Weekly intervals run from Sunday to Sunday, like the resampled series.
        day = current_timestamp.normalize()
        return day - pd.Timedelta(days=(day.dayofweek + 1) % 7)

    frequency = RASTER_FREQUENCIES.get(timeseries_frequency)
    if frequency is None:
        raise ValueError(f"Unsupported forecast frequency: {timeseries_frequency}")
    return current_timestamp.floor(frequency)

def convert_inference_ds_to_ts(ds: typing.List[ray.ObjectRef[ray.data.Dataset]], time_series_frequency: str) -> darts.TimeSeries:
    parsed_datasets: typing.List[ray.data.Dataset] = []

    for ds_ref in ds:
        dataset = ray.get(ds_ref) if isinstance(
                    ds_ref, ray.ObjectRef) else ds_ref

        parsed_dataset = dataset.flat_map(parse_row)
        parsed_datasets.append(parsed_dataset)

    if len(parsed_datasets) == 0:
        raise RuntimeError("Need at least one dataset for inference.")

    merged_dataset = parsed_datasets[0]
    for additional_dataset in parsed_datasets[1:]:
        merged_dataset = merged_dataset.union(additional_dataset)

    sorted_dataset = merged_dataset.sort("ts").materialize()

    num_points = sorted_dataset.count()
    if num_points < 2:
        raise RuntimeError(
            "Need at least two timestamp/value points for inference.")

    inference_timeseries: darts.TimeSeries = convert_dataset_to_timeseries(sorted_dataset, time_series_frequency)
    return inference_timeseries
