import typing
import ray
from util import (
    convert_dataset_to_timeseries,
    extract_timestamp_and_value,
    to_utc_naive_timestamp,
)
import darts
import pandas as pd


PERIOD_FREQUENCIES = {
    "H": "h",
    "D": "D",
    "W": "W-SUN",
    "M": "M",
    "Y": "Y-DEC",
}


def get_period_bounds(timestamp, period):
    current_timestamp = to_utc_naive_timestamp(timestamp)

    if period == "4H":
        period_start = current_timestamp.floor("4h")
        return period_start, period_start + pd.Timedelta(hours=4)

    frequency = PERIOD_FREQUENCIES.get(period)
    if frequency is None:
        raise ValueError(f"Unsupported period: {period}")

    current_period = current_timestamp.to_period(frequency)
    return current_period.start_time, current_period.end_time + pd.Timedelta(
        nanoseconds=1
    )


def check_for_period_change(timestamp, last_timestamp, period):
    current_period_start, _ = get_period_bounds(timestamp, period)
    previous_period_start, _ = get_period_bounds(last_timestamp, period)
    return current_period_start != previous_period_start


def get_period_start(timestamp, period) -> pd.Timestamp:
    period_start, _ = get_period_bounds(timestamp, period)
    return period_start


def get_period_end(timestamp, period) -> pd.Timestamp:
    _, period_end_exclusive = get_period_bounds(timestamp, period)
    return period_end_exclusive - pd.Timedelta(nanoseconds=1)

def parse_row(row):
    parsed = extract_timestamp_and_value(row)

    if parsed is None:
        return []

    return [{
        "ts": parsed[0],
        "value": parsed[1],
    }]

def convert_inference_ds_to_ts(ds: typing.List[ray.ObjectRef[ray.data.Dataset]], time_series_frequency: str) -> darts.TimeSeries:
    parsed_datasets: typing.List[ray.data.Dataset] = []

    for ds_ref in ds:
        dataset = ray.get(ds_ref) if isinstance(
                    ds_ref, ray.ObjectRef) else ds_ref

        parsed_dataset = dataset.flat_map(parse_row)
        parsed_datasets.append(parsed_dataset)

    if len(parsed_datasets) == 0:
        raise RuntimeError("Need at least one dataset to train the model.")

    merged_dataset = parsed_datasets[0]
    for additional_dataset in parsed_datasets[1:]:
        merged_dataset = merged_dataset.union(additional_dataset)

    sorted_dataset = merged_dataset.sort("ts").materialize()

    num_points = sorted_dataset.count()
    if num_points < 2:
        raise RuntimeError(
            "Need at least two timestamp/value points to train the model.")

    inference_timeseries: darts.TimeSeries = convert_dataset_to_timeseries(sorted_dataset, time_series_frequency)
    return inference_timeseries
