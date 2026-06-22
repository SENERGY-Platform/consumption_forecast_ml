import typing
import ray
from util import extract_timestamp_and_value, convert_dataset_to_timeseries
import darts

def check_for_period_change(timestamp, last_timestamp, period):
    period_changed = False
    if period == 'H':
        if timestamp.hour == last_timestamp.hour:
            new_hour = False
        else:
            new_hour = True
        period_changed = new_hour

    if period == '4H':
        if timestamp.floor('4H') == last_timestamp.floor('4H'):
            new_four_hours = False
        else:
            new_four_hours = True
        period_changed = new_four_hours
    
    if period == 'D':
        if timestamp.date() == last_timestamp.date():
            new_day = False
        else:
            new_day = True
        period_changed = new_day
    
    if period == 'W':
        if timestamp.week == last_timestamp.week and timestamp.year == last_timestamp.year:
            new_week = False
        else:
            new_week = True
        period_changed = new_week

    if period == 'M':   
        if timestamp.month == last_timestamp.month and timestamp.year == last_timestamp.year:
            new_month = False
        else:
            new_month = True
        period_changed = new_month
    
    return period_changed

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