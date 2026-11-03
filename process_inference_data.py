import typing
from util import (
    convert_frame_to_timeseries,
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

def convert_inference_frames_to_ts(frames: typing.List[pd.DataFrame], time_series_frequency: str) -> darts.TimeSeries:
    if len(frames) == 0:
        raise RuntimeError("Need at least one dataset for inference.")

    # Missing values as None, which parse_row skips, as it did the nulls of a
    # Ray dataset.
    rows = [
        parsed
        for frame in frames
        for row in frame.astype(object).where(frame.notna(), None).to_dict("records")
        for parsed in parse_row(row)
    ]
    if len(rows) < 2:
        raise RuntimeError(
            "Need at least two timestamp/value points for inference.")

    return convert_frame_to_timeseries(pd.DataFrame(rows), time_series_frequency)
