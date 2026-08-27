from dataclasses import dataclass

import pandas as pd

from process_inference_data import get_period_bounds
from util import to_utc_naive_timestamp


INTERVAL_FREQUENCIES = {
    "H": "h",
    "4H": "4h",
    "D": "D",
    "W": "W-SUN",
}

INTERVAL_DURATIONS = {
    "H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
    "D": pd.Timedelta(days=1),
    "W": pd.Timedelta(days=7),
}


@dataclass(frozen=True)
class PeriodForecast:
    period_consumption: float
    remaining_consumption: float


def unwrap_model(model):
    return model.unwrap_python_model()


def model_last_timestamp(model) -> pd.Timestamp:
    return to_utc_naive_timestamp(unwrap_model(model).last_ts)


def interval_duration(small_period) -> pd.Timedelta:
    try:
        return INTERVAL_DURATIONS[small_period]
    except KeyError as exc:
        raise ValueError(f"Unsupported forecast frequency: {small_period}") from exc


def infer_small_period(series) -> str:
    frequency = str(series.freq_str).lower()
    frequencies = {
        "h": "H",
        "4h": "4H",
        "d": "D",
        "w-sun": "W",
    }
    try:
        return frequencies[frequency]
    except KeyError as exc:
        raise ValueError(f"Unsupported forecast frequency: {series.freq_str}") from exc


def sum_series_over_window(series, window_start, window_end, small_period):
    window_start = to_utc_naive_timestamp(window_start)
    window_end = to_utc_naive_timestamp(window_end)
    if window_start >= window_end:
        return 0.0

    duration = interval_duration(small_period)
    total = 0.0
    for interval_start, value in zip(
        series.time_index,
        series.univariate_values(),
    ):
        interval_start = to_utc_naive_timestamp(interval_start)
        interval_end = interval_start + duration
        overlap_start = max(interval_start, window_start)
        overlap_end = min(interval_end, window_end)
        if overlap_start < overlap_end:
            overlap = overlap_end - overlap_start
            total += float(value) * float(overlap / duration)
    return total


def compute_output_nr(
    small_period,
    target_period,
    last_timestamp_timeseries,
    current_timestamp,
):
    try:
        frequency = INTERVAL_FREQUENCIES[small_period]
    except KeyError as exc:
        raise ValueError(f"Unsupported forecast frequency: {small_period}") from exc

    _, target_end_exclusive = get_period_bounds(
        current_timestamp,
        target_period,
    )
    offset = pd.tseries.frequencies.to_offset(frequency)
    first_forecast_timestamp = (
        to_utc_naive_timestamp(last_timestamp_timeseries) + offset
    )

    if first_forecast_timestamp >= target_end_exclusive:
        return 0

    forecast_timestamps = pd.date_range(
        start=first_forecast_timestamp,
        end=target_end_exclusive,
        freq=offset,
        inclusive="left",
    )
    return len(forecast_timestamps)


def predict_values_through_current_period(
    model,
    payload,
    timeseries_frequency,
    target_period,
):
    missing_steps = compute_output_nr(
        timeseries_frequency,
        target_period,
        model_last_timestamp(model),
        payload["timestamp"],
    )
    if missing_steps < 1:
        raise RuntimeError(
            "The model already ends at or after the requested forecast period."
        )

    predicted_values = model.predict({"missing_steps": missing_steps})
    return (predicted_values + abs(predicted_values)) / 2


def predict_single_output(model, payload, timeseries_frequency, period):
    predicted_values = predict_values_through_current_period(
        model,
        payload,
        timeseries_frequency,
        period,
    )
    period_start, period_end_exclusive = get_period_bounds(
        payload["timestamp"],
        period,
    )
    remaining_start = max(
        to_utc_naive_timestamp(payload["timestamp"]),
        period_start,
    )

    return PeriodForecast(
        period_consumption=sum_series_over_window(
            predicted_values,
            period_start,
            period_end_exclusive,
            timeseries_frequency,
        ),
        remaining_consumption=sum_series_over_window(
            predicted_values,
            remaining_start,
            period_end_exclusive,
            timeseries_frequency,
        ),
    )


def compute_single_output_total(predicted_value, current_value):
    return float(current_value) + predicted_value


def predict_multi_output(
    model,
    payload,
    timeseries_frequency,
    period,
    inference_timeseries,
):
    predicted_values = predict_values_through_current_period(
        model,
        payload,
        timeseries_frequency,
        period,
    )
    return compute_period_forecast(
        predicted_values,
        inference_timeseries,
        period,
        payload["timestamp"],
        timeseries_frequency,
    )


def compute_period_forecast(
    predicted_series,
    true_values_series,
    target_period,
    current_timestamp,
    small_period=None,
):
    if small_period is None:
        small_period = infer_small_period(predicted_series)

    period_start, period_end_exclusive = get_period_bounds(
        current_timestamp,
        target_period,
    )
    current_timestamp = min(
        max(to_utc_naive_timestamp(current_timestamp), period_start),
        period_end_exclusive,
    )

    known_until = period_start
    if len(true_values_series) > 0:
        latest_actual_end = (
            to_utc_naive_timestamp(true_values_series.time_index[-1])
            + interval_duration(small_period)
        )
        known_until = min(
            max(latest_actual_end, period_start),
            current_timestamp,
        )

    actual_consumption = sum_series_over_window(
        true_values_series,
        period_start,
        known_until,
        small_period,
    )
    forecast_consumption = sum_series_over_window(
        predicted_series,
        known_until,
        period_end_exclusive,
        small_period,
    )
    remaining_consumption = sum_series_over_window(
        predicted_series,
        current_timestamp,
        period_end_exclusive,
        small_period,
    )

    return PeriodForecast(
        period_consumption=actual_consumption + forecast_consumption,
        remaining_consumption=remaining_consumption,
    )


def compute_period_pred(
    predicted_series,
    true_values_series,
    target_period,
    current_timestamp,
    small_period=None,
):
    return compute_period_forecast(
        predicted_series,
        true_values_series,
        target_period,
        current_timestamp,
        small_period,
    ).period_consumption
