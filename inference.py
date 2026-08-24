import pandas as pd
import math


def predict_single_output(model, payload):
    n_steps = 1
    model_input = {
        "missing_steps": n_steps,
        "timestamp": payload["timestamp"],
        "value": payload["value"],
    }
    predicted_value = model.predict(model_input).first_value()
    predicted_value = (predicted_value + abs(predicted_value))/2 # This cuts the predicted value at 0 to prevent negative values
    return predicted_value


def compute_single_output_total(predicted_value, current_value):
    return float(current_value) + predicted_value


def predict_multi_output(model, payload, timeseries_frequency, period, inference_timeseries):
    last_ts_model = model._model_impl.python_model.last_ts
    missing_steps, weekly_proportion = compute_output_nr(timeseries_frequency, period, last_ts_model)
    model_input = {
        "missing_steps": missing_steps,
        "timestamp": payload["timestamp"],
        "value": payload["value"],
    }
    predicted_values = model.predict(model_input)
    predicted_values = (predicted_values + abs(predicted_values))/2 # This cuts the predicted timeseries at 0
    predicted_period_consumption = compute_period_pred(
        predicted_values,
        inference_timeseries,
        period,
        payload["timestamp"],
        weekly_proportion,
    )
    return predicted_period_consumption

def compute_output_nr(small_period, target_period, last_timestamp_timeseries):
        end_next_target_period = last_timestamp_timeseries.to_period(target_period).to_timestamp(how="end")
        time_until_next_target_period = end_next_target_period - last_timestamp_timeseries
        
        if small_period == 'H':
            return int(time_until_next_target_period/pd.Timedelta(1,small_period)), None
        elif small_period == '4H':
            return int(time_until_next_target_period/pd.Timedelta(small_period)), None
        elif small_period == 'D':
            return int(time_until_next_target_period/pd.Timedelta(1,small_period)), None
        elif small_period == 'W':
            number_weeks_proportion = time_until_next_target_period/pd.Timedelta(1,'W') 
            return math.ceil(number_weeks_proportion), number_weeks_proportion-math.floor(number_weeks_proportion)
        
def compute_period_pred(
    predicted_series,
    true_values_series,
    target_period,
    current_timestamp,
    weekly_proportion=None,
):
    time_last_value = true_values_series.time_index[-1]
    begin_current_period = (
        pd.Timestamp(current_timestamp)
        .to_period(target_period)
        .to_timestamp(how="start")
    )
    if begin_current_period > time_last_value:
        true_sum_since_current_period_begin = 0
    else:
        true_sum_since_current_period_begin = (
            true_values_series[begin_current_period:]
            .sum(axis=0)
            .first_value()
        )

    if weekly_proportion:
        prediction_for_overlapping_week = (
            weekly_proportion * predicted_series[-1].first_value()
        )
        if len(predicted_series) == 1:
            predictions_for_weeks_inside_period = 0
        else:
            predictions_for_weeks_inside_period = (
                predicted_series[:-1]
                .sum(axis=0)
                .first_value()
            )
        predicted_sum_from_now_until_end_of_period = (
            prediction_for_overlapping_week
            + predictions_for_weeks_inside_period
        )
    else:
        predicted_sum_from_now_until_end_of_period = (
            predicted_series.sum(axis=0).first_value()
        )

    return (
        true_sum_since_current_period_begin
        + predicted_sum_from_now_until_end_of_period
    )
