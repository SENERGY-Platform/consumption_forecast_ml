import unittest

import darts
import numpy as np
import pandas as pd

from inference import (
    compute_output_nr,
    compute_period_forecast,
    predict_multi_output,
    predict_single_output,
    sum_series_over_window,
)


def series(start, freq, values):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return darts.TimeSeries.from_times_and_values(
        index, np.asarray(values, dtype=float)
    )


class FakeModel:
    """Stands in for the MLflow model: forecasts the given values after last_ts."""

    def __init__(self, last_ts, freq, values):
        self.last_ts = pd.Timestamp(last_ts)
        self.freq = freq
        self.values = values

    def unwrap_python_model(self):
        return self

    def predict(self, data):
        steps = data["missing_steps"]
        start = self.last_ts + pd.tseries.frequencies.to_offset(self.freq)
        return series(start, self.freq, self.values[:steps])


class ComputeOutputNrTest(unittest.TestCase):
    def test_counts_steps_from_model_end_through_current_hour(self):
        self.assertEqual(
            compute_output_nr(
                "H",
                "H",
                pd.Timestamp("2026-01-14 09:00"),
                pd.Timestamp("2026-01-14 14:15"),
            ),
            5,
        )

    def test_counts_steps_up_to_end_of_berlin_day(self):
        # The Berlin day ends at 23:00 UTC in winter: hours 09:00 to 22:00.
        self.assertEqual(
            compute_output_nr(
                "H",
                "D",
                pd.Timestamp("2026-01-14 08:00"),
                pd.Timestamp("2026-01-14 14:30"),
            ),
            14,
        )


class SumSeriesOverWindowTest(unittest.TestCase):
    def test_counts_partially_overlapping_intervals_proportionally(self):
        blocks = series("2026-01-13 20:00", "4h", [4.0, 8.0])

        total = sum_series_over_window(
            blocks,
            pd.Timestamp("2026-01-13 23:00"),
            pd.Timestamp("2026-01-14 03:00"),
            "4H",
        )

        self.assertAlmostEqual(total, 1.0 + 6.0)


class PredictSingleOutputTest(unittest.TestCase):
    payload = {"timestamp": pd.Timestamp("2026-01-14 14:15"), "value": 100.0}

    def test_returns_current_hour_and_its_remaining_part(self):
        model = FakeModel("2026-01-14 09:00", "h", [9.0, 9.0, 9.0, 9.0, 2.0])

        forecast = predict_single_output(model, self.payload, "H", "H")

        self.assertAlmostEqual(forecast.period_consumption, 2.0)
        self.assertAlmostEqual(forecast.remaining_consumption, 1.5)

    def test_counts_negative_forecasts_as_zero(self):
        model = FakeModel("2026-01-14 09:00", "h", [1.0, 1.0, 1.0, 1.0, -3.0])

        forecast = predict_single_output(model, self.payload, "H", "H")

        self.assertEqual(forecast.period_consumption, 0.0)


class ComputePeriodForecastTest(unittest.TestCase):
    def test_day_combines_measured_and_predicted_blocks_of_berlin_day(self):
        # The Berlin day runs from 2026-01-13 23:00 to 2026-01-14 23:00 UTC.
        measured = series("2026-01-13 20:00", "4h", [4.0, 4.0, 4.0, 4.0])
        predicted = series("2026-01-14 12:00", "4h", [1.0, 1.0, 1.0])

        forecast = compute_period_forecast(
            predicted,
            measured,
            "D",
            pd.Timestamp("2026-01-14 14:30"),
            "4H",
        )

        # Measured: 1/4 of the 20:00 block and the three blocks until 12:00.
        # Predicted: the 12:00 and 16:00 blocks and 3/4 of the 20:00 block.
        self.assertAlmostEqual(forecast.period_consumption, 13.0 + 2.75)
        self.assertAlmostEqual(
            forecast.remaining_consumption, 0.375 + 1.0 + 0.75
        )


class PredictMultiOutputTest(unittest.TestCase):
    def test_four_hour_block_on_hourly_raster_has_no_partial_hours(self):
        # 23:30 UTC is 00:30 in Berlin: the block runs from 23:00 to 03:00 UTC.
        # The model forecasts the hours 11:00 to 02:00 with values 0 to 15.
        model = FakeModel("2026-01-13 10:00", "h", [float(i) for i in range(16)])
        measured = series("2026-01-13 21:00", "h", [2.0, 2.0])

        forecast = predict_multi_output(
            model,
            {"timestamp": pd.Timestamp("2026-01-13 23:30"), "value": 100.0},
            "H",
            "4H",
            measured,
        )

        self.assertAlmostEqual(forecast.period_consumption, 12 + 13 + 14 + 15)
        self.assertAlmostEqual(
            forecast.remaining_consumption, 6 + 13 + 14 + 15
        )


if __name__ == "__main__":
    unittest.main()
