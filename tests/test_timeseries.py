import unittest

import numpy as np
import pandas as pd

from fakes import FakeDataset, meter_readings
from process_inference_data import convert_inference_frames_to_ts
from util import convert_dataset_to_timeseries, parse_row


def parsed_rows(timestamps, values):
    return FakeDataset(
        {"ts": time.timestamp(), "value": value}
        for time, value in zip(pd.DatetimeIndex(timestamps).tz_localize("UTC"), values)
    )


class ConvertDatasetToTimeseriesTest(unittest.TestCase):
    def test_hourly_consumption_is_meter_difference_per_hour(self):
        timestamps = pd.date_range("2026-01-14 10:07", periods=24, freq="15min")
        values = np.arange(len(timestamps)) * 0.25

        series = convert_dataset_to_timeseries(parsed_rows(timestamps, values), "H")

        self.assertEqual(series.time_index[0], pd.Timestamp("2026-01-14 11:00"))
        np.testing.assert_allclose(series.univariate_values(), [1.0] * 4)

    def test_four_hour_blocks_are_aligned_to_utc_midnight(self):
        timestamps = pd.date_range("2026-01-14 01:00", periods=24, freq="h")

        series = convert_dataset_to_timeseries(
            parsed_rows(timestamps, np.arange(24.0)), "4H"
        )

        self.assertEqual(
            list(series.time_index),
            list(pd.date_range("2026-01-14 04:00", periods=5, freq="4h")),
        )
        np.testing.assert_allclose(series.univariate_values(), [4.0] * 5)

    def test_counter_reset_is_replaced_by_interpolated_consumption(self):
        timestamps = pd.date_range("2026-01-14 00:00", periods=8, freq="h")
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0]

        series = convert_dataset_to_timeseries(parsed_rows(timestamps, values), "H")

        np.testing.assert_allclose(series.univariate_values(), [1.0] * 7)

    def test_rejects_less_than_two_aligned_meter_values(self):
        timestamps = pd.DatetimeIndex(["2026-01-14 10:15", "2026-01-14 10:45"])

        with self.assertRaises(RuntimeError):
            convert_dataset_to_timeseries(parsed_rows(timestamps, [1.0, 2.0]), "H")


class ConvertInferenceFramesToTimeseriesTest(unittest.TestCase):
    def test_hourly_consumption_from_frames_of_several_inputs(self):
        first = pd.DataFrame(meter_readings("2026-01-14 00:00", "2026-01-14 01:45"))
        second = pd.DataFrame(meter_readings("2026-01-14 02:00", "2026-01-14 04:00"))
        second["Consumption"] += 2.0

        series = convert_inference_frames_to_ts([second, first], "H")

        self.assertEqual(series.time_index[0], pd.Timestamp("2026-01-14 00:00"))
        np.testing.assert_allclose(series.univariate_values(), [1.0] * 4)

    def test_readings_without_consumption_are_no_points(self):
        frame = pd.DataFrame(meter_readings("2026-01-14 00:00", "2026-01-14 00:30"))
        frame.loc[1:, "Consumption"] = np.nan

        with self.assertRaisesRegex(RuntimeError, "two timestamp/value points"):
            convert_inference_frames_to_ts([frame], "H")


class ParseRowTest(unittest.TestCase):
    def test_reads_consumption_with_time_as_utc_epoch_seconds(self):
        row = {"time": pd.Timestamp("2026-01-14 12:00"), "Consumption": "3.5"}

        self.assertEqual(
            parse_row(row),
            [{
                "ts": pd.Timestamp("2026-01-14 12:00", tz="UTC").timestamp(),
                "value": 3.5,
            }],
        )

    def test_skips_rows_without_consumption(self):
        row = {"time": pd.Timestamp("2026-01-14 12:00"), "Consumption": None}

        self.assertEqual(parse_row(row), [])


if __name__ == "__main__":
    unittest.main()
