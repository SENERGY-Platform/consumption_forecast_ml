import unittest

import numpy as np
import pandas as pd

from util import convert_dataset_to_timeseries, parse_row


class FakeDataset:
    """Stands in for a ray Dataset, of which only to_pandas() is used."""

    def __init__(self, timestamps, values):
        self.df = pd.DataFrame({
            "ts": pd.DatetimeIndex(timestamps).astype("int64") / 1e9,
            "value": values,
        })

    def to_pandas(self):
        return self.df.copy()


class ConvertDatasetToTimeseriesTest(unittest.TestCase):
    def test_hourly_consumption_is_meter_difference_per_hour(self):
        timestamps = pd.date_range("2026-01-14 10:07", periods=24, freq="15min")
        values = np.arange(len(timestamps)) * 0.25

        series = convert_dataset_to_timeseries(
            FakeDataset(timestamps, values), "H"
        )

        self.assertEqual(series.time_index[0], pd.Timestamp("2026-01-14 11:00"))
        np.testing.assert_allclose(series.univariate_values(), [1.0] * 4)

    def test_four_hour_blocks_are_aligned_to_utc_midnight(self):
        timestamps = pd.date_range("2026-01-14 01:00", periods=24, freq="h")

        series = convert_dataset_to_timeseries(
            FakeDataset(timestamps, np.arange(24.0)), "4H"
        )

        self.assertEqual(
            list(series.time_index),
            list(pd.date_range("2026-01-14 04:00", periods=5, freq="4h")),
        )
        np.testing.assert_allclose(series.univariate_values(), [4.0] * 5)

    def test_counter_reset_is_replaced_by_interpolated_consumption(self):
        timestamps = pd.date_range("2026-01-14 00:00", periods=8, freq="h")
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0]

        series = convert_dataset_to_timeseries(
            FakeDataset(timestamps, values), "H"
        )

        np.testing.assert_allclose(series.univariate_values(), [1.0] * 7)

    def test_rejects_less_than_two_aligned_meter_values(self):
        timestamps = pd.DatetimeIndex(["2026-01-14 10:15", "2026-01-14 10:45"])

        with self.assertRaises(RuntimeError):
            convert_dataset_to_timeseries(
                FakeDataset(timestamps, [1.0, 2.0]), "H"
            )


class ParseRowTest(unittest.TestCase):
    def test_reads_time_column_as_utc_epoch_seconds(self):
        row = {"time": pd.Timestamp("2026-01-14 12:00"), "value": "3.5"}

        self.assertEqual(
            parse_row(row),
            [{
                "ts": pd.Timestamp("2026-01-14 12:00", tz="UTC").timestamp(),
                "value": 3.5,
            }],
        )

    def test_skips_rows_without_value(self):
        row = {"time": pd.Timestamp("2026-01-14 12:00"), "value": None}

        self.assertEqual(parse_row(row), [])


if __name__ == "__main__":
    unittest.main()
