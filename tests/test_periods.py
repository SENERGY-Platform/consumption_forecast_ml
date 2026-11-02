import unittest

import pandas as pd

from process_inference_data import (
    format_period_end,
    get_period_bounds,
    get_period_end,
    get_raster_start,
)


class GetPeriodBoundsTest(unittest.TestCase):
    def assertBounds(self, timestamp, period, start, end):
        self.assertEqual(
            get_period_bounds(pd.Timestamp(timestamp), period),
            (pd.Timestamp(start), pd.Timestamp(end)),
        )

    def test_hour_is_the_utc_hour(self):
        self.assertBounds(
            "2026-01-14 13:30", "H", "2026-01-14 13:00", "2026-01-14 14:00"
        )

    def test_hour_within_repeated_dst_hour_is_a_regular_utc_hour(self):
        # 01:30 UTC on 25 Oct 2026 is the second 02:30 in Berlin.
        self.assertBounds(
            "2026-10-25 01:30", "H", "2026-10-25 01:00", "2026-10-25 02:00"
        )

    def test_four_hour_block_starts_at_berlin_midnight_in_winter(self):
        self.assertBounds(
            "2026-01-13 23:30", "4H", "2026-01-13 23:00", "2026-01-14 03:00"
        )

    def test_four_hour_block_starts_at_berlin_midnight_in_summer(self):
        self.assertBounds(
            "2026-07-14 22:30", "4H", "2026-07-14 22:00", "2026-07-15 02:00"
        )

    def test_four_hour_block_lasts_three_hours_when_clocks_go_forward(self):
        self.assertBounds(
            "2026-03-29 00:30", "4H", "2026-03-28 23:00", "2026-03-29 02:00"
        )

    def test_four_hour_block_lasts_five_hours_when_clocks_go_back(self):
        self.assertBounds(
            "2026-10-25 00:30", "4H", "2026-10-24 22:00", "2026-10-25 03:00"
        )

    def test_day_starts_at_berlin_midnight_in_winter(self):
        self.assertBounds(
            "2026-01-14 12:00", "D", "2026-01-13 23:00", "2026-01-14 23:00"
        )

    def test_day_starts_at_berlin_midnight_in_summer(self):
        self.assertBounds(
            "2026-07-14 12:00", "D", "2026-07-13 22:00", "2026-07-14 22:00"
        )

    def test_late_utc_evening_belongs_to_next_berlin_day(self):
        self.assertBounds(
            "2026-01-14 23:30", "D", "2026-01-14 23:00", "2026-01-15 23:00"
        )

    def test_day_lasts_23_hours_when_clocks_go_forward(self):
        self.assertBounds(
            "2026-03-29 12:00", "D", "2026-03-28 23:00", "2026-03-29 22:00"
        )

    def test_day_lasts_25_hours_when_clocks_go_back(self):
        self.assertBounds(
            "2026-10-25 12:00", "D", "2026-10-24 22:00", "2026-10-25 23:00"
        )

    def test_week_starts_on_monday_in_berlin(self):
        # Wednesday 14 Jan 2026; the week starts on Monday 12 Jan 00:00 CET.
        self.assertBounds(
            "2026-01-14 12:00", "W", "2026-01-11 23:00", "2026-01-18 23:00"
        )

    def test_month_starts_at_berlin_midnight_across_dst_change(self):
        self.assertBounds(
            "2026-03-15 12:00", "M", "2026-02-28 23:00", "2026-03-31 22:00"
        )

    def test_year_starts_at_berlin_midnight(self):
        self.assertBounds(
            "2026-06-01 12:00", "Y", "2025-12-31 23:00", "2026-12-31 23:00"
        )


class GetPeriodEndTest(unittest.TestCase):
    def test_period_end_is_just_before_next_period(self):
        self.assertEqual(
            get_period_end(pd.Timestamp("2026-01-14 12:00"), "D"),
            pd.Timestamp("2026-01-14 23:00") - pd.Timedelta(nanoseconds=1),
        )


class FormatPeriodEndTest(unittest.TestCase):
    def test_writes_end_of_berlin_day_in_utc_with_z(self):
        self.assertEqual(
            format_period_end(pd.Timestamp("2026-01-14 12:00"), "D"),
            "2026-01-14T22:59:59Z",
        )


class GetRasterStartTest(unittest.TestCase):
    def assertRasterStart(self, timestamp, timeseries_frequency, start):
        self.assertEqual(
            get_raster_start(pd.Timestamp(timestamp), timeseries_frequency),
            pd.Timestamp(start),
        )

    def test_hourly_raster_starts_at_full_hour(self):
        self.assertRasterStart("2026-01-14 13:30", "H", "2026-01-14 13:00")

    def test_four_hour_raster_is_aligned_to_utc_midnight(self):
        self.assertRasterStart("2026-01-14 23:30", "4H", "2026-01-14 20:00")

    def test_daily_raster_starts_at_utc_midnight(self):
        self.assertRasterStart("2026-01-14 23:30", "D", "2026-01-14 00:00")

    def test_weekly_raster_starts_on_sunday(self):
        # Wednesday 14 Jan 2026; the resampled weekly series runs from Sunday.
        self.assertRasterStart("2026-01-14 12:00", "W", "2026-01-11 00:00")

    def test_weekly_raster_on_sunday_starts_that_day(self):
        self.assertRasterStart("2026-01-11 12:00", "W", "2026-01-11 00:00")


if __name__ == "__main__":
    unittest.main()
