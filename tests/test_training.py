import logging
import unittest

import darts
import numpy as np
import pandas as pd

from training import build_model, fit_forecaster, time_covariate_encoders


def hourly_series(days):
    index = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    hours = np.arange(len(index))
    return darts.TimeSeries.from_times_and_values(
        index, 1 + 0.5 * np.sin(hours / 24 * 2 * np.pi)
    )


class TimeCovariateEncodersTest(unittest.TestCase):
    def attributes(self, timeseries_frequency):
        encoders = time_covariate_encoders(timeseries_frequency, "future")
        return encoders["datetime_attribute"]["future"]

    def test_hourly_raster_gets_hour_weekday_and_month(self):
        self.assertEqual(self.attributes("H"), ["month", "dayofweek", "hour"])

    def test_daily_raster_gets_weekday_and_month(self):
        self.assertEqual(self.attributes("D"), ["month", "dayofweek"])

    def test_weekly_raster_gets_month_only(self):
        self.assertEqual(self.attributes("W"), ["month"])


class BuildModelTest(unittest.TestCase):
    def test_nhits_takes_time_covariates_as_past_covariates(self):
        # NHiTS supports past covariates only.
        model = build_model("nhits", "D", add_time_covariates=True)

        self.assertEqual(
            list(model.model_params["add_encoders"]["datetime_attribute"]),
            ["past"],
        )

    def test_prophet_takes_time_covariates_as_future_covariates(self):
        model = build_model("prophet", "D", add_time_covariates=True)

        self.assertEqual(
            list(model.model_params["add_encoders"]["datetime_attribute"]),
            ["future"],
        )


class FitForecasterTest(unittest.TestCase):
    def setUp(self):
        # Prophet and cmdstanpy log every fit.
        logging.disable(logging.INFO)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_prophet_with_time_covariates_forecasts_after_training_data(self):
        series = hourly_series(days=3)

        forecaster = fit_forecaster(series, "prophet", "H", add_time_covariates=True)
        forecast = forecaster.model.predict(3)

        self.assertEqual(forecaster.last_ts, series.time_index[-1])
        self.assertTrue(forecaster.add_time_covariates)
        self.assertEqual(
            forecast.time_index[0], series.time_index[-1] + pd.Timedelta(hours=1)
        )
        self.assertEqual(len(forecast), 3)

    def test_rejects_series_shorter_than_the_model_needs(self):
        with self.assertRaises(RuntimeError):
            fit_forecaster(hourly_series(days=1)[:5], "nhits", "H", add_time_covariates=False)


if __name__ == "__main__":
    unittest.main()
