import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import darts
import mlflow
import numpy as np
import pandas as pd
from operator_lib.util import MLOperator, logger

from op import CustomConfig, Operator


class FakeModel:
    """Stands in for the MLflow model wrapping a forecasting model."""

    def __init__(self, last_ts, freq="h", **attributes):
        self.python_model = SimpleNamespace(
            last_ts=pd.Timestamp(last_ts), **attributes
        )
        self.freq = freq

    def unwrap_python_model(self):
        return self.python_model

    def predict(self, data):
        steps = data["missing_steps"]
        start = self.python_model.last_ts + pd.tseries.frequencies.to_offset(
            self.freq
        )
        index = pd.date_range(start, periods=steps, freq=self.freq)
        return darts.TimeSeries.from_times_and_values(index, np.ones(steps))


def make_operator(time_period="H", model_type="prophet"):
    def lib_init(self, *args, **kwargs):
        # The real init connects to Kafka, MLflow and Ray.
        self.config = kwargs["config"]

    operator = Operator()
    with mock.patch.object(MLOperator, "init", lib_init):
        operator.init(config=CustomConfig({
            "time_period": time_period,
            "model_type": model_type,
        }))
    return operator


def receive(operator, timestamp, model=None):
    return operator.infer(
        model,
        {"value": 100.0},
        "",
        "",
        pd.Timestamp(timestamp, tz="UTC").to_pydatetime(),
    )


class CustomConfigTest(unittest.TestCase):
    def test_rejects_unknown_time_period(self):
        with self.assertRaises(ValueError):
            CustomConfig({"time_period": "2H"})

    def test_rejects_unknown_model_type(self):
        with self.assertRaises(ValueError):
            CustomConfig({"model_type": "Prophet"})


class InferTest(unittest.TestCase):
    def test_predicts_current_hour_once(self):
        operator = make_operator("H")
        model = FakeModel("2026-01-14 09:00", timeseries_frequency="H")

        timestamp, output, _ = receive(operator, "2026-01-14 14:15", model)

        self.assertEqual(
            timestamp,
            pd.Timestamp("2026-01-14 14:59:59.999999").to_pydatetime(),
        )
        self.assertEqual(
            output,
            {"HourPrediction": 1.0, "HourPredictionTotal": 100.75},
        )
        self.assertEqual(
            receive(operator, "2026-01-14 14:30", model), (None, None, None)
        )

    def test_does_not_predict_with_four_hour_model_from_before_hourly_raster(self):
        operator = make_operator("4H")
        legacy_model = FakeModel("2026-01-14 08:00", freq="4h")

        self.assertEqual(
            receive(operator, "2026-01-14 14:15", legacy_model),
            (None, None, None),
        )


class NeedRetrainingTest(unittest.TestCase):
    def test_retrains_without_model(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(None))

    def test_retrains_model_older_than_retraining_interval(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel("2026-01-13 10:00", timeseries_frequency="H")
        ))

    def test_keeps_recent_model(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertFalse(operator.need_retraining(
            FakeModel("2026-01-14 10:00", timeseries_frequency="H")
        ))

    def test_retrains_model_with_other_timeseries_frequency(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel("2026-01-14 08:00", freq="4h", timeseries_frequency="4H")
        ))

    def test_retrains_four_hour_model_from_before_hourly_raster(self):
        operator = make_operator("4H")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel("2026-01-14 08:00", freq="4h")
        ))

    def test_keeps_recent_model_of_other_period_without_stored_frequency(self):
        operator = make_operator("D")
        receive(operator, "2026-01-14 12:00")

        self.assertFalse(operator.need_retraining(
            FakeModel("2026-01-14 08:00", freq="4h")
        ))


class TrainTest(unittest.TestCase):
    def fail_training(self, operator):
        with mock.patch(
            "op.provide_historic_data",
            side_effect=RuntimeError("timescale unreachable"),
        ), self.assertLogs(logger, "ERROR"):
            return operator.train(None, None)

    def test_trains_four_hour_period_on_hourly_raster(self):
        operator = make_operator("4H")

        with mock.patch("op.provide_historic_data", return_value=["dataset"]), \
                mock.patch("op.train_forecasting_model") as training, \
                mock.patch("op.ray.get", side_effect=lambda ref: ref):
            operator.train(None, None)

        _, _, timeseries_frequency, _ = training.remote.call_args.args
        self.assertEqual(timeseries_frequency, "H")

    def test_failed_training_returns_no_model_instead_of_raising(self):
        operator = make_operator("H")

        self.assertIsNone(self.fail_training(operator))

    def test_waits_an_hour_after_failed_training_before_retrying(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        with mock.patch("op.time.monotonic", return_value=1000.0):
            self.fail_training(operator)

        with mock.patch("op.time.monotonic", return_value=1000.0 + 59 * 60):
            self.assertFalse(operator.need_retraining(None))
        with mock.patch("op.time.monotonic", return_value=1000.0 + 60 * 60):
            self.assertTrue(operator.need_retraining(None))

    def test_ends_mlflow_run_left_open_by_failed_training(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(Path(tmp.name).as_uri())
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)
        mlflow.set_experiment("test")
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        # operator-lib starts the run before it calls train().
        run = mlflow.start_run()
        self.addCleanup(mlflow.end_run)
        self.fail_training(operator)
        after_retry_delay = time.monotonic() + 60 * 60
        with mock.patch("op.time.monotonic", return_value=after_retry_delay):
            self.assertTrue(operator.need_retraining(None))

        self.assertIsNone(mlflow.active_run())
        self.assertEqual(
            mlflow.MlflowClient().get_run(run.info.run_id).info.status,
            "FAILED",
        )


if __name__ == "__main__":
    unittest.main()
