import datetime
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import mlflow
import pandas as pd
from operator_lib.manifest import build_manifest
from operator_lib.util import MLOperator, logger
from operator_lib.util.helpers import HistoryNeedsRayError

from fakes import FakeDataset, FakeForecaster, FakeModel, meter_readings
from op import CustomConfig, Operator, parse_bool, parse_time_periods


def make_operator(time_periods="H", model="prophet", add_time_covariates="false"):
    def lib_init(self, *args, **kwargs):
        # The real init connects to Kafka, MLflow and Ray.
        self.config = kwargs["config"]

    operator = Operator()
    with mock.patch.object(MLOperator, "init", lib_init):
        operator.init(config=CustomConfig({
            "time_periods": time_periods,
            "model": model,
            "add_time_covariates": add_time_covariates,
        }))
    return operator


def receive(operator, timestamp, model=None, consumption=100.0):
    return operator.infer(
        model,
        {"Consumption": consumption},
        "",
        "",
        pd.Timestamp(timestamp, tz="UTC").to_pydatetime(),
    )


def recent_history(end):
    """Patches the read of recent history: 1 kWh per hour until end."""
    return mock.patch(
        "op.provide_historic_frames_local",
        return_value=[pd.DataFrame(meter_readings("2026-01-13 18:00", end))],
    )


def history_only_kafka_has(end):
    """Patches both reads for an input fed by another operator, whose history needs Ray."""
    without_ray = mock.patch(
        "op.provide_historic_frames_local",
        side_effect=HistoryNeedsRayError("only in Kafka"),
    )
    with_ray = mock.patch(
        "op.provide_historic_data_local",
        return_value=[FakeDataset(meter_readings("2026-01-13 18:00", end))],
    )
    return without_ray, with_ray


def utc(timestamp):
    return pd.Timestamp(timestamp).to_pydatetime()


class CustomConfigTest(unittest.TestCase):
    def test_defaults_match_old_operator(self):
        config = CustomConfig({})

        self.assertEqual(
            (config.time_periods, config.model, config.add_time_covariates),
            ("D", "prophet", False),
        )

    def test_reads_comma_separated_periods_once_each(self):
        self.assertEqual(parse_time_periods(" H, D,H "), ["H", "D"])

    def test_rejects_unknown_period_in_list(self):
        with self.assertRaises(ValueError):
            CustomConfig({"time_periods": "H,2H"})

    def test_rejects_list_without_period(self):
        with self.assertRaises(ValueError):
            CustomConfig({"time_periods": " , "})

    def test_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            CustomConfig({"model": "Prophet"})

    def test_reads_time_covariates_flag_from_string(self):
        self.assertEqual((parse_bool("False"), parse_bool("true")), (False, True))

    def test_rejects_time_covariates_flag_that_is_no_boolean(self):
        with self.assertRaises(ValueError):
            CustomConfig({"add_time_covariates": "maybe"})


class ManifestTest(unittest.TestCase):
    def test_declares_old_operator_input_and_config(self):
        manifest = build_manifest(Operator)

        self.assertEqual(manifest["inputs"], [{"name": "Consumption", "type": "float"}])
        self.assertEqual(manifest["config_values"], [
            {"name": "time_periods", "type": "string"},
            {"name": "model", "type": "string"},
            {"name": "add_time_covariates", "type": "bool"},
            {"name": "logger_level", "type": "string"},
        ])

    def test_declares_every_output_of_every_period(self):
        expected = []
        for name in Operator.period_translation_dict.values():
            expected += [
                {"name": f"{name}Prediction", "type": "float"},
                {"name": f"{name}PredictionTotal", "type": "float"},
                {"name": f"{name}Timestamp", "type": "string"},
            ]

        self.assertEqual(build_manifest(Operator)["outputs"], expected)


class InferTest(unittest.TestCase):
    def assertOutput(self, output, expected):
        self.assertEqual(set(output), set(expected))
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(output[key], value, msg=key)
            else:
                self.assertEqual(output[key], value, msg=key)

    def test_forecasts_current_hour_with_old_output_fields(self):
        operator = make_operator("H")
        model = FakeModel({"H": FakeForecaster("2026-01-14 09:00")})

        timestamp, output, _ = receive(operator, "2026-01-14 14:15", model)

        self.assertEqual(timestamp, utc("2026-01-14 14:15"))
        self.assertOutput(output, {
            "HourPrediction": 1.0,
            "HourPredictionTotal": 100.75,
            "HourTimestamp": "2026-01-14T14:59:59Z",
        })

    def test_does_not_forecast_again_within_same_hour(self):
        operator = make_operator("H")
        model = FakeModel({"H": FakeForecaster("2026-01-14 09:00")})
        receive(operator, "2026-01-14 14:15", model)

        self.assertEqual(
            receive(operator, "2026-01-14 14:30", model), (None, None, None)
        )

    def test_forecasts_again_once_hour_moves_on(self):
        operator = make_operator("H")
        model = FakeModel({"H": FakeForecaster("2026-01-14 09:00")})
        receive(operator, "2026-01-14 14:15", model)

        _, output, _ = receive(operator, "2026-01-14 15:05", model)

        self.assertEqual(output["HourTimestamp"], "2026-01-14T15:59:59Z")

    def test_combines_all_configured_periods_in_one_message(self):
        operator = make_operator("H,D")
        model = FakeModel({
            "H": FakeForecaster("2026-01-14 09:00"),
            "4H": FakeForecaster("2026-01-14 04:00", value=4.0),
        })

        with recent_history("2026-01-14 14:15"):
            _, output, _ = receive(operator, "2026-01-14 14:15", model)

        # The Berlin day ends at 23:00 UTC, 8.75 hours after the message.
        self.assertOutput(output, {
            "HourPrediction": 1.0,
            "HourPredictionTotal": 100.75,
            "HourTimestamp": "2026-01-14T14:59:59Z",
            "DayPrediction": 24.0,
            "DayPredictionTotal": 108.75,
            "DayTimestamp": "2026-01-14T22:59:59Z",
        })

    def test_repeats_unchanged_day_forecast_when_only_hour_moves_on(self):
        operator = make_operator("H,D")
        model = FakeModel({
            "H": FakeForecaster("2026-01-14 09:00"),
            "4H": FakeForecaster("2026-01-14 04:00", value=4.0),
        })

        with recent_history("2026-01-14 15:05") as read_history:
            _, first, _ = receive(operator, "2026-01-14 14:15", model)
            _, second, _ = receive(operator, "2026-01-14 15:05", model)

        self.assertEqual(second["HourTimestamp"], "2026-01-14T15:59:59Z")
        self.assertEqual(
            {key: second[key] for key in first if key.startswith("Day")},
            {key: first[key] for key in first if key.startswith("Day")},
        )
        self.assertEqual(read_history.call_count, 1)

    def test_forecasts_day_again_when_berlin_day_starts(self):
        operator = make_operator("D")
        model = FakeModel({"4H": FakeForecaster("2026-01-14 08:00", value=4.0)})

        with recent_history("2026-01-14 23:10"):
            _, before, _ = receive(operator, "2026-01-14 22:30", model)
            _, after, _ = receive(operator, "2026-01-14 23:10", model)

        self.assertEqual(before["DayTimestamp"], "2026-01-14T22:59:59Z")
        self.assertEqual(after["DayTimestamp"], "2026-01-15T22:59:59Z")
        self.assertAlmostEqual(after["DayPrediction"], 24.0)

    def test_does_not_forecast_with_model_of_other_model_type(self):
        operator = make_operator("H", model="prophet")
        model = FakeModel({"H": FakeForecaster("2026-01-14 09:00", model_type="nhits")})

        self.assertEqual(
            receive(operator, "2026-01-14 14:15", model), (None, None, None)
        )

    def test_ignores_message_without_consumption(self):
        operator = make_operator("H")
        model = FakeModel({"H": FakeForecaster("2026-01-14 09:00")})

        result = operator.infer(
            model, {"value": 1.0}, "", "", utc("2026-01-14 14:15")
        )

        self.assertEqual(result, (None, None, None))


class InferenceHistoryTest(unittest.TestCase):
    model = FakeModel({"4H": FakeForecaster("2026-01-14 04:00", value=4.0)})

    def test_reads_history_without_ray(self):
        operator = make_operator("D")

        with recent_history("2026-01-14 14:15"), \
                mock.patch("op.provide_historic_data_local") as read_with_ray:
            _, output, _ = receive(operator, "2026-01-14 14:15", self.model)

        self.assertAlmostEqual(output["DayPrediction"], 24.0)
        read_with_ray.assert_not_called()

    def test_reads_history_only_kafka_has_with_ray_and_warns(self):
        operator = make_operator("D")
        without_ray, with_ray = history_only_kafka_has("2026-01-14 14:15")

        with without_ray, with_ray, self.assertLogs(logger, "WARNING"):
            _, output, _ = receive(operator, "2026-01-14 14:15", self.model)

        self.assertAlmostEqual(output["DayPrediction"], 24.0)

    def test_tries_without_ray_and_warns_only_once(self):
        operator = make_operator("D")
        without_ray, with_ray = history_only_kafka_has("2026-01-14 23:10")

        with without_ray as read_without_ray, with_ray as read_with_ray, \
                self.assertLogs(logger, "WARNING") as logs:
            receive(operator, "2026-01-14 14:15", self.model)
            receive(operator, "2026-01-14 23:10", self.model)

        self.assertEqual(
            (read_without_ray.call_count, read_with_ray.call_count, len(logs.records)),
            (1, 2, 1),
        )


class NeedRetrainingTest(unittest.TestCase):
    def test_retrains_without_model(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(None))

    def test_retrains_forecaster_older_than_retraining_interval(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel({"H": FakeForecaster("2026-01-13 10:00")})
        ))

    def test_keeps_recent_forecaster(self):
        operator = make_operator("H")
        receive(operator, "2026-01-14 12:00")

        self.assertFalse(operator.need_retraining(
            FakeModel({"H": FakeForecaster("2026-01-14 10:00")})
        ))

    def test_retrains_when_a_configured_period_has_no_forecaster(self):
        operator = make_operator("H,D")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel({"H": FakeForecaster("2026-01-14 10:00")})
        ))

    def test_keeps_weekly_forecaster_for_28_days(self):
        operator = make_operator("Y")
        receive(operator, "2026-03-01 12:00")

        self.assertFalse(operator.need_retraining(
            FakeModel({"W": FakeForecaster("2026-02-08")})
        ))

    def test_retrains_weekly_forecaster_after_28_days(self):
        operator = make_operator("Y")
        receive(operator, "2026-03-01 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel({"W": FakeForecaster("2026-01-29")})
        ))

    def test_retrains_forecaster_trained_without_configured_time_covariates(self):
        operator = make_operator("H", add_time_covariates="true")
        receive(operator, "2026-01-14 12:00")

        self.assertTrue(operator.need_retraining(
            FakeModel({"H": FakeForecaster("2026-01-14 10:00")})
        ))


class TrainTest(unittest.TestCase):
    def train(self, operator, model=None):
        """Trains with history and Ray replaced; returns the result and the training mock."""
        with mock.patch("op.provide_historic_data", return_value=["dataset"]) as read_history, \
                mock.patch("op.train_forecasters") as training, \
                mock.patch("op.ray.get", side_effect=lambda ref: ref):
            training.remote.side_effect = (
                lambda datasets, model_type, frequencies, covariates, mlflow_logger:
                {frequency: f"trained {frequency}" for frequency in frequencies}
            )
            result = operator.train(model, None)
        return result, training, read_history

    def fail_training(self, operator):
        with mock.patch(
            "op.provide_historic_data",
            side_effect=RuntimeError("timescale unreachable"),
        ), self.assertLogs(logger, "ERROR"):
            return operator.train(None, None)

    def test_trains_only_missing_and_outdated_frequencies(self):
        operator = make_operator("H,D")
        receive(operator, "2026-01-14 12:00")
        current = FakeForecaster("2026-01-14 10:00")

        result, training, _ = self.train(operator, FakeModel({"H": current}))

        self.assertEqual(result.forecasters, {"H": current, "4H": "trained 4H"})
        self.assertEqual(training.remote.call_args.args[2], ["4H"])

    def test_reads_history_once_per_training_duration(self):
        operator = make_operator("D,Y")

        _, training, read_history = self.train(operator)

        self.assertEqual(
            [call.args[0] for call in read_history.call_args_list],
            [datetime.timedelta(days=14), datetime.timedelta(days=365)],
        )
        self.assertEqual(
            [call.args[2] for call in training.remote.call_args_list],
            [["4H"], ["W"]],
        )

    def test_trains_four_hour_period_on_hourly_raster(self):
        operator = make_operator("4H")

        _, training, _ = self.train(operator)

        self.assertEqual(training.remote.call_args.args[2], ["H"])

    def test_passes_model_type_and_time_covariates_to_training(self):
        operator = make_operator("D", model="nhits", add_time_covariates="True")

        _, training, _ = self.train(operator)

        _, model_type, _, add_time_covariates, _ = training.remote.call_args.args
        self.assertEqual((model_type, add_time_covariates), ("nhits", True))

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
