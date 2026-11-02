import importlib
import tempfile
import unittest
import warnings
from pathlib import Path

import mlflow
import pandas as pd

import build_models


class FakeDartsModel:
    def __init__(self, first=0):
        self.first = first

    def predict(self, n):
        return list(range(self.first, self.first + n))


def forecaster(first=0):
    return build_models.Forecaster(
        FakeDartsModel(first), "prophet", False, pd.Timestamp("2026-01-14 10:00")
    )


class ForecastingModelsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.model_path = Path(tmp.name, "model").as_posix()

        # Saving a model touches the tracking store, keep it out of the repo.
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(Path(tmp.name).as_uri())
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)

    def save_and_load(self, python_model):
        mlflow.pyfunc.save_model(self.model_path, python_model=python_model)
        return mlflow.pyfunc.load_model(self.model_path)

    def test_reloaded_model_forecasts_with_the_requested_frequency(self):
        model = self.save_and_load(build_models.ForecastingModels({
            "H": forecaster(first=0),
            "4H": forecaster(first=100),
        }))

        self.assertEqual(
            model.predict({"missing_steps": 3, "timeseries_frequency": "4H"}),
            [100, 101, 102],
        )

    def test_reloaded_model_keeps_its_forecasters(self):
        model = self.save_and_load(build_models.ForecastingModels({
            "H": forecaster(),
        }))

        reloaded = model.unwrap_python_model().forecasters["H"]
        self.assertEqual(reloaded.model_type, "prophet")
        self.assertEqual(reloaded.last_ts, pd.Timestamp("2026-01-14 10:00"))

    def test_rejects_frequency_without_forecaster(self):
        model = build_models.ForecastingModels({"H": forecaster()})

        with self.assertRaises(ValueError):
            model.predict({"missing_steps": 3, "timeseries_frequency": "D"})

    def test_predict_signature_is_accepted_by_mlflow(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(build_models)

        # MLflow still recommends adding type hints, but none fits a dict input.
        self.assertEqual(
            [
                str(w.message) for w in caught
                if "invalid parameters" in str(w.message)
                or "not supported" in str(w.message)
            ],
            [],
        )


class GetMissingStepsTest(unittest.TestCase):
    def test_rejects_zero_steps(self):
        with self.assertRaises(ValueError):
            build_models.get_missing_steps({"missing_steps": 0})

    def test_rejects_boolean_steps(self):
        with self.assertRaises(ValueError):
            build_models.get_missing_steps({"missing_steps": True})


if __name__ == "__main__":
    unittest.main()
