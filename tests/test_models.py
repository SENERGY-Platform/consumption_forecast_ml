import importlib
import tempfile
import unittest
import warnings
from pathlib import Path

import mlflow
import pandas as pd

import build_models


class FakeDartsModel:
    def predict(self, n):
        return list(range(n))


class ForecastingModelTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.model_path = Path(tmp.name, "model").as_posix()

        # Saving a model touches the tracking store, keep it out of the repo.
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(
            f"sqlite:///{Path(tmp.name, 'mlflow.db').as_posix()}"
        )
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)

    def save_and_load(self, python_model):
        mlflow.pyfunc.save_model(self.model_path, python_model=python_model)
        return mlflow.pyfunc.load_model(self.model_path)

    def test_reloaded_model_predicts_requested_number_of_steps(self):
        model = self.save_and_load(build_models.ProphetForecastingModel(
            FakeDartsModel(), pd.Timestamp("2026-01-14 10:00"), "H"
        ))

        self.assertEqual(model.predict({"missing_steps": 3}), [0, 1, 2])

    def test_reloaded_model_keeps_its_timeseries_frequency(self):
        model = self.save_and_load(build_models.NHitsForecastingModel(
            FakeDartsModel(), pd.Timestamp("2026-01-14 10:00"), "4H"
        ))

        self.assertEqual(model.unwrap_python_model().timeseries_frequency, "4H")

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
