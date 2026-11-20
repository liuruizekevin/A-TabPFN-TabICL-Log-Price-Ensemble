"""Integrity and output protection for the delivered saved-prediction workflow."""
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ensemble import model


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for directory in ("data/raw", "data/iteration_06/selected", "config"):
            shutil.copytree(model.ROOT / directory, self.root / directory)
        self.patcher = patch.object(model, "ROOT", self.root)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temporary.cleanup()

    def test_replay_and_oof_metrics_match_preserved_results(self):
        result = model.verify()
        self.assertEqual(result["operation"], "saved_prediction_verification")
        self.assertEqual(result["rows"], 1459)
        self.assertLess(result["max_absolute_log_difference"], 1e-12)
        self.assertAlmostEqual(result["validation"]["pooled_oof_rmse"], 0.11688935816387115, places=12)
        prediction = model.predict()
        self.assertEqual(prediction["rows"], 1459)
        self.assertEqual(prediction["operation"], "saved_prediction_blend")
        self.assertIsNone(prediction["new_output_public_score"])
        exported = model.export_scored()
        self.assertEqual(exported["sha256"], result["scored_sha256"])

    def test_changed_source_is_rejected_before_writing(self):
        source = self.root / "data/iteration_06/selected/raw_tabpfn.csv"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            model.predict()
        self.assertFalse((self.root / "outputs").exists())

    def test_id_order_is_checked_even_if_manifest_is_updated(self):
        config_path = self.root / "config/model.json"
        config = json.loads(config_path.read_text())
        source = self.root / config["components"][0]["path"]
        lines = source.read_text().splitlines(keepends=True)
        lines[1], lines[2] = lines[2], lines[1]
        source.write_text("".join(lines))
        config["components"][0]["sha256"] = model.sha256(source)
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "Id order mismatch"):
            model.predict()

    def test_source_and_submission_directory_cannot_be_outputs(self):
        for name in ("data/raw/test.csv", "data/iteration_06/predictions.csv", "../outside.csv"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                model.export_scored(name)

    def test_existing_output_is_not_overwritten(self):
        output = self.root / "outputs/predictions.csv"
        output.parent.mkdir()
        output.write_bytes(b"independent user result")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            model.predict()
        self.assertEqual(output.read_bytes(), b"independent user result")

    def test_output_symlink_is_rejected(self):
        output = self.root / "outputs"
        output.mkdir()
        (output / "scored_submission.csv").symlink_to(self.root / "data/raw/test.csv")
        with self.assertRaises(ValueError):
            model.export_scored()


if __name__ == "__main__":
    unittest.main()
