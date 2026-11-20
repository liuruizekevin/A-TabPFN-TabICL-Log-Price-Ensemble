"""Validate and recompute the selected ensemble from preserved component predictions.

All paths resolve from the checkout, independent of the current working directory.
This module combines the saved component predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WEIGHTS = {"raw_tabpfn": 0.64, "domain_tabpfn": 0.16, "tabicl": 0.20}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Invalid relative path: {relative}")
    result = ROOT / path
    require(result.resolve().is_relative_to(ROOT.resolve()), f"Path escapes checkout: {relative}")
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config():
    config = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    require(config["schema_version"] == 1, "Unsupported configuration schema")
    components = config["components"]
    require(len(components) == 3 and {c["key"]: c["weight"] for c in components} == EXPECTED_WEIGHTS,
            "The selected model must use the fixed 64/16/20 weights")
    require(config["submission"]["rows"] == 1459, "Unexpected official test row count")
    return config


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        columns = reader.fieldnames
    require(columns and len(set(columns)) == len(columns), f"Invalid CSV header: {path}")
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f"Invalid CSV row width: {path}")
    return columns, rows


def checked_inputs(config):
    for item in [*config["inputs"].values(), *config["components"]]:
        require(sha256(local(item["path"])) == item["sha256"], f"SHA-256 mismatch: {item['path']}")
    train_columns, train = read_table(local(config["inputs"]["train"]["path"]))
    test_columns, test = read_table(local(config["inputs"]["test"]["path"]))
    require(len(train) == 1460 and len(train_columns) == 81, "Expected 1460 labeled rows and 79 predictors")
    require(test_columns == [c for c in train_columns if c != "SalePrice"], "Train/test schema mismatch")
    require(len(test_columns) == 80 and "SalePrice" not in test_columns, "Invalid test schema")
    train_ids, ids = [r["Id"] for r in train], [r["Id"] for r in test]
    require(len(set(train_ids)) == 1460, "Duplicate training identifiers")
    require(len(ids) == len(set(ids)) == config["submission"]["rows"], "Invalid test identifiers")
    require(not set(train_ids) & set(ids), "Train and test identifiers overlap")
    return train, ids


def ordered_values(path, ids, column):
    columns, rows = read_table(path)
    require("Id" in columns and column in columns, f"Missing prediction columns: {path}")
    if column == "SalePrice":
        require(columns == ["Id", column], f"Unexpected submission columns: {path}")
    require([r["Id"] for r in rows] == ids, f"Prediction Id order mismatch: {path}")
    values = [float(r[column]) for r in rows]
    require(all(math.isfinite(v) for v in values), f"Non-finite prediction: {path}")
    if column == "SalePrice":
        require(all(v > 0 for v in values), "SalePrice predictions must be positive")
    return values


def prediction_sources(config):
    _, ids = checked_inputs(config)
    logs = {c["key"]: ordered_values(local(c["path"]), ids, "log_prediction") for c in config["components"]}
    scored = local(config["inputs"]["scored"]["path"])
    require(sha256(scored) == config["submission"]["sha256"], "Original scored CSV was changed")
    reference = ordered_values(scored, ids, "SalePrice")
    combined = [sum(c["weight"] * logs[c["key"]][i] for c in config["components"]) for i in range(len(ids))]
    difference = max(abs(value - math.log(old)) for value, old in zip(combined, reference))
    require(difference <= 1e-12, "Saved components do not reproduce the original selected blend")
    return ids, combined, difference


def validation_metrics():
    config = load_config()
    train, _ = checked_inputs(config)
    columns, rows = read_table(local(config["inputs"]["oof"]["path"]))
    require(columns == ["Id", "fold", "log_SalePrice", "log_prediction"], "Invalid OOF schema")
    require([r["Id"] for r in rows] == [r["Id"] for r in train], "OOF training Id order mismatch")
    squared_errors = {fold: [] for fold in range(1, 6)}
    for row, original in zip(rows, train):
        fold = int(row["fold"])
        require(fold in squared_errors, "Unexpected OOF fold")
        truth, prediction = float(row["log_SalePrice"]), float(row["log_prediction"])
        require(math.isfinite(truth) and math.isfinite(prediction), "Non-finite OOF value")
        require(abs(truth - math.log(float(original["SalePrice"]))) <= 1e-12, "OOF target mismatch")
        squared_errors[fold].append((truth - prediction) ** 2)
    require(all(len(errors) == 292 for errors in squared_errors.values()), "Expected five 292-row folds")
    fold_scores = [math.sqrt(statistics.mean(errors)) for errors in squared_errors.values()]
    result = {"mean_fold_rmse": statistics.mean(fold_scores),
              "sample_fold_sd": statistics.stdev(fold_scores),
              "pooled_oof_rmse": math.sqrt(sum(map(sum, squared_errors.values())) / len(rows)),
              "fold_rmse": fold_scores, "rows": len(rows), "operation": "saved_oof_metrics"}
    for key in ("mean_fold_rmse", "sample_fold_sd", "pooled_oof_rmse"):
        require(abs(result[key] - config["validation"][key]) <= 1e-12, f"Recorded metric mismatch: {key}")
    return result


def verify():
    config = load_config()
    ids, _, difference = prediction_sources(config)
    return {"status": "verified", "rows": len(ids), "leaf_weights": EXPECTED_WEIGHTS,
            "scored_sha256": config["submission"]["sha256"],
            "historical_public_score": config["submission"]["public_score"],
            "max_absolute_log_difference": difference, "operation": "saved_prediction_verification",
            "validation": validation_metrics()}


def write_output(content, output):
    path = Path(output)
    if not path.is_absolute():
        path = ROOT / path
    output_root = ROOT / "outputs"
    require(not output_root.is_symlink(), "outputs/ must not be a symlink")
    require(path.resolve().is_relative_to(output_root.resolve()), "Generated files must stay in outputs/")
    require(path != output_root, "Output must be a file inside outputs/")
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != ROOT and p.is_relative_to(ROOT)),
            "Output symlinks are not allowed")
    if path.exists():
        require(path.is_file() and path.read_bytes() == content, f"Refusing to overwrite a different file: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    return path


def predict(output="outputs/predictions.csv"):
    config = load_config()
    ids, combined, difference = prediction_sources(config)
    prices = [math.exp(v) for v in combined]
    require(all(math.isfinite(v) and v > 0 for v in prices), "Invalid recomputed price")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Id", "SalePrice"])
    writer.writerows(zip(ids, prices))
    path = write_output(stream.getvalue().encode("utf-8"), output)
    return {"status": "recomputed_from_saved_components", "output": str(path), "rows": len(ids),
            "max_absolute_log_difference": difference, "operation": "saved_prediction_blend",
            "byte_identical_to_scored_csv": sha256(path) == config["submission"]["sha256"],
            "new_output_public_score": None}


def export_scored(output="outputs/scored_submission.csv"):
    config = load_config()
    prediction_sources(config)
    path = write_output(local(config["inputs"]["scored"]["path"]).read_bytes(), output)
    return {"status": "exported_original_scored_bytes", "output": str(path),
            "sha256": sha256(path), "historical_public_score": config["submission"]["public_score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "predict", "export", "metrics"])
    parser.add_argument("--output", help="Destination within outputs/")
    args = parser.parse_args()
    try:
        if args.command == "predict":
            result = predict(args.output or "outputs/predictions.csv")
        elif args.command == "export":
            result = export_scored(args.output or "outputs/scored_submission.csv")
        else:
            require(args.output is None, "--output is only available for predict and export")
            result = verify() if args.command == "verify" else validation_metrics()
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
