#!/usr/bin/env python3
"""Run nested stacking experiments without changing the delivered model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from house_prices.round4_stacking import run_stacking


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/round4_stacking")
    args = parser.parse_args()
    print(json.dumps(run_stacking(args.data_dir, args.output_dir), indent=2))
