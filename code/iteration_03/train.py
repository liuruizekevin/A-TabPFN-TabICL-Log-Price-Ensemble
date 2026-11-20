#!/usr/bin/env python3
"""Run the small, isolated adaptive scaler/SVR follow-up."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from house_prices.followup import run_followup

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/raw")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"models/followup")
    parser.add_argument("--phase2-dir",type=Path,default=ROOT/"models/optimization")
    args=parser.parse_args()
    print(json.dumps(run_followup(args.data_dir,args.output_dir,args.phase2_dir),indent=2))
