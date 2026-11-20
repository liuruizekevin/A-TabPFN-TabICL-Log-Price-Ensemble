#!/usr/bin/env python3
"""Run bounded phase-two optimization while retaining phase-one artifacts."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from house_prices.optimization import run_optimization

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/raw")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"models/optimization")
    parser.add_argument("--phase1-dir",type=Path,default=ROOT/"models/phase1_snapshot")
    args=parser.parse_args()
    run_optimization(args.data_dir,args.output_dir,args.phase1_dir)
