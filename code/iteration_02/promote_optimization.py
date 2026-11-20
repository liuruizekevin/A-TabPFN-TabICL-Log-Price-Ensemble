#!/usr/bin/env python3
"""Merge experiment evidence and promote the lower-CV phase-two winner."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from house_prices.promotion import promote_optimization

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir",type=Path,default=ROOT/"models")
    parser.add_argument("--optimization-dir",type=Path,default=ROOT/"models/optimization")
    args=parser.parse_args()
    result=promote_optimization(args.phase1_dir,args.optimization_dir)
    print(json.dumps({k:result[k] for k in ["best_experiment","best_cv_mean","experiment_count","absolute_improvement_over_phase1"]},indent=2))
