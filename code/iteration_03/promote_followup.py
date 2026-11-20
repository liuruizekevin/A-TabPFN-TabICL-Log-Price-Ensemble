#!/usr/bin/env python3
"""Merge the eight adaptive follow-up rows and explicitly promote a candidate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from house_prices.followup_promotion import promote_followup

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dir",type=Path,default=ROOT/"models")
    parser.add_argument("--followup-dir",type=Path,default=ROOT/"models/followup")
    parser.add_argument("--selected-experiment")
    parser.add_argument("--selection-basis",default="Lowest primary development CV mean; leaderboard decision recorded separately")
    args=parser.parse_args()
    result=promote_followup(args.main_dir,args.followup_dir,args.selected_experiment,args.selection_basis)
    print(json.dumps({k:result[k] for k in ["best_experiment","best_cv_mean","experiment_count"]},indent=2))
