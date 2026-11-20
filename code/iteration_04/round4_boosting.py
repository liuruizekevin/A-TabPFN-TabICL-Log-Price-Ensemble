#!/usr/bin/env python3
"""Run isolated round-four boosting candidates and fixed blend comparisons."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from house_prices.round4_boosting import run, run_reference_bagging, run_pfn_blends, materialize_blend, audit_outputs

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/raw")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"models/round4_boosting")
    parser.add_argument("--phase2-dir",type=Path,default=ROOT/"models/optimization")
    parser.add_argument("--followup-dir",type=Path,default=ROOT/"models/followup")
    parser.add_argument("--only",default=None,help="Comma-separated exact specification IDs for reproducible subsets")
    parser.add_argument("--reference-bagging",action="store_true",help="Reproduce and bag the existing F07 components only")
    parser.add_argument("--pfn-blends",action="store_true",help="Evaluate fixed convex blends with available TabPFN OOF and test predictions")
    parser.add_argument("--pfn-dir",type=Path,default=ROOT/"models/round4_tabpfn")
    parser.add_argument("--materialize-blend",default=None,help="Save and reload-check a completed fixed blend by its exact ID")
    parser.add_argument("--audit",action="store_true",help="Recompute saved metrics and validate completed outputs")
    args=parser.parse_args()
    if args.audit:
        summary=audit_outputs(args.output_dir)
    elif args.materialize_blend:
        summary=materialize_blend(args.output_dir,args.materialize_blend)
    elif args.pfn_blends:
        summary=run_pfn_blends(args.output_dir,args.pfn_dir)
    elif args.reference_bagging:
        summary=run_reference_bagging(args.data_dir,args.output_dir,args.phase2_dir,args.followup_dir)
    else:
        summary=run(args.data_dir,args.output_dir,args.phase2_dir,args.followup_dir,args.only)
    print(json.dumps(summary,indent=2))
