from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from house_prices.round4_linear import run, extension_specs
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--extension',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'models/round4_linear')
    a=p.parse_args()
    run(ROOT/'data/raw',a.output_dir,ROOT/'models/optimization',specs=extension_specs() if a.extension else None)
