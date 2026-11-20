"""Predict with a saved fitted classical estimator."""
import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from house_prices.preprocessing import read_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=ROOT/'data/raw/test.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'build/predictions.csv')
    args = parser.parse_args()
    data = read_data(args.test)
    if 'SalePrice' in data or 'Id' not in data or not data.Id.is_unique:
        raise ValueError('Expected unique Id values and no SalePrice column.')
    prediction = joblib.load(args.model).predict(data)
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError('Predictions must be finite and positive.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Id':data.Id, 'SalePrice':prediction}).to_csv(args.output, index=False)
    print(f'Saved {len(data)} predictions to {args.output}')
