"""Record manually observed authenticated Kaggle evidence for round six.

Does not submit, infer a score from local metrics, or mutate historical ledgers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--score', type=float, required=True)
    p.add_argument('--screenshot', type=Path, required=True)
    p.add_argument('--rank', type=int)
    a = p.parse_args()
    csv = (ROOT / a.csv).resolve()
    shot = (ROOT / a.screenshot).resolve()
    assert csv.is_file() and shot.is_file()
    frame = pd.read_csv(csv)
    test = pd.read_csv(ROOT / 'data/raw/test.csv')
    assert frame.columns.tolist() == ['Id', 'SalePrice']
    assert frame.Id.equals(test.Id) and len(frame) == 1459
    assert np.isfinite(frame.SalePrice).all() and frame.SalePrice.gt(0).all()
    assert 0 < a.score < 1
    output = ROOT / 'models/round6_review/online_results.json'
    doc = json.loads(output.read_text()) if output.exists() else {
        'baseline_score': 0.11064,
        'baseline_experiment': 'R5E08_noY_domain20',
        'baseline_rank_at_round_start': 86,
        'account': 'kkkkeviiiinnnn',
        'competition': 'house-prices-advanced-regression-techniques',
        'evidence_scope': 'Scores transcribed from authenticated Kaggle UI and tied to exact uploaded file hashes. Rank is time-sensitive. Online selection is adaptive.',
        'submissions': []}
    assert not any(e['experiment_id'] == a.candidate for e in doc['submissions']), 'Already recorded'
    row = {'round_submission_order': len(doc['submissions'])+1,
           'experiment_id': a.candidate, 'local_artifact': str(csv.relative_to(ROOT)),
           'uploaded_filename': csv.name, 'file_sha256': sha(csv), 'rows': len(frame),
           'public_score': a.score, 'status': 'complete',
           'rank_at_observation': a.rank,
           'screenshot': str(shot.relative_to(ROOT)), 'screenshot_sha256': sha(shot),
           'kaggle_submission_id': None}
    doc['submissions'].append(row)
    best = min(doc['submissions'], key=lambda e:e['public_score'])
    doc['best_round_submission'] = best['experiment_id']
    doc['best_round_score'] = best['public_score']
    doc['improved_over_baseline'] = best['public_score'] < doc['baseline_score']
    output.write_text(json.dumps(doc, indent=2)+'\n')
    pd.DataFrame(doc['submissions']).to_csv(output.with_suffix('.csv'), index=False)
    print(json.dumps(row, indent=2))

if __name__ == '__main__':
    main()
