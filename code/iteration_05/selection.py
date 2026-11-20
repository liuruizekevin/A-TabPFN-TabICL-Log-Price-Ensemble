"""Build a round-five promotion manifest from recorded competition submissions."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ledger = json.loads((ROOT/'models/phase5_snapshot/kaggle_submissions.json').read_text())
    best = min((r for r in ledger['submissions'] if r['status']=='complete'), key=lambda r:r['public_score'])
    name = best['experiment_id']
    selected = ROOT / best['local_artifact']
    assert sha(selected) == best['file_sha256']
    ensemble_directory = 'models/round5_ensembles/extension/results'
    candidates = json.loads((ROOT/ensemble_directory/'integration_candidates.json').read_text())
    for candidate in candidates:
        params = candidate['params']
        if isinstance(params.get('weights'), dict):
            mapping = params['weights']
            params['components'] = list(mapping)
            params['weights'] = list(mapping.values())
        candidate['stage'] = 'round5'
        candidate['phase'] = 5
    if name.startswith('R5E'):
        parameters = next(c['params'] for c in candidates if c['experiment_id']==name).copy()
        parameters.update(model='Weighted log-space blend', prediction_type='round5_blend',
                          manifest=ensemble_directory+'/candidate_manifests.json')
    elif name.startswith('R5TP_'):
        parameters = json.loads((ROOT/'models/round5_tabpfn/results'/f'{name}_plan.json').read_text())
        parameters['prediction_type'] = 'round5_tabpfn'
    else:
        # Retain the existing verified recipe if no new candidate improves it.
        previous = json.loads((ROOT/'docs/round4_delivery_manifest.json').read_text())
        assert name == previous['selected_experiment']
        parameters = previous['final_parameters']
    manifest = {
        'selected_experiment': name, 'selected_csv': str(selected.relative_to(ROOT)),
        'selected_model': None, 'run_dirs': [], 'tabpfn_dirs': ['models/round5_tabpfn'],
        'extra_candidates': candidates, 'final_parameters': parameters,
        'catalog_notes':['Five adaptive development phases; official training targets only.',
                         'Fixed-weight primary blends and the recorded fold-isolated model configurations are counted; calibration and median diagnostics remain supplementary.',
                         'All primary validation rows and original seed-42 folds are retained.'],
    }
    output=ROOT/'docs/round5_promotion_plan.json'
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'selected':name,'online_score':best['public_score'],'manifest':str(output)},indent=2))


if __name__=='__main__':
    main()
