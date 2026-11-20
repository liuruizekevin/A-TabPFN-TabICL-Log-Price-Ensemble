"""Check historical source identity separately from the current implementation."""
import hashlib
import json
import os
from pathlib import Path


def verify_source(path, expected, workspace):
    path, workspace = Path(path).resolve(), Path(workspace).resolve()
    projection = path.relative_to(workspace).as_posix()
    repository = Path(os.environ['HOUSE_PRICES_REPOSITORY']).resolve()
    registry = json.loads((repository / 'code/registry.json').read_text())
    matches = [row for row in registry['files'] if row['projection'] == projection]
    if len(matches) != 1:
        raise ValueError(f'Unregistered implementation: {projection}')
    row = matches[0]
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    if current != row['sha256']:
        raise ValueError(f'Current implementation hash mismatch: {projection}')
    historical = row.get('historical_source', {})
    if expected not in (current, historical.get('sha256')):
        raise ValueError(f'Recorded source hash mismatch: {projection}')
    return {'path': projection, 'recorded_sha256': expected, 'current_sha256': current,
            'matches_current_bytes': expected == current,
            'verification': 'Historical record identity and current implementation integrity',
            'historical_source': historical if expected != current else None}
