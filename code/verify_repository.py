"""Verify repository structure, preserved data and recorded experiment metrics."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), f'Invalid headers: {path}')
        require(all(None not in row and all(value is not None for value in row.values()) for row in rows), f'Invalid row width: {path}')
        return rows


def main():
    documents = ROOT / 'documents'
    require({p.name for p in documents.iterdir()} == {'README.md', 'report.pdf', 'slides.pdf'}, 'Expected one report and one slide deck in documents/')
    for name in ('report.pdf', 'slides.pdf'):
        require((documents / name).read_bytes().startswith(b'%PDF-'), f'Not a PDF: {name}')
    skip = {'.git', '.venv', 'outputs', '__pycache__', '.pytest_cache'}
    readme_count = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in skip or part.startswith('.venv-') for part in relative.parts):
            continue
        if path.is_dir():
            require(not re.search(r'final|最终|draft|backup|cache', path.name, re.I), f'Unwanted directory: {relative}')
            require(any(path.iterdir()), f'Empty directory: {relative}')
            require((path / 'README.md').is_file(), f'Missing folder description: {relative}')
            readme_count += 1
        elif path.is_file():
            require(path.suffix.lower() not in {'.tex', '.pptx', '.docx', '.png', '.jpg', '.jpeg', '.svg', '.log', '.zip'}, f'Unwanted document/source format: {relative}')
            require(path.suffix != '.md' or path.name == 'README.md', f'Extra prose document: {relative}')
            if path.name.lower().startswith('readme.'):
                require(path.name == 'README.md', f'Folder descriptions must use README.md: {relative}')
    require(list((ROOT / 'notebooks').glob('*.ipynb')) == [ROOT / 'notebooks/house_prices.ipynb'], 'Expected one Notebook')
    source = json.loads((ROOT / 'config/notebook.json').read_text())['notebook']
    require(source['path'] == 'notebooks/house_prices.ipynb', 'Unexpected Notebook path')
    notebook_path = ROOT / source['path']
    require(hashlib.sha256(notebook_path.read_bytes()).hexdigest() == source['sha256'], 'Notebook checksum mismatch')
    notebook = json.loads(notebook_path.read_text())
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    require(len(notebook['cells']) == source['cells'] == 25 and len(code) == source['code_cells'] == 7, 'Expected 25 Notebook cells, including seven code cells')
    require([c['execution_count'] for c in code] == list(range(1, len(code) + 1)), 'Notebook execution counts are incomplete')
    require(not any(o['output_type'] == 'error' for c in code for o in c['outputs']), 'Notebook contains an execution error')
    text = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    require(all(section in text for section in (
        '## Part A. Algorithms and final prediction procedure',
        '## Part B. Data and saved prediction workflow',
    )), 'Missing an algorithms or workflow section')
    require('## Part C.' not in text and not any(c.get('id', '').startswith('additional-c-') for c in notebook['cells']), 'Part C must be removed')
    contents = next(''.join(c['source']) for c in notebook['cells'] if c.get('id') == 'additional-contents')
    require(not re.search(r'\[[^\]]+\]\([^)]+\)|<a\b', contents, re.I), 'Notebook contents must use plain text without links')
    markdown = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
    require(len(re.findall(r'^# ', markdown, re.M)) == 1, 'Expected one project title')
    require('## Repository' in ''.join(notebook['cells'][0]['source']), 'Repository attribution must appear at the beginning')
    require(notebook['metadata'].get('authors') == [{'name': 'liuruizekevin'}], 'Public Notebook must use the repository maintainer attribution')
    b_sections = re.findall(r'^### B\.(\d+) ', markdown, re.M)
    require(b_sections == [str(n) for n in range(1, 9)], 'Expected the Part B workflow in order from B.1 to B.8')
    catalog = json.loads((ROOT / 'data/catalog.json').read_text())
    seen = set()
    for item in catalog['files']:
        path = ROOT / item['path']
        require(item['path'] not in seen, f'Duplicate catalogue path: {path}')
        seen.add(item['path'])
        require(path.is_file() and path.stat().st_size == item['bytes'], f'Missing or resized input: {path}')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Changed input: {path}')
    configs = read_rows(ROOT / 'data/overview/experiment_results.csv')
    require(len(configs) == len({r['experiment_id'] for r in configs}) == 350, 'Expected 350 distinct configurations')
    counts = []
    for n in range(1, 7):
        expected = [r for r in configs if int(r['phase']) == n]
        actual = read_rows(ROOT / f'data/iteration_{n:02}/tables/configurations.csv')
        require(actual == expected, f'Iteration {n} does not match canonical records')
        counts.append(len(actual))
    require(counts == [21, 29, 8, 263, 17, 12], 'Wrong phase counts')
    oof = read_rows(ROOT / 'data/overview/oof_predictions.csv')
    require(len(oof) == 1460, 'Wrong OOF row count')
    folded = {n: [r for r in oof if int(r['fold']) == n] for n in range(1, 6)}
    require([len(v) for v in folded.values()] == [292] * 5, 'Wrong validation fold sizes')
    fold_rows = read_rows(ROOT / 'data/overview/fold_scores.csv')
    require(len(fold_rows) == 1750, 'Wrong fold-score count')
    fold_lookup = {(r['experiment_id'], int(r['fold'])): float(r['log_rmse']) for r in fold_rows}
    for row in configs:
        name = row['experiment_id'];column = name + '__log_prediction'
        scores = []
        for n, entries in folded.items():
            rmse = math.sqrt(sum((float(r[column]) - float(r['log_SalePrice'])) ** 2 for r in entries) / len(entries))
            require(abs(rmse - fold_lookup[name, n]) < 1e-12, f'Fold RMSE mismatch: {name}, {n}')
            scores.append(rmse)
        require(abs(statistics.mean(scores) - float(row['cv_mean'])) < 1e-12, f'Mean CV mismatch: {name}')
        require(abs(statistics.stdev(scores) - float(row['cv_std'])) < 1e-12, f'Fold SD mismatch: {name}')
        pooled = math.sqrt(sum(x*x for x in scores) / 5)
        require(abs(pooled - float(row['pooled_oof_log_rmse'])) < 1e-12, f'Pooled RMSE mismatch: {name}')
    ledger = json.loads((ROOT / 'data/overview/kaggle_submissions.json').read_text())['submissions']
    require(len(ledger) == 36 and len({x['submission_order'] for x in ledger}) == 36, 'Incorrect submission records')
    origin_map = {origin.removeprefix('source/experiments/'): item for item in catalog['files'] for origin in item['origins']}
    evidence = json.loads((ROOT / 'data/overview/submission_evidence.json').read_text())['files']
    evidence_by_origin = {origin: item for item in evidence for origin in item['aliases']}
    for row in ledger:
        item = origin_map[row['local_artifact']]
        require(item['sha256'] == row['file_sha256'], f'Scored CSV changed: {row["experiment_id"]}')
        record = evidence_by_origin[row['screenshot']]
        require(record['score'] == row['public_score'] and record['experiment_id'] == row['experiment_id'], 'Submission record mismatch')
    print(json.dumps({'status': 'passed', 'experiment_rounds': 6, 'pdfs': 2, 'report_copies': 1, 'slide_deck_copies': 1, 'folder_readmes': readme_count, 'data_files': len(catalog['files']), 'configurations': 350, 'fold_scores': 1750, 'submissions': 36, 'notebook_cells': len(notebook['cells']), 'executed_notebook_cells': len(code)}, indent=2))


if __name__ == '__main__':
    main()
