"""Run the report-linked Notebook in an isolated copy of the current repository layout."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = 'outputs/notebook'
NOTEBOOK_NAME = 'notebooks/house_prices.ipynb'
EXECUTED_NAME = 'notebooks/house_prices.executed.ipynb'


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a relative path: {value}')
    return Path(*path.parts)


def source_bytes(root: Path, item: dict) -> bytes:
    path = root / relative_path(item['path'])
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Expected a file within the repository: {item["path"]}')
    content = path.read_bytes()
    if digest(content) != item['sha256']:
        raise ValueError(f'Source SHA-256 mismatch: {item["path"]}')
    return content


def safe_output(root: Path, output: str) -> Path:
    root = root.resolve()
    relative = relative_path(output)
    if relative.parts[0] != 'outputs' or len(relative.parts) < 2:
        raise ValueError('Notebook workspaces must be inside outputs/.')
    target = root / relative
    if not target.resolve().is_relative_to(root / 'outputs'):
        raise ValueError('Notebook workspace escapes outputs/.')
    for path in (target, *target.parents):
        if path == root:
            break
        if path.is_symlink():
            raise ValueError(f'Notebook output cannot use symlinks: {path}')
    if target.exists() and not target.is_dir():
        raise ValueError(f'Workspace is not a directory: {target}')
    return target


def payloads(root: Path) -> tuple[dict[str, bytes], dict]:
    manifest_bytes = (root / 'config/notebook.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['schema_version'] != 2:
        raise ValueError('Unsupported Notebook manifest; expected canonical repository paths.')
    if manifest['notebook']['path'] != NOTEBOOK_NAME:
        raise ValueError('Unexpected Notebook path in the manifest.')
    notebook_bytes = source_bytes(root, manifest['notebook'])
    notebook = json.loads(notebook_bytes)
    if len(notebook['cells']) != manifest['notebook']['cells'] or sum(
            cell['cell_type'] == 'code' for cell in notebook['cells']) != manifest['notebook']['code_cells']:
        raise ValueError('Notebook cell structure differs from its manifest.')
    result = {NOTEBOOK_NAME: notebook_bytes, 'config/notebook.json': manifest_bytes}

    def add(path: str, content: bytes, expected: str):
        relative_path(path)
        if digest(content) != expected:
            raise ValueError(f'Resource SHA-256 mismatch: {path}')
        if path in result and result[path] != content:
            raise ValueError(f'Conflicting workspace inputs: {path}')
        result[path] = content

    for item in [*manifest['data_files'], *manifest['support_files']]:
        add(item['path'], source_bytes(root, item), item['sha256'])
    asset_item = manifest['embedded_assets']
    asset_bytes = source_bytes(root, asset_item)
    add(asset_item['path'], asset_bytes, asset_item['sha256'])
    assets = json.loads(asset_bytes)
    for item in assets['files']:
        if item['encoding'] == 'base64':
            content = base64.b64decode(item['content'], validate=True)
        elif item['encoding'] == 'utf-8':
            content = item['content'].encode('utf-8')
        else:
            raise ValueError(f'Unknown resource encoding: {item["encoding"]}')
        add(item['path'], content, item['sha256'])
    for item in manifest['figures']:
        images = [output['data']['image/png'] for output in notebook['cells'][item['cell']]['outputs']
                  if 'image/png' in output.get('data', {})]
        content = base64.b64decode(images[item['image_index']], validate=True)
        add(item['path'], content, item['sha256'])
    return result, manifest


def write_payloads(root: Path, workspace: Path, files: dict[str, bytes]) -> None:
    """Check every destination before creating files; never overwrite user changes."""
    destinations = []
    for relative, content in files.items():
        path = workspace / relative_path(relative)
        safe_output(root, path.parent.relative_to(root).as_posix())
        if path.is_symlink():
            raise ValueError(f'Notebook input cannot be a symlink: {path}')
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FileExistsError(f'Refusing to overwrite a different file: {path}')
        destinations.append((path, content))
    for path, content in destinations:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(content)


def prepare(*, root: Path = ROOT, output: str = DEFAULT_OUTPUT) -> dict:
    root = Path(root).resolve()
    workspace = safe_output(root, output)
    files, manifest = payloads(root)
    write_payloads(root, workspace, files)
    return {'status': 'prepared', 'workspace': str(workspace),
            'notebook': str(workspace / NOTEBOOK_NAME), 'input_files': len(files),
            'notebook_sha256': manifest['notebook']['sha256'],
            'tracked_notebook_changed': False, 'operation': 'notebook_workspace_preparation'}


def execute(*, root: Path = ROOT, output: str = DEFAULT_OUTPUT) -> dict:
    """Execute the canonical-layout copy, preserving the tracked Notebook."""
    root = Path(root).resolve()
    workspace = safe_output(root, output)
    destination = workspace / EXECUTED_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Executed output already exists; choose another --output directory: {destination}')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    result = prepare(root=root, output=output)
    manifest = json.loads((root / 'config/notebook.json').read_text(encoding='utf-8'))
    before = source_bytes(root, manifest['notebook'])
    notebook = nbformat.read(result['notebook'], as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(notebook, km=manager, timeout=180, record_timing=False,
                   resources={'metadata': {'path': str(workspace)}}).execute()
    nbformat.validate(notebook)
    if source_bytes(root, manifest['notebook']) != before:
        raise ValueError('The tracked Notebook changed during execution.')
    # The prepared source copy is unchanged; only this second copy gets new outputs.
    with destination.open('x', encoding='utf-8') as stream:
        nbformat.write(notebook, stream)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    return {**result, 'status': 'executed', 'executed_notebook': str(destination),
            'executed_code_cells': len(code_cells), 'operation': 'notebook_execution'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--prepare', action='store_true', help='Prepare an interactive copy without running cells.')
    operation.add_argument('--execute', action='store_true', help='Run the cells and save a separate executed copy.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Workspace path beneath outputs/.')
    args = parser.parse_args()
    try:
        result = execute(output=args.output) if args.execute else prepare(output=args.output)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
