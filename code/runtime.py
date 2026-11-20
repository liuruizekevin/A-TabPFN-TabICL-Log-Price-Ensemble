"""Run recorded iterations in disposable copies of the preserved inputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a relative path inside the run: {value}')
    return Path(*path.parts)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def data_alias(origin: str) -> str | None:
    if origin.startswith('source/experiments/'):
        alias = origin.removeprefix('source/experiments/')
        if alias.startswith('versions/'):
            return None
        return alias
    if origin.startswith('source/data/raw/'):
        return 'data/raw/' + origin.removeprefix('source/data/raw/')
    return None


def input_map(root: Path = ROOT) -> dict[str, dict]:
    result = {}
    for item in read_json(root / 'data/catalog.json')['files']:
        source = root / relative_path(item['path'])
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Input must be a regular file: {item["path"]}')
        for origin in item['origins']:
            alias = data_alias(origin)
            if alias is None:
                continue
            relative_path(alias)
            prior = result.get(alias)
            if prior is not None and prior['sha256'] != item['sha256']:
                raise ValueError(f'Conflicting input versions for {alias}')
            result[alias] = item
    return result


def registry(root: Path = ROOT) -> dict:
    return read_json(root / 'code/registry.json')


def check_inputs(root: Path = ROOT) -> dict:
    aliases = input_map(root)
    checked = {}
    for item in aliases.values():
        source = root / item['path']
        if item['path'] not in checked:
            checked[item['path']] = digest(source)
        if checked[item['path']] != item['sha256'] or source.stat().st_size != item['bytes']:
            raise ValueError(f'Preserved input differs from its catalogue: {item["path"]}')
    code = registry(root)
    projections = set()
    for item in code['files']:
        source = root / relative_path(item['path'])
        relative_path(item['projection'])
        if digest(source) != item['sha256']:
            raise ValueError(f'Implementation differs from its registry: {item["path"]}')
        if item['projection'] in projections:
            raise ValueError(f'Duplicate implementation path: {item["projection"]}')
        projections.add(item['projection'])
    for phase, commands in code['commands'].items():
        if int(phase) not in range(1, 7):
            raise ValueError(f'Invalid iteration: {phase}')
        if not set(commands.values()).issubset(projections):
            raise ValueError(f'Unmapped command in iteration {phase}')
    for path in ('data/raw/train.csv', 'data/raw/test.csv'):
        if path not in aliases:
            raise ValueError(f'Missing competition input: {path}')
    return {'status': 'passed', 'iterations': len(code['commands']),
            'commands': sum(len(value) for value in code['commands'].values()),
            'input_aliases': len(aliases), 'unique_inputs_checked': len(checked),
            'implementations_checked': len(code['files'])}


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


@contextmanager
def project_inputs(root: Path = ROOT, checkpoint_dir: Path | None = None):
    """Create independent input copies; no link targets point into the repository."""
    aliases = input_map(root)
    with tempfile.TemporaryDirectory(prefix='houseprices-iteration-') as directory:
        workspace = Path(directory)
        before = {}
        for alias, item in aliases.items():
            target = workspace / relative_path(alias)
            copy_file(root / item['path'], target)
            before[alias] = item['sha256']
        for item in registry(root)['files']:
            copy_file(root / item['path'], workspace / relative_path(item['projection']))
            before[item['projection']] = item['sha256']
        # Child predictors use the same interpreter and installed packages.
        launcher = '#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n'
        for alias in ('.venv/bin/python', 'build/tabpfn-env/bin/python'):
            target = workspace / alias
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(launcher)
            target.chmod(0o755)
            before[alias] = digest(target)
        if checkpoint_dir is not None:
            checkpoints = {}
            for source in checkpoint_dir.resolve().rglob('*.ckpt'):
                if source.name in checkpoints and digest(source) != digest(checkpoints[source.name]):
                    raise ValueError(f'Conflicting checkpoint names: {source.name}')
                checkpoints[source.name] = source
            for name, source in checkpoints.items():
                folder = 'build/tabicl-checkpoints' if name.startswith('tabicl-') else 'references/tabpfn'
                alias = folder + '/' + name
                copy_file(source, workspace / alias)
                before[alias] = digest(source)
        yield workspace, before


def validate_arguments(arguments: list[str]) -> None:
    for value in arguments:
        candidate = value.split('=', 1)[1] if value.startswith('--') and '=' in value else value
        if candidate.startswith('--'):
            continue
        path = PurePosixPath(candidate)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Iteration arguments must use paths relative to the disposable run.')


def export_name(alias: str) -> Path:
    parts = PurePosixPath(alias).parts
    replacements = {'docs': 'metadata', 'build': 'diagnostics'}
    if parts[0] in replacements:
        parts = (replacements[parts[0]], *parts[1:])
    return Path(*parts)


def export_changes(workspace: Path, before: dict, destination: Path) -> list[dict]:
    changed = []
    for source in sorted(workspace.rglob('*')):
        if not source.is_file() or source.is_symlink() or '__pycache__' in source.parts:
            continue
        if source.suffix in {'.pyc', '.pyo', '.log'}:
            continue
        alias = source.relative_to(workspace).as_posix()
        value = digest(source)
        if before.get(alias) == value:
            continue
        target = destination / export_name(alias)
        if target.exists():
            raise FileExistsError(target)
        copy_file(source, target)
        changed.append({'path': str(target.relative_to(destination)),
                        'sha256': value, 'bytes': target.stat().st_size})
    return changed


def run_iteration(phase: int, command: str, arguments: list[str], *,
                  root: Path = ROOT, checkpoint_dir: Path | None = None) -> int:
    validate_arguments(arguments)
    entry = registry(root)['commands'][str(phase)][command]
    check_inputs(root)
    run_id = uuid4().hex
    destination = root / 'outputs' / f'iteration_{phase:02}' / f'{command}_{run_id}'
    if not destination.resolve().is_relative_to((root.resolve() / 'outputs')):
        raise ValueError('Output directories must not redirect outside outputs/.')
    with project_inputs(root, checkpoint_dir) as (workspace, before):
        if phase == 1 and command == 'train' and not any(x.startswith('--output-dir') for x in arguments):
            arguments = ['--output-dir', 'build/iteration_01', *arguments]
        if phase == 6 and command == 'tabicl' and not any(
                x.startswith('--output-dir') for x in arguments) and not any(
                x in arguments for x in ('--validate-only', '--check-api')):
            arguments = ['--output-dir', f'build/iteration_06/{command}', *arguments]
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['HOUSE_PRICES_REPOSITORY'] = str(root.resolve())
        process = subprocess.run([sys.executable, str(workspace / entry), *arguments], cwd=workspace, env=env)
        changes = export_changes(workspace, before, destination)
    if changes:
        record = {'iteration': phase, 'command': command, 'arguments': arguments,
                  'returncode': process.returncode, 'files': changes}
        (destination / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'output': str(destination.relative_to(root)), 'files': len(changes)}, indent=2))
    return process.returncode


def main(phase: int) -> None:
    commands = registry()['commands'][str(phase)]
    parser = argparse.ArgumentParser(description=f'Iteration {phase:02}: isolated experiment commands.')
    parser.add_argument('--list', action='store_true', help='List available commands.')
    parser.add_argument('--check', action='store_true', help='Verify all preserved inputs and implementation mappings.')
    parser.add_argument('--checkpoint-dir', type=Path, help='Directory containing separately downloaded model checkpoints.')
    parser.add_argument('command', nargs='?', choices=sorted(commands))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_inputs(), indent=2))
        return
    if args.list:
        print('\n'.join(sorted(commands)))
        return
    if args.command is None:
        parser.error('Choose a command, --list or --check.')
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    raise SystemExit(run_iteration(phase, args.command, arguments, checkpoint_dir=args.checkpoint_dir))
