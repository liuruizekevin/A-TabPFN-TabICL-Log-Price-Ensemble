import importlib.util
import ast
import json
import os
from pathlib import Path
import tempfile
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('iteration_runtime', ROOT / 'code/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
provenance_spec = importlib.util.spec_from_file_location('source_provenance', ROOT / 'code/shared/provenance.py')
provenance = importlib.util.module_from_spec(provenance_spec)
provenance_spec.loader.exec_module(provenance)


class IterationPaths(unittest.TestCase):
    def fixture(self, root):
        (root / 'data/raw').mkdir(parents=True)
        (root / 'code').mkdir()
        entries = []
        for name in ('train.csv', 'test.csv'):
            path = root / 'data/raw' / name
            path.write_text('Id,value\n1,2\n')
            entries.append({'path': str(path.relative_to(root)), 'sha256': runtime.digest(path),
                            'bytes': path.stat().st_size,
                            'origins': ['source/experiments/data/raw/' + name]})
        (root / 'data/catalog.json').write_text(json.dumps({'schema_version': 1, 'files': entries}))
        (root / 'code/registry.json').write_text(json.dumps({'commands': {}, 'files': []}))

    def test_projection_copies_and_removes_temporary_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            source = root / 'data/raw/train.csv'
            original = source.read_bytes()
            with runtime.project_inputs(root) as (workspace, before):
                target = workspace / 'data/raw/train.csv'
                self.assertFalse(target.is_symlink())
                self.assertNotEqual(target.stat().st_ino, source.stat().st_ino)
                target.write_text('changed run output\n')
                changes = runtime.export_changes(workspace, before, root / 'outputs/iteration_01/run')
                self.assertEqual(len(changes), 1)
                self.assertEqual(source.read_bytes(), original)
            self.assertFalse(workspace.exists())

    def test_path_escape_is_rejected(self):
        for value in ('/tmp/overwrite.csv', '../data/raw/train.csv', 'a/../../data.csv'):
            with self.assertRaises(ValueError):
                runtime.relative_path(value)
            with self.assertRaises(ValueError):
                runtime.validate_arguments(['--output', value])
        with self.assertRaises(ValueError):
            runtime.validate_arguments(['--output=/tmp/overwrite.csv'])

    def test_conflicting_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / 'data/catalog.json'
            value = json.loads(path.read_text())
            second = dict(value['files'][0], sha256='0' * 64)
            value['files'].append(second)
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                runtime.input_map(root)

    def test_output_symlink_cannot_redirect_a_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repository'
            root.mkdir()
            self.fixture(root)
            implementation = root / 'code/probe.py'
            implementation.write_text("raise RuntimeError('This must not execute')\n")
            mapping = {'commands': {'1': {'probe': 'scripts/probe.py'}},
                       'files': [{'path': 'code/probe.py', 'projection': 'scripts/probe.py',
                                  'sha256': runtime.digest(implementation)}]}
            (root / 'code/registry.json').write_text(json.dumps(mapping))
            outside = Path(directory) / 'outside'
            outside.mkdir()
            (root / 'outputs').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'must not redirect'):
                runtime.run_iteration(1, 'probe', [], root=root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_command_implementations_exist_and_compile(self):
        value = runtime.registry()
        self.assertEqual(set(value['commands']), {str(i) for i in range(1, 7)})
        for entry in value['files']:
            path = ROOT / entry['path']
            self.assertEqual(runtime.digest(path), entry['sha256'])
            compile(path.read_bytes(), str(path), 'exec')

    def test_iteration_two_default_uses_complete_phase_one_snapshot(self):
        implementation = ROOT / 'code/iteration_02/train.py'
        module = ModuleType('house_prices.optimization')
        calls = []
        module.run_optimization = lambda *args, **kwargs: calls.append((args, kwargs))
        filename = '/isolated/scripts/optimize_models.py'
        with patch.dict(sys.modules, {'house_prices.optimization': module}), \
                patch.object(sys, 'argv', ['optimize_models.py']), \
                patch.object(sys, 'path', sys.path[:]):
            exec(compile(implementation.read_text(), filename, 'exec'),
                 {'__name__': '__main__', '__file__': filename})
        self.assertEqual(len(calls), 1)
        phase_one = calls[0][0][2]
        self.assertEqual(phase_one, Path('/isolated/models/phase1_snapshot'))
        aliases = runtime.input_map()
        prefix = phase_one.relative_to('/isolated').as_posix()
        for suffix in ('results/summary.json', 'results/oof_predictions.csv',
                       'results/experiment_results.csv', 'results/fold_assignments.csv',
                       'artifacts/best_model.joblib'):
            record = aliases[prefix + '/' + suffix]
            self.assertEqual(runtime.digest(ROOT / record['path']), record['sha256'])
        summary = json.loads((ROOT / aliases[prefix + '/results/summary.json']['path']).read_text())
        self.assertEqual(summary['experiment_count'], 21)
        self.assertEqual(summary['best_experiment'], 'E11_catboost_depth5')

    def test_historical_source_check_without_git_rejects_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            workspace = root / 'projected'
            source = workspace / 'scripts/probe.py'
            source.parent.mkdir(parents=True)
            source.write_text('value = 1\n')
            old_hash = '1' * 64
            row = {'path': 'code/probe.py', 'projection': 'scripts/probe.py',
                   'sha256': runtime.digest(source),
                   'historical_source': {'sha256': old_hash, 'git_revision': 'a' * 40,
                                         'git_path': 'source/experiments/scripts/probe.py'}}
            (root / 'code/registry.json').write_text(json.dumps({'files': [row]}))
            with patch.dict(os.environ, {'HOUSE_PRICES_REPOSITORY': str(root)}):
                result = provenance.verify_source(source, old_hash, workspace)
                self.assertFalse(result['matches_current_bytes'])
                with self.assertRaisesRegex(ValueError, 'Recorded source hash mismatch'):
                    provenance.verify_source(source, '2' * 64, workspace)
                source.write_text('value = 2\n')
                with self.assertRaisesRegex(ValueError, 'Current implementation hash mismatch'):
                    provenance.verify_source(source, old_hash, workspace)
            self.assertFalse((root / '.git').exists())

    def test_tabicl_dependency_check_accepts_installed_packages_but_rejects_wrong_version(self):
        tree = ast.parse((ROOT / 'code/iteration_06/round6_tabicl.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'verify_dependencies')
        versions = {'tabicl': '2.2.0', 'psutil': '7.2.2'}
        namespace = {'DEPENDENCIES': tuple(versions),
                     'importlib': SimpleNamespace(metadata=SimpleNamespace(version=versions.__getitem__))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<dependency check>', 'exec'), namespace)
        self.assertEqual(namespace['verify_dependencies'](), versions)
        versions['tabicl'] = '0.0.0'
        with self.assertRaisesRegex(RuntimeError, 'requires tabicl'):
            namespace['verify_dependencies']()


if __name__ == '__main__':
    unittest.main()
