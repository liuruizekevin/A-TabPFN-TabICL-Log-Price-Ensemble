"""Protect original Notebook inputs and keep generated files in their workspace."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'code/notebook/prepare.py'
SPEC = importlib.util.spec_from_file_location('original_notebook_runtime', SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class OriginalNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_workspace_cannot_escape_outputs_or_follow_a_symlink(self):
        for name in ('../outside', 'notebooks/runtime', '/tmp/runtime', 'outputs/../notebooks'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runtime.safe_output(self.root, name)
        (self.root / 'outputs').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            runtime.safe_output(self.root, 'outputs/original_notebook')

    def test_changed_source_bytes_are_rejected(self):
        path = self.root / 'notebook.ipynb'
        path.write_bytes(b'original')
        item = {'path': 'notebook.ipynb', 'sha256': runtime.digest(b'original')}
        self.assertEqual(runtime.source_bytes(self.root, item), b'original')
        path.write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            runtime.source_bytes(self.root, item)

    def test_all_existing_files_are_checked_before_writes(self):
        workspace = self.root / 'outputs/original_notebook'
        workspace.mkdir(parents=True)
        existing = workspace / 'input.csv'
        existing.write_bytes(b'user-edited data')
        files = {'new.csv': b'new data', 'input.csv': b'original data'}
        with self.assertRaises(FileExistsError):
            runtime.write_payloads(self.root, workspace, files)
        self.assertFalse((workspace / 'new.csv').exists())
        self.assertEqual(existing.read_bytes(), b'user-edited data')

    def test_identical_prepared_files_can_be_reused(self):
        files = {runtime.NOTEBOOK_NAME: b'original notebook', 'data/input.csv': b'1,2\n'}
        manifest = {'notebook': {'sha256': runtime.digest(files[runtime.NOTEBOOK_NAME])}}
        with patch.object(runtime, 'payloads', return_value=(files, manifest)):
            first = runtime.prepare(root=self.root)
            second = runtime.prepare(root=self.root)
        self.assertEqual(first, second)
        for alias, content in files.items():
            self.assertEqual((Path(first['workspace']) / alias).read_bytes(), content)
        self.assertEqual(first['operation'], 'notebook_workspace_preparation')

    def test_execution_refuses_an_existing_result_before_starting_a_kernel(self):
        folder = self.root / runtime.DEFAULT_OUTPUT
        folder.mkdir(parents=True)
        output = folder / runtime.EXECUTED_NAME
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'previous execution')
        with patch.object(runtime, 'prepare', side_effect=AssertionError('Must not start preparation')):
            with self.assertRaises(FileExistsError):
                runtime.execute(root=self.root)
        self.assertEqual(output.read_bytes(), b'previous execution')


if __name__ == '__main__':
    unittest.main()
