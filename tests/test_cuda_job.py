"""CPU-only tests of job control; never represents GPU validation."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cuda_job


class CudaJobTests(unittest.TestCase):
    def test_output_inside_source_is_rejected_even_in_preview(self):
        out = ROOT / 'scripts' / '_invalid_snapshot_output'
        error_text = io.StringIO()
        with contextlib.redirect_stderr(error_text), self.assertRaises(SystemExit) as caught:
            cuda_job.main(['--output', str(out), '--dry-run'])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('source snapshot directory', error_text.getvalue())
        self.assertFalse(out.exists())

    def test_preview_never_creates_or_launches(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'preview'
            with patch.object(cuda_job.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cuda_job.main(['--phase', 'benchmark', '--output', str(out), '--dry-run']), 0)
            launch.assert_not_called()
            self.assertFalse(out.exists())

    def test_login_node_is_rejected_before_files_are_created(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'rejected'
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cuda_job.main(['--output', str(out)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(out.exists())

    def test_failure_preserves_evidence_and_stops_before_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            plan = [('first', [sys.executable, '-c', "print('test-only marker')"], 10),
                    ('failure', [sys.executable, '-c', 'raise SystemExit(7)'], 10),
                    ('should_not_run', [sys.executable, '-c', 'raise SystemExit(9)'], 10)]
            metadata = {'status': 'running', 'completed_stages': []}
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                cuda_job.execute_plan(plan, out, metadata)
            saved = json.loads((out / 'job_status.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['completed_stages'], ['first'])
            self.assertEqual(saved['current_stage'], 'failure')
            self.assertIn('test-only marker', (out / 'first.stdout').read_text())
            self.assertTrue((out / 'failure.command.json').exists())
            self.assertFalse((out / 'should_not_run.command.json').exists())

    def test_snapshot_excludes_other_runs_and_keeps_reference_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'source'
            hashes = cuda_job.snapshot_source(destination)
            self.assertIn('src/mpi_cuda.cu', hashes)
            self.assertIn('scripts/cuda_env_vanda.sh', hashes)
            self.assertIn('results/serial_reference_64_100.csv', hashes)
            self.assertIn('results/serial_reference_128_200.csv', hashes)
            self.assertNotIn('results/serial_scaling_clean.csv', hashes)
            self.assertNotIn('results/mpi_strong.csv', hashes)
            self.assertFalse((destination / 'results/runs').exists())
            self.assertFalse(any('__pycache__' in name for name in hashes))

    def test_clean_source_snapshot_needs_no_historical_timing_csvs(self):
        """Regression: a clean delivery excludes the old unprovenanced timings."""
        with tempfile.TemporaryDirectory() as temp:
            clean = Path(temp) / 'clean_delivery'
            clean.mkdir()
            for directory in cuda_job.SNAPSHOT_DIRECTORIES:
                shutil.copytree(ROOT / directory, clean / directory,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for filename in ['Makefile.cuda', 'SPEC.md']:
                shutil.copy2(ROOT / filename, clean / filename)
            (clean / 'results').mkdir()
            for filename in ['serial_reference_64_100.csv', 'serial_reference_128_200.csv']:
                shutil.copy2(ROOT / 'results' / filename, clean / 'results' / filename)
            self.assertFalse((clean / 'results/serial_scaling_clean.csv').exists())
            self.assertFalse((clean / 'results/mpi_strong.csv').exists())
            destination = Path(temp) / 'job_source_snapshot'
            with patch.object(cuda_job, 'ROOT', clean):
                hashes = cuda_job.snapshot_source(destination)
            self.assertIn('src/mpi_cuda.cu', hashes)
            self.assertEqual((destination / 'src/serial.cpp').read_bytes(),
                             (clean / 'src/serial.cpp').read_bytes())
            for local in [False, True]:
                plan = cuda_job.make_plan(Path(temp) / 'job', 'benchmark', '', 1, local=local)
                summary = next(command for name, command, _ in plan if name == 'summary')
                self.assertNotIn('--serial-csv', summary)
                self.assertNotIn('--mpi-csv', summary)

    def test_concurrent_plans_keep_both_binaries_separate(self):
        first = cuda_job.make_plan(Path('/jobs/one'), 'benchmark', '', 1)
        second = cuda_job.make_plan(Path('/jobs/two'), 'pilot', '', 1)
        for directory, plan in [('/jobs/one', first), ('/jobs/two', second)]:
            build = next(command for name, command, _ in plan if name == 'build')
            gpu = str(Path(directory) / 'heat_mpi_cuda')
            serial = str(Path(directory) / 'heat_serial')
            self.assertIn('TARGET=' + gpu, build)
            self.assertIn('SERIAL_TARGET=' + serial, build)
            tests = [command for name, command, _ in plan if name.startswith('validate_')]
            for command in tests:
                self.assertEqual(command[command.index('--exe') + 1], gpu)
                self.assertEqual(command[command.index('--serial-exe') + 1], serial)
        self.assertFalse(any(name.startswith('size_') for name, _, _ in second))


if __name__ == '__main__':
    unittest.main()
