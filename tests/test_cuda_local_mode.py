"""CPU-only provenance/control tests. These never claim GPU validation."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_cuda as bench
import cuda_job
import test_cuda as validate_tool


class LocalCudaModeTests(unittest.TestCase):
    def tools(self, output):
        return [
            (cuda_job.main, ['--output', str(output)]),
            (bench.main, ['--n', '64', '--steps', '100', '--allocated-gpus', '1',
                          '--output', str(output)]),
            (validate_tool.main, ['--allocated-gpus', '1', '--output', str(output)]),
        ]

    def test_local_dry_runs_label_origin_without_process_or_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            output = Path(temp) / 'must_not_exist'
            for main, argv in self.tools(output):
                with self.subTest(tool=main.__module__), patch.object(bench.subprocess, 'Popen') as launch:
                    captured = io.StringIO()
                    with contextlib.redirect_stdout(captured):
                        self.assertEqual(main(argv + ['--local', '--dry-run']), 0)
                    launch.assert_not_called()
                    self.assertIn('LOCAL GPU experiment', captured.getvalue())
                    self.assertIn('not an Atlas/Vanda measurement', captured.getvalue())
                    self.assertFalse(output.exists())

    def test_local_with_pbs_is_rejected_even_for_preview(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'PBS_JOBID': 'unit-test-only'}, clear=True):
            output = Path(temp) / 'must_not_exist'
            for main, argv in self.tools(output):
                with self.subTest(tool=main.__module__), contextlib.redirect_stderr(io.StringIO()) as errors:
                    with self.assertRaises(SystemExit) as caught:
                        main(argv + ['--local', '--dry-run'])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn('cannot be combined with PBS_JOBID', errors.getvalue())
                    self.assertFalse(output.exists())

    def test_default_mode_still_requires_allocation_for_every_entry_point(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            output = Path(temp) / 'must_not_exist'
            for main, argv in self.tools(output):
                with self.subTest(tool=main.__module__), contextlib.redirect_stderr(io.StringIO()) as errors:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        main(argv)
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn('allocated PBS GPU job', errors.getvalue())
                    self.assertFalse(output.exists())

    def test_plans_do_not_automatically_mix_historical_cpu_baselines(self):
        local = cuda_job.make_plan(Path('/local/result'), 'benchmark', '', 1, local=True)
        pbs = cuda_job.make_plan(Path('/pbs/result'), 'benchmark', '', 1)
        for name, command, _ in local:
            if name.startswith(('validate_', 'size_')):
                self.assertIn('--local', command)
        local_summary = next(command for name, command, _ in local if name == 'summary')
        self.assertIn('--summary-only', local_summary)
        self.assertNotIn('--serial-csv', local_summary)
        self.assertNotIn('--mpi-csv', local_summary)
        pbs_summary = next(command for name, command, _ in pbs if name == 'summary')
        self.assertNotIn('--serial-csv', pbs_summary)
        self.assertNotIn('--mpi-csv', pbs_summary)
        self.assertFalse(any('--local' in command for _, command, _ in pbs))

    def test_default_local_output_uses_separate_directory(self):
        captured = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(captured):
            self.assertEqual(cuda_job.main(['--local', '--dry-run']), 0)
        output = captured.getvalue().replace('\\', '/')
        self.assertIn('results/local_runs/cuda_local_', output)
        self.assertNotIn('results/runs/cuda_', output)

    def test_local_metadata_has_no_scheduler_record_or_qstat_call(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            fake_binary = Path(temp) / 'cpu_test_fixture_not_a_solver'
            fake_binary.write_bytes(b'Unit-test input for SHA-256 only. Never executed.')
            args = argparse.Namespace(local=True, exe=fake_binary, launcher='mpirun', profile=False)
            with patch.object(bench, 'command_info', return_value={'exit_code': 1, 'stdout': ''}) as query:
                metadata = bench.collect_metadata(args, [])
            commands = [call.args[0] for call in query.call_args_list]
            self.assertFalse(any(command[0] == 'qstat' for command in commands))
            self.assertEqual(metadata['execution_environment'], 'local')
            self.assertIn('not an Atlas/Vanda', metadata['data_origin_note'])
            self.assertEqual(metadata['pbs'], {})
            self.assertIsNone(metadata['job_details'])
            self.assertIsNone(metadata['nodefile'])
            self.assertTrue(metadata['operating_system'])
            self.assertEqual(len(metadata['binary_sha256']), 64)

    def test_local_job_can_execute_cpu_failure_probe_without_fake_pbs(self):
        """Use a real failing CPU child to check metadata; never launch GPU code."""
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            output = Path(temp) / 'local_failure_probe'
            plan = [('intentional_cpu_failure', [sys.executable, '-c', 'raise SystemExit(7)'], 10)]
            with patch.object(cuda_job, 'make_plan', return_value=plan), \
                    patch.object(cuda_job, 'snapshot_source', return_value={'fixture': 'test-only'}), \
                    patch.object(cuda_job.os, 'chdir'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(bench.subprocess.CalledProcessError):
                    cuda_job.main(['--local', '--output', str(output)])
            status = json.loads((output / 'job_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['execution_environment'], 'local')
            self.assertIsNone(status['pbs_job_id'])
            self.assertEqual(status['current_stage'], 'intentional_cpu_failure')
            self.assertTrue((output / 'intentional_cpu_failure.command.json').is_file())


if __name__ == '__main__':
    unittest.main()
