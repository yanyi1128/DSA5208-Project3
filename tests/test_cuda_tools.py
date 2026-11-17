"""CPU-only tests of experiment parsing, statistics and dry-run behavior.

Synthetic rows below are test fixtures, never GPU measurements or deliverables.
"""
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import benchmark_cuda as bench
import test_cuda as validate_tool

spec = importlib.util.spec_from_file_location('cuda_analysis', ROOT / 'analysis' / 'cuda_analysis.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def solver_text(**changes):
    row = dict(implementation='mpi_cuda', n='64', steps='100', r='0.1', ranks='1',
               gpus='1', elapsed_s='0.25', validation='not_run',
               max_abs_error='NA', relative_l2_error='NA')
    row.update(changes)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=bench.FIELDS)
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue()


def fixture_group(directory, mode='strong', timing='ordinary', ranks=1, n=64):
    """Deliberately named unit-test fixture; warm-up outlier must be excluded."""
    key = (str(directory / 'synthetic_test_fixture.csv'), mode, timing, 100, 32, 8, n, ranks)
    records = []
    for repetition, elapsed in enumerate([1000, 1, 2, 3, 4, 100]):
        row = dict(phase='warmup' if repetition == 0 else 'measured', repeat=str(repetition),
                   elapsed_s=str(elapsed / ranks), validation='not_run')
        records.append(row)
    return key, records


class CudaToolsTests(unittest.TestCase):
    def test_exact_solver_csv_and_validation_state(self):
        self.assertEqual(bench.parse_record(solver_text(), 64, 100, 1)['validation'], 'not_run')
        for changes in [{'gpus': '2'}, {'n': '63'}, {'elapsed_s': 'nan'},
                        {'elapsed_s': '-1'}, {'elapsed_s': '0'}, {'r': '0.2'},
                        {'validation': 'pass'}, {'max_abs_error': '0'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.parse_record(solver_text(**changes), 64, 100, 1)
        with self.assertRaises(ValueError):
            bench.parse_record(solver_text() + 'extra output\n', 64, 100, 1)
        self.assertEqual(bench.parse_record(solver_text(validation='pass', max_abs_error='0',
                                                      relative_l2_error='0'), 64, 100, 1, True)['validation'], 'pass')

    def test_scaling_grid_sizes(self):
        self.assertEqual(bench.configurations('strong', 1024, None, [1, 2, 4]),
                         [(1024, 1), (1024, 2), (1024, 4)])
        self.assertEqual(bench.configurations('weak', 1024, None, [1, 4, 9]),
                         [(1024, 1), (2048, 4), (3072, 9)])
        self.assertEqual(bench.configurations('size', None, [256, 512], [1]), [(256, 1), (512, 1)])
        with self.assertRaises(ValueError):
            bench.configurations('strong', 64, None, [2, 4])

    def test_median_retains_outlier_and_excludes_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            groups = dict([fixture_group(Path(temporary), ranks=p) for p in [1, 2]])
            rows = analysis.summarize(groups)
            self.assertEqual(rows[0]['median_s'], 3)
            self.assertEqual(rows[0]['min_s'], 1)
            self.assertEqual(rows[0]['max_s'], 100)
            self.assertEqual(rows[0]['validation'], 'not_run')
            self.assertEqual(rows[1]['speedup_vs_one_gpu'], 2)
            self.assertEqual(rows[1]['efficiency_pct'], 100)
            groups[next(iter(groups))].pop()
            with self.assertRaises(ValueError):
                analysis.summarize(groups)

    def test_weak_efficiency_not_divided_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, av = fixture_group(Path(temporary), mode='weak', ranks=1, n=64)
            b, bv = fixture_group(Path(temporary), mode='weak', ranks=4, n=128)
            for row in bv:
                row['elapsed_s'] = str(float(row['elapsed_s']) * 4)
            rows = analysis.summarize({a: av, b: bv})
            self.assertEqual(rows[1]['efficiency_pct'], 100)
            self.assertEqual(rows[1]['speedup_vs_one_gpu'], 'NA')

    def test_strict_profile_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'unit_test_profile.csv'
            row = dict(rank=0, device=0, n=64, steps=100, block_x=32, block_y=8,
                       kernel_s=0.1, d2h_s=0, h2d_s=0, mpi_s=0, total_s=0.2)
            def write(rows):
                with path.open('w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=bench.PROFILE_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
            write([row])
            self.assertEqual(len(bench.parse_profile(path, 64, 100, 1, 32, 8)), 1)
            write([row, row])
            with self.assertRaises(ValueError):
                bench.parse_profile(path, 64, 100, 2, 32, 8)
            write([dict(row, kernel_s='nan')])
            with self.assertRaises(ValueError):
                bench.parse_profile(path, 64, 100, 1, 32, 8)

    def test_physical_bounds_and_error_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'unit_test_field.csv'
            for value, accepted in [(0.5, True), (1 + 5e-11, False), (math.nan, False)]:
                rows = [[0.0] * 4 for _ in range(4)]
                rows[1][1] = value
                with path.open('w', newline='', encoding='utf-8') as file:
                    csv.writer(file).writerows(rows)
                if accepted:
                    self.assertEqual(len(validate_tool.read_field(path, 4)), 16)
                else:
                    with self.assertRaises(ValueError):
                        validate_tool.read_field(path, 4)
        with self.assertRaises(ValueError):
            validate_tool.compare_fields([0, 0.2], [0, 0.4])

    def test_dry_runs_launch_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'must_not_be_created'
            with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('Dry-run launched a process')):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.main(['--mode', 'size', '--sizes', '256', '512',
                        '--steps', '100', '--allocated-gpus', '1', '--output', str(destination), '--dry-run']), 0)
                    self.assertEqual(validate_tool.main(['--allocated-gpus', '1', '--output', str(destination), '--dry-run']), 0)
            self.assertFalse(destination.exists())

    def test_real_run_requires_allocated_job(self):
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.main(['--n', '64', '--steps', '100', '--allocated-gpus', '1', '--output', 'unused'])

    def test_command_provenance_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / 'real_cpu_process'
            command = [sys.executable, '-c', 'print("cpu-test")']
            self.assertEqual(bench.run_logged(command, prefix, 10).strip(), 'cpu-test')
            saved = json.loads(Path(str(prefix) + '.command.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['argv'], command)

    def test_serial_baselines_are_matched_by_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic_unit_test_serial.csv'
            rows = []
            for n, steps, factor in [(64, 100, 1), (128, 200, 10)]:
                for elapsed in [1, 2, 3, 4, 100]:
                    rows.append(dict(implementation='serial', n=n, steps=steps, r=0.1,
                                     ranks=1, gpus=0, elapsed_s=elapsed * factor,
                                     validation='not_run', max_abs_error='NA', relative_l2_error='NA'))
            def write(records):
                with path.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=bench.FIELDS)
                    writer.writeheader()
                    writer.writerows(records)
            write(rows)
            self.assertEqual(analysis.serial_baselines(path), {(64, 100): 3, (128, 200): 30})
            write(rows[:-1])
            with self.assertRaises(ValueError):
                analysis.serial_baselines(path)

    def test_actual_mpi_baseline_matches_the_selected_cpu_rank_count(self):
        mpi = analysis.mpi_baselines(ROOT / 'results' / 'mpi_strong.csv', 256)
        self.assertAlmostEqual(mpi[(4096, 1000)], 2.002071622)
        with self.assertRaises(ValueError):
            analysis.mpi_baselines(ROOT / 'results' / 'mpi_strong.csv', 3)


if __name__ == '__main__':
    unittest.main()
