import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).parents[1] / 'scripts/benchmark_mpi.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_weak_scaling(self):
        self.assertEqual(benchmark.grid_size('weak', 64, 256), 1024)
        self.assertEqual(benchmark.grid_size('weak', 64, 2), 91)
        self.assertEqual(benchmark.grid_size('strong', 64, 256), 64)

    def test_csv_checks(self):
        header = ','.join(benchmark.FIELDS) + '\n'
        row = 'mpi,8,7,0.1,3,0,0.02,pass,0,0\n'
        self.assertEqual(benchmark.parse_record(header+row, 8, 7, 3, True)['validation'], 'pass')
        for invalid in [row.replace(',pass,', ',fail,'), row.replace('0.02','nan'),
                        row.replace('mpi,8', 'mpi,9'), row+row]:
            with self.assertRaises(ValueError):
                benchmark.parse_record(header+invalid, 8, 7, 3, True)


if __name__ == '__main__':
    unittest.main()
