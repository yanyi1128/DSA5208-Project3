"""Run: python -m unittest discover -s analysis -p test_mpi_analysis.py"""
import unittest
from mpi_analysis import metrics, parse_bindings


class AnalysisTests(unittest.TestCase):
    def test_strong_and_weak_use_different_efficiency_definitions(self):
        strong = metrics([2, 4, 100, 4, 4], 16, 8, 'strong')
        weak = metrics([2, 4, 100, 4, 4], 16, 8, 'weak')
        self.assertEqual(strong['median_s'], 4)
        self.assertEqual(strong['max_s'], 100)  # retain anomalous observation
        self.assertEqual(strong['time_ratio'], 4)
        self.assertEqual(strong['efficiency_pct'], 50)
        self.assertEqual(weak['efficiency_pct'], 400)  # do not clamp efficiency

    def test_core_ids_on_different_hosts_are_distinct(self):
        content = ('[CN-1:12] MCW rank 0 bound to socket 0[core 0[hwt 0]]\n'
                   '[CN-2:13] MCW rank 1 bound to socket 0[core 0[hwt 0]]')
        self.assertEqual(len(parse_bindings(content, 2)), 2)
        with self.assertRaises(ValueError):
            parse_bindings(content.replace('CN-2', 'CN-1'), 2)
        with self.assertRaises(ValueError):
            parse_bindings(content.replace('rank 1', 'rank 0'), 2)


if __name__ == '__main__':
    unittest.main()
