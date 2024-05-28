""" Disk Usage Metric Test"""

import unittest

from avalanche.evaluation.metrics import DiskUsage


class DiskUsageTests(unittest.TestCase):
    def test_basic(self):
        """just checking that directory size is computed without errors."""

        disk = DiskUsage()
        disk.get_dir_size(".")


if __name__ == "__main__":
    unittest.main()
