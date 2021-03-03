import unittest
from avalanche.benchmarks import PermutedMNIST, Experience
from tests.unit_tests_utils import common_setups


class PermutedMNISTTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_PermutedMNIST_scenario(self):

        scenario = PermutedMNIST(3)
        self.assertEqual(3, len(scenario.train_stream))
        self.assertEqual(3, len(scenario.test_stream))

        for task_info in scenario.train_stream:
            self.assertIsInstance(task_info, Experience)
            self.assertEqual(60000, len(task_info.dataset))

        for task_info in scenario.test_stream:
            self.assertIsInstance(task_info, Experience)
            self.assertEqual(10000, len(task_info.dataset))


if __name__ == '__main__':
    unittest.main()
