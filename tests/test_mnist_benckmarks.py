import unittest
from avalanche.benchmarks import PermutedMNIST, Experience, RotatedMNIST
from tests.unit_tests_utils import common_setups

MNIST_DOWNLOADS = 0
MNIST_DOWNLOAD_METHOD = None


class MNISTBenchmarksTests(unittest.TestCase):
    def setUp(self):
        import avalanche.benchmarks.classic.cmnist as cmnist
        global MNIST_DOWNLOAD_METHOD
        common_setups()
        MNIST_DOWNLOAD_METHOD = cmnist._get_mnist_dataset

        def count_downloads(*args, **kwargs):
            global MNIST_DOWNLOADS
            MNIST_DOWNLOADS += 1
            return MNIST_DOWNLOAD_METHOD(*args, **kwargs)

        cmnist._get_mnist_dataset = count_downloads

    def tearDown(self):
        global MNIST_DOWNLOAD_METHOD
        if MNIST_DOWNLOAD_METHOD is not None:
            import avalanche.benchmarks.classic.cmnist as cmnist
            cmnist._get_mnist_dataset = MNIST_DOWNLOAD_METHOD
            MNIST_DOWNLOAD_METHOD = None

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

    def test_PermutedMNIST_scenario_download_once(self):
        global MNIST_DOWNLOADS
        MNIST_DOWNLOADS = 0

        scenario = PermutedMNIST(3)
        self.assertEqual(3, len(scenario.train_stream))
        self.assertEqual(3, len(scenario.test_stream))

        self.assertEqual(1, MNIST_DOWNLOADS)

    def test_RotatedMNIST_scenario_download_once(self):
        global MNIST_DOWNLOADS
        MNIST_DOWNLOADS = 0

        scenario = RotatedMNIST(3)
        self.assertEqual(3, len(scenario.train_stream))
        self.assertEqual(3, len(scenario.test_stream))

        self.assertEqual(1, MNIST_DOWNLOADS)


if __name__ == '__main__':
    unittest.main()
