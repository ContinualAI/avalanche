import unittest

from avalanche.benchmarks import PermutedMNIST, Experience, RotatedMNIST, \
    SplitMNIST
from tests.unit_tests_utils import load_experience_train_eval

MNIST_DOWNLOADS = 0
MNIST_DOWNLOAD_METHOD = None


class MNISTBenchmarksTests(unittest.TestCase):
    def setUp(self):
        import avalanche.benchmarks.classic.cmnist as cmnist
        global MNIST_DOWNLOAD_METHOD
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

    def test_SplitMNIST_scenario(self):
        scenario = SplitMNIST(5)
        self.assertEqual(5, len(scenario.train_stream))
        self.assertEqual(5, len(scenario.test_stream))

        train_sz = 0
        for experience in scenario.train_stream:
            self.assertIsInstance(experience, Experience)
            train_sz += len(experience.dataset)

            # Regression test for 572
            load_experience_train_eval(experience)

        self.assertEqual(60000, train_sz)

        test_sz = 0
        for experience in scenario.test_stream:
            self.assertIsInstance(experience, Experience)
            test_sz += len(experience.dataset)

            # Regression test for 572
            load_experience_train_eval(experience)

        self.assertEqual(10000, test_sz)

    def test_PermutedMNIST_scenario(self):
        scenario = PermutedMNIST(3)
        self.assertEqual(3, len(scenario.train_stream))
        self.assertEqual(3, len(scenario.test_stream))

        for experience in scenario.train_stream:
            self.assertIsInstance(experience, Experience)
            self.assertEqual(60000, len(experience.dataset))

            load_experience_train_eval(experience)

        for experience in scenario.test_stream:
            self.assertIsInstance(experience, Experience)
            self.assertEqual(10000, len(experience.dataset))

            load_experience_train_eval(experience)

    def test_RotatedMNIST_scenario(self):
        scenario = RotatedMNIST(3)
        self.assertEqual(3, len(scenario.train_stream))
        self.assertEqual(3, len(scenario.test_stream))

        for experience in scenario.train_stream:
            self.assertIsInstance(experience, Experience)
            self.assertEqual(60000, len(experience.dataset))

            load_experience_train_eval(experience)

        for experience in scenario.test_stream:
            self.assertIsInstance(experience, Experience)
            self.assertEqual(10000, len(experience.dataset))

            load_experience_train_eval(experience)

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

    # def test_PermutedMNIST_scenario_performance(self):
    #     import time
    #     from torch.utils.data.dataloader import DataLoader
    #     start_time = time.time()
    #     scenario = PermutedMNIST(10)
    #
    #     for experience in scenario.train_stream:
    #         self.assertIsInstance(experience, Experience)
    #         self.assertEqual(60000, len(experience.dataset))
    #         all_targets = sum(experience.dataset.targets)
    #
    #         # dataset = experience.dataset
    #         # loader = DataLoader(dataset, num_workers=4, shuffle=True,
    #         #                     batch_size=256)
    #         # for batch in loader:
    #         #     x, y, t = batch
    #
    #     for experience in scenario.test_stream:
    #         self.assertIsInstance(experience, Experience)
    #         self.assertEqual(10000, len(experience.dataset))
    #         all_targets = sum(experience.dataset.targets)
    #
    #         # dataset = experience.dataset
    #         # loader = DataLoader(dataset, num_workers=4, shuffle=True,
    #         #                     batch_size=256)
    #         # for batch in loader:
    #         #     x, y, t = batch
    #
    #     elapsed_time = time.time() - start_time
    #     print('Elapsed:', elapsed_time)


if __name__ == '__main__':
    unittest.main()
