import unittest

import avalanche.benchmarks.datasets.external_datasets.mnist as mnist_download
import avalanche.benchmarks.classic.cmnist as mnist_benchmark
from avalanche.benchmarks import (
    PermutedMNIST,
    ClassificationExperience,
    RotatedMNIST,
    SplitMNIST,
)
from tests.unit_tests_utils import (
    load_experience_train_eval,
    FAST_TEST,
    is_github_action,
)


MNIST_DOWNLOADS = 0
MNIST_DOWNLOAD_METHOD = None


def count_downloads(*args, **kwargs):
    global MNIST_DOWNLOADS
    MNIST_DOWNLOADS += 1
    return MNIST_DOWNLOAD_METHOD(*args, **kwargs)


class MNISTBenchmarksTests(unittest.TestCase):
    def setUp(self):
        global MNIST_DOWNLOAD_METHOD
        MNIST_DOWNLOAD_METHOD = mnist_download.get_mnist_dataset

        mnist_download.get_mnist_dataset = count_downloads
        mnist_benchmark.get_mnist_dataset = count_downloads

    def tearDown(self):
        global MNIST_DOWNLOAD_METHOD
        if MNIST_DOWNLOAD_METHOD is not None:
            mnist_download.get_mnist_dataset = MNIST_DOWNLOAD_METHOD
            mnist_benchmark.get_mnist_dataset = MNIST_DOWNLOAD_METHOD

            MNIST_DOWNLOAD_METHOD = None

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_SplitMNIST_benchmark(self):
        benchmark = SplitMNIST(5)
        self.assertEqual(5, len(benchmark.train_stream))
        self.assertEqual(5, len(benchmark.test_stream))

        train_sz = 0
        for experience in benchmark.train_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            train_sz += len(experience.dataset)

            # Regression test for 572
            load_experience_train_eval(experience)

        self.assertEqual(60000, train_sz)

        test_sz = 0
        for experience in benchmark.test_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            test_sz += len(experience.dataset)

            # Regression test for 572
            load_experience_train_eval(experience)

        self.assertEqual(10000, test_sz)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_PermutedMNIST_benchmark(self):
        benchmark = PermutedMNIST(3)
        self.assertEqual(3, len(benchmark.train_stream))
        self.assertEqual(3, len(benchmark.test_stream))

        for experience in benchmark.train_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            self.assertEqual(60000, len(experience.dataset))

            load_experience_train_eval(experience)

        for experience in benchmark.test_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            self.assertEqual(10000, len(experience.dataset))

            load_experience_train_eval(experience)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_RotatedMNIST_benchmark(self):
        benchmark = RotatedMNIST(3)
        self.assertEqual(3, len(benchmark.train_stream))
        self.assertEqual(3, len(benchmark.test_stream))

        for experience in benchmark.train_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            self.assertEqual(60000, len(experience.dataset))

            load_experience_train_eval(experience)

        for experience in benchmark.test_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            self.assertEqual(10000, len(experience.dataset))

            load_experience_train_eval(experience)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_PermutedMNIST_benchmark_download_once(self):
        global MNIST_DOWNLOADS
        MNIST_DOWNLOADS = 0

        benchmark = PermutedMNIST(3)
        self.assertEqual(3, len(benchmark.train_stream))
        self.assertEqual(3, len(benchmark.test_stream))

        self.assertEqual(1, MNIST_DOWNLOADS)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_RotatedMNIST_benchmark_download_once(self):
        global MNIST_DOWNLOADS
        MNIST_DOWNLOADS = 0

        benchmark = RotatedMNIST(3)
        self.assertEqual(3, len(benchmark.train_stream))
        self.assertEqual(3, len(benchmark.test_stream))

        self.assertEqual(1, MNIST_DOWNLOADS)

    # def test_PermutedMNIST_benchmark_performance(self):
    #     import time
    #     from torch.utils.data.dataloader import DataLoader
    #     start_time = time.time()
    #     benchmark = PermutedMNIST(10)
    #
    #     for experience in benchmark.train_stream:
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
    #     for experience in benchmark.test_stream:
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


if __name__ == "__main__":
    unittest.main()
