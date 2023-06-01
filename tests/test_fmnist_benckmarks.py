import unittest

import avalanche.benchmarks.datasets.external_datasets.fmnist
from avalanche.benchmarks import ClassificationExperience, SplitFMNIST

from tests.unit_tests_utils import (
    load_experience_train_eval,
    FAST_TEST,
    is_github_action,
)

MNIST_DOWNLOADS = 0
MNIST_DOWNLOAD_METHOD = None


class FMNISTBenchmarksTests(unittest.TestCase):
    def setUp(self):
        import avalanche.benchmarks.classic.cfashion_mnist as cfashion_mnist
        from avalanche.benchmarks.datasets.external_datasets.fmnist import (
            get_fmnist_dataset,
        )

        global MNIST_DOWNLOAD_METHOD
        MNIST_DOWNLOAD_METHOD = get_fmnist_dataset

        def count_downloads(*args, **kwargs):
            global MNIST_DOWNLOADS
            MNIST_DOWNLOADS += 1
            return MNIST_DOWNLOAD_METHOD(*args, **kwargs)

        avalanche.benchmarks.datasets.external_datasets.fmnist.get_fmnist_dataset = (
            count_downloads
        )

    def tearDown(self):
        global MNIST_DOWNLOAD_METHOD
        if MNIST_DOWNLOAD_METHOD is not None:
            import avalanche.benchmarks.classic.cfashion_mnist as cfashion_mnist

            avalanche.benchmarks.datasets.external_datasets.fmnist.get_fmnist_dataset = (
                MNIST_DOWNLOAD_METHOD
            )
            MNIST_DOWNLOAD_METHOD = None

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_SplitFMNIST_benchmark(self):
        benchmark = SplitFMNIST(5)
        self.assertEqual(5, len(benchmark.train_stream))
        self.assertEqual(5, len(benchmark.test_stream))

        train_sz = 0
        for experience in benchmark.train_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            train_sz += len(experience.dataset)

            load_experience_train_eval(experience)

        self.assertEqual(60000, train_sz)

        test_sz = 0
        for experience in benchmark.test_stream:
            self.assertIsInstance(experience, ClassificationExperience)
            test_sz += len(experience.dataset)

            load_experience_train_eval(experience)

        self.assertEqual(10000, test_sz)


if __name__ == "__main__":
    unittest.main()
