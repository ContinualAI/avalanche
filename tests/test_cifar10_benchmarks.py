import unittest

from avalanche.benchmarks import Experience, SplitCIFAR10
from tests.unit_tests_utils import (
    load_experience_train_eval,
    FAST_TEST,
    is_github_action,
)

CIFAR10_DOWNLOADS = 0
CIFAR10_DOWNLOAD_METHOD = None


class CIFAR10BenchmarksTests(unittest.TestCase):
    def setUp(self):
        import avalanche.benchmarks.classic.ccifar10 as ccifar10

        global CIFAR10_DOWNLOAD_METHOD
        CIFAR10_DOWNLOAD_METHOD = ccifar10._get_cifar10_dataset

        def count_downloads(*args, **kwargs):
            global CIFAR10_DOWNLOADS
            CIFAR10_DOWNLOADS += 1
            return CIFAR10_DOWNLOAD_METHOD(*args, **kwargs)

        ccifar10._get_cifar10_dataset = count_downloads

    def tearDown(self):
        global CIFAR10_DOWNLOAD_METHOD
        if CIFAR10_DOWNLOAD_METHOD is not None:
            import avalanche.benchmarks.classic.ccifar10 as ccifar10

            ccifar10._get_cifar10_dataset = CIFAR10_DOWNLOAD_METHOD
            CIFAR10_DOWNLOAD_METHOD = None

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_SplitCifar10_benchmark(self):
        benchmark = SplitCIFAR10(5)
        self.assertEqual(5, len(benchmark.train_stream))
        self.assertEqual(5, len(benchmark.test_stream))

        train_sz = 0
        for experience in benchmark.train_stream:
            self.assertIsInstance(experience, Experience)
            train_sz += len(experience.dataset)

            # Regression test for 575
            load_experience_train_eval(experience)

        self.assertEqual(50000, train_sz)

        test_sz = 0
        for experience in benchmark.test_stream:
            self.assertIsInstance(experience, Experience)
            test_sz += len(experience.dataset)

            # Regression test for 575
            load_experience_train_eval(experience)

        self.assertEqual(10000, test_sz)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_SplitCifar10_benchmark_download_once(self):
        global CIFAR10_DOWNLOADS
        CIFAR10_DOWNLOADS = 0

        benchmark = SplitCIFAR10(5)
        self.assertEqual(5, len(benchmark.train_stream))
        self.assertEqual(5, len(benchmark.test_stream))

        self.assertEqual(1, CIFAR10_DOWNLOADS)


if __name__ == "__main__":
    unittest.main()
