import unittest
from pathlib import Path

from tempfile import TemporaryDirectory
import torch

from avalanche.benchmarks.classic.ctrl import CTrL
from tests.unit_tests_utils import FAST_TEST, is_github_action


def custom_equals(item, other) -> bool:
    """
    Helper function allowing to test if two items are equal.
    The function is called recursively if the items are lists or tuples and
     it uses `torch.equal` if the two items to compare are Tensors.
    """
    if type(item) != type(other):
        return False
    if isinstance(item, (tuple, list)):
        if len(item) != len(other):
            return False
        return all(custom_equals(*elts) for elts in zip(item, other))
    if isinstance(item, torch.Tensor):
        return torch.equal(item, other)
    return item == other


class CTrLTests(unittest.TestCase):
    stream_lengths = dict(
        s_plus=6,
        s_minus=6,
        s_in=6,
        s_out=6,
        s_pl=5,
    )

    long_stream_lengths = [8, 15]

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_length(self):
        for stream, length in self.stream_lengths.items():
            with self.subTest(stream=stream, length=length):
                bench = CTrL(stream)
                self.assertEqual(length, bench.n_experiences)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_length_long(self):
        for n_tasks in self.long_stream_lengths:
            with self.subTest(n_tasks=n_tasks), TemporaryDirectory() as tmp:
                bench = CTrL(
                    "s_long", save_to_disk=True, path=Path(tmp), n_tasks=n_tasks
                )
                self.assertEqual(n_tasks, bench.n_experiences)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_n_tasks_param(self):
        for stream in self.stream_lengths.keys():
            with self.subTest(stream=stream):
                with self.assertRaises(ValueError):
                    CTrL(stream, n_tasks=3)

        with self.subTest(stream="s_long"):
            CTrL("s_long", n_tasks=3)

    @unittest.skipIf(
        FAST_TEST or is_github_action(),
        "We don't want to download large datasets in github actions.",
    )
    def test_determinism(self):
        for stream in self.stream_lengths.keys():
            with self.subTest(stream=stream):
                bench_1 = CTrL(stream, seed=1)
                bench_2 = CTrL(stream, seed=1)

                for exp1, exp2 in zip(
                    bench_1.train_stream, bench_2.train_stream
                ):
                    for sample1, sample2 in zip(exp1.dataset, exp2.dataset):
                        self.assertTrue(custom_equals(sample1, sample2))
