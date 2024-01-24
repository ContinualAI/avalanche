import sys
import unittest
import random

import torch

from avalanche.benchmarks import FixedSizeExperienceSplitter
from avalanche.benchmarks.utils import AvalancheDataset, concat_datasets
from avalanche.benchmarks.utils.classification_dataset import (
    TaskAwareClassificationDataset,
)
from avalanche.benchmarks.utils.flat_data import (
    FlatData,
    LazyRange,
    _flatten_datasets_and_reindex,
    LazyIndices,
)
from avalanche.benchmarks.utils.flat_data import (
    _flatdata_depth,
    _flatdata_print,
)
from avalanche.training import ReservoirSamplingBuffer
from tests.unit_tests_utils import get_fast_benchmark


class AvalancheDatasetTests(unittest.TestCase):
    def test_flatdata_subset_concat_stack_overflow(self):
        d_sz = 5
        x_raw = torch.randint(0, 7, (d_sz,))
        data = FlatData([x_raw])
        dataset_hierarchy_depth = 500

        # prepare random permutations for each step
        perms = []
        for _ in range(dataset_hierarchy_depth):
            idx_permuted = list(range(d_sz))
            random.shuffle(idx_permuted)
            perms.append(idx_permuted)

        # compute expected indices after all permutations
        current_indices = range(d_sz)
        true_indices = []
        true_indices.append(list(current_indices))
        for idx in range(dataset_hierarchy_depth):
            current_indices = [current_indices[x] for x in perms[idx]]
            true_indices.append(current_indices)
        true_indices = list(reversed(true_indices))

        # apply permutations and concatenations iteratively
        curr_dataset = data
        for idx in range(dataset_hierarchy_depth):
            # print(idx)
            # print(idx, "depth: ", _flatdata_depth(curr_dataset))

            subset = curr_dataset.subset(indices=perms[idx])
            # print("SUBSET:")
            # _flatdata_print(subset)

            curr_dataset = subset.concat(curr_dataset)
            # print("CONCAT:")
            # _flatdata_print(curr_dataset)

        self.assertEqual(d_sz * dataset_hierarchy_depth + d_sz, len(curr_dataset))
        for idx in range(dataset_hierarchy_depth):
            leaf_range = range(idx * d_sz, (idx + 1) * d_sz)
            permuted = true_indices[idx]

            x_leaf = torch.stack([curr_dataset[idx] for idx in leaf_range], dim=0)
            self.assertTrue(torch.equal(x_raw[permuted], x_leaf))

        slice_idxs = list(range(d_sz * dataset_hierarchy_depth, len(curr_dataset)))
        x_slice = torch.stack([curr_dataset[idx] for idx in slice_idxs], dim=0)
        self.assertTrue(torch.equal(x_raw, x_slice))

        # If you broke this test it means that dataset merging is not working
        # anymore. you are probably doing something that disable merging
        # (passing custom transforms?)
        # Good luck...
        assert _flatdata_depth(curr_dataset) == 2

    def test_merging(self):
        x = torch.randn(10)
        fdata = FlatData([x])

        dd = fdata
        for i in range(5):
            dd = dd.concat(fdata)
            assert _flatdata_depth(dd) == 2
            assert len(dd._datasets) == 1

            idxs = list(range(len(dd)))
            random.shuffle(idxs)
            dd_old = dd
            dd = dd.subset(idxs[:12])

            for i in range(12):
                assert dd[i] == dd_old[idxs[i]]

            assert _flatdata_depth(dd) == 2
            assert len(dd._indices) == 12
            assert len(dd._datasets) == 1


class FlatteningTests(unittest.TestCase):
    def test_flatten_and_reindex(self):
        bm = get_fast_benchmark()
        D1 = bm.train_stream[0].dataset
        ds, idxs = _flatten_datasets_and_reindex([D1, D1, D1], None)

        print(
            f"len-ds: {len(ds)}, max={max(idxs)}, min={min(idxs)}, "
            f"lens={[len(d) for d in ds]}"
        )
        assert len(ds) == 1
        assert len(idxs) == 3 * len(D1)
        assert max(idxs) == len(D1) - 1
        assert min(idxs) == 0

    def test_concat_flattens_same_dataset(self):
        D = AvalancheDataset(
            [[1, 2, 3]],
        )
        B = concat_datasets([])
        B = B.concat(D)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")

        for _ in range(10):
            B = D.concat(B)
            print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
            # assert _flatdata_depth(B) <= 2
            # assert len(B._datasets) <= 2

    def test_concat_flattens_same_dataset_corner_case(self):
        base_dataset = [1, 2, 3]
        A = FlatData([base_dataset], can_flatten=False, indices=[1, 2])
        B = FlatData([A])
        C = A.concat(B)
        C[3]
        self.assertListEqual([2, 3, 2, 3], list(C))

        A = FlatData([base_dataset], can_flatten=False)
        B = FlatData([A], indices=[1, 2])
        C = A.concat(B)
        self.assertListEqual([1, 2, 3, 2, 3], list(C))

        A = FlatData([base_dataset], can_flatten=False, indices=[1, 2])
        B = FlatData([A])
        C = B.concat(A)
        self.assertListEqual([2, 3, 2, 3], list(C))

        A = FlatData([base_dataset], can_flatten=False)
        B = FlatData([A], indices=[1, 2])
        C = B.concat(A)
        self.assertListEqual([2, 3, 1, 2, 3], list(C))

    def test_concat_flattens_same_classification_dataset(self):
        D = TaskAwareClassificationDataset([[1, 2, 3]])
        B = concat_datasets([])
        B = B.concat(D)
        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) <= 2
        assert len(B._datasets) <= 2
        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) <= 2
        assert len(B._datasets) <= 2

        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) <= 2
        assert len(B._datasets) <= 2

    def test_concat_flattens_nc_scenario_dataset(self):
        benchmark = get_fast_benchmark()
        s = benchmark.train_stream
        B = concat_datasets([s[1].dataset])
        D1 = s[0].dataset

        B1 = D1.concat(B)
        print(f"DATA depth={_flatdata_depth(B1)}, dsets={len(B1._datasets)}")
        assert len(B1._datasets) <= 2
        B2 = D1.concat(B1)
        print(f"DATA depth={_flatdata_depth(B2)}, dsets={len(B2._datasets)}")
        assert len(B2._datasets) <= 2
        B3 = D1.concat(B2)
        print(f"DATA depth={_flatdata_depth(B3)}, dsets={len(B3._datasets)}")
        assert len(B3._datasets) <= 2

    def test_concat_flattens_nc_scenario_dataset2(self):
        bm = get_fast_benchmark()
        s = bm.train_stream

        B = concat_datasets([])  # empty dataset
        D1 = s[0].dataset
        print(repr(D1))

        D2a = s[1].dataset
        D2b = s[1].dataset

        B1 = D1.concat(B)
        print(f"DATA depth={_flatdata_depth(B1)}, dsets={len(B1._datasets)}")
        print(repr(B1))
        assert len(B1._datasets) <= 2
        B2 = D2a.concat(B1)
        print(f"DATA depth={_flatdata_depth(B2)}, dsets={len(B2._datasets)}")
        print(repr(B2))
        assert len(B2._datasets) <= 2
        B3 = D2b.concat(B2)
        print(f"DATA depth={_flatdata_depth(B3)}, dsets={len(B3._datasets)}")
        print(repr(B3))
        assert len(B3._datasets) <= 2

    def test_flattening_replay_ocl(self):
        benchmark = get_fast_benchmark()
        buffer = ReservoirSamplingBuffer(100)

        for t, exp in enumerate(
            FixedSizeExperienceSplitter(benchmark.train_stream[0], 1, None)
        ):
            buffer.update_from_dataset(exp.dataset)
            b = buffer.buffer
            # depths = _flatdata_depth(b)
            # lenidxs = len(b._indices)
            # lendsets = len(b._datasets)
            # print(f"DATA depth={depths}, idxs={lenidxs}, dsets={lendsets}")
            #
            # atts = [b.targets.data, b.targets_task_labels.data]
            # depths = [_flatdata_depth(b) for b in atts]
            # lenidxs = [len(b._indices) for b in atts]
            # lendsets = [len(b._datasets) for b in atts]
            # print(f"(t={t}) ATTS depth={depths}, idxs={lenidxs},
            # dsets={lendsets}")
            if t > 5:
                break
        print(f"DATA depth={_flatdata_depth(b)}, dsets={len(b._datasets)}")
        assert len(b._datasets) <= 2

        for t, exp in enumerate(
            FixedSizeExperienceSplitter(benchmark.train_stream[1], 1, None)
        ):
            buffer.update_from_dataset(exp.dataset)
            b = buffer.buffer
            # depths = _flatdata_depth(b)
            # lenidxs = len(b._indices)
            # lendsets = len(b._datasets)
            # print(f"DATA depth={depths}, idxs={lenidxs}, dsets={lendsets}")
            #
            # atts = [b.targets.data, b.targets_task_labels.data]
            # depths = [_flatdata_depth(b) for b in atts]
            # lenidxs = [len(b._indices) for b in atts]
            # lendsets = [len(b._datasets) for b in atts]
            # print(f"(t={t}) ATTS depth={depths}, idxs={lenidxs},
            # dsets={lendsets}")
            if t > 5:
                break
        print(f"DATA depth={_flatdata_depth(b)}, dsets={len(b._datasets)}")
        assert len(b._datasets) <= 2


class LazyIndicesTests(unittest.TestCase):
    def test_basic(self):
        eager = list(range(10))
        li = LazyIndices(eager)
        self.assertListEqual(eager, list(li))
        self.assertEqual(len(eager), len(li))

        li = LazyIndices(eager, eager)
        self.assertListEqual(eager + eager, list(li))
        self.assertEqual(len(eager) * 2, len(li))

        li = LazyIndices(eager, offset=7)
        self.assertListEqual(list([el + 7 for el in eager]), list(li))
        self.assertEqual(len(eager), len(li))

    def test_range(self):
        eager = list(range(1, 11))
        li = LazyRange(start=1, end=11)
        self.assertListEqual(eager, list(li))
        self.assertEqual(len(eager), len(li))

        eager = list(range(1, 11))
        li = LazyRange(start=0, end=10, offset=1)
        self.assertListEqual(eager, list(li))
        self.assertEqual(len(eager), len(li))

        eager = list(range(8, 18)) + list(range(12, 15))
        a = LazyRange(start=0, end=10, offset=1)
        b = LazyRange(start=2, end=5, offset=3)
        li = LazyIndices(a, b, offset=7)
        self.assertListEqual(eager, list(li))
        self.assertEqual(len(eager), len(li))

    def test_recursion(self):
        eager = list(range(10))

        li = LazyIndices(eager, offset=0)
        # TODO: speed up this test. Can we avoid checking such a high limit?
        limit = sys.getrecursionlimit() * 2 + 10
        for i in range(limit):
            li = LazyIndices(li, eager, offset=0)

        self.assertEqual(len(eager) * (i + 2), len(li))
        for el in li:  # keep this to check recursion error
            pass


if __name__ == "__main__":
    unittest.main()
