import unittest
import random

import torch

from avalanche.benchmarks import fixed_size_experience_split
from avalanche.benchmarks.utils import AvalancheDataset, \
    concat_datasets
from avalanche.benchmarks.utils.classification_dataset import \
    ClassificationDataset
from avalanche.benchmarks.utils.flat_data import FlatData, \
    _flatten_datasets_and_reindex
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

        self.assertEqual(
            d_sz * dataset_hierarchy_depth + d_sz, len(curr_dataset)
        )
        for idx in range(dataset_hierarchy_depth):
            leaf_range = range(idx * d_sz, (idx + 1) * d_sz)
            permuted = true_indices[idx]

            x_leaf = torch.stack(
                [curr_dataset[idx] for idx in leaf_range], dim=0
            )
            self.assertTrue(torch.equal(x_raw[permuted], x_leaf))

        slice_idxs = list(
            range(d_sz * dataset_hierarchy_depth, len(curr_dataset))
        )
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
            dd = dd.subset(idxs[:12])

            assert _flatdata_depth(dd) == 2
            assert len(dd._indices) == 12
            assert len(dd._datasets) == 1


class FlatteningTests(unittest.TestCase):
    def test_flatten_and_reindex(self):
        bm = get_fast_benchmark()
        D1 = bm.train_stream[0].dataset
        ds, idxs = _flatten_datasets_and_reindex([D1, D1, D1], None)

        print(f"len-ds: {len(ds)}, max={max(idxs)}, min={min(idxs)}, "
              f"lens={[len(d) for d in ds]}")
        assert len(ds) == 1
        assert len(idxs) == 3 * len(D1)
        assert max(idxs) == len(D1) - 1
        assert min(idxs) == 0

    def test_concat_flattens_same_dataset(self):
        D = AvalancheDataset([[1, 2, 3]])
        B = concat_datasets([])
        B = B.concat(D)
        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) == 2
        assert len(B._datasets) == 1
        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) == 2
        assert len(B._datasets) == 1

        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) == 2
        assert len(B._datasets) == 1

    def test_concat_flattens_same_classification_dataset(self):
        D = ClassificationDataset([[1, 2, 3]])
        B = concat_datasets([])
        B = B.concat(D)
        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) == 2
        assert len(B._datasets) == 1
        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) == 2
        assert len(B._datasets) == 1

        B = D.concat(B)
        print(f"DATA depth={_flatdata_depth(B)}, dsets={len(B._datasets)}")
        assert _flatdata_depth(B) == 2
        assert len(B._datasets) == 1

    def test_concat_flattens_nc_scenario_dataset(self):
        benchmark = get_fast_benchmark()
        s = benchmark.train_stream
        B = concat_datasets([s[1].dataset])
        D1 = s[0].dataset

        B1 = D1.concat(B)
        print(f"DATA depth={_flatdata_depth(B1)}, dsets={len(B1._datasets)}")
        assert len(B1._datasets) == 2
        B2 = D1.concat(B1)
        print(f"DATA depth={_flatdata_depth(B2)}, dsets={len(B2._datasets)}")
        assert len(B2._datasets) == 2
        B3 = D1.concat(B2)
        print(f"DATA depth={_flatdata_depth(B3)}, dsets={len(B3._datasets)}")
        assert len(B3._datasets) == 2

    def test_concat_flattens_nc_scenario_dataset2(self):
        bm = get_fast_benchmark()
        s = bm.train_stream

        B = concat_datasets([])  # empty dataset
        D1 = s[0].dataset
        D2a = s[1].dataset
        D2b = s[1].dataset

        B1 = D1.concat(B)
        print(f"DATA depth={_flatdata_depth(B1)}, dsets={len(B1._datasets)}")
        assert len(B1._datasets) == 1
        B2 = D2a.concat(B1)
        print(f"DATA depth={_flatdata_depth(B2)}, dsets={len(B2._datasets)}")
        assert len(B2._datasets) == 2
        B3 = D2b.concat(B2)
        print(f"DATA depth={_flatdata_depth(B3)}, dsets={len(B3._datasets)}")
        assert len(B3._datasets) == 2

    def test_flattening_replay_ocl(self):
        benchmark = get_fast_benchmark()
        buffer = ReservoirSamplingBuffer(100)

        for t, exp in enumerate(fixed_size_experience_split(
                benchmark.train_stream[0], 1)):
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
        assert len(b._datasets) == 1

        for t, exp in enumerate(fixed_size_experience_split(
                benchmark.train_stream[1], 1)):
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
        assert len(b._datasets) == 2
