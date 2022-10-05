import unittest
import random

import torch

from avalanche.benchmarks.utils.flattened_data import FlatData
from avalanche.benchmarks.utils.data import _flatdata_depth, _flatdata_print


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

        # If you broke this test it means that dataset merging is not working anymore.
        # you are probably doing something that disable merging (passing custom transforms?)
        # Good luck...
        assert _flatdata_depth(curr_dataset) == 2
