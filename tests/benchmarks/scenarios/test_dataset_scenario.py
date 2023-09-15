import unittest

import torch
from torch.utils.data import TensorDataset, DataLoader

from avalanche.benchmarks import benchmark_from_datasets, benchmark_with_validation_stream, CLScenario, CLStream, \
    split_dataset_random
from avalanche.benchmarks.utils import AvalancheDataset


def get_mbatch(data, batch_size=5):
    dl = DataLoader(
        data, shuffle=False, batch_size=batch_size, collate_fn=data.collate_fn
    )
    return next(iter(dl))


class DatasetScenarioTests(unittest.TestCase):
    def test_split_dataset_by_attribute(self):
        return
        raise NotImplementedError()

    def test_benchmark_from_datasets(self):
        return
        raise NotImplementedError()

    def test_split_dataset_random(self):
        return
        split_dataset_random()
        raise NotImplementedError()


    def test_benchmark_with_validation_stream_fixed_size(self):
        pattern_shape = (3, 32, 32)

        # Definition of training experiences
        # Experience 1
        experience_1_x = torch.zeros(100, *pattern_shape)
        experience_1_y = torch.zeros(100, dtype=torch.long)
        d1 = TensorDataset(experience_1_x, experience_1_y)
        d1 = AvalancheDataset(d1)
        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        d2 = TensorDataset(experience_2_x, experience_2_y)
        d2 = AvalancheDataset(d2)

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)
        dtest = TensorDataset(test_x, test_y)
        dtest = AvalancheDataset(dtest)

        bm = benchmark_from_datasets(train=[d1, d2], test=[dtest])
        valid_bm = benchmark_with_validation_stream(bm, 20, shuffle=False)

        self.assertEqual(2, len(valid_bm.train_stream))
        self.assertEqual(2, len(valid_bm.valid_stream))
        self.assertEqual(1, len(valid_bm.test_stream))

        self.assertEqual(80, len(valid_bm.train_stream[0].dataset))
        self.assertEqual(60, len(valid_bm.train_stream[1].dataset))
        self.assertEqual(20, len(valid_bm.valid_stream[0].dataset))
        self.assertEqual(20, len(valid_bm.valid_stream[1].dataset))

        vd = valid_bm.train_stream[0].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_1_x[:80], mb[0]))
        self.assertTrue(torch.equal(experience_1_y[:80], mb[1]))

        vd = valid_bm.train_stream[1].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_2_x[:60], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[:60], mb[1]))

        vd = valid_bm.valid_stream[0].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_1_x[80:], mb[0]))
        self.assertTrue(torch.equal(experience_1_y[80:], mb[1]))

        vd = valid_bm.valid_stream[1].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(experience_2_x[60:], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[60:], mb[1]))

        vd = valid_bm.test_stream[0].dataset
        mb = get_mbatch(vd, len(vd))
        self.assertTrue(torch.equal(test_x, mb[0]))
        self.assertTrue(torch.equal(test_y, mb[1]))

    def test_benchmark_with_validation_stream_rel_size(self):
        pattern_shape = (3, 32, 32)

        # Definition of training experiences
        # Experience 1
        experience_1_x = torch.zeros(100, *pattern_shape)
        experience_1_y = torch.zeros(100, dtype=torch.long)
        d1 = TensorDataset(experience_1_x, experience_1_y)
        d1 = AvalancheDataset(d1)
        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        d2 = TensorDataset(experience_2_x, experience_2_y)
        d2 = AvalancheDataset(d2)

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)
        dtest = TensorDataset(test_x, test_y)
        dtest = AvalancheDataset(dtest)

        bm = benchmark_from_datasets(train=[d1, d2], test=[dtest])
        valid_bm = benchmark_with_validation_stream(bm, 0.2, shuffle=False)

        true_rel_1_valid = int(100 * 0.2)
        true_rel_1_train = 100 - true_rel_1_valid
        true_rel_2_valid = int(80 * 0.2)
        true_rel_2_train = 80 - true_rel_2_valid

        self.assertEqual(2, len(valid_bm.train_stream))
        self.assertEqual(2, len(valid_bm.valid_stream))
        self.assertEqual(1, len(valid_bm.test_stream))

        ts = valid_bm.train_stream
        self.assertEqual(true_rel_1_train, len(ts[0].dataset))
        self.assertEqual(true_rel_2_train, len(ts[1].dataset))

        stm = valid_bm.valid_stream
        self.assertEqual(true_rel_1_valid, len(stm[0].dataset))
        self.assertEqual(true_rel_2_valid, len(stm[1].dataset))

        dd = valid_bm.train_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_x[:true_rel_1_train], mb[0]))

        dd = valid_bm.train_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_x[:true_rel_2_train], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[:true_rel_2_train], mb[1]))

        dd = valid_bm.train_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_x[:true_rel_1_train], mb[0]))
        self.assertTrue(torch.equal(experience_1_y[:true_rel_1_train], mb[1]))

        dd = valid_bm.valid_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_x[true_rel_2_train:], mb[0]))
        self.assertTrue(torch.equal(experience_2_y[true_rel_2_train:], mb[1]))

        dd = valid_bm.valid_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_y[true_rel_1_train:], mb[1]))

        dd = valid_bm.test_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(test_x, mb[0]))
        self.assertTrue(torch.equal(test_y, mb[1]))

    def test_lazy_benchmark_with_validation_stream_fixed_size(self):
        pattern_shape = (3, 32, 32)

        # Definition of training experiences
        # Experience 1
        experience_1_x = torch.zeros(100, *pattern_shape)
        experience_1_y = torch.zeros(100, dtype=torch.long)
        d1 = TensorDataset(experience_1_x, experience_1_y)
        d1 = AvalancheDataset(d1)
        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        d2 = TensorDataset(experience_2_x, experience_2_y)
        d2 = AvalancheDataset(d2)

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)
        dtest = TensorDataset(test_x, test_y)
        dtest = AvalancheDataset(dtest)

        def train_gen():
            # Lazy generator of the training stream
            for dataset in [d1, d2]:
                yield dataset

        def test_gen():
            # Lazy generator of the test stream
            for dataset in [dtest]:
                yield dataset

        bm = CLScenario([
            CLStream("train", train_gen()),
            CLStream("test", test_gen())
        ])

        valid_bm = benchmark_with_validation_stream(bm, 20, shuffle=False)

        self.assertEqual(2, len(valid_bm.train_stream))
        self.assertEqual(2, len(valid_bm.valid_stream))
        self.assertEqual(1, len(valid_bm.test_stream))

        dd = valid_bm.train_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_1_x[:80],
                mb[0],
            )
        )

        dd = valid_bm.train_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_2_x[:60],
                mb[0],
            )
        )

        dd = valid_bm.train_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_1_y[:80],
                mb[1],
            )
        )

        dd = valid_bm.train_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_2_y[:60],
                mb[1],
            )
        )

        dd = valid_bm.valid_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_x[80:], mb[0]))

        dd = valid_bm.valid_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_x[60:], mb[0]))

        dd = valid_bm.valid_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_y[80:], mb[1]))

        dd = valid_bm.valid_stream[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_y[60:], mb[1]))

        dd = valid_bm.test_stream[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(test_x, mb[0]))
        self.assertTrue(torch.equal(test_y, mb[1]))
