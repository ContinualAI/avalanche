import unittest

import torch
from numpy.testing import assert_almost_equal
from torch.utils.data import TensorDataset, DataLoader

from avalanche.benchmarks import (
    benchmark_from_datasets,
    CLScenario,
    CLStream,
    split_validation_random,
    task_incremental_benchmark,
)
from avalanche.benchmarks.scenarios.validation_scenario import (
    benchmark_with_validation_stream,
)
from avalanche.benchmarks.scenarios.dataset_scenario import (
    DatasetExperience,
    split_validation_class_balanced,
)
from avalanche.benchmarks.utils import AvalancheDataset
from tests.unit_tests_utils import (
    dummy_tensor_dataset,
    get_fast_benchmark,
    DummyImageDataset,
)


def get_mbatch(data, batch_size=5):
    dl = DataLoader(
        data, shuffle=False, batch_size=batch_size, collate_fn=data.collate_fn
    )
    return next(iter(dl))


class DatasetScenarioTests(unittest.TestCase):
    def test_benchmark_from_datasets(self):
        d1 = AvalancheDataset(dummy_tensor_dataset())
        d2 = AvalancheDataset(dummy_tensor_dataset())
        d3 = AvalancheDataset(dummy_tensor_dataset())
        d4 = AvalancheDataset(dummy_tensor_dataset())

        bm = benchmark_from_datasets(train=[d1, d2], test=[d3, d4])

        # train stream
        train_s = bm.streams["train"]
        assert len(train_s) == 2
        for eid, (exp, d_orig) in enumerate(zip(train_s, [d1, d2])):
            assert exp.current_experience == eid
            for ii, (x, y) in enumerate(exp.dataset):
                torch.testing.assert_close(x, d_orig[ii][0])
                torch.testing.assert_close(y, d_orig[ii][1])

        # test stream
        train_s = bm.streams["test"]
        assert len(train_s) == 2
        for eid, (exp, d_orig) in enumerate(zip(train_s, [d3, d4])):
            assert exp.current_experience == eid
            for ii, (x, y) in enumerate(exp.dataset):
                torch.testing.assert_close(x, d_orig[ii][0])
                torch.testing.assert_close(y, d_orig[ii][1])

    def test_benchmark_from_dataset_heterogeneous_data(self):
        d1 = AvalancheDataset(dummy_tensor_dataset())
        d2 = AvalancheDataset(dummy_tensor_dataset())

        d1b = AvalancheDataset(DummyImageDataset(n_classes=10))
        d2b = AvalancheDataset(DummyImageDataset(n_classes=10))

        bm = benchmark_from_datasets(train=[d1, d1b], test=[d2, d2b])


class TaskIncrementalScenarioTests(unittest.TestCase):
    def test_task_incremental_bm_basic(self):
        d1 = AvalancheDataset(dummy_tensor_dataset())
        d2 = AvalancheDataset(dummy_tensor_dataset())
        bm = benchmark_from_datasets(train=[d1, d2])
        bm = task_incremental_benchmark(bm)
        # check task labels attributes and dataset lengths
        s = bm.train_stream
        assert s[0].task_label == 0
        assert len(s[0].dataset) == len(d1)
        assert s[1].task_label == 1
        assert len(s[1].dataset) == len(d2)


class DatasetSplitterTest(unittest.TestCase):
    def test_split_dataset_random(self):
        x = torch.rand(32, 10)
        y = torch.arange(
            32
        )  # we use ordered labels to reconstruct the order after shuffling
        dd = AvalancheDataset([TensorDataset(x, y)])

        d1, d2 = split_validation_random(validation_size=0.5, shuffle=True, dataset=dd)
        assert len(d1) + len(d2) == len(dd)

        # check data is shuffled
        iis = []
        for x, ii in d1:
            iis.append(ii)
        for x, ii in d2:
            iis.append(ii)
        assert not all(b >= a for a, b in zip(iis[:-1], iis[1:]))

        # check d1
        for x, ii in d1:
            torch.testing.assert_close(x, dd[ii.item()][0])
            torch.testing.assert_close(ii, dd[ii.item()][1])
        # check d2
        for x, ii in d2:
            torch.testing.assert_close(x, dd[ii.item()][0])
            torch.testing.assert_close(ii, dd[ii.item()][1])

    def test_split_dataset_class_balanced(self):
        benchmark = get_fast_benchmark(n_samples_per_class=1000)
        exp = benchmark.train_stream[0]
        num_classes = len(exp.classes_in_this_experience)

        train_d, valid_d = split_validation_class_balanced(0.5, exp.dataset)
        assert abs(len(train_d) - len(valid_d)) <= num_classes
        for cid in exp.classes_in_this_experience:
            train_cnt = (torch.as_tensor(train_d.targets) == cid).sum()
            valid_cnt = (torch.as_tensor(valid_d.targets) == cid).sum()
            # print(train_cnt, valid_cnt)
            assert abs(train_cnt - valid_cnt) <= 1

        ratio = 0.123
        len_data = len(exp.dataset)
        train_d, valid_d = split_validation_class_balanced(ratio, exp.dataset)
        assert_almost_equal(len(valid_d) / len_data, ratio, decimal=2)
        for cid in exp.classes_in_this_experience:
            data_cnt = (torch.as_tensor(exp.dataset.targets) == cid).sum()
            valid_cnt = (torch.as_tensor(valid_d.targets) == cid).sum()
            assert_almost_equal(valid_cnt / data_cnt, ratio, decimal=2)

    def test_fixed_size_experience_split_strategy(self):
        x = torch.rand(32, 10)
        y = torch.arange(
            32
        )  # we use ordered labels to reconstruct the order after shuffling
        dd = AvalancheDataset([TensorDataset(x, y)])

        d1, d2 = split_validation_random(validation_size=10, shuffle=True, dataset=dd)
        assert len(d1) + len(d2) == len(dd)

        # check data is shuffled
        iis = []
        for x, ii in d1:
            iis.append(ii)
        for x, ii in d2:
            iis.append(ii)
        assert not all(b >= a for a, b in zip(iis[:-1], iis[1:]))

        # check d1
        for x, ii in d1:
            torch.testing.assert_close(x, dd[ii.item()][0])
            torch.testing.assert_close(ii, dd[ii.item()][1])
        # check d2
        for x, ii in d2:
            torch.testing.assert_close(x, dd[ii.item()][0])
            torch.testing.assert_close(ii, dd[ii.item()][1])


class DatasetWithValidationStreamTests(unittest.TestCase):
    def test_benchmark_with_validation_stream_fixed_size(self):
        pattern_shape = (3, 32, 32)

        # Definition of training experiences
        # Experience 1
        experience_1_x = torch.zeros(100, *pattern_shape)
        experience_1_y = torch.zeros(100, dtype=torch.long)
        d1 = TensorDataset(experience_1_x, experience_1_y)
        d1 = AvalancheDataset([d1])
        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        d2 = TensorDataset(experience_2_x, experience_2_y)
        d2 = AvalancheDataset([d2])

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
        d1 = AvalancheDataset([d1])
        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        d2 = TensorDataset(experience_2_x, experience_2_y)
        d2 = AvalancheDataset([d2])

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)
        dtest = TensorDataset(test_x, test_y)
        dtest = AvalancheDataset([dtest])

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
        d1 = AvalancheDataset([d1])
        # Experience 2
        experience_2_x = torch.zeros(80, *pattern_shape)
        experience_2_y = torch.ones(80, dtype=torch.long)
        d2 = TensorDataset(experience_2_x, experience_2_y)
        d2 = AvalancheDataset([d2])

        # Test experience
        test_x = torch.zeros(50, *pattern_shape)
        test_y = torch.zeros(50, dtype=torch.long)
        dtest = TensorDataset(test_x, test_y)
        dtest = AvalancheDataset([dtest])

        def train_gen():
            # Lazy generator of the training stream
            for dataset in [d1, d2]:
                yield DatasetExperience(dataset=dataset)

        def test_gen():
            # Lazy generator of the test stream
            for dataset in [dtest]:
                yield DatasetExperience(dataset=dataset)

        bm = CLScenario([CLStream("train", train_gen()), CLStream("test", test_gen())])

        valid_bm = benchmark_with_validation_stream(bm, 20, shuffle=False)

        self.assertEqual(2, len(list(valid_bm.train_stream)))
        self.assertEqual(2, len(list(valid_bm.valid_stream)))
        self.assertEqual(1, len(list(valid_bm.test_stream)))

        dd = list(valid_bm.train_stream)[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_1_x[:80],
                mb[0],
            )
        )

        dd = list(valid_bm.train_stream)[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_2_x[:60],
                mb[0],
            )
        )

        dd = list(valid_bm.train_stream)[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_1_y[:80],
                mb[1],
            )
        )

        dd = list(valid_bm.train_stream)[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(
            torch.equal(
                experience_2_y[:60],
                mb[1],
            )
        )

        dd = list(valid_bm.valid_stream)[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_x[80:], mb[0]))

        dd = list(valid_bm.valid_stream)[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_x[60:], mb[0]))

        dd = list(valid_bm.valid_stream)[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_1_y[80:], mb[1]))

        dd = list(valid_bm.valid_stream)[1].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(experience_2_y[60:], mb[1]))

        dd = list(valid_bm.test_stream)[0].dataset
        mb = get_mbatch(dd, len(dd))
        self.assertTrue(torch.equal(test_x, mb[0]))
        self.assertTrue(torch.equal(test_y, mb[1]))

    def test_regressioni1597(args):
        # regression test for issue #1597
        bm = get_fast_benchmark()
        for exp in bm.train_stream:
            assert hasattr(exp, "classes_in_this_experience")
