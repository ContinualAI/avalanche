import unittest

from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.scenarios.task_aware import task_incremental_benchmark
from tests.unit_tests_utils import dummy_classification_datasets


class TestsTaskAware(unittest.TestCase):
    def test_taskaware(self):
        """Common use case: add tas labels to class-incremental benchmark."""        
        num_classes = 10
        d1, d2 = dummy_classification_datasets(n_classes=num_classes)
        bm_ci = class_incremental_benchmark({'train': d1, 'test': d2}, num_experiences=num_classes)
        bm_ti = task_incremental_benchmark(bm_ci)

        assert len(list(bm_ti.train_stream)) == len(list(bm_ci.train_stream))
        assert len(list(bm_ti.test_stream)) == len(list(bm_ci.test_stream))

        ci_train = bm_ci.train_stream
        for eid, exp in enumerate(bm_ti.train_stream):
            assert exp.task_label == eid
            assert len(ci_train[eid].dataset) == len(exp.dataset)


if __name__ == '__main__':
    unittest.main()