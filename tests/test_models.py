import sys

import unittest

import pytorchcv.models.pyramidnet_cifar
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.logging import TextLogger
from avalanche.models import (
    MTSimpleMLP,
    SimpleMLP,
    IncrementalClassifier,
    MultiHeadClassifier,
    SimpleCNN,
    NCMClassifier,
    TrainEvalModel,
    PNN,
)
from avalanche.models.dynamic_optimizers import (
    add_new_params_to_optimizer,
    update_optimizer,
)
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.supervised import Naive
from avalanche.models.pytorchcv_wrapper import (
    vgg,
    resnet,
    densenet,
    pyramidnet,
    get_model,
)
from tests.unit_tests_utils import (
    common_setups,
    get_fast_benchmark,
    load_benchmark,
)


class PytorchcvWrapperTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_vgg(self):
        model = vgg(depth=19, batch_normalization=True, pretrained=False)
        # Batch norm is activated
        self.assertIsInstance(
            model.features.stage1.unit1.bn, torch.nn.BatchNorm2d
        )
        # Check correct depth is loaded
        self.assertEqual(len(model.features.stage5), 5)

    def test_resnet(self):
        model = resnet("cifar10", depth=20)

        # Test input/output sizes
        self.assertEqual(model.in_size, (32, 32))
        self.assertEqual(model.num_classes, 10)

        # Test input/output sizes
        model = resnet("imagenet", depth=12)
        self.assertEqual(model.in_size, (224, 224))
        self.assertEqual(model.num_classes, 1000)

    def test_pyramidnet(self):
        model = pyramidnet("cifar10", depth=110)
        self.assertIsInstance(
            model, pytorchcv.models.pyramidnet_cifar.CIFARPyramidNet
        )
        model = pyramidnet("imagenet", depth=101)
        self.assertIsInstance(model, pytorchcv.models.pyramidnet.PyramidNet)

    def test_densenet(self):
        model = densenet("svhn", depth=40)
        self.assertIsInstance(
            model, pytorchcv.models.densenet_cifar.CIFARDenseNet
        )

    def test_get_model(self):
        # Check general wrapper and whether downloading pretrained model works
        model = get_model("simplepose_resnet18_coco", pretrained=True)
        self.assertIsInstance(
            model, pytorchcv.models.simplepose_coco.SimplePose
        )


class DynamicOptimizersTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def _is_param_in_optimizer(self, param, optimizer):
        for group in optimizer.param_groups:
            for curr_p in group["params"]:
                if hash(curr_p) == hash(param):
                    return True
        return False

    def test_optimizer_update(self):
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=1e-3)
        strategy = Naive(model, optimizer, None)

        # check add_param_group
        p = torch.nn.Parameter(torch.zeros(10, 10))
        add_new_params_to_optimizer(optimizer, p)
        assert self._is_param_in_optimizer(p, strategy.optimizer)

        # check new_param is in optimizer
        # check old_param is NOT in optimizer
        p_new = torch.nn.Parameter(torch.zeros(10, 10))
        update_optimizer(optimizer, [p], [p_new])
        assert self._is_param_in_optimizer(p_new, strategy.optimizer)
        assert not self._is_param_in_optimizer(p, strategy.optimizer)


class DynamicModelsTests(unittest.TestCase):
    def setUp(self):
        common_setups()
        self.benchmark = get_fast_benchmark(
            use_task_labels=False, shuffle=False
        )

    def test_incremental_classifier(self):
        model = SimpleMLP(input_size=6, hidden_size=10)
        model.classifier = IncrementalClassifier(in_features=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = self.benchmark

        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=100,
            train_epochs=1,
            eval_mb_size=100,
            device="cpu",
        )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        print(
            "Current Classes: ",
            benchmark.train_stream[0].classes_in_this_experience,
        )
        print(
            "Current Classes: ",
            benchmark.train_stream[4].classes_in_this_experience,
        )

        # train on first task
        strategy.train(benchmark.train_stream[0])
        w_ptr = model.classifier.classifier.weight.data_ptr()
        b_ptr = model.classifier.classifier.bias.data_ptr()
        opt_params_ptrs = [
            w.data_ptr()
            for group in optimizer.param_groups
            for w in group["params"]
        ]
        # classifier params should be optimized
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # train again on the same task.
        strategy.train(benchmark.train_stream[0])
        # parameters should not change.
        assert w_ptr == model.classifier.classifier.weight.data_ptr()
        assert b_ptr == model.classifier.classifier.bias.data_ptr()
        # the same classifier params should still be optimized
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # update classifier with new classes.
        old_w_ptr, old_b_ptr = w_ptr, b_ptr
        strategy.train(benchmark.train_stream[4])
        opt_params_ptrs = [
            w.data_ptr()
            for group in optimizer.param_groups
            for w in group["params"]
        ]
        new_w_ptr = model.classifier.classifier.weight.data_ptr()
        new_b_ptr = model.classifier.classifier.bias.data_ptr()
        # weights should change.
        assert old_w_ptr != new_w_ptr
        assert old_b_ptr != new_b_ptr
        # Old params should not be optimized. New params should be optimized.
        assert old_w_ptr not in opt_params_ptrs
        assert old_b_ptr not in opt_params_ptrs
        assert new_w_ptr in opt_params_ptrs
        assert new_b_ptr in opt_params_ptrs

    def test_incremental_classifier_weight_update(self):
        model = IncrementalClassifier(in_features=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = self.benchmark

        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=100,
            train_epochs=1,
            eval_mb_size=100,
            device="cpu",
        )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]

        # train on first task
        w_old = model.classifier.weight.clone()
        b_old = model.classifier.bias.clone()

        # adaptation. Increase number of classes
        dataset = benchmark.train_stream[4].dataset
        model.adaptation(dataset)
        w_new = model.classifier.weight.clone()
        b_new = model.classifier.bias.clone()

        # old weights should be copied correctly.
        assert torch.equal(w_old, w_new[: w_old.shape[0]])
        assert torch.equal(b_old, b_new[: w_old.shape[0]])

        # shape should be correct.
        assert w_new.shape[0] == max(dataset.targets) + 1
        assert b_new.shape[0] == max(dataset.targets) + 1

    def test_multihead_head_creation(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = MTSimpleMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=False)

        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=100,
            train_epochs=1,
            eval_mb_size=100,
            device="cpu",
        )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        print(
            "Current Classes: ",
            benchmark.train_stream[4].classes_in_this_experience,
        )
        print(
            "Current Classes: ",
            benchmark.train_stream[0].classes_in_this_experience,
        )

        # head creation
        strategy.train(benchmark.train_stream[0])
        w_ptr = model.classifier.classifiers["0"].classifier.weight.data_ptr()
        b_ptr = model.classifier.classifiers["0"].classifier.bias.data_ptr()
        opt_params_ptrs = [
            w.data_ptr()
            for group in optimizer.param_groups
            for w in group["params"]
        ]
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # head update
        strategy.train(benchmark.train_stream[4])
        w_ptr_t0 = model.classifier.classifiers[
            "0"
        ].classifier.weight.data_ptr()
        b_ptr_t0 = model.classifier.classifiers["0"].classifier.bias.data_ptr()
        w_ptr_new = model.classifier.classifiers[
            "4"
        ].classifier.weight.data_ptr()
        b_ptr_new = model.classifier.classifiers["4"].classifier.bias.data_ptr()
        opt_params_ptrs = [
            w.data_ptr()
            for group in optimizer.param_groups
            for w in group["params"]
        ]

        assert w_ptr not in opt_params_ptrs  # head0 has been updated
        assert b_ptr not in opt_params_ptrs  # head0 has been updated
        assert w_ptr_t0 in opt_params_ptrs
        assert b_ptr_t0 in opt_params_ptrs
        assert w_ptr_new in opt_params_ptrs
        assert b_ptr_new in opt_params_ptrs

    def test_multihead_head_selection(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = MultiHeadClassifier(in_features=6)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=False)

        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=100,
            train_epochs=1,
            eval_mb_size=100,
            device="cpu",
        )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]

        # initialize head
        strategy.train(benchmark.train_stream[0])
        strategy.train(benchmark.train_stream[4])

        # create models with fixed head
        model_t0 = model.classifiers["0"]
        model_t4 = model.classifiers["4"]

        # check head task0
        for x, y, t in DataLoader(benchmark.train_stream[0].dataset):
            y_mh = model(x, t)
            y_t = model_t0(x)
            assert ((y_mh - y_t) ** 2).sum() < 1.0e-7
            break

        # check head task4
        for x, y, t in DataLoader(benchmark.train_stream[4].dataset):
            y_mh = model(x, t)
            y_t = model_t4(x)
            assert ((y_mh - y_t) ** 2).sum() < 1.0e-7
            break


class TrainEvalModelTests(unittest.TestCase):
    def test_classifier_selection(self):
        base_model = SimpleCNN()

        feature_extractor = base_model.features
        classifier1 = base_model.classifier
        classifier2 = NCMClassifier()

        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier1,
            eval_classifier=classifier2,
        )

        model.eval()
        model.adaptation()
        assert model.classifier is classifier2

        model.train()
        model.adaptation()
        assert model.classifier is classifier1

        model.eval_adaptation()
        assert model.classifier is classifier2

        model.train_adaptation()
        assert model.classifier is classifier1


class NCMClassifierTest(unittest.TestCase):
    def test_ncm_classification(self):
        class_means = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float,
        )

        mb_x = torch.tensor(
            [[4, 3, 2, 1], [3, 4, 2, 1], [3, 2, 4, 1], [3, 2, 1, 4]],
            dtype=torch.float,
        )

        mb_y = torch.tensor([0, 1, 2, 3], dtype=torch.float)

        classifier = NCMClassifier(class_means)

        pred = classifier(mb_x)
        assert torch.all(torch.max(pred, 1)[1] == mb_y)


class PNNTest(unittest.TestCase):
    def test_pnn_on_multiple_tasks(self):
        model = PNN(
            num_layers=2,
            in_features=6,
            hidden_features_per_column=10,
            adapter="mlp",
        )
        benchmark = load_benchmark(use_task_labels=True)
        d0 = benchmark.train_stream[0].dataset
        mb0 = iter(DataLoader(d0)).__next__()
        d1 = benchmark.train_stream[1].dataset
        mb1 = iter(DataLoader(d1)).__next__()

        avalanche_model_adaptation(model, d0)
        avalanche_model_adaptation(model, d1)
        model(mb0[0], task_labels=mb0[-1])
        model(mb1[0], task_labels=mb1[-1])
