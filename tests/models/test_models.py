import copy
import os
import sys
import tempfile
import unittest

import numpy as np
import pytorchcv.models.pyramidnet_cifar
import torch
import torch.nn.functional as F
from tests.benchmarks.utils.test_avalanche_classification_dataset import get_mbatch
from tests.unit_tests_utils import common_setups, get_fast_benchmark, load_benchmark
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.logging import TextLogger
from avalanche.models import (
    PNN,
    CosineIncrementalClassifier,
    FeCAMClassifier,
    IncrementalClassifier,
    MTSimpleMLP,
    MultiHeadClassifier,
    NCMClassifier,
    SimpleCNN,
    SimpleMLP,
    TrainEvalModel,
)
from avalanche.models.cosine_layer import CosineLinear, SplitCosineLinear
from avalanche.models.dynamic_optimizers import (
    add_new_params_to_optimizer,
    update_optimizer,
)
from avalanche.models.pytorchcv_wrapper import (
    densenet,
    get_model,
    pyramidnet,
    resnet,
    vgg,
)
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.supervised import Naive


class PytorchcvWrapperTests(unittest.TestCase):
    def setUp(self):
        common_setups()

    def test_vgg(self):
        model = vgg(depth=19, batch_normalization=True, pretrained=False)
        # Batch norm is activated
        self.assertIsInstance(model.features.stage1.unit1.bn, torch.nn.BatchNorm2d)
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
        self.assertIsInstance(model, pytorchcv.models.pyramidnet_cifar.CIFARPyramidNet)
        model = pyramidnet("imagenet", depth=101)
        self.assertIsInstance(model, pytorchcv.models.pyramidnet.PyramidNet)

    def test_densenet(self):
        model = densenet("svhn", depth=40)
        self.assertIsInstance(model, pytorchcv.models.densenet_cifar.CIFARDenseNet)

    def test_get_model(self):
        # Check general wrapper and whether downloading pretrained model works
        model = get_model("resnet10", pretrained=True)
        self.assertIsInstance(model, pytorchcv.models.resnet.ResNet)


class DynamicModelsTests(unittest.TestCase):
    def setUp(self):
        common_setups()
        self.benchmark = get_fast_benchmark(use_task_labels=False, shuffle=False)

    def test_incremental_classifier(self):
        model = SimpleMLP(input_size=6, hidden_size=10)
        model.classifier = IncrementalClassifier(in_features=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = self.benchmark

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
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
            w.data_ptr() for group in optimizer.param_groups for w in group["params"]
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
            w.data_ptr() for group in optimizer.param_groups for w in group["params"]
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
            model=model,
            optimizer=optimizer,
            criterion=criterion,
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
        e = benchmark.train_stream[4]

        class Experience:
            dataset = e.dataset
            classes_in_this_experience = e.classes_in_this_experience

        experience = Experience()

        model.adaptation(experience)
        w_new = model.classifier.weight.clone()
        b_new = model.classifier.bias.clone()

        # old weights should be copied correctly.
        assert torch.equal(w_old, w_new[: w_old.shape[0]])
        assert torch.equal(b_old, b_new[: w_old.shape[0]])

        # shape should be correct.
        assert w_new.shape[0] == max(experience.dataset.targets) + 1
        assert b_new.shape[0] == max(experience.dataset.targets) + 1

    def test_multihead_head_creation(self):
        # Check if the optimizer is updated correctly
        # when heads are created and updated.
        model = MTSimpleMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=False)

        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
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
            w.data_ptr() for group in optimizer.param_groups for w in group["params"]
        ]
        assert w_ptr in opt_params_ptrs
        assert b_ptr in opt_params_ptrs

        # head update
        strategy.train(benchmark.train_stream[4])
        w_ptr_t0 = model.classifier.classifiers["0"].classifier.weight.data_ptr()
        b_ptr_t0 = model.classifier.classifiers["0"].classifier.bias.data_ptr()
        w_ptr_new = model.classifier.classifiers["4"].classifier.weight.data_ptr()
        b_ptr_new = model.classifier.classifiers["4"].classifier.bias.data_ptr()
        opt_params_ptrs = [
            w.data_ptr() for group in optimizer.param_groups for w in group["params"]
        ]

        # assert w_ptr not in opt_params_ptrs  # head0 has NOT been updated
        # assert b_ptr not in opt_params_ptrs  # head0 has NOT been updated
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
            model=model,
            optimizer=optimizer,
            criterion=criterion,
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
        model.masking = False  # disable masking to check output equality
        for x, y, t in DataLoader(benchmark.train_stream[0].dataset):
            y_mh = model(x, t)
            y_t = model_t0(x)

            # We need to pad y_t to dim with zeros
            # because y_mh will have max dim of all heads
            y_t = F.pad(y_t, (0, y_mh.size(1) - y_t.size(1)))

            assert ((y_mh - y_t) ** 2).sum() < 1.0e-7
            break

        # check head task4
        for x, y, t in DataLoader(benchmark.train_stream[4].dataset):
            y_mh = model(x, t)
            y_t = model_t4(x)

            # We need to pad y_t to dim with zeros
            # because y_mh will have max dim of all heads
            y_t = F.pad(y_t, (0, y_mh.size(1) - y_t.size(1)))

            assert ((y_mh - y_t) ** 2).sum() < 1.0e-7
            break

    def test_incremental_classifier_masking(self):
        benchmark = get_fast_benchmark(use_task_labels=False, shuffle=True)
        model = IncrementalClassifier(in_features=6)
        autot = []
        for exp in benchmark.train_stream:
            curr_au = exp.classes_in_this_experience
            autot.extend(curr_au)

            model.adaptation(exp)
            assert torch.all(model.active_units[autot] == 1)

            # print(model.active_units)
            mb = get_mbatch(exp.dataset)[0]
            out = model(mb)
            assert torch.all(out[:, autot] != model.mask_value)
            out_masked = out[:, model.active_units == 0]
            assert torch.all(out_masked == model.mask_value)

    def test_incremental_classifier_update_masking_only_during_training(self):
        benchmark = get_fast_benchmark(use_task_labels=False, shuffle=True)
        model = IncrementalClassifier(in_features=6)
        autot = []
        for exp in benchmark.train_stream:
            curr_au = exp.classes_in_this_experience
            autot.extend(curr_au)

            model.adaptation(exp)

            # eval should NOT modify the mask
            model.eval()
            for ee in benchmark.train_stream:
                model.adaptation(ee)
            model.train()

            assert torch.all(model.active_units[autot] == 1)
            au_copy = torch.clone(model.active_units)
            au_copy[autot] = False
            assert torch.all(~au_copy)

            # print(model.active_units)
            mb = get_mbatch(exp.dataset)[0]
            out = model(mb)
            assert torch.all(out[:, autot] != model.mask_value)
            out_masked = out[:, model.active_units == 0]
            assert torch.all(out_masked == model.mask_value)

    def test_avalanche_adaptation(self):
        # This tests adaptation when done through normal pytorch module
        clf = MultiHeadClassifier(in_features=6)
        model1 = torch.nn.Sequential(clf)
        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=True)
        # Also test sizes here
        sizes = {}
        for t, exp in enumerate(benchmark.train_stream):
            sizes[t] = np.max(exp.classes_in_this_experience) + 1
            avalanche_model_adaptation(model1, exp)
        # Second adaptation should not change anything
        for t, exp in enumerate(benchmark.train_stream):
            avalanche_model_adaptation(model1, exp)
        for t, s in sizes.items():
            self.assertEqual(s, clf.classifiers[str(t)].classifier.out_features)

    def test_recursive_adaptation(self):
        # This tests adaptation when done directly from DynamicModule
        model1 = MultiHeadClassifier(in_features=6)
        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=True)
        # Also test sizes here
        sizes = {}
        for t, exp in enumerate(benchmark.train_stream):
            sizes[t] = np.max(exp.classes_in_this_experience) + 1
            model1.pre_adapt(None, exp)
        # Second adaptation should not change anything
        for t, exp in enumerate(benchmark.train_stream):
            model1.pre_adapt(None, exp)
        for t, s in sizes.items():
            self.assertEqual(s, model1.classifiers[str(t)].classifier.out_features)

    def test_recursive_loop(self):
        model1 = MultiHeadClassifier(in_features=6)
        model2 = MultiHeadClassifier(in_features=6)

        # Create a mess
        model1.layer2 = model2
        model2.layer2 = model1

        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=True)
        model1.pre_adapt(None, benchmark.train_stream[0])

    def test_multi_head_classifier_masking(self):
        benchmark = get_fast_benchmark(use_task_labels=True, shuffle=True)

        # print("task order: ", [e.task_label for e in benchmark.train_stream])
        # print("class order: ", [e.classes_in_this_experience for e in
        # benchmark.train_stream])

        model = MultiHeadClassifier(in_features=6)
        for tid, exp in enumerate(benchmark.train_stream):
            # exclusive task label for each experience
            curr_au = exp.classes_in_this_experience

            model.adaptation(exp)
            curr_mask = model._buffers[f"active_units_T{tid}"]
            nunits = curr_mask.shape[0]
            assert torch.all(curr_mask[curr_au] == 1)

            # print(model._buffers)
            mb, _, tmb = get_mbatch(exp.dataset, batch_size=7)
            out = model(mb, tmb)
            assert torch.all(out[:, curr_au] != model.mask_value)
            assert torch.all(out[:, :nunits][:, curr_mask == 0] == model.mask_value)
        # check masking after adaptation on the entire stream
        for tid, exp in enumerate(benchmark.train_stream):
            curr_au = exp.classes_in_this_experience
            curr_mask = model._buffers[f"active_units_T{tid}"]
            nunits = curr_mask.shape[0]
            assert torch.all(curr_mask[curr_au] == 1)

            # print(model._buffers)
            mb, _, tmb = get_mbatch(exp.dataset)
            out = model(mb, tmb)
            assert torch.all(out[:, curr_au] != model.mask_value)
            assert torch.all(out[:, :nunits][:, curr_mask == 0] == model.mask_value)


class TrainEvalModelTests(unittest.TestCase):
    def test_classifier_selection(self):
        base_model = SimpleCNN()

        feature_extractor = torch.nn.Sequential(base_model.features, torch.nn.Flatten())
        classifier1 = base_model.classifier
        classifier2 = torch.nn.Linear(64, 7)

        x = torch.randn(2, 3, 32, 32)
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier1,
            eval_classifier=classifier2,
        )

        model.train()
        out = model(x)
        assert out.shape[-1] == 10

        model.eval()
        out = model(x)
        assert out.shape[-1] == 7


class NCMClassifierTest(unittest.TestCase):
    def test_ncm_classification(self):
        class_means = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float,
        )
        class_means_dict = {i: el for i, el in enumerate(class_means)}

        mb_x = torch.tensor(
            [[4, 3, 2, 1], [3, 2, 4, 1]],
            dtype=torch.float,
        )

        mb_y = torch.tensor([0, 2], dtype=torch.float)

        classifier = NCMClassifier(normalize=False)
        classifier.update_class_means_dict(class_means_dict)

        pred = classifier(mb_x)
        assert torch.all(torch.max(pred, 1)[1] == mb_y)

    def test_ncm_class_expansion(self):
        class_means = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float,
        )
        class_means_dict = {i: el for i, el in enumerate(class_means)}
        classifier = NCMClassifier()
        classifier.update_class_means_dict(class_means_dict)
        assert classifier.class_means.shape == (3, 4)
        new_mean = torch.randn(
            4,
        )
        classifier.update_class_means_dict({5: new_mean.clone()})
        assert classifier.class_means.shape == (6, 4)
        assert torch.all(
            classifier.class_means[3]
            == torch.zeros(
                4,
            )
        )
        assert torch.all(
            classifier.class_means[4]
            == torch.zeros(
                4,
            )
        )
        assert torch.all(classifier.class_means[5] == new_mean)

    def test_ncm_replace_means(self):
        class_means = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float,
        )
        class_means_dict = {i: el for i, el in enumerate(class_means)}
        classifier = NCMClassifier()
        classifier.update_class_means_dict(class_means_dict)
        class_means = torch.tensor(
            [[2, 0, 0, 0], [2, 1, 0, 0], [2, 0, 1, 0]],
            dtype=torch.float,
        )
        new_dict = {i: el for i, el in enumerate(class_means)}
        classifier.replace_class_means_dict(new_dict)
        assert (classifier.class_means[:, 0] == 2).all()

    def test_ncm_forward_without_class_means(self):
        classifier = NCMClassifier()
        classifier.init_missing_classes(list(range(10)), 7, "cpu")
        logits = classifier(torch.randn(2, 7))
        assert logits.shape == (2, 10)

    def test_ncm_eval_adapt(self):
        benchmark = get_fast_benchmark(use_task_labels=False, shuffle=True)
        model = SimpleMLP(input_size=6)
        train_classifier = model.classifier
        model.classifier = torch.nn.Identity()
        eval_classifier = NCMClassifier()
        model = TrainEvalModel(
            model, train_classifier=train_classifier, eval_classifier=eval_classifier
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = Naive(model, optimizer)

        # Do one first adaptation to know about output shape
        class_means_dict = {0: model.feature_extractor(torch.rand(1, 6)).mean(dim=0)}
        model.eval_classifier.replace_class_means_dict(class_means_dict)

        for exp in benchmark.test_stream:
            strategy.eval(exp)

    def test_ncm_save_load(self):
        classifier = NCMClassifier()
        classifier.update_class_means_dict(
            {
                1: torch.randn(
                    5,
                ),
                2: torch.randn(
                    5,
                ),
            }
        )

        with tempfile.TemporaryFile() as tmpfile:
            torch.save(classifier.state_dict(), tmpfile)
            del classifier
            classifier = NCMClassifier()
            tmpfile.seek(0)
            check = torch.load(tmpfile)
        classifier.load_state_dict(check)
        assert classifier.class_means.shape == (3, 5)
        assert (classifier.class_means[0] == 0).all()
        assert len(classifier.class_means_dict) == 2


class FeCAMClassifierTest(unittest.TestCase):
    def test_fecam_classification(self):
        class_means = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float,
        )
        class_means_dict = {i: el for i, el in enumerate(class_means)}
        class_cov_dict = {
            i: torch.eye(el.size(0)) for i, el in class_means_dict.items()
        }

        mb_x = torch.tensor(
            [[4, 3, 2, 1], [3, 2, 4, 1]],
            dtype=torch.float,
        )

        mb_y = torch.tensor([0, 2], dtype=torch.float)

        classifier = FeCAMClassifier()

        classifier.update_class_means_dict(class_means_dict)
        classifier.update_class_cov_dict(class_cov_dict)

        pred = classifier(mb_x)
        assert torch.all(torch.max(pred, 1)[1] == mb_y)

    def test_fecam_forward_without_class_means(self):
        classifier = FeCAMClassifier()
        classifier.init_missing_classes(list(range(10)), 7, "cpu")
        logits = classifier(torch.randn(2, 7))
        assert logits.shape == (2, 10)

    def test_fecam_save_load(self):
        classifier = FeCAMClassifier()

        classifier.update_class_means_dict(
            {
                1: torch.randn(
                    5,
                ),
                2: torch.randn(
                    5,
                ),
            }
        )

        classifier.update_class_cov_dict(
            {
                1: torch.rand(5, 5),
                2: torch.rand(5, 5),
            }
        )

        with tempfile.TemporaryFile() as tmpfile:
            torch.save(classifier.state_dict(), tmpfile)
            del classifier
            classifier = FeCAMClassifier()
            tmpfile.seek(0)
            check = torch.load(tmpfile)

        classifier.load_state_dict(check)

        assert len(classifier.class_means_dict) == 2

    def test_fecam_eval_adapt(self):
        benchmark = get_fast_benchmark(use_task_labels=False, shuffle=True)
        model = SimpleMLP(input_size=6)
        train_classifier = model.classifier
        model.classifier = torch.nn.Identity()
        eval_classifier = FeCAMClassifier()
        model = TrainEvalModel(
            model, train_classifier=train_classifier, eval_classifier=eval_classifier
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        strategy = Naive(model, optimizer)

        # Do one first adaptation to know about output shape
        class_means_dict = {0: model.feature_extractor(torch.rand(1, 6)).mean(dim=0)}
        class_vars_dict = {0: torch.cov(model.feature_extractor(torch.rand(1000, 6)).T)}
        model.eval_classifier.replace_class_means_dict(class_means_dict)
        model.eval_classifier.replace_class_cov_dict(class_vars_dict)

        for exp in benchmark.test_stream[:2]:
            strategy.eval(exp)


class CosineLayerTest(unittest.TestCase):
    def test_single_cosine(self):
        layer = CosineLinear(32, 10)
        test_input = torch.rand(5, 32)
        out = layer(test_input)
        out.sum().backward()

    def test_split_cosine(self):
        in_feat_1, in_feat_2 = 10, 10
        layer = SplitCosineLinear(32, in_feat_1, in_feat_2)
        test_input = torch.rand(5, 32)
        out = layer(test_input)
        self.assertEqual(out.size(1), in_feat_1 + in_feat_2)
        out.sum().backward()

    def test_cosine_incremental_adaptation(self):
        benchmark = load_benchmark(use_task_labels=False)
        num_classes_0 = np.max(benchmark.train_stream[0].classes_in_this_experience) + 1
        num_classes_1 = np.max(benchmark.train_stream[1].classes_in_this_experience) + 1

        test_input = torch.rand(5, 32)

        # Without initial classes
        layer = CosineIncrementalClassifier(32, num_classes=0)
        avalanche_model_adaptation(layer, benchmark.train_stream[0])
        out = layer(test_input)
        self.assertEqual(out.size(1), num_classes_0)
        avalanche_model_adaptation(layer, benchmark.train_stream[1])
        out = layer(test_input)
        self.assertEqual(out.size(1), max(num_classes_0, num_classes_1))

        # With initial classes
        initial_classes = 5
        layer = CosineIncrementalClassifier(32, num_classes=initial_classes)
        avalanche_model_adaptation(layer, benchmark.train_stream[0])
        out = layer(test_input)
        self.assertEqual(out.size(1), max(num_classes_0, initial_classes))

        # Test backward
        out.sum().backward()


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

        avalanche_model_adaptation(model, benchmark.train_stream[0])
        avalanche_model_adaptation(model, benchmark.train_stream[1])
        model(mb0[0], task_labels=mb0[-1])
        model(mb1[0], task_labels=mb1[-1])


if __name__ == "__main__":
    unittest.main()
