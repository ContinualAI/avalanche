import itertools
import sys

import torch
from torch import nn
import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks import (
    nc_benchmark,
    GenericCLScenario,
    benchmark_with_validation_stream,
)
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import Mean
from avalanche.logging import TextLogger
from avalanche.models import BaseModel, SimpleMLP
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
    EarlyStoppingPlugin,
)
from avalanche.training.plugins.clock import Clock
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.supervised import Naive


class MockPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.activated = [False for _ in range(22)]

    def before_training_exp(self, strategy, **kwargs):
        self.activated[0] = True

    def after_train_dataset_adaptation(self, strategy, **kwargs):
        self.activated[1] = True

    def before_training_epoch(self, strategy, **kwargs):
        self.activated[2] = True

    def before_training_iteration(self, strategy, **kwargs):
        self.activated[3] = True

    def before_forward(self, strategy, **kwargs):
        self.activated[4] = True

    def after_forward(self, strategy, **kwargs):
        self.activated[5] = True

    def before_backward(self, strategy, **kwargs):
        self.activated[6] = True

    def after_backward(self, strategy, **kwargs):
        self.activated[7] = True

    def after_training_iteration(self, strategy, **kwargs):
        self.activated[8] = True

    def before_update(self, strategy, **kwargs):
        self.activated[9] = True

    def after_update(self, strategy, **kwargs):
        self.activated[10] = True

    def after_training_epoch(self, strategy, **kwargs):
        self.activated[11] = True

    def after_training_exp(self, strategy, **kwargs):
        self.activated[12] = True

    def before_eval(self, strategy, **kwargs):
        self.activated[13] = True

    def after_eval_dataset_adaptation(self, strategy, **kwargs):
        self.activated[14] = True

    def before_eval_exp(self, strategy, **kwargs):
        self.activated[15] = True

    def after_eval_exp(self, strategy, **kwargs):
        self.activated[16] = True

    def after_eval(self, strategy, **kwargs):
        self.activated[17] = True

    def before_eval_iteration(self, strategy, **kwargs):
        self.activated[18] = True

    def before_eval_forward(self, strategy, **kwargs):
        self.activated[19] = True

    def after_eval_forward(self, strategy, **kwargs):
        self.activated[20] = True

    def after_eval_iteration(self, strategy, **kwargs):
        self.activated[21] = True


class PluginTests(unittest.TestCase):
    def test_callback_reachability(self):
        # Check that all the callbacks are called during
        # training and test loops.
        model = _PlainMLP(input_size=6, hidden_size=10)
        optimizer = SGD(model.parameters(), lr=1e-3)
        criterion = CrossEntropyLoss()
        benchmark = PluginTests.create_benchmark()

        plug = MockPlugin()
        strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=100,
            train_epochs=1,
            eval_mb_size=100,
            device="cpu",
            plugins=[plug],
        )
        strategy.evaluator.loggers = [TextLogger(sys.stdout)]
        strategy.train(benchmark.train_stream[0], num_workers=0)
        strategy.eval([benchmark.test_stream[0]], num_workers=0)
        assert all(plug.activated)

    @staticmethod
    def create_benchmark(task_labels=False, seed=None, n_samples_per_class=20):
        dataset = make_classification(
            n_samples=10 * n_samples_per_class,
            n_classes=10,
            n_features=6,
            n_informative=6,
            n_redundant=0,
            random_state=seed,
        )

        X = torch.from_numpy(dataset[0]).float()
        y = torch.from_numpy(dataset[1]).long()

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, train_size=0.6, shuffle=True, stratify=y, random_state=seed
        )

        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        return nc_benchmark(
            train_dataset,
            test_dataset,
            5,
            task_labels=task_labels,
            fixed_class_order=list(range(10)),
        )

    def test_scheduler_plugin(self):
        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=True,
            expected=[[4.0, 2.0, 1.0], [4.0, 2.0, 1.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=False,
            reset_scheduler=True,
            expected=[[4.0, 2.0, 1.0], [1.0, 0.5, 0.25]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=False,
            expected=[[4.0, 2.0, 1.0], [4.0, 4.0, 4.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=False,
            reset_scheduler=False,
            expected=[[4.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=False,
            reset_scheduler=True,
            first_exp_only=True,
            expected=[[4.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=False,
            reset_scheduler=True,
            first_epoch_only=True,
            expected=[[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[1, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=False,
            reset_scheduler=True,
            first_epoch_only=True,
            expected=[[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[1, 2],
            base_lr=4.0,
            epochs=3,
            reset_lr=False,
            reset_scheduler=False,
            first_exp_only=True,
            expected=[[2.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[1, 2],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=False,
            first_exp_only=True,
            expected=[[2.0, 1.0, 1.0], [4.0, 4.0, 4.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[1, 2],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=False,
            first_exp_only=True,
            first_epoch_only=True,
            expected=[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[2, 3],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=False,
            first_exp_only=True,
            first_epoch_only=True,
            expected=[[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[1, 4],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=False,
            expected=[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        )

        PluginTests._test_scheduler_multi_step_lr_plugin(
            gamma=1 / 2.0,
            milestones=[3, 4],
            base_lr=4.0,
            epochs=3,
            reset_lr=True,
            reset_scheduler=True,
            expected=[[4.0, 4.0, 2.0], [4.0, 4.0, 2.0]],
        )

    @staticmethod
    def _test_scheduler_multi_step_lr_plugin(
            gamma, milestones, base_lr, epochs, reset_lr, reset_scheduler,
            expected, first_epoch_only=False, first_exp_only=False
    ):

        benchmark = PluginTests.create_benchmark(n_samples_per_class=20)
        model = _PlainMLP(input_size=6, hidden_size=10)
        optim = SGD(model.parameters(), lr=base_lr)
        scheduler = MultiStepLR(optim, milestones=milestones, gamma=gamma)

        PluginTests._test_scheduler_plugin(
            benchmark,
            model,
            optim,
            scheduler,
            epochs,
            reset_lr,
            reset_scheduler,
            expected,
            expected_granularity='epoch',
            max_exps=2,
            first_epoch_only=first_epoch_only,
            first_exp_only=first_exp_only
        )

    def assert_model_equals(self, model1, model2):
        dict1 = model1.state_dict()
        dict2 = model2.state_dict()

        # compare keys
        self.assertSetEqual(set(dict1.keys()), set(dict2.keys()))

        # compare params
        for (k, v) in dict1.items():
            self.assertTrue(torch.equal(v, dict2[k]))

    def assert_benchmark_equals(
        self, bench1: GenericCLScenario, bench2: GenericCLScenario
    ):
        self.assertSetEqual(
            set(bench1.streams.keys()), set(bench2.streams.keys())
        )

        for stream_name in list(bench1.streams.keys()):
            for exp1, exp2 in zip(
                bench1.streams[stream_name], bench2.streams[stream_name]
            ):
                dataset1 = exp1.dataset
                dataset2 = exp2.dataset
                for t_idx in range(3):
                    dataset1_content = dataset1[:][t_idx]
                    dataset2_content = dataset2[:][t_idx]
                    self.assertTrue(
                        torch.equal(dataset1_content, dataset2_content)
                    )

    def _verify_rop_tests_reproducibility(
        self, init_strategy, n_epochs, criterion
    ):
        # This doesn't actually test the support for the specific scheduler
        # (ReduceLROnPlateau), but it's only used to check if:
        # - the same model+benchmark pair can be instantiated in a
        #   deterministic way.
        # - the same results could be obtained in a standard training loop in a
        #   deterministic way.
        models_rnd = []
        benchmarks_rnd = []
        for _ in range(2):
            benchmark, model = init_strategy()
            models_rnd.append(model)
            benchmarks_rnd.append(benchmark)

        self.assert_model_equals(*models_rnd)
        self.assert_benchmark_equals(*benchmarks_rnd)

        expected_lrs_rnd = []
        for _ in range(2):
            benchmark, model = init_strategy()

            expected_lrs = []
            model.train()
            for exp in benchmark.train_stream:
                optimizer = SGD(model.parameters(), lr=0.001)
                scheduler = ReduceLROnPlateau(optimizer)
                expected_lrs.append([])
                train_loss = Mean()
                for epoch in range(n_epochs):
                    train_loss.reset()
                    for x, y, t in TaskBalancedDataLoader(
                        exp.dataset,
                        oversample_small_groups=True,
                        num_workers=0,
                        batch_size=32,
                        shuffle=False,
                        pin_memory=False,
                    ):
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        train_loss.update(loss, weight=len(x))
                        loss.backward()
                        optimizer.step()

                        for group in optimizer.param_groups:
                            expected_lrs[-1].append(group["lr"])
                            break
                    scheduler.step(train_loss.result())

            expected_lrs_rnd.append(expected_lrs)
        self.assertEqual(expected_lrs_rnd[0], expected_lrs_rnd[1])

    def test_scheduler_reduce_on_plateau_plugin(self):
        # Regression test for issue #858
        n_epochs = 20
        criterion = CrossEntropyLoss()

        def _prepare_rng_critical_parts(seed=1234):
            torch.random.manual_seed(seed)
            return (
                PluginTests.create_benchmark(
                    seed=seed, n_samples_per_class=100),
                _PlainMLP(input_size=6, hidden_size=10),
            )

        self._verify_rop_tests_reproducibility(
            _prepare_rng_critical_parts, n_epochs, criterion
        )

        # Everything is in order, now we can test the plugin support for the
        # ReduceLROnPlateau scheduler!

        for reset_lr, reset_scheduler, granularity, first_epoch_only, \
            first_exp_only in \
                itertools.product((True, False), (True, False),
                                  ('iteration', 'epoch'), (True, False),
                                  (True, False)):
            with self.subTest(
                    reset_lr=reset_lr, reset_scheduler=reset_scheduler,
                    granularity=granularity, first_epoch_only=first_epoch_only,
                    first_exp_only=first_exp_only
            ):
                # First, obtain the reference (expected) lr timeline by running
                # a plain PyTorch training loop with ReduceLROnPlateau.
                benchmark, model = _prepare_rng_critical_parts()
                model.train()
                expected_lrs = []

                optimizer = SGD(model.parameters(), lr=0.001)
                scheduler = ReduceLROnPlateau(optimizer)
                for exp_idx, exp in enumerate(benchmark.train_stream):
                    if reset_lr:
                        for group in optimizer.param_groups:
                            group["lr"] = 0.001

                    if reset_scheduler:
                        scheduler = ReduceLROnPlateau(optimizer)

                    expected_lrs.append([])
                    train_loss = Mean()
                    for epoch in range(n_epochs):
                        for x, y, t in TaskBalancedDataLoader(
                            exp.dataset,
                            oversample_small_groups=True,
                            num_workers=0,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=False,
                        ):
                            optimizer.zero_grad()
                            outputs = model(x)
                            loss = criterion(outputs, y)
                            train_loss.update(loss, weight=len(x))
                            loss.backward()
                            optimizer.step()
                            if granularity == 'iteration':
                                if epoch == 0 or not first_epoch_only:
                                    if exp_idx == 0 or not first_exp_only:
                                        scheduler.step(train_loss.result())
                                train_loss.reset()

                            for group in optimizer.param_groups:
                                expected_lrs[-1].append(group["lr"])
                                break

                        if granularity == 'epoch':
                            if epoch == 0 or not first_epoch_only:
                                if exp_idx == 0 or not first_exp_only:
                                    scheduler.step(train_loss.result())
                            train_loss.reset()

                # Now we have the correct timeline stored in expected_lrs.
                # Let's test the plugin!
                benchmark, model = _prepare_rng_critical_parts()
                optimizer = SGD(model.parameters(), lr=0.001)
                scheduler = ReduceLROnPlateau(optimizer)

                PluginTests._test_scheduler_plugin(
                    benchmark,
                    model,
                    optimizer,
                    scheduler,
                    n_epochs,
                    reset_lr,
                    reset_scheduler,
                    expected_lrs,
                    criterion=criterion,
                    metric="train_loss",
                    granularity=granularity,
                    first_exp_only=first_exp_only,
                    first_epoch_only=first_epoch_only
                )

        # Other tests
        benchmark, model = _prepare_rng_critical_parts()
        optimizer = SGD(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer)
        scheduler2 = MultiStepLR(optimizer, [1, 2, 3])

        # The metric must be set
        with self.assertRaises(Exception):
            LRSchedulerPlugin(scheduler, metric=None)

        # Doesn't make sense to set the metric when using a non-metric
        # based scheduler (should warn)
        with self.assertWarns(Warning):
            LRSchedulerPlugin(scheduler2, metric="train_loss")

        # Must raise an error on unsupported metric
        with self.assertRaises(Exception):
            LRSchedulerPlugin(scheduler, metric="cuteness")

    def test_scheduler_reduce_on_plateau_plugin_with_val_stream(self):
        # Regression test for issue #858 (part 2)
        n_epochs = 100
        criterion = CrossEntropyLoss()

        def _prepare_rng_critical_parts(seed=1234):
            torch.random.manual_seed(seed)
            initial_benchmark = PluginTests.create_benchmark(
                seed=seed, n_samples_per_class=100)
            val_benchmark = benchmark_with_validation_stream(
                initial_benchmark, 0.3, shuffle=True
            )
            return (val_benchmark, _PlainMLP(input_size=6, hidden_size=10))

        self._verify_rop_tests_reproducibility(
            _prepare_rng_critical_parts, n_epochs, criterion
        )

        # Everything is in order, now we can test the plugin support for the
        # ReduceLROnPlateau scheduler!
        for reset_lr, reset_scheduler, granularity, first_epoch_only, \
            first_exp_only in \
                itertools.product((True, False), (True, False),
                                  ('iteration', 'epoch'), (True, False),
                                  (True, False)):
            with self.subTest(
                    reset_lr=reset_lr, reset_scheduler=reset_scheduler,
                    granularity=granularity, first_epoch_only=first_epoch_only,
                    first_exp_only=first_exp_only
            ):
                # print('Start subtest', reset_lr, reset_scheduler, granularity,
                #       first_epoch_only, first_exp_only)

                # First, obtain the reference (expected) lr timeline by running
                # a plain PyTorch training loop with ReduceLROnPlateau.
                benchmark, model = _prepare_rng_critical_parts()

                expected_lrs = []

                optimizer = SGD(model.parameters(), lr=0.001)
                scheduler = ReduceLROnPlateau(optimizer)
                for exp_idx, exp in enumerate(benchmark.train_stream):
                    expected_lrs.append([])
                    model.train()
                    if reset_lr:
                        for group in optimizer.param_groups:
                            group["lr"] = 0.001

                    if reset_scheduler:
                        scheduler = ReduceLROnPlateau(optimizer)

                    for epoch in range(n_epochs):

                        val_exp = benchmark.valid_stream[exp_idx]

                        for x, y, t in TaskBalancedDataLoader(
                            exp.dataset,
                            oversample_small_groups=True,
                            num_workers=0,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=False,
                        ):
                            optimizer.zero_grad()
                            outputs = model(x)
                            loss = criterion(outputs, y)
                            loss.backward()
                            optimizer.step()

                            for group in optimizer.param_groups:
                                expected_lrs[-1].append(group["lr"])
                                break

                            if granularity == 'iteration':
                                val_loss = Mean()
                                model.eval()
                                with torch.no_grad():
                                    for x, y, t in DataLoader(
                                        val_exp.dataset,
                                        num_workers=0,
                                        batch_size=100,
                                        pin_memory=False,
                                    ):
                                        outputs = model(x)
                                        loss = criterion(outputs, y)
                                        val_loss.update(loss, weight=len(x))
                                if epoch == 0 or not first_epoch_only:
                                    if exp_idx == 0 or not first_exp_only:
                                        scheduler.step(val_loss.result())

                        if granularity == 'epoch':
                            val_loss = Mean()
                            model.eval()
                            with torch.no_grad():
                                for x, y, t in DataLoader(
                                        val_exp.dataset,
                                        num_workers=0,
                                        batch_size=100,
                                        pin_memory=False,
                                ):
                                    outputs = model(x)
                                    loss = criterion(outputs, y)
                                    val_loss.update(loss, weight=len(x))
                            if epoch == 0 or not first_epoch_only:
                                if exp_idx == 0 or not first_exp_only:
                                    scheduler.step(val_loss.result())

                # Now we have the correct timeline stored in expected_lrs
                # Let's test the plugin!
                benchmark, model = _prepare_rng_critical_parts()
                optimizer = SGD(model.parameters(), lr=0.001)
                scheduler = ReduceLROnPlateau(optimizer)

                PluginTests._test_scheduler_plugin(
                    benchmark,
                    model,
                    optimizer,
                    scheduler,
                    n_epochs,
                    reset_lr,
                    reset_scheduler,
                    expected_lrs,
                    criterion=criterion,
                    metric="val_loss",
                    eval_on_valid_stream=True,
                    granularity=granularity,
                    peval_mode=granularity,
                    first_exp_only=first_exp_only,
                    first_epoch_only=first_epoch_only
                )

    @staticmethod
    def _test_scheduler_plugin(
            benchmark,
            model,
            optim,
            scheduler,
            epochs,
            reset_lr,
            reset_scheduler,
            expected,
            criterion=None,
            metric=None,
            eval_on_valid_stream=False,
            granularity='epoch',
            expected_granularity='iteration',
            peval_mode='epoch',
            first_epoch_only=False,
            first_exp_only=False,
            max_exps=None
    ):
        lr_scheduler_plugin = LRSchedulerPlugin(
            scheduler,
            reset_lr=reset_lr,
            reset_scheduler=reset_scheduler,
            metric=metric,
            step_granularity=granularity,
            first_epoch_only=first_epoch_only,
            first_exp_only=first_exp_only
        )

        verifier_plugin = SchedulerPluginTestPlugin(
            expected, expected_granularity=expected_granularity)

        if criterion is None:
            criterion = CrossEntropyLoss()
        if eval_on_valid_stream:
            cl_strategy = Naive(
                model,
                optim,
                criterion,
                train_mb_size=32,
                train_epochs=epochs,
                eval_mb_size=100,
                plugins=[lr_scheduler_plugin, verifier_plugin],
                eval_every=1,
                peval_mode=peval_mode,
                evaluator=None,
            )

            for exp_id in range(len(benchmark.train_stream)):
                if max_exps is not None and exp_id >= max_exps:
                    break
                cl_strategy.train(
                    benchmark.train_stream[exp_id],
                    shuffle=False,
                    eval_streams=[benchmark.valid_stream[exp_id]],
                )
            # cl_strategy.train(
            #     benchmark.train_stream[1],
            #     shuffle=False,
            #     eval_streams=[benchmark.valid_stream[1]],
            # )
        else:
            cl_strategy = Naive(
                model,
                optim,
                criterion,
                train_mb_size=32,
                train_epochs=epochs,
                eval_mb_size=100,
                plugins=[lr_scheduler_plugin, verifier_plugin],
                peval_mode=peval_mode,
                evaluator=None,
            )

            for exp_id in range(len(benchmark.train_stream)):
                if max_exps is not None and exp_id >= max_exps:
                    break
                cl_strategy.train(benchmark.train_stream[exp_id], shuffle=False)
            # cl_strategy.train(benchmark.train_stream[1], shuffle=False)


class SchedulerPluginTestPlugin(SupervisedPlugin):
    def __init__(self, expected_lrs, expected_granularity='iteration'):
        super().__init__()
        self.expected_lrs = expected_lrs
        self.expected_granularity = expected_granularity
        self.so_far = []

    def after_training_iteration(self, strategy, **kwargs):
        if self.expected_granularity != 'iteration':
            return

        exp_id = strategy.clock.train_exp_counter
        curr_epoch = strategy.clock.train_exp_epochs
        curr_iter = strategy.clock.train_exp_iterations
        expected_lr = self.expected_lrs[exp_id][curr_iter]
        for group in strategy.optimizer.param_groups:
            self.so_far[-1].append(group["lr"])
            if group["lr"] != expected_lr:
                print('Expected vs LRs so far')
                print(self.expected_lrs)
                print(self.so_far)

            assert (
                group["lr"] == expected_lr
            ), f"[it] LR mismatch: {group['lr']} vs {expected_lr} at " \
               f"{exp_id}-{curr_epoch}-{curr_iter}"

    def after_training_epoch(self, strategy, **kwargs):
        if self.expected_granularity != 'epoch':
            return

        exp_id = strategy.clock.train_exp_counter
        curr_epoch = strategy.clock.train_exp_epochs
        curr_iter = strategy.clock.train_exp_iterations
        expected_lr = self.expected_lrs[exp_id][curr_epoch]
        for group in strategy.optimizer.param_groups:
            self.so_far[-1].append(group["lr"])
            if group["lr"] != expected_lr:
                print('Expected vs LRs so far')
                print(self.expected_lrs)
                print(self.so_far)

            assert (
                group["lr"] == expected_lr
            ), f"[ep] LR mismatch: {group['lr']} vs {expected_lr} at " \
               f"{exp_id}-{curr_epoch}-{curr_iter}"

    def before_training_exp(self, strategy, *args, **kwargs):
        self.so_far.append([])


class _PlainMLP(nn.Module, BaseModel):
    """
    An internal MLP implementation without Dropout.

    Needed to reproduce tests for the ReduceLROnPlateau scheduler
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
    ):

        super().__init__()

        layers = nn.Sequential(
            *(nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True))
        )
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                nn.Sequential(
                    *(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(inplace=True),
                    )
                ),
            )

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x


class EvaluationPluginTest(unittest.TestCase):
    def test_publish_metric(self):
        ep = EvaluationPlugin()
        mval = MetricValue(self, "metric", 1.0, 0)
        ep.publish_metric_value(mval)

        # check key exists
        assert len(ep.get_all_metrics()["metric"][1]) == 1

    def test_forward_callbacks(self):
        # The EvaluationPlugin should forward all the callbacks to metrics,
        # even those that are unused by the EvaluationPlugin itself.
        class MetricMock:
            def __init__(self):
                self.x = 0

            def before_blabla(self, strategy):
                self.x += 1

        met = MetricMock()
        evalp = EvaluationPlugin(met)
        evalp.before_blabla(None)

        # it should ignore undefined callbacks
        evalp.after_blabla(None)

        # it should raise error for other undefined attributes
        with self.assertRaises(AttributeError):
            evalp.asd(None)


class EarlyStoppingPluginTest(unittest.TestCase):
    def test_early_stop_epochs(self):
        class MockEvaluator:
            def __init__(self, clock, metrics):
                self.clock = clock
                self.metrics = metrics

            def get_last_metrics(self):
                idx = self.clock.train_exp_iterations
                return {"Top1_Acc_Stream/eval_phase/a": self.metrics[idx]}

        class ESMockStrategy:
            """An empty strategy to test early stopping."""

            def __init__(self, p, metric_vals):
                self.p = p
                self.clock = Clock()
                self.evaluator = MockEvaluator(self.clock, metric_vals)

                self.model = SimpleMLP()

            def before_training_iteration(self):
                self.p.before_training_iteration(self)
                self.clock.before_training_iteration(self)

            def before_training_epoch(self):
                self.p.before_training_epoch(self)
                self.clock.before_training_epoch(self)

            def after_training_iteration(self):
                self.p.after_training_iteration(self)
                self.clock.after_training_iteration(self)

            def after_training_epoch(self):
                self.p.after_training_epoch(self)
                self.clock.after_training_epoch(self)

            def stop_training(self):
                raise StopIteration()

        def run_es(mvals, p):
            strat = ESMockStrategy(p, mvals)
            for t in range(100):
                try:
                    if t % 10 == 0:
                        strat.before_training_epoch()
                    strat.before_training_iteration()
                    strat.after_training_iteration()
                    if t % 10 == 9:
                        strat.after_training_epoch()
                except StopIteration:
                    break
            return strat

        # best on epoch
        metric_vals = list(range(200))
        p = EarlyStoppingPlugin(5, val_stream_name="a")
        run_es(metric_vals, p)
        print(f"best step={p.best_step}, val={p.best_val}")
        assert p.best_step == 9
        assert p.best_val == 90

        # best on iteration
        metric_vals = list(range(200))
        p = EarlyStoppingPlugin(5, val_stream_name="a", peval_mode="iteration")
        run_es(metric_vals, p)
        print(f"best step={p.best_step}, val={p.best_val}")
        assert p.best_step == 99
        assert p.best_val == 99

        # check patience
        metric_vals = list([1 for _ in range(200)])
        p = EarlyStoppingPlugin(5, val_stream_name="a")
        strat = run_es(metric_vals, p)
        print(f"best step={p.best_step}, val={p.best_val}")
        assert p.best_step == 0
        assert strat.clock.train_exp_epochs == p.patience
        assert p.best_val == 1


if __name__ == "__main__":
    unittest.main()
