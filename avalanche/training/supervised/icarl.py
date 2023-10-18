import itertools
from typing import Callable, Optional, List, Union
import torch
from torch.optim import Optimizer

from avalanche.benchmarks.utils import (
    _make_taskaware_tensor_classification_dataset,
    _taskaware_classification_subset,
)
from math import ceil

from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.models import TrainEvalModel, NCMClassifier
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.losses import ICaRLLossPlugin
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch.nn import Module
from torch.utils.data import DataLoader
from avalanche.training.templates import SupervisedTemplate


class ICaRL(SupervisedTemplate):
    """iCaRL Strategy.

    This strategy does not use task identities.
    """

    def __init__(
        self,
        feature_extractor: Module,
        classifier: Module,
        optimizer: Optimizer,
        memory_size,
        buffer_transform,
        fixed_memory,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
    ):
        """Init.

        :param feature_extractor: The feature extractor.
        :param classifier: The differentiable classifier that takes as input
            the output of the feature extractor.
        :param optimizer: The optimizer to use.
        :param memory_size: The nuber of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
            replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """
        model = TrainEvalModel(
            feature_extractor,
            train_classifier=classifier,
            eval_classifier=NCMClassifier(normalize=True),
        )

        criterion = ICaRLLossPlugin()  # iCaRL requires this specific loss (#966)
        icarl = _ICaRLPlugin(memory_size, buffer_transform, fixed_memory)

        if plugins is None:
            plugins = [icarl]
        else:
            plugins += [icarl]

        if isinstance(criterion, SupervisedPlugin):
            plugins += [criterion]

        super().__init__(
            model,
            optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )


class _ICaRLPlugin(SupervisedPlugin):
    """
    iCaRL Plugin.
    iCaRL uses nearest class exemplar classification to prevent
    forgetting to occur at the classification layer. The feature extractor
    is continually learned using replay and distillation. The exemplars
    used for replay and classification are selected through herding.
    This plugin does not use task identities.
    """

    def __init__(self, memory_size, buffer_transform=None, fixed_memory=True):
        """
        :param memory_size: amount of patterns saved in the memory.
        :param buffer_transform: transform applied on buffer elements already
            modified by test_transform (if specified) before being used for
             replay
        :param fixed_memory: If True a memory of size memory_size is
            allocated and partitioned between samples from the observed
            experiences. If False every time a new class is observed
            memory_size samples of that class are added to the memory.
        """
        super().__init__()

        self.memory_size = memory_size
        self.buffer_transform = buffer_transform
        self.fixed_memory = fixed_memory

        self.x_memory = []
        self.y_memory = []
        self.order = []

        self.observed_classes = []
        self.class_means = {}
        self.embedding_size = None
        self.output_size = None
        self.input_size = None

    def after_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.clock.train_exp_counter != 0:
            memory = _make_taskaware_tensor_classification_dataset(
                torch.cat(self.x_memory).cpu(),
                torch.tensor(list(itertools.chain.from_iterable(self.y_memory))),
                transform=self.buffer_transform,
                target_transform=None,
            )

            strategy.adapted_dataset = concat_datasets(
                (strategy.adapted_dataset, memory)
            )

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        tid = strategy.clock.train_exp_counter
        benchmark = strategy.experience.benchmark
        nb_cl = benchmark.n_classes_per_exp[tid]
        previous_seen_classes = sum(benchmark.n_classes_per_exp[:tid])

        self.observed_classes.extend(
            benchmark.classes_order[
                previous_seen_classes : previous_seen_classes + nb_cl
            ]
        )

    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        if self.input_size is None:
            with torch.no_grad():
                self.input_size = strategy.mb_x.shape[1:]
                self.output_size = strategy.model(strategy.mb_x).shape[1]
                self.embedding_size = strategy.model.feature_extractor(
                    strategy.mb_x
                ).shape[1]

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        strategy.model.eval()

        self.construct_exemplar_set(strategy)
        self.reduce_exemplar_set(strategy)
        self.compute_class_means(strategy)
        strategy.model.train()

    def compute_class_means(self, strategy):
        if self.class_means == {}:
            n_classes = sum(strategy.experience.benchmark.n_classes_per_exp)
            self.class_means = {
                c_id: torch.zeros(self.embedding_size, device=strategy.device)
                for c_id in range(n_classes)
            }

        for i, class_samples in enumerate(self.x_memory):
            label = self.y_memory[i][0]
            class_samples = class_samples.to(strategy.device)

            with torch.no_grad():
                mapped_prototypes = strategy.model.feature_extractor(
                    class_samples
                ).detach()
            D = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            if len(class_samples.shape) == 4:
                class_samples = torch.flip(class_samples, [3])

            with torch.no_grad():
                mapped_prototypes2 = strategy.model.feature_extractor(
                    class_samples
                ).detach()

            D2 = mapped_prototypes2.T
            D2 = D2 / torch.norm(D2, dim=0)

            div = torch.ones(class_samples.shape[0], device=strategy.device)
            div = div / class_samples.shape[0]

            m1 = torch.mm(D, div.unsqueeze(1)).squeeze(1)
            m2 = torch.mm(D2, div.unsqueeze(1)).squeeze(1)

            self.class_means[label] = (m1 + m2) / 2
            self.class_means[label] /= torch.norm(self.class_means[label])

        strategy.model.eval_classifier.replace_class_means_dict(self.class_means)

    def construct_exemplar_set(self, strategy: SupervisedTemplate):
        assert strategy.experience is not None
        tid = strategy.clock.train_exp_counter
        benchmark = strategy.experience.benchmark
        nb_cl = benchmark.n_classes_per_exp[tid]
        previous_seen_classes = sum(benchmark.n_classes_per_exp[:tid])

        if self.fixed_memory:
            nb_protos_cl = int(ceil(self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size
        new_classes = self.observed_classes[
            previous_seen_classes : previous_seen_classes + nb_cl
        ]

        dataset = strategy.experience.dataset
        targets = torch.tensor(dataset.targets)
        for iter_dico in range(nb_cl):
            cd = _taskaware_classification_subset(
                dataset, torch.where(targets == new_classes[iter_dico])[0]
            )
            collate_fn = cd.collate_fn if hasattr(cd, "collate_fn") else None

            eval_dataloader = DataLoader(
                cd.eval(), collate_fn=collate_fn, batch_size=strategy.eval_mb_size
            )

            class_patterns = []
            mapped_prototypes = []
            for idx, (class_pt, _, _) in enumerate(eval_dataloader):
                class_pt = class_pt.to(strategy.device)
                class_patterns.append(class_pt)
                with torch.no_grad():
                    mapped_pttp = strategy.model.feature_extractor(class_pt).detach()
                mapped_prototypes.append(mapped_pttp)

            class_patterns_tensor = torch.cat(class_patterns, dim=0)
            mapped_prototypes_tensor = torch.cat(mapped_prototypes, dim=0)

            D = mapped_prototypes_tensor.T
            D = D / torch.norm(D, dim=0)

            mu = torch.mean(D, dim=1)
            order = torch.zeros(class_patterns_tensor.shape[0])
            w_t = mu

            i, added, selected = 0, 0, []
            while not added == nb_protos_cl and i < 1000:
                tmp_t = torch.mm(w_t.unsqueeze(0), D)
                ind_max = torch.argmax(tmp_t)

                if ind_max not in selected:
                    order[ind_max] = 1 + added
                    added += 1
                    selected.append(ind_max.item())

                w_t = w_t + mu - D[:, ind_max]
                i += 1

            pick = (order > 0) * (order < nb_protos_cl + 1) * 1.0
            self.x_memory.append(class_patterns_tensor[torch.where(pick == 1)[0]])
            self.y_memory.append(
                [new_classes[iter_dico]] * len(torch.where(pick == 1)[0])
            )
            self.order.append(order[torch.where(pick == 1)[0]])

    def reduce_exemplar_set(self, strategy: SupervisedTemplate):
        assert strategy.experience is not None
        tid = strategy.clock.train_exp_counter
        nb_cl = strategy.experience.benchmark.n_classes_per_exp

        if self.fixed_memory:
            nb_protos_cl = int(ceil(self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size

        for i in range(len(self.x_memory) - nb_cl[tid]):
            pick = (self.order[i] < nb_protos_cl + 1) * 1.0
            self.x_memory[i] = self.x_memory[i][torch.where(pick == 1)[0]]
            self.y_memory[i] = self.y_memory[i][: len(torch.where(pick == 1)[0])]
            self.order[i] = self.order[i][torch.where(pick == 1)[0]]
