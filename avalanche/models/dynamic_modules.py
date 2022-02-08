################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-04-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Dynamic Modules are Pytorch modules that can be incrementally expanded
    to allow architectural modifications (multi-head classifiers, progressive
    networks, ...).
"""
import torch
from torch.nn import Module

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence


class DynamicModule(Module):
    """
    Dynamic Modules are Avalanche modules that can be incrementally
    expanded to allow architectural modifications (multi-head
    classifiers, progressive networks, ...).

    Compared to pytoch Modules, they provide an additional method,
    `model_adaptation`, which adapts the model given data from the
    current experience.
    """

    def adaptation(self, dataset: AvalancheDataset = None):
        """Adapt the module (freeze units, add units...) using the current
        data. Optimizers must be updated after the model adaptation.

        Avalanche strategies call this method to adapt the architecture
        *before* processing each experience. Strategies also update the
        optimizer automatically.

        .. warning::
            As a general rule, you should NOT use this method to train the
            model. The dataset should be used only to check conditions which
            require the model's adaptation, such as the discovery of new
            classes or tasks.

        :param dataset: data from the current experience.
        :return:
        """
        if self.training:
            self.train_adaptation(dataset)
        else:
            self.eval_adaptation(dataset)

    def train_adaptation(self, dataset: AvalancheDataset):
        """Module's adaptation at training time.

        Avalanche strategies automatically call this method *before* training
        on each experience.
        """
        pass

    def eval_adaptation(self, dataset: AvalancheDataset):
        """Module's adaptation at evaluation time.

        Avalanche strategies automatically call this method *before* evaluating
        on each experience.

        .. warning::
            This method receives the experience's data at evaluation time
            because some dynamic models need it for adaptation. For example,
            an incremental classifier needs to be expanded even at evaluation
            time if new classes are available. However, you should **never**
            use this data to **train** the module's parameters.
        """
        pass


class MultiTaskModule(DynamicModule):
    """
    Multi-task modules are `torch.nn.Modules`s for multi-task
    scenarios. The `forward` method accepts task labels, one for
    each sample in the mini-batch.

    By default the `forward` method splits the mini-batch by task
    and calls `forward_single_task`. Subclasses must implement
    `forward_single_task` or override `forward.

    if `task_labels == None`, the output is computed in parallel
    for each task.
    """

    def __init__(self):
        super().__init__()
        self.known_train_tasks_labels = set()
        self.max_class_label = 0
        """ Set of task labels encountered up to now. """

    def adaptation(self, dataset: AvalancheDataset = None):
        """Adapt the module (freeze units, add units...) using the current
        data. Optimizers must be updated after the model adaptation.

        Avalanche strategies call this method to adapt the architecture
        *before* processing each experience. Strategies also update the
        optimizer automatically.

        .. warning::
            As a general rule, you should NOT use this method to train the
            model. The dataset should be used only to check conditions which
            require the model's adaptation, such as the discovery of new
            classes or tasks.

        :param dataset: data from the current experience.
        :return:
        """
        self.max_class_label = max(self.max_class_label,
                                   max(dataset.targets) + 1)
        if self.training:
            self.train_adaptation(dataset)
        else:
            self.eval_adaptation(dataset)

    def eval_adaptation(self, dataset: AvalancheDataset):
        pass

    def train_adaptation(self, dataset: AvalancheDataset = None):
        """Update known task labels."""
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]
        self.known_train_tasks_labels = self.known_train_tasks_labels.union(
            set(task_labels)
        )

    def forward(
        self, x: torch.Tensor, task_labels: torch.Tensor
    ) -> torch.Tensor:
        """compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """
        if task_labels is None:
            return self.forward_all_tasks(x)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels)
        else:
            unique_tasks = torch.unique(task_labels)

        out = torch.zeros(x.shape[0], self.max_class_label, device=x.device)
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())
            assert len(out_task.shape) == 2,\
                "multi-head assumes mini-batches of 2 dimensions " \
                "<batch, classes>"
            n_labels_head = out_task.shape[1]
            out[task_mask, :n_labels_head] = out_task
        return out

    def forward_single_task(
        self, x: torch.Tensor, task_label: int
    ) -> torch.Tensor:
        """compute the output given the input `x` and task label.

        :param x:
        :param task_label: a single task label.
        :return:
        """
        raise NotImplementedError()

    def forward_all_tasks(self, x: torch.Tensor):
        """compute the output given the input `x` and task label.
        By default, it considers only tasks seen at training time.

        :param x:
        :return: all the possible outputs are returned as a dictionary
            with task IDs as keys and the output of the corresponding
            task as output.
        """
        res = {}
        for task_id in self.known_train_tasks_labels:
            res[task_id] = self.forward_single_task(x, task_id)
        return res


class IncrementalClassifier(DynamicModule):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(self, in_features, initial_out_features=2):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.classifier = torch.nn.Linear(in_features, initial_out_features)

    @torch.no_grad()
    def adaptation(self, dataset: AvalancheDataset):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param dataset: data from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = max(
            self.classifier.out_features, max(dataset.targets) + 1
        )

        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x, **kwargs):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        return self.classifier(x)


class MultiHeadClassifier(MultiTaskModule):
    """Multi-head classifier with separate heads for each task.

    Typically used in task-incremental benchmarks where task labels are
    available and provided to the model.

    .. note::
        Each output head may have a different shape, and the number of
        classes can be determined automatically.

        However, since pytorch doest not support jagged tensors, when you
        compute a minibatch's output you must ensure that each sample
        has the same output size, otherwise the model will fail to
        concatenate the samples together.

        These can be easily ensured in two possible ways:
        - each minibatch contains a single task, which is the case in most
            common benchmarks in Avalanche. Some exceptions to this setting
            are multi-task replay or cumulative strategies.
        - each head has the same size, which can be enforced by setting a
            large enough `initial_out_features`.
    """

    def __init__(self, in_features, initial_out_features=2):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.in_features = in_features
        self.starting_out_features = initial_out_features
        self.classifiers = torch.nn.ModuleDict()

        # needs to create the first head because pytorch optimizers
        # fail when model.parameters() is empty.
        first_head = IncrementalClassifier(
            self.in_features, self.starting_out_features
        )
        self.classifiers["0"] = first_head
        self.max_class_label = max(self.max_class_label,
                                   initial_out_features)

    def adaptation(self, dataset: AvalancheDataset):
        """If `dataset` contains new tasks, a new head is initialized.

        :param dataset: data from the current experience.
        :return:
        """
        super().adaptation(dataset)
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)  # need str keys
            if tid not in self.classifiers:
                new_head = IncrementalClassifier(
                    self.in_features, self.starting_out_features
                )
                new_head.adaptation(dataset)
                self.classifiers[tid] = new_head

    def forward_single_task(self, x, task_label):
        """compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        return self.classifiers[str(task_label)](x)


class TrainEvalModel(DynamicModule):
    """
    TrainEvalModel.
    This module allows to wrap together a common feature extractor and
    two classifiers: one used during training time and another
    used at test time. The classifier is switched when `self.adaptation()`
    is called.
    """

    def __init__(self, feature_extractor, train_classifier, eval_classifier):
        """
        :param feature_extractor: a differentiable feature extractor
        :param train_classifier: a differentiable classifier used
            during training
        :param eval_classifier: a classifier used during testing.
            Doesn't have to be differentiable.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.train_classifier = train_classifier
        self.eval_classifier = eval_classifier

        self.classifier = train_classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def train_adaptation(self, dataset: AvalancheDataset = None):
        self.classifier = self.train_classifier

    def eval_adaptation(self, dataset: AvalancheDataset = None):
        self.classifier = self.eval_classifier


__all__ = [
    "DynamicModule",
    "MultiTaskModule",
    "IncrementalClassifier",
    "MultiHeadClassifier",
    "TrainEvalModel",
]
