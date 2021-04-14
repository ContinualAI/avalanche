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
from typing import Dict

import torch
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence


class DynamicModule(torch.nn.Module):
    def __init__(self):
        """
            Dynamic Modules are Avalanche modules that can be incrementally
            expanded to allow architectural modifications (multi-head
            classifiers, progressive networks, ...).

            Compared to pytoch Modules, they provide an additional method,
            `model_adaptation`, which adapts the model given data from the
            current experience.
        """
        super().__init__()

    def adaptation(self, dataset: AvalancheDataset):
        """ Adapt the module (freeze units, add units...) using the current
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
        """ Module's adaptation at training time.

        Avalanche strategies automatically call this method *before* training
        on each experience.
        """
        pass

    def eval_adaptation(self, dataset: AvalancheDataset):
        """ Module's adaptation at evaluation time.

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


class MultiTaskModule:
    def __init__(self):
        """
            Multi-task dynamic modules are `torch.nn.Modules`s for multi-task
            scenarios. The `forward` method accepts task labels, one for
            each sample in the mini-batch.
        """
        super().__init__()

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor)\
            -> torch.Tensor:
        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample.
        :return:
        """
        unique_tasks = torch.unique(task_labels)
        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(x_task, task.item())

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out

    def forward_single_task(self, x: torch.Tensor, task_label: int)\
            -> torch.Tensor:
        """ compute the output given the input `x` and task label.

        :param x:
        :param task_label: a single task label.
        :return:
        """
        assert NotImplementedError()


class IncrementalClassifier(DynamicModule):
    def __init__(self, in_features, initial_out_features=2):
        """ Output layer that incrementally adds units whenever new classes are
        encountered.

        Typically used in class-incremental scenarios where the number of
        classes grows over time.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.classifier = torch.nn.Linear(in_features, initial_out_features)

    def adaptation(self, dataset: AvalancheDataset):
        """ If `dataset` contains unseen classes the classifier is expanded.

        :param dataset: data from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        out_features = max(self.classifier.out_features,
                           max(dataset.targets) + 1)
        self.classifier = torch.nn.Linear(in_features, out_features)

    def forward(self, x, **kwargs):
        """ compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        return self.classifier(x)


class MultiHeadClassifier(MultiTaskModule, DynamicModule):
    def __init__(self, in_features, initial_out_features=2):
        """ Multi-head classifier with separate classifiers for each task.

        Typically used in task-incremental scenarios where task labels are
        available and provided to the model.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.in_features = in_features
        self.starting_out_features = initial_out_features
        self.classifiers = torch.nn.ModuleDict()

    def adaptation(self, dataset: AvalancheDataset):
        """ If `dataset` contains new tasks, a new head is initialized.

        :param dataset: data from the current experience.
        :return:
        """
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            if tid not in self.classifiers:
                new_head = IncrementalClassifier(self.in_features,
                                                 self.starting_out_features)
                new_head.adaptation(dataset)
                self.classifiers[str(tid)] = new_head

    def forward_single_task(self, x, task_label):
        """ compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        return self.classifiers[str(task_label)](x)


__all__ = [
    'DynamicModule',
    'IncrementalClassifier',
    'MultiHeadClassifier'
]
