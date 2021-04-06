"""
    Dynamic Modules are Pytorch modules that can be incrementally expanded
    to allow architectural modifications (multi-head classifiers, progressive
    networks, ...).
"""
from typing import Dict

import torch
from avalanche.benchmarks.utils import AvalancheDataset


class DynamicModule(torch.nn.Module):
    def __init__(self):
        """
            Dynamic Modules are Pytorch modules that can be incrementally
            expanded to allow architectural modifications (multi-head
            classifiers, progressive networks, ...).

            Compared to pytoch Modules, they provide an additional method,
            `model_adaptation`, which adapts the model given data from the
            current experience.
        """
        super().__init__()

    def model_adaptation(self, dataset: AvalancheDataset):
        """ Adapt the model (freeze units, add units...) using the current
        data. Optimizers must be updated after the model adaptation.

        :param dataset: data from the current experience.
        :return:
        """
        assert NotImplementedError()

    def forward(self, x, task_labels=None):
        """ compute the output given the input `x` and task labels.

        :param x:
        :param task_labels:
        :return:
        """
        assert NotImplementedError()


class IncrementalClassifier(DynamicModule):
    def __init__(self, in_features, starting_out_features):
        """ Output layer that incrementally adds units whenever new classes are
        encountered.

        :param in_features:
        :param starting_out_features:
        """
        super().__init__()
        self.classifier = torch.nn.Linear(in_features, starting_out_features)

    def model_adaptation(self, dataset: AvalancheDataset):
        """ Adapt the model (freeze units, add units...) using the current
        data.

        :param dataset: data from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        out_features = max(self.classifier.out_features,
                           max(dataset.targets) + 1)
        self.classifier = torch.nn.Linear(in_features, out_features)

    def forward(self, x, **kwargs):
        """ compute the output given the input `x`. This module does not use
        task labels.

        :param x:
        :return:
        """
        return self.classifier(x)


class MultiHeadClassifier(DynamicModule):
    def __init__(self, in_features, starting_out_features):
        """ Multi-head classifier with separate heads for each task.

        :param in_features: number of input features.
        :param starting_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.in_features = in_features
        self.starting_out_features = starting_out_features
        self.classifiers: Dict[int, IncrementalClassifier] = dict()

    def model_adaptation(self, dataset: AvalancheDataset):
        """ Adapt the model (freeze units, add units...) using the current
        data.

        :param dataset: data from the current experience.
        :return:
        """
        task_labels = dataset.targets_task_labels
        for tid in task_labels:
            if tid not in self.classifiers:
                new_head = IncrementalClassifier(self.in_features, self.starting_out_features)
                self.classifiers[tid] = new_head

        for head in self.classifiers.values():
            head.model_adaptation(dataset)

    def forward(self, x, **kwargs):
        """ compute the output given the input `x`. This module does not use
        task labels.

        :param x:
        :return:
        """
        return self.classifier(x)


__all__ = [
    'DynamicModule',
    'IncrementalClassifier',
    'MultiHeadClassifier'
]