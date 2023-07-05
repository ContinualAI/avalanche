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
"""Dynamic Modules are Pytorch modules that can be incrementally expanded
to allow architectural modifications (multi-head classifiers, progressive
networks, ...).
"""
import torch
from torch.nn import Module
from typing import Optional

from avalanche.benchmarks.utils.flat_data import ConstantSequence
from avalanche.benchmarks.scenarios import CLExperience


class DynamicModule(Module):
    """Dynamic Modules are Avalanche modules that can be incrementally
    expanded to allow architectural modifications (multi-head
    classifiers, progressive networks, ...).

    Compared to pytoch Modules, they provide an additional method,
    `model_adaptation`, which adapts the model given the current experience.
    """

    def adaptation(self, experience: CLExperience):
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

        :param experience: the current experience.
        :return:
        """
        if self.training:
            self.train_adaptation(experience)
        else:
            self.eval_adaptation(experience)

    def train_adaptation(self, experience: CLExperience):
        """Module's adaptation at training time.

        Avalanche strategies automatically call this method *before* training
        on each experience.
        """
        pass

    def eval_adaptation(self, experience: CLExperience):
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

    @property
    def _adaptation_device(self):
        """
        The device to use when expanding (or otherwise adapting)
        the model. Defaults to the current device of the fist
        parameter listed using :meth:`parameters`.
        """
        return next(self.parameters()).device


class MultiTaskModule(DynamicModule):
    """Base pytorch Module with support for task labels.

    Multi-task modules are ``torch.nn.Module`` for multi-task
    scenarios. The ``forward`` method accepts task labels, one for
    each sample in the mini-batch.

    By default the ``forward`` method splits the mini-batch by task
    and calls ``forward_single_task``. Subclasses must implement
    ``forward_single_task`` or override `forward. If ``task_labels == None``,
    the output is computed in parallel for each task.
    """

    def __init__(self):
        super().__init__()
        self.max_class_label = 0
        self.known_train_tasks_labels = set()
        """ Set of task labels encountered up to now. """

    def adaptation(self, experience: CLExperience):
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

        :param experience: the current experience.
        :return:
        """
        curr_classes = experience.classes_in_this_experience
        self.max_class_label = max(self.max_class_label, max(curr_classes) + 1)
        if self.training:
            self.train_adaptation(experience)
        else:
            self.eval_adaptation(experience)

    def eval_adaptation(self, experience: CLExperience):
        pass

    def train_adaptation(self, experience: CLExperience):
        """Update known task labels."""
        task_labels = experience.task_labels
        self.known_train_tasks_labels = self.known_train_tasks_labels.union(
            set(task_labels)
        )

    def forward(self, x: torch.Tensor, task_labels: torch.Tensor) -> torch.Tensor:
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
            assert len(out_task.shape) == 2, (
                "multi-head assumes mini-batches of 2 dimensions " "<batch, classes>"
            )
            n_labels_head = out_task.shape[1]
            out[task_mask, :n_labels_head] = out_task
        return out

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
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

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        masking=True,
        mask_value=-1000,
    ):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value

        self.classifier = torch.nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        device = self._adaptation_device
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[curr_classes] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x, **kwargs):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)
        return out


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

    def __init__(
        self,
        in_features,
        initial_out_features=2,
        masking=True,
        mask_value=-1000,
    ):
        """Init.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value
        self.in_features = in_features
        self.starting_out_features = initial_out_features
        self.classifiers = torch.nn.ModuleDict()

        # needs to create the first head because pytorch optimizers
        # fail when model.parameters() is empty.
        # masking in IncrementalClassifier is unaware of task labels
        # so we do masking here instead.
        first_head = IncrementalClassifier(
            self.in_features, self.starting_out_features, masking=False
        )
        self.classifiers["0"] = first_head
        self.max_class_label = max(self.max_class_label, initial_out_features)

        au_init = torch.zeros(initial_out_features, dtype=torch.int8)
        self.register_buffer("active_units_T0", au_init)

    @property
    def active_units(self):
        res = {}
        for tid in self.known_train_tasks_labels:
            mask = getattr(self, f"active_units_T{tid}").to(torch.bool)
            au = torch.arange(0, mask.shape[0])[mask].tolist()
            res[tid] = au
        return res

    @property
    def task_masks(self):
        res = {}
        for tid in self.known_train_tasks_labels:
            res[tid] = getattr(self, f"active_units_T{tid}").to(torch.bool)
        return res

    def adaptation(self, experience: CLExperience):
        """If `dataset` contains new tasks, a new head is initialized.

        :param experience: data from the current experience.
        :return:
        """
        super().adaptation(experience)
        device = self._adaptation_device
        curr_classes = experience.classes_in_this_experience
        task_labels = experience.task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)
            # head adaptation
            if tid not in self.classifiers:  # create new head
                new_head = IncrementalClassifier(
                    self.in_features, self.starting_out_features, masking=False
                ).to(device)
                self.classifiers[tid] = new_head

                au_init = torch.zeros(
                    self.starting_out_features, dtype=torch.int8, device=device
                )
                self.register_buffer(f"active_units_T{tid}", au_init)

            self.classifiers[tid].adaptation(experience)

            # update active_units mask for the current task
            if self.masking:
                # TODO: code below assumes a single task for each experience
                # it should be easy to generalize but it may be slower.
                if len(task_labels) > 1:
                    raise NotImplementedError(
                        "Multi-Head unit masking is not supported when "
                        "experiences have multiple task labels. Set "
                        "masking=False in your "
                        "MultiHeadClassifier to disable masking."
                    )

                au_name = f"active_units_T{tid}"
                curr_head = self.classifiers[tid]
                old_nunits = self._buffers[au_name].shape[0]

                new_nclasses = max(
                    curr_head.classifier.out_features, max(curr_classes) + 1
                )
                if old_nunits != new_nclasses:  # expand active_units mask
                    old_act_units = self._buffers[au_name]
                    self._buffers[au_name] = torch.zeros(
                        new_nclasses, dtype=torch.int8, device=device
                    )
                    self._buffers[au_name][: old_act_units.shape[0]] = old_act_units
                # update with new active classes
                if self.training:
                    self._buffers[au_name][curr_classes] = 1

    def forward_single_task(self, x, task_label):
        """compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        device = self._adaptation_device
        task_label = str(task_label)
        out = self.classifiers[task_label](x)
        if self.masking:
            au_name = f"active_units_T{task_label}"
            curr_au = self._buffers[au_name]
            nunits, oldsize = out.shape[-1], curr_au.shape[0]
            if oldsize < nunits:  # we have to update the mask
                old_mask = self._buffers[au_name]
                self._buffers[au_name] = torch.zeros(
                    nunits, dtype=torch.int8, device=device
                )
                self._buffers[au_name][:oldsize] = old_mask
                curr_au = self._buffers[au_name]
            out[..., torch.logical_not(curr_au)] = self.mask_value
        return out


class TrainEvalModel(torch.nn.Module):
    """
    TrainEvalModel.
    This module allows to wrap together a common feature extractor and
    two classifiers: one used during training time and another
    used at test time. The classifier is switched depending on the
    `training` state of the module.
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

    def forward(self, x):
        x = self.feature_extractor(x)
        if self.training:
            return self.train_classifier(x)
        else:
            return self.eval_classifier(x)


__all__ = [
    "DynamicModule",
    "MultiTaskModule",
    "IncrementalClassifier",
    "MultiHeadClassifier",
    "TrainEvalModel",
]
