################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-12-2020                                                             #
# Author(s): Antonio Carta, Andrea Cossu                                       #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
import copy
import random
import quadprog
import logging
from collections import defaultdict
from typing import Dict, Any, Optional, Union, Sequence, List, TYPE_CHECKING

import numpy as np
import torch
from torch.nn import Module, Linear
from torch.utils.data import random_split, ConcatDataset, TensorDataset

from avalanche.benchmarks.scenarios import IStepInfo
from avalanche.evaluation import OnTrainStepStart, EvalData, \
    MetricValue, OnTrainIteration, OnTestStepStart, OnTestIteration, \
    OnTestStepEnd, OnTrainStepEnd, OnTrainEpochStart, OnTrainEpochEnd, \
    EpochAccuracy, EpochLoss
from avalanche.evaluation.abstract_metric import AbstractMetric
from avalanche.evaluation.evaluation_data import OnTestPhaseEnd, \
    OnTestPhaseStart, OnTrainPhaseStart, OnTrainPhaseEnd
from avalanche.extras.logging import Logger

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy
from avalanche.training.utils import copy_params_dict, zerolike_params_dict


class StrategyPlugin:
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses must override
    the callbacks.
    """

    def __init__(self):
        pass

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_step(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def adapt_train_dataset(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_step(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_test(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def adapt_test_dataset(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_test_step(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_test_step(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_test(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_test_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_test_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_test_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_test_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass


class ReplayPlugin(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implements the "adapt_train_dataset" callback to add them to
    the training set.
    This plugin does not use task identities.

    The :mem_size: attribute controls the number of patterns to be stored in 
    the external memory. We assume the training set contains at least 
    :mem_size: data points.
    """

    def __init__(self, mem_size=200):
        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = None
        self.it = 0
        self.rm_add = None

    def adapt_train_dataset(self, strategy, **kwargs):
        """
        Expands the current training set with datapoint from
        the external memory before training.
        """

        # Additional set of the current batch to be concatenated to the ext.
        # memory at the end of the training
        self.rm_add = None

        # how many patterns to save for next iter
        h = min(self.mem_size // (self.it + 1), len(strategy.current_data))

        # We recover it using the random_split method and getting rid of the
        # second split.
        self.rm_add, _ = random_split(
            strategy.current_data, [h, len(strategy.current_data) - h]
        )

        if self.it > 0:
            # We update the train_dataset concatenating the external memory.
            # We assume the user will shuffle the data when creating the data
            # loader.
            strategy.current_data = ConcatDataset([strategy.current_data,
                                                   self.ext_mem])

    def after_training_step(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of
         the current training batch/task. """

        # replace patterns in random memory
        ext_mem = self.ext_mem
        if self.it == 0:
            ext_mem = copy.deepcopy(self.rm_add)
        else:
            _, saved_part = random_split(
                ext_mem, [len(self.rm_add), len(ext_mem) - len(self.rm_add)]
            )
            ext_mem = ConcatDataset([saved_part, self.rm_add])
        self.ext_mem = ext_mem
        self.it += 1


class GDumbPlugin(StrategyPlugin):
    """
    A GDumb plugin. At each step the model
    is trained with all and only the data of the external memory.
    The memory is updated at the end of each step to add new classes or
    new examples of already encountered classes.
    This plugin does not use task identities.

    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.

    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size=200):

        super().__init__()

        self.it = 0
        self.mem_size = mem_size
        self.ext_mem = None
        # count occurrences for each class
        self.counter = defaultdict(int)

    def adapt_train_dataset(self, strategy, **kwargs):
        """ Before training we make sure to organize the memory following
            GDumb approach and updating the dataset accordingly.
        """

        # for each pattern, add it to the memory or not
        for i, (pattern, target_value) in enumerate(strategy.current_data):
            target = torch.tensor(target_value)
            if len(pattern.size()) == 1:
                pattern = pattern.unsqueeze(0)
                
            if self.counter == {}:
                # any positive (>0) number is ok
                patterns_per_class = 1
            else:
                patterns_per_class = int(
                    self.mem_size / len(self.counter.keys())
                )

            if target_value not in self.counter or \
                    self.counter[target_value] < patterns_per_class:
                # full memory: replace item from most represented class
                # with current pattern
                if sum(self.counter.values()) >= self.mem_size:
                    to_remove = max(self.counter, key=self.counter.get)
                    for j in range(len(self.ext_mem.tensors[1])):
                        if self.ext_mem.tensors[1][j].item() == to_remove:
                            self.ext_mem.tensors[0][j] = pattern
                            self.ext_mem.tensors[1][j] = target
                            break
                    self.counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    if self.ext_mem is None:
                        self.ext_mem = TensorDataset(
                            pattern, target.unsqueeze(0))
                    else:
                        self.ext_mem = TensorDataset(
                            torch.cat([
                                pattern,
                                self.ext_mem.tensors[0]], dim=0),

                            torch.cat([
                                target.unsqueeze(0),
                                self.ext_mem.tensors[1]], dim=0)
                        )

                self.counter[target_value] += 1

        strategy.current_data = self.ext_mem


class EvaluationPlugin(StrategyPlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and testing loops of the strategy through callbacks.

    This plugin updates the given metrics and logs them using the provided
    loggers.
    """

    def __init__(self,
                 loggers: Union[Logger, Sequence[Logger]],
                 *metrics: AbstractMetric):
        """
        Creates an instance of the evaluation plugin.

        :param loggers: The loggers to use to log the metric values.
        :param metrics: The metrics to compute.
        """
        super().__init__()

        self.metrics = metrics

        if isinstance(loggers, Logger):
            loggers = [loggers]
        self.loggers = loggers

        # Private state variables
        self._steps_counter: int = -1

        # Training variables
        self._current_train_step_id: Optional[int] = None
        self._train_current_task_id = None

        # Test variables
        self._current_test_step_id: Optional[int] = None
        self._test_current_task_id = None

    def _log_info(self, msg: str):
        for logger in self.loggers:
            logger.log.info(msg)

    def _log_metric_values(self, metric_values: List[MetricValue]):
        for to_be_logged in metric_values:
            for logger in self.loggers:
                logger.log_metric(to_be_logged)

    def _update_metrics(self, evaluation_data: EvalData):
        metric_values = []
        for metric in self.metrics:
            metric_result = metric(evaluation_data)
            # AbstractMetric result can be a single value or a list of values

            if isinstance(metric_result, MetricValue):
                metric_values.append(metric_result)
            elif metric_result is not None:
                metric_values += metric_result
        self._log_metric_values(metric_values)
        return metric_values

    def before_training(self, strategy, **kwargs):
        evaluation_data = OnTrainPhaseStart(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)

    def after_training(self, strategy, **kwargs):
        evaluation_data = OnTrainPhaseEnd(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)

    def before_training_step(self, strategy, joint_training=False, **kwargs):
        step_info = strategy.step_info
        self._steps_counter += 1
        self._current_train_step_id = step_info.current_step
        self._train_current_task_id = step_info.task_label

        evaluation_data = OnTrainStepStart(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)

        # Logging
        if joint_training:
            self._log_info("[Joint Training]")
        else:
            self._log_info("[Training on Step {}, Task {}]"
                           .format(strategy.step_info.current_step,
                                   self._train_current_task_id))

    def after_training_step(self, strategy, **kwargs):
        evaluation_data = OnTrainStepEnd(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)

    def before_training_epoch(self, strategy, **kwargs):
        evaluation_data = OnTrainEpochStart(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, strategy.epoch)

        # Update metrics
        self._update_metrics(evaluation_data)

    def after_training_epoch(self, strategy, **kwargs):
        evaluation_data = OnTrainEpochEnd(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, strategy.epoch)

        # Update metrics
        self._update_metrics(evaluation_data)

    def after_training_iteration(self, strategy, **kwargs):
        epoch = strategy.epoch
        iteration = strategy.mb_it
        train_mb_y = strategy.mb_y
        logits = strategy.logits
        loss = strategy.loss

        evaluation_data = OnTrainIteration(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, epoch, iteration, train_mb_y,
            logits, loss)

        # Update metrics
        metric_values = self._update_metrics(evaluation_data)

        # Logging
        if (iteration + 1) % 100 == 0:
            # TODO: move logger call elsewhere
            # TODO: EpochAccuracy doesn't emit at iteration end
            # TODO: (add ad hoc method or loss?)
            accuracy_metric = next(
                filter(lambda v: isinstance(v.origin, EpochAccuracy),
                       metric_values), None)
            loss_metric = next(
                filter(lambda v: isinstance(v.origin, EpochLoss),
                       metric_values), None)

            accuracy_result = 'N.A.'
            loss_result = 'N.A.'

            if accuracy_metric is not None:
                accuracy_result = '{:.3f}'.format(accuracy_metric.value)

            if loss_metric is not None:
                loss_result = '{:.6f}'.format(loss_metric.value)

            self._log_info(
                '[Training] ==>>> ep: {}, it: {}, avg. loss: {}, '
                'running train acc: {}'.format(
                    strategy.epoch, iteration, loss_result, accuracy_result))

    def before_test_step(self, strategy, **kwargs):
        step_info: IStepInfo = strategy.step_info
        self._current_test_step_id = step_info.current_step
        self._test_current_task_id = step_info.task_label

        evaluation_data = OnTestStepStart(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, self._current_test_step_id,
            self._test_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)

    def after_test_step(self, strategy, **kwargs):
        evaluation_data = OnTestStepEnd(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, self._current_test_step_id,
            self._test_current_task_id)

        # Update metrics
        metric_values = self._update_metrics(evaluation_data)

        # Logging
        # TODO: move logger call elsewhere
        accuracy_metric = next(
            filter(lambda v: isinstance(v.origin, EpochAccuracy),
                   metric_values), None)
        loss_metric = next(
            filter(lambda v: isinstance(v.origin, EpochLoss), metric_values),
            None)

        accuracy_result = 'N.A.'
        loss_result = 'N.A.'

        if accuracy_metric is not None:
            accuracy_result = '{:.3f}'.format(accuracy_metric.value)

        if loss_metric is not None:
            loss_result = '{:.6f}'.format(loss_metric.value)

        self._log_info(
            "[Evaluation] Task {}, Step {}: Avg Loss {}; Avg Acc {}".format(
                self._test_current_task_id, self._current_test_step_id,
                loss_result, accuracy_result))

    def after_test_iteration(self, strategy, **kwargs):
        iteration = strategy.mb_it
        train_mb_y = strategy.mb_y
        logits = strategy.logits
        loss = strategy.loss

        evaluation_data = OnTestIteration(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, self._current_test_step_id,
            self._test_current_task_id, iteration, train_mb_y,
            logits, loss)

        # Update metrics
        self._update_metrics(evaluation_data)

    def before_test(self, strategy, **kwargs):
        evaluation_data = OnTestPhaseStart(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, self._current_test_step_id,
            self._test_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)

    def after_test(self, strategy, **kwargs):
        evaluation_data = OnTestPhaseEnd(
            self._steps_counter, self._current_train_step_id,
            self._train_current_task_id, self._current_test_step_id,
            self._test_current_task_id)

        # Update metrics
        self._update_metrics(evaluation_data)


class CWRStarPlugin(StrategyPlugin):

    def __init__(self, model, second_last_layer_name, num_classes=50):
        """ CWR* Strategy.
        This plugin does not use task identities.

        :param model: trained model
        :param second_last_layer_name: name of the second to last layer.
        :param num_classes: total number of classes
        """
        super().__init__()
        self.log = logging.getLogger("avalanche")
        self.model = model
        self.second_last_layer_name = second_last_layer_name
        self.num_classes = num_classes

        # Model setup
        self.model.saved_weights = {}
        self.model.past_j = {i: 0 for i in range(self.num_classes)}
        self.model.cur_j = {i: 0 for i in range(self.num_classes)}

        # to be updated
        self.cur_class = None

        # State
        self.batch_processed = 0

    def after_training_step(self, strategy, **kwargs):
        CWRStarPlugin.consolidate_weights(self.model, self.cur_class)
        self.batch_processed += 1

    def before_training_step(self, strategy, **kwargs):
        if self.batch_processed == 1:
            self.freeze_lower_layers()

        # Count current classes and number of samples for each of them.
        count = {i: 0 for i in range(self.num_classes)}
        self.curr_classes = set()
        for _, (_, mb_y) in enumerate(strategy.current_dataloader):
            for y in mb_y:
                self.curr_classes.add(int(y))
                count[int(y)] += 1
        self.cur_class = [int(o) for o in self.curr_classes]

        self.model.cur_j = count
        CWRStarPlugin.reset_weights(self.model, self.cur_class)

    def before_test(self, strategy, **kwargs):
        CWRStarPlugin.set_consolidate_weights(self.model)

    @staticmethod
    def consolidate_weights(model, cur_clas):
        """ Mean-shift for the target layer weights"""

        with torch.no_grad():

            globavg = np.average(model.classifier.weight.detach()
                                 .cpu().numpy()[cur_clas])
            for c in cur_clas:
                w = model.classifier.weight.detach().cpu().numpy()[c]

                if c in cur_clas:
                    new_w = w - globavg
                    if c in model.saved_weights.keys():
                        wpast_j = np.sqrt(model.past_j[c] / model.cur_j[c])
                        # wpast_j = model.past_j[c] / model.cur_j[c]
                        model.saved_weights[c] = (model.saved_weights[c] *
                                                  wpast_j
                                                  + new_w) / (wpast_j + 1)
                    else:
                        model.saved_weights[c] = new_w

    @staticmethod
    def set_consolidate_weights(model):
        """ set trained weights """

        with torch.no_grad():
            for c, w in model.saved_weights.items():
                model.classifier.weight[c].copy_(
                    torch.from_numpy(model.saved_weights[c])
                )

    @staticmethod
    def reset_weights(model, cur_clas):
        """ reset weights"""
        with torch.no_grad():
            model.classifier.weight.fill_(0.0)
            for c, w in model.saved_weights.items():
                if c in cur_clas:
                    model.classifier.weight[c].copy_(
                        torch.from_numpy(model.saved_weights[c])
                    )

    def freeze_lower_layers(self):
        for name, param in self.model.named_parameters():
            # tells whether we want to use gradients for a given parameter
            param.requires_grad = False
            self.log.info("Freezing parameter " + name)
            if name == self.second_last_layer_name:
                break


class MultiHeadPlugin(StrategyPlugin):
    def __init__(self, model, classifier_field: str = 'classifier',
                 keep_initial_layer=False):
        """
        MultiHeadPlugin manages a multi-head readout for multi-task
        scenarios and single-head adaptation for incremental tasks.
        The plugin automatically set the correct output head when the task
        changes and adds new heads when a novel task is encountered.
        This plugin *needs task identities* for multi-task scenarios.
        It does not need task identities for single incremental tasks
        (e.g. class incremental).

        By default, a Linear (fully connected) layer is created
        with as many output units as the number of classes in that task. This
        behaviour can be changed by overriding the "create_task_layer" method.

        By default, weights are initialized using the Linear class default
        initialization. This behaviour can be changed by overriding the
        "initialize_new_task_layer" method.

        When dealing with a Single-Incremental-Task scenario, the final layer
        may get dynamically expanded. By default, the initialization provided by
        the Linear class is used and then weights of already existing classes
        are copied (that  is, without adapting the weights of new classes).
        The user can control how the new weights are initialized by overriding
        "initialize_dynamically_expanded_head".

        :param model: PyTorch model
        :param classifier_field: field of the last layer of model.
        :param keep_initial_layer: if True keeps the initial layer for task 0.
        """
        super().__init__()
        if not hasattr(model, classifier_field):
            raise ValueError('The model has no field named ' + classifier_field)

        self.model = model
        self.classifier_field = classifier_field
        self.task_layers: Dict[int, Any] = dict()
        self._optimizer = None

        if keep_initial_layer:
            self.task_layers[0] = getattr(model, classifier_field)

    def before_training_iteration(self, strategy, **kwargs):
        self._optimizer = strategy.optimizer
        self.set_task_layer(strategy, strategy.step_info)

    def before_test_iteration(self, strategy, **kwargs):
        self._optimizer = strategy.optimizer
        self.set_task_layer(strategy, strategy.step_info)

    @torch.no_grad()
    def set_task_layer(self, strategy, step_info: IStepInfo):
        """
        Sets the correct task layer. Creates a new head for previously
        unseen tasks.

        :param strategy: the CL strategy.
        :param step_info: the step info object.
        :return: None
        """

        # task label is set depending on the type of scenario
        # multitask or others
        if hasattr(strategy, 'mb_task_id'):
            task_label = strategy.mb_task_id
        else:
            task_label = step_info.task_label
        n_output_units = max(step_info.dataset.targets) + 1

        if task_label not in self.task_layers:
            # create head for unseen tasks
            task_layer = self.create_task_layer(n_output_units=n_output_units)
            strategy.add_new_params_to_optimizer(task_layer.parameters())
            self.task_layers[task_label] = task_layer.to(strategy.device)
        else:
            # check head expansion
            self.task_layers[task_label] = \
                self.expand_task_layer(strategy, n_output_units,
                                       self.task_layers[task_label])

        # set correct head
        setattr(self.model, self.classifier_field,
                self.task_layers[task_label])

    @torch.no_grad()
    def create_task_layer(self, n_output_units: int, previous_task_layer=None):
        """
        Creates a new task layer.

        By default, this method will create a new :class:`Linear` layer with
        n_output_units" output units. If  "previous_task_layer" is None,
        the name of the classifier field is used to retrieve the amount of
        input features.

        This method will also be used to create a new layer when expanding
        an existing task head.

        This method can be overridden by the user so that a layer different
        from :class:`Linear` can be created.

        :param n_output_units: The number of output units.
        :param previous_task_layer: If not None, the previously created layer
             for the same task.
        :return: The new layer.
        """
        if previous_task_layer is None:
            current_task_layer: Linear = getattr(self.model,
                                                 self.classifier_field)
            in_features = current_task_layer.in_features
            has_bias = current_task_layer.bias is not None
        else:
            in_features = previous_task_layer.in_features
            has_bias = previous_task_layer.bias is not None

        new_layer = Linear(in_features, n_output_units, bias=has_bias)
        self.initialize_new_task_layer(new_layer)
        return new_layer

    @torch.no_grad()
    def initialize_new_task_layer(self, new_layer: Module):
        """
        Initializes a new head.

        This usually is just a weight initialization procedure, but more
        complex operations can be done as well.

        The head can be either a new layer created for a previously
        unseen task or a layer created to expand an existing task layer. In the
        latter case, the user can define a specific weight initialization
        procedure for the expanded part of the head by overriding the
        "initialize_dynamically_expanded_head" method.

        By default, if no custom implementation is provided, no specific
        initialization is done, which means that the default initialization
        provided by the :class:`Linear` class is used.

        :param new_layer: The new layer to adapt.
        :return: None
        """
        pass

    @torch.no_grad()
    def initialize_dynamically_expanded_head(self, prev_task_layer,
                                             new_task_layer):
        """
        Initializes head weights for enw classes.

        This function is called by "adapt_task_layer" only.

        Defaults to no-op, which uses the initialization provided
        by "initialize_new_task_layer" (already called by "adapt_task_layer").

        This method should initialize the weights for new classes. However,
        if the strategy dictates it, this may be the perfect place to adapt
        weights of previous classes, too.

        :param prev_task_layer: New previous, not expanded, task layer.
        :param new_task_layer: The new task layer, with weights from already
            existing classes already set.
        :return:
        """
        # Example implementation of zero-init:
        # new_task_layer.weight[:prev_task_layer.out_features] = 0.0
        pass

    @torch.no_grad()
    def adapt_task_layer(self, prev_task_layer, new_task_layer):
        """
        Adapts the task layer by copying previous weights to the new layer and
        by calling "initialize_dynamically_expanded_head".

        This method is called by "expand_task_layer" only if a new task layer
        was created as the result of encountering a new class for that task.

        :param prev_task_layer: The previous task later.
        :param new_task_layer: The new task layer.
        :return: None.
        """
        to_copy_units = min(prev_task_layer.out_features,
                            new_task_layer.out_features)

        # Weight copy
        new_task_layer.weight[:to_copy_units] = \
            prev_task_layer.weight[:to_copy_units]

        # Bias copy
        if prev_task_layer.bias is not None and \
                new_task_layer.bias is not None:
            new_task_layer.bias[:to_copy_units] = \
                prev_task_layer.bias[:to_copy_units]

        # Initializes the expanded part (and adapts existing weights)
        self.initialize_dynamically_expanded_head(
            prev_task_layer, new_task_layer)

    @torch.no_grad()
    def expand_task_layer(self, strategy, min_n_output_units: int, task_layer):
        """
        Expands an existing task layer.

        This method checks if the layer for a task should be expanded to
        accommodate for "min_n_output_units" output units. If the task layer
        already contains a sufficient amount of output units, no operations are
        done and "task_layer" will be returned as-is.

        If an expansion is needed, "create_task_layer" will be used to create
        a new layer and then "adapt_task_layer" will be called to copy the
        weights of already seen classes and to initialize the weights
        for the expanded part of the layer.

        :param strategy: CL strategy.
        :param min_n_output_units: The number of required output units.
        :param task_layer: The previous task layer.

        :return: The new layer for the task.
        """
        # Expands (creates new) the fully connected layer
        # then calls adapt_task_layer to copy existing weights +
        # initialize the new weights
        if task_layer.out_features >= min_n_output_units:
            return task_layer

        new_layer = self.create_task_layer(
            min_n_output_units,
            previous_task_layer=task_layer)

        self.adapt_task_layer(task_layer, new_layer.to(strategy.device))
        strategy.update_optimizer(task_layer.parameters(),
                                  new_layer.parameters())
        return new_layer


class LwFPlugin(StrategyPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each step.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None
        self.step_id = 0

    def _distillation_loss(self, out, prev_out):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """

        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    def penalty(self, out, x, alpha):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            y_prev = self.prev_model(x).detach()
            dist_loss = self._distillation_loss(out, y_prev)
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        alpha = self.alpha[self.step_id] \
            if isinstance(self.alpha, (list, tuple)) else self.alpha
        penalty = self.penalty(strategy.logits, strategy.mb_x, alpha)
        strategy.loss += penalty

    def after_training_step(self, strategy, **kwargs):
        """
        Save a copy of the model after each step
        """

        self.prev_model = copy.deepcopy(strategy.model)
        self.step_id += 1


class AGEMPlugin(StrategyPlugin):
    """
    Average Gradient Episodic Memory Plugin.
    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous steps. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_step: int, sample_size: int):
        """
        :param patterns_per_step: number of patterns per step in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.patterns_per_step = int(patterns_per_step)
        self.sample_size = int(sample_size)
    
        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """

        if self.memory_x is not None:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            xref, yref = self.sample_from_memory(self.sample_size)
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)
            out = strategy.model(xref)
            loss = strategy.criterion(out, yref)
            loss.backward()
            self.reference_gradients = [ 
                (n, p.grad)
                for n, p in strategy.model.named_parameters()]

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if self.memory_x is not None:
            for (n1, p1), (n2, refg) in zip(strategy.model.named_parameters(),
                                            self.reference_gradients):

                assert n1 == n2, "Different model parameters in AGEM projection"
                assert (p1.grad is not None and refg is not None) \
                    or (p1.grad is None and refg is None)

                if refg is None:
                    continue

                dotg = torch.dot(p1.grad.view(-1), refg.view(-1))
                dotref = torch.dot(refg.view(-1), refg.view(-1))
                if dotg < 0:
                    p1.grad -= (dotg / dotref) * refg

    def after_training_step(self, strategy, **kwargs):
        """
        Save a copy of the model after each step
        """

        self.update_memory(strategy.current_dataloader)

    def sample_from_memory(self, sample_size):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        
        if self.memory_x is None or self.memory_y is None:
            raise ValueError('Empty memory for AGEM.')

        if self.memory_x.size(0) <= sample_size:
            return self.memory_x, self.memory_y
        else:
            idxs = random.sample(range(self.memory_x.size(0)), sample_size)
            return self.memory_x[idxs], self.memory_y[idxs]

    @torch.no_grad()
    def update_memory(self, dataloader):
        """
        Update replay memory with patterns from current step.
        """

        tot = 0
        for x, y in dataloader:
            if tot + x.size(0) <= self.patterns_per_step:
                if self.memory_x is None:
                    self.memory_x = x.clone()
                    self.memory_y = y.clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y), dim=0)
            else:
                diff = self.patterns_per_step - tot
                if self.memory_x is None:
                    self.memory_x = x[:diff].clone()
                    self.memory_y = y[:diff].clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x[:diff]), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y[:diff]), dim=0)
                break
            tot += x.size(0)


class GEMPlugin(StrategyPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous steps. The gradient on the current
    minibatch is projected so that the dot product with all the reference
    gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_step: int, memory_strength: float):
        """
        :param patterns_per_step: number of patterns per step in the memory
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_step = int(patterns_per_step)
        self.memory_strength = memory_strength

        self.memory_x, self.memory_y = {}, {}

        self.G = None

        self.step_id = 0

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all steps.
        """

        if self.step_id > 0:
            G = []
            strategy.model.train()
            for t in range(self.step_id):
                strategy.optimizer.zero_grad()
                xref = self.memory_x[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                out = strategy.model(xref)
                loss = strategy.criterion(out, yref)
                loss.backward()

                G.append(torch.cat([p.grad.flatten()
                         for p in strategy.model.parameters()
                         if p.grad is not None], dim=0))

            self.G = torch.stack(G)  # (steps, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if self.step_id > 0:
            g = torch.cat([p.grad.flatten()
                          for p in strategy.model.parameters()
                          if p.grad is not None], dim=0)

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)
        
            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():

                curr_pars = p.numel()

                if p.grad is None:
                    continue

                p.grad.copy_(v_star[num_pars:num_pars+curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_step(self, strategy, **kwargs):
        """
        Save a copy of the model after each step
        """

        self.update_memory(strategy.current_dataloader)
        self.step_id += 1

    @torch.no_grad()
    def update_memory(self, dataloader):
        """
        Update replay memory with patterns from current step.
        """

        t = self.step_id

        tot = 0
        for x, y in dataloader:
            if tot + x.size(0) <= self.patterns_per_step:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
            else:
                diff = self.patterns_per_step - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]),
                                                 dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]),
                                                 dim=0)
                break
            tot += x.size(0)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and 
        gradients matrix on previous tasks G.
        Taken from original code: 
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np
        
        return torch.from_numpy(v_star).float()


class EWCPlugin(StrategyPlugin):
    """
    Elastic Weight Consolidation (EWC) plugin.
    EWC computes importance of each weight at the end of training on current
    step. During training on each minibatch, the loss is augmented
    with a penalty which keeps the value of the current weights close to the
    value they had on previous steps in proportion to their importance on that
    step. Importances are computed with an additional pass on the training set.
    This plugin does not use task identities.
    """

    def __init__(self, ewc_lambda, mode='separate', decay_factor=None,
                 keep_importance_data=False):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous 
               step. 
               `online` to keep a single penalty summed with a decay factor 
               over all previous tasks.
        :param decay_factor: used only if mode is `online`. 
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """

        super().__init__()

        assert mode == 'separate' or mode == 'online', \
            'Mode must be separate or online.'

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor
        self.step_id = 0

        if self.mode == 'separate':
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """

        if self.step_id == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)
        
        if self.mode == 'separate':
            for step in range(self.step_id):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                            strategy.model.named_parameters(),
                            self.saved_params[step],
                            self.importances[step]):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == 'online':
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[self.step_id],
                        self.importances[self.step_id]):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError('Wrong EWC mode.')

        strategy.loss += self.ewc_lambda * penalty

    def after_training_step(self, strategy, **kwargs):
        """
        Compute importances of parameters after each step.
        """

        importances = self.compute_importances(strategy.model,
                                               strategy.criterion,
                                               strategy.optimizer,
                                               strategy.current_dataloader,
                                               strategy.device)
        self.update_importances(importances)
        self.saved_params[self.step_id] = copy_params_dict(strategy.model)
        # clear previuos parameter values
        if self.step_id > 0 and (not self.keep_importance_data):
            del self.saved_params[self.step_id - 1]
        self.step_id += 1
    
    def compute_importances(self, model, criterion, optimizer,
                            dataloader, device):
        """
        Compute EWC importance matrix for each parameter
        """

        model.train()

        # list of list
        importances = zerolike_params_dict(model)

        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), 
                                          importances):
                assert(k1 == k2)
                imp += p.grad.data.clone().pow(2)
        
        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == 'separate' or self.step_id == 0:
            self.importances[self.step_id] = importances
        elif self.mode == 'online':
            for (k1, old_imp), (k2, curr_imp) in \
                        zip(self.importances[self.step_id - 1], importances):
                assert k1 == k2, 'Error in importance computation.'
                self.importances[self.step_id].append(
                    (k1, (self.decay_factor * old_imp + curr_imp)))
            
            # clear previous parameter importances
            if self.step_id > 0 and (not self.keep_importance_data):
                del self.importances[self.step_id - 1]

        else:
            raise ValueError("Wrong EWC mode.")


__all__ = ['StrategyPlugin', 'ReplayPlugin', 'GDumbPlugin',
           'EvaluationPlugin', 'CWRStarPlugin', 'MultiHeadPlugin', 'LwFPlugin',
           'AGEMPlugin', 'GEMPlugin', 'EWCPlugin']
