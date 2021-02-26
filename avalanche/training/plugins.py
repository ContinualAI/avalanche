################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 03-12-2020                                                             #
# Author(s): Antonio Carta, Andrea Cossu                                       #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
import copy
import logging
import random
from collections import defaultdict
from fnmatch import fnmatch
from typing import Dict, Any, Union, Sequence, TYPE_CHECKING, Optional, List, \
    Tuple, Set

import numpy as np
import quadprog
import torch
import warnings

from torch import Tensor
from torch.nn import Module, Linear
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.data import random_split, ConcatDataset, TensorDataset, \
    DataLoader

from avalanche.benchmarks.scenarios import IExperience
from avalanche.training.strategy_callbacks import StrategyCallbacks
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, \
    get_layers_and_params, freeze_everything, get_last_fc_layer, \
    get_layer_by_name, unfreeze_everything, examples_per_class

if TYPE_CHECKING:
    from avalanche.logging import StrategyLogger
    from avalanche.evaluation import PluginMetric
    from avalanche.training.strategies import BaseStrategy, JointTraining

PluggableStrategy = Union['BaseStrategy', 'JointTraining']


class StrategyPlugin(StrategyCallbacks[Any]):
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses must override
    the callbacks.
    """

    def __init__(self):
        super().__init__()
        pass

    def before_training(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_training_step(self, strategy: PluggableStrategy, **kwargs):
        pass

    def adapt_train_dataset(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_backward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_backward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_update(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_update(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training_step(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_training(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval(self, strategy: PluggableStrategy, **kwargs):
        pass

    def adapt_eval_dataset(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval_step(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval_step(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass

    def before_eval_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval_forward(self, strategy: PluggableStrategy, **kwargs):
        pass

    def after_eval_iteration(self, strategy: PluggableStrategy, **kwargs):
        pass


class ReplayPlugin(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implements the "adapt_train_dataset" callback to add them to
    the training set.

    The :mem_size: attribute controls the number of patterns to be stored in 
    the external memory. In multitask scenarios, mem_size is the memory size
    for each task. We assume the training set contains at least :mem_size: data
    points.
    """

    def __init__(self, mem_size=200):
        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.rm_add = None

    def adapt_train_dataset(self, strategy, **kwargs):
        """
        Expands the current training set with datapoint from
        the external memory before training.
        """
        curr_data = strategy.step_info.dataset

        # Additional set of the current batch to be concatenated to the ext.
        # memory at the end of the training
        self.rm_add = None

        # how many patterns to save for next iter
        h = min(self.mem_size // (strategy.training_step_counter + 1),
                len(curr_data))

        # We recover it using the random_split method and getting rid of the
        # second split.
        self.rm_add, _ = random_split(
            curr_data, [h, len(curr_data) - h]
        )

        if strategy.training_step_counter > 0:
            # We update the train_dataset concatenating the external memory.
            # We assume the user will shuffle the data when creating the data
            # loader.
            for mem_task_id in self.ext_mem.keys():
                mem_data = self.ext_mem[mem_task_id]
                if mem_task_id in strategy.adapted_dataset:
                    cat_data = ConcatDataset([curr_data, mem_data])
                    strategy.adapted_dataset[mem_task_id] = cat_data
                else:
                    strategy.adapted_dataset[mem_task_id] = mem_data

    def after_training_step(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of
         the current training batch/task. """
        curr_task_id = strategy.step_info.task_label

        # replace patterns in random memory
        ext_mem = self.ext_mem
        if curr_task_id not in ext_mem:
            ext_mem[curr_task_id] = self.rm_add
        else:
            rem_len = len(ext_mem[curr_task_id]) - len(self.rm_add)
            _, saved_part = random_split(ext_mem[curr_task_id],
                                         [len(self.rm_add), rem_len]
                                         )
            ext_mem[curr_task_id] = ConcatDataset([saved_part, self.rm_add])
        self.ext_mem = ext_mem


class GDumbPlugin(StrategyPlugin):
    """
    A GDumb plugin. At each step the model
    is trained with all and only the data of the external memory.
    The memory is updated at the end of each step to add new classes or
    new examples of already encountered classes.
    In multitask scenarios, mem_size is the memory size for each task.

    This plugin can be combined with a Naive strategy to obtain the
    standard GDumb strategy.

    https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf
    """

    def __init__(self, mem_size=200):

        super().__init__()

        self.mem_size = mem_size
        self.ext_mem = defaultdict(lambda: None)
        # count occurrences for each class
        self.counter = defaultdict(lambda: defaultdict(int))

    def adapt_train_dataset(self, strategy, **kwargs):
        """ Before training we make sure to organize the memory following
            GDumb approach and updating the dataset accordingly.
        """

        # for each pattern, add it to the memory or not
        dataset = strategy.step_info.dataset
        current_counter = self.counter[strategy.step_info.task_label]
        current_mem = self.ext_mem[strategy.step_info.task_label]
        for i, (pattern, target_value, _) in enumerate(dataset):
            target = torch.tensor(target_value)
            if len(pattern.size()) == 1:
                pattern = pattern.unsqueeze(0)

            if current_counter == {}:
                # any positive (>0) number is ok
                patterns_per_class = 1
            else:
                patterns_per_class = int(
                    self.mem_size / len(current_counter.keys())
                )

            if target_value not in current_counter or \
                    current_counter[target_value] < patterns_per_class:
                # full memory: replace item from most represented class
                # with current pattern
                if sum(current_counter.values()) >= self.mem_size:
                    to_remove = max(current_counter, key=current_counter.get)
                    for j in range(len(current_mem.tensors[1])):
                        if current_mem.tensors[1][j].item() == to_remove:
                            current_mem.tensors[0][j] = pattern
                            current_mem.tensors[1][j] = target
                            break
                    current_counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    if current_mem is None:
                        current_mem = TensorDataset(
                            pattern, target.unsqueeze(0))
                    else:
                        current_mem = TensorDataset(
                            torch.cat([
                                pattern,
                                current_mem.tensors[0]], dim=0),

                            torch.cat([
                                target.unsqueeze(0),
                                current_mem.tensors[1]], dim=0)
                        )

                current_counter[target_value] += 1

        self.ext_mem[strategy.step_info.task_label] = current_mem
        strategy.adapted_dataset = self.ext_mem


class EvaluationPlugin(StrategyPlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and eval loops of the strategy through callbacks.

    This plugin updates the given metrics and logs them using the provided
    loggers.
    """

    def __init__(self,
                 *metrics: Union['PluginMetric', Sequence['PluginMetric']],
                 loggers: Union['StrategyLogger',
                                Sequence['StrategyLogger']] = None,
                 collect_all=True):
        """
        Creates an instance of the evaluation plugin.

        :param metrics: The metrics to compute.
        :param loggers: The loggers to be used to log the metric values.
        :param collect_curves (bool): enables the collection of the metric
            curves. If True `self.metric_curves` stores all the values of
            each curve in a dictionary. Please disable this if you log large
            values (embeddings, parameters) and you want to reduce memory usage.
        """
        super().__init__()
        self.collect_all = collect_all
        flat_metrics_list = []
        for metric in metrics:
            if isinstance(metric, Sequence):
                flat_metrics_list += list(metric)
            else:
                flat_metrics_list.append(metric)
        self.metrics = flat_metrics_list

        if loggers is None:
            loggers = []
        elif not isinstance(loggers, Sequence):
            loggers = [loggers]
        self.loggers: Sequence['StrategyLogger'] = loggers

        if len(self.loggers) == 0:
            warnings.warn('No loggers specified, metrics will not be logged')

        # for each curve  store last emitted value (train/eval separated).
        self.current_metrics = {}
        if self.collect_all:
            # for each curve collect all emitted values.
            self.all_metrics = defaultdict(lambda: ([], []))
        else:
            self.all_metrics = None

    def _update_metrics(self, strategy: PluggableStrategy, callback: str):
        metric_values = []
        for metric in self.metrics:
            metric_result = getattr(metric, callback)(strategy)
            if isinstance(metric_result, Sequence):
                metric_values += list(metric_result)
            elif metric_result is not None:
                metric_values.append(metric_result)

        for metric_value in metric_values:
            name = metric_value.name
            x = metric_value.x_plot
            val = metric_value.value
            self.current_metrics[name] = val
            if self.collect_all:
                self.all_metrics[name][0].append(x)
                self.all_metrics[name][1].append(val)

        for logger in self.loggers:
            getattr(logger, callback)(strategy, metric_values)
        return metric_values

    def before_training(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training')

    def before_training_step(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training_step')

    def adapt_train_dataset(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'adapt_train_dataset')

    def before_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training_epoch')

    def before_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_training_iteration')

    def before_forward(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_forward')

    def after_forward(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_forward')

    def before_backward(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_backward')

    def after_backward(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_backward')

    def after_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training_iteration')

    def before_update(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_update')

    def after_update(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_update')

    def after_training_epoch(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training_epoch')

    def after_training_step(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training_step')

    def after_training(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_training')
        self.current_metrics = {}  # reset current metrics

    def before_eval(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval')

    def adapt_eval_dataset(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'adapt_eval_dataset')

    def before_eval_step(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval_step')

    def after_eval_step(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval_step')

    def after_eval(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval')
        self.current_metrics = {}  # reset current metrics

    def before_eval_iteration(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval_iteration')

    def before_eval_forward(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'before_eval_forward')

    def after_eval_forward(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval_forward')

    def after_eval_iteration(self, strategy: PluggableStrategy, **kwargs):
        self._update_metrics(strategy, 'after_eval_iteration')


class CWRStarPlugin(StrategyPlugin):

    def __init__(self, model, cwr_layer_name=None, freeze_remaining_model=True):
        """
        CWR* Strategy.
        This plugin does not use task identities.

        :param model: the model.
        :param cwr_layer_name: name of the last fully connected layer. Defaults
            to None, which means that the plugin will attempt an automatic
            detection.
        :param freeze_remaining_model: If True, the plugin will freeze (set
            layers in eval mode and disable autograd for parameters) all the
            model except the cwr layer. Defaults to True.
        """
        super().__init__()
        self.log = logging.getLogger("avalanche")
        self.model = model
        self.cwr_layer_name = cwr_layer_name
        self.freeze_remaining_model = freeze_remaining_model

        # Model setup
        self.model.saved_weights = {}
        self.model.past_j = defaultdict(int)
        self.model.cur_j = defaultdict(int)

        # to be updated
        self.cur_class = None

    def after_training_step(self, strategy, **kwargs):
        self.consolidate_weights()
        self.set_consolidate_weights()

    def before_training_step(self, strategy, **kwargs):
        if self.freeze_remaining_model and strategy.training_step_counter > 0:
            self.freeze_other_layers()

        # Count current classes and number of samples for each of them.
        data = strategy.step_info.dataset
        self.model.cur_j = examples_per_class(data.targets)
        self.cur_class = [cls for cls in set(self.model.cur_j.keys()) if
                          self.model.cur_j[cls] > 0]

        self.reset_weights(self.cur_class)

    def consolidate_weights(self):
        """ Mean-shift for the target layer weights"""

        with torch.no_grad():
            cwr_layer = self.get_cwr_layer()
            globavg = np.average(cwr_layer.weight.detach()
                                 .cpu().numpy()[self.cur_class])
            for c in self.cur_class:
                w = cwr_layer.weight.detach().cpu().numpy()[c]

                if c in self.cur_class:
                    new_w = w - globavg
                    if c in self.model.saved_weights.keys():
                        wpast_j = np.sqrt(self.model.past_j[c] /
                                          self.model.cur_j[c])
                        # wpast_j = model.past_j[c] / model.cur_j[c]
                        self.model.saved_weights[c] = \
                            (self.model.saved_weights[c] * wpast_j + new_w) / \
                            (wpast_j + 1)
                    else:
                        self.model.saved_weights[c] = new_w

    def set_consolidate_weights(self):
        """ set trained weights """

        with torch.no_grad():
            cwr_layer = self.get_cwr_layer()
            for c, w in self.model.saved_weights.items():
                cwr_layer.weight[c].copy_(
                    torch.from_numpy(self.model.saved_weights[c])
                )

    def reset_weights(self, cur_clas):
        """ reset weights"""
        with torch.no_grad():
            cwr_layer = self.get_cwr_layer()
            cwr_layer.weight.fill_(0.0)
            for c, w in self.model.saved_weights.items():
                if c in cur_clas:
                    cwr_layer.weight[c].copy_(
                        torch.from_numpy(self.model.saved_weights[c])
                    )

    def get_cwr_layer(self) -> Optional[Linear]:
        result = None
        if self.cwr_layer_name is None:
            last_fc = get_last_fc_layer(self.model)
            if last_fc is not None:
                result = last_fc[1]
        else:
            result = get_layer_by_name(self.model, self.cwr_layer_name)

        return result

    def freeze_other_layers(self):
        cwr_layer = self.get_cwr_layer()
        if cwr_layer is None:
            raise RuntimeError('Can\'t find a the Linear layer')
        freeze_everything(self.model)
        unfreeze_everything(cwr_layer)


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

    def before_eval_iteration(self, strategy, **kwargs):
        self._optimizer = strategy.optimizer
        self.set_task_layer(strategy, strategy.step_info)

    @torch.no_grad()
    def set_task_layer(self, strategy, step_info: IExperience):
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
        alpha = self.alpha[strategy.training_step_counter] \
            if isinstance(self.alpha, (list, tuple)) else self.alpha
        penalty = self.penalty(strategy.logits, strategy.mb_x, alpha)
        strategy.loss += penalty

    def after_training_step(self, strategy, **kwargs):
        """
        Save a copy of the model after each step
        """

        self.prev_model = copy.deepcopy(strategy.model)


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
                assert (p1.grad is not None and refg is not None) or \
                       (p1.grad is None and refg is None)

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
        for batches in dataloader:
            for _, (x, y) in batches.items():
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
                        self.memory_x = torch.cat((self.memory_x,
                                                   x[:diff]), dim=0)
                        self.memory_y = torch.cat((self.memory_y,
                                                   y[:diff]), dim=0)
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

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all steps.
        """

        if strategy.training_step_counter > 0:
            G = []
            strategy.model.train()
            for t in range(strategy.training_step_counter):
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

        if strategy.training_step_counter > 0:
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

                p.grad.copy_(
                    v_star[num_pars:num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_step(self, strategy, **kwargs):
        """
        Save a copy of the model after each step
        """

        self.update_memory(strategy.step_info.dataset,
                           strategy.training_step_counter,
                           strategy.train_mb_size)

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current step.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size)
        tot = 0
        for x, y, _ in dataloader:
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

        if strategy.training_step_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == 'separate':
            for step in range(strategy.training_step_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[step],
                        self.importances[step]):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == 'online':
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[strategy.training_step_counter],
                    self.importances[strategy.training_step_counter]):
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
                                               strategy.step_info.dataset,
                                               strategy.device,
                                               strategy.train_mb_size)
        self.update_importances(importances, strategy.training_step_counter)
        self.saved_params[strategy.training_step_counter] = \
            copy_params_dict(strategy.model)
        # clear previuos parameter values
        if strategy.training_step_counter > 0 and \
                (not self.keep_importance_data):
            del self.saved_params[strategy.training_step_counter - 1]

    def compute_importances(self, model, criterion, optimizer,
                            dataset, device, batch_size):
        """
        Compute EWC importance matrix for each parameter
        """

        model.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, (x, y, _) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(),
                                          importances):
                assert (k1 == k2)
                imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == 'separate' or t == 0:
            self.importances[t] = importances
        elif self.mode == 'online':
            for (k1, old_imp), (k2, curr_imp) in \
                    zip(self.importances[t - 1], importances):
                assert k1 == k2, 'Error in importance computation.'
                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp)))

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")


ParamDict = Dict[str, Tensor]
EwcDataType = Tuple[ParamDict, ParamDict]
SynDataType = Dict[str, Dict[str, Tensor]]


class SynapticIntelligencePlugin(StrategyPlugin):
    """
    The Synaptic Intelligence plugin.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper "Continual Learning Through Synaptic
    Intelligence" (https://arxiv.org/abs/1703.04200).

    This plugin can be attached to existing strategies to achieve a
    regularization effect.

    This plugin will require the strategy `loss` field to be set before the
    `before_backward` callback is invoked. The loss Tensor will be updated to
    achieve the S.I. regularization effect.
    """

    def __init__(self, si_lambda: float,
                 excluded_parameters: Sequence['str'] = None,
                 device: Any = 'as_strategy'):
        """
        Creates an instance of the Synaptic Intelligence plugin.

        :param si_lambda: Synaptic Intelligence lambda term.
        :param device: The device to use to run the S.I. steps. Defaults to
            "as_strategy", which means that the `device` field of the strategy
            will be used. Using a different device may lead to a performance
            drop due to the required data transfer.
        """

        super().__init__()

        warnings.warn("The Synaptic Intelligence plugin is in an alpha stage "
                      "and is not perfectly aligned with the paper "
                      "implementation. Please use at your own risk!")

        if excluded_parameters is None:
            excluded_parameters = []
        self.si_lambda: float = si_lambda
        self.excluded_parameters: Set[str] = set(excluded_parameters)
        self.ewc_data: EwcDataType = (dict(), dict())
        """
        The first dictionary contains the params at loss minimum while the 
        second one contains the parameter importance.
        """

        self.syn_data: SynDataType = {
            'old_theta': dict(),
            'new_theta': dict(),
            'grad': dict(),
            'trajectory': dict(),
            'cum_trajectory': dict()}

        self._device = device

    def before_training_step(self, strategy: PluggableStrategy, **kwargs):
        super().before_training_step(strategy, **kwargs)
        SynapticIntelligencePlugin.create_syn_data(
            strategy.model, self.ewc_data, self.syn_data,
            self.excluded_parameters)

        SynapticIntelligencePlugin.init_batch(
            strategy.model, self.ewc_data, self.syn_data,
            self.excluded_parameters)

    def after_backward(self, strategy: PluggableStrategy, **kwargs):
        super().after_backward(strategy, **kwargs)
        syn_loss = SynapticIntelligencePlugin.compute_ewc_loss(
            strategy.model, self.ewc_data, self.excluded_parameters,
            lambd=self.si_lambda, device=self.device(strategy))

        if syn_loss is not None:
            strategy.loss += syn_loss.to(strategy.device)

    def before_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        super().before_training_iteration(strategy, **kwargs)
        SynapticIntelligencePlugin.pre_update(strategy.model, self.syn_data,
                                              self.excluded_parameters)

    def after_training_iteration(self, strategy: PluggableStrategy, **kwargs):
        super().after_training_iteration(strategy, **kwargs)
        SynapticIntelligencePlugin.post_update(strategy.model, self.syn_data,
                                               self.excluded_parameters)

    def after_training_step(self, strategy: PluggableStrategy, **kwargs):
        super().after_training_step(strategy, **kwargs)
        SynapticIntelligencePlugin.update_ewc_data(
            strategy.model, self.ewc_data, self.syn_data, 0.001,
            self.excluded_parameters, 1)

    def device(self, strategy: PluggableStrategy):
        if self._device == 'as_strategy':
            return strategy.device

        return self._device

    @staticmethod
    @torch.no_grad()
    def create_syn_data(model: Module, ewc_data: EwcDataType,
                        syn_data: SynDataType, excluded_parameters: Set[str]):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters)

        for param_name, param in params:
            if param_name in ewc_data[0]:
                continue

            # Handles added parameters (doesn't manage parameter expansion!)
            ewc_data[0][param_name] = SynapticIntelligencePlugin._zero(param)
            ewc_data[1][param_name] = SynapticIntelligencePlugin._zero(param)

            syn_data['old_theta'][param_name] = \
                SynapticIntelligencePlugin._zero(param)
            syn_data['new_theta'][param_name] = \
                SynapticIntelligencePlugin._zero(param)
            syn_data['grad'][param_name] = \
                SynapticIntelligencePlugin._zero(param)
            syn_data['trajectory'][param_name] = \
                SynapticIntelligencePlugin._zero(param)
            syn_data['cum_trajectory'][param_name] = \
                SynapticIntelligencePlugin._zero(param)

    @staticmethod
    @torch.no_grad()
    def _zero(param: Tensor):
        return torch.zeros(param.numel(), dtype=param.dtype)

    @staticmethod
    @torch.no_grad()
    def extract_weights(model: Module, target: ParamDict,
                        excluded_parameters: Set[str]):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters)

        for name, param in params:
            target[name][...] = param.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def extract_grad(model, target: ParamDict, excluded_parameters: Set[str]):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters)

        # Store the gradients into target
        for name, param in params:
            target[name][...] = param.grad.detach().cpu().flatten()

    @staticmethod
    @torch.no_grad()
    def init_batch(model, ewc_data: EwcDataType, syn_data: SynDataType,
                   excluded_parameters: Set[str]):
        # Keep initial weights
        SynapticIntelligencePlugin. \
            extract_weights(model, ewc_data[0], excluded_parameters)
        for param_name, param_trajectory in syn_data['trajectory'].items():
            param_trajectory.fill_(0.0)

    @staticmethod
    @torch.no_grad()
    def pre_update(model, syn_data: SynDataType,
                   excluded_parameters: Set[str]):
        SynapticIntelligencePlugin.extract_weights(model, syn_data['old_theta'],
                                                   excluded_parameters)

    @staticmethod
    @torch.no_grad()
    def post_update(model, syn_data: SynDataType,
                    excluded_parameters: Set[str]):
        SynapticIntelligencePlugin.extract_weights(model, syn_data['new_theta'],
                                                   excluded_parameters)
        SynapticIntelligencePlugin.extract_grad(model, syn_data['grad'],
                                                excluded_parameters)

        for param_name in syn_data['trajectory']:
            syn_data['trajectory'][param_name] += \
                syn_data['grad'][param_name] * (
                        syn_data['new_theta'][param_name] -
                        syn_data['old_theta'][param_name])

    @staticmethod
    def compute_ewc_loss(model, ewc_data: EwcDataType,
                         excluded_parameters: Set[str], device, lambd=0.0):
        params = SynapticIntelligencePlugin.allowed_parameters(
            model, excluded_parameters)

        loss = None
        for name, param in params:
            weights = param.to(device).flatten()  # Flat, not detached
            param_ewc_data_0 = ewc_data[0][name].to(device)  # Flat, detached
            param_ewc_data_1 = ewc_data[1][name].to(device)  # Flat, detached

            syn_loss: Tensor = torch.dot(
                param_ewc_data_1,
                (weights - param_ewc_data_0) ** 2) * (lambd / 2)

            if loss is None:
                loss = syn_loss
            else:
                loss += syn_loss

        return loss

    @staticmethod
    @torch.no_grad()
    def update_ewc_data(net, ewc_data: EwcDataType, syn_data: SynDataType,
                        clip_to: float, excluded_parameters: Set[str],
                        c=0.0015):
        SynapticIntelligencePlugin.extract_weights(net, syn_data['new_theta'],
                                                   excluded_parameters)
        eps = 0.0000001  # 0.001 in few task - 0.1 used in a more complex setup

        for param_name in syn_data['cum_trajectory']:
            syn_data['cum_trajectory'][param_name] += \
                c * syn_data['trajectory'][param_name] / (
                        np.square(syn_data['new_theta'][param_name] -
                                  ewc_data[0][param_name]) + eps)

        for param_name in syn_data['cum_trajectory']:
            ewc_data[1][param_name] = torch.empty_like(
                syn_data['cum_trajectory'][param_name]).copy_(
                -syn_data['cum_trajectory'][param_name])

        # change sign here because the Ewc regularization
        # in Caffe (theta - thetaold) is inverted w.r.t. syn equation [4]
        # (thetaold - theta)
        for param_name in ewc_data[1]:
            ewc_data[1][param_name] = torch.clamp(
                ewc_data[1][param_name], max=clip_to)
            ewc_data[0][param_name] = \
                syn_data['new_theta'][param_name].clone()

    @staticmethod
    def explode_excluded_parameters(excluded: Set[str]) -> Set[str]:
        """
        Explodes a list of excluded parameters by adding a generic final ".*"
        wildcard at its end.

        :param excluded: The original set of excluded parameters.

        :return: The set of excluded parameters in which ".*" patterns have been
            added.
        """
        result = set()
        for x in excluded:
            result.add(x)
            if not x.endswith('*'):
                result.add(x + '.*')
        return result

    @staticmethod
    def not_excluded_parameters(model: Module, excluded_parameters: Set[str]) \
            -> List[Tuple[str, Tensor]]:
        # Add wildcards ".*" to all excluded parameter names
        result = []
        excluded_parameters = SynapticIntelligencePlugin. \
            explode_excluded_parameters(excluded_parameters)
        layers_params = get_layers_and_params(model)

        for lp in layers_params:
            if isinstance(lp.layer, _NormBase):
                # Exclude batch norm parameters
                excluded_parameters.add(lp.parameter_name)

        for name, param in model.named_parameters():
            accepted = True
            for exclusion_pattern in excluded_parameters:
                if fnmatch(name, exclusion_pattern):
                    accepted = False
                    break

            if accepted:
                result.append((name, param))

        return result

    @staticmethod
    def allowed_parameters(model: Module, excluded_parameters: Set[str]) \
            -> List[Tuple[str, Tensor]]:

        allow_list = SynapticIntelligencePlugin.not_excluded_parameters(
            model, excluded_parameters)

        result = []
        for name, param in allow_list:
            if param.requires_grad:
                result.append((name, param))

        return result


__all__ = [
    'PluggableStrategy',
    'StrategyPlugin',
    'ReplayPlugin',
    'GDumbPlugin',
    'EvaluationPlugin',
    'CWRStarPlugin',
    'MultiHeadPlugin',
    'LwFPlugin',
    'AGEMPlugin',
    'GEMPlugin',
    'EWCPlugin',
    'SynapticIntelligencePlugin'
]
