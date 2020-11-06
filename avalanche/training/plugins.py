import copy
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.nn import Module, Linear
from torch.utils.data import random_split, ConcatDataset, TensorDataset

from avalanche.benchmarks.scenarios import IStepInfo
from avalanche.evaluation.metrics import ACC


class StrategyPlugin:
    """
    Base class for strategy plugins. Implements all the callbacks required
    by the BaseStrategy with an empty function. Subclasses must override
    the callbacks.
    """
    def __init__(self):
        pass

    def before_training(self, strategy, **kwargs):
        pass

    def adapt_train_dataset(self, strategy, **kwargs):
        pass

    def before_training_epoch(self, strategy, **kwargs):
        pass

    def before_training_iteration(self, strategy, **kwargs):
        pass

    def before_forward(self, strategy, **kwargs):
        pass

    def after_forward(self, strategy, **kwargs):
        pass

    def before_backward(self, strategy, **kwargs):
        pass

    def after_backward(self, strategy, **kwargs):
        pass

    def after_training_iteration(self, strategy, **kwargs):
        pass

    def before_update(self, strategy, **kwargs):
        pass

    def after_update(self, strategy, **kwargs):
        pass

    def after_training_epoch(self, strategy, **kwargs):
        pass

    def after_training(self, strategy, **kwargs):
        pass

    def before_test(self, strategy, **kwargs):
        pass

    def adapt_test_dataset(self, strategy, **kwargs):
        pass

    def before_test_step(self, strategy, **kwargs):
        pass

    def after_test_step(self, strategy, **kwargs):
        pass

    def after_test(self, strategy, **kwargs):
        pass

    def before_test_iteration(self, strategy, **kwargs):
        pass

    def before_test_forward(self, strategy, **kwargs):
        pass

    def after_test_forward(self, strategy, **kwargs):
        pass

    def after_test_iteration(self, strategy, **kwargs):
        pass


class ReplayPlugin(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implements the "adapt_train_dataset" callback to add them to
    the training set.

    The :mem_size: attribute controls the number of patterns to be stored in the
    external memory. We assume the training set contains at least :mem_size:
    data points.
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

    def after_training(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of
         the current training batch/task. """

        # replace patterns in random memory
        ext_mem = self.ext_mem
        if self.it == 0:
            ext_mem = copy.deepcopy(self.rm_add)
        else:
            _, saved_part = random_split(
                ext_mem, [len(self.rm_add), len(ext_mem)-len(self.rm_add)]
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

        # helper memory to store new patterns when memory is not full
        # this is necessary since it is not possible to concatenate
        # patterns into an existing TensorDataset
        # (dataset.tensors[0] does not support item assignment)
        ext_mem = [[], []]

        # for each pattern, add it to the memory or not
        for i, (pattern, target) in enumerate(strategy.current_data):
            target_value = target.item()

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


class EvaluationPlugin(StrategyPlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and testing loops of the strategy through callbacks.

    Internally, the evaluation plugin tries uses the "evaluation_protocol"
    (an instance of :class:`EvalProtocol`), to compute the
    required metrics. The "evaluation_protocol" is usually passed as argument
    from the strategy.
    """
    def __init__(self, evaluation_protocol):
        super().__init__()
        self.evaluation_protocol = evaluation_protocol

        # Training
        self._training_dataset_size = None
        self._training_accuracy = None
        self._training_correct_count = 0
        self._training_average_loss = 0
        self._training_total_iterations = 0
        self._train_current_task_id = None

        # Test
        self._test_dataset_size = None
        self._test_average_loss = 0
        self._test_current_task_id = None
        self._test_true_y = None
        self._test_predicted_y = None
        self._test_protocol_results = None

    def get_train_result(self):
        return self._training_average_loss, self._training_accuracy

    def get_test_result(self):
        return self._test_protocol_results

    def before_training(self, strategy, **kwargs):
        _, task_id = strategy.step_info.current_training_set()
        self._training_dataset_size = None
        self._train_current_task_id = task_id
        self._training_accuracy = None
        self._training_correct_count = 0
        self._training_average_loss = 0
        self._training_dataset_size = len(strategy.current_data)

    def after_training_iteration(self, strategy, **kwargs):
        self._training_total_iterations += 1
        epoch = strategy.epoch
        iteration = strategy.mb_it
        train_mb_y = strategy.mb_y
        logits = strategy.logits
        loss = strategy.loss

        # TODO: is this a bug?
        den = ((iteration + 1) * train_mb_y.shape[0] +
               epoch * self._training_dataset_size)

        # Accuracy
        _, predicted_labels = torch.max(logits, 1)
        correct_predictions = torch.eq(predicted_labels, train_mb_y)\
                                   .sum().item()
        self._training_correct_count += correct_predictions
        self._training_accuracy = self._training_correct_count / den

        # Loss
        self._training_average_loss += loss.item()
        self._training_average_loss /= den

        # Logging
        if iteration % 100 == 0:
            print(
                '[Training] ==>>> it: {}, avg. loss: {:.6f}, '
                'running train acc: {:.3f}'.format(
                    iteration, self._training_average_loss,
                    self._training_accuracy))

            self.evaluation_protocol.update_tb_train(
                self._training_average_loss, self._training_accuracy,
                self._training_total_iterations, torch.unique(train_mb_y),
                self._train_current_task_id)

    def before_test_step(self, strategy, **kwargs):
        step_info = strategy.step_info
        step_id = strategy.step_id

        _, task_id = step_info.step_specific_test_set(step_id)
        self._test_protocol_results = dict()
        self._test_dataset_size = len(strategy.current_data)
        self._test_current_task_id = task_id
        self._test_average_loss = 0
        self._test_true_y = []
        self._test_predicted_y = []

    def after_test_iteration(self, strategy, **kwargs):
        _, predicted_labels = torch.max(strategy.logits, 1)
        self._test_true_y.append(strategy.mb_y)
        self._test_predicted_y.append(predicted_labels)
        self._test_average_loss += strategy.loss.item()

    def after_test_step(self, strategy, **kwargs):
        self._test_average_loss /= self._test_dataset_size

        results = self.evaluation_protocol.get_results(
            self._test_true_y, self._test_predicted_y,
            self._train_current_task_id, self._test_current_task_id)
        acc, accs = results[ACC]

        print("[Evaluation] Task {0}: Avg Loss {1}; Avg Acc {2}"
              .format(self._test_current_task_id, self._test_average_loss, acc))

        self._test_protocol_results[self._test_current_task_id] = \
            (self._test_average_loss, acc, accs, results)

    def after_test(self, strategy, **kwargs):
        self.evaluation_protocol.update_tb_test(
            self._test_protocol_results,
            strategy.step_info.current_step)


class CWRStarPlugin(StrategyPlugin):

    def __init__(self, model, second_last_layer_name, num_classes=50):
        """ CWR* Strategy.

        :param model: trained model
        :param second_last_layer_name: name of the second to last layer.
        :param num_classes: total number of classes
        """
        super().__init__()
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

    def after_training(self, strategy, **kwargs):
        CWRStarPlugin.consolidate_weights(self.model, self.cur_class)
        self.batch_processed += 1
        strategy.optimizer

    def before_training(self, strategy, **kwargs):
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
                        model.saved_weights[c] = (model.saved_weights[c] * wpast_j
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
            print("Freezing parameter " + name)
            if name == self.second_last_layer_name:
                break


__all__ = ['StrategyPlugin', 'ReplayPlugin', 'GDumbPlugin',
           'EvaluationPlugin', 'CWRStarPlugin']