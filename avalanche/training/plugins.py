import warnings
from typing import Optional
import copy
from collections import defaultdict

import torch
from torch import Tensor
from torch.utils.data import random_split, ConcatDataset, TensorDataset

from avalanche.benchmarks.scenarios import IStepInfo
from avalanche.evaluation.metrics import ACC
from avalanche.evaluation import EvalProtocol


class StrategyPlugin:
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
                # full memory, remove item from most represented class
                if sum(self.counter.values()) >= self.mem_size:
                    to_remove = max(self.counter, key=self.counter.get)
                    for j in range(len(self.ext_mem.tensors[1])):
                        if self.ext_mem.tensors[1][j].item() == to_remove:
                            self.ext_mem.tensors[0][j] = pattern
                            self.ext_mem.tensors[1][j] = target
                            break
                    self.counter[to_remove] -= 1

                # memory not full, just add the new pattern
                else:
                    ext_mem[0].append(pattern)
                    ext_mem[1].append(target.unsqueeze(0))

                self.counter[target_value] += 1

        # concatenate previous memory with newly added patterns.
        # when ext_mem[0] == [] the memory (self.ext_mem) is full and patterns
        # are inserted into self.ext_mem directly
        if len(ext_mem[0]) > 0:
            memx = torch.cat(ext_mem[0], dim=0)
            memy = torch.cat(ext_mem[1], dim=0)
            if self.ext_mem is None:
                self.ext_mem = TensorDataset(memx, memy)
            else:
                self.ext_mem = TensorDataset(
                    torch.cat([memx, self.ext_mem.tensors[0]], dim=0),
                    torch.cat([memy, self.ext_mem.tensors[1]], dim=0)
                )


class EvaluationPlugin(StrategyPlugin):
    """
    An evaluation plugin that obtains relevant data from the
    training and testing loops of the strategy through callbacks.

    Internally, the evaluation plugin tries to use the "evaluation_protocol"
    namespace value. If found and not None, the evaluation protocol
    (usually an instance of :class:`EvalProtocol`), is used to compute the
    required metrics. The "evaluation_protocol" is usually a field of the main
    strategy.
    """
    def __init__(self):
        super().__init__()

        # Training
        self._training_dataset_size = None
        self._training_accuracy = None
        self._training_correct_count = 0
        self._training_average_loss = 0
        self._training_total_iterations = 0
        self._train_current_task_id = None

        # Testing
        self._test_dataset_size = None
        self._test_average_loss = 0
        self._test_total_iterations = 0
        self._test_current_task_id = None
        self._test_true_y = None
        self._test_predicted_y = None
        self._test_protocol_results = None

    def get_train_result(self):
        return self._training_average_loss, self._training_accuracy

    def get_test_result(self):
        return self._test_protocol_results

    def before_training(self, strategy, **kwargs):
        step_info = strategy.step_info
        _, task_id = step_info.current_training_set()
        self._training_dataset_size = None
        self._train_current_task_id = task_id
        self._training_accuracy = None
        self._training_correct_count = 0
        self._training_average_loss = 0
        self._training_dataset_size = len(strategy.current_data)

    def after_training_iteration(self, strategy, **kwargs):
        self._training_total_iterations += 1
        evaluation_protocol = strategy.evaluation_protocol
        epoch = strategy.epoch
        iteration = strategy.mb_it
        train_mb_y = strategy.mb_y
        logits = strategy.logits
        loss = strategy.loss

        _, predicted_labels = torch.max(logits, 1)
        correct_predictions = torch.eq(predicted_labels,
                                       train_mb_y).sum().item()
        self._training_correct_count += correct_predictions

        torch.eq(predicted_labels, train_mb_y)

        self._training_average_loss += loss.item()
        den = ((iteration + 1) * train_mb_y.shape[0] +
               epoch * self._training_dataset_size)
        self._training_average_loss /= den

        self._training_accuracy = self._training_correct_count / den

        if iteration % 100 == 0:
            print(
                '[Training] ==>>> it: {}, avg. loss: {:.6f}, '
                'running train acc: {:.3f}'.format(
                    iteration, self._training_average_loss,
                    self._training_accuracy))

            evaluation_protocol.update_tb_train(
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
        self._test_total_iterations += 1

        _, predicted_labels = torch.max(strategy.logits, 1)
        self._test_true_y.append(strategy.mb_y)
        self._test_predicted_y.append(predicted_labels)
        self._test_average_loss += strategy.loss.item()

    def after_test_step(self, strategy, **kwargs):
        evaluation_protocol = strategy.evaluation_protocol
        self._test_average_loss /= self._test_dataset_size

        results = evaluation_protocol.get_results(
            self._test_true_y, self._test_predicted_y,
            self._train_current_task_id, self._test_current_task_id)
        acc, accs = results[ACC]

        print("[Evaluation] Task {0}: Avg Loss {1}; Avg Acc {2}"
              .format(self._test_current_task_id, self._test_average_loss, acc))

        self._test_protocol_results[self._test_current_task_id] = \
            (self._test_average_loss, acc, accs, results)

    def after_test(self, strategy, **kwargs):
        strategy.evaluation_protocol.update_tb_test(
            self._test_protocol_results,
            strategy.step_info.current_step)
