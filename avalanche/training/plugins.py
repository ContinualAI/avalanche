import copy
from collections import defaultdict
from typing import Optional, Dict, Any

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


class MultiHeadPlugin(StrategyPlugin):
    def __init__(self, model, classifier_field: str = 'classifier',
                 keep_initial_layer=False):
        """
        MultiHeadPlugin is a plugin that manages a multi-head readout for
        multi-task scenarios. The plugin automatically set the correct
        output head when the task changes and it adds new heads when a novel
        task is encountered. Also works with Single-Incremental-Tasks
        (a.k.a. task-free) to automatically handle the head expansion.

        By default, a Linear (fully connected) layer is created
        with as many output units as the number of classes in that task. This
        behaviour can be changed by overriding the "create_task_layer" method.

        By default, weights are initialized using the Linear class default
        initialization. This behaviour can be changed by overriding the
        "initialize_new_task_layer" method.

        When dealing with a Single-Incremental-Task scenario, the final layer may
        get dynamically expanded. By default, the initialization provided by the
        Linear class is used and then weights of already existing classes are copied
        (that  is, without adapting the weights of new classes). The user can
        control how the new weights are initialized by overriding
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

        if keep_initial_layer:
            self.task_layers[0] = getattr(model, classifier_field)

    def before_training(self, strategy, **kwargs):
        self.set_task_layer(strategy.model, strategy.step_info, strategy.step_id)

    def before_test_step(self, strategy, **kwargs):
        self.set_task_layer(strategy.model, strategy.step_info, strategy.step_id)

    def set_task_layer(self, model: Module, classifier_field: str,
                       step_info: IStepInfo, step_id: Optional[int] = None):
        """
        Sets the correct task layer.

        This method is used by both training and testing flows. This is
        particularly useful when testing on the complete test set, which usually
        includes not already seen tasks.

        By default a Linear layer is created for each task. More info can be
        found at class level documentation.

        :param model: The model to adapt.
        :param classifier_field: The name of the layer (model class field) to
            change.
        :param step_info: The step info object.
        :param step_id: The relevant step id. If None, the current training step
            id is used.
        :return: None
        """
        # TODO: better management of testing flow (especially head expansion)
        # Main idea for the testing flow: just discard (not store) the layer for
        # tasks that were not encountered during a training flow.
        # Can use existing facilities to know if current flow is test or train!
        # Also, do not expand layers during testing !OR! create a new expanded
        # layer (which get discarded) by setting the weights of not already
        # encountered classes to 0!
        # --- end of dev comment ---

        # TODO: move previous layer back to cpu (using .to(...))

        if step_id is None:
            step_id = step_info.current_step

        training_info = step_info.step_specific_training_set(step_id)
        testing_info = step_info.step_specific_test_set(step_id)

        train_dataset, task_label = training_info
        test_dataset, _ = testing_info

        n_output_units = max(max(train_dataset.targets),
                             max(test_dataset.targets)) + 1

        if task_label not in self.task_layers:
            task_layer = self.create_task_layer(
                model, classifier_field, task_label=task_label,
                n_output_units=n_output_units)
            self.task_layers[task_label] = task_layer

        self.task_layers[task_label] = \
            self.expand_task_layer(model, classifier_field,
                                   n_output_units, self.task_layers[task_label])

        self.adapt_model_for_task(
            model, classifier_field,
            self.task_layers[task_label])

    def create_task_layer(self, model: Module, classifier_field: str,
                          n_output_units: int, previous_task_layer=None):
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

        :param model: The model for which the layer will be created.
        :param classifier_field: The name of the classifier field.
        :param n_output_units: The number of output units.
        :param previous_task_layer: If not None, the previously created layer
             for the same task.
        :return: The new layer.
        """
        if previous_task_layer is None:
            current_task_layer: Linear = getattr(model, classifier_field)
            in_features = current_task_layer.in_features
            has_bias = current_task_layer.bias is not None
        else:
            in_features = previous_task_layer.in_features
            has_bias = previous_task_layer.bias is not None

        new_layer = Linear(in_features, n_output_units, bias=has_bias)
        self.initialize_new_task_layer(new_layer)
        return new_layer

    def initialize_new_task_layer(self, new_layer: Module):
        """
        Initializes a new task layer.

        This method should initialize the input layer. This usually is just a
        weight initialization procedure, but more complex operations can be
        done as well.

        The input layer can be either a new layer created for a previously
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
        if prev_task_layer.bias is not None and new_task_layer.bias is not None:
            new_task_layer.bias[:to_copy_units] = \
                prev_task_layer.bias[:to_copy_units]

        # Initializes the expanded part (and adapts existing weights)
        self.initialize_dynamically_expanded_head(
            prev_task_layer, new_task_layer)

    def expand_task_layer(self, model: Module, classifier_field: str,
                          min_n_output_units: int, task_layer):
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

        :param model: The model.
        :param classifier_field: The name of the field to adapt.
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
            model, classifier_field, min_n_output_units,
            previous_task_layer=task_layer)

        self.adapt_task_layer(task_layer, new_layer)
        return new_layer

    def adapt_model_for_task(self, model: Module, classifier_field: str,
                             task_layer):
        """
        Sets the model classifier field for the given task layer
        By default, just sets the model property with the name given by
        the "classifier_field" parameter

        :param model: The model to adapt.
        :param classifier_field: The name of the classifier field.
        :param task_layer: The layer to set.
        """
        setattr(model, classifier_field, task_layer)
