from typing import Dict, Any

import torch
from torch.nn import Linear, Module

from avalanche.benchmarks import Experience
from avalanche.training.plugins.strategy_plugin import StrategyPlugin


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
        self.set_task_layer(strategy, strategy.experience)

    def before_eval_iteration(self, strategy, **kwargs):
        self._optimizer = strategy.optimizer
        self.set_task_layer(strategy, strategy.experience)

    @torch.no_grad()
    def set_task_layer(self, strategy, experience: Experience):
        """
        Sets the correct task layer. Creates a new head for previously
        unseen tasks.

        :param strategy: the CL strategy.
        :param experience: the experience info object.
        :return: None
        """

        # task label is set depending on the type of scenario
        # multitask or others
        if hasattr(strategy, 'mb_task_id'):
            task_label = strategy.mb_task_id
        else:
            task_label = experience.task_label
        n_output_units = max(experience.dataset.targets) + 1

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