#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 11-09-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

from typing import Dict, Any, Optional

import torch
from torch.nn import Linear, Module

from avalanche.benchmarks.scenarios import TStepInfo, IStepInfo, DatasetPart
from avalanche.evaluation import EvalProtocol
from .cl_strategy import StrategyTemplate
from .evaluation_module import EvaluationModule
from .strategy_flow import TrainingFlow, TestingFlow


class DeepLearningStrategy(StrategyTemplate):
    """
    Defines a general Deep Learning strategy.

    This class is usually used as the father class of most strategy
    implementations, although users should consider using
    :class:`MTDeepLearningStrategy`, which also adds automatic multi head
    management for multi-task scenarios.

    This class (or :class:`MTDeepLearningStrategy`) takes care of most
    under-the-hood management. Most users should create an inherited class and
    implement its "training_epoch" and "testing_epoch" methods.
    Optionally, also overriding "adapt_train_dataset" and "adapt_test_dataset"
    may allow the users to adapt (pad, augment, etc.) the training and test
    datasets.

    This class introduces the main parts usually found in a Deep Learning based
    Continual Learning strategy, such as a training loop based on epochs,
    a model adaptation flow group (filled, for instance, by
    :class:`MTDeepLearningStrategy`) and a lot of default callbacks that the
    user can override, such as:

    -   before/after_training
    -   before/after_training_epoch
    -   before/after_testing
    -   before/after_step_testing
    -   before/after_testing_epoch

    And also exposes some useful callback methods that have to be called by
    implementing classes, such as:

    -   before/after_training_iteration
    -   before/after_forward
    -   before/after_backward
    -   before/after_update
    -   before/after_test_iteration
    -   before/after_test_forward
    """
    def __init__(self, train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = 1, device=None,
                 evaluation_protocol: Optional[EvalProtocol] = None):
        """
        Creates a new instance of DeepLearningStrategy.

        This constructor accepts common parameters used to control the minibatch
        size, the number of training epochs and the device used for training.
        It also accepts and instance of :class:`EvalProtocol` which will be used
        to compute the required metrics.

        :param train_mb_size: The size of the training minibatch size. This
            value is used when creating the training data loader. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The size of the test minibatch size. This
            value is used when creating the testing data loader. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param evaluation_protocol: The evaluation protocol used to compute
            the relevant metrics. Defaults to None.
        """
        super(DeepLearningStrategy, self).__init__()

        # Define training skeleton

        # Before training
        # In particular:
        # - set_initial_epoch: sets epoch = 0
        # - before_training: callback the user can override
        self.training_flow.append_part_list([self.set_initial_epoch,
                                             self.before_training],
                                            to_group=self.BeforeTraining,
                                            at_beginning=True)

        # Adds the "adapt_train_dataset" (can be overridden by the user),
        # between make_train_dataset and make_train_dataloader
        self.training_flow.append_part_list([self.adapt_train_dataset],
                                            to_group=self.BeforeTraining,
                                            after_part=self.make_train_dataset)

        # After training
        # - after_training: callback the user can override
        self.training_flow.append_part_list([self.after_training],
                                            to_group=self.AfterTraining,
                                            at_beginning=True)

        # Model training
        # TrainingModelAdaptation is a empty group that can be filled later
        # For instance, by MTDeepLearningStrategy which handles multi-task heads
        self.TrainingModelAdaptation = self.training_flow.append_new_group(
            'TrainingModelAdaptation', [], to_group=self.ModelTraining)

        # before_training_epoch and after_training_epoch are callbacks the user
        # can override, while next_training_epoch sets epoch+=1
        # (if there are any remaining epochs to run, of course)
        training_loop_skeleton = [self.before_training_epoch,
                                  self.after_training_epoch,
                                  self.next_training_epoch]
        self.TrainingLoop = self.training_flow.append_new_group(
            'TrainingLoop', training_loop_skeleton,
            to_group=self.ModelTraining, is_loop=True)

        # The training epoch is actually a group. This was mainly done to
        # allow for more flexibility, but most users will only want to
        # implement the training_epoch method
        self.training_flow.append_new_group(
            'TrainingEpoch', [self.training_epoch],
            to_group=self.TrainingLoop, after_part=self.before_training_epoch)

        # Define testing skeleton
        # The follow ing code adds several callbacks and management parts.
        # The general schema becomes (added elements signaled with "added"):
        # - Before testing
        # --- before_testing (added)
        # --- set_initial_test_step_id
        # - MultiStepTestLoop
        # --- before_step_testing (added)
        # --- TestingModelAdaptation
        # --- StepTesting
        # ------ make_test_dataset
        # ------ adapt_test_dataset (added)
        # ------ make_test_dataloader
        # ------ ModelTesting
        # --------- before_testing_epoch (added)
        # --------- TestingEpoch (added)
        # ------------ testing_epoch (added)
        # --------- after_testing_epoch (added)
        # --- after_step_testing (added)
        # --- next_testing_step
        # - AfterTesting
        # --- after_testing (added)

        # Before testing
        self.testing_flow.append_part_list([self.before_testing],
                                           to_group=self.BeforeTesting,
                                           at_beginning=True)

        # After testing
        self.testing_flow.append_part_list(
            [self.after_testing], self.AfterTesting, at_beginning=True)

        # Testing loop
        self.testing_flow.append_part_list(
            [self.before_step_testing], to_group=self.MultiStepTestLoop,
            at_beginning=True)

        # Adds the "adapt_test_dataset" (can be overridden by the user),
        # between make_test_dataset and make_test_dataloader
        self.testing_flow.append_part_list(
            [self.adapt_test_dataset], to_group=self.StepTesting,
            after_part=self.make_test_dataset)

        self.TestingModelAdaptation = self.testing_flow.append_new_group(
            'TestingModelAdaptation', [], to_group=self.MultiStepTestLoop,
            after_part=self.before_step_testing)

        self.testing_flow.append_part_list(
            [self.before_testing_epoch], to_group=self.ModelTesting,
            at_beginning=True)

        self.TestingEpoch = self.testing_flow.append_new_group(
            'TestingEpoch', [self.testing_epoch],
            to_group=self.ModelTesting, after_part=self.before_testing_epoch)

        self.testing_flow.append_part_list(
            [self.after_testing_epoch], to_group=self.ModelTesting,
            after_part=self.TestingEpoch)

        self.testing_flow.append_part_list(
            [self.after_step_testing], to_group=self.MultiStepTestLoop,
            after_part=self.StepTesting)

        # Store defaults
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.test_mb_size = test_mb_size
        self.device = device

        # Evaluation module
        self.evaluation_module = EvaluationModule()
        self.add_module(self.evaluation_module)
        self.evaluation_protocol = evaluation_protocol

    @TrainingFlow
    def training_epoch(self):
        """
        Runs a training epoch.

        This is the method most users should override, along with
        "testing_epoch".

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def testing_epoch(self):
        """
        Runs a testing epoch.

        This is the method most users should override, along with
        "training_epoch".

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def adapt_train_dataset(self):
        """
        Adapts the training set.

        This method can be safely overridden by users. Defaults to no-op.

        The user should adapt the existing "train_dataset" (that can be
        retrieved in the global namespace). Operations may involve padding,
        merging of replay patterns, ... If the result is an object different
        from the original "train_dataset", the corresponding namespace value
        must be set accordingly using the "update_namespace" method.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def adapt_test_dataset(self):
        """
        Adapts the test set.

        This method can be safely overridden by users. Defaults to no-op.

        The user should adapt the existing "test_dataset" (that can be
        retrieved in the global namespace). If the result is an object different
        from the original "test_dataset", the corresponding namespace value
        must be set accordingly using the "update_namespace" method.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def set_initial_epoch(self):
        """
        Initial utility that sets the initial epoch namespace value to 0.

        Most users shouldn't override this method.
        """
        self.update_namespace(epoch=0)

    @TrainingFlow
    def before_training(self):
        """
        A callback that gets invoked before training.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def after_training(self):
        """
        A callback that gets invoked after training.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def before_training_epoch(self):
        """
        A callback that gets invoked before each training epoch.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def after_training_epoch(self):
        """
        A callback that gets invoked after each training epoch.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def next_training_epoch(self, epoch=0, train_epochs=1):
        """
        Checks if another epoch has to be run and sets the epoch namespace
        value accordingly.

        This method simply checks for the train_epochs parameter for the number
        of training epochs to run.

        This is the last part of the TrainingLoop group, which means that
        returning "False" stops the training loop.

        Most users shouldn't override this method.

        :param epoch: The current epoch.
        :param train_epochs: The number of training epoch to run. This is
        usually taken from the class field with the same name.

        :return: True if other epochs are to be run, False otherwise.
        """
        epoch += 1
        if self.has_training_epochs_left(epoch=epoch,
                                         train_epochs=train_epochs):
            self.update_namespace(epoch=epoch)
            return True
        return False  # Ends training loop

    @TrainingFlow
    def has_training_epochs_left(self, epoch=0, train_epochs=1):
        """
        Checks if there are training epochs left.

        This method simply checks for the train_epochs parameter for the number
        of training epochs to run.

        This method doesn't set any namespace values.

        Most users shouldn't override this method.

        :param epoch: The current epoch.
        :param train_epochs: The number of training epoch to run. This is
        usually taken from the class field with the same name.
        :return: True if other epochs are to be run, False otherwise.
        """
        return epoch < train_epochs

    @TrainingFlow
    def before_training_iteration(self):
        """
        A callback that gets invoked before each training iteration.

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        before running each training iteration.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def after_training_iteration(self):
        """
        A callback that gets invoked after each training iteration.

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        after running each training iteration.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def before_forward(self):
        """
        A callback that gets invoked before running the forward pass on the
        model(s?).

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        before running each forward pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def after_forward(self):
        """
        A callback that gets invoked after running the forward pass on the
        model(s?).

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        after running each forward pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def before_backward(self):
        """
        A callback that gets invoked before running the backward pass on the
        model(s?).

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        before running each backward pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def after_backward(self):
        """
        A callback that gets invoked after running the backward pass on the
        model(s?).

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        after running each backward pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def before_update(self):
        """
        A callback that gets invoked before running the update pass on the
        model(s?).

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        before running each update pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TrainingFlow
    def after_update(self):
        """
        A callback that gets invoked after running the update pass on the
        model(s?).

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        after running each update pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def before_testing(self):
        """
        A callback that gets invoked before testing.

        Beware that another callback method exists, "before_step_testing",
        which gets invoked before testing on each single test set. On the
        contrary, this method gets invoked once at the very beginning of the
        testing flow.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def after_testing(self):
        """
        A callback that gets invoked after testing.

        Beware that another callback method exists, "after_step_testing",
        which gets invoked after testing on each single test set. On the
        contrary, this method gets invoked once at the very end of the
        testing flow.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def before_step_testing(self):
        """
        A callback that gets invoked before testing on a single test set.

        Beware that another callback method exists, "before_testing",
        which gets invoked once at the very beginning of the test flow. On the
        contrary, this method gets invoked before testing on each test set.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def after_step_testing(self):
        """
        A callback that gets invoked after testing on a single test set.

        Beware that another callback method exists, "after_testing",
        which gets invoked once at the very end of the test flow. On the
        contrary, this method gets invoked after testing on each test set.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def before_testing_epoch(self):
        """
        A callback that gets invoked before running a test epoch.

        Consider that, when testing, only one epoch for each test set is
        executed. The main difference between this callback and
        "before_step_testing" is that, when "before_step_testing" is called,
        the model is not already adapted for the current test set. Also,
        the "make_test_dataset", "adapt_test_dataset", "make_test_dataloader"
        are called after "before_step_testing" and before this callback.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def after_testing_epoch(self):
        """
        A callback that gets invoked after running a test epoch.

        Consider that, when testing, only one epoch for each test set is
        executed.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def before_test_iteration(self):
        """
        A callback that gets invoked before running an iteration on a
        test set.

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        before running each test iteration.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def after_test_iteration(self):
        """
        A callback that gets invoked after running an iteration on a
        test set.

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        after running each test iteration.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def before_test_forward(self):
        """
        A callback that gets invoked before running the forward pass on the
        model(s) during testing.

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        before running each forward pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    @TestingFlow
    def after_test_forward(self):
        """
        A callback that gets invoked after running the forward pass on the
        model(s) during testing.

        This callback is not automatically called by the
        :class:`DeepLearningStrategy`, where it's declared. In fact,
        implementing strategies subclasses should explicitly call this method
        after running each forward pass.

        Can be safely overridden by users. Consider calling the super method
        when overriding.

        :return: Strategy specific.
        """
        pass

    def train(self, step_info: TStepInfo, **kwargs):
        """
        Executes an incremental training step on the training data.

        This methods takes a "step_info" as a parameter. The "step_info"
        instance can be used to extract data relevant to the current training
        step, like the training dataset, the current task/batch id, any
        previous or future training or testing test, etc.

        :param step_info: The step info instance.
        :param kwargs: A list of strategy parameters.
        :return: The result of the evaluation protocol (if any).
        """
        super(DeepLearningStrategy, self).train(step_info, **kwargs)
        if self.evaluation_protocol is not None:
            return self.evaluation_module.get_train_result()

    def test(self, step_info: TStepInfo, test_part: DatasetPart, **kwargs):
        """
        Executes a testing procedure step on the testing data.

        This methods takes a "step_info" as a parameter. The "step_info"
        instance can be used to extract data relevant to the current testing
        step, like the test datasets, the current task/batch id, any
        previous or future training or testing test, etc.

        Beware that a dataset part flag must be passed as the second parameter
        (as a value of :class:`DatasetPart`). This flag controls whenever the
        user wants to test on all tasks/batches ("COMPLETE"), only on already
        encountered tasks/batches ("CUMULATIVE"), only on previous ones ("OLD"),
        on the current task/batch only ("CURRENT") or even on future
        tasks/batches ("FUTURE").

        The testing procedure must loop through the different test sets
        to obtain the relevant metrics.

        :param step_info: The step info instance.
        :param kwargs: A list of strategy parameters.
        :return: The result of the evaluation protocol (if any).
        """
        super(DeepLearningStrategy, self).test(step_info, test_part, **kwargs)
        if self.evaluation_protocol is not None:
            return self.evaluation_module.get_test_result()


class MTDeepLearningStrategy(DeepLearningStrategy):
    """
    Defines a common skeleton for Deep Learning strategies supporting
    Multi Task scenarios. This base class can be used as the foundation for
    strategies supporting Single-Incremental-Task (a.k.a. task-free) scenarios
    as well (in which case, it only handles the dynamic head expansion part).

    This class adds several elements to the training and testing flows.
    In particular, the default implementation keeps an internal set of layers,
    one for each task.

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

    In each training/testing step the strategy changes the model final layer
    with the task-specific one. Those behaviours can be changed
    by overriding the appropriate methods in order to achieve more complex
    Multi Task management.
    """
    def __init__(self, model: Module, classifier_field: str = 'classifier',
                 keep_initial_layer=False, train_mb_size=1, train_epochs=1,
                 test_mb_size=None, evaluation_protocol=None, device=None):
        """
        Creates a new MTDeepLearningStrategy instance.

        This constructor is usually invoked by implementing subclasses.

        This class expects a single model to adapt. More complex setups, where
        multiple models are used, must be implemented separately (this
        implementation may still serve as a good starting point). The second
        parameter specifies the name of the model field which will get changed
        when adapting for a different task. The third parameter control whenever
        the existing head should be kept for task 0 or it should be discarded.

        The remaining parameters are passed as constructor arguments for the
        superclass :class:`DeepLearningStrategy`.

        :param model: The model.
        :param classifier_field: The name of the model field to adapt.
        :param keep_initial_layer: If True, the model head found in the original
            model will be used for task 0. Defaults to False, which means that
            the head for task 0 will be created from scratch and the existing
            head will be discarded. Beware that when keeping the original layer
            the weight initialization will not take place. That is, the layer
            will initially be kept as-is.
        :param train_mb_size: The size of the training minibatch size. This
            value is used when creating the training data loader. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The size of the test minibatch size. This
            value is used when creating the testing data loader. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param evaluation_protocol: The evaluation protocol used to compute
            the relevant metrics. Defaults to None.
        """
        super(MTDeepLearningStrategy, self).__init__(
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, evaluation_protocol=evaluation_protocol,
            device=device)

        if not hasattr(model, classifier_field):
            raise ValueError('The model has no field named ' + classifier_field)

        self.training_flow.append_part_list(
            [self.set_task_layer], to_group=self.TrainingModelAdaptation,
            at_beginning=True)
        self.testing_flow.append_part_list(
            [self.set_task_layer], to_group=self.TestingModelAdaptation,
            at_beginning=True)

        self.model = model
        self.classifier_field = classifier_field
        self.task_layers: Dict[int, Any] = dict()

        if keep_initial_layer:
            self.task_layers[0] = getattr(model, classifier_field)

    @TrainingFlow
    @TestingFlow
    @torch.no_grad()
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

    @TrainingFlow
    @TestingFlow
    @torch.no_grad()
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

    @TrainingFlow
    @TestingFlow
    @torch.no_grad()
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

    @TrainingFlow
    @TestingFlow
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

    @TrainingFlow
    @TestingFlow
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
        if prev_task_layer.bias is not None and new_task_layer.bias is not None:
            new_task_layer.bias[:to_copy_units] = \
                prev_task_layer.bias[:to_copy_units]

        # Initializes the expanded part (and adapts existing weights)
        self.initialize_dynamically_expanded_head(
            prev_task_layer, new_task_layer)

    @TrainingFlow
    @TestingFlow
    @torch.no_grad()
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

    @TrainingFlow
    @TestingFlow
    @torch.no_grad()
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


__all__ = ['DeepLearningStrategy', 'MTDeepLearningStrategy']
