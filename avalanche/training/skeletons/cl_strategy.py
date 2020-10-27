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


try:
    from typing import Protocol, List
except ImportError:
    from typing import List
    from typing_extensions import Protocol

from torch.utils.data import DataLoader

from avalanche.benchmarks.scenarios.generic_definitions import DatasetPart, \
    IStepInfo

from .strategy_flow import TrainingFlow, TestingFlow, FlowGroup, StrategyFlow


# TODO: better implementation of flow listeners


class IStrategy(Protocol):
    """
    Define the protocol for a strategy.

    A strategy is a class with train and test methods. Those methods must accept
    a step info object as the parameter. Moreover, the test method should allow
    the user to select which part (complete, cumulative, ...) of the test set
    should be considered by the testing phase.

    """
    def train(self, step_info: IStepInfo, **kwargs):
        """
        Executes an incremental training step on the training data.

        This methods takes a "step_info" as a parameter. The "step_info"
        instance can be used to extract data relevant to the current training
        step, like the training dataset, the current task/batch id, any
        previous or future training or testing test, etc.

        :param step_info: The step info instance.
        :param kwargs: A list of strategy parameters.
        :return: Strategy specific.
        """
        ...

    def test(self, step_info: IStepInfo, test_part: DatasetPart, **kwargs):
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
        :return: Strategy specific.
        """
        ...


class StrategySkeleton(IStrategy):
    """
    Defines the skeleton for a Continual Learning strategy.

    This skeleton introduces the "Training" and "Testing" flows, which are the
    two most commonly used flows. The training flow describes how the strategy
    runs an incremental training step on a new batch/task, while the testing
    flow defines the testing procedure.

    This class is usually not used directly. Consider using
    :class:`avalanche.training.strategies.DeepLearningStrategy` instead.

    This class also defines few utility methods to control the strategy
    lifecycle.
    """
    def __init__(self):
        """
        Creates a strategy skeleton instance.
        """
        # plugins will keep track of the list of plugins
        self.plugins: List['StrategySkeleton'] = []

        # Define empty training and testing flows
        self.training_flow: StrategyFlow = StrategyFlow(
            self, 'training_flow', 'Training')
        self.Training: FlowGroup = self.training_flow.root_group

        # Define testing phase template
        self.testing_flow: StrategyFlow = StrategyFlow(
            self, 'testing_flow', 'Testing')
        self.Testing: FlowGroup = self.testing_flow.root_group

    def update_namespace(self, **update_dict):
        """
        Updates the "results namespace" using the provided parameters.

        For more info refer to the class :class:`StrategyFlow` documentation.

        :param update_dict: The parameters used to update the "results
            namespace".
        :return: None.
        """
        if self.is_training():
            self.training_flow.update_results_namespace(update_dict)
        if self.is_testing():
            self.testing_flow.update_results_namespace(update_dict)

    def is_training(self):
        """
        Check if the training flow is running.
        :return: True if the training flow is running, False otherwise.
        """
        return self.training_flow.is_running()

    def is_testing(self):
        """
        Check if the testing flow is running.
        :return: True if the testing flow is running, False otherwise.
        """
        return self.testing_flow.is_running()

    def add_plugin(self, plugin: 'StrategySkeleton'):
        """
        Adds a plugin to this strategy.

        :param plugin: The plugin to add.
        :return: None
        """
        self.plugins.append(plugin)
        self.training_flow.add_strategy_plugin(plugin)
        self.testing_flow.add_strategy_plugin(plugin)


class StrategyTemplate(StrategySkeleton):
    """
    Defines a common generic template for Continual Learning strategies.

    Being extremely generic, this template can also be used by non Deep
    Learning approaches.

    The training flow is simply divided in three parts: BeforeTraining,
    ModelTraining and AfterTraining. The only implemented facility for training
    is the extraction of the training dataset from the step_info and the
    creation of the DataLoader.

    The testing flow template is more complex as it involves managing a list
    of test subsets (usually one for each task/batch) obtained from step_info.
    By using this template as the  base class for a strategy, the procedures
    needed to loop through the  different test subsets are already implemented
    and integrated in the testing flow.
    """
    def __init__(self):
        """
        Creates and instance of the Strategy Template.
        """
        super().__init__()
        self.training_step_id = 0

        # Define training phase template

        # In fact, the "to_group" argument already defaults to root_group
        # Here it's explicitly set for readability purposes
        # For the same purpose, some at_beginning and after_part arguments were
        # set even when the default value would have sufficed
        self.BeforeTraining = self.training_flow.append_new_group(
            'BeforeTraining',
            [self.make_train_dataset, self.make_train_dataloader],
            to_group=self.Training, at_beginning=True)

        self.ModelTraining = self.training_flow.append_new_group(
            'ModelTraining', [], to_group=self.Training,
            after_part=self.BeforeTraining)

        self.AfterTraining = self.training_flow.append_new_group(
            'AfterTraining', [], to_group=self.Training,
            after_part=self.ModelTraining)

        # Define testing phase template
        # The ideas seen for the training part can be applied to the test
        # flow, too. The only difference is that we have multiple test datasets,
        # usually one for each task.
        #
        # So the general schema becomes:
        # - Before testing
        # --- set_initial_test_step_id
        # - MultiStepTestLoop
        # --- StepTesting
        # ------ make_test_dataset, make_test_dataloader
        # ------ ModelTesting
        # --- next_testing_step
        # - AfterTesting

        self.BeforeTesting = self.testing_flow.append_new_group(
            'BeforeTesting',
            [self.set_initial_test_step_id],
            to_group=self.Testing, at_beginning=True)

        self.MultiStepTestLoop = self.testing_flow.append_new_group(
            'MultiStepTestLoop', [], to_group=self.Testing,
            after_part=self.BeforeTesting, is_loop=True)

        self.StepTesting = self.testing_flow.append_new_group(
            'StepTesting', [self.make_test_dataset, self.make_test_dataloader],
            to_group=self.MultiStepTestLoop)

        self.ModelTesting = self.testing_flow.append_new_group(
            'ModelTesting', [], to_group=self.StepTesting)

        self.testing_flow.append_part_list(
            [self.next_testing_step], to_group=self.MultiStepTestLoop)

        self.AfterTesting = self.testing_flow.append_new_group(
            'AfterTesting', [], to_group=self.Testing,
            after_part=self.MultiStepTestLoop)

    @TrainingFlow
    def make_train_dataset(self, step_info: IStepInfo):
        """
        Returns the training dataset, given the step_info instance.

        This is a part of the training flow. Sets the train_dataset namespace
        value.

        :param step_info: The step info instance, as returned from the CL
            scenario.
        :return: The training dataset.
        """
        train_dataset = step_info.current_training_set()[0]
        self.update_namespace(train_dataset=train_dataset)
        return train_dataset

    @TrainingFlow
    def make_train_dataloader(self, train_dataset, num_workers=0,
                              train_mb_size=1):
        """
        Return a DataLoader initialized with the training dataset.

        This is a part of the training flow. Sets the train_data_loader
        namespace value.

        :param train_dataset: The training dataset. Usually set by the
            make_train_dataset method.
        :param num_workers: The number of workers to use. Defaults to 0.
            Usually set by the user when calling the train method of the
            strategy or as a strategy field.
        :param train_mb_size: The minibatch size. Defaults to 1. Usually set
            as a strategy field.
        :return: The DataLoader for the training set.
        """
        train_data_loader = DataLoader(
            train_dataset, num_workers=num_workers, batch_size=train_mb_size)
        self.update_namespace(train_data_loader=train_data_loader)
        return train_data_loader

    @TestingFlow
    def make_test_dataset(self, step_info: IStepInfo, step_id: int):
        """
        Returns the test dataset, given the step_info instance and the
        identifier of the step (task/batch) to test.

        This is a part of the testing flow. Sets the test_dataset namespace
        value.

        :param step_info: The step info instance, as returned from the CL
            scenario.
        :param step_id: The ID of the step for which to obtain the test set.
        :return: The training dataset.
        """
        test_dataset = step_info.step_specific_test_set(step_id)[0]
        self.update_namespace(test_dataset=test_dataset)
        return test_dataset

    @TestingFlow
    def make_test_dataloader(self, test_dataset, num_workers=0,
                             test_mb_size=1):
        """
        Return a DataLoader initialized with the test dataset.

        This is a part of the testing flow. Sets the test_data_loader
        namespace value.

        :param test_dataset: The test dataset. Usually set by the
            make_test_dataset method.
        :param num_workers: The number of workers to use. Defaults to 0.
            Usually set by the user when calling the test method of the
            strategy or as a strategy field.
        :param test_mb_size: The minibatch size. Defaults to 1. Usually set
            as a strategy field.
        :return: The DataLoader for the test set.
        """
        test_data_loader = DataLoader(
            test_dataset, num_workers=num_workers, batch_size=test_mb_size)
        self.update_namespace(test_data_loader=test_data_loader)
        return test_data_loader

    @TestingFlow
    def set_initial_test_step_id(self, step_info: IStepInfo,
                                 dataset_part: DatasetPart = None):
        """
        An internal method that sets the initial step_id for the testing flow.

        The initial step id depends on the dataset_part passed to the test
        method of the strategy.

        For the complete, cumulative and old parts, the initial step_id will be
        zero. For the future part, the step id will be the current step plus
        one. For the current part, step_id will be set to the current step id.

        :param step_info: The step info instance, as returned from the CL
            scenario.
        :param dataset_part: The dataset part to consider for testing (as passed
            to the test method of the strategy), as a value of
            :class:`DatasetPart`.
        :return: None
        """
        step_id = -1
        if dataset_part is None:
            dataset_part = DatasetPart.COMPLETE

        if dataset_part == DatasetPart.CURRENT:
            step_id = step_info.current_step
        if dataset_part in [DatasetPart.CUMULATIVE, DatasetPart.OLD,
                            DatasetPart.COMPLETE]:
            step_id = 0
        if dataset_part == DatasetPart.FUTURE:
            step_id = step_info.current_step + 1

        if step_id < 0:
            raise ValueError('Invalid dataset part')
        self.update_namespace(step_id=step_id)

    @TestingFlow
    def next_testing_step(self, step_id: int, step_info: IStepInfo,
                          test_part: DatasetPart = None):
        """
        Checks if another testing step has to be done and sets the step_id
        namespace value accordingly.

        :param step_id: The current test step id.
        :param step_info: The step info instance, as returned from the CL
            scenario.
        :param test_part: The dataset part to consider for testing (as passed
            to the test method of the strategy), as a value of
            :class:`DatasetPart`.
        :return: True, if other testing steps are to be executed. False
            otherwise.
        """
        step_id += 1
        if self.has_testing_steps_left(step_id, step_info,
                                       test_part=test_part):
            self.update_namespace(step_id=step_id)
            return True
        return False  # Ends testing loop

    @TestingFlow
    def has_testing_steps_left(self, step_id,
                               step_info: IStepInfo,
                               test_part: DatasetPart = None):
        """
        Checks if another testing step has to be done.

        Doesn't set any namespace value.

        :param step_id: The current test step id.
        :param step_info: The step info instance, as returned from the CL
            scenario.
        :param test_part: The dataset part to consider for testing (as passed
            to the test method of the strategy), as a value of
            :class:`DatasetPart`.
        :return: True, if other testing steps are to be executed. False
            otherwise.
        """
        if test_part is None:
            test_part = DatasetPart.COMPLETE

        if test_part == DatasetPart.CURRENT:
            return step_id == step_info.current_step
        if test_part == DatasetPart.CUMULATIVE:
            return step_id <= step_info.current_step
        if test_part == DatasetPart.OLD:
            return step_id < step_info.current_step
        if test_part == DatasetPart.FUTURE:
            return step_info.current_step < step_id < step_info.n_steps
        if test_part == DatasetPart.COMPLETE:
            return step_id < step_info.n_steps

        raise ValueError('Invalid dataset part')

    def train(self, step_info: IStepInfo, **kwargs):
        if self.is_testing() or self.is_training():
            raise RuntimeError('Another flow is running')
        res = self.training_flow(step_info=step_info, **kwargs)
        self.training_step_id += 1
        return res

    def test(self, step_info: IStepInfo, test_part: DatasetPart, **kwargs):
        if self.is_testing() or self.is_training():
            raise RuntimeError('Another flow is running')
        return self.testing_flow(step_info=step_info,
                                 test_part=test_part, **kwargs)


__all__ = ['IStrategy', 'StrategySkeleton', 'StrategyTemplate']
