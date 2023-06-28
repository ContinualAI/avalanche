################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24/07/2021                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from avalanche.training.plugins import SupervisedPlugin


class Clock(SupervisedPlugin, supports_distributed=True):
    """Counter for strategy events.

    WARNING: Clock needs to be the last plugin, otherwise counters will be
    wrong for plugins called after it.
    """

    def __init__(self):
        """Init."""
        super().__init__()
        # train
        self.train_iterations = 0
        """ Total number of training iterations. """

        self.train_exp_counter = 0
        """ Number of past training experiences. """

        self.train_exp_epochs = 0
        """ Number of training epochs for the current experience. """

        self.train_exp_iterations = 0
        """ Number of training iterations for the current experience. """

        self.train_epoch_iterations = 0
        """ Number of iterations for the current epoch. """

        self.total_iterations = 0
        """ Total number of iterations in training and eval mode. """

    def before_training_exp(self, strategy, **kwargs):
        self.train_exp_iterations = 0
        self.train_exp_epochs = 0

    def before_training_epoch(self, strategy, **kwargs):
        self.train_epoch_iterations = 0

    def after_training_iteration(self, strategy, **kwargs):
        self.train_epoch_iterations += 1
        self.train_exp_iterations += 1
        self.train_iterations += 1
        self.total_iterations += 1

    def after_training_epoch(self, strategy, **kwargs):
        self.train_exp_epochs += 1

    def after_training_exp(self, strategy, **kwargs):
        self.train_exp_counter += 1

    def after_eval_iteration(self, strategy, **kwargs):
        self.total_iterations += 1


__all__ = ["Clock"]
