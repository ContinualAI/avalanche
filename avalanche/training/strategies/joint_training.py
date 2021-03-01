################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-11-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

from typing import Optional, Sequence, TYPE_CHECKING, Union

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset

from avalanche.benchmarks.scenarios import IExperience
from avalanche.logging import default_logger
from avalanche.training.strategies import BaseStrategy

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin


class JointTraining(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger):
        """
        JointStrategy performs joint training on the entire stream of data.
        This means that it is not a continual learning strategy but it can be
        used as an "offline" upper bound for them.

        WARNING: JointTraining adapts its own dataset.
        Please check that the plugins you are using do not implement
        `adapt_trainin_dataset`. Otherwise, they are incompatible with
        `JointTraining`.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        """
        super().__init__(model, optimizer, criterion, train_mb_size,
                         train_epochs, eval_mb_size, device, plugins, evaluator)

    def train(self, experiences: Union[IExperience, Sequence[IExperience]],
              **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.

        :param experiences: single IExperience or sequence.
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        if isinstance(experiences, IExperience):
            experiences = [experiences]

        res = []
        self.before_training(**kwargs)

        self.experience = experiences[0]
        self.train_task_label = self.experience.task_label
        self.model.train()
        self.model.to(self.device)

        self.adapted_dataset = experiences[0].dataset
        self.adapted_dataset.train()
        # DO NOT CALL adapt_train_dataset.
        # JointTraining adapts its own data in a custom manner.
        # TODO: support adapt_train_dataset
        # waiting for https://github.com/vlomonaco/avalanche/issues/320
        # self.adapt_train_dataset(**kwargs)
        ext_mem = {}
        for exp in experiences:
            if exp.task_label in ext_mem:
                curr_task_data = ext_mem[exp.task_label]
                cat_data = ConcatDataset([exp.dataset, curr_task_data])
                ext_mem[exp.task_label] = cat_data
            else:
                ext_mem[exp.task_label] = exp.dataset
        self.adapted_dataset = ext_mem

        self.make_train_dataloader(**kwargs)
        self.before_training_exp(**kwargs)
        self.epoch = 0
        for self.epoch in range(self.train_epochs):
            self.before_training_epoch(**kwargs)
            self.training_epoch(**kwargs)
            self.after_training_epoch(**kwargs)
        self.after_training_exp(**kwargs)

        res.append(self.evaluator.current_metrics.copy())
        self.after_training(**kwargs)
        return res


__all__ = ['JointTraining']
