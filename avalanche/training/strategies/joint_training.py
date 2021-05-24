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

from avalanche.benchmarks.scenarios import Experience
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins.evaluation import default_logger
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
        JointTraining performs joint training (also called offline training) on
        the entire stream of data. This means that it is not a continual
        learning strategy but it can be used as an "offline" upper bound for
        them.

        .. warnings also::
            Currently :py:class:`JointTraining` adapts its own dataset.
            Please check that the plugins you are using do not implement
            :py:meth:`adapt_trainin_dataset`. Otherwise, they are incompatible
            with :py:class:`JointTraining`.

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

    def train(self, experiences: Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience,
                                                    Sequence[
                                                        Experience]]]] = None,
              **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.

        :return: dictionary containing last recorded value for
            each metric name.
        """
        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if isinstance(experiences, Experience):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):
            if isinstance(exp, Experience):
                eval_streams[i] = [exp]

        self._experiences = experiences
        self.before_training(**kwargs)
        for exp in experiences:
            self.train_exp(exp, eval_streams, **kwargs)
            # Joint training only needs a single step because
            # it concatenates all the data at once.
            break
        self.after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def train_dataset_adaptation(self, **kwargs):
        """ Concatenates all the datastream. """
        self.adapted_dataset = self._experiences[0].dataset
        for exp in self._experiences:
            cat_data = AvalancheConcatDataset([self.adapted_dataset,
                                               exp.dataset])
            self.adapted_dataset = cat_data
        self.adapted_dataset = self.adapted_dataset.train()


__all__ = ['JointTraining']
