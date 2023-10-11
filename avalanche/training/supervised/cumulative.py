from typing import Callable, Optional, List, Union
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset

from avalanche.benchmarks.utils import _concat_taskaware_classification_datasets
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate


class Cumulative(SupervisedTemplate):
    """Cumulative training strategy.

    At each experience, train model with data from all previous experiences
        and current experience.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        self.dataset = None  # cumulative dataset

    def train_dataset_adaptation(self, **kwargs):
        """
        Concatenates all the previous experiences.
        """
        exp = self.experience
        assert exp is not None
        if self.dataset is None:
            self.dataset = exp.dataset
        else:
            self.dataset = concat_datasets([self.dataset, exp.dataset])
        self.adapted_dataset = self.dataset
