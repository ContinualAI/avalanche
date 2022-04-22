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

from typing import TYPE_CHECKING, Optional, Sequence, Union

from avalanche.benchmarks.scenarios import ClassificationExperience
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.models import DynamicModule
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.nn import Module
from torch.optim import Optimizer

if TYPE_CHECKING:
    from avalanche.training.plugins import SupervisedPlugin


class AlreadyTrainedError(Exception):
    pass


class JointTraining(SupervisedTemplate):
    """Joint training on the entire stream.

    JointTraining performs joint training (also called offline training) on
    the entire stream of data. This means that it is not a continual
    learning strategy but it can be used as an "offline" upper bound for
    them.

    .. warnings also::
        Currently :py:class:`JointTraining` adapts its own dataset.
        Please check that the plugins you are using do not implement
        :py:meth:`adapt_trainin_dataset`. Otherwise, they are incompatible
        with :py:class:`JointTraining`.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator=default_evaluator,
        eval_every=-1,
    ):
        """Init.

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
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience."""
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )
        # JointTraining can be trained only once.
        self._is_fitted = False

    def train(
        self,
        experiences: Union[
            ClassificationExperience, Sequence[ClassificationExperience]
        ],
        eval_streams: Optional[
            Sequence[
                Union[
                    ClassificationExperience, Sequence[ClassificationExperience]
                ]
            ]
        ] = None,
        **kwargs
    ):
        """Training loop.

        JointTraining concatenates all the experiences together and
        trains on all of them at the same time (a.k.a. offline training).

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

        if self._is_fitted:
            raise AlreadyTrainedError(
                "JointTraining can be trained only once. "
                "Please call the train method once on the entire stream."
            )

        # Normalize training and eval data.
        if isinstance(experiences, ClassificationExperience):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):
            if isinstance(exp, ClassificationExperience):
                eval_streams[i] = [exp]
        self._eval_streams = eval_streams

        self._experiences = experiences
        self._before_training(**kwargs)
        for self.experience in experiences:
            self._before_training_exp(**kwargs)
            self._train_exp(self.experience, eval_streams, **kwargs)
            self._after_training_exp(**kwargs)
            # Joint training only needs a single step because
            # it concatenates all the data at once.
            break
        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        self._is_fitted = True
        return res

    def train_dataset_adaptation(self, **kwargs):
        """Concatenates all the datastream."""
        self.adapted_dataset = self._experiences[0].dataset
        if len(self._experiences) > 1:
            for exp in self._experiences[1:]:
                cat_data = AvalancheConcatDataset(
                    [self.adapted_dataset, exp.dataset]
                )
                self.adapted_dataset = cat_data
        self.adapted_dataset = self.adapted_dataset.train()

    def model_adaptation(self, model=None):
        """Adapts strategy's model for all experiences."""
        if model is None:
            model = self.model

        for experience in self._experiences:
            for module in model.modules():
                if isinstance(module, DynamicModule):
                    module.adaptation(experience.dataset)
            model = model.to(self.device)
        return model


__all__ = ["JointTraining"]
