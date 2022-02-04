import copy
import warnings
from typing import Optional, List, Union, Sequence

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

from avalanche.benchmarks import Experience
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.models import DynamicModule
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate


class SupervisedOnlineTemplate(SupervisedTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        num_passes: int = 1,
        train_mb_size: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=1,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        self.num_passes = num_passes

        warnings.warn(
            "This is an unstable experimental strategy."
            "Some plugins may not work properly."
        )

    def create_sub_experience_list(self, experience):
        """Creates a list of sub-experiences from an experience.
        It returns a list of experiences, where each experience is
        a subset of the original experience.

        :param experience: single Experience.

        :return: list of Experience.
        """

        # Shuffle the indices
        indices = torch.randperm(len(experience.dataset))
        num_sub_exps = len(indices) // self.train_mb_size

        sub_experience_list = []
        for subexp_id in range(num_sub_exps):
            subexp_indices = indices[
                subexp_id
                * self.train_mb_size : (subexp_id + 1)
                * self.train_mb_size
            ]
            sub_experience = copy.copy(experience)
            subexp_ds = AvalancheSubset(
                sub_experience.dataset, indices=subexp_indices
            )
            sub_experience.dataset = subexp_ds
            sub_experience_list.append(sub_experience)

        return sub_experience_list

    def train(
        self,
        experiences: Union[Experience, Sequence[Experience]],
        eval_streams: Optional[
            Sequence[Union[Experience, Sequence[Experience]]]
        ] = None,
        **kwargs
    ):
        """Training loop. if experiences is a single element trains on it.
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
        self._stop_training = False

        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if not isinstance(experiences, Sequence):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        self._eval_streams = eval_streams

        self.num_sub_exps = len(experiences[0].dataset) // self.train_mb_size
        self._before_training(**kwargs)

        # Keep the (full) experience in self.full_experience
        # for model adaptation
        for self.full_experience in experiences:
            sub_experience_list = self.create_sub_experience_list(
                self.full_experience
            )

            # Train for each sub-experience
            for i, sub_experience in enumerate(sub_experience_list):
                self.experience = sub_experience
                is_first_sub_exp = i == 0
                is_last_sub_exp = i == len(sub_experience_list) - 1
                self._train_exp(
                    self.experience,
                    eval_streams,
                    is_first_sub_exp=is_first_sub_exp,
                    is_last_sub_exp=is_last_sub_exp,
                    **kwargs
                )

        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def _train_exp(
        self,
        experience: Experience,
        eval_streams=None,
        is_first_sub_exp=False,
        is_last_sub_exp=False,
        **kwargs
    ):
        """Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param is_first_sub_exp: whether the current sub-experience
            is the first sub-experience.
        :param is_last_sub_exp: whether the current sub-experience
            is the last sub-experience.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units) in the
        # first sub-experience
        if is_first_sub_exp:
            self.model = self.model_adaptation()
            self.make_optimizer()
        self._before_training_exp(**kwargs)
        self._before_training_epoch(**kwargs)

        # if self._stop_training:  # Early stopping
        #     self._stop_training = False
        # break

        for self.n_pass in range(self.num_passes):
            self.training_epoch(**kwargs)

        # if is_last_sub_exp:
        self._after_training_epoch(**kwargs)
        self._after_training_exp(**kwargs)

    def model_adaptation(self, model=None):
        """Adapts the model to the data from the current
           (full) experience.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        for module in model.modules():
            if isinstance(module, DynamicModule):
                module.adaptation(self.full_experience.dataset)
        return model.to(self.device)
