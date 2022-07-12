from typing import Iterable

from avalanche.benchmarks import CLExperience
from avalanche.models.dynamic_optimizers import reset_optimizer


class BatchObservation:
    def _train_exp(
        self, experience: CLExperience, eval_streams=None, **kwargs
    ):
        """Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Iterable):
                eval_streams[i] = [exp]
        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        reset_optimizer(self.optimizer, self.model)

    def maybe_adapt_model_and_make_optimizer(self):
        self.model = self.model_adaptation()
        self.make_optimizer()
