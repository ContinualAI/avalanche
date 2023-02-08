from typing import Iterable

from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.dynamic_optimizers import update_optimizer
from avalanche.models.utils import avalanche_model_adaptation


class OnlineObservation:
    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experience to configure the optimizer.
        """
        # We reset the optimizer's state after an experience only if task
        # boundaries are given and the current experience is the first
        # sub-experience or original experience.
        reset_optimizer(self.optimizer, self.model)

    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        # For training:
        if isinstance(self.experience, OnlineCLExperience):
            # If the strategy has access to task boundaries, adapt the model
            # using the origin_experience.
            avalanche_model_adaptation(model, self.experience.origin_experience)

        # For evaluation, the experience is not necessarily an online
        # experience:
        else:
            avalanche_model_adaptation(model, self.experience)

        return model.to(self.device)

    def check_model_and_optimizer(self):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        if self.experience.access_task_boundaries:
            if self.experience.is_first_subexp:
                self.model = self.model_adaptation()
                self.make_optimizer()
