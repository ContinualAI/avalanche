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
        # We reset the optimizer's state after each experience if task
        # boundaries are given, otherwise it updates the optimizer only if
        # new parameters are added to the model after each adaptation step.

        # We assume the current experience is an OnlineCLExperience:
        if self.experience.access_task_boundaries:
            reset_optimizer(self.optimizer, self.model)

        else:
            update_optimizer(self.optimizer,
                             self.model_params_before_adaptation,
                             self.model.parameters(),
                             reset_state=False)

    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        # For training:
        if isinstance(self.experience, OnlineCLExperience):
            # If the strategy has access to task boundaries, adapt the model
            # for the whole origin experience to add the
            if self.experience.access_task_boundaries:
                avalanche_model_adaptation(model,
                                           self.experience.origin_experience)
            else:
                self.model_params_before_adaptation = list(model.parameters())
                avalanche_model_adaptation(model, self.experience)

        # For evaluation, the experience is not necessarily an online
        # experience:
        else:
            avalanche_model_adaptation(model, self.experience)

        return model.to(self.device)

    def check_model_and_optimizer(self):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online (sub-)stream,
        # then adapt the model with the full origin experience:
        if self.experience.access_task_boundaries:
            if self.experience.is_first_subexp:
                self.model = self.model_adaptation()
                self.make_optimizer()
        # Otherwise, adapt to the current sub-experience:
        else:
            self.model = self.model_adaptation()
            self.make_optimizer()
