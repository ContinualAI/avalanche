from typing import Iterable

from avalanche.benchmarks import CLExperience
from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_model_adaptation


class BatchObservation:
    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model
        avalanche_model_adaptation(model, self.experience)
        return model.to(self.device)

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        reset_optimizer(self.optimizer, self.model)

    def check_model_and_optimizer(self):
        self.model = self.model_adaptation()
        self.make_optimizer()
