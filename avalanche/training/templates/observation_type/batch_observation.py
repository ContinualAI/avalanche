from typing import final

from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_model_adaptation


class BatchObservation:

    @final
    def model_adaptation(self, model=None):
        """Adapts the model to the current data.
        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        This method should not be overridden by child classes.
        Consider overriding :meth:`_model_adaptation` instead.
        """
        with self.use_local_model():
            return self._model_adaptation(model=model)

    def _model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model
        avalanche_model_adaptation(model, self.experience)

        return model.to(self.device)

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experience to configure the optimizer.
        """
        # we reset the optimizer's state after each experience.
        # This allows to add new parameters (new heads) and
        # freezing old units during the model's adaptation phase.
        reset_optimizer(self.optimizer, self.model)

    def check_model_and_optimizer(self):
        with self.use_local_model():
            self.model = self.model_adaptation()
            self.model = self.wrap_distributed_model(self.model)
        self.make_optimizer()


__all__ = [
    'BatchObservation'
]
