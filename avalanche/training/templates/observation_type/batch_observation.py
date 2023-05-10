from avalanche.models.dynamic_optimizers import reset_optimizer
from avalanche.models.utils import avalanche_model_adaptation
from avalanche.training.templates.strategy_mixin_protocol import \
    SGDStrategyProtocol
from avalanche.benchmarks import OnlineCLExperience
from avalanche.models.dynamic_optimizers import update_optimizer


class BatchObservation(SGDStrategyProtocol):
    def model_adaptation(self, model=None):
        """Adapts the model to the current data.

        Calls the :class:`~avalanche.models.DynamicModule`s adaptation.
        """
        if model is None:
            model = self.model

        assert self.experience is not None

        # For training:
        if isinstance(self.experience, OnlineCLExperience) and self.is_training:
            # If the strategy has access to task boundaries, adapt the model
            # for the whole origin experience to add the
            if self.experience.access_task_boundaries:
                avalanche_model_adaptation(model,
                                           self.experience.origin_experience)
            else:
                avalanche_model_adaptation(model, self.experience)

        # For evaluation, the experience is not necessarily an online
        # experience:
        else:
            avalanche_model_adaptation(model, self.experience)

        return model.to(self.device)

    def make_optimizer(self):
        """Optimizer initialization.

        Called before each training experiene to configure the optimizer.
        """
        if isinstance(self.experience, OnlineCLExperience):
            if self.experience.access_task_boundaries:
                assert self.experience.is_first_subexp
                reset_optimizer(self.optimizer, self.model)

            # Otherwise, update the optimizer
            else:
                update_optimizer(
                    self.optimizer,
                    self.model_params_before_adaptation,  # type: ignore
                    self.model.parameters(),
                    reset_state=False)
        else:
            reset_optimizer(self.optimizer, self.model)

    def check_model_and_optimizer(self):
        # If strategy has access to the task boundaries, and the current
        # sub-experience is the first sub-experience in the online stream,
        # then adapt the model with the full origin experience:
        assert self.experience is not None
        if isinstance(self.experience, OnlineCLExperience):
            if self.experience.access_task_boundaries:
                if self.experience.is_first_subexp:
                    self.model = self.model_adaptation()
                    self.make_optimizer()
        else:
            self.model = self.model_adaptation()
            self.make_optimizer()


__all__ = [
    'BatchObservation'
]
