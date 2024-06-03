from avalanche.models import avalanche_forward
from avalanche.training.templates.strategy_mixin_protocol import (
    SupervisedStrategyProtocol,
    TSGDExperienceType,
    TMBInput,
    TMBOutput,
)


# Types are perfectly ok for MyPy
# Also confirmed here: https://stackoverflow.com/a/70907644
# PyLance just does not understand it
class SupervisedProblem(
    SupervisedStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mb_x(self):
        """Current mini-batch input."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3, "Task label not found."
        return mbatch[-1]

    def criterion(self):
        """Loss function for supervised problems."""
        return self._criterion(self.mb_output, self.mb_y)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        # use task-aware forward only for task-aware benchmarks
        if hasattr(self.experience, "task_labels"):
            return avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        else:
            return self.model(self.mb_x)

    def _unpack_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        mbatch = self.mbatch
        assert mbatch is not None

        if isinstance(mbatch, tuple):
            mbatch = list(mbatch)
            self.mbatch = mbatch

        for i in range(len(mbatch)):
            mbatch[i] = mbatch[i].to(self.device, non_blocking=True)  # type: ignore


__all__ = ["SupervisedProblem"]
