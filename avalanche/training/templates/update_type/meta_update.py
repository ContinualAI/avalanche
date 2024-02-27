from avalanche.training.templates.strategy_mixin_protocol import (
    MetaLearningStrategyProtocol,
    TSGDExperienceType,
    TMBInput,
    TMBOutput,
)
from avalanche.training.utils import trigger_plugins


class MetaUpdate(MetaLearningStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Inner updates
            self._before_inner_updates(**kwargs)
            self._inner_updates(**kwargs)
            self._after_inner_updates(**kwargs)

            # Outer update
            self._before_outer_update(**kwargs)
            self._outer_update(**kwargs)
            self._after_outer_update(**kwargs)

            self.mb_output = self.forward()

            self._after_training_iteration(**kwargs)

    def _before_inner_updates(self, **kwargs):
        trigger_plugins(self, "before_inner_updates", **kwargs)

    def _inner_updates(self, **kwargs):
        raise NotImplementedError()

    def _after_inner_updates(self, **kwargs):
        trigger_plugins(self, "after_inner_updates", **kwargs)

    def _before_outer_update(self, **kwargs):
        trigger_plugins(self, "before_outer_update", **kwargs)

    def _outer_update(self, **kwargs):
        raise NotImplementedError()

    def _after_outer_update(self, **kwargs):
        trigger_plugins(self, "after_outer_update", **kwargs)


__all__ = ["MetaUpdate"]
