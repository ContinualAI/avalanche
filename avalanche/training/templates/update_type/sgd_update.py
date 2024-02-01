from avalanche.training.templates.strategy_mixin_protocol import (
    SGDStrategyProtocol,
    TSGDExperienceType,
    TMBInput,
    TMBOutput,
)


class SGDUpdate(SGDStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]):
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

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)


__all__ = ["SGDUpdate"]
