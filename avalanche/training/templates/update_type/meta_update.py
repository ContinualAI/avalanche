
class MetaUpdate:
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
            self.loss = 0

            # Fast updates
            self._before_fast_update(**kwargs)
            self._after_fast_updates(**kwargs)

            # Slow updates
            self._before_slow_update(**kwargs)
            self._after_slow_updates(**kwargs)

            self._after_training_iteration(**kwargs)
