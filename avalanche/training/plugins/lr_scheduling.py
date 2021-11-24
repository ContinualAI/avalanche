from avalanche.training.plugins import StrategyPlugin


class LRSchedulerPlugin(StrategyPlugin):
    """ Learning Rate Scheduler Plugin.

    This plugin manages learning rate scheduling inside of a strategy using the
    PyTorch scheduler passed to the constructor. The step() method of the
    scheduler is called after each training epoch.
    """

    def __init__(self, scheduler, reset_scheduler=True, reset_lr=True):
        """
        Creates a ``LRSchedulerPlugin`` instance.

        :param scheduler: a learning rate scheduler that can be updated through
            a step() method and can be reset by setting last_epoch=0
        :param reset_scheduler: If True, the scheduler is reset at the end of
            the experience.
            Defaults to True.
        :param reset_lr: If True, the optimizer learning rate is reset to its
            original value.
            Default to True.
        """
        super().__init__()
        self.scheduler = scheduler
        self.reset_scheduler = reset_scheduler
        self.reset_lr = reset_lr

    def after_training_epoch(self, strategy, **kwargs):
        self.scheduler.step()

    def after_training_exp(self, strategy, **kwargs):
        param_groups = strategy.optimizer.param_groups
        base_lrs = self.scheduler.base_lrs

        if self.reset_lr:
            for group, lr in zip(param_groups, base_lrs):
                group['lr'] = lr

        if self.reset_scheduler:
            self.scheduler.last_epoch = 0
