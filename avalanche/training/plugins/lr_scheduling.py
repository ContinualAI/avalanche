from avalanche.training.plugins import StrategyPlugin


class LRSchedulerPlugin(StrategyPlugin):
    def __init__(self, scheduler, reset_scheduler=True, reset_lr=True):
        super().__init__()
        self.scheduler = scheduler
        self.reset_scheduler = reset_scheduler
        self.reset_lr = reset_lr

    def after_training_epoch(self, strategy, **kwargs):
        self.scheduler.step()

    def after_training_exp(self, strategy, **kwargs):
        if self.reset_lr:
            for group, lr in zip(strategy.optimizer.param_groups, self.scheduler.base_lrs):
                group['lr'] = lr

        if self.reset_scheduler:
            self.scheduler.last_epoch = 0
