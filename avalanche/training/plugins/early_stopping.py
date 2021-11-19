import operator
from copy import deepcopy

from avalanche.training.plugins import StrategyPlugin


class EarlyStoppingPlugin(StrategyPlugin):
    """ Early stopping plugin.

    Simple plugin stopping the training when the accuracy on the
    corresponding validation metric stopped progressing for a few epochs.
    The state of the best model is saved after each improvement on the
    given metric and is loaded back into the model before stopping the
    training procedure.
    """
    def __init__(self, patience: int, val_stream_name: str,
                 metric_name: str = 'Top1_Acc_Stream', mode: str = 'max'):
        """
        :param patience: Number of epochs to wait before stopping the training.
        :param val_stream_name: Name of the validation stream to search in the
        metrics. The corresponding stream will be used to keep track of the
        evolution of the performance of a model.
        :param metric_name: The name of the metric to watch as it will be
        reported in the evaluator.
        :param mode: Must be "max" or "min". max (resp. min) means that the
        given metric should me maximized (resp. minimized).
        """
        super().__init__()
        self.val_stream_name = val_stream_name
        self.patience = patience
        self.metric_name = metric_name
        self.metric_key = f'{self.metric_name}/eval_phase/' \
                          f'{self.val_stream_name}'
        if mode not in ('max', 'min'):
            raise ValueError(f'Mode must be "max" or "min", got {mode}.')
        self.operator = operator.gt if mode == 'max' else operator.lt

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_epoch = None

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_epoch = None

    def before_training_epoch(self, strategy, **kwargs):
        self._update_best(strategy)
        if strategy.clock.train_exp_epochs - self.best_epoch >= self.patience:
            strategy.model.load_state_dict(self.best_state)
            strategy.stop_training()

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        val_acc = res.get(self.metric_key)
        if self.best_val is None or self.operator(val_acc, self.best_val):
            self.best_state = deepcopy(strategy.model.state_dict())
            self.best_val = val_acc
            self.best_epoch = strategy.clock.train_exp_epochs
