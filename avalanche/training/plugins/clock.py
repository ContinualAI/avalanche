from avalanche.training.plugins import StrategyPlugin


class Clock(StrategyPlugin):
    def __init__(self):
        """ Counter for strategy events. """
        super().__init__()
        # train
        self.train_iterations = 0
        """ Total number of training iterations. """

        self.train_exp_counter = 0
        """ Number of past training experiences. """

        self.train_exp_epochs = 0
        """ Number of training epochs for the current experience. """

        self.train_exp_iterations = 0
        """ Number of training iterations for the current experience. """

        self.train_epoch_iterations = 0
        """ Number of iteartions for the current epoch. """

        self.total_iterations = 0
        """ Total number of iterations in training and eval mode. """

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.train_exp_epochs = 0

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self.train_exp_iterations = 0

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self.train_epoch_iterations += 1
        self.train_iterations += 1
        self.total_iterations += 1

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.train_exp_counter += 1

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self.total_iterations += 1

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self.train_epoch_iterations = 0
