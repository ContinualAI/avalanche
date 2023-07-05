from avalanche.training import LearningWithoutForgetting
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class LwFPlugin(SupervisedPlugin):
    """Learning without Forgetting plugin.

    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """
        super().__init__()
        self.lwf = LearningWithoutForgetting(alpha, temperature)

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """

        strategy.loss += self.lwf(strategy.mb_x, strategy.mb_output, strategy.model)

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.lwf.update(strategy.experience, strategy.model)
