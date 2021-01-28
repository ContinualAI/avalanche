################################################################################
# Copyright (c) 2020 ContinualAI Research                                      #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta, Andrea Cossu                                       #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################
from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from torch.utils.data import ConcatDataset

from avalanche.logging import default_logger
from avalanche.models.mobilenetv1 import MobilenetV1
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins import StrategyPlugin, \
    CWRStarPlugin, ReplayPlugin, GDumbPlugin, LwFPlugin, AGEMPlugin, \
    GEMPlugin, EWCPlugin, EvaluationPlugin, SynapticIntelligencePlugin


class Naive(BaseStrategy):
    """
    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class CWRStar(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 second_last_layer_name, num_classes=50,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ CWR* Strategy.
        This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param second_last_layer_name: name of the second to last layer
                (layer just before the classifier).
        :param num_classes: total number of classes.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """
        cwsp = CWRStarPlugin(model, second_last_layer_name, num_classes)
        if plugins is None:
            plugins = [cwsp]
        else:
            plugins.append(cwsp)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class Replay(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ Experience replay strategy. See ReplayPlugin for more details.
        This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """
        rp = ReplayPlugin(mem_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class GDumb(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ GDumb strategy. See GDumbPlugin for more details.
        This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        gdumb = GDumbPlugin(mem_size)
        if plugins is None:
            plugins = [gdumb]
        else:
            plugins.append(gdumb)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class Cumulative(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ Cumulative strategy. At each step,
            train model with data from all previous steps and current step.
            This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)

        self.dataset = None  # cumulative dataset

    def adapt_train_dataset(self, **kwargs):

        super().adapt_train_dataset(**kwargs)

        if self.dataset is None:
            self.dataset = self.current_data
        else:
            self.dataset = ConcatDataset([self.dataset, self.current_data])
            self.current_data = self.dataset


class LwF(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 alpha: Union[float, Sequence[float]], temperature: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ Learning without Forgetting strategy. 
            See LwF plugin for details.
            This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each step.
        :param temperature: softmax temperature for distillation
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        lwf = LwFPlugin(alpha, temperature)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class AGEM(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 patterns_per_step: int, sample_size: int = 64,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ Average Gradient Episodic Memory (A-GEM) strategy. 
            See AGEM plugin for details.
            This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_step: number of patterns per step in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.        
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        agem = AGEMPlugin(patterns_per_step, sample_size)
        if plugins is None:
            plugins = [agem]
        else:
            plugins.append(agem)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class GEM(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 patterns_per_step: int, memory_strength: float = 0.5,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ Gradient Episodic Memory (GEM) strategy. 
            See GEM plugin for details.
            This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_step: number of patterns per step in the memory
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        gem = GEMPlugin(patterns_per_step, memory_strength)
        if plugins is None:
            plugins = [gem]
        else:
            plugins.append(gem)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class EWC(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 ewc_lambda: float, mode: str = 'standard',
                 decay_factor: Optional[float] = None,
                 keep_importance_data: bool = False,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ Elastic Weight Consolidation (EWC) strategy.
            See EWC plugin for details.
            This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `standard` to keep a separate penalty for each previous 
               step. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`. 
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)


class AR1(BaseStrategy):
    """
    TODO: doc
    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(self, criterion=None, lr: float = 0.001,
                 init_update_rate: float = 0.01, inc_update_rate=0.00005,
                 max_r_max=1.25, max_d_max=0.5, inc_step=4.1e-05, momentum=0.9,
                 l2=0.0005, rm_sz: int = 1500,
                 freeze_below_layer: str = "lat_features.19.bn.beta",
                 latent_layer_num: int = 19, ewc_lambda: float = 0,
                 train_mb_size: int = 128,
                 train_epochs: int = 128, test_mb_size: int = 128,
                 device=None,
                 plugins: Optional[Sequence[StrategyPlugin]] = None):
        """
        TODO: doc
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param lr:
        :param criterion: The loss criterion to use. Defaults to None, in which
            case the cross entropy loss is used.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        """

        if plugins is None:
            plugins = []

        # Model setup
        model = MobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
        replace_bn_with_brn(
            model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
            max_r_max=max_r_max, max_d_max=max_d_max
        )

        model.saved_weights = {}
        model.past_j = {i: 0 for i in range(50)}
        model.cur_j = {i: 0 for i in range(50)}

        if ewc_lambda != 0:
            plugins.append(SynapticIntelligencePlugin(ewc_lambda))

        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=l2)
        if criterion is None:
            criterion = CrossEntropyLoss()

        self.ewc_lambda = ewc_lambda
        self.freeze_below_layer = freeze_below_layer
        self.rm_sz = rm_sz
        self.inc_update_rate = inc_update_rate
        self.max_r_max = max_r_max
        self.max_d_max = max_d_max
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.rm = None

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins)

        # TODO: implement callbacks


__all__ = ['Naive', 'CWRStar', 'Replay', 'GDumb', 'Cumulative', 'LwF', 'AGEM',
           'GEM', 'EWC']
