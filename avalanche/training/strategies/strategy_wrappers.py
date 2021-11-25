################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta, Andrea Cossu                                       #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins import StrategyPlugin, CWRStarPlugin, \
    ReplayPlugin, GDumbPlugin, LwFPlugin, AGEMPlugin, GEMPlugin, EWCPlugin, \
    EvaluationPlugin, SynapticIntelligencePlugin, CoPEPlugin, \
    GSS_greedyPlugin, LFLPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy


class Naive(BaseStrategy):
    """ Naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class PNNStrategy(BaseStrategy):
    """ Progressive Neural Network strategy. """

    def __init__(self, num_layers: int, in_features: int,
                 hidden_features_per_column: int,
                 lr: float, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, adapter='mlp',
                 criterion=CrossEntropyLoss(),
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Progressive Neural Network strategy.

        :param num_layers: Number of layers for the PNN architecture.
        :param in_features: Number of input features.
        :param hidden_features_per_column: Number of hidden units for
            each column of the PNN architecture.
        :param lr: learning rate
        :param momentum: momentum factor (default: 0)
        :param weight_decay: weight decay (L2 penalty) (default: 0)
        :param dampening: dampening for momentum (default: 0)
        :param nesterov: enables Nesterov momentum (default: False)
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        model = PNN(
            num_layers=num_layers,
            in_features=in_features,
            hidden_features_per_column=hidden_features_per_column,
            adapter=adapter
        )
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum,
                        weight_decay=weight_decay, dampening=dampening,
                        nesterov=nesterov)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class CWRStar(BaseStrategy):
    """ CWR* Strategy. """
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 cwr_layer_name: str, train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ 

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param cwr_layer_name: name of the CWR layer. Defaults to None, which
            means that the last fully connected layer will be used.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        cwsp = CWRStarPlugin(model, cwr_layer_name, freeze_remaining_model=True)
        if plugins is None:
            plugins = [cwsp]
        else:
            plugins.append(cwsp)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class Replay(BaseStrategy):
    """ Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities.
    """
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        rp = ReplayPlugin(mem_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, 
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, 
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class GSS_greedy(BaseStrategy):
    """ Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities. 
    """
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200, mem_strength=1, input_size=[],
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param n: memory random set size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        rp = GSS_greedyPlugin(mem_size=mem_size,
                              mem_strength=mem_strength, input_size=input_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class GDumb(BaseStrategy):
    """ GDumb strategy.

    See GDumbPlugin for more details.
    This strategy does not use task identities.
    """
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param mem_size: replay buffer size.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        gdumb = GDumbPlugin(mem_size)
        if plugins is None:
            plugins = [gdumb]
        else:
            plugins.append(gdumb)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class LwF(BaseStrategy):
    """ Learning without Forgetting (LwF) strategy.

    See LwF plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 alpha: Union[float, Sequence[float]], temperature: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        lwf = LwFPlugin(alpha, temperature)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class AGEM(BaseStrategy):
    """ Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """
    
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 patterns_per_exp: int, sample_size: int = 64,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        agem = AGEMPlugin(patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [agem]
        else:
            plugins.append(agem)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class GEM(BaseStrategy):
    """ Gradient Episodic Memory (GEM) strategy.

    See GEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 patterns_per_exp: int, memory_strength: float = 0.5,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        gem = GEMPlugin(patterns_per_exp, memory_strength)
        if plugins is None:
            plugins = [gem]
        else:
            plugins.append(gem)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class EWC(BaseStrategy):
    """ Elastic Weight Consolidation (EWC) strategy.

    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 ewc_lambda: float, mode: str = 'separate',
                 decay_factor: Optional[float] = None,
                 keep_importance_data: bool = False,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
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
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        ewc = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class SynapticIntelligence(BaseStrategy):
    """ Synaptic Intelligence strategy.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper
    "Continuous Learning in Single-Incremental-Task Scenarios"
    (https://arxiv.org/abs/1806.08568)

    The original implementation has been proposed in the paper
    "Continual Learning Through Synaptic Intelligence"
    (https://arxiv.org/abs/1703.04200).

    The Synaptic Intelligence regularization can also be used in a different
    strategy by applying the :class:`SynapticIntelligencePlugin` plugin.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 si_lambda: Union[float, Sequence[float]],
                 eps: float = 0.0000001, train_mb_size: int = 1,
                 train_epochs: int = 1, eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):
        """ Init.

        Creates an instance of the Synaptic Intelligence strategy.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param si_lambda: Synaptic Intelligence lambda term.
            If list, one lambda for each experience. If the list has less
            elements than the number of experiences, last lambda will be
            used for the remaining experiences.
        :param eps: Synaptic Intelligence damping parameter.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        if plugins is None:
            plugins = []

        # This implementation relies on the S.I. Plugin, which contains the
        # entire implementation of the strategy!
        plugins.append(SynapticIntelligencePlugin(si_lambda=si_lambda, eps=eps))

        super(SynapticIntelligence, self).__init__(
            model, optimizer, criterion, train_mb_size, train_epochs,
            eval_mb_size, device=device, plugins=plugins, evaluator=evaluator,
            eval_every=eval_every
        )


class CoPE(BaseStrategy):
    """ Continual Prototype Evolution strategy.

    See CoPEPlugin for more details.
    This strategy does not use task identities during training.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 mem_size: int = 200, n_classes: int = 10, p_size: int = 100,
                 alpha: float = 0.99, T: float = 0.1,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger,
                 eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: Loss criterion to use. Standard overwritten by
        PPPloss (see CoPEPlugin).
        :param mem_size: replay buffer size.
        :param n_classes: total number of classes that will be encountered. This
        is used to output predictions for all classes, with zero probability
        for unseen classes.
        :param p_size: The prototype size, which equals the feature size of the
        last layer.
        :param alpha: The momentum for the exponentially moving average of the
        prototypes.
        :param T: The softmax temperature, used as a concentration parameter.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """
        copep = CoPEPlugin(mem_size, n_classes, p_size, alpha, T)
        if plugins is None:
            plugins = [copep]
        else:
            plugins.append(copep)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


class LFL(BaseStrategy):
    """ Less Forgetful Learning strategy.

    See LFL plugin for details.
    Refer Paper: https://arxiv.org/pdf/1607.00122.pdf
    This strategy does not use task identities.
    """
    
    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 lambda_e: Union[float, Sequence[float]],
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param lambda_e: euclidean loss hyper parameter. It can be either a
                float number or a list containing lambda_e for each experience.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that 
            `eval` is called every `eval_every` epochs and at the end of the 
            learning experience.
        """

        lfl = LFLPlugin(lambda_e)
        if plugins is None:
            plugins = [lfl]
        else:
            plugins.append(lfl)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)


__all__ = [
    'Naive',
    'PNNStrategy',
    'CWRStar',
    'Replay',
    'GDumb',
    'LwF',
    'AGEM',
    'GEM',
    'EWC',
    'SynapticIntelligence',
    'GSS_greedy',
    'CoPE',
    'LFL'
]
