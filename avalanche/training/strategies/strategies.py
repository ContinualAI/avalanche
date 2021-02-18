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
import warnings
from typing import Optional, Sequence, List, Union

import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.nn.modules.batchnorm import _NormBase
from torch.optim import Optimizer, SGD
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from avalanche.logging import default_logger
from avalanche.models.batch_renorm import BatchRenorm2D
from avalanche.models.mobilenetv1 import MobilenetV1
from avalanche.training.plugins import StrategyPlugin, CWRStarPlugin,\
    ReplayPlugin, GDumbPlugin, LwFPlugin, AGEMPlugin, GEMPlugin, EWCPlugin, \
    EvaluationPlugin, SynapticIntelligencePlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.utils import get_last_fc_layer, replace_bn_with_brn, \
    change_brn_pars, freeze_up_to, LayerAndParameter, examples_per_class


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
                 cwr_layer_name: str, train_mb_size: int = 1,
                 train_epochs: int = 1, test_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """ CWR* Strategy.
        This strategy does not use task identities.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param cwr_layer_name: name of the CWR layer. Defaults to None, which
            means that the last fully connected layer will be used.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param test_mb_size: The test minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """
        cwsp = CWRStarPlugin(model, cwr_layer_name, freeze_remaining_model=True)
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

        self.dataset = {}  # cumulative dataset

    def adapt_train_dataset(self, **kwargs):

        super().adapt_train_dataset(**kwargs)

        curr_task_id = self.step_info.task_label
        curr_data = self.step_info.dataset
        if curr_task_id in self.dataset:
            cat_data = ConcatDataset([self.dataset[curr_task_id],
                                      curr_data])
            self.dataset[curr_task_id] = cat_data
        else:
            self.dataset[curr_task_id] = curr_data
        self.adapted_dataset = self.dataset


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
                 ewc_lambda: float, mode: str = 'separate',
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


class SynapticIntelligence(BaseStrategy):
    """
    The Synaptic Intelligence strategy.

    This is the Synaptic Intelligence PyTorch implementation of the
    algorithm described in the paper "Continual Learning Through Synaptic
    Intelligence" (https://arxiv.org/abs/1703.04200).

    The Synaptic Intelligence regularization can also be used in a different
    strategy by applying the :class:`SynapticIntelligencePlugin` plugin.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 si_lambda: float, train_mb_size: int = 1,
                 train_epochs: int = 1, test_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger):
        """
        Creates an instance of the Synaptic Intelligence strategy.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param si_lambda: Synaptic Intelligence lambda term.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param test_mb_size: mini-batch size for test.
        :param device: PyTorch device to run the model.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """
        if plugins is None:
            plugins = []

        # This implementation relies on the S.I. Plugin, which contains the
        # entire implementation of the strategy!
        plugins.append(SynapticIntelligencePlugin(si_lambda))

        super(SynapticIntelligence, self).__init__(
            model, optimizer, criterion, train_mb_size, train_epochs,
            test_mb_size, device=device, plugins=plugins, evaluator=evaluator)


class AR1(BaseStrategy):
    """
    The AR1 strategy with Latent Replay.

    This implementations allows for the use of both Synaptic Intelligence and
    Latent Replay to protect the lower level of the model from forgetting.

    While the original papers show how to use those two techniques in a mutual
    exclusive way, this implementation allows for the use of both of them
    concurrently. This behaviour is controlled by passing proper constructor
    arguments).
    """

    def __init__(self, criterion=None, lr: float = 0.001, momentum=0.9,
                 l2=0.0005, train_epochs: int = 4,
                 init_update_rate: float = 0.01,
                 inc_update_rate=0.00005,
                 max_r_max=1.25, max_d_max=0.5, inc_step=4.1e-05,
                 rm_sz: int = 1500,
                 freeze_below_layer: str = "lat_features.19.bn.beta",
                 latent_layer_num: int = 19, ewc_lambda: float = 0,
                 train_mb_size: int = 128, test_mb_size: int = 128, device=None,
                 plugins: Optional[Sequence[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger):
        """
        Creates an instance of the AR1 strategy.

        :param criterion: The loss criterion to use. Defaults to None, in which
            case the cross entropy loss is used.
        :param lr: The learning rate (SGD optimizer).
        :param momentum: The momentum (SGD optimizer).
        :param l2: The L2 penalty used for weight decay.
        :param train_epochs: The number of training epochs. Defaults to 4.
        :param init_update_rate: The initial update rate of BatchReNorm layers.
        :param inc_update_rate: The incremental update rate of BatchReNorm
            layers.
        :param max_r_max: The maximum r value of BatchReNorm layers.
        :param max_d_max: The maximum d value of BatchReNorm layers.
        :param inc_step: The incremental step of r and d values of BatchReNorm
            layers.
        :param rm_sz: The size of the replay buffer. The replay buffer is shared
            across classes. Defaults to 1500.
        :param freeze_below_layer: A string describing the name of the layer
            to use while freezing the lower (nearest to the input) part of the
            model. The given layer is not frozen (exclusive).
        :param latent_layer_num: The number of the layer to use as the Latent
            Replay Layer. Usually this is the same of `freeze_below_layer`.
        :param ewc_lambda: The Synaptic Intelligence lambda term. Defaults to
            0, which means that the Synaptic Intelligence regularization
            will not be applied.
        :param train_mb_size: The train minibatch size. Defaults to 128.
        :param test_mb_size: The test minibatch size. Defaults to 128.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        """

        warnings.warn("The AR1 strategy implementation is in an alpha stage "
                      "and is not perfectly aligned with the paper "
                      "implementation. Please use at your own risk!")

        if plugins is None:
            plugins = []

        # Model setup
        model = MobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
        replace_bn_with_brn(
            model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
            max_r_max=max_r_max, max_d_max=max_d_max)

        fc_name, fc_layer = get_last_fc_layer(model)

        if ewc_lambda != 0:
            # Synaptic Intelligence is not applied to the last fully
            # connected layer (and implicitly to "freeze below" ones.
            plugins.append(SynapticIntelligencePlugin(
                ewc_lambda, excluded_parameters=[fc_name]))

        self.cwr_plugin = CWRStarPlugin(model, cwr_layer_name=fc_name,
                                        freeze_remaining_model=False)
        plugins.append(self.cwr_plugin)

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
        self.cur_acts: Optional[Tensor] = None
        self.replay_mb_size = 0

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            test_mb_size=test_mb_size, device=device, plugins=plugins,
            evaluator=evaluator)

    def before_training_step(self, **kwargs):
        self.model.eval()
        self.model.end_features.train()
        self.model.output.train()

        if self.training_step_counter > 0:
            # In AR1 batch 0 is treated differently as the feature extractor is
            # left more free to learn.
            # This if is executed for batch > 0, in which we freeze layers
            # below "self.freeze_below_layer" (which usually is the latent
            # replay layer!) and we also change the parameters of BatchReNorm
            # layers to a more conservative configuration.

            # "freeze_up_to" will freeze layers below "freeze_below_layer"
            # Beware that Batch ReNorm layers are not frozen!
            freeze_up_to(self.model, freeze_until_layer=self.freeze_below_layer,
                         layer_filter=AR1.filter_bn_and_brn)

            # Adapt the parameters of BatchReNorm layers
            change_brn_pars(self.model, momentum=self.inc_update_rate,
                            r_d_max_inc_step=0, r_max=self.max_r_max,
                            d_max=self.max_d_max)

            # Adapt the model and optimizer
            self.model = self.model.to(self.device)
            self.optimizer = SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum,
                weight_decay=self.l2)

        # super()... will run S.I. and CWR* plugin callbacks
        super().before_training_step(**kwargs)

        # Update cur_j of CWR* to consider latent patterns
        if self.training_step_counter > 0:
            for class_id, count in examples_per_class(self.rm[1]).items():
                self.model.cur_j[class_id] += count
            self.cwr_plugin.cur_class = [
                cls for cls in set(self.model.cur_j.keys())
                if self.model.cur_j[cls] > 0]
            self.cwr_plugin.reset_weights(self.cwr_plugin.cur_class)

    def make_train_dataloader(self, num_workers=0, shuffle=True, **kwargs):
        """
        Called after the dataset instantiation. Initialize the data loader.

        For AR1 a "custom" dataloader is used: instead of using
        `self.train_mb_size` as the batch size, the data loader batch size will
        be computed ad `self.train_mb_size - latent_mb_size`. `latent_mb_size`
        is in turn computed as:

        `
        len(train_dataset) // ((len(train_dataset) + len(replay_buffer)
        // self.train_mb_size)
        `

        so that the number of iterations required to run an epoch on the current
        batch is equal to the number of iterations required to run an epoch
        on the replay buffer.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """

        current_batch_mb_size = self.train_mb_size

        if self.training_step_counter > 0:
            train_patterns = len(self.adapted_dataset)
            current_batch_mb_size = train_patterns // (
                    (train_patterns + self.rm_sz) // self.train_mb_size)

        current_batch_mb_size = max(1, current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)

        # AR1 only supports SIT scenarios (no task labels).
        assert len(self.adapted_dataset.keys()) == 1
        curr_data = list(self.adapted_dataset.values())[0]
        self.current_dataloader = DataLoader(
            curr_data, num_workers=num_workers,
            batch_size=current_batch_mb_size, shuffle=shuffle)

    def training_epoch(self, **kwargs):
        for self.mb_it, (self.mb_x, self.mb_y) in \
                enumerate(self.current_dataloader):
            self.before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.mb_x = self.mb_x.to(self.device)
            self.mb_y = self.mb_y.to(self.device)

            if self.training_step_counter > 0:
                lat_mb_x = self.rm[0][self.mb_it * self.replay_mb_size:
                                      (self.mb_it + 1) * self.replay_mb_size]
                lat_mb_x = lat_mb_x.to(self.device)
                lat_mb_y = self.rm[1][self.mb_it * self.replay_mb_size:
                                      (self.mb_it + 1) * self.replay_mb_size]
                lat_mb_y = lat_mb_y.to(self.device)
                self.mb_y = torch.cat((self.mb_y, lat_mb_y), 0)
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0), which
            # means that lat_acts.shape[0] == self.mb_x[0].
            self.before_forward(**kwargs)
            self.logits, lat_acts = self.model(
                self.mb_x, latent_input=lat_mb_x, return_lat_acts=True)

            if self.epoch == 0:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()
                if self.mb_it == 0:
                    self.cur_acts = lat_acts
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
            self.after_forward(**kwargs)

            # Loss & Backward
            # We don't need to handle latent replay, as self.mb_y already
            # contains both current and replay labels.
            self.loss = self.criterion(self.logits, self.mb_y)
            self.before_backward(**kwargs)
            self.loss.backward()
            self.after_backward(**kwargs)

            # Optimization step
            self.before_update(**kwargs)
            self.optimizer.step()
            self.after_update(**kwargs)

            self.after_training_iteration(**kwargs)

    def after_training_step(self, **kwargs):
        h = min(self.rm_sz // (self.training_step_counter + 1),
                self.cur_acts.size(0))

        curr_data = self.step_info.dataset
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        rm_add_y = torch.tensor(
            [curr_data.targets[idx_cur] for idx_cur in idxs_cur])

        rm_add = [self.cur_acts[idxs_cur], rm_add_y]

        # replace patterns in random memory
        if self.training_step_counter == 0:
            self.rm = rm_add
        else:
            idxs_2_replace = torch.randperm(self.rm[0].size(0))[:h]
            for j, idx in enumerate(idxs_2_replace):
                idx = int(idx)
                self.rm[0][idx] = rm_add[0][j]
                self.rm[1][idx] = rm_add[1][j]

        self.cur_acts = None

        # Runs S.I. and CWR* plugin callbacks
        super().after_training_step(**kwargs)

    @staticmethod
    def filter_bn_and_brn(param_def: LayerAndParameter):
        return not isinstance(param_def.layer, (_NormBase, BatchRenorm2D))


__all__ = [
    'Naive',
    'CWRStar',
    'Replay',
    'GDumb',
    'Cumulative',
    'LwF',
    'AGEM',
    'GEM',
    'EWC',
    'SynapticIntelligence',
    'AR1'
]
