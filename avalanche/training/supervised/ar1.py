import warnings
from typing import Optional, Sequence, List

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.batchnorm import _NormBase
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.models import MobilenetV1
from avalanche.models.batch_renorm import BatchRenorm2D
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CWRStarPlugin,
)
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.utils import (
    replace_bn_with_brn,
    get_last_fc_layer,
    freeze_up_to,
    change_brn_pars,
    examples_per_class,
    LayerAndParameter,
)
from avalanche.training.plugins.evaluation import default_evaluator


class AR1(SupervisedTemplate):
    """AR1 with Latent Replay.

    This implementations allows for the use of both Synaptic Intelligence and
    Latent Replay to protect the lower level of the model from forgetting.

    While the original papers show how to use those two techniques in a mutual
    exclusive way, this implementation allows for the use of both of them
    concurrently. This behaviour is controlled by passing proper constructor
    arguments).
    """

    def __init__(
        self,
        criterion=None,
        lr: float = 0.001,
        momentum=0.9,
        l2=0.0005,
        train_epochs: int = 4,
        init_update_rate: float = 0.01,
        inc_update_rate=0.00005,
        max_r_max=1.25,
        max_d_max=0.5,
        inc_step=4.1e-05,
        rm_sz: int = 1500,
        freeze_below_layer: str = "lat_features.19.bn.beta",
        latent_layer_num: int = 19,
        ewc_lambda: float = 0,
        train_mb_size: int = 128,
        eval_mb_size: int = 128,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
    ):
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
        :param eval_mb_size: The eval minibatch size. Defaults to 128.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        warnings.warn(
            "The AR1 strategy implementation is in an alpha stage "
            "and is not perfectly aligned with the paper "
            "implementation. Please use at your own risk!"
        )

        if plugins is None:
            plugins = []

        # Model setup
        model = MobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
        replace_bn_with_brn(
            model,
            momentum=init_update_rate,
            r_d_max_inc_step=inc_step,
            max_r_max=max_r_max,
            max_d_max=max_d_max,
        )

        fc_name, fc_layer = get_last_fc_layer(model)

        if ewc_lambda != 0:
            # Synaptic Intelligence is not applied to the last fully
            # connected layer (and implicitly to "freeze below" ones.
            plugins.append(
                SynapticIntelligencePlugin(
                    ewc_lambda, excluded_parameters=[fc_name]
                )
            )

        self.cwr_plugin = CWRStarPlugin(
            model, cwr_layer_name=fc_name, freeze_remaining_model=False
        )
        plugins.append(self.cwr_plugin)

        optimizer = SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=l2
        )

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
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

    def _before_training_exp(self, **kwargs):
        self.model.eval()
        self.model.end_features.train()
        self.model.output.train()

        if self.clock.train_exp_counter > 0:
            # In AR1 batch 0 is treated differently as the feature extractor is
            # left more free to learn.
            # This if is executed for batch > 0, in which we freeze layers
            # below "self.freeze_below_layer" (which usually is the latent
            # replay layer!) and we also change the parameters of BatchReNorm
            # layers to a more conservative configuration.

            # "freeze_up_to" will freeze layers below "freeze_below_layer"
            # Beware that Batch ReNorm layers are not frozen!
            freeze_up_to(
                self.model,
                freeze_until_layer=self.freeze_below_layer,
                layer_filter=AR1.filter_bn_and_brn,
            )

            # Adapt the parameters of BatchReNorm layers
            change_brn_pars(
                self.model,
                momentum=self.inc_update_rate,
                r_d_max_inc_step=0,
                r_max=self.max_r_max,
                d_max=self.max_d_max,
            )

            # Adapt the model and optimizer
            self.model = self.model.to(self.device)
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.l2,
            )

        # super()... will run S.I. and CWR* plugin callbacks
        super()._before_training_exp(**kwargs)

        # Update cur_j of CWR* to consider latent patterns
        if self.clock.train_exp_counter > 0:
            for class_id, count in examples_per_class(self.rm[1]).items():
                self.model.cur_j[class_id] += count
            self.cwr_plugin.cur_class = [
                cls
                for cls in set(self.model.cur_j.keys())
                if self.model.cur_j[cls] > 0
            ]
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

        if self.clock.train_exp_counter > 0:
            train_patterns = len(self.adapted_dataset)
            current_batch_mb_size = train_patterns // (
                (train_patterns + self.rm_sz) // self.train_mb_size
            )

        current_batch_mb_size = max(1, current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)

        # AR1 only supports SIT scenarios (no task labels).
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=current_batch_mb_size,
            shuffle=shuffle,
        )

    def training_epoch(self, **kwargs):
        for mb_it, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            if self.clock.train_exp_counter > 0:
                lat_mb_x = self.rm[0][
                    mb_it
                    * self.replay_mb_size : (mb_it + 1)
                    * self.replay_mb_size
                ]
                lat_mb_x = lat_mb_x.to(self.device)
                lat_mb_y = self.rm[1][
                    mb_it
                    * self.replay_mb_size : (mb_it + 1)
                    * self.replay_mb_size
                ]
                lat_mb_y = lat_mb_y.to(self.device)
                self.mbatch[1] = torch.cat((self.mb_y, lat_mb_y), 0)
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0), which
            # means that lat_acts.shape[0] == self.mb_x[0].
            self._before_forward(**kwargs)
            self.mb_output, lat_acts = self.model(
                self.mb_x, latent_input=lat_mb_x, return_lat_acts=True
            )

            if self.clock.train_exp_epochs == 0:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()
                if mb_it == 0:
                    self.cur_acts = lat_acts
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
            self._after_forward(**kwargs)

            # Loss & Backward
            # We don't need to handle latent replay, as self.mb_y already
            # contains both current and replay labels.
            self.loss = self._criterion(self.mb_output, self.mb_y)
            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        h = min(
            self.rm_sz // (self.clock.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        curr_data = self.experience.dataset
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        rm_add_y = torch.tensor(
            [curr_data.targets[idx_cur] for idx_cur in idxs_cur]
        )

        rm_add = [self.cur_acts[idxs_cur], rm_add_y]

        # replace patterns in random memory
        if self.clock.train_exp_counter == 0:
            self.rm = rm_add
        else:
            idxs_2_replace = torch.randperm(self.rm[0].size(0))[:h]
            for j, idx in enumerate(idxs_2_replace):
                idx = int(idx)
                self.rm[0][idx] = rm_add[0][j]
                self.rm[1][idx] = rm_add[1][j]

        self.cur_acts = None

        # Runs S.I. and CWR* plugin callbacks
        super()._after_training_exp(**kwargs)

    @staticmethod
    def filter_bn_and_brn(param_def: LayerAndParameter):
        return not isinstance(param_def.layer, (_NormBase, BatchRenorm2D))
