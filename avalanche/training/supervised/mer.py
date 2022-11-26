from typing import Sequence, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from copy import deepcopy

from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import OnlineSupervisedMetaLearningTemplate
from avalanche.models.utils import avalanche_forward
from avalanche.training.storage_policy import ReservoirSamplingBuffer


class MERBuffer:
    def __init__(self, max_buffer_size=100, buffer_mb_size=10,
                 device=torch.device("cpu")):
        self.storage_policy = ReservoirSamplingBuffer(max_size=max_buffer_size)
        self.buffer_mb_size = buffer_mb_size
        self.device = device

    def update(self, strategy):
        self.storage_policy.update(strategy)

    def __len__(self):
        return len(self.storage_policy.buffer)

    def get_batch(self, x, y, t):
        if len(self) == 0:
            return x, y, t

        bsize = min(len(self), self.buffer_mb_size)
        rnd_ind = torch.randperm(len(self))[:bsize]
        buff_x = torch.cat([self.storage_policy.buffer[i][0].unsqueeze(0)
                            for i in rnd_ind]).to(self.device)
        buff_y = torch.LongTensor([self.storage_policy.buffer[i][1]
                                   for i in rnd_ind]).to(self.device)
        buff_t = torch.LongTensor([self.storage_policy.buffer[i][2]
                                   for i in rnd_ind]).to(self.device)

        mixed_x = torch.cat([x, buff_x], dim=0)
        mixed_y = torch.cat([y, buff_y], dim=0)
        mixed_t = torch.cat([t, buff_t], dim=0)

        return mixed_x, mixed_y, mixed_t


class MER(OnlineSupervisedMetaLearningTemplate):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        max_buffer_size=200,
        buffer_mb_size=10,
        n_inner_steps=5,
        beta=0.1,
        gamma=0.1,
        train_mb_size: int = 1,
        train_passes: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
        """Implementation of Look-ahead MAML (LaMAML) algorithm in Avalanche
            using Higher library for applying fast updates.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param max_buffer_size: maximum size of the buffer.
        :param buffer_mb_size: number of samples to retrieve from buffer
            for each sample.
        :param n_inner_steps: number of inner updates per sample.
        :param beta: coefficient for within-batch Reptile update.
        :param gamma: coefficient for within-task Reptile update.

        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_passes,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        self.buffer = MERBuffer(max_buffer_size=max_buffer_size,
                                buffer_mb_size=buffer_mb_size,
                                device=self.device)
        self.n_inner_steps = n_inner_steps
        self.beta = beta
        self.gamma = gamma

    def _before_inner_updates(self, **kwargs):
        self.w_bef = deepcopy(self.model.state_dict())
        super()._before_inner_updates(**kwargs)

    def _inner_updates(self, **kwargs):
        for inner_itr in range(self.n_inner_steps):
            x, y, t = self.mb_x, self.mb_y, self.mb_task_id
            x, y, t = self.buffer.get_batch(x, y, t)

            # Inner updates
            w_bef_t = deepcopy(self.model.state_dict())
            for idx in range(x.shape[0]):
                x_b = x[idx].unsqueeze(0)
                y_b = y[idx].unsqueeze(0)
                t_b = t[idx].unsqueeze(0)
                self.model.zero_grad()
                pred = avalanche_forward(self.model, x_b, t_b)
                loss = self._criterion(pred, y_b)
                loss.backward()
                self.optimizer.step()

            # Within-batch Reptile update
            w_aft_t = self.model.state_dict()
            self.model.load_state_dict(
                {name: w_bef_t[name] + ((w_aft_t[name] - w_bef_t[name])
                                        * self.beta)
                 for name in w_bef_t}
            )

    def _outer_update(self, **kwargs):
        w_aft = self.model.state_dict()
        self.model.load_state_dict(
            {name: self.w_bef[name] + ((w_aft[name] - self.w_bef[name])
                                       * self.gamma) for name in self.w_bef}
        )
        with torch.no_grad():
            pred = self.forward()
            self.loss = self._criterion(pred, self.mb_y)

    def _after_training_exp(self, **kwargs):
        self.buffer.update(self)
        super()._after_training_exp(**kwargs)
