from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy

from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)


class DualNet(SupervisedTemplate):
    """
    DualNet strategy as proposed in
    "DualNet: Continual Learning, Fast and Slow" by Quang Pham et. al.
    When task_agnostic_fast_updates is False, the strategy is equivalent to 
    "task-agnostic" version of the strategy proposed in the paper where only 
    the slow-updates are task-agnostic and the fast-updates are task-aware. 
    When task_agnostic_fast_updates is True, the strategy becomes fully 
    task-agnostic without the need fot task-identity labels during training.
    https://arxiv.org/abs/2110.00175
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_bt: torch.optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),
        inner_steps: int = 2,
        n_outer: int = 3,
        batch_size_mem: int = 10,
        mem_size: int = 200,
        memory_strength: float = 10.0,
        temperature: float = 2.0,
        beta: float = 0.05,
        task_agnostic_fast_updates: bool = False,
        img_size: int = 84,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin,
            Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="experience",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param inner_steps: number of inner steps for the fast updates.
        :param n_outer: number of outer steps for the slow updates.
        :param memory_strength: Regularization coefficient for the KLDiv loss.
        :param mem_size: Buffer size.
        :param temperature: Temperature for the KLDiv loss.
            mode.
        :param beta: Update rate for the slow updates.
        :param task_agnostic_fast_updates: whether to perform the fast updates 
            in task-agnostic mode or task-aware mode.
        :param img_size: Dataset image size.
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        torch.autograd.set_detect_anomaly(True)

        # Set strategy parameters
        self.reg = memory_strength
        self.temp = temperature
        self.beta = beta
        self.batch_size_mem = batch_size_mem
        self.inner_steps = inner_steps
        self.n_outer = n_outer
        self.task_agnostic_fast_updates = task_agnostic_fast_updates

        # Optimizer
        self.optimizer = optimizer
        self.optimizer_bt = optimizer_bt

        # Set transforms
        self.transforms_0, self.transforms_1, self.transforms_2 = \
            get_dualnet_transforms(img_size=img_size)

        # Initialize Losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        # Initialize buffer
        self.storage_policy = ReservoirSamplingBuffer(max_size=mem_size)

        # Initialize random number generator for sampling from the buffer
        self.rng = torch.Generator()
        self.rng.manual_seed(0)

    def sample_from_buffer(self, ssl=False):
        """ Sample from buffer and return a batch of samples. """
        if ssl:
            n_samples = 32
        else:
            n_samples = self.batch_size_mem 

        buffer = self.storage_policy.buffer
        indices = torch.arange(len(buffer))

        if len(indices) == 0:
            x_b, y_b, l_b = torch.FloatTensor([]), torch.LongTensor([])
            return x_b.to(self.device), y_b.to(self.device), l_b.to(self.device)

        # Shuffle indices
        rnd_idx = torch.randperm(len(indices), generator=self.rng)
        indices = indices[rnd_idx][:n_samples]

        # Create buffer batch
        samples = [buffer[i.item()] for i in indices]
        x_b = torch.stack([s[0] for s in samples])
        y_b = torch.LongTensor([s[1] for s in samples])
        l_b = torch.stack([s[3] for s in samples])

        return x_b.to(self.device), y_b.to(self.device), l_b.to(self.device)

    def kl(self, y1, y2, temp): 
        return F.kl_div(
                F.log_softmax(y1 / temp, dim=-1),
                F.softmax(y2 / temp, dim=-1), reduce=True) * y1.shape[0]
    
    def slow_update(self):
        # Performs slow updates to the model's representation backbone
        for j in range(self.n_outer):
            w_before = deepcopy(self.model.state_dict())
            for _ in range(self.inner_steps):
                self.optimizer_bt.zero_grad()
                if self.task_agnostic_fast_updates:
                    if len(self.storage_policy.buffer) > 0:
                        x_, y_, l_ = self.sample_from_buffer(ssl=True)
                    else:
                        x_ = self.mb_x.to(self.device)
                    x1, x2 = self.transforms_1(x_), self.transforms_2(x_)
                    loss = self.model.BarlowTwins(x1, x2)
                    loss.backward()
                    self.optimizer_bt.step()
                else:
                    raise NotImplementedError()
            w_after = self.model.state_dict()
            new_params = {
                k : w_before[k] + ((w_after[k] - w_before[k]) * self.beta) 
                for k in w_before.keys()
            }
            self.model.load_state_dict(new_params)

    def fast_update(self):
        # Performs fast updates to the adaptor model
        x, y = self.mb_x.to(self.device), self.mb_y.to(self.device)

        for _ in range(self.inner_steps):
            self.optimizer.zero_grad()

            if self.task_agnostic_fast_updates:
                pred = self.model(self.transforms_0(x))
                loss = self.bce(pred, y)

                if len(self.storage_policy.buffer) > 0:
                    x_, y_, l_ = self.sample_from_buffer()
                    pred = self.model(self.transforms_0(x_))
                    loss += self.bce(pred, y_)
                    loss += self.reg * self.kl(pred , l_ , self.temp)
            else:
                raise NotImplementedError()

            loss.backward()
            self.optimizer.step()

        # Set mb_output for logging as the last prediction
        self.mb_output = pred
        self.loss = loss

        return 0.

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()
            self._before_training_iteration(**kwargs)

            self.slow_update()
            self.fast_update()

            self._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        self._update_buffer()

    def _update_buffer(self):
        if not self.task_agnostic_fast_updates:
            raise NotImplementedError()

        # Get dataset from the current experience
        new_data: AvalancheDataset = self.experience.dataset

        # Compute dataset logits
        training_mode = self.model.training is True
        self.model.eval()
        logits = []
        loader = torch.utils.data.DataLoader(
            new_data,
            batch_size=64,
            shuffle=False,
        )

        for x, _, _ in loader:
            x = x.to(self.device)
            out = self.model(self.transforms_0(x))
            logits.extend(list(out.detach().cpu()))

        # Add logits to the dataset
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(logits, name="logits", use_in_getitem=True)
            ],
        )
        # Update storage policy
        self.storage_policy.update_from_dataset(new_data_with_logits)

        # Restore training mode
        if training_mode:
            self.model.train()


def get_dualnet_transforms(img_size=84):
    class RandomGaussianBlur:
        def __init__(self, kernel_size, sigma=(0.1, 2.0), p=0.5):
            self.p = p
            self.transform = transforms.GaussianBlur(kernel_size, sigma)

        def __call__(self, x):
            if torch.rand(1).item() < self.p:
                return self.transform(x)
            return x

    class RandomColorJitter:
        def __init__(self, brightness=0.4, contrast=0.4,
                     saturation=0.2, hue=0.1, p=0.5):
            self.p = p
            self.transform = transforms.ColorJitter(
                brightness=brightness, contrast=contrast,
                saturation=saturation, hue=hue)

        def __call__(self, x):
            if torch.rand(1).item() < self.p:
                return self.transform(x)
            return x

    transforms_0 = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(torch.FloatTensor((0.5, 0.5, 0.5)),
                                     torch.FloatTensor((0.5, 0.5, 0.5)))
    ])

    transforms_1 = transforms.Compose([
                transforms.RandomCrop((img_size, img_size)), 
                transforms.RandomHorizontalFlip(),
                RandomColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.2, hue=0.1, p=0.8),
                transforms.RandomGrayscale(p=0.2),
                RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0),
                transforms.Normalize(torch.FloatTensor((0.5, 0.5, 0.5)), 
                                     torch.FloatTensor((0.5, 0.5, 0.5)))
    ])

    transforms_2 = transforms.Compose([
                transforms.RandomCrop((img_size, img_size)), 
                transforms.RandomHorizontalFlip(),
                RandomColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.2, hue=0.1, p=0.8),
                RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.1),
                transforms.Normalize(torch.FloatTensor((0.5, 0.5, 0.5)), 
                                     torch.FloatTensor((0.5, 0.5, 0.5)))
    ])

    return transforms_0, transforms_1, transforms_2
