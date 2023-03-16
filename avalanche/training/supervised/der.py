import copy
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

from avalanche.benchmarks.utils import (classification_subset,
                                        make_tensor_classification_dataset)
from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import (BalancedExemplarsBuffer,
                                               ReservoirSamplingBuffer)
from avalanche.training.templates import SupervisedTemplate


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def create_tensor_dataset(
    dataset,
    model=None,
    batch_size=128,
    device="cuda",
    transforms=None,
    add_logits=True,
):
    logits = []
    data = []
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    if add_logits:
        model.eval()
    for x, _, _ in loader:
        x = x.to(device)
        data.append(x.cpu())

        if add_logits:
            out = model(x)
            logits.append(out.cpu())
    data = torch.cat(data)
    if add_logits:
        logits = torch.cat(logits)
        dataset = make_tensor_classification_dataset(
            data,
            torch.tensor(dataset.targets),
            torch.tensor(dataset.targets_task_labels),
            logits,
            transform=transforms,
        )
    else:
        dataset = make_tensor_classification_dataset(
            data,
            torch.tensor(dataset.targets),
            torch.tensor(dataset.targets_task_labels),
            transform=transforms,
        )
    return dataset


class ClassBalancedBufferWithLogits(BalancedExemplarsBuffer):
    """
    ClassBalancedBuffer that also stores the logits
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: int = None,
        transforms: Callable = None,
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        :param transforms: transformation to be applied to the buffer
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()
        self.transforms = transforms

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset.eval()

        new_data_with_logits = create_tensor_dataset(
            new_data,
            strategy.model,
            strategy.train_mb_size,
            strategy.device,
            self.transforms,
            add_logits=True,
        )

        # Get sample idxs per class
        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = classification_subset(new_data_with_logits, indices=c_idxs)
            cl_datasets[c] = subset
        # Update seen classes
        self.seen_classes.update(cl_datasets.keys())

        # associate lengths to classes
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        # update buffers with new data
        for class_id, new_data_c in cl_datasets.items():
            ll = class_to_len[class_id]
            if class_id in self.buffer_groups:
                old_buffer_c = self.buffer_groups[class_id]
                # Here it uses underlying dataset
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.buffer_groups[class_id] = new_buffer

        # resize buffers
        for class_id, class_buf in self.buffer_groups.items():
            self.buffer_groups[class_id].resize(strategy, 
                                                class_to_len[class_id])


_default_der_transforms = Compose([RandomCrop(32, padding=4), 
                                   RandomHorizontalFlip()])


class DER(SupervisedTemplate):
    """
    Implements the DER and the DER++ Strategy,
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: int = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        transforms: Callable = _default_der_transforms,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param alpha: float : Hyperparameter weighting the MSE loss
        :param beta: float : Hyperparameter weighting the CE loss,
                             when more than 0, DER++ is used instead of DER
        :param transforms: Callable: Transformations to use for
                                     both the dataset and the buffer data, on 
                                     top of already existing 
                                     test transformations.
                                     If any supplementary transformations 
                                     are applied to the 
                                     input data, it will be 
                                     overwritten by this argument 
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
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size
        self.storage_policy = ClassBalancedBufferWithLogits(
            self.mem_size, adaptive_size=True, transforms=transforms
        )
        self.replay_loader = None
        self.alpha = alpha
        self.beta = beta
        self.transforms = transforms

    def _before_training_exp(self, **kwargs):
        buffer = self.storage_policy.buffer
        if len(buffer) >= self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)

    def train_dataset_adaptation(self, **kwargs):
        self.adapted_dataset = create_tensor_dataset(
            self.experience.dataset.eval(),
            model=None,
            batch_size=self.train_mb_size,
            device=self.device,
            transforms=self.transforms,
            add_logits=False,
        )

    def _after_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        super()._after_training_exp(**kwargs)

    def _before_backward(self, **kwargs):
        super()._before_backward(**kwargs)
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits, _ = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(self.device),
            batch_y.to(self.device),
            batch_tid.to(self.device),
            batch_logits.to(self.device),
        )

        out_buffer = avalanche_forward(self.model, batch_x, batch_tid)

        if batch_logits.shape[1] != out_buffer.shape[1]:
            raise AssertionError(
                "DER Strategy requires the use of "
                "a fixed size classifier, but a classifier of"
                f"varying size has been found during execution"
            )

        self.loss += self.alpha * F.mse_loss(
            out_buffer, batch_logits[:, : out_buffer.shape[1]]
        )

        self.loss += self.beta * F.cross_entropy(out_buffer, batch_y)
