import copy
from typing import List, Optional

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
def dataset_with_logits(dataset, model, batch_size, device, _max_size=300):
    model = copy.deepcopy(model)
    model.train()
    logits = []
    data = []
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False)

    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        logits.append(out)
        data.append(x)

    logits = torch.cat(logits)
    data = torch.cat(data)

    logits = F.pad(logits, (0, _max_size - logits.shape[1]), value=0)

    if len(data.shape) == 4:
        transform = Compose(
            [RandomCrop(data[0].shape[2], padding=4), RandomHorizontalFlip()]
        )
    else:
        transform = None

    dataset = make_tensor_classification_dataset(
        data,
        torch.tensor(dataset.targets),
        torch.tensor(dataset.targets_task_labels),
        logits,
        transform=transform,
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
    ):
        """Init.

        :param max_size: The max capacity of the replay memory.
        :param adaptive_size: True if mem_size is divided equally over all
                            observed experiences (keys in replay_mem).
        :param total_num_classes: If adaptive size is False, the fixed number
                                  of classes to divide capacity over.
        """
        if not adaptive_size:
            assert (
                total_num_classes > 0
            ), """When fixed exp mem size, total_num_classes should be > 0."""

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        new_data = strategy.experience.dataset.eval()

        new_data_with_logits = dataset_with_logits(
            new_data, strategy.model, strategy.train_mb_size, strategy.device,
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
            self.buffer_groups[class_id].resize(
                strategy, 
                class_to_len[class_id])


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
        derpp: bool = False,
        alpha: float = 0.1,
        beta: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="epoch",
    ):
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
            self.mem_size, adaptive_size=True
        )
        self.replay_loader = None
        self.derpp = derpp
        self.alpha = alpha
        self.beta = beta

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

        self.loss += self.alpha * F.mse_loss(
            out_buffer, batch_logits[:, : out_buffer.shape[1]]
        )

        if self.derpp:
            self.loss += self.beta * F.cross_entropy(out_buffer, batch_y)
