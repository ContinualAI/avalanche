from collections import defaultdict
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    SupportsInt,
    Union,
)

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_attribute import TensorDataAttribute
from avalanche.benchmarks.utils.flat_data import FlatData
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.utils import cycle
from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.training.storage_policy import (
    BalancedExemplarsBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.training.templates import SupervisedTemplate


@torch.no_grad()
def compute_dataset_logits(dataset, model, batch_size, device, num_workers=0):
    was_training = model.training
    model.eval()

    logits = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        out = out.detach().cpu()

        for row in out:
            logits.append(torch.clone(row))

    if was_training:
        model.train()

    return logits


class ClassBalancedBufferWithLogits(BalancedExemplarsBuffer):
    """
    ClassBalancedBuffer that also stores the logits
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        total_num_classes: Optional[int] = None,
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
                total_num_classes is not None and total_num_classes > 0
            ), "When fixed exp mem size, total_num_classes should be > 0."

        super().__init__(max_size, adaptive_size, total_num_classes)
        self.adaptive_size = adaptive_size
        self.total_num_classes = total_num_classes
        self.seen_classes: Set[int] = set()

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data: AvalancheDataset = strategy.experience.dataset

        logits = compute_dataset_logits(
            new_data.eval(),
            strategy.model,
            strategy.train_mb_size,
            strategy.device,
            num_workers=kwargs.get("num_workers", 0),
        )
        new_data_with_logits = make_avalanche_dataset(
            new_data,
            data_attributes=[
                TensorDataAttribute(
                    FlatData([logits], discard_elements_not_in_indices=True),
                    name="logits",
                    use_in_getitem=True,
                )
            ],
        )
        # Get sample idxs per class
        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        targets: Sequence[SupportsInt] = getattr(new_data, "targets")
        for idx, target in enumerate(targets):
            # Conversion to int may fix issues when target
            # is a single-element torch.tensor
            target = int(target)
            cl_idxs[target].append(idx)

        # Make AvalancheSubset per class
        cl_datasets = {}
        for c, c_idxs in cl_idxs.items():
            subset = new_data_with_logits.subset(c_idxs)
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
            self.buffer_groups[class_id].resize(strategy, class_to_len[class_id])


class DER(SupervisedTemplate):
    """
    Implements the DER and the DER++ Strategy,
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: Optional[int] = None,
        alpha: float = 0.1,
        beta: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
        **kwargs
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
            **kwargs
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
                    num_workers=kwargs.get("num_workers", 0),
                )
            )
        else:
            self.replay_loader = None

        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.replay_loader = None  # Allow DER to be checkpointed
        self.storage_policy.update(self, **kwargs)
        super()._after_training_exp(**kwargs)

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        if self.replay_loader is None:
            return None

        batch_x, batch_y, batch_tid, batch_logits = next(self.replay_loader)
        batch_x, batch_y, batch_tid, batch_logits = (
            batch_x.to(self.device),
            batch_y.to(self.device),
            batch_tid.to(self.device),
            batch_logits.to(self.device),
        )
        self.mbatch[0] = torch.cat((batch_x, self.mbatch[0]))
        self.mbatch[1] = torch.cat((batch_y, self.mbatch[1]))
        self.mbatch[2] = torch.cat((batch_tid, self.mbatch[2]))
        self.batch_logits = batch_logits

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            if self.replay_loader is not None:
                # DER Loss computation

                self.loss += F.cross_entropy(
                    self.mb_output[self.batch_size_mem :],
                    self.mb_y[self.batch_size_mem :],
                )

                self.loss += self.alpha * F.mse_loss(
                    self.mb_output[: self.batch_size_mem],
                    self.batch_logits,
                )
                self.loss += self.beta * F.cross_entropy(
                    self.mb_output[: self.batch_size_mem],
                    self.mb_y[: self.batch_size_mem],
                )

                # They are a few difference compared to the autors impl:
                # - Joint forward pass vs. 3 forward passes
                # - One replay batch vs two replay batches
                # - Logits are stored from the non-transformed sample
                #   after training on task vs instantly on transformed sample

            else:
                self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)


__all__ = ["DER"]
