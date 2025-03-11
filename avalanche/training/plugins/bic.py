from collections import defaultdict
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    SupportsInt,
    Union,
)

from copy import deepcopy
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models.bic_model import BiasLayer

from avalanche.training.templates import SupervisedTemplate


class BiCPlugin(SupervisedPlugin):
    """
    Bias Correction (BiC) plugin.

    Technique introduced in:
    "Wu, Yue, et al. "Large scale incremental learning." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition. 2019"

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/bic.py
    """

    def __init__(
        self,
        mem_size: int = 2000,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        val_percentage: float = 0.1,
        T: int = 2,
        stage_2_epochs: int = 200,
        lamb: float = -1,
        lr: float = 0.1,
        num_workers: Union[int, Literal["as_strategy"]] = "as_strategy",
        verbose: bool = False,
    ):
        """
        :param mem_size: replay buffer size.
        :param batch_size: the size of the data batch. If set to `None`, it
            will be set equal to the strategy's batch size.
        :param batch_size_mem: the size of the memory batch. If
            `task_balanced_dataloader` is set to True, it must be greater than
            or equal to the number of tasks. If its value is set to `None`
            (the default value), it will be automatically set equal to the
            data batch size.
        :param task_balanced_dataloader: if True, buffer data loaders will be
                task-balanced, otherwise it will create a single dataloader for
                the buffer samples.
        :param storage_policy: The policy that controls how to add new exemplars
                            in memory
        :param val_percentage: hyperparameter used to set the
                percentage of exemplars in the val set.
        :param T: hyperparameter used to set the temperature
                used in stage 1.
        :param stage_2_epochs: hyperparameter used to set the
                amount of epochs of stage 2.
        :param lamb: hyperparameter used to balance the distilling
                loss and the classification loss.
        :param lr: hyperparameter used as a learning rate for
                the second phase of training.
        :param num_workers: number of workers using during stage 2 data loading.
            Defaults to "as_strategy", which means that the number of workers
            will be the same as the one used by the strategy.
        :param verbose: if True, prints additional info regarding the stage 2 stage
        """

        # Replay (Phase 1)
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

        # Train Bias (Phase 2)
        self.val_percentage = val_percentage
        self.stage_2_epochs = stage_2_epochs
        self.T = T
        self.lamb = lamb
        self.mem_size = mem_size
        self.lr = lr
        self.num_workers: Union[int, Literal["as_strategy"]] = num_workers

        self.seen_classes: Set[int] = set()
        self.class_to_tasks: Dict[int, int] = {}
        self.bias_layer: Optional[BiasLayer] = None
        self.model_old: Optional[Module] = None
        self.val_buffer: Dict[int, ReservoirSamplingBuffer] = {}

        self.is_first_experience: bool = True

        self.verbose: bool = verbose

    def before_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        assert not isinstance(
            strategy.model, MultiTaskModule
        ), "BiC only supported for Class Incremetnal Learning (single head)"

    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        assert strategy.experience is not None
        new_data: AvalancheDataset = strategy.experience.dataset
        task_id = strategy.clock.train_exp_counter

        cl_idxs: Dict[int, List[int]] = defaultdict(list)
        targets: Sequence[SupportsInt] = getattr(new_data, "targets")
        for idx, target in enumerate(targets):
            # Conversion to int may fix issues when target
            # is a single-element torch.tensor
            target = int(target)
            cl_idxs[target].append(idx)

        for c in cl_idxs.keys():
            self.class_to_tasks[c] = task_id

        self.seen_classes.update(cl_idxs.keys())
        lens = self.get_group_lengths(len(self.seen_classes))
        class_to_len = {}
        for class_id, ll in zip(self.seen_classes, lens):
            class_to_len[class_id] = ll

        train_data = []
        for class_id in cl_idxs.keys():
            ll = class_to_len[class_id]
            new_data_c = new_data.subset(cl_idxs[class_id][:ll])
            if class_id in self.val_buffer:
                old_buffer_c = self.val_buffer[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.val_buffer[class_id] = new_buffer

            train_data.append(new_data.subset(cl_idxs[class_id][ll:]))

        # resize buffers
        for class_id, class_buf in self.val_buffer.items():
            class_buf.resize(strategy, class_to_len[class_id])

        strategy.experience.dataset = concat_datasets(train_data)

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        assert strategy.adapted_dataset is not None

        # During the distillation phase this layer is not trained and is only
        # used to correct the bias of the classes encountered in the previous experience.
        # It will be unlocked in the bias correction phase.
        if self.bias_layer is not None:
            for param in self.bias_layer.parameters():
                param.requires_grad = False

        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def after_eval_forward(self, strategy, **kwargs):
        if self.is_first_experience:
            # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L488
            return

        strategy.mb_output = self.bias_forward(strategy.mb_output)

    def bias_forward(self, input_data: Tensor) -> Tensor:
        if self.bias_layer is None:
            return input_data

        return self.bias_layer(input_data)

    def before_backward(self, strategy, **kwargs):
        # Distillation
        if self.model_old is not None:  # That is, from the second experience onwards
            distillation_loss = self.make_distillation_loss(strategy)

            # Count the number of already seen classes (i.e., classes from previous experiences)
            initial_classes, previous_classes, current_classes = self._classes_groups(
                strategy
            )

            # Make old_classes and all_classes
            old_clss: Set[int] = set(initial_classes) | set(previous_classes)
            all_clss: Set[int] = old_clss | set(current_classes)

            if self.lamb == -1:
                lamb = len(old_clss) / len(all_clss)
                strategy.loss = (1.0 - lamb) * strategy.loss + lamb * distillation_loss
            else:
                strategy.loss = strategy.loss + self.lamb * distillation_loss

    def after_training_exp(self, strategy, **kwargs):
        self.is_first_experience = False

        # Make sure that the old_model is frozen (including batch norm layers)
        # requires_grad=False is not sufficient to freeze BN layers,
        # we also need eval()
        self.model_old = None
        self.model_old = deepcopy(strategy.model)
        self.model_old.eval()
        for param in self.model_old.parameters():
            param.requires_grad = False

        task_id = strategy.clock.train_exp_counter

        self.storage_policy.update(strategy, **kwargs)

        if task_id > 0:
            num_workers = (
                int(kwargs.get("num_workers", 0))
                if self.num_workers == "as_strategy"
                else self.num_workers
            )
            persistent_workers = (
                False if num_workers == 0 else kwargs.get("persistent_workers", False)
            )

            self.bias_correction_step(
                strategy,
                persistent_workers=persistent_workers,
                num_workers=num_workers,
            )

    def cross_entropy(self, new_outputs, old_outputs):
        """Calculates cross-entropy with temperature scaling"""
        # logp = torch.nn.functional.log_softmax(new_outputs / self.T, dim=1)
        # pre_p = torch.nn.functional.softmax(old_outputs / self.T, dim=1)
        # return -torch.mean(torch.sum(pre_p * logp, dim=1)) * self.T * self.T

        # The previous implementation (above), multiplied the final loss by T^2, which is not correct.
        # In addition, this is more aligned to how it's done in the original implementation.
        dis_logits_soft = torch.nn.functional.softmax(old_outputs / 2, dim=0)
        loss_distill = torch.nn.functional.cross_entropy(
            new_outputs / 2, dis_logits_soft
        )
        return loss_distill

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        max_size = int(self.val_percentage * self.mem_size)
        lengths = [max_size // num_groups for _ in range(num_groups)]
        # distribute remaining size among experiences.
        rem = max_size - sum(lengths)
        for i in range(rem):
            lengths[i] += 1

        return lengths

    def make_distillation_loss(self, strategy):
        assert self.model_old is not None
        initial_classes, previous_classes, current_classes = self._classes_groups(
            strategy
        )
        # print('initial_classes', initial_classes, 'previous_classes', previous_classes, 'current_classes', current_classes)

        # Forward current minibatch through the old model
        with torch.no_grad():
            out_old: Tensor = self.model_old(strategy.mb_x)

        if len(initial_classes) == 0:
            # We are in the second experience, no need to correct the bias
            # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L561
            pass
        else:
            # We are in the third experience or later
            # bias_forward will apply the bias correction to the output of the old model for the classes
            # found in previous_classes (bias correction is not applied to initial_classes or current_classes)!
            # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L564
            assert self.bias_layer is not None
            assert set(self.bias_layer.clss.tolist()) == set(previous_classes)
            with torch.no_grad():
                # out_old_before = out_old.clone()
                out_old = self.bias_forward(out_old)

                # Asserts commented out for performance reasons.
                # Remove the comments if you want to check that the bias correction is applied correctly.
                # assert torch.equal(out_old_before[:, initial_classes], out_old[:, initial_classes])
                # assert torch.equal(out_old_before[:, current_classes], out_old[:, current_classes])
                # assert not torch.equal(out_old_before[:, previous_classes], out_old[:, previous_classes])

        # To compute the distillation loss, we need the output of the new model
        # without the bias correction. During train, the output of the new model
        # does not undergo bias correction, so we can use mb_output directly.
        out_new: Tensor = strategy.mb_output

        # Union of initial_classes and previous_classes: needed to select the logits of all the old classes
        old_clss: List[int] = sorted(set(initial_classes) | set(previous_classes))

        # Distillation loss on the logits of the old classes
        return self.cross_entropy(out_new[:, old_clss], out_old[:, old_clss])

    def bias_correction_step(
        self,
        strategy: SupervisedTemplate,
        persistent_workers: bool = False,
        num_workers: int = 0,
    ):
        # --- Prepare the models ---
        # Freeze the base model, only train the new bias layer
        strategy.model.eval()

        # Note: we use torch.no_grad for this.
        # In this way, we don't need to store the status of each requires_grad
        # which is useful when we have multiple parameters with different
        # requires_grad status.
        # for param in strategy.model.parameters():
        #     param.requires_grad = False

        # Create the bias layer of the current experience
        targets = getattr(strategy.adapted_dataset, "targets")
        self.bias_layer = BiasLayer(targets.uniques)
        self.bias_layer.to(strategy.device)
        self.bias_layer.train()
        for param in self.bias_layer.parameters():
            param.requires_grad = True

        bic_optimizer = torch.optim.SGD(
            self.bias_layer.parameters(), lr=self.lr, momentum=0.9
        )

        # Typing note: verbose here is actually correct
        # The PyTorch type stubs for MultiStepLR are broken in some versions
        scheduler = MultiStepLR(
            bic_optimizer, milestones=[50, 100, 150], gamma=0.1, verbose=False
        )  # type: ignore

        # --- Prepare the dataloader for the validation set ---
        list_subsets: List[AvalancheDataset] = []
        for _, class_buf in self.val_buffer.items():
            list_subsets.append(class_buf.buffer)

        stage_set = concat_datasets(list_subsets)
        stage_loader = DataLoader(
            stage_set,
            batch_size=strategy.train_mb_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

        # Loop epochs
        for e in range(self.stage_2_epochs):
            total, t_acc, t_loss = 0, 0, 0
            for inputs in stage_loader:
                x = inputs[0].to(strategy.device)
                y_real = inputs[1].to(strategy.device)

                with torch.no_grad():
                    outputs = strategy.model(x)

                outputs = self.bias_layer(outputs)

                loss = torch.nn.functional.cross_entropy(outputs, y_real)

                _, preds = torch.max(outputs, 1)
                t_acc += torch.sum(preds == y_real.data)
                t_loss += loss.item() * x.size(0)
                total += x.size(0)

                # Hand-made L2 loss
                # https://github.com/wuyuebupt/LargeScaleIncrementalLearning/blob/7f687a323ae3629109b35c369b547af74a94e73d/resnet.py#L636
                loss += 0.1 * ((self.bias_layer.beta.sum() ** 2) / 2)

                bic_optimizer.zero_grad()
                loss.backward()
                bic_optimizer.step()

            scheduler.step()
            if self.verbose and (self.stage_2_epochs // 4) > 0:
                if (e + 1) % (self.stage_2_epochs // 4) == 0:
                    print(
                        "| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |".format(
                            e + 1, t_loss / total, 100 * t_acc / total
                        )
                    )

        # Freeze the bias layer
        self.bias_layer.eval()
        for param in self.bias_layer.parameters():
            param.requires_grad = False

        if self.verbose:
            print(
                "Bias correction done: alpha={}, beta={}".format(
                    self.bias_layer.alpha.item(), self.bias_layer.beta.item()
                )
            )

    def _classes_groups(self, strategy: SupervisedTemplate):
        current_experience: int = strategy.experience.current_experience
        # Split between
        # - "initial" classes: seen between in experiences [0, current_experience-2]
        # - "previous" classes: seen in current_experience-1
        # - "current" classes: seen in current_experience

        # "initial" classes
        initial_classes: Set[int] = (
            set()
        )  # pre_initial_cl in the original implementation
        previous_classes: Set[int] = set()  # pre_new_cl in the original implementation
        current_classes: Set[int] = set()  # new_cl in the original implementation
        # Note: pre_initial_cl + pre_new_cl is "initial_cl" in the original implementation

        for cls, exp_id in self.class_to_tasks.items():
            assert exp_id >= 0
            assert exp_id <= current_experience

            if exp_id < current_experience - 1:
                initial_classes.add(cls)
            elif exp_id == current_experience - 1:
                previous_classes.add(cls)
            else:
                current_classes.add(cls)

        return (
            sorted(initial_classes),
            sorted(previous_classes),
            sorted(current_classes),
        )


__all__ = [
    "BiCPlugin",
]
