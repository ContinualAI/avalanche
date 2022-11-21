from typing import Optional, TYPE_CHECKING

from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.benchmarks.utils import classification_subset, \
                                    concat_classification_datasets
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
    ReservoirSamplingBuffer,
)
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models.bic_model import BiasLayer

if TYPE_CHECKING:
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
        batch_size: int = None,
        batch_size_mem: int = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,

        val_percentage: float = 0.1,
        T: int = 2, 
        stage_2_epochs: int = 200,
        lamb: float = -1, 
        lr: float = 0.1,
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

        self.seen_classes = set()
        self.class_to_tasks = {}
        self.bias_layer = {}
        self.model_old = None
        self.val_buffer = {}

    @property
    def ext_mem(self):
        return self.storage_policy.buffer_groups  # a Dict<task_id, Dataset>

    def before_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        assert not isinstance(strategy.model, MultiTaskModule), \
               "BiC only supported for Class Incremetnal Learning (single head)"

    def before_train_dataset_adaptation(
        self, 
        strategy: "SupervisedTemplate", 
        **kwargs
    ):
        new_data = strategy.experience.dataset
        task_id = strategy.experience.current_experience

        cl_idxs = {k : [] for k in new_data.targets.uniques}
        for idx, target in enumerate(new_data.targets):
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
            new_data_c = classification_subset(
                                            new_data,
                                            cl_idxs[class_id][:ll])
            if class_id in self.val_buffer:
                old_buffer_c = self.val_buffer[class_id]
                old_buffer_c.update_from_dataset(new_data_c)
                old_buffer_c.resize(strategy, ll)
            else:
                new_buffer = ReservoirSamplingBuffer(ll)
                new_buffer.update_from_dataset(new_data_c)
                self.val_buffer[class_id] = new_buffer

            train_data.append(classification_subset(
                                            new_data,
                                            cl_idxs[class_id][ll:]))

        # resize buffers
        for class_id, class_buf in self.val_buffer.items():
            class_buf.resize(
                strategy, class_to_len[class_id]
            )

        strategy.experience.dataset = concat_classification_datasets(train_data)

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
        task_id = strategy.experience.current_experience

        if task_id not in self.bias_layer:
            self.bias_layer[task_id] = BiasLayer(
                                strategy.device, 
                                list(strategy.adapted_dataset.targets.uniques)
                            )

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

    def after_forward(self, strategy, **kwargs):
        for t in self.bias_layer.keys():
            strategy.mb_output = self.bias_layer[t](strategy.mb_output)
    
    def after_eval_forward(self, strategy, **kwargs):
        for t in self.bias_layer.keys():
            strategy.mb_output = self.bias_layer[t](strategy.mb_output)

    def before_backward(self, strategy, **kwargs):
        # Distill
        task_id = strategy.experience.current_experience

        if self.model_old is not None:
            out_old = self.model_old(strategy.mb_x.to(strategy.device))
            out_new = strategy.model(strategy.mb_x.to(strategy.device))

            old_clss = []
            for c in self.class_to_tasks.keys():
                if self.class_to_tasks[c] < task_id:
                    old_clss.append(c)
            
            loss_dist = self.cross_entropy(out_new[:, old_clss],
                                           out_old[:, old_clss])
            if self.lamb == -1:
                lamb = len(old_clss) / len(self.seen_classes)
                return (1.0 - lamb) * strategy.loss + lamb * loss_dist
            else:
                return strategy.loss + self.lamb * loss_dist

    def after_training_exp(self, strategy, **kwargs):
        self.model_old = deepcopy(strategy.model)
        task_id = strategy.experience.current_experience
        self.storage_policy.update(strategy, **kwargs)

        if task_id > 0:
            list_subsets = []
            for _, class_buf in self.val_buffer.items():
                list_subsets.append(class_buf.buffer)
            
            stage_set = concat_classification_datasets(list_subsets)
            stage_loader = DataLoader(
                                stage_set, 
                                batch_size=strategy.train_mb_size, 
                                shuffle=True,
                                num_workers=4)
            
            bic_optimizer = torch.optim.SGD(
                                self.bias_layer[task_id].parameters(), 
                                lr=self.lr, momentum=0.9)

            scheduler = MultiStepLR(bic_optimizer, milestones=[50, 100, 150], 
                                    gamma=0.1, verbose=False)
            
            # Loop epochs
            for e in range(self.stage_2_epochs):
                total, t_acc, t_loss = 0, 0, 0
                for inputs in stage_loader:
                    x = inputs[0].to(strategy.device)
                    y_real = inputs[1].to(strategy.device)

                    outputs = strategy.model(x)
                    for t in self.bias_layer.keys():
                        outputs = self.bias_layer[t](outputs)

                    loss = torch.nn.functional.cross_entropy(
                                                            outputs, 
                                                            y_real)
                        
                    _, preds = torch.max(outputs, 1)
                    t_acc += torch.sum(preds == y_real.data)
                    t_loss += loss.item() * x.size(0)
                    total += x.size(0)

                    loss += 0.1 * ((self.bias_layer[task_id].beta.sum() 
                                    ** 2) / 2)

                    bic_optimizer.zero_grad()
                    loss.backward()
                    bic_optimizer.step()
                
                scheduler.step()
                if (e + 1) % (int(self.stage_2_epochs / 4)) == 0:
                    print('| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |'
                          .format(e + 1, t_loss / total,
                                  100 * t_acc / total))
    
    def cross_entropy(self, outputs, targets):
        """Calculates cross-entropy with temperature scaling"""
        logp = torch.nn.functional.log_softmax(outputs/self.T, dim=1)
        pre_p = torch.nn.functional.softmax(targets/self.T, dim=1)
        return -torch.mean(torch.sum(pre_p * logp, dim=1)) * self.T * self.T
    
    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        max_size = int(self.val_percentage * self.mem_size)
        lengths = [max_size // num_groups for _ in range(num_groups)]
        # distribute remaining size among experiences.
        rem = max_size - sum(lengths)
        for i in range(rem):
            lengths[i] += 1

        return lengths
