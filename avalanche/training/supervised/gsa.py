from typing import List, Optional

from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
import torchvision.transforms as transforms

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.templates import OnlineSupervisedTemplate


class GSA(OnlineSupervisedTemplate):
    """
    GSA, as originally proposed in
    "Dealing with Cross-Task Class Discrimination in Online Continual Learning"
    by Yiduo Guo et. al.
    https://www.cs.uic.edu/~liub/publications/CVPR-2023-Yidou-Guo.pdf
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: int = 64,
        batch_size_mem_cur: int = 22,
        crop_size: int = 32,
        train_mb_size: int = 1,
        train_passes: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
        peval_mode="experience",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int : Fixed memory size
        :param batch_size_mem: int : Buffer batch size for previous tasks.
        :param batch_size_mem_cur: int : Buffer batch size for the current task.
        :param crop_size: int : Crop size for augmentation.
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
            train_passes,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        self.batch_size_mem_cur = batch_size_mem_cur
        self.storage_policy = ReservoirSamplingBuffer(max_size=self.mem_size)

        # Initialize random generator for sampling
        self.rng = torch.Generator()
        self.rng.manual_seed(0)

        # Transformationss
        self.aug_trans = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0), 
                                         antialias=True),
        ])

        # Initialize 
        self.num_repeats = 2  # Number of repeats based on number of flips
        self.neg_logits_sum_global = None  # Sum of negative logits
        self.pos_logits_sum_global = None  # Sum of negative logits
        self.n_class_obs_global = None  # Number of class observations
        self.seen_classes = None  # Classes seen so far

    def random_flip(self, x):
        return torch.cat([self.aug_trans(x), torch.flip(x, dims=[3])], dim=0)

    def _before_training_exp(self, **kwargs):
        # Only at task boundaries
        if self.experience.is_first_subexp:
            # Task label
            self.current_task = self.experience.task_labels[0]

            # Classes in the current task
            self.current_classes = torch.LongTensor(
                self.experience.origin_experience.classes_in_this_experience
            ).unique().to(self.device)

            # Classes seen so far
            if self.seen_classes is None:
                self.seen_classes = self.current_classes.clone().to(self.device)
            else:
                self.seen_classes = torch.cat(
                    [self.seen_classes, self.current_classes], 
                    dim=0).unique().to(self.device)

            # Reser values for the current experience
            self.neg_logits_sum_exp = None
            self.pos_logits_sum_exp = None
            self.sub_exp_counter = 0

        super()._before_training_exp(**kwargs)

    def cross_entropy_with_oh_targets(self, outputs, targets, eps=1e-5):
        """ Calculates cross-entropy with temperature scaling, 
        targets can also be soft targets but they must sum to 1 """
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        # outputs += eps
        ce = -(targets * torch.log(outputs + eps)).sum(1)
        ce = ce.mean()

        return ce

    def sample_from_buffer(self, n_samples, current_task=False):
        """ Sample from buffer and return a batch of samples. 
        If current_task is True, only samples from the current task are 
        returned otherwise samples from previous tasks are returned.
        :param current_task: bool
        :return: x_b, y_b
        """
        buffer = self.storage_policy.buffer
        indices = torch.arange(len(buffer))

        # Get task labels for buffer samples
        buffer_tasks = torch.LongTensor(buffer.targets_task_labels)

        # Create mask for buffer samples according to the task they belong to
        if current_task:
            mask = buffer_tasks.eq(self.current_task)
        else:
            if self.current_task == 0:
                mask = torch.tensor([False]*len(indices))
            else:
                prev_tasks = torch.LongTensor(list(range(self.current_task)))
                mask = buffer_tasks.eq(prev_tasks.view(-1, 1))
                mask = torch.sum(mask, dim=0).bool()

        # Filter indices
        indices = indices[mask]

        # Shuffle indices
        rnd_idx = torch.randperm(len(indices), generator=self.rng)
        indices = indices[rnd_idx][:n_samples]

        if len(indices) == 0:
            x_b, y_b = torch.FloatTensor([]), torch.LongTensor([])
            return x_b.to(self.device), y_b.to(self.device)

        # Create buffer batch
        samples = [buffer[i.item()] for i in indices]
        x_b = torch.stack([s[0] for s in samples])
        y_b = torch.LongTensor([s[1] for s in samples])

        return x_b.to(self.device), y_b.to(self.device)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        # Mask logits for unseen classes
        logits = self.model(self.mb_x)
        mask = torch.zeros_like(logits)
        mask[:, self.seen_classes] = 1
        mask = ~mask.bool()
        logits[mask] = -torch.inf

        return logits

    def forward_without_buffer(self):
        # Aumgment current batch
        x = self.random_flip(self.mb_x)
        y = self.mb_y.repeat(self.num_repeats)

        # Model forward
        x_aug = self.aug_trans(x)
        logits = self.model(x_aug)

        # Set self.mb_output for logging
        self.mb_output = logits[:self.mb_x.shape[0]]

        # One-hot targets for the current batch
        oh_y = F.one_hot(y, num_classes=logits.shape[1])

        # Selected indices for the current task only
        logits_cur = logits.index_select(dim=1, index=self.current_classes)
        oh_y_cur = oh_y.index_select(dim=1, index=self.current_classes)

        # Calculate loss
        self.loss += self.cross_entropy_with_oh_targets(logits_cur, oh_y_cur)

    def forward_with_buffer(self):
        # ==================== >  Cross-entropy loss for the current task
        # Sample from buffer and concatenate with current batch
        x_b_cur, y_b_cur = self.sample_from_buffer(self.batch_size_mem_cur,
                                                   current_task=True)
        x_cur = torch.cat([self.mb_x, x_b_cur], dim=0)
        y_cur = torch.cat([self.mb_y, y_b_cur], dim=0)

        # Aumgment current batch and make predictions
        x_cur_rf = self.random_flip(x_cur)
        y_cur_rf = y_cur.repeat(self.num_repeats)
        logits_cur = self.model(self.aug_trans(x_cur_rf))

        # Set self.mb_output for logging
        self.mb_output = logits_cur[:self.mb_x.shape[0]]

        # One-hot targets for the current batch
        y_cur_rf_oh = F.one_hot(y_cur_rf, num_classes=logits_cur.shape[1])

        # Selected indices for the current task only
        logits_cur = logits_cur.index_select(dim=1, index=self.current_classes)
        y_cur_rf_oh = y_cur_rf_oh.index_select(dim=1, 
                                               index=self.current_classes)

        # Calculate loss
        loss_p = self.cross_entropy_with_oh_targets(logits_cur, y_cur_rf_oh)

        # ==================== >   Cross-entropy loss for other tasks

        rate = len(self.current_classes) / len(self.seen_classes)

        # Sample from previous tasks
        x_b_prev, y_b_prev = self.sample_from_buffer(self.batch_size_mem,
                                                     current_task=False)
        x_mem = torch.cat([x_b_prev[:int(self.batch_size_mem*(1-rate))],
                           x_cur[:int(self.batch_size_mem*(rate))]], dim=0)
        y_mem = torch.cat([y_b_prev[:int(self.batch_size_mem*(1-rate))],
                           y_cur[:int(self.batch_size_mem*(rate))]], dim=0)

        # Permute indices in x_mem
        rnd_idx = torch.randperm(len(x_mem), generator=self.rng)
        x_mem, y_mem = x_mem[rnd_idx], y_mem[rnd_idx]

        # Random flip
        x_mem = self.random_flip(x_mem)
        y_mem = y_mem.repeat(self.num_repeats)

        # Model forward
        logits_mixed = self.model(x_mem)

        # Logits to probabilities
        logits_mixed = F.softmax(logits_mixed, dim=1)

        # One-hot matrix for the current batch
        category_matrix_new = F.one_hot(y_mem, 
                                        num_classes=logits_mixed.shape[1])

        # Calculate negative and positive gradients for the current batch 
        positive_prob = torch.zeros(logits_mixed.shape).to(self.device)
        false_prob = deepcopy(logits_mixed.detach()).to(self.device)
        for i_t in range(int(logits_mixed.shape[0])):
            false_prob[i_t][y_mem[i_t]] = 0
            positive_prob[i_t][y_mem[i_t]
                               ] = logits_mixed[i_t][y_mem[i_t]].detach()

        # Update task negative and positive gradients
        if self.neg_logits_sum_exp is None:
            self.neg_logits_sum_exp = torch.sum(false_prob, 
                                                dim=0).to(self.device)
            self.pos_logits_sum_exp = torch.sum(positive_prob, 
                                                dim=0).to(self.device)
            if self.current_task == 0:
                self.n_class_obs_global = torch.sum(category_matrix_new, 
                                                    dim=0).to(self.device)
            else:
                self.n_class_obs_global += torch.sum(category_matrix_new, dim=0)
            self.n_class_obs_exp = torch.sum(category_matrix_new, 
                                             dim=0).to(self.device)
        else:
            self.n_class_obs_global += torch.sum(category_matrix_new, dim=0)
            self.n_class_obs_exp += torch.sum(category_matrix_new, dim=0)
            self.neg_logits_sum_exp += torch.sum(false_prob, dim=0)
            self.pos_logits_sum_exp += torch.sum(positive_prob, dim=0)

        # Update global negative and positive gradients
        if self.neg_logits_sum_global is None:
            self.neg_logits_sum_global = torch.sum(false_prob, dim=0)
            self.pos_logits_sum_global = torch.sum(positive_prob, dim=0)
        else:
            self.neg_logits_sum_global += torch.sum(false_prob, dim=0)
            self.pos_logits_sum_global += torch.sum(positive_prob, dim=0)

        if self.sub_exp_counter < 5:
            ANT = torch.ones(len(self.seen_classes))
        else:
            cat_sum = self.n_class_obs_global.index_select(
                dim=0, index=self.seen_classes)
            pos_log = self.pos_logits_sum_global.index_select(
                dim=0, index=self.seen_classes)
            neg_log = self.neg_logits_sum_global.index_select(
                dim=0, index=self.seen_classes)
            ANT = (cat_sum - pos_log) / neg_log

        # One-hot targets for the all seen classes
        # oh_y_mixed = F.one_hot(y_mem, num_classes=logits_mixed.shape[1])
        oh_y_mixed = torch.zeros(logits_mixed.shape).to(self.device)
        for i in range(y_mem.shape[0]):
            if y_mem[i] >= len(ANT):
                oh_y_mixed[i][y_mem[i]] = 1
            else:
                oh_y_mixed[i][y_mem[i]] = 2 / (1+torch.exp(1-(ANT[y_mem[i]])))

        # Mask indices based on seen classes
        logits_mixed_seen = logits_mixed.index_select(dim=1, 
                                                      index=self.seen_classes)
        oh_y_mixed_seen = oh_y_mixed.index_select(dim=1, 
                                                  index=self.seen_classes)

        # Calculate loss for the mixed batch
        loss_n = self.cross_entropy_with_oh_targets(logits_mixed_seen, 
                                                    oh_y_mixed_seen)
        
        # Final loss
        self.loss = 2 * loss_n + 1 * loss_p

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
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            if len(self.storage_policy.buffer) > 0:
                self.forward_with_buffer()
            else:
                self.forward_without_buffer()
            self._after_forward(**kwargs)

            # Backward
            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.storage_policy.update_from_dataset(self.experience.dataset)
        self.sub_exp_counter += 1
        return super()._after_training_exp(**kwargs)
