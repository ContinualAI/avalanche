import random
from copy import deepcopy

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import torch

from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader


_default_cifar100_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ),
    ]
)



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
        self, val_percentage: float = 0.1, T: int = 2, 
        mem_size: int = 2000, stage_2_epochs: int = 200, lamb: float = -1, 
        verbose: bool = False, lr: float = 0.1
    ):
        """
        :param val_percentage: hyperparameter used to set the 
                percentage of exemplars in the val set.
        :param T: hyperparameter used to set the temperature 
                used in stage 1.
        :param stage_2_epochs: hyperparameter used to set the 
                amount of epochs of stage 2.
        :param lamb: hyperparameter used to balance the distilling 
                loss and the classification loss.
        :param verbose: when True, the computation of the influence
        """

        # Init super class
        super().__init__()
        
        self.val_percentage = val_percentage
        self.stage_2_epochs = stage_2_epochs

        self.T = T
        self.lamb = lamb
        self.mem_size = mem_size
        self.lr = lr

        self.verbose = verbose

        self.seen_classes = set()
        self.class_to_tasks = {}

        self.train_data = {}
        self.val_data = {}

        self.model_old = None

        self.task_balanced_dataloader = False
        self.batch_size = None
        self.batch_size_mem = None

    def before_train_dataset_adaptation(self, strategy, **kwargs):
        new_data = strategy.experience.dataset
        task_id = strategy.experience.current_experience

        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        for c in cl_idxs.keys():
            self.class_to_tasks[c] = task_id
        
        if strategy.experience.current_experience > 0  and isinstance(strategy.model, MultiTaskModule):
            strategy.model.adaptation(strategy.experience)

        strategy.model.add_bias_layer(strategy.device, cl_idxs.keys())

        self.seen_classes.update(cl_idxs.keys())
        num_per_class = self.mem_size // len(self.seen_classes)
        num_val_cls = int(self.val_percentage * num_per_class)

        # Remove extra exemplars from previous tasks
        for k in self.val_data.keys():
            for i, subset in enumerate(self.val_data[k]):
                self.val_data[k][i] = AvalancheSubset(subset, torch.arange(num_val_cls))

            for i, subset in enumerate(self.train_data[k]):
                num_train = num_per_class - num_val_cls
                self.train_data[k][i] = AvalancheSubset(subset, torch.arange(num_train))
        
        # Add elements from the current task
        self.train_data[task_id] = []
        self.val_data[task_id] = []

        for c in cl_idxs.keys():
            # w = torch.rand(len(cl_idxs[c]))
            # _, sorted_idx = w.sort(descending=True)
            
            self.val_data[task_id].append(AvalancheSubset(new_data, 
                                            cl_idxs[c][:num_val_cls], 
                                            task_labels=task_id))
            self.train_data[task_id].append(AvalancheSubset(new_data, 
                                            cl_idxs[c][num_val_cls:]))
        
        current_subsets = []
        for k in self.train_data.keys():
            for subset in self.train_data[k]:
                current_subsets.append(subset)

        # strategy.experience.dataset = AvalancheConcatDataset(current_subsets)

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
        t = strategy.experience.current_experience

        if t == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        current_subsets = []
        past_subsets = []
        for k in self.train_data.keys():
            for subset in self.train_data[k]:
                if k == t:
                    current_subsets.append(subset)
                else:
                    past_subsets.append(subset)

        current_data = AvalancheConcatDataset(current_subsets)
        past_data = AvalancheConcatDataset(past_subsets)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        strategy.dataloader = ReplayDataLoader(
            current_data,
            past_data,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def before_backward(self, strategy, **kwargs):
        t = strategy.experience.current_experience

        if self.model_old is not None:
            if isinstance(self.model_old, MultiTaskModule):
                out_old = []
                out_new = []
                for i in strategy.mb_task_id.unique():
                    if i != t:
                        mask = (strategy.mb_task_id == i)
                        out_old.append(self.model_old.forward_single_task(
                                    strategy.mb_x[mask].to(strategy.device), 
                                    int(i)))
                        out_new.append(strategy.model.forward_single_task(
                                    strategy.mb_x[mask].to(strategy.device), 
                                    int(i)))

                if len(out_old) == 0:
                    return strategy.loss
                out_old = torch.cat(out_old, dim=0)
                out_new = torch.cat(out_new, dim=0)
                
            else:
                out_old = self.model_old(strategy.mb_x.to(strategy.device))
                out_new = strategy.mb_output

            old_clss = []
            for c in self.class_to_tasks.keys():
                if self.class_to_tasks[c] < t:
                    old_clss.append(c)

            if isinstance(self.model_old, MultiTaskModule):
                loss_dist = self.cross_entropy(out_new,
                                               out_old, 
                                               exp=1.0 / self.T)
            else:
                loss_dist = self.cross_entropy(out_new[:, old_clss],
                                               out_old[:, old_clss], 
                                               exp=1.0 / self.T)

            if self.lamb == -1:
                lamb = len(old_clss) / len(self.seen_classes)
                return (1.0 - lamb) * strategy.loss + lamb * loss_dist
            else:
                return strategy.loss + self.lamb * loss_dist

    def after_training_exp(self, strategy, **kwargs):
        self.model_old = deepcopy(strategy.model)
        # Stage 2: Bias Correction
        t = strategy.experience.current_experience
        if t > 0:
            list_subsets = []
            for k in self.val_data.keys():
                for subset in self.val_data[k]:
                    list_subsets.append(subset)
            
            stage_set = AvalancheConcatDataset(list_subsets)
            stage_loader = DataLoader(
                                stage_set, 
                                batch_size=strategy.train_mb_size, 
                                shuffle=True,
                                num_workers=4)

            bic_optimizer = torch.optim.SGD(
                                strategy.model.bias_layers[t].parameters(), 
                                lr=self.lr, momentum=0.9)

            scheduler = MultiStepLR(bic_optimizer, milestones=[50,100, 150], 
                                    gamma=0.1, verbose=False)

            # Loop epochs
            for e in range(self.stage_2_epochs):
                total, t_acc, t_loss = 0, 0, 0
                for inputs in stage_loader:
                    x = inputs[0].to(strategy.device)
                    y_real = inputs[1].to(strategy.device)
                    task_id = inputs[2].to(strategy.device)
                    
                    loss = 0
                    if isinstance(self.model_old, MultiTaskModule):
                        mask = (task_id == t)
                        if mask.sum() > 0:
                            out = strategy.model.forward_single_task(
                                            x[mask], t)
                            loss += torch.nn.functional.cross_entropy(
                                                                out, 
                                                                y_real[mask])
                            _, preds = torch.max(out, 1)
                            t_acc += torch.sum(preds == y_real[mask].data)
                            t_loss += loss.item() * mask.sum()
                            total += mask.sum()

                    else:
                        outputs = strategy.model(x)
                        loss = torch.nn.functional.cross_entropy(
                                                            outputs, 
                                                            y_real)
                        
                        _, preds = torch.max(outputs, 1)
                        t_acc += torch.sum(preds == y_real.data)
                        t_loss += loss.item() * x.size(0)
                        total += x.size(0)

                    loss += 0.1 * ((strategy.model.bias_layers[t].beta[0] 
                                    ** 2) / 2)

                    bic_optimizer.zero_grad()
                    loss.backward()
                    bic_optimizer.step()
                
                scheduler.step()
                if (e + 1) % (int(self.stage_2_epochs / 4)) == 0:
                    print('| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |'
                          .format(e + 1, t_loss / total,
                                  100 * t_acc / total))
                                  
    def cross_entropy(self, outputs, targets, exp=1.0, 
                      eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        # out = torch.nn.functional.softmax(outputs, dim=1)
        # tar = torch.nn.functional.softmax(targets, dim=1)
        # if exp != 1:
        #     out = out.pow(exp)
        #     out = out / out.sum(1).view(-1, 1).expand_as(out)
        #     tar = tar.pow(exp)
        #     tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        # out = out + eps / out.size(1)
        # out = out / out.sum(1).view(-1, 1).expand_as(out)
        # ce = -(tar * out.log()).sum(1)
        # ce = ce.mean()
        logp = torch.nn.functional.log_softmax(outputs/self.T, dim=1)
        pre_p = torch.nn.functional.softmax(targets/self.T, dim=1)
        return -torch.mean(torch.sum(pre_p * logp, dim=1)) * self.T * self.T
