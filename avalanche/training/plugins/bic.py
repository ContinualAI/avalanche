import random
from copy import deepcopy

from torch.utils.data import DataLoader
from avalanche.models.dynamic_modules import MultiTaskModule
import torch

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset


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
        mem_size: int = 200, stage_2_epochs: int = 200, lamb: float = -1, 
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

        self.examplars_data = {}
        self.examplars_idx = {}
        self.seen_classes = set()
        self.class_to_tasks = {}

        self.model_old = None

    def before_train_dataset_adaptation(self, strategy, **kwargs):
        # Create a SubSet for the task with num_per_class elements
        # Remove extra elements from past SubSets (max num_per_class)
        # Iterate over all (past and current) tasks and create val and train
        #   Past tasks use self.val_percentage
        #   Current task num_val_cls and the rest 

        # self.bias_layers.append(BiasLayer(strategy.device))

        new_data = strategy.experience.dataset
        task_id = strategy.experience.current_experience

        cl_idxs = {}
        for idx, target in enumerate(new_data.targets):
            if target not in cl_idxs:
                cl_idxs[target] = []
            cl_idxs[target].append(idx)

        for c in cl_idxs.keys():
            self.class_to_tasks[c] = task_id

        strategy.model.add_bias_layer(strategy.device, cl_idxs.keys())
        self.seen_classes.update(cl_idxs.keys())
        num_per_class = self.mem_size // len(self.seen_classes)

        for t in self.examplars_idx.keys():  # If class is in multiple exp
            for c in self.examplars_idx[t].keys():
                if len(self.examplars_idx[t][c]) > num_per_class:
                    self.examplars_idx[t][c] = \
                            self.examplars_idx[t][c][:num_per_class]
        
        self.exemplar_stage_2 = {}
        self.exemplar_stage_1 = {}
        if task_id > 0:
            num_val_cls = int(self.val_percentage * num_per_class)
            for t in self.examplars_idx.keys():
                self.exemplar_stage_2[t] = []
                self.exemplar_stage_1[t] = []
                for c in self.examplars_idx[t].keys():
                    self.exemplar_stage_2[t] += \
                        self.examplars_idx[t][c][:num_val_cls]
                    self.exemplar_stage_1[t] += \
                        self.examplars_idx[t][c][num_val_cls:]
            
            self.exemplar_stage_2[task_id] = []
            self.exemplar_stage_1[task_id] = []
            for k in cl_idxs.keys():
                w = torch.rand(len(cl_idxs[k]))
                _, sorted_idx = w.sort(descending=True)
                self.exemplar_stage_2[task_id] += sorted_idx[:num_val_cls]
                self.exemplar_stage_1[task_id] += sorted_idx[num_val_cls:]

        self.examplars_idx[task_id] = {}
        for k in cl_idxs.keys():
            self.examplars_idx[task_id][k] = random.sample(cl_idxs[k], 
                                                           num_per_class)

        self.examplars_data[task_id] = new_data

        if task_id > 0:
            list_dataset = []
            for k in self.exemplar_stage_1.keys():
                list_dataset.append(AvalancheSubset(self.examplars_data[k],
                                                    self.exemplar_stage_1[k]))
            
            strategy.experience.dataset = AvalancheConcatDataset(list_dataset)

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
            list_dataset = []
            for k in self.exemplar_stage_2.keys():
                list_dataset.append(AvalancheSubset(self.examplars_data[k],
                                                    self.exemplar_stage_2[k],
                                                    task_labels=k))
            
            stage_set = AvalancheConcatDataset(list_dataset)
            stage_loader = DataLoader(
                                stage_set, 
                                batch_size=strategy.train_mb_size, 
                                shuffle=True,
                                num_workers=4)

            bic_optimizer = torch.optim.SGD(
                                strategy.model.bias_layers[t].parameters(), 
                                lr=self.lr, momentum=0.9)

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
                
                if (e + 1) % (self.stage_2_epochs / 4) == 0:
                    print('| E {:3d} | Train: loss={:.3f}, S2 acc={:5.1f}% |'
                          .format(e + 1, t_loss / total,
                                  100 * t_acc / total))
                                  
    def cross_entropy(self, outputs, targets, exp=1.0, 
                      size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce
