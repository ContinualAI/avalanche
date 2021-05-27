import copy
import itertools
from typing import TYPE_CHECKING
import torch
from avalanche.benchmarks.utils import AvalancheConcatDataset, \
    AvalancheTensorDataset, AvalancheSubset
from torch import nn
from math import ceil
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from torch.nn import BCELoss
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class ICaRLPlugin(StrategyPlugin):
    def __init__(self, memory_size, diff_transform=None, fixed_memory=True):
        super().__init__()

        self.memory_size = memory_size
        self.diff_transform = diff_transform
        self.fixed_memory = fixed_memory

        self.x_memory = []
        self.y_memory = []
        self.order = []

        self.old_model = None
        self.observed_classes = []
        self.class_means = None
        self.embedding_size = None
        self.output_size = None
        self.input_size = None

        self.setup = False

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        if not self.setup:
            strategy.model = ICaRLModel(strategy.model.feature_extractor,
                                        train_head=strategy.model.classifier,
                                        eval_head=NCMClassifier())
            self.setup = True

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):

        if strategy.training_exp_counter != 0:
            memory = AvalancheTensorDataset(
                torch.cat(self.x_memory).cpu(),
                list(itertools.chain.from_iterable(self.y_memory)),
                transform=self.diff_transform, target_transform=None)

            strategy.adapted_dataset = \
                AvalancheConcatDataset((strategy.adapted_dataset, memory))

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        tid = strategy.training_exp_counter
        scenario = strategy.experience.scenario
        nb_cl = scenario.n_classes_per_exp[tid]

        self.observed_classes.extend(
            scenario.classes_order[tid * nb_cl:(tid + 1) * nb_cl])

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        if self.input_size is None:
            with torch.no_grad():
                self.input_size = strategy.mb_x.shape[1:]
                self.output_size = strategy.model(strategy.mb_x).shape[1]
                self.embedding_size = strategy.model.feature_extractor(
                    strategy.mb_x).shape[1]

        tid = strategy.training_exp_counter

        if tid > 0:
            scenario = strategy.experience.scenario
            old_classes = scenario.classes_in_exp_range(0, tid)
            self.old_model.feature_extractor.eval()
            self.old_model.feature_extractor.eval()
            with torch.no_grad():
                embds = self.old_model.feature_extractor(strategy.mb_x)
                old_logits = self.old_model.train_head(embds)

            strategy.criterion.set_old(old_classes, old_logits)

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):

        if strategy.training_exp_counter == 0:
            old_model = copy.deepcopy(strategy.model)
            self.old_model = old_model.to(strategy.device)

        self.old_model.load_state_dict(strategy.model.state_dict())

        strategy.model.eval()

        self.construct_exemplar_set(strategy)
        self.reduce_exemplar_set(strategy)
        self.compute_class_means(strategy)

    def compute_class_means(self, strategy):
        if self.class_means is None:
            n_classes = sum(strategy.experience.scenario.n_classes_per_exp)
            self.class_means = torch.zeros(
                (self.embedding_size, n_classes)).to(strategy.device)

        for i, class_samples in enumerate(self.x_memory):
            l = self.y_memory[i][0]
            class_samples = class_samples.to(strategy.device)

            with torch.no_grad():
                mapped_prototypes = strategy.model.feature_extractor(
                    class_samples).detach()
            D = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            with torch.no_grad():
                mapped_prototypes2 = strategy.model.feature_extractor(
                    torch.flip(class_samples, [3])).detach()

            D2 = mapped_prototypes2.T
            D2 = D2 / torch.norm(D2, dim=0)

            div = torch.ones(class_samples.shape[0], device=strategy.device) \
                  / class_samples.shape[0]

            m1 = torch.mm(D, div.unsqueeze(1)).squeeze(1)
            m2 = torch.mm(D2, div.unsqueeze(1)).squeeze(1)
            self.class_means[:, l] = (m1 + m2) / 2
            self.class_means[:, l] /= torch.norm(self.class_means[:, l])

            strategy.model.eval_head.class_means = self.class_means

    def construct_exemplar_set(self, strategy):
        tid = strategy.training_exp_counter
        nb_cl = strategy.experience.scenario.n_classes_per_exp

        if self.fixed_memory:
            nb_protos_cl = int(ceil(
                self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size
        new_classes = self.observed_classes[tid*nb_cl[tid]:(tid+1)*nb_cl[tid]]

        dataset = strategy.experience.dataset
        targets = torch.tensor(dataset.targets)
        for iter_dico in range(nb_cl[tid]):
            cd = AvalancheSubset(dataset,
                                 torch.where(targets == new_classes[iter_dico])
                                 [0])

            class_patterns, _, _ = next(iter(
                DataLoader(cd.eval(), batch_size=len(cd))))
            class_patterns = class_patterns.to(strategy.device)

            with torch.no_grad():
                mapped_prototypes = strategy.model.feature_extractor(
                    class_patterns).detach()
            D = mapped_prototypes.T
            D = D / torch.norm(D, dim=0)

            mu = torch.mean(D, dim=1)
            order = torch.zeros(class_patterns.shape[0])
            w_t = mu

            i, added, selected = 0, 0, []
            while not added == nb_protos_cl and i < 1000:
                tmp_t = torch.mm(w_t.unsqueeze(0), D)
                ind_max = torch.argmax(tmp_t)

                if ind_max not in selected:
                    order[ind_max] = 1 + added
                    added += 1
                    selected.append(ind_max.item())

                w_t = w_t + mu - D[:, ind_max]
                i += 1

            pick = (order > 0) * (order < nb_protos_cl + 1) * 1.
            self.x_memory.append(class_patterns[torch.where(pick == 1)[0]])
            self.y_memory.append(
                [new_classes[iter_dico]]*len(torch.where(pick == 1)[0]))
            self.order.append(order[torch.where(pick == 1)[0]])

    def reduce_exemplar_set(self, strategy):
        tid = strategy.training_exp_counter
        nb_cl = strategy.experience.scenario.n_classes_per_exp

        if self.fixed_memory:
            nb_protos_cl = int(ceil(
                self.memory_size / len(self.observed_classes)))
        else:
            nb_protos_cl = self.memory_size

        for i in range(len(self.x_memory) - nb_cl[tid]):
            pick = (self.order[i] < nb_protos_cl + 1) * 1.
            self.x_memory[i] = self.x_memory[i][torch.where(pick == 1)[0]]
            self.y_memory[i] = self.y_memory[i][:len(torch.where(pick==1)[0])]
            self.order[i] = self.order[i][torch.where(pick == 1)[0]]


class DistillationLoss:
    def __init__(self):
        self.criterion = BCELoss()
        self.old_classes = None
        self.old_logits = None

    def set_old(self, old_classes, old_logits):
        self.old_classes = old_classes
        self.old_logits = old_logits

    def __call__(self, logits, targets):
        predictions = torch.sigmoid(logits)

        one_hot = torch.zeros(targets.shape[0], logits.shape[1]
                              , dtype=torch.float, device=logits.device)
        one_hot[range(len(targets)), targets.long()] = 1

        if self.old_classes is not None:
            old_predictions = torch.sigmoid(self.old_logits)
            one_hot[:, self.old_classes] = old_predictions[:, self.old_classes]
            self.old_classes, self.old_logits = None, None

        return self.criterion(predictions, one_hot)


class NCMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_means = None

    def forward(self, x):
        pred_inter = (x.T / torch.norm(x.T, dim=0)).T
        sqd = torch.cdist(self.class_means[:, :].T, pred_inter)
        return (-sqd).T


class ICaRLModel(nn.Module):
    def __init__(self, feature_extractor, train_head, eval_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.train_head = train_head
        self.eval_head = eval_head

        self.classifier = train_head

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def train(self, mode=True):
        super().train(mode)
        self.classifier = self.train_head if mode else self.eval_head
