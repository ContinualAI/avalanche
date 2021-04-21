from typing import Dict

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch.nn.modules import Module

from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheConcatDataset, AvalancheDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.plugins.replay import ClassBalancedStoragePolicy
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskMultiBatchDataLoader


class CoPEPlugin(StrategyPlugin):
    """
    Continual Prototype Evolution
    This plugin does not use task identities during training or testing, it is data incremental.


    CoPE is build for online processing, implemented here as each batch being a new experience.
    Multiple epochs = multiple iterations on the same processing mini-batch.
    """

    def __init__(self, mem_size=200, n_classes=10, p_size=100,
                 alpha=0.99, T=0.1):
        """

        Online processing: batch-wise
        alpha= prototype update momentum
        """
        super().__init__()
        self.n_classes = n_classes

        # Operational memory: replay memory
        self.replay_mem = {}
        self.mem_size = mem_size  # replay memory size
        self.storage_policy = ClassBalancedStoragePolicy(
            replay_mem=self.replay_mem,
            mem_size=self.mem_size,
            adaptive_size=True)

        # Operational memory: Prototypical memory
        self.p_mem = {}  # Scales with nb classes * feature size
        self.p_size = p_size  # Prototype size determined on runtime
        self.tmp_p_mem = {}  # Intermediate when processing a batch for multiple times
        self.alpha = alpha

        # PPP-loss
        self.T = T
        self.loss = PPPloss(self.p_mem, T=self.T)

    def before_training(self, strategy, **kwargs):
        """ Enforce using the PPP-loss and add a NN-classifier."""
        strategy.criterion = self.loss
        print("Using the Pseudo-Prototypical-Proxy loss for CoPE.")

        # The network should contain a 'classifier' (typically used for sofmtax)
        assert hasattr(strategy.model, 'classifier')

        strategy.model.classifier = self._nearest_neigbor_classifier(
            strategy.model.classifier)

    def _nearest_neigbor_classifier(self, last_layer):
        nn = NearestNeigborClassifier(self.p_mem, self.n_classes)
        return torch.nn.Sequential(last_layer, nn)

    def before_forward(self, strategy, num_workers=0, shuffle=True,
                       **kwargs):
        """
        Random retrieval from a class-balanced memory at each batch.
        Dataloader builds batches containing examples from both memories and
        the training dataset.
        """
        if len(self.replay_mem) == 0:
            return
        mem_dataloader = MultiTaskMultiBatchDataLoader(
            AvalancheConcatDataset(self.replay_mem.values()),
            oversample_small_tasks=False,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,  # A batch of replay samples
            shuffle=shuffle
        )

        mb_x, mb_y, mb_task_id = next(iter(mem_dataloader))[0]
        mb_x, mb_y = mb_x.to(strategy.device), mb_y.to(strategy.device)

        # Add to current batch
        strategy.mb_x = torch.cat((strategy.mb_x, mb_x))
        strategy.mb_y = torch.cat((strategy.mb_y, mb_y))

    def after_forward(self, strategy, **kwargs):
        """
        After the forward we can use the representations to update our running avg of the prototypes.
        This is in case we do multiple iterations of processing on the same batch.

        New prototypes are initialized for previously unseen classes.
        """
        # Initialize prototypes for unseen classes
        self._init_new_prototypes(strategy)

        # Update batch info over multiple epochs (multiple iterations same batch)
        # Actual update is in after_exp
        self._update_running_prototypes(strategy)

    @torch.no_grad()
    def _init_new_prototypes(self, strategy):
        """Initialize prototypes for previously unseen classes."""
        y_unique = torch.unique(strategy.mb_y).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c = y_unique[idx].item()
            if c not in self.p_mem:  # Init new prototype
                self.p_mem[c] = normalize(
                    torch.empty((1, self.p_size)).uniform_(0, 1), p=2,
                    dim=1).detach().to(strategy.device)

    @torch.no_grad()
    def _update_running_prototypes(self, strategy):
        """ Accumulate seen outputs of the network and keep counts. """
        y_unique = torch.unique(strategy.mb_y).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c = y_unique[idx].item()
            idxs = torch.nonzero(strategy.mb_y == c).squeeze(1)
            p_tmp_batch = strategy.logits[idxs].sum(dim=0).unsqueeze(0).to(
                strategy.device)

            p_init, cnt_init = self.tmp_p_mem[c] if c in self.tmp_p_mem else (
                0, 0)
            self.tmp_p_mem[c] = (p_init + p_tmp_batch, cnt_init + len(idxs))

    def after_update(self, strategy, **kwargs):
        """ After the update on current batch, update prototypes and store samples.
        """
        self._update_prototypes()  # Update prototypes

        batch_data = AvalancheDataset(
            TensorDataset(strategy.mb_x, strategy.mb_y))
        self.storage_policy(batch_data)  # Update memory

    @torch.no_grad()
    def _update_prototypes(self):
        """ Update the prototypes based on the running averages. """
        for c, (p_sum, p_cnt) in self.tmp_p_mem.items():
            incr_p = normalize(p_sum / p_cnt, p=2, dim=1)  # L2 normalized
            old_p = self.p_mem[c].clone()
            new_p_momentum = self.alpha * old_p + (
                    1 - self.alpha) * incr_p  # Momentum update
            self.tmp_p_mem[c] = normalize(new_p_momentum, p=2, dim=1).detach()
        self.tmp_p_mem = {}


class NearestNeigborClassifier(Module):
    """
    At training time is Identity function. At evaluation time, matches
    representation with prototypes to produce similarity scores.
    """

    def __init__(self, p_mem, n_classes):
        super().__init__()
        self.p_mem = p_mem  # Reference to prototype dict
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:
        if self.training:  # Identity
            return x
        else:
            return self.nearest_neigbor(x)

    def nearest_neigbor(self, x: Tensor) -> Tensor:
        """ Deployment forward. Find closest prototype for samples in batch. """
        ns = x.size(0)
        nd = x.view(ns, -1).shape[-1]

        # Get prototypes
        seen_c = len(self.p_mem.keys())
        if seen_c == 0:  # no prototypes yet, output uniform distr. all classes
            return torch.Tensor(ns, self.n_classes
                                ).fill_(1.0 / self.n_classes).to(x.device)
        means = torch.ones(seen_c, nd).to(x.device) * float('inf')
        for c, c_proto in self.p_mem.items():
            means[c] = c_proto  # Class idx gets allocated its prototype

        # Predict nearest mean
        classpred = torch.LongTensor(ns)
        for s_idx in range(ns):  # Per sample
            dist = - torch.mm(means, x[s_idx].unsqueeze(-1))  # Dot product
            _, ii = dist.min(0)  # Min dist (no proto = inf)
            ii = ii.squeeze()
            classpred[s_idx] = ii.item()  # Allocate class idx

        # Convert to 1-hot
        out = torch.zeros(ns, self.n_classes).to(x.device)
        for s_idx in range(ns):
            out[s_idx, classpred[s_idx]] = 1
        return out  # return 1-of-C code, ns x nc


class PPPloss(object):
    """ Pseudo-Prototypical Proxy loss.
    Uses relations in the batch to construct
    """

    def __init__(self, p_mem: Dict, T=0.1):
        """
        :param p_mem: dictionary with keys the prototype identifier and
                      values the prototype tensors.
        :param T: temperature of the softmax, serving as concentration
                  density parameter.
        """
        self.T = T
        self.p_mem = p_mem

    def __call__(self, x, y):
        """
        The loss is calculated with one-vs-rest batches Bc and Bk,
        split into the attractor and repellor loss terms.
        We iterate over the possible batches while accumulating the losses per
        class c vs other-classes k.
        """
        loss = None
        bs = x.size(0)
        x = x.view(bs, -1)  # Batch x feature size
        y_unique = torch.unique(y).squeeze().view(-1)
        include_repellor = len(y_unique.size()) <= 1  # When at least 2 classes

        # All prototypes
        p_y = torch.tensor(
            [c for c in self.p_mem.keys()]).to(x.device).detach()
        p_x = torch.cat(
            [self.p_mem[c.item()] for c in p_y]).to(x.device).detach()

        for label_idx in range(y_unique.size(0)):  # Per-class operation
            c = y_unique[label_idx]

            print("Class {}:".format(str(c.item())), end='')  # TODO rm

            # Make all-vs-rest batches per class (Bc=attractor, Bk=repellor set)
            Bc = x.index_select(0, torch.nonzero(y == c).squeeze(dim=1))
            Bk = x.index_select(0, torch.nonzero(y != c).squeeze(dim=1))

            p_idx = torch.nonzero(p_y == c).squeeze(dim=1)  # Prototypes
            pc = p_x[p_idx]  # Class proto
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]])  # Other class proto

            # Accumulate loss for instances of class c
            sum_logLc = self.attractor(pc, pk, Bc)
            sum_logLk = self.repellor(pc, pk, Bc, Bk) if include_repellor else 0
            Loss_c = -sum_logLc - sum_logLk  # attractor + repellor for class c
            loss = Loss_c if loss is None else loss + Loss_c  # Update loss

            print("{: >20}".format(
                "| TOTAL: {:.1f} + {:.1f} = {:.1f}".format(float(-sum_logLc),
                                                           float(-sum_logLk),
                                                           float(Loss_c))))

            # print(
            #     f'sum_logLc={sum_logLc}\tsum_logLk={sum_logLk}\tLoss_c={Loss_c}')

            # Checks
            try:
                assert sum_logLc <= 0, f"{sum_logLc}"
                assert sum_logLk <= 0, f"{sum_logLk}"
                assert loss >= 0
            except:
                import traceback
                traceback.print_exc()
                exit(1)

        print()
        print("-" * 40)  # TODO RM
        return loss / bs  # Make independent batch size

    def attractor(self, pc, pk, Bc):
        """
        Get the attractor loss terms for all instances in xc.
        :param pc: Prototype of the same class c.
        :param pk: Prototoypes of the other classes.
        :param Bc: Batch of instances of the same class c.
        :return: Sum_{i, the part of same class c} log P(c|x_i^c)
        """
        m = torch.cat([Bc.clone(), pc, pk]).detach()  # Incl other-class proto
        pk_idx = m.shape[0] - pk.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms
        D = torch.mm(m, Bc.t()).div_(self.T).exp_()  # Distance matrix exp terms
        mask = torch.eye(*D.shape).bool().to(Bc.device)  # Exclude self-product
        Dm = D.masked_fill(mask, 0)  # Masked out products with self

        Lc_n, Lk_d = Dm[:pk_idx], Dm[pk_idx:].sum(dim=0)  # Num/denominator
        Pci = Lc_n / (Lc_n + Lk_d)  # Get probabilities per instance
        E_Pc = Pci.sum(0) / Bc.shape[0]  # Expectation over pseudo-prototypes
        # return E_Pc.log_().sum()  # sum over all instances (sum i) # TODO PUT BACK
        final = E_Pc.log_().sum()  # sum over all instances (sum i)

        print("(#c) {:.1f}/({:.1f} + {:.1f}) ".format(  # TODO rm
            float(Lc_n.mean().item()), float(Lc_n.mean().item()),
            float(Lk_d.mean().item())),
            end='')

        return final

    def repellor(self, pc, pk, Bc, Bk):
        """
        Get the repellor loss terms for all pseudo-prototype instances in Bc.
        :param pc: Actual prototype of the same class c.
        :param pk: Prototoypes of the other classes (k).
        :param Bc: Batch of instances of the same class c. Acting as pseudo-prototypes.
        :param Bk: Batch of instances of other-than-c classes (k).
        :return: Sum_{i, the part of same class c} Sum_{x_j^k} log 1 - P(c|x_j^k)
        """
        union_ck = torch.cat([Bc, pc, pk])
        pk_idx = union_ck.shape[0] - pk.shape[0]

        # Distance of other-class-k to prototypes (pc/pk) and pseudo-prototype (xc)
        D = torch.mm(union_ck, Bk.t()).div_(self.T).exp_()

        Lk_d = D[pk_idx:].sum(dim=0).unsqueeze(0)  # Numerator/denominator terms
        Lc_n = D[:pk_idx]
        Pki = Lc_n / (Lc_n + Lk_d)  # probability

        E_Pk = (Pki[:-1] + Pki[-1].unsqueeze(
            0)) / 2  # Expectation pseudo/prototype
        inv_E_Pk = E_Pk.mul_(-1).add_(1).log_()  # log( (1 - Pk))

        print(" + (#k) {:.1f}/({:.1f} + {:.1f})".format(  # TODO rm
            float(Lc_n.mean().item()),
            float(Lc_n.mean().item()),
            float(Lk_d.mean().item())), end='')

        return inv_E_Pk.sum()  # Sum over (pseudo-prototypes), and instances
