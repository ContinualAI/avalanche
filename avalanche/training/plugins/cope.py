from typing import Dict
import torch
from torch.nn.functional import normalize

from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.plugins.replay import ClassBalancedStoragePolicy
from avalanche.benchmarks.utils.data_loader import \
    MultiTaskJoinedBatchDataLoader


class CoPEPlugin(StrategyPlugin):
    """
    Continual Prototype Evolution
    This plugin does not use task identities during training or testing, it is data incremental.


    CoPE is build for online processing, implemented here as each batch being a new experience.
    Multiple epochs = multiple iterations on the same processing mini-batch.
    """

    def __init__(self, mem_size=200, p_size=100, alpha=0.9, T=0.1):
        """

        Online processing: batch-wise
        alpha= prototype update momentum
        """
        super().__init__()
        self.T = T  # PPP-loss

        # Operational memory: replay memory
        self.replay_mem = {}
        self.mem_size = mem_size  # replay memory size
        self.storage_policy = ClassBalancedStoragePolicy(replay_mem=self.replay_mem,
                                                         mem_size=self.mem_size,
                                                         adaptive_size=True)

        # Operational memory: Prototypical memory
        self.p_mem = {}  # Scales with nb classes * feature size
        self.p_size = p_size  # Prototype size determined on runtime
        self.tmp_p_mem = {}  # Intermediate when processing a batch for multiple times
        self.alpha = alpha

    def before_training(self, strategy, **kwargs):
        strategy.criterion = PPPloss(self.p_mem, T=self.T)
        print("Using the Pseudo-Prototypical-Proxy loss for CoPE.")

    def before_training_exp(self, strategy, num_workers=0, shuffle=True,
                            **kwargs):
        """
        Random retrieval from a class-balanced memory.
        Dataloader builds batches containing examples from both memories and
        the training dataset.
        """
        if len(self.replay_mem) == 0:
            return
        strategy.current_dataloader = MultiTaskJoinedBatchDataLoader(
            strategy.adapted_dataset,
            AvalancheConcatDataset(self.replay_mem.values()),
            oversample_small_tasks=False,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size * 2,  # bs = 10: 10 for new samples, 10 from replay
            shuffle=shuffle
        )

    def after_forward(self, strategy, **kwargs):
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
                    torch.empty((1, self.p_size)).uniform_(0, 1), p=2, dim=1).detach().to(strategy.device)

    @torch.no_grad()
    def _update_running_prototypes(self, strategy):
        """ Accumulate seen outputs of the network and keep counts. """
        y_unique = torch.unique(strategy.mb_y).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c = y_unique[idx].item()
            idxs = (strategy.mb_y == c).nonzero().squeeze(1)
            p_tmp_batch = strategy.logits[idxs].sum(dim=0).unsqueeze(0).to(strategy.device)

            p_init, cnt_init = self.tmp_p_mem[c] if c in self.tmp_p_mem else (0, 0)
            self.tmp_p_mem[c] = (p_init + p_tmp_batch, cnt_init + len(idxs))

    def after_training_exp(self, strategy, **kwargs):
        """ After the experience (online processing batch). """
        self._update_prototypes()  # Update prototypes
        self.storage_policy(strategy)  # Update memory

    @torch.no_grad()
    def _update_prototypes(self):
        for c, (p_sum, p_cnt) in self.tmp_p_mem.items():
            incr_p = p_sum / p_cnt
            old_p = self.p_mem[c].clone()
            new_p_momentum = self.alpha * old_p + (1 - self.alpha) * incr_p  # Momentum update
            self.tmp_p_mem[c] = normalize(new_p_momentum, p=2, dim=1).detach()  # Update L2 normalized
        self.tmp_p_mem = {}


class PPPloss(object):
    """ Pseudo-Prototypical Proxy loss.
    Uses relations in the batch to construct
    """

    def __init__(self, p_mem: Dict, T=0.1):
        """
        :param p_mem: dictionary with keys the prototype identifier and values the prototype tensors.
        :param T: temperature of the softmax, serving as concentration density parameter.
        """
        self.T = T
        self.p_mem = p_mem

    def __call__(self, x, y):
        """
        The loss is calculated with one-vs-rest batches Bc and Bk,
        split into the attractor and repellor loss terms.
        We iterate over the possible batches while accumulating the losses per class c vs other-classes k.
        """
        loss = None
        bs = x.size(0)
        x = x.view(bs, -1)  # Batch x feature size
        y_unique = torch.unique(y).squeeze().view(-1)
        include_repellor = len(y_unique.size()) <= 1  # If only from the same class, there is no neg term

        # All prototypes
        p_y = torch.tensor([c for c in self.p_mem.keys()]).to(x.device).detach()
        p_x = torch.cat([self.p_mem[c.item()] for c in p_y]).to(x.device).detach()

        for label_idx in range(y_unique.size(0)):  # Per-class operation
            c = y_unique[label_idx]

            # Make all-vs-rest batches per class
            Bc = x.index_select(0, (y == c).nonzero().squeeze(dim=1))  # Attractor set (same class)
            Bk = x.index_select(0, (y != c).nonzero().squeeze(dim=1))  # Repellor set (Other classes)

            p_idx = (p_y == c).nonzero().squeeze(dim=1)  # Prototypes
            pc = p_x[p_idx]  # Class prototype
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]])  # Other class prototypes

            # Accumulate loss for instances of class c
            sum_logLc = self.attractor(pc, pk, Bc)
            sum_logLk = self.repellor(pc, pk, Bc, Bk) if include_repellor else 0
            Loss_c = -sum_logLc - sum_logLk  # attractor + repellor for class c
            loss = Loss_c if loss is None else loss + Loss_c  # Update loss

            # Checks
            try:
                assert sum_logLc <= 0
                assert sum_logLk <= 0
                assert loss >= 0
            except:
                exit(1)
        return loss / bs  # Make independent batch size

    def attractor(self, pc, pk, Bc):
        """
        Get the attractor loss terms for all instances in xc.
        :param pc: Prototype of the same class c.
        :param pk: Prototoypes of the other classes.
        :param Bc: Batch of instances of the same class c.
        :return: Sum_{i, the part of same class c} log P(c|x_i^c)
        """
        m = torch.cat([Bc.clone(), pc, pk]).detach()  # Include all other-class prototypes p_k
        pk_idx = m.shape[0] - pk.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms (don't include self => diagonal)
        D = torch.mm(m, Bc.t()).div_(self.T).exp_()  # Distance matrix in exp terms
        mask = torch.eye(*D.shape).bool().to(Bc.device)  # Exclude product with self
        Dm = D.masked_fill(mask, 0)  # Masked out products with self

        Lc_n, Lk_d = Dm[:pk_idx], Dm[pk_idx:].sum(dim=0)  # Numerator/denominator terms
        Pci = Lc_n / (Lc_n + Lk_d)  # Get probabilities per instance
        E_Pc = Pci.sum(0) / (Bc.shape[0])  # Expectation over pseudo-prototypes
        return E_Pc.log_().sum()  # sum over all instances (sum i)

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

        E_Pk = (Pki[:-1] + Pki[-1].unsqueeze(0)) / 2  # Expectation pseudo/prototype
        inv_E_Pk = E_Pk.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        return inv_E_Pk.sum()  # Sum over (pseudo-prototypes), and instances
