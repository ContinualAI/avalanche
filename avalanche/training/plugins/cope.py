from typing import Dict

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch.nn.modules import Module

from avalanche.training.utils import get_last_fc_layer, swap_last_fc_layer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader


class CoPEPlugin(SupervisedPlugin):
    """Continual Prototype Evolution plugin.

    Each class has a prototype for nearest-neighbor classification.
    The prototypes are updated continually with an exponentially moving average,
    using class-balanced replay to keep the prototypes up-to-date.
    The embedding space is optimized using the PseudoPrototypicalProxy-loss,
    exploiting both prototypes and batch information.

    This plugin doesn't use task identities in training or eval
    (data incremental) and is designed for online learning (1 epoch per task).
    """

    def __init__(
        self,
        mem_size=200,
        n_classes=10,
        p_size=100,
        alpha=0.99,
        T=0.1,
        max_it_cnt=1,
    ):
        """
        :param mem_size: max number of input samples in the replay memory.
        :param n_classes: total number of classes that will be encountered. This
            is used to output predictions for all classes, with zero probability
            for unseen classes.
        :param p_size: The prototype size, which equals the feature size of the
            last layer.
        :param alpha: The momentum for the exponentially moving average of the
            prototypes.
        :param T: The softmax temperature, used as a concentration parameter.
        :param max_it_cnt: How many processing iterations per batch (experience)
        """
        super().__init__()
        self.n_classes = n_classes
        self.it_cnt = 0
        self.max_it_cnt = max_it_cnt

        # Operational memory: replay memory
        self.mem_size = mem_size  # replay memory size
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )

        # Operational memory: Prototypical memory
        # Scales with nb classes * feature size
        self.p_mem: Dict[int, Tensor] = {}
        self.p_size = p_size  # Prototype size determined on runtime
        self.tmp_p_mem = {}  # Intermediate to process batch for multiple times
        self.alpha = alpha
        self.p_init_adaptive = False  # Only create proto when class seen

        # PPP-loss
        self.T = T
        self.ppp_loss = PPPloss(self.p_mem, T=self.T)

        self.initialized = False

    def before_training(self, strategy, **kwargs):
        """Enforce using the PPP-loss and add a NN-classifier."""
        if not self.initialized:
            strategy._criterion = self.ppp_loss
            print("Using the Pseudo-Prototypical-Proxy loss for CoPE.")

            # Normalize representation of last layer
            swap_last_fc_layer(
                strategy.model,
                torch.nn.Sequential(
                    get_last_fc_layer(strategy.model)[1], L2Normalization()
                ),
            )

            # Static prototype init
            # Create prototypes for all classes at once
            if not self.p_init_adaptive and len(self.p_mem) == 0:
                self._init_new_prototypes(
                    torch.arange(0, self.n_classes).to(strategy.device)
                )

            self.initialized = True

    def before_training_exp(self, strategy, num_workers=0, shuffle=True, **kwargs):
        """
        Random retrieval from a class-balanced memory.
        Dataloader builds batches containing examples from both memories and
        the training dataset.
        This implementation requires the use of early stopping, otherwise the
        entire memory will be iterated.
        """
        if len(self.storage_policy.buffer) == 0:
            return
        self.it_cnt = 0
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=False,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            batch_size_mem=strategy.train_mb_size,
            shuffle=shuffle,
        )

    def after_training_iteration(self, strategy, **kwargs):
        """
        Implements early stopping, determining how many subsequent times a
        batch can be used for updates. The dataloader contains only data for
        the current experience (batch) and the entire memory.
        Multiple iterations will hence result in the original batch with new
        exemplars sampled from the memory for each iteration.
        """
        self.it_cnt += 1
        if self.it_cnt == self.max_it_cnt:
            strategy.stop_training()  # Stop processing the new-data batch

    def after_forward(self, strategy, **kwargs):
        """
        After the forward we can use the representations to update our running
        avg of the prototypes. This is in case we do multiple iterations of
        processing on the same batch.

        New prototypes are initialized for previously unseen classes.
        """

        if self.p_init_adaptive:  # Init prototypes for unseen classes in batch
            self._init_new_prototypes(strategy.mb_y)

        # Update batch info (when multiple iterations on same batch)
        self._update_running_prototypes(strategy)

    @torch.no_grad()
    def _init_new_prototypes(self, targets: Tensor):
        """Initialize prototypes for previously unseen classes.
        :param targets: The targets Tensor to make prototypes for.
        """
        y_unique: Tensor = torch.unique(targets).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c: int = y_unique[idx].item()
            if c not in self.p_mem:  # Init new prototype
                self.p_mem[c] = (
                    normalize(
                        torch.empty((1, self.p_size)).uniform_(-1, 1),
                        p=2,
                        dim=1,
                    )
                    .detach()
                    .to(targets.device)
                )

    @torch.no_grad()
    def _update_running_prototypes(self, strategy):
        """Accumulate seen outputs of the network and keep counts."""
        y_unique = torch.unique(strategy.mb_y).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c = y_unique[idx].item()
            idxs = torch.nonzero(strategy.mb_y == c).squeeze(1)
            p_tmp_batch = (
                strategy.mb_output[idxs].sum(dim=0).unsqueeze(0).to(strategy.device)
            )

            p_init, cnt_init = self.tmp_p_mem[c] if c in self.tmp_p_mem else (0, 0)
            self.tmp_p_mem[c] = (p_init + p_tmp_batch, cnt_init + len(idxs))

    def after_training_exp(self, strategy, **kwargs):
        """After the current experience (batch), update prototypes and
        store observed samples for replay.
        """
        self._update_prototypes()  # Update prototypes
        self.storage_policy.update(strategy)  # Update memory

    @torch.no_grad()
    def _update_prototypes(self):
        """Update the prototypes based on the running averages."""
        for c, (p_sum, p_cnt) in self.tmp_p_mem.items():
            incr_p = normalize(p_sum / p_cnt, p=2, dim=1)  # L2 normalized
            old_p = self.p_mem[c].clone()
            new_p_momentum = (
                self.alpha * old_p + (1 - self.alpha) * incr_p
            )  # Momentum update
            self.p_mem[c] = normalize(new_p_momentum, p=2, dim=1).detach()
        self.tmp_p_mem = {}

    def after_eval_iteration(self, strategy, **kwargs):
        """Convert output scores to probabilities for other metrics like
        accuracy and forgetting. We only do it at this point because before
        this,we still need the embedding outputs to obtain the PPP-loss."""
        strategy.mb_output = self._get_nearest_neigbor_distr(strategy.mb_output)

    def _get_nearest_neigbor_distr(self, x: Tensor) -> Tensor:
        """
        Find closest prototype for output samples in batch x.
        :param x: Batch of network logits.
        :return: one-hot representation of the predicted class.
        """
        ns = x.size(0)
        nd = x.view(ns, -1).shape[-1]

        # Get prototypes
        seen_c = len(self.p_mem.keys())
        if seen_c == 0:  # no prototypes yet, output uniform distr. all classes
            return (
                torch.Tensor(ns, self.n_classes)
                .fill_(1.0 / self.n_classes)
                .to(x.device)
            )
        means = torch.ones(seen_c, nd).to(x.device) * float("inf")
        for c, c_proto in self.p_mem.items():
            means[c] = c_proto  # Class idx gets allocated its prototype

        # Predict nearest mean
        classpred = torch.LongTensor(ns)
        for s_idx in range(ns):  # Per sample
            dist = -torch.mm(means, x[s_idx].unsqueeze(-1))  # Dot product
            _, ii = dist.min(0)  # Min dist (no proto = inf)
            ii = ii.squeeze()
            classpred[s_idx] = ii.item()  # Allocate class idx

        # Convert to 1-hot
        out = torch.zeros(ns, self.n_classes).to(x.device)
        for s_idx in range(ns):
            out[s_idx, classpred[s_idx]] = 1
        return out  # return 1-of-C code, ns x nc


class L2Normalization(Module):
    """Module to L2-normalize the input. Typically used in last layer to
    normalize the embedding."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=1)


class PPPloss(object):
    """Pseudo-Prototypical Proxy loss (PPP-loss).
    This is a contrastive loss using prototypes and representations of the
    samples in the batch to optimize the embedding space.
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
        p_y = torch.tensor([c for c in self.p_mem.keys()]).to(x.device).detach()
        p_x = torch.cat([self.p_mem[c.item()] for c in p_y]).to(x.device).detach()

        for label_idx in range(y_unique.size(0)):  # Per-class operation
            c = y_unique[label_idx]

            # Make all-vs-rest batches per class (Bc=attractor, Bk=repellor set)
            Bc = x.index_select(0, torch.nonzero(y == c).squeeze(dim=1))
            Bk = x.index_select(0, torch.nonzero(y != c).squeeze(dim=1))

            p_idx = torch.nonzero(p_y == c).squeeze(dim=1)  # Prototypes
            pc = p_x[p_idx]  # Class proto
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1 :]]).clone().detach()

            # Accumulate loss for instances of class c
            sum_logLc = self.attractor(pc, pk, Bc)
            sum_logLk = self.repellor(pc, pk, Bc, Bk) if include_repellor else 0
            Loss_c = -sum_logLc - sum_logLk  # attractor + repellor for class c
            loss = Loss_c if loss is None else loss + Loss_c  # Update loss
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
        return E_Pc.log_().sum()  # sum over all instances (sum i)

    def repellor(self, pc, pk, Bc, Bk):
        """
        Get the repellor loss terms for all pseudo-prototype instances in Bc.
        :param pc: Actual prototype of the same class c.
        :param pk: Prototoypes of the other classes (k).
        :param Bc: Batch of instances of the same class c. Acting as
        pseudo-prototypes.
        :param Bk: Batch of instances of other-than-c classes (k).
        :return: Sum_{i, part of same class c} Sum_{x_j^k} log 1 - P(c|x_j^k)
        """
        union_ck = torch.cat([Bc.clone(), pc, pk]).detach()
        pk_idx = union_ck.shape[0] - pk.shape[0]

        # Distance other-class-k to prototypes (pc/pk) and pseudo-prototype (xc)
        D = torch.mm(union_ck, Bk.t()).div_(self.T).exp_()

        Lk_d = D[pk_idx:].sum(dim=0).unsqueeze(0)  # Numerator/denominator terms
        Lc_n = D[:pk_idx]
        Pki = Lc_n / (Lc_n + Lk_d)  # probability

        E_Pk = (Pki[:-1] + Pki[-1].unsqueeze(0)) / 2  # Exp. pseudo/prototype
        inv_E_Pk = E_Pk.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        return inv_E_Pk.sum()  # Sum over (pseudo-prototypes), and instances
