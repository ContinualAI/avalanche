import random

import torch

from avalanche.training.plugins.strategy_plugin import StrategyPlugin


class AGEMPlugin(StrategyPlugin):
    """
    Average Gradient Episodic Memory Plugin.
    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.sample_size = int(sample_size)

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """

        if self.memory_x is not None:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            xref, yref = self.sample_from_memory(self.sample_size)
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)
            out = strategy.model(xref)
            loss = strategy._criterion(out, yref)
            loss.backward()
            self.reference_gradients = [
                (n, p.grad)
                for n, p in strategy.model.named_parameters()]

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if self.memory_x is not None:
            for (n1, p1), (n2, refg) in zip(strategy.model.named_parameters(),
                                            self.reference_gradients):

                assert n1 == n2, "Different model parameters in AGEM projection"
                assert (p1.grad is not None and refg is not None) or \
                       (p1.grad is None and refg is None)

                if refg is None:
                    continue

                dotg = torch.dot(p1.grad.view(-1), refg.view(-1))
                dotref = torch.dot(refg.view(-1), refg.view(-1))
                if dotg < 0:
                    p1.grad -= (dotg / dotref) * refg

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.update_memory(strategy.dataloader)

    def sample_from_memory(self, sample_size):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """

        if self.memory_x is None or self.memory_y is None:
            raise ValueError('Empty memory for AGEM.')

        if self.memory_x.size(0) <= sample_size:
            return self.memory_x, self.memory_y
        else:
            idxs = random.sample(range(self.memory_x.size(0)), sample_size)
            return self.memory_x[idxs], self.memory_y[idxs]

    @torch.no_grad()
    def update_memory(self, dataloader):
        """
        Update replay memory with patterns from current experience.
        """

        tot = 0
        for mbatch in dataloader:
            x, y, _ = mbatch
            if tot + x.size(0) <= self.patterns_per_experience:
                if self.memory_x is None:
                    self.memory_x = x.clone()
                    self.memory_y = y.clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y), dim=0)
            else:
                diff = self.patterns_per_experience - tot
                if self.memory_x is None:
                    self.memory_x = x[:diff].clone()
                    self.memory_y = y[:diff].clone()
                else:
                    self.memory_x = torch.cat((self.memory_x,
                                               x[:diff]), dim=0)
                    self.memory_y = torch.cat((self.memory_y,
                                               y[:diff]), dim=0)
                break
            tot += x.size(0)
