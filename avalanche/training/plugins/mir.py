import copy
from typing import TYPE_CHECKING
import torch
from avalanche.benchmarks.utils import concat_datasets
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def update_temp(model, grad, lr):
    model_copy = copy.deepcopy(model)
    for g, p in zip(grad, model_copy.parameters()):
        if g is not None:
            p.data = p.data - lr * g
    return model_copy


class MIRPlugin(SupervisedPlugin):
    """
    Maximally Interfered Retrieval plugin,
    Implements the strategy defined in
    "Online Continual Learning with Maximally Interfered Retrieval"
    https://arxiv.org/abs/1908.04742

    This strategy has been designed and tested in the
    Online Setting (OnlineCLScenario). However, it
    can also be used in non-online scenarios
    """

    def __init__(
        self,
        batch_size_mem: int,
        mem_size: int = 200,
        subsample: int = 200,
    ):
        """
        mem_size: int       : Fixed memory size
        subsample: int      : Size of the sample from which to look
                              for highest interfering exemplars
        batch_size_mem: int : Size of the batch sampled from
                              the bigger subsample batch
        """
        super().__init__()
        self.mem_size = mem_size
        self.subsample = subsample
        self.batch_size_mem = batch_size_mem
        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

    def before_backward(self, strategy, **kwargs):
        if self.replay_loader is None:
            return
        samples_x, samples_y, samples_tid = next(self.replay_loader)
        samples_x, samples_y, samples_tid = (
            samples_x.to(strategy.device),
            samples_y.to(strategy.device),
            samples_tid.to(strategy.device),
        )
        # Perform the temporary update with current data
        grad = torch.autograd.grad(
            strategy.loss,
            strategy.model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        model_updated = update_temp(
            strategy.model, grad, strategy.optimizer.param_groups[0]["lr"]
        )
        # Selection of the most interfering samples, no grad required
        # plus we put the model in eval mode so that the additional
        # forward pass don't influence the batch norm statistics
        # strategy.model.eval()
        # model_updated.eval()
        with torch.no_grad():
            _old_red_strategy = strategy._criterion.reduction
            strategy._criterion.reduction = "none"
            old_output = avalanche_forward(strategy.model, samples_x, samples_tid)
            old_loss = strategy._criterion(old_output, samples_y)
            new_output = avalanche_forward(model_updated, samples_x, samples_tid)
            new_loss = strategy._criterion(new_output, samples_y)
            loss_diff = new_loss - old_loss
            chosen_samples_indexes = torch.argsort(loss_diff)[
                len(samples_x) - self.batch_size_mem :
            ]
            strategy._criterion.reduction = _old_red_strategy
        # strategy.model.train()
        # Choose the samples and add their loss to the current loss
        chosen_samples_x, chosen_samples_y, chosen_samples_tid = (
            samples_x[chosen_samples_indexes],
            samples_y[chosen_samples_indexes],
            samples_tid[chosen_samples_indexes],
        )
        replay_output = avalanche_forward(
            strategy.model, chosen_samples_x, chosen_samples_tid
        )
        replay_loss = strategy._criterion(replay_output, chosen_samples_y)
        strategy.loss += replay_loss

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)
        # Exclude classes that were in the last batch
        buffer = concat_datasets(
            [
                self.storage_policy.buffer_groups[key].buffer
                for key, _ in self.storage_policy.buffer_groups.items()
                if int(key) not in torch.unique(strategy.mb_y).cpu()
            ]
        )
        if len(buffer) > self.batch_size_mem:
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.subsample,
                    shuffle=True,
                )
            )
        else:
            self.replay_loader = None


__all__ = ["MIRPlugin"]
