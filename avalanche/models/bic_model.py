from typing import Iterable, SupportsInt
import torch


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters

    Bias layers used in Bias Correction (BiC) plugin.
    "Wu, Yue, et al. "Large scale incremental learning." Proceedings
    of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition. 2019"
    """

    def __init__(self, clss: Iterable[SupportsInt]):
        """
        :param clss: list of classes of the current layer. This are use
            to identify the columns which are multiplied by the Bias
            correction Layer.
        """
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

        unique_classes = list(sorted(set(int(x) for x in clss)))

        self.register_buffer("clss", torch.tensor(unique_classes, dtype=torch.long))

    def forward(self, x):
        alpha = torch.ones_like(x)
        beta = torch.zeros_like(x)

        alpha[:, self.clss] = self.alpha
        beta[:, self.clss] = self.beta

        return alpha * x + beta


__all__ = ["BiasLayer"]
