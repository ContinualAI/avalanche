import torch


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self, device, clss):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
        self.beta = torch.nn.Parameter(torch.zeros(1, device=device))

        self.clss = torch.Tensor(list(clss)).long().to(device)
        self.not_clss = None

    def forward(self, x):
        alpha = torch.ones_like(x)
        beta = torch.ones_like(x)

        alpha[:, self.clss] = self.alpha
        beta[:, self.clss] = self.beta

        return alpha * x + beta
