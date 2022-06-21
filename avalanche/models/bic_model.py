import torch


class BiCAdapter(torch.nn.Module):
    def __init__(self, model) -> None:
        super(BiCAdapter, self).__init__()

        self.model = model
        self.bias_layers = []

    def add_bias_layer(self, device, cls):
        self.bias_layers.append(BiasLayer(device, cls))
    
    def forward(self, x):
        out = self.model(x)

        for layer in self.bias_layers:
            out = layer(out)
        
        return out

    def forward_logits(self, x):
        return self.model(x)


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self, device, cls):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
        self.beta = torch.nn.Parameter(torch.zeros(1, device=device))

        self.cls = list(cls)
        self.device = device

    def forward(self, x):
        a = torch.ones(x.size(1), device=self.device)
        a[self.cls] = self.alpha

        b = torch.ones(x.size(1), device=self.device)
        b[self.cls] = self.beta

        # return self.alpha * x + self.beta
        return a * x + b
