import torch
from avalanche.models.dynamic_modules import (
    MultiTaskModule,
    MultiHeadClassifier,
)


class BiCAdapter(torch.nn.Module):
    def __init__(self, model) -> None:
        super(BiCAdapter, self).__init__()

        self.model = model
        self.bias_layers = []

    def add_bias_layer(self, device, cls):
        self.bias_layers.append(BiasLayer(device, cls, False))
    
    def forward(self, x):
        out = self.model(x)

        for layer in self.bias_layers:
            out = layer(out)
        
        return out

    def forward_logits(self, x):
        return self.model(x)


class BiCAdapterMH(MultiTaskModule):
    def __init__(self, model) -> None:
        super(BiCAdapterMH, self).__init__()

        self.model = model
        self.bias_layers = []

        self.model.fc = MultiHeadClassifier(64)

    def add_bias_layer(self, device, cls):
        self.bias_layers.append(BiasLayer(device, cls, True))
    
    def forward_single_task(self, x, task_label):
        out = self.model.forward_rep(x)
        out = self.model.fc(out, task_label)
        
        if isinstance(task_label, int):
            # fast path. mini-batch is single task.
            return self.bias_layers[task_label](out)
        else:
            unique_tasks = torch.unique(task_label)

            out = torch.zeros_like(out)

            for task in unique_tasks:
                task_mask = task_label == task
                x_task = out[task_mask]
                out[task_mask] = self.bias_layers[task](x_task)
            return out

    def forward_logits(self, x):
        return self.model(x)


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self, device, cls, task_incremental=False):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
        self.beta = torch.nn.Parameter(torch.zeros(1, device=device))

        self.cls = torch.Tensor(list(cls)).long().to(device)
        self.device = device
        self.task_incremental = task_incremental

    def forward(self, x):
        if self.task_incremental:
            a = self.alpha
            b = self.beta
        else:
            a = torch.ones(x.size(1), device=self.device)
            a[self.cls] = self.alpha

            b = torch.ones(x.size(1), device=self.device)
            b[self.cls] = self.beta

        # return self.alpha * x + self.beta
        return a * x + b
