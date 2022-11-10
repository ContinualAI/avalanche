from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.models.dynamic_modules import MultiTaskModule, DynamicModule
import torch.nn as nn
from collections import OrderedDict

from avalanche.benchmarks.scenarios import CLExperience


def avalanche_forward(model, x, task_labels):
    if isinstance(model, MultiTaskModule):
        return model(x, task_labels)
    else:  # no task labels
        return model(x)


def avalanche_model_adaptation(model: nn.Module, experience: CLExperience):
    for module in model.modules():
        if isinstance(module, DynamicModule):
            module.adaptation(experience)


class FeatureExtractorBackbone(nn.Module):
    """
    This PyTorch module allows us to extract features from a backbone network
    given a layer name.
    """

    def __init__(self, model, output_layer_name):
        super(FeatureExtractorBackbone, self).__init__()
        self.model = model
        self.output_layer_name = output_layer_name
        self.output = None  # this will store the layer output
        self.add_hooks(self.model)

    def forward(self, x):
        self.model(x)
        return self.output

    def get_name_to_module(self, model):
        name_to_module = {}
        for m in model.named_modules():
            name_to_module[m[0]] = m[1]
        return name_to_module

    def get_activation(self):
        def hook(model, input, output):
            self.output = output.detach()

        return hook

    def add_hooks(self, model):
        """
        :param model:
        :param outputs: Outputs from layers specified in `output_layer_names`
        will be stored in `output` variable
        :param output_layer_names:
        :return:
        """
        name_to_module = self.get_name_to_module(model)
        name_to_module[self.output_layer_name].register_forward_hook(
            self.get_activation()
        )


class Flatten(nn.Module):
    """
    Simple nn.Module to flatten each tensor of a batch of tensors.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    """
    Simple nn.Module to create a multi-layer perceptron
    with BatchNorm and ReLU activations.

    :param hidden_size: An array indicating the number of neurons in each layer.
    :type hidden_size: int[]
    :param last_activation: Indicates whether to add BatchNorm and ReLU
                            after the last layer.
    :type last_activation: Boolean
    """

    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or (
                (i == len(hidden_size) - 2) and (last_activation)
            ):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


__all__ = ["avalanche_forward", "FeatureExtractorBackbone", "MLP", "Flatten"]
