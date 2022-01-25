from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models.dynamic_modules import MultiTaskModule, DynamicModule
import torch.nn as nn


def avalanche_forward(model, x, task_labels):
    if isinstance(model, MultiTaskModule):
        return model(x, task_labels)
    else:  # no task labels
        return model(x)


def avalanche_model_adaptation(model: nn.Module, dataset: AvalancheDataset):
    for module in model.modules():
        if isinstance(module, DynamicModule):
            module.adaptation(dataset)


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


__all__ = ["avalanche_forward", "FeatureExtractorBackbone"]
