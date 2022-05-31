import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models.base_model import BaseModel


class ExpertGatePlugin(SupervisedPlugin):
    """Expert Gate Plugin.
    """
