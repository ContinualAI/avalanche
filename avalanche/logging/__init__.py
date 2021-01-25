from .strategy_logger import *
from .tensorboard_logger import *
from .text_logging import TextLogger
from .interactive_logging import InteractiveLogger

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics

default_logger = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, task=True),
    loss_metrics(minibatch=True, epoch=True, task=True),
    loggers=[InteractiveLogger()])
