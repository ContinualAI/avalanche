"""
The :py:mod:`models` module provides a set of (eventually pre-trained) models
that can be used for your continual learning experiments and applications.
These models are mostly `torchvision.models
<https://pytorch.org/vision/0.8/models.html#torchvision-models>`_ and `pytorchcv
<https://pypi.org/project/pytorchcv/>`_ but we plan to add more architectures in
the near future.
"""

from .simple_cnn import *
from .simple_mlp import *
from .mlp_tiny_imagenet import SimpleMLP_TinyImageNet
from .mobilenetv1 import MobilenetV1
from .dynamic_modules import *
from .utils import *
from .slda_resnet import SLDAResNetModel
from .icarl_resnet import *
from .ncm_classifier import NCMClassifier
from .base_model import BaseModel
from .helper_method import as_multitask
from .pnn import PNN
