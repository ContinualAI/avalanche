"""
The :py:mod:`models` module provides a set of (eventually pre-trained) models
that can be used for your continual learning experiments and applications.
These models are mostly `torchvision.models
<https://pytorch.org/vision/0.8/models.html#torchvision-models>`_ and `pytorchcv
<https://pypi.org/project/pytorchcv/>`_ but we plan to add more architectures in
 the near future.
"""

from .simple_cnn import SimpleCNN
from .simple_mlp import SimpleMLP
from .mlp_tiny_imagenet import SimpleMLP_TinyImageNet
from .mobilenetv1 import MobilenetV1
