from typing import Union, Sequence, Callable

import torch
from torch.nn import (
    Module,
    Sequential,
    BatchNorm2d,
    Conv2d,
    ReLU,
    ConstantPad3d,
    Identity,
    AdaptiveAvgPool2d,
    Linear,
)
from torch import Tensor
from torch.nn.init import zeros_, kaiming_normal_
from torch.nn.modules.flatten import Flatten
import torch.nn.functional as F


class IdentityShortcut(Module):
    def __init__(self, transform_function: Callable[[Tensor], Tensor]):
        super(IdentityShortcut, self).__init__()
        self.transform_function = transform_function

    def forward(self, x: Tensor) -> Tensor:
        return self.transform_function(x)


def conv3x3(in_planes: int, out_planes: int, stride: Union[int, Sequence[int]] = 1):
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def batch_norm(num_channels: int) -> BatchNorm2d:
    return BatchNorm2d(num_channels)


class ResidualBlock(Module):
    def __init__(
        self,
        input_num_filters: int,
        increase_dim: bool = False,
        projection: bool = False,
        last: bool = False,
    ):
        super().__init__()
        self.last: bool = last

        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        self.direct = Sequential(
            conv3x3(input_num_filters, out_num_filters, stride=first_stride),
            batch_norm(out_num_filters),
            ReLU(True),
            conv3x3(out_num_filters, out_num_filters, stride=(1, 1)),
            batch_norm(out_num_filters),
        )

        self.shortcut: Module

        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                self.shortcut = Sequential(
                    Conv2d(
                        input_num_filters,
                        out_num_filters,
                        kernel_size=(1, 1),
                        stride=(2, 2),
                        bias=False,
                    ),
                    batch_norm(out_num_filters),
                )
            else:
                # identity shortcut, as option A in paper
                self.shortcut = Sequential(
                    IdentityShortcut(lambda x: x[:, :, ::2, ::2]),
                    ConstantPad3d(
                        (
                            0,
                            0,
                            0,
                            0,
                            out_num_filters // 4,
                            out_num_filters // 4,
                        ),
                        0.0,
                    ),
                )
        else:
            self.shortcut = Identity()

    def forward(self, x):
        if self.last:
            return self.direct(x) + self.shortcut(x)
        else:
            return torch.relu(self.direct(x) + self.shortcut(x))


class IcarlNet(Module):
    def __init__(self, num_classes: int, n=5, c=3):
        super().__init__()

        self.is_train = True
        input_dims = c
        output_dims = 16

        first_conv = Sequential(
            conv3x3(input_dims, output_dims, stride=(1, 1)),
            batch_norm(16),
            ReLU(True),
        )

        input_dims = output_dims
        output_dims = 16

        # first stack of residual blocks, output is 16 x 32 x 32
        layers_list = []
        for _ in range(n):
            layers_list.append(ResidualBlock(input_dims))
        first_block = Sequential(*layers_list)

        input_dims = output_dims
        output_dims = 32

        # second stack of residual blocks, output is 32 x 16 x 16
        layers_list = [ResidualBlock(input_dims, increase_dim=True)]
        for _ in range(1, n):
            layers_list.append(ResidualBlock(output_dims))
        second_block = Sequential(*layers_list)

        input_dims = output_dims
        output_dims = 64

        # third stack of residual blocks, output is 64 x 8 x 8
        layers_list = [ResidualBlock(input_dims, increase_dim=True)]
        for _ in range(1, n - 1):
            layers_list.append(ResidualBlock(output_dims))
        layers_list.append(ResidualBlock(output_dims, last=True))
        third_block = Sequential(*layers_list)
        final_pool = AdaptiveAvgPool2d(output_size=(1, 1))

        self.feature_extractor = Sequential(
            first_conv,
            first_block,
            second_block,
            third_block,
            final_pool,
            Flatten(),
        )

        input_dims = output_dims
        output_dims = num_classes

        self.classifier = Linear(input_dims, output_dims)

    def forward(self, x):
        x = self.feature_extractor(x)  # Already flattened
        x = self.classifier(x)
        return x


def make_icarl_net(num_classes: int, n=5, c=3) -> IcarlNet:
    """Create :py:class:`IcarlNet` network, the ResNet used in
    ICarl.
    :param num_classes: number of classes, network output size
    :param n: depth of each residual blocks stack
    :param c: number of input channels
    """
    return IcarlNet(num_classes, n=n, c=c)


def initialize_icarl_net(m: Module):
    """Initialize the input network based on `kaiming_normal`
    with `mode=fan_in` for `Conv2d` and `Linear` blocks.
    Biases are initialized to zero.
    :param m: input network (should be IcarlNet).
    """
    if isinstance(m, Conv2d):
        kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            zeros_(m.bias.data)

    elif isinstance(m, Linear):
        kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="sigmoid")
        if m.bias is not None:
            zeros_(m.bias.data)


__all__ = ["initialize_icarl_net", "make_icarl_net", "IcarlNet"]
