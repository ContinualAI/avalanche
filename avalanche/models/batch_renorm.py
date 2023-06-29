################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

""" Batch Renorm implementation """

from torch.nn import Module
import torch


class BatchRenorm2D(Module):
    def __init__(
        self,
        num_features,
        gamma=None,
        beta=None,
        running_mean=None,
        running_var=None,
        eps=1e-05,
        momentum=0.01,
        r_d_max_inc_step=0.0001,
        r_max=1.0,
        d_max=0.0,
        max_r_max=3.0,
        max_d_max=5.0,
    ):
        super(BatchRenorm2D, self).__init__()

        self.eps = eps
        self.num_features = num_features
        self.momentum = torch.tensor(momentum, requires_grad=False)

        if gamma is None:
            self.gamma = torch.nn.Parameter(
                torch.ones((1, num_features, 1, 1)), requires_grad=True
            )
        else:
            self.gamma = torch.nn.Parameter(gamma.view(1, -1, 1, 1))
        if beta is None:
            self.beta = torch.nn.Parameter(
                torch.zeros((1, num_features, 1, 1)), requires_grad=True
            )
        else:
            self.beta = torch.nn.Parameter(beta.view(1, -1, 1, 1))

        if running_mean is None:
            self.running_avg_mean = torch.ones(
                (1, num_features, 1, 1), requires_grad=False
            )
            self.running_avg_std = torch.zeros(
                (1, num_features, 1, 1), requires_grad=False
            )
        else:
            self.running_avg_mean = running_mean.view(1, -1, 1, 1)
            self.running_avg_std = torch.sqrt(running_var.view(1, -1, 1, 1))

        self.max_r_max = max_r_max
        self.max_d_max = max_d_max

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.r_max = r_max
        self.d_max = d_max

    def forward(self, x):
        device = self.gamma.device

        self.r_max = (
            self.r_max if isinstance(self.r_max, float) else self.r_max.to(device)
        )
        self.d_max = (
            self.d_max if isinstance(self.d_max, float) else self.d_max.to(device)
        )

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True).to(device)
        batch_ch_std = torch.sqrt(
            torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False) + self.eps
        )
        batch_ch_std = batch_ch_std.to(device)

        self.running_avg_std = self.running_avg_std.to(device)
        self.running_avg_mean = self.running_avg_mean.to(device)
        self.momentum = self.momentum.to(device)

        if self.training:
            r = (
                torch.clamp(
                    batch_ch_std / self.running_avg_std,
                    1.0 / self.r_max,
                    self.r_max,
                )
                .to(device)
                .data.to(device)
            )
            d = (
                torch.clamp(
                    (batch_ch_mean - self.running_avg_mean) / self.running_avg_std,
                    -self.d_max,
                    self.d_max,
                )
                .to(device)
                .data.to(device)
            )

            x = ((x - batch_ch_mean) * r) / batch_ch_std + d
            x = self.gamma * x + self.beta

            if self.r_max < self.max_r_max:
                self.r_max += self.r_max_inc_step * x.shape[0]

            if self.d_max < self.max_d_max:
                self.d_max += self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (
                batch_ch_mean.data.to(device) - self.running_avg_mean
            )
            self.running_avg_std = self.running_avg_std + self.momentum * (
                batch_ch_std.data.to(device) - self.running_avg_std
            )

        else:
            x = (x - self.running_avg_mean) / self.running_avg_std
            x = self.gamma * x + self.beta

        return x
