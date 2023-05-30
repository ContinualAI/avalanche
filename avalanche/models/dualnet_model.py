# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from itertools import chain
from copy import deepcopy
import torch.nn.functional as F
from copy import deepcopy


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        sizes = [int(x) for x in sizes]
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class block(nn.Module):
    def __init__(self, n_in, n_out):
        super(block, self).__init__()
        self.net = nn.Sequential(*[nn.Linear(n_in, n_out), nn.ReLU()])
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(CustomResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))
        self.linear = NCM_classifier(nf*8*block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

    def update_means(self, x, y, alpha=0):
        self.linear.update_means(x, y, alpha=alpha)

    def predict(self, x, t):
        out = self.linear(x, t)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        bsz = x.size(0)
        if x.dim() < 4:
            x = x.view(bsz, 3, 32, 32)
        else:
            assert x.size(-1) == 84
        out = self.conv1(x)
        out = relu(self.bn1(out))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)
        if return_feat:
            return feat
        y = self.linear(feat) 
        return y


def cResNet18(num_classes, nf=20):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)


def ResNet18(num_classes, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)


def ResNet32(num_classes, nf=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, nf)


def Flatten(x):
    return x.view(x.size(0), -1)


class noReLUBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(noReLUBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class MaskNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(MaskNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))

        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.predictor = nn.Sequential(*layers)

        self.f_conv1 = self._make_conv2d_layer(3, nf, max_pool=True, padding=1)
        self.f_conv2 = self._make_conv2d_layer(
            nf*1, nf*2, padding=1, max_pool=True)
        self.f_conv3 = self._make_conv2d_layer(
            nf*2, nf*4, padding=1, max_pool=True)
        self.f_conv4 = self._make_conv2d_layer(
            nf*4, nf*8, padding=1, max_pool=True)
        self.relu = nn.ReLU()

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps, max_pool=False, padding=1):
        layers = [nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1,
                            padding=padding),
                  nn.BatchNorm2d(out_maps), nn.ReLU()]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, 
                                       ceil_mode=True))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def slow_learner(self):
        param = chain(self.conv1.parameters(), self.layer1.parameters(), 
                      self.layer2.parameters(),
                      self.layer3.parameters(), self.layer4.parameters(), 
                      self.projector.parameters())
        for p in param:
            yield p

    def fast_learner(self):
        param = chain(self.f_conv1.parameters(), self.f_conv3.parameters(),
                      self.f_conv4.parameters(), self.linear.parameters())
        for p in param:
            yield p

    def train_slow(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = False

    def train_all(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = True

    def train_fast(self):
        for p in self.slow_learner():
            p.requires_grad = False
        for p in self.fast_learner():
            p.requires_grad = True

    def forward(self, x, return_feat=False):
        # bsz = x.size(0)
        # if x.dim() < 4:
        #     x = x.view(bsz, 3, 32, 32)
        # else:
        #     assert x.size(-1) == 84
        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        h0 = self.maxpool(h0)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        if return_feat:
            feat = self.avgpool(h4)
            return feat.view(feat.size(0), -1)

        m1_ = self.f_conv1(x)
        m1 = m1_ * h1
        m2_ = self.f_conv2(m1)
        m2 = m2_ * h2
        m3_ = self.f_conv3(m2)
        m3 = m3_ * h3
        m4_ = self.f_conv4(m3)
        m4 = m4_ * h4
        out = self.avgpool(m4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    def BarlowTwins(self, y1, y2):

        z1 = self.projector(self(y1, True))
        z2 = self.projector(self(y2, True))
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        c_diff = (c_ - torch.eye(D).to(c_.device)).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()   
        return loss

    def SimCLR(self, y1, y2, temp=100, eps=1e-6):
        z1 = self.projector(self(y1, True))
        z2 = self.projector(self(y2, True))
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)

        out = torch.cat([z_a, z_b], dim=0)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / temp)
        neg = sim.sum(dim=1)

        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1/temp)).cuda()
        neg = torch.clamp(neg - row_sub, min=eps)
        pos = torch.exp(torch.sum(z_a * z_b, dim=-1) / temp)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        return loss

    def SimSiam(self, y1, y2):
        def D(p, z):
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

        z1, z2 = self.projector(self(y1, True)), self.projector(self(y2, True))
        p1, p2 = self.predictor(z1), self.predictor(z2)

        loss = (D(p1, z2).mean() + D(p2, z1).mean()) * 0.5
        return loss

    def BYOL(self, y1, y2):
        def D(p, z):
            p = F.normalize(p, dim=-1, p=2)
            z = F.normalize(z, dim=-1, p=2)
            return 2 - 2 * (p*z).sum(dim=-1)

        z1, z2 = self.projector(self(y1, True)), self.projector(self(y2, True))
        p1, p2 = self.predictor(z1), self.predictor(z2)

        loss = (D(z1, p2.detach()).mean() + D(z2, p1.detach()).mean()) * 0.5
        return loss


def MaskNet18(num_classes, nf=20):
    return MaskNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)


class SlowNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(SlowNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nf * 8 * block.expansion, int(num_classes))

        sizes = [nf*8] + [256, nf*8]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        self.projector2 = deepcopy(self.projector)

        self.relu = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def slow_learner(self):
        param = chain(self.conv1.parameters(), self.layer1.parameters(),
                      self.layer2.parameters(),
                      self.layer3.parameters(), self.layer4.parameters(),
                      self.projector.parameters())
        for p in param:
            yield p

    def train_slow(self):
        for p in self.slow_learner():
            p.requires_grad = True
        for p in self.fast_learner():
            p.requires_grad = False

    def forward(self, x, return_feat=False):
        bsz = x.size(0)
        if x.dim() < 4:
            x = x.view(bsz, 3, 32, 32)
        else:
            assert x.size(-1) == 84
        h0 = self.conv1(x)
        h0 = relu(self.bn1(h0))
        h0 = self.maxpool(h0)
        h1 = self.layer1(h0)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        feat = self.avgpool(h4)
        feat = feat.view(feat.size(0), -1)
        if return_feat:
            return feat
        y = self.linear(feat)
        return y

    def BarlowTwins(self, y1, y2):

        z1 = self.projector(self(y1, True))
        z2 = self.projector(self(y2, True))
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        c_diff = (c_ - torch.eye(D).cuda()).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()   
        return loss

    def SimCLR(self, y1, y2, temp=100, eps=1e-6):
        z1 = self.projector(self(y1, True))
        z2 = self.projector(self(y2, True))
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)

        out = torch.cat([z_a, z_b], dim=0)
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / temp)
        neg = sim.sum(dim=1)

        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1/temp)).cuda()
        neg = torch.clamp(neg - row_sub, min=eps)
        pos = torch.exp(torch.sum(z_a * z_b, dim=-1) / temp)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        return loss


def SlowNet18(num_classes, nf=20):
    return SlowNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)


def MaskNet18(num_classes, nf=20):
    return MaskNet(BasicBlock, [2, 2, 2, 2], num_classes, nf)
