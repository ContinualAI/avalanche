import warnings
from typing import Optional, Sequence

import os
import torch
from torch import nn
import torchvision.models as models

from avalanche.training import default_logger
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training.utils import get_last_fc_layer


class ModelWrapper(nn.Module):
    """
    This class allows us to extract features from a backbone network
    """

    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        self.add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, x):
        self.model(x)
        output_vals = [self.outputs[output_layer_name] for output_layer_name in
                       self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals

    def get_name_to_module(self, model):
        name_to_module = {}
        for m in model.named_modules():
            name_to_module[m[0]] = m[1]
        return name_to_module

    def get_activation(self, all_outputs, name):
        def hook(model, input, output):
            all_outputs[name] = output.detach()

        return hook

    def add_hooks(self, model, outputs, output_layer_names):
        """
        :param model:
        :param outputs: Outputs from layers specified in `output_layer_names`
        will be stored in `output` variable
        :param output_layer_names:
        :return:
        """
        name_to_module = self.get_name_to_module(model)
        for output_layer_name in output_layer_names:
            name_to_module[output_layer_name].register_forward_hook(
                self.get_activation(outputs, output_layer_name))


class StreamingLDA(nn.Module):
    """
    This is an implementation of the Deep Streaming Linear Discriminant
    Analysis algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, test_batch_size=1024,
                 shrinkage_param=1e-4, streaming_update_sigma=True,
                 arch='resnet18', imagenet_pretrained=True,
                 device='cuda',
                 plugins: Optional[Sequence[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """
        Init function for the SLDA model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        :param test_batch_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        :param arch: backbone architecture (default is resnet-18, but others
        can be used by modifying layer for
        feature extraction in `self.feature_extraction_wrapper'
        :param imagenet_pretrained: True if initializing backbone with imagenet
        pre-trained weights else False
        :param imagenet_pretrained: device to use for experiment
        """

        super(StreamingLDA, self).__init__()

        warnings.warn(
            "The Deep SLDA implementation is not perfectly aligned with "
            "the paper implementation (i.e., it does not use a base "
            "initialization phase here and instead starts streming from "
            "pre-trained weights).")

        if plugins is None:
            plugins = []

        # SLDA parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

        # setup feature extraction model pre-trained on imagenet
        feat_extractor = self.get_feature_extraction_model(arch,
                                                           imagenet_pretrained)
        # layer 4.1 is the final layer in resnet18 (need to change this code
        # for other architectures)
        self.feature_extraction_wrapper = ModelWrapper(
            feat_extractor.eval().to(self.device),
            ['layer4.1'], return_single=True).eval()

    def get_feature_extraction_model(self, arch, imagenet_pretrained):
        feature_extraction_model = models.__dict__[arch](
            pretrained=imagenet_pretrained)
        return feature_extraction_model

    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():

            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (
                        self.num_updates + 1)

            # update class means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(
                1)
            self.cK[y] += 1
            self.num_updates += 1

    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                Lambda = torch.pinverse(
                    (
                            1 - self.shrinkage_param) * self.Sigma +
                    self.shrinkage_param * torch.eye(
                        self.input_shape).to(
                        self.device))
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze().long()

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(
            self.device)

    def pool_feat(self, features):
        feat_size = features.shape[-1]
        num_channels = features.shape[1]
        features2 = features.permute(0, 2, 3,
                                     1)  # 1 x feat_size x feat_size x
        # num_channels
        features3 = torch.reshape(features2, (
            features.shape[0], feat_size * feat_size, num_channels))
        feat = features3.mean(1)  # mb x num_channels
        return feat

    def train_model(self, train_loader):

        for train_x, train_y, _ in train_loader:
            batch_x_feat = self.feature_extraction_wrapper(
                train_x.to(self.device))
            batch_x_feat = self.pool_feat(batch_x_feat)

            # train one sample at a time
            for x_pt, y_pt in zip(batch_x_feat, train_y):
                self.fit(x_pt.cpu(), y_pt.view(1, ))

    def evaluate_model(self, test_loader):
        preds = []
        correct = 0

        for it, (test_x, test_y, _) in enumerate(test_loader):
            batch_x_feat = self.feature_extraction_wrapper(
                test_x.to(self.device))
            batch_x_feat = self.pool_feat(batch_x_feat)

            logits = self.predict(batch_x_feat, return_probas=True)

            _, pred_label = torch.max(logits, 1)
            correct += (pred_label == test_y).sum()
            preds += list(pred_label.numpy())

        acc = correct.item() / len(test_loader.dataset)
        return acc, preds

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['muK'] = self.muK.cpu()
        d['cK'] = self.cK.cpu()
        d['Sigma'] = self.Sigma.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_path, save_name):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_path, save_name + '.pth'))
        self.muK = d['muK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.Sigma = d['Sigma'].to(self.device)
        self.num_updates = d['num_updates']
