import warnings
from typing import Optional, Sequence

import os
import torch
from torch import nn
import torchvision.models as models

from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from avalanche.training import default_logger
from avalanche.models.dynamic_modules import MultiTaskModule


class ModelWrapper(nn.Module):
    """
    This PyTorch module allows us to extract features from a backbone network
    given a layer name.
    """

    def __init__(self, model, output_layer_name):
        super(ModelWrapper, self).__init__()
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
            self.get_activation())


class SLDAResNetModel(nn.Module):
    """
    This is a model wrapper to reproduce experiments from the original
    paper of Deep Streaming Linear Discriminant Analysis by using
    a pretrained ResNet model.
    """

    def __init__(self, arch='resnet18', output_layer_name='layer4.1',
                 imagenet_pretrained=True, device='cpu'):
        """
        :param arch: backbone architecture (default is resnet-18, but others
        can be used by modifying layer for
        feature extraction in `self.feature_extraction_wrapper'
        :param imagenet_pretrained: True if initializing backbone with imagenet
        pre-trained weights else False
        :param output_layer_name: name of the layer from feature extractor
        :param device: cpu, gpu or other device
        """

        super(SLDAResNetModel, self).__init__()

        feat_extractor = models.__dict__[arch](
            pretrained=imagenet_pretrained).to(device).eval()
        self.feature_extraction_wrapper = ModelWrapper(
            feat_extractor, output_layer_name).eval()

        warnings.warn(
            "The Deep SLDA implementation is not perfectly aligned with "
            "the paper implementation (i.e., it does not use a base "
            "initialization phase here and instead starts streming from "
            "pre-trained weights).")

    @staticmethod
    def pool_feat(features):
        feat_size = features.shape[-1]
        num_channels = features.shape[1]
        features2 = features.permute(0, 2, 3,
                                     1)  # 1 x feat_size x feat_size x
        # num_channels
        features3 = torch.reshape(features2, (
            features.shape[0], feat_size * feat_size, num_channels))
        feat = features3.mean(1)  # mb x num_channels
        return feat

    def forward(self, x):
        """
        :param x: raw x data
        """
        feat = self.feature_extraction_wrapper(x)
        feat = SLDAResNetModel.pool_feat(feat)
        return feat


class StreamingLDA(BaseStrategy):
    """
    Deep Streaming Linear Discriminant Analysis.
    This strategy does not use backpropagation.
    Minibatches are first passed to the pretrained feature extractor.
    The result is processed one element at a time to fit the
    LDA.
    """
    def __init__(self, slda_model, criterion,
                 input_size, num_classes, output_layer_name=None,
                 shrinkage_param=1e-4, streaming_update_sigma=True,
                 train_epochs: int = 1, eval_mb_size: int = 1, device='cpu',
                 plugins: Optional[Sequence['StrategyPlugin']] = None,
                 evaluator=default_logger, eval_every=-1):
        """
        Init function for the SLDA model.
        :param slda_model: a PyTorch model
        :param criterion: loss function
        :param output_layer_name: if not None, wrap model to retrieve
            only the `output_layer_name` output. If None, the strategy
            assumes that the model already produces a valid output.
            You can use `ModelWrapper` class to create your custom
            SLDA-compatible model.
        :param input_size: feature dimension
        :param num_classes: number of total classes in stream
        :param eval_mb_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        feature extraction in `self.feature_extraction_wrapper'
        :param plugins: list of StrategyPlugins
        :param evaluator: Evaluation Plugin instance
        :param eval_every: run eval every `eval_every` epochs.
            See `BaseStrategy` for details.
        """

        if plugins is None:
            plugins = []

        if output_layer_name is not None:
            slda_model = ModelWrapper(slda_model.to(device),
                                      output_layer_name).eval()

        super(StreamingLDA, self).__init__(
            slda_model, None, criterion, eval_mb_size, train_epochs,
            eval_mb_size, device=device, plugins=plugins, evaluator=evaluator,
            eval_every=eval_every)

        # SLDA parameters
        self.input_size = input_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        # setup weights for SLDA
        self.muK = torch.zeros((num_classes, input_size)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Sigma = torch.ones((input_size, input_size)).to(self.device)
        self.num_updates = 0
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.prev_num_updates = -1

    def forward(self, return_features=False):
        if isinstance(self.model, MultiTaskModule):
            feat = self.model.forward(self.mb_x, self.mb_task_id)
        else:  # no task labels
            feat = self.model.forward(self.mb_x)
        out = self.predict(feat)
        if return_features:
            return out, feat
        else:
            return out

    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        for self.mb_it, self.mbatch in \
                enumerate(self.dataloader):
            self.before_training_iteration(**kwargs)

            self.loss = 0
            for self.mb_x, self.mb_y, self.mb_task_id in self.mbatch.values():
                self.mb_x = self.mb_x.to(self.device)
                self.mb_y = self.mb_y.to(self.device)

                # Forward
                self.before_forward(**kwargs)

                # compute output on entire minibatch
                self.logits, feats = self.forward(return_features=True)

                # process one element at a time
                for f, y in zip(feats, self.mb_y):
                    self.fit(f.unsqueeze(0), y.unsqueeze(0))

                self.after_forward(**kwargs)

                # Loss
                self.loss += self.criterion(self.logits, self.mb_y)

            self.after_training_iteration(**kwargs)

    def make_optimizer(self):
        pass

    @torch.no_grad()
    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """

        # covariance updates
        if self.streaming_update_sigma:
            x_minus_mu = (x - self.muK[y])
            mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates / (self.num_updates + 1)
            self.Sigma = (self.num_updates * self.Sigma + delta) / (
                    self.num_updates + 1)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def predict(self, X):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """

        # compute/load Lambda matrix
        if self.prev_num_updates != self.num_updates:
            # there have been updates to the model, compute Lambda
            self.Lambda = torch.pinverse(
                (
                        1 - self.shrinkage_param) * self.Sigma +
                self.shrinkage_param * torch.eye(
                    self.input_size, device=self.device))
            self.prev_num_updates = self.num_updates

        # parameters for predictions
        M = self.muK.transpose(1, 0)
        W = torch.matmul(self.Lambda, M)
        c = 0.5 * torch.sum(M * W, dim=0)

        scores = torch.matmul(X, W) - c

        # return predictions or probabilities
        return scores

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print('\nFitting Base...')

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


__all__ = [
    'ModelWrapper',
    'StreamingLDA',
    'SLDAResNetModel',
]