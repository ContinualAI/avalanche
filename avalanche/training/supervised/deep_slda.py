import warnings
from typing import Callable, Optional, Sequence, Union

import os
import torch

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import (
    EvaluationPlugin,
    default_evaluator,
)
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models import FeatureExtractorBackbone


class StreamingLDA(SupervisedTemplate):
    """Deep Streaming Linear Discriminant Analysis.

    This strategy does not use backpropagation.
    Minibatches are first passed to the pretrained feature extractor.
    The result is processed one element at a time to fit the LDA.
    Original paper:
    "Hayes et. al., Lifelong Machine Learning with Deep Streaming Linear
    Discriminant Analysis, CVPR Workshop, 2020"
    https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf
    """

    def __init__(
        self,
        slda_model,
        criterion,
        input_size,
        num_classes,
        output_layer_name=None,
        shrinkage_param=1e-4,
        streaming_update_sigma=True,
        train_epochs: int = 1,
        train_mb_size: int = 1,
        eval_mb_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
    ):
        """Init function for the SLDA model.

        :param slda_model: a PyTorch model
        :param criterion: loss function
        :param output_layer_name: if not None, wrap model to retrieve
            only the `output_layer_name` output. If None, the strategy
            assumes that the model already produces a valid output.
            You can use `FeatureExtractorBackbone` class to create your custom
            SLDA-compatible model.
        :param input_size: feature dimension
        :param num_classes: number of total classes in stream
        :param train_mb_size: batch size for feature extractor during
            training. Fit will be called on a single pattern at a time.
        :param eval_mb_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
            feature extraction in `self.feature_extraction_wrapper`.
        :param plugins: list of StrategyPlugins
        :param evaluator: Evaluation Plugin instance
        :param eval_every: run eval every `eval_every` epochs.
            See `BaseTemplate` for details.
        """

        if plugins is None:
            plugins = []

        slda_model = slda_model.eval()
        if output_layer_name is not None:
            slda_model = FeatureExtractorBackbone(
                slda_model.to(device), output_layer_name
            ).eval()

        super(StreamingLDA, self).__init__(
            slda_model,
            None,  # type: ignore
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

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
        """Compute the model's output given the current mini-batch."""
        self.model.eval()
        if isinstance(self.model, MultiTaskModule):
            feat = self.model(self.mb_x, self.mb_task_id)
        else:  # no task labels
            feat = self.model(self.mb_x)
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
        for _, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            # compute output on entire minibatch
            self.mb_output, feats = self.forward(return_features=True)
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            # Optimization step
            self._before_update(**kwargs)
            # process one element at a time
            for f, y in zip(feats, self.mb_y):
                self.fit(f.unsqueeze(0), y.unsqueeze(0))
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def make_optimizer(self, **kwargs):
        """Empty function.
        Deep SLDA does not need a Pytorch optimizer."""
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
            x_minus_mu = x - self.muK[y]
            mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates / (self.num_updates + 1)
            self.Sigma = (self.num_updates * self.Sigma + delta) / (
                self.num_updates + 1
            )

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
                (1 - self.shrinkage_param) * self.Sigma
                + self.shrinkage_param * torch.eye(self.input_size, device=self.device)
            )
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
        print("\nFitting Base...")

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        print("\nEstimating initial covariance matrix...")
        from sklearn.covariance import OAS

        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d["muK"] = self.muK.cpu()
        d["cK"] = self.cK.cpu()
        d["Sigma"] = self.Sigma.cpu()
        d["num_updates"] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + ".pth"))

    def load_model(self, save_path, save_name):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_path, save_name + ".pth"))
        self.muK = d["muK"].to(self.device)
        self.cK = d["cK"].to(self.device)
        self.Sigma = d["Sigma"].to(self.device)
        self.num_updates = d["num_updates"]

    def _check_plugin_compatibility(self):
        """Check that the list of plugins is compatible with the template.

        This means checking that each plugin impements a subset of the
        supported callbacks.
        """
        # TODO: ideally we would like to check the argument's type to check
        #  that it's a supertype of the template.
        # I don't know if it's possible to do it in Python.
        ps = self.plugins

        def get_plugins_from_object(obj):
            def is_callback(x):
                return x.startswith("before") or x.startswith("after")

            return filter(is_callback, dir(obj))

        cb_supported = set(get_plugins_from_object(self.PLUGIN_CLASS))
        cb_supported.remove("before_backward")
        cb_supported.remove("after_backward")
        for p in ps:
            cb_p = set(get_plugins_from_object(p))

            if not cb_p.issubset(cb_supported):
                warnings.warn(
                    f"Plugin {p} implements incompatible callbacks for template"
                    f" {self}. This may result in errors. Incompatible "
                    f"callbacks: {cb_p - cb_supported}",
                )
                return


__all__ = ["StreamingLDA"]
