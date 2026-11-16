from typing import Callable, Dict, Optional, Union
import warnings

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.ewc import EWCPlugin
from avalanche.training.utils import zerolike_params_dict, ParamData


class IEWCPlugin(EWCPlugin):
    """
    Improved Elastic Weight Consolidation (IEWC) plugin.

    IEWC keeps the EWC penalty and changes how the importance is computed.
    EWC's empirical Fisher sums squared per-sample gradients, so each old
    sample is weighted by the scale of its loss gradient and well-fitted
    samples barely contribute. IEWC divides each per-sample term by the
    squared norm of the loss gradient with respect to the model output,
    plus a small task-relative damping::

        F = 1/N sum_n h_n * g_n**2 / (s_n + tau),   s_n = ||dl_n / df_n||**2
        tau = kappa * median{s_i : s_i > 0}

    where `g_n` is the per-sample parameter gradient and `h_n` are optional
    sample weights with mean one. The importance is invariant to a global
    rescaling of the loss and works with any differentiable criterion.

    Importances are computed with two passes on the training set of the
    experience (the first gives `s_n` and `tau`, the second the per-sample
    parameter gradients), so this is slower than :class:`EWCPlugin`. The
    inputs of the experience are kept in memory between the two passes.
    The penalty is the same as in :class:`EWCPlugin`,
    ``ewc_lambda * sum_i F_i (theta_i - theta*_i)**2``, so `ewc_lambda`
    corresponds to lambda / 2 in the paper. This plugin does not use task
    identities.

    Technique introduced in:
    "Improved Elastic Weight Consolidation as an Optimization Constraint for
    Continual Learning" by D. Wiest (2026).
    https://arxiv.org/abs/2611.16039
    """

    def __init__(
        self,
        ewc_lambda,
        kappa: float = 1e-2,
        loss_scale_fn: Optional[Callable[[Tensor], Tensor]] = None,
        sample_weighting: Union[str, Callable[[Tensor, Tensor, Tensor], float]] = (
            "uniform"
        ),
        rdw_max_basis: int = 64,
        mode="separate",
        decay_factor=None,
        keep_importance_data=False,
    ):
        """
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param kappa: relative damping. The damping added to each denominator
               is `kappa` times the median of the positive `s_n` of the
               experience.
        :param loss_scale_fn: optional function mapping the output gradient
               `r_n` of a single sample (shape of one model output) to the
               scalar `s_n = r_n^T G_n^{-1} r_n`. Defaults to the squared
               Euclidean norm (`G_n = I`).
        :param sample_weighting: `uniform` gives every sample weight one.
               `rdw` (residual-diversity weighting) weights each sample by the
               squared residual of its unit gradient direction outside the
               span of up to `rdw_max_basis` earlier residual directions,
               normalized to mean one. RDW depends on the dataset order.
               A callable `fn(x, y, grad)` is called once per sample with its
               input, target and flattened parameter gradient and returns a
               non-negative weight; the weights are normalized to mean one.
        :param rdw_max_basis: maximum number of directions kept by `rdw`.
               The basis is kept on the CPU.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param decay_factor: used only if mode is `online`.
               It specifies the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """
        super().__init__(
            ewc_lambda,
            mode=mode,
            decay_factor=decay_factor,
            keep_importance_data=keep_importance_data,
        )
        assert kappa >= 0, "kappa must be non-negative."
        assert callable(sample_weighting) or sample_weighting in (
            "uniform",
            "rdw",
        ), "sample_weighting must be uniform, rdw or a callable."
        assert rdw_max_basis >= 0, "rdw_max_basis must be non-negative."

        self.kappa = kappa
        self.loss_scale_fn = loss_scale_fn
        self.sample_weighting = sample_weighting
        self.rdw_max_basis = rdw_max_basis

        # damping and sample weights of the last importance computation
        self.damping: Optional[float] = None
        self.sample_weights: Optional[Tensor] = None

    def _loss_scales(self, output_grads: Tensor) -> Tensor:
        """Compute `s_n` from a batch of per-sample output gradients."""
        if self.loss_scale_fn is None:
            return output_grads.pow(2).flatten(1).sum(dim=1)
        return torch.stack([self.loss_scale_fn(r).reshape(()) for r in output_grads])

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size, num_workers=0
    ) -> Dict[str, ParamData]:
        """
        Compute IEWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if torch.device(device).type == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        importances = zerolike_params_dict(model)
        params = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        # first pass: s_n for every sample. Batches are cached so that the
        # second pass sees the same inputs even with random transforms.
        batches = []
        scales = []
        for batch in dataloader:
            x, y, task_labels = batch[0], batch[1], batch[-1]
            batches.append((x, y, task_labels))
            with torch.no_grad():
                out = avalanche_forward(model, x.to(device), task_labels)
            out.requires_grad_(True)
            y = y.to(device)
            loss = sum(criterion(out[i : i + 1], y[i : i + 1]) for i in range(len(y)))
            output_grads = torch.autograd.grad(loss, out)[0]
            scales.append(self._loss_scales(output_grads.detach()))
        scales = torch.cat(scales)
        positive = scales[scales > 0]
        damping = (
            self.kappa * torch.quantile(positive.double(), 0.5).item()
            if positive.numel() > 0
            else 0.0
        )

        # second pass: per-sample parameter gradients
        basis = None
        weights = []
        n = 0
        for x, y, task_labels in batches:
            x, y = x.to(device), y.to(device)
            for i in range(len(y)):
                denom = scales[n].item() + damping
                n += 1
                out = avalanche_forward(model, x[i : i + 1], task_labels[i : i + 1])
                loss = criterion(out, y[i : i + 1])
                grads = torch.autograd.grad(
                    loss, [p for _, p in params], allow_unused=True
                )
                if callable(self.sample_weighting):
                    flat = torch.cat(
                        [
                            (g if g is not None else torch.zeros_like(p))
                            .detach()
                            .flatten()
                            for (_, p), g in zip(params, grads)
                        ]
                    )
                    h = float(self.sample_weighting(x[i : i + 1], y[i : i + 1], flat))
                elif self.sample_weighting == "rdw":
                    h, basis = self._rdw_weight(params, grads, basis)
                else:
                    h = 1.0
                weights.append(h)
                if denom == 0 or h == 0:
                    continue
                for (k, _), g in zip(params, grads):
                    if g is not None:
                        importances[k].data += (h / denom) * g.detach().pow(2)

        # sample weights have mean one
        weights = torch.tensor(weights)
        total = weights.sum().item()
        for _, imp in importances.items():
            imp.data /= total if total > 0 else 1.0
        self.damping = damping
        self.sample_weights = weights / weights.mean() if total > 0 else weights

        model.train()

        return importances

    def _rdw_weight(self, params, grads, basis):
        """Residual of the unit gradient direction outside the span of the
        stored basis. The residual direction is added to the basis while it
        has fewer than `rdw_max_basis` rows."""
        u = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).detach().flatten()
                for (_, p), g in zip(params, grads)
            ]
        ).cpu()
        norm = u.norm()
        if norm == 0:
            return 0.0, basis
        u = u / norm
        if basis is not None:
            u = u - basis.t() @ (basis @ u)
        h = u.pow(2).sum().item()
        rows = 0 if basis is None else basis.shape[0]
        if h > 1e-12 and rows < self.rdw_max_basis:
            u = (u / h**0.5).unsqueeze(0)
            basis = u if basis is None else torch.cat([basis, u])
        return h, basis


__all__ = ["IEWCPlugin"]
