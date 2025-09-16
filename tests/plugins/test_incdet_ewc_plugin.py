import unittest
import os
import importlib.util
import sys
import types
import torch
from torch.nn import Linear


def _inject_pkg_stubs():
    for name in ["avalanche", "avalanche.training", "avalanche.training.plugins"]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            setattr(mod, "__path__", [])
            sys.modules[name] = mod
    # stub strategy_plugin with minimal SupervisedPlugin base
    sp_name = "avalanche.training.plugins.strategy_plugin"
    if sp_name not in sys.modules:
        sp_mod = types.ModuleType(sp_name)
        class _SupervisedPlugin:
            def __init__(self):
                pass
        sp_mod.SupervisedPlugin = _SupervisedPlugin
        sys.modules[sp_name] = sp_mod
    # stub models.utils to avoid pulling full models
    mu_name = "avalanche.models.utils"
    if mu_name not in sys.modules:
        mu_mod = types.ModuleType(mu_name)
        def _forward(m, x, t=None):
            return m(x)
        mu_mod.avalanche_forward = _forward
        sys.modules[mu_name] = mu_mod
    # stub training.utils to avoid importing benchmarks
    tu_name = "avalanche.training.utils"
    if tu_name not in sys.modules:
        tu_mod = types.ModuleType(tu_name)
        class ParamData:
            def __init__(self, name, shape=None, init_tensor=None, device="cpu"):
                self.name = name
                self.shape = shape
                self.device = device
                self._data = init_tensor if init_tensor is not None else torch.zeros(())
            @property
            def data(self):
                return self._data
            def expand(self, new_shape, padding_fn=torch.zeros):
                if self._data.numel() == 0:
                    self._data = padding_fn(new_shape)
                return self._data
        def copy_params_dict(model, copy_grad=False):
            out = {}
            for k, p in model.named_parameters():
                tensor = p.grad.data.clone() if copy_grad and p.grad is not None else p.data.clone()
                out[k] = ParamData(k, p.shape, init_tensor=tensor, device=p.device)
            return out
        def zerolike_params_dict(model):
            out = {}
            for k, p in model.named_parameters():
                out[k] = ParamData(k, p.shape, init_tensor=torch.zeros_like(p.data), device=p.device)
            return out
        tu_mod.ParamData = ParamData
        tu_mod.copy_params_dict = copy_params_dict
        tu_mod.zerolike_params_dict = zerolike_params_dict
        sys.modules[tu_name] = tu_mod


def _preload_ewc_module():
    here = os.path.dirname(__file__)
    ewc_path = os.path.abspath(
        os.path.join(here, "..", "..", "avalanche", "training", "plugins", "ewc.py")
    )
    spec = importlib.util.spec_from_file_location("avalanche.training.plugins.ewc", ewc_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules["avalanche.training.plugins.ewc"] = module


def _inject_ex_model_stub():
    mod = types.ModuleType("avalanche.benchmarks.classic.ex_model")
    class _Dummy:
        pass
    mod.LeNet5 = _Dummy
    mod.SlimResNet18 = _Dummy
    sys.modules["avalanche.benchmarks.classic.ex_model"] = mod


def _load_huber_plugin():
    _inject_pkg_stubs()
    _inject_ex_model_stub()
    _preload_ewc_module()
    here = os.path.dirname(__file__)
    plugin_path = os.path.abspath(
        os.path.join(here, "..", "..", "avalanche", "training", "plugins", "incdet_ewc.py")
    )
    spec = importlib.util.spec_from_file_location("incdet_ewc_local", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.HuberEWCPlugin


class HuberEWCPluginTest(unittest.TestCase):
    def test_penalty_nonzero_after_prev_exp(self):
        torch.manual_seed(0)
        model = Linear(5, 3)
        HuberEWCPlugin = _load_huber_plugin()
        plugin = HuberEWCPlugin(ewc_lambda=1.0, beta=0.5)

        class _Clock:
            def __init__(self):
                self.train_exp_counter = 0

        class _Strategy:
            def __init__(self, model: torch.nn.Module):
                self.model = model
                self.clock = _Clock()
                self.device = torch.device("cpu")
                self.loss = torch.tensor(0.0)

        strategy = _Strategy(model)

        plugin.saved_params[0] = {k: torch.zeros_like(p.data) for k, p in model.named_parameters()}
        plugin.importances[0] = {k: torch.ones_like(p.data) for k, p in model.named_parameters()}

        strategy.clock.train_exp_counter = 1

        plugin.before_backward(strategy)

        self.assertGreater(float(strategy.loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
