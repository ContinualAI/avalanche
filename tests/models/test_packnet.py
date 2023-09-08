import unittest
from avalanche.models.packnet import PackNetModel, packnet_simple_mlp
from avalanche.training.supervised.strategy_wrappers import PackNet
from torch.optim import SGD
import torch
import os

from tests.unit_tests_utils import get_fast_benchmark
from functools import partial


class TestPackNet(unittest.TestCase):
    _EXPECTATIONS = {
        "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000": 0.75,
        "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp001": 0.45,
        "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp002": 0.33,
        "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp003": 0.0886,
        "Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp004": 0.0,
    }

    def test_PackNetPlugin(self):
        torch.manual_seed(0)
        if "USE_GPU" in os.environ:
            use_gpu = os.environ["USE_GPU"].lower() in ["true"]
        else:
            use_gpu = False
        scenario = get_fast_benchmark(True, seed=0)
        construct_model = partial(
            packnet_simple_mlp,
            num_classes=10,
            drop_rate=0,
            input_size=6,
            hidden_size=20,
        )
        model = construct_model()
        optimizer = SGD(model.parameters(), lr=0.1)
        strategy = PackNet(
            model,
            prune_proportion=0.5,
            post_prune_epochs=1,
            optimizer=optimizer,
            train_epochs=2,
            train_mb_size=10,
            eval_mb_size=10,
            device="cuda" if use_gpu else "cpu",
        )
        x_test = torch.rand(10, 6)
        t_test = torch.ones(10, dtype=torch.long)
        task_ouputs = []

        # Train
        for i, experience in enumerate(scenario.train_stream):
            strategy.train(experience)
            # Store the model output for each task
            task_ouputs.append(model.forward(x_test, t_test * i))

        # Check that the model achieves the expected accuracy
        self.assert_eval(strategy.eval(scenario.test_stream))

        # Verify the model can be saved and loaded from a state dict
        new_model = construct_model()
        missing_keys, unexpected_keys = new_model.load_state_dict(
            strategy.model.state_dict()
        )
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)
        strategy.model = new_model

        # Check that the loaded model achieves the expected accuracy
        self.assert_eval(strategy.eval(scenario.test_stream))

        # Ensure that given the same inputs, the model produces the same outputs
        for i, task_out in enumerate(task_ouputs):
            out = model.forward(x_test, t_test * i)
            self.assertTrue(torch.allclose(out, task_out))

    def test_unsupported_exception(self):
        """Expect an exception when trying to wrap an unsupported module"""

        class UnsupportedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = torch.nn.Parameter(torch.rand(10, 10))

        with self.assertRaises(ValueError):
            PackNetModel(UnsupportedModule())

        with self.assertRaises(ValueError):
            PackNetModel(torch.nn.BatchNorm2d(10))

    def assert_eval(self, last_eval):
        for metric, value in self._EXPECTATIONS.items():
            self.assertAlmostEqual(last_eval[metric], value, places=2)


if __name__ == "__main__":
    unittest.main()
