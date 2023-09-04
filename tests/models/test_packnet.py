import unittest
from avalanche.models.packnet import PackNetPlugin, PackNetSimpleMLP
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitMNIST
import pdb
import torch
import os


class TestPackNet(unittest.TestCase):
    _EXPECTATIONS = {
        "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000": 0.9981105,
        "Top1_Acc_Exp/eval_phase/test_stream/Task001/Exp001": 0.9911458,
        "Top1_Acc_Exp/eval_phase/test_stream/Task002/Exp002": 0.9796929,
        "Top1_Acc_Exp/eval_phase/test_stream/Task003/Exp003": 0.9912998,
        "Top1_Acc_Exp/eval_phase/test_stream/Task004/Exp004": 0.9763819,
    }
    """Because PackNet is a parameter isolation method using task identities, we
    expect accuracies in the high 90s for an easy dataset like SplitMNIST.
    """

    def test_PackNetPlugin(self):
        torch.manual_seed(0)
        if "USE_GPU" in os.environ:
            use_gpu = os.environ["USE_GPU"].lower() in ["true"]
        else:
            use_gpu = False

        scenario = SplitMNIST(n_experiences=5, return_task_id=True)
        # Disabled dropout ensures the logits are the same given the same input
        model = PackNetSimpleMLP(drop_rate=0)
        optimizer = SGD(model.parameters(), lr=0.1)
        plugin = PackNetPlugin(1)
        strategy = Naive(
            model,
            optimizer,
            plugins=[plugin],
            train_epochs=2,
            train_mb_size=128,
            eval_mb_size=128,
            device="cuda" if use_gpu else "cpu",
        )
        # Train
        for experience in scenario.train_stream:
            strategy.train(experience)

        # Check that the model achieves the expected accuracy
        self.assert_eval(strategy.eval(scenario.test_stream))

        # Verify the model can be saved and loaded from a state dict
        new_model = PackNetSimpleMLP(drop_rate=0)
        missing_keys, unexpected_keys = new_model.load_state_dict(model.state_dict())
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)
        strategy.model = new_model

        # Check that the loaded model achieves the expected accuracy
        self.assert_eval(strategy.eval(scenario.test_stream))

    def assert_eval(self, last_eval):
        for metric, value in self._EXPECTATIONS.items():
            self.assertAlmostEqual(last_eval[metric], value, places=3)


if __name__ == "__main__":
    unittest.main()
