import unittest
import torch
from torch.nn import Linear

from avalanche.training.plugins.pseudo_annotation import PseudoAnnotationPlugin


class PseudoAnnotationPluginTest(unittest.TestCase):
    def test_no_prev_model_noop(self):
        plugin = PseudoAnnotationPlugin(confidence_thresh=0.99)

        class _Strategy:
            def __init__(self):
                self.device = torch.device("cpu")
                self.adapted_dataset = None
                self.experience = type("E", (), {"classes_in_this_experience": [0]})

        strat = _Strategy()
        # Should not raise
        plugin.after_train_dataset_adaptation(strat)

    def test_after_training_sets_prev_model(self):
        plugin = PseudoAnnotationPlugin(confidence_thresh=0.99)

        class _Strategy:
            def __init__(self):
                self.model = Linear(4, 2)
                self.device = torch.device("cpu")
                self.experience = type("E", (), {"classes_in_this_experience": [0, 1]})

        strat = _Strategy()
        plugin.after_training_exp(strat)
        self.assertIsNotNone(plugin._prev_model)


if __name__ == "__main__":
    unittest.main()


