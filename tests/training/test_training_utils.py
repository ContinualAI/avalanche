import unittest
import torch
from avalanche.training.utils import ParamData


class TestParamData(unittest.TestCase):
    def test_init(self):
        device = "cpu"
        with self.assertRaises(AssertionError):
            # different shapes between input shape
            # and init tensor
            p = ParamData('test', shape=(3, 3),
                          device=device,
                          init_tensor=torch.randn(3, 2))

        with self.assertRaises(AssertionError):
            # missing either shape or init tensor
            p = ParamData('test', device=device)

        p = ParamData('test', device=device,
                      init_tensor=torch.randn(2, 3))
        self.assertEqual(p.data.shape, p.shape)

        p = ParamData('test', device=device,
                      shape=(2, 3))
        self.assertEqual(p.data.shape, p.shape)

        p = ParamData('test', device=device,
                      shape=(2, 3),
                      init_function=torch.ones)
        self.assertEqual(p.data.shape, p.shape)
        self.assertTrue((torch.ones(2, 3) == p.data).all())

    def test_expand(self):
        device = 'cpu'
        p = ParamData('test', device=device, shape=(2, 3),
                      init_function=torch.ones)
        with self.assertRaises(AssertionError):
            p.expand((3, 4))
        with self.assertRaises(AssertionError):
            p.expand((1, 4))
        with self.assertRaises(AssertionError):
            p.expand((1, 4, 5))
        new_p = p.expand((2, 3))
        self.assertTrue(new_p.shape == (2, 3))
        new_p = p.expand((2, 5))
        self.assertTrue((p.data[:2, :3] == 1).all())
        self.assertTrue((p.data[2:, 3:] == 0).all())
        self.assertTrue(new_p.shape == (2, 5))
        self.assertTrue(p.data.shape == (2, 5))
        self.assertTrue(p.shape == (2, 5))
        p = ParamData('test', device=device, shape=(2, 3),
                      init_function=torch.ones)
        new_p = p.expand((5, 3))
        self.assertTrue((p.data[:2, :3] == 1).all())
        self.assertTrue((p.data[2:, 3:] == 0).all())
        self.assertTrue(new_p.shape == (5, 3))
        self.assertTrue(p.data.shape == (5, 3))
        self.assertTrue(p.shape == (5, 3))
        p = ParamData('test', device=device, shape=(2, 3))
        p.expand((2, 5), padding_fn=torch.ones)
        self.assertTrue((p.data[:2, :3] == 0).all())
        self.assertTrue((p.data[2:, 3:] == 1).all())

    def test_reset(self):
        device = 'cpu'
        p = ParamData('test', device=device, shape=(2, 3))
        p.reset_like((3, 4))
        self.assertTrue(p.shape == (3, 4))
        self.assertTrue(p.data.shape == (3, 4))
        p.reset_like((4, 5), init_function=torch.ones)
        self.assertTrue((p.data == 1).all())
        self.assertTrue(p.init_function == torch.zeros)
