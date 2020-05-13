# This is a short test to check PyEtaler works.
# Note: This test is only aimed to test the wrapper and the added pythonic functionality.
# A thorough test in Etaler's C++ repository

from etaler import et
import numpy as np
import unittest

class TestShape(unittest.TestCase):
    def test_creation(self):
        s = et.Shape([60, 60])
        self.assertEqual(s.volume(), 3600)
        self.assertEqual(s.size(), 2)
        self.assertEqual(len(s), 2) # cppyy should have wrapped this

    def test_pythonic_ops(self):
        s = et.Shape((1, 2, 3, 4, 5))
        self.assertEqual(s[3], 4)
        self.assertEqual(s[:2], et.Shape([1, 2]))

        s[:2] = [7, 7]
        self.assertEqual(s, et.Shape([7, 7, 3, 4, 5]))

    def test_to_list(self):
        s = et.Shape((4, 3, 2, 1))
        l = s.to_list()
        self.assertEqual(type(l), list)
        self.assertEqual(len(l), s.size())
        for i in range(4):
            self.assertEqual(l[i], s[i])

class TestTensor(unittest.TestCase):
    def test_creation(self):
        a = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        t = et.Tensor.from_numpy(a)
        self.assertEqual(t.shape(), et.Shape([3, 2]))

    def test_pythonic_ops(self):
        a = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2)
        t = et.Tensor.from_numpy(a)
        self.assertEqual(t[0:2].shape(), et.Shape([2, 2]))
        self.assertEqual(t[0::2].shape(), et.Shape([2, 2]))
        self.assertEqual(t[0, 0].item(), 1)
        self.assertEqual(len(t.toHost()), 6)

    def test_numpy(self):
        t = et.ones((4, 4))
        a = t.numpy()
        self.assertEqual(len(a.shape), 2)
        

    def test_assignment(self):
        a = et.ones((4, 4))
        b = et.zeros((4,))
        a[0] = b
        self.assertEqual(a.sum().item(), 12)
        self.assertEqual(a[:, 0].realize().sum().item(), 3)

    def test_reshape(self):
        a = et.ones((4, 4))
        a = a.reshape((16, ))
        self.assertEqual(a.shape(), et.Shape([16, ]))

    def test_tohost(self):
        a = et.ones((3, 3))
        v = a.toHost()
        self.assertEqual(v.size(), a.size())

class TestKeywordArguments(unittest.TestCase):
    def test_keyword_arg(self):
        # NOTE: Make sure to pass a tuple/list instead of a integer.
        sp = et.SpatialPooler(input_shape=(2048,), output_shape=(1024,))
        self.assertEqual(sp.connections().shape()[0], 1024)

class TestEncoder(unittest.TestCase):
    def test_gc1d(self):
        try:
            sdr = et.encoder.gridCell1d(42)
        except:
            self.fail("Failed to encode 1D grid cell")
    
    def test_gc2d(self):
        try:
            sdr = et.encoder.gridCell2d((42, 0))
        except:
            self.fail("Failed to encode 2D grid cell")


if __name__ == '__main__':
    unittest.main()
