# This is a short test to check PyEtaler works.

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
        

if __name__ == '__main__':
    unittest.main()
