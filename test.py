# This is a short test to check PyEtaler works.
# Note: This test is only aimed to test the wrapper and the added pythonic functionality.
# A thorough test is in Etaler's C++ repository

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

    def test_stringify(self):
        self.assertEqual('<' in str(et.Shape()), False)

    def test_python_interop(self):
        s = et.Shape([5, 6, 7, 8])
        self.assertEqual(sum(s), 26)

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

        try:
            t.item()
            self.fail("et.Tensor.item() should fail when tensor is not a scalar")
        except:
            pass

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

    def test_tolist(self):
        a = et.ones((9, 3))
        self.assertEqual(len(a.tolist()), 9)
        self.assertEqual(len(a.tolist()[0]), 3)
        self.assertEqual(type(a.tolist()), list)

    def test_sanity(self):
        self.assertEqual(et.zeros((5, 3), et.DType.Bool).numpy().sum(), 0)

    def test_stringify(self):
        # The '<' character is never used in the string conversion
        # But in the defauly Python printer. This makes sure we have overrwriten
        # the default one
        self.assertEqual('<' in str(et.zeros((3, 3))), False)

    def test_view(self):
        a = et.ones((4, 4))
        b = a[:2, :2]
        b[:] -= 1
        self.assertEqual(a.sum().item(), 12)

class TestKeywordArguments(unittest.TestCase):
    def test_keyword_arg(self):
        # NOTE: Make sure to pass a tuple/list instead of a integer.
        sp = et.SpatialPooler(input_shape=(2048,), output_shape=(1024,))
        self.assertEqual(sp.connections().shape()[0], 1024)

class TestEncoder(unittest.TestCase):
    def test_gc1d(self):
        try:
            et.encoder.gridCell1d(42)
        except:
            self.fail("Failed to encode 1D grid cell")
    
    def test_gc2d(self):
        try:
            et.encoder.gridCell2d((42, 0))
        except:
            self.fail("Failed to encode 2D grid cell")

class TestException(unittest.TestCase):
    def test_raise_exception(self):
        try:
            raise et.EtError("TEST")
        except et.EtError:
            pass
        else:
            self.fail("et.EtError thrown in Python but not catched by Python")

    def test_exception_from_cpp(self):
        try:
            et.ones((2, )).item() # This is going to throw
        except et.EtError:
            pass
        else:
            self.fail("Exception thrown from C++ not catched by Python")

class TestSP(unittest.TestCase):
    def test_sp(self):
        sp = et.SpatialPooler((64, ), (64, ))
        sp.setGlobalDensity(0.5)
        self.assertEqual(sp.globalDensity(), 0.5)
    
    def test_sp_density(self):
        sp = et.SpatialPooler((256, ), (256, ))
        sp.setGlobalDensity(0.15)
        density = sp.compute(et.encoder.gridCell1d(1)).sum().item()/256
        diff = abs(density - 0.15)
        if(diff > 0.05):
            self.fail("Spatial Pooler does not respect density settings. Expecting 0.15, get {}".format(density))


if __name__ == '__main__':
    unittest.main()

