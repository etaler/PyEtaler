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
        self.assertIn(3, s)
        self.assertNotIn(9000, s)

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

    def test_creatyion_type(self):
        for create_type in (int, float, bool, np.half):
            t = et.Tensor().from_numpy(np.zeros(2).astype(create_type))
            self.assertEqual(t.dtype(), et.typeToDType[create_type]())

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

    def test_scalar_assignment(self):
        t = et.ones((2, 2))
        try:
            t[:2, :2] = 1
        except et.EtError:
            self.fail("Scalar assignment failed")

    def test_tensor_assignment(self):
        t = et.ones((2, 2))
        try:
            t[:2, :2] = et.zeros((1, 1))
        except et.EtError:
            self.fail("Tensor assignment failed")

    def test_numpy(self):
        t = et.ones((4, 4))
        a = t.numpy()
        self.assertEqual(len(a.shape), 2)

    def test_contruct_numpy_array(self):
        t = et.ones((4, 4))
        a = np.array(t)
        self.assertEqual(len(a.shape), 2)
        self.assertEqual(a.shape[0], 4)
        self.assertEqual(a.shape[1], 4)

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

    def test_integer_division_type(self):
        a = et.ones((1))
        b = et.ones((1))

        # Unlike Python, PyEtaler follows the C/C++ opetor type conversion
        self.assertEqual((a/b).dtype(), et.DType.Int32)

class TestKeywordArguments(unittest.TestCase):
    def test_keyword_arg(self):
        # NOTE: Make sure to pass a tuple/list instead of a integer.
        sp = et.SpatialPooler(input_shape=(2048,), output_shape=(1024,))
        self.assertEqual(sp.connections().shape()[0], 1024)

        sdr1 = et.encoder.gridCell1d(3.1415, seed=90, scale_range=(1, 5.0))
        sdr2 = et.encoder.gridCell1d(3.1415, scale_range=(1, 5.0), seed=90)
        self.assertEqual(sdr1.isSame(sdr2), True)


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


class TestStateDict(unittest.TestCase):
    def test_works_in_py(self):
        s = et.StateDict()
        self.assertEqual(s.empty(), True)
        s['asd'] = 123
        s['qwe'] = 456
        self.assertEqual(s.size(), 2)
        self.assertEqual(len(s), 2)

        # Check searching still works (using C++ iterators)
        self.assertEqual(s.find('zxc'), s.end())
        self.assertNotEqual(s.find('asd'), s.end())

        # Check the python stuff also works
        self.assertIn('asd', s)
        self.assertNotIn('zxc', s)
        for k, _ in s: # for loop in Python should work like it is in C++17 i.e. `for(auto [k, v] : s)`
            self.assertIn(k, ['asd', 'qwe'])

    def test_converting_from_py_dict(self):
        d = {'asd':123, 'qwe':456}
        try:
            s = et.StateDict(d)
        except:
            self.fail("Converting from python dict to et.StateDict failed")

        self.assertEqual(s.size(), 2)
        self.assertNotEqual(s.find('asd'), s.end())
        self.assertEqual(s.find('zxc'), s.end())

    # NOTE: This is a particular behaivoir of C++ map/unordered_map.
    def test_cpp_querk(self):
        s = et.StateDict()
        self.assertEqual(s.size(), 0)
        try:
            _ = s['asd']
        except:
            self.fail("The C++ behaivor of creating entery when reading a non-existance one disappeared")

        self.assertEqual(s.size(), 1)

    def test_failed_lookup(self):
        try:
            s = et.StateDict()
            s.at('zxc')
        except:
            pass
        else:
            self.fail("Should have thrown a C++ exception upon bad at() call")



if __name__ == '__main__':
    unittest.main()

