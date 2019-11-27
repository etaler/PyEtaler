import cppyy
import os

src_dir = os.path.dirname(os.path.realpath(__file__))
cppyy.load_reflection_info(os.path.join(src_dir, "etaler_rflx.so"))


from cppyy.gbl import et
from cppyy.gbl import std

# Override the default __repr__ with Etaler's function
et.Shape.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
et.Tensor.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)

# Override the __setitem__ and __getitem__ function of et.Tensor
# To allow Python style subscription
def get_tensor_view(self, slices):

    tup = None
    if type(slices) is int:
        tup = (slices,)
    elif type(slices) is range or type(slices) is slice:
        tup = (slices,)
    else:
        tup = slices

    rgs = et.svector[et.Range](len(tup))
    shape = self.shape()
    for i, r in enumerate(tup):
        if type(r) is int:
            rgs[i] = et.Range(r)
        elif type(r) is range or type(r) is slice:
            if r.step is not None and r.step != 1:
                raise NotImplementedError("Etaler does not support non 1 steps")
            start = 0 if r.start is None else r.start
            stop = shape[i] if r.stop is None else r.stop
            rgs[i] = et.Range(start, stop, start < 0, stop < 0)

    return self.view(rgs)


et.Tensor.__getitem__ = get_tensor_view
et.Tensor.__setitem__ = lambda self, slices, value: self.__getitem__(slices).assign(value)
