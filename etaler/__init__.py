import cppyy
import os

src_dir = os.path.dirname(os.path.realpath(__file__))
cppyy.load_reflection_info(os.path.join(src_dir, "etaler_rflx.so"))


from cppyy.gbl import et
from cppyy.gbl import std

# cppyy does not print reflections like ROOT does
# But Python doesn not allow overwriting an enum's __repr__
# So we supply one to cppyy
#FIXME: cppyy cannot find et::DType in C++ for some reason
#cppyy.cppdef("""
#namespace cling {
#std::string printValue(const et::DType* dtype) {
#    auto d = *dtype;
#    if(d == et::DType::Float)
#        return "Float";
#    else if(d == et::DType::Bool)
#        return "Bool";
#    else if(d == et::DType::Int)
#        return "Int";
#    else if(d == et::DType::Half)
#        return "Half";
#    return "Unknown";
#}
#}
#""")

# helper functions

def type_from_dtype(dtype):
    if dtype == et.DType.Bool:
        return bool
    elif dtype == et.DType.Int:
        return int
    elif dtype == et.DType.Float:
        return float
    elif dtype == et.DType.Half:
        return et.half
    else:
        raise ValueError("DType {} not recognized".format(dtype))

et.dtypeToType = type_from_dtype

def in_bound(idx: int, size: int):
    if idx is None:
        return True
    elif idx >= 0:
        return idx < size
    else:
        return -idx >= size

def is_iteratable(obj):
    try:
        it = iter(obj)
        return True
    except TypeError:
        return False

# Override the default __repr__ with Etaler's function
et.Shape.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
et.Tensor.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
et.half.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)

def is_index_good(self, idx):
    if type(idx) is int:
        if in_bound(idx, self.size()) is False:
            raise IndexError("index {} it our of range".format(idx))
    elif in_bound(idx.stop, self.size()) is False:
        raise IndexError("Stop index {} is out of range".format(idx.stop))
    elif in_bound(idx.start, self.size()) is False:
        raise IndexError("Start index {} is out of range".format(idx.stop))
    elif idx.step == 0:
        raise IndexError("Cannot have step size of 0")
et.Shape.is_index_good = is_index_good

# Override et.Shape's __getitem__ and __setitem__
def get_subshape(self: et.Shape, idx):
    self.is_index_good(idx)

    if type(idx) is int:
        if idx < 0:
            return self.data()[len(self) + idx]
        else:
            return self.data()[idx]
    elif type(idx) is not slice and type(idx) is not range:
        raise TypeError("Cannot index with type {}".format(type(idx)))
   
    # Unfortunatelly iterators doesn't work like in C++ :(
    start = idx.start if idx.start is not None else 0
    stop = idx.stop if idx.stop is not None else self.size()
    step = idx.step if idx.step is not None else 1
    res = et.Shape()
    for i in range(start, stop, step):
        res.push_back(self[i])
    return res

def set_subshape(self: et.Shape, idx, value):
    self.is_index_good(idx)
    idx = idx if type(idx) is not int else range(idx, idx+1, 1)

    rng = range(
            idx.start if idx.start is not None else 0,
            idx.stop if idx.stop is not None else self.size(),
            idx.step if idx.step is not None else 1) # incase idx is slice
    values = (value,) if is_iteratable(value) is False else value

    if len(rng) != len(values):
        raise ValueError('Cannot assign {} values into {} variables'.format(len(values), len(rng)))

    for i, j in enumerate(rng):
        self.data()[j] = values[i] if values[i] is not None else -1 # -1 is None in Etaler

et.Shape.__getitem__ = get_subshape
et.Shape.__setitem__ = set_subshape

# Override the __setitem__ and __getitem__ function of et.Tensor
# To allow Python style subscription
def get_tensor_view(self: et.Tensor, slices) -> et.Tensor:

    tup = None
    if type(slices) is int or type(slices) is range or type(slices) is slice:
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
            # No need to check to out-of-bounds access. The C++ side does that
        else:
            raise TypeError("indices must be a int, range or slice")

    return self.view(rgs)

et.Tensor.__getitem__ = get_tensor_view
et.Tensor.__setitem__ = lambda self, slices, value: self.__getitem__(slices).assign(value)

# Override the default C++ item<T>() with a Python one
cpp_get_tensor_item = et.Tensor.item
def get_tensor_item(self: et.Tensor):
    return cpp_get_tensor_item[et.dtypeToType(self.dtype())](self)
et.Tensor.item = get_tensor_item

# Override the default C++ toHost<T> with a Python one
# TODO: Should the function return a list/np.array instead of a std.vector?
cpp_tensor_to_host = et.Tensor.toHost
def tensor_to_host(self: et.Tensor):
    return cpp_tensor_to_host[et.dtypeToType(self.dtype())](self)
et.Tensor.toHost = tensor_to_host

# Handle the misleading __bool__ function generated by cppyy
def tensor_trueness(self: et.Tensor) -> bool:
    if self.size() == 0:
        raise ValueError("The true-ness of a empty tensor is not defined.")
    elif self.size() > 1:
        raise  ValueError("The true-ness of a non-scalar is ambiguous.")
    return self.item() != 0 if self.dtype() == et.DType.Half else bool(self.item()) 

et.Tensor.__bool__ = tensor_trueness

# Implement out __len__ to match numpy's behaivour
et.Tensor.__len__ = lambda self: self.shape()[0] if self.has_value else 0
