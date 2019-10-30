import cppyy
cppyy.load_reflection_info("etaler_rflx.so")

from cppyy.gbl import et

# Override the default __repr__ with Etaler's function
et.Shape.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
et.Tensor.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
