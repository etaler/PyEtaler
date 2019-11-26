import cppyy
import os

src_dir = os.path.dirname(os.path.realpath(__file__))
cppyy.load_reflection_info(os.path.join(src_dir, "etaler_rflx.so"))


from cppyy.gbl import et

# Override the default __repr__ with Etaler's function
et.Shape.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
et.Tensor.__repr__ = lambda self: cppyy.gbl.cling.printValue(self)
