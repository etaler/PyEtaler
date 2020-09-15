from etaler import et
from brainblocks.blocks import ScalarEncoder
se_0 = ScalarEncoder(num_s=1024, num_as=128)
se_0.compute(0)
a = et.Tensor.from_numpy(se_0.output.bits)
print(a.tolist())
print(a.to_brainblocks())
