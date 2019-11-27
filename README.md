# PyEtaler

This is the offical Python binding for Etaler. PyEtaler generates Python binging using [cppyy](https://cppyy.readthedocs.io/en/latest/) and adds additional feature on top of the automatically generated bindings.


**Note:** As of now, installing cppyy (thus PyEtaler) will cause [ROOT](https://root.cern.ch) to fail to load due to dependency clash.

## Building from source

## Installation

**Note:** You must have Etaler and cppyy installed globally before building the binding.
**Note:** Since the binding is generated to load the actual Etaler installation. You'll need to re-compile the binding everytime Etaler is updated.

If you are building from source (building via directlt interacting with the generator).

```shell
pip install cppyy # must installed globally
python3 genbinding.py
cp *.so etaler/
cp *.pcm etaler/
# Then copy the resulting files into your package directory
```

Locally build via PIP

```shell
pip3 install .
```

Alternativelly you can install it directly via PyPI.

```shell
pip install pyetaler
```

## Usage

After installation, you can use Etaler from python. The API is exactly like it is in C++.

```python
>>> from etaler import et
>>> et.ones([2, 2])
{{ 1, 1}, 
 { 1, 1}}
>>> sp = et.SpatialPooler([128], [32])
>>> x = et.encoder.scalar(0.1, 0, 1, 128, 12)
>>> sp.compute(x)
{ 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
```

### Hacking PyEtaler

In case that you need to use C++ STL - maybe because the wrapper is doing something stupid. You can access the STL using `etaler.std`.

For example

```Python
>>> from etaler import std
>>> std.vector[int](10)
<cppyy.gbl.std.vector<int> object at 0x1ef112c0>
```
