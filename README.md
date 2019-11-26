# PyEtaler

This is the offical Python binding for Etaler. PyEtaler generates Python binging using [cppyy](https://cppyy.readthedocs.io/en/latest/) and adds additional feature on top of the automatically generated bindings.

## Building from source

**Note:** You must have Etaler and cppyy installed globally before building the binding.
```python
pip install cppyy # must installed globally
python3 genbinding.py
```

## Installation
Please copy the files to the proper location for now. We're still working on a PyPI package.

## Usage

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
