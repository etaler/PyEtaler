# PyEtaler

This is the offical Python binding for Etaler. (WIP)

## Building from source

```python
pip install cppyy # must installed globally
python3 genbinding.py
```

## Installation
Please copy the files to the proper location for now. We're still getting stuff working.

## Usage

```python
>>> from etaler import et
>>> et.ones([2, 2])
{{ 1, 1}, 
 { 1, 1}}
```
