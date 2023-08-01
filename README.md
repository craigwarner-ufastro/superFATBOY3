# superFATBOY3
Python 3 version of superFATBOY GPU accelerated data pipeline for IR and optical astronomical data

## Requirements
- numpy
- scipy
- astropy
- matplotlib (optional)
- sep or sextractor (optional)
- CUDA, PyCUDA, and CuPy (optional)
- deepCR (optional)

## Installation
- To install optional GPU libraries, first use Makefile to build CUDA code, first make sure that you set environment variables `CUDA_HOME` and `PYTHON_INCLUDE`
to point to your install directory for CUDA and your python3 include dir, respectively (e.g. `/usr/local/cuda` and `/usr/include/python3.8` are typical values).
Then
```
cd superFATBOY3/superFATBOY
make gpu3
```

- To install as sudo:
```
cd superFATBOY3
sudo python3 setup.py install
```
