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
- To install optional GPU libraries, first use Makefile to build CUDA code, first make sure that you set environment variable `CUDA_HOME` to point to your install directory for CUDA (e.g. `/usr/local/cuda`).  Optionally you may also set`PYTHON3_INCLUDE` to e.g. `/usr/include/python3.8` but this *should* be auto-detected and not necessary to set manually unless it complains that it can't find Python.h.
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

## HPC Environments
- To install and run superFATBOY3 on HiperGator or other HPC environments, try this guide.
- YMMV on versions of GCC, CUDA, and Python depending on your particular HPC environment but the below works on HiperGator Nov 2023.  Note that you must choose a compatible GCC and CUDA (sometimes the newest GCC is too new for the newest CUDA).
- #### To BUILD:
```
module load conda gcc/9.3.0 cuda/11.4.3
conda create --name sFB3 python=3.9 cupy numpy scipy astropy matplotlib pycuda
conda activate sFB3
cd superFATBOY3/superFATBOY/
make gpu3
cd ..
python setup.py install
```
- #### To RUN subsequently:
```
module load conda gcc/9.3.0 cuda/11.4.3
conda activate sFB3
superFatboy3.py [-list] [filename.xml]
```
