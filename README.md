<p align="center">
  <img src="docs/images/superFATBOY.png" alt="superFATBOY mascot" width="160">
</p>

# superFATBOY3
Python 3 version of superFATBOY GPU accelerated data pipeline for IR and optical astronomical data

## Documentation
Full documentation is in [`docs/`](docs/README.md):

- [Quick start](docs/quickstart.md): install, run, and the template XML files for each instrument
- [Instruments and templates](docs/instruments.md): which instruments have been validated
- [XML style guide](docs/xml-guide.md)
- [Process guide](docs/processes/README.md) and [options reference](docs/options-reference.md)
- [MIRADAS guide](docs/miradas.md)
- [API guide](docs/api.md): scripting superFATBOY and writing your own processes and datatypes

Validated end-to-end in both GPU and CPU mode: FLAMINGOS-1 (imaging and MOS), OSIRIS and KAST (longslit), and MIRADAS (SOL, SOS, MOS).
Templates for these are in [`superFATBOY/data/templates/`](superFATBOY/data/templates/).

## Requirements
- numpy
- scipy
- astropy
- matplotlib (optional)
- sep or sextractor (optional)
- CUDA and CuPy (optional)
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
conda create --name sFB3 python=3.9 cupy numpy scipy astropy matplotlib
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
