# SuperFATBOY3 Pipeline Guide

## Introduction
SuperFATBOY3 is a GPU-accelerated astronomical data pipeline framework for IR and optical imaging and spectroscopic data reduction, specifically designed as a Python-based successor to IRAF pipelines (SLIM - Sans Lousy IRAF Mistakes).

## Installation
The pipeline is designed to run on Linux.

### Requirements
- Python 2.7 (Note: superFATBOY v1.1 and MIRADAS DRP are based on Python 2).
- NumPy
- SciPy
- PyFits (or Astropy)
- SEP (preferred) or Sextractor
- CUDA, PyCUDA, PyFFT (for GPU acceleration)

### Building and Installation
1.  **Unpackage:** `tar xvfz superFATBOY-MIRADAS_DRP.tar.gz`
2.  **GPU Library Build (Optional):**
    From the `superFATBOY/superFATBOY` directory:
    - Set environment variables if needed (`CUDA_HOME`, `PYTHON_INCLUDE`, `NUMPY_INCLUDE`).
    - Run `make gpu`.
3.  **System-wide Install (Recommended):**
    From the top-level directory: `sudo python setup.py install`

## How to Run
SuperFATBOY3 can be run via a GUI or the command line.

### Command Line
For spectroscopy:
```bash
python f2spec.py param_file.dat
```
For imaging:
```bash
python flam2pipeline.py param_file.dat
```
For MIRADAS data using XML configuration:
```bash
python superFatboy.py config_file.xml
```

## XML Style Guide
The XML configuration file structure:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<fatboy>
  <queries>
    <!-- Dataset definition -->
  </queries>

  <processes>
    <!-- Process definitions in execution order -->
  </processes>

  <parameters>
    <!-- Global pipeline parameters -->
  </parameters>
</fatboy>
```
Refer to the detailed "XML Guide" in the original documentation for specifics on `<dataset>`, `<object>`, `<calib>`, and `<option>` tags.
