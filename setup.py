#!/usr/bin/python -u
# setup.py
# build command : python setup.py build build_ext --inplace
#from numpy.distutils.core import setup, Extension
from setuptools import setup, find_packages, Extension
import os, numpy

name = 'superFATBOY'
version = '2.3.0'
cmod_name = 'superFATBOY/fatboyclib'
sources = ['superFATBOY/fatboyclibmodule.cpp']

include_dirs = [ numpy.get_include() ]

setup( name = name,
  version=version,
  description='superFATBOY DRP',
  author='Craig Warner',
  include_dirs = include_dirs,
  include_package_data=True,
  packages=find_packages(),
  package_data={'': ['*.so', 'data/linelists/*.dat', 'data/config/*.*']},
  install_requires=['numpy>=1.0', 'scipy>=0.5'],
  zip_safe = False,
  ext_modules = [Extension(cmod_name, sources)],
  scripts=['superFATBOY/superFatboy3.py', 'superFATBOY/fitsUpdate.py', 'superFATBOY/getHead.py']
  )
