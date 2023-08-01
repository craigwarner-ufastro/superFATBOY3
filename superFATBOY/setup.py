#!/usr/bin/python -u
# setup.py
# build command : python setup.py build build_ext --inplace
from numpy.distutils.core import setup, Extension
import os, numpy

name = 'fatboyclib'
sources = ['fatboyclibmodule.cpp']

include_dirs = [ numpy.get_include() ]

setup( name = name,
  include_dirs = include_dirs,
  ext_modules = [Extension(name, sources)]
  )
