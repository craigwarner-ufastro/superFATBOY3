# RCS: 
# $Name:  $ $Id: Makefile,v 1.14 2021/02/23 23:23:41 cwarner Exp $
#
# Makefile for CIRCE
# Macros:

SHELL := /bin/tcsh -f

DOMAIN := $(shell domainname)

#reset RCSMASTER & DEV:
RCSDEV := $(shell \pwd)
RCSMASTER := $(shell \ls -l | \grep RCS | \cut -d'>' -f2 | \cut -c2- | sed 's^/RCS*^^')

UFDIRS := fatboyProcesses datatypeExtensions data superFATBOYPlot
JAVA := superFATBOYPlot

ifndef CUDA_HOME
  CUDA_HOME = /usr/local/cuda
  ifeq "$(wildcard $(CUDA_HOME))" ""
    $(warning CUDA_HOME not defined)
  endif
endif

ifndef CUDA_LIB
  CUDA_LIB = $(CUDA_HOME)/lib64
  ifeq "$(wildcard $(CUDA_LIB))" ""
    CUDA_LIB = $(CUDA_HOME)/lib
    ifeq "$(wildcard $(CUDA_LIB))" ""
      $(warning CUDA_LIB not defined)
    endif
  endif
endif 

ifndef PYTHON_INCLUDE
  PYTHON_INCLUDE = /usr/include/python2.7
  ifeq "$(wildcard $(PYTHON_INCLUDE))" ""
    $(warning PYTHON_INCLUDE not defined)
  endif
endif

ifndef NUMPY_INCLUDE
  NUMPY_INCLUDE = $(shell python2 -c 'import numpy; print numpy.get_include()')
  ifeq "$(wildcard $(NUMPY_INCLUDE))" ""
    $(warning NUMPY_INCLUDE not defined)
  endif
endif

ifneq "$(shell echo $PYTHON_INCLUDE | grep 3)" "" 
  PYTHON3_INCLUDE = $(PYTHON_INCLUDE)
  NUMPY3_INCLUDE = $(NUMPY_INCLUDE)
endif

ifndef PYTHON3_INCLUDE
  PYTHON3_INCLUDE = $(shell python3-config --includes | awk '{print $1}' | cut -c 3-)
  ifeq "$(wildcard $(PYTHON3_INCLUDE))" ""
    $(warning PYTHON3_INCLUDE not defined)
  endif
endif

ifndef NUMPY3_INCLUDE
  NUMPY3_INCLUDE = $(shell python3 -c 'import numpy; print(numpy.get_include())')
  ifeq "$(wildcard $(NUMPY3_INCLUDE))" ""
    $(warning NUMPY3_INCLUDE not defined)
  endif
endif

# Targets:
init: initco
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k init; popd;)

initco: 
	if( -e RCS ) co -q RCS/*,v
	$(foreach i, $(UFDIRS), if( ! -e $i ) mkdir $i;)
	$(foreach i, $(UFDIRS), pushd $i; if ( ! -e RCS && -e $(RCSMASTER) ) ln -s $(RCSMASTER)/$i/RCS; popd;)
	$(foreach i, $(UFDIRS), pushd $i; if( -e RCS ) co -q RCS/*,v; popd;)

clean:
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k clean; popd;)

cpu:
	python setup.py build build_ext --inplace
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k install; popd;)

cpu3:
	python3 setup.py build build_ext --inplace
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k install; popd;)

gpu:
	nvcc -O2 -c fatboycudalibmodule.cu --compiler-options '-fPIC --shared' -shared -I$(NUMPY_INCLUDE) -I$(PYTHON_INCLUDE) -o temp.so
	g++ -fPIC -shared -o fatboycudalib.so temp.so -L$(CUDA_LIB) -lcuda -lcudart
	rm temp.so
	nvcc -O2 -c cp_selectmodule.cu --compiler-options '-fPIC --shared' -shared -I$(NUMPY_INCLUDE) -I$(PYTHON_INCLUDE) -o temp.so
	g++ -fPIC -shared -o cp_select.so temp.so -L/$(CUDA_LIB) -lcuda -lcudart
	rm temp.so

gpu3:
	nvcc -O2 -c fatboycudalibmodule.cu --compiler-options '-fPIC --shared' -shared -I$(NUMPY3_INCLUDE) -I$(PYTHON3_INCLUDE) -o temp.so
	g++ -fPIC -shared -o fatboycudalib.so temp.so -L$(CUDA_LIB) -lcuda -lcudart
	rm temp.so
	nvcc -O2 -c cp_selectmodule.cu --compiler-options '-fPIC --shared' -shared -I$(NUMPY3_INCLUDE) -I$(PYTHON3_INCLUDE) -o temp.so
	g++ -fPIC -shared -o cp_select.so temp.so -L/$(CUDA_LIB) -lcuda -lcudart
	rm temp.so

java:
	$(foreach i, $(UFDIRS), pushd $i; $(MAKE) -i -k install; popd;)	

all: cpu gpu
