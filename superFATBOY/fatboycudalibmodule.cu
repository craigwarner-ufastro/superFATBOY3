#define _NPY_NO_DEPRECATIONS //for NPY_CHAR error 
#include <math.h>
#include <stdio.h>

#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include <thrust/sort.h>

#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/system_error.h>
#include <thrust/system/system_error.h>


extern "C" {
  static PyObject * gpumedian(PyObject *self, PyObject *args, PyObject *keywds);
  static PyObject * gpumedianS(PyObject *self, PyObject *args, PyObject *keywds);
  static PyObject * gpumedian2d(PyObject *self, PyObject *args, PyObject *keywds);

  void gpusort_float(float *data, int n) {
    thrust::device_vector<float> d_x(data, data+n);
    thrust::sort(d_x.begin(), d_x.end());
    thrust::copy(d_x.begin(), d_x.end(), data);
  } 

  void gpusort_int(int *data, int n) {
    thrust::device_vector<int> d_x(data, data+n);
    thrust::sort(d_x.begin(), d_x.end());
    thrust::copy(d_x.begin(), d_x.end(), data);
  }

  void gpusort_long(long *data, int n) {
    thrust::device_vector<long> d_x(data, data+n);
    thrust::sort(d_x.begin(), d_x.end());
    thrust::copy(d_x.begin(), d_x.end(), data);
  }

  void gpusort_double(double *data, int n) {
    thrust::device_vector<double> d_x(data, data+n);
    thrust::sort(d_x.begin(), d_x.end());
    thrust::copy(d_x.begin(), d_x.end(), data);
  }

  static PyMethodDef module_methods[] = {
    {"gpumedian", (PyCFunction)gpumedian, METH_VARARGS | METH_KEYWORDS, "Calculate median using thrust gpu radix sort"},
    {"gpumedianS", (PyCFunction)gpumedianS, METH_VARARGS | METH_KEYWORDS, "Calculate medians using thrust gpu radix sort"},
    {"gpumedian2d", (PyCFunction)gpumedian2d, METH_VARARGS | METH_KEYWORDS, "Calculate median using thrust gpu radix sort"},
    {NULL, NULL, 0, NULL} /* Sentinel */
  };

  #ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
    #define PyMODINIT_FUNC void
  #endif

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "fatboycudalib",     /* m_name */
      "Module with CUDA methods for sFB",  /* m_doc */
      -1,                  /* m_size */
      module_methods,     /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
#endif

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC initfatboycudalib(void) {
        PyObject* m;
        m = Py_InitModule3("fatboycudalib", module_methods, "Example module that creates an extension type.");
        if (m == NULL){ return; }
        import_array(); /* required NumPy initialization */
    }
#else
    PyMODINIT_FUNC PyInit_fatboycudalib(void) {
        PyObject* m;
        m = PyModule_Create(&moduledef);
        if (m == NULL){ return m; }
        import_array(); /* required NumPy initialization */
        return m;
    }
#endif

  static PyObject * gpumedian2d(PyObject *self, PyObject *args, PyObject *keywds) {
    PyArrayObject *array;
    double med=INT_MIN;
    double lthresh=INT_MIN, hthresh=INT_MAX;
    int nlow=0, nhigh=0;
    bool nonzero=false, even=false;
    int n, k;
    static char *kwlist[] = {"array", "lthreshold", "hthreshold", "nlow", "nhigh", "nonzero", "even", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|ddiibb", kwlist, &PyArray_Type, &array, &lthresh, &hthresh, &nlow, &nhigh, &nonzero, &even)) return NULL;

    n = array->dimensions[0];
    int ny = array->dimensions[1];
    k = n/2;
    if (array->descr->type_num == PyArray_FLOAT) {
      float* data = (float*)PyArray_DATA(array);
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < n; j++) {
        gpusort_float(&data[j*ny], ny);
        if (n%2 == 1 || !even) {
          med = (double)(data[k]);
        } else {
          med = (double)(data[k]+data[k-1])/2;
        }
      }
    }
    return PyFloat_FromDouble(med);
  }

  static PyObject * gpumedianS(PyObject *self, PyObject *args, PyObject *keywds) {
    PyArrayObject *array, *smaskarray, *medianarray;
    int nslits, rows, slitMaxIdx;
    bool nonzero=false, even=false;
    int n, k;
    static char *kwlist[] = {"array", "slitmask", "medians", "nslits", "nonzero", "even", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!i|bb", kwlist, &PyArray_Type, &array, &PyArray_Type, &smaskarray, &PyArray_Type, &medianarray, &nslits, &nonzero, &even)) return NULL;

    rows = array->dimensions[0];
    n = array->dimensions[0];
    for (int j = 1; j < array->nd; j++) n *= array->dimensions[j];
    k = n/2;
    if (PyArray_ISBYTESWAPPED(array)) PyArray_Byteswap(array, true);
    int* slitmask = (int*)PyArray_DATA(smaskarray);
    float *medians = (float*)PyArray_DATA(medianarray);

    if (array->descr->type_num == PyArray_FLOAT) {
      float* data = (float*)PyArray_DATA(array);
      float *temp = data;
      int lastIdx = 0;
      int i = 0;
      bool foundLast = false;
      int s1 = 0, s2 = 0;
      for (int s = 0; s < nslits; s++) {
	slitMaxIdx = lastIdx; //reset slitMaxIdx
        data = new float[n];
        i = 0;
        s1 = s+1;
        s2 = s+2;
	foundLast = false;
        if (nonzero) {
          //only nonzero
          for (int j = lastIdx; j < n; j++) {
            if (temp[j] == 0) continue;
	    if (slitmask[j] == s1) {
	      //this slitlet
	      data[i++] = temp[j]; 
	      slitMaxIdx = j; 
	    } else if (!foundLast && slitmask[j] == s2) {
	      //first index of next slitlet
	      lastIdx = j;
	      foundLast = true;
	    } else if (slitmask[j] > s2 && j-slitMaxIdx > 2*rows && i > 0) {
	      //two slitlets above; break
	      //also check that there has been at least 2*rows
	      //datapoints since finding one in this slit
	      //also check at least one nonzero datapoint was found so that
	      //slitMaxIdx has been updated
	      break;
	    }
          }
        } else { 
          for (int j = lastIdx; j < n; j++) {
            if (slitmask[j] == s1) {
              //this slitlet
              data[i++] = temp[j];
              slitMaxIdx = j;
            } else if (!foundLast && slitmask[j] == s2) {
              //first index of next slitlet
              lastIdx = j;
              foundLast = true;
            } else if (slitmask[j] > s2 && j-slitMaxIdx > 2*rows && i > 0) {
              //two slitlets above; break
              //also check that there has been at least 2*rows
              //datapoints since finding one in this slit
              //two slitlets above; break
              //also check at least one nonzero datapoint was found so that
              //slitMaxIdx has been updated
              break;
            }
	  }
        }
	k = i/2;
	if (n == 0) {
	  medians[s] = 0.0f;
	} else {
	  gpusort_float(data, i);
	  if (n%2 == 1 || !even) {
	    medians[s] = data[k];
	  } else {
	    medians[s] = (data[k]+data[k-1])/2;
	  }
	}
      }
      free(data);
    }
    return PyFloat_FromDouble(1.0);
  }

  static PyObject * gpumedian(PyObject *self, PyObject *args, PyObject *keywds) {
    PyArrayObject *array;
    double med=INT_MIN;
    double lthresh=INT_MIN, hthresh=INT_MAX;
    int nlow=0, nhigh=0;
    bool nonzero=false, even=false;
    int n, k;
    static char *kwlist[] = {"array", "lthreshold", "hthreshold", "nlow", "nhigh", "nonzero", "even", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|ddiibb", kwlist, &PyArray_Type, &array, &lthresh, &hthresh, &nlow, &nhigh, &nonzero, &even)) return NULL;

    n = array->dimensions[0];
    for (int j = 1; j < array->nd; j++) n *= array->dimensions[j];
    k = n/2;
    if (PyArray_ISBYTESWAPPED(array)) PyArray_Byteswap(array, true);

    if (array->descr->type_num == PyArray_FLOAT) {
      float* data = (float*)PyArray_DATA(array);
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
        float *temp = data;
        data = new float[n];
        int i = 0;
	if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
	  //only nonzero
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0) data[i++] = temp[j];
          }
	} else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
	  //only lthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh) data[i++] = temp[j];
          }
	} else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //only hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] <= hthresh) data[i++] = temp[j];
          }
	} else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //nonzero and lthresh 
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[0] >= lthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //nonzero and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //lthresh and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
	  //all three
	  for (int j = 0; j < n; j++) {
	    if (temp[j] != 0 && temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
	  }
	}
	n = i;
      }
      k = (n-nhigh-nlow)/2+nlow;
      if (n == 0) {
	med = (double)0;
	if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
	return PyFloat_FromDouble(med);
      }
      gpusort_float(data, n);
      if (n%2 == 1 || !even) {
	med = (double)(data[k]);
      } else {
	med = (double)(data[k]+data[k-1])/2;
      }
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data); 
    } else if (array->descr->type_num == PyArray_DOUBLE) {
      double* data = (double*)PyArray_DATA(array);
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
        double *temp = data;
        data = new double[n];
        int i = 0;
        if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
          //only nonzero
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //only lthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //only hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //nonzero and lthresh 
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[0] >= lthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //nonzero and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //lthresh and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //all three
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        }
        n = i;
      }
      k = (n-nhigh-nlow)/2+nlow;
      if (n == 0) {
        med = (double)0;
        if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
        return PyFloat_FromDouble(med);
      }
      gpusort_double(data, n);
      if (n%2 == 1 || !even) {
        med = (double)(data[k]);
      } else {
        med = (double)(data[k]+data[k-1])/2;
      }
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
    } else if (array->descr->type_num == PyArray_INT32) {
      int* data = (int*)PyArray_DATA(array);
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
        int *temp = data;
        data = new int[n];
        int i = 0;
        if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
          //only nonzero
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //only lthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //only hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //nonzero and lthresh 
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[0] >= lthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //nonzero and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //lthresh and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //all three
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        }
        n = i;
      }
      k = (n-nhigh-nlow)/2+nlow;
      if (n == 0) {
        med = (double)0;
        if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
        return PyFloat_FromDouble(med);
      }
      gpusort_int(data, n);
      if (n%2 == 1 || !even) {
        med = (double)(data[k]);
      } else {
        med = (double)(data[k]+data[k-1])/2;
      }
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
    } else if (array->descr->type_num == PyArray_INT64) {
      long* data = (long*)PyArray_DATA(array);
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
        long *temp = data;
        data = new long[n];
        int i = 0;
        if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
          //only nonzero
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //only lthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //only hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
          //nonzero and lthresh 
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[0] >= lthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
          //nonzero and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //lthresh and hthresh
          for (int j = 0; j < n; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
          //all three
          for (int j = 0; j < n; j++) {
            if (temp[j] != 0 && temp[j] >= lthresh && temp[j] <= hthresh) data[i++] = temp[j];
          }
        }
        n = i;
      }
      k = (n-nhigh-nlow)/2+nlow;
      if (n == 0) {
        med = (double)0;
        if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
        return PyFloat_FromDouble(med);
      }
      gpusort_long(data, n);
      if (n%2 == 1 || !even) {
        med = (double)(data[k]);
      } else {
        med = (double)(data[k]+data[k-1])/2;
      }
      if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) free(data);
    } else {
      PyErr_SetString(PyExc_ValueError, "Invalid array data type");
      return NULL;
    }
    return PyFloat_FromDouble(med); 
  }
}
