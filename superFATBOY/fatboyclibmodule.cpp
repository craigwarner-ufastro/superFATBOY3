#include <math.h>
#include <stdio.h>
#include <iostream>

#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#if defined(__linux__)
#include <byteswap.h>
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define bswap_16 OSSwapInt16
#define bswap_32 OSSwapInt32
#define bswap_64 OSSwapInt64
#else
#include <byteswap.h>
#endif

//the above is a fix for macs not having byteswap.h  The else should catch UNIX machines

static PyObject * median(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject * median2d(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject * median3d(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject * dcr(PyObject *self, PyObject *args, PyObject *keywds);
static PyObject * fluxblend(PyObject *self, PyObject *args, PyObject *keywds);

/* Some sample C code for the quickselect algorithm, 
   taken from Numerical Recipes in C. */

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;

float quickselect(float *arr, int n, int k) {
  int i,ir,j,l,mid;
  float a,temp;

  l=0;
  ir=n-1;
  for(;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
      }
      return arr[k];
    }
    else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
      }
      i=l+1;
      j=ir;
      a=arr[l+1];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
      }
      arr[l+1]=arr[j];
      arr[j]=a;
      if (j >= k) ir=j-1;
      if (j <= k) l=i;
    }
  }
}

double quickselect(double *arr, int n, int k) {
  int i,ir,j,l,mid;
  double a,temp;

  l=0;
  ir=n-1;
  for(;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
      }
      return arr[k];
    }
    else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
      }
      i=l+1;
      j=ir;
      a=arr[l+1];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
      }
      arr[l+1]=arr[j];
      arr[j]=a;
      if (j >= k) ir=j-1;
      if (j <= k) l=i;
    }
  }
}

long quickselect(long *arr, int n, int k) {
  int i,ir,j,l,mid;
  long a,temp;

  l=0;
  ir=n-1;
  for(;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
      }
      return arr[k];
    }
    else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
      }
      i=l+1;
      j=ir;
      a=arr[l+1];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
      }
      arr[l+1]=arr[j];
      arr[j]=a;
      if (j >= k) ir=j-1;
      if (j <= k) l=i;
    }
  }
}

int quickselect(int *arr, int n, int k) {
  int i,ir,j,l,mid;
  int a,temp;

  l=0;
  ir=n-1;
  for(;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
      }
      return arr[k];
    }
    else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
      }
      i=l+1;
      j=ir;
      a=arr[l+1];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
      }
      arr[l+1]=arr[j];
      arr[j]=a;
      if (j >= k) ir=j-1;
      if (j <= k) l=i;
    }
  }
}

float mean_float(float *arr, int n) {
    float sum = 0;
    for (int j = 0; j < n; j++) sum += arr[j];
    sum/=n;
    return sum;
}

float mean_int(int *arr, int n) {
    float sum = 0;
    for (int j = 0; j < n; j++) sum += arr[j];
    sum/=n;
    return sum;
}

double mean_double(double *arr, int n) {
    double sum = 0;
    for (int j = 0; j < n; j++) sum += arr[j];
    sum/=n;
    return sum;
}

float mean_long(long *arr, int n) {
    float sum = 0;
    for (int j = 0; j < n; j++) sum += arr[j];
    sum/=n;
    return sum;
}

float std_float(float *arr, int n, float m) {
    float var = 0;
    for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
    float sd = sqrt(var/(n-1));
    return sd;
}

float std_int(int *arr, int n, float m) {
    float var = 0;
    for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
    float sd = sqrt(var/(n-1));
    return sd;
}

double std_double(double *arr, int n, double m) {
    double var = 0;
    for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
    double sd = sqrt(var/(n-1));
    return sd;
}

float std_long(long *arr, int n, float m) {
    float var = 0;
    for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
    float sd = sqrt(var/(n-1));
    return sd;
}


/***************** DCR subroutines ************************/
typedef struct {
  double thresh;
  int xrad, yrad, npass, diaxis;
  int lrad, urad, grad, verbose;
} dcr_options;

/*--------------------------------------------------------*/
void calc_mean(float *data, int nx, int ny, double *mean, double *sdev)
{
        int     i,            /* loop numerator           */
                j;            /* loop numerator           */
        double  s,
                ss;

  s = ss = 0.0;
  for (i=0; i<ny; i++)
  {
    for (j=0; j<nx; j++)
    {
      s+=data[i*nx+j];
      ss+=data[i*nx+j]*data[i*nx+j];
    }
  }
  *mean=s/nx/ny;
  *sdev=sqrt((ss-s*s/nx/ny)/nx/ny);

  return;
}
/*--------------------------------------------------------*/
int calc_submean(dcr_options opts, float *data, int nx, int xstart, int ystart, int dx, int dy,
                 double *mean, double *sdev) {
        int     k;
        double  s,
                ss,
                thu,
                thl;

  s = ss = 0.0;
  for (int i=ystart; i<ystart+dy; i++) {
    for (int j=xstart; j<xstart+dx; j++) {
      s+=data[i*nx+j];
      ss+=data[i*nx+j]*data[i*nx+j];
    }
  }

  *mean=s/dx/dy;
  *sdev=sqrt((ss-s*s/dx/dy)/dx/dy);

  if (*sdev == 0.0) return(0);

  thu=*mean+opts.thresh*(*sdev);
  thl=*mean-opts.thresh*(*sdev);

  //sigma clipping
  k=0;
  s = ss = 0.0;
  for (int i=ystart; i<ystart+dy; i++) {
    for (int j=xstart; j<xstart+dx; j++) {
      if ((data[i*nx+j] < thu) && (data[i*nx+j] > thl)) {
        k++;
        s+=data[i*nx+j];
        ss+=data[i*nx+j]*data[i*nx+j];
      }
    }
  }
  *mean=s/k;
  *sdev=sqrt((ss-s*s/k)/k);

  return(k);
}
/*--------------------------------------------------------*/
float max(float *data, int nx, int xs, int ys, int *xmax, int *ymax) {
        int   i,              /* rows loop numerator      */
              j;              /* columns loop numerator   */
        float maxc;           /* maximum count            */

  maxc=data[0];
  *xmax=xs;
  *ymax=ys;
  for (i=0; i<ys; i++)
  {
    for (j=0; j<xs; j++)
      if (data[i*nx+j] > maxc) { maxc=data[i*nx+j]; *xmax=j; *ymax=i; }
  }

  return(maxc);
}
/*--------------------------------------------------------*/
void minmax(float *data, int nx, int xs, int ys, int dx, int dy,
            float *minc, float *maxc) {
        int i,      /* rows loop numerator                */
            j;      /* columns loop numerator             */

  *minc = *maxc = data[ys*nx+xs];
  for (i=ys; i<ys+dy; i++)
  {
    for (j=xs; j<xs+dx; j++)
    {
      if (data[i*nx+j] < *minc) *minc=data[i*nx+j];
      if (data[i*nx+j] > *maxc) *maxc=data[i*nx+j];
    }
  }

  return;
}
/*--------------------------------------------------------*/
int make_hist(int xs, int ys, float *data, int nx, int dx, int dy, float min,
  int hs, float bin_width, int *hbuf) {
        int     i,            /* loop numerator           */
                j,            /* loop numerator           */
                hi;           /* histogram buffer index   */

  memset((void *)hbuf, 0, hs*sizeof(int));
  for (i=ys; i<ys+dy; i++) {
    for (j=xs; j<xs+dx; j++) {
      hi=(int)((data[i*nx+j]-min)/bin_width);
      if (hi < 0) {
        printf("\n\tERROR! make_hist(): (hi= %ld) < 0\n", (long)hi);
        return(EXIT_FAILURE);
      }
      if (hi > hs-1) {
        printf("\n\tERROR! make_hist(): (hi= %d) > (hs= %d)\n", hi, hs-1);
        return(EXIT_FAILURE);
      }
      hbuf[hi]++;
    }
  }

  return(EXIT_SUCCESS);
}
/*--------------------------------------------------------*/
int detect(dcr_options opts, float *data, int nx, int ny, int xs, int ys,
  int dx, int dy, char *pixmap)
{
        int     ix, iy,       /* loop numerator           */
                kx, ky,       /* loop numerator           */
                x, y,         /* whole frame coordinates  */
                k,            /* num. pixels in submean   */
                hmax,         /* hbuf maximum value       */
                n,            /* number of pixels cleaned */
                nmax,         /* limit of pixels cleaned  */
                hbs,          /* size of histogram buffer */
                mode,         /* histogram  mode          */
                i,            /* loop numerator           */
                j;            /* loop numerator           */
        float   minc,         /* minimum data             */
                maxc,         /* maximum data             */
                hw,           /* width of histogram bin   */
                th;           /* threshold                */
        double  mean,         /* data mean                */
                sdev;         /* std. deviation           */

  minmax(data, nx, xs, ys, dx, dy, &minc, &maxc);
  if (minc == maxc) return(0);

  if ((k=calc_submean(opts, data, nx, xs, ys, dx, dy, &mean, &sdev)) <= 0) {
    printf("\n\tERROR! calc_submean() failed\n");
    return(-1);
  }

  hw=1.0; /* width of histogram bin - this is valid for the counts in the image */
  hbs=(int)(maxc-minc)/hw+1; //size of histogram buffer

  int* hbuf = new int[(int)hbs]; //historgram buffer

  if (make_hist(xs, ys, data, nx, dx, dy, minc, hbs, hw, hbuf) != EXIT_SUCCESS) {
    return(-1);
  }

/** find mode = maximum peak of histogram **/
  mode=0;
  hmax=hbuf[0];
  for (i=0; i<hbs; i++) {
    if (hbuf[i] > hmax) {
      hmax=hbuf[i];
      mode=i;
    }
  }

/** determine clean threshold **/
  if (mode == 0) mode=(int)((mean-minc)/hw);

  j=0;
  for (i=mode; i<hbs; i++) {
    if ((hbuf[i]) == 0) j++; else j=0;
    if (j > (int)(opts.thresh*sdev/hw)) break;
  }
  th=minc+(float)i*hw;

/** count number of pixels to be cleaned **/
  n=0;
  for (j=i; j<hbs; j++) n+=hbuf[j];

  free(hbuf);

  nmax=(int)sqrt((double)dx*dy);
  if (n > nmax) {
    if (opts.verbose) printf("\n\tWARNING: [%d:%d,%d:%d] number of pixels to be cleaned: %d > %d\n", xs+1, xs+dx, ys+1, ys+dy, n, nmax);
    return(0);
  }

/** detect **/
  n=0;
  for (iy=0; iy<dy; iy++) {
    y=ys+iy;
    for (ix=0; ix<dx; ix++) {
      x=xs+ix;

      if (data[y*nx+x] > th) {
        n++;
        for (ky=-opts.grad; ky<=opts.grad; ky++) {
          if ((y+ky >= 0) && (y+ky < ny)) {
            for (kx=-opts.grad; kx<=opts.grad; kx++) {
              if ((x+kx >= 0) && (x+kx < nx)) {
                pixmap[(y+ky)*nx+x+kx]=1;
              }
            }
          }
        }
      }
    }
  }

  if ((n != 0) && (opts.verbose > 1)) {
    printf("  min= %.1f max= %.1f mean= %.1f+-%.1f (npix= %d/%d) mode= %.1f\n", minc, maxc, mean, sdev, k, dx*dy, minc+(float)mode*hw);
    printf("  threshold= %.1f -> %d pixels to be cleaned\n", th, n);
  }

  return(n);
}
/*--------------------------------------------------------*/
int make_map(dcr_options opts, float *data, int nx, int ny, char *pixmap)
{
        int     xs,
                ys,
                dx,
                dy,
                imax,
                jmax,
                nc,           /* number of pixels cleaned */
                n;            /* number of pixels cleaned */

  dx=2*opts.xrad;
  if (dx > nx) {
    printf("\n\tERROR! make_map(): x-radius of the box too large\n");
    return(-1);
  }
  imax=nx/opts.xrad-1;

  dy=2*opts.yrad;
  if (dy > ny) {
    printf("\n\tERROR! make_map(): y-radius of the box too large\n");
    dy = ny;
    //return(-1);
  }
  jmax=ny/opts.yrad-1;

/** clean most of the frame **/
  n = nc = 0;
  for (int j=0; j<jmax; j++) {
    ys=j*opts.yrad;
    for (int i=0; i<imax; i++) {
      xs=i*opts.xrad;

      nc=detect(opts, data, nx, ny, xs, ys, dx, dy, pixmap);
      if (nc < 0) return(-1);
      if (nc > 0) {
        n+=nc;
        if (opts.verbose > 1) printf("[%d:%d,%d:%d]\n", xs+1, xs+dx, ys+1, ys+dy);
      }
    }
  }

/** clean margins of the frame **/
/** left margin **/
  xs=nx-dx;
  for (int j=0; j<jmax; j++) {
    ys=j*opts.yrad;

    nc=detect(opts, data, nx, ny, xs, ys, dx, dy, pixmap);
    if (nc < 0) return(-1);
    if (nc > 0) {
      n+=nc;
      if (opts.verbose > 1) printf("[%d:%d,%d:%d]\n", xs+1, xs+dx, ys+1, ys+dy);
    }
  }

/** top margin **/
  ys=ny-dy;
  for (int i=0; i<imax; i++) {
    xs=i*opts.xrad;

    nc=detect(opts, data, nx, ny, xs, ys, dx, dy, pixmap);
    if (nc < 0) return(-1);
    if (nc > 0) {
      n+=nc;
      if (opts.verbose > 1) printf("[%d:%d,%d:%d]\n", xs+1, xs+dx, ys+1, ys+dy);
    }
  }

/** left-top margin **/
  xs=nx-dx;
  ys=ny-dy;

  nc=detect(opts, data, nx, ny, xs, ys, dx, dy, pixmap);
  if (nc < 0) return(-1);
  if (nc > 0) {
    n+=nc;
    if (opts.verbose > 1) printf("[%d:%d,%d:%d]\n", xs+1, xs+dx, ys+1, ys+dy);
  }

  return(n);
}
/*--------------------------------------------------------*/
void clean_xdisp(dcr_options opts, float *data, int i, int j, int nx, int ny, double mean, char *pixmap, float *cf) {
        int     mx, ns;
        double  s;

  ns=0;
  s=0.0;

  for (mx=-opts.urad; mx<=-opts.lrad; mx++) {
    if (j+mx < 0) continue;
    if (pixmap[i*nx+j+mx]) continue;
    s+=data[i*nx+j+mx];
    ns++;
  }

  for (mx=opts.lrad; mx<=opts.urad; mx++) {
    if (j+mx >= nx) break;
    if (pixmap[i*nx+j+mx]) continue;
    s+=data[i*nx+j+mx];
    ns++;
  }

  if (ns) s/=ns;
  else    s=mean;

  cf[i*nx+j]+=data[i*nx+j]-s;
  data[i*nx+j]=s;

  return;
}
/*--------------------------------------------------------*/
void clean_ydisp(dcr_options opts, float *data, int i, int j, int nx, int ny, double mean, char *pixmap, float *cf) {
        int     my, ns;
        double  s;

  ns=0;
  s=0.0;

  for (my=-opts.urad; my<=-opts.lrad; my++) {
    if (i+my < 0) continue;
    if (pixmap[(i+my)*nx+j]) continue;
    s+=data[(i+my)*nx+j];
    ns++;
  }

  for (my=opts.lrad; my<=opts.urad; my++) {
    if (i+my >= ny) break;
    if (pixmap[(i+my)*nx+j]) continue;
    s+=data[(i+my)*nx+j];
    ns++;
  }

  if (ns) s/=ns;
  else    s=mean;

  cf[i*nx+j]+=data[i*nx+j]-s;
  data[i*nx+j]=s;

  return;
}
/*--------------------------------------------------------*/
void clean_nodisp(dcr_options opts, float *data, int i, int j, int nx, int ny, double mean, char *pixmap, float *cf) {
        int     mx, my, ns;
        float   d;
        double  s;

  ns=0;
  s=0.0;

  for (my=opts.lrad; my<=opts.urad; my++) {
    for (mx=opts.lrad; mx<=opts.urad; mx++) {
      d=sqrt((float)mx*mx+(float)my*my);
      if ((d > (float)opts.urad) || (d < (float)opts.lrad)) continue;

      if ((j+mx < nx) && (i+my < ny)) {
        if (!pixmap[(i+my)*nx+j+mx]) {
          s+=data[(i+my)*nx+j+mx];
          ns++;
        }
      }

      if ((j-mx >= 0) && (i+my < ny)) {
        if (!pixmap[(i+my)*nx+j-mx]) {
          s+=data[(i+my)*nx+j-mx];
          ns++;
        }
      }

      if ((j+mx < nx) && (i-my >= 0)) {
        if (!pixmap[(i-my)*nx+j+mx]) {
          s+=data[(i-my)*nx+j+mx];
          ns++;
        }
      }
      if ((j-mx >= 0) && (i-my >= 0)) {
        if (!pixmap[(i-my)*nx+j-mx]) {
          s+=data[(i-my)*nx+j-mx];
          ns++;
        }
      }
    }
  }

  if (ns) s/=ns;
  else    s=mean;

  cf[i*nx+j]+=data[i*nx+j]-s;
  data[i*nx+j]=s;

  return;
}
/*--------------------------------------------------------*/
int clean(dcr_options opts, float *data, int nx, int ny, double mean, char *pixmap, float *cf) {
  for (int i=0; i<ny; i++) {
    for (int j=0; j<nx; j++) {
      if (pixmap[i*nx+j]) {
        switch (opts.diaxis) {
          case 1:  clean_xdisp(opts, data, i, j, nx, ny, mean, pixmap, cf);
                   break;
          case 2:  clean_ydisp(opts, data, i, j, nx, ny, mean, pixmap, cf);
                   break;
          default: clean_nodisp(opts, data, i, j, nx, ny, mean, pixmap, cf);
                   break;
        }
      }
    }
  }

  return(EXIT_SUCCESS);
}
/*--------------------------------------------------------*/


static PyMethodDef module_methods[] = {
  {"median", (PyCFunction)median, METH_VARARGS | METH_KEYWORDS, "Calculate median using quickselect on CPU"},
  {"median2d", (PyCFunction)median2d, METH_VARARGS | METH_KEYWORDS, "Calculate median using quickselect on CPU"},
  {"median3d", (PyCFunction)median3d, METH_VARARGS | METH_KEYWORDS, "Calculate median using quickselect on CPU"},
  {"dcr", (PyCFunction)dcr, METH_VARARGS | METH_KEYWORDS, "Remove cosmic rays using DCR algorithim, see Pych, W., 2004, PASP, 116, 148"},
  {"fluxblend", (PyCFunction)fluxblend, METH_VARARGS | METH_KEYWORDS, "Sum fluxes in a with common indices in b into array c"},
  {NULL, NULL, 0, NULL} /* Sentinel */
};

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
  #define PyMODINIT_FUNC void
#endif

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "fatboyclib",     /* m_name */
      "Module with C++ methods for sFB",  /* m_doc */
      -1,                  /* m_size */
      module_methods,     /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
#endif

#if PY_MAJOR_VERSION < 3
    PyMODINIT_FUNC initfatboyclib(void) {
        PyObject* m;
        m = Py_InitModule3("fatboyclib", module_methods, "Example module that creates an extension type.");
        if (m == NULL){ return; }
        import_array(); /* required NumPy initialization */
    }
#else
    PyMODINIT_FUNC PyInit_fatboyclib(void) {
        PyObject* m;
        m = PyModule_Create(&moduledef);
        if (m == NULL){ return m; }
        import_array(); /* required NumPy initialization */
	return m;
    }
#endif


static PyObject * median3d(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *array;
  PyObject *medArray = NULL;
  double lthresh=INT_MIN, hthresh=INT_MAX;
  int nlow=0, nhigh=0;
  bool nonzero=false, even=false;
  int n, k;
  static char *kwlist[] = {"array", "lthreshold", "hthreshold", "nlow", "nhigh", "nonzero", "even", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|ddiibb", kwlist, &PyArray_Type, &array, &lthresh, &hthresh, &nlow, &nhigh, &nonzero, &even)) return NULL;

  if (array->nd < 3) {
    PyErr_SetString(PyExc_ValueError, "Array must have 3 dimesions!");
    return NULL;
  }
  n = array->dimensions[2];
  int ny = array->dimensions[0]*array->dimensions[1];
  k = n/2;
  npy_intp dims[2] = {array->dimensions[0], array->dimensions[1]};
  //if (PyArray_ISBYTESWAPPED(array)) PyArray_Byteswap(array, true);

  if (array->descr->type_num == PyArray_FLOAT) {
    float* data = (float*)PyArray_DATA(array);
    float* medVals = new float[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      float *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new float[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
	  if (k == 0 || i == 0) {
	    medVals[j] = 0;
	    continue;
	  }
	  if (i %2 == 1 || !even) {
	    medVals[j] = (float)quickselect(temp, i, k);
	  } else {
	    medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = (float)quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (float)(quickselect(&data[j*n], n, k) + quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, medVals);
  } else if (array->descr->type_num == PyArray_DOUBLE) {
    double* data = (double*)PyArray_DATA(array);
    double* medVals = new double[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      double *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new double[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
	  if (i %2 == 1 || !even) {
	    medVals[j] = quickselect(temp, i, k);
	  } else {
	    medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (quickselect(&data[j*n], n, k) + quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, medVals);
  } else if (array->descr->type_num == PyArray_INT32) {
    int* data = (int*)PyArray_DATA(array);
    float* medVals = new float[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      int *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new int[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
	  if (k == 0 || i == 0) {
	    medVals[j] = 0;
	    continue;
	  }
	  if (i %2 == 1 || !even) {
	    medVals[j] = (float)quickselect(temp, i, k);
	  } else {
	    medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = (float)quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (float)(quickselect(&data[j*n], n, k) + quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, medVals);
  } else if (array->descr->type_num == PyArray_INT64) {
    long* data = (long*)PyArray_DATA(array);
    float* medVals = new float[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      long *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new long[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
	  if (k == 0 || i == 0) {
	    medVals[j] = 0;
	    continue;
	  }
	  if (i %2 == 1 || !even) {
	    medVals[j] = (float)quickselect(temp, i, k);
	  } else {
	    medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = (float)quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (float)(quickselect(&data[j*n], n, k) + quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, medVals);
  } else {
    PyErr_SetString(PyExc_ValueError, "Invalid array data type");
    return NULL;
  }
  return medArray; 
}

static PyObject * median2d(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *array;
  PyObject *medArray = NULL;
  double lthresh=INT_MIN, hthresh=INT_MAX, lsigma=3, hsigma=3;
  int nlow=0, nhigh=0, niter=3;
  bool nonzero=false, even=false, sigclip=false, mclip=false;
  int n, k;
  static char *kwlist[] = {"array", "lthreshold", "hthreshold", "nlow", "nhigh", "nonzero", "even", "sigclip", "lsigma", "hsigma", "mclip", "niter", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|ddiibbbddbi", kwlist, &PyArray_Type, &array, &lthresh, &hthresh, &nlow, &nhigh, &nonzero, &even, &sigclip, &lsigma, &hsigma, &mclip, &niter)) return NULL;

  n = array->dimensions[1];
  int ny = array->dimensions[0];
  k = n/2;
  npy_intp dims[1] = {ny};
  //if (PyArray_ISBYTESWAPPED(array)) PyArray_Byteswap(array, true);

  if (array->descr->type_num == PyArray_FLOAT) {
    float* data = (float*)PyArray_DATA(array);
    float* medVals = new float[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      float *temp = NULL;
      int i = 0;
      temp = new float[n];
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  //temp = new float[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
	  if (k == 0 || i == 0) {
	    medVals[j] = 0;
	    continue;
	  }
	  if (i %2 == 1 || !even) {
	    medVals[j] = (float)quickselect(temp, i, k);
	  } else {
	    medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          //temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          //temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          //temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          //temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          //temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          //temp = new float[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } 
      n = i;
      delete temp;
    } else if (sigclip) {
      float *temp = NULL;
      int i = 0;
      temp = new float[n];
      int iter = 0;
      int nold = n+1;
      float m = 0, sd = 0;
      for (int j = 0; j < ny; j++) {
	i = 0;
        iter = 0;
        m = 0;
	sd = 0;
	for (int l = 0; l < n; l++) temp[i++] = data[j*n+l];
        nold = i+1;
        while (nold > i && i > 2 && iter < niter) {
          iter++;
          k = i/2;
          if (!mclip) {
            //mean
            m = mean_float(temp, i);
          } else {
            if (i % 2 == 1 || !even) {
              m = quickselect(temp, i, k);
            } else {
              m = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
            }
          }
          sd = std_float(temp, i, m);
          nold = i;
          i = 0;
          lthresh = m-lsigma*sd;
          hthresh = m+hsigma*sd;
          for (int j = 0; j < nold; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) {
              temp[i++] = temp[j];
            }
          }
        }
        k = (i-nhigh-nlow)/2+nlow;
        if (k == 0 || i == 0) {
          medVals[j] = 0;
          continue;
        }
        if (i %2 == 1 || !even) {
          medVals[j] = (float)quickselect(temp, i, k);
        } else {
          medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
        }
      }
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = (float)quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (float)(quickselect(&data[j*n], n, k) + (float)quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, medVals);
  } else if (array->descr->type_num == PyArray_DOUBLE) {
    double* data = (double*)PyArray_DATA(array);
    double* medVals = new double[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      double *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new double[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
	  if (i %2 == 1 || !even) {
	    medVals[j] = quickselect(temp, i, k);
	  } else {
	    medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new double[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = quickselect(temp, i, k);
          } else {
            medVals[j] = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else if (sigclip) {
      double *temp = NULL;
      int i = 0;
      temp = new double[n];
      int iter = 0;
      int nold = n+1;
      double m = 0, sd = 0;
      for (int j = 0; j < ny; j++) {
        i = 0;
        iter = 0;
        m = 0;
        sd = 0;
        for (int l = 0; l < n; l++) temp[i++] = data[j*n+l];
        nold = i+1;
        while (nold > i && i > 2 && iter < niter) {
          iter++;
          k = i/2;
          if (!mclip) {
            //mean
            m = mean_double(temp, i);
          } else {
            if (i % 2 == 1 || !even) {
              m = quickselect(temp, i, k);
            } else {
              m = (quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
            }
          }
          sd = std_double(temp, i, m);
          nold = i;
          i = 0;
          lthresh = m-lsigma*sd;
          hthresh = m+hsigma*sd;
          for (int j = 0; j < nold; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) {
              temp[i++] = temp[j];
            }
          }
        }
        k = (i-nhigh-nlow)/2+nlow;
        if (k == 0 || i == 0) {
          medVals[j] = 0;
          continue;
        }
        if (i %2 == 1 || !even) {
          medVals[j] = (float)quickselect(temp, i, k);
        } else {
          medVals[j] = (float)(quickselect(temp, i, k) + quickselect(temp, i, k-1))/2;
        }
      }
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (quickselect(&data[j*n], n, k) + quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, medVals);
  } else if (array->descr->type_num == PyArray_INT32) {
    int* data = (int*)PyArray_DATA(array);
    float* medVals = new float[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      int *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new int[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
	  if (k == 0 || i == 0) {
	    medVals[j] = 0;
	    continue;
	  }
	  if (i %2 == 1 || !even) {
	    medVals[j] = (float)quickselect(temp, i, k);
	  } else {
	    medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new int[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else if (sigclip) {
      int *temp = NULL;
      int i = 0;
      temp = new int[n];
      int iter = 0;
      int nold = n+1;
      float m = 0, sd = 0;
      for (int j = 0; j < ny; j++) {
        i = 0;
        iter = 0;
        m = 0;
        sd = 0;
        for (int l = 0; l < n; l++) temp[i++] = data[j*n+l];
        nold = i+1;
        while (nold > i && i > 2 && iter < niter) {
          iter++;
          k = i/2;
          if (!mclip) {
            //mean
            m = mean_int(temp, i);
          } else {
            if (i % 2 == 1 || !even) {
              m = quickselect(temp, i, k);
            } else {
              m = (quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
            }
          }
          sd = std_int(temp, i, m);
          nold = i;
          i = 0;
          lthresh = m-lsigma*sd;
          hthresh = m+hsigma*sd;
          for (int j = 0; j < nold; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) {
              temp[i++] = temp[j];
            }
          }
        }
        k = (i-nhigh-nlow)/2+nlow;
        if (k == 0 || i == 0) {
          medVals[j] = 0;
          continue;
        }
        if (i %2 == 1 || !even) {
          medVals[j] = (float)quickselect(temp, i, k);
        } else {
          medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
        }
      }
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = (float)quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (float)(quickselect(&data[j*n], n, k) + (float)quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, medVals);
  } else if (array->descr->type_num == PyArray_INT64) {
    long* data = (long*)PyArray_DATA(array);
    float* medVals = new float[ny];
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
      long *temp = NULL;
      int i = 0;
      if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
        //only nonzero
        for (int j = 0; j < ny; j++) {
	  temp = new long[n];
	  i = 0;
	  for (int l = 0; l < n; l++) {
	    if (data[j*n+l] != 0) temp[i++] = data[j*n+l];
	  }
	  k = (i-nhigh-nlow)/2+nlow;
	  if (k == 0 || i == 0) {
	    medVals[j] = 0;
	    continue;
	  }
	  if (i %2 == 1 || !even) {
	    medVals[j] = (float)quickselect(temp, i, k);
	  } else {
	    medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
	  }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //only lthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //only hthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
        //nonzero and lthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
        //nonzero and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //lthresh and hthresh
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
        //all three
        for (int j = 0; j < ny; j++) {
          temp = new long[n];
          i = 0;
          for (int l = 0; l < n; l++) {
            if (data[j*n+l] != 0 && data[j*n+l] >= lthresh && data[j*n+l] <= hthresh) temp[i++] = data[j*n+l];
          }
          k = (i-nhigh-nlow)/2+nlow;
          if (k == 0 || i == 0) {
            medVals[j] = 0;
            continue;
          }
          if (i %2 == 1 || !even) {
            medVals[j] = (float)quickselect(temp, i, k);
          } else {
            medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
          }
        }
      }
      n = i;
      delete temp;
    } else if (sigclip) {
      long *temp = NULL;
      int i = 0;
      temp = new long[n];
      int iter = 0;
      int nold = n+1;
      float m = 0, sd = 0;
      for (int j = 0; j < ny; j++) {
        i = 0;
        iter = 0;
        m = 0;
        sd = 0;
        for (int l = 0; l < n; l++) temp[i++] = data[j*n+l];
        nold = i+1;
        while (nold > i && i > 2 && iter < niter) {
          iter++;
          k = i/2;
          if (!mclip) {
            //mean
            m = mean_long(temp, i);
          } else {
            if (i % 2 == 1 || !even) {
              m = (float)quickselect(temp, i, k);
            } else {
              m = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
            }
          }
          sd = std_long(temp, i, m);
          nold = i;
          i = 0;
          lthresh = m-lsigma*sd;
          hthresh = m+hsigma*sd;
          for (int j = 0; j < nold; j++) {
            if (temp[j] >= lthresh && temp[j] <= hthresh) {
              temp[i++] = temp[j];
            }
          }
        }
        k = (i-nhigh-nlow)/2+nlow;
        if (k == 0 || i == 0) {
          medVals[j] = 0;
          continue;
        }
        if (i %2 == 1 || !even) {
          medVals[j] = (float)quickselect(temp, i, k);
        } else {
          medVals[j] = (float)(quickselect(temp, i, k) + (float)quickselect(temp, i, k-1))/2;
        }
      }
      delete temp;
    } else {
      k = (n-nhigh-nlow)/2+nlow;
      for (int j = 0; j < ny; j++) {
        if (n%2 == 1 || !even) {
          medVals[j] = (float)quickselect(&data[j*n], n, k);
        } else {
          medVals[j] = (float)(quickselect(&data[j*n], n, k) + (float)quickselect(&data[j*n], n, k-1))/2;
        }
      }
    }
    medArray = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, medVals);
  } else {
    PyErr_SetString(PyExc_ValueError, "Invalid array data type");
    return NULL;
  }
  return medArray; 
}

static PyObject * median(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *array;
  double med=INT_MIN;
  double lthresh=INT_MIN, hthresh=INT_MAX, lsigma=3, hsigma=3;
  int nlow=0, nhigh=0, niter=3;
  bool nonzero=false, even=false, sigclip=false, mclip=false;
  int n, k;
  static char *kwlist[] = {"array", "lthreshold", "hthreshold", "nlow", "nhigh", "nonzero", "even", "sigclip", "lsigma", "hsigma", "mclip", "niter", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|ddiibbbddbi", kwlist, &PyArray_Type, &array, &lthresh, &hthresh, &nlow, &nhigh, &nonzero, &even, &sigclip, &lsigma, &hsigma, &mclip, &niter)) return NULL;

  n = array->dimensions[0];
  for (int j = 1; j < array->nd; j++) n *= array->dimensions[j];
  k = n/2; 
  //if (PyArray_ISBYTESWAPPED(array)) PyArray_Byteswap(array, true);

  if (array->descr->type_num == PyArray_FLOAT) {
    float* data = (float*)PyArray_DATA(array);
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) {
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
          if (temp[j] != 0 && temp[j] >= lthresh) data[i++] = temp[j];
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
      } else {
	//none, just sigclip
	for (int j = 0; j < n; j++) data[i++] = temp[j];
      }
      n = i;
    }
    if (sigclip) {
      int iter = 0;
      int nold = n+1;
      float m = 0, sd = 0;
      while (nold > n && n > 2 && iter < niter) {
        iter++;
        k = n/2;
        if (!mclip) {
          //mean
          m = mean_float(data, n);
        } else {
          if (n % 2 == 1 || !even) {
            m = quickselect(data, n, k);
          } else {
            m = (quickselect(data, n, k) + quickselect(data, n, k-1))/2;
          }
        }
        sd = std_float(data, n, m);
        nold = n;
        n = 0;
        lthresh = m-lsigma*sd;
        hthresh = m+hsigma*sd;
        for (int j = 0; j < nold; j++) {
          if (data[j] >= lthresh && data[j] <= hthresh) {
            data[n++] = data[j];
          }
        }
      }
    }
    k = (n-nhigh-nlow)/2+nlow;
    if (n%2 == 1 || !even) {
      med = (double)quickselect(data, n, k);
    } else {
      med = (double)(quickselect(data, n, k) + (double)quickselect(data, n, k-1))/2;
    }
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) delete data; 
  } else if (array->descr->type_num == PyArray_DOUBLE) {
    double* data = (double*)PyArray_DATA(array);
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) {
      double* temp = data;
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
          if (temp[j] != 0 && temp[j] >= lthresh) data[i++] = temp[j];
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
      } else {
        //none, just sigclip
        for (int j = 0; j < n; j++) data[i++] = temp[j];
      }
      n = i;
    }
    if (sigclip) {
      int iter = 0;
      int nold = n+1;
      double m = 0, sd = 0;
      while (nold > n && n > 2 && iter < niter) {
        iter++;
        k = n/2;
        if (!mclip) {
          //mean
          m = mean_double(data, n);
        } else {
          if (n % 2 == 1 || !even) {
            m = quickselect(data, n, k);
          } else {
            m = (quickselect(data, n, k) + quickselect(data, n, k-1))/2;
          }
        }
        sd = std_double(data, n, m);
        nold = n;
        n = 0;
        lthresh = m-lsigma*sd;
        hthresh = m+hsigma*sd;
        for (int j = 0; j < nold; j++) {
          if (data[j] >= lthresh && data[j] <= hthresh) {
            data[n++] = data[j];
          }
        }
      }
    }
    k = (n-nhigh-nlow)/2+nlow;
    if (n%2 == 1 || !even) {
      med = (double)quickselect(data, n, k);
    } else {
      med = (double)(quickselect(data, n, k) + (double)quickselect(data, n, k-1))/2;
    }
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) delete data;
  } else if (array->descr->type_num == PyArray_INT32) {
    int* data = (int*)PyArray_DATA(array);
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) {
      int* temp = data;
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
          if (temp[j] != 0 && temp[j] >= lthresh) data[i++] = temp[j];
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
      } else {
        //none, just sigclip
        for (int j = 0; j < n; j++) data[i++] = temp[j];
      }
      n = i;
    }
    if (sigclip) {
      int iter = 0;
      int nold = n+1;
      float m = 0, sd = 0;
      while (nold > n && n > 2 && iter < niter) {
        iter++;
        k = n/2;
        if (!mclip) {
          //mean
          m = mean_int(data, n);
        } else {
          if (n % 2 == 1 || !even) {
            m = (float)quickselect(data, n, k);
          } else {
            m = (float)(quickselect(data, n, k) + (float)quickselect(data, n, k-1))/2;
          }
        }
        sd = std_int(data, n, m);
        nold = n;
        n = 0;
        lthresh = m-lsigma*sd;
        hthresh = m+hsigma*sd;
        for (int j = 0; j < nold; j++) {
          if (data[j] >= lthresh && data[j] <= hthresh) {
            data[n++] = data[j];
          }
        }
      }
    }
    k = (n-nhigh-nlow)/2+nlow;
    if (n%2 == 1 || !even) {
      med = (double)quickselect(data, n, k);
    } else {
      med = (double)(quickselect(data, n, k) + (double)quickselect(data, n, k-1))/2;
    }
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) delete data;
  } else if (array->descr->type_num == PyArray_INT64) {
    long* data = (long*)PyArray_DATA(array);
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) {
      long* temp = data;
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
          if (temp[j] != 0 && temp[j] >= lthresh) data[i++] = temp[j];
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
      } else {
        //none, just sigclip
        for (int j = 0; j < n; j++) data[i++] = temp[j];
      }
      n = i;
    }
    if (sigclip) {
      int iter = 0;
      int nold = n+1;
      float m = 0, sd = 0;
      while (nold > n && n > 2 && iter < niter) {
        iter++;
        k = n/2;
        if (!mclip) {
          //mean
          m = mean_long(data, n);
        } else {
          if (n % 2 == 1 || !even) {
            m = (float)quickselect(data, n, k);
          } else {
            m = (float)(quickselect(data, n, k) + (float)quickselect(data, n, k-1))/2;
          }
        }
        sd = std_long(data, n, m);
        nold = n;
        n = 0;
        lthresh = m-lsigma*sd;
        hthresh = m+hsigma*sd;
        for (int j = 0; j < nold; j++) {
          if (data[j] >= lthresh && data[j] <= hthresh) {
            data[n++] = data[j];
          }
        }
      }
    }
    k = (n-nhigh-nlow)/2+nlow;
    if (n%2 == 1 || !even) {
      med = (double)quickselect(data, n, k);
    } else {
      med = (double)(quickselect(data, n, k) + quickselect(data, n, k-1))/2;
    }
    if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX || sigclip) delete data;
  } else {
    PyErr_SetString(PyExc_ValueError, "Invalid array data type");
    return NULL;
  }
  return PyFloat_FromDouble(med);
}

static PyObject * dcr(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *array;
  PyObject *cfArray = NULL;
  double thresh = 4;
  int xrad=9, yrad=9, npass=5, diaxis=1;
  int lrad=1, urad=3, grad=1, verbose=1;
  int nx, ny, nc, np, xmax, ymax; 
  float maxc;
  double mean, sdev;
  static char *kwlist[] = {"array", "thresh", "xrad", "yrad", "npass", "diaxis", "lrad", "urad", "grad", "verbose", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!|diiiiiiii", kwlist, &PyArray_Type, &array, &thresh, &xrad, &yrad, &npass, &diaxis, &lrad, &urad, &grad, &verbose)) return NULL;

  if (array->nd != 2) {
    PyErr_SetString(PyExc_ValueError, "Array must have 2 dimesions!");
    return NULL;
  }
  nx = array->dimensions[1];
  ny = array->dimensions[0];

  //set up dcr_options struct
  dcr_options opts;
  opts.thresh = thresh;
  opts.xrad = xrad;
  opts.yrad = yrad;
  opts.npass = npass;
  opts.diaxis = diaxis;
  opts.lrad = lrad;
  opts.urad = urad;
  opts.grad = grad;
  opts.verbose = verbose;

  npy_intp dims[2] = {array->dimensions[0], array->dimensions[1]};
  float* data = (float*)PyArray_DATA(array);
  float* cf = new float[nx*ny];
  for (int i = 0; i < nx*ny; i++) cf[i] = 0; //initialize cf array to all zeros
  char* pixmap = new char[nx*ny];

  if (verbose) {
    printf("Whole frame before cleaning:\n");
    calc_mean(data, nx, ny, &mean, &sdev);
    printf("mean= %f +- %f\n", mean, sdev);
    maxc=max(data, nx, nx, ny, &xmax, &ymax);
    printf("max count= %f (%d,%d)\n", maxc, xmax+1, ymax+1);
  }

  nc=0;
  for (int ipass=0; ipass < npass; ipass++) {
    for (int i=0; i<ny; i++) {
      for (int j=0; j<nx; j++) {
        pixmap[i*nx+j]=0;
      }
    }

    if ((np=make_map(opts, data, nx, ny, pixmap)) < 0) {
      PyErr_SetString(PyExc_ValueError, "Error making map");
      return NULL;
    }

    clean(opts, data, nx, ny, mean, pixmap, cf);

    nc+=np;
    if (verbose) printf("Pass %d: %d pixels cleaned\n", ipass, np);
    if (verbose > 1) printf("--------------------------\n");
    if (np == 0) break;
  }

  free(pixmap);

  if (verbose) {
    printf("Total number of pixels/cleaned= %d/%d (%.1f%%)\n",
            nx*ny, nc, nc*100.0/nx/ny);
    printf("Whole frame after cleaning:\n");
    calc_mean(data, nx, ny, &mean, &sdev);
    printf("mean= %f +- %f\n", mean, sdev);
    maxc=max(data, nx, nx, ny, &xmax, &ymax);
    printf("max count= %f (%d,%d)\n", maxc, xmax+1, ymax+1);
  }

  cfArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, cf);
  //return tuple with number of pixels cleaned and cf array
  return Py_BuildValue("(i,O)", nc, cfArray);
}

static PyObject * fluxblend(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *flux, *indices;
  PyObject *output = NULL;
  int n, nout;
  static char *kwlist[] = {"flux", "indices", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!", kwlist, &PyArray_Type, &flux, &PyArray_Type, &indices)) return NULL;
  //if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &flux, &PyArray_Type, &indices)) return NULL;

  n = flux->dimensions[0];
  if (indices->dimensions[0] != n) {
    PyErr_SetString(PyExc_ValueError, "Array must have 2 dimesions!");
    return NULL;
  }

  if (flux->descr->type_num == PyArray_DOUBLE) {
    double* a = (double*)PyArray_DATA(flux);
    int* b = (int*)PyArray_DATA(indices);

    //find array size for outputs
    nout = 0;
    for (int i = 0; i < n; i++) if (b[i] >= nout) nout = b[i]+1;
    double* c = new double[nout];
    npy_intp dims[1] = {nout};

    for (int i = 0; i < nout; i++) c[i] = 0; //initialize c array to all zeros
    for (int i = 0; i < n; i++) c[b[i]] += a[i];
    output = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, c);
  } else { 
    float* a = (float*)PyArray_DATA(flux);
    int* b = (int*)PyArray_DATA(indices);

    //find array size for outputs
    nout = 0;
    for (int i = 0; i < n; i++) if (b[i] >= nout) nout = b[i]+1;
    double* c = new double[nout];
    npy_intp dims[1] = {nout};

    for (int i = 0; i < nout; i++) c[i] = 0; //initialize c array to all zeros
    for (int i = 0; i < n; i++) c[b[i]] += a[i];
    output = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, c);
  }
  return output;
}
