hasCuda = True
try:
    import superFATBOY
    if (not superFATBOY.gpuEnabled()):
        hasCuda = False
    else:
        import pycuda.driver as drv
        import pycuda.tools
        if (not superFATBOY.threaded()):
            #If not threaded mode, import autoinit.  Otherwise assume context exists.
            #Code will crash if in threaded mode and context does not exist.
            import pycuda.autoinit
        from pycuda.compiler import SourceModule
except Exception:
    print("gpu_arraymedian> WARNING: PyCUDA not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)
import numpy
import numpy.linalg as la

import math, time
from numpy import *
from superFATBOY import fatboyclib
defaultKernel = fatboyclib.median
try:
    from superFATBOY import cp_select
    defaultKernel = cp_select.cpmedian
except Exception:
    print("gpu_arraymedian> WARNING: cp_select not installed!")
try:
    from superFATBOY import fatboycudalib
    defaultKernel = fatboycudalib.gpumedian
except Exception:
    print("gpu_arraymedian> WARNING: fatboycudalib not installed!")

blocks = 2048*4
block_size = 512

def get_mod():
    mod = None
    if (hasCuda and superFATBOY.gpuEnabled()):
        try:
            mod = SourceModule("""

        __device__ void bubblesort_float(float* arr, int n) {
          float temp;
          for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
              if (arr[j] < arr[i]) {
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
              }
            }
          }
        }

        __device__ void bubblesort_int(int* arr, int n) {
          int temp;
          for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
              if (arr[j] < arr[i]) {
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
              }
            }
          }
        }

        __device__ void bubblesort_double(double* arr, int n) {
          double temp;
          for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
              if (arr[j] < arr[i]) {
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
              }
            }
          }
        }

        __device__ void bubblesort_long(long* arr, int n) {
          long temp;
          for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
              if (arr[j] < arr[i]) {
                temp = arr[j];
                arr[j] = arr[i];
                arr[i] = temp;
              }
            }
          }
        }

        __device__ void quicksort_float(float* arr, int left, int right) {
          int i = left, j = right;
          float tmp;
          float pivot = arr[(left + right) / 2];

          /* partition */
          while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
              tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
            }
          };

          /* recursion */
          if (left < j) quicksort_float(arr, left, j);
          if (i < right) quicksort_float(arr, i, right);
        }

        __device__ void quicksort_int(int* arr, int left, int right) {
          int i = left, j = right;
          int tmp;
          int pivot = arr[(left + right) / 2];

          /* partition */
          while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
              tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
            }
          };

          /* recursion */
          if (left < j) quicksort_int(arr, left, j);
          if (i < right) quicksort_int(arr, i, right);
        }

        __device__ void quicksort_double(double* arr, int left, int right) {
          int i = left, j = right;
          double tmp;
          double pivot = arr[(left + right) / 2];

          /* partition */
          while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
              tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
            }
          };

          /* recursion */
          if (left < j) quicksort_double(arr, left, j);
          if (i < right) quicksort_double(arr, i, right);
        }

        __device__ void quicksort_long(long* arr, int left, int right) {
          int i = left, j = right;
          long tmp;
          long pivot = arr[(left + right) / 2];

          /* partition */
          while (i <= j) {
            while (arr[i] < pivot) i++;
            while (arr[j] > pivot) j--;
            if (i <= j) {
              tmp = arr[i];
              arr[i] = arr[j];
              arr[j] = tmp;
              i++;
              j--;
            }
          };

          /* recursion */
          if (left < j) quicksort_long(arr, left, j);
          if (i < right) quicksort_long(arr, i, right);
        }

        __device__ float quickselect_float(float* arr, int n, int k) {
          int i,ir,j,l,mid;
          float a,temp;

          l = 0;
          ir=n-1;
          for(;;) {
            if (ir <= l+1) {
              if (ir == l+1 && arr[ir] < arr[l]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              return arr[k];
            }
            else {
              mid=(l+ir) >> 1;
              temp = arr[mid];
              arr[mid] = arr[l+1];
              arr[l+1] = temp;
              if (arr[l] > arr[ir]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l+1] > arr[ir]) {
                temp = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l] > arr[l+1]) {
                temp = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = temp;
              }
              i=l+1;
              j=ir;
              a=arr[l+1];
              for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
              arr[l+1]=arr[j];
              arr[j]=a;
              if (j >= k) ir=j-1;
              if (j <= k) l=i;
            }
          }
        }

        __device__ int quickselect_int(int* arr, int n, int k) {
          int i,ir,j,l,mid;
          int a,temp;

          l = 0;
          ir=n-1;
          for(;;) {
            if (ir <= l+1) {
              if (ir == l+1 && arr[ir] < arr[l]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              return arr[k];
            }
            else {
              mid=(l+ir) >> 1;
              temp = arr[mid];
              arr[mid] = arr[l+1];
              arr[l+1] = temp;
              if (arr[l] > arr[ir]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l+1] > arr[ir]) {
                temp = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l] > arr[l+1]) {
                temp = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = temp;
              }
              i=l+1;
              j=ir;
              a=arr[l+1];
              for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
              arr[l+1]=arr[j];
              arr[j]=a;
              if (j >= k) ir=j-1;
              if (j <= k) l=i;
            }
          }
        }

        __device__ long quickselect_long(long* arr, int n, int k) {
          int i,ir,j,l,mid;
          long a,temp;

          l = 0;
          ir=n-1;
          for(;;) {
            if (ir <= l+1) {
              if (ir == l+1 && arr[ir] < arr[l]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              return arr[k];
            }
            else {
              mid=(l+ir) >> 1;
              temp = arr[mid];
              arr[mid] = arr[l+1];
              arr[l+1] = temp;
              if (arr[l] > arr[ir]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l+1] > arr[ir]) {
                temp = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l] > arr[l+1]) {
                temp = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = temp;
              }
              i=l+1;
              j=ir;
              a=arr[l+1];
              for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
              arr[l+1]=arr[j];
              arr[j]=a;
              if (j >= k) ir=j-1;
              if (j <= k) l=i;
            }
          }
        }

        __device__ double quickselect_double(double* arr, int n, int k) {
          int i,ir,j,l,mid;
          double a,temp;

          l = 0;
          ir=n-1;
          for(;;) {
            if (ir <= l+1) {
              if (ir == l+1 && arr[ir] < arr[l]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              return arr[k];
            }
            else {
              mid=(l+ir) >> 1;
              temp = arr[mid];
              arr[mid] = arr[l+1];
              arr[l+1] = temp;
              if (arr[l] > arr[ir]) {
                temp = arr[l];
                arr[l] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l+1] > arr[ir]) {
                temp = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = temp;
              }
              if (arr[l] > arr[l+1]) {
                temp = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = temp;
              }
              i=l+1;
              j=ir;
              a=arr[l+1];
              for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
              arr[l+1]=arr[j];
              arr[j]=a;
              if (j >= k) ir=j-1;
              if (j <= k) l=i;
            }
          }
        }

        __device__ float gpumean_float(float *arr, int n) {
          float sum = 0;
          for (int j = 0; j < n; j++) sum += arr[j];
          sum/=n;
          return sum;
        }

        __device__ float gpumean_int(int *arr, int n) {
          float sum = 0;
          for (int j = 0; j < n; j++) sum += arr[j];
          sum/=n;
          return sum;
        }

        __device__ double gpumean_double(double *arr, int n) {
          double sum = 0;
          for (int j = 0; j < n; j++) sum += arr[j];
          sum/=n;
          return sum;
        }

        __device__ float gpumean_long(long *arr, int n) {
          float sum = 0;
          for (int j = 0; j < n; j++) sum += arr[j];
          sum/=n;
          return sum;
        }

        __device__ float gpustd_float(float *arr, int n, float m) {
          float var = 0;
          for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
          float sd = sqrt(var/(n-1));
          return sd;
        }

        __device__ float gpustd_int(int *arr, int n, float m) {
          float var = 0;
          for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
          float sd = sqrt(var/(n-1));
          return sd;
        }

        __device__ double gpustd_double(double *arr, int n, double m) {
          double var = 0;
          for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
          double sd = sqrt(var/(n-1));
          return sd;
        }

        __device__ float gpustd_long(long *arr, int n, float m) {
          float var = 0;
          for (int j = 0; j < n; j++) var += (arr[j]-m)*(arr[j]-m);
          float sd = sqrt(var/(n-1));
          return sd;
        }

        __global__ void median3dX_float(float *data, float *output, int elem, int depth, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= elem) return;
            int k;
            int n = depth;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*depth;
              int n0 = i*depth;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] != 0) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < depth; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            float temp[10];
            for (int j = 0; j < depth; j++) {
              temp[j] = data[i+j*elem];
            }
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_float(&temp[0], n, k);
            } else {
              output[i] = (quickselect_float(&temp[0], n, k) + quickselect_float(&temp[0], n, k-1))/2;
            }
          }

        __global__ void median2d_float(float *data, float *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_float(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_int(int *data, float *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_int(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_long(long *data, float *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_long(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_double(double *data, double *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_double(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_float_w(float *data, float *output, float *w, float *wmap, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) wmap[i] += w[j];
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_float(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_int_w(int *data, float *output, int *w, int *wmap, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) wmap[i] += w[j];
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_int(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_long_w(long *data, float *output, long *w, long *wmap, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) wmap[i] += w[j];
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_long(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_double_w(double *data, double *output, double *w, double *wmap, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero, int even)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                    wmap[i] += w[j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) wmap[i] += w[j];
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = (n-nhigh-nlow)/2+nlow;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_double(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_float_sigclip(float *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_float(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_float(&data[i*cols], n, k);
                } else {
                  m = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_float(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_float(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_int_sigclip(int *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_int(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_int(&data[i*cols], n, k);
                } else {
                  m = (quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_int(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_int(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_double_sigclip(double *data, double *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            double m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_double(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_double(&data[i*cols], n, k);
                } else {
                  m = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_double(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_double(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_long_sigclip(long *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_long(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_long(&data[i*cols], n, k);
                } else {
                  m = (quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_long(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_long(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_float_sigclip_w(float *data, float *output, float *w, float *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_float(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_float(&data[i*cols], n, k);
                } else {
                  m = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_float(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) wmap[i] += w[i*cols+j];
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_float(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_int_sigclip_w(int *data, float *output, int *w, int *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_int(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_int(&data[i*cols], n, k);
                } else {
                  m = (quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_int(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) wmap[i] += w[i*cols+j];
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_int(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_double_sigclip_w(double *data, double *output, double *w, double *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            double m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_double(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_double(&data[i*cols], n, k);
                } else {
                  m = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_double(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) wmap[i] += w[i*cols+j];
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = quickselect_double(&data[i*cols], n, k);
            } else {
              output[i] = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void median2d_long_sigclip_w(long *data, float *output, long *w, long *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int even, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              int n0 = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_long(&data[i*cols], n);
              } else {
                if (n % 2 == 1 || !even) {
                  m = quickselect_long(&data[i*cols], n, k);
                } else {
                  m = (quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_long(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) wmap[i] += w[i*cols+j];
            if (n == 0) {
              output[i] = 0;
              return;
            }
            k = n/2;
            if (n % 2 == 1 || !even) {
              output[i] = (float)quickselect_long(&data[i*cols], n, k);
            } else {
              output[i] = (float)(quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
            }
          }

        __global__ void transpose_float(float *data, float *output, int rows, int cols)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            int row = i/cols;
            int col = i % cols;
            output[col*rows+row] = data[i];
          }

        __global__ void transpose_int(int *data, int *output, int rows, int cols)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            int row = i/cols;
            int col = i % cols;
            output[col*rows+row] = data[i];
          }

        __global__ void transpose_long(long *data, long *output, int rows, int cols)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            int row = i/cols;
            int col = i % cols;
            output[col*rows+row] = data[i];
          }

        __global__ void transpose_double(double *data, double *output, int rows, int cols)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            int row = i/cols;
            int col = i % cols;
            output[col*rows+row] = data[i];
          }

        __global__ void transpose3d_int(int *data, int *output, int rows, int cols, int depth, int mode)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols*depth) return;
            int row = i/(cols*depth);
            int col = (i/depth) % cols;
            int d = i % depth;
            if (mode == 0) output[col*rows*depth+row*depth+d] = data[i];
            else if (mode == 1) output[d*rows*cols+col*rows+row] = data[i];
            else if (mode == 2) output[row*cols*depth+d*cols+col] = data[i];
          }

        __global__ void transpose3d_float(float *data, float *output, int rows, int cols, int depth, int mode)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols*depth) return;
            int row = i/(cols*depth);
            int col = (i/depth) % cols;
            int d = i % depth;
            if (mode == 0) output[col*rows*depth+row*depth+d] = data[i];
            else if (mode == 1) output[d*rows*cols+col*rows+row] = data[i];
            else if (mode == 2) output[row*cols*depth+d*cols+col] = data[i];
          }

        __global__ void transpose3d_long(long *data, long *output, int rows, int cols, int depth, int mode)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols*depth) return;
            int row = i/(cols*depth);
            int col = (i/depth) % cols;
            int d = i % depth;
            if (mode == 0) output[col*rows*depth+row*depth+d] = data[i];
            else if (mode == 1) output[d*rows*cols+col*rows+row] = data[i];
            else if (mode == 2) output[row*cols*depth+d*cols+col] = data[i];
          }

        __global__ void transpose3d_double(double *data, double *output, int rows, int cols, int depth, int mode)
          {
            const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            //const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols*depth) return;
            int row = i/(cols*depth);
            int col = (i/depth) % cols;
            int d = i % depth;
            if (mode == 0) output[col*rows*depth+row*depth+d] = data[i];
            else if (mode == 1) output[d*rows*cols+col*rows+row] = data[i];
            else if (mode == 2) output[row*cols*depth+d*cols+col] = data[i];
          }

        __global__ void medianfilter_int(int *data, int *output, int n, int algorithm, int boxsize, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= n) return;
            int temp[51];
            int k = boxsize/2;
            int n0 = i-k;
            if (i < k+1) {
              n0 = 0;
            } else if (i >= n-(k+1)) {
              n0 = n-boxsize;
            }
            if (nonzero) {
              int j = 0;
              int idx = 0;
              while ((j < boxsize || idx < minpts) && (j+n0 < n)) {
                if (data[n0+j] != 0) {
                  temp[idx] = data[n0+j];
                  idx++;
                }
                j++;
              }
              if (idx == 0) {
                //special case
                idx++;
                temp[0] = 0;
              }
              boxsize = idx;
              k = boxsize/2;
            } else {
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j];
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_int(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_int(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_int(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter_float(float *data, float *output, int n, int algorithm, int boxsize, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= n) return;
            float temp[51];
            int k = boxsize/2;
            int n0 = i-k;
            if (i < k+1) {
              n0 = 0;
            } else if (i >= n-(k+1)) {
              n0 = n-boxsize;
            }
            if (nonzero) {
              int j = 0;
              int idx = 0;
              while ((j < boxsize || idx < minpts) && (j+n0 < n)) {
                if (data[n0+j] != 0) {
                  temp[idx] = data[n0+j];
                  idx++;
                }
                j++;
              }
              if (idx == 0) {
                //special case
                idx++;
                temp[0] = 0;
              }
              boxsize = idx;
              k = boxsize/2;
            } else {
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j];
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_float(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_float(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_float(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter_long(long *data, long *output, int n, int algorithm, int boxsize, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= n) return;
            long temp[51];
            int k = boxsize/2;
            int n0 = i-k;
            if (i < k+1) {
              n0 = 0;
            } else if (i >= n-(k+1)) {
              n0 = n-boxsize;
            }
            if (nonzero) {
              int j = 0;
              int idx = 0;
              while ((j < boxsize || idx < minpts) && (j+n0 < n)) {
                if (data[n0+j] != 0) {
                  temp[idx] = data[n0+j];
                  idx++;
                }
                j++;
              }
              if (idx == 0) {
                //special case
                idx++;
                temp[0] = 0;
              }
              boxsize = idx;
              k = boxsize/2;
            } else {
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j];
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_long(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_long(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_long(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter_double(double *data, double *output, int n, int algorithm, int boxsize, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= n) return;
            double temp[51];
            int k = boxsize/2;
            int n0 = i-k;
            if (i < k+1) {
              n0 = 0;
            } else if (i >= n-(k+1)) {
              n0 = n-boxsize;
            }
            if (nonzero) {
              int j = 0;
              int idx = 0;
              while ((j < boxsize || idx < minpts) && (j+n0 < n)) {
                if (data[n0+j] != 0) {
                  temp[idx] = data[n0+j];
                  idx++;
                }
                j++;
              }
              if (idx == 0) {
                //special case
                idx++;
                temp[0] = 0;
              }
              boxsize = idx;
              k = boxsize/2;
            } else {
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j];
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_double(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_double(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_double(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter2d_int(int *data, int *output, int rows, int cols, int algorithm, int boxsize, int direction, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            int temp[51];
            int k = boxsize/2;
            int n0 = i-k;

            if (direction == 1) {
              int row = i / cols;
              n0 = i-k*cols;
              if (row < k+1) {
                n0 = i-row*cols;
              } else if (row >= rows-(k+1)) {
                n0 = i-row*cols+(rows-boxsize)*cols;
              }
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j*cols];
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0/cols < rows)) {
                  if (data[n0+j*cols] != 0) {
                    temp[idx] = data[n0+j*cols];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j*cols];
                }
              }
            } else {
              int col = i % cols;
              if (col < k+1) {
                n0 = i-col;
              } else if (col >= cols-(k+1)) {
                n0 = i-col+cols-boxsize;
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0%cols < cols)) {
                  if (data[n0+j] != 0) {
                    temp[idx] = data[n0+j];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j];
                }
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_int(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_int(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_int(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter2d_float(float *data, float *output, int rows, int cols, int algorithm, int boxsize, int direction, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            float temp[51];
            int k = boxsize/2;
            int n0 = i-k;

            if (direction == 1) {
              int row = i / cols;
              n0 = i-k*cols;
              if (row < k+1) {
                n0 = i-row*cols;
              } else if (row >= rows-(k+1)) {
                n0 = i-row*cols+(rows-boxsize)*cols;
              }
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j*cols];
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0/cols < rows)) {
                  if (data[n0+j*cols] != 0) {
                    temp[idx] = data[n0+j*cols];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j*cols];
                }
              }
            } else {
              int col = i % cols;
              if (col < k+1) {
                n0 = i-col;
              } else if (col >= cols-(k+1)) {
                n0 = i-col+cols-boxsize;
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0%cols < cols)) {
                  if (data[n0+j] != 0) {
                    temp[idx] = data[n0+j];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j];
                }
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_float(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_float(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_float(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter2d_long(long *data, long *output, int rows, int cols, int algorithm, int boxsize, int direction, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            long temp[51];
            int k = boxsize/2;
            int n0 = i-k;

            if (direction == 1) {
              int row = i / cols;
              n0 = i-k*cols;
              if (row < k+1) {
                n0 = i-row*cols;
              } else if (row >= rows-(k+1)) {
                n0 = i-row*cols+(rows-boxsize)*cols;
              }
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j*cols];
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0/cols < rows)) {
                  if (data[n0+j*cols] != 0) {
                    temp[idx] = data[n0+j*cols];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j*cols];
                }
              }
            } else {
              int col = i % cols;
              if (col < k+1) {
                n0 = i-col;
              } else if (col >= cols-(k+1)) {
                n0 = i-col+cols-boxsize;
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0%cols < cols)) {
                  if (data[n0+j] != 0) {
                    temp[idx] = data[n0+j];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j];
                }
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_long(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_long(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_long(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void medianfilter2d_double(double *data, double *output, int rows, int cols, int algorithm, int boxsize, int direction, int nonzero, int minpts, int nhigh, int nlow) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows*cols) return;
            double temp[51];
            int k = boxsize/2;
            int n0 = i-k;

            if (direction == 1) {
              int row = i / cols;
              n0 = i-k*cols;
              if (row < k+1) {
                n0 = i-row*cols;
              } else if (row >= rows-(k+1)) {
                n0 = i-row*cols+(rows-boxsize)*cols;
              }
              for (int j = 0; j < boxsize; j++) {
                temp[j] = data[n0+j*cols];
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0/cols < rows)) {
                  if (data[n0+j*cols] != 0) {
                    temp[idx] = data[n0+j*cols];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j*cols];
                }
              }
            } else {
              int col = i % cols;
              if (col < k+1) {
                n0 = i-col;
              } else if (col >= cols-(k+1)) {
                n0 = i-col+cols-boxsize;
              }
              if (nonzero) {
                int j = 0;
                int idx = 0;
                while ((j < boxsize || idx < minpts) && (j + n0%cols < cols)) {
                  if (data[n0+j] != 0) {
                    temp[idx] = data[n0+j];
                    idx++;
                  }
                  j++;
                }
                if (idx == 0) {
                  //special case
                  idx++;
                  temp[0] = 0;
                }
                boxsize = idx;
                k = boxsize/2;
              } else {
                for (int j = 0; j < boxsize; j++) {
                  temp[j] = data[n0+j];
                }
              }
            }
            if (nhigh > 0 || nlow > 0) k = (boxsize-nhigh-nlow)/2+nlow;
            if (algorithm == 0) {
              //Quick select
              output[i] = data[i]-quickselect_double(temp, boxsize, k);
            } else if (algorithm == 1) {
              //Quick sort
              quicksort_double(temp, 0, boxsize-1);
              output[i] = data[i] - temp[k];
            } else if (algorithm == 2) {
              //Bubble sort
              bubblesort_double(temp, boxsize);
              output[i] = data[i] - temp[k];
            }
          }

        __global__ void mean2d_float(float *data, float *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            float divisor = (float)cols;
            if (nlow > 0 || nhigh > 0) {
              quicksort_float(&data[n0], 0, cols-1);
              if (nlow > 0) n0 += nlow;
              if (nhigh > 0) n_end -= nhigh;
              divisor -= (nlow+nhigh);
            }
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_int(int *data, float *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            float divisor = (float)cols;
            if (nlow > 0 || nhigh > 0) {
              quicksort_int(&data[n0], 0, cols-1);
              if (nlow > 0) n0 += nlow;
              if (nhigh > 0) n_end -= nhigh;
              divisor -= (nlow+nhigh);
            }
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_double(double *data, double *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            double sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            double divisor = (double)cols;
            if (nlow > 0 || nhigh > 0) {
              quicksort_double(&data[n0], 0, cols-1);
              if (nlow > 0) n0 += nlow;
              if (nhigh > 0) n_end -= nhigh;
              divisor -= (nlow+nhigh);
            }
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_long(long *data, float *output, int rows, int cols, float lthresh, float hthresh, int nlow, int nhigh, int nonzero)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            float divisor = (float)cols;
            if (nlow > 0 || nhigh > 0) {
              quicksort_long(&data[n0], 0, cols-1);
              if (nlow > 0) n0 += nlow;
              if (nhigh > 0) n_end -= nhigh;
              divisor -= (nlow+nhigh);
            }
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor--;
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }


        __global__ void wmean2d_float(float *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, float *weights, float divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_int(int *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, float *weights, float divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_double(double *data, double *output, int rows, int cols, float lthresh, float hthresh, int nonzero, double *weights, double divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            double sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_long(long *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, float *weights, float divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) sum += data[j]; else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) sum += data[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_float_w(float *data, float *output, float *w, float *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, float *weights, float divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) {
                sum += data[j];
                wmap[i] += w[j-n0];
              }
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_int_w(int *data, float *output, int *w, int *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, float *weights, float divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) {
                sum += data[j];
                wmap[i] += w[j-n0];
              }
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_double_w(double *data, double *output, double *w, double *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, double *weights, double divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            double sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) {
                sum += data[j];
                wmap[i] += w[j-n0];
              }
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_long_w(long *data, float *output, long *w, long *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, float *weights, float divisor)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            float sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            wmap[i] = 0;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    sum += data[j];
                    wmap[i] += w[j-n0];
                  } else divisor-=weights[j-n0];
                }
              }
            } else {
              for (int j = n0; j < n_end; j++) {
                sum += data[j];
                wmap[i] += w[j-n0];
              }
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_float_sigclip(float *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_float(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_float(&data[i*cols], n, k);
                } else {
                  m = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_float(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
            }
            divisor = (float)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_int_sigclip(int *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_int(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_int(&data[i*cols], n, k);
                } else {
                  m = (quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_int(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
            }
            divisor = (float)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_double_sigclip(double *data, double *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            double m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_double(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_double(&data[i*cols], n, k);
                } else {
                  m = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_double(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
            }
            divisor = (double)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_long_sigclip(long *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_long(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_long(&data[i*cols], n, k);
                } else {
                  m = (quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_long(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
            }
            divisor = (float)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_float_sigclip(float *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, float *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            float weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_float(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_float(&data[i*cols], n, k);
                } else {
                  m = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_float(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_int_sigclip(int *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, float *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            float weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_int(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_int(&data[i*cols], n, k);
                } else {
                  m = (quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_int(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_double_sigclip(double *data, double *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, double *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            double m = 0, sd = 0, divisor = 0, sum = 0;
            double weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_double(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_double(&data[i*cols], n, k);
                } else {
                  m = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_double(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_long_sigclip(long *data, float *output, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, float *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            float weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_long(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_long(&data[i*cols], n, k);
                } else {
                  m = (quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_long(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_float_sigclip_w(float *data, float *output, float *w, float *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_float(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_float(&data[i*cols], n, k);
                } else {
                  m = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_float(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  w[n] = w[j];
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
              wmap[i] += w[j];
            }
            divisor = (float)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_int_sigclip_w(int *data, float *output, int *w, int *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_int(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_int(&data[i*cols], n, k);
                } else {
                  m = (quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_int(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  w[n] = w[j];
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
              wmap[i] += w[j];
            }
            divisor = (float)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_double_sigclip_w(double *data, double *output, double *w, double *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            double m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_double(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_double(&data[i*cols], n, k);
                } else {
                  m = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_double(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  w[n] = w[j];
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
              wmap[i] += w[j];
            }
            divisor = (double)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void mean2d_long_sigclip_w(long *data, float *output, long *w, long *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            int n0 = i*cols;
            int n_end = n0+cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = n0; j < n_end; j++) {
                  if (data[j] != 0 && data[j] >= lthresh && data[j] <= hthresh) {
                    w[n] = w[j];
                    data[n++] = data[j];
                  }
                }
              }
              n -= n0;
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_long(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_long(&data[i*cols], n, k);
                } else {
                  m = (quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_long(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = n0; j < n0+nold; j++) {
                if (data[j] >= lthresh && data[j] <= hthresh) {
                  w[n] = w[j];
                  data[n++] = data[j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = n0; j < n0+n; j++) {
              sum += data[j];
              wmap[i] += w[j];
            }
            divisor = (float)n;
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_float_sigclip_w(float *data, float *output, float *w, float *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, float *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            float weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_float(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_float(&data[i*cols], n, k);
                } else {
                  m = (quickselect_float(&data[i*cols], n, k) + quickselect_float(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_float(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
              wmap[i] += w[j+n0];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_int_sigclip_w(int *data, float *output, int *w, int *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, float *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            float weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_int(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_int(&data[i*cols], n, k);
                } else {
                  m = (quickselect_int(&data[i*cols], n, k) + quickselect_int(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_int(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
              wmap[i] += w[j+n0];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_double_sigclip_w(double *data, double *output, double *w, double *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, double *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            double m = 0, sd = 0, divisor = 0, sum = 0;
            double weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_double(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_double(&data[i*cols], n, k);
                } else {
                  m = (quickselect_double(&data[i*cols], n, k) + quickselect_double(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_double(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
              wmap[i] += w[j+n0];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

        __global__ void wmean2d_long_sigclip_w(long *data, float *output, long *w, long *wmap, int rows, int cols, float lthresh, float hthresh, int nonzero, int niter, float lsigma, float hsigma, int mclip, float *weights)
          {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= rows) return;
            int k;
            int n = cols;
            int iter = 0;
            float m = 0, sd = 0, divisor = 0, sum = 0;
            float weights_i[4096];
            int n0 = i*cols;
            if (nonzero || lthresh != INT_MIN || hthresh != INT_MAX) {
              n = i*cols;
              if (nonzero && lthresh == INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh == INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh == INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (!nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              } else if (nonzero && lthresh != INT_MIN && hthresh != INT_MAX) {
                for (int j = 0; j < cols; j++) {
                  if (data[n0+j] != 0 && data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                    w[n] = w[n0+j];
                    weights_i[n-n0] = weights[j];
                    data[n++] = data[n0+j];
                  }
                }
              }
              n -= n0;
            } else {
              for (int j = 0; j < cols; j++) weights_i[j] = weights[j];
            }
            int nold = n+1;
            while (nold > n && n > 2 && iter < niter) {
              iter++;
              k = n/2;
              if (mclip == 0) {
                //mean
                m = gpumean_long(&data[i*cols], n);
              } else {
                if (n % 2 == 1) {
                  m = quickselect_long(&data[i*cols], n, k);
                } else {
                  m = (quickselect_long(&data[i*cols], n, k) + quickselect_long(&data[i*cols], n, k-1))/2;
                }
              }
              sd = gpustd_long(&data[i*cols], n, m);
              int n0 = i*cols;
              nold = n;
              n = i*cols;
              lthresh = m-lsigma*sd;
              hthresh = m+hsigma*sd;
              for (int j = 0; j < nold; j++) {
                if (data[n0+j] >= lthresh && data[n0+j] <= hthresh) {
                  w[n] = w[n0+j];
                  weights_i[n-n0] = weights_i[j];
                  data[n++] = data[n0+j];
                }
              }
              n -= n0;
            }
            wmap[i] = 0;
            for (int j = 0; j < n; j++) {
              sum += data[j+n0];
              divisor += weights_i[j];
              wmap[i] += w[j+n0];
            }
            if (divisor == 0) {
              output[i] = 0;
              return;
            }
            output[i] = sum/divisor;
          }

         """)

        except Exception as ex:
            print("gpu_arraymedian> WARNING: CUDA median libs not installed!")
            print(ex)
    return mod
#end get_mod()

mod = get_mod()

INT_MIN = -2**31
INT_MAX = 2**31-1
ALGORITHM_QUICKSELECT = 0
ALGORITHM_QUICKSORT = 1
ALGORITHM_BUBBLESORT = 2

def gputranspose(data):
    if (data.size < 2048):
        return ascontiguousarray(data.transpose())
    rows = data.shape[0]
    cols = data.shape[1]
    blocks = data.size//block_size
    if (data.size % block_size != 0):
        blocks += 1
    blocky = 1
    while(blocks > 65535):
        blocky *= 2
        blocks //= 2
    outtype = data.dtype
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (data.dtype == float32):
        transpose = mod.get_function("transpose_float")
    elif (data.dtype == int32):
        transpose = mod.get_function("transpose_int")
    elif (data.dtype == int64):
        transpose = mod.get_function("transpose_long")
    elif (data.dtype == float64):
        transpose = mod.get_function("transpose_double")
    else:
        print("Invalid datatype: ", data.dtype)
        return None
    output = empty((cols, rows), outtype)
    transpose(drv.In(data), drv.Out(output), int32(rows), int32(cols), grid=(blocks,blocky), block=(block_size,1,1))
    return output

def gputranspose3d(data, axis1, axis2):
    if (len(data.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    if (data.size < 2048):
        if (axis1 == 0 and axis2 == 1):
            return ascontiguousarray(swapaxes(data, 0, 1))
        elif (axis1 == 0 and axis2 == 2):
            return ascontiguousarray(swapaxes(data, 0, 2))
        elif (axis1 == 1 and axis2 == 2):
            return ascontiguousarray(swapaxes(data, 1, 2))
    rows = data.shape[0]
    cols = data.shape[1]
    depth = data.shape[2]
    blocks = data.size//block_size
    if (data.size % block_size != 0):
        blocks += 1
    blocky = 1
    while(blocks > 65535):
        blocky *= 2
        blocks //= 2
    outtype = data.dtype
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (data.dtype == float32):
        transpose = mod.get_function("transpose3d_float")
    elif (data.dtype == int32):
        transpose = mod.get_function("transpose3d_int")
    elif (data.dtype == int64):
        transpose = mod.get_function("transpose3d_long")
    elif (data.dtype == float64):
        transpose = mod.get_function("transpose3d_double")
    else:
        print("Invalid datatype: ", data.dtype)
        return None
    if (axis1 == 0 and axis2 == 1):
        output = empty((cols, rows, depth), outtype)
        transpose(drv.In(data), drv.Out(output), int32(rows), int32(cols), int32(depth), int32(0), grid=(blocks,blocky), block=(block_size,1,1))
    elif (axis1 == 0 and axis2 == 2):
        output = empty((depth, cols, rows), outtype)
        transpose(drv.In(data), drv.Out(output), int32(rows), int32(cols), int32(depth), int32(1), grid=(blocks,blocky), block=(block_size,1,1))
    elif (axis1 == 1 and axis2 == 2):
        output = empty((rows, depth, cols), outtype)
        transpose(drv.In(data), drv.Out(output), int32(rows), int32(cols), int32(depth), int32(2), grid=(blocks,blocky), block=(block_size,1,1))
    else:
        print("Invalid axes!")
        return None
    return output

def gpumedianfilter(data, algorithm=ALGORITHM_BUBBLESORT, boxsize=51, nonzero=False, minpts=0, nhigh=0, nlow=0):
    if (len(data.shape) != 1):
        print("Invalid shape!  Must be 1-d array!")
        return None
    if (boxsize > 51):
        print("Using maximum boxsize of 51!")
        boxsize = 51
    elif (boxsize % 2 == 0):
        boxsize += 1
        print("Boxsize must be odd!  Using "+str(boxsize))
    n = data.size
    block_size = 512
    if (n < 2048):
        block_size = 32
    blocks = n//block_size
    if (n % block_size != 0):
        blocks += 1
    outtype = data.dtype
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (data.dtype == float32):
        medianfilter = mod.get_function("medianfilter_float")
    elif (data.dtype == int32):
        medianfilter = mod.get_function("medianfilter_int")
    elif (data.dtype == int64):
        medianfilter = mod.get_function("medianfilter_long")
    elif (data.dtype == float64):
        medianfilter = mod.get_function("medianfilter_double")
    else:
        print("Invalid datatype: ", data.dtype)
        return None
    output = empty(n, outtype)
    medianfilter(drv.In(data), drv.Out(output), int32(n), int32(algorithm), int32(boxsize), int32(nonzero), int32(minpts), int32(nhigh), int32(nlow), grid=(blocks,1), block=(block_size,1,1))
    return output
#end gpumedianfilter

def gpumedianfilter2d(data, algorithm=ALGORITHM_BUBBLESORT, boxsize=51, axis="X", nonzero=False, minpts=0, nhigh=0, nlow=0):
    if (len(data.shape) != 2):
        print("Invalid shape!  Must be 2-d array!")
        return None
    if (boxsize > 51):
        print("Using maximum boxsize of 51!")
        boxsize = 51
    elif (boxsize % 2 == 0):
        boxsize += 1
        print("Boxsize must be odd!  Using "+str(boxsize))
    direction = 0
    if (axis == "Y"):
        direction = 1
    rows = data.shape[0]
    cols = data.shape[1]
    blocks = data.size//block_size
    if (data.size % block_size != 0):
        blocks += 1
    outtype = data.dtype
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (data.dtype == float32):
        medianfilter2d = mod.get_function("medianfilter2d_float")
    elif (data.dtype == int32):
        medianfilter2d = mod.get_function("medianfilter2d_int")
    elif (data.dtype == int64):
        medianfilter2d = mod.get_function("medianfilter2d_long")
    elif (data.dtype == float64):
        medianfilter2d = mod.get_function("medianfilter2d_double")
    else:
        print("Invalid datatype: ", data.dtype)
        return None
    output = empty((rows,cols), outtype)
    medianfilter2d(drv.In(data), drv.Out(output), int32(rows), int32(cols), int32(algorithm), int32(boxsize), int32(direction), int32(nonzero), int32(minpts), int32(nhigh), int32(nlow), grid=(blocks,1), block=(block_size,1,1))
    return output
#end gpumedianfilter2d

def gpumedianS(data, slitmask, nslits, nonzero=False, even=False, kernel=None, trans=False):
    if (kernel is None):
        kernel = fatboycudalib.gpumedianS
    #Return array containing median values of each slitlet
    medians = zeros(nslits, dtype=float32)
    if (trans):
        #Transpose data first
        kernel(gputranspose(float32(data)), slitmask=gputranspose(int32(slitmask)), medians=medians, nslits=nslits, nonzero=nonzero, even=even)
    else:
        kernel(float32(data), slitmask=int32(slitmask), medians=medians, nslits=nslits, nonzero=nonzero, even=even)
    return medians
#end gpumedianS

def gpumedian3d(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nlow=0, nhigh=0, nonzero=False, even=False, axis="Z"):
    if (len(input.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    rows = input.shape[0]
    cols = input.shape[1]
    depth = input.shape[2]
    outtype = float32
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (input.dtype == float32):
        median2d = mod.get_function("median2d_float")
    elif (input.dtype == int32):
        median2d = mod.get_function("median2d_int")
    elif (input.dtype == int64):
        median2d = mod.get_function("median2d_long")
    elif (input.dtype == float64):
        median2d = mod.get_function("median2d_double")
        outtype = float64
    else:
        print("Invalid datatype: ", input.dtype)
        return None
    if (axis == "X"):
        median2d = mod.get_function("median3dX_float")
    block_size = 512
    if (rows*cols <= 2048):
        block_size = 32
    blocks = rows*cols//block_size
    if (rows*cols % block_size != 0):
        blocks += 1
    medVals = empty((rows,cols), outtype)
    median2d(drv.In(input), drv.Out(medVals), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), int32(even), grid=(blocks,1), block=(block_size,1,1))
    return medVals

def gpumedian3d_w(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nlow=0, nhigh=0, nonzero=False, even=False, w=None, wmap=None):
    if (len(input.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    rows = input.shape[0]
    cols = input.shape[1]
    depth = input.shape[2]
    outtype = float32
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (input.dtype == float32):
        median2d_w = mod.get_function("median2d_float_w")
    elif (input.dtype == int32):
        median2d_w = mod.get_function("median2d_int_w")
    elif (input.dtype == int64):
        median2d_w = mod.get_function("median2d_long_w")
    elif (input.dtype == float64):
        median2d_w = mod.get_function("median2d_double_w")
        outtype = float64
    else:
        print("Invalid datatype: ", input.dtype)
        return None
    block_size = 512
    if (rows*cols <= 2048):
        block_size = 32
    blocks = rows*cols//block_size
    if (rows*cols % block_size != 0):
        blocks += 1
    medVals = empty((rows,cols), outtype)
    median2d_w(drv.In(input), drv.Out(medVals), drv.In(w), drv.Out(wmap), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), int32(even), grid=(blocks,1), block=(block_size,1,1))
    return medVals

def gpumedian3d_sigclip(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nonzero=False, even=False, w=None, wmap=None, niter=5, lsigma=3, hsigma=3, mclip=0):
    if (len(input.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    if (isinstance(mclip, str)):
        if (mclip == 'median' or mclip == 'med'):
            mclip = 1
        else:
            mclip = 0
    if (mclip != 0 and mclip != 1):
        print("Invalid mclip: "+str(mclip)+".  Using mean (mclip = 0).")
        mclip = 0
    rows = input.shape[0]
    cols = input.shape[1]
    depth = input.shape[2]
    outtype = float32
    medVals = empty((rows,cols), outtype)
    block_size = 512
    if (rows*cols <= 2048):
        block_size = 32
    blocks = rows*cols//block_size
    if (rows*cols % block_size != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (w is None and wmap is None):
        if (input.dtype == float32):
            median2d_sig = mod.get_function("median2d_float_sigclip")
        elif (input.dtype == int32):
            median2d_sig = mod.get_function("median2d_int_sigclip")
        elif (input.dtype == int64):
            median2d_sig = mod.get_function("median2d_long_sigclip")
        elif (input.dtype == float64):
            median2d_sig = mod.get_function("median2d_double_sigclip")
            outtype = float64
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        median2d_sig(drv.In(input), drv.Out(medVals), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(even), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    else:
        if (input.dtype == float32):
            median2d_sig = mod.get_function("median2d_float_sigclip_w")
        elif (input.dtype == int32):
            median2d_sig = mod.get_function("median2d_int_sigclip_w")
        elif (input.dtype == int64):
            median2d_sig = mod.get_function("median2d_long_sigclip_w")
        elif (input.dtype == float64):
            median2d_sig = mod.get_function("median2d_double_sigclip_w")
            outtype = float64
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        if (len(w.shape) != 3):
            w2 = empty(input.shape, input.dtype)
            w2[:,:] = w
            w = w2
        median2d_sig(drv.In(input), drv.Out(medVals), drv.In(w), drv.Out(wmap), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(even), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    return medVals

def gpumedian2d(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nlow=0, nhigh=0, nonzero=False, even=False):
    rows = input.shape[0]
    cols = input.shape[1]
    outtype = float32
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (input.dtype == float32):
        median2d = mod.get_function("median2d_float")
    elif (input.dtype == int32):
        median2d = mod.get_function("median2d_int")
    elif (input.dtype == int64):
        median2d = mod.get_function("median2d_long")
    elif (input.dtype == float64):
        median2d = mod.get_function("median2d_double")
        outtype = float64
    else:
        print("Invalid datatype: ", input.dtype)
        return None
    block_size = 512
    if (rows <= 2048):
        block_size = 32
    blocks = rows//block_size
    if (rows % block_size != 0):
        blocks += 1
    medVals = empty(rows, outtype)
    median2d(drv.In(input), drv.Out(medVals), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), int32(even), grid=(blocks,1), block=(block_size,1,1))
    return medVals

def gpumedian2d_w(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nlow=0, nhigh=0, nonzero=False, even=False, w=None, wmap=None):
    rows = input.shape[0]
    cols = input.shape[1]
    outtype = float32
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (input.dtype == float32):
        median2d_w = mod.get_function("median2d_float_w")
    elif (input.dtype == int32):
        median2d_w = mod.get_function("median2d_int_w")
    elif (input.dtype == int64):
        median2d_w = mod.get_function("median2d_long_w")
    elif (input.dtype == float64):
        median2d_w = mod.get_function("median2d_double_w")
        outtype = float64
    else:
        print("Invalid datatype: ", input.dtype)
        return None
    block_size = 512
    if (rows <= 2048):
        block_size = 32
    blocks = rows//block_size
    if (rows % block_size != 0):
        blocks += 1
    medVals = empty(rows, outtype)
    median2d_w(drv.In(input), drv.Out(medVals), drv.In(w), drv.Out(wmap), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), int32(even), grid=(blocks,1), block=(block_size,1,1))
    return medVals

def gpumedian2d_sigclip(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nonzero=False, even=False, w=None, wmap=None, niter=5, lsigma=3, hsigma=3, mclip=0):
    if (len(input.shape) != 2):
        print("Invalid shape!  Must be 2-d array!")
        return None
    if (isinstance(mclip, str)):
        if (mclip == 'median' or mclip == 'med'):
            mclip = 1
        else:
            mclip = 0
    if (mclip != 0 and mclip != 1):
        print("Invalid mclip: "+str(mclip)+".  Using mean (mclip = 0).")
        mclip = 0
    rows = input.shape[0]
    cols = input.shape[1]
    outtype = float32
    medVals = empty(rows, outtype)
    block_size = 512
    if (rows <= 2048):
        block_size = 32
    blocks = rows//block_size
    if (rows % block_size != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (w is None and wmap is None):
        if (input.dtype == float32):
            median2d_sig = mod.get_function("median2d_float_sigclip")
        elif (input.dtype == int32):
            median2d_sig = mod.get_function("median2d_int_sigclip")
        elif (input.dtype == int64):
            median2d_sig = mod.get_function("median2d_long_sigclip")
        elif (input.dtype == float64):
            median2d_sig = mod.get_function("median2d_double_sigclip")
            outtype = float64
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        median2d_sig(drv.In(input), drv.Out(medVals), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(even), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    else:
        if (input.dtype == float32):
            median2d_sig = mod.get_function("median2d_float_sigclip_w")
        elif (input.dtype == int32):
            median2d_sig = mod.get_function("median2d_int_sigclip_w")
        elif (input.dtype == int64):
            median2d_sig = mod.get_function("median2d_long_sigclip_w")
        elif (input.dtype == float64):
            median2d_sig = mod.get_function("median2d_double_sigclip_w")
            outtype = float64
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        if (len(w.shape) != 2):
            w2 = empty(input.shape, input.dtype)
            w2[:] = w
            w = w2
        median2d_sig(drv.In(input), drv.Out(medVals), drv.In(w), drv.Out(wmap), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(even), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    return medVals

def gpu_arraymedian(input, axis="both", lthreshold=None, hthreshold=None, nlow=0, nhigh=0, nonzero=False, even=False, kernel=None, kernel2d=gpumedian2d, kernel3d=gpumedian3d, w=None, wmap=None, sigclip=False, niter=5, lsigma=3, hsigma=3, mclip=0):
    if (not input.dtype.isnative):
        #Byteswap
        print("Byteswapping data!")
        input = float32(input)
    if (input.dtype != float32 and input.dtype != int32 and input.dtype != float64 and input.dtype != int64):
        input = input.astype(float32) #Cast for example uint16 as float32
    t = time.time()
    n = input.size
    if (n < 2**16 or not hasCuda):
        kernel = fatboyclib.median
        kernel2d = fatboyclib.median2d
        kernel3d = fatboyclib.median3d
        input = input.copy()
    dims = len(input.shape)
    if (n == 0):
        return 0
    if (axis == "both" or dims == 1):
        if (kernel is None):
            #Check after above test for array size so that it doesn't crash on CPU-only machines
            #Where no kernel is listed for taking median of smaller arrays
            kernel = defaultKernel #Use defaultKernel, which depends on imports
        input = input.copy()
        if (lthreshold is None and hthreshold is None):
            med = kernel(input, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
        elif (lthreshold is None):
            med = kernel(input, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
        elif (hthreshold is None):
            med = kernel(input, lthreshold=lthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
        else:
            med = kernel(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
        #print "Median time: ", time.time()-t
        return med
    else:
        medVals = None
        if (lthreshold is None):
            lthreshold = INT_MIN
        if (hthreshold is None):
            hthreshold = INT_MAX
        if (sigclip):
            #median with sigma clipping
            if (kernel2d == fatboyclib.median2d):
                if (dims == 2):
                    if (axis == "Y"):
                        medVals = kernel2d(ascontiguousarray(input.transpose()), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    elif (axis == "X"):
                        medVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
            else:
                #Update kernel - median with sigma clipping on GPU
                kernel2d = gpumedian2d_sigclip
                kernel3d = gpumedian3d_sigclip
                if (dims == 2):
                    if (axis == "Y"):
                        medVals = kernel2d(gputranspose(input), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    elif (axis == "X"):
                        medVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                elif (dims == 3):
                    if (axis == "Z"):
                        medVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    elif (axis == "Y"):
                        medVals = kernel3d(gputranspose3d(input, 0, 2), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    elif (axis == "X"):
                        medVals = kernel3d(gputranspose3d(input, 1, 2), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, even=even, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
        elif (w is not None and wmap is not None):
            #w = weights or exptimes for weighting/exposure map
            kernel2d = gpumedian2d_w
            kernel3d = gpumedian3d_w
            if (dims == 2):
                if (axis == "Y"):
                    medVals = kernel2d(gputranspose(input), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even, w=w, wmap=wmap)
                elif (axis == "X"):
                    medVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even, w=w, wmap=wmap)
            elif (dims == 3):
                if (axis == "Z"):
                    medVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even, w=w, wmap=wmap)
                elif (axis == "Y"):
                    medVals = kernel3d(gputranspose3d(input, 0, 2), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even, w=w, wmap=wmap)
                elif (axis == "X"):
                    medVals = kernel3d(gputranspose3d(input, 1, 2), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even, w=w, wmap=wmap)
        else:
            #Regular median
            if (dims == 2):
                if (kernel2d == fatboyclib.median2d and axis == "Y"):
                    medVals = kernel2d(ascontiguousarray(input.transpose()), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
                elif (axis == "Y"):
                    medVals = kernel2d(gputranspose(input), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
                elif (axis == "X"):
                    medVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
            elif (dims == 3):
                if (axis == "Z"):
                    medVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
                elif (axis == "Y"):
                    medVals = kernel3d(gputranspose3d(input, 0, 2), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
                elif (axis == "X"):
                    medVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even, axis="X")
                    #medVals = kernel3d(gputranspose3d(input, 1, 2), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, even=even)
        #print "Median time: ", time.time()-t
        return medVals

###### ---------------------   MEAN FUNCTIONS   --------------------########

def gpumean3d(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nlow=0, nhigh=0, nonzero=False, weights=None):
    if (len(input.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    rows = input.shape[0]
    cols = input.shape[1]
    depth = input.shape[2]
    block_size = 512
    blocks = input.size//block_size
    if (input.size % block_size != 0):
        blocks += 1
    outtype = float32
    meanVals = empty((rows,cols), outtype)
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (weights is None):
        if (input.dtype == float32):
            mean2d = mod.get_function("mean2d_float")
        elif (input.dtype == int32):
            mean2d = mod.get_function("mean2d_int")
        elif (input.dtype == int64):
            mean2d = mod.get_function("mean2d_long")
        elif (input.dtype == float64):
            mean2d = mod.get_function("mean2d_double")
            outtype = float64
            meanVals = empty((rows,cols), outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        mean2d(drv.In(input), drv.Out(meanVals), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), grid=(blocks,1), block=(block_size,1,1))
    else:
        if (nlow > 0 or nhigh > 0):
            print("Weighting not supported for minmax rejection!")
            return None
        if (input.dtype == float32):
            mean2d = mod.get_function("wmean2d_float")
        elif (input.dtype == int32):
            mean2d = mod.get_function("wmean2d_int")
        elif (input.dtype == int64):
            mean2d = mod.get_function("wmean2d_long")
        elif (input.dtype == float64):
            mean2d = mod.get_function("wmean2d_double")
            outtype = float64
            meanVals = empty((rows,cols), outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        divisor = weights.sum()
        mean2d(drv.In(input), drv.Out(meanVals), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), drv.In(weights), outtype(divisor), grid=(blocks,1), block=(block_size,1,1))
    return meanVals

def gpumean3d_w(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nonzero=False, w=None, wmap=None, weights=None):
    if (len(input.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    rows = input.shape[0]
    cols = input.shape[1]
    depth = input.shape[2]
    block_size = 512
    blocks = input.size//block_size
    if (input.size % block_size != 0):
        blocks += 1
    outtype = float32
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (input.dtype == float32):
        mean2d_w = mod.get_function("mean2d_float_w")
    elif (input.dtype == int32):
        mean2d_w = mod.get_function("mean2d_int_w")
    elif (input.dtype == int64):
        mean2d_w = mod.get_function("mean2d_long_w")
    elif (input.dtype == float64):
        mean2d_w = mod.get_function("mean2d_double_w")
        outtype = float64
    else:
        print("Invalid datatype: ", input.dtype)
        return None
    if (weights is None):
        weights = ones(depth, outtype)
    divisor = weights.sum()
    meanVals = empty((rows,cols), outtype)
    mean2d_w(drv.In(input), drv.Out(meanVals), drv.In(w), drv.Out(wmap), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), drv.In(weights), outtype(divisor), grid=(blocks,1), block=(block_size,1,1))
    return meanVals

def gpumean3d_sigclip(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nonzero=False, w=None, wmap=None, niter=5, lsigma=3, hsigma=3, mclip=0, weights=None):
    if (len(input.shape) != 3):
        print("Invalid shape!  Must be 3-d array!")
        return None
    if (isinstance(mclip, str)):
        if (mclip == 'mean' or mclip == 'med'):
            mclip = 1
        else:
            mclip = 0
    if (mclip != 0 and mclip != 1):
        print("Invalid mclip: "+str(mclip)+".  Using mean (mclip = 0).")
        mclip = 0
    rows = input.shape[0]
    cols = input.shape[1]
    depth = input.shape[2]
    block_size = 512
    blocks = input.size//block_size
    if (input.size % block_size != 0):
        blocks += 1
    outtype = float32
    meanVals = empty((rows,cols), outtype)
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (weights is None and w is None and wmap is None):
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("mean2d_float_sigclip")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("mean2d_int_sigclip")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("mean2d_long_sigclip")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("mean2d_double_sigclip")
            outtype = float64
            meanVals = empty((rows,cols), outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        mean2d_sig(drv.In(input), drv.Out(meanVals), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    elif (w is None and wmap is None):
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("wmean2d_float_sigclip")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("wmean2d_int_sigclip")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("wmean2d_long_sigclip")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("wmean2d_double_sigclip")
            outtype = float64
            meanVals = empty((rows,cols), outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        mean2d_sig(drv.In(input), drv.Out(meanVals), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), drv.In(weights), grid=(blocks,1), block=(block_size,1,1))
    elif (weights is None):
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("mean2d_float_sigclip_w")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("mean2d_int_sigclip_w")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("mean2d_long_sigclip_w")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("mean2d_double_sigclip_w")
            outtype = float64
            meanVals = empty((rows,cols), outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        if (len(w.shape) != 3):
            w2 = empty(input.shape, input.dtype)
            w2[:,:] = w
            w = w2
        mean2d_sig(drv.In(input), drv.Out(meanVals), drv.In(w), drv.Out(wmap), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    else:
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("wmean2d_float_sigclip_w")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("wmean2d_int_sigclip_w")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("wmean2d_long_sigclip_w")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("wmean2d_double_sigclip_w")
            outtype = float64
            meanVals = empty((rows,cols), outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        if (len(w.shape) != 3):
            w2 = empty(input.shape, input.dtype)
            w2[:,:] = w
            w = w2
        mean2d_sig(drv.In(input), drv.Out(meanVals), drv.In(w), drv.Out(wmap), int32(rows*cols), int32(depth), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), drv.In(weights), grid=(blocks,1), block=(block_size,1,1))
    return meanVals

def gpumean2d(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nlow=0, nhigh=0, nonzero=False, weights=None):
    rows = input.shape[0]
    cols = input.shape[1]
    block_size = 512
    blocks = input.size//block_size
    if (input.size % block_size != 0):
        blocks += 1
    outtype = float32
    meanVals = empty(rows, outtype)
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (weights is None):
        if (input.dtype == float32):
            mean2d = mod.get_function("mean2d_float")
        elif (input.dtype == int32):
            mean2d = mod.get_function("mean2d_int")
        elif (input.dtype == int64):
            mean2d = mod.get_function("mean2d_long")
        elif (input.dtype == float64):
            mean2d = mod.get_function("mean2d_double")
            outtype = float64
            meanVals = empty(rows, outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        mean2d(drv.In(input), drv.Out(meanVals), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), grid=(blocks,1), block=(block_size,1,1))
    else:
        if (nlow > 0 or nhigh > 0):
            print("Weighting not supported for minmax rejection!")
            return None
        if (input.dtype == float32):
            mean2d = mod.get_function("wmean2d_float")
        elif (input.dtype == int32):
            mean2d = mod.get_function("wmean2d_int")
        elif (input.dtype == int64):
            mean2d = mod.get_function("wmean2d_long")
        elif (input.dtype == float64):
            mean2d = mod.get_function("wmean2d_double")
            outtype = float64
            meanVals = empty(rows, outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        divisor = weights.sum()
        mean2d(drv.In(input), drv.Out(meanVals), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nlow), int32(nhigh), int32(nonzero), drv.In(weights), outtype(divisor), grid=(blocks,1), block=(block_size,1,1))
    return meanVals

def gpumean2d_w(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nonzero=False, w=None, wmap=None, weights=None):
    rows = input.shape[0]
    cols = input.shape[1]
    block_size = 512
    blocks = input.size//block_size
    if (input.size % block_size != 0):
        blocks += 1
    outtype = float32
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (input.dtype == float32):
        mean2d_w = mod.get_function("mean2d_float_w")
    elif (input.dtype == int32):
        mean2d_w = mod.get_function("mean2d_int_w")
    elif (input.dtype == int64):
        mean2d_w = mod.get_function("mean2d_long_w")
    elif (input.dtype == float64):
        mean2d_w = mod.get_function("mean2d_double_w")
        outtype = float64
    else:
        print("Invalid datatype: ", input.dtype)
        return None
    if (weights is None):
        weights = ones(cols, outtype)
    divisor = weights.sum()
    meanVals = empty(rows, outtype)
    mean2d_w(drv.In(input), drv.Out(meanVals), drv.In(w), drv.Out(wmap), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), drv.In(weights), outtype(divisor), grid=(blocks,1), block=(block_size,1,1))
    return meanVals

def gpumean2d_sigclip(input, lthreshold=INT_MIN, hthreshold=INT_MAX, nonzero=False, w=None, wmap=None, niter=5, lsigma=3, hsigma=3, mclip=0, weights=None):
    if (len(input.shape) != 2):
        print("Invalid shape!  Must be 2-d array!")
        return None
    if (isinstance(mclip, str)):
        if (mclip == 'mean' or mclip == 'med'):
            mclip = 1
        else:
            mclip = 0
    if (mclip != 0 and mclip != 1):
        print("Invalid mclip: "+str(mclip)+".  Using mean (mclip = 0).")
        mclip = 0
    rows = input.shape[0]
    cols = input.shape[1]
    block_size = 512
    blocks = input.size//block_size
    if (input.size % block_size != 0):
        blocks += 1
    outtype = float32
    meanVals = empty(rows, outtype)
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    if (weights is None and w is None and wmap is None):
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("mean2d_float_sigclip")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("mean2d_int_sigclip")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("mean2d_long_sigclip")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("mean2d_double_sigclip")
            outtype = float64
            meanVals = empty(rows, outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        mean2d_sig(drv.In(input), drv.Out(meanVals), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    elif (w is None and wmap is None):
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("wmean2d_float_sigclip")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("wmean2d_int_sigclip")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("wmean2d_long_sigclip")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("wmean2d_double_sigclip")
            outtype = float64
            meanVals = empty(rows, outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        mean2d_sig(drv.In(input), drv.Out(meanVals), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), drv.In(weights), grid=(blocks,1), block=(block_size,1,1))
    elif (weights is None):
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("mean2d_float_sigclip_w")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("mean2d_int_sigclip_w")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("mean2d_long_sigclip_w")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("mean2d_double_sigclip_w")
            outtype = float64
            meanVals = empty(rows, outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        if (len(w.shape) != 2):
            w2 = empty(input.shape, input.dtype)
            w2[:] = w
            w = w2
        mean2d_sig(drv.In(input), drv.Out(meanVals), drv.In(w), drv.Out(wmap), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), grid=(blocks,1), block=(block_size,1,1))
    else:
        if (input.dtype == float32):
            mean2d_sig = mod.get_function("wmean2d_float_sigclip_w")
        elif (input.dtype == int32):
            mean2d_sig = mod.get_function("wmean2d_int_sigclip_w")
        elif (input.dtype == int64):
            mean2d_sig = mod.get_function("wmean2d_long_sigclip_w")
        elif (input.dtype == float64):
            mean2d_sig = mod.get_function("wmean2d_double_sigclip_w")
            outtype = float64
            meanVals = empty(rows, outtype)
        else:
            print("Invalid datatype: ", input.dtype)
            return None
        if (len(w.shape) != 2):
            w2 = empty(input.shape, input.dtype)
            w2[:] = w
            w = w2
        mean2d_sig(drv.In(input), drv.Out(meanVals), drv.In(w), drv.Out(wmap), int32(rows), int32(cols), float32(lthreshold), float32(hthreshold), int32(nonzero), int32(niter), float32(lsigma), float32(hsigma), int32(mclip), drv.In(weights), grid=(blocks,1), block=(block_size,1,1))
    return meanVals

def gpu_arraymean(input, axis="both", lthreshold=None, hthreshold=None, nlow=0, nhigh=0, nonzero=False, kernel=None, kernel2d=gpumean2d, kernel3d=gpumean3d, w=None, wmap=None, sigclip=False, niter=5, lsigma=3, hsigma=3, mclip=0, weights=None):
    t = time.time()
    n = input.size
    dims = len(input.shape)
    if (n == 0):
        return 0
    if (axis == "both" or dims == 1):
        print("Use UFGPUOps.gpumean for 1-d mean!")
        return None
    else:
        meanVals = None
        if (lthreshold is None):
            lthreshold = INT_MIN
        if (hthreshold is None):
            hthreshold = INT_MAX
        if (sigclip):
            #mean with sigma clipping
            kernel2d = gpumean2d_sigclip
            kernel3d = gpumean3d_sigclip
            if (dims == 2):
                if (axis == "Y"):
                    meanVals = kernel2d(gputranspose(input), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip, weights=weights)
                elif (axis == "X"):
                    meanVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip, weights=weights)
            elif (dims == 3):
                if (axis == "Z"):
                    meanVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip, weights=weights)
                elif (axis == "Y"):
                    meanVals = kernel3d(gputranspose3d(input, 0, 2), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip, weights=weights)
                elif (axis == "X"):
                    meanVals = kernel3d(gputranspose3d(input, 1, 2), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip, weights=weights)
        elif (w is not None and wmap is not None):
            #w = weights or exptimes for weighting/exposure map
            kernel2d = gpumean2d_w
            kernel3d = gpumean3d_w
            if (dims == 2):
                if (axis == "Y"):
                    meanVals = kernel2d(gputranspose(input), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, weights=weights)
                elif (axis == "X"):
                    meanVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, weights=weights)
            elif (dims == 3):
                if (axis == "Z"):
                    meanVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, weights=weights)
                elif (axis == "Y"):
                    meanVals = kernel3d(gputranspose3d(input, 0, 2), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, weights=weights)
                elif (axis == "X"):
                    meanVals = kernel3d(gputranspose3d(input, 1, 2), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero, w=w, wmap=wmap, weights=weights)
        else:
            #Regular mean
            if (dims == 2):
                if (axis == "Y"):
                    meanVals = kernel2d(gputranspose(input), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, weights=weights)
                elif (axis == "X"):
                    meanVals = kernel2d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, weights=weights)
            elif (dims == 3):
                if (axis == "Z"):
                    meanVals = kernel3d(input, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, weights=weights)
                elif (axis == "Y"):
                    meanVals = kernel3d(gputranspose3d(input, 0, 2), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, weights=weights)
                elif (axis == "X"):
                    meanVals = kernel3d(gputranspose3d(input, 1, 2), lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, nonzero=nonzero, weights=weights)
        print("Mean time: ", time.time()-t)
        return meanVals

#For backwards compatibility
arraymedian = gpu_arraymedian
arraymean = gpu_arraymean
