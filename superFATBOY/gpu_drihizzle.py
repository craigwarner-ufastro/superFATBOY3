#!/usr/bin/python -u
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
    print("gpu_drihizzle> WARNING: PyCUDA not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)

from math import *
from numpy import *
from .fatboyDataUnit import *

blocks = 2048*4
block_size = 512

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction

MODE_FDU_USE_INDIVIDUAL_GPMS = 5 #Use individual GPMs (inverse of BPMs) for input masks

def get_mod():
    mod = None
    if (hasCuda and superFATBOY.gpuEnabled()):
        mod = SourceModule("""

        __global__ void calcXin(float *xin, int nx, float offset) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          xin[i] = (i%nx) + offset;
        }

        __global__ void calcYin(float *yin, int nx, float offset) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          yin[i] = (i/nx) + offset;
        }

        __global__ void calcXYin(float *xin, float *yin, int nx, int ny, float xoff, float yoff) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= nx*ny) {
            xin[i] = 0;
            yin[i] = 0;
            return;
          }
          xin[i] = (i%nx) + xoff;
          yin[i] = (i/nx) + yoff;
        }

        __global__ void calcTransOpt(float *xout, float *yout, float *xin, float *yin, float *xcoeffs, float *ycoeffs, int order, float offset, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) {
            xout[i] = 0;
            yout[i] = 0;
            return;
          }
          int n = 0;
          float xp [20];
          float yp [20];
          xp[0] = 1; yp[0] = 1;
          xp[1] = xin[i]; yp[1] = yin[i];
          if (order >= 2) {
            xp[2] = xin[i]*xin[i];
            yp[2] = yin[i]*yin[i];
          }
          for (int j = 3; j <= order; j++) {
            xp[j] = pow(xin[i], j);
            yp[j] = pow(yin[i], j);
          }
          xout[i] = offset;
          yout[i] = offset;
          for (int l = 0; l <= order; l++) {
            for (int k = 0; k <= l; k++) {
              xout[i] += xcoeffs[n]*xp[l-k]*yp[k];
              yout[i] += ycoeffs[n]*xp[l-k]*yp[k];
              n++;
            }
          }
        }

        __global__ void multArrFloatIntScalar(float *data, int* mask, float factor, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          data[i] *= mask[i]*factor;
        }

        __global__ void divFloatArrays(float *dividend, float* divisor, int size)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          if (divisor[i] != 0) dividend[i] /= divisor[i];
        }

        __global__ void pointKernel(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh+0.5);
          int inty = (int)(yout[i]+ysh+0.5);
          int idx = intx+xsize*inty;
          atomicAdd(&newdata[idx], data[i]);
          atomicAdd(&expmap[idx], tmpexp[i]);
        }

        __global__ void pointKernelPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh+0.5);
          int inty = (int)(yout[i]+ysh+0.5);
          int idx = intx+xsize*inty;
          atomicAdd(&newdata[idx], data[i]);
          atomicAdd(&expmap[idx], tmpexp[i]);
          atomicAdd(&pixmap[idx], 1);
        }

        __global__ void uniformKernel(int *newdata, int *data, float *xout, float *yout, int *mask, int nxsize, int nsize)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= nsize) return;
          int intx = (int)(xout[i]);
          int inty = (int)(yout[i]);
          int idx = intx+nxsize*inty;
          if (idx < 0) return;

          atomicMax(&newdata[idx], data[i]*mask[i]);
          atomicMax(&newdata[idx+1], data[i]*mask[i]);
          atomicMax(&newdata[idx+nxsize], data[i]*mask[i]);
          atomicMax(&newdata[idx+nxsize+1], data[i]*mask[i]);
        }

        __global__ void turboKernel(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float fracx = xout[i]+xsh-intx;
          float fracy = yout[i]+ysh-inty;
          int idx = intx+xsize*inty;

          if (dropsize < 1) {
            float idrop = 1.f/dropsize;
            float fx1 = ((1.f+dropsize)/2-fracx)*idrop;
            float fy1 = ((1.f+dropsize)/2-fracy)*idrop;
            fracx = (fracx-(1.f-dropsize)/2)*idrop;
            fracy = (fracy-(1.f-dropsize)/2)*idrop;

            atomicAdd(&newdata[idx], data[i]*fy1*fx1);
            atomicAdd(&newdata[idx+1], data[i]*fy1*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*fracy*fx1);
            atomicAdd(&newdata[idx+xsize+1], data[i]*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*fy1*fx1);
            atomicAdd(&expmap[idx+1], tmpexp[i]*fy1*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*fracy*fx1);
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*fracy*fracx);
          } else {
            //dropsize = 1
            atomicAdd(&newdata[idx], data[i]*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&newdata[idx+1], data[i]*(1.f-fracy)*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*fracy*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize+1], data[i]*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&expmap[idx+1], tmpexp[i]*(1.f-fracy)*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*fracy*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*fracy*fracx);
          }
        }

        __global__ void turboKernelPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float fracx = xout[i]+xsh-intx;
          float fracy = yout[i]+ysh-inty;
          int idx = intx+xsize*inty;

          if (dropsize < 1) {
            float idrop = 1.f/dropsize;
            float fx1 = ((1.f+dropsize)/2-fracx)*idrop;
            float fy1 = ((1.f+dropsize)/2-fracy)*idrop;
            fracx = (fracx-(1.f-dropsize)/2)*idrop;
            fracy = (fracy-(1.f-dropsize)/2)*idrop;

            atomicAdd(&newdata[idx], data[i]*fy1*fx1);
            atomicAdd(&newdata[idx+1], data[i]*fy1*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*fracy*fx1);
            atomicAdd(&newdata[idx+xsize+1], data[i]*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*fy1*fx1);
            atomicAdd(&expmap[idx+1], tmpexp[i]*fy1*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*fracy*fx1);
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*fracy*fracx);

            atomicAdd(&pixmap[idx], 1);
            atomicAdd(&pixmap[idx+1], 1);
            atomicAdd(&pixmap[idx+xsize], 1);
            atomicAdd(&pixmap[idx+xsize+1], 1);
          } else {
            //dropsize = 1
            atomicAdd(&newdata[idx], data[i]*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&newdata[idx+1], data[i]*(1.f-fracy)*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*fracy*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize+1], data[i]*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&expmap[idx+1], tmpexp[i]*(1.f-fracy)*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*fracy*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*fracy*fracx);

            atomicAdd(&pixmap[idx], 1);
            atomicAdd(&pixmap[idx+1], 1);
            atomicAdd(&pixmap[idx+xsize], 1);
            atomicAdd(&pixmap[idx+xsize+1], 1);
          }
        }

        __global__ void tophatKernel(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize)
        {
          //Flux spread equally among pixels whose centers lie inside circle with r = dropsize/2
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rnddrop = (int)(rintf(dropsize/2.f));
          const int ceildrop = (int)(ceilf(dropsize/2.f));
          float counts = 0.f;
          float rx, ry;

          //Loop over box containing circle
          //Count how many output pixels each input pixel contributes to
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue; //pixel outside r
              rx = (intx+jx+0.5f-ox2);
              ry = (inty+jy+0.5f-oy2);
              if (sqrt(rx*rx+ry*ry) <= dropsize/2.f) counts++;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue; //pixel outside r
              rx = (intx+jx+0.5f-ox2);
              ry = (inty+jy+0.5f-oy2);
              if (sqrt(rx*rx+ry*ry) <= dropsize/2.f) {
                atomicAdd(&newdata[idx+jx+xsize*jy], data[i]);
                atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]);
              }
            }
          }
        }

        __global__ void tophatKernelPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize)
        {
          //Flux spread equally among pixels whose centers lie inside circle with r = dropsize/2
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rnddrop = (int)(rintf(dropsize/2.f));
          const int ceildrop = (int)(ceilf(dropsize/2.f));
          float counts = 0.f;
          float rx, ry;

          //Loop over box containing circle
          //Count how many output pixels each input pixel contributes to
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue; //pixel outside r
              rx = (intx+jx+0.5f-ox2);
              ry = (inty+jy+0.5f-oy2);
              if (sqrt(rx*rx+ry*ry) <= dropsize/2.f) counts++;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue; //pixel outside r
              rx = (intx+jx+0.5f-ox2);
              ry = (inty+jy+0.5f-oy2);
              if (sqrt(rx*rx+ry*ry) <= dropsize/2.f) {
                atomicAdd(&newdata[idx+jx+xsize*jy], data[i]);
                atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]);
                atomicAdd(&pixmap[idx+jx+xsize*jy], 1);
              }
            }
          }
        }

        __global__ void gaussianKernel(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize)
        {
          //Flux weighted by 2-D gaussian with FWHM dropsize
          //Cutoff at 2.5 sigma for time considerations (same as in IRAFs drizzle)
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rndsig = (int)(rintf(dropsize*2.5/2.3548));
          const int ceilsig = (int)(ceilf(dropsize*2.5/2.3548));
          float counts = 0.f;
          float zx, zy, gauss;

          //Loop over box containing Gaussian
          //Count up total weight given to each input pixel in output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              zx = 2.3548/dropsize*(intx+jx+0.5f-ox2);
              zy = 2.3548/dropsize*(inty+jy+0.5f-oy2);
              gauss = exp(-0.5*(zx*zx+zy*zy));
              counts += gauss;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              zx = 2.3548/dropsize*(intx+jx+0.5f-ox2);
              zy = 2.3548/dropsize*(inty+jy+0.5f-oy2);
              gauss = exp(-0.5*(zx*zx+zy*zy));
              atomicAdd(&newdata[idx+jx+xsize*jy], data[i]*gauss);
              atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]*gauss);
            }
          }
        }

        __global__ void gaussianKernelPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize)
        {
          //Flux weighted by 2-D gaussian with FWHM dropsize
          //Cutoff at 2.5 sigma for time considerations (same as in IRAFs drizzle)
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rndsig = (int)(rintf(dropsize*2.5/2.3548));
          const int ceilsig = (int)(ceilf(dropsize*2.5/2.3548));
          float counts = 0.f;
          float zx, zy, gauss;

          //Loop over box containing Gaussian
          //Count up total weight given to each input pixel in output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              zx = 2.3548/dropsize*(intx+jx+0.5f-ox2);
              zy = 2.3548/dropsize*(inty+jy+0.5f-oy2);
              gauss = exp(-0.5*(zx*zx+zy*zy));
              counts += gauss;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              zx = 2.3548/dropsize*(intx+jx+0.5f-ox2);
              zy = 2.3548/dropsize*(inty+jy+0.5f-oy2);
              gauss = exp(-0.5*(zx*zx+zy*zy));
              atomicAdd(&newdata[idx+jx+xsize*jy], data[i]*gauss);
              atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]*gauss);
              atomicAdd(&pixmap[idx+jx+xsize*jy], 1);
            }
          }
        }

        __global__ void fastGaussKernel(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize, float *gausslut, int gausscen)
        {
          //Flux weighted by 2-D gaussian with FWHM dropsize
          //Cutoff at 2.5 sigma for time considerations same as in IRAFs drizzle
          //1D lookup table with points every 0.001 pixels used to estimate Gaussian
          //rather than calculating for every pixel.  Results in an overall
          //25% increase in speed versus gaussian with nearly identical results
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rndsig = (int)(rintf(dropsize*2.5/2.3548));
          const int ceilsig = (int)(ceilf(dropsize*2.5/2.3548));
          float counts = 0.f;
          float gauss;
          int indx = (int)(gausscen+1000*(intx+0.5f-ox2));
          int indy = (int)(gausscen+1000*(inty+0.5f-oy2));

          //Loop over box containing Gaussian
          //Count up total weight given to each input pixel in output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              gauss = gausslut[indx+jx*1000]*gausslut[indy+jy*1000];
              counts += gauss;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              gauss = gausslut[indx+jx*1000]*gausslut[indy+jy*1000];
              atomicAdd(&newdata[idx+jx+xsize*jy], data[i]*gauss);
              atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]*gauss);
            }
          }
        }

        __global__ void fastGaussKernelPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize, float *gausslut, int gausscen)
        {
          //Flux weighted by 2-D gaussian with FWHM dropsize
          //Cutoff at 2.5 sigma for time considerations same as in IRAFs drizzle
          //1D lookup table with points every 0.001 pixels used to estimate Gaussian
          //rather than calculating for every pixel.  Results in an overall
          //25% increase in speed versus gaussian with nearly identical results
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rndsig = (int)(rintf(dropsize*2.5/2.3548));
          const int ceilsig = (int)(ceilf(dropsize*2.5/2.3548));
          float counts = 0.f;
          float gauss;
          int indx = (int)(gausscen+1000*(intx+0.5f-ox2));
          int indy = (int)(gausscen+1000*(inty+0.5f-oy2));

          //Loop over box containing Gaussian
          //Count up total weight given to each input pixel in output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              gauss = gausslut[indx+jx*1000]*gausslut[indy+jy*1000];
              counts += gauss;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rndsig; jx <= rndsig; jx++) {
            for (int jy = -1*rndsig; jy <= rndsig; jy++) {
              if (jx*jx + jy*jy > ceilsig*ceilsig) continue; //outside 2.5 sigma
              gauss = gausslut[indx+jx*1000]*gausslut[indy+jy*1000];
              atomicAdd(&newdata[idx+jx+xsize*jy], data[i]*gauss);
              atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]*gauss);
              atomicAdd(&pixmap[idx+jx+xsize*jy], 1);
            }
          }
        }

        __global__ void lanczosKernel(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize, float *lanclut, int lanccen)
        {
          //Flux weighted by 2-D lanczos sinc function with width
          //determined by dropsize.
          //1D lookup table with points every 0.001 pixels used to estimate function
          //same as in IRAFs drizzle but IRAF only uses 0.01 pixel accuracy
          //rather than calculating function for every pixel.

          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rnddrop = (int)(rintf(dropsize));
          const int ceildrop = (int)(ceilf(dropsize));
          float counts = 0.f;
          float lanc;
          int indx = (int)(lanccen+1000*(intx+0.5f-ox2));
          int indy = (int)(lanccen+1000*(inty+0.5f-oy2));

          //Loop over box containing Lanczos function
          //Count up total weight given to each input pixel in output grid
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue;
              lanc = lanclut[indx+jx*1000]*lanclut[indy+jy*1000];
              counts += lanc;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue;
              lanc = lanclut[indx+jx*1000]*lanclut[indy+jy*1000];
              atomicAdd(&newdata[idx+jx+xsize*jy], data[i]*lanc);
              atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]*lanc);
            }
          }
        }

        __global__ void lanczosKernelPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float xsh, float ysh, int xsize, int size, float dropsize, float *lanclut, int lanccen)
        {
          //Flux weighted by 2-D lanczos sinc function with width
          //determined by dropsize.
          //1D lookup table with points every 0.001 pixels used to estimate function
          //same as in IRAFs drizzle but IRAF only uses 0.01 pixel accuracy
          //rather than calculating function for every pixel.

          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          float ox2 = xout[i]+xsh;
          float oy2 = yout[i]+ysh;
          int idx = intx+xsize*inty;
          const int rnddrop = (int)(rintf(dropsize));
          const int ceildrop = (int)(ceilf(dropsize));
          float counts = 0.f;
          float lanc;
          int indx = (int)(lanccen+1000*(intx+0.5f-ox2));
          int indy = (int)(lanccen+1000*(inty+0.5f-oy2));

          //Loop over box containing Lanczos function
          //Count up total weight given to each input pixel in output grid
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue;
              lanc = lanclut[indx+jx*1000]*lanclut[indy+jy*1000];
              counts += lanc;
            }
          }
          //Normalize input data to conserve flux
          if (counts == 0.f) counts = 1.f;
          data[i] /= counts;
          tmpexp[i] /= counts;

          //Calculate output grid
          for (int jx = -1*rnddrop; jx <= rnddrop; jx++) {
            for (int jy = -1*rnddrop; jy <= rnddrop; jy++) {
              if (jx*jx + jy*jy > ceildrop*ceildrop) continue;
              lanc = lanclut[indx+jx*1000]*lanclut[indy+jy*1000];
              atomicAdd(&newdata[idx+jx+xsize*jy], data[i]*lanc);
              atomicAdd(&expmap[idx+jx+xsize*jy], tmpexp[i]*lanc);
              atomicAdd(&pixmap[idx+jx+xsize*jy], 1);
            }
          }
        }

        //DRIHIZZLE3d specific optimized kernels
        __global__ void calcXYZin(float *xin, float *yin, float *zin, int nx, int ny, int nz, float xoff, float yoff, float zoff) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= nx*ny*nz) {
            xin[i] = 0;
            yin[i] = 0;
            zin[i] = 0;
            return;
          }
          xin[i] = (i%nx) + xoff;
          yin[i] = (i % (nx*ny))/nx + yoff;
          zin[i] = i/(nx*ny) + zoff;
        }

        __global__ void calcTransOpt3d(float *xout, float *yout, float *zout, float *xin, float *yin, float *zin, float *xcoeffs, float *ycoeffs, float *zcoeffs, int order, float offset, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) {
            xout[i] = 0;
            yout[i] = 0;
            zout[i] = 0;
            return;
          }
          int n = 0;
          float xp [10];
          float yp [10];
          float zp [10];
          xp[0] = 1; yp[0] = 1; zp[0] = 1;
          xp[1] = xin[i]; yp[1] = yin[i]; zp[1] = zin[i];
          if (order >= 2) {
            xp[2] = xin[i]*xin[i];
            yp[2] = yin[i]*yin[i];
            zp[2] = zin[i]*zin[i];
          }
          for (int j = 3; j <= order; j++) {
            xp[j] = pow(xin[i], j);
            yp[j] = pow(yin[i], j);
            zp[j] = pow(zin[i], j);
          }
          xout[i] = offset;
          yout[i] = offset;
          zout[i] = offset;
          for (int j = 0; j <= order; j++) {
            for (int l = 1; l < j+2; l++) {
              for (int k = 1; k <= l; k++) {
                xout[i] += xcoeffs[n]*xp[j-l+1]*yp[l-k]*zp[k-1];
                yout[i] += ycoeffs[n]*xp[j-l+1]*yp[l-k]*zp[k-1];
                zout[i] += zcoeffs[n]*xp[j-l+1]*yp[l-k]*zp[k-1];
                n++;
              }
            }
          }
        }

        __global__ void pointKernel3d(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float *zout, float xsh, float ysh, float zsh, int xsize, int ysize, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh+0.5);
          int inty = (int)(yout[i]+ysh+0.5);
          int intz = (int)(zout[i]+zsh+0.5);
          int idx = intx+xsize*inty+xsize*ysize*intz;
          atomicAdd(&newdata[idx], data[i]);
          atomicAdd(&expmap[idx], tmpexp[i]);
        }

        __global__ void pointKernel3dPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float *zout, float xsh, float ysh, float zsh, int xsize, int ysize, int size)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh+0.5);
          int inty = (int)(yout[i]+ysh+0.5);
          int intz = (int)(zout[i]+zsh+0.5);
          int idx = intx+xsize*inty+xsize*ysize*intz;
          atomicAdd(&newdata[idx], data[i]);
          atomicAdd(&expmap[idx], tmpexp[i]);
          atomicAdd(&pixmap[idx], 1);
        }

        __global__ void turboKernel3d(float *newdata, float *expmap, float *data, float *tmpexp, float *xout, float *yout, float *zout, float xsh, float ysh, float zsh, int xsize, int ysize, int size, float dropsize)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          int intz = (int)(zout[i]+zsh);
          float fracx = xout[i]+xsh-intx;
          float fracy = yout[i]+ysh-inty;
          float fracz = zout[i]+zsh-intz;
          int idx = intx+xsize*inty+xsize*ysize*intz;

          if (dropsize < 1) {
            float idrop = 1.f/dropsize;
            float fx1 = ((1.f+dropsize)/2-fracx)*idrop;
            float fy1 = ((1.f+dropsize)/2-fracy)*idrop;
            float fz1 = ((1.f+dropsize)/2-fracz)*idrop;
            fracx = (fracx-(1.f-dropsize)/2)*idrop;
            fracy = (fracy-(1.f-dropsize)/2)*idrop;
            fracz = (fracz-(1.f-dropsize)/2)*idrop;

            atomicAdd(&newdata[idx], data[i]*fz1*fy1*fx1);
            atomicAdd(&newdata[idx+1], data[i]*fz1*fy1*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*fz1*fracy*fx1);
            atomicAdd(&newdata[idx+xsize+1], data[i]*fz1*fracy*fracx);

            atomicAdd(&newdata[idx+xsize*ysize], data[i]*fracz*fy1*fx1);
            atomicAdd(&newdata[idx+xsize*ysize+1], data[i]*fracz*fy1*fracx);
            atomicAdd(&newdata[idx+xsize*ysize+xsize], data[i]*fracz*fracy*fx1);
            atomicAdd(&newdata[idx+xsize*ysize+xsize+1], data[i]*fracz*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*fz1*fy1*fx1);
            atomicAdd(&expmap[idx+1], tmpexp[i]*fz1*fy1*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*fz1*fracy*fx1);
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*fz1*fracy*fracx);

            atomicAdd(&expmap[idx+xsize*ysize], tmpexp[i]*fracz*fy1*fx1);
            atomicAdd(&expmap[idx+xsize*ysize+1], tmpexp[i]*fracz*fy1*fracx);
            atomicAdd(&expmap[idx+xsize*ysize+xsize], tmpexp[i]*fracz*fracy*fx1);
            atomicAdd(&expmap[idx+xsize*ysize+xsize+1], tmpexp[i]*fracz*fracy*fracx);
          } else {
            //dropsize = 1
            atomicAdd(&newdata[idx], data[i]*(1.f-fracz)*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&newdata[idx+1], data[i]*(1.f-fracz)*(1.f-fracy)*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*(1.f-fracz)*fracy*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize+1], data[i]*(1.f-fracz)*fracy*fracx);

            atomicAdd(&newdata[idx+xsize*ysize], data[i]*fracz*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize*ysize+1], data[i]*fracz*(1.f-fracy)*fracx);
            atomicAdd(&newdata[idx+xsize*ysize+xsize], data[i]*fracz*fracy*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize*ysize+xsize+1], data[i]*fracz*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*(1.f-fracz)*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&expmap[idx+1], tmpexp[i]*(1.f-fracz)*(1.f-fracy)*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*(1.f-fracz)*fracy*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*(1.f-fracz)*fracy*fracx);

            atomicAdd(&expmap[idx+xsize*ysize], tmpexp[i]*fracz*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize*ysize+1], tmpexp[i]*fracz*(1.f-fracy)*fracx);
            atomicAdd(&expmap[idx+xsize*ysize+xsize], tmpexp[i]*fracz*fracy*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize*ysize+xsize+1], tmpexp[i]*fracz*fracy*fracx);
          }
        }

        __global__ void turboKernel3dPix(float *newdata, float *expmap, int *pixmap, float *data, float *tmpexp, float *xout, float *yout, float *zout, float xsh, float ysh, float zsh, int xsize, int ysize, int size, float dropsize)
        {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          int intx = (int)(xout[i]+xsh);
          int inty = (int)(yout[i]+ysh);
          int intz = (int)(zout[i]+zsh);
          float fracx = xout[i]+xsh-intx;
          float fracy = yout[i]+ysh-inty;
          float fracz = zout[i]+zsh-intz;
          int idx = intx+xsize*inty+xsize*ysize*intz;

          if (dropsize < 1) {
            float idrop = 1.f/dropsize;
            float fx1 = ((1.f+dropsize)/2-fracx)*idrop;
            float fy1 = ((1.f+dropsize)/2-fracy)*idrop;
            float fz1 = ((1.f+dropsize)/2-fracz)*idrop;
            fracx = (fracx-(1.f-dropsize)/2)*idrop;
            fracy = (fracy-(1.f-dropsize)/2)*idrop;
            fracz = (fracz-(1.f-dropsize)/2)*idrop;

            atomicAdd(&newdata[idx], data[i]*fz1*fy1*fx1);
            atomicAdd(&newdata[idx+1], data[i]*fz1*fy1*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*fz1*fracy*fx1);
            atomicAdd(&newdata[idx+xsize+1], data[i]*fz1*fracy*fracx);

            atomicAdd(&newdata[idx+xsize*ysize], data[i]*fracz*fy1*fx1);
            atomicAdd(&newdata[idx+xsize*ysize+1], data[i]*fracz*fy1*fracx);
            atomicAdd(&newdata[idx+xsize*ysize+xsize], data[i]*fracz*fracy*fx1);
            atomicAdd(&newdata[idx+xsize*ysize+xsize+1], data[i]*fracz*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*fz1*fy1*fx1);
            atomicAdd(&expmap[idx+1], tmpexp[i]*fz1*fy1*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*fz1*fracy*fx1);
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*fz1*fracy*fracx);

            atomicAdd(&expmap[idx+xsize*ysize], tmpexp[i]*fracz*fy1*fx1);
            atomicAdd(&expmap[idx+xsize*ysize+1], tmpexp[i]*fracz*fy1*fracx);
            atomicAdd(&expmap[idx+xsize*ysize+xsize], tmpexp[i]*fracz*fracy*fx1);
            atomicAdd(&expmap[idx+xsize*ysize+xsize+1], tmpexp[i]*fracz*fracy*fracx);

            atomicAdd(&pixmap[idx], 1);
            atomicAdd(&pixmap[idx+1], 1);
            atomicAdd(&pixmap[idx+xsize], 1);
            atomicAdd(&pixmap[idx+xsize+1], 1);

            atomicAdd(&pixmap[idx+xsize*ysize], 1);
            atomicAdd(&pixmap[idx+xsize*ysize+1], 1);
            atomicAdd(&pixmap[idx+xsize*ysize+xsize], 1);
            atomicAdd(&pixmap[idx+xsize*ysize+xsize+1], 1);
          } else {
            //dropsize = 1
            atomicAdd(&newdata[idx], data[i]*(1.f-fracz)*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&newdata[idx+1], data[i]*(1.f-fracz)*(1.f-fracy)*fracx);
            atomicAdd(&newdata[idx+xsize], data[i]*(1.f-fracz)*fracy*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize+1], data[i]*(1.f-fracz)*fracy*fracx);

            atomicAdd(&newdata[idx+xsize*ysize], data[i]*fracz*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize*ysize+1], data[i]*fracz*(1.f-fracy)*fracx);
            atomicAdd(&newdata[idx+xsize*ysize+xsize], data[i]*fracz*fracy*(1.f-fracx));
            atomicAdd(&newdata[idx+xsize*ysize+xsize+1], data[i]*fracz*fracy*fracx);

            atomicAdd(&expmap[idx], tmpexp[i]*(1.f-fracz)*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&expmap[idx+1], tmpexp[i]*(1.f-fracz)*(1.f-fracy)*fracx);
            atomicAdd(&expmap[idx+xsize], tmpexp[i]*(1.f-fracz)*fracy*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize+1], tmpexp[i]*(1.f-fracz)*fracy*fracx);

            atomicAdd(&expmap[idx+xsize*ysize], tmpexp[i]*fracz*(1.f-fracy)*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize*ysize+1], tmpexp[i]*fracz*(1.f-fracy)*fracx);
            atomicAdd(&expmap[idx+xsize*ysize+xsize], tmpexp[i]*fracz*fracy*(1.f-fracx));
            atomicAdd(&expmap[idx+xsize*ysize+xsize+1], tmpexp[i]*fracz*fracy*fracx);

            atomicAdd(&pixmap[idx], 1);
            atomicAdd(&pixmap[idx+1], 1);
            atomicAdd(&pixmap[idx+xsize], 1);
            atomicAdd(&pixmap[idx+xsize+1], 1);

            atomicAdd(&pixmap[idx+xsize*ysize], 1);
            atomicAdd(&pixmap[idx+xsize*ysize+1], 1);
            atomicAdd(&pixmap[idx+xsize*ysize+xsize], 1);
            atomicAdd(&pixmap[idx+xsize*ysize+xsize+1], 1);
          }
        }

       """)
    return mod
#end get_mod()

mod = get_mod()

def leadZeros(n, x):
    s = str(x)
    while (len(s) < n):
        s = '0'+s
    return s

def drihizzle(frames, outfile=None, weightfile=None, inmask=None, weight='exptime', kernel='point', dropsize=1, geomDist=None, xsh=None, ysh=None, inunits='counts', outunits='cps', expkey='EXP_TIME', keepImages='no', imgdir='', log=None, gui=None, xtrans=None, ytrans=None, mef=0, inmef=None, pixfile=None, doPix=True, scale=1, mode=None, returnHeader=False, dataTag=None, pixscale_keyword='PIXSCALE', rotpa_keyword='ROT_PA', inport_keyword='INPORT', updateFDUs=False, expmaps=None):
    #frames = input data -- filename, list of filenames, or array
    #outfile = output data
    #weightfile = output exposure map
    #inmask = input good pixel mask
    #weight = weighting -- can be a number or 'exptime'
    #kernel = drizzle kernel
    #dropsize = drizzle dropsize
    #geomDist = file with geometric distortion correction parameters
    #xsh = x shifts (number or list)
    #ysh = y shifts (number or list)
    #inunits = input units, counts or cps
    #outunits = output units, counts or cps
    #expkey = FITS keyword for exposure time
    #keepImages = keep individual output frames
    #imgdir = directory to store individual output frames in
    #logfile = logfile to be appended
    #xtrans = x-transformation to be applied.  overrides geomDist.
    #ytrans = y-transformation to be applied.  overrides geomDist.
    #mef = fits extension
    #inmef = fits extension for input mask
    #pixfile = output pixel map (number of input pixels contributing to each output pixel)
    #scale = the factor of subsampling to do in the output image.  scale = 2 means each input pixel is a 2x2 grid in output.
    #mode = mode - FITS files, raw data, FDUs
    #returnHeader = Create new dict that can be used to update header
    #dataTag = for MODE_FDU_TAG
    #pixscale_keyword = keyword for pixel scale
    #rotpa_keyword = keyword for rotation angle
    #inport_keyword = keyword for inport (F2)
    #updateFDUs = tag FDUs with results and expmap to keep in memory
    #expmaps = list of input exposure maps for use with 'cps' inputs

    t = time.time()
    print("Drihizzle's the bizzle fo' shizzle!")
    _verbosity = fatboyLog.NORMAL

    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()

    #set log type
    logtype = LOGTYPE_NONE
    if (log is not None):
        if (isinstance(log, str)):
        #log given as a string
            log = open(log,'a')
            logtype = LOGTYPE_ASCII
        elif(isinstance(log, fatboyLog)):
            logtype = LOGTYPE_FATBOY
            _verbosity = fatboyLog._verbosity

    #Filelist
    if (isinstance(frames, str) and os.access(frames, os.F_OK)):
        frames = readFileIntoList(frames)
    elif (isinstance(frames, ndarray)):
        frames = [frames]
    elif (not isinstance(frames, list)):
        frames = [frames]

    nframes = len(frames)
    if (mode == MODE_FDU_DIFFERENCE):
        nframes -= 1

    #Find type
    if (mode is None):
        mode = MODE_FITS
        if (isinstance(frames[0], str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], ndarray)):
            mode = MODE_RAW
        elif (isinstance(frames[0], fatboyDataUnit)):
            mode = MODE_FDU


    if (gui is not None):
        if (len(gui) != 5): gui = None

    if (inmef is None):
        inmef = mef

    if (pixfile is not None):
        doPix = True

    if (outfile is not None and os.access(outfile, os.F_OK)):
        outExists = True
        if (weightfile is None or not os.access(weightfile, os.F_OK)):
            print("gpu_drihizzle> Error: Weighting image must exist if output image exists.")
            write_fatboy_log(log, logtype, "Weighting image must exist if output image exists.", __name__, messageType=fatboyLog.ERROR)
            return None
        if (doPix and (pixfile is None or not os.access(pixfile, os.F_OK))):
            print("gpu_drihizzle> Error: If output image exists and a pixmap is desired, pixmap must also exist.")
            write_fatboy_log(log, logtype, "If output image exists and a pixmap is desired, pixmap must also exist.", __name__, messageType=fatboyLog.ERROR)
            return None
    else:
        outExists = False
        if (weightfile is not None and os.access(weightfile, os.F_OK)):
            os.unlink(weightfile)
        if (doPix and pixfile is not None and os.access(pixfile, os.F_OK)):
            os.unlink(pixfile)

    doIndividualMasks = False
    if (isinstance(inmask, int) and inmask == MODE_FDU_USE_INDIVIDUAL_GPMS):
        doIndividualMasks = True
        print("gpu_drihizzle> Using individual good pixel masks.")
        write_fatboy_log(log, logtype, "Using individual good pixel masks.", __name__)
    elif (isinstance(inmask, str)):
        temp = pyfits.open(inmask)
        inmask = temp[inmef].data
        temp.close()
        del temp
    elif (isinstance(inmask, fatboyDataUnit)):
        inmask = inmask.getData()
    elif (not isinstance(inmask, ndarray)):
        inmask = None

    if (dropsize <= 0.01 and kernel != 'point'):
        kernel = 'point'
        print("gpu_drihizzle> Dropsize <= 0.01.  Using the point kernel.")
        write_fatboy_log(log, logtype, "Dropsize <= 0.01.  Using the point kernel.", __name__)
    if (dropsize > 1 and kernel == 'turbo'):
        dropsize = 1
        print("gpu_drihizzle> Dropsize > 1 not allowed for "+kernel+" kernel.  Using dropsize = 1.")
        write_fatboy_log(log, logtype, "Dropsize > 1 not allowed for "+kernel+" kernel.  Using dropsize = 1.", __name__)
    if (dropsize < sqrt(2) and kernel == 'tophat'):
        print("Dropsize < sqrt(2) should not be used for the tophat kernel.  Using dropsize = sqrt(2).")
        write_fatboy_log(log, logtype, "Dropsize < sqrt(2) should not be used for the tophat kernel.  Using dropsize = sqrt(2).", __name__)
        dropsize = sqrt(2)

    xcoeffs = []
    ycoeffs = []

    #Read geom distortion file or parse list
    if (geomDist is not None):
        if (isinstance(geomDist, str) and os.access(geomDist, os.F_OK)):
            f = open(geomDist, 'r')
            temp = f.read().split('\n')
            f.close()
            temp.pop(0)
            currCoeff = temp.pop(0)
            while (currCoeff != ''):
                xcoeffs.append(float(currCoeff))
                currCoeff = temp.pop(0)
            while (len(temp) > 0):
                currCoeff = temp.pop(0)
                if (currCoeff != ''):
                    ycoeffs.append(float(currCoeff))
        elif (isinstance(geomDist, list)):
            if (len(geomDist) == 2):
                xcoeffs = geomDist[0]
                ycoeffs = geomDist[1]
        else:
            xcoeffs = [0,1,0]
            ycoeffs = [0,0,1]
    else:
        #xin = xout, yin = yout
        xcoeffs = [0,1,0]
        ycoeffs = [0,0,1]

    xcoeffs = array(xcoeffs).astype(float32)
    ycoeffs = array(ycoeffs).astype(float32)

    ncoeff = len(xcoeffs)
    order = 0
    while (ncoeff > 0):
        ncoeff -= (order+1)
        if (ncoeff > 0): order+=1

    #Shifts
    if (xsh is None):
        xsh = zeros(nframes)
    elif (not isinstance(xsh, list) and not isinstance(xsh, ndarray)):
        xsh = [xsh]

    if (ysh is None):
        ysh = zeros(nframes)
    elif (not isinstance(ysh, list) and not isinstance(ysh, ndarray)):
        ysh = [ysh]

    if (_verbosity == fatboyLog.VERBOSE):
        print("Initialize: ",time.time()-t)
    tt = time.time()
    #Setup output image structure
    totexp = 0.
    numframes = 0
    hasWCS = False
    newHeader = dict() #setup new header

    #output file exists, open for updating
    if (outExists):
        outimage = pyfits.open(outfile, 'update')
        outexp = pyfits.open(weightfile, 'update')
        if ('DHZUNITS' in outimage[0].header):
            if (outimage[0].header['DHZUNITS'] == 'cps'):
                outimage[mef].data *= outexp[mef].data
            if (outimage[0].header['DHZUNITS'] == 'cps' and expkey in outimage[0].header):
                #outimage[mef].data*=float(outimage[0].header[expkey])
                totexp = float(outimage[0].header[expkey])
        if ('DHZNFRMS' in outimage[0].header):
            numframes = int(outimage[0].header['DHZNFRMS'])
        if (doPix):
            outpix = pyfits.open(pixfile, 'update')

    if (_verbosity == fatboyLog.VERBOSE):
        print("Open Files: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Get look up tables if necessary
    if (kernel == 'fastgauss'):
        gausslut = getGaussLut(dropsize, 5, 0.001)
        gausscen = gausslut.shape[0]//2
    if (kernel == 'lanczos'):
        lanclut = getLanczosLut(dropsize, 0.001)
        lanccen = lanclut.shape[0]//2

    ##Now start nx and ny at 0 and compare with size of each frame to get max size
    nx = 0
    ny = 0
    for j in range(nframes):
        if (mode == MODE_FITS):
            temp = pyfits.open(frames[j])
            if ('NAXIS2' in temp[mef].header):
                ny = max(ny, int(temp[mef].header['NAXIS2']))
            else:
                ny = max(ny, temp[mef].data.shape[0])
            if ('NAXIS1' in temp[mef].header):
                nx = max(nx, int(temp[mef].header['NAXIS1']))
            else:
                nx = max(nx, temp[mef].data.shape[1])
        elif (mode == MODE_RAW):
            ny = max(ny, frames[j].shape[0])
            nx = max(nx, frames[j].shape[1])
        elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE):
            ny = max(ny, frames[j].getShape()[0])
            nx = max(nx, frames[j].getShape()[1])
        elif (mode == MODE_FDU_TAG):
            ny = max(ny, frames[j].getData(tag=dataTag).shape[0])
            nx = max(nx, frames[j].getData(tag=dataTag).shape[1])

    for j in range(nframes):
        #Set weighting factor if not FITS keyword
        numframes+=1
        exptime = 1.0
        scalefac = 1.0
        if (not isinstance(weight, str)):
            scalefac = weight
        xrefin = -1
        yrefin = -1

        exptime = -1
        #wcs and FITS header
        xrefin = 0
        yrefin = 0
        crval1 = 0
        crval2 = 0
        cd11 = 0
        cd12 = 0
        cd21 = 0
        cd22 = 0
        pixscale = 0
        theta = 0
        inport = 0

        #Read data and wcs info
        if (mode == MODE_FITS):
            name = frames[j]
            temp = pyfits.open(frames[j])
            data = temp[mef].data.astype(float32)
            if (expkey in temp[0].header):
                exptime = float(temp[0].header[expkey])
            if ('CRPIX1' in temp[0].header):
                xrefin = float(temp[0].header['CRPIX1'])-1
            if ('CRPIX2' in temp[0].header):
                yrefin = float(temp[0].header['CRPIX2'])-1
            if (j == 0):
                if ('CRVAL1' in temp[0].header):
                    crval1 = float(temp[0].header['CRVAL1'])
                if ('CRVAL2' in temp[0].header):
                    crval2 = float(temp[0].header['CRVAL2'])
            if ('CD1_1' in temp[0].header):
                hasWCS = True
                cd11 = float(temp[0].header['CD1_1'])
            if ('CD1_2' in temp[0].header):
                cd12 = float(temp[0].header['CD1_2'])
            if ('CD2_1' in temp[0].header):
                cd21 = float(temp[0].header['CD2_1'])
            if ('CD2_2' in temp[0].header):
                cd22 = float(temp[0].header['CD2_2'])
            if (pixscale_keyword in temp[0].header):
                pixscale = float(temp[0].header[pixscale_keyword])
            if (rotpa_keyword in temp[0].header):
                theta = float(temp[0].header[rotpa_keyword])
            if (inport_keyword in temp[0].header):
                inport = float(temp[0].header[inport_keyword])
            temp.close()
        elif (mode == MODE_RAW):
            name = "frame number "+str(j)
            data = frames[j].astype(float32)
        elif (mode == MODE_FDU):
            name = frames[j].getFullId()
            data = frames[j].getData().astype(float32)
        elif (mode == MODE_FDU_DIFFERENCE):
            name = frames[j+1].getFullId()+"-"+frames[j].getFullId()
            data = frames[j+1].getData()-frames[j].getData()
        elif (mode == MODE_FDU_TAG):
            name = frames[j].getFullId()+":"+dataTag
            data = frames[j].getData(tag=dataTag).astype(float32)
        if (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
            if (doIndividualMasks):
                inmask = (1-frames[j].getBadPixelMask().getData()).astype("int32")
            exptime = frames[j].exptime
            if (frames[j].hasHeaderValue('CRPIX1')):
                xrefin = float(frames[j].getHeaderValue('CRPIX1'))
            if (frames[j].hasHeaderValue('CRPIX2')):
                yrefin = float(frames[j].getHeaderValue('CRPIX2'))
            if (j == 0):
                if (frames[j].hasHeaderValue('CRVAL1')):
                    crval1 = float(frames[j].getHeaderValue('CRVAL1'))
                if (frames[j].hasHeaderValue('CRVAL2')):
                    crval2 = float(frames[j].getHeaderValue('CRVAL2'))
            if (frames[j].hasHeaderValue('CD1_1')):
                hasWCS = True
                cd11 = float(frames[j].getHeaderValue('CD1_1'))
            if (frames[j].hasHeaderValue('CD1_2')):
                cd12 = float(frames[j].getHeaderValue('CD1_2'))
            if (frames[j].hasHeaderValue('CD2_1')):
                cd21 = float(frames[j].getHeaderValue('CD2_1'))
            if (frames[j].hasHeaderValue('CD2_2')):
                cd22 = float(frames[j].getHeaderValue('CD2_2'))
            if (frames[j].hasHeaderValue(pixscale_keyword)):
                pixscale = float(frames[j].getHeaderValue(pixscale_keyword))
            if (frames[j].hasHeaderValue(rotpa_keyword)):
                theta = float(frames[j].getHeaderValue(rotpa_keyword))
            if (frames[j].hasHeaderValue(inport_keyword)):
                inport = float(frames[j].getHeaderValue(inport_keyword))
        print("\tDrihizzling "+name+" with kernel "+kernel+"...")
        write_fatboy_log(log, logtype, "Drihizzling "+name+" with kernel "+kernel+"...", __name__, printCaller=False, tabLevel=1)
        lz = leadZeros(3, numframes)
        newHeader['DHZ'+lz+'FL'] = name
        newHeader['DHZ'+lz+'XS'] = xsh[j]
        newHeader['DHZ'+lz+'YS'] = ysh[j]
        if (_verbosity == fatboyLog.VERBOSE):
            print("\tRead Data: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        if (exptime == -1):
            if (weight == 'exptime'):
                print("gpu_drihizzle> Warning: Could not find exposure time.  Using exptime = 1.0.")
                write_fatboy_log(log, logtype, "Could not find exposure time.  Using exptime = 1.0.", __name__, messageType=fatboyLog.WARNING)
            exptime = 1
        if (weight == 'exptime'):
            scalefac = exptime

        #Handle F1/F2 data without CD matrix
        if (not outExists):
            if (cd11 == 0 and cd12 == 0 and cd12 == 0 and cd22 == 0):
                #CD matrix not set -- F2 data?
                if (pixscale != 0):
                    hasWCS = True
                    pixscale /= 3600.
                    theta = theta*pi/180.
                    if (inport == 2):
                        #side port
                        cd11 = -1*pixscale*sin(theta)
                        cd21 = -1*pixscale*cos(theta)
                        cd12 = -1*pixscale*cos(theta)
                        cd22 = pixscale*sin(theta)
                    else:
                        #up port, F1
                        cd11 = -1*pixscale*cos(theta)
                        cd21 = pixscale*sin(theta)
                        cd12 = -1*pixscale*sin(theta)
                        cd22 = -1*pixscale*cos(theta)

        tmpexp = None
        if (inunits == 'cps'):
            #Convert from counts to cps
            if (mode == MODE_FITS):
                #If a FITS file, look for exposure map passed as argument
                if (isinstance(expmaps, list) and len(expmaps) >= j and os.access(expmaps[j], os.F_OK)):
                    tempexpmap = pyfits.open(expmaps[j])
                    tmpexp = tempexpmap[mef].data
                    data *= tmpexp
                    tempexpmap.close()
                elif (os.access(frames[j].replace('as_','exp_'), os.F_OK)):
                    #Else look for exp_id.fits if frames format is as_id.fits
                    tempexpmap = pyfits.open(frames[j].replace('as_','exp_'))
                    tmpexp = tempexpmap[mef].data
                    data *= tmpexp
                    tempexpmap.close()
                else:
                    print("gpu_drihizzle> Warning: Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.")
                    write_fatboy_log(log, logtype, "Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.", __name__, messageType=fatboyLog.WARNING)
                    data*=exptime
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                #For FDUs, look for a property "exposure_map"
                if (frames[j].hasProperty("exposure_map")):
                    tmpexp = frames[j].getProperty("exposure_map")
                    data *= tmpexp
                else:
                    print("gpu_drihizzle> Warning: Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.")
                    write_fatboy_log(log, logtype, "Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.", __name__, messageType=fatboyLog.WARNING)
                    data*=exptime
            else:
                data*=exptime
            data = data.astype(float32)

        if (j == 0):
            #Setup input mask if not given already
            if (inmask is None):
                inmask = ones(data.shape, int32)
            else:
                inmask = inmask.astype(int32)

            #Setup input arrays to calculate transformation
            #Do this for the first pass only
            #ny = data.shape[0]
            #nx = data.shape[1]
            #nx and ny are set above looping over all images as max size.  blocks should be defined for this frame's size
            blocks = data.size//512
            blocks = nx*ny//512
            if (data.size % 512 != 0):
                blocks += 1

            calcXYin = mod.get_function("calcXYin")
            xin = empty(shape=(ny,nx), dtype=float32)
            yin = empty(shape=(ny,nx), dtype=float32)

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tAllocate input mask/arrays: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            offset = 0.0
            if (kernel == 'tophat' or kernel.find('gauss') != -1 or kernel == 'lanczos'):
                offset = 0.5
            calcXYin(drv.Out(xin), drv.Out(yin), int32(nx), int32(ny), float32(offset-nx//2), float32(offset-ny//2), grid=(blocks,1), block=(block_size,1,1))
            xout = empty(shape=(ny,nx), dtype=float32)
            yout = empty(shape=(ny,nx), dtype=float32)

            #Reference pixels for CRPIX
            if (xrefin == -1):
                xrefin = nx/2+0.
            if (yrefin == -1):
                yrefin = ny/2+0.
            xrefout = 1.
            yrefout = 1.

            #Update WCS if applicable
            if (hasWCS):
                denom = xcoeffs[1]*ycoeffs[2]-xcoeffs[2]*ycoeffs[1]
                new11 = (cd11*ycoeffs[2] + cd21*ycoeffs[1])/denom
                new21 = (cd21*ycoeffs[2] - cd11*ycoeffs[1])/denom
                new12 = (cd12*xcoeffs[1] - cd22*xcoeffs[2])/denom
                new22 = (cd12*xcoeffs[2] + cd22*xcoeffs[1])/denom
                newHeader['CD1_1'] = new11
                newHeader['CD2_2'] = new22
                newHeader['CD1_2'] = new12
                newHeader['CD2_1'] = new21
            newHeader['CRVAL1'] = crval1
            newHeader['CRVAL2'] = crval2

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCalc input arrays: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()
            #Compute transformation
            #Do this for the first pass only
            calcTransOpt = mod.get_function("calcTransOpt")
            calcTransOpt(drv.InOut(xout), drv.InOut(yout), drv.In(xin), drv.In(yin), drv.In(xcoeffs), drv.In(ycoeffs), int32(order), float32(offset), int32(nx*ny), grid=(blocks,1), block=(block_size,1,1))
            #Compute transformation on refrerence pixels in python
            n = 0
            for l in range(order+1):
                for k in range(l+1):
                    xrefout+=xcoeffs[n]*xrefin**(l-k)*yrefin**k
                    yrefout+=ycoeffs[n]*xrefin**(l-k)*yrefin**k
                    n+=1

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCalc transformation: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            if (xtrans is not None):
                xout = xtrans.astype(float32)
            if (ytrans is not None):
                yout = ytrans.astype(float32)

            #Release memory
            del xin
            del yin

            #Apply scaling if necessary
            if (scale != 1):
                xout *= scale
                yout *= scale
                xsh = array(xsh)*scale
                ysh = array(ysh)*scale
                xrefout *= scale
                yrefout *= scale

            #Calculate mins, maxes
            #xmin = int(floor(xout[:,0].min()))
            #xmax = int(floor(xout[:,-1].max()))+2
            #ymin = int(floor(yout[0,:].min()))
            #ymax = int(floor(yout[-1,:].max()))+2
            xmin = int(floor(xout.min()))
            xmax = int(floor(xout.max()))+2
            ymin = int(floor(yout.min()))
            ymax = int(floor(yout.max()))+2
            #xmin = int(floor(xout[inmask != 0].min()))
            #xmax = int(floor(xout[inmask != 0].max()))+2
            #ymin = int(floor(yout[inmask != 0].min()))
            #ymax = int(floor(yout[inmask != 0].max()))+2

            if (kernel == 'tophat' or kernel.find('gauss') != -1 or kernel == 'lanczos'):
                xmin -= int(dropsize)
                xmax += int(dropsize)
                ymin -= int(dropsize)
                ymax += int(dropsize)

            xshmin = min(xsh)
            yshmin = min(ysh)
            xshrange = int(ceil(max(xsh)-min(xsh)))
            yshrange = int(ceil(max(ysh)-min(ysh)))
            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCalc min/max output: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            #xsh[0] and ysh[0] may not be zero!  We shift image by this pixel amount so WCS needs to be shifted by same amount!
            xrefout += xsh[0]-xshmin
            yrefout += ysh[0]-yshmin

            #Subtract mins from xout, yout
            xout -= xmin
            yout -= ymin
            if (_verbosity == fatboyLog.VERBOSE):
                print("\tSubtract min/maxes: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            #Create new image array only the first time through
            #Take into account max shifts
            newdata = zeros((ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
            expmap = zeros((ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
            if (doPix):
                pixmap = zeros((ymax-ymin+yshrange, xmax-xmin+xshrange), int32)

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCreate new image array: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()
            #Handle existing data
            if (outExists):
                outxsh = 0.
                outysh = 0.
                if ('DHZMINXS' in outimage[0].header):
                    outxsh = float(outimage[0].header['DHZMINXS'])
                if ('DHZMINYS' in outimage[0].header):
                    outysh = float(outimage[0].header['DHZMINYS'])
                xrefout = float(outimage[0].header['CRPIX1'])
                yrefout = float(outimage[0].header['CRPIX2'])
                #Case exisiting image is as large or larger than needed
                if (outxsh <= xshmin and outysh <= yshmin and outimage[mef].data.shape[1]+outxsh >= xmax-xmin+max(xsh) and outimage[mef].data.shape[0] >= ymax-ymin+max(ysh)):
                    newdata = outimage[mef].data.astype(float32)
                    expmap = outexp[mef].data.astype(float32)
                    if (doPix):
                        pixmap = outpix[mef].data.astype(int32)
                    xsh = array(xsh)-outxsh
                    ysh = array(ysh)-outysh
                    xshmin -= outxsh
                    yshmin -= outysh
                else:
                    #Need to enlarge image
                    xshmin = min(xshmin, outxsh)
                    yshmin = min(yshmin, outysh)
                    xmax = max(max(xsh)+xmax-xmin, outxsh+outimage[mef].data.shape[1])
                    ymax = max(max(ysh)+ymax-ymin, outysh+outimage[mef].data.shape[0])
                    x_range = int(ceil(xmax-xshmin))
                    y_range = int(ceil(ymax-yshmin))
                    newdata = zeros((y_range, x_range), float32)
                    expmap = zeros((y_range, x_range), float32)
                    if (doPix):
                        pixmap = zeros((y_range, x_range), int32)
                    if (outxsh != xshmin):
                        xshmin = floor(xshmin)
                    x1 = int(outxsh - xshmin)
                    x2 = x1+outimage[mef].data.shape[1]
                    xrefout += x1
                    if (outysh != yshmin):
                        yshmin = floor(yshmin)
                    y1 = int(outysh - yshmin)
                    y2 = y1+outimage[mef].data.shape[0]
                    yrefout += y1
                    newdata[y1:y2,x1:x2] = outimage[mef].data.astype(float32)
                    expmap[y1:y2,x1:x2] = outexp[mef].data.astype(float32)
                    if (doPix):
                        pixmap[y1:y2,x1:x2] = outpix[mef].data.asytpe(int32)

            #Update reference min shifts, ref pixels in image header
            newHeader['DHZMINXS'] = xshmin
            newHeader['DHZMINYS'] = yshmin
            newHeader['CRPIX1'] = xrefout
            newHeader['CRPIX2'] = yrefout

            #Copy over to pyfits ojects
            if (outExists):
                outimage[mef].data = newdata.astype(float32)
                outexp[mef].data = expmap.astype(float32)
                if (doPix):
                    outpix[mef].data = pixmap.astype(int32)
            else:
                #Create output arrays
                imagedata = zeros((ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
                expdata = zeros((ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
                if (doPix):
                    pixdata = zeros((ymax-ymin+yshrange, xmax-xmin+xshrange), int32)

        if (_verbosity == fatboyLog.VERBOSE):
            print("\tCopy over to pyfits and update: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        #Now set blocks based on data size
        blocks = data.size//512
        if (data.size % 512 != 0):
            blocks += 1
        #Scale data by inmask and weight factor
        multArrFloatIntScalar = mod.get_function("multArrFloatIntScalar")
        multArrFloatIntScalar(drv.InOut(data), drv.In(inmask), float32(scalefac/exptime), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
        if (tmpexp is None):
            #Exposure map should be exposure time * good pixel mask unless a previous exposure map has been loaded for inunits = cps
            tmpexp = float32(inmask*scalefac)
        totexp+=exptime

        if (_verbosity == fatboyLog.VERBOSE):
            print("\tScale data and expmask: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        #Ensure xout, yout shape matches data before sending to GPU
        if (xout.shape == data.shape):
            xout_g = xout
            yout_g = yout
        else:
            xout_g = ascontiguousarray(xout[:data.shape[0], :data.shape[1]])
            yout_g = ascontiguousarray(yout[:data.shape[0], :data.shape[1]])

        xsize = newdata.shape[1]
        if (kernel == 'turbo'):
            if (doPix):
                turboKernel = mod.get_function("turboKernelPix")
                turboKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
            else:
                turboKernel = mod.get_function("turboKernel")
                turboKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'point'):
            if (doPix):
                pointKernel = mod.get_function("pointKernelPix")
                pointKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
            else:
                pointKernel = mod.get_function("pointKernel")
                pointKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'tophat'):
            if (doPix):
                tophatKernel = mod.get_function("tophatKernelPix")
                tophatKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
            else:
                tophatKernel = mod.get_function("tophatKernel")
                tophatKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'gaussian'):
            if (doPix):
                gaussianKernel = mod.get_function("gaussianKernelPix")
                gaussianKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
            else:
                gaussianKernel = mod.get_function("gaussianKernel")
                gaussianKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'fastgauss'):
            if (doPix):
                fastGaussKernel = mod.get_function("fastGaussKernelPix")
                fastGaussKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), drv.In(gausslut), int32(gausscen), grid=(blocks,1), block=(block_size,1,1))
            else:
                fastGaussKernel = mod.get_function("fastGaussKernel")
                fastGaussKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), drv.In(gausslut), int32(gausscen), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'lanczos'):
            #Flux weighted by 2-D lanczos sinc function with width
            #determined by dropsize.
            #1D lookup table with points every 0.001 pixels used to estimate function
            #(same as in IRAF's drizzle but IRAF only uses 0.01 pixel accuracy)
            #rather than calculating function for every pixel.
            if (doPix):
                lanczosKernel = mod.get_function("lanczosKernelPix")
                lanczosKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), drv.In(lanclut), int32(lanccen), grid=(blocks,1), block=(block_size,1,1))
            else:
                lanczosKernel = mod.get_function("lanczosKernel")
                lanczosKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), int32(xsize), int32(data.size), float32(dropsize), drv.In(lanclut), int32(lanccen), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'uniform'):
            data = int32(data)
            newdata = int32(newdata)
            imagedata = int32(imagedata)
            nxsize = newdata.shape[1]
            idx = xout.astype(int32)+nxsize*yout.astype(int32)
            uniformKernel = mod.get_function("uniformKernel")
            uniformKernel(drv.InOut(newdata), drv.In(data), drv.In(xout_g), drv.In(yout_g), drv.In(inmask), int32(nxsize), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
        tt = time.time()

        #If requested, update FDUs here
        if (updateFDUs and (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG)):
            #Make copies with astype()
            drihizzled_data = newdata.astype(float32)
            expmap_data = expmap.astype(float32)
            divFloatArrays = mod.get_function("divFloatArrays")
            #Apply weighting
            blocks = newdata.size//block_size
            if (newdata.size % block_size != 0):
                blocks += 1
            blocky = 1
            while(blocks > 65535):
                blocky *= 2
                blocks //= 2
            #Apply weighting
            if (weight == 'exptime'):
                if (outunits == 'cps'):
                    divFloatArrays(drv.InOut(drihizzled_data), drv.In(expmap_data), int32(drihizzled_data.size), grid=(blocks, blocky), block=(block_size,1,1))
                else:
                    divFloatArrays(drv.InOut(drihizzled_data), drv.In(expmap_data), int32(drihizzled_data.size), grid=(blocks, blocky), block=(block_size,1,1))
                    if (outunits == 'counts'):
                        drihizzled_data*=exptime
            frames[j].tagDataAs("drihizzled", data=drihizzled_data)
            frames[j].tagDataAs("exposure_map", data=expmap_data)

        #If keeping individual images, write them out here
        if (keepImages.lower() == 'yes'):
            if (not os.access(imgdir, os.F_OK)):
                imgdir = ''
            if (imgdir != '' and imgdir[-1] != '/'):
                imgdir += '/'
            if (mode == MODE_FITS):
                shortfn = frames[j][frames[j].rfind('/')+1:]
                temp = pyfits.open(frames[j])
            elif (mode == MODE_RAW):
                shortfn = 'frame'+str(j)+'.fits'
                temp = pyfits.HDUList()
                temp.append(pyfits.PrimaryHDU())
            elif (mode == MODE_FDU):
                shortfn = frames[j].getFullId()
                temp = pyfits.open(frames[j].getFilename())
            elif (mode == MODE_FDU_DIFFERENCE):
                shortfn = frames[j+1]._id+frames[j+1]._index+"-"+frames[j]._index+".fits"
                temp = pyfits.open(frames[j+1].getFilename())
            elif (mode == MODE_FDU_TAG):
                shortfn = frames[j]._id+"_"+dataTag+frames[j]._index+".fits"
                temp = pyfits.open(frames[j].getFilename())
            temp[mef].data = newdata.astype(float32)
            tempexpmap = expmap.astype(float32)
            divFloatArrays = mod.get_function("divFloatArrays")
            #Apply weighting
            blocks = newdata.size//block_size
            if (newdata.size % block_size != 0):
                blocks += 1
            blocky = 1
            while(blocks > 65535):
                blocky *= 2
                blocks //= 2
            if (weight == 'exptime'):
                if (outunits == 'cps'):
                    divFloatArrays(drv.InOut(temp[mef].data), drv.In(tempexpmap), int32(temp[mef].data.size), grid=(blocks, blocky), block=(block_size,1,1))
                else:
                    divFloatArrays(drv.InOut(temp[mef].data), drv.In(tempexpmap), int32(temp[mef].data.size), grid=(blocks, blocky), block=(block_size,1,1))
                    if (outunits == 'counts'):
                        temp[mef].data*=exptime
            temp.verify('silentfix')
            if (os.access(imgdir+'drihiz_'+shortfn, os.F_OK)):
                os.unlink(imgdir+'drihiz_'+shortfn)
            if (os.access(imgdir+'expmap_'+shortfn, os.F_OK)):
                os.unlink(imgdir+'expmap_'+shortfn)
            #Update header info
            updateHeader(temp[0].header, newHeader)
            #Update WCS if applicable
            if (hasWCS):
                if (j != 0):
                    denom = xcoeffs[1]*ycoeffs[2]-xcoeffs[2]*ycoeffs[1]
                    new11 = (cd11*ycoeffs[2] + cd21*ycoeffs[1])/denom
                    new21 = (cd21*ycoeffs[2] - cd11*ycoeffs[1])/denom
                    new12 = (cd12*xcoeffs[1] - cd22*xcoeffs[2])/denom
                    new22 = (cd12*xcoeffs[2] + cd22*xcoeffs[1])/denom
                updateHeaderEntry(temp[0].header, 'CD1_1', new11)
                updateHeaderEntry(temp[0].header, 'CD2_2', new22)
                updateHeaderEntry(temp[0].header, 'CD1_2', new12)
                updateHeaderEntry(temp[0].header, 'CD2_1', new21)
            #Write out new files
            #Get rid of extraneous extensions in data like CIRCE/Newfirm
            prepMefForWriting(temp, mef)
            temp.writeto(imgdir+'drihiz_'+shortfn, output_verify='silentfix')
            temp[mef].data = expmap
            temp.writeto(imgdir+'expmap_'+shortfn, output_verify='silentfix')
            temp.close()
            if (_verbosity == fatboyLog.VERBOSE):
                print("\tWrite Indiv Images: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

        ##Update overall images
        if (outExists):
            outimage[mef].data += newdata
            outexp[mef].data += expmap
            if (doPix):
                outpix[mef].data += pixmap
        else:
            imagedata += newdata
            expdata += expmap
            if (doPix):
                pixdata += pixmap

        #Reset arrays
        if (j < nframes-1):
            newdata[:,:] = 0.
            expmap[:,:] = 0.
        if (doPix):
            if (j < nframes-1):
                pixmap[:,:] = 0

        #GUI message:
        if (gui is not None):
            gui = (gui[0], gui[1]+1., gui[2], gui[3], gui[4])
            if (gui[0]): print("PROGRESS: "+str(int(gui[3]+gui[1]/gui[2]*gui[4])))

    if (_verbosity == fatboyLog.VERBOSE):
        print("Process indiv frames: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()
    #Free memory
    del xout
    del yout
    del inmask
    del data
    del tmpexp

    if (outExists):
        imagedata = outimage[mef].data
        expdata = outexp[mef].data

    blocks = imagedata.size//block_size
    if (imagedata.size % block_size != 0):
        blocks += 1
    blocky = 1
    while(blocks > 65535):
        blocky *= 2
        blocks //= 2

    divFloatArrays = mod.get_function("divFloatArrays")
    #Apply weighting
    if (weight == 'exptime' and kernel != 'uniform'):
        if (outunits == 'cps'):
            divFloatArrays(drv.InOut(imagedata), drv.In(expdata), int32(imagedata.size), grid=(blocks,blocky), block=(block_size,1,1))
    elif (kernel != 'uniform'):
        divFloatArrays(drv.InOut(imagedata), drv.In(expdata), int32(imagedata.size), grid=(blocks,blocky), block=(block_size,1,1))
        if (outunits == 'counts'):
            imagedata *= totexp

    if (_verbosity == fatboyLog.VERBOSE):
        print("Apply Weighting: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Update header
    newHeader['DHZUNITS'] = outunits
    newHeader[expkey] = totexp
    newHeader['DHZKERNL'] = kernel
    newHeader['DHZNFRMS'] = numframes

    #output file
    if (outfile is not None):
        print("\tOutput file: "+outfile)
        write_fatboy_log(log, logtype, "\tOutput file: "+outfile, __name__, printCaller=False, tabLevel=1)
        if (not outExists):
            if (mode == MODE_FITS):
                outimage = pyfits.open(frames[0])
            elif (mode == MODE_RAW):
                hdu = pyfits.PrimaryHDU(outtype(out))
                outimage = pyfits.HDUList([hdu])
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                outimage = pyfits.open(frames[0].filename)
        outimage[mef].data = imagedata
        #update header
        updateHeader(outimage[0].header, newHeader)
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(outimage, mef)
        if (outExists):
            outimage.flush()
        else:
            outimage.writeto(outfile, output_verify='silentfix')
        outimage.close()
        del outimage
    #Exp map file
    if (weightfile is not None):
        print("\tWeight file: "+weightfile)
        write_fatboy_log(log, logtype, "\tWeight file: "+weightfile, __name__, printCaller=False, tabLevel=1)
        if (not outExists):
            if (mode == MODE_FITS):
                outexp = pyfits.open(frames[0])
            elif (mode == MODE_RAW):
                hdu = pyfits.PrimaryHDU(outtype(out))
                outexp = pyfits.HDUList([hdu])
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                outexp = pyfits.open(frames[0].filename)
        outexp[mef].data = expdata
        #update header
        updateHeader(outexp[0].header, newHeader)
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(outexp, mef)
        if (outExists):
            outexp.flush()
        else:
            outexp.writeto(weightfile, output_verify='silentfix')
        outexp.close()
        del outexp
    if (pixfile is not None):
        print("\tPix file: "+pixfile)
        write_fatboy_log(log, logtype, "\tPix file: "+pixfile, __name__, printCaller=False, tabLevel=1)
        if (not outExists):
            if (mode == MODE_FITS):
                outpix = pyfits.open(frames[0])
            elif (mode == MODE_RAW):
                hdu = pyfits.PrimaryHDU(outtype(out))
                outpix = pyfits.HDUList([hdu])
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                outpix = pyfits.open(frames[0].filename)
        outpix[mef].data = pixdata
        #update header
        updateHeader(outpix[0].header, newHeader)
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(outpix, mef)
        if (outExists):
            outpix.flush()
        else:
            outpix.writeto(pixfile, output_verify='silentfix')
        outpix.close()
        del outpix
    if (_verbosity == fatboyLog.VERBOSE):
        print("Write data: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()
    print("Drihizzled "+str(nframes)+" files with "+kernel+" kernel.  Total time (s): "+str(time.time()-t))
    write_fatboy_log(log, logtype, "Drihizzled "+str(nframes)+" files with "+kernel+" kernel.  Total time (s): "+str(time.time()-t), __name__)
    #Return 4-tuple
    return (imagedata, newHeader, expdata, pixdata)

def getGaussLut(fwhm, nsig, inc):
    endx = fwhm*nsig*2/2.3548
    sz = int(ceil(endx/inc))*2
    x = (arange(sz,dtype=float32)-sz//2)*inc
    z = 2.3548/fwhm*x
    y = exp(-0.5*z*z)
    y[x > endx/2] = 0.
    y[x < -endx/2] = 0.
    return y

def getLanczosLut(order, inc):
    endx = order*2
    sz = int(ceil(endx/inc))*2
    x = (arange(sz,dtype=float32)-sz//2)*inc
    y = zeros(x.shape, float32)
    y[x == 0] = 1
    b = (x != 0)
    y[b] = order*sin(pi*x[b]) * sin(pi*x[b]/order) / (pi**2*x[b]*x[b])
    y[x >= order] = 0.
    y[x <+ -order] = 0.
    return y

def drihizzle3d(frames, outfile=None, weightfile=None, inmask=None, weight='exptime', kernel='point', dropsize=1, geomDist=None, xsh=None, ysh=None, zsh=None, inunits='counts', outunits='cps', expkey='EXP_TIME', keepImages='no', imgdir='', log=None, gui=None, xtrans=None, ytrans=None, ztrans=None, mef=0, inmef=None, pixfile=None, doPix=True, scale=1, mode=None, returnHeader=False, dataTag=None, pixscale_keyword='PIXSCALE', rotpa_keyword='ROT_PA', inport_keyword='INPORT', updateFDUs=False, expmaps=None):
    #frames = input data -- filename, list of filenames, or array
    #outfile = output data
    #weightfile = output exposure map
    #inmask = input good pixel mask
    #weight = weighting -- can be a number or 'exptime'
    #kernel = drizzle kernel
    #dropsize = drizzle dropsize
    #geomDist = file with geometric distortion correction parameters
    #xsh = x shifts (number or list)
    #ysh = y shifts (number or list)
    #zsh = z shifts (number or list)
    #inunits = input units, counts or cps
    #outunits = output units, counts or cps
    #expkey = FITS keyword for exposure time
    #keepImages = keep individual output frames
    #imgdir = directory to store individual output frames in
    #logfile = logfile to be appended
    #xtrans = x-transformation to be applied.  overrides geomDist.
    #ytrans = y-transformation to be applied.  overrides geomDist.
    #ztrans = z-transformation to be applied.  overrides geomDist.
    #mef = fits extension
    #inmef = fits extension for input mask
    #pixfile = output pixel map (number of input pixels contributing to each output pixel)
    #scale = the factor of subsampling to do in the output image.  scale = 2 means each input pixel is a 2x2 grid in output.
    #mode = mode - FITS files, raw data, FDUs
    #returnHeader = Create new dict that can be used to update header
    #dataTag = for MODE_FDU_TAG
    #pixscale_keyword = keyword for pixel scale
    #rotpa_keyword = keyword for rotation angle
    #inport_keyword = keyword for inport (F2)
    #updateFDUs = tag FDUs with results and expmap to keep in memory
    #expmaps = list of input exposure maps for use with 'cps' inputs

    t = time.time()
    print("Drihizzle's the bizzle fo' shizzle!")
    _verbosity = fatboyLog.NORMAL

    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()

    #set log type
    logtype = LOGTYPE_NONE
    if (log is not None):
        if (isinstance(log, str)):
            #log given as a string
            log = open(log,'a')
            logtype = LOGTYPE_ASCII
        elif(isinstance(log, fatboyLog)):
            logtype = LOGTYPE_FATBOY
            _verbosity = fatboyLog._verbosity

    #Filelist
    if (isinstance(frames, str) and os.access(frames, os.F_OK)):
        frames = readFileIntoList(frames)
    elif (isinstance(frames, ndarray)):
        frames = [frames]
    elif (not isinstance(frames, list)):
        frames = [frames]

    nframes = len(frames)
    if (mode == MODE_FDU_DIFFERENCE):
        nframes -= 1

    #Find type
    if (mode is None):
        mode = MODE_FITS
        if (isinstance(frames[0], str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], ndarray)):
            mode = MODE_RAW
        elif (isinstance(frames[0], fatboyDataUnit)):
            mode = MODE_FDU

    if (gui is not None):
        if (len(gui) != 5): gui = None

    if (inmef is None):
        inmef = mef

    if (pixfile is not None):
        doPix = True

    if (outfile is not None and os.access(outfile, os.F_OK)):
        outExists = True
        if (weightfile is None or not os.access(weightfile, os.F_OK)):
            print("gpu_drihizzle3d> Error: Weighting image must exist if output image exists.")
            write_fatboy_log(log, logtype, "Weighting image must exist if output image exists.", __name__, messageType=fatboyLog.ERROR)
            return None
        if (doPix and (pixfile is None or not os.access(pixfile, os.F_OK))):
            print("gpu_drihizzle3d> Error: If output image exists and a pixmap is desired, pixmap must also exist.")
            write_fatboy_log(log, logtype, "If output image exists and a pixmap is desired, pixmap must also exist.", __name__, messageType=fatboyLog.ERROR)
            return None
    else:
        outExists = False
        if (weightfile is not None and os.access(weightfile, os.F_OK)):
            os.unlink(weightfile)
        if (doPix and pixfile is not None and os.access(pixfile, os.F_OK)):
            os.unlink(pixfile)

    doIndividualMasks = False
    if (isinstance(inmask, int) and inmask == MODE_FDU_USE_INDIVIDUAL_GPMS):
        doIndividualMasks = True
        print("gpu_drihizzle3d> Using individual good pixel masks.")
        write_fatboy_log(log, logtype, "Using individual good pixel masks.", __name__)
    elif (isinstance(inmask, str)):
        temp = pyfits.open(inmask)
        inmask = temp[inmef].data
        temp.close()
        del temp
    elif (isinstance(inmask, fatboyDataUnit)):
        inmask = inmask.getData()
    elif (not isinstance(inmask, ndarray)):
        inmask = None

    if (dropsize <= 0.01 and kernel != 'point'):
        kernel = 'point'
        print("gpu_drihizzle3d> Dropsize <= 0.01.  Using the point kernel.")
        write_fatboy_log(log, logtype, "Dropsize <= 0.01.  Using the point kernel.", __name__)
    if (dropsize > 1 and kernel == 'turbo'):
        dropsize = 1
        print("gpu_drihizzle3d> Dropsize > 1 not allowed for "+kernel+" kernel.  Using dropsize = 1.")
        write_fatboy_log(log, logtype, "Dropsize > 1 not allowed for "+kernel+" kernel.  Using dropsize = 1.", __name__)

    xcoeffs = []
    ycoeffs = []
    zcoeffs = []

    #Read geom distortion file or parse list
    if (geomDist is not None):
        if (isinstance(geomDist, str) and os.access(geomDist, os.F_OK)):
            f = open(geomDist, 'r')
            temp = f.read().split('\n')
            f.close()
            temp.pop(0)
            currCoeff = temp.pop(0)
            while (currCoeff != ''):
                xcoeffs.append(float(currCoeff))
                currCoeff = temp.pop(0)
            currCoeff = temp.pop(0)
            while (currCoeff != ''):
                ycoeffs.append(float(currCoeff))
                currCoeff = temp.pop(0)
            while (len(temp) > 0):
                currCoeff = temp.pop(0)
                if (currCoeff != ''):
                    zcoeffs.append(float(currCoeff))
        elif (isinstance(geomDist, list)):
            if (len(geomDist) == 3):
                xcoeffs = geomDist[0]
                ycoeffs = geomDist[1]
                zcoeffs = geomDist[2]
        else:
            xcoeffs = [0,1,0,0]
            ycoeffs = [0,0,1,0]
            zcoeffs = [0,0,0,1]
    else:
        #xin = xout, yin = yout
        xcoeffs = [0,1,0,0]
        ycoeffs = [0,0,1,0]
        zcoeffs = [0,0,0,1]

    xcoeffs = array(xcoeffs).astype(float32)
    ycoeffs = array(ycoeffs).astype(float32)
    zcoeffs = array(zcoeffs).astype(float32)

    ncoeff = len(xcoeffs)
    order = 0
    while (ncoeff > 0):
        ncoeff -= (order+1)*(order+2)/2
        if (ncoeff > 0): order+=1

    #Shifts
    if (xsh is None):
        xsh = zeros(nframes)
    elif (not isinstance(xsh, list) and not isinstance(xsh, ndarray)):
        xsh = [xsh]

    if (ysh is None):
        ysh = zeros(nframes)
    elif (not isinstance(ysh, list) and not isinstance(ysh, ndarray)):
        ysh = [ysh]

    if (zsh is None):
        zsh = zeros(nframes)
    elif (not isinstance(zsh, list) and not isinstance(zsh, ndarray)):
        zsh = [zsh]

    if (_verbosity == fatboyLog.VERBOSE):
        print("Initialize: ",time.time()-t)
    tt = time.time()
    #Setup output image structure
    totexp = 0.
    numframes = 0
    hasWCS = False
    newHeader = dict() #setup new header

    #output file exists, open for updating
    if (outExists):
        outimage = pyfits.open(outfile, 'update')
        outexp = pyfits.open(weightfile, 'update')
        if ('DHZUNITS' in outimage[0].header):
            if (outimage[0].header['DHZUNITS'] == 'cps'):
                outimage[mef].data *= outexp[mef].data
            if (outimage[0].header['DHZUNITS'] == 'cps' and expkey in outimage[0].header):
                #outimage[mef].data*=float(outimage[0].header[expkey])
                totexp = float(outimage[0].header[expkey])
        if ('DHZNFRMS' in outimage[0].header):
            numframes = int(outimage[0].header['DHZNFRMS'])
        if (doPix):
            outpix = pyfits.open(pixfile, 'update')

    if (_verbosity == fatboyLog.VERBOSE):
        print("Open Files: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    ##Now start nx and ny at 0 and compare with size of each frame to get max size
    nx = 0
    ny = 0
    nz = 0
    for j in range(nframes):
        if (mode == MODE_FITS):
            temp = pyfits.open(frames[j])
            if ('NAXIS3' in temp[mef].header):
                nz = max(nz, int(temp[mef].header['NAXIS3']))
            else:
                nz = max(nz, temp[mef].data.shape[0])
            if ('NAXIS2' in temp[mef].header):
                ny = max(ny, int(temp[mef].header['NAXIS2']))
            else:
                ny = max(ny, temp[mef].data.shape[1])
            if ('NAXIS1' in temp[mef].header):
                nx = max(nx, int(temp[mef].header['NAXIS1']))
            else:
                nx = max(nx, temp[mef].data.shape[2])
        elif (mode == MODE_RAW):
            nz = max(nz, frames[j].shape[0])
            ny = max(ny, frames[j].shape[1])
            nx = max(nx, frames[j].shape[2])
        elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE):
            nz = max(nz, frames[j].getShape()[0])
            ny = max(ny, frames[j].getShape()[1])
            nx = max(nx, frames[j].getShape()[2])
        elif (mode == MODE_FDU_TAG):
            nz = max(nz, frames[j].getData(tag=dataTag).shape[0])
            ny = max(ny, frames[j].getData(tag=dataTag).shape[1])
            nx = max(nx, frames[j].getData(tag=dataTag).shape[2])

    for j in range(nframes):
        #Set weighting factor if not FITS keyword
        numframes+=1
        exptime = 1.0
        scalefac = 1.0
        if (not isinstance(weight, str)):
            scalefac = weight
        xrefin = -1
        yrefin = -1
        zrefin = -1

        exptime = -1
        #wcs and FITS header
        xrefin = 0
        yrefin = 0
        zrefin = 0
        crval1 = 0
        crval2 = 0
        crval3 = 0
        cd11 = 0
        cd12 = 0
        cd21 = 0
        cd22 = 0
        pixscale = 0
        theta = 0
        inport = 0

        #Read data and wcs info
        if (mode == MODE_FITS):
            name = frames[j]
            temp = pyfits.open(frames[j])
            data = temp[mef].data.astype(float32)
            if (expkey in temp[0].header):
                exptime = float(temp[0].header[expkey])
            if ('CRPIX1' in temp[0].header):
                xrefin = float(temp[0].header['CRPIX1'])-1
            if ('CRPIX2' in temp[0].header):
                yrefin = float(temp[0].header['CRPIX2'])-1
            if (j == 0):
                if ('CRVAL1' in temp[0].header):
                    crval1 = float(temp[0].header['CRVAL1'])
                if ('CRVAL2' in temp[0].header):
                    crval2 = float(temp[0].header['CRVAL2'])
            if ('CD1_1' in temp[0].header):
                hasWCS = True
                cd11 = float(temp[0].header['CD1_1'])
            if ('CD1_2' in temp[0].header):
                cd12 = float(temp[0].header['CD1_2'])
            if ('CD2_1' in temp[0].header):
                cd21 = float(temp[0].header['CD2_1'])
            if ('CD2_2' in temp[0].header):
                cd22 = float(temp[0].header['CD2_2'])
            if (pixscale_keyword in temp[0].header):
                pixscale = float(temp[0].header[pixscale_keyword])
            if (rotpa_keyword in temp[0].header):
                theta = float(temp[0].header[rotpa_keyword])
            if (inport_keyword in temp[0].header):
                inport = float(temp[0].header[inport_keyword])
            temp.close()
        elif (mode == MODE_RAW):
            name = "frame number "+str(j)
            data = frames[j].astype(float32)
        elif (mode == MODE_FDU):
            name = frames[j].getFullId()
            data = frames[j].getData().astype(float32)
        elif (mode == MODE_FDU_DIFFERENCE):
            name = frames[j+1].getFullId()+"-"+frames[j].getFullId()
            data = frames[j+1].getData()-frames[j].getData()
        elif (mode == MODE_FDU_TAG):
            name = frames[j].getFullId()+":"+dataTag
            data = frames[j].getData(tag=dataTag).astype(float32)
        if (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
            if (doIndividualMasks):
                inmask = (1-frames[j].getBadPixelMask().getData()).astype("int32")
            exptime = frames[j].exptime
            if (frames[j].hasHeaderValue('CRPIX1')):
                xrefin = float(frames[j].getHeaderValue('CRPIX1'))
            if (frames[j].hasHeaderValue('CRPIX2')):
                yrefin = float(frames[j].getHeaderValue('CRPIX2'))
            if (j == 0):
                if (frames[j].hasHeaderValue('CRVAL1')):
                    crval1 = float(frames[j].getHeaderValue('CRVAL1'))
                if (frames[j].hasHeaderValue('CRVAL2')):
                    crval2 = float(frames[j].getHeaderValue('CRVAL2'))
            if (frames[j].hasHeaderValue('CD1_1')):
                hasWCS = True
                cd11 = float(frames[j].getHeaderValue('CD1_1'))
            if (frames[j].hasHeaderValue('CD1_2')):
                cd12 = float(frames[j].getHeaderValue('CD1_2'))
            if (frames[j].hasHeaderValue('CD2_1')):
                cd21 = float(frames[j].getHeaderValue('CD2_1'))
            if (frames[j].hasHeaderValue('CD2_2')):
                cd22 = float(frames[j].getHeaderValue('CD2_2'))
            if (frames[j].hasHeaderValue(pixscale_keyword)):
                pixscale = float(frames[j].getHeaderValue(pixscale_keyword))
            if (frames[j].hasHeaderValue(rotpa_keyword)):
                theta = float(frames[j].getHeaderValue(rotpa_keyword))
            if (frames[j].hasHeaderValue(inport_keyword)):
                inport = float(frames[j].getHeaderValue(inport_keyword))
        print("\tDrihizzling "+name+" with kernel "+kernel+"...")
        write_fatboy_log(log, logtype, "Drihizzling "+name+" with kernel "+kernel+"...", __name__, printCaller=False, tabLevel=1)
        lz = leadZeros(3, numframes)
        newHeader['DHZ'+lz+'FL'] = name
        newHeader['DHZ'+lz+'XS'] = xsh[j]
        newHeader['DHZ'+lz+'YS'] = ysh[j]
        newHeader['DHZ'+lz+'ZS'] = zsh[j]
        if (_verbosity == fatboyLog.VERBOSE):
            print("\tRead Data: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        if (exptime == -1):
            print("gpu_drihizzle3d> Warning: Could not find exposure time.  Using exptime = 1.0.")
            write_fatboy_log(log, logtype, "Could not find exposure time.  Using exptime = 1.0.", __name__, messageType=fatboyLog.WARNING)
            exptime = 1
        if (weight == 'exptime'):
            scalefac = exptime

        tmpexp = None
        if (inunits == 'cps'):
            #Convert from counts to cps
            if (mode == MODE_FITS):
                #If a FITS file, look for exposure map passed as argument
                if (isinstance(expmaps, list) and len(expmaps) >= j and os.access(expmaps[j], os.F_OK)):
                    tempexpmap = pyfits.open(expmaps[j])
                    tmpexp = tempexpmap[mef].data
                    data *= tmpexp
                    tempexpmap.close()
                elif (os.access(frames[j].replace('as_','exp_'), os.F_OK)):
                    #Else look for exp_id.fits if frames format is as_id.fits
                    tempexpmap = pyfits.open(frames[j].replace('as_','exp_'))
                    tmpexp = tempexpmap[mef].data
                    data *= tmpexp
                    tempexpmap.close()
                else:
                    print("gpu_drihizzle3d> Warning: Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.")
                    write_fatboy_log(log, logtype, "Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.", __name__, messageType=fatboyLog.WARNING)
                    data*=exptime
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                #For FDUs, look for a property "exposure_map"
                if (frames[j].hasProperty("exposure_map")):
                    tmpexp = frames[j].getProperty("exposure_map")
                    data *= tmpexp
                else:
                    print("gpu_drihizzle3d> Warning: Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.")
                    write_fatboy_log(log, logtype, "Could not find exposure map.  Using exptime = "+str(exptime)+" to convert from cps to counts.", __name__, messageType=fatboyLog.WARNING)
                    data*=exptime
            else:
                data*=exptime
            data = data.astype(float32)

        if (j == 0):
            #Setup input mask if not given already
            if (inmask is None):
                inmask = ones(data.shape, int32)
            else:
                inmask = inmask.astype(int32)

            #Setup input arrays to calculate transformation
            #Do this for the first pass only
            #ny = data.shape[0]
            #nx = data.shape[1]
            #nx and ny are set above looping over all images as max size.  blocks should be defined for this frame's size
            blocks = data.size//512
            blocks = nx*ny*nz//512
            if (data.size % 512 != 0):
                blocks += 1

            calcXYZin = mod.get_function("calcXYZin")
            xin = empty(shape=(nz,ny,nx), dtype=float32)
            yin = empty(shape=(nz,ny,nx), dtype=float32)
            zin = empty(shape=(nz,ny,nx), dtype=float32)

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tAllocate input mask/arrays: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            offset = 0.0
            calcXYZin(drv.Out(xin), drv.Out(yin), drv.Out(zin), int32(nx), int32(ny), int32(nz), float32(offset-nx//2), float32(offset-ny//2), float32(offset-nz//2), grid=(blocks,1), block=(block_size,1,1))
            xout = empty(shape=(nz,ny,nx), dtype=float32)
            yout = empty(shape=(nz,ny,nx), dtype=float32)
            zout = empty(shape=(nz,ny,nx), dtype=float32)

            #Reference pixels for CRPIX
            if (xrefin == -1):
                xrefin = nx/2+0.
            if (yrefin == -1):
                yrefin = ny/2+0.
            if (zrefin == -1):
                zrefin = nz/2+0.
            xrefout = 1.
            yrefout = 1.
            zrefout = 1.

            #Update WCS if applicable
            if (hasWCS):
                denom = xcoeffs[1]*ycoeffs[2]-xcoeffs[2]*ycoeffs[1]
                new11 = (cd11*ycoeffs[2] + cd21*ycoeffs[1])/denom
                new21 = (cd21*ycoeffs[2] - cd11*ycoeffs[1])/denom
                new12 = (cd12*xcoeffs[1] - cd22*xcoeffs[2])/denom
                new22 = (cd12*xcoeffs[2] + cd22*xcoeffs[1])/denom
                newHeader['CD1_1'] = new11
                newHeader['CD2_2'] = new22
                newHeader['CD1_2'] = new12
                newHeader['CD2_1'] = new21
            newHeader['CRVAL1'] = crval1
            newHeader['CRVAL2'] = crval2

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCalc input arrays: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()
            #Compute transformation
            #Do this for the first pass only
            calcTransOpt3d = mod.get_function("calcTransOpt3d")
            calcTransOpt3d(drv.InOut(xout), drv.InOut(yout), drv.InOut(zout), drv.In(xin), drv.In(yin), drv.In(zin), drv.In(xcoeffs), drv.In(ycoeffs), drv.In(zcoeffs), int32(order), float32(offset), int32(nx*ny*nz), grid=(blocks,1), block=(block_size,1,1))
            #Compute transformation on refrerence pixels in python
            n = 0
            for i in range(order+1):
                for l in range(1,i+2):
                    for k in range(1,l+1):
                        xrefout+=xcoeffs[n]*xrefin**(i-l+1)*yrefin**(l-k)*zrefin**(k-1)
                        yrefout+=ycoeffs[n]*xrefin**(i-l+1)*yrefin**(l-k)*zrefin**(k-1)
                        zrefout+=ycoeffs[n]*xrefin**(i-l+1)*yrefin**(l-k)*zrefin**(k-1)
                        n+=1

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCalc transformation: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            if (xtrans is not None):
                xout = xtrans.astype(float32)
            if (ytrans is not None):
                yout = ytrans.astype(float32)
            if (ztrans is not None):
                zout = ztrans.astype(float32)

            #Release memory
            del xin
            del yin
            del zin

            #Apply scaling if necessary
            if (scale != 1):
                xout *= scale
                yout *= scale
                zout *= scale
                xsh = array(xsh)*scale
                ysh = array(ysh)*scale
                zsh = array(zsh)*scale
                xrefout *= scale
                yrefout *= scale
                zrefout *= scale

            #Calculate mins, maxes
            xmin = int(floor(xout.min()))
            xmax = int(floor(xout.max()))+2
            ymin = int(floor(yout.min()))
            ymax = int(floor(yout.max()))+2
            zmin = int(floor(zout.min()))
            zmax = int(floor(zout.max()))+2

            xshmin = min(xsh)
            yshmin = min(ysh)
            zshmin = min(zsh)
            xshrange = int(ceil(max(xsh)-min(xsh)))
            yshrange = int(ceil(max(ysh)-min(ysh)))
            zshrange = int(ceil(max(zsh)-min(zsh)))
            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCalc min/max output: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            #xsh[0] and ysh[0] may not be zero!  We shift image by this pixel amount so WCS needs to be shifted by same amount!
            xrefout += xsh[0]-xshmin
            yrefout += ysh[0]-yshmin
            zrefout += zsh[0]-zshmin

            #Subtract mins from xout, yout
            xout -= xmin
            yout -= ymin
            zout -= zmin
            if (_verbosity == fatboyLog.VERBOSE):
                print("\tSubtract min/maxes: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            #Create new image array only the first time through
            #Take into account max shifts
            newdata = zeros((zmax-zmin+zshrange, ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
            expmap = zeros((zmax-zmin+zshrange, ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
            if (doPix):
                pixmap = zeros((zmax-zmin+zshrange, ymax-ymin+yshrange, xmax-xmin+xshrange), int32)

            if (_verbosity == fatboyLog.VERBOSE):
                print("\tCreate new image array: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()
            #Handle existing data
            if (outExists):
                outxsh = 0.
                outysh = 0.
                outzsh = 0.
                if ('DHZMINXS' in outimage[0].header):
                    outxsh = float(outimage[0].header['DHZMINXS'])
                if ('DHZMINYS' in outimage[0].header):
                    outysh = float(outimage[0].header['DHZMINYS'])
                if ('DHZMINZS' in outimage[0].header):
                    outzsh = float(outimage[0].header['DHZMINZS'])
                xrefout = float(outimage[0].header['CRPIX1'])
                yrefout = float(outimage[0].header['CRPIX2'])
                zrefout = float(outimage[0].header['CRPIX3'])
                #Case exisiting image is as large or larger than needed
                if (outxsh <= xshmin and outysh <= yshmin and outzsh <= zshmin and outimage[mef].data.shape[2]+outxsh >= xmax-xmin+max(xsh) and outimage[mef].data.shape[1] >= ymax-ymin+max(ysh) and outimage[mef].data.shape[0] >= zmax-zmin+max(zsh)):
                    newdata = outimage[mef].data.astype(float32)
                    expmap = outexp[mef].data.astype(float32)
                    if (doPix):
                        pixmap = outpix[mef].data.astype(int32)
                    xsh = array(xsh)-outxsh
                    ysh = array(ysh)-outysh
                    zsh = array(zsh)-outzsh
                    xshmin -= outxsh
                    yshmin -= outysh
                    zshmin -= outzsh
                else:
                    #Need to enlarge image
                    xshmin = min(xshmin, outxsh)
                    yshmin = min(yshmin, outysh)
                    zshmin = min(zshmin, outzsh)
                    xmax = max(max(xsh)+xmax-xmin, outxsh+outimage[mef].data.shape[2])
                    ymax = max(max(ysh)+ymax-ymin, outysh+outimage[mef].data.shape[1])
                    zmax = max(max(zsh)+zmax-zmin, outzsh+outimage[mef].data.shape[0])
                    x_range = int(ceil(xmax-xshmin))
                    y_range = int(ceil(ymax-yshmin))
                    z_range = int(ceil(zmax-zshmin))
                    newdata = zeros((z_range, y_range, x_range), float32)
                    expmap = zeros((z_range, y_range, x_range), float32)
                    if (doPix):
                        pixmap = zeros((z_range, y_range, x_range), int32)
                    if (outxsh != xshmin):
                        xshmin = floor(xshmin)
                    x1 = int(outxsh - xshmin)
                    x2 = x1+outimage[mef].data.shape[1]
                    xrefout += x1
                    if (outysh != yshmin):
                        yshmin = floor(yshmin)
                    y1 = int(outysh - yshmin)
                    y2 = y1+outimage[mef].data.shape[0]
                    yrefout += y1
                    if (outzsh != zshmin):
                        zshmin = floor(zshmin)
                    z1 = int(outzsh - zshmin)
                    z2 = z1+outimage[mef].data.shape[0]
                    zrefout += z1
                    newdata[z1:z2,y1:y2,x1:x2] = outimage[mef].data.astype(float32)
                    expmap[z1:z2,y1:y2,x1:x2] = outexp[mef].data.astype(float32)
                    if (doPix):
                        pixmap[z1:z2,y1:y2,x1:x2] = outpix[mef].data.asytpe(int32)

            #Update reference min shifts, ref pixels in image header
            newHeader['DHZMINXS'] = xshmin
            newHeader['DHZMINYS'] = yshmin
            newHeader['DHZMINZS'] = zshmin
            newHeader['CRPIX1'] = xrefout
            newHeader['CRPIX2'] = yrefout
            newHeader['CRPIX3'] = zrefout

            #Copy over to pyfits ojects
            if (outExists):
                outimage[mef].data = newdata.astype(float32)
                outexp[mef].data = expmap.astype(float32)
                if (doPix):
                    outpix[mef].data = pixmap.astype(int32)
            else:
                #Create output arrays
                imagedata = zeros((zmax-zmin+zshrange, ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
                expdata = zeros((zmax-zmin+zshrange, ymax-ymin+yshrange, xmax-xmin+xshrange), float32)
                if (doPix):
                    pixdata = zeros((zmax-zmin+zshrange, ymax-ymin+yshrange, xmax-xmin+xshrange), int32)

        if (_verbosity == fatboyLog.VERBOSE):
            print("\tCopy over to pyfits and update: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        #Now set blocks based on data size
        blocks = data.size//512
        if (data.size % 512 != 0):
            blocks += 1
        #Scale data by inmask and weight factor
        multArrFloatIntScalar = mod.get_function("multArrFloatIntScalar")
        multArrFloatIntScalar(drv.InOut(data), drv.In(inmask), float32(scalefac/exptime), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
        if (tmpexp is None):
            #Exposure map should be exposure time * good pixel mask unless a previous exposure map has been loaded for inunits = cps
            tmpexp = float32(inmask*scalefac)
        totexp+=exptime

        if (_verbosity == fatboyLog.VERBOSE):
            print("\tScale data and expmask: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        #Ensure xout, yout shape matches data before sending to GPU
        if (xout.shape == data.shape):
            xout_g = xout
            yout_g = yout
            zout_g = zout
        else:
            xout_g = ascontiguousarray(xout[:data.shape[0], :data.shape[1], :data.shape[2]])
            yout_g = ascontiguousarray(yout[:data.shape[0], :data.shape[1], :data.shape[2]])
            zout_g = ascontiguousarray(zout[:data.shape[0], :data.shape[1], :data.shape[2]])

        xsize = newdata.shape[2]
        ysize = newdata.shape[1]
        if (kernel == 'turbo'):
            if (doPix):
                turboKernel = mod.get_function("turboKernel3dPix")
                turboKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), drv.In(zout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), float32(zsh[j]-zshmin), int32(xsize), int32(ysize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
            else:
                turboKernel = mod.get_function("turboKernel3d")
                turboKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), drv.In(zout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), float32(zsh[j]-zshmin), int32(xsize), int32(ysize), int32(data.size), float32(dropsize), grid=(blocks,1), block=(block_size,1,1))
        elif (kernel == 'point'):
            if (doPix):
                pointKernel = mod.get_function("pointKernel3dPix")
                pointKernel(drv.InOut(newdata), drv.InOut(expmap), drv.InOut(pixmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), drv.In(zout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), float32(zsh[j]-zshmin), int32(xsize), int32(ysize), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
            else:
                pointKernel = mod.get_function("pointKernel3d")
                pointKernel(drv.InOut(newdata), drv.InOut(expmap), drv.In(data), drv.In(tmpexp), drv.In(xout_g), drv.In(yout_g), drv.In(zout_g), float32(xsh[j]-xshmin), float32(ysh[j]-yshmin), float32(zsh[j]-zshmin), int32(xsize), int32(ysize), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
        tt = time.time()

        #If requested, update FDUs here
        if (updateFDUs and (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG)):
            #Make copies with astype()
            drihizzled_data = newdata.astype(float32)
            expmap_data = expmap.astype(float32)
            divFloatArrays = mod.get_function("divFloatArrays")
            #Apply weighting
            blocks = newdata.size//block_size
            if (newdata.size % block_size != 0):
                blocks += 1
            blocky = 1
            while(blocks > 65535):
                blocky *= 2
                blocks //= 2
            #Apply weighting
            if (weight == 'exptime'):
                if (outunits == 'cps'):
                    divFloatArrays(drv.InOut(drihizzled_data), drv.In(expmap_data), int32(drihizzled_data.size), grid=(blocks, blocky), block=(block_size,1,1))
                else:
                    divFloatArrays(drv.InOut(drihizzled_data), drv.In(expmap_data), int32(drihizzled_data.size), grid=(blocks, blocky), block=(block_size,1,1))
                    if (outunits == 'counts'):
                        drihizzled_data*=exptime
            frames[j].tagDataAs("drihizzled", data=drihizzled_data)
            frames[j].tagDataAs("exposure_map", data=expmap_data)

        #If keeping individual images, write them out here
        if (keepImages.lower() == 'yes'):
            if (not os.access(imgdir, os.F_OK)):
                imgdir = ''
            if (imgdir != '' and imgdir[-1] != '/'):
                imgdir += '/'
            if (mode == MODE_FITS):
                shortfn = frames[j][frames[j].rfind('/')+1:]
                temp = pyfits.open(frames[j])
            elif (mode == MODE_RAW):
                shortfn = 'frame'+str(j)+'.fits'
                temp = pyfits.HDUList()
                temp.append(pyfits.PrimaryHDU())
            elif (mode == MODE_FDU):
                shortfn = frames[j].getFullId()
                temp = pyfits.open(frames[j].getFilename())
            elif (mode == MODE_FDU_DIFFERENCE):
                shortfn = frames[j+1]._id+frames[j+1]._index+"-"+frames[j]._index+".fits"
                temp = pyfits.open(frames[j+1].getFilename())
            elif (mode == MODE_FDU_TAG):
                shortfn = frames[j]._id+"_"+dataTag+frames[j]._index+".fits"
                temp = pyfits.open(frames[j].getFilename())
            temp[mef].data = newdata.astype(float32)
            tempexpmap = expmap.astype(float32)
            divFloatArrays = mod.get_function("divFloatArrays")
            #Apply weighting
            blocks = newdata.size//block_size
            if (newdata.size % block_size != 0):
                blocks += 1
            blocky = 1
            while(blocks > 65535):
                blocky *= 2
                blocks //= 2
            if (weight == 'exptime'):
                if (outunits == 'cps'):
                    divFloatArrays(drv.InOut(temp[mef].data), drv.In(tempexpmap), int32(temp[mef].data.size), grid=(blocks, blocky), block=(block_size,1,1))
                else:
                    divFloatArrays(drv.InOut(temp[mef].data), drv.In(tempexpmap), int32(temp[mef].data.size), grid=(blocks, blocky), block=(block_size,1,1))
                    if (outunits == 'counts'):
                        temp[mef].data*=exptime
            temp.verify('silentfix')
            if (os.access(imgdir+'drihiz_'+shortfn, os.F_OK)):
                os.unlink(imgdir+'drihiz_'+shortfn)
            if (os.access(imgdir+'expmap_'+shortfn, os.F_OK)):
                os.unlink(imgdir+'expmap_'+shortfn)
            #Update header info
            updateHeader(temp[0].header, newHeader)
            #Update WCS if applicable
            if (hasWCS):
                if (j != 0):
                    denom = xcoeffs[1]*ycoeffs[2]-xcoeffs[2]*ycoeffs[1]
                    new11 = (cd11*ycoeffs[2] + cd21*ycoeffs[1])/denom
                    new21 = (cd21*ycoeffs[2] - cd11*ycoeffs[1])/denom
                    new12 = (cd12*xcoeffs[1] - cd22*xcoeffs[2])/denom
                    new22 = (cd12*xcoeffs[2] + cd22*xcoeffs[1])/denom
                updateHeaderEntry(temp[0].header, 'CD1_1', new11)
                updateHeaderEntry(temp[0].header, 'CD2_2', new22)
                updateHeaderEntry(temp[0].header, 'CD1_2', new12)
                updateHeaderEntry(temp[0].header, 'CD2_1', new21)
            #Write out new files
            #Get rid of extraneous extensions in data like CIRCE/Newfirm
            prepMefForWriting(temp, mef)
            temp.writeto(imgdir+'drihiz_'+shortfn, output_verify='silentfix')
            temp[mef].data = expmap
            temp.writeto(imgdir+'expmap_'+shortfn, output_verify='silentfix')
            temp.close()
            if (_verbosity == fatboyLog.VERBOSE):
                print("\tWrite Indiv Images: ",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

        ##Update overall images
        if (outExists):
            outimage[mef].data += newdata
            outexp[mef].data += expmap
            if (doPix):
                outpix[mef].data += pixmap
        else:
            imagedata += newdata
            expdata += expmap
            if (doPix):
                pixdata += pixmap

        #Reset arrays
        if (j < nframes-1):
            newdata[:,:,:] = 0.
            expmap[:,:,:] = 0.
        if (doPix):
            if (j < nframes-1):
                pixmap[:,:,:] = 0

        #GUI message:
        if (gui is not None):
            gui = (gui[0], gui[1]+1., gui[2], gui[3], gui[4])
            if (gui[0]): print("PROGRESS: "+str(int(gui[3]+gui[1]/gui[2]*gui[4])))

    if (_verbosity == fatboyLog.VERBOSE):
        print("Process indiv frames: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()
    #Free memory
    del xout
    del yout
    del inmask
    del data
    del tmpexp

    if (outExists):
        imagedata = outimage[mef].data
        expdata = outexp[mef].data

    blocks = imagedata.size//block_size
    if (imagedata.size % block_size != 0):
        blocks += 1
    blocky = 1
    while(blocks > 65535):
        blocky *= 2
        blocks //= 2

    divFloatArrays = mod.get_function("divFloatArrays")
    #Apply weighting
    if (weight == 'exptime' and kernel != 'uniform'):
        if (outunits == 'cps'):
            divFloatArrays(drv.InOut(imagedata), drv.In(expdata), int32(imagedata.size), grid=(blocks,blocky), block=(block_size,1,1))
    elif (kernel != 'uniform'):
        divFloatArrays(drv.InOut(imagedata), drv.In(expdata), int32(imagedata.size), grid=(blocks,blocky), block=(block_size,1,1))
        if (outunits == 'counts'):
            imagedata *= totexp

    if (_verbosity == fatboyLog.VERBOSE):
        print("Apply Weighting: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Update header
    newHeader['DHZUNITS'] = outunits
    newHeader[expkey] = totexp
    newHeader['DHZKERNL'] = kernel
    newHeader['DHZNFRMS'] = numframes

    #output file
    if (outfile is not None):
        print("\tOutput file: "+outfile)
        write_fatboy_log(log, logtype, "\tOutput file: "+outfile, __name__, printCaller=False, tabLevel=1)
        if (not outExists):
            if (mode == MODE_FITS):
                outimage = pyfits.open(frames[0])
            elif (mode == MODE_RAW):
                hdu = pyfits.PrimaryHDU(outtype(out))
                outimage = pyfits.HDUList([hdu])
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                outimage = pyfits.open(frames[0].filename)
        outimage[mef].data = imagedata
        #update header
        updateHeader(outimage[0].header, newHeader)
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(outimage, mef)
        if (outExists):
            outimage.flush()
        else:
            outimage.writeto(outfile, output_verify='silentfix')
        outimage.close()
        del outimage
    #Exp map file
    if (weightfile is not None):
        print("\tWeight file: "+weightfile)
        write_fatboy_log(log, logtype, "\tWeight file: "+weightfile, __name__, printCaller=False, tabLevel=1)
        if (not outExists):
            if (mode == MODE_FITS):
                outexp = pyfits.open(frames[0])
            elif (mode == MODE_RAW):
                hdu = pyfits.PrimaryHDU(outtype(out))
                outexp = pyfits.HDUList([hdu])
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                outexp = pyfits.open(frames[0].filename)
        outexp[mef].data = expdata
        #update header
        updateHeader(outexp[0].header, newHeader)
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(outexp, mef)
        if (outExists):
            outexp.flush()
        else:
            outexp.writeto(weightfile, output_verify='silentfix')
        outexp.close()
        del outexp
    if (pixfile is not None):
        print("\tPix file: "+pixfile)
        write_fatboy_log(log, logtype, "\tPix file: "+pixfile, __name__, printCaller=False, tabLevel=1)
        if (not outExists):
            if (mode == MODE_FITS):
                outpix = pyfits.open(frames[0])
            elif (mode == MODE_RAW):
                hdu = pyfits.PrimaryHDU(outtype(out))
                outpix = pyfits.HDUList([hdu])
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                outpix = pyfits.open(frames[0].filename)
        outpix[mef].data = pixdata
        #update header
        updateHeader(outpix[0].header, newHeader)
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(outpix, mef)
        if (outExists):
            outpix.flush()
        else:
            outpix.writeto(pixfile, output_verify='silentfix')
        outpix.close()
        del outpix
    if (_verbosity == fatboyLog.VERBOSE):
        print("Write data: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()
    print("Drihizzled "+str(nframes)+" files with "+kernel+" kernel.  Total time (s): "+str(time.time()-t))
    write_fatboy_log(log, logtype, "Drihizzled "+str(nframes)+" files with "+kernel+" kernel.  Total time (s): "+str(time.time()-t), __name__)
    #Return 4-tuple
    return (imagedata, newHeader, expdata, pixdata)
