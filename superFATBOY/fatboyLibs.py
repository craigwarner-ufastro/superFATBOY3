## @package superFATBOY
#  Documentation for pipeline.
#
#
hasCuda = True
useAstropy = False
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
        import pycuda.gpuarray as gpuarray
        from pycuda.reduction import ReductionKernel
except Exception:
    print("fatboyLibs> WARNING: PyCUDA not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)

from superFATBOY.fatboyLog import *
from superFATBOY.gpu_arraymedian import *

from numpy import *
import numpy as np
import scipy
from scipy.optimize import leastsq
try:
    import pyfits
except ImportError as ex:
    import astropy
    import astropy.io.fits as pyfits
    useAstropy = True
import os, time
import xml.dom.minidom
from xml.dom.minidom import Node

#constants
LOGTYPE_NONE = 0
LOGTYPE_ASCII = 1
LOGTYPE_FATBOY = 2

blocks = 2048*4
block_size = 512
nbr_values = blocks * block_size

def get_fatboy_mod():
    fatboy_mod = None
    if (hasCuda and superFATBOY.gpuEnabled()):
        fatboy_mod = SourceModule("""

      /***** Device functions ***/

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

      __device__ float fwhm1d_float(float* data, float halfMax, int size) {
        //estimate fwhm of 1d data
        float maxVal = 0.0f;
        int maxIdx = 0;
        for (int i = 0; i < size; i++) {
          if (maxVal == 0 || data[i] > maxVal) {
            maxVal = data[i];
            maxIdx = i;
          }
        }
        if (halfMax == 0) halfMax = maxVal/2.0f;

        //Find last data point < half max
        int startIdx = 0;
        for (int i = maxIdx-1; i >=0; i--) {
          if (data[i] < halfMax) {
            startIdx = i;
            break;
          }
        }

        //Find first data point after max that is < half max
        int endIdx = size-1;
        for (int i = maxIdx+1; i < size; i++) {
          if (data[i] < halfMax) {
            endIdx = i;
            break;
          }
        }

        if (startIdx >= size-1) startIdx = size-2;
        if (endIdx <= 0) endIdx = 1;

        float xstart = startIdx + (halfMax - data[startIdx]) / (data[startIdx+1] - data[startIdx]);
        float xend = endIdx - (halfMax - data[endIdx]) / (data[endIdx-1] - data[endIdx]);

        if (xstart < startIdx-1 || xstart > endIdx+1) xstart = startIdx;
        if (xend > endIdx+1 || xend < startIdx-1) xend = endIdx;
        return abs(xend-xstart);
      }


      /****** Global functions *****/

      __global__ void applyObjMask_float(float* image, int* objMask) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (objMask[i] > 0) image[i] = 0;
      }

      __global__ void blkavg_float(float *data, float *output, int faccol, int facrow, int cols, int rows) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= rows*cols) return;
        int row = i/cols;
        int col = i%cols;
        output[i] = 0;
        for (int j = 0; j < faccol; j++) {
          for (int l = 0; l < facrow; l++) {
            output[i] += data[(row*facrow+l)*(cols*faccol)+(col*faccol+j)];
          }
        }
        output[i] /= (facrow*faccol);
      }

      __global__ void blkrep_float(float *data, float *output, int faccol, int facrow, int cols, int rows) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= rows*cols) return;
        int row = i/cols;
        int col = i%cols;
        for (int j = 0; j < faccol; j++) {
          for (int l = 0; l < facrow; l++) {
            output[(row*facrow+l)*(cols*faccol)+(col*faccol+j)] = data[i];
          }
        }
      }

      __global__ void calcTrans3d(float *yout, float *xin, float *yin, float *z, float *ycoeffs, int order, int xsize, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        if (z[i] == 0) {
          yout[i] = 0;
          return;
        }
        int xi = i % xsize;
        int n = 0;
        float xp [10];
        float yp [10];
        float zp [10];
        xp[0] = 1; yp[0] = 1; zp[0] = 1;
        xp[1] = xin[xi]; yp[1] = yin[i]; zp[1] = z[i];
        if (order >= 2) {
          xp[2] = xin[xi]*xin[xi];
          yp[2] = yin[i]*yin[i];
          zp[2] = z[i]*z[i];
        }
        for (int j = 3; j <= order; j++) {
          xp[j] = pow(xin[xi], j);
          yp[j] = pow(yin[i], j);
          zp[j] = pow(z[i], j);
        }
        yout[i] = 0;
        for (int x = 0; x <= order; x++) {
          for (int l = 1; l <= x+1; l++) {
            for (int k = 1; k <= l; k++) {
              yout[i] += ycoeffs[n]*xp[x-l+1]*yp[l-k]*zp[k-1];
              n++;
            }
          }
        }
      }

      __global__ void calcXin(float *xin, int nx, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        xin[i] = (i%nx);
      }

      __global__ void calcYin(float *yin, int nx, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        yin[i] = (i/nx);
      }

      __global__ void convolve2d_float(float *data, float *output, float *kernel, int rows, int cols, int kny, int knx, int bnd, int maskNeg) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        int row = i/cols - kny/2;
        int col = i % cols - knx/2;
        int currRow = 0;
        int currCol = 0;
        output[i] = 0;
        if (bnd == 1) {
          for (int j = 0; j < knx; j++) {
            for (int k = 0; k < kny; k++) {
              currCol = col + j;
              if (currCol < 0) currCol = 0;
              if (currCol >= cols) currCol = cols-1;
              currRow = row + k;
              if (currRow < 0) currRow = 0;
              if (currRow >= rows) currRow = rows-1;
              output[i] += kernel[j+k*knx]*data[currRow*cols+currCol];
            }
          }
        } else {
          for (int j = 0; j < knx; j++) {
            for (int k = 0; k < kny; k++) {
              currCol = col + j;
              currRow = row + k;
              if (currCol < 0 || currCol >= cols || currRow < 0 || currRow >= rows) continue;
              output[i] += kernel[j+k*knx]*data[currRow*cols+currCol];
            }
          }
        }
        if (maskNeg == 1 && output[i] < 0) output[i] = 0;
      }

      __global__ void convolve2dAndBlk_float(float *data, float *output, float *kernel, int rows, int cols, int kny, int knx, int bnd, int maskNeg, int facrow, int faccol) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= rows*cols) return;
        int outRow = i/cols;
        int outCol = i%cols;

        int inRow = i/cols - kny/2;
        int inCol = i % cols - knx/2;
        int inCols = cols*faccol;
        int inRows = rows*facrow;

        int currRow = 0;
        int currCol = 0;
        output[i] = 0;
        float pix;
        if (bnd == 1) {
          for (int c = 0; c < faccol; c++) {
            for (int r = 0; r < facrow; r++) {
              inRow = outRow*facrow+r-kny/2;
              inCol = outCol*faccol+c-knx/2;
              pix = 0;
              for (int j = 0; j < knx; j++) {
                for (int k = 0; k < kny; k++) {
                  currCol = inCol + j;
                  if (currCol < 0) currCol = 0;
                  if (currCol >= inCols) currCol = inCols-1;
                  currRow = inRow + k;
                  if (currRow < 0) currRow = 0;
                  if (currRow >= inRows) currRow = inRows-1;
                  pix += kernel[j+k*knx]*data[currRow*inCols+currCol];
                }
              }
              if (pix > 0 || maskNeg == 0) output[i] += pix;
            }
          }
        } else {
          for (int c = 0; c < faccol; c++) {
            for (int r = 0; r < facrow; r++) {
              inRow = outRow*facrow+r-kny/2;
              inCol = outCol*faccol+c-knx/2;
              pix = 0;
              for (int j = 0; j < knx; j++) {
                for (int k = 0; k < kny; k++) {
                  currCol = inCol + j;
                  currRow = inRow + k;
                  if (currCol < 0 || currCol >= inCols || currRow < 0 || currRow >= inRows) continue;
                  pix += kernel[j+k*knx]*data[currRow*inCols+currCol];
                }
              }
              if (pix > 0 || maskNeg == 0) output[i] += pix;
            }
          }
        }
        output[i] /= (facrow*faccol);
      }

      __global__ void cosmicRayRemoval_float(float *data, float *output, int rows, int cols, int *ict) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= rows*cols) return;
        int row = i/cols;
        int col = i % cols;
        output[i] = data[i];
        if (row == 0 || col == 0 || row == rows-1 || col == cols-1) return;
        float mean = 0, sq = 0;
        for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
            if (j == 0 && k == 0) continue;
            mean += data[i+cols*j+k];
            sq += data[i+cols*j+k]*data[i+cols*j+k];
          }
        }
        mean /= 8;
        sq /= 8;
        float sd = sqrt(sq-mean*mean);
        if (abs(data[i]-mean) > 5*sd) {
          float temp[8];
          int idx = 0;
          for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
              if (j == 0 && k == 0) continue;
              temp[idx++] = data[i+cols*j+k];
            }
          }
          bubblesort_float(temp, 8);
          output[i] = (temp[3]+temp[4])/2.0f;
          atomicAdd(&ict[0], 1);
        }
      }

      __global__ void createObjMask(int* objMask) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        objMask[i] = objMask[i] == 0 ? 1 : 0;
      }

      __global__ void createSlitmask(int *slitmask, int *rslitHi, int *rslitLo, int cols, int rows, int nslits, int horizontal, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        slitmask[i] = 0;
        int yind = i / cols;
        int xind = i % cols;
        int idx = 0;
        if (horizontal) {
          for (int j = 0; j < nslits; j++) {
            idx = j*cols+xind;
            if (yind >= rslitLo[idx] && yind <= rslitHi[idx]) {
              slitmask[i] = (j+1);
            }
          }
        } else {
          for (int j = 0; j < nslits; j++) {
            //idx = j*cols+yind;
            idx = j*rows+yind;
            if (xind >= rslitLo[idx] && xind <= rslitHi[idx]) {
              slitmask[i] = (j+1);
            }
          }
        }
      }

      __global__ void divideArrays_float(float* dividend, float* divisor, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        if (divisor[i] == 0) {
          dividend[i] = 0;
          return;
        }
        dividend[i] /= divisor[i];
      }

      __global__ void fwhm2d_cube_float(float* data, int* flag, int depth, int nx, int ny, int estimateBackground, float* fwhms) {
        //estimate 2-d fwhm of data
        //returned array contains fwhm_mean, fwhm_stddev, array of 4 FWHMs, background value used as zero level
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= depth) return;
        if (flag[i] == 0) {
          for (int j = i*7; j < i*7+6; j++) fwhms[j] = -1;
          fwhms[i*7+6] = 0;
          return;
        }

        int startIdx = i*nx*ny;
        int endIdx = (i+1)*nx*ny;
        int npix = nx*ny;
        float dmax = 0.0f;
        int xpos = -1;
        int ypos = -1;
        for (int j = startIdx; j < endIdx; j++) {
          if (dmax == 0 || data[j] > dmax) {
            dmax = data[j];
            xpos = (j-startIdx) % nx;
            ypos = (j-startIdx) / nx;
          }
        }

        float dtot = 0.0f;
        float bkgrnd = 0.0f;
        float halfmax = (dmax + bkgrnd)/2;
        float dtcen = 0.0f;

        if (estimateBackground > 0) {
          //sum
          for (int j = startIdx; j < endIdx; j++) dtot += data[j];
          for (int yi = ypos-1; yi < ypos+2; yi++) {
            for (int xi = xpos-1; xi = xpos+2; xi++) {
              dtcen += data[startIdx + yi*nx + xi];
            }
          }
          bkgrnd = (dtot - dtcen)/(npix - 9);
        }

        //X and Y cuts
        //extern __shared__ float xcut[];
        //extern __shared__ float ycut[];
        //float xcut[nx];
        //float ycut[ny];
        float xcut[100];
        float ycut[100];
        for (int j = 0; j < nx; j++) xcut[j] = data[startIdx + ypos*nx + j];
        //for (int j = 0; j < nx; j++) fwhms[i*nx+j] = xcut[j];
        for (int j = 0; j < ny; j++) ycut[j] = data[startIdx + j*nx + xpos];
        fwhms[i*7+2] = fwhm1d_float(xcut, halfmax, nx);
        fwhms[i*7+3] = fwhm1d_float(ycut, halfmax, ny);

        //diagonals are sqrt(2) times larger:
        float sq2 = 1.41421356f;
        int xystart = min(xpos, ypos);
        int xyend = min(nx-1-xpos, ny-1-ypos);
        int ncut = xystart+xyend+1;
        //y=x
        //extern __shared__ float dcutxy[];
        //float dcutxy[ncut];
        float dcutxy[100];
        for (int j = 0; j < ncut; j++) dcutxy[j] = data[startIdx + (ypos-xystart+j)*nx + (xpos-xystart+j)];
        fwhms[i*7+4] = sq2*fwhm1d_float(dcutxy, halfmax, ncut);

        int negxystart = min(xpos, ny-1-ypos);
        int negxyend = min( nx-1-xpos, ypos );
        ncut = negxystart+negxyend+1;
        //y=-x
        //extern __shared__ float dcutnegxy[];
        //float dcutnegxy[ncut];
        float dcutnegxy[100];
        for (int j = 0; j < ncut; j++) dcutnegxy[j] = data[startIdx + (ypos+negxystart-j)*nx + (xpos-negxystart+j)];
        fwhms[i*7+5] = sq2 * fwhm1d_float(dcutnegxy, halfmax, ncut);

        fwhms[i*7+6] = bkgrnd;
        fwhms[i*7] = (fwhms[i*7+2]+fwhms[i*7+3]+fwhms[i*7+4]+fwhms[i*7+5])/4.0f;
        fwhms[i*7+1] = 19;
      }


      __global__ void generateQAData_float(float *data, float *xcoords, float *ycoords, float *sylo, float *syhi, int ncoords, int npoints, int nslits, int cols, int horizontal) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= npoints) return;
        //i = {0 : 18*nslits*len(xcoords)}
        int icoord = i%ncoords;
        int islit = i/(ncoords*18);
        int i2 = (i-(ncoords*islit*18))/ncoords; //i2 = {0:18}
        int xi = (i2%3)-1; // {xi = {-1:1}
        int yi = (i2%9)/3-1; // {yi = {-1:1}
        int xval = (int)(xcoords[icoord]+0.5)+xi;
        float xdist = xcoords[icoord]-xval;
        int yval = 0;
        float ydist = 0;
        if (i2 >= 9) {
          yval = (int)(ycoords[icoord]+sylo[islit]+0.5)+yi;
          ydist = ycoords[icoord]+sylo[islit]-yval;
        } else {
          yval = (int)(ycoords[icoord]+syhi[islit]+0.5)+yi;
          ydist = ycoords[icoord]+syhi[islit]-yval;
        }
        float dist = sqrt(ydist*ydist+xdist*xdist);
        if (horizontal == 1) {
          data[yval*cols+xval] = -50000/((1+dist)*(1+dist));
        } else {
          data[xval*cols+yval] = -50000/((1+dist)*(1+dist));
          //data[yval*cols+xval] = -50000/((1+dist)*(1+dist));
        }
      }

      __global__ void getCentroid_cube_float(float* data, int* flag, int mx, int my, float fwhm, int depth, int xsize, int ysize, float* cens) {
        //Input (mx,my) is assummed to be location of maximum pixel in image,
        //and fwhm is either guess or actual FWHM.
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= depth) return;
        //initialize output vals
        cens[i*2] = -1;
        cens[i*2+1] = -1;
        if (flag[i] == 0) {
          return;
        }
        int startIdx = i*xsize*ysize; //start of this cut
        int nhalf = (int)(0.637*fwhm);
        if (nhalf < 2) nhalf = 2;
        int nbox = 2*nhalf+1;

        if( (mx < nhalf) || ((mx + nhalf) >= xsize) || (my < nhalf) || ((my + nhalf) >= ysize) ) {
          //Too near edge of image, return -1, -1
          return;
        }

        //define arrays here to all be up to 9x9
        float starbox[81];
        float deriv[81];
        float dd[9];
        float ddsq[9];
        float w[9];
        float derivtot[9];

        int j = 0;
        for (int iy=my-nhalf; iy<my+nhalf+1; iy++) {
          for (int ix = mx-nhalf; ix<mx+nhalf+1; ix++) {
            starbox[j++] = data[startIdx+iy*xsize+ix];
          }
        }

        int ir = max(nhalf-1, 1);
        float sumc = 0;
        for (int j = 0; j < nbox-1; j++) {
          dd[j] = j+0.5f-nhalf;
          ddsq[j] = dd[j]*dd[j];
          w[j] = (1.0f - 0.5f*(abs(dd[j])-0.5) / (nhalf-0.5));
          sumc += w[j];
        }

        int boxelem = nbox*nbox; //total elements of starbox

        //Y partial derivative:
        for (int j = 0; j < boxelem; j++) deriv[j] = starbox[(j+nbox)%boxelem] - starbox[j];
        float sumd = 0;
        float sumxd = 0;
        float sumxsq = 0;
        for (int iy = 0; iy < nbox-1; iy++) {
          derivtot[iy] = 0;
          for (int ix = nhalf-ir; ix < nhalf+ir+1; ix++) {
            derivtot[iy] += deriv[iy*nbox+ix];
          }
          sumd += w[iy]*derivtot[iy];
          sumxd += w[iy]*dd[iy]*derivtot[iy];
          sumxsq += w[iy]*ddsq[iy];
        }

        if (sumxd < 0) {
          float dy = sumxsq*sumd/(sumc*sumxd);
          if (abs(dy) <= nhalf) cens[i*2+1] = my-dy; //do not add 0.5 by convention
        }

        //X partial derivative:
        for (int iy = 0; iy < nbox; iy++) {
          for (int ix = 0; ix < nbox; ix++) {
            deriv[iy*nbox+ix] = starbox[iy*nbox+((ix+1)%nbox)] - starbox[iy*nbox+ix];
          }
        }
        sumd = 0;
        sumxd = 0;
        sumxsq = 0;
        for (int ix = 0; ix < nbox-1; ix++) {
          derivtot[ix] = 0;
          for (int iy = nhalf-ir; iy < nhalf+ir+1; iy++) {
            derivtot[ix] += deriv[iy*nbox+ix];
          }
          sumd += w[ix]*derivtot[ix];
          sumxd += w[ix]*dd[ix]*derivtot[ix];
          sumxsq += w[ix]*ddsq[ix];
        }

        if (sumxd < 0) {
          float dx = sumxsq*sumd/(sumc*sumxd);
          if (abs(dx) <= nhalf) cens[i*2] = mx-dx; //do not add 0.5 by convention
        }
        return;
      }

      __global__ void growApplyObjMask_float(float *image, int *objMask, int rows, int cols, int w, float rejectLevel) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        float smoothed = 0;
        int row = i/cols;
        int col = i % cols;
        int npts = 0;
        int start_row = row >= w ? row-w : 0;
        int end_row = row < rows-w ? row+w : rows-1;
        int start_col = col >= w ? col-w : 0;
        int end_col = col < cols-w ? col+w : cols-1;
        for (int j = start_row; j <= end_row; j++) {
          for (int k = start_col; k <= end_col; k++) {
            smoothed += objMask[cols*j+k];
            npts++;
          }
        }
        if (smoothed/npts < rejectLevel) image[i] = 0;
      }

      /************** LA cosmic (lacos) helepr methods ***********/

      __global__ void lacosFirstSel_float(float *sigmap, float *med5, float *firstsel, float sigclip) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        //subtract median filtered sigmap
        sigmap[i] -= med5[i];
        firstsel[i] = 0;
        if (sigmap[i] > sigclip) firstsel[i] = sigmap[i];
        if (firstsel[i] > 0.1) firstsel[i] = 1.0f;
      }

      __global__ void lacosNoiseModel_float(float *med5, float *deriv2, float *noise, float *sigmap, float gain, float readn) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        //create noise model
        noise[i] = sqrt(abs(med5[i]*gain+readn*readn))/gain;
        sigmap[i] = deriv2[i]/noise[i]/2;
      }

      __global__ void lacosSelect_float(float *sel, float *sigmap, float lower1, float sigclip, float lower2) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (sel[i] > lower1) sel[i] = 1.0f;
        sel[i] *= sigmap[i];
        if (sel[i] <= sigclip) sel[i] = 0;
        if (sel[i] > 0.1) sel[i] = 1.0f;
      }

      __global__ void lacosSelectAndCount_float(float *sel, float *sigmap, float lower1, float sigclip, float lower2, float *mask, float *inputmask, float *oldoutput, int *npix) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (sel[i] > lower1) sel[i] = 1.0f;
        sel[i] *= sigmap[i];
        if (sel[i] <= sigclip) sel[i] = 0;
        if (sel[i] > 0.1) sel[i] = 1.0f;
        if ((1-mask[i])*sel[i] >= 0.5) atomicAdd(&npix[0], 1);
        mask[i] += sel[i];
        if (mask[i] > 1) mask[i] = 1;
        inputmask[i] = (1-10000.0*mask[i])*oldoutput[i];
      }

      __global__ void lacosStarReject_float(float *med3, float *med7, float *noise, float *firstsel, float *sigmap, float objlim) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        med3[i] = (med3[i]-med7[i])/noise[i];
        if (med3[i] < 0.01) med3[i] = 0.01;
        //compare CR flux to object flux
        float starreject = (firstsel[i]*sigmap[i])/med3[i];

        //discard if CR flux <= objlim * object flux
        if (starreject <= objlim) starreject = 0;
        if (starreject > 0.5) starreject = 1.0f;
        firstsel[i]*=starreject;
      }

      __global__ void lacosUpdateOutput_float(float *oldoutput, float *tempOmask, float *med5, float *skymod) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        med5[i] *= tempOmask[i];
        oldoutput[i] = (1-tempOmask[i])*oldoutput[i] + med5[i];
        oldoutput[i] += skymod[i];
      }
      /************end LA cosmic (lacos) methods **************/

      __global__ void linInterp_float(float *data, float *output, int* gpm, float x, int rows, int cols, int *ict, int *nfound) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= rows*cols) return;
        int row = i/cols;
        int col = i % cols;
        if (data[i] != x || gpm[i] == 0) {
          //return if data is nonzero or it is a bad pixel
          output[i] = data[i];
          return;
        }
        float temp[8];
        atomicAdd(&nfound[0], 1);
        int npts = 0;
        int start_row = row >= 1 ? row-1 : 0;
        int end_row = row < rows-1 ? row+1 : rows-1;
        int start_col = col >= 1 ? col-1 : 0;
        int end_col = col < cols-1 ? col+1 : cols-1;
        for (int j = start_row; j <= end_row; j++) {
          for (int k = start_col; k <= end_col; k++) {
            if (j == 0 && k == 0) continue;
            if (data[cols*j+k] != x) temp[npts++] = data[cols*j+k];
          }
        }
        if (npts < 2) {
          //not enough neighboring pixels found!
          output[i] = data[i];
          return;
        }
        bubblesort_float(temp, npts);
        atomicAdd(&ict[0], 1);
        if (npts % 2 == 0) {
          output[i] = (temp[npts/2]+temp[npts/2-1])/2.0f;
        } else {
          output[i] = temp[npts/2];
        }
      }

      __global__ void maskNegatives_float(float *image, float *output) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (image[i] < 0) output[i] = 0; else output[i] = image[i];
      }

      __global__ void maskNegatives_int(int *image, int *output) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (image[i] < 0) output[i] = 0; else output[i] = image[i];
      }

      __global__ void maskNegatives_double(double *image, double *output) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (image[i] < 0) output[i] = 0; else output[i] = image[i];
      }

      __global__ void maskNegatives_long(long *image, long *output) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (image[i] < 0) output[i] = 0; else output[i] = image[i];
      }

      __global__ void maskNegativesAndZeros_float(float *image, float *output, float zeroRep, float negRep) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (image[i] == 0) {
          output[i] = zeroRep;
        } else if (image[i] < 0) {
          output[i] = negRep;
        } else output[i] = image[i];
      }

      __global__ void medfilt2d_float(float *data, float *smoothed, int rows, int cols, int w, int zlo, int zhi) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        float temp[49];
        smoothed[i] = 0;
        int row = i/cols;
        int col = i % cols;
        int npts = 0;
        int start_row = row >= w ? row-w : 0;
        int end_row = row < rows-w ? row+w : rows-1;
        int start_col = col >= w ? col-w : 0;
        int end_col = col < cols-w ? col+w : cols-1;
        for (int j = start_row; j <= end_row; j++) {
          for (int k = start_col; k <= end_col; k++) {
            if (zlo != 0 && data[cols*j+k] < zlo) continue;
            if (zhi != 0 && data[cols*j+k] > zhi) continue;
            temp[npts++] = data[cols*j+k];
          }
        }
        bubblesort_float(temp, npts);
        if (npts % 2 == 0) {
          smoothed[i] = (temp[npts/2]+temp[npts/2-1])/2.0f;
        } else {
          smoothed[i] = temp[npts/2];
        }
      }

      __global__ void medianNeighborReplaceBadPix(float *data, float *output, bool *badpix, bool *out_badpix, int rows, int cols, int *ct) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= rows*cols) return;
        int row = i/cols;
        int col = i % cols;
        output[i] = data[i];
        out_badpix[i] = badpix[i];

        if (badpix[i]) {
          float temp[8];
          int npts = 0;
          for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {
              if (row+j < 0 || row+j >= rows || col+k < 0 || col+k >= cols) continue;
              if (j == 0 && k == 0) continue;
              if (badpix[i+cols*j+k]) continue;
              if (data[i+cols*j+k] == 0) continue;
              temp[npts++] = data[i+cols*j+k];
            }
          }
          if (npts < 2) return; //not enough pixels found
          bubblesort_float(temp, npts);
          if (npts % 2 == 0) {
            output[i] = (temp[npts/2]+temp[npts/2-1])/2.0f;
          } else {
            output[i] = temp[npts/2];
          }
          atomicAdd(&ct[0], 1);
          out_badpix[i] = 0; //update badpix mask here for multiple iterations
        }
      }

      __global__ void mediansmooth1d2d_float(float *data, float *smoothed, int rows, int cols, int w, int axis) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        int row = i/cols;
        int col = i%cols;
        float temp[31];
        int n = 0;
        smoothed[i] = 0;
        if (axis == 0) {
          if (col < w) {
            for (int j = row*cols; j <= i+w; j++) {
              temp[n++] = data[j];
            }
          } else if (col >= cols-w) {
            for (int j = i-w; j < (row+1)*cols; j++) {
              temp[n++] = data[j];
            }
          } else {
            for (int j = i-w; j <= i+w; j++) {
              temp[n++] = data[j];
            }
          }
        } else {
          if (row < w) {
            for (int j = col; j <= i+w*cols; j+=cols) {
              temp[n++] = data[j];
            }
          } else if (row >= rows-w) {
            for (int j = i-w*cols; j < rows*cols+col; j+=cols) {
              temp[n++] = data[j];
            }
          } else {
            for (int j = i-w*cols; j <= i+w*cols; j+=cols) {
              temp[n++] = data[j];
            }
          }
        }
        bubblesort_float(temp, n);
        if (n % 2 == 0) {
          smoothed[i] = (temp[n/2]+temp[n/2-1])/2.0f;
        } else {
          smoothed[i] = temp[n/2];
        }
      }

      /************** Noisemap methods ***********/
      __global__ void createNoisemaps_int(int* image, float* nm, float gain, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        nm[i] = sqrt(abs(image[i]/gain));
      }

      __global__ void createNoisemaps_float(float* image, float* nm, float gain, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        nm[i] = sqrt(abs(image[i]/gain));
      }

      __global__ void noisemaps_ds_float(float* nm, float* nmdark, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        nm[i] = sqrt(nm[i]*nm[i]+nmdark[i]*nmdark[i]);
      }

      __global__ void noisemaps_fd_float(float* image, float* nm, float* oldimage, float *nmflat, float* masterFlat, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        if (oldimage[i] == 0 || masterFlat[i] == 0) {
          nm[i] = 0;
        } else {
          nm[i] = abs(image[i])*sqrt( (nm[i]*nm[i]) / (oldimage[i]*oldimage[i]) + (nmflat[i]*nmflat[i]) / (masterFlat[i]*masterFlat[i]));
        }
      }

      __global__ void noisemaps_mflat_dome_on_off_float(float* nm, float* on, float* off, float ncomb1, float ncomb2, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        nm[i] = sqrt(abs(on[i]/ncomb1)+abs(off[i]/ncomb2));
      }

      __global__ void noisemaps_sqrtAndDivide_float(float* dividend, float* divisor, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        if (divisor[i] == 0) {
          dividend[i] = 0;
          return;
        }
        dividend[i] = sqrt(dividend[i])/divisor[i];
      }

      /************end noisemap methods **************/

      __global__ void normalizeFlat_float(float *flat, float med, float lowThresh, float lowRep, float hiThresh, float hiRep, int *lowct, int *hict, int doNM, float *nm, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        flat[i] /= med;
        if (doNM == 1) {
          nm[i] /= med;
        }
        if (lowThresh != 0 && flat[i] < lowThresh) {
          flat[i] = lowRep;
          atomicAdd(&lowct[0], 1);
        }
        if (hiThresh != 0 && flat[i] > hiThresh) {
          flat[i] = hiRep;
          atomicAdd(&hict[0], 1);
        }
      }

      __global__ void normalizeMOSFlat_float(float *flat, int *slitmask, float *medians, float lowThresh, float lowRep, float hiThresh, float hiRep, int *lowct, int *hict, int doNM, float *nm, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        if (slitmask[i] == 0) {
          flat[i] = 0;
          if (doNM == 1) {
            nm[i] = 0;
          }
          return;
        }

        flat[i] /= medians[slitmask[i]-1];
        if (doNM == 1) {
          nm[i] /= medians[slitmask[i]-1];
        }
        if (lowThresh != 0 && flat[i] < lowThresh) {
          flat[i] = lowRep;
          atomicAdd(&lowct[0], 1);
        }
        if (hiThresh != 0 && flat[i] > hiThresh) {
          flat[i] = hiRep;
          atomicAdd(&hict[0], 1);
        }
      }

      __global__ void normalizeMOSSource_float(float *source, int *slitmask, float *medians, int size) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        if (slitmask[i] == 0) {
          source[i] = 0;
          return;
        }
        source[i] /= medians[slitmask[i]-1];
      }

      __global__ void rawToFlatDivided(float *data, float *output, float *linCoeffs, int nCoeffs, float coaddRead, float *masterDark,
                                    float *masterFlat, int *bpm, int *steps) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        int n = 1;
        float x;
        if (coaddRead != 1) {
            //Divide by coadds * nreads
            data[i] /= coaddRead;
        }
        if (steps[0] == 1) {
            //Do linearity
            n = 1;
            x = data[i]*linCoeffs[0];
            for (int j = 1; j < nCoeffs; j++) {
                n++;
                x += linCoeffs[j] * pow(data[i], n);
            }
            data[i] = x;
        }
        if (steps[1] == 1) {
            //Do dark subtract
            data[i] -= masterDark[i];
        }
        if (steps[2] == 1) {
            //Do flat divide
            if (masterFlat[i] == 0) {
              data[i] = 0;
            } else {
              data[i] /= masterFlat[i];
            }
        }
        if (steps[3] == 1) {
            //Apply bpm
            if (bpm[i] == 1) {
              data[i] = 0;
            }
        }
        output[i] = data[i];
      }

      __global__ void shiftAddSlitmask(int *slitmask, int *ylo, int *yhi, int cols, int nslits, int size, int horizontal) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= size) return;
        int row = i/cols;
        if (!horizontal) row = i%cols;
        for (int k = 0; k < nslits; k++) {
          if (slitmask[i] == (k+1)) {
            atomicMin(&ylo[k], row);
            atomicMax(&yhi[k], row);
          }
        }
      }

      __global__ void smooth1d_float(float *data, float *smoothed, int n, int w) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int width = 2*w+1;
        smoothed[i] = 0;
        if (i < w) {
          for (int j = 0; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (i+w+1);
        } else if (i >= n-w) {
          for (int j = i-w; j < n; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (n-i+w);
        } else {
          for (int j = i-w; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= width;
        }
      }

      __global__ void smooth1d_int(int *data, float *smoothed, int n, int w) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int width = 2*w+1;
        smoothed[i] = 0;
        if (i < w) {
          for (int j = 0; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (i+w+1);
        } else if (i >= n-w) {
          for (int j = i-w; j < n; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (n-i+w);
        } else {
          for (int j = i-w; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= width;
        }
      }

      __global__ void smooth1d_long(long *data, float *smoothed, int n, int w) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int width = 2*w+1;
        smoothed[i] = 0;
        if (i < w) {
          for (int j = 0; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (i+w+1);
        } else if (i >= n-w) {
          for (int j = i-w; j < n; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (n-i+w);
        } else {
          for (int j = i-w; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= width;
        }
      }

      __global__ void smooth1d_double(double *data, double *smoothed, int n, int w) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int width = 2*w+1;
        smoothed[i] = 0;
        if (i < w) {
          for (int j = 0; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (i+w+1);
        } else if (i >= n-w) {
          for (int j = i-w; j < n; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= (n-i+w);
        } else {
          for (int j = i-w; j <= i+w; j++) {
            smoothed[i] += data[j];
          }
          smoothed[i] /= width;
        }
      }

      __global__ void smooth1d2d_float(float *data, float *smoothed, int rows, int cols, int w, int axis) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        int width = 2*w+1;
        int row = i/cols;
        int col = i%cols;
        smoothed[i] = 0;
        if (axis == 0) {
          if (col < w) {
            for (int j = row*cols; j <= i+w; j++) {
              smoothed[i] += data[j];
            }
            smoothed[i] /= (col+w+1);
          } else if (col >= cols-w) {
            for (int j = i-w; j < (row+1)*cols; j++) {
              smoothed[i] += data[j];
            }
            smoothed[i] /= (cols-col+w);
          } else {
            for (int j = i-w; j <= i+w; j++) {
              smoothed[i] += data[j];
            }
            smoothed[i] /= width;
          }
        } else {
          if (row < w) {
            for (int j = col; j <= i+w*cols; j+=cols) {
              smoothed[i] += data[j];
            }
            smoothed[i] /= (row+w+1);
          } else if (row >= rows-w) {
            for (int j = i-w*cols; j < rows*cols+col; j+=cols) {
              smoothed[i] += data[j];
            }
            smoothed[i] /= (rows-row+w);
          } else {
            for (int j = i-w*cols; j <= i+w*cols; j+=cols) {
              smoothed[i] += data[j];
            }
            smoothed[i] /= width;
          }
        }
      }

      __global__ void smooth2d_float(float *data, float *smoothed, int rows, int cols, int w) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        smoothed[i] = 0;
        int row = i/cols;
        int col = i % cols;
        int npts = 0;
        int start_row = row >= w ? row-w : 0;
        int end_row = row < rows-w ? row+w : rows-1;
        int start_col = col >= w ? col-w : 0;
        int end_col = col < cols-w ? col+w : cols-1;
        for (int j = start_row; j <= end_row; j++) {
          for (int k = start_col; k <= end_col; k++) {
            smoothed[i] += data[cols*j+k];
            npts++;
          }
        }
        smoothed[i] /= npts;
      }

      __global__ void subtractArrays_float(float* image1, float* image2) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        image1[i] -= image2[i];
      }

      __global__ void subtractArrays_gpm_float(float* image1, float* image2, int *gpm) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (gpm[i] != 0) {
          image1[i] -= image2[i];
        }
      }

      __global__ void subtractArrays_scaled_float(float* image1, float* image2, float scale) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        image1[i] -= image2[i]*scale;
      }

      __global__ void whereEqual_float(float *data, float val, int *idx) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (data[i] == val) atomicExch(&idx[0], i);
      }

      __global__ void whereEqual_int(int *data, int val, int *idx) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (data[i] == val) atomicExch(&idx[0], i);
      }

      __global__ void whereEqual_double(double *data, double val, int *idx) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (data[i] == val) atomicExch(&idx[0], i);
      }

      __global__ void whereEqual_long(long *data, long val, int *idx) {
        const int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (data[i] == val) atomicExch(&idx[0], i);
      }
    """)
    return fatboy_mod
#end get_fatboy_mod

fatboy_mod = get_fatboy_mod()

def doItAll(fdu, params, log, linCoeffs=None, darkFdu=None, flatFdu=None, bpm=None, skyFdu=None, stepsToDo=dict()):
    #1. Reject
    #2. linearity/nreads
    #3. dark subtract
    #4. flat divide
    #5. bpm
    #6. sky subtract
    #7. remove cosmic rays
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("rawToFlatDivided")
    t = time.time()

    #Rejection
    if (stepsToDo.get('rejection')):
        if (params['MIN_FRAME_VALUE'] != 0 or params['MAX_FRAME_VALUE'] != 0):
            keep = True
            #check min/max
            if (fdu.getMedian() < params['MIN_FRAME_VALUE'] and not fdu.isDark):
                keep = False
            if (params['MAX_FRAME_VALUE'] != 0 and fdu.getMedian() > params['MAX_FRAME_VALUE']):
                keep = False
            if (not keep):
                print("Warning: Ignoring file "+fdu.filename)
                log.writeLog(__name__, "Ignoring file "+fdu.filename, type=fatboyLog.WARNING)
                fdu.disable()
                return False

    t2 = time.time()

    #Get data
    data = fdu.getData()
    if (data.dtype != float32):
        data = float32(data)

    #Create empty output array
    output = empty(data.shape, float32)
    rows = data.shape[0]
    cols = data.shape[1]
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    #Steps
    steps = zeros(4, int32)

    #Coadds and nreads
    coaddRead = 1
    if (params['DIVIDE_BY_COADD'].lower() == "yes"):
        coaddRead = fdu.coadds*fdu.nreads

    #Linearity
    nCoeffs = 0
    if (stepsToDo.get('linearity') and linCoeffs is not None):
        nCoeffs = len(linCoeffs)
        if (nCoeffs != 1 or linCoeffs[0] != 1):
            steps[0] = 1
            linCoeffs = float32(linCoeffs)
    else:
        linCoeffs = ones(1, float32)

    t3 = time.time()

    #Dark subtraction
    if (stepsToDo.get('darkSubtract') and darkFdu is not None):
        masterDark = darkFdu.getData()
        steps[1] = 1
    else:
        masterDark = zeros(1, float32)

    #Flat division
    if (stepsToDo.get('flatDivide') and flatFdu is not None):
        if (bpm is not None):
            flatFdu.renormalize(bpm)
        masterFlat = flatFdu.getData()
        steps[2] = 1
    else:
        masterFlat = zeros(1, float32)

    #Bad pixel mask
    if (stepsToDo.get('badPixelMask') and bpm is not None):
        bpm = int32(bpm)
        steps[3] = 1
    else:
        bpm = zeros(1, int32)

    t4 = time.time()
    #Run rawToFlatDivided
    kernel(drv.In(data), drv.Out(output), drv.In(linCoeffs), int32(nCoeffs), float32(coaddRead), drv.In(masterDark), drv.In(masterFlat), drv.In(bpm), drv.In(steps), grid=(blocks,1), block=(block_size,1,1))
    t5 = time.time()

    #Update fdu
    data = output
    fdu.updateData(data)

    #Sky subtraction
    if (stepsToDo.get('skySubtract') and skyFdu is not None):
        skySubtractImage = fatboy_mod.get_function("subtractArrays_scaled_float")
        masterSky = skyFdu.getData()
        skyScale = fdu.getMedian()/skyFdu.getMedian()
        skySubtractImage(drv.InOut(data), drv.In(masterSky), float32(skyScale), grid=(blocks,1), block=(block_size,1,1))

    t6 = time.time()

    #Cosmic ray removal
    if (stepsToDo.get('removeCosmicRays')):
        cosmicRayRemoval = fatboy_mod.get_function("cosmicRayRemoval_float")
        crpass = params['COSMIC_RAY_PASSES']
        ict = zeros(1, int32)
        for j in range(crpass):
            cosmicRayRemoval(drv.In(data), drv.Out(output), int32(rows), int32(cols), drv.InOut(ict), grid=(blocks,1), block=(block_size,1,1))
            print(ict,' replaced.')
            if (log is not None):
                log.writeLog(__name__, str(ict)+" replaced.", printCaller=False, tabLevel=1)
            data = output
            ict = zeros(1, int32)

    t7 = time.time()
    #print "Time: setup = "+str(t2-t)+", kernel = "+str(time.time()-t2)+", total = "+str(time.time()-t)
    print("Time: ",t2-t, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t)
    return output
#end doItAll

#apply an object mask from sextractor to an image
def applyObjMask(image, objMask):
    blocks = image.size//block_size
    if (image.size % block_size != 0):
        blocks += 1
    #Make sure obj mask is little endian 32 bit int
    objMask = objMask.astype("int32")
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    applyObjMaskFunc = fatboy_mod.get_function("applyObjMask_float")
    applyObjMaskFunc(drv.InOut(image), drv.In(objMask), grid=(blocks,1), block=(block_size,1,1))
    return image
#end applyObjMask

#grow an object mask from sextractor with a smoothing function and apply it to an image
def apply2PassObjMask(image, objMask, boxcarSize, rejectLevel):
    rows = image.shape[0]
    cols = image.shape[1]
    if (boxcarSize % 2 == 0):
        boxcarSize+=1
    w = boxcarSize//2
    blocks = image.size//block_size
    if (image.size % block_size != 0):
        blocks += 1

    #Make sure obj mask is little endian 32 bit int
    objMask = objMask.astype("int32")
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    createObjMaskFunc = fatboy_mod.get_function("createObjMask")
    createObjMaskFunc(drv.InOut(objMask), grid=(blocks,1), block=(block_size,1,1))
    growApplyObjMaskFunc = fatboy_mod.get_function("growApplyObjMask_float")
    growApplyObjMaskFunc(drv.InOut(image), drv.In(objMask), int32(rows), int32(cols), int32(w), float32(rejectLevel), grid=(blocks,1), block=(block_size,1,1))
    return image
#end apply2PassObjMask

#Turn an x*fac1 by y*fac2 array into an x by y array by averaging pixels
def blkavg(data, outfile=None, faccol=1, facrow=1, mef=0, log=None):
    #Turn an x*fac1 by y*fac2 array into an x by y array by averaging pixels
    if (isinstance(data, str)):
        if (os.access(data, os.F_OK)):
            outimage = pyfits.open(data)
            data = outimage[mef].data
        else:
            print("blkavg> Error: File "+data+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
    elif (isinstance(data, ndarray)):
        if (outfile is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)
    else:
        print("blkavg> Error: Input must be a FITS file or a raw array.")
        if (log is not None):
            log.writeLog(__name__, "Input must be a FITS file or a raw array.", type=fatboyLog.ERROR)
        return None

    rows = data.shape[0]
    cols = data.shape[1]
    data = data.astype(float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    blkavgFunc = fatboy_mod.get_function("blkavg_float")
    outtype = float32
    out = empty((rows//facrow, cols//faccol), outtype)
    blocks = out.size//512
    if (out.size % 512 != 0):
        blocks += 1
    blkavgFunc(drv.In(data), drv.Out(out), int32(faccol), int32(facrow), int32(cols//faccol), int32(rows//facrow), grid=(blocks,1), block=(block_size,1,1))

    if (outfile is not None):
        outimage[mef].data = out
        outimage.verify('silentfix')
        outimage.writeto(outfile)
        outimage.close()
    return out
#end blkavg

#Turn an x by y array into an x*fac1 by y*fac2 array by replicating pixels
def blkrep(data, outfile=None, faccol=1, facrow=1, mef=0, log=None):
    #Turn an x by y array into an x*fac1 by y*fac2 array by replicating pixels
    if (isinstance(data, str)):
        if (os.access(data, os.F_OK)):
            outimage = pyfits.open(data)
            data = outimage[mef].data
        else:
            print("blkrep> Error: File "+data+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
    elif (isinstance(data, ndarray)):
        if (outfile is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)
    else:
        print("blkrep> Error: Input must be a FITS file or a raw array.")
        if (log is not None):
            log.writeLog(__name__, "Input must be a FITS file or a raw array.", type=fatboyLog.ERROR)
        return None

    rows = data.shape[0]
    cols = data.shape[1]
    data = data.astype(float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    blkrepFunc = fatboy_mod.get_function("blkrep_float")
    outtype = float32
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    out = empty((rows*facrow, cols*faccol), outtype)
    blkrepFunc(drv.In(data), drv.Out(out), int32(faccol), int32(facrow), int32(cols), int32(rows), grid=(blocks,1), block=(block_size,1,1))

    if (outfile is not None):
        outimage[mef].data = out
        outimage.verify('silentfix')
        outimage.writeto(outfile)
        outimage.close()
    return out
#end blkrep

#GPU equivalent of surface3dFunction
def calcTrans3d(xin, yin, z, ycoeffs, order):
    xsize = xin.size
    ycoeffs = float32(ycoeffs)
    yout = empty(shape=yin.shape, dtype=float32)
    blocks = (yout.size)//512
    if (yout.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("calcTrans3d")
    kernel(drv.Out(yout), drv.In(xin), drv.In(yin), drv.In(z), drv.In(ycoeffs), int32(order), int32(xsize), int32(yin.size), grid=(blocks,1), block=(block_size,1,1))
    return yout
#end calcTrans3d

#Calculate an array where every value is its X coordinate
def calcXin(xsize, ysize):
    xin = empty(shape=(ysize,xsize), dtype=float32)
    blocks = (xin.size)//512
    if (xin.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("calcXin")
    kernel(drv.Out(xin), int32(xsize), int32(xin.size), grid=(blocks,1), block=(block_size,1,1))
    return xin
#end calcXin

#Calculate an array where every value is its Y coordinate
def calcYin(xsize, ysize):
    yin = empty(shape=(ysize,xsize), dtype=float32)
    blocks = (yin.size)//512
    if (yin.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("calcYin")
    kernel(drv.Out(yin), int32(xsize), int32(yin.size), grid=(blocks,1), block=(block_size,1,1))
    return yin
#end calcYin

#Perform a 2-d convolution of data with kernel
def convolve2d(data, kernel, outfile=None, boundary="nearest", mef=0, maskNegative=False, log=None):
    if (isinstance(data, str)):
        if (os.access(data, os.F_OK)):
            outimage = pyfits.open(data)
            data = outimage[mef].data
        else:
            print("convolve2d> Error: File "+data+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
    elif (isinstance(data, ndarray)):
        if (outfile is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)
    else:
        print("convolve2d> Error: Input must be a FITS file or a raw array.")
        if (log is not None):
            log.writeLog(__name__, "Input must be a FITS file or a raw array.", type=fatboyLog.ERROR)
        return None

    ## INVERT KERNEL and ensure that it is float32 ##
    kernel = kernel[::-1,::-1].astype(float32)
    rows = data.shape[0]
    cols = data.shape[1]
    kny = kernel.shape[0]
    knx = kernel.shape[1]
    bnd = 0
    if (boundary == "nearest"):
        bnd = 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    convolveFunc = fatboy_mod.get_function("convolve2d_float")
    outtype = float32
    data = data.astype(float32)
    out = empty(data.shape, outtype)
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    convolveFunc(drv.In(data), drv.Out(out), drv.In(kernel), int32(rows), int32(cols), int32(kny), int32(knx), int32(bnd), int32(maskNegative), grid=(blocks,1), block=(block_size,1,1))

    if (outfile is not None):
        outimage[mef].data = out
        outimage.verify('silentfix')
        outimage.writeto(outfile)
        outimage.close()
    return out
#end convolve2d

#Perform a 2-d convolution of data with kernel and blkavg result
def convolve2dAndBlk(data, kernel, outfile=None, facrow=1, faccol=1, boundary="nearest", mef=0, maskNegative=False, log=None):
    if (isinstance(data, str)):
        if (os.access(data, os.F_OK)):
            outimage = pyfits.open(data)
            data = outimage[mef].data
        else:
            print("convolve2dAndBlk> Error: File "+data+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
    elif (isinstance(data, ndarray)):
        if (outfile is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)
    else:
        print("convolve2dAndBlk> Error: Input must be a FITS file or a raw array.")
        if (log is not None):
            log.writeLog(__name__, "Input must be a FITS file or a raw array.", type=fatboyLog.ERROR)
        return None

    rows = data.shape[0]
    cols = data.shape[1]
    kny = kernel.shape[0]
    knx = kernel.shape[1]
    bnd = 0
    if (boundary == "nearest"):
        bnd = 1

    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    convolveFunc = fatboy_mod.get_function("convolve2dAndBlk_float")
    outtype = float32
    data = data.astype(float32)
    out = empty((rows//facrow, cols//faccol), outtype)
    blocks = out.size//512
    if (out.size % 512 != 0):
        blocks += 1
    convolveFunc(drv.In(data), drv.Out(out), drv.In(kernel), int32(rows//facrow), int32(cols//faccol), int32(kny), int32(knx), int32(bnd), int32(maskNegative), int32(facrow), int32(faccol), grid=(blocks,1), block=(block_size,1,1))

    if (outfile is not None):
        outimage[mef].data = out
        outimage.verify('silentfix')
        outimage.writeto(outfile)
        outimage.close()
    return out
#end convolve2dAndBlk

#Create a binary FITS table
def createFitsTable(columns):
    if (useAstropy):
        tbhdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(columns))
    elif (hasattr(pyfits, '__version__') and pyfits.__version__ >= '3.1'):
        tbhdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs(columns))
    else:
        tbhdu = pyfits.new_table(pyfits.ColDefs(columns))
    return tbhdu
#end createFitsTable

#Create a noisemap
def createNoisemap(image, gain=1.0):
    blocks = image.size//512
    if (image.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    gpu_noisemap = fatboy_mod.get_function("createNoisemaps_float")
    if (image.dtype == int32):
        gpu_noisemap = fatboy_mod.get_function("createNoisemaps_int")
    nm = empty(image.shape, float32)
    gpu_noisemap(drv.In(image), drv.Out(nm), float32(gain), int32(image.size), grid=(blocks,1), block=(block_size,1,1))
    return nm
#end createNoisemap

#Create a slitmask using the GPU
def createSlitmask(shp, rslitHi, rslitLo, nslits, horizontal):
    slitmask = empty(shape=shp, dtype=int32)
    rows = shp[0]
    cols = shp[1]
    blocks = slitmask.size//512
    if (slitmask.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("createSlitmask")
    kernel(drv.Out(slitmask), drv.InOut(int32(rslitHi)), drv.In(int32(rslitLo)), int32(cols), int32(rows), int32(nslits), int32(horizontal), int32(slitmask.size), grid=(blocks,1), block=(block_size,1,1))
    return slitmask
#end createSlitmask

def dcr(image, clean_file=None, crfile=None, slitmask=None, thresh=4.0, xrad=9, yrad=9, npass=5, diaxis=1, lrad=1, urad=3, grad=1, verbose=1, mef=-1, log=None):
    if (isinstance(image, str)):
        if (os.access(image, os.F_OK)):
            outimage = pyfits.open(image)
            if (mef == -1):
                #find first data extension
                mef = findMef(outimage)
            data = outimage[mef].data.astype(float32) #make sure to convert to little endian 32-bit float
        else:
            print("dcr> Error: could not find image "+image)
            if (log is not None):
                log.writeLog(__name__, "could not find image "+image, type=fatboyLog.ERROR)
            return None
    elif (isinstance(image, ndarray)):
        data = image.astype(float32) #copy image to data and make sure to convert to little endian 32-bit float
        if (clean_file is not None or crfile is not None):
            #create new pyfits HDUlist object for output
            outimage = pyfits.HDUList()
            outimage.append(pyfits.PrimaryHDU())
            mef = 0
    else:
        print("dcr> Error: invalid datatype.  Must be a FITS file or numpy ndarray.")
        if (log is not None):
            log.writeLog(__name__, "invalid datatype.  Must be a FITS file or numpy ndarray.", type=fatboyLog.ERROR)
        return None

    npix = 0
    if (slitmask is None):
        #process full frame. data will be modified to have cleaned data and return value cr has cr data
        (npix, cr_data) = fatboyclib.dcr(data, thresh=thresh, xrad=xrad, yrad=yrad, npass=npass, diaxis=diaxis, lrad=lrad, urad=urad, grad=grad, verbose=verbose)
    else:
        #Multi-object (or multi-order) spectroscopy.  Use slitmask, which is array same size as data where each pixel is an integer from 1 to n
        #representing the slitlet that pixel belongs to.
        if (isinstance(slitmask, str)):
            if (os.access(slitmask, os.F_OK)):
                sm = pyfits.open(slitmask)
                #find first data extension
                mef = findMef(sm)
                smdata = sm[mef].data
            else:
                print("dcr> Error: could not find slitmask "+slitmask)
                if (log is not None):
                    log.writeLog(__name__, "could not find slitmask "+slitmask, type=fatboyLog.ERROR)
                return None
        elif (isinstance(slitmask, ndarray)):
            smdata = slitmask
        else:
            print("dcr> Error: invalid slitmask datatype.  Must be a FITS file or numpy ndarray.")
            if (log is not None):
                log.writeLog(__name__, "invalid slitmask datatype.  Must be a FITS file or numpy ndarray.", type=fatboyLog.ERROR)
            return None
        if (smdata.shape != data.shape):
            print("dcr> Error: slitmask shape "+str(smdata.shape)+" is differnt from data shape "+str(data.shape))
            if (log is not None):
                log.writeLog(__name__, "slitmask shape "+str(smdata.shape)+" is differnt from data shape "+str(data.shape), type=fatboyLog.ERROR)
            return None
        #Create arrays for clean_data and cr_data
        clean_data = zeros(smdata.shape, dtype=float32)
        cr_data = zeros(smdata.shape, dtype=float32)
        nslits = smdata.max() #number of slits
        #Loop over slitlets
        for j in range(nslits):
            slit = where(smdata == (j+1))
            if (diaxis != 2):
                #Default is to assume horizontal dispersion for slits
                ylo = slit[0].min()
                yhi = slit[0].max()+1
                tempMask = smdata[ylo:yhi,:] == (j+1)
                slit = (data[ylo:yhi,:]*tempMask).astype(float32)
                if (yrad > (yhi-ylo)/2):
                    yrad = int((yhi-ylo)/2)
                #Run DCR on this one slit.  Put return value into cr_data array. slit will contain cleaned_data
                (np, cr_slit) = fatboyclib.dcr(slit, thresh=thresh, xrad=xrad, yrad=yrad, npass=npass, diaxis=diaxis, lrad=lrad, urad=urad, grad=grad, verbose=verbose)
                npix += np
                cr_data[ylo:yhi,:][tempMask] = cr_slit[tempMask].astype(float32)
                clean_data[ylo:yhi,:][tempMask] = slit[tempMask]
            else:
                #Vertical dispersion for slits
                xlo = slit[1].min()
                xhi = slit[1].max()+1
                tempMask = smdata[:,xlo:xhi] == (j+1)
                slit = (data[:,xlo:xhi]*tempMask).astype(float32)
                if (xrad > (xhi-xlo)/2):
                    xrad = int((xhi-xlo)/2)
                #Run DCR on this one slit.  Put return value into cr_data array. slit will contain cleaned_data
                (np, cr_slit) = fatboyclib.dcr(slit, thresh=thresh, xrad=xrad, yrad=yrad, npass=npass, diaxis=diaxis, lrad=lrad, urad=urad, grad=grad, verbose=verbose)
                npix += np
                cr_data[:,xlo:xhi][tempMask] = cr_slit[tempMask].astype(float32)
                clean_data[:,xlo:xhi][tempMask] = slit[tempMask]
            if (verbose > 0):
                print("\tSlit "+str((j+1))+": cleaned "+str(np)+" pixels.")
                if (log is not None):
                    log.writeLog(__name__, "Slit "+str((j+1))+": cleaned "+str(np)+" pixels.", printCaller=False, tabLevel=1)
        #Copy clean_data array back to data
        data = clean_data
    print("dcr> Cleaned "+str(npix)+" pixels.")
    if (log is not None):
        log.writeLog(__name__, "Cleaned "+str(npix)+" pixels.")
    if (clean_file is not None):
        #write out clean file
        outimage[mef].data = data
        outimage.verify('silentfix')
        outimage.writeto(clean_file, output_verify='silentfix')
    if (crfile is not None):
        outimage[mef].data = cr_data
        outimage.verify('silentfix')
        outimage.writeto(crfile, output_verify='silentfix')
        outimage.close()
    #Returns 3-element list [npix, clean_data, cr_data]
    return [npix, data, cr_data]
#end dcr

#Divide two arrays using the GPU
def divideArraysFloatGPU(dividend, divisor, log=None):
    if (dividend.size != divisor.size):
        print("fatboyLibs::divideArraysFloatGPU> Error: array size mismatch!")
        if (log is not None):
            log.writeLog(__name__, "array size mismatch!", type=fatboyLog.ERROR)
        #Return None
        return None
    blocks = dividend.size//512
    if (dividend.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    divArrays = fatboy_mod.get_function("divideArrays_float")
    divArrays(drv.InOut(dividend), drv.In(divisor), int32(dividend.size), grid=(blocks,1), block=(block_size,1,1))
    return dividend
#end divideArraysFloatGPU

#extract non-zero regions from 1-d data
#used for auto-finding slitlets in normalized master flats
def extractNonzeroRegions(data, width, nspec=0, sort=False):
    specIdx = where(data > 0)[0]
    if (len(specIdx) == 0):
        #Nothing found greater than zero
        return None
    cw = 0 #continuum width
    last = specIdx[0]

    specList = []
    meanVals = []
    for j in range(1, len(specIdx)):
        if (specIdx[j]-specIdx[j-1] != 1):
            #There is a gap between this point and last significant point
            if (cw >= width):
                #width of last continuum is above minimum threshold
                #append [ylo, yhi] to list of spectra
                ylo = specIdx[j-cw-1]
                yhi = specIdx[j-1]
                specList.append([ylo, yhi])
                #Append mean value of ylo to yhi, inclusive
                meanVals.append(data[ylo:yhi+1].mean())
            #set continuum width = 0
            cw = 0
        else:
            #These points are consecutive.  Add to continuum width
            cw+=1

    #Loop has finished, add last continuum to list if > min width
    if (cw >= width):
        ylo = specIdx[-1-cw]
        yhi = specIdx[-1]
        specList.append([ylo, yhi])
        #Append mean value of ylo to yhi, inclusive
        meanVals.append(data[ylo:yhi+1].mean())
    if (len(specList) == 0):
        return None

    specList = array(specList)
    meanVals = array(meanVals)
    if (nspec == 0):
        #return all spectra, don't sort unless asked
        if (not sort):
            return specList
        else:
            b = meanVals.argsort()[::-1]
            return specList[b]
    else:
        #return nspec brightest spectra
        #sort and throw away "extra" spectra
        b = meanVals.argsort()[::-1][:nspec]
        return specList[b]
#end extractNonzeroRegions

#extract spectra from 1-d data
def extractSpectra(data, sigma, width, nspec=0, sort=False, minFluxPct=0.001, allowZeros=True):
    n = data.size
    if (allowZeros):
        data = data.copy()
        data[data == 0] = abs(data[data != 0]).min()/10. #Set zeros to 0.1*min value
    norig = (data != 0).sum()
    #array of indices
    ind = arange(n)
    nold = n+1
    b = data != 0 #points to use as background for std dev calcs
    #Mask out 5 pixel box around highest datapoint before first pass
    bmax = where(data == data.max())[0][0]
    b[max(bmax-2,0):min(bmax+3,len(b))] = False
    #Iterative sigma clipping
    niter = 0
    sd = data[b].std()
    while (n < nold and ((b.sum() > norig//4 and b.sum() > 5) or niter == 0)):
        #Break here and don't recalculate medVal and sd if 5 or fewer points
        if (b.sum() <= 5 and niter > 0):
            b = bold
            break
        niter += 1
        #will use CPU for median
        medVal = gpu_arraymedian(data[b].copy(), kernel=fatboyclib.median)
        sdold = sd
        sd = data[b].std()
        bold = b
        #Break here if sd = 0 and use old sd value
        if (sd == 0):
            sd = sdold
            b = bold
            break
        #Update background points to be within +/- 2 sd of medVal
        b = (data < medVal+2*sd) * (data > medVal-2*sd) * (data != 0)
        #update n and nold
        nold = n
        n = data[b].size
        #print niter, medVal, sd, b.sum(), n, nold
    specIdx = where(data > medVal+sigma*sd)[0]
    #print "medVal = ", medVal, "sigma = ", sd, "max = ", (data.max()-medVal)/sd, "npoints = ", len(specIdx), "back pts = ", b.sum()
    #print specIdx
    if (len(specIdx) == 0):
        #Nothing found greater than +sigma
        return None
    cw = 0 #continuum width
    last = specIdx[0]
    specList = []
    meanVals = []
    specMax = data[specIdx].max()
    for j in range(1, len(specIdx)):
        if (specIdx[j]-specIdx[j-1] != 1):
            #There is a gap between this point and last significant point
            if (cw >= width):
                #width of last continuum is above minimum threshold
                #append [ylo, yhi] to list of spectra
                ylo = specIdx[j-cw-1]
                yhi = specIdx[j-1]
                specList.append([ylo, yhi])
                #Append mean value of ylo to yhi, inclusive
                meanVals.append(data[ylo:yhi+1].mean())
            #set continuum width = 0
            cw = 0
        else:
            #These points are consecutive.  Check flux level.
            if (data[specIdx[j]] < specMax*minFluxPct):
                #Less than 0.1% (default) of max value. Create a break here.
                if (cw >= width):
                    #width of last continuum is above minimum threshold
                    #append [ylo, yhi] to list of spectra
                    ylo = specIdx[j-cw-1]
                    yhi = specIdx[j-1]
                    specList.append([ylo, yhi])
                    #Append mean value of ylo to yhi, inclusive
                    meanVals.append(data[ylo:yhi+1].mean())
                #set continuum width = 0
                cw = 0
            else:
                #These points are consecutive.  Add to continuum width
                cw+=1

    #Loop has finished, add last continuum to list if > min width
    if (cw >= width):
        ylo = specIdx[-1-cw]
        yhi = specIdx[-1]
        specList.append([ylo, yhi])
        #Append mean value of ylo to yhi, inclusive
        meanVals.append(data[ylo:yhi+1].mean())
    if (len(specList) == 0):
        return None

    specList = array(specList)
    meanVals = array(meanVals)
    if (nspec == 0):
        #return all spectra, don't sort unless asked
        if (not sort):
            return specList
        else:
            b = meanVals.argsort()[::-1]
            return specList[b]
    else:
        #return nspec brightest spectra
        #sort and throw away "extra" spectra
        b = meanVals.argsort()[::-1][:nspec]
        return specList[b]
#end extractSpectra

#find and fit emission lines in 1D spectrum with Gaussian
#returns list of Gaussian params
def findAndFitLines(oned, nlines=-1, sigthresh=2.0, thresh=None, squareData=True, gaussWidth=2):
    if (thresh is None):
        thresh = gpu_arraymedian(oned)+sigthresh*oned.std()
    lines = []
    refCut = oned.copy()
    while (refCut.max() > thresh and (nlines < 0 or nlines > len(lines))):
        blref = where(refCut == refCut.max())[0][0]
        if (blref < 10 or blref > len(refCut)-11):
            refCut[blref] = 0
            continue
        tempCut = refCut[blref-10:blref+11]
        if (squareData):
            tempCut = tempCut**2
        p = zeros(4, dtype=float64)
        p[0] = max(tempCut)
        p[1] = 10
        p[2] = gaussWidth/sqrt(2)
        p[3] = gpu_arraymedian(tempCut)
        try:
            lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
        except Exception as ex:
            #Error centroiding, continue to next line
            refCut -= gaussFunction(p, arange(len(refCut), dtype=float32))
            refCut[refCut < 0] = 0
            continue
        p = zeros(4)
        p[0] = sqrt(abs(lsq[0][0]))
        p[1] = lsq[0][1]+blref-10
        p[2] = abs(lsq[0][2]*sqrt(2))
        refCut -= gaussFunction(p, arange(len(refCut), dtype=float32))
        refCut[refCut < 0] = 0
        lines.append(p)
    return lines
#end findAndFitLines

#find extension in a MEF
def findMef(image):
    #find mef extension
    for j in range(len(image)):
        if (image[j].data is not None):
            return j
    #no data found
    return -1
#end findMef

#find "regions" from a slitmask
def findRegions(data, nslits, fdu, gpu=True, log=None, regFile=None):
    #First read region file if given to get slitx, slitw.
    #Do first so it doesn't overwrite sylo, syhi
    if (regFile is not None):
        #Read region file
        if (regFile.endswith(".reg")):
            (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=log)
        elif (regFile.endswith(".txt")):
            (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=log)
        elif (regFile.endswith(".xml")):
            (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=log)
            print("fatboyLibs::findRegions> Warning: Invalid extension for region file "+regFile+"!")
            if (log is not None):
                log.writeLog(__name__, "Invalid extension for region file "+regFile+"!", type=fatboyLog.WARNING)
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            slitx = [data.shape[1]//2]*nslits
        else:
            slitx = [data.shape[0]//2]*nslits
        slitw = [3]*nslits
    else:
        if (fdu.hasProperty("regions")):
            (yl, yh, slitx, slitw) = fdu.getProperty("regions")
        else:
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                slitx = [data.shape[1]//2]*nslits
            else:
                slitx = [data.shape[0]//2]*nslits
            slitw = [3]*nslits

    #Now update sylo, syhi
    if (gpu):
        (sylo, syhi) = shiftAddSlitmask(data, nslits, horizontal=(fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
    else:
        sylo = []
        syhi = []
        for slitidx in range(nslits):
            b = where(data == slitidx+1)
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                sylo.append(b[0].min())
                syhi.append(b[0].max())
            else:
                sylo.append(b[1].min())
                syhi.append(b[1].max())
    return (array(sylo), array(syhi), array(slitx), array(slitw))
#end findRegions

#Use scipy.polyval to fit 1d polynomial along axis of 2-d data
def fit1d(input, outfile=None, axis="X", order=3, lsigma=None, hsigma=None, niter=5, mef=0, mask=None, log=None):
    if (isinstance(input, str)):
        if (os.access(input, os.F_OK)):
            outimage = pyfits.open(input)
            input = outimage[mef].data
        else:
            print("fit1d> Error: File "+input+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
    elif (isinstance(input, ndarray)):
        if (outfile is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)
    else:
        print("fit1d> Error: Input must be a FITS file or a raw array.")
        if (log is not None):
            log.writeLog(__name__, "Input must be a FITS file or a raw array.", type=fatboyLog.ERROR)
        return None
    ny = input.shape[0]
    nx = input.shape[1]
    out = zeros((ny,nx), dtype=float32)
    if (axis == "X"):
        xs = arange(nx, dtype=float32)
        for j in range(ny):
            iter = 0
            nold = 0
            b = ones(nx,bool)
            if (mask is not None):
                b *= mask[j,:]
            n = b.sum()
            fit = zeros(nx)
            while (iter < niter and n != nold and n > 100):
                coeffs = np.polyfit(xs[b], input[j,b], order)
                fit = np.polyval(coeffs, xs)
                resid = input[j,:]-fit
                m = resid[b].sum()*(1./n)
                sd = sqrt(add.reduce(resid[b]*resid[b])*(1./(n-1))-m*n*(1./(n-1)))
                if (lsigma is not None):
                    b *= resid >= m-lsigma*sd
                if (hsigma is not None):
                    b *= resid <= m+hsigma*sd
                nold = n
                n = b.sum()
                iter+=1
            out[j,:] = fit
    elif (axis == "Y"):
        ys = arange(ny, dtype=float32)
        for j in range(nx):
            iter = 0
            nold = 0
            b = ones(ny,bool)
            if (mask is not None):
                b *= mask[:,j]
            n = b.sum()
            fit = zeros(ny)
            while (iter < niter and n != nold and n > 100):
                coeffs = scipy.polyfit(ys[b], input[b,j], order)
                fit = scipy.polyval(coeffs, ys)
                resid = input[:,j]-fit
                m = resid[b].sum()*(1./n)
                sd = sqrt(add.reduce(resid[b]*resid[b])*(1./(n-1))-m*n*(1./(n-1)))
                if (lsigma is not None):
                    b *= resid >= m-lsigma*sd
                if (hsigma is not None):
                    b *= resid <= m+hsigma*sd
                nold = n
                n = b.sum()
                iter+=1
            out[:,j] = fit
    if (outfile is not None):
        outimage[mef].data = out
        outimage.verify('silentfix')
        outimage.writeto(outfile)
        outimage.close()
    return out
#end fit1d

#Fit a 1-d Gaussian to data
def fitGaussian(data, maskNeg=False, maxWidth=None, guess=None):
    if (len(data) == 0):
        return [ zeros(4, float64), False ]
    if (guess is not None):
        p = guess
    else:
        p = zeros(4, float64)
        p[0] = data.max()
        b = where(data == p[0])
        if (b[0].size != 0):
            p[1] = b[0][0]
        p[2] = fwhm1d(data)*0.425 #convert from FWHM in pixels
    if (maxWidth is not None):
        if (p[2] > maxWidth):
            p[2] = maxWidth
    if (maskNeg):
        data = data.copy()
        data[data < 0] = 0
        p[3] = 0
    elif (guess is not None):
        p[3] = gpu_arraymedian(data)
    xs = arange(len(data), dtype=float64)
    try:
        lsq = leastsq(gaussResiduals, p, args=(xs, data))
    except Exception as ex:
        #Return False as lsq[1]
        lsq = [ p, False ]
    return lsq
#end fitGaussian

#Fit a 2-d Gaussian to data
def fitGaussian2d(data, maskNeg=False, maxWidth=None, guess=None):
    if (len(data) == 0):
        return [ zeros(5, float64), False ]
    if (guess is not None):
        p = guess
    else:
        p = zeros(5, float64)
        p[0] = data.max()
        b = where(data == p[0])
        if (b[0].size != 0):
            p[1] = b[1][0]
            p[2] = b[0][0]
        p[3] = fwhm2d(data)[0]*0.425 #convert from FWHM in pixels
    if (maxWidth is not None):
        if (p[3] > maxWidth):
            p[3] = maxWidth
    if (maskNeg):
        data = data.copy()
        data[data < 0] = 0
        p[4] = 0
    elif (guess is not None):
        p[4] = gpu_arraymedian(data)
    xin = (arange(data.size) % data.shape[1]).astype(float64)
    yin = (arange(data.size) // data.shape[1]).astype(float64)
    try:
        lsq = leastsq(gaussResiduals2d, p, args=(xin, yin, data.ravel()))
    except Exception as ex:
        #Return False as lsq[1]
        lsq = [ p, False ]
    return lsq
#end fitGaussian

def fitLines(data, nlines, quartileFilter=True, maskNeg=False, maxWidth=2, guess=None, edge=25, linesToMatch=[]):
    fitParams = []
    totalWidth = 0
    nfit = 0

    fitData = zeros(data.size, dtype=float64)
    if (maskNeg):
        #Correct for big negative values
        data[where(data < -100)] = 1.e-6
    if (quartileFilter):
        #Filter the 1-d cut!
        #Use quartile instead of median to get better estimate of background levels!
        #Use 2 passes of quartile filter
        badpix = data == 0 #First find bad pixels
        for i in range(2):
            tempcut = zeros(len(data))
            nh = 25-badpix[:51].sum()//2 #Instead of defaulting to 25 for quartile, use median of bottom half of *nonzero* pixels
            for k in range(25):
                tempcut[k] = data[k] - gpu_arraymedian(data[:51],nonzero=True,nhigh=nh)
            for k in range(25,len(data)-25):
                nh = 25-badpix[k-25:k+26].sum()//2
                tempcut[k] = data[k] - gpu_arraymedian(data[k-25:k+26],nonzero=True,nhigh=nh)
            nh = 25-badpix[len(data)-50:].sum()//2
            for k in range(len(data)-25,len(data)):
                tempcut[k] = data[k] - gpu_arraymedian(data[len(data)-50:],nonzero=True,nhigh=nh)
            #Set zero values to small positive number to avoid being flagged
            tempcut[tempcut == 0] = 1.e-6
            if (maskNeg):
                #Correct for big negative values
                tempcut[where(tempcut < -100)] = 1.e-6
            data = tempcut
        #Set bad pixels back to 0
        data[badpix] = 0
        data[:edge] = 0
        data[-1*edge:] = 0

        #Find and fit brightest line in resid
        for i in range(nlines):
            resid = data - fitData
            blref = where(resid == max(resid[edge+5:-1*edge-5]))[0][0]
            keepLine = False
            while (not keepLine):
                keepLine = True
                for k in range(len(fitParams)):
                    if (abs(blref-fitParams[k][1]) < 15):
                        #Too close to another line
                        keepLine = False
                if (not keepLine):
                    resid[blref-2:blref+3] = 0
                    blref = where(resid == max(resid[edge+5:-1*edge-5]))[0][0]
            #Lines given to match
            if (len(linesToMatch) > i):
                blref = int(linesToMatch[i][1]+0.5)
            #Centroid line for subpixel accuracy
            #Square data to ensure bright line dominates fit
            tempCut = resid[blref-10:blref+11]**2
            p = zeros(4, dtype=float64)
            p[0] = max(tempCut)
            p[1] = 10
            p[2] = 2
            if (nfit > 0):
                p[2] = totalWidth/nfit/sqrt(2)
            p[3] = gpu_arraymedian(tempCut)
            lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
            mcor = lsq[0][1]
            #Check each component's width
            currWidth = abs(lsq[0][2]*sqrt(2))
            if (currWidth > maxWidth*1.25):
                #2.5 -> 1.5 default
                currWidth = maxWidth*0.75
            elif (currWidth > maxWidth):
                #2 -> 1.75 default
                currWidth = maxWidth*0.875
            totalWidth += currWidth
            nfit += 1
            #Add line to lineParams
            p = zeros(4)
            p[0] = sqrt(abs(lsq[0][0]))
            p[1] = blref+mcor-10
            p[2] = currWidth
            fitData += gaussFunction(p, arange(len(fitData), dtype=float64))
            p[3] = sqrt(abs(lsq[0][3]))
            #Keep track of Gaussian params for line
            fitParams.append(p)
        return fitParams
#end fitLines

#format a list as a string including each number therein
def formatList(x, ndec=3):
    if (not isinstance(x, list) and not isinstance(x, ndarray)):
        return str(x)
    s = '['
    for j in range(len(x)):
        if (j > 0):
            s += ', '
        s += formatNum(x[j], ndec=ndec)
    s += ']'
    return s
#end formatList

#format a number for string output
def formatNum(val, ndec=3):
    s = str(val)
    dpos = s.find('.')
    if (dpos == -1):
        #integer
        return s
    epos = s.find('e')
    if (epos == -1):
        #normal decimal
        return s[:dpos+ndec+1]
    elif (epos <= dpos+ndec+1):
        return s
    else:
        return s[:dpos+ndec+1]+s[epos:]
#end formatNum

#estimate fwhm of 1d data
def fwhm1d(data, halfMax=None):
    if (halfMax is None):
        halfMax = data.max()/2.0
    maxIdx = where(data == data.max())[0][0]

    if (halfMax < 0):
        print("fwhm1d> WARNING: Could not calculate FWHM")
        return 1

    #Find last data point < half max
    b = where(data[:maxIdx] < halfMax)
    if (len(b[0]) == 0):
        startIdx = 0
    else:
        startIdx = b[0][-1]

    #Find first data point after max that is < half max
    b = where(data[maxIdx+1:] < halfMax)
    if (len(b[0]) == 0):
        endIdx = len(data)-1
    else:
        endIdx = b[0][0]+maxIdx+1

    if (startIdx >= len(data)-1):
        startIdx = len(data)-2
    if (endIdx <= 0):
        endIdx = 1

    try:
        xstart = startIdx + (halfMax - data[startIdx]) / (data[startIdx+1] - data[startIdx])
        xend = endIdx - (halfMax - data[endIdx]) / (data[endIdx-1] - data[endIdx])
    except Exception as ex:
        #Divide by 0 error
        print("fwhm1d> WARNING: Could not calculate FWHM")
        return 1
    if (xstart < startIdx-1 or xstart > endIdx+1):
        xstart = startIdx
    if (xend > endIdx+1 or xend < startIdx-1):
        xend = endIdx
    return abs(xend-xstart)
#end fwhm1d

#estimate 2-d fwhm of data
def fwhm2d(data, estimateBackground=False):
    #returned tuple contains: (fwhm_mean, fwhm_stddev, array of 4 FWHMs, background value used as zero level)
    # fwhm1ds[0] = FWHM of X cut.
    # fwhm1ds[1] = FWHM of Y cut.
    # fwhm1ds[2] = FWHM of Y = X cut.
    # fwhm1ds[3] = FWHM of Y = -X cut.

    fwhm1ds = zeros(4, float32)

    b = where(data == data.max())
    xpos = b[1][0]
    ypos = b[0][0]

    nx = data.shape[1]
    ny = data.shape[0]
    npix = data.size

    dmax = data[ypos,xpos]
    dtot = 0
    bkgrnd = 0
    halfmax = (dmax + bkgrnd)/2

    if (dmax == 0):
        #return 0
        return [0, 0, fwhm1ds, 0]

    if (estimateBackground):
        dtot = data.sum()
        dtcen = data[ypos-1:ypos+2, xpos-1:xpos+2].sum()
        bkgrnd = (dtot - dtcen)/(npix - 9)

    #X and Y cuts
    fwhm1ds[0] = fwhm1d(data[ypos,:], halfMax=halfmax)
    fwhm1ds[1] = fwhm1d(data[:,xpos], halfMax=halfmax)

    #diagonals are sqrt(2) times larger:
    sq2 = sqrt(2)
    xystart = min(xpos, ypos)
    xyend = min( nx-1-xpos, ny-1-ypos );
    ncut = xystart+xyend+1;
    dcutxy = zeros(ncut, float32)
    for j in range(ncut):
        dcutxy[j] = data[ypos-xystart+j, xpos-xystart+j] #y=x
    fwhm1ds[2] = sq2*fwhm1d(dcutxy, halfMax=halfmax)

    negxystart = min(xpos, ny-1-ypos)
    negxyend = min( nx-1-xpos, ypos );
    ncut = negxystart+negxyend+1
    dcutnegxy = zeros(ncut, float32)
    for j in range(ncut):
        dcutnegxy[j] = data[ypos+negxystart-j, xpos-negxystart+j] #y=-x
    fwhm1ds[3] = sq2 * fwhm1d(dcutnegxy, halfMax=halfmax)

    if (estimateBackground):
        #recompute average background rejecting larger region using new estimate of FWHM:
        rejSize = (int)(ceil(fwhm1ds[0] + fwhm1ds[1]))
        if (rejSize > 1):
            xmin = xpos - rejSize
            xmax = xpos + rejSize
            ymin = ypos - rejSize
            ymax = ypos + rejSize
            if (xmin < 1):
                xmin = 1;
            if (ymin < 1):
                ymin = 1;
            if (xmax > (nx-2)):
                xmax = nx - 2;
            if (ymax > (ny-2)):
                ymax = ny - 2;
            dcen = data[ymin:ymax+1, xmin:xmax+1]
            dtcen = dcen.sum()
            npcen = dcen.size
            bkgrnd = (dtot - dtcen)/(npix - npcen)
            halfmax = (dmax + bkgrnd)/2
            fwhm1ds[0] = fwhm1d(data[ypos,:], halfMax=halfmax)
            fwhm1ds[1] = fwhm1d(data[:,xpos], halfMax=halfmax)
            fwhm1ds[2] = sq2*fwhm1d(dcutxy, halfMax=halfmax)
            fwhm1ds[3] = sq2*fwhm1d(dcutnegxy, halfMax=halfmax)

    return (fwhm1ds.mean(), fwhm1ds.std(), fwhm1ds, bkgrnd)
#end fwhm2d

#Compute fwhm2ds of a data cube
def fwhm2d_cube_gpu(data, flag=None, estimateBackground=False, log=None):
    if (len(data.shape) != 3):
        print("fatboyLibs::fwhm2d_cube_gpu> Error: data array must have 3 dimensions")
        if (log is not None):
            log.writeLog(__name__, "data array must have 3 dimensions", type=fatboyLog.ERROR)
        #Return None
        return None
    if (flag is None):
        flag = ones(data.shape, dtype=int32)
    (depth, ny, nx) = data.shape
    blocks = depth//512
    if (depth % 512 != 0):
        blocks += 1
    fwhms = zeros((depth, 7), dtype=float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    fwhm2d_cube_float = fatboy_mod.get_function("fwhm2d_cube_float")
    fwhm2d_cube_float(drv.In(data.astype(float32)), drv.In(flag.astype(int32)), int32(depth), int32(nx), int32(ny), int32(estimateBackground), drv.Out(fwhms), grid=(blocks,1), block=(block_size,1,1))
    fwhms[:,1] = fwhms[:,2:6].std(1) #calc std dev here, only takes 1-2ms
    return fwhms
#end fwhm2d_cube_gpu

def gaussFunction(p, x):
    f = zeros(len(x), float64)
    x = x.astype(float64)
    z = (x-p[1])/p[2]
    f = p[3]+p[0]*math.e**(-z**2/2)
    return f
#end gaussFunction

def gaussFunction2d(p, x, y):
    f = zeros(x.shape, float64)
    x = x.astype(float64)
    y = y.astype(float64)
    zx = (x-p[1])/p[3]
    zy = (y-p[2])/p[3]
    f = p[4]+p[0]*math.e**(-(zx**2/2+zy**2/2))
    return f
#end gaussFunction2d

def gaussResiduals(p, x, out):
    f = zeros(len(x), float64)
    x = x.astype(float64)
    z = (x-p[1])/p[2]
    f = p[3]+p[0]*math.e**(-z**2/2)
    err = out-f
    return err
#end gaussResiduals

def gaussResiduals2d(p, x, y, out):
    f = zeros(len(x), float64)
    x = x.astype(float64)
    y = y.astype(float64)
    zx = (x-p[1])/p[3]
    zy = (y-p[2])/p[3]
    f = p[4]+p[0]*math.e**(-(zx**2/2+zy**2/2))
    err = out-f
    return err
#end gaussResiduals2d

def generateQAData(data, xcoords, ycoords, sylo, syhi, horizontal=True):
    data = float32(data)
    ycoords = float32(ycoords)
    xcoords = float32(xcoords)
    blocks = data.size//block_size
    if (data.size % 512 != 0):
        blocks += 1
    if (horizontal):
        cols = data.shape[1]
    else:
        cols = data.shape[0]
    cols = data.shape[1]
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    generateQADataFunc = fatboy_mod.get_function("generateQAData_float")
    output = empty(data.shape, data.dtype)
    generateQADataFunc(drv.InOut(data), drv.In(xcoords), drv.In(ycoords), drv.In(float32(sylo)), drv.In(float32(syhi)), int32(xcoords.size), int32(xcoords.size*sylo.size*18), int32(sylo.size), int32(cols), int32(horizontal), grid=(blocks,1), block=(block_size,1,1))
    return data
#end generateQAData

#Input (mx,my) is assummed to be location of maximum pixel in image,
#  and fwhm is either guess or actual FWHM.
def getCentroid(img, mx, my, fwhm, verbose=False):
    (ysize, xsize) = img.shape
    nhalf = (int)(0.637*fwhm)
    if (nhalf < 2):
        nhalf = 2
    nbox = 2*nhalf+1
    xcen = -1
    ycen = -1

    spos = " ( " + str(mx) + ", " + str(my) + " ) "

    if( (mx < nhalf) or ((mx + nhalf) >= xsize) or (my < nhalf) or ((my + nhalf) >= ysize) ):
        #Too near edge of image, return -1, -1
        if(verbose):
            print("Position" + spos + "too near edge of image")
        return (xcen, ycen)

    starbox = img[my-nhalf:my+nhalf+1, mx-nhalf:mx+nhalf+1]
    ir = max(nhalf-1, 1)
    dd = arange(nbox-1, dtype=float64)+0.5-nhalf
    ddsq = dd*dd

    w = (1.0 - 0.5 * (abs(dd)-0.5) / (nhalf-0.5))
    sumc = w.sum()

    # Y partial derivative:
    deriv = roll(starbox, -1, 0) - starbox
    deriv = deriv[0:nbox-1, nhalf-ir:nhalf+ir+1]
    derivtot = deriv.sum(1)

    sumd = (w*derivtot).sum()
    sumxd = (w*dd*derivtot).sum()
    sumxsq = (w*ddsq).sum()

    if (sumxd < 0):
        dy = sumxsq*sumd/(sumc*sumxd)
        if (abs(dy) <= nhalf):
            #ycen = my-dy+0.5
            ycen = my-dy #Do not add 0.5 by convention
        elif (verbose):
            print("Computed Y centroid for position" + spos + "out of range")
    elif (verbose):
        print("Unable to compute Y centroid around position" + spos)

    # X partial derivative:
    deriv = roll(starbox, -1, 1) - starbox
    deriv = deriv[nhalf-ir:nhalf+ir+1, 0:nbox-1]
    derivtot = deriv.sum(0)

    sumd = (w*derivtot).sum()
    sumxd = (w*dd*derivtot).sum()
    sumxsq = (w*ddsq).sum()

    if( sumxd < 0 ):
        dx = sumxsq*sumd/(sumc*sumxd)
        if (abs(dx) <= nhalf):
            #xcen = mx-dx+0.5
            xcen = mx-dx #Do not add 0.5 by convention
        elif (verbose):
            print("Computed X centroid for position" + spos + "out of range")
    elif (verbose):
        print("Unable to compute X centroid around position" + spos)
    return (xcen, ycen)
#end getCentroid

#Use GPU to compute centroids of cuts in 3-d data cube
def getCentroid_cube_gpu(data, mx, my, fwhm, flag=None, verbose=False, log=None):
    if (len(data.shape) != 3):
        print("fatboyLibs::getCentroid_cube_gpu> Error: data array must have 3 dimensions")
        if (log is not None):
            log.writeLog(__name__, "data array must have 3 dimensions", type=fatboyLog.ERROR)
        #Return None
        return None
    if (flag is None):
        flag = ones(data.shape, dtype=int32)
    (depth, ysize, xsize) = data.shape
    blocks = depth//512
    if (depth % 512 != 0):
        blocks += 1
    cens = zeros((depth, 2), dtype=float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    getcentroid_cube_float = fatboy_mod.get_function("getCentroid_cube_float")
    getcentroid_cube_float(drv.In(data.astype(float32)), drv.In(flag.astype(int32)), int32(mx), int32(my), float32(fwhm), int32(depth), int32(xsize), int32(ysize), drv.Out(cens), grid=(blocks,1), block=(block_size,1,1))
    return cens
#end getCentroid_cube_gpu

## Wrapper function to get a tuple with pyfits/astropy version
def getPyfitsVersion():
    pname = "pyfits"
    version = "0"
    if (useAstropy):
        pname = "astropy"
        if (hasattr(astropy, '__version__')):
            version = astropy.__version__
    elif (hasattr(pyfits, '__version__')):
        version = pyfits.__version__
    return (pname, version)
#end getPyfitsVersion

## Get RA or Dec
def getRADec(s,log=None, rel=False, dec=False, file=None):
    if (isinstance(s, float) or isinstance(s, int)):
        if (rel or rel == 'yes'):
            if (dec):
                return s/3600.
            else:
                return s/(3600.*15)
        else:
            if (dec):
                return s
            else:
                return s/15.
    try:
        n = s.find(':')
        x = float(s[:n])
        if (x < 0):
            sign = -1
        else:
            sign = 1
        x = x+sign*float(s[n+1:s.find(':',n+1)])/60.
        x = x+sign*float(s[s.rfind(':')+1:])/3600.
    except Exception:
        print("fatboyLibs::getRADec> Error: malformed RA or Dec: "+s)
        if (log is not None):
            log.writeLog(__name__, "Malformed RA or Dec: "+s+" in "+self.filename+"!", type=fatboyLog.ERROR)
        x = 0.
    return x
#end getRADec

#return a 1-d array with wavelengths from the wavelength solution info in the header
def getWavelengthSolution(fdu, islit, xsize):
    xs = arange(xsize, dtype=float32)
    #Calculate wavelength solution
    wave = zeros(xsize, dtype=float32)
    if (isinstance(fdu, pyfits.header.Header) or isinstance(fdu, dict)):
        if ('PORDER' in fdu):
            wave[:] = fdu['PCOEFF_0']
            for i in range(1, fdu['PORDER']+1):
                wave += fdu['PCOEFF_'+str(i)]*xs**i
            return wave
        if ('PORDER01' in fdu):
            #Use PORDERxx and PCFi_Sxx
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            wave[:] = fdu['PCF0_S'+slitStr]
            for i in range(1, fdu['PORDER'+slitStr]+1):
                wave += fdu['PCF'+str(i)+'_S'+slitStr]*xs**i
            return wave
        if ('PORDER01_SEG0' in fdu):
            #multiple segments
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            nseg = fdu['NSEG_'+slitStr]
            stride = xsize//nseg
            for seg in range(nseg):
                startidx = seg*stride
                endidx = (seg+1)*stride
                if (seg == nseg-1 and xsize % nseg != 0):
                    stride = xsize - startidx
                    endidx = xsize
                segws = zeros(stride, dtype=float32)
                segws[:] = fdu['PCF0_S'+slitStr+'_SEG'+str(seg)]
                for i in range(1, fdu['PORDER'+slitStr+'_SEG'+str(seg)]+1):
                    segws += fdu['PCF'+str(i)+'_S'+slitStr+'_SEG'+str(seg)]*xs[:stride]**i
                wave[startidx:endidx] = segws
                #wave[seg*stride:(seg+1)*stride] = segws
            return wave
        if ('CRVALS01_SEG0' in fdu):
            #multiple segments, resampled
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            nseg = fdu['NSEG_'+slitStr]
            stride = xsize//nseg
            for seg in range(nseg):
                startidx = seg*stride
                endidx = (seg+1)*stride
                if (seg == nseg-1 and xsize % nseg != 0):
                    stride = xsize - startidx
                    endidx = xsize
                segws = fdu['CRVALS'+slitStr+'_SEG'+str(seg)]+xs[:stride]*fdu['CDELTS'+slitStr+'_SEG'+str(seg)]
                wave[startidx:endidx] = segws
                #wave[seg*stride:(seg+1)*stride] = segws
            return wave
        if ('CRVALS01' in fdu):
            #multiple slitlets, one segment
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            wave = fdu['CRVALS'+slitStr]+xs*fdu['CDELTS'+slitStr]
            return wave
        if ('CRVAL1' in fdu):
            #linear
            wave = fdu['CRVAL1']+xs*fdu['CDELT1']
            return wave
        print("fatboyLibs::getWavelengthSolution> Error: could not find wavelength solution.")
        return arange(xsize)
    try:
        if (fdu.hasProperty("wcHeader")):
            hd = fdu.getProperty("wcHeader")
            if ('PORDER' in hd):
                wave[:] = hd['PCOEFF_0']
                for i in range(1, hd['PORDER']+1):
                    wave += hd['PCOEFF_'+str(i)]*xs**i
                return wave
            if ('PORDER01' in hd):
                #Use PORDERxx and PCFi_Sxx
                slitStr = str(islit+1)
                if (islit+1 < 10):
                    slitStr = '0'+slitStr
                wave[:] = hd['PCF0_S'+slitStr]
                for i in range(1, hd['PORDER'+slitStr]+1):
                    wave += hd['PCF'+str(i)+'_S'+slitStr]*xs**i
                return wave
            if ('HIERARCH PORDER01_SEG0' in hd):
                #multiple segments
                slitStr = str(islit+1)
                if (islit+1 < 10):
                    slitStr = '0'+slitStr
                nseg = hd['NSEG_'+slitStr]
                stride = xsize//nseg
                for seg in range(nseg):
                    startidx = seg*stride
                    endidx = (seg+1)*stride
                    if (seg == nseg-1 and xsize % nseg != 0):
                        stride = xsize - startidx
                        endidx = xsize
                    segws = zeros(stride, dtype=float32)
                    segws[:] = hd['HIERARCH PCF0_S'+slitStr+'_SEG'+str(seg)]
                    for i in range(1, hd['HIERARCH PORDER'+slitStr+'_SEG'+str(seg)]+1):
                        segws += hd['HIERARCH PCF'+str(i)+'_S'+slitStr+'_SEG'+str(seg)]*xs[:stride]**i
                    wave[startidx:endidx] = segws
                    #wave[seg*stride:(seg+1)*stride] = segws
                return wave
        if (fdu.hasProperty("resampledHeader")):
            hd = fdu.getProperty("resampledHeader")
            if ('HIERARCH CRVALS01_SEG0' in hd):
                #multiple segments, resampled
                slitStr = str(islit+1)
                if (islit+1 < 10):
                    slitStr = '0'+slitStr
                nseg = hd['NSEG_'+slitStr]
                stride = xsize//nseg
                for seg in range(nseg):
                    startidx = seg*stride
                    endidx = (seg+1)*stride
                    if (seg == nseg-1 and xsize % nseg != 0):
                        stride = xsize - startidx
                        endidx = xsize
                    segws = hd['HIERARCH CRVALS'+slitStr+'_SEG'+str(seg)]+xs[:stride]*hd['CDELTS'+slitStr+'_SEG'+str(seg)]
                    wave[startidx:endidx] = segws
                    #wave[seg*stride:(seg+1)*stride] = segws
                return wave
            if ('CRVALS01' in hd):
                #multiple slitlets, one segment
                slitStr = str(islit+1)
                if (islit+1 < 10):
                    slitStr = '0'+slitStr
                wave = hd['CRVALS'+slitStr]+xs*hd['CDELTS'+slitStr]
                return wave
            if ('CRVAL1' in hd):
                #linear
                wave = hd['CRVAL1']+xs*hd['CDELT1']
                return wave
        if (fdu.hasHeaderValue("PORDER")):
            wave[:] = fdu.getHeaderValue('PCOEFF_0')
            for i in range(1, fdu.getHeaderValue('PORDER')+1):
                wave += fdu.getHeaderValue('PCOEFF_'+str(i))*xs**i
            return wave
        if (fdu.hasHeaderValue("PORDER01")):
            #Use PORDERxx and PCFi_Sxx
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            wave[:] = fdu.getHeaderValue('PCF0_S'+slitStr)
            for i in range(1, fdu.getHeaderValue('PORDER'+slitStr)+1):
                wave += fdu.getHeaderValue('PCF'+str(i)+'_S'+slitStr)*xs**i
            return wave
        if (fdu.hasHeaderValue("PORDER01_SEG0")):
            #multiple segments
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            nseg = fdu.getHeaderValue("NSEG_"+slitStr)
            stride = xsize//nseg
            for seg in range(nseg):
                startidx = seg*stride
                endidx = (seg+1)*stride
                if (seg == nseg-1 and xsize % nseg != 0):
                    stride = xsize - startidx
                    endidx = xsize
                segws = zeros(stride, dtype=float32)
                segws[:] = fdu.getHeaderValue('PCF0_S'+slitStr+"_SEG"+str(seg))
                for i in range(1, fdu.getHeaderValue('PORDER'+slitStr+"_SEG"+str(seg))+1):
                    segws += fdu.getHeaderValue('PCF'+str(i)+'_S'+slitStr+"_SEG"+str(seg))*xs[:stride]**i
                wave[startidx:endidx] = segws
                #wave[seg*stride:(seg+1)*stride] = segws
            return wave
        if (fdu.hasHeaderValue("CRVALS01_SEG0")):
            #multiple segments, resampled
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            nseg = fdu.getHeaderValue("NSEG_"+slitStr)
            stride = xsize//nseg
            for seg in range(nseg):
                startidx = seg*stride
                endidx = (seg+1)*stride
                if (seg == nseg-1 and xsize % nseg != 0):
                    stride = xsize - startidx
                    endidx = xsize
                segws = fdu.getHeaderValue('CRVALS'+slitStr+'_SEG'+str(seg))+xs[:stride]*fdu.getHeaderValue('CDELTS'+slitStr+'_SEG'+str(seg))
                wave[startidx:endidx] = segws
                #wave[seg*stride:(seg+1)*stride] = segws
            return wave
        if (fdu.hasHeaderValue("CRVALS01")):
            #multiple slitlets, one segment
            slitStr = str(islit+1)
            if (islit+1 < 10):
                slitStr = '0'+slitStr
            wave = fdu.getHeaderValue('CRVALS'+slitStr)+xs*fdu.getHeaderValue('CDELTS'+slitStr)
            return wave
        if (fdu.hasHeaderValue("CRVAL1")):
            #linear
            wave = fdu.getHeaderValue('CRVAL1')+xs*fdu.getHeaderValue('CDELT1')
            return wave
        print("fatboyLibs::getWavelengthSolution> Error: could not find wavelength solution.")
        return arange(xsize)
    except Exception as ex:
        print(str(ex))
        print("fatboyLibs::getWavelengthSolution> Error: datatype must be fatboyDataUnit or pyfits Header object.")
    return arange(xsize)
#end getWavelengthSolution

def gpusum(data, lthreshold=None, hthreshold=None, nonzero=False):
    gpu_data = gpuarray.to_gpu(data)
    if (lthreshold is None and hthreshold is None and not nonzero):
        x = gpuarray.sum(gpu_data, float64)
    else:
        expr = ""
        argu = "float *x"
        if (data.dtype == float64):
            argu = "double *x"
        elif (data.dtype == int32):
            argu = "int *x"
        elif (data.dtype == int64):
            argu = "long *x"
        if (lthreshold is not None and hthreshold is not None and nonzero):
            expr = "x[i] >= "+str(lthreshold)+" && x[i] <= "+str(hthreshold)+" && x[i] != 0 ? x[i]:0"
        elif (lthreshold is not None and hthreshold is not None):
            expr = "x[i] >= "+str(lthreshold)+" && x[i] <= "+str(hthreshold)+" ? x[i]:0"
        elif (lthreshold is not None and nonzero):
            expr = "x[i] >= "+str(lthreshold)+" && x[i] != 0 ? x[i]:0"
        elif (hthreshold is not None and nonzero):
            expr = "x[i] <= "+str(hthreshold)+" && x[i] != 0 ? x[i]:0"
        elif (lthreshold is not None):
            expr = "x[i] >= "+str(lthreshold)+" ? x[i]:0"
        elif (hthreshold is not None):
            expr = "x[i] <= "+str(hthreshold)+" ? x[i]:0"
        elif (nonzero):
            expr = "x[i] != 0 ? x[i]:0"
        sumKernel = ReductionKernel(float64, neutral="0", reduce_expr="a+b", map_expr = expr, arguments = argu)
        x = sumKernel(gpu_data)
    return float(x.get())
#end gpusum

#return if an FDU has multiple wavelength solutions for individual slitlets
def hasMultipleWavelengthSolutions(fdu):
    if (isinstance(fdu, pyfits.header.Header)):
        if ('PORDER01' in fdu):
            return True
        if ('PORDER01_SEG0' in fdu):
            return True
        return False
    try:
        if (fdu.hasProperty("wcHeader")):
            hd = fdu.getProperty("wcHeader")
            if ('PORDER01' in hd):
                return True
            if ('HIERARCH PORDER01_SEG0' in hd):
                return True
        if (fdu.hasHeaderValue("PORDER01")):
            return True
        if (fdu.hasHeaderValue("PORDER01_SEG0")):
            return True
    except Exception as ex:
        print("fatboyLibs::hasMultipleWavelengthSolutions> Error: datatype must be fatboyDataUnit or pyfits Header object.")
    return False
#end hasMultipleWavelengthSolutions

#return if an FDU has a wavelength solution
def hasWavelengthSolution(fdu):
    if (isinstance(fdu, pyfits.header.Header)):
        if ('PORDER' in fdu):
            return True
        if ('PORDER01' in fdu):
            return True
        if ('PORDER01_SEG0' in fdu):
            return True
        return False
    try:
        if (fdu.hasProperty("wcHeader")):
            hd = fdu.getProperty("wcHeader")
            if ('PORDER' in hd):
                return True
            if ('PORDER01' in hd):
                return True
            if ('HIERARCH PORDER01_SEG0' in hd):
                return True
        if (fdu.hasHeaderValue("PORDER")):
            return True
        if (fdu.hasHeaderValue("PORDER01")):
            return True
        if (fdu.hasHeaderValue("PORDER01_SEG0")):
            return True
    except Exception as ex:
        print("fatboyLibs::hasWavelengthSolution> Error: datatype must be fatboyDataUnit or pyfits Header object.")
    return False
#end hasWavelengthSolution

def isAlpha(char):
    #upper case
    if (ord(char) >= 65 and ord(char) <= 90):
        return True
    #lower case
    if (ord(char) >= 97 and ord(char) <= 122):
        return True
    return False
#end isAlpha

def isDigit(char):
    if (ord(char) >= 48 and ord(char) <= 57):
        return True
    return False
#end isDigit

#detect if a string contains an integer
def isInt(value):
    for j in range(len(value)):
        char = value[j]
        if (ord(char) == 45):
            if (j == 0):
                continue
            else:
                #- only allowed on first char
                return False
        if (ord(char) < 48):
            return False
        if (ord(char) > 57):
            return False
    return True
#end isInt

#detect if a string contains a float
def isFloat(value):
    ndec = 0
    eprev = False #prev char was E/e
    ne = 0 #number of es
    for j in range(len(value)):
        char = value[j]
        #scientific notation
        if (eprev):
            if (ord(char) == 43 or ord(char) == 45):
                #We already know that there is at least one char after this (if its not a number, it will of course fail)
                # +/-
                eprev = False
                continue
            else:
                return False
        if (j > 0 and ne == 0 and not eprev and char.lower() == 'e'):
            if (len(value) > j+2):
                # +/-
                if (ord(value[j+1]) == 43 or ord(value[j+1]) == 45):
                    eprev = True
                    ne = 1
                    continue
        if (ord(char) == 45 and j != 0):
            #- only allowed on first char or part of scientific notation
            return False
        if (ord(char) == 46):
            # .
            if (ne != 0):
                #. not allowed after e+/-
                return False
            if (len(value) == 1):
                # only a .
                return False
            ndec += 1
            if (ndec > 1):
                #only one . allowed
                return False
            else:
                continue
        if (ord(char) < 48):
            return False
        if (ord(char) > 57):
            return False
    return True
#end isFloat

def isValidFilter(filter):
    invalid = ['open','clear','none']
    filter = str(filter)
    if (invalid.count(filter.lower()) > 0):
        return False
    return True
#end isValidFilter

#### LA Cosmic Routine - GPU version ####
def lacos_spec(indata, outfile, outmask, mef=0, gain=4.1, readn=30.0, xorder = 9, yorder = 3, sigclip = 10, sigfrac = 0.3, objlim = 5, niter = 1, writeOut=True, mask=None, log=None, verbose=False):
    t = time.time()
    mef = 0
    outimage = None
    if (isinstance(indata, str)):
        #Input is FITS file
        bpos = indata.find('[')
        if (bpos == -1):
            fname = indata
        else:
            fname = indata[:bpos]
        if (not os.access(fname, os.F_OK)):
            print("lacos_spec> Error: File "+fname+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
        outimage = pyfits.open(fname)
        mef = findMef(outimage)
        data = outimage[mef].data
    elif (isinstance(indata, ndarray)):
        #Input is raw array
        data = indata
        if (outfile is not None or outmask is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)

    writeMask = True
    if (outmask is None):
        writeMask = False
    if (outfile is None):
        writeOut = False
    if (writeOut and os.access(outfile, os.F_OK)):
        os.unlink(outfile)
    if (outmask is not None and os.access(outmask, os.F_OK)):
        os.unlink(outmask)
    if (not isDigit(str(gain)[0]) and outimage is not None):
        if (gain in outimage[0].header):
            gain = float(outimage[0].header[gain])
    if (not isDigit(str(readn)[0]) and outimage is not None):
        if (readn in outimage[0].header):
            readn = float(outimage[0].header[readn])

    if (gain is None):
        gain = 4.1
    if (readn is None):
        readn = 30.0

    #Mask
    if (mask is None):
        mask = ones(data.shape, dtype=bool)
    elif (isinstance(mask, str)):
        temp = pyfits.open(mask)
        mmef = findMef(temp)
        mask = temp[mmef].data == 0
        temp.close()
    mask = mask.astype(bool)

    # create Laplacian kernel
    kernel = array([[0,-1,0],[-1,4,-1],[0,-1,0]], float32)

    # create growth kernel
    gkernel = ones((3,3), float32)

    # initialize loop
    i = 1
    stop = False
    previous = 0

    shp = data.shape
    oldoutput = float32(data)
    tempOmask = zeros(data.shape, dtype=float32)

    print(oldoutput.mean())

    #subtract object spectra if desired
    if (xorder > 0):
        if (verbose):
            print("\tSubtracting object spectra...")
        galaxy = fit1d(oldoutput,axis="X",order=xorder-1,lsigma=4,hsigma=4,niter=3,mask=mask)
        oldoutput -= galaxy
    elif (xorder == -1):
        galaxy = mediansmooth1d2d(oldoutput, 7, axis=0)
        galaxy = smooth1d2d(galaxy, 5, 10, axis=0)
        oldoutput -= galaxy
    else:
        galaxy = zeros(shp, float32)

    #subtract sky lines
    if (yorder > 0):
        if (verbose):
            print("\tSubtracting sky lines...")
        skymod = fit1d(oldoutput,axis="Y",order=yorder-1,lsigma=4,hsigma=4,niter=3,mask=mask)
        oldoutput -= skymod
    elif (yorder == -1):
        skymod = mediansmooth1d2d(oldoutput, 7, axis=1)
        skymod = smooth1d2d(skymod, 5, 10, axis=1)
        oldoutput -= skymod
    else:
        skymod = zeros(shp, float32)

    print(oldoutput.mean())
    #add object spectra to sky model
    skymod += galaxy
    del galaxy

    #start iterations
    nptotal = 0
    while (not stop):
        print("\tIteration ",i,"...")
        if (log is not None):
            log.writeLog(__name__, "Iteration "+str(i)+"...", printCaller=False, tabLevel=1)
        #add median of residuals to sky model
        med5 = medfilt2d(oldoutput, 5)
        med5 += skymod
        # take second-order derivative (Laplacian) of input image
        # kernel is convolved with subsampled image, in order to remove negative
        # pattern around high pixels
        if (verbose):
            print("\tConvolving with Laplacian Kernel...")
        blk = blkrep(oldoutput,None,2,2)
        deriv2 = convolve2dAndBlk(blk, kernel, facrow=2, faccol=2, maskNegative=True)
        print(med5.mean(), blk.mean(), deriv2.mean())
        del blk
        #create noise model
        if (verbose):
            print("\tCreating noise model...")
        (noise, sigmap) = lacosNoiseModel(med5, deriv2, gain, readn)
        print(noise.mean(), sigmap.mean())
        del deriv2
        #Laplacian of blkreplicated image counts edges twice
        #removal of large structure (bright, extended objects)
        med5 = medfilt2d(sigmap, 5)
        #find all candidate cosmic rays
        #this selection includes sharp features such as stars and HII regions
        if (verbose):
            print("\tSelecting candidate cosmic rays...")
        (sigmap, firstsel) = lacosFirstSel(sigmap, med5, sigclip)
        #compare candidate CRs to median filtered image
        #this step rejects bright, compact sources from the initial CR list
        if (verbose):
            print("\tRemoving suspected emission lines...")
        #subtract background and smooth component of objects
        med3 = medfilt2d(oldoutput, 3)
        med7 = medfilt2d(med3, 7)
        print(med3.mean(), med5.mean(), med7.mean())

        #compare CR flux to object flux
        firstsel = lacosStarReject(med3, med7, noise, firstsel, sigmap, objlim)
        print(firstsel.mean())
        del noise
        del med3
        del med7
        #discard if CR flux <= objlim * object flux

        #grow CRs by one pixel and check in original sigma map
        gfirstsel = float32(convolve2d(firstsel,gkernel))
        del firstsel
        lacosSelect(gfirstsel, sigmap, 0.5, sigclip, 0.1, doCount=False)
        print(gfirstsel.mean(), sigmap.mean())
        #grow CRs by one pixel and lower detection limit
        sigcliplow = sigfrac*sigclip
        if (verbose):
            print("\tFinding neighboring pixels affected by cosmic rays...")
        finalsel = float32(convolve2d(gfirstsel,gkernel))
        del gfirstsel
        (inputmask, npix) = lacosSelect(finalsel, sigmap, 0.5, sigcliplow, 0.1, doCount=True, mask=tempOmask, oldoutput=oldoutput)
        print(finalsel.mean(), inputmask.mean())
        nptotal += npix
        #determine number of CRs found in this iteration
        #create cleaned output image; use 3x3 median with CRs excluded
        if (verbose):
            print("\tCreating output...")
        del finalsel
        med5 = medfilt2d(inputmask, 5, zlo=-9999)

        lacosUpdateOutput(oldoutput, tempOmask, med5, skymod)
        print(tempOmask.mean(), med5.mean(), skymod.mean(), oldoutput.mean())
        del inputmask

        if (i > 1 and writeOut):
            os.unlink(outfile)
        del med5
        if (writeOut):
            outimage[mef].data = oldoutput
            outimage.writeto(outfile)
            outimage.close()
        #clean up and get ready for next iteration
        print("\tFound ",npix," cosmic rays...")
        if (log is not None):
            log.writeLog(__name__, "Found "+str(npix)+" cosmic rays...", printCaller=False, tabLevel=1)
        if (npix == 0):
            stop = True
        i+=1
        if (i > niter):
            stop = True
        #delete temp files
    crmask = 1-tempOmask.astype(int16)
    if (writeMask):
        outimage[mef].data = crmask
        outimage.verify('silentfix')
        outimage.writeto(outmask)
        outimage.close()
    if (verbose):
        print("Total time: ", time.time()-t)
    #Returns tuple of mask, output data
    return (nptotal, crmask, oldoutput)
#end lacos_spec

############# LA Cosmic helper routines #################
def lacosFirstSel(sigmap, med5, sigclip):
    #Returns (sigmap, firstsel)
    sigmap = sigmap.astype(float32)
    firstsel = empty(sigmap.shape, dtype=float32)
    blocks = sigmap.size//512
    if (sigmap.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("lacosFirstSel_float")
    kernel(drv.InOut(sigmap), drv.In(med5), drv.Out(firstsel), float32(sigclip), grid=(blocks,1), block=(block_size,1,1))
    return (sigmap, firstsel)
#end lacosFirstSel

def lacosNoiseModel(med5, deriv2, gain, readn):
    #Returns (noise, sigmap)
    noise = empty(med5.shape, dtype=float32)
    sigmap = empty(med5.shape, dtype=float32)
    med5 = med5.astype(float32)
    deriv2 = deriv2.astype(float32)
    blocks = med5.size//512
    if (med5.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("lacosNoiseModel_float")
    kernel(drv.In(med5), drv.In(deriv2), drv.Out(noise), drv.Out(sigmap), float32(gain), float32(readn), grid=(blocks,1), block=(block_size,1,1))
    return (noise, sigmap)
#end lacosNoiseModel

def lacosSelect(sel, sigmap, lower1, sigclip, lower2, doCount=False, mask=None, oldoutput=None):
    #Returns count if doCount = True
    blocks = sel.size//512
    sel = sel.astype(float32)
    sigmap = sigmap.astype(float32)
    if (sel.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    if (doCount and mask is not None):
        npix = zeros(1, dtype=int32)
        inputmask = empty(sel.shape, dtype=float32)
        kernel = fatboy_mod.get_function("lacosSelectAndCount_float")
        kernel(drv.In(sel), drv.In(sigmap), float32(lower1), float32(sigclip), float32(lower2), drv.InOut(mask), drv.Out(inputmask), drv.In(oldoutput), drv.InOut(npix), grid=(blocks,1), block=(block_size,1,1))
        return (inputmask, npix[0])
    else:
        kernel = fatboy_mod.get_function("lacosSelect_float")
        kernel(drv.InOut(sel), drv.In(sigmap), float32(lower1), float32(sigclip), float32(lower2), grid=(blocks,1), block=(block_size,1,1))
#end lacosSelect

def lacosStarReject(med3, med7, noise, firstsel, sigmap, objlim):
    #Returns firstsel
    med3 = med3.astype(float32)
    med7 = med7.astype(float32)
    noise = noise.astype(float32)
    blocks = med3.size//512
    if (med3.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("lacosStarReject_float")
    kernel(drv.In(med3), drv.In(med7), drv.In(noise), drv.InOut(firstsel), drv.In(sigmap), float32(objlim), grid=(blocks,1), block=(block_size,1,1))
    return firstsel
#end lacosStarReject

def lacosUpdateOutput(oldoutput, tempOmask, med5, skymod):
    oldoutput = oldoutput.astype(float32)
    tempOmask = tempOmask.astype(float32)
    med5 = med5.astype(float32)
    skymod = skymod.astype(float32)
    blocks = oldoutput.size//512
    if (oldoutput.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("lacosUpdateOutput_float")
    kernel(drv.InOut(oldoutput), drv.In(tempOmask), drv.In(med5), drv.In(skymod), grid=(blocks,1), block=(block_size,1,1))
#end lacosUpdateOutput

############# end LA Cosmic helper routines #################


#residuals to linear fit for use with leastsq
def linResiduals(p, x, out):
    f = (p[0]+p[1]*x).astype(float64)
    x = x.astype(float64)
    err = out-f
    return err
#end linResiduals

#GPU-ized linear interpolation across a data value (0's for all practical purposes)
def linterp_gpu(data, x, gpm, iter=100, log=None):
    t = time.time()
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    linInterp = fatboy_mod.get_function("linInterp_float")
    output = empty(data.shape, float32)
    rows = data.shape[0]
    cols = data.shape[1]
    gpm = gpm.astype(int32)
    ict = zeros(1, int32)
    nfound = zeros(1, int32)
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    p = 0

    #set log type
    logtype = LOGTYPE_NONE
    if (log is not None):
        if (isinstance(log, str)):
            #log given as a string
            log = open(log,'a')
            logtype = LOGTYPE_ASCII
        elif(isinstance(log, fatboyLog)):
            logtype = LOGTYPE_FATBOY

    print("Interpolating across "+str(x)+"'s")
    write_fatboy_log(log, logtype, "Interpolating across " + str(x) + "'s", __name__)

    #iterate
    while (p < iter):
        p += 1
        print("\tPass "+str(p))
        linInterp(drv.In(data.astype(float32)), drv.Out(output), drv.In(gpm), float32(x), int32(rows), int32(cols), drv.InOut(ict), drv.InOut(nfound), grid=(blocks,1), block=(block_size,1,1))
        print("\t\t"+str(nfound) + " found; "+str(ict)+" replaced.")
        write_fatboy_log(log, logtype, "Pass "+str(p)+": "+str(nfound) + " found; "+str(ict)+" replaced.", __name__, printCaller=False, tabLevel=1)
        if (ict == 0 or ict == nfound):
            break
        data = output
        ict = zeros(1, int32)
        nfound = zeros(1, int32)
    print("Interpolation time: "+str(time.time()-t))
    return output
#end linterp_gpu

#CPU linear interpolation across a data value (0's for all practical purposes)
def linterp_cpu(data, x, gpm, iter=100, log=None):
    nx = shape(data)[0]
    ny = shape(data)[1]
    z = -1
    initys = arange(nx*ny).reshape(nx,ny) % ny
    initxs = arange(nx*ny).reshape(nx,ny) // ny
    p = 0

    #set log type
    logtype = LOGTYPE_NONE
    if (log is not None):
        if (isinstance(log, str)):
            #log given as a string
            log = open(log,'a')
            logtype = LOGTYPE_ASCII
        elif(isinstance(log, fatboyLog)):
            logtype = LOGTYPE_FATBOY

    print("Interpolating across "+str(x)+"'s")
    write_fatboy_log(log, logtype, "Interpolating across " + str(x) + "'s", __name__)

    #iterate
    while (z != 0 and p < iter):
        p+=1
        print("\tPass "+str(p))
        b = (data == x)*gpm.astype('bool')
        ys = initys[b]
        xs = initxs[b]
        newData = data.copy()
        if (len(xs) == 0):
            z = 0
            break
        ict = 0
        for i in range(xs.size):
            j = xs[i]
            l = ys[i]
            temp = []
            for k in range(-2,3,1):
                for r in range(-2,3,1):
                    if (k == 0 and r == 0):
                        continue
                    xc = k+j
                    yc = r+l
                    if (xc >= 0 and xc < nx and yc >= 0 and yc < nx):
                        if (data[xc,yc] != 0):
                            temp.append(data[xc,yc])
            if (len(temp) != 0):
                temp = array(temp)
                newData[j,l] = gpu_arraymedian(temp, kernel=fatboyclib.median)
                ict+=1
        data = newData
        print("\t\t"+str(len(xs))+" found; "+str(ict)+" replaced.")
        write_fatboy_log(log, logtype, "Pass "+str(p)+": "+str(len(xs))+" found; "+str(ict)+" replaced.", __name__, printCaller=False, tabLevel=1)

        if (ict == 0):
            break
    return newData
#end linterp_cpu

def maskNegatives(data):
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    maskNegativesFunc = fatboy_mod.get_function("maskNegatives_float")
    outtype = float32
    if (data.dtype == int32):
        maskNegativesFunc = fatboy_mod.get_function("maskNegatives_int")
    elif (data.dtype == int64):
        maskNegativesFunc = fatboy_mod.get_function("maskNegatives_long")
    elif (data.dtype == float64):
        maskNegativesFunc = fatboy_mod.get_function("maskNegatives_double")
        outtype = float64
    blocks = data.size//block_size
    if (data.size % 512 != 0):
        blocks += 1
    output = empty(data.shape, data.dtype)
    maskNegativesFunc(drv.In(data), drv.Out(output), grid=(blocks,1), block=(block_size,1,1))
    return output
#end maskNegatives

def maskNegativesAndZeros(data, zeroRep, negRep):
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    maskNegativesFunc = fatboy_mod.get_function("maskNegativesAndZeros_float")
    data = float32(data)
    blocks = data.size//block_size
    if (data.size % 512 != 0):
        blocks += 1
    output = empty(data.shape, data.dtype)
    maskNegativesFunc(drv.In(data), drv.Out(output), float32(zeroRep), float32(negRep), grid=(blocks,1), block=(block_size,1,1))
    return output
#end maskNegativesAndZeros

def maskNegativesCPU(data):
    #make a copy so as not to destroy original data
    output = data.copy()
    output[output < 0] = 0
    return output
#end maskNegativesCPU

def maskNegativesAndZerosCPU(data, zeroRep, negRep):
    #make a copy so as not to destroy original data
    output = data.copy()
    output[data < 0] = negRep
    output[data == 0] = zeroRep
    return output
#end maskNegativesAndZerosCPU

#Perform 2-d median filter
def medfilt2d(data, width, outfile=None, zlo=0, zhi=0, mef=0, log=None):
    t = time.time()
    if (isinstance(data, str)):
        if (os.access(data, os.F_OK)):
            outimage = pyfits.open(data)
            data = outimage[mef].data
        else:
            print("medfilt2d> Error: File "+input+" does not exist!")
            if (log is not None):
                log.writeLog(__name__, "File "+data+" does not exist!", type=fatboyLog.ERROR)
            return None
    elif (isinstance(data, ndarray)):
        if (outfile is not None):
            outimage = pyfits.HDUList()
            hdu = pyfits.PrimaryHDU()
            outimage.append(hdu)
    else:
        print("medfilt2d> Error: Input must be a FITS file or a raw array.")
        if (log is not None):
            log.writeLog(__name__, "Input must be a FITS file or a raw array.", type=fatboyLog.ERROR)
        return None

    rows = data.shape[0]
    cols = data.shape[1]
    w = width//2
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    medfiltFunc = fatboy_mod.get_function("medfilt2d_float")
    outtype = float32
    out = empty(data.shape, outtype)
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    medfiltFunc(drv.In(data), drv.Out(out), int32(rows), int32(cols), int32(w), int32(zlo), int32(zhi), grid=(blocks,1), block=(block_size,1,1))

    if (outfile is not None):
        outimage[mef].data = out
        outimage.verify('silentfix')
        outimage.writeto(outfile)
        outimage.close()
    return out
#end medfilt2d

def medianfilterCPU(cut, boxsize=25, nhigh=0):
    tempcut = zeros(len(cut))
    tempcut[:boxsize] = cut[:boxsize] - arraymedian(cut[:2*boxsize], kernel=fatboyclib.median, nhigh=nhigh)
    for j in range(boxsize, len(cut)-boxsize):
        tempcut[j] = cut[j] - arraymedian(cut[j-boxsize:j+boxsize+1], kernel=fatboyclib.median, nhigh=nhigh)
    tempcut[-boxsize:] = cut[-boxsize:] - arraymedian(cut[-2*boxsize:], kernel=fatboyclib.median, nhigh=nhigh)
    return tempcut
#end medianfilterCPU

def medianfilter2dCPU(data, axis="X", boxsize=51, nhigh=0):
    if (boxsize > 51):
        print("Using maximum boxsize of 51!")
        boxsize = 51
    elif (boxsize % 2 == 0):
        boxsize += 1
        print("Boxsize must be odd!  Using "+str(boxsize))
    temp = zeros(data.shape)
    bshalf = boxsize//2
    bshalfplus = bshalf+1
    if (axis == "Y"):
        for j in range(bshalf):
            temp[j,:] = data[j,:] - arraymedian(data[:boxsize,:].copy(), axis="Y", kernel=fatboyclib.median, kernel2d=fatboyclib.median2d, nhigh=nhigh)
        for j in range(bshalf, data.shape[0]-bshalfplus):
            temp[j,:] = data[j,:] - arraymedian(data[j-bshalf:j+bshalfplus,:].copy(), axis="Y", kernel=fatboyclib.median, kernel2d=fatboyclib.median2d, nhigh=nhigh)
        for j in range(data.shape[0]-bshalfplus, data.shape[0]):
            temp[j,:] = data[j,:] - arraymedian(data[-boxsize:,:].copy(), axis="Y", kernel=fatboyclib.median, kernel2d=fatboyclib.median2d, nhigh=nhigh)
        return temp
    #Default = X-axis
    for j in range(bshalf):
        temp[:,j] = data[:,j] - arraymedian(data[:,:boxsize].copy(), axis="X", kernel=fatboyclib.median, kernel2d=fatboyclib.median2d, nhigh=nhigh)
    for j in range(bshalf, data.shape[1]-bshalfplus):
        temp[:,j] = data[:,j] - arraymedian(data[:,j-bshalf:j+bshalfplus].copy(), axis="X", kernel=fatboyclib.median, kernel2d=fatboyclib.median2d, nhigh=nhigh)
    for j in range(data.shape[1]-bshalfplus, data.shape[1]):
        temp[:,j] = data[:,j] - arraymedian(data[:,-boxsize:].copy(), axis="X", kernel=fatboyclib.median, kernel2d=fatboyclib.median2d, nhigh=nhigh)
    return temp
#end medianfilter2dCPU

def mediansmooth1d(data, width):
    n = len(data)
    if (width % 2 == 0):
        width += 1
    w = width//2
    newdata = zeros((width, n), float32)
    newdata [w,:] = data[:]
    for j in range(w):
        newdata[j,w-j:] = data[:j-w]
        newdata[2*w-j,:j-w] = data[w-j:]
    out = gpu_arraymedian(newdata, axis="Y", nonzero=True, kernel2d=fatboyclib.median2d)
    return out
#end mediansmooth1d

#median smooth 2d data along 1 axis
def mediansmooth1d2d(data, width, axis=0):
    n = len(data)
    rows = data.shape[0]
    cols = data.shape[1]
    if (width % 2 == 0):
        width+=1
    w = width//2
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    smoothFunc = fatboy_mod.get_function("mediansmooth1d2d_float")
    outtype = float32
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    smoothed = empty(data.shape, outtype)
    smoothFunc(drv.In(data), drv.Out(smoothed), int32(rows), int32(cols), int32(w), int32(axis), grid=(blocks,1), block=(block_size,1,1))
    return smoothed
#end mediansmooth1d2d

#replace masked with median of neighboring pixels
def medReplace2d(data, mask, boxsize):
    out = data.copy()
    z = int(boxsize/2) #boxsize typically 3 = 8 neighboring pixels
    b = where(mask[z:-z,z:-z]) #mask should be true if pixel is zero (or otherwise should be replaced)
    repVals = zeros((b[0].size, boxsize**2-1))
    i = 0
    for j in range(boxsize):
        for l in range(boxsize):
            if (j == z and l == z):
                continue
            repVals[:,i] = data[b[0]+j, b[1]+l]
            i+=1
    out[b[0]+z, b[1]+z] = gpu_arraymedian(repVals, axis="X", nonzero=True)
    return out
#end medReplace2d

#Used for point_replace.  Calculate scaling of neighboring pixels in point vs turbo images
def medScale2d(data1, data2, mask, boxsize):
    out = zeros(data1.shape)
    z = int(boxsize/2) #boxsize typically 3 = 8 neighboring pixels
    b = where(mask[z:-z,z:-z]) #mask should be true if pixel is zero in point image and nonzero in turbo
    scaleVals1 = zeros((b[0].size, boxsize**2-1))
    scaleVals2 = zeros((b[0].size, boxsize**2-1))
    i = 0
    for j in range(boxsize):
        for l in range(boxsize):
            if (j == z and l == z):
                continue
            scaleVals1[:,i] = data1[b[0]+j, b[1]+l]
            scaleVals2[:,i] = data2[b[0]+j, b[1]+l]*(scaleVals1[:,i] != 0)
            i+=1
    #output is median of neighboring pixels in point image to that in turbo image for scaling replaced pixels
    out[b[0]+z, b[1]+z] = gpu_arraymedian(scaleVals1, axis="X", nonzero=True)/gpu_arraymedian(scaleVals2, axis="X", nonzero=True)
    #Handle NaNs where there is no neighboring data
    out[isnan(out)] = 0
    return out
#end medScale2d

def noisemaps_ds_gpu(nm, nmdark):
    blocks = nm.size//512
    if (nm.size % 512 != 0):
        blocks += 1
    if (nm.dtype != 'float32'):
        nm = nm.astype(float32)
    if (nmdark.dtype != 'float32'):
        nmdark = nmdark.astype(float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("noisemaps_ds_float")
    kernel(drv.InOut(nm), drv.In(nmdark), int32(nm.size), grid=(blocks,1), block=(block_size,1,1))
    return nm
#end noisemaps_ds_gpu

def noisemaps_fd_gpu(image, nm, oldimage, nmflat, masterFlat):
    blocks = nm.size//512
    if (nm.size % 512 != 0):
        blocks += 1
    if (nm.dtype != 'float32'):
        nm = nm.astype(float32)
    if (image.dtype != 'float32'):
        image = image.astype(float32)
    if (nmflat.dtype != 'float32'):
        nmflat = nmflat.astype(float32)
    if (oldimage.dtype != 'float32'):
        oldimage = oldimage.astype(float32)
    if (masterFlat.dtype != 'float32'):
        masterFlat = masterFlat.astype(float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("noisemaps_fd_float")
    kernel(drv.In(image), drv.InOut(nm), drv.In(oldimage), drv.In(nmflat), drv.In(masterFlat), int32(nm.size), grid=(blocks,1), block=(block_size,1,1))
    return nm
#end noisemaps_fd_gpu

def noisemaps_mflat_dome_on_off_gpu(on, off, ncomb1, ncomb2):
    nm = empty(on.shape, float32)
    blocks = nm.size//512
    if (nm.size % 512 != 0):
        blocks += 1
    if (nm.dtype != 'float32'):
        nm = nm.astype(float32)
    if (on.dtype != 'float32'):
        on = on.astype(float32)
    if (off.dtype != 'float32'):
        off = off.astype(float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("noisemaps_mflat_dome_on_off_float")
    kernel(drv.Out(nm), drv.In(on), drv.In(off), float32(ncomb1), float32(ncomb2), int32(nm.size), grid=(blocks,1), block=(block_size,1,1))
    return nm
#end noisemaps_mflat_dome_on_off_gpu

#Take the sqrt(dividend) / divisor
def noisemaps_sqrtAndDivide_float(dividend, divisor):
    blocks = dividend.size//512
    if (dividend.size % 512 != 0):
        blocks += 1
    if (dividend.dtype != 'float32'):
        dividend = dividend.astype(float32)
    if (divisor.dtype != 'float32'):
        divisor = divisor.astype(float32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("noisemaps_sqrtAndDivide_float")
    kernel(drv.InOut(dividend), drv.In(divisor), int32(dividend.size), grid=(blocks,1), block=(block_size,1,1))
    return dividend
#end noisemaps_sqrtAndDivide_float

#Normalize a longslit flat
def normalizeFlat(masterFlat, medVal, lowThresh, lowReplace, hiThresh, hiReplace, log=None):
    flat = masterFlat.getData()
    doNM = False
    if (masterFlat.hasProperty("noisemap")):
        doNM = True
        nm = masterFlat.getProperty("noisemap")
    else:
        nm = empty(1, dtype=float32)
    blocks = (flat.size)//512
    if (flat.size % 512 != 0):
        blocks += 1
    lowct = zeros(1, int32)
    hict = zeros(1, int32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("normalizeFlat_float")
    kernel(drv.InOut(flat), float32(medVal), float32(lowThresh), float32(lowReplace), float32(hiThresh), float32(hiReplace), drv.InOut(lowct), drv.InOut(hict), int32(doNM), drv.InOut(nm), int32(flat.size), grid=(blocks,1), block=(block_size,1,1))
    if (lowct > 0):
        print("normalizeFlat> Replaced "+str(lowct)+" pixels below "+str(lowThresh))
        if (log is not None):
            log.writeLog(__name__, "Replaced "+str(lowct)+" pixels below "+str(lowThresh))
    if (hict > 0):
        print("normalizeFlat> Replaced "+str(hict)+" pixels above "+str(hiThresh))
        if (log is not None):
            log.writeLog(__name__, "Replaced "+str(hict)+" pixels above "+str(lowThresh))
    if (doNM):
        #Update noisemap
        masterFlat.tagDataAs("noisemap", nm)
    masterFlat.updateData(flat)
#end normalizeFlat

#Normalize a MOS flat to have each slitlet's median value be 1
def normalizeMOSFlat(masterFlat, slitmask, nslits, lowThresh=0, lowReplace=0, hiThresh=0, hiReplace=0, log=None):
    flat = masterFlat.getData()
    doNM = False
    if (masterFlat.hasProperty("noisemap")):
        doNM = True
        nm = masterFlat.getProperty("noisemap")
    else:
        nm = empty(1, dtype=float32)
    blocks = (flat.size)//512
    if (flat.size % 512 != 0):
        blocks += 1
    lowct = zeros(1, int32)
    hict = zeros(1, int32)
    trans = False
    if (masterFlat.dispersion == masterFlat.DISPERSION_VERTICAL):
        trans = True #transpose data first
    #First find medians of each slitlet
    medians = gpumedianS(flat, slitmask, nslits, nonzero=True, trans=trans)
    for j in range(nslits):
        key = ''
        if (j+1 < 10):
            key += '0'
        key += str(j+1)
        if (medians[j] == 1 and 'NORMAL'+key in masterFlat._header):
            continue
        updateHeaderEntry(masterFlat._header, 'NORMAL'+key, medians[j])
        masterFlat.setHistory('renormalize_'+key, medians[j])
        #masterFlat._header.update(key, medians[j])
    #Normalize and replace
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("normalizeMOSFlat_float")
    kernel(drv.InOut(flat), drv.In(int32(slitmask)), drv.In(medians), float32(lowThresh), float32(lowReplace), float32(hiThresh), float32(hiReplace), drv.InOut(lowct), drv.InOut(hict), int32(doNM), drv.InOut(nm), int32(flat.size), grid=(blocks,1), block=(block_size,1,1))
    if (lowct > 0):
        print("normalizeMOSFlat> Replaced "+str(lowct)+" pixels below "+str(lowThresh))
        if (log is not None):
            log.writeLog(__name__, "Replaced "+str(lowct)+" pixels below "+str(lowThresh))
    if (hict > 0):
        print("normalizeMOSFlat> Replaced "+str(hict)+" pixels above "+str(hiThresh))
        if (log is not None):
            log.writeLog(__name__, "Replaced "+str(hict)+" pixels above "+str(lowThresh))
    if (doNM):
        #Update noisemap
        masterFlat.tagDataAs("noisemap", nm)
    masterFlat.updateData(flat)
#end normalizeMOSFlat

#Normalize a MOS source frame to have each slitlet's median value be 1
def normalizeMOSSource(sourceFDU, slitmask, nslits, log=None):
    data = sourceFDU.getData()
    blocks = (data.size)//512
    if (data.size % 512 != 0):
        blocks += 1
    trans = False
    if (sourceFDU.dispersion == sourceFDU.DISPERSION_VERTICAL):
        trans = True #transpose data first
    #First find medians of each slitlet
    medians = gpumedianS(data, slitmask, nslits, nonzero=True, trans=trans)
    #Normalize
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("normalizeMOSSource_float")
    kernel(drv.InOut(data), drv.In(int32(slitmask)), drv.In(medians), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
    sourceFDU.updateData(data)
#end normalizeMOSSource

def polyFunction(p, x, order):
    if (isinstance(x, float) or isinstance(x, int)):
        f = 0.
    else:
        f = zeros(x.shape, float64)
        x = x.astype(float64)
    for j in range(0,order+1):
        f+=p[j]*x**j
    return f
#end polyFunction

def polyResiduals(p, x, out, order):
    f = zeros(x.shape, float64)
    x = x.astype(float64)
    for j in range(0,order+1):
        f+=p[j]*x**j
    err = out-f
    return err
#end polyResiduals

def prepMefForWriting(hdu, mef):
    #Get rid of extraneous extensions in data like CIRCE/Newfirm
    while (len(hdu) > mef+1):
        hdu.pop(mef+1)
    if (mef > 0):
        hdu[0].data = None
        updateHeaderEntry(hdu[0].header, 'NAXIS', 0)
        if ('NAXIS1' in hdu[0].header):
            del hdu[0].header['NAXIS1']
        if ('NAXIS2' in hdu[0].header):
            del hdu[0].header['NAXIS2']
        if ('NAXIS3' in hdu[0].header):
            del hdu[0].header['NAXIS3']
    if 'BZERO' in hdu[mef].header:
        if (hdu[mef].data.dtype != 'uint16'):
            hdu[mef].header.pop('BZERO')
    hdu.verify('silentfix')
#end prepMefForWriting

#read a rectification/geometric distortion correction file
#return [xcoeffs, ycoeffs], where xcoeffs and ycoeffs are lists
def readCoeffsFile(coeffFile, log=None):
    xcoeffs = []
    ycoeffs = []
    if (isinstance(coeffFile, str) and os.access(coeffFile, os.F_OK)):
        #Leave empty lines in list
        f = open(coeffFile, 'r')
        lines = f.read().split('\n')
        f.close()
        try:
            lines.pop(0) #remove header line
            currCoeff = lines.pop(0)
            #Read xcoeffs until blank line
            while (currCoeff != ''):
                xcoeffs.append(float(currCoeff))
                currCoeff = lines.pop(0)
            #Read ycoeffs until lines is empty
            while (len(lines) > 0):
                currCoeff = lines.pop(0)
                if (currCoeff != ''):
                    ycoeffs.append(float(currCoeff))
            return [xcoeffs, ycoeffs]
        except Exception as ex:
            print("fatboyLibs::readCoeffsFile> Error: misformatted file "+coeffFile+"! "+str(ex))
            if (log is not None):
                log.writeLog(__name__, "misformatted file "+coeffFile+"! "+str(ex), type=fatboyLog.ERROR)
            return None
    print("fatboyLibs::readCoeffsFile> Error: file not found "+str(coeffFile))
    if (log is not None):
        log.writeLog(__name__, "file not found "+str(coeffFile), type=fatboyLog.ERROR)
    return None
#end readCoeffsFile

#read an ascii file into a list
def readFileIntoList(fname):
    f = open(fname, 'r')
    x = f.read().split('\n')
    f.close()
    removeEmpty(x)
    return x
#end readFileIntoList

#read a ds9 style region file
#Note that ds9 counts starting from 1 instead of 0
def readRegionFile(regfile, horizontal=True, log=None):
    #horizontal gives dispersion direction, true = horizontal, false = vertical
    sylo = []
    syhi = []
    slitx = []
    slitw = []
    lines = readFileIntoList(regfile)
    for line in lines:
        if (line.startswith("image;box") or line.startswith("box(")):
            temp=line[line.find('(')+1:]
            temp = temp.split(',')
            if (temp[-1].find(')') != -1):
                temp[-1] = temp[-1][:temp[-1].find(')')]
            try:
                for j in range(4):
                    temp[j] = float(temp[j])
            except Exception as ex:
                print("fatboyLibs::readRegionFileText> Warning: misformatted line in "+regfile+": "+line)
                if (log is not None):
                    log.writeLog(__name__, "misformatted line in "+regfile+": "+line, type=fatboyLog.WARNING)
            if (horizontal):
                slitx.append(float(temp[0])-1)
                sylo.append(int(float(temp[1])-float(temp[3])/2.-1))
                syhi.append(int(float(temp[1])+float(temp[3])/2.-1))
                slitw.append(float(temp[2]))
            else:
                slitx.append(float(temp[1])-1)
                sylo.append(int(float(temp[0])-float(temp[2])/2.-1))
                syhi.append(int(float(temp[0])+float(temp[2])/2.-1))
                slitw.append(float(temp[3]))
    #Deal with non-sequential region file entries by sorting
    sorted = array(sylo).argsort()
    sylo = array(sylo)[sorted]
    syhi = array(syhi)[sorted]
    slitx = array(slitx)[sorted]
    slitw = array(slitw)[sorted]
    return (sylo, syhi, slitx, slitw)
#end readRegionFile

#read a text-sytle region file
#text files are zero-indexed
def readRegionFileText(regfile, horizontal=True, log=None):
    #horizontal gives dispersion direction, true = horizontal, false = vertical
    sylo = []
    syhi = []
    slitx = []
    slitw = []
    lines = readFileIntoList(regfile)
    for line in lines:
        #Ignore comments
        if (not line.startswith("#")):
            temp = line.split()
            try:
                for j in range(4):
                    temp[j] = float(temp[j])
            except Exception as ex:
                print("fatboyLibs::readRegionFileText> Warning: misformatted line in "+regfile+": "+line)
                if (log is not None):
                    log.writeLog(__name__, "misformatted line in "+regfile+": "+line, type=fatboyLog.WARNING)
            if (horizontal):
                slitx.append(float(temp[0]))
                sylo.append(int(float(temp[1])-float(temp[3])/2.))
                syhi.append(int(float(temp[1])+float(temp[3])/2.))
                slitw.append(float(temp[2]))
            else:
                slitx.append(float(temp[1]))
                sylo.append(int(float(temp[0])-float(temp[2])/2.))
                syhi.append(int(float(temp[0])+float(temp[2])/2.))
                slitw.append(float(temp[3]))
    #Deal with non-sequential region file entries by sorting
    sorted = array(sylo).argsort()
    sylo = array(sylo)[sorted]
    syhi = array(syhi)[sorted]
    slitx = array(slitx)[sorted]
    slitw = array(slitw)[sorted]
    return (sylo, syhi, slitx, slitw)
#end readRegionFileText

#read an XML style region file
#XML files are zero-indexed
def readRegionFileXML(regfile, horizontal=True, log=None):
    #horizontal gives dispersion direction, true = horizontal, false = vertical
    sylo = []
    syhi = []
    slitx = []
    slitw = []
    #doc = xml config file
    try:
        doc = xml.dom.minidom.parse(regfile)
    except Exception as ex:
        print("fatboyLibs::readRegionFileXML> Error parsing XML region file "+regfile+": "+str(ex))
        if (log is not None):
            log.writeLog(__name__, "Error parsing XML region file "+regfile+": "+str(ex), type=fatboyLog.ERROR)
        return (sylo, syhi, slitx, slitw)

    #get all slitlet nodes
    slitletNodes = doc.getElementsByTagName('slitlet')
    #loop over query nodes
    for node in slitletNodes:
        if (not node.hasAttribute("xcenter") or not node.hasAttribute("ycenter") or not node.hasAttribute("width") or not node.hasAttribute("height")):
            print("fatboyLibs::readRegionFileXML> Warning: misformatted line in "+regfile+": missing required attributed!")
            if (log is not None):
                log.writeLog(__name__, "misformatted line in "+regfile+": missing required attributes", type=fatboyLog.WARNING)
            continue
        try:
            xcenter = float(node.getAttribute("xcenter"))
            ycenter = float(node.getAttribute("ycenter"))
            width = float(node.getAttribute("width"))
            height = float(node.getAttribute("height"))
        except Exception as ex:
            print("fatboyLibs::readRegionFileXML> Warning: misformatted line in "+regfile+": error parsing attributes!")
            if (log is not None):
                log.writeLog(__name__, "misformatted line in "+regfile+": error parsing attributes!", type=fatboyLog.WARNING)
            continue
        if (horizontal):
            slitx.append(xcenter)
            sylo.append(int(ycenter-height/2.))
            syhi.append(int(ycenter+height/2.))
            slitw.append(width)
        else:
            slitx.append(ycenter)
            sylo.append(int(xcenter-width/2.))
            syhi.append(int(xcenter+width/2.))
            slitw.append(height)
    #Deal with non-sequential region file entries by sorting
    sorted = array(sylo).argsort()
    sylo = array(sylo)[sorted]
    syhi = array(syhi)[sorted]
    slitx = array(slitx)[sorted]
    slitw = array(slitw)[sorted]
    return (sylo, syhi, slitx, slitw)
#end readRegionFileXML

def removeEmpty(s):
    while(s.count('') > 0):
        s.remove('')
#end removeEmpty

#Find the low and high row/column values of each slit
def shiftAddSlitmask(slitmask, nslits, horizontal=True):
    slitmask = int32(slitmask)
    rows = slitmask.shape[0]
    cols = slitmask.shape[1]
    blocks = slitmask.size//512
    ylo = ones(nslits, int32)*rows
    yhi = zeros(nslits, int32)
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    kernel = fatboy_mod.get_function("shiftAddSlitmask")
    kernel(drv.In(slitmask), drv.InOut(ylo), drv.InOut(yhi), int32(cols), int32(nslits), int32(slitmask.size), int32(horizontal), grid=(blocks,1), block=(block_size,1,1))
    return (ylo, yhi)
#end shiftAddSlitmask

#Return a mean, median, and std using a sigma clipping algorithm
def sigmaFromClipping(origData, sig, iter):
    n = 0
    oldstddev = 0
    data = origData.copy()
    while (n < iter and len(data) > 0):
        mean = data.mean()
        stddev = data.std()
        if (stddev == oldstddev):
            break
        lo = mean-sig*stddev
        hi = mean+sig*stddev
        oldstddev = stddev
        data = data[logical_and(data > lo, data < hi)]
        n+=1
    med = gpu_arraymedian(data)
    return [mean, med, stddev]
#end sigmaFromClipping

#CPU verision of 1d mean smoothing algorithm
def smooth1dCPU(data, width, niter):
    if (niter == 0):
        return data
    n = len(data)
    if (width % 2 == 0):
        width+=1
    w = width//2
    divisor = ones(data.shape, dtype=float32)*width
    divisor[:w] -= (w-arange(w))
    divisor[-w:] -= (1+arange(w))
    for k in range(niter):
        newdata = data.copy()
        for j in range(1,w+1):
            newdata[j:] += data[:-j]
            newdata[:-j] += data[j:]
        newdata/=divisor
        data = newdata
    return data
#end smooth1dCPU

#GPU version of 1d mean smoothing algorithm
def smooth1d(data, width, niter):
    if (niter == 0):
        return data
    n = len(data)
    if (n <= 16384):
        return smooth1dCPU(data, width, niter)
    if (width % 2 == 0):
        width+=1
    w = width//2
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    smooth1dFunc = fatboy_mod.get_function("smooth1d_float")
    outtype = float32
    if (data.dtype == int32):
        smooth1dFunc = fatboy_mod.get_function("smooth1d_int")
    elif (data.dtype == int64):
        smooth1dFunc = fatboy_mod.get_function("smooth1d_long")
    elif (data.dtype == float64):
        smooth1dFunc = fatboy_mod.get_function("smooth1d_double")
        outtype = float64
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    for k in range(niter):
        smoothed = empty(data.shape, outtype)
        smooth1dFunc(drv.In(data), drv.Out(smoothed), int32(n), int32(w), grid=(blocks,1), block=(block_size,1,1))
        data = smoothed
    return data
#end smooth1d

#Smooth 2-d data along 1 axis
def smooth1d2d(data, width, niter, axis=0):
    if (niter == 0):
        return data
    n = len(data)
    rows = data.shape[0]
    cols = data.shape[1]
    if (width % 2 == 0):
        width+=1
    w = width//2
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    smoothFunc = fatboy_mod.get_function("smooth1d2d_float")
    outtype = float32
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    for k in range(niter):
        smoothed = empty(data.shape, outtype)
        smoothFunc(drv.In(data), drv.Out(smoothed), int32(rows), int32(cols), int32(w), int32(axis), grid=(blocks,1), block=(block_size,1,1))
        data = smoothed
    return data
#end smooth1d2d

#CPU version of 2d smoothing algorithm
def smooth_cpu(data, width, niter, type = float32):
    if (niter == 0):
        return data
    m = data.shape[1]
    n = data.shape[0]
    if (width % 2 == 0): width+=1
    w = width//2
    tot = zeros((n+2*w,m+2*w), type)
    sdata = zeros(data.shape, type)
    a = zeros(data.shape, float32)
    b = zeros(data.shape, float32)
    c = zeros(data.shape, float32)
    d = zeros(data.shape, float32)
    for i in range(niter):
        for l in range(m):
            if (l > 0):
                tot[w:-w,l+w] = tot[w:-w,l+w-1] + data[:,l]
            else:
                tot[w:-w,l+w] = data[:,l]
        for l in range(w):
            tot[:,l] = tot[:,w+1]
            tot[:,m+w+l] = tot[:,m+w-1]
        for j in range(n-2, -1, -1):
            tot[j+w,:]+=tot[j+w+1,:]
        for j in range(w):
            tot[n+w+j,:] = tot[n+w-1,:]
            tot[j,:] = tot[w+1,:]
        a[:,:] = tot[0:n,2*w:]
        b[0:-w-1,:] = tot[2*w+1:n+w,2*w:]
        c[:,w+1:] = tot[0:n,w:m-1]
        d[0:-w-1,w+1:] = tot[2*w+1:n+w,w:m-1]
        j = arange(n*m).reshape(n,m) % m
        l = arange(n*m).reshape(n,m) // m
        jw1 = j+w+1
        jw = j-w
        lw = l+w
        lw1 = l-(w+1)
        jw1[:,-w:] = m
        jw[:,:w] = 0
        lw[-w:,:] = n-1
        lw1[:w,:] = -1
        pts = (jw1-jw)*(lw-lw1)
        sdata[:,:] = (a-b-c+d)/pts
        data = sdata
    return sdata
#end smooth_cpu

#GPU verion of 2d smoothing algorithm
def smooth2d(data, width, niter):
    if (niter == 0):
        return data
    rows = data.shape[0]
    cols = data.shape[1]
    if (width % 2 == 0):
        width+=1
    w = width//2
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    smooth2dFunc = fatboy_mod.get_function("smooth2d_float")
    outtype = float32
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    for k in range(niter):
        smoothed = empty(data.shape, outtype)
        smooth2dFunc(drv.In(data), drv.Out(smoothed), int32(rows), int32(cols), int32(w), grid=(blocks,1), block=(block_size,1,1))
        data = smoothed
    return data
#end smooth2d

def subtractImages(image1, image2, gpm=None, scale=None):
    blocks = image1.size//512
    if (image1.size % 512 != 0):
        blocks += 1
    t = time.time()
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    if (scale is not None):
        subArrays = fatboy_mod.get_function("subtractArrays_scaled_float")
        subArrays(drv.InOut(image1), drv.In(image2), float32(scale), grid=(blocks,1), block=(block_size,1,1))
    elif (gpm is not None):
        subArrays = fatboy_mod.get_function("subtractArrays_gpm_float")
        subArrays(drv.InOut(image1), drv.In(image2), drv.In(gpm), grid=(blocks,1), block=(block_size,1,1))
    else:
        subArrays = fatboy_mod.get_function("subtractArrays_float")
        subArrays(drv.InOut(image1), drv.In(image2), grid=(blocks,1), block=(block_size,1,1))
    print("Subtraction time: ",time.time()-t)
    return image1
#end subtractImages

def surface3dFunction(coeffs, x, y, z, order):
    out = zeros(x.shape, float64)
    x = x.astype(float64)
    y = y.astype(float64)
    z = z.astype(float64)
    i = 0
    for j in range(order+1):
        for l in range(1,j+2):
            for k in range(1,l+1):
                out+=coeffs[i]*x**(j-l+1)*y**(l-k)*z**(k-1)
                i+=1
    return out
#end surface3dFunction

def surface3dResiduals(p, x, y, z, out, order):
    f = zeros(x.shape, float64)
    x = x.astype(float64)
    y = y.astype(float64)
    z = z.astype(float64)
    n = 1
    for j in range(1,order+1):
        for l in range(1,j+2):
            for k in range(1,l+1):
                f+=p[n]*x**(j-l+1)*y**(l-k)*z**(k-1)
                n+=1
    err = out - f
    return err
#end surface3dResiduals

def surfaceFunction(coeffs, x, y, order):
    out = zeros(x.shape, float64)
    if (len(out.shape) == 1 or out.shape[0] == out.size):
        out = zeros(y.shape, float64)
    x = x.astype(float64)
    y = y.astype(float64)
    i = 0
    for j in range(order+1):
        for l in range(j+1):
            out+=coeffs[i]*x**(j-l)*y**l
            i+=1
    return out
#end surfaceFunction

def surfaceResiduals(p, x, y, out, order):
    f = zeros(x.shape, float64)
    if (len(f.shape) == 1 or f.shape[0] == f.size):
        f = zeros(y.shape, float64)
    x = x.astype(float64)
    y = y.astype(float64)
    n = 1
    for j in range(1,order+1):
        for l in range(j+1):
            f+=p[n]*x**(j-l)*y**l
            n+=1
    err = out - f
    return err
#end surfaceResiduals

def surfaceResidualsWithOffset(p, x, y, out, order):
    f = zeros(x.shape, float64)
    if (len(f.shape) == 1 or f.shape[0] == f.size):
        f = zeros(y.shape, float64)
    x = x.astype(float64)
    y = y.astype(float64)
    f += p[0]
    n = 1
    for j in range(1,order+1):
        for l in range(j+1):
            f+=p[n]*x**(j-l)*y**l
            n+=1
    err = out - f
    return err
#end surfaceResidualsWithOffset

def updateHeader(header, extHeader):
    #extHeader = dict()
    for key in extHeader:
        if (extHeader[key] is None):
            extHeader[key] = 'None'
    if (useAstropy):
        #need to get obstype due to stupid astropy bug that updates to a number
        obst = None
        if ('OBSTYPE' in extHeader):
            obst = str(extHeader['OBSTYPE'])
        try:
            header.update(extHeader)
        except Exception as ex:
            print("WARNING: exception updating header: "+str(ex))
        if (obst is not None):
            #now copy back over string of obstype to headers
            if (header['OBSTYPE'] != obst):
                header['OBSTYPE'] = obst
            if (extHeader['OBSTYPE'] != obst):
                extHeader['OBSTYPE'] = obst
    elif (hasattr(pyfits, '__version__') and pyfits.__version__ >= '3.1'):
        header.update(extHeader)
    else:
        for key in sorted(extHeader):
            if (key == 'HISTORY'):
                #Handle history separately
                continue
            header.update(key, extHeader[key])
        if ((hasattr(pyfits, 'header') and isinstance(extHeader, pyfits.header.Header)) or (hasattr(pyfits, 'core') and isinstance(extHeader, pyfits.core.Header))):
            for extHist in extHeader.get_history():
                if (not isinstance(extHist, str)):
                    extHist = extHist.value
                hasThisHistory = False
                for hist in header.get_history():
                    if (not isinstance(hist, str)):
                        hist = hist.value
                    if (hist == extHist):
                        hasThisHistory = True
                        break
                if (not hasThisHistory):
                    header.add_history(extHist)
#end updateHeader

def updateHeaderEntry(header, key, value):
    if (value is None):
        value = 'None'
    if (useAstropy):
        #new style
        header[key] = value
    elif (hasattr(pyfits, '__version__') and pyfits.__version__ >= '3.1'):
        #new style
        header[key] = value
    else:
        #old style
        if (key == 'HISTORY'):
            header.add_history(value)
        else:
            header.update(key, value)
#end updateHeaderEntry

def whereEqual(data, val):
    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    whereEqualFunc = fatboy_mod.get_function("whereEqual_float")
    outtype = float32
    idx = int32([-1])
    if (data.dtype == int32):
        whereEqualFunc = fatboy_mod.get_function("whereEqual_int")
    elif (data.dtype == int64):
        whereEqualFunc = fatboy_mod.get_function("whereEqual_long")
    elif (data.dtype == float64):
        whereEqualFunc = fatboy_mod.get_function("whereEqual_double")
        outtype = float64
    blocks = data.size//block_size
    if (data.size % block_size != 0):
        blocks += 1
    whereEqualFunc(drv.In(data), outtype(val), drv.InOut(idx), grid=(blocks,1), block=(block_size,1,1))
    if (len(data.shape) == 1):
        idx = idx,
    elif (len(data.shape) == 2):
        idx = idx//data.shape[1], idx%data.shape[1]
    elif (len(data.shape) == 3):
        x = idx//(data.shape[1]*data.shape[2])
        rem = idx-x*(data.shape[1]*data.shape[2])
        idx = x, rem//data.shape[2], rem%data.shape[2]
    return idx
#end whereEqual

def write_fatboy_log(log, logtype, message, name, printCaller=True, tabLevel=0, verbosity=None, messageType=fatboyLog.INFO):
    if (logtype == LOGTYPE_NONE):
        return
    elif (logtype == LOGTYPE_ASCII):
        log.write(message+"\n")
        return
    elif (logtype == LOGTYPE_FATBOY):
        log.writeLog(name, message, printCaller=printCaller, tabLevel=tabLevel, verbosity=verbosity, callerLevel=2, type=messageType)
#end write_fatboy_log

def write_fits_file(filename, data, dtype="float32", header=None, headerExt=None, overwrite=False, fitsobj=None, mef=0, log=None):
    if (fitsobj is None):
        #hdulist is already given
        fitsobj = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        fitsobj.append(hdu)
        while (len(fitsobj) <= mef):
            #add extensions
            hdu = pyfits.ImageHDU()
            fitsobj.append(hdu)
    else:
        #Get rid of extraneous extensions in data like CIRCE/Newfirm
        prepMefForWriting(fitsobj, mef)
        hdu = fitsobj[mef]
    hdu.data = data.astype(dtype)
    if (header is not None):
        updateHeader(fitsobj[0].header, header)
    if (headerExt is not None):
        #Add keywords from extended header
        updateHeader(fitsobj[0].header, headerExt)
    try:
        fitsobj.verify('silentfix')
        if (overwrite and os.access(filename, os.F_OK)):
            os.unlink(filename)
        elif (os.access(filename, os.F_OK)):
            print("fatboyLibs::write_fits_file> ERROR: "+filename+" already exists!")
            if (log is not None):
                log.writeLog(__name__, filename+" already exists!", type=fatboyLog.ERROR)
            return False
        fitsobj.writeto(filename, output_verify='silentfix')
        fitsobj.close()
    except Exception as ex:
        print("fatboyLibs::write_fits_file> ERROR writing file "+filename+": "+str(ex))
        if (log is not None):
            log.writeLog(__name__, "ERROR writing file "+filename+": "+str(ex), type=fatboyLog.ERROR)
        return False
    return True
#end write_fits_file

#read a ds9 style region file
#Note that ds9 counts starting from 1 instead of 0
def writeRegionFile(regfile, sylo, syhi, slitx, slitw, horizontal=True):
    #Output new region file
    f = open(regfile,'w')
    f.write('# Region file format: DS9 version 3.0\n')
    f.write('global color=green select=1 edit=1 move=1 delete=1 include=1 fixed=0\n')
    if (horizontal):
        for i in range(len(sylo)):
            f.write('image;box('+str(slitx[i])+','+str((sylo[i]+syhi[i])/2)+','+str(slitw[i])+','+str(syhi[i]-sylo[i])+')\n')
    else:
        for i in range(len(sylo)):
            f.write('image;box('+str((sylo[i]+syhi[i])/2)+','+str(slitx[i])+','+str(syhi[i]-sylo[i])+','+str(slitw[i])+')\n')
    f.close()
#end writeRegionFile

#read an XML style region file
#XML files are zero-indexed
def writeRegionFileXML(regfile, sylo, syhi, slitx, slitw, horizontal=True):
    #Output new XML region file
    f = open(regfile,'w')
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<fatboy>\n')
    if (horizontal):
        for i in range(len(sylo)):
            f.write('<slitlet xcenter="'+str(slitx[i])+'" ycenter="'+str((sylo[i]+syhi[i])/2)+'" width="'+str(slitw[i])+'" height="'+str(syhi[i]-sylo[i])+'"/>\n')
    else:
        for i in range(len(sylo)):
            f.write('<slitlet xcenter="'+str((sylo[i]+syhi[i])/2)+'" ycenter="'+str(slitx[i])+'" width="'+str(syhi[i]-sylo[i])+'" height="'+str(slitw[i])+'"/>\n')
    f.write('</fatboy>\n')
    f.close()
#end writeRegionFileXML

#write sextractor param file
def writeSexParam(filename,style=1):
    f = open(filename,'w')
    if (style == 1):
        #normal
        f.write("NUMBER\nFLUXERR_ISO\nFLUX_AUTO\nFLUXERR_AUTO\nX_IMAGE\nY_IMAGE\nALPHA_J2000\nDELTA_J2000\nFLAGS\n")
    elif (style == 2):
        #for scamp usage
        f.write("NUMBER\nFLUX_APER(1)\nFLUXERR_APER(1)\nFLUX_AUTO\nFLUXERR_AUTO\nXWIN_IMAGE\nYWIN_IMAGE\nAWIN_IMAGE\nERRAWIN_IMAGE\nBWIN_IMAGE\nERRBWIN_IMAGE\nTHETAWIN_IMAGE\nERRTHETAWIN_IMAGE\nFLAGS\nIMAFLAGS_ISO(1)\nFLUX_RADIUS\n")
    f.close()
#end writeSexParam

#write sextractor conv file
def writeSexConv(filename):
    f = open(filename,'w')
    f.write("CONV NORM\n")
    f.write("# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n")
    f.write("1 2 1\n")
    f.write("2 4 2\n")
    f.write("1 2 1\n")
    f.close()
#end writeSexConv
