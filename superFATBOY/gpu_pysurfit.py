hasCuda = True
try:
    import superFATBOY
    if (not superFATBOY.gpuEnabled()):
        hasCuda = False
    else:
        import cupy as cp
except Exception:
    print("gpu_pysurfit> WARNING: CuPy not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)

import numpy as np
import scipy
import time
from scipy.optimize import leastsq
from .fatboyDataUnit import *
from .fatboyLibs import gpusum

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction

blocks = 2048*4
block_size = 512

def get_mod():
    mod = None
    if (hasCuda and superFATBOY.gpuEnabled()):
        code = r"""
        extern "C" {
        __global__ void calcXin(float *xin, int nx, float offset) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          xin[i] = (i%nx) + offset;
        }

        __global__ void calcYin(float *yin, int nx, float offset) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          yin[i] = (i/nx) + offset;
        }

        __global__ void calcXYin(float *xin, float *yin, int nx, float xoff, float yoff) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          xin[i] = (i%nx) + xoff;
          yin[i] = (i/nx) + yoff;
        }

        __global__ void binData(float *data, float *xin, float *yin, float *d2, bool *inmask, bool *outmask, int hasMask, int bin, int nx) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          int x = i % (nx/bin) * bin;
          int y = i / (nx/bin) * bin;
          d2[i] = 0;
          int tempmask = 0;
          if (hasMask == 0) {
            for (int j = 0; j < bin; j++) {
              for (int l = 0; l < bin; l++) {
                d2[i] += data[x+j+(y+l)*nx];
              }
            }
            outmask[i] = 1;
          } else {
            for (int j = 0; j < bin; j++) {
              for (int l = 0; l < bin; l++) {
                d2[i] += data[x+j+(y+l)*nx];
                if (inmask[x+j+(y+l)*nx]) tempmask++;
              }
            }
            if (tempmask == (bin*bin)) outmask[i] = 1; else outmask[i] = 0;
          }
          d2[i] /= (bin*bin);
          xin[i] = x + (bin/2.0-0.5);
          yin[i] = y + (bin/2.0-0.5);
        }

        __global__ void calcPysurfaceResid(float *x, float *y, float *data, double *resid, double *p, int order, int size) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          double f = p[0];
          if (order >= 1) {
            f += p[1]*x[i];
            f += p[2]*y[i];
          }
          int n = 3;
          for (int j = 2; j < order+1; j++) {
            for (int l = 0; l < j+1; l++) {
              f += p[n] * powf((float)(x[i]), (float)(j-l)) * powf((float)(y[i]), (float)(l));
              n++;
            }
          }
          resid[i] = data[i]-f;
        }

        __global__ void calcPysurfaceResid_float(float *x, float *y, float *data, float *resid, double *p, int order, int size) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          double f = p[0];
          if (order >= 1) {
            f += p[1]*x[i];
            f += p[2]*y[i];
          }
          int n = 3;
          for (int j = 2; j < order+1; j++) {
            for (int l = 0; l < j+1; l++) {
              f += p[n] * powf((float)(x[i]), (float)(j-l)) * powf((float)(y[i]), (float)(l));
              n++;
            }
          }
          resid[i] = (float)(data[i]-f);
        }

        __global__ void calcPysurface(float *fit, int nx, double *lsq, int order, int size) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          float x = (i%nx);
          float y = (i/nx);
          fit[i] = lsq[0];
          if (order >= 1) {
            fit[i] += lsq[1]*x;
            fit[i] += lsq[2]*y;
          }
          int n = 3;
          for (int j = 2; j < order+1; j++) {
            for (int l = 0; l < j+1; l++) {
              fit[i] += lsq[n] * powf((float)(x), (float)(j-l)) * powf((float)(y), (float)(l));
              n++;
            }
          }
        }

        __global__ void updateMask(bool *inmask, double* resid, float tempmean, float tempstddev, float upper, float lower, int size) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= size) return;
          if (inmask[i] == false) return;
          double sigma = (resid[i]-tempmean)/tempstddev;
          if (sigma > upper || sigma < -1*lower) {
            inmask[i] = false;
          }
        }
        }
        """
        mod = cp.RawModule(code=code)
    return mod
#end get_mod()

mod = get_mod()

def pysurfit(input, out=None, order=1, niter=3, lower=2.5, upper=2.5, inmask=None, log=None, bin=2, mef=0, mode=None, dataTag=None):
    t = time.time()
    _verbosity = fatboyLog.NORMAL
    #set log type
    logtype = LOGTYPE_NONE
    if (log is not None):
        if (isinstance(log, str)):
            #log given as a string
            log = open(log,'a')
            logtype = LOGTYPE_ASCII
        elif(isinstance(log, fatboyLog)):
            logtype = LOGTYPE_FATBOY
            _verbosity = log.verbosity

    if (mode is None):
        if (isinstance(input, fatboyDataUnit)):
            mode = MODE_FDU
        elif (isinstance(input, list)):
            mode = MODE_FDU_DIFFERENCE
        elif (isinstance(input, str)):
            mode = MODE_FITS
        else:
            mode = MODE_RAW

    if (mode == MODE_FITS):
        outimage = pyfits.open(input)
        data = outimage[mef].data
    elif (mode == MODE_RAW):
        data = input
    elif (mode == MODE_FDU):
        data = input.getData()
    elif (mode == MODE_FDU_DIFFERENCE):
        #data is difference frame
        data = input[0].getData() - input[1].getData()
    elif (mode == MODE_FDU_TAG):
        data = input.getData(tag=dataTag)

    blocks = data.size//block_size
    if (data.size % block_size != 0):
        blocks += 1

    #input mask
    if (isinstance(inmask, str)):
        temp = pyfits.open(inmask)
        inmask = temp[mef].data.astype(bool)
        temp.close()
        temp = 0
        del temp
        nonzero = True
    elif (isinstance(inmask, fatboyDataUnit)):
        inmask = inmask.getData()
    elif (not isinstance(inmask, np.ndarray) and (cp is None or not isinstance(inmask, cp.ndarray))):
        inmask = None
    else:
        nonzero = True

    ny = data.shape[0]
    nx = data.shape[1]

    if (_verbosity == fatboyLog.VERBOSE):
        print("\tInitialize pysurfit: ",time.time()-t)
    tt = time.time()

    #binning
    binData = mod.get_function("binData")
    n = (ny*nx)//(bin*bin)
    
    is_cpu = isinstance(data, np.ndarray)
    data_gpu = cp.asarray(data).astype(cp.np.float32)
    
    xin_gpu = cp.np.empty(n, cp.np.float32)
    yin_gpu = cp.np.empty(n, cp.np.float32)
    d2_gpu = cp.np.empty(n, cp.np.float32)
    hasMask = True
    if (inmask is None):
        inmask_gpu = cp.np.zeros(data.size, dtype=bool) # Original size before binning? 
        # Actually binData takes data of size nx*ny and outputs d2 of size n.
        # It takes inmask of size nx*ny.
        hasMask = False
    else:
        inmask_gpu = cp.asarray(inmask).astype(bool)
        
    binmask_gpu = cp.np.empty(n, bool)
    
    binData((blocks//(bin*bin), 1), (block_size, 1, 1), (data_gpu, xin_gpu, yin_gpu, d2_gpu, inmask_gpu, binmask_gpu, np.int32(hasMask), np.int32(bin), np.int32(nx)))
    
    inmask_gpu = binmask_gpu # Now it's binned mask
    
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tBinning / Input mask / Input Arrays: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Setup initial guess for params
    terms = 0
    for j in range(order+2):
        terms+=j
    p = np.zeros(terms, np.float64)
    # Get d2 and inmask to CPU for mean calc if needed, or use CuPy
    p[0] = float(d2_gpu[inmask_gpu].mean())

    nkeep = int(gpusum(inmask_gpu))
    nkeepold = 0
    curriter = 0
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tInitial Guesses: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    calcPysurfaceResid = mod.get_function("calcPysurfaceResid")
    updateMask = mod.get_function("updateMask")

    while (curriter < niter and nkeep != nkeepold):
        tt = time.time()
        print("\tgpu_pysurfit: iteration "+str(curriter))
        write_fatboy_log(log, logtype, "pysurfit: iteration "+str(curriter), __name__, printCaller=False, tabLevel=1)
        
        xb_cpu = xin_gpu[inmask_gpu].get()
        yb_cpu = yin_gpu[inmask_gpu].get()
        d2b_cpu = d2_gpu[inmask_gpu].get()
        
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tMasking: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()
        
        lsq = leastsq(surfaceResiduals, p, args=(xb_cpu, yb_cpu, d2b_cpu, order))
        p = lsq[0]
        
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tCalc Fit: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        print("\t\tFit params: "+str(p))
        write_fatboy_log(log, logtype, "Fit params: "+str(p), __name__, printCaller=False, tabLevel=1)
        
        resid_gpu = cp.np.empty(d2_gpu.shape, cp.np.float64)
        p_gpu = cp.asarray(p).astype(cp.np.float64)
        
        calcPysurfaceResid((n//512+1,1), (block_size,1,1), (xin_gpu, yin_gpu, d2_gpu, resid_gpu, p_gpu, np.int32(order), np.int32(n)))

        residb_gpu = resid_gpu[inmask_gpu]
        tempmean = float(residb_gpu.mean())
        tempstddev = float(residb_gpu.std())
        
        print("\t\tData - fit    mean: "+str(tempmean) + "   sigma: "+str(tempstddev))
        write_fatboy_log(log, logtype, "Data - fit    mean: "+str(tempmean) + "   sigma: "+str(tempstddev), __name__, printCaller=False, tabLevel=1)
        
        updateMask((n//512+1,1), (block_size,1,1), (inmask_gpu, resid_gpu, np.float32(tempmean), np.float32(tempstddev), np.float32(upper), np.float32(lower), np.int32(n)))
        
        curriter+=1
        nkeepold = nkeep
        nkeep = int(gpusum(inmask_gpu))
        
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tCalc Resid: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

    #reconstruct fit from original data size
    calcPysurface = mod.get_function("calcPysurface")
    fit_gpu = cp.np.zeros((ny,nx), dtype=cp.np.float32)
    p_gpu = cp.asarray(p).astype(cp.np.float64)
    
    calcPysurface((data.size//block_size+1, 1), (block_size, 1, 1), (fit_gpu, np.int32(nx), p_gpu, np.int32(order), np.int32(data.size)))
    
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tApply Fit: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    fit = fit_gpu.get()

    if (out is not None):
        print("\tOutput file: "+out)
        write_fatboy_log(log, logtype, "\tOutput file: "+out, __name__, printCaller=False, tabLevel=1)
        if (mode == MODE_FITS):
            outimage[mef].data = fit
        elif (mode == MODE_RAW):
            hdu = pyfits.PrimaryHDU(fit)
            outimage = pyfits.HDUList([hdu])
        elif (mode == MODE_FDU or mode == MODE_FDU_TAG):
            outimage = pyfits.open(input.getFilename())
            outimage[mef].data = fit
        elif (mode == MODE_FDU_DIFFERENCE):
            outimage = pyfits.open(input[0].getFilename())
            outimage[mef].data = fit
        outimage.verify('silentfix')
        outimage.writeto(out, overwrite=True) # use overwrite instead of verify
        outimage.close()
        
    print("gpu_pysurfit: Total Time = "+str(time.time()-t)+" s")
    write_fatboy_log(log, logtype, "Total Time = "+str(time.time()-t)+" s", __name__, printCaller=False, tabLevel=1)
    return fit

def pysurfaceResiduals(p, x, y, data, order):
    # This is for leastsq on CPU
    # p = coeffs
    # x, y, data are CPU arrays
    f = p[0]
    if (order >= 1):
        f += p[1]*x
        f += p[2]*y
    n = 3
    for j in range(2, order+1):
        for l in range(j+1):
            f += p[n] * (x**(j-l)) * (y**l)
            n += 1
    return data - f
