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
    print("gpu_pysurfit> WARNING: PyCUDA not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)

import scipy
from scipy.optimize import leastsq
from .fatboyDataUnit import *

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
        mod = SourceModule("""

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
              f += p[n] * pow(x[i], j-l) * pow(y[i], l);
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
              f += p[n] * pow(x[i], j-l) * pow(y[i], l);
              n++;
            }
          }
          resid[i] = (float)(data[i]-f);
        }


        __global__ void calcPysurface(float *fit, int nx, float *lsq, int order, int size) {
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
              fit[i] += lsq[n] * pow(x, j-l) * pow(y, l);
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

    """)
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
            _verbosity = log._verbosity

    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()

    #Find type
    if (mode is None):
        mode = MODE_FITS
        if (isinstance(frames[0], str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], ndarray)):
            mode = MODE_RAW
        elif (isinstance(frames[0], fatboyDataUnit)):
            mode = MODE_FDU

    #Process input
    if (mode == MODE_FITS):
        if (os.access(input, os.F_OK)):
            outimage = pyfits.open(input)
            data = outimage[mef].data.astype(float32)
        else:
            print("gpu_pysurfit> Could not find file "+input)
            write_fatboy_log(log, logtype, "Could not find file "+input, __name__)
            return None
    elif (mode == MODE_RAW):
        data = input.astype(float32)
    elif (mode == MODE_FDU):
        data = input.getData()
    elif (mode == MODE_FDU_DIFFERENCE):
        data = input[1].getData()-input[0].getData()
        #input should be a list in this case
    elif (mode == MODE_FDU_TAG):
        data = input.getData(tag=dataTag)
    else:
        print("gpu_pysurfit> Invalid input!  Exiting!")
        write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
        return None

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
    elif (not isinstance(inmask, ndarray)):
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
    xin = empty(n, float32)
    yin = empty(n, float32)
    d2 = empty(n, float32)
    hasMask = True
    if (inmask is None):
        inmask = empty(n, bool)
        hasMask = False
    else:
        inmask = inmask.astype(bool)
    binmask = empty(n, bool)
    binData(drv.In(data), drv.Out(xin), drv.Out(yin), drv.Out(d2), drv.In(inmask), drv.Out(binmask), int32(hasMask), int32(bin), int32(nx), grid=(blocks//(bin*bin),1), block=(block_size,1,1))
    inmask = binmask
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tBinning / Input mask / Input Arrays: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Setup initial guess for params
    terms = 0
    for j in range(order+2):
        terms+=j
    p = zeros(terms, float64)
    p[0] = d2[inmask].mean(dtype=float64)

    nkeep = gpusum(inmask)
    nkeepold = 0
    curriter = 0
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tInitial Guesses: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    while (curriter < niter and nkeep != nkeepold):
        tt = time.time()
        print("\tgpu_pysurfit: iteration "+str(curriter))
        write_fatboy_log(log, logtype, "pysurfit: iteration "+str(curriter), __name__, printCaller=False, tabLevel=1)
        b = inmask
        xb = xin[b]
        yb = yin[b]
        d2b = d2[b]
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tMasking: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()
        lsq = leastsq(pysurfaceResiduals, p, args=(xb,yb,d2b,order))
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tCalc Fit: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        print("\t\tFit params: "+str(lsq[0]))
        write_fatboy_log(log, logtype, "Fit params: "+str(lsq[0]), __name__, printCaller=False, tabLevel=1)
        resid = empty(d2.shape, float64)
        calcPysurfaceResid = mod.get_function("calcPysurfaceResid")
        calcPysurfaceResid(drv.In(xin), drv.In(yin), drv.In(d2), drv.Out(resid), drv.In(lsq[0]), int32(order), int32(resid.size), grid=(resid.size//512+1,1), block=(block_size,1,1))

        residb = empty(d2b.shape, float64)
        calcPysurfaceResid(drv.In(xb), drv.In(yb), drv.In(d2b), drv.Out(residb), drv.In(lsq[0]), int32(order), int32(residb.size), grid=(residb.size//512+1,1), block=(block_size,1,1))

        tempmean = residb.mean()
        tempstddev = residb.std()
        print("\t\tData - fit    mean: "+str(tempmean) + "   sigma: "+str(tempstddev))
        write_fatboy_log(log, logtype, "Data - fit    mean: "+str(tempmean) + "   sigma: "+str(tempstddev), __name__, printCaller=False, tabLevel=1)
        updateMask = mod.get_function("updateMask")
        updateMask(drv.InOut(inmask), drv.In(resid), float32(tempmean), float32(tempstddev), float32(upper), float32(lower), int32(resid.size), grid=(resid.size//512+1,1), block=(block_size,1,1))
        curriter+=1
        nkeepold = nkeep
        nkeep = gpusum(inmask)
        p = lsq[0]
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tCalc Resid: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

    #reconstruct fit from original data size
    calcPysurface = mod.get_function("calcPysurface")
    #fit = empty(shape=(ny,nx), dtype=float32)
    fit = zeros(shape=(ny,nx), dtype=float32)
    calcPysurface(drv.Out(fit), int32(nx), drv.In(float32(lsq[0])), int32(order), int32(fit.size), grid=(blocks,1), block=(block_size,1,1))
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tApply Fit: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    if (out is not None):
        print("\tOutput file: "+outfile)
        write_fatboy_log(log, logtype, "\tOutput file: "+outfile, __name__, printCaller=False, tabLevel=1)
        if (mode == MODE_FITS):
            outimage[mef].data = fit
        elif (mode == MODE_RAW):
            hdu = pyfits.PrimaryHDU(float32(fit))
            outimage = pyfits.HDUList([hdu])
        elif (mode == MODE_FDU or mode == MODE_FDU_TAG):
            outimage = pyfits.open(input.getFilename())
            outimage[mef].data = fit
        elif (mode == MODE_FDU_DIFFERENCE):
            outimage = pyfits.open(input[0].getFilename())
            outimage[mef].data = fit
        outimage.verify('silentfix')
        outimage.writeto(out, output_verify='silentfix')
        outimage.close()
    print("gpu_pysurfit: Total Time = "+str(time.time()-t)+" s")
    write_fatboy_log(log, logtype, "Total Time = "+str(time.time()-t)+" s", __name__, printCaller=False, tabLevel=1)
    return fit

def pysurfaceResiduals(p, x, y, out, order):
    resid = empty(x.shape, float64)
    blocks = x.size//512+1
    if (not superFATBOY.threaded()):
        global mod
    else:
        mod = get_mod()
    calcPysurfaceResid = mod.get_function("calcPysurfaceResid")
    calcPysurfaceResid(drv.In(x), drv.In(y), drv.In(out), drv.Out(resid), drv.In(p), int32(order), int32(x.size), grid=(blocks,1), block=(block_size,1,1))
    return resid
