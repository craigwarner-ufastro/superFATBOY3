#Notes - source can also be .fits file or file list
#in get calibs, if use_indiv_slit, check for slitmask - use badPixelMaskSpecs process
#in createBadPixelMask, use slitmask to find bad pixels in each slit
#in execute, add interpolating logic

from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
hasCuda = True
try:
    import superFATBOY
    if (not superFATBOY.gpuEnabled()):
        hasCuda = False
    else:
        import pycuda.driver as drv
        if (not superFATBOY.threaded()):
            #If not threaded mode, import autoinit.  Otherwise assume context exists.
            #Code will crash if in threaded mode and context does not exist.
            import pycuda.autoinit
        from pycuda.compiler import SourceModule
except Exception:
    print("badPixelMaskSpecProcess> Warning: PyCUDA not installed")
    hasCuda = False
from numpy import *
import os, time
from scipy.optimize import leastsq
from scipy.interpolate import interp1d, interp2d, griddata, bisplrep, bisplev

block_size = 512

#### Bad pixel mask replacement algorithms ###

def bpm_replace_linterp_x(data, bpm=None, niter=1, arg=None):
    #1d linear interpolation along X axis
    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            if (x-1 > 0 and x+1 < data.shape[1] and data[y,x-1] != 0 and data[y,x+1] != 0 and z[y,x-1] == False and z[y,x+1] == False):
                newdata[y,x] = (data[y,x-1]+data[y,x+1])/2.
            elif (x-1 > 0 and x+2 < data.shape[1] and data[y,x-1] != 0 and data[y,x+2] != 0 and z[y,x-1] == False and z[y,x+2] == False):
                newdata[y,x] = (data[y,x-1]+data[y,x+2])/2.
            elif (x-2 > 0 and x+1 < data.shape[1] and data[y,x-2] != 0 and data[y,x+1] != 0 and z[y,x-2] == False and z[y,x+1] == False):
                newdata[y,x] = (data[y,x-2]+data[y,x+1])/2.
            elif (x-1 > 0 and data[y,x-1] != 0 and z[y,x-1] == False):
                newdata[y,x] = data[y,x-1]
            elif (x+1 < data.shape[1] and data[y,x+1] != 0 and z[y,x+1] == False):
                newdata[y,x] = data[y,x+1]
            else:
                continue
            nreplace += 1
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_linterp_x

def bpm_replace_linterp_y(data, bpm=None, niter=1, arg=None):
    #1d linear interpolation along Y axis
    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            if (y-1 > 0 and y+1 < data.shape[0] and data[y-1,x] != 0 and data[y+1,x] != 0 and z[y-1,x] == False and z[y+1,x] == False):
                newdata[y,x] = (data[y-1,x]+data[y+1,x])/2.
            elif (y-1 > 0 and y+2 < data.shape[0] and data[y-1,x] != 0 and data[y+2,x] != 0 and z[y-1,x] == False and z[y+2,x] == False):
                newdata[y,x] = (data[y-1,x]+data[y+2,x])/2.
            elif (y-2 > 0 and y+1 < data.shape[0] and data[y-2,x] != 0 and data[y+1,x] != 0 and z[y-2,x] == False and z[y+1,x] == False):
                newdata[y,x] = (data[y-2,x]+data[y+1,x])/2.
            elif (y-1 > 0 and data[y-1,x] != 0 and z[y-1,x] == False):
                newdata[y,x] = data[y-1,x]
            elif (y+1 < data.shape[0] and data[y+1,x] != 0 and z[y+1,x] == False):
                newdata[y,x] = data[y+1,x]
            else:
                continue
            nreplace += 1
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_linterp_y

def bpm_replace_linterp_2d(data, bpm=None, niter=1, arg=None):
    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            pts = []
            if (y-1 > 0 and data[y-1,x] != 0 and z[y-1,x] == False):
                pts.append(data[y-1,x])
            elif (y-2 > 0 and data[y-2,x] != 0 and z[y-2,x] == False):
                pts.append(data[y-2,x])
            if (y+1 < data.shape[0] and data[y+1,x] != 0 and z[y+1,x] == False):
                pts.append(data[y+1,x])
            elif (y+2 < data.shape[0] and data[y+2,x] != 0 and z[y+2,x] == False):
                pts.append(data[y+2,x])
            if (x-1 > 0 and data[y,x-1] != 0 and z[y,x-1] == False):
                pts.append(data[y,x-1])
            elif (x-2 > 0 and data[y,x-2] != 0 and z[y,x-2] == False):
                pts.append(data[y,x-2])
            if (x+1 < data.shape[1] and data[y,x+1] != 0 and z[y,x+1] == False):
                pts.append(data[y,x+1])
            if (x+2 < data.shape[1] and data[y,x+2] != 0 and z[y,x+2] == False):
                pts.append(data[y,x+2])
            npts = float(len(pts))
            if (npts > 0):
                newdata[y,x] = sum(pts)/npts
                nreplace += 1
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_linterp_2d

def bpm_replace_median_neighbor(data, bpm=None, niter=1, arg=None):
    npts = 1
    if (arg is not None):
        npts = arg

    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        ny = data.shape[0]
        nx = data.shape[1]
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            temp = []
            for k in range(-1*npts,npts+1):
                for r in range(-1*npts,npts+1):
                    if (k == 0 and r == 0):
                        continue
                    xc = k+x
                    yc = r+y
                    if (xc >= 0 and xc < nx and yc >= 0 and yc < ny):
                        if (data[yc,xc] != 0 and z[yc,xc] == False):
                            temp.append(data[yc,xc])
            if (len(temp) != 0):
                temp = array(temp)
                newdata[y,x] = gpu_arraymedian(temp, kernel=fatboyclib.median, even=True)
                nreplace += 1
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_median_neighbor

def bpm_replace_median_neighbor_gpu(data, bpm=None, niter=1, arg=None):
    if (bpm is None):
        bpm = data == 0

    bpm = bpm.astype(bool)
    data[bpm] = 0

    if (not superFATBOY.threaded()):
        global fatboy_mod
    else:
        fatboy_mod = get_fatboy_mod()
    badPixRemoval = fatboy_mod.get_function("medianNeighborReplaceBadPix")
    output = empty(data.shape, float32)
    out_bpm = empty(data.shape, bool)
    rows = data.shape[0]
    cols = data.shape[1]
    ct = zeros(1, int32)
    blocks = data.size//512
    if (data.size % 512 != 0):
        blocks += 1
    nreplace = 0
    for iter in range(niter):
        badPixRemoval(drv.In(data), drv.Out(output), drv.In(bpm), drv.Out(out_bpm), int32(rows), int32(cols), drv.InOut(ct), grid=(blocks,1), block=(block_size,1,1))
        data = output
        bpm = out_bpm
        nreplace += ct[0]
        ct = zeros(1, int32)
    return (data, nreplace)
#end bpm_replace_median_neighbor_gpu

def bpm_replace_scipy_interp2d(data, bpm=None, niter=1, arg=None):
    npts = 2
    if (arg is not None):
        npts = arg

    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            tempdata = []
            tempx = []
            tempy = []
            xmin = max(0, x-npts)
            xmax = min(data.shape[1], x+npts+1)
            ymin = max(0, y-npts)
            ymax = min(data.shape[0], y+npts+1)
            for i in range(ymin, ymax):
                for k in range(xmin, xmax):
                    if (data[i, k] != 0 and z[i, k] == False):
                        tempy.append(i)
                        tempx.append(k)
                        tempdata.append(data[i, k])
            if (len(tempy) < 4):
                #Not enough datapoints found
                continue
            tempx = array(tempx, dtype=float64)
            tempy = array(tempy, dtype=float64)
            tempdata = array(tempdata, dtype=float64)

            try:
                nz = len(tempdata)
                (tck, fp, ier, msg) = bisplrep(tempx, tempy, tempdata, full_output=1, kx=1, ky=1, s=nz, nxest=nz//2, nyest=nz//2)
                if (ier > 0):
                    #Error
                    (tck, fp, ier, msg) = bisplrep(tempx, tempy, tempdata, full_output=1, kx=1, ky=1, s=nz*3, nxest=nz, nyest=nz)
                    if (ier > 0):
                        continue
                newdata[y,x] = bisplev(x, y, tck)
                nreplace += 1
            except Exception as ex:
                continue
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_scipy_interp2d

def bpm_replace_scipy_griddata_cubic(data, bpm=None, niter=1, arg=None):
    npts = 2
    if (arg is not None):
        npts = arg

    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            tempdata = []
            tempx = []
            tempy = []
            xmin = max(0, x-npts)
            xmax = min(data.shape[1], x+npts+1)
            ymin = max(0, y-npts)
            ymax = min(data.shape[0], y+npts+1)
            for i in range(ymin, ymax):
                for k in range(xmin, xmax):
                    if (i == y and k == x):
                        #This is the bad pixel itself
                        continue
                    #Check if data == 0 instead of if bpm != 1 because on iteration 2
                    #bpm is still 1 on data that has been replaced on iteration 1
                    #Don't use data value 1.e-6 that were zeros for interpolation either
                    #Probably outside of slitlet so masked out.
                    if (data[i, k] != 0 and z[i, k] == False):
                        tempy.append(i)
                        tempx.append(k)
                        tempdata.append(data[i, k])
            if (len(tempy) < 5):
                #Not enough datapoints found
                continue
            tempx = array(tempx, dtype=float64)
            tempy = array(tempy, dtype=float64)
            tempdata = array(tempdata, dtype=float64)
            #For edge conditions, use median as fill value
            fv = gpu_arraymedian(tempdata)

            try:
                newdata[y,x] = griddata(array([tempx, tempy]).transpose(), tempdata, (x,y), method='cubic', fill_value=fv)
                nreplace += 1
            except Exception as ex:
                print(ex)
                continue
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_scipy_griddata_cubic

def bpm_replace_scipy_interp2d_quintic(data, bpm=None, niter=1, arg=None):
    npts = 3
    nyest = len(data)//2
    nxest = len(data)//2
    if (arg is not None):
        npts = arg

    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            tempdata = []
            tempx = []
            tempy = []
            xmin = max(0, x-npts)
            xmax = min(data.shape[1], x+npts+1)
            ymin = max(0, y-npts)
            ymax = min(data.shape[0], y+npts+1)
            for i in range(ymin, ymax):
                for k in range(xmin, xmax):
                    if (data[i, k] != 0 and z[i, k] == False):
                        tempy.append(i)
                        tempx.append(k)
                        tempdata.append(data[i, k])
            if (len(tempy) < 5):
                #Not enough datapoints found
                continue
            tempx = array(tempx, dtype=float64)
            tempy = array(tempy, dtype=float64)
            tempdata = array(tempdata, dtype=float64)

            try:
                nz = len(tempdata)
                (tck, fp, ier, msg) = bisplrep(tempx, tempy, tempdata, full_output=1, kx=5, ky=5, s=nz, nxest=nz//2, nyest=nz//2)
                if (ier > 0):
                    #Error
                    (tck, fp, ier, msg) = bisplrep(tempx, tempy, tempdata, full_output=1, kx=5, ky=5, s=nz*3, nxest=nz, nyest=nz)
                    if (ier > 0):
                        continue
                newdata[y,x] = bisplev(x, y, tck)
                nreplace += 1
            except Exception as ex:
                print(ex)
                continue
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_scipy_interp2d_quintic

def bpm_replace_weighted(data, bpm=None, niter=1, arg=None):
    kernel = array([[0,2,0],[0,5,0],[1,0,1],[0,5,0],[0,2,0]])
    if (arg is not None):
        #Validate that this is array([...]) before using eval
        if (arg.startswith("array([") and arg.endswith("])")):
            kernel = eval(arg)

    if (bpm is None):
        bpm = data == 0

    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        ny = data.shape[0]
        nx = data.shape[1]
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            temp = zeros((5,3))
            for k in range(3):
                for r in range(5):
                    xc = k+x-1
                    yc = r+y-2
                    if (k == x and r == y):
                        continue
                    if (xc >= 0 and xc < nx and yc >= 0 and yc < ny):
                        if (data[yc,xc] != 0 and z[yc,xc] == False):
                            temp[r,k] = data[yc,xc]
            wgts = (temp != 0)*kernel
            if (wgts.sum() != 0):
                newdata[y,x] = (wgts*temp).sum()/wgts.sum()
                nreplace += 1
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_weighted

def bpm_replace_biharmonic_2d(data, bpm=None, niter=1, arg=None):
    npts = 2
    if (arg is not None):
        npts = arg

    if (bpm is None):
        bpm = data == 0
    #Set zeros not flagged as bad pixels to be 1.e-6 and all bad pixels to be zero
    z = (bpm == 0)*(data == 0)
    data[z] = 1.e-6
    data[bpm] = 0

    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        #Check if data == 0 instead of if bpm != 1 because on iteration 2
        #bpm is still 1 on data that has been replaced on iteration 1
        b = where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            ws = []
            eqs = []
            tempx = []
            tempy = []
            xmin = max(0, x-npts)
            xmax = min(data.shape[1], x+npts+1)
            ymin = max(0, y-npts)
            ymax = min(data.shape[0], y+npts+1)
            for i in range(ymin, ymax):
                for k in range(xmin, xmax):
                    if (data[i, k] != 0 and z[i, k] == False):
                        ws.append(data[i, k])
                        eqs.append([])
                        for yj in range(ymin, ymax):
                            for xj in range(xmin, xmax):
                                if (xj == k and yj == i):
                                    eqs[-1].append(0)
                                elif (data[yj, xj] != 0 and z[yj, xj] == False):
                                    eqs[-1].append(((k-xj)**2+(i-yj)**2)*(2*math.log(math.sqrt((k-xj)**2+(i-yj)**2))-1))
            if (len(ws) < 2):
                #Not enough datapoints found
                continue
            eqs = array(eqs, dtype=float64)
            ws = array(ws, dtype=float64)
            alphas = linalg.solve(eqs, ws)
            wxy = []
            for i in range(ymin, ymax):
                for k in range(xmin, xmax):
                    if (i == y and k == x):
                        continue
                    if (data[i, k] == 0 or z[i, k] != False):
                        continue
                    wxy.append(((x-k)**2+(y-i)**2)*(2*math.log(math.sqrt((x-k)**2+(y-i)**2))-1))
            newdata[y,x] = dot(wxy, alphas)
            nreplace += 1
        data = newdata.copy()
    #Reset data to 0 that was zero but not bad pixels
    data[z] = 0
    return (data, nreplace)
#end bpm_replace_biharmonic_2d

class badPixelMaskSpecProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    def get_bpm_mod(self):
        bpm_mod = None
        if (hasCuda):
            bpm_mod = SourceModule("""
          __global__ void gpu_bad_pixel_mask(int *output, float *input, int nx, int ny, float lo, float hi, int edge, float radius) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= nx*ny) return;
            float xin = (i%nx);
            float yin = (i/nx);

            output[i] = 0; //false
            if (input[i] < lo || input[i] > hi) {
              output[i] = 1;
              return;
            }
            if (edge > 0) {
              if (xin < edge || yin < edge || xin >= nx-edge || yin >= ny-edge) {
                output[i] = 1;
                return;
              }
            }
            if (radius > 0) {
              xin -= ((nx-1)/2.);
              yin -= ((ny-1)/2.);
              float r = sqrt(xin*xin+yin*yin);
              if (r > radius) output[i] = 1;
            }
            return;
          }

          __global__ void gpu_bad_pixel_mask_mos(int *output, float *input, int *slitmask, int nx, int ny, float *lo, float *hi, int edge, float radius) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= nx*ny) return;
            float xin = (i%nx);
            float yin = (i/nx);

            output[i] = 0; //false

            if (edge > 0) {
              if (xin < edge || yin < edge || xin >= nx-edge || yin >= ny-edge) {
                output[i] = 1;
                return;
              }
            }
            if (radius > 0) {
              xin -= ((nx-1)/2.);
              yin -= ((ny-1)/2.);
              float r = sqrt(xin*xin+yin*yin);
              if (r > radius) output[i] = 1;
            }

            int slit = slitmask[i];
            if (slit == 0) return; //not in any slit
            if (input[i] < lo[slit-1] || input[i] > hi[slit-1]) {
              output[i] = 1;
              return;
            }
            return;
          }
          """)
        return bpm_mod
    #end get_fatboy_mod

    #Convenience method for median combining source frames
    def combineSourceFrames(self, source, fdu):
        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            (data, header) = gpu_imcombine.imcombine(source, method="median", returnHeader=True, log=self._log)
        else:
            (data, header) = imcombine.imcombine(source, method="median", returnHeader=True, log=self._log)
        sourceFDU = fatboySpecCalib(self._pname, "bpm_source", fdu, data=data, log=self._log)
        return sourceFDU
    #end combineSourceFrames

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createBadPixelMask(self, fdu, sourceFDU, calibs):
        badPixelMask = None
        bpmfilename = None
        bpmname = "badPixelMasks/BPM-"+str(fdu.filter).replace(" ","_")+"-"+str(fdu._id)
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (fdu.getTag(mode="composite") is not None):
            bpmname += "-"+fdu.getTag(mode="composite").replace(" ","_")
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/badPixelMasks", os.F_OK)):
                os.mkdir(outdir+"/badPixelMasks",0o755)
        #Check to see if bad pixel mask exists already from a previous run
        bpmfilename = outdir+"/"+bpmname+".fits"
        if (os.access(bpmfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(bpmfilename)
        elif (os.access(bpmfilename, os.F_OK)):
            #file already exists.  #Use fdu as source header
            print("badPixelMaskSpecProcess::createBadPixelMask> Bad pixel mask "+bpmfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Bad pixel mask "+bpmfilename+" already exists!  Re-using...")
            badPixelMask = fatboySpecCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, filename=bpmfilename, log=self._log)
            return badPixelMask

        print("badPixelMaskSpecProcess::createBadPixelMask> Using "+str(sourceFDU.getFilename())+" to calculate bad pixel mask...")
        self._log.writeLog(__name__, "Using "+str(sourceFDU.getFilename())+" to calculate bad pixel mask...")
        clippingMethod = self.getOption('clipping_method', fdu.getTag()).lower()
        doNormalize = self.getOption('normalize_source', fdu.getTag()).lower()
        try:
            clipping_high = float(self.getOption('clipping_high', fdu.getTag()))
            clipping_low = float(self.getOption('clipping_low', fdu.getTag()))
            clipping_sigma = float(self.getOption('clipping_sigma', fdu.getTag()))
            edge_reject = int(self.getOption('edge_reject', fdu.getTag()))
            radius_reject = float(self.getOption('radius_reject', fdu.getTag()))
        except ValueError as ex:
            print("badPixelMaskSpecProcess::createBadPixelMask> Error: invalid bad pixel mask options: "+str(ex))
            self._log.writeLog(__name__, " invalid bad pixel mask options: "+str(ex), type=fatboyLog.ERROR)
            return None

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
                if (doNormalize == "yes" and not sourceFDU.hasProperty("normalized")):
                    #need to normalize source first
                    sourceFDU.renormalize()
                    sourceFDU.setProperty("normalized", True)
                if (clippingMethod == "values"):
                    lo = clipping_low
                    hi = clipping_high
                elif (clippingMethod == "sigma"):
                    sigclip = sigmaFromClipping(sourceFDU.getData(), clipping_sigma, 5)
                    med = sigclip[1]
                    stddev = sigclip[2]
                    lo = med-sig*stddev
                    hi = med+sig*stddev
                #Create bad pixel mask
                (nx, ny) = fdu.getShape()
                gpu_bad_pixel_mask = self.get_bpm_mod().get_function("gpu_bad_pixel_mask")
                data = empty(fdu.getShape(), int32)
                blocks = data.size//block_size
                if (data.size % 512 != 0):
                    blocks += 1
                gpu_bad_pixel_mask(drv.Out(data), drv.In(sourceFDU.getData()), int32(nx), int32(ny), float32(lo), float32(hi), int32(edge_reject), float32(radius_reject), grid=(blocks,1), block=(block_size,1,1))
            else:
                if (not 'slitmask' in calibs or not 'nslits' in calibs):
                    #Can't find slitmask, error
                    print("badPixelMaskSpecProcess::createBadPixelMask> Error: could not find slitmask for "+sourceFDU.getFullId())
                    self._log.writeLog(__name__, "could not find slitmask for "+sourceFDU.getFullId())
                    return None
                if (doNormalize == "yes" and not sourceFDU.hasProperty("normalized")):
                    #need to normalize source first
                    normalizeMOSSource(sourceFDU, calibs['slitmask'].getData(), calibs['nslits'], self._log)
                    sourceFDU.setProperty("normalized", True)

                if (clippingMethod == "values"):
                    lo = [clipping_low]*calibs['nslits']
                    hi = [clipping_high]*calibs['nslits']
                elif (clippingMethod == "sigma"):
                    lo = []
                    hi = []
                    for j in range(calibs['nslits']):
                        slit = calibs['slitmask'].getData() == (j+1)
                        sigclip = sigmaFromClipping(sourceFDU.getData()[slit], clipping_sigma, 5)
                        med = sigclip[1]
                        stddev = sigclip[2]
                        lo.append(med-sig*stddev)
                        hi.append(med+sig*stddev)
                lo = array(lo, dtype=float32)
                hi = array(hi, dtype=float32)
                #Create bad pixel mask
                (nx, ny) = fdu.getShape()
                gpu_bad_pixel_mask_mos = self.get_bpm_mod().get_function("gpu_bad_pixel_mask_mos")
                data = empty(fdu.getShape(), int32)
                blocks = data.size//block_size
                if (data.size % 512 != 0):
                    blocks += 1
                gpu_bad_pixel_mask_mos(drv.Out(data), drv.In(sourceFDU.getData()), drv.In(calibs['slitmask'].getData().astype(int32)), int32(nx), int32(ny), drv.In(lo), drv.In(hi), int32(edge_reject), float32(radius_reject), grid=(blocks,1), block=(block_size,1,1))
            #Convert to bool here
            data = data.astype(bool)
        else:
            if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
                if (doNormalize == "yes" and not sourceFDU.hasProperty("normalized")):
                    #need to normalize source first
                    sourceFDU.renormalize()
                    sourceFDU.setProperty("normalized", True)
                if (clippingMethod == "values"):
                    lo = clipping_low
                    hi = clipping_high
                elif (clippingMethod == "sigma"):
                    sigclip = sigmaFromClipping(sourceFDU.getData(), clipping_sigma, 5)
                    med = sigclip[1]
                    stddev = sigclip[2]
                    lo = med-sig*stddev
                    hi = med+sig*stddev
                #Create bad pixel mask
                data = logical_or(sourceFDU.getData() < lo, sourceFDU.getData() > hi)
            else:
                if (not 'slitmask' in calibs or not 'nslits' in calibs):
                    #Can't find slitmask, error
                    print("badPixelMaskSpecProcess::createBadPixelMask> Error: could not find slitmask for "+sourceFDU.getFullId())
                    self._log.writeLog(__name__, "could not find slitmask for "+sourceFDU.getFullId())
                    return None
                if (doNormalize == "yes" and not sourceFDU.hasProperty("normalized")):
                    #need to normalize source first
                    sourceFDU.renormalize(slitmask=calibs['slitmask'])
                    sourceFDU.setProperty("normalized", True)
                #Create bad pixel mask slitlet by slitlet
                #Initialize as all zeros
                data = zeros(sourceFDU.getData().shape, bool)
                for j in range(calibs['nslits']):
                    slit = calibs['slitmask'].getData() == (j+1)
                    if (clippingMethod == "values"):
                        lo = clipping_low
                        hi = clipping_high
                    elif (clippingMethod == "sigma"):
                        sigclip = sigmaFromClipping(sourceFDU.getData()[slit], clipping_sigma, 5)
                        med = sigclip[1]
                        stddev = sigclip[2]
                        lo = med-sig*stddev
                        hi = med+sig*stddev
                    #Update bad pixel mask for this slit
                    data[slit] = logical_or(sourceFDU.getData()[slit] < lo, sourceFDU.getData()[slit] > hi)

            #Handle edge and radius here - no need to split based on LONGSLIT or MOS
            if (edge_reject > 0):
                data[0:edge_reject,:] = True
                data[:,0:edge_reject] = True
                data[-1*edge_reject:,:] = True
                data[:,-1*edge_reject:] = True
            if (radius_reject > 0):
                (nx, ny) = data.shape
                xs = arange(ny*nx).reshape(ny,nx) % nx - ((nx-1)/2.)
                ys = arange(ny*nx).reshape(ny,nx) // nx - ((ny-1)/2.)
                rs = sqrt(xs**2+ys**2)
                data[rs > radius_reject] = True
        #Column/row reject regardless of data format and GPU mode
        if (self.getOption('column_reject', fdu.getTag()) is not None):
            #Column reject of format "54, 320:384, 947, 1024:1080"
            column_reject = self.getOption('column_reject', fdu.getTag())
            try:
                #Parse out into list
                column_reject = column_reject.split(",")
                for j in range(len(column_reject)):
                    column_reject[j] = column_reject[j].strip().split(":")
                    #Mask out data
                    if (len(column_reject[j]) == 1):
                        data[:,int(column_reject[j][0])] = True
                    elif (len(column_reject[j]) == 2):
                        data[:,int(column_reject[j][0]):int(column_reject[j][1])] = True
            except ValueError as ex:
                print("badPixelMaskSpecProcess::createBadPixelMask> Error: invalid format in column_reject: "+str(ex))
                self._log.writeLog(__name__, " invalid format in column_reject: "+str(ex), type=fatboyLog.ERROR)
                return None
        if (self.getOption('row_reject', fdu.getTag()) is not None):
            #Row reject of format "54, 320:384, 947, 1024:1080"
            row_reject = self.getOption('row_reject', fdu.getTag())
            try:
                #Parse out into list
                row_reject = row_reject.split(",")
                for j in range(len(row_reject)):
                    row_reject[j] = row_reject[j].strip().split(":")
                    #Mask out data
                    if (len(row_reject[j]) == 1):
                        data[int(row_reject[j][0]),:] = True
                    elif (len(row_reject[j]) == 2):
                        data[int(row_reject[j][0]):int(row_reject[j][1]),:] = True
            except ValueError as ex:
                print("badPixelMaskSpecProcess::createBadPixelMask> Error: invalid format in row_reject: "+str(ex))
                self._log.writeLog(__name__, " invalid format in row_reject: "+str(ex), type=fatboyLog.ERROR)
                return None

        print("badPixelMaskSpecProcess::createBadPixelMask> Found "+str(data.sum())+" bad pixels...")
        self._log.writeLog(__name__, " Found "+str(data.sum())+" bad pixels...")
        pct = 100*float(data.sum())/float(data.size)
        if (pct > 25):
            print("badPixelMaskSpecProcess::createBadPixelMask> WARNING: "+str(pct)+"% of pixels found to be bad.  Check your flat field!")
            self._log.writeLog(__name__, str(pct)+"% of pixels found to be bad.  Check your flat field!", type=fatboyLog.WARNING)
        #Use fdu as source - this will copy over filter/exptime/section
        badPixelMask = fatboySpecCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, data=data, tagname=bpmname, log=self._log)
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            if (os.access(bpmfilename, os.F_OK)):
                os.unlink(bpmfilename)
            #Optionally save if write_calib_output = yes
            badPixelMask.writeTo(bpmfilename)
        return badPixelMask
    #end createBadPixelMask

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Bad Pixel Mask")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For badPixelMask, this dict should have one entry 'badPixelMask' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'badPixelMask' in calibs):
            #Failed to obtain master flat frame
            #Issue error message and disable this FDU
            print("badPixelMaskSpecProcess::execute> ERROR: Bad pixel mask not applied for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Bad pixel mask not applied for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        bafile = "badPixelMaskApplied/ba_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, bafile)):
            return True

        #get bad pixel mask
        badPixelMask = calibs['badPixelMask']

        if (self.getOption("behavior", fdu.getTag()).lower() == "interpolate"):
            #interpolate fdu, masterFlat, any other frames found
            algorithm = self.getOption("interpolation_algorithm", fdu.getTag())
            niter = int(self.getOption("interpolation_iterations", fdu.getTag()))
            arg = self.getOption("interpolation_arg", fdu.getTag())
            if (arg is not None and isInt(arg)):
                arg = int(arg)

            #Default is median_neighbor
            bpm_replace_algorithm = bpm_replace_median_neighbor
            if (algorithm == "linear_1d_x"):
                bpm_replace_algorithm = bpm_replace_linterp_x
            elif (algorithm == "linear_1d_y"):
                bpm_replace_algorithm = bpm_replace_linterp_y
            elif (algorithm == "linear_2d"):
                bpm_replace_algorithm = bpm_replace_linterp_2d
            elif (algorithm == "median_neighbor"):
                bpm_replace_algorithm = bpm_replace_median_neighbor
                if (self._fdb.getGPUMode()):
                    bpm_replace_algorithm = bpm_replace_median_neighbor_gpu
            elif (algorithm == "linear_spline"):
                bpm_replace_algorithm = bpm_replace_scipy_interp2d
            elif (algorithm == "cubic_spline"):
                bpm_replace_algorithm = bpm_replace_scipy_griddata_cubic
            elif (algorithm == "quintic_spline"):
                bpm_replace_algorithm = bpm_replace_scipy_interp2d_quintic
            elif (algorithm == "weighted"):
                bpm_replace_algorithm = bpm_replace_weighted
            elif (algorithm == "biharmonic_2d"):
                bpm_replace_algorithm = bpm_replace_biharmonic_2d

            npix = badPixelMask.getData().sum()
            print("badPixelMaskSpecProcess::execute> Replacing "+str(npix)+" bad pixels with algorithm "+algorithm+"...")
            self._log.writeLog(__name__, "Replacing "+str(npix)+" bad pixels with algorithm "+algorithm+"...")

            t = time.time()
            (data, nreplace) = bpm_replace_algorithm(float32(fdu.getData()), badPixelMask.getData(), niter, arg)
            t2 = time.time()-t

            print("badPixelMaskSpecProcess::execute> Replaced "+str(nreplace)+" of "+str(npix)+" bad pixels using "+str(niter)+" iterations in "+str(t2)+" seconds.")
            self._log.writeLog(__name__, "Replaced "+str(nreplace)+" of "+str(npix)+" bad pixels using "+str(niter)+" iterations in "+str(t2)+" seconds.")
            fdu.updateData(data)
            fdu._header.add_history("Interpolated "+str(nreplace)+" of "+str(npix)+" bad pixels with algorithm "+algorithm+" niter="+str(niter))
        else:
            if ('masterFlat' in calibs):
                #apply bad pixel mask to master flat and renormalize
                #get master flat
                masterFlat = calibs['masterFlat']
                if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
                    #Renormalize master flat to median value of 1
                    masterFlat.renormalize()
                    if (not masterFlat.hasHistory('renormalized_bpm')):
                        masterFlat.renormalize(bpm=badPixelMask.getData())
                    #Update FDU with new median value of master flat
                    scaleFactor = masterFlat.getHistory('renormalized_bpm')
                    fdu.updateData(float32(fdu.getData())*scaleFactor)
                    fdu.setHistory('rescaled_for_bad_pixels', scaleFactor)
                    fdu._header.add_history('rescaled for bad pixels by '+str(scaleFactor))
                else:
                    if (not 'slitmask' in calibs or not 'nslits' in calibs):
                        #Can't find slitmask, error
                        print("badPixelMaskSpecProcess::execute> Error: could not find slitmask for "+fdu.getFullId())
                        self._log.writeLog(__name__, "could not find slitmask for "+fdu.getFullId())
                        return False
                    #Renormalize master flat to median value of 1
                    masterFlat.renormalize(slitmask=calibs['slitmask'])
                    if (not masterFlat.hasHistory('renormalized_bpm_01')):
                        masterFlat.renormalize(slitmask=calibs['slitmask'], bpm=badPixelMask.getData())
                    #Update FDU with new median value of master flat
                    data = float32(fdu.getData())
                    for j in range(calibs['nslits']):
                        key = ''
                        if (j+1 < 10):
                            key += '0'
                        key += str(j+1)
                        scaleFactor = masterFlat.getHistory('renormalized_bpm_'+key)
                        slit = calibs['slitmask'].getData() == (j+1)
                        data[slit] *= scaleFactor
                        fdu.setHistory('rescaled_for_bad_pixels_'+key, scaleFactor)
                        fdu._header.add_history('rescaled slitlet '+key+' for bad pixels by '+str(scaleFactor))
                    fdu.updateData(data)
                if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
                    #Optionally save renormalized flat if write_calib_output = yes
                    outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
                    if (not os.access(outdir+"/badPixelMasks", os.F_OK)):
                        os.mkdir(outdir+"/badPixelMasks",0o755)
                    mffilename = outdir+"/badPixelMasks/renorm_"+masterFlat.getFullId()
                    if (os.access(mffilename, os.F_OK)):
                        os.unlink(mffilename)
                    #Optionally save if write_calib_output = yes
                    masterFlat.writeTo(mffilename)

            #Apply bad pixel mask
            #9/28/18 This should only be done for behavior=mask
            fdu.applyBadPixelMask(badPixelMask)
            fdu._header.add_history('Applied bad pixel mask '+badPixelMask.filename)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for matching grism_keyword, specmode, and flat_method
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")

        #First find a master flat frame to also apply bad pixel mask to
        #1) Check for an already created master flat frame matching specmode/filter/grism and TAGGED for this object
        masterFlat = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype="master_flat", filter=fdu.filter, properties=properties, headerVals=headerVals)
        if (masterFlat is not None):
            #Found master flat
            calibs['masterFlat'] = masterFlat
        else:
            #2) Check for an already created master flat frame matching specmode/filter/grism
            masterFlat = self._fdb.getMasterCalib(pname=None, obstype="master_flat", filter=fdu.filter, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
            if (masterFlat is not None):
                #Found master flat.
                calibs['masterFlat'] = masterFlat
            else:
                #3) Look at previous master flats to see if any has a history of being used as master flat for
                #this _id and filter combination from step 7 below.
                masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
                for mflat in masterFlats:
                    if (mflat.hasHistory('master_flat::'+fdu._id+'::'+str(fdu.filter)+'::'+str(fdu.grism)+'::'+str(fdu.getProperty("specmode")))):
                        #Use this master flat
                        print("badPixelMaskSpecProcess::getCalibs> Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter)+", grism "+str(mflat.grism)+", specmode "+str(mflat.getProperty("specmode")))
                        self._log.writeLog(__name__, "Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter)+", grism "+str(mflat.grism)+", specmode "+str(mflat.getProperty("specmode")))
                        #Already in _calibs, no need to appendCalib
                        calibs['masterFlat'] = mflat

        #If master flat not found, perhaps it needs to be created
        #Use flatDivideSpecProcess.getCalibs to get masterFlat and create if necessary
        #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
        #Returns None on a failure
        if (not 'masterFlat' in calibs):
            fds_process = self._fdb.getProcessByName("flatDivideSpec")
            if (fds_process is None or not isinstance(fds_process, fatboyProcess)):
                print("badPixelMaskSpecProcess::getCalibs> ERROR: could not find process flatDivideSpec!  Check your XML file!")
                self._log.writeLog(__name__, "could not find process flatDivideSpec!  Check your XML file!", type=fatboyLog.ERROR)
            else:
                #Call setDefaultOptions and getCalibs on flatDivideSpecProcess
                fds_process.setDefaultOptions()
                calibs = fds_process.getCalibs(fdu, prevProc)
                if (not 'masterFlat' in calibs):
                    #Failed to obtain master flat frame
                    print("badPixelMaskSpecProcess::execute> ERROR: Master flat not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+", grism="+str(fdu.grism)+", specmode="+str(fdu.getProperty("specmode"))+")!")
                    self._log.writeLog(__name__, "Master flat not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+", grism="+str(fdu.grism)+", specmode="+str(fdu.getProperty("specmode"))+")!", type=fatboyLog.ERROR)

        #Look for slitmask (for method=median only)
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("skySubtractSpecProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("skySubtractSpecProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        elif (fdu.hasProperty('slitmask')):
            calibs['slitmask'] = fdu.getProperty('slitmask')

        #If not longslit data and use_individual_slitlets, find slitmask
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and self.getOption("use_individual_slitlets", fdu.getTag()).lower() == "yes"):
            if (not 'slitmask' in calibs):
                #Use findSlitletProcess.getCalibs to get slitmask and create if necessary
                #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
                #Returns None on a failure
                fs_process = self._fdb.getProcessByName("findSlitlets")
                if (fs_process is None or not isinstance(fs_process, fatboyProcess)):
                    print("badPixelMaskSpecProcess::getCalibs> ERROR: could not find process findSlitlet!  Check your XML file!")
                    self._log.writeLog(__name__, "could not find process findSlitlet!  Check your XML file!", type=fatboyLog.ERROR)
                    return calibs
                #Call setDefaultOptions and getCalibs on findSlitletProcess
                fs_process.setDefaultOptions()
                fs_calibs = fs_process.getCalibs(fdu, prevProc)
                if (not 'slitmask' in fs_calibs):
                    #Failed to obtain slitmask
                    #Issue error message.  FDU will be disabled in execute()
                    print("badPixelMaskSpecProcess::execute> ERROR: Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+", grism="+str(fdu.grism)+", specmode="+str(fdu.getProperty("specmode"))+")!")
                    self._log.writeLog(__name__, "Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+", grism="+str(fdu.grism)+", specmode="+str(fdu.getProperty("specmode"))+")!", type=fatboyLog.ERROR)
                    return calibs
                calibs['slitmask'] = fs_calibs['slitmask']
            if (not calibs['slitmask'].hasProperty("nslits")):
                calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
            calibs['nslits'] = calibs['slitmask'].getProperty("nslits")

        bpmfilename = self.getCalib("badPixelMask", fdu.getTag())
        if (bpmfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(bpmfilename, os.F_OK)):
                print("badPixelMaskSpecProcess::getCalibs> Using bad pixel mask "+bpmfilename+"...")
                self._log.writeLog(__name__, "Using bad pixel mask "+bpmfilename+"...")
                calibs['badPixelMask'] = fatboySpecCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, filename=bpmfilename, log=self._log)
                return calibs
            else:
                print("badPixelMaskSpecProcess::getCalibs> Warning: Could not find bad pixel mask "+bpmfilename+"...")
                self._log.writeLog(__name__, "Could not find bad pixel mask "+bpmfilename+"...", type=fatboyLog.WARNING)

        #Next find or create bad pixel mask
        #1) Check for an already created bad pixel mask matching section and TAGGED for this object.  filter does not need to match.
        badPixelMask = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK)
        if (badPixelMask is not None):
            #Found bpm.  Return here
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #2) Check for an already created bad pixel mask matching specmode/filter/grism
        badPixelMask = self._fdb.getMasterCalib(self._pname, obstype=fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, filter=fdu.filter, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
        if (badPixelMask is not None):
            #Found bpm.  Return here
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #3) Check default_bad_pixel_mask for matching specmode/filter/grism before using source, master_flat, or master_dark
        defaultBPMs = []
        if (self.getOption('default_bad_pixel_mask', fdu.getTag()) is not None):
            dbpmlist = self.getOption('default_bad_pixel_mask', fdu.getTag())
            ignoreHeader = self.getOption('default_bpm_ignore_header', fdu.getTag()).lower()
            if (dbpmlist.count(',') > 0):
                #comma separated list
                defaultBPMs = dbpmlist.split(',')
                removeEmpty(defaultBPMs)
                for j in range(len(defaultBPMs)):
                    defaultBPMs[j] = defaultBPMs[j].strip()
            elif (dbpmlist.endswith('.fit') or dbpmlist.endswith('.fits')):
                #FITS file given
                defaultBPMs.append(dbpmlist)
            elif (dbpmlist.endswith('.dat') or dbpmlist.endswith('.list') or dbpmlist.endswith('.txt')):
                #ASCII file list
                defaultBPMs = readFileIntoList(dbpmlist)
            for bpmfile in defaultBPMs:
                #Loop over list of bad pixel masks
                #badPixelMask = fatboyImage(bpmfile, log=self._log)
                badPixelMask = fatboySpecCalib(self._pname, "bad_pixel_mask", fdu, filename=bpmfile, log=self._log)
                #read header and initialize
                badPixelMask.readHeader()
                badPixelMask.initialize()
                if (ignoreHeader == "yes"):
                    #set filter, section to match.  One BPM will be appended below for each filter/section combination.
                    #BPM will then be found in #2 getMasterCalib above for other FDUs with same filter/section
                    badPixelMask.filter = fdu.filter
                    badPixelMask.grism = fdu.grism
                    badPixelMask.setProperty("specmode", fdu.getProperty("specmode"))
                if (badPixelMask.filter is not None and badPixelMask.filter != fdu.filter):
                    #does not match filter
                    continue
                if (badPixelMask.grism is not None and badPixelMask.grism != fdu.grism):
                    #does not match grism
                    continue
                if (badPixelMask.getProperty("specmode") is not None and badPixelMask.getProperty("specmode") != fdu.getProperty("specmode")):
                    #does not match specmode
                    continue
                if (fdu.section is not None):
                    #check section if applicable
                    section = -1
                    if (badPixelMask.hasHeaderValue('SECTION')):
                        section = badPixelMask.getHeaderValue('SECTION')
                    else:
                        idx = badPixelMask.getFilename().rfind('.fit')
                        if (badPixelMask.getFilename()[idx-2] == 'S' and isDigit(badPixelMask.getFilename()[idx-1])):
                            section = int(badPixelMask.getFilename()[idx-1])
                    if (section != fdu.section):
                        continue
                badPixelMask.setType("bad_pixel_mask")
                #Found matching master flat
                print("badPixelMaskSpecProcess::getCalibs> Using bad pixel mask "+badPixelMask.getFilename())
                self._log.writeLog(__name__, " Using bad pixel mask "+badPixelMask.getFilename())
                self._fdb.appendCalib(badPixelMask)
                calibs['badPixelMask'] = badPixelMask
                return calibs
        #4) Check bad_pixel_mask source before trying master_flat and master_dark.  Source could be master_flat or master_dark as well
        if (self.getOption('bad_pixel_mask_source', fdu.getTag()) is not None):
            source = self.getOption('bad_pixel_mask_source', fdu.getTag())
            sourceFDU = None
            #1) Check if bad_pixel_mask_source is an individual FITS file or file list
            if (source.count(',') > 0):
                #comma separated list
                source = source.split(',')
                removeEmpty(source)
                for j in range(len(source)):
                    source[j] = source[j].strip()
                sourceFDU = combineSourceFrames(source)
            elif (source.endswith('.fit') or source.endswith('.fits')):
                #FITS file given
                sourceFDU = fatboySpecCalib(self._pname, "bpm_source", fdu, filename=source, log=self._log)
            elif (source.endswith('.dat') or source.endswith('.list') or source.endswith('.txt')):
                #ASCII file list
                source = readFileIntoList(source)
                sourceFDU = combineSourceFrames(source)

            if (sourceFDU is None):
                #obstype given
                #1) Check for an already created frame of type source matching section and TAGGED for this object
                #No need to check specmode/filter/grism here as it should be TAGGED for this specific id
                sourceFDU = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype=source)
            if (sourceFDU is None):
                #2) Check for an already created frame of type source matching specmode/filter/grism
                sourceFDU = self._fdb.getMasterCalib(self._pname, obstype=source, filter=fdu.filter, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
            if (sourceFDU is None):
                #3) Check for an already created frame of type source matching exptime/nreads/section
                sourceFDU = self._fdb.getMasterCalib(self._pname, obstype=source, exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
            if (sourceFDU is not None):
                #Found source.  Create bad pixel mask.
                #convenience method
                badPixelMask = self.createBadPixelMask(fdu, sourceFDU, calibs)
                if (badPixelMask is None):
                    return calibs
                self._fdb.appendCalib(badPixelMask)
                calibs['badPixelMask'] = badPixelMask
                return calibs
        #5) Try master_flat.  Already should have obtained master_flat above if it exists
        if ('masterFlat' in calibs):
            #Create bad pixel mask
            #convenience method
            print("badPixelMaskSpecProcess::getCalibs> Creating bad pixel mask for filter: "+str(fdu.filter)+", grism: "+str(fdu.grism)+", specmode: "+str(fdu.getProperty("specmode"))+" using FLAT "+calibs['masterFlat'].getFilename())
            self._log.writeLog(__name__, " Creating bad pixel mask for filter: "+str(fdu.filter)+", grism: "+str(fdu.grism)+", specmode: "+str(fdu.getProperty("specmode"))+" using FLAT "+calibs['masterFlat'].getFilename())
            badPixelMask = self.createBadPixelMask(fdu, calibs['masterFlat'], calibs)
            if (badPixelMask is None):
                return calibs
            self._fdb.appendCalib(badPixelMask)
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #6) Try master_dark
        #Search for a TAGGED master_dark first
        masterDark = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype="master_dark", exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section)
        if (masterDark is None):
            masterDark = self._fdb.getMasterCalib(pname=None, obstype="master_dark", exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
        if (masterDark is not None):
            #Found source.  Create bad pixel mask.
            #convenience method
            print("badPixelMaskSpecProcess::getCalibs> Creating bad pixel mask for filter: "+str(fdu.filter)+" using DARK "+masterDark.getFilename())
            self._log.writeLog(__name__, " Creating bad pixel mask for filter: "+str(fdu.filter)+" using DARK "+masterDark.getFilename())
            badPixelMask = self.createBadPixelMask(fdu, masterDark, calibs)
            if (badPixelMask is None):
                return calibs
            self._fdb.appendCalib(badPixelMask)
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #7) Prompt user for source file
        print("List of master calibration frames, types, filters, exptimes, grisms, specmodes")
        masterCalibs = self._fdb.getMasterCalibs(obstype=fatboyDataUnit.FDU_TYPE_MASTER_CALIB)
        for mcalib in masterCalibs:
            print(mcalib.getFilename(), mcalib.getObsType(), mcalib.filter, mcalib.exptime, mcalib.grism, mcalib.getProperty("specmode"))
        tmp = input("Select a filename to use as a source to create bad pixel mask: ")
        calibfilename = tmp
        #Now find if input matches one of these filenames
        for mcalib in masterCalibs:
            if (mcalib.getFilename() == calibfilename):
                #Found matching master calib
                print("badPixelMaskSpecProcess::getCalibs> Using master calib "+mcalib.getFilename())
                self._log.writeLog(__name__, " Using master calib "+mcalib.getFilename())
                #create bad pixel mask
                #convenience method
                badPixelMask = self.createBadPixelMask(fdu, mcalib, calibs)
                if (badPixelMask is None):
                    return calibs
                self._fdb.appendCalib(badPixelMask)
                calibs['badPixelMask'] = badPixelMask
                return calibs
            return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_bad_pixel_mask', None)
        self._options.setdefault('default_bpm_ignore_header', 'no')
        self._options.setdefault('bad_pixel_mask_source', None)
        self._options.setdefault('behavior', 'mask')
        self._optioninfo.setdefault('behavior', 'mask | interpolate')
        self._options.setdefault('clipping_method','values')
        self._optioninfo.setdefault('clipping_method', 'values | sigma')
        self._options.setdefault('clipping_high','2.0')
        self._options.setdefault('clipping_low','0.5')
        self._options.setdefault('clipping_sigma','5')
        self._options.setdefault('column_reject', None)
        self._optioninfo.setdefault('column_reject', 'supports slicing, e.g. 320:384, 500, 752:768')
        self._options.setdefault('edge_reject','5')
        self._options.setdefault('interpolation_algorithm', 'median_neighbor')
        self._optioninfo.setdefault('interpolation_algorithm', 'Algorithm used to interpolate replacement value for bad pixels\nlinear_1d_x | linear_1d_y | linear_2d | median_neighbor |\nlinear_spline | cubic_spline | quintic_spline | weighted | biharmonic_2d')
        self._options.setdefault('interpolation_arg', None)
        self._optioninfo.setdefault('interpolation_arg', 'interpolation algorithm specific argument(s)')
        self._options.setdefault('interpolation_iterations', '1')
        self._optioninfo.setdefault('interpolation_iterations', 'Max number of iterations for interpolation algorithms')
        self._options.setdefault('normalize_source', 'yes')
        self._optioninfo.setdefault('normalize_source', 'ensure that bad pixel mask source is normalized\nbefore applying clipping_high/clipping_low')
        self._options.setdefault('radius_reject', '0')
        self._options.setdefault('row_reject', None)
        self._optioninfo.setdefault('row_reject', 'supports slicing, e.g. 320:384, 500, 752:768')
        self._options.setdefault('use_individual_slitlets', 'yes')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/badPixelMaskApplied", os.F_OK)):
            os.mkdir(outdir+"/badPixelMaskApplied",0o755)
        #Create output filename
        bafile = outdir+"/badPixelMaskApplied/ba_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(bafile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(bafile)
        if (not os.access(bafile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(bafile)
    #end writeOutput
