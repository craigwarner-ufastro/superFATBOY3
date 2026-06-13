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
import numpy as np
import cupy as cp
import os, time, math
from scipy.optimize import leastsq
from scipy.interpolate import interp1d, interp2d, griddata, bisplrep, bisplev

hasCuda = True
try:
    import superFATBOY
    if (not superFATBOY.gpuEnabled()):
        hasCuda = False
except Exception:
    print("badPixelMaskSpecProcess> Warning: CuPy not installed")
    hasCuda = False

block_size = 512

#### Bad pixel mask replacement algorithms ###

def bpm_replace_linterp_x(data, bpm=None, niter=1, arg=None):
    if (bpm is None):
        bpm = data == 0
    z = (bpm == 0) * (data == 0)
    data[z] = 1.e-6
    data[bpm] = 0
    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        b = np.where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            if (x - 1 > 0 and x + 1 < data.shape[1] and data[y, x - 1] != 0 and data[y, x + 1] != 0 and z[y, x - 1] == False and z[y, x + 1] == False):
                newdata[y, x] = (data[y, x - 1] + data[y, x + 1]) / 2.
            elif (x - 1 > 0 and x + 2 < data.shape[1] and data[y, x - 1] != 0 and data[y, x + 2] != 0 and z[y, x - 1] == False and z[y, x + 2] == False):
                newdata[y, x] = (data[y, x - 1] + data[y, x + 2]) / 2.
            elif (x - 2 > 0 and x + 1 < data.shape[1] and data[y, x - 2] != 0 and data[y, x + 1] != 0 and z[y, x - 2] == False and z[y, x + 1] == False):
                newdata[y, x] = (data[y, x - 2] + data[y, x + 1]) / 2.
            elif (x - 1 > 0 and data[y, x - 1] != 0 and z[y, x - 1] == False):
                newdata[y, x] = data[y, x - 1]
            elif (x + 1 < data.shape[1] and data[y, x + 1] != 0 and z[y, x + 1] == False):
                newdata[y, x] = data[y, x + 1]
            else:
                continue
            nreplace += 1
        data = newdata.copy()
    data[z] = 0
    return (data, nreplace)

def bpm_replace_linterp_y(data, bpm=None, niter=1, arg=None):
    if (bpm is None):
        bpm = data == 0
    z = (bpm == 0) * (data == 0)
    data[z] = 1.e-6
    data[bpm] = 0
    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        b = np.where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            if (y - 1 > 0 and y + 1 < data.shape[0] and data[y - 1, x] != 0 and data[y + 1, x] != 0 and z[y - 1, x] == False and z[y + 1, x] == False):
                newdata[y, x] = (data[y - 1, x] + data[y + 1, x]) / 2.
            elif (y - 1 > 0 and y + 2 < data.shape[0] and data[y - 1, x] != 0 and data[y + 2, x] != 0 and z[y - 1, x] == False and z[y + 2, x] == False):
                newdata[y, x] = (data[y - 1, x] + data[y + 2, x]) / 2.
            elif (y - 2 > 0 and y + 1 < data.shape[0] and data[y - 2, x] != 0 and data[y + 1, x] != 0 and z[y - 2, x] == False and z[y + 1, x] == False):
                newdata[y, x] = (data[y - 2, x] + data[y + 1, x]) / 2.
            elif (y - 1 > 0 and data[y - 1, x] != 0 and z[y - 1, x] == False):
                newdata[y, x] = data[y - 1, x]
            elif (y + 1 < data.shape[0] and data[y + 1, x] != 0 and z[y + 1, x] == False):
                newdata[y, x] = data[y + 1, x]
            else:
                continue
            nreplace += 1
        data = newdata.copy()
    data[z] = 0
    return (data, nreplace)

def bpm_replace_linterp_2d(data, bpm=None, niter=1, arg=None):
    if (bpm is None):
        bpm = data == 0
    z = (bpm == 0) * (data == 0)
    data[z] = 1.e-6
    data[bpm] = 0
    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        b = np.where(data == 0)
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            pts = []
            if (y - 1 > 0 and data[y - 1, x] != 0 and z[y - 1, x] == False):
                pts.append(data[y - 1, x])
            elif (y - 2 > 0 and data[y - 2, x] != 0 and z[y - 2, x] == False):
                pts.append(data[y - 2, x])
            if (y + 1 < data.shape[0] and data[y + 1, x] != 0 and z[y + 1, x] == False):
                pts.append(data[y + 1, x])
            elif (y + 2 < data.shape[0] and data[y + 2, x] != 0 and z[y + 2, x] == False):
                pts.append(data[y + 2, x])
            if (x - 1 > 0 and data[y, x - 1] != 0 and z[y, x - 1] == False):
                pts.append(data[y, x - 1])
            elif (x - 2 > 0 and data[y, x - 2] != 0 and z[y, x - 2] == False):
                pts.append(data[y, x - 2])
            if (x + 1 < data.shape[1] and data[y, x + 1] != 0 and z[y, x + 1] == False):
                pts.append(data[y, x + 1])
            if (x + 2 < data.shape[1] and data[y, x + 2] != 0 and z[y, x + 2] == False):
                pts.append(data[y, x + 2])
            npts = float(len(pts))
            if (npts > 0):
                newdata[y, x] = sum(pts) / npts
                nreplace += 1
        data = newdata.copy()
    data[z] = 0
    return (data, nreplace)

def bpm_replace_median_neighbor(data, bpm=None, niter=1, arg=None):
    npts = 1
    if (arg is not None):
        npts = arg
    if (bpm is None):
        bpm = data == 0
    z = (bpm == 0) * (data == 0)
    data[z] = 1.e-6
    data[bpm] = 0
    nreplace = 0
    for iter in range(niter):
        newdata = data.copy()
        b = np.where(data == 0)
        ny = data.shape[0]
        nx = data.shape[1]
        for j in range(len(b[0])):
            x = b[1][j]
            y = b[0][j]
            temp = []
            for k in range(-1 * npts, npts + 1):
                for r in range(-1 * npts, npts + 1):
                    if (k == 0 and r == 0):
                        continue
                    xc = k + x
                    yc = r + y
                    if (xc >= 0 and xc < nx and yc >= 0 and yc < ny):
                        if (data[yc, xc] != 0 and z[yc, xc] == False):
                            temp.append(data[yc, xc])
            if (len(temp) != 0):
                temp = np.array(temp)
                newdata[y, x] = gpu_arraymedian(temp, kernel=fatboyclib.median, even=True)
                nreplace += 1
        data = newdata.copy()
    data[z] = 0
    return (data, nreplace)

def bpm_replace_median_neighbor_gpu(data, bpm=None, niter=1, arg=None):
    if (bpm is None):
        bpm = data == 0
    data_gpu = cp.asarray(data, dtype=cp.float32)
    bpm_gpu = cp.asarray(bpm, dtype=bool)
    data_gpu[bpm_gpu] = 0
    fatboy_mod = get_fatboy_mod()
    badPixRemoval = fatboy_mod.get_function("medianNeighborReplaceBadPix")
    output_gpu = cp.empty(data_gpu.shape, dtype=cp.float32)
    out_bpm_gpu = cp.empty(data_gpu.shape, dtype=bool)
    rows = data_gpu.shape[0]
    cols = data_gpu.shape[1]
    ct = cp.zeros(1, dtype=np.int32)
    blocks = data_gpu.size // 512
    if (data_gpu.size % 512 != 0):
        blocks += 1
    nreplace = 0
    grid = (blocks, 1, 1)
    block = (block_size, 1, 1)
    for iter in range(niter):
        badPixRemoval(grid, block, (data_gpu, output_gpu, bpm_gpu, out_bpm_gpu, rows, cols, ct))
        data_gpu = output_gpu
        bpm_gpu = out_bpm_gpu
        nreplace += int(ct[0])
        ct = cp.zeros(1, dtype=np.int32)
    return (cp.asnumpy(data_gpu), nreplace)

# ... (the rest of the functions bpm_replace_scipy_interp2d, bpm_replace_scipy_griddata_cubic, etc. remain the same)
