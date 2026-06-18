#!/usr/bin/python -u
import superFATBOY
import time
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

## Create wrapper functions here for FFTs based on cupy
## Using cupy as standard for GPU acceleration

def gpu_fft2(x, return_gpuarray=True):
    ## IMPORTANT - first need to syncrhonize context to this thread ##
    # With cupy, context management is handled by cupy
    if (isinstance(x, cp.ndarray)):
        cx = x
    else:
        cx = cp.array(x)
    #Execute FFT and get data to return in numpy np.array
    cfftx = cp.fft.fft2(cx)
    if (return_gpuarray):
        return cfftx
    fftx = cp.asnumpy(cfftx)
    return fftx

def gpu_ifft2(fftx, return_gpuarray=True):
    ## IMPORTANT - first need to syncrhonize context to this thread ##
    if (isinstance(fftx, cp.ndarray)):
        cfftx = fftx
    else:
        cfftx = cp.array(fftx)
    #Execute FFT and get data to return in numpy np.array
    cx = cp.fft.ifft2(cfftx)
    if (return_gpuarray):
        return cx
    x = cp.asnumpy(cx)
    return x

def gpu_real_fftshift(fftx):
    ## IMPORTANT - first need to syncrhonize context to this thread ##
    if (isinstance(fftx, cp.ndarray)):
        cfftx = fftx
    else:
        cfftx = cp.array(fftx)
    rifft = cp.asnumpy(cp.real(cp.fft.fftshift(cfftx)))
    return rifft

def cleanup_ffts():
    # Cupy handles cleanup automatically.
    pass

def gpu_correlate2d(x, y, verbose=False):
    t = time.time()
    nx = 2
    ny = 2
    while (nx < x.shape[1] or nx < y.shape[1]):
        nx = nx << 1
    while (ny < x.shape[0] or ny < y.shape[0]):
        ny = ny << 1
    newshape = (ny, nx)
    if (verbose):
        print("\tGC Initialize: ",time.time()-t)
    tt = time.time()

    if (newshape != x.shape):
        #pad to power of 2 shape for GPUs
        padx = np.zeros(newshape, x.dtype)
        padx[:x.shape[0],:x.shape[1]] = x
        x = padx
    fftx = gpu_fft2(x)

    if (verbose):
        print("\tGC Copy data and FFT x: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    if (newshape != y.shape):
        #pad to power of 2 shape for GPUs
        pady = np.zeros(newshape, y.dtype)
        pady[:y.shape[0],:y.shape[1]] = y
        y = pady
    ffty = gpu_fft2(y[::-1,::-1])
    if (verbose):
        print("\tGC Copy data and FFT inverse y: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    fftprod = fftx*ffty
    if (verbose):
        print("\tGC Multiply FFTs: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    ix = gpu_ifft2(fftprod)
    if (verbose):
        print("\tGC Inverse FFT: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    rifft = gpu_real_fftshift(ix)
    if (verbose):
        print("\tGC FFT shift: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time();

    cleanup_ffts()
    if (verbose):
        print("\tGC cleanup: ",time.time()-tt,"; Total: ",time.time()-t)
    else:
        print("\tgpu_correlate2d: total time(s) = ",time.time()-t)

    return np.array(rifft)
