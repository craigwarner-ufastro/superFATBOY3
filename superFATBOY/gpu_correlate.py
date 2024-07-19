#!/usr/bin/python -u
import superFATBOY
if (not superFATBOY.threaded()):
    #If not threaded mode, import autoinit.  Otherwise assume context exists.
    #Code will crash if in threaded mode and context does not exist.
    import pycuda.autoinit
import time
from numpy import *
import numpy.fft
## Create wrapper functions here for FFTs based on cupy/pyfft
## Try cupy first
mode = 0 # cupy = 0, pyfft = 1, error = -1
try:
    import cupy
    try:
        cp_ndarray = cupy.ndarray #new location
    except Exception as ex:
        cp_ndarray = cp_ndarray #old location
except Exception as ex:
    try:
        import pycuda.gpuarray as gpuarray
        from pyfft.cuda import Plan
        mode = 1
        print("gpu_correlate2d> Using pyfft instead of cupy.")
    except Exception as ex2:
        mode = -1
        print("gpu_correlate2d> ERROR: Could not find cupy or pyfft.  Will use numpy fft.")

if (mode == 0):
    def gpu_fft2(x, return_gpuarray=True):
        ## IMPORTANT - first need to syncrhonize context to this thread ##
        with cupy.cuda.Device(0):
            cupy.cuda.Device().synchronize()
            if (isinstance(x, cp_ndarray)):
                cx = x
            else:
                cx = cupy.array(x)
            #Execute FFT and get data to return in numpy array
            cfftx = cupy.fft.fft2(cx)
            if (return_gpuarray):
                return cfftx
            fftx = cupy.asnumpy(cfftx)
            return fftx

    def gpu_ifft2(fftx, return_gpuarray=True):
        ## IMPORTANT - first need to syncrhonize context to this thread ##
        with cupy.cuda.Device(0):
            cupy.cuda.Device().synchronize()
            if (isinstance(fftx, cp_ndarray)):
                cfftx = fftx
            else:
                cfftx = cupy.array(fftx)
            #Execute FFT and get data to return in numpy array
            cx = cupy.fft.ifft2(cfftx)
            if (return_gpuarray):
                return cx
            x = cupy.asnumpy(cx)
            return x

    def gpu_real_fftshift(fftx):
        ## IMPORTANT - first need to syncrhonize context to this thread ##
        with cupy.cuda.Device(0):
            cupy.cuda.Device().synchronize()
            if (isinstance(fftx, cp_ndarray)):
                cfftx = fftx
            else:
                cfftx = cupy.array(fftx)
            rifft = cupy.asnumpy(real(cupy.fft.fftshift(cfftx)))
            return rifft

    def cleanup_ffts():
        superFATBOY.popGPUContext()
        superFATBOY.createGPUContext()
        if (not superFATBOY.threaded()):
            superFATBOY.popGPUContext()
            import pycuda.autoinit
elif (mode == 1):
    def gpu_fft2(x, return_gpuarray=False):
        #Create plan with new power of 2 shape
        plan = Plan(x.shape)
        #Special case, no padding needed
        if (isinstance(x, pycuda.gpuarray.GPUArray)):
            gpu_data = x
        else:
            gpu_data = gpuarray.to_gpu(complex64(x))
        #Execute FFT and get data to return
        plan.execute(gpu_data)
        if (return_gpuarray):
            return gpu_data
        fftx = gpu_data.get()
        return fftx
    def gpu_ifft2(fftx, return_gpuarray=False):
        #Create plan with new power of 2 shape
        plan = Plan(fftx.shape)
        #Special case, no padding needed
        if (isinstance(fftx, pycuda.gpuarray.GPUArray) and not reverse):
            gpu_data = fftx
        else:
            gpu_data = gpuarray.to_gpu(complex64(fftx))
        #Execute inverse FFT and get data to return
        plan.execute(gpu_data, inverse=True)
        if (return_gpuarray):
            return gpu_data
        x = gpu_data.get()
        return x
    def gpu_real_fftshift(fftx):
        return real(numpy.fft.fftshift(fftx))
    def cleanup_ffts():
        pass
else:
    def gpu_fft2(x, return_gpuarray=False):
        #Using CPU here, no power of 2 advantage so do not pad
        return numpy.fft.fft2(x)
    def gpu_ifft2(x, return_gpuarray=False):
        #Using CPU here, no power of 2 advantage so do not pad
        return numpy.fft.ifft2(x)
    def gpu_real_fftshift(fftx):
        return real(numpy.fft.fftshift(fftx))
    def cleanup_ffts():
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

    if (newshape != x.shape and mode != -1):
        #pad to power of 2 shape for GPUs
        padx = zeros(newshape, x.dtype)
        padx[:x.shape[0],:x.shape[1]] = x
        x = padx
    fftx = gpu_fft2(x)

    if (verbose):
        print("\tGC Copy data and FFT x: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    if (newshape != y.shape and mode != -1):
        #pad to power of 2 shape for GPUs
        pady = zeros(newshape, y.dtype)
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

    return array(rifft)
