#print "__init__ main"
#from fatboyDatabase import fatboyDatabase
__version = "2.1.8"
__version__ = __version
__build = "7/23/24"
__build__ = __build
__threaded = False
__gpuenabled = True
__ctx = None

def setGPUEnabled(isEnabled):
    global __gpuenabled
    __gpuenabled = isEnabled

def setThreaded(isThreaded):
    global __threaded
    __threaded = isThreaded

def gpuEnabled():
    global __gpuenabled
    return __gpuenabled

def threaded():
    global __threaded
    return __threaded

def createGPUContext():
    global __ctx
    import os
    import pycuda.driver as drv
    drv.init()
    devnum = 0
    if ('CUDA_DEVICE' in os.environ):
        devnum = int(os.environ['CUDA_DEVICE'])
    dev = drv.Device(devnum)
    __ctx = dev.make_context()

def hasGPUContext():
    global __ctx
    if (ctx is None):
        return False
    return True

def setGPUContext(ctx):
    global __ctx
    __ctx = ctx

def popGPUContext():
    global __ctx
    if (__ctx is not None):
        __ctx.pop()
    __ctx = None
