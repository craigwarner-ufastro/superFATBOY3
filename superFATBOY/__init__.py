#print "__init__ main"
#from fatboyDatabase import fatboyDatabase
__version = "2.3.13"
__version__ = __version
__build = "6/14/25"
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
    # CuPy manages GPU contexts automatically.
    pass

def hasGPUContext():
    # CuPy manages GPU contexts automatically.
    return True

def setGPUContext(ctx):
    # CuPy manages GPU contexts automatically.
    pass

def popGPUContext():
    # CuPy manages GPU contexts automatically.
    pass
