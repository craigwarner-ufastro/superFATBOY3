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
    print("gpu_imcombine> WARNING: PyCUDA not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)

# frames = array of fits filenames or one ASCII file containing a list of
#       filenames or an array of FDUs
# method = 'median', 'mean', 'max', 'min', or 'sum'
# outfile = filename of output FITS file
# reject = rejection type if using method = 'mean'
# sigma = rejection threshold if using method = 'mean'
# weight = weighting option: 'uniform', 'exptime', 'mean', 'median',
#       'mean+sigma', or 'median+sigma'
# exptime = FITS keyword corresponding to exposure time

from .gpu_arraymedian import gpu_arraymedian
from .fatboyDataUnit import *
import sys
from numpy import *

blocks = 2048*4
block_size = 512

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction
MODE_FDU_DIFF_PAIRING = 5 #for CIRCE data twilight flats

def get_mod():
    mod = None
    if (hasCuda and superFATBOY.gpuEnabled()):
        mod = SourceModule("""
        __global__ void multArrVector_float(float *data, float *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] *= factor[i%stride];
        }

        __global__ void multArrVector_int(int *data, int *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] *= factor[i%stride];
        }

        __global__ void multArrVector_double(double *data, double *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] *= factor[i%stride];
        }

        __global__ void multArrVector_long(long *data, long *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] *= factor[i%stride];
        }

        __global__ void subArrVector_float(float *data, float *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] -= factor[i%stride];
        }

        __global__ void subArrVector_int(int *data, int *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] -= factor[i%stride];
        }

        __global__ void subArrVector_double(double *data, double *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] -= factor[i%stride];
        }

        __global__ void subArrVector_long(long *data, long *factor, int stride)
        {
          const int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
          //const int i = blockDim.x*blockIdx.x + threadIdx.x;
          data[i] -= factor[i%stride];
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

def space(x):
    return ' '*x

nx = 64

def imcombine(frames, outfile=None, expmask=None, method='median', reject='none', lsigma=3, hsigma=3, weight='none', lthreshold=None, hthreshold=None, scale='none', zero='none', nlow=0, nhigh=0, mclip='mean', qsfile=None, nonzero=False, even=True, inmask=None, expkey='EXP_TIME', niter=5, log=None, mef=0, outtype=float32, mode=None, returnHeader=False, dataTag=None):
    t = time.time()
    _verbosity = fatboyLog.NORMAL
    if (isinstance(frames, str) and os.access(frames, os.F_OK)):
        frames = readFileIntoList(frames)
    #if (len(frames) <= nx):
    #  nx = max(len(frames)-1, 1)
    nframes = len(frames)
    if (outfile is not None and os.access(outfile, os.F_OK)):
        os.unlink(outfile)
    if (expmask is not None and os.access(expmask, os.F_OK)):
        os.unlink(expmask)
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

    if (expmask is not None and reject == 'minmax'):
        print("Exposure masks not supported for minmax rejection!")
        write_fatboy_log(log, logtype, "Exposure masks not supported for minmax rejection!", __name__)
        expmask = None
    if (weight != 'none' and reject == 'minmax'):
        print("Weighting not supported for minmax rejection!")
        write_fatboy_log(log, logtype, "Weighting not supported for minmax rejection!", __name__)
        weight = 'none'

    if (mode == MODE_FITS):
        temp = pyfits.open(frames[0])
        data = temp[mef].data
        temp.close()
    elif (mode == MODE_RAW):
        data = frames[0]
    elif (mode == MODE_FDU):
        data = frames[0].getData()
    elif (mode == MODE_FDU_DIFFERENCE):
        data = frames[1].getData()-frames[0].getData()
        #2-1, 3-2, etc.
        nframes -= 1
    elif (mode == MODE_FDU_TAG):
        data = frames[0].getData(tag=dataTag)
    elif (mode == MODE_FDU_DIFF_PAIRING):
        sourceIndices = []
        pairIndices = []
        for i in range(len(frames)-1):
            for j in range(i+1, len(frames)):
                if (frames[i].ramp == frames[j].ramp and frames[i]._index != frames[j]._index):
                    #Found a pair
                    sourceIndices.append(i)
                    pairIndices.append(j)
                    break
        nframes = len(sourceIndices)
        data = frames[pairIndices[0]].getData()-frames[sourceIndices[0]].getData()

    if (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG or mode == MODE_FDU_DIFF_PAIRING):
        #Get MEF from FDU
        mef = frames[0]._mef

    imsize = float(data.size)
    totpix = nframes*imsize
    chunks = 1
    while (totpix > 2048.*2048.*nx):
        chunks *= 2
        totpix /= 2
    origsz = data.shape
    csize = origsz[0]//chunks
    totcols = origsz[0]
    out = empty(origsz, outtype)
    if (expmask is not None):
        exp = empty(origsz, outtype)
        exptimes = []
    w = ones(nframes, outtype)
    zpts = zeros(nframes, outtype)
    scl = ones(nframes, outtype)
    inp = empty((csize, origsz[1], nframes), outtype)
    mm = []
    del data

    #input mask
    if (isinstance(inmask, str)):
        temp = pyfits.open(inmask)
        inmask = temp[mef].data
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

    #Read quick start file
    hasqs = False
    qsDict = dict()
    if (qsfile is not None and os.access(qsfile, os.F_OK)):
        hasqs = True
        qslines = readFileIntoList(qsfile)
        for line in qslines:
            #Check for bad data
            qstokens = line.split(':')
            if (len(qstokens) != 2):
                #Should be filename options:median
                continue
            qsfilename = qstokens[0]
            qsmed = float(qstokens[1])
            #Add to dict
            qsDict[qsfilename] = qsmed

    if (_verbosity == fatboyLog.VERBOSE):
        print("Initialize: ",time.time()-t)
    tt = time.time()

    for l in range(nframes):
        if (mode == MODE_FITS):
            mm.append(pyfits.open(frames[l],memmap=1))
        if (expmask is not None):
            if (mode == MODE_FITS):
                if (expkey in mm[l][0].header):
                    exptimes.append(int(mm[l][0].header[expkey]))
                else:
                    exptimes.append(1)
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                exptimes.append(frames[l].exptime)
            elif (mode == MODE_FDU_DIFF_PAIRING):
                exptimes.append(frames[sourceIndices[l]].exptime)
            else:
                exptimes.append(1)
    if (expmask is not None):
        exptimes = outtype(exptimes)

    s = "GPU IMCOMBINE: method="+method
    if (zero != 'none'):
        s+=", zero="+zero
    if (scale != 'none'):
        s+=", scale="+scale
    if (weight != 'none'):
        s+=", weight="+weight
    s+="\n\treject="+reject
    if (reject == 'minmax'):
        s+="\tnlow="+str(nlow)+"\tnhigh="+str(nhigh)
    elif (reject == 'sigma' or reject == 'sigclip'):
        s+="\tlsigma="+str(lsigma)+"\thsigma="+str(hsigma)
    if (reject == 'sigclip'):
        s+="\tniter="+str(niter)
    s+="\n\tlthreshold="+str(lthreshold)+"\ththreshold="+str(hthreshold)
    if (nonzero):
        s+="\tnonzero=True"
    print(s)
    write_fatboy_log(log, logtype, s, __name__, printCaller=False, tabLevel=1)
    if (_verbosity == fatboyLog.VERBOSE):
        print("Open files: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()
    for j in range(chunks):
        if (j == chunks-1 and chunks*csize != totcols):
            endpos = totcols
            startpos = j*csize
            inp = empty((endpos-j*csize, origsz[1], nframes), outtype)
            csize = endpos-startpos
        else:
            endpos = (j+1)*csize
            startpos = j*csize
        blocks = inp.size//512
        blocky = 1
        while(blocks > 65535):
            blocky *= 2
            blocks //= 2
        if (j == 0):
            s = ""
            if (mode == MODE_FITS):
                s = "\tImages:"+space(len(frames[0])-6)
            elif (mode == MODE_FDU):
                s = "\tImages:"+space(len(frames[0].getFullId())-6)
            elif (mode == MODE_FDU_DIFFERENCE):
                s = "\tDifference Images:"+space(len(frames[1].getFullId())+len(frames[0].getFullId())-16)
            elif (mode == MODE_FDU_TAG):
                s = "\tImages:"+space(len(frames[j].getFullId()+":"+dataTag)-6)
            elif (mode == MODE_FDU_DIFF_PAIRING):
                s = "\tDifference Images:"+space(len(frames[pairIndices[0]].getFullId())+len(frames[sourceIndices[0]].getFullId())-16)
            if (zero != 'none'):
                s+="\tZero:"
            if (scale != 'none'):
                s+="\tScale:"
            if (weight != 'none'):
                s+="\tWeight:"
            if (s != ""):
                print(s)
                write_fatboy_log(log, logtype, s, __name__, printCaller=False, tabLevel=1)
        for l in range(nframes):
            if (mode == MODE_FDU_DIFFERENCE):
                #for twilight flats
                diffFrame = frames[l+1].getData()-frames[l].getData()
            elif (mode == MODE_FDU_DIFF_PAIRING):
                #for CIRCE twilight flats
                diffFrame = frames[pairIndices[l]].getData()-frames[sourceIndices[l]].getData()
            #Zeros, scaling, and weighting
            if (j == 0):
                if (inmask is not None):
                    #apply mask here
                    if (mode == MODE_FITS):
                        mm[l][mef].data*=inmask
                    elif (mode == MODE_FDU or mode == MODE_FDU_TAG):
                        frames[l].setMask(inmask)
                    elif (mode == MODE_RAW):
                        frames[l] *= inmask
                    elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                        diffFrame *= inmask
                #Calculate mean of image and reject if necessary
                if (zero.find('mean') != -1 or scale.find('mean') != -1 or weight.find('mean') != -1 or zero.find('sigma') !=-1 or scale.find('sigma') != -1 or weight.find('sigma') != -1):
                    qsstring = None
                    if (mode == MODE_FITS):
                        qsstring = frames[l]+' mean '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU):
                        qsstring = frames[l].getName()+' mean '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU_TAG):
                        qsstring = frames[l].getName()+":"+dataTag+' mean '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    #Check qsDict
                    if (hasqs and qsstring in qsDict):
                        immean = qsDict[qsstring]
                    elif (mode == MODE_FITS):
                        #rejection added to gpumean function
                        immean = gpumean(mm[l][mef].data, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = immean
                    elif (mode == MODE_FDU):
                        immean = gpumean(frames[l].getMaskedData(), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = immean
                    elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                        immean = gpumean(diffFrame, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                    elif (mode == MODE_FDU_TAG):
                        immean = gpumean(frames[l].getMaskedData(tag=dataTag), lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = immean
                    else:
                        immean = gpumean(frames[l], lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)

                #Calculate median
                if (zero.find('median') != -1 or scale.find('median') != -1 or weight.find('median') != -1):
                    qsstring = None
                    if (mode == MODE_FITS):
                        qsstring = frames[l]+' median '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU):
                        qsstring = frames[l].getName()+' median '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU_TAG):
                        qsstring = frames[l].getName()+":"+dataTag+' median '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    #Check qsDict
                    if (hasqs and qsstring in qsDict):
                        immed = qsDict[qsstring]
                    elif (mode == MODE_FITS):
                        byteswapped = False
                        if (not mm[l][mef].data.dtype.isnative):
                            #Byteswap
                            mm[l][mef].data = outtype(mm[l][mef].data)
                            byteswapped = True
                        #rejection added to median function
                        immed = gpu_arraymedian(mm[l][mef].data, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = immed
                        if (byteswapped):
                            mm[l] = pyfits.open(frames[l],memmap=1)
                    elif (mode == MODE_FDU):
                        immed = frames[l].getMaskedMedian() #use getMaskedMedian method to apply mask if applicable and find median
                        qsDict[qsstring] = immed
                    elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                        immed = gpu_arraymedian(diffFrame, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                    elif (mode == MODE_FDU_TAG):
                        immed = frames[l].getMaskedMedian(tag=dataTag) #use getMaskedMedian method to apply mask if applicable and find median
                        qsDict[qsstring] = immed
                    else:
                        immed = gpu_arraymedian(frames[l], lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)

                if (zero.find('sigma') != -1 or scale.find('sigma') != -1 or weight.find('sigma') != -1):
                    qsstring = None
                    if (mode == MODE_FITS):
                        qsstring = frames[l]+' sigma '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU):
                        qsstring = frames[l].getName()+' sigma '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU_TAG):
                        qsstring = frames[l].getName()+":"+dataTag+' sigma '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    #Check qsDict
                    if (hasqs and qsstring in qsDict):
                        imstd = qsDict[qsstring]
                    elif (mode == MODE_FITS):
                        #rejection added to gpustd function
                        imstd = gpustd(mm[l][mef].data, mean=immean, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = imstd
                    elif (mode == MODE_FDU):
                        imstd = gpustd(frames[l].getMaskedData(), mean=immean, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = imstd
                    elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                        imstd = gpustd(diffFrame, mean=immean, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                    elif (mode == MODE_FDU_TAG):
                        imstd = gpustd(frames[l].getMaskedData(tag=dataTag), mean=immean, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = imstd
                    else:
                        imstd = gpustd(frames[l], mean=immean, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)

                s = ""
                if (mode == MODE_FITS):
                    s = "\t"+frames[l]
                elif (mode == MODE_FDU):
                    s = "\t"+frames[l].getFullId()
                elif (mode == MODE_FDU_DIFFERENCE):
                    s = "\t"+frames[l+1].getFullId()+"-"+frames[l].getFullId()
                elif (mode == MODE_FDU_TAG):
                    s = "\t"+frames[l].getFullId()+":"+dataTag
                elif (mode == MODE_FDU_DIFF_PAIRING):
                    s = "\t"+frames[pairIndices[l]].getFullId()+"-"+frames[sourceIndices[l]].getFullId()
                #Zeros
                if (zero != 'none'):
                    if (zero == 'mean'):
                        zpts[l] = immean
                    elif (zero == 'median'):
                        zpts[l] = immed
                    elif (zero == 'mean+sigma'):
                        zpts[l] = immean+imstd
                    elif (zero == 'median+sigma'):
                        zpts[l] = immed+imstd
                    elif (mode == MODE_FITS and zero in mm[l][0].header):
                        zpts[l] = mm[l][0].header[zero]
                    elif ((mode == MODE_FDU or mode == MODE_FDU_TAG) and frames[l].hasHeaderValue(zero)):
                        zpts[l] = frames[l].getHeaderValue(zero)
                    s+="\t"+str(zpts[l])
                #Scaling
                if (scale != 'none'):
                    if (scale == 'mean'):
                        scl[l] = immean
                    elif (scale == 'median'):
                        scl[l] = immed
                    elif (scale == 'mean+sigma'):
                        scl[l] = immean+imstd
                    elif (scale == 'median+sigma'):
                        scl[l] = immed+imstd
                    elif (mode == MODE_FITS and scale in mm[l][0].header):
                        scl[l] = mm[l][0].header[scale]
                    elif ((mode == MODE_FDU or mode == MODE_FDU_TAG) and frames[l].hasHeaderValue(scale)):
                        scl[l] = frames[l].getHeaderValue(scale)
                    if (l != 0 and zpts[0] != 0):
                        scl[l] -= scl[0]*(zpts[l]-zpts[0])/zpts[0]
                    s+="\t"+str(scl[l])
                #Weights
                if (weight != 'none'):
                    if (weight == 'mean'):
                        w[l] = immean
                    if (weight == 'median'):
                        w[l] = immed
                    if (weight == 'mean+sigma'):
                        w[l] = immean+imstd
                    if (weight == 'median+sigma'):
                        w[l] = immed+imstd
                    elif (mode == MODE_FITS and weight in mm[l][0].header):
                        w[l] = mm[l][0].header[weight]
                    elif ((mode == MODE_FDU or mode == MODE_FDU_TAG) and frames[l].hasHeaderValue(weight)):
                        w[l] = frames[l].getHeaderValue(weight)
                    s+="\t"+str(w[l])

                if (s != ""):
                    print(s)
                    write_fatboy_log(log, logtype, s, __name__, printCaller=False, tabLevel=1)

            ttt = time.time()
            if (mode == MODE_FITS):
                inp[:,:,l] = mm[l][mef].data[startpos:endpos,:]
            elif (mode == MODE_RAW):
                inp[:,:,l] = frames[l][startpos:endpos,:]
            elif (mode == MODE_FDU):
                inp[:,:,l] = frames[l].getMaskedData()[startpos:endpos,:]
            elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                inp[:,:,l] = diffFrame[startpos:endpos, :]
            elif (mode == MODE_FDU_TAG):
                inp[:,:,l] = frames[l].getMaskedData(tag=dataTag)[startpos:endpos,:]
            #print "------------",time.time()-ttt

        if (_verbosity == fatboyLog.VERBOSE):
            print("Data copying: ", time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()
        if (j == 0):
            sys.stdout.write("\tUsing "+str(chunks)+" chunks: ")
        sys.stdout.write(str(j)+" ")
        sys.stdout.flush()
        #Actual combining: median and mean, 4 rejection types = 8 cases
        if (method == 'median'):
            sz = inp.shape
            nfiles = sz[2]
            nfint = int(nfiles)
            dothresh = False
            if (lthreshold is not None or hthreshold is not None or nonzero):
                dothresh = True
            #Apply zero, scale, weight
            if (zero != 'none'):
                fac = (zpts-zpts[0])
                subArrVector = mod.get_function("subArrVector_float")
                subArrVector(drv.InOut(inp), drv.In(fac), int32(nfint), grid=(blocks,blocky), block=(block_size,1,1))
            if (scale != 'none' or weight != 'none'):
                fac = ones(nfint, outtype)
                if (scale != 'none'):
                    fac *= (scl[0]/scl)
                if (weight != 'none'):
                    fac *= (w/w[0])
                multArrVector = mod.get_function("multArrVector_float")
                multArrVector(drv.InOut(inp), drv.In(fac), int32(nfint), grid=(blocks,blocky), block=(block_size,1,1))

            if (_verbosity == fatboyLog.VERBOSE):
                print("Scaling: ", time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            if (reject == 'none'):
                if (dothresh):
                    if (expmask is None):
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, even=even)
                    else:
                        #expmask calculated from median2d_w function
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], even=even)
                else:
                    out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", even=even)
                    if (expmask is not None):
                        exp[startpos:endpos,:] = exptimes.sum()
            elif (reject == 'minmax'):
                if (dothresh):
                    out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh, even=even)
                else:
                    out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nlow=nlow, nhigh=nhigh, even=even)
            elif (reject == 'sigma'):
                if (dothresh):
                    if (expmask is None):
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, even=even, sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], even=even, sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                else:
                    if (expmask is None):
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", even=even, sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", w=exptimes, wmap=exp[startpos:endpos,:], even=even, sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
            elif (reject == 'sigclip'):
                if (dothresh):
                    if (expmask is None):
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, even=even, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], even=even, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                else:
                    if (expmask is None):
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", even=even, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        out[startpos:endpos,:] = gpu_arraymedian(inp,axis="Z", w=exptimes, wmap=exp[startpos:endpos,:], even=even, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
        elif (method == 'mean'):
            sz = inp.shape
            nfiles = sz[2]
            nfint = int(nfiles)
            dothresh = False
            if (lthreshold is not None or hthreshold is not None or nonzero):
                dothresh = True
            #Apply zero, scale, weight
            if (zero != 'none'):
                fac = (zpts-zpts[0])
                subArrVector = mod.get_function("subArrVector_float")
                subArrVector(drv.InOut(inp), drv.In(fac), int32(nfint), grid=(blocks,blocky), block=(block_size,1,1))
            if (scale != 'none' or weight != 'none'):
                fac = ones(nfint, outtype)
                if (scale != 'none'):
                    fac *= (scl[0]/scl)
                if (weight != 'none'):
                    fac *= (w/w[0])
                multArrVector = mod.get_function("multArrVector_float")
                multArrVector(drv.InOut(inp), drv.In(fac), int32(nfint), grid=(blocks,blocky), block=(block_size,1,1))
            if (_verbosity == fatboyLog.VERBOSE):
                print("Scaling: ", time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            if (reject == 'none'):
                if (weight != 'none'):
                    if (dothresh):
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, weights=(w/w[0]))
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], weights=(w/w[0]))
                    else:
                        out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", weights=(w/w[0]))
                        if (expmask is not None):
                            exp[startpos:endpos,:] = exptimes.sum()
                else:
                    if (dothresh):
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:])
                    else:
                        out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z")
                        if (expmask is not None):
                            exp[startpos:endpos,:] = exptimes.sum()
            elif (reject == 'minmax'):
                if (dothresh):
                    out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, nlow=nlow, nhigh=nhigh)
                else:
                    out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nlow=nlow, nhigh=nhigh)
            elif (reject == 'sigma'):
                if (weight != 'none'):
                    if (dothresh):
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, weights=(w/w[0]), sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], weights=(w/w[0]), sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", weights=(w/w[0]), sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", w=exptimes, wmap=exp[startpos:endpos,:], weights=(w/w[0]), sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                else:
                    if (dothresh):
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", w=exptimes, wmap=exp[startpos:endpos,:], sigclip=True, niter=1, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
            elif (reject == 'sigclip'):
                if (weight != 'none'):
                    if (dothresh):
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, weights=(w/w[0]), sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], weights=(w/w[0]), sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", weights=(w/w[0]), sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", w=exptimes, wmap=exp[startpos:endpos,:], weights=(w/w[0]), sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                else:
                    if (dothresh):
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:], sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                    else:
                        if (expmask is None):
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
                        else:
                            #expmask calculated from mean2d_w function
                            out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", w=exptimes, wmap=exp[startpos:endpos,:], sigclip=True, niter=niter, lsigma=lsigma, hsigma=hsigma, mclip=mclip)
        elif (method == 'sum'):
            sz = inp.shape
            nfiles = sz[2]
            nfint = int(nfiles)
            dothresh = False
            if (lthreshold is not None or hthreshold is not None or nonzero):
                dothresh = True
            #Apply zero, scale, weight
            if (zero != 'none'):
                fac = (zpts-zpts[0])
                subArrVector = mod.get_function("subArrVector_float")
                subArrVector(drv.InOut(inp), drv.In(fac), int32(nfint), grid=(blocks,blocky), block=(block_size,1,1))
            if (scale != 'none' or weight != 'none'):
                fac = ones(nfint, outtype)
                if (scale != 'none'):
                    fac *= (scl[0]/scl)
                if (weight != 'none'):
                    fac *= (w/w[0])
                multArrVector = mod.get_function("multArrVector_float")
                multArrVector(drv.InOut(inp), drv.In(fac), int32(nfint), grid=(blocks,blocky), block=(block_size,1,1))

            if (_verbosity == fatboyLog.VERBOSE):
                print("Scaling: ", time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            if (reject == 'none'):
                if (dothresh):
                    if (expmask is None):
                        out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold)
                    else:
                        #expmask calculated from mean2d_w function
                        out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z", nonzero=nonzero, lthreshold=lthreshold, hthreshold=hthreshold, w=exptimes, wmap=exp[startpos:endpos,:])
                else:
                    out[startpos:endpos,:] = gpu_arraymean(inp,axis="Z")
                    if (expmask is not None):
                        exp[startpos:endpos,:] = exptimes.sum()
                out[startpos:endpos,:] *= nfiles

        #endif
        if (_verbosity == fatboyLog.VERBOSE):
            print("Chunk "+str(j)+": ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()
    #endfor
    del inp

    print("")
    #Write out quick start file
    if (hasqs and len(qsDict) > 0):
        f = open(qsfile, 'w')
        for qsstring in qsDict:
            f.write(qsstring+'\n')
        f.close()

    #Create new dict that can be used to update header
    newHeader = dict()
    newHeader['FILENAME'] = outfile
    if (outfile is not None and outfile.rfind('/') != -1):
        newHeader['FILENAME'] = outfile[outfile.rfind('/')+1:]
    newHeader['NCOMBINE'] = nframes
    newHeader['IMCBMETH'] = method
    newHeader['IMCBREJ'] = reject
    newHeader['IMCBSCL'] = scale
    newHeader['IMCBZERO'] = zero
    newHeader['IMCBWGHT'] = weight
    for j in range(nframes):
        if (mode == MODE_FITS):
            newHeader['IMCB'+leadZeros(3,j)] = frames[j]
        elif (mode == MODE_FDU):
            newHeader['IMCB'+leadZeros(3,j)] = frames[j].getFullId()
        elif (mode == MODE_FDU_DIFFERENCE):
            newHeader['IMCB'+leadZeros(3,j)] = frames[j+1].getFullId()+"-"+frames[j].getFullId()
        elif (mode == MODE_FDU_TAG):
            newHeader['IMCB'+leadZeros(3,j)] = frames[j].getFullId()+":"+dataTag
        elif (mode == MODE_FDU_DIFF_PAIRING):
            newHeader['IMCB'+leadZeros(3,j)] = frames[pairIndices[j]].getFullId()+"-"+frames[sourceIndices[j]].getFullId()

    if (outfile is not None):
        print("\tOutput file: "+outfile)
        write_fatboy_log(log, logtype, "\tOutput file: "+outfile, __name__, printCaller=False, tabLevel=1)
        header = None
        hdulist = None
        if (mode == MODE_FITS):
            #Open FITS file and use exisiting header and mef
            hdulist = pyfits.open(frames[0])
        elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG or mode == MODE_FDU_DIFF_PAIRING):
            header = frames[0]._header
        write_fits_file(outfile, out, dtype=outtype, header=header, headerExt=newHeader, fitsobj=hdulist, mef=mef, log=log)

    if (expmask is not None and expmask != 'return_expmask'):
        print("\tExposure mask: "+expmask)
        write_fatboy_log(log, logtype, "\tExposure mask: "+expmask, __name__, printCaller=False, tabLevel=1)
        header = None
        hdulist = None
        if (mode == MODE_FITS):
            #Open FITS file and use exisiting header and mef
            hdulist = pyfits.open(frames[0])
        elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG or mode == MODE_FDU_DIFF_PAIRING):
            header = frames[0]._header
        write_fits_file(expfile, exp, dtype="float32", header=header, headerExt=newHeader, fitsobj=hdulist, mef=mef, log=log)
        del exp
    if (_verbosity == fatboyLog.VERBOSE):
        print("Write data: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()
    print("\tCombined "+str(nfint)+" files.  Total time (s): "+str(time.time()-t))
    write_fatboy_log(log, logtype, "\tCombined "+str(nfint)+" files.  Total time (s): "+str(time.time()-t), __name__, printCaller=False, tabLevel=1)
    if (logtype == LOGTYPE_ASCII):
        log.close()
    #for j in range(len(mm)):
        #mm[j].close()
    if (expmask == 'return_expmask'):
        #return tuple
        return (out, exp, newHeader)
    elif (returnHeader):
        #return tuple
        return (out, newHeader)
    return out
