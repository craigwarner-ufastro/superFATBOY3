#!/usr/bin/python -u
import scipy
from scipy.optimize import leastsq
from .fatboyDataUnit import *

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction

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
            print("pysurfit> Could not find file "+input)
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
        print("pysurfit> Invalid input!  Exiting!")
        write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
        return None

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
    d2 = data.reshape(nx//bin,bin,ny//bin,bin,).sum(1).sum(2)/bin/bin
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tBinning: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Setup input mask if not given already
    if (inmask is None):
        inmask = ones(d2.shape).astype(bool)
    else:
        inmask = (inmask.reshape(nx//bin,bin,ny//bin,bin,).sum(1).sum(2)//bin//bin).astype(bool)
    if (_verbosity == fatboyLog.VERBOSE):
        print("Input mask: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    #Setup input arrays to calculate surface
    xin = arange(ny*nx//bin//bin) % (nx//bin)*bin+(bin//2-0.5)
    yin = arange(ny*nx//bin//bin) // (nx//bin)*bin+(bin//2-0.5)
    xin = xin.astype(float64)
    yin = yin.astype(float64)
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tInput Arrays: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    d2 = d2.ravel()
    inmask = inmask.ravel()

    #Setup initial guess for params
    terms = 0
    for j in range(order+2):
        terms+=j
    p = zeros(terms)
    p[0] = d2[inmask].mean()

    keep = ones(d2.shape).astype(bool)
    nkeep = (keep*inmask).sum()
    nkeepold = 0
    curriter = 0
    if (_verbosity == fatboyLog.VERBOSE):
        print("\tInitial Guesses: ",time.time()-tt,"; Total: ",time.time()-t)
    tt = time.time()

    while (curriter < niter and nkeep != nkeepold):
        tt = time.time()
        print("\tpysurfit: iteration "+str(curriter))
        write_fatboy_log(log, logtype, "pysurfit: iteration "+str(curriter), __name__, printCaller=False, tabLevel=1)
        b = keep*inmask
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
        fit = d2*0.
        fit += lsq[0][0]
        n = 1
        for j in range(1,order+1):
            for l in range(j+1):
                fit+=lsq[0][n]*xin**(j-l)*yin**l
                n+=1
        resid = d2b-fit[b]
        tempmean = resid.sum()/nkeep
        tempstddev = sqrt((resid*resid).sum()*(1./(nkeep-1))-tempmean*tempmean*nkeep/(nkeep-1))
        print("\t\tData - fit    mean: "+str(tempmean) + "   sigma: "+str(tempstddev))
        write_fatboy_log(log, logtype, "Data - fit    mean: "+str(tempmean) + "   sigma: "+str(tempstddev), __name__, printCaller=False, tabLevel=1)
        keep *= logical_and((d2-fit-tempmean)*(1./tempstddev) <= upper, (d2-fit-tempmean)*(1./tempstddev) >= -lower)
        curriter+=1
        nkeepold = nkeep
        nkeep = (keep*inmask).sum()
        p = lsq[0]
        if (_verbosity == fatboyLog.VERBOSE):
            print("\t\tCalc Resid: ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

    #reconstruct fit from original data size
    xin = arange(ny*nx).reshape(ny,nx) % nx + 0.
    yin = arange(ny*nx).reshape(ny,nx) // nx + 0.
    fit = data*0.
    fit += lsq[0][0]
    n = 1
    for j in range(1,order+1):
        for l in range(j+1):
            fit+=lsq[0][n]*xin**(j-l)*yin**l
            n+=1
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

    print("pysurfit: Total Time = "+str(time.time()-t)+" s")
    write_fatboy_log(log, logtype, "Total Time = "+str(time.time()-t)+" s", __name__, printCaller=False, tabLevel=1)
    return fit

def pysurfaceResiduals(p, x, y, out, order):
    f = x*0.
    f+=p[0]
    if (order >= 1):
        f+=p[1]*x
        f+=p[2]*y
    n = 3
    for j in range(2,order+1):
        for l in range(j+1):
            f+=p[n]*x**(j-l)*y**l
            n+=1
    err = out - f
    return err
