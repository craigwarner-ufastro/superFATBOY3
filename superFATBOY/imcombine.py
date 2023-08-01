#!/usr/bin/python -u
# frames = array of fits filenames or one ASCII file containing a list of
#       filenames or an array of FDUs
# method = 'median' or 'mean'
# outfile = filename of output FITS file
# reject = rejection type if using method = 'mean'
# sigma = rejection threshold if using method = 'mean'
# weight = weighting option: 'uniform', 'exptime', 'mean', 'median',
#       'mean+sigma', or 'median+sigma'
# exptime = FITS keyword corresponding to exposure time

from .fatboyDataUnit import *
from .arraymedian import arraymedian
import sys
from numpy import *
from functools import reduce

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction
MODE_FDU_DIFF_PAIRING = 5 #for CIRCE data twilight flats

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
        #2-1, 3-1, etc.
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
    inp = zeros((nframes,csize,origsz[1]), outtype)
    mm = []

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
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE):
                exptimes.append(frames[j].exptime)
            elif (mode == MODE_FDU_DIFF_PAIRING):
                exptimes.append(frames[sourceIndices[l]].exptime)
            else:
                exptimes.append(1)
    if (expmask is not None):
        exptimes = outtype(exptimes)

    s = "IMCOMBINE: method="+method
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
            inp = empty((nframes, endpos-j*csize, origsz[1]), outtype)
            csize = endpos-startpos
        else:
            endpos = (j+1)*csize
            startpos = j*csize
        blocks = inp.size//512
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
                    useqs = False
                    qsstring = None
                    if (mode == MODE_FITS):
                        qsstring = frames[l]+' mean '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU or mode == MODE_FDU_TAG):
                        qsstring = frames[l].getName()+' mean '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    b = None
                    #Check quick start file
                    if (hasqs and qsstring is not None):
                        bqs = where(qskeys == qsstring)[0]
                        if (len(bqs) != 0):
                            immean = qsvals[bqs[0]]
                            useqs = True
                    if (not useqs):
                        if (mode == MODE_FITS):
                            data = mm[l][mef].data
                        elif (mode == MODE_FDU):
                            data = frames[l].getMaskedData()
                        elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                            data = diffFrame
                        elif (mode == MODE_FDU_TAG):
                            data = frames[l].getMaskedData(tag=dataTag)
                        else:
                            data = frames[l]
                        if (lthreshold is not None and hthreshold is not None):
                            b = logical_and(data >= lthreshold, data <= hthreshold)
                            if (nonzero):
                                b *= data != 0
                            nb = b.sum()
                            immean = ((data+0.)*b).sum()/nb
                        elif (lthreshold is not None):
                            b = data >= lthreshold
                            if (nonzero):
                                b *= data != 0
                            nb = b.sum()
                            immean = ((data+0.)*b).sum()/nb
                        elif (hthreshold is not None):
                            b = data <= hthreshold
                            if (nonzero):
                                b *= data != 0
                            nb = b.sum()
                            immean = ((data+0.)*b).sum()/nb
                        elif (nonzero):
                            b = data != 0
                            nb = b.sum()
                            immean = ((data+0.)*b).sum()/nb
                        else:
                            immean = (data+0.).mean()
                        if (qsstring is not None):
                            qslist.append(qsstring+': '+str(immean))

                #Calculate median
                if (zero.find('median') != -1 or scale.find('median') != -1 or weight.find('median') != -1):
                    qsstring = None
                    if (mode == MODE_FITS):
                        qsstring = frames[l]+' median '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU or mode == MODE_FDU_TAG):
                        qsstring = frames[l].getName()+' median '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    #Check qsDict
                    if (hasqs and qsstring in qsDict):
                        immed = qsDict[qsstring]
                    elif (mode == MODE_FITS):
                        #rejection added to median function
                        immed = arraymedian(mm[l][mef].data, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                        qsDict[qsstring] = immed
                    elif (mode == MODE_FDU):
                        immed = frames[l].getMaskedMedian() #use getMaskedMedian method to apply mask if applicable and find median
                        qsDict[qsstring] = immed
                    elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                        immed = arraymedian(diffFrame, lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)
                    elif (mode == MODE_FDU_TAG):
                        immed = frames[l].getMaskedMedian(tag=dataTag) #use getMaskedMedian method to apply mask if applicable and find median
                        qsDict[qsstring] = immed
                    else:
                        immed = arraymedian(frames[l], lthreshold=lthreshold, hthreshold=hthreshold, nonzero=nonzero)

                if (zero.find('sigma') != -1 or scale.find('sigma') != -1 or weight.find('sigma') != -1):
                    useqs = False
                    qsstring = None
                    if (mode == MODE_FITS):
                        qsstring = frames[l]+' sigma '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    elif (mode == MODE_FDU or mode == MODE_FDU_TAG):
                        qsstring = frames[l].getName()+' sigma '+str(lthreshold)+' '+str(hthreshold)+' '+str(nonzero)
                    #Check quick start file
                    if (hasqs and qsstring is not None):
                        bqs = where(qskeys == qsstring)[0]
                        if (len(bqs) != 0):
                            imstd = qsvals[bqs[0]]
                            useqs = True
                    if (not useqs):
                        if (mode == MODE_FITS):
                            data = mm[l][mef].data
                        elif (mode == MODE_FDU):
                            data = frames[l].getMaskedData()
                        elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                            data = diffFrame
                        elif (mode == MODE_FDU_TAG):
                            data = frames[l].getMaskedData(tag=dataTag)
                        else:
                            data = frames[l]
                        if (lthreshold is not None and hthreshold is not None):
                            if (b is None):
                                b = logical_and(data >= lthreshold, data <= hthreshold)
                                if (nonzero):
                                    b *= data != 0
                                nb = b.sum()
                            imstd = sqrt((data*data*b+0.).sum()*1./(nb-1)-immean*immean*nb/(nb-1))
                        elif (lthreshold is not None):
                            if (b is None):
                                b = data >= lthreshold
                                if (nonzero):
                                    b *= data != 0
                                nb = b.sum()
                            imstd = sqrt((data*data*b+0.).sum()*1./(nb-1)-immean*immean*nb/(nb-1))
                        elif (hthreshold is not None):
                            if (b is None):
                                b = data <= hthreshold
                                if (nonzero):
                                    b *= data != 0
                                nb = b.sum()
                            imstd = sqrt((data*data*b+0.).sum()*1./(nb-1)-immean*immean*nb/(nb-1))
                        elif (nonzero):
                            if (b is None):
                                b = data != 0
                                nb = b.sum()
                            imstd = sqrt((data*data*b+0.).sum()*1./(nb-1)-immean*immean*nb/(nb-1))
                        else:
                            imstd = sqrt((data*data+0.).sum()/(imsize-1)-immean*immean*imsize/(imsize-1))
                        if (qsstring is not None):
                            qslist.append(qsstring+': '+str(imstd))

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
                inp[l,:,:] = mm[l][mef].data[startpos:endpos,:]
            elif (mode == MODE_RAW):
                inp[l,:,:] = frames[l][startpos:endpos,:]
            elif (mode == MODE_FDU):
                inp[l,:,:] = frames[l].getMaskedData()[startpos:endpos,:]
            elif (mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_DIFF_PAIRING):
                inp[l,:,:] = diffFrame[startpos:endpos, :]
            elif (mode == MODE_FDU_TAG):
                inp[l,:,:] = frames[l].getMaskedData(tag=dataTag)[startpos:endpos,:]
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
            nfiles = float(sz[0])
            nfint = int(nfiles)
            if (lthreshold is not None and hthreshold is not None):
                b = logical_and(inp <= hthreshold, inp >= lthreshold)
                dothresh = True
            elif (lthreshold is not None):
                b = inp >= lthreshold
                dothresh = True
            elif (hthreshold is not None):
                b = inp <= hthreshold
                dothresh = True
            elif (nonzero):
                b = True
                dothresh = True
            else:
                dothresh = False

            #Apply zero, scale, weight
            for l in range(nfint):
                if (zero != 'none'):
                    inp[l,:,:]-=(zpts[l]-zpts[0])
                if (scale != 'none'):
                    inp[l,:,:]*=(scl[0]/scl[l])
                if (weight != 'none'):
                    inp[l,:,:]*=(w[l]/w[0])

            if (reject == 'none'):
                if (dothresh):
                    inp*=b
                    out[startpos:endpos,:] = arraymedian(inp, axis="Y", nonzero=True)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
                else:
                    out[startpos:endpos,:] = arraymedian(inp,axis="Y")
                    if (expmask is not None):
                        exp[startpos:endpos,:] = exptimes.sum()
            elif (reject == 'minmax'):
                if (dothresh):
                    inp*=b
                    out[startpos:endpos,:] = arraymedian(inp, axis="Y", nonzero=True, nlow=nlow, nhigh=nhigh)
                else:
                    out[startpos:endpos,:] = arraymedian(inp, axis="Y", nlow=nlow, nhigh=nhigh)
            elif (reject == 'sigma'):
                if (dothresh):
                    if (nonzero):
                        b *= inp != 0
                    tmask = reduce(add,b,0)
                    tmask[tmask == 0] = 1
                    inp *= b
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y",nonzero=True)
                    else:
                        avg = reduce(add,inp)*(1./tmask)
                    #nm1 = n-1
                    nm1 = tmask-1
                    nm1[nm1 == 0] = 1
                    sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*tmask/nm1)
                    lower = avg-lsigma*sd
                    upper = avg+hsigma*sd
                    out[startpos:endpos,:] = arraymedian(inp,axis="Y", lthreshold=lower, hthreshold=upper, nonzero=True)
                    if (expmask is not None):
                        b *= logical_and(inp <= upper, inp >= lower)
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
                else:
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y")
                    else:
                        avg = reduce(add,inp)*(1./nfiles)
                    sd = sqrt(reduce(add,inp*inp)*(1./(nfiles-1))-avg*avg*nfiles/(nfiles-1))
                    lower = avg-lsigma*sd
                    upper = avg+hsigma*sd
                    out[startpos:endpos,:] = arraymedian(inp,axis="Y", lthreshold=lower, hthreshold=upper)
                    if (expmask is not None):
                        b = logical_and(inp <= upper, inp >= lower)
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
            elif (reject == 'sigclip'):
                if (not dothresh):
                    avg = reduce(add,inp)*(1./nfiles)
                    sd = sqrt(reduce(add,inp*inp)*(1./(nfiles-1))-avg*avg*nfiles/(nfiles-1))
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y")
                    keep = logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                    n = reduce(add, keep, 0)
                    inp*=keep
                    #nm1 = n-1 for sd purposes
                    nm1 = n-1
                    n[n==0]=1
                    nm1[nm1 < 1] = 1
                    nold = zeros((sz[1],sz[2]))+nfiles
                    c = 1
                    while ((n != nold).max() == 1 and c < niter):
                        b = n != nold
                        if ((b+0).sum() < .2*b.size):
                            inpb = inp[:,b]
                            nb = n[b]
                            nm1b = maximum(nb-1,1)
                            avgb = reduce(add,inpb)*(1./nb)
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sdb = sqrt(reduce(add,inpb*inpb)*(1./nm1b)-avgb*avgb*nb/nm1b+1.e-6)
                            if (mclip == 'median'):
                                avgb = arraymedian(inpb,axis="Y",nonzero=True)
                            keepb = logical_and(inpb >= -lsigma*sdb+avgb, inpb <= sdb*hsigma+avgb)
                            inp[:,b]*=keepb
                            n[b] = reduce(add, keepb,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                        else:
                            avg = reduce(add,inp)/n
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*n/nm1+1.e-6)
                            if (mclip == 'median'):
                                avg = arraymedian(inp,axis="Y",nonzero=True)
                            keep = logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                            inp*=keep
                            n = reduce(add, keep,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                    if (mclip == 'median'):
                        out[startpos:endpos,:] = avg
                    else:
                        out[startpos:endpos,:] = arraymedian(inp, axis="Y", nonzero=True)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += keep[l,:,:]*exptimes[l]
                    keep = None
                else:
                    if (nonzero):
                        b *= inp != 0
                    inp*=b
                    tmask = reduce(add,b,0)
                    tmask[tmask == 0] = 1
                    avg = reduce(add,inp)*(1./tmask)
                    #nm1 = n-1
                    nm1 = tmask-1
                    nm1[nm1 == 0] = 1
                    sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*tmask/nm1)
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y",nonzero=True)
                    keep = b*logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                    n = reduce(add, keep, 0)
                    inp*=keep
                    #nm1 = n-1 for sd purposes
                    nm1 = n-1
                    n[n==0]=1
                    nm1[nm1 < 1] = 1
                    nold = zeros((sz[1],sz[2]))+nfiles
                    c = 1
                    while ((n != nold).max() == 1 and c < niter):
                        b = n != nold
                        if ((b+0).sum() < .15*b.size):
                            inpb = inp[:,b]
                            nb = n[b]
                            nm1b = maximum(nb-1,1)
                            avgb = reduce(add,inpb)*(1./nb)
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sdb = sqrt(reduce(add,inpb*inpb)*(1./nm1b)-avgb*avgb*nb/nm1b+1.e-6)
                            if (mclip == 'median'):
                                avgb = arraymedian(inpb,axis="Y",nonzero=True)
                            keepb = keep[:,b]*logical_and(inpb >= -lsigma*sdb+avgb, inpb <= sdb*hsigma+avgb)
                            inp[:,b]*=keepb
                            n[b] = reduce(add, keepb,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                        else:
                            avg = reduce(add,inp)/n
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*n/nm1+1.e-6)
                            if (mclip == 'median'):
                                avg = arraymedian(inp,axis="Y",nonzero=True)
                            keep *= logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                            inp*=keep
                            n = reduce(add, keep,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                    if (mclip == 'median'):
                        out[startpos:endpos,:] = avg
                    else:
                        out[startpos:endpos,:] = arraymedian(inp, axis="Y", nonzero=True)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += keep[l,:,:]*exptimes[l]
                    keep = None
        elif (method == 'mean'):
            sz = inp.shape
            nfiles = float(sz[0])
            nfint = int(nfiles)
            if (lthreshold is not None and hthreshold is not None):
                b = logical_and(inp <= hthreshold, inp >= lthreshold)
                dothresh = True
            elif (lthreshold is not None):
                b = inp >= lthreshold
                dothresh = True
            elif (hthreshold is not None):
                b = inp <= hthreshold
                dothresh = True
            elif (nonzero):
                b = True
                dothresh = True
            else:
                dothresh = False

            #Apply zero, scale, weight
            for l in range(nfint):
                if (zero != 'none'):
                    inp[l,:,:]-=(zpts[l]-zpts[0])
                if (scale != 'none'):
                    inp[l,:,:]*=(scl[0]/scl[l])
                if (weight != 'none'):
                    inp[l,:,:]*=(w[l]/w[0])
            #Correct for weighting
            if (weight != 'none'):
                nfiles/=(reduce(add,w)/(nfiles*w[0]))

            if (reject == 'none'):
                if (dothresh):
                    if (nonzero):
                        b *= inp != 0
                    tmask = reduce(add,b,0)
                    tmask[tmask == 0] = 1
                    out[startpos:endpos,:] = reduce(add,inp*b)*(1./tmask)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
                else:
                    out[startpos:endpos,:] = reduce(add,inp)*(1./nfiles)
                    if (expmask is not None):
                        exp[startpos:endpos,:] = exptimes.sum()
            elif (reject == 'minmax'):
                if (dothresh):
                    if (nonzero):
                        b *= inp != 0
                    dmin = inp.min()
                    inp[b == False] = dmin-2
                    inp.sort(0)
                    inp = reshape(inp, (nfint, csize*sz[2]))
                    #Calculate keep mask on sorted data
                    b = inp >= dmin-1
                    inp*=b
                    tmask = reduce(add,b,0)
                    nlowmask = ((nlow+0.)/nfiles*tmask).astype("int8")
                    nhighmask = ((nhigh+0.)/nfiles*tmask).astype("int8")
                    i = arange(csize*sz[2])
                    #tmask = number of valid points
                    #zero out adjusted nlow points
                    while (nlowmask.max() > 0):
                        b = nlowmask > 0
                        inp[(nfint-tmask)[b],i[b]] = 0
                        nlowmask -= 1*b
                        tmask -= 1*b
                    #zero out adjusted nhigh points
                    ihigh = 0
                    while (nhighmask.max() > 0):
                        ihigh += 1
                        b = nhighmask > 0
                        inp[nfint-ihigh, i[b]] = 0
                        nhighmask -= 1*b
                        tmask -= 1*b
                    tmask[tmask == 0] = 1
                    inp = reshape(inp, (nfint, csize, sz[2]))
                    tmask = reshape(tmask, (csize,sz[2]))
                    out[startpos:endpos,:] = reduce(add,inp)*(1./tmask)
                else:
                    inp.sort(0)
                    npts = nfint-nlow-nhigh
                    out[startpos:endpos,:] = reduce(add,inp[nlow:nfint-nhigh,:,:])*(1./npts)
            elif (reject == 'sigma'):
                if (not dothresh):
                    avg = reduce(add,inp)*(1./nfiles)
                    sd = sqrt(reduce(add,inp*inp)*(1./(nfiles-1))-avg*avg*nfiles/(nfiles-1))
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y")
                    b = logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                    inp*=b
                    n = reduce(add,b,0)
                    n[n==0] = 1
                    out[startpos:endpos,:] = reduce(add,inp)*(1./n)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
                else:
                    if (nonzero):
                        b *= inp != 0
                    inp*=b
                    tmask = reduce(add,b,0)
                    tmask[tmask == 0] = 1
                    avg = reduce(add,inp)*(1./tmask)
                    #nm1 = n-1
                    nm1 = tmask-1
                    nm1[nm1 == 0] = 1
                    sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*tmask/nm1)
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y",nonzero=True)
                    b *= logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                    tempsum = reduce(add, inp*b)
                    n = reduce(add,b,0)
                    n[n==0] = 1
                    out[startpos:endpos,:] = tempsum*(1./n)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
            elif (reject == 'sigclip'):
                if (not dothresh):
                    avg = reduce(add,inp)*(1./nfiles)
                    sd = sqrt(reduce(add,inp*inp)*(1./(nfiles-1))-avg*avg*nfiles/(nfiles-1))
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y")
                    keep = logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                    n = reduce(add, keep, 0)
                    inp*=keep
                    #nm1 = n-1 for sd purposes
                    nm1 = n-1
                    n[n==0]=1
                    nm1[nm1 < 1] = 1
                    nold = zeros((sz[1],sz[2]))+nfiles
                    c = 1
                    while ((n != nold).max() == 1 and c < niter):
                        b = n != nold
                        if ((b+0).sum() < .2*b.size):
                            inpb = inp[:,b]
                            nb = n[b]
                            nm1b = maximum(nb-1,1)
                            avgb = reduce(add,inpb)*(1./nb)
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sdb = sqrt(reduce(add,inpb*inpb)*(1./nm1b)-avgb*avgb*nb/nm1b+1.e-6)
                            if (mclip == 'median'):
                                avgb = arraymedian(inpb,axis="Y",nonzero=True)
                            keepb = logical_and(inpb >= -lsigma*sdb+avgb, inpb <= sdb*hsigma+avgb)
                            inp[:,b]*=keepb
                            n[b] = reduce(add, keepb,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                        else:
                            avg = reduce(add,inp)/n
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*n/nm1+1.e-6)
                            if (mclip == 'median'):
                                avg = arraymedian(inp,axis="Y",nonzero=True)
                            keep = logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                            inp*=keep
                            n = reduce(add, keep,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                    out[startpos:endpos,:] = reduce(add,inp)*(1./n)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += keep[l,:,:]*exptimes[l]
                    keep = None
                else:
                    if (nonzero):
                        b *= inp != 0
                    inp*=b
                    tmask = reduce(add,b,0)
                    tmask[tmask == 0] = 1
                    avg = reduce(add,inp)*(1./tmask)
                    #nm1 = n-1
                    nm1 = tmask-1
                    nm1[nm1 == 0] = 1
                    sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*tmask/nm1)
                    if (mclip == 'median'):
                        avg = arraymedian(inp,axis="Y",nonzero=True)
                    keep = b*logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                    n = reduce(add, keep, 0)
                    inp*=keep
                    #nm1 = n-1 for sd purposes
                    nm1 = n-1
                    n[n==0]=1
                    nm1[nm1 < 1] = 1
                    nold = zeros((sz[1],sz[2]))+nfiles
                    c = 1
                    while ((n != nold).max() == 1 and c < niter):
                        b = n != nold
                        if ((b+0).sum() < .15*b.size):
                            inpb = inp[:,b]
                            nb = n[b]
                            nm1b = maximum(nb-1,1)
                            avgb = reduce(add,inpb)*(1./nb)
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sdb = sqrt(reduce(add,inpb*inpb)*(1./nm1b)-avgb*avgb*nb/nm1b+1.e-6)
                            if (mclip == 'median'):
                                avgb = arraymedian(inpb,axis="Y",nonzero=True)
                            keepb = keep[:,b]*logical_and(inpb >= -lsigma*sdb+avgb, inpb <= sdb*hsigma+avgb)
                            inp[:,b]*=keepb
                            n[b] = reduce(add, keepb,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                        else:
                            avg = reduce(add,inp)/n
                            nold = n+0
                            #1.e-6 for floating point rounding errors
                            sd = sqrt(reduce(add,inp*inp)*(1./nm1)-avg*avg*n/nm1+1.e-6)
                            if (mclip == 'median'):
                                avg = arraymedian(inp,axis="Y",nonzero=True)
                            keep *= logical_and(inp >= -lsigma*sd+avg, inp <= sd*hsigma+avg)
                            inp*=keep
                            n = reduce(add, keep,0)
                            nm1 = n-1
                            n[n==0]=1
                            nm1[nm1 < 1] = 1
                            c+=1
                    out[startpos:endpos,:] = reduce(add,inp)*(1./n)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += keep[l,:,:]*exptimes[l]
                    keep = None
        elif (method == 'sum'):
            sz = inp.shape
            nfiles = float(sz[2])
            nfint = int(nfiles)
            if (lthreshold is not None and hthreshold is not None):
                b = logical_and(inp <= hthreshold, inp >= lthreshold)
                dothresh = True
            elif (lthreshold is not None):
                b = inp >= lthreshold
                dothresh = True
            elif (hthreshold is not None):
                b = inp <= hthreshold
                dothresh = True
            elif (nonzero):
                b = True
                dothresh = True
            else:
                dothresh = False

            #Apply zero, scale, weight
            for l in range(nfint):
                if (zero != 'none'):
                    inp[l,:,:]-=(zpts[l]-zpts[0])
                if (scale != 'none'):
                    inp[l,:,:]*=(scl[0]/scl[l])
                if (weight != 'none'):
                    inp[l,:,:]*=(w[l]/w[0])
            #Correct for weighting
            if (weight != 'none'):
                nfiles/=(reduce(add,w)/(nfiles*w[0]))

            if (reject == 'none'):
                if (dothresh):
                    if (nonzero):
                        b *= inp != 0
                    out[startpos:endpos,:] = reduce(add,inp*b)
                    if (expmask is not None):
                        for l in range(nfint):
                            exp[startpos:endpos,:] += b[l,:,:]*exptimes[l]
                else:
                    out[startpos:endpos,:] = reduce(add,inp)
                    if (expmask is not None):
                        exp[startpos:endpos,:] = exptimes.sum()
        #endif
        if (_verbosity == fatboyLog.VERBOSE):
            print("Chunk "+str(j)+": ",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()
    #endfor

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
