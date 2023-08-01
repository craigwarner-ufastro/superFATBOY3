hasCuda = True
try:
    import superFATBOY
    from .gpu_correlate import gpu_correlate2d
    from .gpu_arraymedian import *
except Exception:
    print("gpu_xregister> WARNING: PyCUDA not installed!")
    hasCuda = False
    superFATBOY.setGPUEnabled(False)

hasSep = True
try:
    import sep
except Exception:
    print("skySubtractProcess> Warning: sep not installed")
    hasSep = False

import scipy, os, time
from scipy.optimize import leastsq
from .fatboyLibs import *
from .fatboyLog import *
from .fatboyDataUnit import *

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction

METHOD_REGULAR = 0
METHOD_CONSTRAINED = 1
METHOD_SEP = 2
METHOD_SEP_CONSTRAINED = 3
METHOD_GUESSES = 4
METHOD_SEP_CENTROID = 5
METHOD_SEP_CENTROID_CONSTRAINED = 6

def xregister(frames, outfile=None, xcenter=-1, ycenter=-1, xboxsize=-1, yboxsize=-1, log=None, mef=0, gui=None, refframe=0, mode=None, dataTag=None, ra_keyword=None, dec_keyword=None, pixscale_keyword=None, rotpa_keyword=None, constrain_boxsize=256, constrain_guesses=None, median_filter2d=True, doMaskNegatives=False, doSmoothCorrelation=False, doFit2dGaussian=False, sepmask=None, sepDetectThresh=3, sepfwhm='a', method=METHOD_REGULAR):
    t = time.time()
    _verbosity = fatboyLog.NORMAL
    #Set booleans
    constrain = False
    useSep = False
    if (method == METHOD_CONSTRAINED or method == METHOD_SEP_CONSTRAINED or method == METHOD_GUESSES or method == METHOD_SEP_CENTROID_CONSTRAINED):
        constrain = True
    if (method == METHOD_SEP or method == METHOD_SEP_CONSTRAINED or method == METHOD_SEP_CENTROID or method == METHOD_SEP_CENTROID_CONSTRAINED):
        if (hasSep):
            useSep = True
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

    nframes = len(frames)
    #Find type
    if (mode is None):
        mode = MODE_FITS
        if (isinstance(frames, str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], ndarray)):
            mode = MODE_RAW
        elif (isinstance(frames[0], fatboyDataUnit)):
            mode = MODE_FDU

    if (mode == MODE_FITS):
        if (isinstance(frames, str)):
            if (os.access(frames, os.F_OK)):
                filelist = readFileIntoList(frames)
                nframes = len(filelist)
            else:
                print("gpu_xregister> Could not find file "+frames)
                write_fatboy_log(log, logtype, "Could not find file "+frames, __name__, messageType=fatboyLog.ERROR)
                return None
        else:
            filelist = frames
        #find refframe
        if (isinstance(refframe, str)):
            for j in range(len(filelist)):
                if (filelist[j].find(refframe) != -1):
                    refframe = j
                    print("gpu_xregister> Using "+filelist[j]+" as reference frame.")
                    write_fatboy_log(log, logtype, "Using "+filelist[j]+" as reference frame.", __name__)
                    break
            if (isinstance(refframe, str)):
                print("gpu_xregister> Could not find reference frame: "+refframe+"!  Using frame 0 = "+filelist[0])
                write_fatboy_log(log, logtype, "Could not find reference frame: "+refframe+"!  Using frame 0 = "+filelist[0], __name__, messageType=fatboyLog.WARNING)
                refframe = 0
    elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
        #find refframe
        if (isinstance(refframe, str)):
            for j in range(len(frames)):
                if (frames[j].getFullId().find(refframe) != -1):
                    refframe = j
                    print("gpu_xregister> Using "+frames[j].getFullId()+" as reference frame.")
                    write_fatboy_log(log, logtype, "Using "+frames[j].getFullId()+" as reference frame.", __name__)
                    break
            if (isinstance(refframe, str)):
                print("gpu_xregister> Could not find reference frame: "+refframe+"!  Using frame 0 = "+frames[0].getFullId())
                write_fatboy_log(log, logtype, "Could not find reference frame: "+refframe+"!  Using frame 0 = "+frames[0].getFullId(), __name__, messageType=fatboyLog.WARNING)
                refframe = 0

    #Look for constrain guess file
    if (method == METHOD_GUESSES):
        if (isinstance(constrain_guesses, list)):
            guesses = array(constrain_guesses)
        elif (isinstance(constrain_guesses, str) and os.access(constrain_guesses, os.F_OK)):
            guesses = loadtxt(constrain_guesses)
        else:
            print("gpu_xregister> Error: invalid constrain_guesses")
            write_fatboy_log(log, logtype, "invalid constrain_guesses", __name__, messageType=fatboyLog.ERROR)
            return None
        if (len(guesses) != nframes):
            print("gpu_xregister> Error: guesses file has length "+str(len(guesses))+" instead of length "+str(nframes))
            write_fatboy_log(log, logtype, "guesses file has length "+str(len(guesses))+" instead of length "+str(nframes), __name__, messageType=fatboyLog.ERROR)
            return None

    #Get reference frame
    if (mode == MODE_FITS):
        if (os.access(filelist[refframe], os.F_OK)):
            temp = pyfits.open(filelist[refframe])
            refData = temp[mef].data
            if (not refData.dtype.isnative):
                print("gpu_register> Byteswapping "+filelist[refframe])
                refData = float32(refData)
            refName = filelist[refframe]
            if (constrain and method != METHOD_GUESSES):
                refRA = getRADec(temp[0].header[ra_keyword])*15
                refDec = getRADec(temp[0].header[dec_keyword], dec=True)
                pixelScale = temp[0].header[pixscale_keyword]
                theta = temp[0].header[rotpa_keyword]
            temp.close()
        else:
            print("gpu_xregister> Could not find file "+filelist[refframe])
            write_fatboy_log(log, logtype, "Could not find file "+filelist[refframe], __name__, messageType=fatboyLog.ERROR)
            return None
    elif (mode == MODE_RAW):
        refData = frames[refframe]
        refName = "index "+str(refframe)
        constrain = False #not applicable
    elif (mode == MODE_FDU):
        refData = frames[refframe].getData()
        refName = frames[refframe].getFullId()
        if (constrain and method != METHOD_GUESSES):
            refRA = frames[refframe].ra
            refDec = frames[refframe].dec
            pixelScale = frames[refframe].getHeaderValue('pixscale_keyword')
            theta = frames[refframe].getHeaderValue('rotpa_keyword')
    elif (mode == MODE_FDU_DIFFERENCE):
        refData = frames[refframe+1].getData()-frames[refframe].getData()
        refName = frames[refframe+1].getFullId()+"-"+frames[refframe].getFullId()
        constrain = False #not applicable
        #2-1, 3-2, etc.
        nframes -= 1
    elif (mode == MODE_FDU_TAG):
        refData = frames[refframe].getData(tag=dataTag)
        refName = frames[refframe].getFullId()+":"+dataTag
        if (constrain and method != METHOD_GUESSES):
            refRA = frames[refframe].ra
            refDec = frames[refframe].dec
            pixelScale = frames[refframe].getHeaderValue('pixscale_keyword')
            theta = frames[refframe].getHeaderValue('rotpa_keyword')
    else:
        print("gpu_xregister> Invalid input!  Exiting!")
        write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
        return None

    shp = refData.shape

    if (xcenter == -1):
        xcenter = shp[1]//2
    if (ycenter == -1):
        ycenter = shp[0]//2
    if (xboxsize == -1):
        xboxsize = shp[1]
    if (yboxsize == -1):
        yboxsize = shp[0]

    print("Using ("+str(xboxsize)+", "+str(yboxsize)+") pixel wide box centered at ("+str(xcenter)+", "+str(ycenter)+").")
    write_fatboy_log(log, logtype, "Using ("+str(xboxsize)+", "+str(yboxsize)+") pixel wide box centered at ("+str(xcenter)+", "+str(ycenter)+").", __name__)

    x1 = xcenter-xboxsize//2
    x2 = xcenter+xboxsize//2
    y1 = ycenter-yboxsize//2
    y2 = ycenter+yboxsize//2
    if (x1 < 0):
        x1 = 0
    if (y1 < 0):
        y1 = 0
    if (x2 > shp[1]):
        x2 = shp[1]
    if (y2 > shp[0]):
        y2 = shp[0]

    xshifts = [0]
    yshifts = [0]
    ccmax = [0]

    #Create dummy image if using sep
    if (useSep):
        halfPeak = refData.max()/2
        if (mode == MODE_FDU):
            mask=frames[0].getBadPixelMask().getData().astype(bool)
        elif (sepmask is not None):
            mask = sepmask
        else:
            mask = zeros(refData.shape, bool)
        bkg = sep.Background(refData)
        print("\tsep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms))
        write_fatboy_log(log, logtype, "sep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms), __name__)
        thresh = sepDetectThresh*bkg.globalrms #Default = 3
        #subtract background from data
        bkg.subfrom(refData)
        #extract objects
        objects = sep.extract(refData, thresh, minarea=9)
        print("\tsep extracted "+str(len(objects))+" objects using thresh = "+str(sepDetectThresh)+"*rms; fwhm = "+str(sepfwhm))
        write_fatboy_log(log, logtype, "sep extracted "+str(len(objects))+" objects using thresh = "+str(sepDetectThresh)+"*rms; fwhm = "+str(sepfwhm), __name__)
        if (method == METHOD_SEP_CENTROID or method == METHOD_SEP_CENTROID_CONSTRAINED):
            #Save refobjects for these methods
            refObjects = array([objects['x'], objects['y']])
        #Create new blank image
        refData = zeros(refData.shape, float32)
        p = zeros(5)
        p[0] = halfPeak
        p[4] = bkg.globalback
        for j in range(len(objects['x'])):
            #Fill in image with 2-D gaussians
            p[1] = objects['x'][j]
            p[2] = objects['y'][j]
            if (sepfwhm == 'a'):
                p[3] = objects['a'][j]
            else:
                p[3] = float(sepfwhm)
            xmin = max(int(p[1])-30, 0)
            xmax = min(int(p[1])+31, refData.shape[1])
            ymin = max(int(p[2])-30, 0)
            ymax = min(int(p[2])+31, refData.shape[0])
            xin = arange((xmax-xmin)*(ymax-ymin)).reshape((ymax-ymin,xmax-xmin)) % (xmax-xmin) + xmin
            yin = arange((xmax-xmin)*(ymax-ymin)).reshape((ymax-ymin,xmax-xmin)) // (xmax-xmin) + ymin
            refData[ymin:ymax, xmin:xmax] += gaussFunction2d(p, xin, yin)

    #Get rid of negative datapoints
    if (doMaskNegatives):
        refData = maskNegatives(refData)

    if (outfile is not None):
        f = open(outfile,'w')

    if (_verbosity == fatboyLog.VERBOSE):
        print("Initialize: ",time.time()-t)
    tt = time.time()

    for j in range(nframes):
        if (j == refframe):
            continue
        #Get data
        if (mode == MODE_FITS):
            if (os.access(filelist[j], os.F_OK)):
                temp = pyfits.open(filelist[j])
                currData = temp[mef].data
                if (not currData.dtype.isnative):
                    print("gpu_register> Byteswapping "+filelist[j])
                    currData = float32(currData)
                currName = filelist[j]
                if (constrain and method != METHOD_GUESSES):
                    currRA = getRADec(temp[0].header[ra_keyword])*15
                    currDec = getRADec(temp[0].header[dec_keyword], dec=True)
                temp.close()
            else:
                print("gpu_xregister> Could not find file "+filelist[j])
                write_fatboy_log(log, logtype, "Could not find file "+filelist[j], __name__)
                return None
        elif (mode == MODE_RAW):
            currData = frames[j]
            currName = "index "+str(j)
        elif (mode == MODE_FDU):
            currData = frames[j].getData()
            currName = frames[j].getFullId()
            if (constrain and method != METHOD_GUESSES):
                currRA = frames[j].ra
                currDec = frames[j].dec
        elif (mode == MODE_FDU_DIFFERENCE):
            currData = frames[j+1].getData()-frames[j].getData()
            currName = frame[j+1].getFullId()+"-"+frame[j].getFullId()
            #2-1, 3-2, etc.
        elif (mode == MODE_FDU_TAG):
            currData = frames[j].getData(tag=dataTag)
            currName = frames[j].getFullId()+":"+dataTag
            if (constrain and method != METHOD_GUESSES):
                currRA = frames[j].ra
                currDec = frames[j].dec
        else:
            print("gpu_xregister> Invalid input!  Exiting!")
            write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
            return None

        #Create dummy image if using sep
        if (useSep):
            halfPeak = refData.max()/2
            if (mode == MODE_FDU):
                mask=frames[0].getBadPixelMask().getData().astype(bool)
            else:
                mask = zeros(currData.shape, bool)
            bkg = sep.Background(currData)
            print("\tsep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms))
            write_fatboy_log(log, logtype, "sep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms), __name__)
            thresh = sepDetectThresh*bkg.globalrms #Default = 3
            #subtract background from data
            bkg.subfrom(currData)
            #extract objects
            objects = sep.extract(currData, thresh, minarea=9)
            print("\tsep extracted "+str(len(objects))+" objects using thresh = "+str(sepDetectThresh)+"*rms; fwhm = "+str(sepfwhm))
            write_fatboy_log(log, logtype, "sep extracted "+str(len(objects))+" objects using thresh = "+str(sepDetectThresh)+"*rms; fwhm = "+str(sepfwhm), __name__)
            #Create new blank image
            currData = zeros(currData.shape, float32)
            p = zeros(5)
            p[0] = halfPeak
            p[4] = bkg.globalback
            for l in range(len(objects['x'])):
                #Fill in image with 2-D gaussians
                p[1] = objects['x'][l]
                p[2] = objects['y'][l]
                if (sepfwhm == 'a'):
                    p[3] = objects['a'][l]
                else:
                    p[3] = float(sepfwhm)
                xmin = max(int(p[1])-30, 0)
                xmax = min(int(p[1])+31, currData.shape[1])
                ymin = max(int(p[2])-30, 0)
                ymax = min(int(p[2])+31, currData.shape[0])
                xin = arange((xmax-xmin)*(ymax-ymin)).reshape((ymax-ymin,xmax-xmin)) % (xmax-xmin) + xmin
                yin = arange((xmax-xmin)*(ymax-ymin)).reshape((ymax-ymin,xmax-xmin)) // (xmax-xmin) + ymin
                currData[ymin:ymax, xmin:xmax] += gaussFunction2d(p, xin, yin)

        #Mask negatives
        if (doMaskNegatives):
            currData = maskNegatives(currData)
        if (_verbosity == fatboyLog.VERBOSE):
            print("Read data, mask negatives "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        #Do 2d cross correlation
        ccor = gpu_correlate2d(refData[y1:y2,x1:x2], currData[y1:y2,x1:x2])
        if (_verbosity == fatboyLog.VERBOSE):
            print("Correlate2d "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        if (constrain and method != METHOD_GUESSES):
            diffRA = int((currRA-refRA)*math.cos(refDec*math.pi/180)*3600/pixelScale)
            diffDec = int((currDec-refDec)*3600/pixelScale)
            xguess = diffRA
            yguess = diffDec
        elif (constrain and method == METHOD_GUESSES):
            xguess = guesses[j][0]
            yguess = guesses[j][1]
        if (constrain):
            print("gpu_xregister> Initial guess from "+refName+" to "+currName+" is ("+str(xguess)+", "+str(yguess)+").")
            write_fatboy_log(log, logtype, "Initial guess from "+refName+" to "+currName+" is ("+str(xguess)+", "+str(yguess)+").", __name__)
            cby1 = ccor.shape[0]//2+yguess-constrain_boxsize//2
            cby2 = ccor.shape[0]//2+yguess+constrain_boxsize//2
            cbx1 = ccor.shape[1]//2+xguess-constrain_boxsize//2
            cbx2 = ccor.shape[1]//2+xguess+constrain_boxsize//2
            if (cby1 < 0):
                yguess -= cby1
                cby1 = 0
                cby2 = min(constrain_boxsize, ccor.shape[0])
            if (cbx1 < 0):
                xguess -= cbx1
                cbx1 = 0
                cbx2 = min(constrain_boxsize, ccor.shape[1])
            if (cby2 > ccor.shape[0]):
                yguess -= (cby2-ccor.shape[0])
                cby2 = ccor.shape[0]
                cby1 = max(0, cby2-constrain_boxsize)
            if (cbx2 > ccor.shape[1]):
                xguess -= (cbx2-ccor.shape[1])
                cbx2 = ccor.shape[1]
                cbx1 = max(0, cbx2-constrain_boxsize)
            ccor = ccor[cby1:cby2, cbx1:cbx2].copy()
            ccor = ascontiguousarray(ccor)
            #Use .copy() to make sure data is contiguous for GPU

        if (median_filter2d):
            #Median filter resulting matrix
            boxsize = min(25, min(ccor.shape)//2+1)
            ccor = gpumedianfilter2d(ccor, boxsize=boxsize)
            if (_verbosity == fatboyLog.VERBOSE):
                print("MedianFilter2d "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
            write_fatboy_log(log, logtype, "median filtering cross correlation result", __name__)
            tt = time.time()

        cmax = ccor.max()
        b = whereEqual(ccor, cmax)
        if (doSmoothCorrelation):
            ccor_smooth = smooth2d(ccor, 3, 1)
            cmax = ccor_smooth.max()
            b = whereEqual(ccor_smooth, cmax)
        ccmax.append(cmax)
        if (_verbosity == fatboyLog.VERBOSE):
            print("Where "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        if (doFit2dGaussian):
            #fit one 2-d Gaussian
            p = zeros(5, float64)
            p[0] = ccor.max()
            p[1] = b[1][0]
            p[2] = b[0][0]
            p[3] = 3
            p[4] = 0
            out = ccor.ravel()
            xin = (arange(out.size) % ccor.shape[1]).astype(float64)
            yin = (arange(out.size) // ccor.shape[0]).astype(float64)
            lsq = leastsq(gaussResiduals2d, p, args=(xin, yin, out))
            xshift = lsq[0][1] - (ccor.shape[1]//2-1)
            yshift = lsq[0][2] - (ccor.shape[0]//2-1)
            if (_verbosity == fatboyLog.VERBOSE):
                print("Fit 2d "+str(j)+": ",time.time()-tt,"; Total: ",time.time()-t)
        else:
            #Fit Gaussians to determine shift with subpixel accuracy
            p = zeros(4, float64)
            p[0] = ccor.max()
            p[1] = b[1][0]
            p[2] = 3
            p[3] = 0
            y = ccor[b[0][0],:]
            x = arange(len(y), dtype=float64)
            lsq = leastsq(gaussResiduals, p, args=(x, y))
            xshift = lsq[0][1] - (len(x)//2-1)
            if (_verbosity == fatboyLog.VERBOSE):
                print("Fit Xshift "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

            p = zeros(4, float64)
            p[0] = ccor.max()
            p[1] = b[0][0]
            p[2] = 3
            p[3] = 0
            y = ccor[:,b[1][0]]
            if (median_filter2d):
                #Median filter 1-d cut
                y = gpumedianfilter(array(y))
            x = arange(len(y), dtype=float64)
            lsq = leastsq(gaussResiduals, p, args=(x, y))
            yshift = lsq[0][1] - (len(x)//2-1)
            if (_verbosity == fatboyLog.VERBOSE):
                print("Fit Yshift "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
            tt = time.time()

        if (constrain):
            xshift += xguess
            yshift += yguess

        if (method == METHOD_SEP_CENTROID or method == METHOD_SEP_CENTROID_CONSTRAINED):
            #Use xshift, yshift as initial guess and attempt to match up objects found via sep
            currObjects = array([objects['x'], objects['y']])
            matchRef = []
            matchCurr = []
            for l in range(len(refObjects[0])):
                for i in range(len(currObjects[0])):
                    #object i in currFrame is within 5 pixels in x and y of object l in refFrame
                    if (abs(refObjects[0,l]-currObjects[0,i]-xshift) < 5 and abs(refObjects[1,l]-currObjects[1,i]-yshift) < 5):
                        matchRef.append(l)
                        matchCurr.append(i)
            xdiff = refObjects[0,:][matchRef]-currObjects[0,:][matchCurr]
            ydiff = refObjects[1,:][matchRef]-currObjects[1,:][matchCurr]
            print("gpu_xregister> Found "+str(len(xdiff))+" matching objects.  xshift = "+str(xdiff.mean())+" +/- "+str(xdiff.std())+"; yshift = "+str(ydiff.mean())+" +/- "+str(ydiff.std()))
            write_fatboy_log(log, logtype, "Found "+str(len(xdiff))+" matching objects.  xshift = "+str(xdiff.mean())+" +/- "+str(xdiff.std())+"; yshift = "+str(ydiff.mean())+" +/- "+str(ydiff.std()), __name__)
            niter = 0
            nprev = 0
            #Do iterative sigma clipping of differences of centroids
            while (niter < 5 and nprev != len(xdiff)):
                nprev = len(xdiff)
                niter += 1
                b = (abs(xdiff-xdiff.mean()) < 2*xdiff.std()) * (abs(ydiff-ydiff.mean()) < 2*ydiff.std())
                xdiff = xdiff[b]
                ydiff = ydiff[b]
            xshift = xdiff.mean()
            yshift = ydiff.mean()
            print("gpu_xregister> Used "+str(len(xdiff))+" matching objects.  xshift = "+str(xshift)+" +/- "+str(xdiff.std())+"; yshift = "+str(yshift)+" +/- "+str(ydiff.std()))
            write_fatboy_log(log, logtype, "Used "+str(len(xdiff))+" matching objects.  xshift = "+str(xshift)+" +/- "+str(xdiff.std())+"; yshift = "+str(yshift)+" +/- "+str(ydiff.std()), __name__)
            del matchRef
            del matchCurr

        print("Shift from "+refName+" to "+currName+" is ("+str(xshift)+", "+str(yshift)+").")
        write_fatboy_log(log, logtype, "Shift from "+refName+" to "+currName+" is ("+str(xshift)+", "+str(yshift)+").", __name__)
        if (outfile is not None):
            f.write(str(xshift)+'\t'+str(yshift)+'\n')
        xshifts.append(xshift)
        yshifts.append(yshift)

        #GUI message:
        if (gui is not None):
            gui = (gui[0], gui[1]+1., gui[2], gui[3], gui[4])
            if (gui[0]): print("PROGRESS: "+str(int(gui[3]+gui[1]/gui[2]*gui[4])))

    #Check for low ccor values
    for j in range(1, len(ccmax)):
        if (ccmax[j] < 0.3*max(ccmax)):
            if (mode == MODE_FITS):
                name = filelist[j]
            elif (mode == MODE_RAW):
                name = "frame "+str(j)
            elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
                name = frames[j].getFullId()
            print("WARNING: "+name+" may not have been properly aligned.  Check your box size!")
            write_fatboy_log(log, logtype, "WARNING: "+name+" may not have been properly aligned.  Check your box size!", __name__)

    if (outfile is not None):
        f.close()

    if (_verbosity == fatboyLog.VERBOSE):
        print("Cleanup "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
    print("Xregistered "+str(nframes)+" frames. Total time (s): "+str(time.time()-t))
    write_fatboy_log(log, logtype, "Xregistered "+str(nframes)+" frames. Total time (s): "+str(time.time()-t), __name__)
    return [xshifts, yshifts]
