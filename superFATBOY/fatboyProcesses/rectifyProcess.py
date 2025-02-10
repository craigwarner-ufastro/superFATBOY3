from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_imcombine, imcombine
from numpy import *
from scipy.optimize import leastsq
from superFATBOY import gpu_drihizzle, drihizzle

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

class rectifyProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    #Calculate longslit rectification transformation
    #Returns [xcoeffs, ycoeffs] and optionally writes coeffs to file
    def calcLongslitContinuaRectification(self, fdu, rctfdus):
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/rectified", os.F_OK)):
                os.mkdir(outdir+"/rectified",0o755)

        #Read options
        fit_order = int(self.getOption("fit_order", fdu.getTag()))
        maxSpectra = int(self.getOption("max_continua_per_slit", fdu.getTag()))
        thresh = float(self.getOption("min_threshold", fdu.getTag()))
        minCovFrac = float(self.getOption("min_coverage_fraction", fdu.getTag()))
        xinit = self.getOption("continuum_trace_xinit", fdu.getTag())
        trace_ylo = self.getOption("continuum_trace_ylo", fdu.getTag())
        trace_yhi = self.getOption("continuum_trace_yhi", fdu.getTag())
        find_xlo = self.getOption("continuum_find_xlo", fdu.getTag())
        find_xhi = self.getOption("continuum_find_xhi", fdu.getTag())
        maxSlope = float(self.getOption("sky_max_slope", fdu.getTag()))
        min_gauss_width = float(self.getOption("min_continuum_fwhm", fdu.getTag()))
        if (self.getOption("use_zero_as_center_fitting", fdu.getTag()).lower() == "yes"):
            useCenterAsZero = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        #Set default values for xinit, trace_ylo, trace_yhi, convert to int
        if (xinit is None):
            xinit = xsize//2
        xinit = int(xinit)
        if (trace_ylo is None):
            trace_ylo = 0
        trace_ylo = int(trace_ylo)
        if (trace_yhi is None):
            trace_yhi = ysize
        trace_yhi = int(trace_yhi)
        if (find_xlo is None):
            find_xlo = 0
        find_xlo = int(find_xlo)
        if (find_xhi is None):
            find_xhi = xsize
        find_xhi = int(find_xhi)

        print("rectifyProcess::calcLongslitContinuaRectification> Searching for continua to trace out...")
        self._log.writeLog(__name__, "Searching for continua to trace out...")

        ncont = 0
        for j in range(len(rctfdus)):
            #Median filter then find continuum at sigma > 3, min width 5px
            if (rctfdus[j].dispersion == fdu.DISPERSION_HORIZONTAL):
                oned = mediansmooth1d(sum(rctfdus[j].getData(tag="cleanFrame")[:,find_xlo:find_xhi], 1), 5)
            elif (rctfdus[j].dispersion == fdu.DISPERSION_VERTICAL):
                oned = mediansmooth1d(sum(rctfdus[j].getData(tag="cleanFrame")[find_xlo:find_xhi,:], 0), 5)
            #continuaList is an n x 2 array of [[ylo1, yhi1], [ylo2, yhi2], ...]
            continuaList = extractSpectra(oned, sigma=thresh, width=5, nspec=maxSpectra)
            if (not rctfdus[j].hasProperty("use_only_positive")):
                #Check the inverse of the data too for continua from the frame paired with
                #this one for sky subtraction
                troughList = extractSpectra(-1*oned, sigma=thresh, width=5, nspec=maxSpectra)
                if (continuaList is None and troughList is not None):
                    continuaList = -1*troughList
                elif (troughList is not None):
                    #Use numpy concatenate
                    continuaList = concatenate((continuaList, -1*troughList))
            if (continuaList is None):
                continue
            #Reject continua outside of [trace_ylo:trace_yhi] range
            keep = ones(len(continuaList), dtype=bool)
            for i in range(len(continuaList)):
                ylo = continuaList[i][0]
                yhi = continuaList[i][1]
                ycen = abs(ylo+yhi)//2
                if (trace_ylo is not None and ycen < trace_ylo):
                    keep[i] = False
                elif (trace_yhi is not None and ycen > trace_yhi):
                    keep[i] = False
                if (ylo == 0 or yhi == ysize or yhi == -1*ysize):
                    keep[i] = False
                if (keep[i]):
                    if (ylo > 0):
                        print("\tFound continuum ["+str(ylo)+":"+str(yhi)+"] in "+rctfdus[j].getFullId()+".")
                        self._log.writeLog(__name__, "Found continuum ["+str(ylo)+":"+str(yhi)+"] in "+rctfdus[j].getFullId()+".", printCaller=False, tabLevel=1)
                    else:
                        #This is a trough
                        print("\tFound negative continuum ["+str(-1*ylo)+":"+str(-1*yhi)+"] in "+rctfdus[j].getFullId()+".")
                        self._log.writeLog(__name__, "Found negative continuum ["+str(-1*ylo)+":"+str(-1*yhi)+"] in "+rctfdus[j].getFullId()+".", printCaller=False, tabLevel=1)
                    ncont += 1
                    if (xinit == -1 and ylo > 0):
                        #xinit == -1 => find brightest part of continuum within middle half of chip
                        #Use first kept spectrum for this purpose
                        if (rctfdus[j].dispersion == fdu.DISPERSION_HORIZONTAL):
                            zcut = mediansmooth1d(sum(rctfdus[j].getData(tag="cleanFrame")[ylo:yhi+1,:],0), 5)
                        elif (rctfdus[j].dispersion == fdu.DISPERSION_VERTICAL):
                            zcut = mediansmooth1d(sum(rctfdus[j].getData(tag="cleanFrame")[:,ylo:yhi+1], 1), 5)
                        xlo = int(zcut.size//4)
                        xhi = int(zcut.size*3//4)
                        xinit = where(zcut == max(zcut[xlo:xhi]))[0][0]
                else:
                    print("rectifyProcess::calcLongslitContinuaRectification> Continuum ["+str(ylo)+":"+str(yhi)+"] in "+rctfdus[j].getFullId()+" is outside of range ["+str(trace_ylo)+":"+str(trace_yhi)+"].  Ignoring!")
                    self._log.writeLog(__name__, "Continuum ["+str(ylo)+":"+str(yhi)+"] in "+rctfdus[j].getFullId()+" is outside of range ["+str(trace_ylo)+":"+str(trace_yhi)+"].  Ignoring!")
            if (keep.sum() > 0):
                #Update fdu property "continua_list"
                rctfdus[j].setProperty("continua_list", continuaList[keep])

        print("rectifyProcess::calcLongslitContinuaRectification> Using "+str(ncont)+ " continua to trace out rectification with xinit="+str(xinit)+"...")
        self._log.writeLog(__name__, "Using "+str(ncont)+ " continua to trace out rectification with xinit="+str(xinit)+"...")

        #setup output lists
        xin = []
        yin = []
        yout = []
        ncont = 0
        #setup input lists
        #xs = x values (dispersion direction) to cross correlate at
        #Start at middle and trace to end then to beginning
        #Set this up before looping over orders
        step = 5
        xs = list(range(xinit, xsize-50, step))+list(range(xinit-step, 50, -1*step))
        #Index array used for fitting
        yind = arange(ysize, dtype=float64)

        #Loop over FDUs
        for currFDU in rctfdus:
            if (not currFDU.hasProperty("continua_list")):
                continue
            #Get qa data here
            qaData = currFDU.getData(tag="cleanFrame").copy()

            #Loop over continua_list
            for (cylo, cyhi) in currFDU.getProperty("continua_list"):
                #Create new copy of data here
                currData = currFDU.getData(tag="cleanFrame").copy()
                isInverted = False

                if (cylo < 0):
                    #We need to trace out a trough here so invert data
                    currData *= -1
                    cylo *= -1
                    cyhi *= -1
                    isInverted = True

                #Attempt to "blank out" any troughs due to sky subtraction
                #Median filter then find continuum at sigma > 3, min width 5px
                if (currFDU.dispersion == fdu.DISPERSION_HORIZONTAL):
                    oned = -1*mediansmooth1d(sum(currData, 1), 5)
                elif (currFDU.dispersion == fdu.DISPERSION_VERTICAL):
                    oned = -1*mediansmooth1d(sum(currData, 0), 5)
                #continuaList is an n x 2 array of [[ylo1, yhi1], [ylo2, yhi2], ...]
                troughList = extractSpectra(oned, sigma=thresh, width=5, nspec=maxSpectra)
                if (troughList is not None):
                    for j in range(len(troughList)):
                        if (currFDU.dispersion == fdu.DISPERSION_HORIZONTAL):
                            currData[troughList[j][0]:troughList[j][1]+1,:] = 0
                        elif (currFDU.dispersion == fdu.DISPERSION_VERTICAL):
                            currData[:,troughList[j][0]:troughList[j][1]+1] = 0

                ycen = (cyhi+cylo)//2
                yboxsize = (cyhi-cylo)//2
                if (yboxsize < 4):
                    #Minimum boxsize
                    yboxsize = 4
                if (yboxsize > 50):
                    #Maximum boxsize
                    yboxsize = 50
                #Setup lists and arrays for within each loop
                #xcoords and ycoords contain lists of fit (x,y) points
                xcoords = []
                ycoords = []
                #peak values of fits are kept and used as rejection criteria later
                peaks = []
                #Up to last 10 (x,y) pairs are kept and used in various rejection criteria
                lastXs = []
                lastYs = []
                currX = xs[0] #current X value
                currY = ycen #shift in cross-dispersion direction at currX relative to Y at X=xinit
                gaussWidth = yboxsize/3.
                if (gaussWidth < 2):
                    #Minimum 2 pixels
                    gaussWidth = 2

                #Loop over xs every 5 pixels and try to fit Gaussian across continuum
                for j in range(len(xs)):
                    if (xs[j] == xinit-step):
                        #We have finished tracing to the end, starting back at middle to trace in other direction
                        if (len(ycoords) == 0):
                            #print "ERR1B"
                            break
                        #Reset currY, lastYs, lastXs
                        currY = ycoords[0]
                        lastYs = [ycoords[0]]
                        lastXs = [xcoords[0]]
                    if (j == 0):
                        #Use a wider y-range for the first datapoint
                        #Also use median of 15 pixel box instead of sum of 5 pixel box
                        ylo = int(currY - 4*yboxsize)
                        yhi = int(currY + 4*yboxsize)
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            data = currData[:,xs[j]-7:xs[j]+8].copy()
                            y = gpu_arraymedian(data, axis="X")
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            data = currData[xs[j]-7:xs[j]+8,:].copy()
                            y = gpu_arraymedian(data, axis="Y")
                        #Mask negative pixels for first datapoint to get rid of troughs
                        y[y < 0] = 0
                    else:
                        ylo = int(currY - 2*yboxsize)
                        yhi = int(currY + 2*yboxsize)
                        #Sum 5 pixel box and fit 1-d Gaussian
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            y = sum(currData[:, xs[j]-2:xs[j]+3], 1, dtype=float64)
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            y = sum(currData[xs[j]-2:xs[j]+3, :], 0, dtype=float64)
                    #Make sure it doesn't move off chip
                    if (ylo < 0):
                        ylo = 0
                    if (yhi >= ysize):
                        yhi = ysize-1
                    #Initial guesses for Gaussian fit
                    p = zeros(4, dtype=float64)
                    p[0] = max(y[ylo:yhi])
                    p[1] = (where(y[ylo:yhi] == p[0]))[0][0]+ylo
                    p[2] = gaussWidth
                    p[3] = gpu_arraymedian(y.copy())
                    #Range of pixels above and below continuum used for calculating std dev of background
                    stdrng = list(range(ylo,ylo+yboxsize+1))+list(range(yhi-yboxsize-1,yhi))
                    if (p[0]-p[3] < 7*y[stdrng].std() and j != 0):
                        #Set minimum threshold at 7 sigma significance to be continnum
                        continue
                    if (abs(p[1]-currY) > yboxsize and j != 0):
                        #Highest value is > yboxsize pixels from the previously fit peak.  Throw out this point
                        continue
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(yind[ylo:yhi], y[ylo:yhi]))
                    except Exception as ex:
                        continue

                    #Error checking results of leastsq call
                    if (lsq[1] == 5):
                        #exceeded max number of calls = ignore
                        continue
                    if (lsq[0][0]+lsq[0][3] < 0):
                        #flux less than zero = ignore
                        continue
                    if (lsq[0][2] < 0 and j != 0):
                        #negative fwhm = ignore unless first datapoint
                        continue
                    if (j == 0):
                        #First datapoint -- update currX, currY, append to all lists
                        currY = lsq[0][1]
                        currX = xs[0]
                        peaks.append(lsq[0][0])
                        xcoords.append(xs[j])
                        ycoords.append(lsq[0][1])
                        lastXs.append(xs[j])
                        lastYs.append(lsq[0][1])
                        #update gaussWidth to be actual fit FWHM
                        if (lsq[0][2] < gaussWidth):
                            #1.5 pixel minimum default
                            gaussWidth = max(abs(lsq[0][2]), min_gauss_width)
                    else:
                        #FWHM is over a factor of 2 different than first fit.  Throw this point out
                        if (lsq[0][2] > 2*gaussWidth or lsq[0][2] < 0.5*gaussWidth):
                            continue
                        #Sanity check
                        #Calculate predicted "ref" value of Y based on slope of previous
                        #fit datapoints
                        wavg = 0.
                        wavgx = 0.
                        wavgDivisor = 0.
                        #Compute weighted avg of previously fitted values
                        #Weight by 1 over sqrt of delta-x
                        #Compare current y fit value to weighted avg instead of just
                        #previous value.
                        for i in range(len(lastYs)):
                            wavg += lastYs[i]/sqrt(abs(lastXs[i]-xs[j]))
                            wavgx += lastXs[i]/sqrt(abs(lastXs[i]-xs[j]))
                            wavgDivisor += 1./sqrt(abs(lastXs[i]-xs[j]))
                        if (wavgDivisor != 0):
                            wavg = wavg/wavgDivisor
                            wavgx = wavgx/wavgDivisor
                        else:
                            #We seem to have no datapoints in lastYs.  Simply use previous value
                            wavg = currY
                            wavgx = currX
                        #More than 50 pixels in deltaX between weight average of last 10
                        #datapoints and current X
                        #And not the discontinuity in middle of xs where we jump from end back to center
                        #because abs(xs[j]-xs[j-1]) == step
                        if (abs(xs[j]-xs[j-1]) == step and abs(wavgx-xs[j]) > 50):
                            if (len(lastYs) > 1):
                                #Fit slope to lastYs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                                slope = lin[0][1]
                            else:
                                #Only 1 datapoint, use +/- maxSlope as slope
                                slope = -1*abs(maxSlope)
                                if ((lsq[0][1]-wavg)/(xs[j]-wavgx) > 0):
                                    slope = abs(maxSlope)
                            #Calculate guess for refX and max acceptable error
                            #err = 1+maxSlope*deltaY, with a max value of 3.
                            refY = wavg+slope*(xs[j]-wavgx)
                            maxerr = min(1+int(abs(xs[j]-wavgx)*.02),3)
                        else:
                            if (len(lastYs) > 3):
                                #Fit slope to lastYs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                                slope = lin[0][1]
                            else:
                                #Less than 4 datapoints, use +/- maxSlope as slope
                                slope = -1*abs(maxSlope)
                                if ((lsq[0][1]-wavg)/(xs[j]-wavgx) > 0):
                                    slope = abs(maxSlope)
                            #Calculate guess for refX and max acceptable error
                            #0.5 <= maxerr <= 2 in this case.  Use slope*50 if it falls in that range
                            refY = wavg+slope*(xs[j]-wavgx)
                            maxerr = max(min(abs(slope*50),2),0.5)
                        #Discontinuity point in xs. Keep if within +/-1.
                        if (xs[j] == xinit-step and abs(lsq[0][1]-currY) < 1):
                            #update currX, currY, append to all lists
                            currY = lsq[0][1]
                            currX = xs[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                        elif (abs(lsq[0][1] - refY) < maxerr):
                            #Regular datapoint.  Apply sanity check rejection criteria here
                            #Discard if farther than maxerr away from refY
                            if (abs(xs[j]-currX) < 4*step and maxerr > 1 and abs(lsq[0][1]-currY) > maxerr):
                                #Also discard if < 20 pixels in X from last fit datapoint, and deltaY > 1
                                continue
                            #update currX, currY, append to all lists
                            currY = lsq[0][1]
                            currX = xs[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            #keep lastXs and lastYs at 10 elements or less
                            if (len(lastYs) > 10):
                                lastXs.pop(0)
                                lastYs.pop(0)
                    #print xs[j], p[1], lsq[0][1], lsq[0][0], lsq[0][2]
                print("rectifyProcess::calcLongslitContinuaRectification> Continuum centered at "+str(ycen)+" in "+currFDU.getFullId()+": found "+str(len(ycoords))+" datapoints.")
                self._log.writeLog(__name__, "Continuum centered at "+str(ycen)+" in "+currFDU.getFullId()+": found "+str(len(ycoords))+" datapoints.")
                #Check coverage fraction
                covfrac = len(ycoords)*100.0/len(xs)
                if (covfrac < minCovFrac):
                    print("rectifyProcess::calcLongslitContinuaRectification> Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping continuum!")
                    self._log.writeLog(__name__, "Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping continuum!")
                    continue

                #Phase 2 of rejection criteria after continua have been traced
                #Find outliers > 2.5 sigma in peak values and remove them
                #First store first value as yout
                peaks = array(peaks)
                peakmed = arraymedian(peaks)
                peaksd = peaks.std()
                b = (peaks > peakmed-2.5*peaksd)*(peaks < peakmed+2.5*peaksd)
                xcoords = array(xcoords)[b]
                ycoords = array(ycoords)[b]
                print("\trejecting outliers (phase 2) - kept "+str(len(ycoords))+" datapoints.")
                self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(ycoords))+" datapoints.", printCaller=False, tabLevel=1)

                #Fit 2nd order order polynomial to datapoints, Y = f(X)
                order = 2
                p = zeros(order+1, float64)
                p[0] = ycoords[0]
                try:
                    lsq = leastsq(polyResiduals, p, args=(xcoords,ycoords,order))
                except Exception as ex:
                    print("rectifyProcess::calcLongslitContinuaRectification> Could not fit continuum: "+str(ex))
                    self._log.writeLog(__name__, "Could not fit continuum: "+str(ex))
                    continue

                #Compute output offsets and residuals from actual datapoints
                yprime = polyFunction(lsq[0], xcoords, order)
                yresid = yprime-ycoords
                xcen = xsize//2
                currYout = polyFunction(lsq[0], xcen, order) #yout at xcenter
                #Remove outliers and refit
                b = abs(yresid) < yresid.mean()+2.5*yresid.std()
                xcoords = xcoords[b]
                ycoords = ycoords[b]
                print("\trejecting outliers (phase 3). Sigma = "+str(yresid.std())[:5]+". Using "+str(len(ycoords))+" datapoints to fit continuum.")
                self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+str(yresid.std())[:5]+". Using "+str(len(ycoords))+" datapoints to fit continuum.", printCaller=False, tabLevel=1)

                #Check coverage fraction
                covfrac = len(ycoords)*100.0/len(xs)
                if (covfrac >= minCovFrac):
                    xin.extend(xcoords)
                    yin.extend(ycoords)
                    yout.extend([currYout]*len(ycoords))
                    ncont += 1
                    qavalue = -50000
                    if (isInverted):
                        #use +50000 for qa data to show up
                        qavalue = 50000
                    #Generate qa data
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        for i in range(len(xcoords)):
                            yval = int(ycoords[i]+.5)
                            xval = int(xcoords[i]+.5)
                            for yi in range(yval-1,yval+2):
                                for xi in range(xval-1,xval+2):
                                    dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                    qaData[yi,xi] = qavalue/((1+dist)**2)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        for i in range(len(xcoords)):
                            yval = int(ycoords[i]+.5)
                            xval = int(xcoords[i]+.5)
                            for yi in range(yval-1,yval+2):
                                for xi in range(xval-1,xval+2):
                                    dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                    qaData[xi,yi] = qavalue/((1+dist)**2)
                print("\tY Center: "+formatNum(currYout)+"\t Xref: "+str(xs[0])+"\t Cov. Frac: "+formatNum(covfrac))
                self._log.writeLog(__name__, "Y Center: "+formatNum(currYout)+"\t Xref: "+str(xs[0])+"\t Cov. Frac: "+formatNum(covfrac), printCaller=False, tabLevel=1)
                del currData

            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            #Create output filename
            qafile = outdir+"/rectified/qa_"+currFDU.getFullId()

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                if (os.access(qafile, os.F_OK)):
                    os.unlink(qafile)
            #Write out qa file
            if (not os.access(qafile, os.F_OK)):
                currFDU.tagDataAs("slitqa", qaData)
                currFDU.writeTo(qafile, tag="slitqa")
                currFDU.removeProperty("slitqa")
            del qaData

        print("rectifyProcess::calcLongslitContinuaRectification> Successfully traced out "+str(ncont)+ " continua.  Fitting transformation...")
        self._log.writeLog(__name__, "Successfully traced out "+str(ncont)+ " continua.  Fitting transformation...")
        #Convert to arrays
        xin = array(xin, dtype=float32)
        yin = array(yin, dtype=float32)
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #Need to swap xin, yin here because surfaceResiduals and surfaceFunction expect out = f(xin, yin) not f(yin, xin).
            tmp = xin
            xin = yin
            yin = tmp
        yout = array(yout)
        #Fit entire continuum transformation
        #Calculate number of terms based on order
        terms = 0
        for j in range(fit_order+2):
            terms+=j
        p = zeros(terms)
        #Initial guess is f(x_in, y_in) = y_in
        p[2] = 1
        #Want to define center to be (0,0) before fitting
        ycen = float(ysize-1)*0.5
        xcen = float(xsize-1)*0.5
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yout -= ycen
            xin -= xcen
            yin -= ycen
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #xin, yin have been swapped
            yout -= ycen
            xin -= ycen
            yin -= xcen
        try:
            lsq = leastsq(surfaceResiduals, p, args=(xin, yin, yout, fit_order))
        except Exception as ex:
            print("rectifyProcess::calcLongslitContinuaRectification> ERROR performing least squares fit: "+str(ex)+"! Continua will NOT be rectified!")
            self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Continua will NOT be rectified!", type=fatboyLog.ERROR)
            ycoeffs = array([0, 0, 1])
            if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                ycoeffs = array([0, 1, 0])
            return ycoeffs

        #Compute output offsets and residuals from actual datapoints
        yprime = surfaceFunction(lsq[0], xin, yin, fit_order)
        yresid = yprime-yout
        residmean = yresid.mean()
        residstddev = yresid.std()
        print("\tFound "+str(len(yout))+" datapoints.  Fit: "+formatList(lsq[0]))
        print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
        self._log.writeLog(__name__, "Found "+str(len(yout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
        self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

        print("rectifyProcess::calcLongslitContinuaRectification> Performing iterative sigma clipping to throw away outliers...")
        self._log.writeLog(__name__, "Performing iterative sigma clipping to throw away outliers...")

        #Throw away outliers starting at 2 sigma significance
        sigThresh = 2
        niter = 0
        norig = len(yout)
        bad = where(abs(yresid-residmean)/residstddev > sigThresh)
        while (len(bad[0]) > 0):
            niter += 1
            good = (abs(yresid-residmean)/residstddev <= sigThresh)
            xin = xin[good]
            yin = yin[good]
            yout = yout[good]
            #Refit, use last actual fit coordinates as input guess
            p = lsq[0]
            try:
                lsq = leastsq(surfaceResiduals, p, args=(xin, yin, yout, fit_order))
            except Exception as ex:
                print("rectifyProcess::calcLongslitContinuaRectification> ERROR performing least squares fit: "+str(ex)+"! Continua will NOT be rectified!")
                self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Continua will NOT be rectified!", type=fatboyLog.ERROR)
                ycoeffs = array([0, 0, 1])
                if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    ycoeffs = array([0, 1, 0])
                return ycoeffs

            #Compute output offsets and residuals from actual datapoints
            yprime = surfaceFunction(lsq[0], xin, yin, fit_order)
            yresid = yprime-yout
            residmean = yresid.mean()
            residstddev = yresid.std()
            if (niter > 2):
                #Gradually increase sigma threshold
                sigThresh += 0.2
            bad = where(abs(yresid-residmean)/residstddev > sigThresh)
        print("\tAfter "+str(niter)+" passes, kept "+str(len(yout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
        print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
        self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(yout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
        self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

        #Write out more qa data
        qafile = outdir+"/rectified/qa_"+fdu._id+"-continua.dat"
        f = open(qafile,'w')
        for i in range(len(yout)):
            f.write(str(xin[i])+'\t'+str(yin[i])+'\t'+str(yout[i])+'\t'+str(yprime[i])+'\n')
        f.close()

        ycoeffs = lsq[0]
        return ycoeffs
    #end calcLongslitContinuaRectification

    def calcLongslitSkylineRectification(self, fdu, skyFDU):
        if (skyFDU is None):
            print("rectifyProcess::calcLongslitSkylineRectification> Warning: Could not find clean sky frame associated with "+fdu.getFullId()+"!  Skylines will NOT be rectified!")
            self._log.writeLog(__name__, "Could not find clean sky frame associated with "+fdu.getFullId()+"!  Skylines will NOT be rectified!", type=fatboyLog.WARNING)
            xcoeffs = array([0, 1, 0])
            if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                xcoeffs = array([0, 0, 1])
            return xcoeffs

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/rectified", os.F_OK)):
                os.mkdir(outdir+"/rectified",0o755)

        #Read options
        fit_order = int(self.getOption("sky_fit_order", fdu.getTag()))
        thresh = float(self.getOption("min_sky_threshold", fdu.getTag()))
        minCovFrac = float(self.getOption("min_coverage_fraction", fdu.getTag()))
        skybox = float(self.getOption("sky_boxsize", fdu.getTag()))
        min_gauss_width = float(self.getOption("min_continuum_fwhm", fdu.getTag()))
        yinit = self.getOption("skyline_trace_yinit", fdu.getTag())
        find_ylo = self.getOption("skyline_find_ylo", fdu.getTag())
        find_yhi = self.getOption("skyline_find_yhi", fdu.getTag())
        useTwoPasses = False
        if (self.getOption("sky_two_pass_detection", fdu.getTag()).lower() == "yes"):
            useTwoPasses = True
        useArclamps = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True

        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        if (find_ylo is None):
            find_ylo = 0
        find_ylo = int(find_ylo)
        if (find_yhi is None):
            find_yhi = ysize
        find_yhi = int(find_yhi)

        print("rectifyProcess::calcLongslitSkylineRectification> Searching for sky/arclamp to trace out, using "+skyFDU.getFullId()+"...")
        self._log.writeLog(__name__, "Searching for sky/arclamp to trace out, using "+skyFDU.getFullId()+"...")

        skyData = skyFDU.getData().copy()
        xcenters = []
        #Take 1-d sum and median filter
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            oned = sum(skyData[find_ylo:find_yhi,:], 0)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            oned = sum(skyData[:,find_ylo:find_yhi], 1)

        if (self._fdb.getGPUMode()):
            oned = gpumedianfilter(oned)
        else:
            oned = medianfilterCPU(oned)

        #Next smooth the 1-d cut
        if (self._fdb.getGPUMode()):
            z = smooth1d(oned,5,1)
        else:
            z = smooth1dCPU(oned, 5, 1)
        medVal = arraymedian(z, nonzero=True)
        sig = z[where(z != 0)].std()

        if (usePlot and (debug or writePlots)):
            plt.plot(z)

        #Zero out the first and last 15 pixels
        z[:15] = 0.
        z[-15:] = 0.

        #First pass at globally finding sky lines
        while (max(z) > medVal+thresh*sig):
            if (max(z) <= 0):
                #All points have been zeroed out
                break
            #Find brightest point
            b = (where(z == max(z)))[0][0]
            #Check that it matches min threshold
            if (z[b] > medVal+thresh*sig):
                valid = True
                #If data within +/-5 pixels has been zeroed out already, this is
                #too near another line
                for l in range(b-5,b+6):
                    if (z[l] == 0):
                        valid = False
                #This line is valid -- it is at least 6 pixels away from another line
                if valid:
                    #Append to linelist
                    xcenters.append(b)
                    print("\tFound skyline: Center = "+str(b)+"; sigma = "+str((z[b]-medVal)/sig))
                    self._log.writeLog(__name__, "Found skyline: Center = "+str(b)+"; sigma = "+str((z[b]-medVal)/sig), printCaller=False, tabLevel=1)
                #Zero out +/-15 pixels from this line
                z[b-15:b+15] = 0
                #Update median and std dev
                medVal = arraymedian(z, nonzero=True)
                sig = z[where(z != 0)].std()
        #Split into 4 sections for two pass detection
        if (useTwoPasses):
            print("rectifyProcess::calcLongslitSkylineRectification> Pass 2: Searching for additional sky/arclamp lines...")
            self._log.writeLog(__name__, "Pass 2: Searching for additional sky/arclamp lines...")
            xs = []
            for k in range(50,xsize-49,(xsize-100)//3):
                xs.append(k)
            #Loop over each section
            for k in range(len(xs)-1):
                #Create new array with "z" value for just this section and zero out edges
                zlocal = z[xs[k]:xs[k+1]]
                zlocal[0:15] = 0.
                zlocal[-15:] = 0.
                medVal = arraymedian(zlocal, nonzero=True)
                #Calculate sigma if there are nonzero points left in this section
                if (len(where(zlocal != 0)[0]) != 0):
                    sig = zlocal[where(zlocal != 0)].std()
                else:
                    #set sigma to 10000 - just a way to fail condition of while loop
                    sig=10000.
                #Always try to get at least one line if there is any nonzero data
                firstPass = True
                while (max(zlocal) > medVal+thresh*sig or firstPass):
                    if (max(zlocal) <= 0):
                        #All points have been zeroed out
                        break
                    #Find brightest point
                    b = (where(zlocal == max(zlocal)))[0][0]
                    #If first pass, lower threshold to just 1.5 sigma so that at least one point
                    #can be used to anchor fit in this section
                    if (firstPass and zlocal[b] < medVal+1.5*sig):
                        #Only if < 1.5 sigma, discard
                        firstPass = False
                        continue
                    #If this line matches min threshold OR is the first pass and is >= 1.5 sigma, keep
                    if (zlocal[b] > medVal+thresh*sig or firstPass):
                        valid = True
                        #If data within +/-5 pixels has been zeroed out already, this is
                        #too near another line
                        for l in range(b-5,b+6):
                            if (zlocal[l] == 0):
                                valid = False
                        #print b+xs[k], len(where(zlocal != 0)[0]), (zlocal[b]-med)/sig
                        firstPass = False
                        #This line is valid -- it is at least 6 pixels away from another line
                        if valid:
                            #Append to linelist
                            xcenters.append(b+xs[k])
                            print("\tFound skyline: Center = "+str(b+xs[k])+"; sigma = "+str((zlocal[b]-medVal)/sig))
                            self._log.writeLog(__name__, "Found skyline: Center = "+str(b+xs[k])+"; sigma = "+str((zlocal[b]-medVal)/sig), printCaller=False, tabLevel=1)
                        #Zero out +/-15 pixels from this line
                        zlocal[b-15:b+15] = 0
                        if (len(where(zlocal != 0)[0]) == 0):
                            break
                        #update median and sd
                        medVal = arraymedian(zlocal, nonzero=True)
                        sig = zlocal[where(zlocal != 0)].std()

        #Use brightest skyline to trace out range in y where skylines are visible
        #(Usually they cut off before the top/bottom of the chip)
        nlines = len(xcenters)
        skyData = skyFDU.getData()
        bcen = int(xcenters[0])
        #Sum 1-d cut at 5 pixels centered around skyline
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            bline = sum(skyData[:,bcen-2:bcen+3],1, dtype=float64)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            bline = sum(skyData[bcen-2:bcen+3,:],0, dtype=float64)
        #Calculate median and std dev
        blmed = arraymedian(bline)
        blsig = bline.std()

        #Default values
        ylo = 0
        yhi = len(bline)-1

        #Find lower cutoff
        isylo = False
        currY = ysize//2
        while (not isylo and currY > 5):
            #If 5 consecutive points in 1-d cut are below -2 sigma, we've found lower cutoff
            if (alltrue(bline[currY-4:currY+1] < blmed-2*blsig)):
                ylo = currY
                isylo = True
            currY-=1
        #Find upper cutoff
        isyhi = False
        currY = ysize//2
        while (not isyhi and currY < ysize-5):
            #If 5 consecutive points in 1-d cut are below -2 sigma, we've found upper cutoff
            if (alltrue(bline[currY:currY+5] < blmed-2*blsig)):
                yhi = currY
                isyhi = True
            currY+=1

        print("rectifyProcess::calcLongslitSkylineRectification> Using "+str(nlines)+ " lines to trace out skyline rectification over y-range = ["+str(ylo)+":"+str(yhi)+"]...")
        self._log.writeLog(__name__, "Using "+str(nlines)+ " lines to trace out skyline rectification over y-range = ["+str(ylo)+":"+str(yhi)+"]...")

        if (usePlot and (debug or writePlots)):
            plt.xlabel("Pixel; Nlines = "+str(nlines))
            plt.ylabel("Flux; y-range = ["+str(ylo)+":"+str(yhi)+"]")
            if (writePlots):
                plt.savefig(outdir+"/rectified/skylines_"+fdu._id+".png", dpi=200)
            if (debug):
                plt.show()
            plt.close()

        #Setup list of y positions every 10 pixels
        if (yinit is None):
            yinit = (ylo+yhi)//2
        yinit = int(yinit)
        step = 10
        ys = list(range(yinit, ylo-step, -1*step))+list(range(yinit+step, yhi, step))
        #Index array used for fitting
        xind = arange(xsize, dtype=float64)

        #Setup output lists
        xin = []
        yin = []
        xout = []
        nlines = 0

        #Median filter whole image to bring out skylines first
        if (self._fdb.getGPUMode()):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                skyData = gpumedianfilter2d(skyData)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                skyData = gpumedianfilter2d(skyData, axis="Y")
        else:
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                skyData = medianfilter2dCPU(skyData)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                skyData = medianfilter2dCPU(skyData, axis="Y")
        #qa data
        qaData = skyFDU.getData().copy()

        #Loop over list of skylines
        for xcen in xcenters:
            currX = xcen
            currY = ys[0]
            lastY = ys[0]
            #Setup lists and arrays for within each loop
            #xcoords and ycoords contain lists of fit (x,y) points
            xcoords = []
            ycoords = []
            #peak values of fits are kept and used as rejection criteria later
            peaks = []
            #Up to last 10 (x,y) pairs are kept and used in various rejection criteria
            lastXs = []
            lastYs = []
            gaussWidth = skybox/3.
            if (gaussWidth < 2):
                #Minimum 2 pixels
                gaussWidth = 2
            gaussWidth /= sqrt(2)

            if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("LINE", xcen, yinit)
            #Loop over ys every 10 pixels and try to fit Gaussian across skyline
            for j in range(len(ys)):
                if (ys[j] == yinit+step):
                    if (len(xcoords) == 0):
                        #No data for first half of line including first point
                        break
                    #We have finished tracing to the end, starting back at middle to trace in other direction
                    #Reset currX, lastYs, lastXs
                    currX = xcoords[0]
                    lastYs = [ycoords[0]]
                    lastXs = [xcoords[0]]
                xlo = int(currX - skybox)
                xhi = int(currX + skybox)+1
                #Sum 11 pixel box and fit 1-d Gaussian
                #Use 11 instead of 5 to wash out noise more and obtain better fit
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    x = sum(skyData[ys[j]-5:ys[j]+6,xlo:xhi], 0, dtype=float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    x = sum(skyData[xlo:xhi,ys[j]-5:ys[j]+6], 1, dtype=float64)
                if (len(x) < skybox):
                    continue
                #Make sure it doesn't move off chip
                if (xlo < 0):
                    xlo = 0
                if (xhi >= xsize):
                    xhi = xsize-1
                x[x<0] = 0

                #Initial guesses for Gaussian fit
                p = zeros(4, dtype=float64)
                p[0] = max(x)
                p[1] = (where(x == p[0]))[0][0]+xlo
                p[2] = gaussWidth
                p[3] = gpu_arraymedian(x.copy())
                if (abs(p[1]-currX) > skybox):
                    #Highest value is > skybox pixels from the previously fit peak.  Throw out this point
                    continue
                try:
                    lsq = leastsq(gaussResiduals, p, args=(xind[xlo:xhi], x))
                except Exception as ex:
                    continue

                #print ("\t",xcen, ys[j], xlo, xhi, lsq)
                #if (j == 0):
                #    plt.plot(arange(len(x))+xlo, x)
                #    plt.show()

                #Error checking results of leastsq call
                if (lsq[1] == 5):
                    #exceeded max number of calls = ignore
                    continue
                if (lsq[0][0]+lsq[0][3] < 0 and j != 0):
                    #flux less than zero = ignore unless first datapoint
                    continue
                if (lsq[0][2] < 0 and j != 0):
                    #negative fwhm = ignore unless first datapoint
                    continue
                if (j == 0):
                    #First datapoint -- update currX, currY, append to all lists
                    currX = lsq[0][1]
                    currY = ys[0]
                    peaks.append(lsq[0][0])
                    xcoords.append(lsq[0][1])
                    ycoords.append(ys[j])
                    lastXs.append(lsq[0][1])
                    lastYs.append(ys[j])
                    #update gaussWidth to be actual fit FWHM
                    if (lsq[0][2] < gaussWidth and lsq[0][2] > 1):
                        #1.5 pixel minimum default
                        gaussWidth = max(abs(lsq[0][2]), min_gauss_width)
                else:
                    #FWHM is over a factor of 2 different than first fit.  Throw this point out
                    if (lsq[0][2] > 2*gaussWidth or lsq[0][2] < 0.5*gaussWidth):
                        if (gaussWidth == skybox/3. and lsq[0][2] < 0.5*gaussWidth and lsq[0][2] > 1):
                            #Special case, gaussWidth did not update on first pass because it was super narrow.
                            #Update here.
                            gaussWidth = max(abs(lsq[0][2]), 1.5)
                        else:
                            #print ("ERR6", gaussWidth, lsq[0][2])
                            continue
                    #Sanity check
                    #Calculate predicted "ref" value of X based on slope of previous
                    #fit datapoints
                    wavg = 0.
                    wavgy = 0.
                    wavgDivisor = 0.
                    #Compute weighted avg of previously fitted values
                    #Weight by 1 over sqrt of delta-y
                    #Compare current x fit value to weighted avg instead of just
                    #previous value.
                    for i in range(len(lastXs)):
                        wavg += lastXs[i]/sqrt(abs(lastYs[i]-ys[j]))
                        wavgy += lastYs[i]/sqrt(abs(lastYs[i]-ys[j]))
                        wavgDivisor += 1./sqrt(abs(lastYs[i]-ys[j]))
                    if (wavgDivisor != 0):
                        wavg = wavg/wavgDivisor
                        wavgy = wavgy/wavgDivisor
                    else:
                        #We seem to have no datapoints in lastXs.  Simply use previous value
                        wavg = currX
                        wavgy = currY
                    #More than 50 pixels in deltaY between weight average of last 10
                    #datapoints and current Y
                    #And not the discontinuity in middle of ys where we jump from end back to center
                    #because abs(ys[j]-ys[j-1]) == step
                    if (abs(ys[j]-ys[j-1]) == step and abs(wavgy-ys[j]) > 50):
                        if (len(lastXs) > 1):
                            #Fit slope to lastXs
                            lin = leastsq(linResiduals, [0.,0.], args=(array(lastYs),array(lastXs)))
                            slope = lin[0][1]
                        else:
                            #Only 1 datapoint, use -0.12 as slope
                            slope = -0.12
                        #Calculate guess for refX and max acceptable error
                        #err = 1+0.12*deltaY, with a max value of 3.
                        refX = wavg+slope*(ys[j]-wavgy)
                        maxerr = min(1+int(abs(ys[j]-wavgy)*.02),3)
                    else:
                        if (len(lastXs) > 3):
                            #Fit slope to lastXs
                            lin = leastsq(linResiduals, [0.,0.], args=(array(lastYs),array(lastXs)))
                            slope = lin[0][1]
                        else:
                            #Less than 4 datapoints, use -0.12 as slope
                            slope = -0.12
                        #Calculate guess for refX and max acceptable error
                        #0.5 <= maxerr <= 2 in this case.  Use slope*50 if it falls in that range
                        refX = wavg+slope*(ys[j]-wavgy)
                        maxerr = max(min(abs(slope*50),2),0.5)
                    #Discontinuity point in ys. Keep if within +/-1.
                    if (ys[j] == yinit+step and abs(lsq[0][1]-currX) < 1):
                        #update currX, currY, append to all lists
                        currX = lsq[0][1]
                        currY = ys[j]
                        peaks.append(lsq[0][0])
                        xcoords.append(lsq[0][1])
                        ycoords.append(ys[j])
                        lastXs.append(lsq[0][1])
                        lastYs.append(ys[j])
                    elif (abs(lsq[0][1] - refX) < maxerr):
                        #Regular datapoint.  Apply sanity check rejection criteria here
                        #Discard if farther than maxerr away from refX
                        if (abs(ys[j]-currY) < 4*step and maxerr > 1 and abs(lsq[0][1]-currX) > maxerr):
                            #Also discard if < 20 pixels in Y from last fit datapoint, and deltaX > 1
                            #print ("ERR3")
                            continue
                        #update currX, currY, append to all lists
                        currX = lsq[0][1]
                        currY = ys[j]
                        peaks.append(lsq[0][0])
                        xcoords.append(lsq[0][1])
                        ycoords.append(ys[j])
                        lastXs.append(lsq[0][1])
                        lastYs.append(ys[j])
                        #keep lastXs and lastYs at 10 elements or less
                        if (len(lastYs) > 10):
                            lastXs.pop(0)
                            lastYs.pop(0)
                #print ys[j], p[1], lsq[0][1], lsq[0][0], lsq[0][2]
            print("rectifyProcess::calcLongslitSkylineRectification> Line centered at "+str(xcen)+" in "+skyFDU.getFullId()+": found "+str(len(xcoords))+" datapoints.")
            self._log.writeLog(__name__, "Line centered at "+str(xcen)+" in "+skyFDU.getFullId()+": found "+str(len(xcoords))+" datapoints.")
            #Check coverage fraction
            covfrac = len(xcoords)*100.0/len(ys)
            if (covfrac < minCovFrac):
                print("rectifyProcess::calcLongslitSkylineRectification> Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping this line!")
                self._log.writeLog(__name__, "Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping this line!")
                continue

            #Phase 2 of rejection criteria after lines have been traced
            #Find outliers > 2.5 sigma in peak values and remove them
            #First store first value as xout
            #currXout = xcoords[0]
            peaks = array(peaks)
            peakmed = arraymedian(peaks)
            peaksd = peaks.std()
            b = (peaks > peakmed-2.5*peaksd)*(peaks < peakmed+2.5*peaksd)
            xcoords = array(xcoords)[b]
            ycoords = array(ycoords)[b]
            print("\trejecting outliers (phase 2) - kept "+str(len(xcoords))+" datapoints.")
            self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(xcoords))+" datapoints.", printCaller=False, tabLevel=1)

            #Fit 2nd order order polynomial to datapoints, X = f(Y)
            order = 2
            p = zeros(order+1, float64)
            p[0] = xcoords[0]
            try:
                lsq = leastsq(polyResiduals, p, args=(ycoords,xcoords,order))
            except Exception as ex:
                print("rectifyProcess::calcLongslitSkylineRectification> Could not fit line: "+str(ex))
                self._log.writeLog(__name__, "Could not fit line: "+str(ex))
                continue

            #Compute output offsets and residuals from actual datapoints
            xprime = polyFunction(lsq[0], ycoords, order)
            xresid = xprime-xcoords
            ycen = ysize//2
            currXout = polyFunction(lsq[0], ycen, order) #yout at ycenter
            #Remove outliers and refit
            b = abs(xresid) < xresid.mean()+2.5*xresid.std()
            xcoords = xcoords[b]
            ycoords = ycoords[b]
            print("\trejecting outliers (phase 3). Sigma = "+formatNum(xresid.std())+". Using "+str(len(xcoords))+" datapoints to fit slitlets.")
            self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+formatNum(xresid.std())+". Using "+str(len(xcoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)

            #Check coverage fraction
            covfrac = len(xcoords)*100.0/len(ys)
            if (covfrac >= minCovFrac):
                xin.extend(xcoords)
                yin.extend(ycoords)
                xout.extend([currXout]*len(xcoords))
                nlines += 1
                qavalue = -50000
                #Generate qa data
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    for i in range(len(xcoords)):
                        yval = int(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(yval-1,yval+2):
                            for xi in range(xval-1,xval+2):
                                dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                qaData[yi,xi] = qavalue/((1+dist)**2)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    for i in range(len(xcoords)):
                        yval = int(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(yval-1,yval+2):
                            for xi in range(xval-1,xval+2):
                                dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                qaData[xi,yi] = qavalue/((1+dist)**2)
            print("\tX Center: "+formatNum(currXout)+"\t Yref: "+str(ys[0])+"\t Cov. Frac: "+formatNum(covfrac))
            self._log.writeLog(__name__, "X Center: "+formatNum(currXout)+"\t Yref: "+str(ys[0])+"\t Cov. Frac: "+formatNum(covfrac), printCaller=False, tabLevel=1)

        print("rectifyProcess::calcLongslitSkylineRectification> Successfully traced out "+str(nlines)+ " skylines.  Fitting transformation...")
        self._log.writeLog(__name__, "Successfully traced out "+str(nlines)+ " skylines.  Fitting transformation...")
        #Convert to arrays
        xin = array(xin, dtype=float32)
        yin = array(yin, dtype=float32)
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #Need to swap xin, yin here because surfaceResiduals and surfaceFunction expect out = f(xin, yin) not f(yin, xin).
            tmp = xin
            xin = yin
            yin = tmp
        xout = array(xout)
        #Fit entire skyline transformation
        #Calculate number of terms based on order
        terms = 0
        for j in range(fit_order+2):
            terms+=j
        p = zeros(terms)
        #Initial guess is f(x_in, y_in) = x_in
        p[1] = 1
        #Want to define center to be (0,0) before fitting
        ycen = float(ysize-1)*0.5
        xcen = float(xsize-1)*0.5
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xout -= xcen
            xin -= xcen
            yin -= ycen
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #xin, yin have been swapped
            xout -= xcen
            xin -= ycen
            yin -= xcen
        try:
            lsq = leastsq(surfaceResiduals, p, args=(xin, yin, xout, fit_order))
        except Exception as ex:
            print("rectifyProcess::calcLongslitSkylineRectification> ERROR performing least squares fit: "+str(ex)+"! Skylines will NOT be rectified!")
            self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Skylines will NOT be rectified!", type=fatboyLog.ERROR)
            xcoeffs = array([0, 1, 0])
            if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                xcoeffs = array([0, 0, 1])
            return xcoeffs

        #Compute output offsets and residuals from actual datapoints
        xprime = surfaceFunction(lsq[0], xin, yin, fit_order)
        xresid = xprime-xout
        residmean = xresid.mean()
        residstddev = xresid.std()
        print("\tFound "+str(len(xout))+" datapoints.  Fit: "+formatList(lsq[0]))
        print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
        self._log.writeLog(__name__, "Found "+str(len(xout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
        self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

        print("rectifyProcess::calcLongslitSkylineRectification> Performing iterative sigma clipping to throw away outliers...")
        self._log.writeLog(__name__, "Performing iterative sigma clipping to throw away outliers...")

        #Throw away outliers starting at 2 sigma significance
        sigThresh = 2
        niter = 0
        norig = len(xout)
        bad = where(abs(xresid-residmean)/residstddev > sigThresh)
        while (len(bad[0]) > 0):
            niter += 1
            good = (abs(xresid-residmean)/residstddev <= sigThresh)
            xin = xin[good]
            yin = yin[good]
            xout = xout[good]
            #Refit, use last actual fit coordinates as input guess
            p = lsq[0]
            try:
                lsq = leastsq(surfaceResiduals, p, args=(xin, yin, xout, fit_order))
            except Exception as ex:
                print("rectifyProcess::calcLongslitSkylineRectification> ERROR performing least squares fit: "+str(ex)+"! Skylines will NOT be rectified!")
                self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Skylines will NOT be rectified!", type=fatboyLog.ERROR)
                xcoeffs = array([0, 1, 0])
                if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    xcoeffs = array([0, 0, 1])
                return xcoeffs

            #Compute output offsets and residuals from actual datapoints
            xprime = surfaceFunction(lsq[0], xin, yin, fit_order)
            xresid = xprime-xout
            residmean = xresid.mean()
            residstddev = xresid.std()
            if (niter > 2):
                #Gradually increase sigma threshold
                sigThresh += 0.2
            bad = where(abs(xresid-residmean)/residstddev > sigThresh)
        print("\tAfter "+str(niter)+" passes, kept "+str(len(xout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
        print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
        self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(xout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
        self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

        #Create output qa filename
        qafile = outdir+"/rectified/qa_"+skyFDU.getFullId()

        #Remove existing files if overwrite = yes
        if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            if (os.access(qafile, os.F_OK)):
                os.unlink(qafile)
        #Write out qa file
        if (not os.access(qafile, os.F_OK)):
            skyFDU.tagDataAs("slitqa", qaData)
            skyFDU.writeTo(qafile, tag="slitqa")
            skyFDU.removeProperty("slitqa")
        del qaData

        #Write out more qa data
        qafile = outdir+"/rectified/qa_"+fdu._id+"-skylines.dat"
        f = open(qafile,'w')
        for i in range(len(xout)):
            f.write(str(xin[i])+'\t'+str(yin[i])+'\t'+str(xout[i])+'\t'+str(xprime[i])+'\n')
        f.close()

        xcoeffs = lsq[0]
        return xcoeffs
    #end calcLongslitSkylineRectification

    #Calculate longslit rectification transformation
    #Returns [xcoeffs, ycoeffs] and optionally writes coeffs to file
    def calcLongslitRectification(self, fdu, rctfdus, skyFDU, calibs):
        if ('longslit_continua_frames' in calibs):
            #Use frames passed from XML to trace out continua
            ycoeffs = self.calcLongslitContinuaRectification(fdu, calibs['longslit_continua_frames']).tolist()
            #Free memory
            for calib in calibs['longslit_continua_frames']:
                calib.disable()
        elif (self.getOption("rectify_continua", fdu.getTag()).lower() == "yes"):
            #Use data itself to trace out continuum rectification
            ycoeffs = self.calcLongslitContinuaRectification(fdu, rctfdus).tolist()
        else:
            print("rectifyProcess::calcLongslitRectification> Skipping continua rectification...")
            self._log.writeLog(__name__, "Skipping continua rectification...")
            ycoeffs = [0, 0, 1]
            if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                ycoeffs = [0, 1, 0]

        if (self.getOption("rectify_sky", fdu.getTag()).lower() == "yes"):
            xcoeffs = self.calcLongslitSkylineRectification(fdu, skyFDU).tolist()
        else:
            print("rectifyProcess::calcLongslitRectification> Skipping sky rectification...")
            self._log.writeLog(__name__, "Skipping sky rectification...")
            xcoeffs = [0, 1, 0]
            if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                xcoeffs = [0, 0, 1]
        while (len(ycoeffs) < len(xcoeffs)):
            ycoeffs.append(0)
        while (len(xcoeffs) < len(ycoeffs)):
            xcoeffs.append(0)

        #Order depends on dispersion direction
        rect_coeffs = [xcoeffs, ycoeffs]
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            rect_coeffs = [ycoeffs, xcoeffs]

        #Write coeffs to disk if requested
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/rectified", os.F_OK)):
                os.mkdir(outdir+"/rectified",0o755)
            coeff_file = outdir+"/rectified/rect_coeffs_"+fdu._id+".dat"
            #Overwrite if overwrite_files = yes
            if (os.access(coeff_file, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(coeff_file)
            if (not os.access(coeff_file, os.F_OK)):
                fit_order = int(self.getOption("fit_order", fdu.getTag()))
                sky_fit_order = int(self.getOption("sky_fit_order", fdu.getTag()))
                f = open(coeff_file, 'w')
                f.write('poly ' + str(max(fit_order, sky_fit_order)) + "\n")
                for j in range(len(rect_coeffs[0])):
                    f.write(str(rect_coeffs[0][j])+"\n")
                f.write("\n") #Blank line
                for j in range(len(rect_coeffs[1])):
                    f.write(str(rect_coeffs[1][j])+"\n")
                f.close()

        return rect_coeffs
    #end calcLongslitRectification

    def calculateMOSContinuaTrans(self, fdu, coords, mosMode, calibs):
        if (isinstance(coords, str) and os.access(coords, os.F_OK)):
            #This is a coord_list filename
            coord_list = loadtxt(coords)
            if (mosMode == "use_slitpos"):
                (xin, yin, yout, xslitin) = (coord_list[:,0], coord_list[:,1], coord_list[:,2], coord_list[:,3])
            elif (mosMode == "independent_slitlets"):
                (xin, yin, yout, islit, iseg) = (coord_list[:,0], coord_list[:,1], coord_list[:,2], coord_list[:,3])
                if (coord_list.shape[1] > 4):
                    iseg = coord_list[:4]
                else:
                    iseg = zeros(len(islit))
            elif (mosMode == "whole_chip"):
                (xin, yin, yout) = (coord_list[:,0], coord_list[:,1], coord_list[:,2])
            else:
                print("rectifyProcess::calculateMOSContinuaTrans> ERROR: Invalid mos_mode: "+mosMode)
                self._log.writeLog(__name__, "Invalid mos_mode: "+mosMode, type=fatboyLog.ERROR)
        elif (isinstance(coords, tuple)):
            #This is a tuple returned from traceMOSContinuaRectification
            if (mosMode == "use_slitpos"):
                (xin, yin, yout, xslitin) = coords
            elif (mosMode == "independent_slitlets"):
                (xin, yin, yout, islit, iseg) = coords
            elif (mosMode == "whole_chip"):
                (xin, yin, yout) = coords
        else:
            print("rectifyProcess::calculateMOSContinuaTrans> ERROR: Invalid arguments received.  coords must be a string or a tuple.")
            self._log.writeLog(__name__, "Invalid arguments received.  coords must be a string or a tuple.", type=fatboyLog.ERROR)
            return None

        fit_order = int(self.getOption("mos_fit_order", fdu.getTag()))
        maxSlitWidth = float(self.getOption("mos_max_slit_width", fdu.getTag()))
        n_segments = int(self.getOption("n_segments", fdu.getTag()))
        useCenterAsZero = False
        if (self.getOption("use_zero_as_center_fitting", fdu.getTag()).lower() == "yes"):
            useCenterAsZero = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        if (not calibs['slitmask'].hasProperty("nslits")):
            calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
        nslits = calibs['slitmask'].getProperty("nslits")
        if (calibs['slitmask'].hasProperty("regions")):
            (sylo, syhi, slitx, slitw) = calibs['slitmask'].getProperty("regions")
        else:
            #Get region file for this FDU
            if (fdu.hasProperty("region_file")):
                regFile = fdu.getProperty("region_file")
            else:
                regFile = self.getCalib("region_file", fdu.getTag())
            #Check that region file exists
            if (regFile is None or not os.access(regFile, os.F_OK)):
                print("rectifyProcess::calculateMOSContinuaTrans> No region file given.  Calculating regions from slitmask...")
                self._log.writeLog(__name__, "No region file given.  Calculating regions from slitmask...")
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    cut1d = calibs['slitmask'].getData()[:,xsize//2].astype(float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    cut1d = calibs['slitmask'].getData()[xsize//2,:].astype(float64)
                #detect nonzero points in 1-d cut to find regions
                slitlets = extractNonzeroRegions(cut1d, 10) #min_width = 10
                if (slitlets is None):
                    print("rectifyProcess::calculateMOSContinuaTrans> ERROR: Could not find region file or calculate regions associated with "+fdu.getFullId()+"! Discarding Image!")
                    self._log.writeLog(__name__, "Could not find region file or calculate regions associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
                sylo = slitlets[:,0]
                syhi = slitlets[:,1]
                slitx = array([xsize//2]*len(sylo))
                slitw = array([3]*len(sylo))
            else:
                #Read region file
                if (regFile.endswith(".reg")):
                    (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".txt")):
                    (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".xml")):
                    (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                else:
                    print("rectifyProcess::calculateMOSContinuaTrans> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                    self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
            calibs['slitmask'].setProperty("regions", (sylo, syhi, slitx, slitw))

        ytransData = zeros(calibs['slitmask'].getData().shape, dtype=float32)
        #Use GPU to calculuate xind
        xind = arange(xsize, dtype=float32)
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            xind = xind.reshape((xsize,1))
        if (self._fdb.getGPUMode()):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                yind = calcYin(xsize, ysize)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                yind = calcXin(ysize, xsize)
        else:
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                yind = (arange(xsize*ysize).reshape(ysize,xsize) // xsize).astype(float32)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                yind = arange(xsize*ysize, dtype=float32).reshape(xsize,ysize) % ysize
        #+1 because region files start at corner=(1,1) not (0,0)
        rylo = zeros(len(slitx), dtype=float32)+1.
        ryhi = zeros(len(slitx), dtype=float32)+1.

        if (mosMode == "use_slitpos"):
            #Convert to arrays
            xin = array(xin)
            yin = array(yin)
            yout = array(yout)
            xslitin = array(xslitin)
            if (useCenterAsZero):
                #Subtract x0
                x0 = xsize//2
                xin -= x0
                xind -= x0

            #Fit 3d surface to data
            terms = 0
            nterms = 0
            for i in range(fit_order+2):
                nterms+=i
                terms+=nterms
            p = zeros(terms)
            p[2] = 1
            try:
                lsq = leastsq(surface3dResiduals, p, args=(xin, yin, xslitin, yout, fit_order))
            except Exception as ex:
                print("rectifyProcess::calculateMOSContinuaTrans> ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return None

            coeffs = lsq[0]

            #Compute output offsets and residuals from actual datapoints
            yprime = surface3dFunction(lsq[0], xin, yin, xslitin, fit_order)
            yresid = yprime-yout
            residmean = yresid.mean()
            residstddev = yresid.std()
            print("\tUsing "+str(len(yout))+" datapoints.  Fit: "+formatList(lsq[0]))
            print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
            self._log.writeLog(__name__, "Using "+str(len(yout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
            self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

            #Throw away outliers starting at 2 sigma significance
            sigThresh = 2
            niter = 0
            norig = len(yout)
            bad = where(abs(yresid-residmean)/residstddev > sigThresh)
            while (len(bad[0]) > 0):
                niter += 1
                good = (abs(yresid-residmean)/residstddev <= sigThresh)
                xin = xin[good]
                yin = yin[good]
                yout = yout[good]
                xslitin = xslitin[good]
                #Refit, use last actual fit coordinates as input guess
                p = coeffs
                try:
                    lsq = leastsq(surface3dResiduals, p, args=(xin, yin, xslitin, yout, fit_order))
                except Exception as ex:
                    print("rectifyProcess::calculateMOSContinuaTrans> ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!")
                    self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None

                coeffs = lsq[0]
                #Compute output offsets and residuals from actual datapoints
                yprime = surface3dFunction(lsq[0], xin, yin, xslitin, fit_order)
                yresid = yprime-yout
                residmean = yresid.mean()
                residstddev = yresid.std()
                if (niter > 2):
                    #Gradually increase sigma threshold
                    sigThresh += 0.2
                bad = where(abs(yresid-residmean)/residstddev > sigThresh)
            print("\tAfter "+str(niter)+" passes, kept "+str(len(yout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
            print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
            self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(yout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
            self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)


            #Create "z" indicies = slitx for each slitlet
            z = zeros(calibs['slitmask'].getData().shape, dtype=float32)
            for slitidx in range(nslits):
                z[calibs['slitmask'].getData() == (slitidx+1)] = slitx[slitidx]

            if (self._fdb.getGPUMode()):
                #Use GPU for ytransData calculation, CPU for others
                ytransData = calcTrans3d(xind, yind, z, coeffs, fit_order)
            i = 0
            for x in range(fit_order+1):
                for l in range(1,x+2):
                    for k in range(1,l+1):
                        if (useCenterAsZero):
                            rylo+=coeffs[i]*(slitx-x0)**(x-l+1)*sylo**(l-k)*slitx**(k-1)
                            ryhi+=coeffs[i]*(slitx-x0)**(x-l+1)*syhi**(l-k)*slitx**(k-1)
                        else:
                            rylo+=coeffs[i]*slitx**(x-l+1)*sylo**(l-k)*slitx**(k-1)
                            ryhi+=coeffs[i]*slitx**(x-l+1)*syhi**(l-k)*slitx**(k-1)
                        if (not self._fdb.getGPUMode()):
                            ytransData+=coeffs[i]*xind**(x-l+1)*yind**(l-k)*z**(k-1)
                        i+=1
            if (not self._fdb.getGPUMode()):
                #Zero out anything not in a slitlet
                ytransData *= (z != 0)
        elif (mosMode == "whole_chip"):
            #Convert to arrays
            xin = array(xin)
            yin = array(yin)
            yout = array(yout)
            if (useCenterAsZero):
                #Subtract x0
                x0 = xsize//2
                xin -= x0
                xind -= x0

            terms = 0
            for j in range(fit_order+2):
                terms+=j
            p = zeros(terms)
            #Initial guess is f(x_in, y_in) = x_in
            p[1] = 1

            if (len(xin) < 10):
                print("rectifyProcess::calculateMOSContinuaTrans> ERROR: Could not find continua to trace out! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Could not find continua to trace out! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return None

            #Fit surface to data
            try:
                lsq = leastsq(surfaceResiduals, p, args=(xin, yin, yout, fit_order))
            except Exception as ex:
                print("rectifyProcess::calculateMOSContinuaTrans> ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return None

            coeffs = lsq[0]

            #Compute output offsets and residuals from actual datapoints
            yprime = surfaceFunction(lsq[0], xin, yin, fit_order)
            yresid = yprime-yout
            residmean = yresid.mean()
            residstddev = yresid.std()
            print("\tUsing "+str(len(yout))+" datapoints.  Fit: "+formatList(lsq[0]))
            print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
            self._log.writeLog(__name__, "Using "+str(len(yout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
            self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

            #Throw away outliers starting at 2 sigma significance
            sigThresh = 2
            niter = 0
            norig = len(yout)
            bad = where(abs(yresid-residmean)/residstddev > sigThresh)
            while (len(bad[0]) > 0):
                niter += 1
                good = (abs(yresid-residmean)/residstddev <= sigThresh)
                xin = xin[good]
                yin = yin[good]
                yout = yout[good]
                #Refit, use last actual fit coordinates as input guess
                p = coeffs
                try:
                    lsq = leastsq(surfaceResiduals, p, args=(xin, yin, yout, fit_order))
                except Exception as ex:
                    print("rectifyProcess::calculateMOSContinuaTrans> ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!")
                    self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None

                coeffs = lsq[0]
                #Compute output offsets and residuals from actual datapoints
                yprime = surfaceFunction(lsq[0], xin, yin, fit_order)
                yresid = yprime-yout
                residmean = yresid.mean()
                residstddev = yresid.std()
                if (niter > 2):
                    #Gradually increase sigma threshold
                    sigThresh += 0.2
                bad = where(abs(yresid-residmean)/residstddev > sigThresh)
            print("\tAfter "+str(niter)+" passes, kept "+str(len(yout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
            print("\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
            self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(yout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
            self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=1)

            ytransData = surfaceFunction(lsq[0], xind, yind, fit_order)
            i = 0
            for j in range(fit_order+1):
                for l in range(j+1):
                    if (useCenterAsZero):
                        rylo += lsq[0][i]*(slitx-x0)**(j-l)*sylo**l
                        ryhi += lsq[0][i]*(slitx-x0)**(j-l)*syhi**l
                    else:
                        rylo += lsq[0][i]*slitx**(j-l)*sylo**l
                        ryhi += lsq[0][i]*slitx**(j-l)*syhi**l
                    i+=1
        elif (mosMode == "independent_slitlets"):
            #Convert to arrays
            xin = array(xin)
            yin = array(yin)
            yout = array(yout)
            islit = array(islit)
            iseg = array(iseg)

            #Use helper method to all ylo, yhi for each slit in each frame
            (ylos, yhis, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)

            for slitidx in range(nslits):
                if (slitw[slitidx] > maxSlitWidth):
                    print("\tSlit "+str(slitidx+1)+" is a guide star box.  Skipping!")
                    self._log.writeLog(__name__, "Slit "+str(slitidx+1)+" is a guide star box.  Skipping!", printCaller=False, tabLevel=1)
                    continue
                ylo = ylos[slitidx]
                yhi = yhis[slitidx]

                #Loop over segments here
                for seg in range(n_segments):
                    seg_name = ""
                    if (n_segments > 1):
                        seg_name = "segment "+str(seg)+" of "
                    xstride = xsize//n_segments
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)

                    #Find the data corresponding to this slit and take 1-d cut
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        currMask = calibs['slitmask'].getData()[ylo:yhi,sxlo:sxhi] == (slitidx+1)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        currMask = calibs['slitmask'].getData()[sxlo:sxhi,ylo:yhi] == (slitidx+1)

                    b = (islit == slitidx+1)*(iseg == seg)
                    if (b.sum() == 0):
                        #No data fit for this slitlet.
                        print("rectifyProcess::calculateMOSContinuaTrans> Warning: No data found to rectify "+seg_name+"slitlet "+str(slitidx+1)+"!  This slitlet will not be rectified!")
                        self._log.writeLog(__name__, "No data found to rectify "+seg_name+"slitlet "+str(slitidx+1)+"!  This slitlet will not be rectified!", type=fatboyLog.WARNING)
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            ytransData[ylo:yhi,sxlo:sxhi][currMask] = yind[ylo:yhi,sxlo:sxhi][currMask]
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            ytransData[sxlo:sxhi,ylo:yhi][currMask] = yind[sxlo:sxhi,ylo:yhi][currMask]
                        rylo[slitidx] = sylo[slitidx]
                        ryhi[slitidx] = syhi[slitidx]
                        continue

                    slitxin = xin[b]
                    slityin = yin[b]
                    slityout = yout[b]
                    if (useCenterAsZero):
                        #Subtract x0
                        x0 = slitxin[0]
                        slitxin -= x0

                    #Fit transformation to line in this slitlet/segment
                    #Calculate number of terms based on order
                    terms = 0
                    for j in range(fit_order+2):
                        terms+=j
                    p = zeros(terms)
                    #Initial guess is f(x_in, y_in) = x_in
                    p[1] = 1
                    try:
                        print ("P",p)
                        print (len(slitxin), len(slityin), len(slityout), fit_order, terms)
                        print ("SLITIDX", slitidx+1, seg)
                        lsq = leastsq(surfaceResiduals, p, args=(slitxin, slityin, slityout, fit_order))
                    except Exception as ex:
                        print("rectifyProcess::calculateMOSContinuaTrans> ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!")
                        self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                        #disable this FDU
                        fdu.disable()
                        return None

                    coeffs = lsq[0]

                    #Compute output offsets and residuals from actual datapoints
                    slityprime = surfaceFunction(lsq[0], slitxin, slityin, fit_order)
                    yresid = slityprime-slityout
                    residmean = yresid.mean()
                    residstddev = yresid.std()
                    print("\t"+seg_name+"Slit "+str(slitidx+1)+":  using "+str(len(slityout))+" datapoints.  Fit: "+formatList(lsq[0]))
                    print("\t\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
                    self._log.writeLog(__name__, seg_name+"Slit "+str(slitidx+1)+": using "+str(len(slityout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
                    self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=2)

                    #Throw away outliers starting at 2 sigma significance
                    sigThresh = 2
                    niter = 0
                    norig = len(slityout)
                    bad = where(abs(yresid-residmean)/residstddev > sigThresh)
                    while (len(bad[0]) > 0):
                        niter += 1
                        good = (abs(yresid-residmean)/residstddev <= sigThresh)
                        slitxin = slitxin[good]
                        slityin = slityin[good]
                        slityout = slityout[good]
                        #Refit, use last actual fit coordinates as input guess
                        p = coeffs
                        try:
                            lsq = leastsq(surfaceResiduals, p, args=(slitxin, slityin, slityout, fit_order))
                        except Exception as ex:
                            print("rectifyProcess::calculateMOSContinuaTrans> ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!")
                            self._log.writeLog(__name__, "ERROR performing least squares fit: "+str(ex)+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                            #disable this FDU
                            fdu.disable()
                            return None

                        coeffs = lsq[0]
                        #Compute output offsets and residuals from actual datapoints
                        slityprime = surfaceFunction(lsq[0], slitxin, slityin, fit_order)
                        yresid = slityprime-slityout
                        residmean = yresid.mean()
                        residstddev = yresid.std()
                        if (niter > 2):
                            #Gradually increase sigma threshold
                            sigThresh += 0.2
                        bad = where(abs(yresid-residmean)/residstddev > sigThresh)
                    print("\t\tAfter "+str(niter)+" passes, kept "+str(len(slityout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
                    print("\t\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
                    self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(slityout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
                    self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=2)
                    #f = open('rect_cont_sigma.txt','a')
                    #f.write(str(slitidx)+'\t'+str(seg)+'\t'+str(fit_order)+'\t'+str(len(slityout))+'\t'+str(norig)+'\t'+str(residstddev)+'\n')
                    #f.close()

                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        if (useCenterAsZero):
                            ytransData[ylo:yhi,sxlo:sxhi][currMask] = surfaceFunction(coeffs, xind[sxlo:sxhi]-x0, yind[ylo:yhi,sxlo:sxhi], fit_order)[currMask]
                        else:
                            ytransData[ylo:yhi,sxlo:sxhi][currMask] = surfaceFunction(coeffs, xind[sxlo:sxhi], yind[ylo:yhi,sxlo:sxhi], fit_order)[currMask]
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        if (useCenterAsZero):
                            ytransData[sxlo:sxhi,ylo:yhi][currMask] = surfaceFunction(coeffs, xind[sxlo:sxhi]-x0, yind[sxlo:sxhi,ylo:yhi], fit_order)[currMask]
                        else:
                            ytransData[sxlo:sxhi,ylo:yhi][currMask] = surfaceFunction(coeffs, xind[sxlo:sxhi], yind[sxlo:sxhi,ylo:yhi], fit_order)[currMask]

                    i = 0
                    seg_rylo = 1 #+1 because region files start at corner=(1,1) not (0,0)
                    seg_ryhi = 1
                    for j in range(fit_order+1):
                        for l in range(j+1):
                            if (useCenterAsZero):
                                seg_rylo += lsq[0][i]*(slitx[slitidx]-x0)**(j-l)*sylo[slitidx]**l
                                seg_ryhi += lsq[0][i]*(slitx[slitidx]-x0)**(j-l)*syhi[slitidx]**l
                            else:
                                seg_rylo += lsq[0][i]*(slitx[slitidx])**(j-l)*sylo[slitidx]**l
                                seg_ryhi += lsq[0][i]*(slitx[slitidx])**(j-l)*syhi[slitidx]**l
                            i+=1
                    if (seg == 0):
                        rylo[slitidx] = seg_rylo
                        ryhi[slitidx] = seg_ryhi
                    else:
                        rylo[slitidx] = min(rylo[slitidx], seg_rylo)
                        ryhi[slitidx] = max(ryhi[slitidx], seg_ryhi)

        #create fatboySpecCalib and add to calibs dict
        ytrans_name = "ytrans_rect"
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #for vertical dispersion, this is actually ytrans
            ytrans_name = "xtrans_rect"
        ytrans_rect = fatboySpecCalib(self._pname, ytrans_name, fdu, data=ytransData, tagname=fdu._id, log=self._log)
        ytrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
        ytrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
        self._fdb.appendCalib(ytrans_rect)

        #Update slitmask properties to save rylo and ryhi for use later in removing guide star boxes from slitmask
        if ((ytransData != 0).sum() > 0):
            ytrans_min = floor(ytransData[ytransData != 0].min())
            if (ytrans_min < 0):
                rylo -= ytrans_min
                ryhi -= ytrans_min
        calibs['slitmask'].setProperty("rylo", rylo)
        calibs['slitmask'].setProperty("ryhi", ryhi)

        #Get rid of guide star boxes and any entirely negative slits (shouldn't happen)
        b = logical_and(slitw < maxSlitWidth, ryhi > 0)
        rylo = rylo[b]
        ryhi = ryhi[b]
        slitx = slitx[b]
        slitw = slitw[b]

        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/rectified", os.F_OK)):
            os.mkdir(outdir+"/rectified",0o755)

        #Output new region file
        rctRegFile = outdir+"/rectified/region_"+fdu._id+".reg"
        f = open(rctRegFile,'w')
        f.write('# Region file format: DS9 version 3.0\n')
        f.write('global color=green select=1 edit=1 move=1 delete=1 include=1 fixed=0\n')
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            for i in range(len(rylo)):
                f.write('image;box('+str(slitx[i])+','+str((rylo[i]+ryhi[i])/2)+','+str(slitw[i])+','+str(ryhi[i]-rylo[i])+')\n')
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            for i in range(len(rylo)):
                f.write('image;box('+str((rylo[i]+ryhi[i])/2)+','+str(slitx[i])+','+str(ryhi[i]-rylo[i])+','+str(slitw[i])+')\n')
        f.close()

        #Output new XML region file
        rctRegFile = outdir+"/rectified/region_"+fdu._id+".xml"
        f = open(rctRegFile,'w')
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<fatboy>\n')
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            for i in range(len(rylo)):
                f.write('<slitlet xcenter="'+str(slitx[i])+'" ycenter="'+str((rylo[i]+ryhi[i])/2)+'" width="'+str(slitw[i])+'" height="'+str(ryhi[i]-rylo[i])+'"/>\n')
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            for i in range(len(rylo)):
                f.write('<slitlet xcenter="'+str((rylo[i]+ryhi[i])/2)+'" ycenter="'+str(slitx[i])+'" width="'+str(ryhi[i]-rylo[i])+'" height="'+str(slitw[i])+'"/>\n')
        f.write('</fatboy>\n')
        f.close()

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Create output filename
            ytransfile = outdir+"/rectified/"+ytrans_name+"_"+fdu._id+".fits"

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes" and os.access(ytransfile, os.F_OK)):
                os.unlink(ytransfile)

            #Write out ytrans
            if (not os.access(ytransfile, os.F_OK)):
                ytrans_rect.writeTo(ytransfile)

        return ytrans_rect
    #end calculateMOSContinuaTrans

    def calculateMOSSkylineTrans(self, fdu, coords, calibs):
        if (isinstance(coords, str) and os.access(coords, os.F_OK)):
            #This is a coord_list filename
            coord_list = loadtxt(coords)
            (xin, yin, xout, islit) = (coord_list[:,0], coord_list[:,1], coord_list[:,2], coord_list[:3])
            if (coord_list.shape[1] > 5):
                iseg = coord_list[:5]
            else:
                iseg = zeros(len(islit))
        elif (isinstance(coords, tuple)):
            #This is a tuple returned from traceMOSSkylineRectification
            (xin, yin, xout, islit, iseg) = coords
        else:
            print("rectifyProcess::calculateMOSSkylineTrans> ERROR: Invalid arguments received.  coords must be a string or a tuple.")
            self._log.writeLog(__name__, "Invalid arguments received.  coords must be a string or a tuple.", type=fatboyLog.ERROR)
            return None

        #Convert to arrays
        xin = array(xin)
        yin = array(yin)
        xout = array(xout)
        islit = array(islit)
        iseg = array(iseg)

        fit_order = int(self.getOption("mos_sky_fit_order", fdu.getTag()))
        maxSlitWidth = float(self.getOption("mos_max_slit_width", fdu.getTag()))
        n_segments = int(self.getOption("n_segments", fdu.getTag()))
        useCenterAsZero = False
        if (self.getOption("use_zero_as_center_fitting", fdu.getTag()).lower() == "yes"):
            useCenterAsZero = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        if (not calibs['slitmask'].hasProperty("nslits")):
            calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
        nslits = calibs['slitmask'].getProperty("nslits")
        if (calibs['slitmask'].hasProperty("regions")):
            (sylo, syhi, slitx, slitw) = calibs['slitmask'].getProperty("regions")
        else:
            #Get region file for this FDU
            if (fdu.hasProperty("region_file")):
                regFile = fdu.getProperty("region_file")
            else:
                regFile = self.getCalib("region_file", fdu.getTag())
            #Check that region file exists
            if (regFile is None or not os.access(regFile, os.F_OK)):
                print("rectifyProcess::calculateMOSSkylineTrans> No region file given.  Calculating regions from slitmask...")
                self._log.writeLog(__name__, "No region file given.  Calculating regions from slitmask...")
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    cut1d = calibs['slitmask'].getData()[:,xsize//2].astype(float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    cut1d = calibs['slitmask'].getData()[xsize//2,:].astype(float64)
                #detect nonzero points in 1-d cut to find regions
                slitlets = extractNonzeroRegions(cut1d, 10) #min_width = 10
                if (slitlets is None):
                    print("rectifyProcess::calculateMOSSkylineTrans> ERROR: Could not find region file or calculate regions associated with "+fdu.getFullId()+"! Discarding Image!")
                    self._log.writeLog(__name__, "Could not find region file or calculate regions associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
                sylo = slitlets[:,0]
                syhi = slitlets[:,1]
                slitx = array([xsize//2]*len(sylo))
                slitw = array([3]*len(sylo))
            else:
                #Read region file
                if (regFile.endswith(".reg")):
                    (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".txt")):
                    (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".xml")):
                    (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                else:
                    print("rectifyProcess::calculateMOSSkylineTrans> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                    self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
            calibs['slitmask'].setProperty("regions", (sylo, syhi, slitx, slitw))

        #Use GPU to calculuate xind
        yind = arange(ysize, dtype=float32).reshape(ysize,1)
        if (self._fdb.getGPUMode()):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                xind = calcXin(xsize, ysize)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                xind = calcYin(ysize, xsize)
        else:
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                xind = arange(xsize*ysize, dtype=float32).reshape(ysize,xsize) % xsize
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                xind = (arange(xsize*ysize).reshape(xsize,ysize) // ysize).astype(float32)

        #Use helper method to all ylo, yhi for each slit in each frame
        (ylos, yhis, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)


        xtransData = zeros(calibs['slitmask'].getData().shape, dtype=float32)

        for slitidx in range(nslits):
            if (slitw[slitidx] > maxSlitWidth):
                print("\tSlit "+str(slitidx+1)+" is a guide star box.  Skipping!")
                self._log.writeLog(__name__, "Slit "+str(slitidx+1)+" is a guide star box.  Skipping!", printCaller=False, tabLevel=1)
                continue
            ylo = ylos[slitidx]
            yhi = yhis[slitidx]

            #Loop over segments here
            for seg in range(n_segments):
                seg_name = ""
                if (n_segments > 1):
                    seg_name = "segment "+str(seg)+" of "
                xstride = xsize//n_segments
                sxlo = xstride*seg
                sxhi = xstride*(seg+1)

                #Find the data corresponding to this slit and take 1-d cut
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    currMask = calibs['slitmask'].getData()[ylo:yhi+1,sxlo:sxhi] == (slitidx+1)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    currMask = calibs['slitmask'].getData()[sxlo:sxhi,ylo:yhi+1] == (slitidx+1)

                b = (islit == slitidx+1)*(iseg == seg)
                if (b.sum() == 0):
                    #No data fit for this slitlet.
                    print("rectifyProcess::calculateMOSSkylineTrans> Warning: No data found to rectify skylines in "+seg_name+"slitlet "+str(slitidx+1)+"!  This slitlet will not be rectified!")
                    self._log.writeLog(__name__, "No data found to rectify skylines in "+seg_name+"slitlet "+str(slitidx+1)+"!  This slitlet will not be rectified!", type=fatboyLog.WARNING)
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        xtransData[ylo:yhi+1,sxlo:sxhi][currMask] = xind[ylo:yhi+1,sxlo:sxhi][currMask]
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        xtransData[sxlo:sxhi,ylo:yhi+1][currMask] = xind[sxlo:sxhi,ylo:yhi+1][currMask]
                    continue

                slitxin = xin[b]
                slityin = yin[b]
                slitxout = xout[b]
                if (useCenterAsZero):
                    #Subtract y0
                    y0 = slityin[0]
                    slityin -= y0

                #Fit transformation to line in this slitlet
                #Calculate number of terms based on order
                terms = 0
                for j in range(fit_order+2):
                    terms+=j
                p = zeros(terms)
                #Initial guess is f(x_in, y_in) = x_in
                p[1] = 1
                try:
                    lsq = leastsq(surfaceResiduals, p, args=(slitxin, slityin, slitxout, fit_order))
                except Exception as ex:
                    print("rectifyProcess::calculateMOSSkylineTrans> Warning: Error performing least squares fit: "+str(ex)+"! This slitlet will not be rectified!")
                    self._log.writeLog(__name__, "Error performing least squares fit: "+str(ex)+"! This slitlet will not be rectified!", type=fatboyLog.WARNING)
                    continue

                coeffs = lsq[0]

                #Compute output offsets and residuals from actual datapoints
                slitxprime = surfaceFunction(lsq[0], slitxin, slityin, fit_order)
                xresid = slitxprime-slitxout
                residmean = xresid.mean()
                residstddev = xresid.std()
                print("\t"+seg_name+"Slit "+str(slitidx+1)+":  using "+str(len(slitxout))+" datapoints.  Fit: "+formatList(lsq[0]))
                print("\t\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
                self._log.writeLog(__name__, seg_name+"Slit "+str(slitidx+1)+": using "+str(len(slitxout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=1)
                self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=2)

                #Throw away outliers starting at 2 sigma significance
                sigThresh = 2
                niter = 0
                norig = len(slitxout)
                bad = where(abs(xresid-residmean)/residstddev > sigThresh)
                while (len(bad[0]) > 0):
                    niter += 1
                    good = (abs(xresid-residmean)/residstddev <= sigThresh)
                    slitxin = slitxin[good]
                    slityin = slityin[good]
                    slitxout = slitxout[good]
                    #Refit, use last actual fit coordinates as input guess
                    p = coeffs
                    try:
                        lsq = leastsq(surfaceResiduals, p, args=(slitxin, slityin, slitxout, fit_order))
                    except Exception as ex:
                        print("rectifyProcess::calculateMOSSkylineTrans> Warning: Error performing least squares fit: "+str(ex)+"! This slitlet will not be rectified!")
                        self._log.writeLog(__name__, "Error performing least squares fit: "+str(ex)+"! This slitlet will not be rectified!", type=fatboyLog.WARNING)
                        continue

                    coeffs = lsq[0]
                    #Compute output offsets and residuals from actual datapoints
                    slitxprime = surfaceFunction(lsq[0], slitxin, slityin, fit_order)
                    xresid = slitxprime-slitxout
                    residmean = xresid.mean()
                    residstddev = xresid.std()
                    if (niter > 2):
                        #Gradually increase sigma threshold
                        sigThresh += 0.2
                    bad = where(abs(xresid-residmean)/residstddev > sigThresh)
                print("\t\tAfter "+str(niter)+" passes, kept "+str(len(slitxout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
                print("\t\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
                self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(slitxout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
                self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=2)
                #f = open('rect_lines_sigma.txt','a')
                #f.write(str(slitidx+1)+'\t'+str(seg)+'\t'+str(fit_order)+'\t'+str(len(slitxout))+'\t'+str(norig)+'\t'+str(residstddev)+'\n')
                #f.close()

                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (useCenterAsZero):
                        xtransData[ylo:yhi+1,sxlo:sxhi][currMask] = surfaceFunction(coeffs, xind[ylo:yhi+1,sxlo:sxhi], yind[ylo:yhi+1,:]-y0, fit_order)[currMask]
                    else:
                        xtransData[ylo:yhi+1,sxlo:sxhi][currMask] = surfaceFunction(coeffs, xind[ylo:yhi+1,sxlo:sxhi], yind[ylo:yhi+1,:], fit_order)[currMask]
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    #yind is yshape x 1 array
                    if (useCenterAsZero):
                        xtransData[sxlo:sxhi,ylo:yhi+1][currMask] = surfaceFunction(coeffs, xind[sxlo:sxhi,ylo:yhi+1], yind[ylo:yhi+1,0]-y0, fit_order)[currMask]
                    else:
                        xtransData[sxlo:sxhi,ylo:yhi+1][currMask] = surfaceFunction(coeffs, xind[sxlo:sxhi,ylo:yhi+1], yind[ylo:yhi+1,0], fit_order)[currMask]

        #create fatboySpecCalib and add to calibs dict
        xtrans_name = "xtrans_rect"
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #for vertical dispersion, this is actually ytrans
            xtrans_name = "ytrans_rect"
        xtrans_rect = fatboySpecCalib(self._pname, xtrans_name, fdu, data=xtransData, tagname=fdu._id, log=self._log)
        xtrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
        xtrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
        self._fdb.appendCalib(xtrans_rect)

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/rectified", os.F_OK)):
                os.mkdir(outdir+"/rectified",0o755)
            #Create output filename
            xtransfile = outdir+"/rectified/"+xtrans_name+"_"+fdu._id+".fits"

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes" and os.access(xtransfile, os.F_OK)):
                os.unlink(xtransfile)

            #Write out xtrans
            if (not os.access(xtransfile, os.F_OK)):
                xtrans_rect.writeTo(xtransfile)

        return xtrans_rect
    #end calculateMOSSkylineTrans

    #Calculate MOS rectification transformation
    #Adds 'xtrans_rect' and 'ytrans_rect' to calibs and returns calibs
    def calcMOSRectification(self, fdu, rctfdus, skyFDU, calibs):
        mosMode = self.getOption("mos_mode", fdu.getTag())
        if (not 'xtrans_rect' in calibs):
            calibs['xtrans_rect'] = None
        if (not 'ytrans_rect' in calibs):
            calibs['ytrans_rect'] = None

        if (not 'slitmask' in calibs):
            print("rectifyProcess::calcMOSRectification> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to rectify!")
            self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to rectify!", type=fatboyLog.ERROR)
            return calibs

        #First look for 'xtrans_coord_list' and 'ytrans_coord_list' property or calib
        if (fdu.hasProperty("xtrans_coord_list")):
            xtransCoordList = fdu.getProperty("xtrans_coord_list")
        else:
            xtransCoordList = self.getCalib("xtrans_coord_list", fdu.getTag())
        if (fdu.hasProperty("ytrans_coord_list")):
            ytransCoordList = fdu.getProperty("ytrans_coord_list")
        else:
            ytransCoordList = self.getCalib("ytrans_coord_list", fdu.getTag())

        #First calculate continuum rectification
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL and calibs['ytrans_rect'] is not None):
            print("rectifyProcess::calcMOSRectification> Using previous continuum transformation "+calibs['ytrans_rect'].getFullId())
            self._log.writeLog(__name__, "Using previous continuum transformation "+calibs['ytrans_rect'].getFullId())
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL and calibs['xtrans_rect'] is not None):
            print("rectifyProcess::calcMOSRectification> Using previous continuum transformation "+calibs['xtrans_rect'].getFullId())
            self._log.writeLog(__name__, "Using previous continuum transformation "+calibs['xtrans_rect'].getFullId())
        elif (fdu.dispersion == fdu.DISPERSION_HORIZONTAL and ytransCoordList is not None and os.access(ytransCoordList, os.F_OK)):
            calibs['ytrans_rect'] = self.calculateMOSContinuaTrans(fdu, ytransCoordList, mosMode, calibs)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL and xtransCoordList is not None and os.access(xtransCoordList, os.F_OK)):
            calibs['xtrans_rect'] = self.calculateMOSContinuaTrans(fdu, xtransCoordList, mosMode, calibs)
        elif (self.getOption("rectify_continua", fdu.getTag()).lower() == "yes"):
            if ('mos_continua_frames' in calibs):
                #Use frames passed from XML to trace out continua
                coords = self.traceMOSContinuaRectification(fdu, calibs['mos_continua_frames'], mosMode, calibs)
                #Free memory
                for calib in calibs['mos_continua_frames']:
                    calib.disable()
            else:
                #Use data frames themselves to trace out continua
                coords = self.traceMOSContinuaRectification(fdu, rctfdus, mosMode, calibs)
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                calibs['ytrans_rect'] = self.calculateMOSContinuaTrans(fdu, coords, mosMode, calibs)
            else:
                calibs['xtrans_rect'] = self.calculateMOSContinuaTrans(fdu, coords, mosMode, calibs)
        else:
            print("rectifyProcess::calcMOSRectification> Skipping continua rectification...")
            self._log.writeLog(__name__, "Skipping continua rectification...")
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (self._fdb.getGPUMode()):
                    ytransData = calcYin(xsize, ysize)
                else:
                    ytransData = (arange(xsize*ysize).reshape(ysize,xsize) // xsize).astype(float32)
                ytrans_rect = fatboySpecCalib(self._pname, "ytrans_rect", fdu, data=ytransData, tagname=fdu._id, log=self._log)
                ytrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
                ytrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
                self._fdb.appendCalib(ytrans_rect)
                calibs['ytrans_rect'] = ytrans_rect
            else:
                if (self._fdb.getGPUMode()):
                    xtransData = calcXin(xsize, ysize)
                else:
                    xtransData = arange(xsize*ysize, dtype=float32).reshape(ysize,xsize) % xsize
                xtrans_rect = fatboySpecCalib(self._pname, "xtrans_rect", fdu, data=xtransData, tagname=fdu._id, log=self._log)
                xtrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
                xtrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
                self._fdb.appendCalib(xtrans_rect)
                calibs['xtrans_rect'] = xtrans_rect

        #Second calculate skyline rectification
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL and calibs['xtrans_rect'] is not None):
            print("rectifyProcess::calcMOSRectification> Using previous line transformation "+calibs['xtrans_rect'].getFullId())
            self._log.writeLog(__name__, "Using previous line transformation "+calibs['xtrans_rect'].getFullId())
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL and calibs['ytrans_rect'] is not None):
            print("rectifyProcess::calcMOSRectification> Using previous line transformation "+calibs['ytrans_rect'].getFullId())
            self._log.writeLog(__name__, "Using previous line transformation "+calibs['ytrans_rect'].getFullId())
        elif (fdu.dispersion == fdu.DISPERSION_HORIZONTAL and xtransCoordList is not None and os.access(xtransCoordList, os.F_OK)):
            calibs['xtrans_rect'] = self.calculateMOSSkylineTrans(fdu, xtransCoordList, calibs)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL and ytransCoordList is not None and os.access(ytransCoordList, os.F_OK)):
            calibs['ytrans_rect'] = self.calculateMOSSkylineTrans(fdu, ytransCoordList, calibs)
        elif (self.getOption("rectify_sky", fdu.getTag()).lower() == "yes"):
            coords = self.traceMOSSkylineRectification(fdu, skyFDU, calibs)
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                calibs['xtrans_rect'] = self.calculateMOSSkylineTrans(fdu, coords, calibs)
            else:
                calibs['ytrans_rect'] = self.calculateMOSSkylineTrans(fdu, coords, calibs)
        else:
            print("rectifyProcess::calcMOSRectification> Skipping sky rectification...")
            self._log.writeLog(__name__, "Skipping sky rectification...")
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (self._fdb.getGPUMode()):
                    xtransData = calcXin(xsize, ysize)
                else:
                    xtransData = arange(xsize*ysize, dtype=float32).reshape(ysize,xsize) % xsize
                xtrans_rect = fatboySpecCalib(self._pname, "xtrans_rect", fdu, data=xtransData, tagname=fdu._id, log=self._log)
                xtrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
                xtrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
                self._fdb.appendCalib(xtrans_rect)
                calibs['xtrans_rect'] = xtrans_rect
            else:
                if (self._fdb.getGPUMode()):
                    ytransData = calcYin(xsize, ysize)
                else:
                    ytransData = (arange(xsize*ysize).reshape(ysize,xsize) // xsize).astype(float32)
                ytrans_rect = fatboySpecCalib(self._pname, "ytrans_rect", fdu, data=ytransData, tagname=fdu._id, log=self._log)
                ytrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
                ytrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
                self._fdb.appendCalib(ytrans_rect)
                calibs['ytrans_rect'] = ytrans_rect

        return calibs
    #end calcMOSRectification

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            #Make sure rectify isn't called on offsource skies when re-reading data from disk
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_CONTINUUM_SOURCE):
            #Also rectify for continuum source calibs
            return True
        return False
    #end checkValidDatatype

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Rectify")
        print(fdu._identFull)

        #Get original shape before checkOutputExists
        origShape = fdu.getShape()

        #Check if output exists first and update from disk
        rctfile = "rectified/rct_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, rctfile)):
            #Also check if "cleanFrame" exists
            cleanfile = "rectified/clean_rct_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "exposure map" exists
            expfile = "rectified/exp_rct_"+fdu.getFullId()
            self.checkOutputExists(fdu, expfile, tag="exposure_map")
            #Also check if noisemap exists
            nmfile = "rectified/NM_rct_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")

            #Need to get calibration frames - cleanSky, masterLamp, and slitmask to update from disk too
            calibs = dict()
            headerVals = dict()
            headerVals['grism_keyword'] = fdu.grism
            properties = dict()
            properties['specmode'] = fdu.getProperty("specmode")
            properties['dispersion'] = fdu.getProperty("dispersion")
            if (not 'cleanSky' in calibs):
                #Check for an already created clean sky frame matching specmode/filter/grism/ident
                #cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (cleanSky is not None):
                    #add to calibs for rectification below
                    calibs['cleanSky'] = cleanSky

            if (not 'masterLamp' in calibs):
                #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
                masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
                if (masterLamp is None):
                    #2) Check for an already created master arclamp frame matching specmode/filter/grism
                    masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (masterLamp is not None):
                    #add to calibs for rectification below
                    calibs['masterLamp'] = masterLamp

            #Calculate rectification for longslit here
            if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
                #Find slitmask associated with this fdu
                #Use new fdu.getSlitmask method
                #First check for a rectified slitmask
                slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                if (slitmask is None):
                    #Now check for one with origShape -- will be updated below
                    slitmask = fdu.getSlitmask(pname=None, shape=origShape, properties=properties, headerVals=headerVals)
                if (slitmask is None):
                    print("rectifyProcess::execute> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to rectify!")
                    self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to rectify!", type=fatboyLog.ERROR)
                    return calibs
                calibs['slitmask'] = slitmask

            #Check for cleanSky and masterLamp frames to update from disk too
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("rectified")):
                #Check if output exists
                rctfile = "rectified/rct_"+calibs['cleanSky'].getFullId()
                if (self.checkOutputExists(calibs['cleanSky'], rctfile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                    calibs['cleanSky'].setProperty("rectified", True)

            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("rectified")):
                #Check if output exists first
                rctfile = "rectified/rct_"+calibs['masterLamp'].getFullId()
                if (self.checkOutputExists(calibs['masterLamp'], rctfile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                    calibs['masterLamp'].setProperty("rectified", True)

            #Check for slitmask frames to update from disk too
            if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("rectified")):
                #Check if output exists
                rctfile = "rectified/rct_"+calibs['slitmask'].getFullId()
                #This will append new slitmask
                if (self.checkOutputExists(calibs['slitmask'], rctfile)):
                    #Now get new slitmask with correct shape
                    calibs['slitmask'] = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                    #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                    calibs['slitmask'].setProperty("rectified", True)
                    #Update nslits property
                    nslits = calibs['slitmask'].getData().max()
                    calibs['slitmask'].setProperty("nslits", nslits)
                    #Update regions
                    if (calibs['slitmask'].hasProperty("regions")):
                        (sylo, syhi, slitx, slitw) = calibs['slitmask'].getProperty("regions")
                        #Get rid of guide star boxes and any entirely negative slits (shouldn't happen)
                        maxSlitWidth = float(self.getOption("mos_max_slit_width", fdu.getTag()))
                        b = logical_and(slitw < maxSlitWidth, syhi > 0)
                        sylo = sylo[b]
                        syhi = syhi[b]
                        slitx = slitx[b]
                        slitw = slitw[b]
                        #Use helper method to all ylo, yhi for each slit in each frame
                        #Keep original slitx, slitw - use temp vars to receive return values
                        (sylo, syhi, tempx, tempw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)
                    else:
                        #Use helper method to all ylo, yhi for each slit in each frame
                        (sylo, syhi, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)
                    calibs['slitmask'].setProperty("regions", (sylo, syhi, slitx, slitw))
            return True
        #############

        #Call get calibs to return dict() of calibration frames.
        #For rectification of longslit data, this dict should have 1 entry: 'rect_coeffs'
        #These are obtained by tracing slitlets using the master flat
        calibs = self.getCalibs(fdu, prevProc)

        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            if (calibs['rect_coeffs'] is None):
                #Failed to obtain rectification coefficients
                #Issue error message and disable this FDU
                print("rectifyProcess::execute> ERROR: Rectification coefficients not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
                self._log.writeLog(__name__, "Rectification coefficients not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return False
        else:
            if (calibs['xtrans_rect'] is None or calibs['ytrans_rect'] is None):
                #Failed to obtain rectification coefficients
                #Issue error message and disable this FDU
                print("rectifyProcess::execute> ERROR: Rectification coefficients not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
                self._log.writeLog(__name__, "Rectification coefficients not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return False

        #Perform rectification transformation
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            #Call helper method to rectify
            self.rectifyLongslit(fdu, calibs)
        else:
            #Call helper method to rectify
            self.rectifyMOS(fdu, calibs)
        fdu._header.add_history('Rectified')
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each master calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("rectifyProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("rectifyProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        csfilename = self.getCalib("master_clean_sky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("rectifyProcess::getCalibs> Using master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using master clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, log=self._log)
            else:
                print("rectifyProcess::getCalibs> Warning: Could not find master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find master clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("rectifyProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("rectifyProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Could not find master arclamp frame "+mlfilename+"...", type=fatboyLog.WARNING)
        #Look for list of frames passed as longslit continua frames
        lsFrames = self.getCalib("longslit_continua_frames", fdu.getTag())
        if (lsFrames is not None):
            frameList = []
            if (lsFrames.count(',') > 0):
                #comma separated list
                frameList = lsFrames.split(',')
                removeEmpty(frameList)
                for j in range(len(frameList)):
                    frameList[j] = frameList[j].strip()
            elif (lsFrames.endswith('.fit') or lsFrames.endswith('.fits')):
                #FITS file given
                frameList.append(lsFrames)
            elif (lsFrames.endswith('.dat') or lsFrames.endswith('.list') or lsFrames.endswith('.txt')):
                #ASCII file list
                frameList = readFileIntoList(lsFrames)
            frameFDUs = []
            for frame in frameList:
                if (os.access(frame, os.F_OK)):
                    frameFDUs.append(fatboySpecCalib(self._pname, "rectification_frame", fdu, filename=frame, log=self._log))
            if (len(frameFDUs) > 0):
                calibs['longslit_continua_frames'] = frameFDUs
        #Look for list of frames passed as MOS continua frames
        mosFrames = self.getCalib("mos_continua_frames", fdu.getTag())
        if (mosFrames is not None):
            frameList = []
            if (mosFrames.count(',') > 0):
                #comma separated list
                frameList = mosFrames.split(',')
                removeEmpty(frameList)
                for j in range(len(frameList)):
                    frameList[j] = frameList[j].strip()
            elif (mosFrames.endswith('.fit') or mosFrames.endswith('.fits')):
                #FITS file given
                frameList.append(mosFrames)
            elif (mosFrames.endswith('.dat') or mosFrames.endswith('.list') or mosFrames.endswith('.txt')):
                #ASCII file list
                frameList = readFileIntoList(mosFrames)
            frameFDUs = []
            for frame in frameList:
                if (os.access(frame, os.F_OK)):
                    frameFDUs.append(fatboySpecCalib(self._pname, "rectification_frame", fdu, filename=frame, log=self._log))
            if (len(frameFDUs) > 0):
                calibs['mos_continua_frames'] = frameFDUs

        #Find master clean sky and master arclamp associated with this object
        #Do this once in getCalibs rather than having to get them again in
        #rectifyLongslit and rectifyMOS
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #Set xtrans and ytrans to None here
        calibs['xtrans_rect'] = None
        calibs['ytrans_rect'] = None
        calibs['rect_coeffs'] = None

        #Define rct_id to be fdu._id by default unless property "rectify_object" exists
        rct_id = fdu._id
        if (fdu.hasProperty("rectify_object")):
            rct_id = fdu.getProperty("rectify_object")
        else:
            #1a) check for continuum source frames matching specmode/filter/grism and TAGGED for this object
            csources = self._fdb.getTaggedCalibs(fdu._id, obstype=fdu.FDU_TYPE_CONTINUUM_SOURCE, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
            if (len(csources) > 0):
                #Found continuum sources associated with this fdu. Recursively process - tag as rct_id
                print("rectifyProcess::getCalibs> Found continuum sources for tagged object "+fdu._id+", filter "+str(fdu.filter)+"...")
                self._log.writeLog(__name__, "Found continuum sources for tagged object "+fdu._id+", filter "+str(fdu.filter)+"...")
                rct_id = csources[0]._id
            #2) Check for individual continuum sources matching specmode/filter/grism
            csources = self._fdb.getCalibs(obstype=fdu.FDU_TYPE_CONTINUUM_SOURCE, filter=fdu.filter, tag=fdu.getTag(), section=fdu.section, properties=properties, headerVals=headerVals)
            if (len(csources) > 0):
                #Found continuum sources associated with this fdu. Recursively process - tag as rct_id
                print("rectifyProcess::getCalibs> Found continuum sources for object "+fdu._id+", filter "+str(fdu.filter)+"...")
                self._log.writeLog(__name__, "Found continuum sources for object "+fdu._id+", filter "+str(fdu.filter)+"...")
                rct_id = csources[0]._id

        #Arclamp / clean sky frame
        if (not 'cleanSky' in calibs):
            #Check for an already created clean sky frame matching specmode/filter/grism/ident
            #cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            #Shape can be different - if it has been already rectified and has transformation saved as property
            cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL and cleanSky is not None and cleanSky.hasProperty('xtrans_rect')):
                calibs['xtrans_rect'] = cleanSky.getProperty('xtrans_rect')
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL and cleanSky is not None and cleanSky.hasProperty('ytrans_rect')):
                calibs['ytrans_rect'] = cleanSky.getProperty('ytrans_rect')
            elif (fdu._specmode == fdu.FDU_TYPE_LONGSLIT and cleanSky is not None and cleanSky.hasProperty('rect_coeffs')):
                calibs['rect_coeffs'] = cleanSky.getProperty('rect_coeffs')
            else:
                #Check for an already created clean sky frame matching specmode/filter/grism, also matching shape
                cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", shape=fdu.getShape(), section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (cleanSky is not None):
                #add to calibs for rectification below
                calibs['cleanSky'] = cleanSky

        if (not 'masterLamp' in calibs):
            #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, shape=fdu.getShape(), section=fdu.section, properties=properties, headerVals=headerVals)
            if (masterLamp is None):
                #2a) Check for an already created master arclamp frame matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL and masterLamp is not None and masterLamp.hasProperty('xtrans_rect')):
                    calibs['xtrans_rect'] = masterLamp.getProperty('xtrans_rect')
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL and masterLamp is not None and masterLamp.hasProperty('ytrans_rect')):
                    calibs['ytrans_rect'] = masterLamp.getProperty('ytrans_rect')
                elif (fdu._specmode == fdu.FDU_TYPE_LONGSLIT and masterLamp is not None and masterLamp.hasProperty('rect_coeffs')):
                    calibs['rect_coeffs'] = masterLamp.getProperty('rect_coeffs')
                else:
                    #2b) Check for an already created master arclamp frame matching specmode/filter/grism, also matching shape
                    masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, shape=fdu.getShape(), section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is not None):
                #add to calibs for rectification below
                calibs['masterLamp'] = masterLamp

        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            #First look for FDU property rect_coeffs
            if (fdu.hasProperty('rect_coeffs')):
                calibs['rect_coeffs'] = fdu.getProperty('rect_coeffs')
                return calibs

            #Second check for property or calib rect_coeffs_file from XML
            if (fdu.hasProperty("rect_coeffs_file")):
                rectCoeffsFile = fdu.getProperty("rect_coeffs_file")
            else:
                rectCoeffsFile = self.getCalib("rect_coeffs_file", fdu.getTag())
                if (rectCoeffsFile is None):
                    rectCoeffsFile = self.getOption("rect_coeffs_file", fdu.getTag())
            #Check that rect coeffs file exists
            if (rectCoeffsFile is not None and os.access(rectCoeffsFile, os.F_OK)):
                #Read in coeffs into [xcoeffs, ycoeffs] list of lists
                calibs['rect_coeffs'] = readCoeffsFile(rectCoeffsFile, self._log)
                #Check that it was read correctly
                if (calibs['rect_coeffs'] is not None):
                    return calibs
                else:
                    #File exists but wasn't read properly; issue warning
                    print("rectifyProcess::getCalibs> Warning: Could not read rect_coeffs_file "+rectCoeffsFile+" associated with "+fdu.getFullId()+"! Calculating rectification from data!")
                    self._log.writeLog(__name__, "Could not read rect_coeffs_file "+rectCoeffsFile+" associated with "+fdu.getFullId()+"!  Calculating rectification from data!", type=fatboyLog.WARNING)
            #If rect_coeffs exist, return them here
            if (calibs['rect_coeffs'] is not None):
                return calibs
        else:
            #First look for slitmask
            if (not 'slitmask' in calibs):
                #Use new fdu.getSlitmask method
                slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                if (slitmask is not None):
                    calibs['slitmask'] = slitmask

            #Is x-transformation passed as calib?
            #1) Look for FDU property xtrans_rect
            if (fdu.hasProperty('xtrans_rect')):
                calibs['xtrans_rect'] = fdu.getProperty('xtrans_rect')
            else:
                #2) Look for master calibration frame xtrans_rect matching rct_id for this FDU
                xtransRect = self._fdb.getMasterCalib(ident=rct_id, filter=fdu.filter, obstype="xtrans_rect", section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (xtransRect is not None):
                    calibs['xtrans_rect'] = xtransRect
                else:
                    #3) Look for property or calib "xtrans_rect_file" that is a FITS file
                    if (fdu.hasProperty("xtrans_rect_file")):
                        xtransCoeffsFile = fdu.getProperty("xtrans_rect_file")
                    else:
                        xtransCoeffsFile = self.getCalib("xtrans_rect_file", fdu.getTag())
                    if (xtransCoeffsFile is not None and os.access(xtransCoeffsFile, os.F_OK)):
                        xtrans_rect = fatboySpecCalib(self._pname, "xtrans_rect", fdu, filename=xtransCoeffsFile, log=self._log)
                        xtrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
                        xtrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
                        self._fdb.appendCalib(xtrans_rect)
                        calibs['xtrans_rect'] = xtrans_rect

            #Is y-transformation passed as calib?
            #1) Look for FDU property ytrans_rect
            if (fdu.hasProperty('ytrans_rect')):
                calibs['ytrans_rect'] = fdu.getProperty('ytrans_rect')
            else:
                #2) Look for master calibration frame ytrans_rect matching rct_id for this FDU
                ytransRect = self._fdb.getMasterCalib(ident=rct_id, filter=fdu.filter, obstype="ytrans_rect", section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (ytransRect is not None):
                    calibs['ytrans_rect'] = ytransRect
                else:
                    #3) Look for property or calib "ytrans_rect_file" that is a FITS file
                    if (fdu.hasProperty("ytrans_rect_file")):
                        ytransCoeffsFile = fdu.getProperty("ytrans_rect_file")
                    else:
                        ytransCoeffsFile = self.getCalib("ytrans_rect_file", fdu.getTag())
                    if (ytransCoeffsFile is not None and os.access(ytransCoeffsFile, os.F_OK)):
                        ytrans_rect = fatboySpecCalib(self._pname, "ytrans_rect", fdu, filename=ytransCoeffsFile, log=self._log)
                        ytrans_rect.setProperty("specmode", fdu.getProperty("specmode"))
                        ytrans_rect.setProperty("dispersion", fdu.getProperty("dispersion"))
                        self._fdb.appendCalib(ytrans_rect)
                        calibs['ytrans_rect'] = ytrans_rect

            #If xcoeffs and ycoeffs exist, return them here
            if (calibs['xtrans_rect'] is not None and calibs['ytrans_rect'] is not None):
                return calibs

        #Check for individual FDUs matching specmode/filter/grism/ident to calculate rectification
        #Also section
        #rctfdus can not be [] as it will always at least return the current FDU itself
        rctfdus = self._fdb.getFDUs(ident = rct_id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())

        #Next check for master arclamp frame or clean sky frame to rectify skylines
        #These should have been found above and added to calibs dict
        skyFDU = None
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            if ('masterLamp' in calibs):
                skyFDU = calibs['masterLamp']
            else:
                print("rectifyProcess::getCalibs> Warning: Could not find master arclamp associated with "+fdu.getFullId()+"! Attempting to use clean sky frame for skyline rectification!")
                self._log.writeLog(__name__, "Could not find master arclamp associated with "+fdu.getFullId()+"! Attempting to use clean sky frame for skyline rectification!", type=fatboyLog.WARNING)
        if (skyFDU is None and 'cleanSky' in calibs):
            #Either use_arclamps = no or master arclamp not found
            skyFDU = calibs['cleanSky']

        print("rectifyProcess::getCalibs> Calculating rectification for "+fdu._id+" ...")
        self._log.writeLog(__name__, "Calculating rectification for "+fdu._id+" ...")
        #First recursively process (through sky subtraction presumably)
        self.recursivelyExecute(rctfdus, prevProc)
        #Loop over rctfdus and pop out any that have been disabled at sky subtraction stage by pairing up
        #Loop backwards!
        for j in range(len(rctfdus)-1, -1, -1):
            currFDU = rctfdus[j]
            if (not currFDU.inUse):
                rctfdus.remove(currFDU)

        #Calculate rectification for longslit here
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            #Returns [0,1,0] and [0,0,1] if rctfdus is [] or skyFDU is None, respectively
            calibs['rect_coeffs'] = self.calcLongslitRectification(fdu, rctfdus, skyFDU, calibs)
            #loop over rctfdus and set property 'rect_coeffs'
            for j in range(len(rctfdus)):
                rctfdus[j].setProperty('rect_coeffs', calibs['rect_coeffs'])
            if (rct_id != fdu._id):
                #Also loop over fdus for this object and set property 'rect_coeffs'
                objfdus = self._fdb.getFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                for j in range(len(objfdus)):
                    objfdus[j].setProperty('rect_coeffs', calibs['rect_coeffs'])
            if (self.getOption("rectify_sky", fdu.getTag()).lower() == "yes" and skyFDU is not None):
                #Finally set skyFDU property for transformation
                skyFDU.setProperty('rect_coeffs', calibs['rect_coeffs'])
        else:
            if (not 'slitmask' in calibs):
                #Find slitmask associated with this fdu
                #Use new fdu.getSlitmask method
                slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                if (slitmask is None):
                    print("rectifyProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to rectify!")
                    self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to rectify!", type=fatboyLog.ERROR)
                    return calibs
                calibs['slitmask'] = slitmask

            #Calculate MOS rectification -- this method edits and returns calibs
            calibs = self.calcMOSRectification(fdu, rctfdus, skyFDU, calibs)
            #loop over rctfdus and set properties 'xtrans_rect' and 'ytrans_rect'
            for j in range(len(rctfdus)):
                rctfdus[j].setProperty('xtrans_rect', calibs['xtrans_rect'])
                rctfdus[j].setProperty('ytrans_rect', calibs['ytrans_rect'])
            if (rct_id != fdu._id):
                #Also loop over fdus for this object and set properties 'xtrans_rect' and ytrans_rect'
                objfdus = self._fdb.getFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                for j in range(len(objfdus)):
                    objfdus[j].setProperty('xtrans_rect', calibs['xtrans_rect'])
                    objfdus[j].setProperty('ytrans_rect', calibs['ytrans_rect'])
            if (self.getOption("rectify_sky", fdu.getTag()).lower() == "yes" and skyFDU is not None):
                #Finally set skyFDU property for transformation
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    skyFDU.setProperty('xtrans_rect', calibs['xtrans_rect'])
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    skyFDU.setProperty('ytrans_rect', calibs['ytrans_rect'])

        return calibs
    #end getCalibs

    #Perform actual rectification of longslit data
    def rectifyLongslit(self, fdu, calibs):
        #Get options
        drihizzleKernel = self.getOption("drihizzle_kernel", fdu.getTag()).lower()
        dropsize = float(self.getOption("drihizzle_dropsize", fdu.getTag()))

        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/rectified", os.F_OK)):
            os.mkdir(outdir+"/rectified",0o755)

        #Check for cleanSky and masterLamp frames to update from disk too
        if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("rectified")):
            #Check if output exists
            rctfile = "rectified/rct_"+calibs['cleanSky'].getFullId()
            if (self.checkOutputExists(calibs['cleanSky'], rctfile)):
                #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                calibs['cleanSky'].setProperty("rectified", True)

        if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("rectified")):
            #Check if output exists first
            rctfile = "rectified/rct_"+calibs['masterLamp'].getFullId()
            if (self.checkOutputExists(calibs['masterLamp'], rctfile)):
                #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                calibs['masterLamp'].setProperty("rectified", True)

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle

        #inmask = fdu.crMask if cosmic ray method = mask
        crMask = None
        if (fdu.hasProperty("crmask")):
            crMask = fdu.getProperty("crmask")

        #Look for "cleanSky" frame to rectify
        if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("rectified")):
            cleanSky = calibs['cleanSky']
            #Use turbo kernel for cleanSky
            (data, header, expmap, pixmap) = drihizzle_method(cleanSky, None, None, inmask=crMask, weight='exptime', kernel='turbo', dropsize=dropsize, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #update data, header, set "rectified" property
            cleanSky.updateData(data)
            cleanSky.updateHeader(header)
            cleanSky.setProperty("rectified", True)
            #Write to disk if requested
            if (writeCalibs):
                rctfile = outdir+"/rectified/rct_"+cleanSky.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(rctfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(rctfile)
                #Write to disk
                if (not os.access(rctfile, os.F_OK)):
                    cleanSky.writeTo(rctfile)

        #Look for "masterLamp" frame to rectify
        if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("rectified")):
            masterLamp = calibs['masterLamp']
            #Use turbo kernel for masterLamp
            (data, header, expmap, pixmap) = drihizzle_method(masterLamp, None, None, inmask=crMask, weight='exptime', kernel='turbo', dropsize=dropsize, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #update data, header, set "rectified" property
            masterLamp.updateData(data)
            masterLamp.updateHeader(header)
            masterLamp.setProperty("rectified", True)
            #Write to disk if requested
            if (writeCalibs):
                rctfile = outdir+"/rectified/rct_"+masterLamp.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(rctfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(rctfile)
                #Write to disk
                if (not os.access(rctfile, os.F_OK)):
                    masterLamp.writeTo(rctfile)

        if (drihizzleKernel == "point_replace"):
            #Look for "cleanFrame" to rectify
            if (fdu.hasProperty("cleanFrame")):
                #First use point to drizzle
                #Update "cleanFrame" data tag
                (cleanData, header, point_expmap, point_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="point", dropsize=0.01, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Next use turbo
                (turbo_data, header, turbo_expmap, turbo_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="turbo", dropsize=1, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Next scale any output pixel where more than one input pixel contributed
                #And replace any output pixel with 0 contribution with turbo value
                b = (cleanData == 0)*(turbo_expmap != 0)*(turbo_data != 0)
                nonzeroPoints = turbo_expmap != 0
                turbo_data[nonzeroPoints] /= turbo_expmap[nonzeroPoints]
                #Scale turbo value by median of neighboring pixels in point / neighboring pixels in turbo before replacing
                scale = medScale2d(cleanData, turbo_data, b, 3)
                scale[scale > 2*fdu.exptime] = fdu.exptime
                scale[scale < 0] = 0
                cleanData[b] = turbo_data[b]*scale[b]
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)

            #Rectify noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, rectify, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, point_expmap, point_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="point", dropsize=0.01, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Update noisemap for pixels contributing twice or more
                b = point_expmap != 0
                point_expmap[b] *= fdu.exptime/point_expmap[b]
                #Update noisemap for pixels with 0 contribution
                b = point_expmap == 0
                nmData = medReplace2d(nmData, b, 3)
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))

            #First use point to drizzle
            (point_data, header, point_expmap, point_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="point", dropsize=0.01, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Next use turbo
            (turbo_data, header, turbo_expmap, turbo_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="turbo", dropsize=1, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Next scale any output pixel where more than one input pixel contributed
            #And replace any output pixel with 0 contribution with turbo value
            b = (point_data == 0)*(turbo_expmap != 0)*(turbo_data != 0)
            nonzeroPoints = turbo_expmap != 0
            turbo_data[nonzeroPoints] /= turbo_expmap[nonzeroPoints]
            #Scale turbo value by median of neighboring pixels in point / neighboring pixels in turbo before replacing
            scale = medScale2d(point_data, turbo_data, b, 3)
            scale[scale > 2*fdu.exptime] = fdu.exptime
            scale[scale < 0] = 0

            #Compute difference image
            diffImage = zeros(point_data.shape)
            diffImage[b] = turbo_data[b]*scale[b] - point_data[b]
            point_data[b] = turbo_data[b]*scale[b]
            point_expmap[b] = fdu.exptime
            print("\t\tReplaced "+str(b.sum())+" pixels with turbo kernel results.")
            self._log.writeLog(__name__, "Replaced "+str(b.sum())+" pixels with turbo kernel results.", printCaller=False, tabLevel=2)
            #Write out difference image
            diffFile = outdir+'/rectified/diff_'+fdu.getFullId()
            fdu.tagDataAs("diffImage", data=diffImage)
            fdu.writeTo(diffFile, tag="diffImage")
            fdu.removeProperty("diffImage")

            #Still zeros around edges
            b = point_expmap != 0
            #Adjust expmap for double contributions
            point_expmap[b] *= fdu.exptime/point_expmap[b]
            #Update FDU data and expmap
            fdu.updateData(point_data)
            fdu.updateHeader(header)
            fdu.tagDataAs("exposure_map", data=point_expmap)
            if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
                #only tag this if writing output -- then free up memory after writing output
                fdu.tagDataAs("pixel_map", data=point_pixmap)
        else:
            #All other rectify options
            #Look for "cleanFrame" to rectify
            if (fdu.hasProperty("cleanFrame")):
                (cleanData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel=drihizzleKernel, dropsize=dropsize, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)

            #Rectify noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, rectify, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel=drihizzleKernel, dropsize=dropsize, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))

                #Rectify actual data frame
            (data, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel=drihizzleKernel, dropsize=dropsize, geomDist=calibs['rect_coeffs'], inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            if (drihizzleKernel == "point"):
                #Adjust expmap for double contributions
                b = expmap > fdu.exptime
                expmap[b] = fdu.exptime
            fdu.updateData(data)
            fdu.updateHeader(header)
            fdu.tagDataAs("exposure_map", data=expmap)

            if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
                #only tag this if writing output -- then free up memory after writing output
                fdu.tagDataAs("pixel_map", data=pixmap)
    #end rectifyLongslit

    #Perform actual rectification of longslit data
    def rectifyMOS(self, fdu, calibs):
        #Get options
        drihizzleKernel = self.getOption("drihizzle_kernel", fdu.getTag()).lower()
        dropsize = float(self.getOption("drihizzle_dropsize", fdu.getTag()))
        maxSlitWidth = float(self.getOption("mos_max_slit_width", fdu.getTag()))

        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/rectified", os.F_OK)):
            os.mkdir(outdir+"/rectified",0o755)

        #Check for cleanSky and masterLamp frames to update from disk too
        if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("rectified")):
            #Check if output exists
            rctfile = "rectified/rct_"+calibs['cleanSky'].getFullId()
            if (self.checkOutputExists(calibs['cleanSky'], rctfile)):
                #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                calibs['cleanSky'].setProperty("rectified", True)

        if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("rectified")):
            #Check if output exists first
            rctfile = "rectified/rct_"+calibs['masterLamp'].getFullId()
            if (self.checkOutputExists(calibs['masterLamp'], rctfile)):
                #output file already exists and overwrite = no.  Update data from disk and set "rectified" = True
                calibs['masterLamp'].setProperty("rectified", True)

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle

        #inmask = 0 in between slitlets
        crMask = (calibs['xtrans_rect'].getData() != 0)*(calibs['ytrans_rect'].getData() != 0)
        #inmask *= fdu.crMask if cosmic ray method = mask
        if (fdu.hasProperty("crmask")):
            crMask *= fdu.getProperty("crmask") #crmask is good pixel mask

        #First update slitmask before anything else.  Use "uniform" kernel.
        if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("rectified")):
            slitmask = calibs['slitmask']
            (data, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=crMask, weight='exptime', kernel="uniform", xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Now update slitmask to remove guide star boxes

            if (not slitmask.hasProperty("nslits")):
                slitmask.setProperty("nslits", slitmask.getData().max())
            nslits = slitmask.getProperty("nslits")
            if (slitmask.hasProperty("regions")):
                (sylo, syhi, slitx, slitw) = slitmask.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (sylo, syhi, slitx, slitw) = findRegions(slitmask.getData(), nslits, slitmask, gpu=self._fdb.getGPUMode(), log=self._log)
                slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
            if (slitmask.hasProperty("rylo")):
                #Use previously saved rylo and ryhi
                rylo = slitmask.getProperty("rylo")
                ryhi = slitmask.getProperty("ryhi")
                #Get rid of guide star boxes and any entirely negative slits (shouldn't happen)
                b = logical_and(slitw < maxSlitWidth, ryhi > 0)
                sylo = rylo[b]
                syhi = ryhi[b]
                slitx = slitx[b]
                slitw = slitw[b]
            else:
                #Must be using previous ytrans file.  rylo, ryhi not calculated, use sylo, syhi
                #Get rid of guide star boxes and any entirely negative slits (shouldn't happen)
                b = logical_and(slitw < maxSlitWidth, syhi > 0)
                sylo = sylo[b]
                syhi = syhi[b]
                slitx = slitx[b]
                slitw = slitw[b]
            #Update slitmask to eliminate guide star boxes that have been thrown out so that slit numbering is consecutive!
            idx = 0
            for islit in range(nslits):
                #This input slit index exists in the output data
                if ((islit+1) in data):
                    #If idx doesn't match islit, we've skipped a guide star box and need to update data in this slitlet
                    #to be a lower index, e.g. 2 instead of 3.
                    if (islit != idx):
                        data[data == islit+1] = idx+1
                    #Always increment idx if data is found in this slitlet
                    idx += 1
            #idx now represents the number of slitlets
            nslits = idx
            #Need to update sylo and syhi to new rectified data
            #Use helper method to all ylo, yhi for each slit in each frame
            #Keep original slitx, slitw -- use temp vars to get return values
            (sylo, syhi, tempx, tempw) = findRegions(data, nslits, slitmask, gpu=self._fdb.getGPUMode(), log=self._log)

            rctSlitmask = self._fdb.addNewSlitmask(slitmask, data, self._pname)
            #update properties
            rctSlitmask.setProperty("nslits", nslits)
            rctSlitmask.setProperty("regions", (sylo, syhi, slitx, slitw))

            #update data, header, set "rectified" property
            rctSlitmask.updateData(data)
            rctSlitmask.updateHeader(header)
            rctSlitmask.setProperty("rectified", True)
            slitmask.setProperty("rectified", True)
            #Write to disk if requested
            if (writeCalibs):
                rctfile = outdir+"/rectified/rct_"+rctSlitmask.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(rctfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(rctfile)
                #Write to disk
                if (not os.access(rctfile, os.F_OK)):
                    rctSlitmask.writeTo(rctfile)

        #Look for "cleanSky" frame to rectify
        if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("rectified")):
            cleanSky = calibs['cleanSky']
            #Use turbo kernel for cleanSky
            (data, header, expmap, pixmap) = drihizzle_method(cleanSky, None, None, inmask=crMask, weight='exptime', kernel='turbo', dropsize=dropsize, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #update data, header, set "rectified" property
            cleanSky.updateData(data)
            cleanSky.updateHeader(header)
            cleanSky.setProperty("rectified", True)
            #Write to disk if requested
            if (writeCalibs):
                rctfile = outdir+"/rectified/rct_"+cleanSky.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(rctfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(rctfile)
                #Write to disk
                if (not os.access(rctfile, os.F_OK)):
                    cleanSky.writeTo(rctfile)

        #Look for "masterLamp" frame to rectify
        if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("rectified")):
            masterLamp = calibs['masterLamp']
            #Use turbo kernel for masterLamp
            (data, header, expmap, pixmap) = drihizzle_method(masterLamp, None, None, inmask=crMask, weight='exptime', kernel='turbo', dropsize=dropsize, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #update data, header, set "rectified" property
            masterLamp.updateData(data)
            masterLamp.updateHeader(header)
            masterLamp.setProperty("rectified", True)
            #Write to disk if requested
            if (writeCalibs):
                rctfile = outdir+"/rectified/rct_"+masterLamp.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(rctfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(rctfile)
                #Write to disk
                if (not os.access(rctfile, os.F_OK)):
                    masterLamp.writeTo(rctfile)

        #Split processing of rest of frames based on drizzle kernel
        if (drihizzleKernel == "point_replace"):
            #Look for "cleanFrame" to rectify
            if (fdu.hasProperty("cleanFrame")):
                #First use point to drizzle
                #Update "cleanFrame" data tag
                (cleanData, header, point_expmap, point_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="point", dropsize=0.01, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Next use turbo
                (turbo_data, header, turbo_expmap, turbo_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="turbo", dropsize=1, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Next scale any output pixel where more than one input pixel contributed
                #And replace any output pixel with 0 contribution with turbo value
                b = (cleanData == 0)*(turbo_expmap != 0)*(turbo_data != 0)
                nonzeroPoints = turbo_expmap != 0
                turbo_data[nonzeroPoints] /= turbo_expmap[nonzeroPoints]
                #Scale turbo value by median of neighboring pixels in point / neighboring pixels in turbo before replacing
                scale = medScale2d(cleanData, turbo_data, b, 3)
                scale[scale > 2*fdu.exptime] = fdu.exptime
                scale[scale < 0] = 0
                cleanData[b] = turbo_data[b]*scale[b]
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)

            #Rectify noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, rectify, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, point_expmap, point_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="point", dropsize=0.01, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Update noisemap for pixels contributing twice or more
                b = point_expmap != 0
                point_expmap[b] *= fdu.exptime/point_expmap[b]
                #Update noisemap for pixels with 0 contribution
                b = point_expmap == 0
                nmData = medReplace2d(nmData, b, 3)
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))

            #First use point to drizzle
            (point_data, header, point_expmap, point_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="point", dropsize=0.01, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Next use turbo
            (turbo_data, header, turbo_expmap, turbo_pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel="turbo", dropsize=1, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Next scale any output pixel where more than one input pixel contributed
            #And replace any output pixel with 0 contribution with turbo value
            b = (point_data == 0)*(turbo_expmap != 0)*(turbo_data != 0)
            nonzeroPoints = turbo_expmap != 0
            turbo_data[nonzeroPoints] /= turbo_expmap[nonzeroPoints]
            #Scale turbo value by median of neighboring pixels in point / neighboring pixels in turbo before replacing
            scale = medScale2d(point_data, turbo_data, b, 3)
            scale[scale > 2*fdu.exptime] = fdu.exptime
            scale[scale < 0] = 0

            #Compute difference image
            diffImage = zeros(point_data.shape)
            diffImage[b] = turbo_data[b]*scale[b] - point_data[b]
            point_data[b] = turbo_data[b]*scale[b]
            point_expmap[b] = fdu.exptime
            print("\t\tReplaced "+str(b.sum())+" pixels with turbo kernel results.")
            self._log.writeLog(__name__, "Replaced "+str(b.sum())+" pixels with turbo kernel results.", printCaller=False, tabLevel=2)
            #Write out difference image
            diffFile = outdir+'/rectified/diff_'+fdu.getFullId()
            fdu.tagDataAs("diffImage", data=diffImage)
            fdu.writeTo(diffFile, tag="diffImage")
            fdu.removeProperty("diffImage")

            #Still zeros around edges
            b = point_expmap != 0
            #Adjust expmap for double contributions
            point_expmap[b] *= fdu.exptime/point_expmap[b]
            #Update FDU data and expmap
            fdu.updateData(point_data)
            fdu.updateHeader(header)
            fdu.tagDataAs("exposure_map", data=point_expmap)
            if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
                #only tag this if writing output -- then free up memory after writing output
                fdu.tagDataAs("pixel_map", data=point_pixmap)
        else:
            #Look for "cleanFrame" to rectify
            if (fdu.hasProperty("cleanFrame")):
                (cleanData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel=drihizzleKernel, dropsize=dropsize, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)

            #Rectify noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, rectify, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel=drihizzleKernel, dropsize=dropsize, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))

            #Rectify actual data frame
            (data, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=crMask, weight='exptime', kernel=drihizzleKernel, dropsize=dropsize, xtrans=calibs['xtrans_rect'].getData(), ytrans=calibs['ytrans_rect'].getData(), inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU)
            if (drihizzleKernel == "point"):
                #Adjust expmap for double contributions
                b = expmap > fdu.exptime
                expmap[b] = fdu.exptime
            fdu.updateData(data)
            fdu.updateHeader(header)
            fdu.tagDataAs("exposure_map", data=expmap)
            if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
                #only tag this if writing output -- then free up memory after writing output
                fdu.tagDataAs("pixel_map", data=pixmap)
    #end rectifyMOS

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('continuum_find_xlo', None)
        self._optioninfo.setdefault('continuum_find_xlo', 'Defaults to 0.  Used to specify a range of the chip to sum\na 1-d cut across and attemt to find any continua.\nIn the case of highly curved orders, a narrower range is needed.')
        self._options.setdefault('continuum_find_xhi', None)
        self._optioninfo.setdefault('continuum_find_xhi', 'Defaults to xsize.  Used to specify a range of the chip to sum\na 1-d cut across and attemt to find any continua.\nIn the case of highly curved orders, a narrower range is needed.')

        self._options.setdefault('continuum_trace_xinit', None)
        self._options.setdefault('continuum_trace_ylo', None)
        self._optioninfo.setdefault('continuum_trace_ylo', 'Used in longslit only.  Lowest point on chip to trace\nout continua in case of bad area on chip.')
        self._options.setdefault('continuum_trace_yhi', None)
        self._optioninfo.setdefault('continuum_trace_yhi', 'Used in longslit only.  Highest point on chip to trace\nout continua in case of bad area on chip.')

        self._options.setdefault('drihizzle_dropsize', '1')
        self._options.setdefault('drihizzle_kernel', 'turbo')
        self._optioninfo.setdefault('drihizzle_kernel', 'turbo | point | point_replace | tophat | gaussian | fastgauss | lanczos')
        self._options.setdefault('fit_order', '2')
        self._optioninfo.setdefault('fit_order', 'Longslit only.  Order of polynomial to use to fit continua.')

        self._options.setdefault('longslit_continua_frames', None)
        self._optioninfo.setdefault('longslit_continua_frames', 'FITS file or ASCII list of FITS files to use to trace out continua')
        self._options.setdefault('max_continua_per_slit', '1')
        self._optioninfo.setdefault('max_continua_per_slit', 'Max number of continua to be traced out in each slitlet/order')
        self._options.setdefault('min_continuum_fwhm', '1.5')
        self._optioninfo.setdefault('min_continuum_fwhm', 'Minimum FWHM in pixels for tracing out continua.\nChange to lower than 1.5 if continua are narrow.')
        self._options.setdefault('min_coverage_fraction', '30')
        self._optioninfo.setdefault('min_coverage_fraction', 'Minimum percentage of continuum or skyline that must\nbe traced out in order to be included in fit.')
        self._options.setdefault('min_sky_threshold', '2.5')
        self._optioninfo.setdefault('min_sky_threshold', 'Minimum sigma threshold compared to noise for sky/lamp line\nto be detected')
        self._options.setdefault('min_threshold', '5')
        self._optioninfo.setdefault('min_threshold', 'Minimum sigma threshold compared to noise for continuum\nto be detected')

        self._options.setdefault('mos_continua_frames', None)
        self._optioninfo.setdefault('mos_continua_frames', 'FITS file or ASCII list of FITS files to use to trace out continua')
        self._options.setdefault('mos_continuum_boundary_size', 50)
        self._optioninfo.setdefault('mos_continuum_boundary_size', 'MOS only! Step size in pixels for boundary\nat edges to not attempt to trace continua.\nDefault: 50')
        self._options.setdefault('mos_continuum_step_size', 5)
        self._optioninfo.setdefault('mos_continuum_step_size', 'MOS only! Step size in pixels for tracing\n MOS continua within each slitlet.\nDefault: 5')
        self._options.setdefault('mos_double_subtract_continua', 'yes')
        self._optioninfo.setdefault('mos_double_subtract_continua', 'Set to yes if input data has been sky subtracted.\nIt will double subtract data before tracing out continua\tto increase S/N ratio.')
        self._options.setdefault('mos_find_lines_alternate_boxsize', '11')
        self._optioninfo.setdefault('mos_find_lines_alternate_boxsize', 'The boxsize in pixels in the center of the slitlet\nto use to find sky/lamp lines using alternate method.')
        self._options.setdefault('mos_find_lines_alternate_method', 'no')
        self._optioninfo.setdefault('mos_find_lines_alternate_method', 'Use alternate method to find sky/lamp lines in slitlets.\nShould be yes if slits are extremely curved like fire data.')
        self._options.setdefault('mos_fit_order', 2)
        self._optioninfo.setdefault('mos_fit_order', 'MOS only.  Order of polynomial to use to fit continua.')
        self._options.setdefault('mos_max_slit_width', 10)
        self._optioninfo.setdefault('mos_max_slit_width', 'Anything with a greater width is assumed to be a guide star\nbox and will be blanked out at this stage.')
        self._options.setdefault('mos_mode', 'use_slitpos')
        self._optioninfo.setdefault('mos_mode', 'independent_slitlets | use_slitpos | whole_chip')
        self._options.setdefault('mos_sky_fit_order', 2)
        self._optioninfo.setdefault('mos_sky_fit_order', 'MOS only! Fit order for MOS skyline\nrectification within each slitlet')
        self._options.setdefault('mos_sky_step_size', 5)
        self._optioninfo.setdefault('mos_sky_step_size', 'MOS only! Step size in pixels for tracing\n MOS skylines within each slitlet')

        self._options.setdefault('n_segments', '1')
        self._optioninfo.setdefault('n_segments', 'Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases.')

        self._options.setdefault('rectify_continua', 'yes')
        self._optioninfo.setdefault('rectify_continua', 'Turn off to skip continua and only rectify sky.')
        self._options.setdefault('rectify_sky', 'yes')
        self._optioninfo.setdefault('rectify_sky', 'Turn off to skip sky and only rectify continua.')

        self._options.setdefault('rect_coeffs_file', None)
        self._optioninfo.setdefault('rect_coeffs_file', 'file describing rectification coefficients')
        self._options.setdefault('region_file', None)
        self._optioninfo.setdefault('region_file', '.reg, .xml, or .txt file describing slitlets')

        self._options.setdefault('sky_boxsize', '6')
        self._optioninfo.setdefault('sky_boxsize', 'Boxsize in pixels for tracing out sky/lamp lines')
        self._options.setdefault('sky_fit_order', '4')
        self._optioninfo.setdefault('sky_fit_order', 'Longslit only!  Fit order for longslit skyline rectification.')
        self._options.setdefault('skyline_mask_radius', '15')
        self._optioninfo.setdefault('skyline_mask_radius', 'Number of pixels to mask on either side of a found skyline before attempting to find next one.  Set to 0 to fit and subtract Gaussian instead.')
        self._options.setdefault('sky_max_slope', 0.04)
        self._optioninfo.setdefault('sky_max_slope', 'Maximum slope of skylines.\nChange this if skylines are very tilted.')
        self._options.setdefault('sky_two_pass_detection', 'yes')
        self._optioninfo.setdefault('sky_two_pass_detection', 'Use 2-pass detection to try to ensure skylines are\nfound in both halves of image')

        self._options.setdefault('skyline_find_ylo', None)
        self._optioninfo.setdefault('skyline_find_ylo', 'Defaults to 0.  Used to specify a range of the chip to sum\na 1-d cut across and attemt to find any sky or lamp lines.\nIn the case of highly curved orders, a narrower range is needed.')
        self._options.setdefault('skyline_find_yhi', None)
        self._optioninfo.setdefault('skyline_find_yhi', 'Defaults to ysize.  Used to specify a range of the chip to sum\na 1-d cut across and attemt to find any sky or lamp lines.\nIn the case of highly curved orders, a narrower range is needed.')
        self._options.setdefault('skyline_trace_yinit', None)

        self._options.setdefault('use_arclamps', 'no')
        self._optioninfo.setdefault('use_arclamps', 'no = use master "clean sky", yes = use master arclamp')
        self._options.setdefault('use_zero_as_center_fitting', 'no')
        self._optioninfo.setdefault('use_zero_as_center_fitting', 'Use (0,0) as the center of the chip/slitlet when fitting\n(e.g. subtract x0 and y0 before least squares fit)')
        self._options.setdefault('write_noisemaps', 'no')

        self._options.setdefault('xtrans_rect_file', None)
        self._options.setdefault('ytrans_rect_file', None)

        self._options.setdefault('debug_mode','no')
        self._options.setdefault('write_plots','no')
    #end setDefaultOptions

    ## Trace out continua in each individual order/slitlet
    def traceMOSContinuaRectification(self, fdu, rctfdus, mosMode, calibs):
        ###*** For purposes of trace algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        ###*** It will trace out and fit Y = f(X) ***###
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/rectified", os.F_OK)):
            os.mkdir(outdir+"/rectified",0o755)

        #Read options
        fit_order = int(self.getOption("mos_fit_order", fdu.getTag()))
        thresh = float(self.getOption("min_threshold", fdu.getTag()))
        minCovFrac = float(self.getOption("min_coverage_fraction", fdu.getTag()))
        maxSlitWidth = float(self.getOption("mos_max_slit_width", fdu.getTag()))
        xinit = self.getOption("continuum_trace_xinit", fdu.getTag())
        find_xlo = self.getOption("continuum_find_xlo", fdu.getTag())
        find_xhi = self.getOption("continuum_find_xhi", fdu.getTag())
        maxSpectra = int(self.getOption("max_continua_per_slit", fdu.getTag()))
        n_segments = int(self.getOption("n_segments", fdu.getTag()))
        step = int(self.getOption("mos_continuum_step_size", fdu.getTag()))
        bndry = int(self.getOption("mos_continuum_boundary_size", fdu.getTag()))
        min_gauss_width = float(self.getOption("min_continuum_fwhm", fdu.getTag()))
        doDS = True
        if (self.getOption("mos_double_subtract_continua", fdu.getTag()).lower() == "no"):
            doDS = False

        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        #Set default values for xinit, find_xlo, find_xhi, convert to int
        if (xinit is None):
            xinit = xsize//2
        xinit = int(xinit)
        if (find_xlo is None):
            find_xlo = 0
        find_xlo = int(find_xlo)
        if (find_xhi is None):
            find_xhi = xsize
        find_xhi = int(find_xhi)
        #Get xstride
        xstride = xsize//n_segments

        if (not calibs['slitmask'].hasProperty("nslits")):
            calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
        nslits = calibs['slitmask'].getProperty("nslits")
        if (calibs['slitmask'].hasProperty("regions")):
            (sylo, syhi, slitx, slitw) = calibs['slitmask'].getProperty("regions")
        else:
            #Get region file for this FDU
            if (fdu.hasProperty("region_file")):
                regFile = fdu.getProperty("region_file")
            else:
                regFile = self.getCalib("region_file", fdu.getTag())
            #Check that region file exists
            if (regFile is None or not os.access(regFile, os.F_OK)):
                print("rectifyProcess::traceMOSContinuaRectification> No region file given.  Calculating regions from slitmask...")
                self._log.writeLog(__name__, "No region file given.  Calculating regions from slitmask...")
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    cut1d = calibs['slitmask'].getData()[:,xinit].astype(float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    cut1d = calibs['slitmask'].getData()[xinit,:].astype(float64)
                #detect nonzero points in 1-d cut to find regions
                slitlets = extractNonzeroRegions(cut1d, 10) #min_width = 10
                if (slitlets is None):
                    print("rectifyProcess::traceMOSContinuaRectification> ERROR: Could not find region file or calculate regions associated with "+fdu.getFullId()+"! Discarding Image!")
                    self._log.writeLog(__name__, "Could not find region file or calculate regions associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
                sylo = slitlets[:,0]
                syhi = slitlets[:,1]
                slitx = array([xinit]*len(sylo))
                slitw = array([3]*len(sylo))
            else:
                #Read region file
                if (regFile.endswith(".reg")):
                    (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".txt")):
                    (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".xml")):
                    (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                else:
                    print("rectifyProcess::traceMOSContinuaRectification> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                    self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
            calibs['slitmask'].setProperty("regions", (sylo, syhi, slitx, slitw))

        #Use GPU to calculuate xind
        yind = arange(ysize, dtype=float32).reshape(ysize,1)
        if (self._fdb.getGPUMode()):
            xind = calcXin(xsize, ysize)
        else:
            xind = arange(xsize*ysize, dtype=float32).reshape(ysize,xsize) % xsize

        #Use helper method to all ylo, yhi for each slit in each frame
        (ylos, yhis, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)

        ncont = 0
        for currFDU in rctfdus:
            shift = 0
            if (doDS):
                #Find shift for double subtraction
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    oned = sum(currFDU.getData(tag="cleanFrame")[:,find_xlo:find_xhi], 1)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    oned = sum(currFDU.getData(tag="cleanFrame")[find_xlo:find_xhi,:], 0)

                #Loop over slitlets
                for slitidx in range(nslits):
                    if (slitw[slitidx] > maxSlitWidth):
                        #Mask out guide star boxes
                        oned[ylos[slitidx]:yhis[slitidx]+1] = 0

                #Median filter positive and negative 1-d cuts
                if (self._fdb.getGPUMode()):
                    posCut = gpumedianfilter(oned)
                    negCut = gpumedianfilter(-1*oned)
                else:
                    posCut = medianfilterCPU(oned)
                    negCut = medianfilterCPU(-1*oned)

                #Mask out negative datapoints and cross correlate to find double subtract shift
                posCut[posCut < 0] = 0
                negCut[negCut < 0] = 0
                ccor = correlate(posCut,negCut,mode='same')
                mcor = where(ccor == max(ccor))[0]
                shift = len(ccor)//2-mcor[0]
                print("rectifyProcess::traceMOSContinuaRectification> Double subtract shift = "+str(shift)+"; guess was +/-"+str(currFDU.getProperty("double_subtract_guess")))
                self._log.writeLog(__name__, "Double subtract shift = "+str(shift)+"; guess was +/-"+str(currFDU.getProperty("double_subtract_guess")))

            #Loop over slitlets
            for slitidx in range(nslits):
                if (slitw[slitidx] > maxSlitWidth):
                    print("\tSlit "+str(slitidx+1)+" is a guide star box.  Skipping!")
                    self._log.writeLog(__name__, "Slit "+str(slitidx+1)+" is a guide star box.  Skipping!", printCaller=False, tabLevel=1)
                    continue
                ylo = ylos[slitidx]
                yhi = yhis[slitidx]
                print("\tSlit "+str(slitidx+1)+" = ["+str(ylo)+":"+str(yhi)+"] ...")
                self._log.writeLog(__name__, "Slit "+str(slitidx+1)+" = ["+str(ylo)+":"+str(yhi)+"] ...", printCaller=False, tabLevel=1)
                slitSize = yhi-ylo

                #Find the data corresponding to this slit and take 1-d cut
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    currMask = calibs['slitmask'].getData()[ylo:yhi+1,find_xlo:find_xhi] == (slitidx+1)
                    slit = currFDU.getData(tag="cleanFrame")[ylo:yhi+1,find_xlo:find_xhi]*currMask
                    oned = sum(slit, 1)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    currMask = calibs['slitmask'].getData()[find_xlo:find_xhi,ylo:yhi+1] == (slitidx+1)
                    slit = currFDU.getData(tag="cleanFrame")[find_xlo:find_xhi,ylo:yhi+1]*currMask
                    oned = sum(slit, 0)

                if (doDS):
                    #Median filter and mask out negative datapoints
                    posCut = medianfilterCPU(oned)
                    negCut = medianfilterCPU(-1*oned)
                    posCut[posCut < 0] = 0
                    negCut[negCut < 0] = 0

                    #Find double subtract shift and compare to overall shift found above
                    ccor = correlate(posCut,negCut,mode='same')
                    mcor = where(ccor == max(ccor))[0]
                    slitshift = len(ccor)//2-mcor[0]
                    if (abs(slitshift-shift) > 2):
                        #This shift is > 3 pixel away from the shift calculated for the whole image and is likely wrong.
                        #Probably there is no bright continuum to dominate correlation
                        continue

                    #Double subtract to bring out continuum more
                    #Use overall shift as its more likely accurate
                    posOffset = 0-min(0,shift)
                    negOffset = shift-min(0,shift)
                    dsCut = posCut[posOffset:len(posCut)-negOffset] + negCut[negOffset:len(posCut)-posOffset]
                    #Mask zeros as small positive number so they don't get flagged as bad pixels
                    dsCut[dsCut < 0] = 1.e-6
                else:
                    #Do not double subtract, just median filter and mask out zeros
                    dsCut = medianfilterCPU(oned)
                    #Mask zeros as small positive number so they don't get flagged as bad pixels
                    dsCut[dsCut < 0] = 1.e-6
                    posOffset = 0
                continuaList = extractSpectra(dsCut, sigma=thresh, width=4, nspec=maxSpectra)
                if (usePlot and (debug or writePlots)):
                    plt.plot(dsCut)
                    if (debug):
                        print(continuaList)
                    #plt.show()

                if (continuaList is None or len(continuaList) == 0):
                    continue
                #Loop over continua list
                for i in range(len(continuaList)):
                    cylo = continuaList[i][0]+ylo+posOffset
                    cyhi = continuaList[i][1]+ylo+posOffset
                    ycen = abs(cylo+cyhi)//2
                    print("\tFound continuum ["+str(cylo)+":"+str(cyhi)+"] in "+currFDU.getFullId()+"; slit "+str(slitidx+1))
                    self._log.writeLog(__name__, "Found continuum ["+str(cylo)+":"+str(cyhi)+"] in "+currFDU.getFullId()+"; slit "+str(slitidx+1), printCaller=False, tabLevel=1)
                    ncont += 1
                    if (xinit == -1 and cylo > 0):
                        #xinit == -1 => find brightest part of continuum within middle half of chip
                        #Use first kept spectrum for this purpose
                        if (currFDU.dispersion == fdu.DISPERSION_HORIZONTAL):
                            zcut = mediansmooth1d(sum(currFDU.getData(tag="cleanFrame")[cylo:cyhi+1,:],0), 5)
                        elif (currFDU.dispersion == fdu.DISPERSION_VERTICAL):
                            zcut = mediansmooth1d(sum(currFDU.getData(tag="cleanFrame")[:,cylo:cyhi+1], 1), 5)
                        xlo = int(zcut.size//4)
                        xhi = int(zcut.size*3//4)
                        xinit = where(zcut == max(zcut[xlo:xhi]))[0][0]
                    #Update fdu property "continua_list"
                    if (not currFDU.hasProperty("continua_list")):
                        currFDU.setProperty("continua_list", [(cylo, cyhi, slitidx, shift)])
                    else:
                        currFDU.getProperty("continua_list").append((cylo, cyhi, slitidx, shift))
                #endfor loop over continuaList

        if (usePlot and (debug or writePlots)):
            plt.xlabel("Pixel within slitlet")
            plt.ylabel("Flux of 1D cut of continuum")
            if (writePlots):
                plt.savefig(outdir+"/rectified/continua_"+fdu._id+".png", dpi=200)
            if (debug):
                plt.show()
            plt.close()
        print("rectifyProcess::traceMOSContinuaRectification> Using "+str(ncont)+ " continua to trace out rectification with xinit="+str(xinit)+"...")
        self._log.writeLog(__name__, "Using "+str(ncont)+ " continua to trace out rectification with xinit="+str(xinit)+"...")

        #setup output lists
        xin = []
        yin = []
        yout = []
        islit = []
        xslitin = []
        yprime = []
        iseg = []
        ncont = 0
        #setup input lists
        #xs = x values (dispersion direction) to cross correlate at
        #Start at middle and trace to end then to beginning
        #Set this up before looping over orders
        #step = 5
        #Create xs piecewise if multiple segments
        xs = list(range(xinit,xstride*(xinit//xstride+1)-bndry, step))+list(range(xinit-step, xstride*(xinit//xstride)+bndry, -1*step))
        first_seg = xinit//xstride
        #Piece together in consecutively higher then consecutively lower segments
        for seg in range(first_seg+1, n_segments):
            #Higher x vals
            xs += list(range(xstride*seg+bndry, xstride*(seg+1)-bndry, step))
        for seg in range(first_seg-1, -1, -1):
            #Lower x vals
            xs += list(range(xstride*(seg+1)-bndry, xstride*seg+bndry, -1*step))

        #Index array used for fitting
        yind = arange(ysize, dtype=float64)

        #Loop over FDUs
        for currFDU in rctfdus:
            if (not currFDU.hasProperty("continua_list")):
                continue
            #Get qa data here
            qaData = currFDU.getData(tag="cleanFrame").copy()

            #Loop over continua_list
            for (cylo, cyhi, slitidx, shift) in currFDU.getProperty("continua_list"):
                #Create new copy of data here
                currData = currFDU.getData(tag="cleanFrame").copy()
                if (doDS):
                    #Double subtract
                    negData = -1.0*currData
                    currData[currData < 0] = 0
                    negData[negData < 0] = 0

                    #Perform double subtraction
                    posOffset = 0-min(0,shift)
                    negOffset = shift-min(0,shift)
                    if (currFDU.dispersion == fdu.DISPERSION_HORIZONTAL):
                        currData[posOffset:currData.shape[0]-negOffset,:] += negData[negOffset:currData.shape[0]-posOffset,:]
                    elif (currFDU.dispersion == fdu.DISPERSION_VERTICAL):
                        currData[:,posOffset:currData.shape[1]-negOffset] += negData[:,negOffset:currData.shape[1]-posOffset]
                else:
                    #Just mask out negatives
                    currData[currData < 0] = 0

                ycen = (cyhi+cylo)//2
                yboxsize = (cyhi-cylo)//2
                if (yboxsize < 4):
                    #Minimum boxsize
                    yboxsize = 4
                if (yboxsize > 50):
                    #Maximum boxsize
                    yboxsize = 50
                #Setup lists and arrays for within each loop
                #xcoords and ycoords contain lists of fit (x,y) points
                xcoords = []
                ycoords = []
                #peak values of fits are kept and used as rejection criteria later
                peaks = []
                initpeak = 0
                #Up to last 10 (x,y) pairs are kept and used in various rejection criteria
                lastXs = []
                lastYs = []
                currX = xs[0] #current X value
                currY = ycen #shift in cross-dispersion direction at currX relative to Y at X=xinit
                gaussWidth = yboxsize/3.
                if (gaussWidth < 2):
                    #Minimum 2 pixels
                    gaussWidth = 2

                #Find shifts between segments
                seg_shifts = []
                first_seg = xinit//xstride
                for seg in range(n_segments):
                    if (seg == first_seg):
                        seg_shifts.append(0)
                    else:
                        if (seg > first_seg):
                            x1 = xstride*seg+bndry
                            x2 = xstride*seg+bndry+(find_xhi-find_xlo)
                            x3 = xstride*seg-bndry-(find_xhi-find_xlo)
                            x4 = xstride*seg-bndry
                        else:
                            x1 = xstride*(seg+1)-bndry-(find_xhi-find_xlo)
                            x2 = xstride*(seg+1)-bndry
                            x3 = xstride*(seg+1)+bndry
                            x4 = xstride*(seg+1)+bndry+(find_xhi-find_xlo)
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            currMask = calibs['slitmask'].getData()[ylos[slitidx]:yhis[slitidx]+1,x1:x2] == (slitidx+1)
                            #oned_seg1 = sum(currData[ylos[slitidx]:yhis[slitidx]+1,x1:x2]*currMask, 1) #1-d cut of curr segment
                            oned_seg1 = gpu_arraymedian(currData[ylos[slitidx]:yhis[slitidx]+1,x1:x2]*currMask, axis="X", nonzero=True) #1-d cut of curr segment
                            currMask = calibs['slitmask'].getData()[ylos[slitidx]:yhis[slitidx]+1,x3:x4] == (slitidx+1)
                            #oned_seg0 = sum(currData[ylos[slitidx]:yhis[slitidx]+1,x3:x4]*currMask, 1) #1-d cut of last segment
                            oned_seg0 = gpu_arraymedian(currData[ylos[slitidx]:yhis[slitidx]+1,x3:x4]*currMask, axis="X", nonzero=True) #1-d cut of curr segment
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            currMask = calibs['slitmask'].getData()[x1:x2,ylos[slitidx]:yhis[slitidx]+1] == (slitidx+1)
                            #oned_seg1 = sum(currData[x1:x2,ylos[slitidx]:yhis[slitidx]+1]*currMask, 0) #1-d cut of curr segment
                            oned_seg1 = gpu_arraymedian(currData[x1:x2,ylos[slitidx]:yhis[slitidx]+1]*currMask, axis="Y") #1-d cut of curr segment
                            currMask = calibs['slitmask'].getData()[x3:x4,ylos[slitidx]:yhis[slitidx]+1] == (slitidx+1)
                            #oned_seg0 = sum(currData[x3:x4,ylos[slitidx]:yhis[slitidx]+1]*currMask, 0) #1-d cut of curr segment
                            oned_seg0 = gpu_arraymedian(currData[x3:x4,ylos[slitidx]:yhis[slitidx]+1]*currMask, axis="Y") #1-d cut of curr segment
                        ccor = correlate(oned_seg0, oned_seg1, mode='same')
                        mcor = where(ccor == max(ccor))[0]
                        seg_shifts.append(len(ccor)//2-mcor[0])

                #Find the data corresponding to this slit and take 1-d cut
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    currMask = calibs['slitmask'].getData()[ylo:yhi+1,find_xlo:find_xhi] == (slitidx+1)
                    slit = currFDU.getData(tag="cleanFrame")[ylo:yhi+1,find_xlo:find_xhi]*currMask
                    oned = sum(slit, 1)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    currMask = calibs['slitmask'].getData()[find_xlo:find_xhi,ylo:yhi+1] == (slitidx+1)
                    slit = currFDU.getData(tag="cleanFrame")[find_xlo:find_xhi,ylo:yhi+1]*currMask
                    oned = sum(slit, 0)

                lastSeg = xinit//xstride #reset lastSeg
                #Loop over xs every 5 pixels and try to fit Gaussian across continuum
                for j in range(len(xs)):
                    currSeg = xs[j]//xstride
                    if (xs[j] == xinit-step):
                        #We have finished tracing to the end, starting back at middle to trace in other direction
                        if (len(ycoords) == 0):
                            #print "ERR1B"
                            break
                        #Reset currY, lastYs, lastXs
                        currY = ycoords[0]
                        lastYs = [ycoords[0]]
                        lastXs = [xcoords[0]]
                    elif (currSeg != lastSeg):
                        lastIdx = where(abs(array(xcoords)-xs[j]) == min(abs(array(xcoords)-xs[j])))[0][0]
                        currY = ycoords[lastIdx]+seg_shifts[currSeg]
                        lastYs = [ycoords[lastIdx]+seg_shifts[currSeg]]
                        lastXs = [xcoords[lastIdx]]

                    ylo = int(currY - yboxsize)
                    yhi = int(currY + yboxsize)
                    #Make sure it doesn't move off chip
                    if (ylo < 0):
                        ylo = 0
                    if (yhi >= ysize):
                        yhi = ysize-1

                    if (currFDU.dispersion == fdu.DISPERSION_HORIZONTAL):
                        outerbox = gpu_arraymedian(currData[max(0, ylo-yboxsize):yhi+yboxsize+1,xs[j]-5:xs[j]+6].copy(), axis="X")
                        y = gpu_arraymedian(currData[ylo:yhi+1,xs[j]-5:xs[j]+6].copy(), axis="X")
                    elif (currFDU.dispersion == fdu.DISPERSION_VERTICAL):
                        outerbox = gpu_arraymedian(currData[xs[j]-5:xs[j]+6,max(0, ylo-yboxsize):yhi+yboxsize+1].copy(), axis="Y")
                        y = gpu_arraymedian(currData[xs[j]-5:xs[j]+6,ylo:yhi+1].copy(), axis="Y")

                    #if (j == 0):
                    #  print slitidx, ylo, yhi, xs[j]
                    #  plt.plot(y)
                    #  plt.show()

                    #Rejection criteria
                    #outerbox = y[max(0,int(ylo-yboxsize)):int(yhi+yboxsize)]
                    noData = False
                    #Test 1: mean of inner box must be higher
                    if (y.mean() <= outerbox.mean()):
                        noData = True
                    #Test 2: median of inner box must be higher
                    if (gpu_arraymedian(y) <= gpu_arraymedian(outerbox)):
                        noData = True
                    #Test 3: max value of inner box after 3 pixel smoothing
                    #must be greater than median+1*sigma of outer box
                    #Don't let first point fail this test
                    if (j != 0 and max(smooth1dCPU(y,3,1)) < arraymedian(outerbox) + outerbox.std()):
                        noData = True
                    #If first iteration break
                    if (noData and j == 0):
                        #print "ERR1", xs[j], y.mean(), outerbox.mean(), gpu_arraymedian(y), gpu_arraymedian(outerbox), max(smooth1dCPU(y,3,1)), arraymedian(outerbox) + outerbox.std()
                        break
                    elif (noData):
                        #print "ERR1A", xs[j], y.mean(), outerbox.mean(), gpu_arraymedian(y), gpu_arraymedian(outerbox), max(smooth1dCPU(y,3,1)), arraymedian(outerbox) + outerbox.std()
                        continue

                    ylo -= yboxsize
                    yhi += yboxsize
                    if (ylo < 0):
                        ylo = 0

                    if (currFDU.dispersion == fdu.DISPERSION_HORIZONTAL):
                        y = sum(currData[ylo:yhi+1,xs[j]-5:xs[j]+6], 1, dtype=float64)
                    elif (currFDU.dispersion == fdu.DISPERSION_VERTICAL):
                        y = sum(currData[xs[j]-5:xs[j]+6,ylo:yhi+1], 0, dtype=float64)
                    y[y<0] = 0
                    y*=y

                    #if (j == 0):
                    #  print ylo, yhi, xs[j]
                    #  plt.plot(y)
                    #  plt.show()

                    #Initial guesses for Gaussian fit
                    p = zeros(4, dtype=float64)
                    p[0] = max(y)
                    p[1] = (where(y == p[0]))[0][0]+ylo
                    p[2] = gaussWidth
                    p[3] = gpu_arraymedian(y.copy())
                    if (initpeak > 0 and sqrt(p[0]) < 0.02*initpeak):
                        #peak flux must be >= 2% of highest for any 1-d cut
                        #print "ERR2A", xs[j], initpeak, p[0]
                        continue
                    if (abs(p[1]-currY) > yboxsize):
                        #Highest value is > yboxsize pixels from the previously fit peak.  Throw out this point
                        #print "ERR2", xs[j]
                        continue
                    #Range of pixels above and below continuum used for calculating std dev of background
                    stdrng = list(range(0,yboxsize+1))+list(range(len(y)-yboxsize,len(y)))
                    #if (p[0]-p[3] < 5*y[stdrng].std() and j != 0):
                    #  #Set minimum threshold at 5 sigma significance to be continnum
                    #  print xs[j], p[0]-p[3], y[stdrng].std(), (p[0]-p[3])/y[stdrng].std(), len(stdrng)
                    #  print "ERR3"
                    #  continue
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(yind[ylo:yhi+1], y))
                    except Exception as ex:
                        continue

                    #Error checking results of leastsq call
                    if (lsq[1] == 5):
                        #exceeded max number of calls = ignore
                        #print "ERR4"
                        continue
                    if (lsq[0][0]+lsq[0][3] < 0):
                        #flux less than zero = ignore
                        #print "ERR5"
                        continue
                    if (lsq[0][2] < 0 and j != 0):
                        #negative fwhm = ignore unless first datapoint
                        #print "ERR6"
                        continue
                    if (j == 0):
                        #First datapoint -- update currX, currY, append to all lists
                        currY = lsq[0][1]
                        currX = xs[0]
                        peaks.append(lsq[0][0])
                        initpeak = sqrt(lsq[0][0])
                        xcoords.append(xs[j])
                        ycoords.append(lsq[0][1])
                        lastXs.append(xs[j])
                        lastYs.append(lsq[0][1])
                        #update gaussWidth to be actual fit FWHM
                        if (lsq[0][2] < gaussWidth):
                            #1.5 pixel minimum default
                            gaussWidth = max(abs(lsq[0][2]), min_gauss_width)
                    else:
                        #FWHM is over a factor of 2 different than first fit.  Throw this point out
                        if (lsq[0][2] > 2*gaussWidth or lsq[0][2] < 0.5*gaussWidth):
                            #print "ERR7", gaussWidth, lsq[0][2]
                            continue
                        if (lastSeg != currSeg):
                            #update currX, currY, append to all lists
                            currY = lsq[0][1]
                            currX = xs[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            lastSeg = currSeg
                            continue
                        #Sanity check
                        #Calculate predicted "ref" value of Y based on slope of previous
                        #fit datapoints
                        wavg = 0.
                        wavgx = 0.
                        wavgDivisor = 0.
                        #Compute weighted avg of previously fitted values
                        #Weight by 1 over sqrt of delta-x
                        #Compare current y fit value to weighted avg instead of just
                        #previous value.
                        for i in range(len(lastYs)):
                            wavg += lastYs[i]/sqrt(abs(lastXs[i]-xs[j]))
                            wavgx += lastXs[i]/sqrt(abs(lastXs[i]-xs[j]))
                            wavgDivisor += 1./sqrt(abs(lastXs[i]-xs[j]))
                        if (wavgDivisor != 0):
                            wavg = wavg/wavgDivisor
                            wavgx = wavgx/wavgDivisor
                        else:
                            #We seem to have no datapoints in lastYs.  Simply use previous value
                            wavg = currY
                            wavgx = currX
                        #More than 50 pixels in deltaX between weight average of last 10
                        #datapoints and current X
                        #And not the discontinuity in middle of xs where we jump from end back to center
                        #because abs(xs[j]-xs[j-1]) == step
                        if (abs(xs[j]-xs[j-1]) == step and abs(wavgx-xs[j]) > 50):
                            if (len(lastYs) > 1):
                                #Fit slope to lastYs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                                slope = lin[0][1]
                            else:
                                #Only 1 datapoint, use -0.12 as slope
                                slope = -0.12
                            #Calculate guess for refY and max acceptable error
                            #err = 1+0.02*deltaX, with a max value of 3.
                            refY = wavg+slope*(xs[j]-wavgx)
                            maxerr = min(1+int(abs(xs[j]-wavgx)*.02),3)
                        else:
                            if (len(lastYs) > 3):
                                #Fit slope to lastYs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                                slope = lin[0][1]
                            else:
                                #Less than 4 datapoints, use -0.12 as slope
                                slope = -0.12
                            #Calculate guess for refY and max acceptable error
                            #0.5 <= maxerr <= 2 in this case.  Use slope*50 if it falls in that range
                            refY = wavg+slope*(xs[j]-wavgx)
                            maxerr = max(min(abs(slope*50),2),0.5)
                        #Discontinuity point in xs. Keep if within +/-1.
                        if (xs[j] == xinit-step and abs(lsq[0][1]-currY) < 1):
                            #update currX, currY, append to all lists
                            currY = lsq[0][1]
                            currX = xs[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            lastSeg = currSeg
                        elif (abs(lsq[0][1] - refY) < maxerr):
                            #Regular datapoint.  Apply sanity check rejection criteria here
                            #Discard if farther than maxerr away from refY
                            if (abs(xs[j]-currX) < 4*step and maxerr > 1 and abs(lsq[0][1]-currY) > maxerr):
                                #Also discard if < 20 pixels in X from last fit datapoint, and deltaY > 1
                                continue
                            #update currX, currY, append to all lists
                            currY = lsq[0][1]
                            currX = xs[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            lastSeg = currSeg
                            #keep lastXs and lastYs at 10 elements or less
                            if (len(lastYs) > 10):
                                lastXs.pop(0)
                                lastYs.pop(0)
                        else:
                            #More than maxerr away from refY.  Don't save datapoint except for use in lastXs and lastYs
                            #print "ERR8", lsq[0][1], refY, abs(lsq[0][1] - refY), maxerr
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            #keep lastXs and lastYs at 10 elements or less
                            if (len(lastYs) > 10):
                                lastXs.pop(0)
                                lastYs.pop(0)
                    #print xs[j], p[1], lsq[0][1], lsq[0][0], lsq[0][2]
                print("rectifyProcess::traceMOSContinuaRectification> Continuum centered at "+str(ycen)+" in "+currFDU.getFullId()+": found "+str(len(ycoords))+" datapoints.")
                self._log.writeLog(__name__, "Continuum centered at "+str(ycen)+" in "+currFDU.getFullId()+": found "+str(len(ycoords))+" datapoints.")
                #Check coverage fraction
                covfrac = len(ycoords)*100.0/len(xs)
                if (covfrac < minCovFrac):
                    print("rectifyProcess::traceMOSContinuaRectification> Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping continuum!")
                    self._log.writeLog(__name__, "Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping continuum!")
                    continue
########NOTES: Possibly let lines be traced out as-is and only start applying segments here.  Return isegment along with islit which can be used for
########calculating separate tranformations.

                #Phase 2 of rejection criteria after continua have been traced
                #Find outliers > 2.5 sigma in peak values and remove them
                #First store first value as yout
                currYout = ycoords[0]
                peaks = array(peaks)
                xcoords = array(xcoords)
                ycoords = array(ycoords)
                xc_keep = [] #Create new lists for xcoords and ycoords that will be kept
                yc_keep = []
                iseg_keep = [] #And for segment number of those kept datapoints

                for seg in range(n_segments):
                    xstride = xsize//n_segments
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                    segmask = (xcoords >= sxlo)*(xcoords < sxhi)
                    peakmed = arraymedian(peaks[segmask])
                    peaksd = peaks[segmask].std()
                    b = (peaks[segmask] > peakmed-2.5*peaksd)*(peaks[segmask] < peakmed+2.5*peaksd)
                    seg_xcoords = xcoords[segmask][b]
                    seg_ycoords = ycoords[segmask][b]

                    if (len(seg_xcoords) < 5):
                        continue

                    if (n_segments > 1):
                        print("\tSegment "+str(seg)+": rejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" of "+str(len(ycoords[segmask]))+" datapoints.")
                        self._log.writeLog(__name__, "Segment "+str(seg)+": rejecting outliers (phase 2) - kept "+str(len(ycoords))+" of "+str(len(ycoords[segmask]))+" datapoints.", printCaller=False, tabLevel=1)
                    else:
                        print("\trejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" datapoints.")
                        self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" datapoints.", printCaller=False, tabLevel=1)

                    #Fit fit_order order order polynomial to datapoints, Y = f(X)
                    #order = 2
                    order = fit_order
                    p = zeros(order+1, float64)
                    p[0] = ycoords[0]
                    try:
                        lsq = leastsq(polyResiduals, p, args=(seg_xcoords,seg_ycoords,order))
                    except Exception as ex:
                        continue

                    #Compute output offsets and residuals from actual datapoints
                    yprime = polyFunction(lsq[0], seg_xcoords, order)
                    yresid = yprime-seg_ycoords
                    #Remove outliers and refit
                    b = abs(yresid) < yresid.mean()+2.5*yresid.std()
                    seg_xcoords = seg_xcoords[b]
                    seg_ycoords = seg_ycoords[b]
                    #Append to keep lists
                    xc_keep.extend(seg_xcoords)
                    yc_keep.extend(seg_ycoords)
                    iseg_keep.extend([seg]*len(seg_ycoords))
                    if (n_segments > 1):
                        print("\tSegment "+str(seg)+": rejecting outliers (phase 3). Sigma = "+str(yresid.std())[:5]+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.")
                        self._log.writeLog(__name__, "Segment "+str(seg)+": rejecting outliers (phase 3). Sigma = "+str(yresid.std())[:5]+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)
                    else:
                        print("\trejecting outliers (phase 3). Sigma = "+str(yresid.std())[:5]+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.")
                        self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+str(yresid.std())[:5]+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)

                #Copy back over to xcoords, ycoords
                xcoords = array(xc_keep)
                ycoords = array(yc_keep)
                #Check coverage fraction
                covfrac = len(ycoords)*100.0/len(xs)
                if (covfrac >= minCovFrac):
                    xin.extend(xcoords)
                    yin.extend(ycoords)
                    yout.extend([currYout]*len(ycoords))
                    if (mosMode == "use_slitpos"):
                        xslitin.extend([slitx[slitidx]]*len(ycoords))
                    elif (mosMode == "independent_slitlets"):
                        islit.extend([slitidx+1]*len(ycoords))
                        iseg.extend(iseg_keep)
                    ncont += 1
                    qavalue = -50000
                    #Generate qa data
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        for i in range(len(xcoords)):
                            yval = int(ycoords[i]+.5)
                            xval = int(xcoords[i]+.5)
                            for yi in range(yval-1,yval+2):
                                for xi in range(xval-1,xval+2):
                                    dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                    qaData[yi,xi] = qavalue/((1+dist)**2)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        for i in range(len(xcoords)):
                            yval = int(ycoords[i]+.5)
                            xval = int(xcoords[i]+.5)
                            for yi in range(yval-1,yval+2):
                                for xi in range(xval-1,xval+2):
                                    dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                    qaData[xi,yi] = qavalue/((1+dist)**2)

                print("\tY Center: "+formatNum(currYout)+"\t Xref: "+str(xs[0])+"\t Cov. Frac: "+formatNum(covfrac))
                self._log.writeLog(__name__, "Y Center: "+formatNum(currYout)+"\t Xref: "+str(xs[0])+"\t Cov. Frac: "+formatNum(covfrac), printCaller=False, tabLevel=1)
                del currData

            #Create output filename
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/rectified", os.F_OK)):
                os.mkdir(outdir+"/rectified",0o755)
            qafile = outdir+"/rectified/qa_"+currFDU.getFullId()

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                if (os.access(qafile, os.F_OK)):
                    os.unlink(qafile)
            #Write out qa file
            if (not os.access(qafile, os.F_OK)):
                currFDU.tagDataAs("slitqa", qaData)
                currFDU.writeTo(qafile, tag="slitqa")
                currFDU.removeProperty("slitqa")
            del qaData

        print("rectifyProcess::traceMOSContinuaRectification> Successfully traced out "+str(len(yout))+" datapoints in "+str(ncont)+ " continua.")
        self._log.writeLog(__name__, "Successfully traced out "+str(len(yout))+" datapoints in "+str(ncont)+ " continua.")
        #Convert to arrays
        xin = array(xin)
        yin = array(yin)
        yout = array(yout)
        if (mosMode == "use_slitpos"):
            xslitin = array(xslitin)
        elif (mosMode == "independent_slitlets"):
            islit = array(islit)

        #Write out more qa data
        qafile = outdir+"/rectified/qa_"+fdu._id+"-continua.dat"
        f = open(qafile,'w')
        if (mosMode == "use_slitpos"):
            for i in range(len(yout)):
                f.write(str(xin[i])+'\t'+str(yin[i])+'\t'+str(yout[i])+'\t'+str(xslitin[i])+'\n')
        elif (mosMode == "independent_slitlets"):
            for i in range(len(yout)):
                f.write(str(xin[i])+'\t'+str(yin[i])+'\t'+str(yout[i])+'\t'+str(islit[i])+'\t'+str(iseg[i])+'\n')
        elif (mosMode == "whole_chip"):
            for i in range(len(yout)):
                f.write(str(xin[i])+'\t'+str(yin[i])+'\t'+str(yout[i])+'\n')
        f.close()

        if (mosMode == "use_slitpos"):
            return (xin, yin, yout, xslitin)
        elif (mosMode == "independent_slitlets"):
            return (xin, yin, yout, islit, iseg)
        return (xin, yin, yout)
    #end traceMOSContinuaRectification


    ## Trace out skylines in each individual order/slitlet
    def traceMOSSkylineRectification(self, fdu, skyFDU, calibs):
        ###*** For purposes of trace algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        ###*** It will trace out and fit X = f(Y) ***###

        if (skyFDU is None):
            print("rectifyProcess::traceMOSSkylineRectification> Warning: Could not find clean sky frame associated with "+fdu.getFullId()+"!  Skylines will NOT be rectified!")
            self._log.writeLog(__name__, "Could not find clean sky frame associated with "+fdu.getFullId()+"!  Skylines will NOT be rectified!", type=fatboyLog.WARNING)
            return None

        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/rectified", os.F_OK)):
            os.mkdir(outdir+"/rectified",0o755)

        #Read options
        fit_order = int(self.getOption("mos_sky_fit_order", fdu.getTag()))
        thresh = float(self.getOption("min_sky_threshold", fdu.getTag()))
        skylineMaskRadius = int(self.getOption("skyline_mask_radius", fdu.getTag()))
        minCovFrac = float(self.getOption("min_coverage_fraction", fdu.getTag()))
        skybox = int(self.getOption("sky_boxsize", fdu.getTag()))
        maxSlitWidth = float(self.getOption("mos_max_slit_width", fdu.getTag()))
        maxSlope = float(self.getOption("sky_max_slope", fdu.getTag()))
        step = int(self.getOption("mos_sky_step_size", fdu.getTag()))
        n_segments = int(self.getOption("n_segments", fdu.getTag()))
        min_gauss_width = float(self.getOption("min_continuum_fwhm", fdu.getTag()))
        useTwoPasses = False
        if (self.getOption("sky_two_pass_detection", fdu.getTag()).lower() == "yes"):
            useTwoPasses = True
        useArclamps = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True
        useAlternate = False
        if (self.getOption("mos_find_lines_alternate_method", fdu.getTag()).lower() == "yes"):
            useAlternate = True
        alternateBoxsize = int(self.getOption("mos_find_lines_alternate_boxsize", fdu.getTag()))

        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        print("rectifyProcess::traceMOSSkylineRectification> Searching for sky/arclamp to trace out, using "+skyFDU.getFullId()+"...")
        self._log.writeLog(__name__, "Searching for sky/arclamp to trace out, using "+skyFDU.getFullId()+"...")

        skyData = skyFDU.getData().copy()

        if (not calibs['slitmask'].hasProperty("nslits")):
            calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
        nslits = calibs['slitmask'].getProperty("nslits")
        if (calibs['slitmask'].hasProperty("regions")):
            (sylo, syhi, slitx, slitw) = calibs['slitmask'].getProperty("regions")
        else:
            #Get region file for this FDU
            if (fdu.hasProperty("region_file")):
                regFile = fdu.getProperty("region_file")
            else:
                regFile = self.getCalib("region_file", fdu.getTag())
            #Check that region file exists
            if (regFile is None or not os.access(regFile, os.F_OK)):
                print("rectifyProcess::traceMOSSkylineRectification> No region file given.  Calculating regions from slitmask...")
                self._log.writeLog(__name__, "No region file given.  Calculating regions from slitmask...")
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    cut1d = calibs['slitmask'].getData()[:,xsize//2].astype(float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    cut1d = calibs['slitmask'].getData()[xsize//2,:].astype(float64)
                #detect nonzero points in 1-d cut to find regions
                slitlets = extractNonzeroRegions(cut1d, 10) #min_width = 10
                if (slitlets is None):
                    print("rectifyProcess::traceMOSSkylineRectification> ERROR: Could not find region file or calculate regions associated with "+fdu.getFullId()+"! Discarding Image!")
                    self._log.writeLog(__name__, "Could not find region file or calculate regions associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
                sylo = slitlets[:,0]
                syhi = slitlets[:,1]
                slitx = array([xsize//2]*len(sylo))
                slitw = array([3]*len(sylo))
            else:
                #Read region file
                if (regFile.endswith(".reg")):
                    (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".txt")):
                    (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                elif (regFile.endswith(".xml")):
                    (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
                else:
                    print("rectifyProcess::traceMOSSkylineRectification> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                    self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return None
            calibs['slitmask'].setProperty("regions", (sylo, syhi, slitx, slitw))

        #Use GPU to calculuate xind
        yind = arange(ysize, dtype=float32).reshape(ysize,1)
        if (self._fdb.getGPUMode()):
            xind = calcXin(xsize, ysize)
        else:
            xind = arange(xsize*ysize, dtype=float32).reshape(ysize,xsize) % xsize

        #Use helper method to all ylo, yhi for each slit in each frame
        (ylos, yhis, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)

        #Setup output lists
        xin = []
        yin = []
        xout = []
        islit = []
        iseg = []
        xprime = []
        #qa data
        qaData = skyFDU.getData().copy()

        #Loop over slitlets
        for slitidx in range(nslits):
            if (slitw[slitidx] > maxSlitWidth):
                print("\tSlit "+str(slitidx+1)+" is a guide star box.  Skipping!")
                self._log.writeLog(__name__, "Slit "+str(slitidx+1)+" is a guide star box.  Skipping!", printCaller=False, tabLevel=1)
                continue
            ylo = ylos[slitidx]
            yhi = yhis[slitidx]
            print("\tSlit "+str(slitidx+1)+" = ["+str(ylo)+":"+str(yhi)+"] ...")
            self._log.writeLog(__name__, "Slit "+str(slitidx+1)+" = ["+str(ylo)+":"+str(yhi)+"] ...", printCaller=False, tabLevel=1)
            slitSize = yhi-ylo
            xcenters = []

            #Find the data corresponding to this slit and take 1-d cut
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                currMask = calibs['slitmask'].getData()[ylo:yhi+1,:] == (slitidx+1)
                slit = skyData[ylo:yhi+1,:]*currMask
                oned = sum(slit, 0)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                currMask = calibs['slitmask'].getData()[:,ylo:yhi+1] == (slitidx+1)
                slit = skyData[:,ylo:yhi+1]*currMask
                oned = sum(slit, 1)

            #write_fits_file('slit.fits', slit)
            if (useAlternate):
                halfbox = alternateBoxsize//2
                #Alternate method is finding n (default 11) central pixels of slit and taking a median cut at each x
                for j in range(len(oned)):
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        b = where(calibs['slitmask'].getData()[ylo:yhi+1,j] == slitidx+1)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        b = where(calibs['slitmask'].getData()[j,ylo:yhi+1] == slitidx+1)
                    if (len(b[0]) < alternateBoxsize):
                        oned[j] = 0
                        continue
                    ycen = (b[0].min()+b[0].max())//2
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        oned[j] = gpu_arraymedian(slit[ycen-halfbox:ycen+halfbox+1,j])
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        oned[j] = gpu_arraymedian(slit[j,ycen-halfbox:ycen+halfbox+1])

            #Median filter 1-d cut
            if (self._fdb.getGPUMode()):
                oned = gpumedianfilter(oned)
            else:
                oned = medianfilterCPU(oned)

            #Next smooth the 1-d cut
            if (self._fdb.getGPUMode()):
                z = smooth1d(oned,5,1)
            else:
                z = smooth1dCPU(oned, 5, 1)
            medVal = arraymedian(z, nonzero=True)
            sig = z[where(z != 0)].std()

            if (usePlot and (debug or writePlots)):
                plt.plot(z)

            #Zero out the first and last 15 pixels
            z[:15] = 0.
            z[-15:] = 0.

            #First pass at globally finding sky lines
            while (max(z) > medVal+thresh*sig):
                if (max(z) <= 0):
                    #All points have been zeroed out
                    break
                #Find brightest point
                b = (where(z == max(z)))[0][0]
                #Check that it matches min threshold
                if (z[b] > medVal+thresh*sig):
                    valid = True
                    #If data within +/-5 pixels has been zeroed out already, this is
                    #too near another line
                    for l in range(b-5,b+6):
                        if (z[l] == 0):
                            valid = False
                    #This line is valid -- it is at least 6 pixels away from another line
                    if valid:
                        #Append to linelist
                        xcenters.append(b)
                        if (self._fdb._verbosity == fatboyLog.VERBOSE):
                            print("\t\tFound skyline: Center = "+str(b)+"; sigma = "+str((z[b]-medVal)/sig))
                            self._log.writeLog(__name__, "Found skyline: Center = "+str(b)+"; sigma = "+str((z[b]-medVal)/sig), printCaller=False, tabLevel=2, verbosity=fatboyLog.VERBOSE)
                    #Zero out +/-skylineMaskRadius (default 15) pixels from this line
                    if (skylineMaskRadius != 0):
                        z[b-skylineMaskRadius:b+skylineMaskRadius] = 0
                    else:
                        #Fit and subtract gaussian
                        refCut = z[b-10:b+11]**2
                        p = zeros(4, dtype=float64)
                        p[0] = max(refCut)
                        p[1] = 10
                        p[2] = 2
                        p[3] = gpu_arraymedian(refCut)
                        lsq = leastsq(gaussResiduals, p, args=(arange(len(refCut), dtype=float64), refCut))
                        #Subtract line from z
                        p = zeros(4)
                        p[0] = sqrt(abs(lsq[0][0]))
                        p[1] = b+lsq[0][1]-10
                        p[2] = abs(lsq[0][2]*sqrt(2))
                        z -= gaussFunction(p, arange(len(z), dtype=float32))
                        z[int(b-lsq[0][2]):int(b+lsq[0][2]+0.5)] = 0
                    #Update median and std dev
                    medVal = arraymedian(z, nonzero=True)
                    sig = z[where(z != 0)].std()
            #Split into 4 sections for two pass detection
            if (useTwoPasses):
                if (self._fdb._verbosity == fatboyLog.VERBOSE):
                    print("rectifyProcess::traceMOSSkylineRectification> Pass 2: Searching for additional sky/arclamp lines...")
                    self._log.writeLog(__name__, "Pass 2: Searching for additional sky/arclamp lines...", verbosity=fatboyLog.VERBOSE)
                xs = []
                for k in range(50,xsize-49,(xsize-100)//3):
                    xs.append(k)
                #Loop over each section
                for k in range(len(xs)-1):
                    #Create new array with "z" value for just this section and zero out edges
                    zlocal = z[xs[k]:xs[k+1]]
                    zlocal[0:15] = 0.
                    zlocal[-15:] = 0.
                    medVal = arraymedian(zlocal, nonzero=True)
                    #Calculate sigma if there are nonzero points left in this section
                    if (len(where(zlocal != 0)[0]) != 0):
                        sig = zlocal[where(zlocal != 0)].std()
                    else:
                        #set sigma to 10000 - just a way to fail condition of while loop
                        sig=10000.
                    #Always try to get at least one line if there is any nonzero data
                    firstPass = True
                    while (max(zlocal) > medVal+thresh*sig or firstPass):
                        if (max(zlocal) <= 0):
                            #All points have been zeroed out
                            break
                        #Find brightest point
                        b = (where(zlocal == max(zlocal)))[0][0]
                        #If first pass, lower threshold to just 1.5 sigma so that at least one point
                        #can be used to anchor fit in this section
                        if (firstPass and zlocal[b] < medVal+1.5*sig):
                            #Only if < 1.5 sigma, discard
                            firstPass = False
                            continue
                        #If this line matches min threshold OR is the first pass and is >= 1.5 sigma, keep
                        if (zlocal[b] > medVal+thresh*sig or firstPass):
                            valid = True
                            #If data within +/-5 pixels has been zeroed out already, this is
                            #too near another line
                            for l in range(b-5,b+6):
                                if (zlocal[l] == 0):
                                    valid = False
                            #print b+xs[k], len(where(zlocal != 0)[0]), (zlocal[b]-med)/sig
                            firstPass = False
                            #This line is valid -- it is at least 6 pixels away from another line
                            if valid:
                                #Append to linelist
                                xcenters.append(b+xs[k])
                                if (self._fdb._verbosity == fatboyLog.VERBOSE):
                                    print("\t\tFound skyline: Center = "+str(b+xs[k])+"; sigma = "+str((zlocal[b]-medVal)/sig))
                                    self._log.writeLog(__name__, "Found skyline: Center = "+str(b+xs[k])+"; sigma = "+str((zlocal[b]-medVal)/sig), printCaller=False, tabLevel=2, verbosity=fatboyLog.VERBOSE)
                            #Zero out +/-15 pixels from this line in second pass
                            zlocal[b-15:b+15] = 0
                            if (len(where(zlocal != 0)[0]) == 0):
                                break
                            #update median and sd
                            medVal = arraymedian(zlocal, nonzero=True)
                            sig = zlocal[where(zlocal != 0)].std()

            #Use brightest skyline to trace out range in y where skylines are visible
            #(In case they cut off before the top/bottom of the slitlet/order)
            nlines = len(xcenters)
            bcen = int(xcenters[0])
            #Sum 1-d cut at 5 pixels centered around skyline
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                bline = sum(slit[:,bcen-2:bcen+3],1, dtype=float64)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                bline = sum(slit[bcen-2:bcen+3,:],0, dtype=float64)
            #Calculate median and std dev
            blmed = arraymedian(bline)
            blsig = bline.std()

            #Default values
            slitylo = 0
            slityhi = len(bline)-1

            #Find lower cutoff
            isslitylo = False
            currY = slitSize//2
            while (not isslitylo and currY > 5):
                #If 5 consecutive points in 1-d cut are below -2 sigma, we've found lower cutoff
                if (alltrue(bline[currY-4:currY+1] < blmed-2*blsig)):
                    slitylo = currY
                    isslitylo = True
                currY-=1
            #Find upper cutoff
            isslityhi = False
            currY = slitSize//2
            while (not isslityhi and currY < slitSize-5):
                #If 5 consecutive points in 1-d cut are below -2 sigma, we've found upper cutoff
                if (alltrue(bline[currY:currY+5] < blmed-2*blsig)):
                    slityhi = currY
                    isslityhi = True
                currY+=1

            print("\t\tUsing "+str(nlines)+ " lines to trace out skyline rectification over y-range = ["+str(slitylo+ylo)+":"+str(slityhi+ylo)+"]...")
            self._log.writeLog(__name__, "Using "+str(nlines)+ " lines to trace out skyline rectification over y-range = ["+str(slitylo+ylo)+":"+str(slityhi+ylo)+"]...", printCaller=False, tabLevel=2)

            if (usePlot and (debug or writePlots)):
                plt.xlabel("Pixel, Slitlet "+str(slitidx)+"; Nlines = "+str(nlines))
                plt.ylabel("Flux; y-range = ["+str(slitylo+ylo)+":"+str(slityhi+ylo)+"]")
                if (writePlots):
                    plt.savefig(outdir+"/rectified/skylines_"+fdu._id+"_slit_"+str(slitidx)+".png", dpi=200)
                if (debug):
                    plt.show()
                plt.close()

            #Setup list of y positions every 5 pixels
            yinit = (slitylo+slityhi)//2
            #step = 5
            ys = list(range(yinit, slitylo-step, -1*step))+list(range(yinit+step, slityhi, step))
            #Index array used for fitting
            slitxind = arange(xsize, dtype=float64)

            #Setup output lists
            slitxin = []
            slityin = []
            slitxout = []
            nlines = 0

            #Median filter whole image to bring out skylines first
            if (self._fdb.getGPUMode()):
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    slit = gpumedianfilter2d(slit)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    slit = gpumedianfilter2d(slit, axis="Y")
            else:
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    slit = medianfilter2dCPU(slit)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    slit = medianfilter2dCPU(slit, axis="Y")

            #write_fits_file('slit_mf.fits', slit)
            lineMask = ones(slit.shape, int32) #Array to mask that have been found to subtract from data

            #Loop over list of skylines
            for xcen in xcenters:
                if (useAlternate):
                    #Update yinit and ys here based on this individual line
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        b = where(calibs['slitmask'].getData()[ylo:yhi+1,xcen] == slitidx+1)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        b = where(calibs['slitmask'].getData()[xcen,ylo:yhi+1] == slitidx+1)
                    lineylo = max(b[0].min(), 5)
                    lineyhi = min(b[0].max(), ysize-6)
                    yinit = (lineylo+lineyhi)//2
                    ys = list(range(yinit, lineylo-1, -1*step))+list(range(yinit+step, lineyhi+1, step))

                currX = xcen
                currY = ys[0]
                lastY = ys[0]
                #Setup lists and arrays for within each loop
                #xcoords and ycoords contain lists of fit (x,y) points
                xcoords = []
                ycoords = []
                #peak values of fits are kept and used as rejection criteria later
                peaks = []
                #Up to last 10 (x,y) pairs are kept and used in various rejection criteria
                lastXs = []
                lastYs = []
                gaussWidth = skybox/3.
                if (gaussWidth < 2):
                    #Minimum 2 pixels
                    gaussWidth = 2
                gaussWidth /= sqrt(2)

                if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                    print("LINE", xcen, yinit)
                #Loop over ys every 5 pixels and try to fit Gaussian across skyline
                for j in range(len(ys)):
                    if (ys[j] == yinit+step):
                        if (len(xcoords) == 0):
                            #No data for first half of line including first point
                            break
                        #We have finished tracing to the end, starting back at middle to trace in other direction
                        #Reset currX, lastYs, lastXs
                        currX = xcoords[0]
                        lastYs = [yinit]
                        lastXs = [xcoords[0]]
                    xlo = int(currX - skybox)
                    xhi = int(currX + skybox)+1
                    if (xlo < 0):
                        continue

                    #multiply by lineMask once here
                    slitm = slit*lineMask
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        outerbox = gpu_arraymedian((slitm)[ys[j]-5:ys[j]+6,max(0, xlo-skybox):xhi+skybox].copy(), axis="Y")
                        x = gpu_arraymedian((slitm)[ys[j]-5:ys[j]+6,xlo:xhi].copy(), axis="Y")
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        outerbox = gpu_arraymedian((slitm)[max(0, xlo-skybox):xhi+skybox,ys[j]-step:ys[j]+step+1].copy(), axis="X")
                        x = gpu_arraymedian((slitm)[xlo:xhi,ys[j]-step:ys[j]+step+1].copy(), axis="X")
                    if (type(x) == int):
                        continue
                    #if (j == 0 and slitidx == 2):
                        #plt.plot(arange(len(outerbox))+xlo-skybox, outerbox)
                        #plt.plot(arange(len(x))+xlo, x)
                        #plt.show()

                    #Rejection criteria
                    #outerbox = x[max(0,int(xlo-skybox)):int(xhi+skybox)]
                    noData = False
                    #Test 1: mean of inner box must be higher
                    if (x.mean() < outerbox.mean()):
                        noData = True
                    #Test 2: median of inner box must be higher
                    if (gpu_arraymedian(x, nonzero=True) < gpu_arraymedian(outerbox, nonzero=True)):
                        noData = True
                    #Test 3: max value of inner box after 3 pixel smoothing
                    #must be greater than median+1*sigma of outer box
                    #Don't let first point fail this test
                    if (j != 0 and max(smooth1dCPU(x,3,1)) < arraymedian(outerbox) + outerbox.std()):
                        noData = True
                    #If first iteration break
                    if (noData and j == 0):
                        #print "ERR1", x.mean(), outerbox.mean(), gpu_arraymedian(x), gpu_arraymedian(outerbox)
                        break
                    elif (noData):
                        #print "ERR1A", x.mean(), outerbox.mean(), gpu_arraymedian(x), gpu_arraymedian(outerbox), max(smooth1dCPU(x,3,1)), outerbox.std()
                        continue

                    xlo -= skybox
                    xhi += skybox
                    if (xlo < 0):
                        xlo = 0

                    #Sum 11 pixel box and fit 1-d Gaussian
                    #Use 11 instead of 5 to wash out noise more and obtain better fit
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        x = sum((slitm)[ys[j]-step:ys[j]+step+1,xlo:xhi], 0, dtype=float64)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        x = sum((slitm)[xlo:xhi,ys[j]-step:ys[j]+step+1], 1, dtype=float64)
                    if (len(x) < 3*skybox):
                        continue
                    x[x<0] = 0

                    #Initial guesses for Gaussian fit
                    p = zeros(4, dtype=float64)
                    p[0] = max(x[skybox:-skybox])
                    p[1] = (where(x == p[0]))[0][0]+xlo
                    p[2] = gaussWidth
                    p[3] = gpu_arraymedian(x.copy())
                    if (abs(p[1]-currX) > skybox):
                        #Highest value is > skybox pixels from the previously fit peak.  Throw out this point
                        #print "ERR2"
                        continue
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(slitxind[xlo:xhi], x))
                    except Exception as ex:
                        continue

                    #Error checking results of leastsq call
                    if (lsq[1] == 5):
                        #exceeded max number of calls = ignore
                        #print "ERR3"
                        continue
                    if (lsq[0][0]+lsq[0][3] < 0 and j != 0):
                        #flux less than zero = ignore unless first datapoint
                        #print "ERR4"
                        continue
                    if (lsq[0][2] < 0 and j != 0):
                        #negative fwhm = ignore unless first datapoint
                        #print "ERR5"
                        continue
                    if (j == 0):
                        #First datapoint -- update currX, currY, append to all lists
                        currX = lsq[0][1]
                        currY = ys[0]
                        peaks.append(lsq[0][0])
                        xcoords.append(lsq[0][1])
                        ycoords.append(ys[j]+ylo)
                        lastXs.append(lsq[0][1])
                        lastYs.append(ys[j])
                        #update gaussWidth to be actual fit FWHM
                        if (lsq[0][2] < gaussWidth and lsq[0][2] > 1):
                            #1.5 pixel minimum default
                            gaussWidth = max(abs(lsq[0][2]), min_gauss_width)
                    else:
                        #FWHM is over a factor of 2 different than first fit.  Throw this point out
                        if (lsq[0][2] > 2*gaussWidth or lsq[0][2] < 0.5*gaussWidth):
                            if (gaussWidth == skybox/3. and lsq[0][2] < 0.5*gaussWidth and lsq[0][2] > 1):
                                #Special case, gaussWidth did not update on first pass because it was super narrow.
                                #Update here.
                                gaussWidth = max(abs(lsq[0][2]), 1.5)
                            else:
                                #print "ERR6", gaussWidth, lsq[0][2]
                                continue
                        #Sanity check
                        #Calculate predicted "ref" value of X based on slope of previous
                        #fit datapoints
                        wavg = 0.
                        wavgy = 0.
                        wavgDivisor = 0.
                        #Compute weighted avg of previously fitted values
                        #Weight by 1 over sqrt of delta-y
                        #Compare current x fit value to weighted avg instead of just
                        #previous value.
                        for i in range(len(lastXs)):
                            wavg += lastXs[i]/sqrt(abs(lastYs[i]-ys[j]))
                            wavgy += lastYs[i]/sqrt(abs(lastYs[i]-ys[j]))
                            wavgDivisor += 1./sqrt(abs(lastYs[i]-ys[j]))
                        if (wavgDivisor != 0):
                            wavg = wavg/wavgDivisor
                            wavgy = wavgy/wavgDivisor
                        else:
                            #We seem to have no datapoints in lastXs.  Simply use previous value
                            wavg = currX
                            wavgy = currY
                        #More than 50 pixels in deltaY between weight average of last 10
                        #datapoints and current Y
                        #And not the discontinuity in middle of ys where we jump from end back to center
                        #because abs(ys[j]-ys[j-1]) == step
                        if (abs(ys[j]-ys[j-1]) == step and abs(wavgy-ys[j]) > 50):
                            if (len(lastXs) > 1):
                                #Fit slope to lastXs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastYs),array(lastXs)))
                                slope = lin[0][1]
                            else:
                                #Only 1 datapoint, use +/- maxSlope as slope
                                slope = -1*abs(maxSlope)
                                if ((lsq[0][1]-wavg)/(ys[j]-wavgy) > 0):
                                    slope = abs(maxSlope)
                            #Calculate guess for refX and max acceptable error
                            #err = 1+maxSlope*deltaY, with a max value of 3.
                            refX = wavg+slope*(ys[j]-wavgy)
                            maxerr = min(1+int(abs(ys[j]-wavgy)*.02),max(3,10*maxSlope))
                        else:
                            if (len(lastXs) > 2):
                                #Fit slope to lastXs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastYs),array(lastXs)))
                                slope = lin[0][1]
                            else:
                                #Less than 4 datapoints, use +/-maxSlope as slope
                                slope = -1*abs(maxSlope)
                                if ((lsq[0][1]-wavg)/(ys[j]-wavgy) > 0):
                                    slope = abs(maxSlope)
                            #Calculate guess for refX and max acceptable error
                            #0.5 <= maxerr <= 2 in this case.  Use slope*50 if it falls in that range
                            refX = wavg+slope*(ys[j]-wavgy)
                            maxerr = max(min(abs(slope*50),max(2,10*maxSlope)),0.5)
                        #Discontinuity point in ys. Keep if within +/- 1 (or step*maxSlope)
                        if (ys[j] == yinit+step and abs(lsq[0][1]-currX) < max(step*abs(maxSlope), 1)):
                            #update currX, currY, append to all lists
                            currX = lsq[0][1]
                            currY = ys[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(lsq[0][1])
                            ycoords.append(ys[j]+ylo)
                            lastXs.append(lsq[0][1])
                            lastYs.append(ys[j])
                        elif (abs(lsq[0][1] - refX) < maxerr):
                            #Regular datapoint.  Apply sanity check rejection criteria here
                            #Discard if farther than maxerr away from refX
                            if (abs(ys[j]-currY) < 4*step and maxerr > 1 and abs(lsq[0][1]-currX) > maxerr):
                                #Also discard if < 20 pixels in Y from last fit datapoint, and deltaX > 1
                                #print "ERRX", abs(ys[j]-currY), 4*step, maxerr, lsq[0][1], currX, abs(lsq[0][1]-currX), refX
                                continue
                            #update currX, currY, append to all lists
                            currX = lsq[0][1]
                            currY = ys[j]
                            peaks.append(lsq[0][0])
                            xcoords.append(lsq[0][1])
                            ycoords.append(ys[j]+ylo)
                            lastXs.append(lsq[0][1])
                            lastYs.append(ys[j])
                            #keep lastXs and lastYs at 10 elements or less
                            if (len(lastYs) > 10):
                                lastXs.pop(0)
                                lastYs.pop(0)
                        #else:
                            #print "ERR8", lsq[0][1], refX, abs(lsq[0][1] - refX), maxerr
                    #print ys[j], p[1], lsq[0][1], lsq[0][0], lsq[0][2]
                if (self._fdb._verbosity == fatboyLog.VERBOSE):
                    print("\t\tLine centered at "+str(xcen)+" in "+skyFDU.getFullId()+": found "+str(len(xcoords))+" datapoints.")
                    self._log.writeLog(__name__, "Line centered at "+str(xcen)+" in "+skyFDU.getFullId()+": found "+str(len(xcoords))+" datapoints.", printCaller=False, tabLevel=2, verbosity=fatboyLog.VERBOSE)
                #Check coverage fraction
                covfrac = len(xcoords)*100.0/len(ys)
                if (covfrac < minCovFrac):
                    if (self._fdb._verbosity == fatboyLog.VERBOSE):
                        print("rectifyProcess::traceMOSSkylineRectification> Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping this line!")
                        self._log.writeLog(__name__, "Coverage fraction of "+formatNum(covfrac)+"% is below minimnum threshold.  Skipping this line!", verbosity=fatboyLog.VERBOSE)
                    continue

                #Phase 2 of rejection criteria after lines have been traced
                #Find outliers > 2.5 sigma in peak values and remove them
                #First store first value as xout
                currXout = xcoords[0]
                peaks = array(peaks)
                peakmed = arraymedian(peaks)
                peaksd = peaks.std()
                b = (peaks > peakmed-2.5*peaksd)*(peaks < peakmed+2.5*peaksd)
                xcoords = array(xcoords)[b]
                ycoords = array(ycoords)[b]

                #Check coverage fraction
                if (len(xcoords) < 3):
                    if (self._fdb._verbosity == fatboyLog.VERBOSE):
                        print("rectifyProcess::traceMOSSkylineRectification> Coverage fraction is below minimnum threshold.  Skipping this line!")
                        self._log.writeLog(__name__, "Coverage fraction is below minimnum threshold.  Skipping this line!", verbosity=fatboyLog.VERBOSE)
                    continue
                if (self._fdb._verbosity == fatboyLog.VERBOSE):
                    print("\t\trejecting outliers (phase 2) - kept "+str(len(xcoords))+" datapoints.")
                    self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(xcoords))+" datapoints.", printCaller=False, tabLevel=2, verbosity=fatboyLog.VERBOSE)

                #Fit 2nd order order polynomial to datapoints, X = f(Y)
                order = 2
                p = zeros(order+1, float64)
                p[0] = xcoords[0]
                try:
                    lsq = leastsq(polyResiduals, p, args=(ycoords,xcoords,order))
                except Exception as ex:
                    continue

                #Compute output offsets and residuals from actual datapoints
                slitxprime = polyFunction(lsq[0], ycoords, order)
                xresid = slitxprime-xcoords
                #Remove outliers and refit
                b = abs(xresid) < xresid.mean()+2.5*xresid.std()
                xcoords = xcoords[b]
                ycoords = ycoords[b]
                if (self._fdb._verbosity == fatboyLog.VERBOSE):
                    print("\t\trejecting outliers (phase 3). Sigma = "+formatNum(xresid.std())+". Using "+str(len(xcoords))+" datapoints to fit line.")
                    self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+formatNum(xresid.std())+". Using "+str(len(xcoords))+" datapoints to fit line.", printCaller=False, tabLevel=2, verbosity=fatboyLog.VERBOSE)

                #Check coverage fraction
                covfrac = len(xcoords)*100.0/len(ys)
                if (covfrac >= minCovFrac):
                    slitxin.extend(xcoords)
                    slityin.extend(ycoords)
                    if (useAlternate):
                        #update currXout - needs to be based on a consistent yinit
                        #Fit 2nd order polynomial to remaining datapoints and evaluate at yref
                        yref = (slitylo+slityhi)//2+ylo
                        try:
                            lsq = leastsq(polyResiduals, p, args=(ycoords,xcoords,order))
                        except Exception as ex:
                            continue
                        currXout = polyFunction(lsq[0], yref, order)
                    slitxout.extend([currXout]*len(xcoords))
                    nlines += 1
                    qavalue = -50000
                    #Generate qa data
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        for i in range(len(xcoords)):
                            yval = int(ycoords[i]+.5)
                            xval = int(xcoords[i]+.5)
                            if (yval < 1 or yval > ysize-2 or xval < 1 or xval > xsize-2):
                                continue
                            for yi in range(yval-1,yval+2):
                                for xi in range(xval-1,xval+2):
                                    dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                    qaData[yi,xi] = qavalue/((1+dist)**2)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        for i in range(len(xcoords)):
                            yval = int(ycoords[i]+.5)
                            xval = int(xcoords[i]+.5)
                            if (yval < 1 or yval > ysize-2 or xval < 1 or xval > xsize-2):
                                continue
                            for yi in range(yval-1,yval+2):
                                for xi in range(xval-1,xval+2):
                                    dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                    qaData[xi,yi] = qavalue/((1+dist)**2)
                    inys = arange(min(ys), max(ys)+1)
                    outxs = polyFunction(lsq[0], inys+ylo, order)

                    #Mask out line based on FWHM and 2nd order fit to this individual line
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        xind = arange(lineMask.size).reshape(lineMask.shape) % lineMask.shape[1]
                        outxs = outxs.reshape((outxs.size,1))
                        b = (xind[inys,:] >= int32(outxs-gaussWidth))*(xind[inys,:] <= int32(outxs+gaussWidth+0.5))
                        tempMask = lineMask[inys, :]
                        tempMask[b] = 0
                        lineMask[inys, :] = tempMask
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        xind = arange(lineMask.size).reshape(lineMask.shape)//lineMask.shape[1]
                        b = (xind[:,inys] >= int32(outxs-gaussWidth))*(xind[:,inys] <= int32(outxs+gaussWidth+0.5))
                        tempMask = lineMask[:, inys]
                        tempMask[b] = 0
                        lineMask[:, inys] = tempMask

                if (self._fdb._verbosity == fatboyLog.VERBOSE):
                    print("\t\tX Center: "+formatNum(currXout)+"\t Yref: "+str(ys[0])+"\t Cov. Frac: "+formatNum(covfrac))
                    self._log.writeLog(__name__, "X Center: "+formatNum(currXout)+"\t Yref: "+str(ys[0])+"\t Cov. Frac: "+formatNum(covfrac), printCaller=False, tabLevel=2, verbosity=fatboyLog.VERBOSE)

            #write_fits_file('linemask.fits',lineMask,int32)
            #write_fits_file('masked.fits',slit*lineMask)
            if (nlines == 0):
                print("rectifyProcess::traceMOSSkylineRectification> Warning: Could not trace out any skylines in slit "+str(slitidx+1))
                self._log.writeLog(__name__, "Could not trace out any skylines in slit "+str(slitidx+1), type=fatboyLog.WARNING)
                continue
            print("\tSlit "+str(slitidx+1)+": Successfully traced out "+str(nlines)+ " skylines.  Fitting transformation...")
            self._log.writeLog(__name__, "\tSlit "+str(slitidx+1)+": Successfully traced out "+str(nlines)+ " skylines.  Fitting transformation...", printCaller=False, tabLevel=1)
            #Convert to arrays
            slitxin = array(slitxin)
            slityin = array(slityin)
            slitxout = array(slitxout)

            for seg in range(n_segments):
                xstride = xsize//n_segments
                sxlo = xstride*seg
                sxhi = xstride*(seg+1)
                segmask = (slitxin >= sxlo)*(slitxin < sxhi)
                seg_slitxin = slitxin[segmask]
                seg_slityin = slityin[segmask]
                seg_slitxout = slitxout[segmask]

                if (len(seg_slitxin) < 5):
                    continue

                #Fit transformation to line in this slitlet
                #Calculate number of terms based on order
                terms = 0
                for j in range(fit_order+2):
                    terms+=j
                p = zeros(terms)
                #Initial guess is f(x_in, y_in) = x_in
                p[1] = 1
                if (len(seg_slitxin) <= terms):
                    continue
                try:
                    lsq = leastsq(surfaceResiduals, p, args=(seg_slitxin, seg_slityin, seg_slitxout, fit_order))
                except Exception as ex:
                    continue

                #Compute output offsets and residuals from actual datapoints
                slitxprime = surfaceFunction(lsq[0], seg_slitxin, seg_slityin, fit_order)
                xresid = slitxprime-seg_slitxout
                residmean = xresid.mean()
                residstddev = xresid.std()
                seg_name = ""
                if (n_segments > 1):
                    seg_name = "Segment "+str(seg)+": "
                print("\t\t"+seg_name+"Found "+str(len(seg_slitxout))+" datapoints.  Fit: "+formatList(lsq[0]))
                print("\t\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
                self._log.writeLog(__name__, seg_name+"Found "+str(len(seg_slitxout))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
                self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=2)

                print("\t\tPerforming iterative sigma clipping to throw away outliers...")
                self._log.writeLog(__name__, "Performing iterative sigma clipping to throw away outliers...", printCaller=False, tabLevel=2)

                #Throw away outliers starting at 2 sigma significance
                sigThresh = 2
                niter = 0
                norig = len(seg_slitxout)
                bad = where(abs(xresid-residmean)/residstddev > sigThresh)
                while (len(bad[0]) > 0):
                    niter += 1
                    good = (abs(xresid-residmean)/residstddev <= sigThresh)
                    seg_slitxin = seg_slitxin[good]
                    seg_slityin = seg_slityin[good]
                    seg_slitxout = seg_slitxout[good]
                    #Refit, use last actual fit coordinates as input guess
                    p = lsq[0]
                    try:
                        lsq = leastsq(surfaceResiduals, p, args=(seg_slitxin, seg_slityin, seg_slitxout, fit_order))
                    except Exception as ex:
                        continue
                    #Compute output offsets and residuals from actual datapoints
                    slitxprime = surfaceFunction(lsq[0], seg_slitxin, seg_slityin, fit_order)
                    xresid = slitxprime-seg_slitxout
                    residmean = xresid.mean()
                    residstddev = xresid.std()
                    if (niter > 2):
                        #Gradually increase sigma threshold
                        sigThresh += 0.2
                    bad = where(abs(xresid-residmean)/residstddev > sigThresh)
                print("\t\tAfter "+str(niter)+" passes, kept "+str(len(seg_slitxout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
                print("\t\tData - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev))
                self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(seg_slitxout))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
                self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residmean)+"\tsigma: "+formatNum(residstddev), printCaller=False, tabLevel=2)

                xin.extend(seg_slitxin)
                yin.extend(seg_slityin)
                xout.extend(seg_slitxout)
                islit.extend([slitidx+1]*len(seg_slitxout))
                iseg.extend([seg]*len(seg_slitxout))
                xprime.extend(slitxprime)

        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        #Create output qa filename
        qafile = outdir+"/rectified/qa_"+skyFDU.getFullId()

        #Remove existing files if overwrite = yes
        if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            if (os.access(qafile, os.F_OK)):
                os.unlink(qafile)
        #Write out qa file
        if (not os.access(qafile, os.F_OK)):
            skyFDU.tagDataAs("slitqa", qaData)
            skyFDU.writeTo(qafile, tag="slitqa")
            skyFDU.removeProperty("slitqa")
        del qaData

        #Write out more qa data
        qafile = outdir+"/rectified/qa_"+fdu._id+"-skylines.dat"
        f = open(qafile,'w')
        for i in range(len(xout)):
            f.write(str(xin[i])+'\t'+str(yin[i])+'\t'+str(xout[i])+'\t'+str(islit[i])+'\t'+str(xprime[i])+'\t'+str(iseg[i])+'\n')
        f.close()

        return (xin, yin, xout, islit, iseg)
    #end traceMOSSkylineRectification

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/rectified", os.F_OK)):
            os.mkdir(outdir+"/rectified",0o755)
        #Create output filename
        rctfile = outdir+"/rectified/rct_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(rctfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(rctfile)
        if (not os.access(rctfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(rctfile)
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/rectified/clean_rct_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write out exposure map if it exists
        if (fdu.hasProperty("exposure_map")):
            expfile = outdir+"/rectified/exp_rct_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(expfile)
            if (not os.access(expfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(expfile, tag="exposure_map")
        #Write out exposure map if it exists
        if (fdu.hasProperty("pixel_map")):
            pixfile = outdir+"/rectified/pix_rct_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(pixfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(pixfile)
            if (not os.access(pixfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(pixfile, tag="pixel_map")
            fdu.removeProperty("pixel_map")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/rectified/NM_rct_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
