from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_drihizzle, drihizzle
from numpy import *
from scipy.optimize import leastsq
import inspect

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

class wavelengthCalibrateProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]
    _datadir = os.path.dirname(inspect.getfile(superFATBOY))+"/data/"

    ## Just apply a wavelength calibration from an existing lamp/clean sky
    def applyWavelengthCalibration(self, skyFDU, fdu, calibs):
        ###*** For purposes of wavelengthCalibrate algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        print("wavelengthCalibrateProcess::applyWavelengthCalibration> APPLYING previous wavelength calibration from "+skyFDU.getFullId()+" to "+fdu.getFullId()+"...")
        self._log.writeLog(__name__, "APPLYING previous wavelength calibration from "+skyFDU.getFullId()+" to "+fdu.getFullId()+"...")
        #Read options
        doIndividualSlitlets = False
        if (self.getOption("calibrate_slitlets_individually", fdu.getTag()).lower() == "yes"):
            doIndividualSlitlets = True
        resampleCommonScale = False
        if (self.getOption("resample_to_common_scale", fdu.getTag()).lower() == "yes"):
            resampleCommonScale = True
        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #Create output dir if it doesn't exist
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/wavelengthCalibrated", os.F_OK)):
            os.mkdir(outdir+"/wavelengthCalibrated",0o755)

        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and (fdu.hasProperty('slitmask') or 'slitmask' in calibs)):
            ###MOS/IFU data -- get slitmask
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (doIndividualSlitlets):
                #Find nslits and regions
                if (not fdu.hasProperty("nslits")):
                    fdu.setProperty("nslits", slitmask.getData().max())
                nslits = fdu.getProperty("nslits")
                if (fdu.hasProperty("regions")):
                    (ylos_data, yhis_data, slitx_data, slitw_data) = fdu.getProperty("regions")
                else:
                    #Use helper method to all ylo, yhi for each slit in each frame
                    (ylos_data, yhis_data, slitx_data, slitw_data) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                    fdu.setProperty("regions", (ylos_data, yhis_data, slitx_data, slitw_data))
            else:
                #doIndividualSlitlets = False - treat MOS data as 1 slit
                nslits = 1
                b = where(slitmask.getData() != 0)
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    ylos_data = [b[0].min()]
                    yhis_data = [b[0].max()]
                else:
                    ylos_data = [b[1].min()]
                    yhis_data = [b[1].max()]

        wcHeader = dict()
        wcHeader['CTYPE'] = 'WAVE-PLY'
        minLambda = None
        minLambdaList = []
        maxLambdaList = []
        fitParams = []
        for j in range(nslits):
            #Add new list of fitParams for this slitlet
            fitParams.append([])
            slitStr = str(j+1)
            if (j+1 < 10):
                slitStr = '0'+slitStr
            nseg = 1
            multseg = False
            if (skyFDU.hasHeaderValue('NSEG_'+slitStr)):
                nseg = skyFDU.getHeaderValue('NSEG_'+slitStr)
                wcHeader['NSEG_'+slitStr] = nseg
            if (nseg > 1):
                multseg = True
            if (multseg):
                for seg in range(nseg):
                    if (skyFDU.hasHeaderValue("PORDER"+slitStr+"_SEG"+str(seg))):
                        fit_order = skyFDU.getHeaderValue("PORDER"+slitStr+"_SEG"+str(seg))
                        coeffs = []
                        wcHeader["HIERARCH PORDER"+slitStr+"_SEG"+str(seg)] = fit_order
                        for i in range(fit_order+1):
                            term = skyFDU.getHeaderValue("PCF"+str(i)+"_S"+slitStr+"_SEG"+str(seg))
                            coeffs.append(term)
                            wcHeader["HIERARCH PCF"+str(i)+"_S"+slitStr+"_SEG"+str(seg)] = term
                        fitParams[j].append(coeffs)
                    else:
                        fitParams[j].append([])
            elif (nslits == 1 and skyFDU.hasHeaderValue("PORDER")):
                fit_order = skyFDU.getHeaderValue("PORDER")
                coeffs = []
                wcHeader["PORDER"] = fit_order
                for i in range(fit_order+1):
                    term = skyFDU.getHeaderValue("PCOEFF_"+str(i))
                    coeffs.append(term)
                    wcHeader["PCOEFF_"+str(i)] = term
                fitParams[j].append(coeffs)
            elif (skyFDU.hasHeaderValue("PORDER"+slitStr)):
                fit_order = skyFDU.getHeaderValue("PORDER"+slitStr)
                coeffs = []
                wcHeader["PORDER"+slitStr] = fit_order
                for i in range(fit_order+1):
                    term = skyFDU.getHeaderValue("PCF"+str(i)+"_S"+slitStr)
                    coeffs.append(term)
                    wcHeader["PCF"+str(i)+"_S"+slitStr] = term
                fitParams[j].append(coeffs)
            else:
                fitParams[j].append([])

        #Update header
        fdu.setProperty("wcHeader", wcHeader)
        #fdu.updateHeader(wcHeader)

        #Now loop over fitParams and find min/max lambda for resampling
        for j in range(nslits):
            nseg = len(fitParams[j])
            for seg in range(nseg):
                #Use xstride to split into equal length segments here
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    xstride = skyFDU.getShape()[1]//nseg
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    xstride = skyFDU.getShape()[0]//nseg
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                coeffs = fitParams[j][seg]
                if (len(coeffs) == 0):
                    #No solution to this segment
                    continue
                scale = coeffs[1]
                fit_order = len(coeffs)-1
                if (minLambda is None):
                    #First pass
                    if (scale > 0):
                        minLambda = polyFunction(coeffs, 0, fit_order)
                        maxLambda = polyFunction(coeffs, xstride-1, fit_order)
                    else:
                        maxLambda = polyFunction(coeffs, 0, fit_order)
                        minLambda = polyFunction(coeffs, xstride-1, fit_order)
                    minLambdaList.append(minLambda)
                    maxLambdaList.append(maxLambda)
                else:
                    if (scale > 0):
                        localMinLambda = polyFunction(coeffs, 0, fit_order)
                        localMaxLambda = polyFunction(coeffs, xstride-1, fit_order)
                        minLambda = min(minLambda, polyFunction(coeffs, sxlo, fit_order))
                        maxLambda = max(maxLambda, polyFunction(coeffs, sxhi-1, fit_order))
                    else:
                        localMaxLambda = polyFunction(coeffs, 0, fit_order)
                        localMinLambda = polyFunction(coeffs, xstride-1, fit_order)
                    minLambda = min(minLambda, localMinLambda)
                    maxLambda = max(maxLambda, localMaxLambda)
                    minLambdaList.append(localMinLambda)
                    maxLambdaList.append(localMaxLambda)

        if (minLambda is None):
            #If minLambda is None here than no slits were wavlength calibrated
            print("wavelengthCalibrateProcess::applyWavelengthCalibration> ERROR: Could not calibrate any slitlets for "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not calibrate any slitlets for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return

        xout_data = zeros(fdu.getShape(), dtype=float32)
        #Linear wavelength scale
        scale_data = (maxLambda-minLambda)/xsize
        xs_data = arange(xsize, dtype=float32)
        #New header keywords for resampled data
        resampHeader = dict()
        resampHeader['RESAMPLD'] = 'YES'
        resampHeader['CRVAL1'] = minLambda
        if (scale_data < 0):
            resampHeader['CRVAL1'] = maxLambda
        resampHeader['CDELT1'] = scale_data
        resampHeader['CRPIX1'] = 1
        if (skyFDU.hasHeaderValue('CRVAL1') and skyFDU.hasHeaderValue('CDELT1')):
            resampHeader['CRVAL1'] = skyFDU.getHeaderValue('CRVAL1')
            resampHeader['CDELT1'] = skyFDU.getHeaderValue('CDELT1')
            scale = skyFDU.getHeaderValue('CDELT1')
            scale_data = skyFDU.getHeaderValue('CDELT1')
        elif (skyFDU.hasProperty("resampledHeader")):
            resampHeader['CRVAL1'] = skyFDU.getProperty("resampledHeader")['CRVAL1']
            resampHeader['CDELT1'] = skyFDU.getProperty("resampledHeader")['CDELT1']
            scale = skyFDU.getProperty("resampledHeader")['CDELT1']
            scale_data = skyFDU.getProperty("resampledHeader")['CDELT1']

        for j in range(nslits):
            n_segments = len(fitParams[j])
            #mult_seg = True if multiple segments
            mult_seg = (n_segments > 1)

            if (not resampleCommonScale):
                #Use min/max wavelength from this input slitlet for all segments
                maxLambda = max(maxLambdaList[j*n_segments:(j+1)*n_segments])
                minLambda = min(minLambdaList[j*n_segments:(j+1)*n_segments])
                scale_data = (maxLambda-minLambda)/(float)(xsize)

            #Now loop over segments.  If n_segments == 1, this loop will only be done once (most data)
            for seg in range(n_segments):
                pass_name = " for order "+str(j+1)+" of "
                if (mult_seg):
                    pass_name = " for order "+str(j+1)+", segment "+str(seg+1)+" of "
                if (len(fitParams[j][seg]) == 0):
                    print("wavelengthCalibrateProcess::applyWavelengthCalibration> Warning: Could not find wavelength solution"+pass_name+fdu.getFullId()+". Skipping.")
                    self._log.writeLog(__name__, "Could not find wavelength solution"+pass_name+fdu.getFullId()+". Skipping.", type=fatboyLog.WARNING)
                    #WC failed!
                    continue

                xstride_data = xsize//n_segments
                sxlo_data = xstride_data*seg
                sxhi_data = xstride_data*(seg+1)
                xs_data = arange(xstride_data, dtype=float32)

                #Calculate wavelength input scale
                lambdaIn_data = polyFunction(fitParams[j][seg], xs_data, len(fitParams[j][seg])-1)

                if (not resampleCommonScale):
                    #Use min/max wavelength from this input slitlet and segment
                    #maxLambda = lambdaIn_data.max()
                    #minLambda = lambdaIn_data.min()
                    #scale_data = (maxLambda-minLambda)/(float)(xstride_data)
                    #Use CRVALSxx and CDELTSxx for header keywords
                    slitStr = str(j+1)
                    if (j+1 < 10):
                        slitStr = '0'+slitStr
                    if (not mult_seg):
                        resampHeader['CRVALS'+slitStr] = minLambda
                        if (scale_data < 0):
                            resampHeader['CRVALS'+slitStr] = maxLambda
                        resampHeader['CDELTS'+slitStr] = scale_data
                    else:
                        #Multiple segments, use hierarchical keywords CRVALS_xx_SEGy and CDELTS_xx_SEGy
                        slitStr += '_SEG' + str(seg)
                        resampHeader['HIERARCH CRVALS'+slitStr] = minLambda
                        if (scale_data < 0):
                            resampHeader['HIERARCH CRVALS'+slitStr] = maxLambda
                        resampHeader['HIERARCH CDELTS'+slitStr] = scale_data

                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (slitmask is not None):
                        #MOS data - add to xout array, mutiply input by currMask
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        if (doIndividualSlitlets):
                            currMask = slitmask.getData()[int(ylos_data[j]):int(yhis_data[j]+1),sxlo_data:sxhi_data] == (j+1)
                        else:
                            currMask = slitmask.getData()[int(ylos_data[j]):int(yhis_data[j]+1),sxlo_data:sxhi_data] != 0
                        xout_data[int(ylos_data[j]):int(yhis_data[j]+1),sxlo_data:sxhi_data] += ((lambdaIn_data-minLambda)/scale_data+xstride*seg)*currMask
                    else:
                        #Longslit data, calculate entire xout frame at once
                        xout_data[:,sxlo_data:sxhi_data] = (lambdaIn_data-minLambda)/scale_data+xstride*seg
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    if (slitmask is not None):
                        #Use this FDU's slitmask to calculate currMask for xout_data
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        if (doIndividualSlitlets):
                            currMask = slitmask.getData()[sxlo_data:sxhi_data,int(ylos_data[j]):int(yhis_data[j]+1)] == (j+1)
                        else:
                            currMask = slitmask.getData()[sxlo_data:sxhi_data,int(ylos_data[j]):int(yhis_data[j]+1)] != 0
                        xout_data[sxlo_data:sxhi_data,int(ylos_data[j]):int(yhis_data[j]+1)] += ((lambdaIn_data.reshape((len(lambdaIn_data), 1))-minLambda)/scale_data+xstride*seg)*currMask
                    else:
                        #Longslit data, calculate entire xout frame at once
                        xout_data[sxlo_data:sxhi_data,:] = (lambdaIn_data.reshape((len(lambdaIn_data), 1))-minLambda)/scale+xstride*seg

        #Add header as property of FDU since it won't be written out until writeOutput
        fdu.setProperty("resampledHeader", resampHeader)
        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle
        #Resampled frames should be in cps not counts!
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            #Use drihizzle to resample with xtrans=xout_data
            inMask = (xout_data != 0).astype(int32)
            (resampData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits="counts", outunits="cps", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Save as data tag "resampled"
            expmap[expmap == 0] = 1
            fdu.tagDataAs("resampled", resampData)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #ytrans = xout_data
            inMask = (xout_data != 0).astype(int32)
            (resampData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits="counts", outunits="cps", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            expmap[expmap == 0] = 1
            fdu.tagDataAs("resampled", resampData)
    #end applyWavelengthCalibration

    def checkFinalRejectionCriteria(self, fdu, mcor, refbox, diff, wlines, currWave, reflines, currLine, scale, delta):
        if (abs(mcor-refbox) > max(0.05*diff, 2)):
            #Rejection 9: Max difference between initial guess and fitted
            #value is greater than 5% of the distance between this line and
            #the reference line in the reference slitlet (or 2 pixels, whichever is greater)

            #Also check with delta added in - delta is the difference in mcor position if
            #linear scale from nearest 2 lines instead of overall scale is used to calculate
            #guess at position
            if (abs(mcor+delta-refbox) > max(0.05*diff, 2)):
                if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                    print("FAIL 9 - diff between initial guess and fit value "+str(abs(mcor-refbox))+" is > 5% of distance between this line and ref line ("+str(diff)+")")
                return False

        if (wlines.count(currWave) != 0):
            #Rejection 10: Do not use a line from the line list that has already been used!!
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 10 - line has already been used")
            return False

        #11/6/19 new criteria 11 - check std of residuals with and without new line
        #If they are >= 5x bigger, reject
        rl = array(reflines)
        wl = array(wlines)
        nlines = len(rl)
        if (nlines < 8):
            fit_order = 1
        else:
            fit_order = 2
        #Fit polynomial of order fit_order to lines
        p = zeros(fit_order+1, dtype=float32)
        p[1] = scale
        try:
            lsq = leastsq(polyResiduals, p, args=(rl, wl, fit_order))
        except Exception as ex:
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("Least squares failed with exception "+str(ex))
            return False
        #Calculate residuals
        residLines = polyFunction(lsq[0], rl, fit_order)-wl
        sd_old = residLines.std()

        #Append new lines
        rl = append(reflines, currLine)
        wl = append(wlines, currWave)
        try:
            lsq = leastsq(polyResiduals, p, args=(rl, wl, fit_order))
        except Exception as ex:
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("Least squares failed with exception "+str(ex))
            return False
        #Calculate residuals
        residLines = polyFunction(lsq[0], rl, fit_order)-wl
        sd_new = residLines.std()

        #print "SD NEW", sd_new, "SD OLD", sd_old, "FAC", sd_new/sd_old, "THIS LINE", abs(residLines[-1])
        if (sd_new / sd_old >= 5 and nlines > 4):
            #Rejection 11: Line increases residuals in fit by 5x or more!
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 11 - std dev increasted 5x", currLine, currWave, sd_new, sd_old)
            return False
        elif (sd_new / sd_old >= 4 and nlines > 4 and diff < 100):
            #For difference of < 100 pixels, limit to 4x increase
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 11A - std dev increased 4x", currLine, currWave, sd_new, sd_old)
            return False
        elif (sd_new / sd_old >= 2 and nlines > 8 and diff < 30):
            #Within 30 pixels of another line and > 8 lines - second order,
            #should never more than double residuals std.
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 11B - std dev increased 2x", currLine, currWave, sd_new, sd_old)
            return False
        elif (nlines > 8 and abs(residLines[-1]) > 15*sd_old):
            #This line is > 15 sigma of old fit
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 11C - std dev of this line is 15 * old std dev", currLine, currWave, sd_new, sd_old, abs(residLines[-1]))
            return False

        #passed rejection criteria 9 and 10
        return True
    #end checkFinalRejectionCriteria

    def checkPostfitRejectionCriteria(self, fdu, lsq, obsFlag, blref, ccor):
        if (lsq[1] == 5):
            #Rejection 7: Exceeded max # of calls
            #Blank out line
            obsFlag[blref-3:blref+4] = 0
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 7 - least squares failed to converge")
            return False

        if (abs(where(ccor == max(ccor))[0][0] - lsq[0][1]) > 2):
            #Rejection 8: should be within +/- 2 pixels from guess
            #Blank out line
            obsFlag[blref-3:blref+4] = 0
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 8 - fit to cross correlation not within 2 pixels of guess")
            return False
        #Passed criteria 7 and 8
        return True
    #end checkPostfitRejectionCriteria

    def checkPrefitRejectionCriteria(self, fdu, blref, resid, reflines, wlines, obsFlag, inloop, refbox, searchbox, nlines, dummyWave, dummyFlux, min_threshold):
        success = True
        templines = array(reflines)
        #Find closest line in pixel space to this one to use for initial guesses
        refline = where(abs(templines-blref) == min(abs(templines-blref)))[0][0]
        #Find index of dummyWave closest to actual wavelength of refline
        refIdx = where(abs(dummyWave-wlines[refline]) == min(abs(dummyWave-wlines[refline])))[0][0]
        #Guess at index offset between oned cut and dummyFlux cut
        xoffGuess = int(refIdx-reflines[refline])
        refCut = resid[blref-refbox:blref+refbox+1]

        #Rejection 0: Out of range - at edge of image
        if (blref+xoffGuess <= 5 or blref+xoffGuess > len(dummyFlux)-5):
            obsFlag[blref-3:blref+4] = 0
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 0 - out of range - edge of image")
                print("\tindex guess = ",blref+xoffGuess, "dummy spectrum length = ", len(dummyFlux))
            return (False, inloop, 0)

        #Rejection 1: If current line is within 3 pixels of a previous line
        if (abs(reflines[refline]-blref) < 4):
            #Too close!  Blank out 3 pixels and try again
            obsFlag[blref-1:blref+2] = 0
            inloop+=1
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 1 - within 3 pixels of a previous line")
            return (False, inloop, 0)

        #Rejection 2: If there is more than one zeroed out value
        #within +/- 3 pixels of line's center in oned cut
        if (len(where(refCut[refbox-3:refbox+4] == 0)[0]) > 1):
            obsFlag[blref-1:blref+2] = 0
            if (nlines > 3):
                inloop+=1
            else:
                inloop+=0.1
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 2 - more than 1 zero value within +/- 3 pixels")
            return (False, inloop, 0)

        #Rejection 3: If there is not a nonzero value within +/- 3
        #pixels of guess at line's center in dummy cut
        if (len(where(dummyFlux[blref+xoffGuess-3:blref+xoffGuess+4] != 0)[0]) == 0):
            obsFlag[blref-3:blref+4] = 0
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 3 - no nonzero values within 3 pixels of guess in dummy spectrum")
            return (False, inloop, 0)

        #Find nonzero points within +/- 2*search box but excluding +/- 5 points from line center
        leftpts = where(resid[blref-2*searchbox:blref-5] != 0)[0]+blref-2*searchbox
        rightpts = where(resid[blref+6:blref+2*searchbox+1] != 0)[0]+blref+6
        sigpts = concatenate((leftpts, rightpts))
        #Rejection 4: If there are 3 or less such nonzero points on either the left or right side
        if (len(where(sigpts < blref)[0]) <= 3 or len(where(sigpts > blref)[0]) <= 3):
            obsFlag[blref-1:blref+2] = 0
            if (nlines > 3):
                inloop+=1
            else:
                inloop+=0.1
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 4 - 3 or less nonzero points in stats area")
            return (False, inloop, 0)

        #Calculate local sigma using median and std dev of sigpts.
        #Since all sigpts are nonzero, no need to pass nonzero=True.
        lsigma = (resid[blref]-gpu_arraymedian(resid[sigpts]))/resid[sigpts].std()
        #Rejection 5: If local sigma < min_threshold
        if (lsigma < min_threshold):
            obsFlag[blref-1:blref+2] = 0
            if (nlines > 3):
                inloop+=1
            else:
                inloop+=0.1
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 5 - local sigma", lsigma, "< threshold", min_threshold)
            return (False, inloop, lsigma)

        #Rejection 6: out of range
        if (blref-searchbox+xoffGuess < 0 or blref+(0.75*searchbox)+xoffGuess > len(dummyFlux)):
            obsFlag[blref-1:blref+2] = 0
            if (nlines > 3):
                inloop+=1
            else:
                inloop+=0.1
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("FAIL 6 - out of range")
            return (False, inloop, lsigma)
        return (success, inloop, lsigma)
    #end checkPrefitRejectionCriteria

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Wavelength Calibrate")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        wcfile = "wavelengthCalibrated/wc_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, wcfile, headerTag="wcHeader")):
            #Also check if "cleanFrame" exists
            cleanfile = "wavelengthCalibrated/clean_wc_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "noisemap" exists
            nmfile = "wavelengthCalibrated/NM_wc_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Also check if "resampled" exists
            resampfile = "wavelengthCalibrated/resamp_wc_"+fdu.getFullId()
            self.checkOutputExists(fdu, resampfile, tag="resampled", headerTag="resampledHeader")
            #Also check if "resampled_slitmask" exists
            resampfile =  "wavelengthCalibrated/resamp_slitmask_"+fdu.getFullId()
            self.checkOutputExists(fdu, resampfile, tag="resampled_slitmask", headerTag="resampledHeader")

            #Need to get calibration frames - cleanSky, masterLamp, and slitmask to update from disk too
            calibs = dict()
            headerVals = dict()
            headerVals['grism_keyword'] = fdu.grism
            properties = dict()
            properties['specmode'] = fdu.getProperty("specmode")
            properties['dispersion'] = fdu.getProperty("dispersion")
            skyShape = None
            if (not 'cleanSky' in calibs):
                #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
                #cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (cleanSky is not None):
                    #add to calibs for rectification below
                    calibs['cleanSky'] = cleanSky
                    skyShape = cleanSky.getShape()
                    if (cleanSky.hasProperty("presampled_shape")):
                        skyShape = cleanSky.getProperty("presampled_shape")

            if (not 'masterLamp' in calibs):
                #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
                masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
                if (masterLamp is None):
                    #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                    masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (masterLamp is not None):
                    #add to calibs for rectification below
                    calibs['masterLamp'] = masterLamp
                    skyShape = masterLamp.getShape()
                    if (masterLamp.hasProperty("presampled_shape")):
                        skyShape = masterLamp.getProperty("presampled_shape")

            if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
                #Use new fdu.getSlitmask method
                slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
                if (slitmask is not None):
                    #Found slitmask
                    calibs['slitmask'] = slitmask

            #Check for cleanSky and masterLamp frames to update from disk too
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("wavelengthCalibrated")):
                #Check if output exists
                wcfile = "wavelengthCalibrated/wc_"+calibs['cleanSky']._id+".fits"
                if (self.checkOutputExists(calibs['cleanSky'], wcfile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "wavelengthCalibrated" = True
                    calibs['cleanSky'].setProperty("wavelengthCalibrated", True)
                    #Also check if "resampled" exists
                    resampfile = "wavelengthCalibrated/resamp_wc_"+calibs['cleanSky']._id+".fits"
                    self.checkOutputExists(calibs['cleanSky'], resampfile, tag="resampled", headerTag="resampledHeader")

            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("wavelengthCalibrated")):
                #Check if output exists first
                wcfile = "wavelengthCalibrated/wc_"+calibs['masterLamp'].getFullId()
                if (self.checkOutputExists(calibs['masterLamp'], wcfile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "wavelengthCalibrated" = True
                    calibs['masterLamp'].setProperty("wavelengthCalibrated", True)
                    #Also check if "resampled" exists
                    resampfile = "wavelengthCalibrated/resamp_wc_"+calibs['masterLamp']._id+".fits"
                    self.checkOutputExists(calibs['masterLamp'], resampfile, tag="resampled", headerTag="resampledHeader")

            if ('slitmask' in calibs):
                #Check if output exists first
                resampfile = "wavelengthCalibrated/resamp_wc_"+calibs['slitmask'].getFullId()
                self.checkOutputExists(calibs['slitmask'], resampfile, tag="resampled", headerTag="resampledHeader")
            return True

        #Call get calibs to return dict() of calibration frames.
        #For wavelengthCalibrated, this dict should have cleanSky and/or masterLamp
        #and optionally slitmask if this is not a property of the FDU at this point.
        #These are obtained by tracing slitlets using the master flat
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'masterLamp' in calibs and not 'cleanSky' in calibs):
            #Failed to obtain master lamp or cleanSky
            #Issue error message and disable this FDU
            print("wavelengthCalibrateProcess::execute> ERROR: Wavelength solution not calibrated for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Wavelength solution not calibrated for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        ##Check here if wavelength solution already exists in calibration frame and just needs to be applied
        useArclamps = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True
        if ('masterLamp' in calibs and useArclamps):
            if (calibs['masterLamp'].hasHeaderValue('CTYPE') and calibs['masterLamp'].getHeaderValue('CTYPE') == 'WAVE-PLY'):
                self.applyWavelengthCalibration(calibs['masterLamp'], fdu, calibs)
                return True
        elif ('cleanSky' in calibs and not useArclamps):
            if (calibs['cleanSky'].hasHeaderValue('CTYPE') and calibs['cleanSky'].getHeaderValue('CTYPE') == 'WAVE-PLY'):
                self.applyWavelengthCalibration(calibs['cleanSky'], fdu, calibs)
                return True

        #call wavelengthCalibrate helper function to do gpu/cpu calibration
        self.wavelengthCalibrate(fdu, calibs)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("wavelengthCalibrateProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("wavelengthCalibrateProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        csfilename = self.getCalib("master_clean_sky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("wavelengthCalibrateProcess::getCalibs> Using master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using master clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, log=self._log)
            else:
                print("wavelengthCalibrateProcess::getCalibs> Warning: Could not find master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find master clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("wavelengthCalibrateProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("wavelengthCalibrateProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Could not find master arclamp frame "+mlfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and dispersion
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        skyShape = None
        if (not 'cleanSky' in calibs):
            #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
            #cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (cleanSky is not None):
                #add to calibs for rectification below
                calibs['cleanSky'] = cleanSky
                skyShape = cleanSky.getShape()
                if (cleanSky.hasProperty("presampled_shape")):
                    skyShape = cleanSky.getProperty("presampled_shape")

        if (not 'masterLamp' in calibs):
            #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
            if (masterLamp is None):
                #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is not None):
                #add to calibs for rectification below
                calibs['masterLamp'] = masterLamp
                skyShape = masterLamp.getShape()
                if (masterLamp.hasProperty("presampled_shape")):
                    skyShape = masterLamp.getProperty("presampled_shape")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        return calibs
    #end getCalibs

    #parse wcinfo data structure correctly for this slitidx and optionally segment
    def getWCParam(self, wcinfo, keyword, slitidx, mult_seg=False, segidx=0):
        if (not mult_seg):
            return wcinfo[keyword][slitidx]
        elif (not isinstance(wcinfo[keyword][slitidx], list)):
            #multiple segments but no overriding lines in XML file for individual segments
            return wcinfo[keyword][slitidx]
        else:
            if (keyword == "wavelength_scale_guess" and not isinstance(wcinfo[keyword][slitidx][0], list)):
                return wcinfo[keyword][slitidx]
            return wcinfo[keyword][slitidx][segidx]
    #end getWCParam

    #Match 3 brightest lines in data (out of n) to lines from linelist (out of m)
    #Default n = 5, m = 14
    def match3BrightestLines(self, coeffs, dlines, dpeak, dwave, dummyFlux, dummyOrder, dummyWave, fdu, oned, nonlinear, scale, usePlot, wccentroids, wclines):
        currLines = []
        #read option
        max_separation = self.getOption("max_bright_line_separation", fdu.getTag())
        if (max_separation is not None):
            max_separation = int(max_separation)
        #Get all combinations of 3 brightest out of n lines - [0,1,2], [0,1,3],...
        combs = []
        for ix in range(len(wclines)-2):
            for iy in range(ix+1, len(wclines)-1):
                for iz in range(iy+1, len(wclines)):
                    combs.append([ix, iy, iz])
        dumPeak = max(dpeak) #Set a default
        for c in combs:
            #Match up 3 lines in image with corresponding lines in template
            curr_wclines = array(wclines)[c]
            curr_wccentroids = array(wccentroids)[c]
            #Sort 3 lines by wavelength so no confusion as to < > comparisons with pos/neg values
            idx = array(c)[curr_wclines.argsort()]
            curr_wclines.sort()
            curr_wccentroids.sort()
            #Check max_separation
            if (max_separation is not None):
                #lines are sorted so if 2-0 < max, all 3 deltas are < max
                if (curr_wccentroids[2]-curr_wccentroids[0] > max_separation):
                    print("wavelengthCalibrateProcess::match3BrightestLines> Skipping combination "+str(curr_wclines)+" - over max separation "+str(max_separation))
                    self._log.writeLog(__name__, "Skipping combination "+str(curr_wclines)+" - over max separation "+str(max_separation))
                    continue
            print("wavelengthCalibrateProcess::match3BrightestLines> Trying lines "+str(curr_wclines))
            self._log.writeLog(__name__, "Trying lines "+str(curr_wclines))
            #delta wavelengths reference frame
            deltaWref = []
            for i in range(1,len(curr_wclines)):
                for k in range(i-1,-1,-1):
                    #Guess at wavelength separation betweeen each pair of lines, 1-0, 2-1, 2-0
                    if (not nonlinear):
                        deltaWref.append(scale*(curr_wclines[i]-curr_wclines[k]))
                    else:
                        #Use coeffs not scale
                        deltaWref.append(polyFunction(coeffs, curr_wclines[i], dummyOrder)-polyFunction(coeffs, curr_wclines[k], dummyOrder))

            if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("Data line pixels, deltaLambdas: ", curr_wclines, deltaWref)
                print("Dummy line pixels: ", dlines)
                print("Dummy line wavlengths: ", dummyWave[dlines].tolist())
                plt.plot(oned, '#1f77b4', linewidth=2.0)
                plt.plot(dummyFlux, '#ff7f0e', linewidth=2.0)
                plt.show()
                plt.close()
            maxCor = 0
            currLines = []
            sign = (scale > 0)*2-1 #sign = 1 | -1

            #Triple loop to look at all potential 3-line combinations
            for l in range(len(dlines)):
                for k in range(len(dlines)):
                    #Look at each possible pairing of lines in dummy spectrum, 1-0, 2-1, 2-0, 3-2, etc.
                    if (l == k):
                        #Same line
                        continue
                    #Rejection 1: not within +/- 25% of initial guess at scale
                    deltaWdum = [dwave[k]-dwave[l]]
                    #deltaWdum = [dummyWave[dlines[k]]-dummyWave[dlines[l]]]
                    #Calculate linear wavelength scale based on wavelength separation between these
                    #two lines in dummy spectrum (known wavelengths) and pixel separation of
                    #brightest 2 lines in image
                    #1/29/16 Use centroids and actual wavelengths not rounded values
                    tempScale = [deltaWdum[0]/(curr_wccentroids[1]-curr_wccentroids[0])]
                    if (sign*deltaWdum[0] < sign*0.75*deltaWref[0] or sign*deltaWdum[0] > sign*1.25*deltaWref[0]):
                        continue
                    #Passed 1 -- look for potential 3rd lines
                    for i in range(len(dlines)):
                        #Reset lists at each iteration through potential 3rd lines
                        deltaWdum = [deltaWdum[0]]
                        tempScale = [tempScale[0]]
                        if (i == l or i == k):
                            #This is one of the first two lines
                            continue
                        #Rejection 2: Check delta12 and delta02 ... reject if not
                        #within +/- 25% of initial guess at scale
                        #deltaWdum.append(dummyWave[dlines[i]]-dummyWave[dlines[k]])
                        deltaWdum.append(dwave[i]-dwave[k])
                        tempScale.append(deltaWdum[1]/(curr_wccentroids[2]-curr_wccentroids[1]))
                        if (sign*deltaWdum[1] < sign*0.75*deltaWref[1] or sign*deltaWdum[1] > sign*1.25*deltaWref[1]):
                            continue
                        #deltaWdum.append(dummyWave[dlines[i]]-dummyWave[dlines[l]])
                        deltaWdum.append(dwave[i]-dwave[l])
                        tempScale.append(deltaWdum[2]/(curr_wccentroids[2]-curr_wccentroids[0]))
                        if (sign*deltaWdum[2] < sign*0.75*deltaWref[2] or sign*deltaWdum[2] > sign*1.25*deltaWref[2]):
                            continue
                        #Rejection 3: Compare calculated linear wavelength scales based on each
                        #pairing and if any is not within +/- 10% of others, reject this trio
                        if (sign*tempScale[1] < sign*0.9*tempScale[0] or sign*tempScale[1] > sign*1.1*tempScale[0]):
                            continue
                        if (sign*tempScale[2] < sign*0.9*tempScale[0] or sign*tempScale[2] > sign*1.1*tempScale[0]):
                            continue
                        if (sign*tempScale[2] < sign*0.9*tempScale[1] or sign*tempScale[2] > sign*1.1*tempScale[1]):
                            continue
                        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                            print("Passed Rejection 3.  Indices ", l, k, i, " Dummy pixels ", dlines[l], dlines[k], dlines[i])
                            print("\tWavelengths ", dwave[l], dwave[k], dwave[i])
                        #These 3 lines have passed the first 3 rejection criteria
                        tempLines = [dlines[l], dlines[k], dlines[i]]
                        tempPeak = dpeak[l]+dpeak[k]+dpeak[i]
                        #cross correlate +/- 100 px box around each line in refCut and dumCut to find shifts
                        #Rejection 4: max ccor should be right near 100, in middle of search box.
                        #Since lines are sorted, lambda0 < lambda1 < lambda2 for wclines
                        #and lambd l < lambda k < lambda i
                        isValid = True
                        for r in range(3):
                            dumCut = dummyFlux[tempLines[r]-100:tempLines[r]+101].copy()
                            refCut = oned[curr_wclines[r]-100:curr_wclines[r]+101].copy()
                            #Handle edge cases where a brighter line might be in the refCut
                            #That was ignored as a bright line because its near the edge
                            if (curr_wclines[r] < 200):
                                refCut[:200-curr_wclines[r]] = 0
                                dumCut[:200-curr_wclines[r]] = 0
                            elif (len(oned)-curr_wclines[r] < 200):
                                refCut[len(oned)-curr_wclines[r]:] = 0
                                dumCut[len(oned)-curr_wclines[r]:] = 0
                            ccor = correlate(refCut, dumCut, mode='same')
                            mcor = where(ccor == max(ccor))[0][0]
                            if (mcor < 90 or mcor > 110):
                                isValid = False
                        if (not isValid):
                            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                print("Failed Rejection 4.  Indices ", l, k, i, " Dummy pixels ", dlines[l], dlines[k], dlines[i])
                                print("\tWavelengths ", dwave[l], dwave[k], dwave[i])
                            #At least one line failed Rejection test #4.
                            continue
                        #cross correlate sections of refCut and dumCut spanning entire range of 3 lines
                        #and 50 pixels beyond each (after subtracting median values of cuts)
                        dumCut = dummyFlux[min(tempLines)-50:max(tempLines)+51].copy()
                        dumCut -= gpu_arraymedian(dumCut)
                        refCut = oned[min(curr_wclines)-50:max(curr_wclines)+51].copy()
                        refCut -= gpu_arraymedian(refCut)
                        ccor = correlate(refCut, dumCut, mode='same')
                        mcor = where(ccor == max(ccor))[0][0]
                        #Rejection 5: check max of ccor function.  Must be higher than
                        #previous max and also be within central 10% of search box.
                        if (max(ccor) > 1.001*maxCor and mcor > len(ccor)//2*0.9 and mcor < len(ccor)//2*1.1):
                            currLines = tempLines
                            dumPeak = tempPeak
                            maxCor = max(ccor)
                            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                print("New Match")
                                print(l,k,i,dlines[l],dlines[k],dlines[i], max(ccor), where(ccor == max(ccor)), len(refCut), len(dumCut), len(ccor))
                                print("\tWavelengths ", dwave[l], dwave[k], dwave[i])
                        else:
                            if (max(ccor) > 1.001*maxCor):
                                #Try second highest peak
                                ccor[mcor-5:mcor+6] = 0
                                mcor = where(ccor == max(ccor))[0][0]
                                if (max(ccor) > 1.001*maxCor and mcor > len(ccor)//2*0.9 and mcor < len(ccor)//2*1.1):
                                    currLines = tempLines
                                    dumPeak = tempPeak
                                    maxCor = max(ccor)
                                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                        print("New Match 2nd pass")
                                        print(l,k,i,dlines[l],dlines[k],dlines[i], max(ccor), where(ccor == max(ccor)), len(refCut), len(dumCut), len(ccor))
                                        print("\tWavelengths ", dwave[l], dwave[k], dwave[i])
                                else:
                                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                        print("Failed Rejection 5.  Indices ", l, k, i, " Dummy pixels ", dlines[l], dlines[k], dlines[i])
                                        print("\tWavelengths ", dwave[l], dwave[k], dwave[i])
                            else:
                                if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                    print("Failed Rejection 5.  Indices ", l, k, i, " Dummy pixels ", dlines[l], dlines[k], dlines[i])
                                    print("\tWavelengths ", dwave[l], dwave[k], dwave[i])
            if (len(currLines) == 3):
                #Matched 3 brightest lines
                print("wavelengthCalibrateProcess::match3BrightestLines> Successfully matched with "+str(currLines))
                self._log.writeLog(__name__, "Successfully matched with "+str(currLines))
                return (True, currLines, dumPeak, idx)
        return (False, currLines, dumPeak, array([0,1,2]))
    #end match3BrightestLines

    #Add gaussians for each line in line list
    def populateDummyFlux(self, masterFlux, masterWave, dummyWave, dummyFlux, scale, gaussWidth, wlines=[]):
        #Set up boolean arrays to find lines that are > 5 pixels out of wavelength regime
        out_of_regime_low = array(masterWave) < min(dummyWave)-5*scale
        out_of_regime_high = array(masterWave) > max(dummyWave)+5*scale
        idx_range = max(50, int(gaussWidth*10)) #index range for creating Gaussians
        for i in range(len(masterFlux)):
            if (masterFlux[i] < 0):
                continue
            #Skip lines out of wavelength regime
            if (out_of_regime_low[i] or out_of_regime_high[i]):
                continue
            #Skip lines within 2*scale of lines that have been fit
            #As pixels will be blanked out in obsFlag
            skip = False
            for k in range(len(wlines)):
                if (abs(wlines[k]-masterWave[i]) < 2*abs(scale)):
                    skip = True
            if (skip):
                continue
            p = zeros(4)
            p[0] = masterFlux[i]
            p[1] = masterWave[i]
            p[2] = gaussWidth*abs(scale)
            #Find index nearest this wavelength as fast as possible
            if (scale < 0):
                if (masterWave[i] < dummyWave.min()):
                    idx = 0
                else:
                    idx = where(dummyWave < masterWave[i])[0][0]
            else:
                if (masterWave[i] > dummyWave.max()):
                    idx = dummyWave.size-1
                else:
                    idx = where(dummyWave > masterWave[i])[0][0]
            #Indices to take a subarray of dummyWave to use to create Gaussian
            #Saves 10x time and Gaussian contributions out of this range are
            #effectively 0
            idx1 = max(0, idx-idx_range)
            idx2 = min(dummyWave.size, idx+idx_range)
            currLine = gaussFunction(p, dummyWave[idx1:idx2])
            dummyFlux[idx1:idx2] += currLine
        return dummyFlux
    #endPopulateDummyFlux

    #read a line list and return a tuple of arrays
    def readLineList(self, line_list):
        if (not os.access(line_list, os.F_OK) and os.access(self._datadir+"/linelists/"+line_list, os.F_OK)):
            line_list = self._datadir+"/linelists/"+line_list
        #Read in line list
        lines = readFileIntoList(line_list)

        #Parse line list into masterWave, masterFlux, and masterFlag arrays
        masterWave = []
        masterFlux = []
        #0 = use for fit, nonzero = don't use for fit
        masterFlag = []
        for line in lines:
            if (line[0] == '#'):
                #Comment line
                continue
            temp = line.split()
            masterWave.append(float(temp[0]))
            masterFlux.append(float(temp[1]))
            if (len(temp) > 2 and temp[2][0] != '#'):
                masterFlag.append(int(temp[2]))
            else:
                masterFlag.append(0)
        #Convert lists to arrays
        masterWave = array(masterWave)
        masterFlux = array(masterFlux)
        masterFlag = array(masterFlag)
        return (masterWave, masterFlux, masterFlag)
    #endreadLineList

    #read an XML style region file
    def readWavelengthCalibrationFile(self, wcfile, nslits, fdu):
        #Set defaults
        def_fit_order = int(self.getOption("fit_order", fdu.getTag()))
        def_line_list = self.getOption("line_list", fdu.getTag())
        def_max_wavelength = float(self.getOption("max_wavelength", fdu.getTag()))
        def_min_wavelength = float(self.getOption("min_wavelength", fdu.getTag()))
        def_min_threshold = float(self.getOption("min_threshold", fdu.getTag()))
        def_wavelength_scale_guess = self.getOption("wavelength_scale_guess", fdu.getTag())
        if (def_wavelength_scale_guess is None):
            try:
                lambda1 = float(self.getOption("wavelength_line_1", fdu.getTag()))
                lambda2 = float(self.getOption("wavelength_line_2", fdu.getTag()))
                delta = float(self.getOption("wavelength_line_separation", fdu.getTag()))
                def_wavelength_scale_guess = abs(lambda1-lambda2)/delta
            except Exception as ex:
                def_wavelength_scale_guess = None
        else:
            def_wavelength_scale_guess = def_wavelength_scale_guess.split()
            for j in range(len(def_wavelength_scale_guess)):
                def_wavelength_scale_guess[j] = float(def_wavelength_scale_guess[j])
        def_n_segments = int(self.getOption("n_segments", fdu.getTag()))
        def_min_lines_nonlinear = int(self.getOption("min_lines_to_refine_nonlinear_guess", fdu.getTag()))

        #Path
        currPath = ""
        if (wcfile.find('/') != -1):
            currPath = wcfile[:wcfile.rfind('/')+1]

        #Check if in config dir
        if (not os.access(wcfile, os.F_OK) and os.access(self._datadir+"/config/"+wcfile, os.F_OK)):
            wcfile = self._datadir+"/config/"+wcfile

        wcinfo = dict()
        #doc = xml config file
        try:
            doc = xml.dom.minidom.parse(wcfile)
        except Exception as ex:
            print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Error parsing XML config file "+wcfile+": "+str(ex))
            self._log.writeLog(__name__, "Error parsing XML config file "+wcfile+": "+str(ex), type=fatboyLog.ERROR)
            return None

        #get top level dataset node (should only be 1)
        datasetNodes = doc.getElementsByTagName('dataset')
        for node in datasetNodes:
            if (node.hasAttribute("fit_order")):
                try:
                    def_fit_order = int(node.getAttribute("fit_order"))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing fit_order.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing fit_order.", type=fatboyLog.WARNING)
                    def_fit_order = 3
            if (node.hasAttribute("line_list")):
                def_line_list = currPath+node.getAttribute("line_list")
                if (not os.access(def_line_list, os.F_OK) and not os.access(self._datadir+"/linelists/"+def_line_list, os.F_OK)):
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: could not find line list "+def_line_list)
                    self._log.writeLog(__name__, "could not find line list "+def_line_list, type=fatboyLog.WARNING)
                    def_list_list = None
            if (node.hasAttribute("max_wavelength")):
                try:
                    def_max_wavelength = float(node.getAttribute("max_wavelength"))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing max_wavelength.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing max_wavelength.", type=fatboyLog.WARNING)
                    def_max_wavelength = 18500
            if (node.hasAttribute("min_wavelength")):
                try:
                    def_min_wavelength = float(node.getAttribute("min_wavelength"))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing min_wavelength.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing min_wavelength.", type=fatboyLog.WARNING)
                    def_min_wavelength = 10000
            if (node.hasAttribute("min_threshold")):
                try:
                    def_min_threshold = float(node.getAttribute("min_threshold"))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing min_threshold.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing min_threshold.", type=fatboyLog.WARNING)
                    def_min_threshold = 3
            if (node.hasAttribute("wavelength_scale_guess")):
                try:
                    def_wavelength_scale_guess = node.getAttribute("wavelength_scale_guess").split()
                    for j in range(len(def_wavelength_scale_guess)):
                        def_wavelength_scale_guess[j] = float(def_wavelength_scale_guess[j])
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing wavelength_scale_guess.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing wavelength_scale_guess.", type=fatboyLog.WARNING)
                    def_wavelength_scale_guess = None
            elif (node.hasAttribute("wavelength_line_1") and node.hasAttribute("wavelength_line_2") and node.hasAttribute("wavelength_line_separation")):
                try:
                    lambda1 = float(node.getAttribute("wavelength_line_1"))
                    lambda2 = float(node.getAttribute("wavelength_line_2"))
                    delta = float(node.getAttribute("wavelength_line_separation"))
                    def_wavelength_scale_guess = abs(lambda1-lambda2)/delta
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error calculating wavelength_scale_guess.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error calculating wavelength_scale_guess.", type=fatboyLog.WARNING)
                    def_wavelength_scale_guess = None
            else:
                print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: Could not find wavelength_scale_guess or wavelength_line_1, wavelength_line_2, and wavelength_line_separation.")
                self._log.writeLog(__name__, "Could not find wavelength_scale_guess or wavelength_line_1, wavelength_line_2, and wavelength_line_separation.", type=fatboyLog.WARNING)
            if (node.hasAttribute("n_segments")):
                try:
                    def_n_segments = int(node.getAttribute("n_segments"))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing n_segments.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing n_segments.", type=fatboyLog.WARNING)
                    def_n_segments = 1
            if (node.hasAttribute("min_lines_to_refine_nonlinear_guess")):
                try:
                    def_min_lines_nonlinear = int(node.getAttribute("min_lines_to_refine_nonlinear_guess"))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing min_lines_to_refine_nonlinear_guess.")
                    self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing min_lines_to_refine_nonlinear_guess.", type=fatboyLog.WARNING)
                    def_min_lines_nonlinear = 12
            #Now create list entries nslits long in wcinfo dict
            wcinfo['fit_order'] = [def_fit_order]*nslits
            wcinfo['line_list'] = [def_line_list]*nslits
            wcinfo['max_wavelength'] = [def_max_wavelength]*nslits
            wcinfo['min_wavelength'] = [def_min_wavelength]*nslits
            wcinfo['min_threshold'] = [def_min_threshold]*nslits
            wcinfo['wavelength_scale_guess'] = [def_wavelength_scale_guess]*nslits
            wcinfo['n_segments'] = [def_n_segments]*nslits
            wcinfo['min_lines_nonlinear'] = [def_min_lines_nonlinear]*nslits

            #Now loop over child nodes
            if (not node.hasChildNodes()):
                break
            islit = 0
            for orderNode in node.childNodes:
                #Ignore non element nodes and nodes not named <order>
                if (orderNode.nodeType != Node.ELEMENT_NODE):
                    continue
                if (orderNode.nodeName != 'order'):
                    continue
                #This is an order node look for slitlet attribute
                if (orderNode.hasAttribute("slitlet")):
                    #This node has a slitlet attribute.  Use this as the index.
                    #Subtract 1 from index to make it zero-ordered
                    try:
                        idx = int(orderNode.getAttribute("slitlet"))-1
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing slitlet index")
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing slitlet index", type=fatboyLog.WARNING)
                        idx = islit
                else:
                    #Use islit as index, assume orders are in order
                    idx = islit
                if (idx >= nslits):
                    print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: index "+str(idx+1)+" is greater than the number of slitlets! Skipping this line!")
                    self._log.writeLog(__name__, "index "+str(idx+1)+" is greater than the number of slitlets! Skipping this line!", type=fatboyLog.WARNING)
                    islit += 1
                    continue
                #Now parse options
                segment = -1
                if (orderNode.hasAttribute("segment")):
                    #This node has a segment attribute.  Use this as the secondary index.
                    #Subtract 1 from index to make it zero-ordered
                    try:
                        segment = int(orderNode.getAttribute("segment"))-1
                        if (not isinstance(wcinfo['fit_order'][idx], list)):
                            #Update wcinfo structure to have lists for each slitidx
                            wcinfo['fit_order'][idx] = [wcinfo['fit_order'][idx]]*wcinfo['n_segments'][idx]
                            wcinfo['line_list'][idx] = [wcinfo['line_list'][idx]]*wcinfo['n_segments'][idx]
                            wcinfo['max_wavelength'][idx] = [wcinfo['max_wavelength'][idx]]*wcinfo['n_segments'][idx]
                            wcinfo['min_wavelength'][idx] = [wcinfo['min_wavelength'][idx]]*wcinfo['n_segments'][idx]
                            wcinfo['min_threshold'][idx] = [wcinfo['min_threshold'][idx]]*wcinfo['n_segments'][idx]
                            wcinfo['wavelength_scale_guess'][idx] = [wcinfo['wavelength_scale_guess'][idx]]*wcinfo['n_segments'][idx]
                            wcinfo['min_lines_nonlinear'][idx] = [wcinfo['min_lines_nonlinear'][idx]]*wcinfo['n_segments'][idx]
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing segment for slitlet "+str(idx+1)+" - "+str(ex))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing segment for slitlet "+str(idx+1)+" - "+str(ex), type=fatboyLog.WARNING)
                    if (segment >= wcinfo['n_segments'][idx]):
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: segment "+str(segment+1)+" is greater than the number of segments!  Skipping this segment!")
                        self._log.writeLog(__name__, "segment "+str(segment+1)+" is greater than the number of segments!  Skipping this segment!", type=fatboyLog.WARNING)
                        islit += 1
                        continue
                if (orderNode.hasAttribute("fit_order")):
                    try:
                        if (segment == -1):
                            wcinfo['fit_order'][idx] = int(orderNode.getAttribute("fit_order"))
                        else:
                            wcinfo['fit_order'][idx][segment] = int(orderNode.getAttribute("fit_order"))
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing fit_order for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing fit_order for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("line_list")):
                    currList = currPath+orderNode.getAttribute("line_list")
                    if (os.access(currList, os.F_OK) or os.access(self._datadir+"/linelists/"+currList, os.F_OK)):
                        if (segment == -1):
                            wcinfo['line_list'][idx] = currList
                        else:
                            wcinfo['line_list'][idx][segment] = currList
                    else:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: could not find line list "+currList+" for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "could not find line list "+currList+" for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("max_wavelength")):
                    try:
                        if (segment == -1):
                            wcinfo['max_wavelength'][idx] = float(orderNode.getAttribute("max_wavelength"))
                        else:
                            wcinfo['max_wavelength'][idx][segment] = float(orderNode.getAttribute("max_wavelength"))
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing max_wavelength for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing max_wavelength for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("min_wavelength")):
                    try:
                        if (segment == -1):
                            wcinfo['min_wavelength'][idx] = float(orderNode.getAttribute("min_wavelength"))
                        else:
                            wcinfo['min_wavelength'][idx][segment] = float(orderNode.getAttribute("min_wavelength"))
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing min_wavelength for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing min_wavelength for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("min_threshold")):
                    try:
                        if (segment == -1):
                            wcinfo['min_threshold'][idx] = float(orderNode.getAttribute("min_threshold"))
                        else:
                            wcinfo['min_threshold'][idx][segment] = float(orderNode.getAttribute("min_threshold"))
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing min_threshold for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing min_threshold for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("wavelength_scale_guess")):
                    try:
                        curr_scale_guess = orderNode.getAttribute("wavelength_scale_guess").split()
                        for j in range(len(curr_scale_guess)):
                            curr_scale_guess[j] = float(curr_scale_guess[j])
                        if (segment == -1):
                            wcinfo['wavelength_scale_guess'][idx] = curr_scale_guess
                        else:
                            wcinfo['wavelength_scale_guess'][idx][segment] = curr_scale_guess
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing wavelength_scale_guess for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing wavelength_scale_guess for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                elif (orderNode.hasAttribute("wavelength_line_1") and orderNode.hasAttribute("wavelength_line_2") and orderNode.hasAttribute("wavelength_line_separation")):
                    try:
                        lambda1 = float(orderNode.getAttribute("wavelength_line_1"))
                        lambda2 = float(orderNode.getAttribute("wavelength_line_2"))
                        delta = float(orderNode.getAttribute("wavelength_line_separation"))
                        if (segment == -1):
                            wcinfo['wavelength_scale_guess'][idx] = abs(lambda1-lambda2)/delta
                        else:
                            wcinfo['wavelength_scale_guess'][idx][segment] = abs(lambda1-lambda2)/delta
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error calculating wavelength_scale_guess for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error calculating wavelength_scale_guess for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("min_lines_to_refine_nonlinear_guess")):
                    try:
                        if (segment == -1):
                            wcinfo['min_lines_nonlinear'][idx] = float(orderNode.getAttribute("min_lines_to_refine_nonlinear_guess"))
                        else:
                            wcinfo['min_lines_nonlinear'][idx][segment] = float(orderNode.getAttribute("min_lines_to_refine_nonlinear_guess"))
                    except Exception as ex:
                        print("wavelengthCalibrateProcess::readWavelengthCalibrationFile> Warning: misformatted line in "+wcfile+": error parsing min_lines_to_refine_nonlinear_guess for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+wcfile+": error parsing min_lines_to_refine_nonlinear_guess for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                #Update islit
                islit += 1
            #There should be only one <dataset> tag so break here
            break
        return wcinfo
    #end readWavelengthCalibrationFile

    def refineWavelengthScale(self, npass, nlines, nonlinear, reflines, wlines, scale, min_wavelength, max_wavelength, dummySize, min_lines_nonlinear, coeffs=[]):
        success = True
        if ((npass == 1 or nlines < 8) and not nonlinear):
            #Use linear scale for first pass until range is wide enough / and enough lines to use 2nd order solution
            #Unless nonlinear = True
            p = zeros(2, dtype=float32)
            p[0] = wlines[0]-scale*reflines[0]
            p[1] = scale
            try:
                lsq = leastsq(linResiduals, p, args=(array(reflines), array(wlines)))
                scale = lsq[0][1]
                print("\t\tGuess at wavelength scale (units/pixel): "+formatNum(scale))
                self._log.writeLog(__name__, "Guess at wavelength scale (units/pixel): "+formatNum(scale), printCaller=False, tabLevel=2)
            except Exception:
                #Print error statement back in wavelengthCalibrate when result obtained
                success = False
            dummySize = int((max_wavelength-min_wavelength)/abs(scale))
            dummyWave = arange(dummySize, dtype=float32)*scale+min_wavelength
            if (scale < 0):
                #Negative scale, need wavelenghts descending
                dummyWave = arange(dummySize, dtype=float32)*scale+max_wavelength
        elif (not nonlinear):
            #Fit order 2 polynomial to current data
            p =  zeros(3, dtype=float32)
            p[0] = wlines[0]-scale*reflines[0]
            p[1] = scale
            try:
                lsq = leastsq(polyResiduals, p, args=(array(reflines), array(wlines), len(p)-1))
                scale = lsq[0][1]
                print("\t\tGuess at 2nd order solution: "+formatList(lsq[0]))
                self._log.writeLog(__name__, "Guess at 2nd order solution: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
            except Exception:
                #Print error statement back in wavelengthCalibrate when result obtained
                success = False
            #Refine dummy arrays with new 2nd order solution
            #conservative starting point
            dummySize = int(dummySize//2)
            while (min_wavelength+abs(lsq[0][1]*dummySize+lsq[0][2]*dummySize**2) < max_wavelength):
                dummySize += 1
            xs = arange(dummySize, dtype=float32)
            dummyWave = min_wavelength+lsq[0][1]*xs+lsq[0][2]*xs**2
            if (dummyWave[-1] < dummyWave[0]):
                #Negative scale, need wavelenghts descending
                dummyWave = max_wavelength+lsq[0][1]*xs+lsq[0][2]*xs**2
        elif (npass > 1 and nlines >= min_lines_nonlinear):
            #Found at least min_lines_nonlinear and not first pass
            #Try calculating solution from current data
            dummyOrder = len(coeffs)-1
            p = array(coeffs, dtype=float32)
            p[0] = wlines[0]-scale*reflines[0]
            try:
                lsq = leastsq(polyResiduals, p, args=(array(reflines), array(wlines), dummyOrder))
                coeffs[1:] = lsq[0][1:]
                #coeffs = lsq[0]
                #coeffs[0] = 0
                print("\t\tGuess at order "+str(dummyOrder)+" solution: "+formatList(lsq[0]))
                self._log.writeLog(__name__, "Guess at order "+str(dummyOrder)+" solution: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
            except Exception:
                #Print error statement back in wavelengthCalibrate when result obtained
                success = False
            #Refine dummy arrays with higher order solution
            #conservative starting point
            dummySize = int(dummySize//2)
            while (min_wavelength+abs(polyFunction(coeffs, dummySize, dummyOrder)) < max_wavelength):
                #Higher order terms can do wonky stuff
                if (abs(polyFunction(coeffs, dummySize, dummyOrder)) < abs(polyFunction(coeffs, dummySize-10, dummyOrder))):
                    break
                dummySize += 10
            xs = arange(dummySize, dtype=float32)
            dummyWave = min_wavelength+polyFunction(coeffs, xs, dummyOrder)
            if (polyFunction(coeffs, dummySize, dummyOrder) < 0):
                #Negative scale, need wavelengths descending
                dummyWave = max_wavelength+polyFunction(coeffs, xs, dummyOrder)
        else:
            dummyOrder = len(coeffs)-1
            xs = arange(dummySize, dtype=float32)
            dummyWave = min_wavelength+polyFunction(coeffs, xs, dummyOrder)
            if (polyFunction(coeffs, dummySize, dummyOrder) < 0):
                #Negative scale, need wavelengths descending
                dummyWave = max_wavelength+polyFunction(coeffs, xs, dummyOrder)
        #Create new dummyFlux array.
        dummyFlux = zeros(dummySize, dtype=float32)
        #Coeffs should be updated without returning
        return (success, scale, dummySize, dummyWave, dummyFlux)
    #end refineWavelengthScale

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('bright_line_searchbox_max', '-100')
        self._optioninfo.setdefault('bright_line_searchbox_max', 'Max pixel value of search box, negative notation allowed.')
        self._options.setdefault('bright_line_searchbox_min', '100')
        self._optioninfo.setdefault('bright_line_searchbox_min', 'Min pixel value of search box.')
        self._options.setdefault('calibrate_slitlets_individually', 'no')
        self._optioninfo.setdefault('calibrate_slitlets_individually', 'Set to yes to handle each slitlet individually if image has not been "jailbarred"')
        self._options.setdefault('debug_mode', 'no')
        self._optioninfo.setdefault('debug_mode', 'Show plots of each slitlet and print out debugging information.')
        self._options.setdefault('fit_order', '3')
        self._optioninfo.setdefault('fit_order', 'Order of polynomial to use to fit wavelength solution.\nRecommended value = 3.')
        self._options.setdefault('line_list', None)
        self._optioninfo.setdefault('ASCII file containing line wavelengths and relative intensities.\nOptional 3rd column contains flag of -1 for blended lines\nthat should not be used in final fit.')
        self._options.setdefault('max_bright_line_separation', None)
        self._optioninfo.setdefault('max_bright_line_separation', 'Maximum separation in pixels for any two bright lines\nto be used in matching up 3 brightest.\nUsed when data is nonlinear.')
        self._options.setdefault('max_shift_tolerance', None)
        self._optioninfo.setdefault('max_shift_tolerance', 'Maximum tolerance in pixels for the shift between the initial\nguess as to a lines position and the peak of the cross-correlation\nbetween data and dummy spectrum. Cross-correlation\nwindow is 50 pixels wide for reference. Suggested value\n15 to 20 if extraneous lines in\nline list')
        self._options.setdefault('max_wavelength', '18500')
        self._optioninfo.setdefault('max_wavelength', 'Maximum wavelength of data coverage, for use in\nconstructing "dummy" spectrum.')
        self._options.setdefault('min_bright_line_separation', '15')
        self._optioninfo.setdefault('min_bright_line_separation', 'Minimum separation in pixels for two bright lines\nto be used in matching up 3 brightest.')
        self._options.setdefault('min_intensity_percent', '0.5')
        self._optioninfo.setdefault('min_intensity_percent', 'Minimum intensity as a percent of the brightest line\nfor a line to be detected')
        self._options.setdefault('min_lines_to_refine_nonlinear_guess', '12')
        self._optioninfo.setdefault('min_lines_to_refine_nonlinear_guess', 'Minimum number of lines that must be matched\nbefore refining a nonlinear initial guess.')
        self._options.setdefault('min_threshold', '3')
        self._optioninfo.setdefault('min_threshold', 'Minimum local sigma threshold compared to noise for line\nto be detected')
        self._options.setdefault('min_wavelength', '10000')
        self._optioninfo.setdefault('min_wavelength', 'Minimum wavelength of data coverage, for use in\nconstructing "dummy" spectrum.')
        self._options.setdefault('n_brightest_data', '5')
        self._optioninfo.setdefault('n_brightest_data', 'If the 3 brightest lines in the image cannot be matched,\ntry permutations of up to the n brightest.\nShould rarely need to be changed.\nUseful if line is missing from line list\nor scaling is vastly off for a line.')
        self._options.setdefault('n_brightest_lines', '14')
        self._optioninfo.setdefault('n_brightest_lines', 'Use the n brightest lines in the line list to compare\nto the 3 brightest in the image.\nShould rarely need to be changed.')
        self._options.setdefault('n_segments', '1')
        self._optioninfo.setdefault('n_segments', 'Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases.')
        self._options.setdefault('resample_to_common_scale', 'yes')
        self._optioninfo.setdefault('resample_to_common_scale', 'Resample all slitlets to common scale for MOS data.\nIf set to no, each slitlet will be resampled to an\nindependent linear scale.')
        self._options.setdefault('slitlets_to_debug', None)
        self._optioninfo.setdefault('slitlets_to_debug', 'Set to a list - 3,5,7 - to debug only certain slitlets')
        self._options.setdefault('slitlets_to_write_plots', None)
        self._optioninfo.setdefault('slitlets_to_write_plots', 'Set to a list - 3,5,7 - to plot only certain slitlets')
        self._options.setdefault('use_arclamps', 'no')
        self._optioninfo.setdefault('use_arclamps', 'no = use master "clean sky", yes = use master arclamp')
        self._options.setdefault('use_initial_guess_on_fail', 'no')
        self._optioninfo.setdefault('use_initial_guess_on_fail', 'Use the initial guess to the wavelength solution if\na solution cannot be found.')
        self._options.setdefault('wavelength_calibration_file', None)
        self._optioninfo.setdefault('wavelength_calibration_file', 'An XML file with wavelength calibration info for the image as a whole\nor for each order individually.\nAny options can be passed as attributes of <dataset> tag\nAnd <order> subtag, which also has optional attribute "slitlet",\nwhich refers to index in slitmask')
        self._options.setdefault('wavelength_line_1', None)
        self._optioninfo.setdefault('wavelength_line_1', 'The wavelength (in output units) of a particular line,\nfor use in constructing "dummy" spectrum.')
        self._options.setdefault('wavelength_line_2', None)
        self._optioninfo.setdefault('wavelength_line_2', 'The wavelength (in output units) of a particular line,\nfor use in constructing "dummy" spectrum.')
        self._options.setdefault('wavelength_line_separation', None)
        self._optioninfo.setdefault('wavelength_separation', 'The separation in pixels between line_1 and line_2,\nfor use in constructing "dummy" spectrum.')
        self._options.setdefault('wavelength_scale_guess', None)
        self._optioninfo.setdefault('wavelength_scale_guess', 'Initial guess of linear wavelength scale,\nfor use in constructing "dummy" spectrum.\nCan also be space delmited list of\npolynomail coefficients, starting with linear term.')
        self._options.setdefault('write_noisemaps', 'no')
        self._options.setdefault('write_plots', 'no')
    #end setDefaultOptions

    ## Wavelength Calibrate data
    def wavelengthCalibrate(self, fdu, calibs):
        ###*** For purposes of wavelengthCalibrate algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        #Read options
        doIndividualSlitlets = False
        if (self.getOption("calibrate_slitlets_individually", fdu.getTag()).lower() == "yes"):
            doIndividualSlitlets = True
        useArclamps = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True
        n_brightest_lines = int(self.getOption("n_brightest_lines", fdu.getTag()))
        n_brightest_data = int(self.getOption("n_brightest_data", fdu.getTag()))
        min_separation = int(self.getOption("min_bright_line_separation", fdu.getTag()))
        bl_min = int(self.getOption("bright_line_searchbox_min", fdu.getTag()))
        bl_max = int(self.getOption("bright_line_searchbox_max", fdu.getTag()))
        min_intensity_pct = float(self.getOption("min_intensity_percent", fdu.getTag()))/100.0
        use_tolerance = False
        if (self.getOption("max_shift_tolerance", fdu.getTag()) is not None):
            use_tolerance = True
            shift_tol = int(self.getOption("max_shift_tolerance", fdu.getTag()))
        resampleCommonScale = False
        if (self.getOption("resample_to_common_scale", fdu.getTag()).lower() == "yes"):
            resampleCommonScale = True
        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        useInitialOnFail = False
        if (self.getOption("use_initial_guess_on_fail", fdu.getTag()).lower() == "yes"):
            useInitialOnFail = True
        debugSlitlets = False
        if (self.getOption("slitlets_to_debug", fdu.getTag()) is not None):
            debugSlitlets = True
            slitletsToDebug = self.getOption("slitlets_to_debug", fdu.getTag()).split(",")
            try:
                for j in range(len(slitletsToDebug)):
                    slitletsToDebug[j] = int(slitletsToDebug[j].strip())
            except Exception as ex:
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Misformatted option slitlets_to_debug: "+self.getOption("slitlets_to_debug", fdu.getTag()))
                self._log.writeLog(__name__, "Misformatted option slitlets_to_debug: "+self.getOption("slitlets_to_debug", fdu.getTag()), type=fatboyLog.WARNING)
        plotSlitlets = False
        if (self.getOption("slitlets_to_write_plots", fdu.getTag()) is not None):
            plotSlitlets = True
            slitletsToPlot = self.getOption("slitlets_to_write_plots", fdu.getTag()).split(",")
            try:
                for j in range(len(slitletsToPlot)):
                    slitletsToPlot[j] = int(slitletsToPlot[j].strip())
            except Exception as ex:
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Misformatted option slitlets_to_write_plots: "+self.getOption("slitlets_to_write_plots", fdu.getTag()))
                self._log.writeLog(__name__, "Misformatted option slitlets_to_write_plots: "+self.getOption("slitlets_to_write_plots", fdu.getTag()), type=fatboyLog.WARNING)

        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and 'slitmask' in calibs):
            ###MOS/IFU data -- get slitmask
            #Use calibration frame slitmask, which corresponds to master lamp and clean sky
            #And not "slitmask" property of each FDU which has been shifted and added and has different shape
            slitmask = calibs['slitmask']
            if (doIndividualSlitlets):
                if (not slitmask.hasProperty("nslits")):
                    slitmask.setProperty("nslits", slitmask.getData().max())
                nslits = slitmask.getProperty("nslits")
                if (slitmask.hasProperty("regions")):
                    (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
                else:
                    #Use helper method to all ylo, yhi for each slit in each frame
                    (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                    slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))
            else:
                #doIndividualSlitlets = False - treat MOS data as 1 slit
                nslits = 1
                b = where(slitmask.getData() != 0)
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    ylos = [b[0].min()]
                    yhis = [b[0].max()]
                else:
                    ylos = [b[1].min()]
                    yhis = [b[1].max()]

        useWCfile = False
        wcfile = self.getOption("wavelength_calibration_file", fdu.getTag())
        if (fdu.hasProperty("wavelength_calibration_file")):
            #If this individual fdu has a property wavelength_calibration_file, it overrides option
            wcfile = fdu.getProperty("wavelength_calibration_file")
        if (wcfile is not None):
            if (os.access(wcfile, os.F_OK) or os.access(self._datadir+"/config/"+wcfile, os.F_OK)):
                wcinfo = self.readWavelengthCalibrationFile(wcfile, nslits, fdu)
                if (wcinfo is not None):
                    useWCfile = True
            else:
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Could not find wavelength_calibration_file "+wcfile+" for "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Could not find wavelength_calibration_file "+wcfile+" for "+fdu.getFullId()+"!", type=fatboyLog.WARNING)

        if (not useWCfile):
            #Read other options only if not using a wc file!
            fit_order = int(self.getOption("fit_order", fdu.getTag()))
            max_wavelength = float(self.getOption("max_wavelength", fdu.getTag()))
            min_wavelength = float(self.getOption("min_wavelength", fdu.getTag()))
            min_threshold = float(self.getOption("min_threshold", fdu.getTag()))
            n_segments = int(self.getOption("n_segments", fdu.getTag()))
            min_lines_nonlinear = int(self.getOption("min_lines_to_refine_nonlinear_guess", fdu.getTag()))
            line_list = self.getOption("line_list", fdu.getTag())
            if (line_list is None):
                print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No line_list given for "+fdu.getFullId()+"! Discarding Image!")
                self._log.writeLog(__name__, "No line_list given for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return
            if (not os.access(line_list, os.F_OK) and not os.access(self._datadir+"/linelists/"+line_list, os.F_OK)):
                print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not find line_list "+line_list+" for "+fdu.getFullId()+"! Discarding Image!")
                self._log.writeLog(__name__, "Could not find line_list "+line_list+" for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return
            wavelength_scale_guess = self.getOption("wavelength_scale_guess", fdu.getTag())
            if (wavelength_scale_guess is None):
                try:
                    lambda1 = float(self.getOption("wavelength_line_1", fdu.getTag()))
                    lambda2 = float(self.getOption("wavelength_line_2", fdu.getTag()))
                    delta = float(self.getOption("wavelength_line_separation", fdu.getTag()))
                    #Allow to be negative
                    wavelength_scale_guess = (lambda1-lambda2)/delta
                except Exception as ex:
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No wavelength_scale_guess or (wavelength_line_1, wavelength_line_2, and wavelength_line_separation) found for "+fdu.getFullId()+"! Discarding Image!")
                    self._log.writeLog(__name__, "No wavelength_scale_guess or (wavelength_line_1, wavelength_line_2, and wavelength_line_separation) found for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return
            else:
                wavelength_scale_guess = wavelength_scale_guess.split()
                for j in range(len(wavelength_scale_guess)):
                    wavelength_scale_guess[j] = float(wavelength_scale_guess[j])

        #Next check for master arclamp frame or clean sky frame to rectify skylines
        #These should have been found above and added to calibs dict
        skyFDU = None
        if (useArclamps):
            if ('masterLamp' in calibs):
                skyFDU = calibs['masterLamp']
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Using master ARCLAMP "+skyFDU.getFullId())
                self._log.writeLog(__name__, "Using master ARCLAMP "+skyFDU.getFullId())
            else:
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Could not find master arclamp associated with "+fdu.getFullId()+"! Attempting to use clean sky frame for wavelength calibration!")
                self._log.writeLog(__name__, "Could not find master arclamp associated with "+fdu.getFullId()+"! Attempting to use clean sky frame for wavelength calibration!", type=fatboyLog.WARNING)
                useArclamps = False
        if (skyFDU is None and 'cleanSky' in calibs):
            #Either use_arclamps = no or master arclamp not found
            skyFDU = calibs['cleanSky']
            print("wavelengthCalibrateProcess::wavelengthCalibrate> Using master CLEAN SKY "+skyFDU.getFullId())
            self._log.writeLog(__name__, "Using master CLEAN SKY "+skyFDU.getFullId())
        if (skyFDU is None):
            print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not find clean sky frame associated with "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not find clean sky frame associated with "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        #Create output dir if it doesn't exist
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/wavelengthCalibrated", os.F_OK)):
            os.mkdir(outdir+"/wavelengthCalibrated",0o755)

        #Create new header dict
        wcHeader = dict()
        #Get sky data for qadata
        qadata = skyFDU.getData().copy()
        #Create fitParams list to store fit parameters for resampling data
        fitParams = []
        #Create qaParams list to store qa data on nlines fit and sigma
        qaParams = []
        minLambda = None #set minLambda to None here so it can be checked later in case first slit is not fit
        minLambdaList = []
        maxLambdaList = []
        coeffs = []

        #Auto-trim for longslit data
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                y = skyFDU.getData().copy().sum(1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                y = skyFDU.getData().copy().sum(0)
            trimsec = extractSpectra(y, sigma=5, width=5, nspec=1)
            if (trimsec is not None and (trimsec[0][1]-trimsec[0][0]) > ysize/10):
                ylos = [trimsec[0][0]]
                yhis = [trimsec[0][1]]
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Auto-trimmed to use section "+str(trimsec[0]))
                self._log.writeLog(__name__, "Auto-trimmed to use section "+str(trimsec[0]))

        #Loop over nslits
        for j in range(nslits):
            #Reset scale to original wavelength_scale_guess in each iteration
            if (not useWCfile):
                scale = wavelength_scale_guess[0]
                fit_order = int(self.getOption("fit_order", fdu.getTag()))
                nonlinear = False
                if (len(wavelength_scale_guess) > 1):
                    nonlinear = True
                    #Create coeffs list with 0 constant term
                    coeffs = [0]
                    coeffs.extend(wavelength_scale_guess)
            else:
                n_segments = self.getWCParam(wcinfo, 'n_segments', j)
            #mult_seg = True if multiple segments
            mult_seg = (n_segments > 1)
            fitParams.append([]) #Append new empty list for this order

            #Check slitlets_to_debug
            if (debugSlitlets):
                if ((j+1) in slitletsToDebug):
                    self.setOption("debug_mode", "yes")
                else:
                    self.setOption("debug_mode", "no")
            #Check slitlets_to_write_plots
            if (plotSlitlets):
                if ((j+1) in slitletsToPlot):
                    self.setOption("write_plots", "yes")
                else:
                    self.setOption("write_plots", "no")

            #Now loop over segments.  If n_segments == 1, this loop will only be done once (most data)
            for seg in range(n_segments):
                pass_name = " for order "+str(j+1)+" of "
                #define output specfile, residfile
                specfile = outdir+"/wavelengthCalibrated/spec_"+fdu._id
                residfile = outdir+"/wavelengthCalibrated/resid_"+fdu._id
                if (nslits > 1):
                    #Append slit number
                    specfile += "_slitlet_"+str(j+1)
                    residfile += "_slitlet_"+str(j+1)
                if (mult_seg):
                    pass_name = " for order "+str(j+1)+", segment "+str(seg+1)+" of "
                    #Append segment number
                    specfile += "_segment_"+str(seg+1)
                    residfile += "_segment_"+str(seg+1)
                specfile += ".dat"
                residfile += ".dat"

                if (useWCfile):
                    #Update options for this slit (or entire image) if using wc file
                    fit_order = self.getWCParam(wcinfo, 'fit_order', j, mult_seg, seg)
                    max_wavelength = self.getWCParam(wcinfo, 'max_wavelength', j, mult_seg, seg)
                    min_wavelength = self.getWCParam(wcinfo, 'min_wavelength', j, mult_seg, seg)
                    min_threshold = self.getWCParam(wcinfo, 'min_threshold', j, mult_seg, seg)
                    min_lines_nonlinear = self.getWCParam(wcinfo, 'min_lines_nonlinear', j, mult_seg, seg)
                    line_list = self.getWCParam(wcinfo, 'line_list', j, mult_seg, seg)
                    scale_guess = self.getWCParam(wcinfo, 'wavelength_scale_guess', j, mult_seg, seg)

                    scale = scale_guess[0]
                    nonlinear = False
                    if (len(scale_guess) > 1):
                        nonlinear = True
                    #Create coeffs list with 0 constant term
                    coeffs = [0]
                    coeffs.extend(scale_guess)
                    if (line_list is None):
                        print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not find line_list "+line_list+pass_name+fdu.getFullId()+"! Skipping order!")
                        self._log.writeLog(__name__, "Could not find line_list "+line_list+pass_name+fdu.getFullId()+"! Skipping order!", type=fatboyLog.ERROR)
                        continue
                    if (scale is None):
                        print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No wavelength_scale_guess "+pass_name+fdu.getFullId()+"! Skipping order!")
                        self._log.writeLog(__name__, "No wavelength_scale_guess "+pass_name+fdu.getFullId()+"! Skipping order!", type=fatboyLog.ERROR)
                        continue

                (masterWave, masterFlux, masterFlag) = self.readLineList(line_list)
                if (nslits > 1):
                    if (mult_seg):
                        print("wavelengthCalibrateProcess::wavelengthCalibrate> Finding wavelength solution to slitlet "+str(j+1)+", segment "+str(seg+1)+"...")
                        self._log.writeLog(__name__, "Finding wavelength solution to slitlet "+str(j+1)+", segment "+str(seg+1)+"...")
                    else:
                        print("wavelengthCalibrateProcess::wavelengthCalibrate> Finding wavelength solution to slitlet "+str(j+1)+"...")
                        self._log.writeLog(__name__, "Finding wavelength solution to slitlet "+str(j+1)+"...")

                #Take 1-d cut of skyFDU (cleanSky or masterLamp) data
                #Ensure no negative index values
                if (ylos[j] < 0):
                    ylos[j] = 0
                currMask = None

                #Select kernel for 2d median
                kernel2d = fatboyclib.median2d
                if (self._fdb.getGPUMode()):
                    #Use GPU for medians
                    kernel2d=gpumedian2d

                #Use xstride to split into equal length segments here
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    xstride = skyFDU.getShape()[1]//n_segments
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                    slit = skyFDU.getData()[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi].copy()
                    if (slitmask is not None):
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        if (doIndividualSlitlets):
                            currMask = slitmask.getData()[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi] == (j+1)
                        else:
                            currMask = slitmask.getData()[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi] != 0
                        slit *= currMask
                    if (int(ylos[j]) == int(yhis[j])):
                        oned = slit.ravel()
                    elif (self._fdb.getGPUMode()):
                        #Use GPU
                        oned = gpu_arraymedian(slit, axis="Y", nonzero=True, kernel2d=kernel2d, even=True)
                    else:
                        #Use CPU
                        oned = kernel2d(slit.transpose().copy(), nonzero=True, even=True)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    xstride = skyFDU.getShape()[0]//n_segments
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                    slit = skyFDU.getData()[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)].copy()
                    if (slitmask is not None):
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        if (doIndividualSlitlets):
                            currMask = slitmask.getData()[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)] == (j+1)
                        else:
                            currMask = slitmask.getData()[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)] != 0
                        slit *= currMask
                    if (int(ylos[j]) == int(yhis[j])):
                        oned = slit.ravel()
                    else:
                        oned = gpu_arraymedian(slit, axis="X", nonzero=True, kernel2d=kernel2d)

                #Filter the 1-d cut!
                #Use quartile instead of median to get better estimate of background levels!
                #Use 2 passes of quartile filter
                badpix = oned == 0 #First find bad pixels
                #Correct for big negative values
                oned[where(oned < -100)] = 1.e-6
                for i in range(2):
                    tempcut = zeros(len(oned))
                    nh = 25-badpix[:51].sum()//2 #Instead of defaulting to 25 for quartile, use median of bottom half of *nonzero* pixels
                    for k in range(25):
                        #tempcut[k] = oned[k] - gpu_arraymedian(oned[:51],nonzero=True,nhigh=25)
                        tempcut[k] = oned[k] - gpu_arraymedian(oned[:51],nonzero=True,nhigh=nh)
                    for k in range(25,len(oned)-25):
                        nh = 25-badpix[k-25:k+26].sum()//2
                        tempcut[k] = oned[k] - gpu_arraymedian(oned[k-25:k+26],nonzero=True,nhigh=nh)
                    nh = 25-badpix[len(oned)-50:].sum()//2
                    for k in range(len(oned)-25,len(oned)):
                        tempcut[k] = oned[k] - gpu_arraymedian(oned[len(oned)-50:],nonzero=True,nhigh=nh)
                    #Set zero values to small positive number to avoid being flagged
                    tempcut[tempcut == 0] = 1.e-6
                    #Correct for big negative values
                    tempcut[where(tempcut < -100)] = 1.e-6
                    oned = tempcut
                #Set bad pixels back to 0
                oned[badpix] = 0
                oned[:10] = 0
                oned[-10:] = 0

                #Find brightest line in image
                blref = where(oned == max(oned))[0][0]
                maxPeak = max(oned)
                #Fit a Gaussian to find shape.  Square data first to ensure that
                #bright line dominates fit
                refCut = oned[blref-10:blref+11]**2
                p = zeros(4, dtype=float64)
                p[0] = max(refCut)
                p[1] = 10
                p[2] = 2
                p[3] = gpu_arraymedian(refCut, nonzero=True)
                try:
                    lsq = leastsq(gaussResiduals, p, args=(arange(len(refCut), dtype=float64), refCut))
                except Exception as ex:
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> WARNING: leastsq fit failed at pixel "+str(blref)+"; Using gaussWidth=1.5.")
                    self._log.writeLog(__name__, "leastsq fit failed at pixel "+str(blref)+"; Using gaussWidth=1.5.", type=fatboyLog.WARNING)
                    gaussWidth = 1.5
                #Multiply by sqrt(2) because we fit squared data.  This will
                #give us the width of the emission lines.
                gaussWidth = abs(lsq[0][2]*sqrt(2))
                if (gaussWidth > 2.5):
                    gaussWidth = 1.5
                    #If its a broad line for some reason, use 1.5 as default
                elif (gaussWidth > 2):
                    gaussWidth = 1.75

                if (min_wavelength > max_wavelength):
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> WARNING: min wavelength "+str(min_wavelength)+" is greater than max wavelength "+str(max_wavelength)+"!  Flipping them and proceeding...")
                    self._log.writeLog(__name__, "min wavelength "+str(min_wavelength)+" is greater than max wavelength "+str(max_wavelength)+"!  Flipping them and proceeding...", type=fatboyLog.WARNING)
                    tmp = min_wavelength
                    min_wavelength = max_wavelength
                    max_wavelength = tmp

                #Create 1st guess at dummy 1-d cuts
                dummySize = int((max_wavelength-min_wavelength)/abs(scale))
                dummyFlux = zeros(dummySize, dtype=float32)
                dummyWave = arange(dummySize, dtype=float32)*scale+min_wavelength
                if (scale < 0):
                    #Negative scale, need wavelengths descending
                    dummyWave = arange(dummySize, dtype=float32)*scale+max_wavelength

                dummyOrder = 1
                if (nonlinear):
                    dummyOrder = len(coeffs)-1
                    #Refine dummy arrays with higher order solution
                    #conservative starting point
                    dummySize = int(dummySize//2)
                    while (min_wavelength+abs(polyFunction(coeffs, dummySize, dummyOrder)) < max_wavelength):
                        dummySize += 10
                    dummyFlux = zeros(dummySize, dtype=float32)
                    xs = arange(dummySize, dtype=float32)
                    dummyWave = min_wavelength+polyFunction(coeffs, xs, dummyOrder)
                    if (polyFunction(coeffs, dummySize, dummyOrder) < 0):
                    #Negative scale, need wavelengths descending
                        dummyWave = max_wavelength+polyFunction(coeffs, xs, dummyOrder)

                #Add gaussians for each line in line list
                dummyFlux = self.populateDummyFlux(masterFlux, masterWave, dummyWave, dummyFlux, scale, gaussWidth)
                if (dummyFlux[100:-100].max() == 0):
                    #If this happened, no lines were found in the given wavelength range!  Print error and skip slitlet
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No lines found in wavelength range ["+str(min_wavelength)+":"+str(max_wavelength)+"] "+pass_name+fdu.getFullId()+"! Skipping order!")
                    self._log.writeLog(__name__, "No lines found in wavelength range ["+str(min_wavelength)+":"+str(max_wavelength)+"] "+pass_name+fdu.getFullId()+"! Skipping order!", type=fatboyLog.ERROR)
                    if (useInitialOnFail):
                        #Update header info
                        if (len(coeffs) == 0):
                            coeffs = [0]
                        coeff_order = len(coeffs)-1
                        coeffs[0] = min_wavelength
                        if (scale < 0):
                            coeffs[0] = max_wavelength
                        if ((j == 0 and seg == 0) or minLambda is None):
                            #Update CTYPE on first pass
                            wcHeader['CTYPE'] = 'WAVE-PLY'
                            if (scale > 0):
                                minLambda = polyFunction(coeffs, 0, coeff_order)
                                maxLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            else:
                                maxLambda = polyFunction(coeffs, 0, coeff_order)
                                minLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            minLambdaList.append(minLambda)
                            maxLambdaList.append(maxLambda)
                        else:
                            if (scale > 0):
                                localMinLambda = polyFunction(coeffs, 0, coeff_order)
                                localMaxLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            else:
                                localMaxLambda = polyFunction(coeffs, 0, coeff_order)
                                localMinLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            minLambda = min(minLambda, localMinLambda)
                            maxLambda = max(maxLambda, localMaxLambda)
                            minLambdaList.append(localMinLambda)
                            maxLambdaList.append(localMaxLambda)

                        if (nslits == 1 and not mult_seg):
                            #If only one slit, use PORDER and PCOEFF_i
                            wcHeader['PORDER'] = coeff_order
                            wcHeader['NSEG_01'] = 1
                            for i in range(coeff_order+1):
                                wcHeader['PCOEFF_'+str(i)] = coeffs[i]
                        else:
                            #Use PORDERxx and PCFi_Sxx
                            slitStr = str(j+1)
                            if (j+1 < 10):
                                slitStr = '0'+slitStr
                            wcHeader['NSEG_'+slitStr] = n_segments
                            if (not mult_seg):
                                wcHeader['PORDER'+slitStr] = coeff_order
                                for i in range(coeff_order+1):
                                    wcHeader['PCF'+str(i)+'_S'+slitStr] = coeffs[i]
                            else:
                                #Multiple segments, use hierarchical keywords PORDER_xx_SEGy and PCFi_Sxx_SEGy
                                slitStr += '_SEG' + str(seg)
                                wcHeader['HIERARCH PORDER'+slitStr] = coeff_order
                                for i in range(coeff_order+1):
                                    wcHeader['HIERARCH PCF'+str(i)+'_S'+slitStr] = coeffs[i]
                        fitParams[j].append(coeffs) #Use initial guess
                    else:
                        #Add empty list to fitParams
                        fitParams[j].append([])
                        minLambdaList.append(0)
                        maxLambdaList.append(xsize-1)
                    continue

                #Scale template
                fluxScale = oned[100:-100].max()/dummyFlux[100:-100].max()
                dummyFlux *= fluxScale
                #Set zero values to small positive number so they're not considered flagged
                dummyFlux[dummyFlux == 0] = 1.e-6
                #Correct for big negative values
                dummyFlux[where(dummyFlux < -100)] = 1.e-6

                #Find n_brightest_data (default 3) brightest lines in image, n_brightest_lines (default 14) brightest in template
                wclines = []
                wccentroids = [] #Keep track of actual centroids of lines for calculating wavelength scale
                lineParams = [] #Keep track of Gaussian parameters for each line
                refCut = oned.copy()
                lineWidths = []
                linePeaks = []
                #Only look from 100 to length-100
                #Only look from search_min to search_max - defaults 100, -100
                refCut[0:bl_min] = 0
                refCut[bl_max:] = 0
                #Use 3 passes to find and fit brightest line in refCut
                #Then zero out 21 pixels centered around line
                for i in range(n_brightest_data):
                    blref = where(refCut == max(refCut))[0][0]
                    if (blref < bl_min or blref >= bl_max % len(refCut)):
                        refCut[blref] = refCut.min()-1
                        continue
                    keepLine = False
                    while (not keepLine):
                        keepLine = True
                        for k in range(len(wclines)):
                            if (abs(blref-wclines[k]) < min_separation):
                                #Too close to another line
                                keepLine = False
                        if (not keepLine):
                            #Fit Gaussian and remove line 8/29/19
                            tempCut = refCut[blref-10:blref+11]**2
                            p = zeros(4, dtype=float64)
                            p[0] = max(tempCut)
                            p[1] = 10
                            p[2] = gaussWidth/sqrt(2)
                            p[3] = gpu_arraymedian(tempCut)
                            try:
                                lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
                            except Exception as ex:
                                #Should not happen; use initial guess
                                lsq = [p]
                            p = zeros(4)
                            p[0] = sqrt(abs(lsq[0][0]))
                            p[1] = lsq[0][1]+blref-10
                            p[2] = abs(lsq[0][2]*sqrt(2))
                            refCut -= gaussFunction(p, arange(len(refCut), dtype=float32))
                            refCut[refCut < 0] = 0
                            #refCut[blref-2:blref+3] = 0
                            blref = where(refCut == max(refCut))[0][0]

                    #Centroid line for subpixel accuracy
                    #Square data to ensure bright line dominates fit
                    tempCut = refCut[blref-10:blref+11]**2
                    p = zeros(4, dtype=float64)
                    p[0] = max(tempCut)
                    p[1] = 10
                    p[2] = gaussWidth/sqrt(2)
                    p[3] = gpu_arraymedian(tempCut)
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
                    except Exception as ex:
                        #Error centroiding, continue to next line
                        refCut -= gaussFunction(p, arange(len(refCut), dtype=float32))
                        refCut[refCut < 0] = 0
                        continue
                    mcor = lsq[0][1]
                    wccentroids.append(blref+mcor-10) #Actual centroid
                    wclines.append(int(blref+mcor-9.5)) #Rounded to nearest pixel
                    #Check each component's width rather than the average
                    currWidth = abs(lsq[0][2]*sqrt(2))
                    if (currWidth > 2.5):
                        currWidth = 1.5
                    elif (currWidth > 2):
                        currWidth = 1.75
                    lineWidths.append(currWidth)
                    #Add line to lineParams
                    p = zeros(4)
                    p[0] = sqrt(abs(lsq[0][0]))
                    p[1] = lsq[0][1]+blref-10
                    p[2] = abs(lsq[0][2]*sqrt(2))
                    #subtract Gaussian fitted to line rather than zeroing out 21 pixel
                    #box 8/29/19.  Also ensure no negative points
                    refCut -= gaussFunction(p, arange(len(refCut), dtype=float32))
                    refCut[refCut < 0] = 0
                    p[2] = currWidth
                    linePeaks.append(sqrt(abs(lsq[0][0])))
                    #Keep track of Gaussian params for line
                    lineParams.append(p)
                    #zero out 21 pixel box centered at this line
                    #refCut[blref-10:blref+11] = 0

                #Find n_brightest_lines (default 14) brightest lines in "dummy" template
                dlines = []
                dpeak = []
                dwave = [] #Keep track of actual wavelengths from line lists
                dumCut = dummyFlux.copy()
                #Only look from 100 to length-100
                dumCut[0:100] = 0
                dumCut[-100:] = 0
                #Use n_brightest_lines (default 14) passes to find and fit brightest line in dumCut
                #Then zero out 21 pixels centered around line
                for i in range(n_brightest_lines):
                    blref = where(dumCut == max(dumCut))[0][0]
                    if (blref < 100 or blref > len(dumCut)-100):
                        dumCut[blref] = dumCut.min()-1
                        continue
                    #Centroid line for subpixel accuracy
                    #Square data to ensure bright line dominates fit
                    tempCut = dumCut[blref-10:blref+11]**2
                    p = zeros(4, dtype=float64)
                    p[0] = max(tempCut)
                    p[1] = 10
                    p[2] = gaussWidth/sqrt(2)
                    p[3] = gpu_arraymedian(tempCut)
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
                    except Exception as ex:
                        #Error centroiding, continue to next line
                        dumCut -= gaussFunction(p, arange(len(dumCut), dtype=float32))
                        dumCut[dumCut < 0] = 0
                        continue
                    mcor = lsq[0][1]
                    #blref = line center rounded to nearest pixel
                    blref = int(blref+mcor-9.5)
                    dlines.append(blref)
                    #Append flux peak to dpeak list
                    dpeak.append(sqrt(lsq[0][0]))
                    #Get wavelgnth from masterWave -- dummyWave is now an approximation
                    #Find closest wavelength in masterWave
                    dwave.append(masterWave[where(abs(masterWave-dummyWave[blref]) == min(abs(masterWave-dummyWave[blref])))][0])
                    #subtract Gaussian fitted to line rather than zeroing out 21 pixel
                    #box 1/9/20.  Also ensure no negative points
                    p = zeros(4)
                    p[0] = sqrt(abs(lsq[0][0]))
                    p[1] = lsq[0][1]+blref-10
                    p[2] = abs(lsq[0][2]*sqrt(2))
                    #subtract Gaussian fitted to line rather than zeroing out 21 pixel
                    #box 8/29/19.  Also ensure no negative points
                    dumCut -= gaussFunction(p, arange(len(dumCut), dtype=float32))
                    dumCut[dumCut < 0] = 0
                    #zero out 21 pixel box centered at this line
                    #dumCut[blref-10:blref+11] = 0

                #Use helper method to match 3 brightest lines in image with
                #corresponding lines in template
                (success, currLines, dumPeak, idx) = self.match3BrightestLines(coeffs, dlines, dpeak, dwave, dummyFlux, dummyOrder, dummyWave, fdu, oned, nonlinear, scale, usePlot, wccentroids, wclines)
                #reflines = pixel value in image, wlines = wavelength taken from masterWave
                reflines = []
                wlines = []
                if (success):
                    #idx is already sorted for wavelength so confusion as to < > comparisons with pos/neg values
                    wclines = array(wclines)[idx]
                    wccentroids = array(wccentroids)[idx]
                    lineParams = array(lineParams)[idx].tolist()
                    sumWidth = array(lineWidths)[idx].sum()
                    sumPeak = array(linePeaks)[idx].sum()
                else:
                    #Could not match 3 brightest lines
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not match 3 brightest lines "+pass_name+fdu.getFullId()+"! Skipping order!")
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> Check "+specfile+" for data.")
                    self._log.writeLog(__name__, "Could not match 3 brightest lines "+pass_name+fdu.getFullId()+"! Skipping order!", type=fatboyLog.ERROR)
                    self._log.writeLog(__name__, "Check "+specfile+" for data.")
                    #Output "spec" file
                    xs = arange(len(oned), dtype=float32)*scale+min_wavelength
                    if (scale < 0):
                        #Negative scale, need wavelenghts descending
                        xs = arange(len(oned), dtype=float32)*scale+max_wavelength
                    if (nonlinear):
                        xs = min_wavelength+polyFunction(coeffs, arange(len(oned), dtype=float32), dummyOrder)
                        if (polyFunction(coeffs, dummySize, dummyOrder) < 0):
                            #Negative scale, need wavelengths descending
                            xs = max_wavelength+polyFunction(coeffs, arange(len(oned), dtype=float32), dummyOrder)

                    dummyFlux = zeros(len(oned), dtype=float32)
                    dummyWave = xs
                    dummyFlux = self.populateDummyFlux(masterFlux, masterWave, dummyWave, dummyFlux, scale, gaussWidth)
                    #Scale template
                    dummyFlux *= fluxScale
                    f = open(specfile,'w')
                    f.write("#lambda\t1-d cut\tref\tfit\n")
                    for i in range(len(oned)):
                        f.write(str(xs[i])+'\t'+str(oned[i])+'\t'+str(dummyFlux[i])+'\t0.0\n')
                    f.close()
                    if (useInitialOnFail):
                        #Update header info
                        if (len(coeffs) == 0):
                            coeffs = [0]
                        coeff_order = len(coeffs)-1
                        coeffs[0] = min_wavelength
                        if (scale < 0):
                            coeffs[0] = max_wavelength
                        if ((j == 0 and seg == 0) or minLambda is None):
                            #Update CTYPE on first pass
                            wcHeader['CTYPE'] = 'WAVE-PLY'
                            if (scale > 0):
                                minLambda = polyFunction(coeffs, 0, coeff_order)
                                maxLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            else:
                                maxLambda = polyFunction(coeffs, 0, coeff_order)
                                minLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            minLambdaList.append(minLambda)
                            maxLambdaList.append(maxLambda)
                        else:
                            if (scale > 0):
                                localMinLambda = polyFunction(coeffs, 0, coeff_order)
                                localMaxLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            else:
                                localMaxLambda = polyFunction(coeffs, 0, coeff_order)
                                localMinLambda = polyFunction(coeffs, xstride-1, coeff_order)
                            minLambda = min(minLambda, localMinLambda)
                            maxLambda = max(maxLambda, localMaxLambda)
                            minLambdaList.append(localMinLambda)
                            maxLambdaList.append(localMaxLambda)

                        if (nslits == 1 and not mult_seg):
                            #If only one slit, use PORDER and PCOEFF_i
                            wcHeader['PORDER'] = coeff_order
                            wcHeader['NSEG_01'] = 1
                            for i in range(coeff_order+1):
                                wcHeader['PCOEFF_'+str(i)] = coeffs[i]
                        else:
                            #Use PORDERxx and PCFi_Sxx
                            slitStr = str(j+1)
                            if (j+1 < 10):
                                slitStr = '0'+slitStr
                            wcHeader['NSEG_'+slitStr] = n_segments
                            if (not mult_seg):
                                wcHeader['PORDER'+slitStr] = coeff_order
                                for i in range(coeff_order+1):
                                    wcHeader['PCF'+str(i)+'_S'+slitStr] = coeffs[i]
                            else:
                                #Multiple segments, use hierarchical keywords PORDER_xx_SEGy and PCFi_Sxx_SEGy
                                slitStr += '_SEG' + str(seg)
                                wcHeader['HIERARCH PORDER'+slitStr] = coeff_order
                                for i in range(coeff_order+1):
                                    wcHeader['HIERARCH PCF'+str(i)+'_S'+slitStr] = coeffs[i]
                        fitParams[j].append(coeffs) #Use initial guess
                    else:
                        #Add empty list to fitParams
                        fitParams[j].append([])
                        minLambdaList.append(0)
                        maxLambdaList.append(xsize-1)
                    continue

                #oned = one-d cut of image; obsSpec = gaussians of found lines;
                #obsFlag = flagged pixels; resid = used for finding next line
                obsSpec = zeros(len(oned))
                resid = zeros(len(oned))
                obsFlag = ones(len(oned))
                #Add actual wavelengths of 3 lines to wlines array
                for i in range(len(currLines)):
                    #Get wavelgnth from masterWave -- dummyWave is now an approximation
                    #Find closest wavelength in masterWave
                    currWave =  masterWave[where(abs(masterWave-dummyWave[currLines[i]]) == min(abs(masterWave-dummyWave[currLines[i]])))][0]
                    wlines.append(currWave)
                #Add pixel centroid of 3 lines to reflines array
                for i in range(len(wclines)):
                    reflines.append(wccentroids[i])
                    #Add line to obsSpec
                    obsSpec += gaussFunction(lineParams[i], arange(len(obsSpec), dtype=float32))
                #Refine Gaussian width, flux scale by taking average from 3 lines
                gaussWidth = sumWidth/3.
                maxPeak = sumPeak/3.
                fluxScale *= sumPeak/dumPeak
                #Refine scale, dummy arrays
                #With 3 datapoints, use linear approximation
                scale = (wlines[2]-wlines[0])/(reflines[2]-reflines[0])
                p = zeros(2, dtype=float32)
                p[0] = wlines[0]-scale*reflines[0]
                p[1] = scale
                try:
                    lsq = leastsq(linResiduals, p, args=(array(reflines), array(wlines)))
                    scale = lsq[0][1]
                except Exception as ex:
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: exception "+str(ex)+" while doing least squares fit to linear scale.")
                    self._log.writeLog(__name__, "exception "+str(ex)+" while doing least squares fit to linear scale.", type=fatboyLog.WARNING)

                #Print and write to log the wavelengths and pixel values of these 3 lines
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Matched up brightest 3 lines "+pass_name+fdu.getFullId())
                self._log.writeLog(__name__, "Matched up brightest 3 lines "+pass_name+fdu.getFullId())
                for i in range(len(reflines)):
                    print("\tLine ("+str(i)+"): pixel="+formatNum(reflines[i])+", wavelength="+formatNum(wlines[i]))
                    self._log.writeLog(__name__, "Line ("+str(i)+"): pixel="+formatNum(reflines[i])+", wavelength="+formatNum(wlines[i]), printCaller=False, tabLevel=1)
                nlines = 3

                #Force arrays to be updated on first pass
                oldnlines = 0
                findLines = True
                inloop = 0
                xlo = int(max(min(reflines)-200, 50))
                xhi = int(min(max(reflines)+200, len(oned)-51))
                #Loop over other lines in +/- 200 px area

                while (findLines and inloop < 20):
                    #Refine scale with linear approximation if new line found
                    #Do not update if nonlinear until range expands below and at least min_lines_to_refine_nonlinear_guess lines found
                    if (nlines > oldnlines):
                        #Use helper method refineWavelengthScale.  npass = 1.
                        (success, scale, dummySize, dummyWave, dummyFlux) = self.refineWavelengthScale(1, nlines, nonlinear, reflines, wlines, scale, min_wavelength, max_wavelength, dummySize, min_lines_nonlinear, coeffs)
                        if (not success):
                            print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Unable to refine wavelength solution "+pass_name+fdu.getFullId())
                            self._log.writeLog(__name__, "Unable to refine wavelength solution "+pass_name+fdu.getFullId(), type=fatboyLog.WARNING)
                        #Add gaussians for each line in line list
                        dummyFlux = self.populateDummyFlux(masterFlux, masterWave, dummyWave, dummyFlux, scale, gaussWidth, wlines)
                        #Scale template
                        dummyFlux *= fluxScale
                    oldnlines = nlines

                    #Set up resid array = residuals of (oned - found lines) * obsFlag
                    resid = (oned-obsSpec)*obsFlag
                    #If peak in resid array is <= 2*min_intensity_pct (default 1%) of peak of brightest line, break out of loop
                    if (resid[xlo+5:xhi-4].max() <= maxPeak*min_intensity_pct*2):
                        findLines = False
                        break
                    #blref = line center of brightest line remaining in resid, rounded to nearest pixel
                    blref = where(resid[xlo:xhi+1] == max(resid[xlo+5:xhi-4]))[0][0]+xlo
                    #Edge cases - ensure peak of line found
                    if (blref-xlo < 10):
                        blref = where(resid[blref-5:blref+1] == max(resid[blref-5:blref+1]))[0][0]+blref-5
                    if (xhi-blref < 10):
                        blref = where(resid[blref:blref+6] == max(resid[blref:blref+6]))[0][0]+blref
                    if (blref-xlo < 30):
                        resid[blref-50:blref-30] = 0
                    elif (xhi-blref < 30):
                        resid[blref+30:blref+50] = 0
                    refbox = 25
                    searchbox = 25
                    templines = array(reflines)
                    #Find closest line in pixel space to this one to use for initial guesses
                    refline = where(abs(templines-blref) == min(abs(templines-blref)))[0][0]
                    #Find index of dummyWave closest to actual wavelength of refline
                    refIdx = where(abs(dummyWave-wlines[refline]) == min(abs(dummyWave-wlines[refline])))[0][0]
                    #Guess at index offset between oned cut and dummyFlux cut
                    xoffGuess = int(refIdx-reflines[refline])
                    refCut = resid[blref-refbox:blref+refbox+1]

                    if (nonlinear):
                        waveguess = polyFunction(coeffs, blref, dummyOrder)-polyFunction(coeffs, reflines[refline], dummyOrder)+dummyWave[refIdx]
                        guessIdx = where(abs(dummyWave-waveguess) == min(abs(dummyWave-waveguess)))[0][0]
                        xoffGuess = guessIdx-blref
                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        print("Line at ref pixel BLREF ", blref, "nearest line", reflines[refline], "wavelength=",wlines[refline])
                        print("\tguess at offset=", xoffGuess)

                    #Rejection criteria 0-6 now use helper method
                    (success, inloop, lsigma) = self.checkPrefitRejectionCriteria(fdu, blref, resid, reflines, wlines, obsFlag, inloop, refbox, searchbox, nlines, dummyWave, dummyFlux, min_threshold)
                    if (success == False):
                        continue

                    #Cross-correlate
                    dumCut = dummyFlux[blref-searchbox+xoffGuess:blref+searchbox+1+xoffGuess]
                    n = min(len(refCut),len(dumCut))
                    #Make sure refCut and dumCut are same length
                    refCut = refCut[:n]
                    dumCut = dumCut[:n]
                    ccor = correlate(refCut, dumCut, mode='same')
                    #plt.plot(refCut)
                    #plt.plot(dumCut)
                    #plt.show()
                    #plt.plot(ccor)
                    #plt.show()
                    #Fit cross correlation function with a Gaussian
                    p = zeros(4, dtype=float64)
                    p[0] = max(ccor)
                    p[1] = where(ccor == max(ccor))[0][0]
                    if (use_tolerance and abs(p[1]-len(ccor)//2) > shift_tol):
                        #Maybe a line in list that is not in the data?
                        #Zero out 5 pixels around max
                        ccor[max(int(p[1])-2, 0):min(int(p[1])+3, len(ccor))] = 0
                        #Examine second highest peak
                        cmax2 = max(ccor)
                        peak2 = where(ccor == max(ccor))[0][0]
                        if (cmax2 >= p[0]*0.25 and abs(peak2-len(ccor)//2) <= shift_tol):
                            p[0] = cmax2
                            p[1] = peak2
                    p[2] = gaussWidth
                    p[3] = gpu_arraymedian(ccor)
                    llo = max(0, int(p[1]-5))
                    lhi = min(len(ccor), int(p[1]+6))
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(arange(lhi-llo, dtype=float64)+llo, ccor[llo:lhi]))
                    except Exception as ex:
                        #Flag and continue
                        obsFlag[blref-1:blref+2] = 0
                        continue
                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        print("\tleast squares fit to ccor pixel value=",lsq[0][1],"guess param=",p[1])
                    inloop += 1

                    success = self.checkPostfitRejectionCriteria(fdu, lsq, obsFlag, blref, ccor)
                    if (success == False):
                        continue

                    mcor = lsq[0][1]
                    #Centroided position in dummyFlux, dummyWave arrays
                    currDummy = blref+xoffGuess-mcor+searchbox
                    #Centroid line for subpixel accuracy
                    #Square data to ensure bright line dominates fit
                    refCut = resid[blref-10:blref+11]**2
                    p = zeros(4, dtype=float64)
                    p[0] = max(refCut)
                    p[1] = 10
                    p[2] = gaussWidth/sqrt(2)
                    p[3] = gpu_arraymedian(refCut)
                    try:
                        lsq = leastsq(gaussResiduals, p, args=(arange(len(refCut), dtype=float64), refCut))
                    except Exception as ex:
                        #Flag and continue
                        obsFlag[blref-1:blref+2] = 0
                        continue
                    #Actual centroid of line in image, in pixel space
                    currLine = blref+lsq[0][1]-10
                    #Add line to obsSpec
                    p = zeros(4)
                    p[0] = sqrt(abs(lsq[0][0]))
                    p[1] = currLine
                    p[2] = abs(lsq[0][2]*sqrt(2))
                    #Common sense check of paramaters
                    if (p[0] > 1.5*max(resid[blref-5:blref+6]) or p[2] > 5*gaussWidth):
                        #If width or height seems weird, use actual peak vaule and gaussWidth from 3 brightest lines
                        p[0] = max(resid[blref-5:blref+6])
                        p[1] = blref
                        p[2] = gaussWidth
                    obsSpec += gaussFunction(p, arange(len(obsSpec), dtype=float32))
                    #Get wavelength from masterWave array.  Use int(currDummy) as index
                    #and find wavelength from line list closest to wavelength of dummyWave at this index.
                    currWave =  masterWave[where(abs(masterWave-dummyWave[int(currDummy)]) == min(abs(masterWave-dummyWave[int(currDummy)])))][0]
                    diff = abs(reflines[refline] - currLine)

                    closestLinesIndices = argsort(abs(templines-blref))[:2]
                    #scale from closest 2 lines
                    slocal = (wlines[closestLinesIndices[1]]-wlines[closestLinesIndices[0]])/(reflines[closestLinesIndices[1]]-reflines[closestLinesIndices[0]])
                    delta = (wlines[refline]-currWave)/slocal - (wlines[refline]-currWave)/scale

                    success = self.checkFinalRejectionCriteria(fdu, mcor, refbox, diff, wlines, currWave, reflines, currLine, scale, delta)
                    if (success == False):
                        obsFlag[blref-1:blref+2] = 0
                        continue

                    #Reset inloop array
                    inloop = 0
                    #Append this line's pixel centroid to reflines and wavelength to wlines and its parameters to lineParams.
                    reflines.append(currLine)
                    wlines.append(currWave)
                    lineParams.append(p)
                    print("\tLine found ("+str(nlines)+"): pixel="+formatNum(currLine)+", wavelength="+formatNum(currWave)+" (sigma = "+formatNum(lsigma)+")")
                    self._log.writeLog(__name__, "Line ("+str(nlines)+"): pixel="+formatNum(currLine)+", wavelength="+formatNum(currWave)+" (sigma = "+formatNum(lsigma)+")", printCaller=False, tabLevel=1)
                    #Increment nlines
                    nlines+=1
                    #If peak in resid array is <= 2*min_intensity_pct (default 1%) of peak of brightest line, break out of loop
                    if (resid[xlo+5:xhi-4].max() <= maxPeak*min_intensity_pct*2):
                        findLines = False

                if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                    print("Expanding search box...")
                #Expand search range out incrementally until entire image reached
                npass = 0
                #oldnlines = nlines-1
                while (xlo > 25 or xhi < len(oned)-26):
                    findLines = True
                    inloop = 0
                    #Incrementally expand range by 50 pixels per pass
                    xlo = int(max(min(reflines)-(250+npass*50), 25))
                    xhi = int(min(max(reflines)+(250+npass*50), len(oned)-26))
                    npass+=1
                    #Within each pass, loop over current range until found all lines in area
                    while (findLines and inloop < 25):
                        #Refine wavelength scale
                        if (nlines > oldnlines):
                            #Use helper method refineWavelengthScale
                            (success, scale, dummySize, dummyWave, dummyFlux) = self.refineWavelengthScale(npass, nlines, nonlinear, reflines, wlines, scale, min_wavelength, max_wavelength, dummySize, min_lines_nonlinear, coeffs)
                            if (not success):
                                print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Unable to refine wavelength solution "+pass_name+fdu.getFullId())
                                self._log.writeLog(__name__, "Unable to refine wavelength solution "+pass_name+fdu.getFullId(), type=fatboyLog.WARNING)
                            #Add gaussians for each line in line list
                            dummyFlux = self.populateDummyFlux(masterFlux, masterWave, dummyWave, dummyFlux, scale, gaussWidth, wlines)
                            #Scale template
                            dummyFlux *= fluxScale
                        oldnlines = nlines

                        #Set up resid array = residuals of (oned - found lines) * obsFlag
                        resid = (oned-obsSpec)*obsFlag
                        #If peak in resid array is <= min_intensity_pct (default 0.5%) of peak of brightest line, break out of loop
                        #continue to next pass in expanding range
                        if (resid[xlo+5:xhi-4].max() <= maxPeak*min_intensity_pct):
                            findLines = False
                            break
                        blref = where(resid[xlo:xhi+1] == max(resid[xlo+5:xhi-4]))[0][0]+xlo
                        #Edge cases - ensure peak of line found
                        if (blref-xlo < 10):
                            blref = where(resid[blref-5:blref+1] == max(resid[blref-5:blref+1]))[0][0]+blref-5
                        if (xhi-blref < 10):
                            blref = where(resid[blref:blref+6] == max(resid[blref:blref+6]))[0][0]+blref
                        if (xlo == 25 and blref < 75):
                            resid[:20] = 0
                        elif (xhi == len(oned)-26 and blref > len(oned)-76):
                            resid[-20:] = 0
                        elif (blref-xlo < 30):
                            resid[blref-50:blref-30] = 0
                        elif (xhi-blref < 30):
                            resid[blref+30:blref+50] = 0
                        refbox = 25
                        searchbox = 25
                        templines = array(reflines)
                        #Find closest line in pixel space to this one to use for initial guesses
                        refline = where(abs(templines-blref) == min(abs(templines-blref)))[0][0]
                        #Find index of dummyWave closest to actual wavelength of ref line
                        refIdx = where(abs(dummyWave-wlines[refline]) == min(abs(dummyWave-wlines[refline])))[0][0]
                        #Guess at index offset between oned cut and dummyFlux cut
                        xoffGuess = int(refIdx-reflines[refline])
                        refCut = resid[blref-refbox:blref+refbox+1]
                        if (nonlinear):
                            waveguess = polyFunction(coeffs, blref, dummyOrder)-polyFunction(coeffs, reflines[refline], dummyOrder)+dummyWave[refIdx]
                            guessIdx = where(abs(dummyWave-waveguess) == min(abs(dummyWave-waveguess)))[0][0]
                            xoffGuess = guessIdx-blref
                        elif ((abs(reflines[refline]-blref) > 200 and nlines >= 8) or (abs(reflines[refline]-blref) > 100 and nlines >= 32)):
                            #Try 2nd order guess if > 200 pixels from closest matched line and at least 8 lines matched
                            p = zeros(3, dtype=float32)
                            p[1] = scale
                            lsq = leastsq(polyResiduals, p, args=(array(reflines), array(wlines), 2))
                            waveguess = polyFunction(lsq[0], blref, 2)-polyFunction(lsq[0], reflines[refline], 2)+dummyWave[refIdx]
                            #Find nearest line
                            guessIdx = where(abs(dummyWave-waveguess) == min(abs(dummyWave-waveguess)))[0][0]
                            xoffGuess = guessIdx-blref
                            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                print("2nd order XOFF guess", lsq[0], waveguess, guessIdx, xoffGuess)
                        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                            print("Line at ref pixel BLREF ", blref, "nearest line", reflines[refline], "wavelength=",wlines[refline])
                            print("\tguess at offset=", xoffGuess)

                        #Rejection criteria 0-6 now use helper method
                        (success, inloop, lsigma) = self.checkPrefitRejectionCriteria(fdu, blref, resid, reflines, wlines, obsFlag, inloop, refbox, searchbox, nlines, dummyWave, dummyFlux, min_threshold)
                        if (success == False):
                            continue

                        if (blref-searchbox+xoffGuess < 0):
                            #Too close to edge!  Blank out 3 pixels and try again
                            obsFlag[blref-1:blref+2] = 0
                            inloop+=1
                            continue

                        #Cross-correlate
                        dumCut = dummyFlux[blref-searchbox+xoffGuess:blref+searchbox+1+xoffGuess]
                        n = min(len(refCut),len(dumCut))
                        #Make sure refCut and dumCut are same length
                        refCut = refCut[:n]
                        dumCut = dumCut[:n]
                        ccor = correlate(refCut, dumCut, mode='same')
                        #if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        #  plt.plot(refCut, 'r')
                        #  plt.plot(dumCut, 'g')
                        #  plt.show()
                        #  plt.plot(ccor)
                        #  plt.show()
                        #Fit cross correlation function with a Gaussian
                        p = zeros(4, dtype=float64)
                        p[0] = max(ccor)
                        p[1] = where(ccor == max(ccor))[0][0]
                        p[2] = gaussWidth
                        p[3] = gpu_arraymedian(ccor)
                        llo = max(0, int(p[1]-5))
                        lhi = min(len(ccor), int(p[1]+6))
                        try:
                            lsq = leastsq(gaussResiduals, p, args=(arange(lhi-llo, dtype=float64)+llo, ccor[llo:lhi]))
                        except Exception as ex:
                            #Flag and continue
                            obsFlag[blref-1:blref+2] = 0
                            continue
                        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                            print("\tleast squares fit to ccor pixel value=",lsq[0][1],"guess param=",p[1])
                        inloop += 1

                        success = self.checkPostfitRejectionCriteria(fdu, lsq, obsFlag, blref, ccor)
                        if (success == False):
                            continue

                        mcor = lsq[0][1]
                        #Centroided position in dummyFlux, dummyWave arrays
                        currDummy = blref+xoffGuess-mcor+searchbox
                        #Centroid line for subpixel accuracy
                        #Square data to ensure bright line dominates fit
                        refCut = resid[blref-10:blref+11]**2
                        p = zeros(4, dtype=float64)
                        p[0] = max(refCut)
                        p[1] = 10
                        p[2] = gaussWidth/sqrt(2)
                        p[3] = gpu_arraymedian(refCut)
                        try:
                            lsq = leastsq(gaussResiduals, p, args=(arange(len(refCut), dtype=float64), refCut))
                        except Exception as ex:
                            #Flag and continue
                            obsFlag[blref-1:blref+2] = 0
                            continue
                        currLine = blref+lsq[0][1]-10
                        #Add line to obsSpec
                        p = zeros(4)
                        p[0] = sqrt(abs(lsq[0][0]))
                        p[1] = currLine
                        p[2] = abs(lsq[0][2]*sqrt(2))
                        #Common sense check of paramaters
                        if (p[0] > 1.5*max(resid[blref-5:blref+6]) or p[2] > 5*gaussWidth):
                            #If width or height seems weird, use actual peak vaule and gaussWidth from 3 brightest lines
                            p[0] = max(resid[blref-5:blref+6])
                            p[1] = blref
                            p[2] = gaussWidth
                        obsSpec += gaussFunction(p, arange(len(obsSpec))+0.)
                        if (int(currDummy-1) < 0 or int(currDummy+1) > len(dummyWave)):
                            #Out of range!
                            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                                print("FAIL X - out of range")
                            continue
                        #Get wavelength from masterWave array.  Use int(currDummy) as index
                        #and find wavelength from line list closest to wavelength of dummyWave at this index.
                        currWave =  masterWave[where(abs(masterWave-dummyWave[int(currDummy)]) == min(abs(masterWave-dummyWave[int(currDummy)])))][0]
                        diff = abs(reflines[refline] - currLine)

                        closestLinesIndices = argsort(abs(templines-blref))[:2]
                        #scale from closest 2 lines
                        slocal = (wlines[closestLinesIndices[1]]-wlines[closestLinesIndices[0]])/(reflines[closestLinesIndices[1]]-reflines[closestLinesIndices[0]])
                        delta = (wlines[refline]-currWave)/slocal - (wlines[refline]-currWave)/scale

                        success = self.checkFinalRejectionCriteria(fdu, mcor, refbox, diff, wlines, currWave, reflines, currLine, scale, delta)
                        if (success == False):
                            obsFlag[blref-1:blref+2] = 0
                            continue

                        #Reset inloop array
                        inloop = 0
                        #Append this line's pixel centroid to reflines and wavelength to wlines and its parameters to lineParams.
                        reflines.append(currLine)
                        wlines.append(currWave)
                        lineParams.append(p)
                        print("\tLine found ("+str(nlines)+"): pixel="+formatNum(currLine)+", wavelength="+formatNum(currWave)+" (sigma = "+formatNum(lsigma)+")")
                        self._log.writeLog(__name__, "Line ("+str(nlines)+"): pixel="+formatNum(currLine)+", wavelength="+formatNum(currWave)+" (sigma = "+formatNum(lsigma)+")", printCaller=False, tabLevel=1)
                        #Increment nlines
                        nlines+=1
                        #If peak in resid array is <= min_intensity_pct (default 0.5%) of peak of brightest line, break out of loop
                        if (resid[xlo+5:xhi-4].max() <= maxPeak*min_intensity_pct):
                            findLines = False

                #Fit polynomial to find wavelength solution
                #Convert lists to arrays
                reflines = array(reflines)
                wlines = array(wlines)
                lineParams = array(lineParams)
                #Throw out flagged lines to not use in fit
                flags = zeros(len(wlines))
                for i in range(len(wlines)):
                    b = where(masterWave == wlines[i])
                    #0 = use for fit, nonzero = don't use for fit
                    flags[i] = masterFlag[b[0][0]]
                if ((flags == 0).sum() > 3):
                    nflagged = (flags != 0).sum()
                    print("\tThrowing out "+str(nflagged)+" flagged lines.")
                    self._log.writeLog(__name__, "Throwing out "+str(nflagged)+" flagged lines.", printCaller=False, tabLevel=1)
                    #good = lines to use in fit, subscript arrays
                    good = (flags == 0)
                    reflines = reflines[good]
                    wlines = wlines[good]
                    lineParams = lineParams[good]
                else:
                    print("\tWarning: Could not throw out flagged lines because not enough lines would remain to perform fit.")
                    self._log.writeLog(__name__, "Could not throw out flagged lines because not enough lines would remain to perform fit.", type=fatboyLog.WARNING, printCaller=False, tabLevel=1)

                #Fit polynomial of order fit_order to lines
                p = zeros(fit_order+1, dtype=float32)
                p[1] = scale
                if (len(reflines) <= fit_order):
                    print("\tWarning: Only found "+str(len(reflines))+" lines.  Using fit order = 1.")
                    self._log.writeLog(__name__, "Only found "+str(len(reflines))+" lines.  Using fit order = 1.", type=fatboyLog.WARNING, printCaller=False, tabLevel=1)
                    fit_order = 1
                    p = p[:2]
                lsq = leastsq(polyResiduals, p, args=(reflines, wlines, fit_order))
                #Calculate residuals
                residLines = polyFunction(lsq[0], reflines, fit_order)-wlines
                print("\t\tFound "+str(len(reflines))+" datapoints.  Fit: "+formatList(lsq[0]))
                print("\t\tData - fit mean: "+formatNum(residLines.mean())+"\tsigma: "+formatNum(residLines.std()))
                self._log.writeLog(__name__, "Found "+str(len(reflines))+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
                self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residLines.mean())+"\tsigma: "+formatNum(residLines.std()), printCaller=False, tabLevel=2)

                #Throw away outliers starting at 2 sigma significance
                sigThresh = 2
                niter = 0
                norig = len(reflines)
                bad = where(abs(residLines-residLines.mean())/residLines.std() > sigThresh)
                print("\t\tPerforming iterative sigma clipping to throw away outliers...")
                self._log.writeLog(__name__, "Performing iterative sigma clipping to throw away outliers...", printCaller=False, tabLevel=2)
                #Iterative sigma clipping
                while (len(bad[0]) > 0):
                    niter += 1
                    good = where(abs(residLines-residLines.mean())/residLines.std() <= sigThresh)
                    if (len(good[0]) < fit_order):
                        break
                    reflines = reflines[good]
                    wlines = wlines[good]
                    lineParams = lineParams[good]
                    #Refit, use last actual fit coordinates as input guess
                    p = lsq[0]
                    try:
                        lastLsq = lsq
                        lsq = leastsq(polyResiduals, p, args=(reflines, wlines, fit_order))
                    except Exception as ex:
                        lsq = lastLsq
                        break
                    #Calculate residuals
                    residLines = polyFunction(lsq[0], reflines, fit_order)-wlines
                    if (niter > 2):
                        #Gradually increase sigma threshold
                        sigThresh += 0.2
                    bad = where(abs(residLines-residLines.mean())/residLines.std() > sigThresh)
                print("\t\tAfter "+str(niter)+" passes, kept "+str(len(reflines))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]))
                print("\t\tData - fit mean: "+formatNum(residLines.mean())+"\tsigma: "+formatNum(residLines.std()))
                self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(reflines))+" of "+str(norig)+" datapoints.  Fit: "+formatList(lsq[0]), printCaller=False, tabLevel=2)
                self._log.writeLog(__name__, "Data - fit mean: "+formatNum(residLines.mean())+"\tsigma: "+formatNum(residLines.std()), printCaller=False, tabLevel=2)

                #Output "spec" file
                xs = polyFunction(lsq[0], arange(len(oned), dtype=float32), fit_order)
                dummyFlux = zeros(len(oned), dtype=float32)
                #Add gaussians for each line in line list
                dummyFlux = self.populateDummyFlux(masterFlux, masterWave, xs, dummyFlux, lsq[0][1], gaussWidth)
                #Scale template
                dummyFlux *= fluxScale
                f = open(specfile,'w')
                f.write("#lambda\t1-d cut\tref\tfit\n")
                for i in range(len(oned)):
                    f.write(str(xs[i])+'\t'+str(oned[i])+'\t'+str(dummyFlux[i])+'\t'+str(obsSpec[i])+'\n')
                f.close()
                #Output "resid" file with reference wavelength and residuals
                f = open(residfile,'w')
                f.write("#Ref wavelength\tresidual\n")
                for i in range(len(wlines)):
                    f.write(str(wlines[i])+'\t\t'+str(residLines[i])+'\n')
                f.close()

                #Update qadata data to show where lines were found
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    slit[:5,:] = 0.
                    slit[-5:,:] = 0.
                    for i in range(len(reflines)):
                        slit[:5,:] += gaussFunction(lineParams[i], arange(len(obsSpec), dtype=float32))
                        slit[-5:,:] += gaussFunction(lineParams[i], arange(len(obsSpec), dtype=float32))
                    if (currMask is None):
                        qadata[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi] = slit
                    else:
                        qadata[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi][currMask] = slit[currMask]
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    slit[:,:5] = 0.
                    slit[:,-5:] = 0.
                    for i in range(len(reflines)):
                        slit[:,:5] += gaussFunction(lineParams[i], arange(len(obsSpec), dtype=float32).reshape((len(obsSpec), 1)))
                        slit[:,-5:] += gaussFunction(lineParams[i], arange(len(obsSpec), dtype=float32).reshape((len(obsSpec), 1)))
                    if (currMask is None):
                        qadata[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)] = slit
                    else:
                        qadata[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)][currMask] = slit[currMask]

                if (usePlot and (self.getOption("debug_mode", fdu.getTag()).lower() == "yes" or self.getOption("write_plots", fdu.getTag()).lower() == "yes")):
                    plt.plot(oned, '#1f77b4', linewidth=2.0)
                    plt.plot(dummyFlux, '#ff7f0e', linewidth=2.0)
                    plt.plot(obsSpec, '#2ca02c', linewidth=2.0)
                    plt.legend(['Data', 'Line list', 'Fit'], loc=2)
                    plt.xlabel('Pixel')
                    plt.ylabel('Flux')
                    if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
                        pltfile = outdir+"/wavelengthCalibrated/qa_"+skyFDU._id+"_slit_"+str(j+1)
                        if (mult_seg):
                            pltfile += "_seg_"+str(seg+1)
                        pltfile += ".png"
                        plt.savefig(pltfile, dpi=200)
                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        plt.show()
                    plt.close()

                #Update header info
                if ((j == 0 and seg == 0) or minLambda is None):
                    #Update CTYPE on first pass
                    wcHeader['CTYPE'] = 'WAVE-PLY'
                    if (scale > 0):
                        minLambda = polyFunction(lsq[0], 0, fit_order)
                        maxLambda = polyFunction(lsq[0], xstride-1, fit_order)
                    else:
                        maxLambda = polyFunction(lsq[0], 0, fit_order)
                        minLambda = polyFunction(lsq[0], xstride-1, fit_order)
                    minLambdaList.append(minLambda)
                    maxLambdaList.append(maxLambda)
                else:
                    if (scale > 0):
                        localMinLambda = polyFunction(lsq[0], 0, fit_order)
                        localMaxLambda = polyFunction(lsq[0], xstride-1, fit_order)
                    else:
                        localMaxLambda = polyFunction(lsq[0], 0, fit_order)
                        localMinLambda = polyFunction(lsq[0], xstride-1, fit_order)
                    minLambda = min(minLambda, localMinLambda)
                    maxLambda = max(maxLambda, localMaxLambda)
                    minLambdaList.append(localMinLambda)
                    maxLambdaList.append(localMaxLambda)

                if (nslits == 1 and not mult_seg):
                    #If only one slit, use PORDER and PCOEFF_i
                    wcHeader['PORDER'] = fit_order
                    wcHeader['NSEG_01'] = 1
                    for i in range(fit_order+1):
                        wcHeader['PCOEFF_'+str(i)] = lsq[0][i]
                else:
                    #Use PORDERxx and PCFi_Sxx
                    slitStr = str(j+1)
                    if (j+1 < 10):
                        slitStr = '0'+slitStr
                    wcHeader['NSEG_'+slitStr] = n_segments
                    if (not mult_seg):
                        wcHeader['PORDER'+slitStr] = fit_order
                        for i in range(fit_order+1):
                            wcHeader['PCF'+str(i)+'_S'+slitStr] = lsq[0][i]
                    else:
                        #Multiple segments, use hierarchical keywords PORDER_xx_SEGy and PCFi_Sxx_SEGy
                        slitStr += '_SEG' + str(seg)
                        wcHeader['HIERARCH PORDER'+slitStr] = fit_order
                        for i in range(fit_order+1):
                            wcHeader['HIERARCH PCF'+str(i)+'_S'+slitStr] = lsq[0][i]
                #Append fit params to list
                fitParams[j].append(lsq[0])
                #Append qa params - slitlet, segment, norig, n lines used in fit, sigma
                #Format as string
                qaParams.append(str(j+1)+"\t"+str(seg+1)+"\t"+str(norig)+"\t"+str(len(reflines))+"\t"+formatNum(residLines.std()))

        #Update header
        fdu.setProperty("wcHeader", wcHeader)
        skyFDU.setProperty("wcHeader", wcHeader)
        #fdu.updateHeader(wcHeader)
        ##8/6/18 use updateHeader for sky/lamp
        skyFDU.updateHeader(wcHeader)
        #Write out qadata
        qafile = outdir+"/wavelengthCalibrated/qa_"+skyFDU.getFullId()
        #Remove existing files if overwrite = yes
        if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            if (os.access(qafile, os.F_OK)):
                os.unlink(qafile)
        #Write out qa file
        if (not os.access(qafile, os.F_OK)):
            skyFDU.tagDataAs("wcqa", qadata)
            skyFDU.writeTo(qafile, tag="wcqa")
            skyFDU.removeProperty("wcqa")
        del qadata
        #Output qa file with stats about each slitlet/segment
        qafile = outdir+"/wavelengthCalibrated/qa_"+skyFDU._id+".dat"
        f = open(qafile,'w')
        f.write("Slitlet\tSegment\tn lines\tn used\tsigma\n")
        for i in range(len(qaParams)):
            f.write(qaParams[i]+"\n")
        f.close()

        #Resample data for both master lamp / clean sky and FDU
        #Calculate ylo, yhi for FDU if different from skyFDU
        if (doIndividualSlitlets):
            if (fdu.hasProperty("slitmask")):
                if (fdu.hasProperty("regions")):
                    (ylos_data, yhis_data, slitx_data, slitw_data) = fdu.getProperty("regions")
                else:
                    #Use helper method to all ylo, yhi for each slit in each frame
                    (ylos_data, yhis_data, slitx_data, slitw_data) = findRegions(fdu.getSlitmask().getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                    fdu.setProperty("regions", (ylos_data, yhis_data, slitx_data, slitw_data))
            else:
                ylos_data = ylos
                yhis_data = yhis
        else:
            #doIndividualSlitlets = False - treat MOS data as 1 slit
            nslits = 1
            if (fdu.hasProperty("slitmask")):
                b = where(fdu.getSlitmask().getData() != 0)
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    ylos_data = [b[0].min()]
                    yhis_data = [b[0].max()]
                else:
                    ylos_data = [b[1].min()]
                    yhis_data = [b[1].max()]
            else:
                ylos_data = ylos
                yhis_data = yhis

        if (minLambda is None):
            #If minLambda is None here than no slits were wavlength calibrated
            print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not calibrate any slitlets for "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not calibrate any slitlets for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return

        #Output x-pixels
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            sky_xsize = skyFDU.getShape()[1]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            sky_xsize = skyFDU.getShape()[0]
        xout = zeros(skyFDU.getShape(), dtype=float32)
        xout_data = zeros(fdu.getShape(), dtype=float32)
        #Linear wavelength scale
        #scale = (maxLambda-minLambda)/sky_xsize
        #scale_data = (maxLambda-minLambda)/xsize
        scale_data = scale
        xs = arange(sky_xsize, dtype=float32)
        xs_data = arange(xsize, dtype=float32)
        #New header keywords for resampled data
        resampHeader = dict()
        resampHeader['RESAMPLD'] = 'YES'
        resampHeader['CRVAL1'] = minLambda
        if (scale < 0):
            resampHeader['CRVAL1'] = maxLambda
        resampHeader['CDELT1'] = scale
        resampHeader['CRPIX1'] = 1

        for j in range(nslits):
            if (useWCfile):
                n_segments = self.getWCParam(wcinfo, 'n_segments', j)
            #mult_seg = True if multiple segments
            mult_seg = (n_segments > 1)

            if (not resampleCommonScale):
                #Use min/max wavelength from this input slitlet for all segments
                maxLambda = max(maxLambdaList[j*n_segments:(j+1)*n_segments])
                minLambda = min(minLambdaList[j*n_segments:(j+1)*n_segments])
                scale = (maxLambda-minLambda)/(float)(sky_xsize)
                scale_data = (maxLambda-minLambda)/(float)(xsize)

            #Now loop over segments.  If n_segments == 1, this loop will only be done once (most data)
            for seg in range(n_segments):
                pass_name = " for order "+str(j+1)+" of "
                if (mult_seg):
                    pass_name = " for order "+str(j+1)+", segment "+str(seg+1)+" of "
                if (len(fitParams[j][seg]) == 0):
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> Warning: Could not fit wavelength solution"+pass_name+fdu.getFullId()+". Skipping.")
                    self._log.writeLog(__name__, "Could not fit wavelength solution"+pass_name+fdu.getFullId()+". Skipping.", type=fatboyLog.WARNING)
                    #WC failed!
                    continue

                xstride = sky_xsize//n_segments
                xstride_data = xsize//n_segments
                sxlo = xstride*seg
                sxhi = xstride*(seg+1)
                sxlo_data = xstride_data*seg
                sxhi_data = xstride_data*(seg+1)
                xs = arange(xstride, dtype=float32)
                xs_data = arange(xstride_data, dtype=float32)

                #Calculate wavelength input scale
                lambdaIn = polyFunction(fitParams[j][seg], xs, len(fitParams[j][seg])-1)
                lambdaIn_data = polyFunction(fitParams[j][seg], xs_data, len(fitParams[j][seg])-1)

                if (not resampleCommonScale):
                    #Use min/max wavelength from this input slitlet and segment
                    #maxLambda = lambdaIn.max()
                    #minLambda = lambdaIn.min()
                    #scale = (maxLambda-minLambda)/(float)(xstride)
                    #scale_data = (maxLambda-minLambda)/(float)(xstride_data)
                    #Use CRVALSxx and CDELTSxx for header keywords
                    slitStr = str(j+1)
                    if (j+1 < 10):
                        slitStr = '0'+slitStr
                    if (not mult_seg):
                        resampHeader['CRVALS'+slitStr] = minLambda
                        if (scale < 0):
                            resampHeader['CRVALS'+slitStr] = maxLambda
                        resampHeader['CDELTS'+slitStr] = scale
                    else:
                        #Multiple segments, use hierarchical keywords CRVALS_xx_SEGy and CDELTS_xx_SEGy
                        slitStr += '_SEG' + str(seg)
                        resampHeader['HIERARCH CRVALS'+slitStr] = minLambda
                        if (scale < 0):
                            resampHeader['HIERARCH CRVALS'+slitStr] = maxLambda
                        resampHeader['HIERARCH CDELTS'+slitStr] = scale

                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (slitmask is not None):
                        #MOS data - add to xout array, mutiply input by currMask
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        if (doIndividualSlitlets):
                            currMask = slitmask.getData()[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi] == (j+1)
                        else:
                            currMask = slitmask.getData()[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi] != 0
                        xout[int(ylos[j]):int(yhis[j]+1),sxlo:sxhi] += ((lambdaIn-minLambda)/scale)*currMask
                        if (fdu.hasProperty("slitmask")):
                            #Use this FDU's slitmask to calculate currMask for xout_data
                            #Apply mask to slit - based on if individual slitlets are being calibrated or not
                            if (doIndividualSlitlets):
                                currMask = fdu.getSlitmask().getData()[int(ylos_data[j]):int(yhis_data[j]+1),sxlo_data:sxhi_data] == (j+1)
                            else:
                                currMask = fdu.getSlitmask().getData()[int(ylos_data[j]):int(yhis_data[j]+1),sxlo_data:sxhi_data] != 0
                        xout_data[int(ylos_data[j]):int(yhis_data[j]+1),sxlo_data:sxhi_data] += ((lambdaIn_data-minLambda)/scale_data)*currMask
                    else:
                        #Longslit data, calculate entire xout frame at once
                        xout[:,sxlo:sxhi] = (lambdaIn-minLambda)/scale
                        xout_data[:,sxlo_data:sxhi_data] = (lambdaIn_data-minLambda)/scale_data
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    if (slitmask is not None):
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        if (doIndividualSlitlets):
                            currMask = slitmask.getData()[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)] == (j+1)
                        else:
                            currMask = slitmask.getData()[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)] != 0
                        #Use reshape to broadcast 1-d array to (n, 1) shape
                        xout[sxlo:sxhi,int(ylos[j]):int(yhis[j]+1)] += ((lambdaIn.reshape((len(lambdaIn), 1))-minLambda)/scale)*currMask
                        if (fdu.hasProperty("slitmask")):
                            #Use this FDU's slitmask to calculate currMask for xout_data
                            #Apply mask to slit - based on if individual slitlets are being calibrated or not
                            if (doIndividualSlitlets):
                                currMask = fdu.getSlitmask().getData()[sxlo_data:sxhi_data,int(ylos_data[j]):int(yhis_data[j]+1)] == (j+1)
                            else:
                                currMask = fdu.getSlitmask().getData()[sxlo_data:sxhi_data,int(ylos_data[j]):int(yhis_data[j]+1)] != 0
                        xout_data[sxlo_data:sxhi_data,int(ylos_data[j]):int(yhis_data[j]+1)] += ((lambdaIn_data.reshape((len(lambdaIn_data), 1))-minLambda)/scale)*currMask
                    else:
                        #Longslit data, calculate entire xout frame at once
                        xout[sxlo:sxhi,:] = (lambdaIn.reshape((len(lambdaIn), 1))-minLambda)/scale
                        xout_data[sxlo_data:sxhi_data,:] = (lambdaIn_data.reshape((len(lambdaIn_data), 1))-minLambda)/scale

        #Add header as property of FDU since it won't be written out until writeOutput
        fdu.setProperty("resampledHeader", resampHeader)
        skyFDU.setProperty("resampledHeader", resampHeader)
        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle
        #Resampled frames should be in cps not counts!
        hasResampledSlitmask = False
        smprops = dict()
        smprops["resampled"] = True
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            #Use drihizzle to resample with xtrans=xout_data
            inMask = (xout_data != 0).astype(int32)
            (resampData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits="counts", outunits="cps", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            #Save as data tag "resampled"
            expmap[expmap == 0] = 1
            fdu.tagDataAs("resampled", resampData)
            #Resample this FDU's slitmask
            if (fdu.hasProperty("slitmask")):
                (resampsm, header, expmap, pixmap) = drihizzle_method(fdu.getSlitmask(), None, None, inmask=inMask, kernel="uniform", xtrans=xout_data, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                fdu.setSlitmask(resampsm, pname=self._pname, tagname="resampled_slitmask", properties=smprops)
            #Use drihizzle to resample "sky" with xtrans=xout
            inMask = (xout != 0).astype(int32)
            (resampSky, header, expmap, pixmap) = drihizzle_method(skyFDU, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout, inunits="counts", outunits="cps", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            expmap[expmap == 0] = 1
            skyFDU.tagDataAs("resampled", resampSky)
            #Resample main slitmask
            if (slitmask is not None):
                (resampsm, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", xtrans=xout, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                resampSlitmask = self._fdb.addNewSlitmask(slitmask, resampsm, self._pname)
                resampSlitmask.updateHeader(header)
                hasResampledSlitmask = True
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #ytrans = xout_data
            inMask = (xout_data != 0).astype(int32)
            (resampData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits="counts", outunits="cps", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            expmap[expmap == 0] = 1
            fdu.tagDataAs("resampled", resampData)
            #Resample this FDU's slitmask
            if (fdu.hasProperty("slitmask")):
                (resampsm, header, expmap, pixmap) = drihizzle_method(fdu.getSlitmask(), None, None, inmask=inMask, dataTag="slitmask", kernel="uniform", ytrans=xout_data, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                fdu.setSlitmask(resampsm, pname=self._pname, tagname="resampled_slitmask", properties=smprops)
            #ytrans = xout
            inMask = (xout != 0).astype(int32)
            (resampSky, header, expmap, expmap) = drihizzle_method(skyFDU, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout, inunits="counts", outunits="cps", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            expmap[expmap == 0] = 1
            skyFDU.tagDataAs("resampled", resampSky)
            #Resample main slitmask
            if (slitmask is not None):
                (resampsm, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", ytrans=xout, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                resampSlitmask = self._fdb.addNewSlitmask(slitmask, resampsm, self._pname)
                resampSlitmask.updateHeader(header)
                hasResampledSlitmask = True

##TODO
        fdu.tagDataAs("xout_data", xout_data)
        fdu.writeTo(outdir+"/wavelengthCalibrated/xout_data_"+fdu.getFullId(), tag="xout_data")

        #Write out resampled sky/arclamp if requested
        if (writeCalibs):
            if (useArclamps):
                resampfile = outdir+"/wavelengthCalibrated/resamp_wc_"+calibs['masterLamp']._id+".fits"
                skyfile = outdir+"/wavelengthCalibrated/wc_"+calibs['masterLamp']._id+".fits"
            else:
                resampfile = outdir+"/wavelengthCalibrated/resamp_wc_"+calibs['cleanSky']._id+".fits"
                skyfile = outdir+"/wavelengthCalibrated/wc_"+calibs['cleanSky']._id+".fits"
            #Remove existing files if overwrite = yes
            if (os.access(skyfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(skyfile)
            if (os.access(resampfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(resampfile)
            #Write to disk
            if (not os.access(skyfile, os.F_OK)):
                skyFDU.writeTo(skyfile, headerExt=fdu.getProperty("wcHeader"))
            if (not os.access(resampfile, os.F_OK)):
                #Write with resampHeader as header extension
                skyFDU.writeTo(resampfile, tag="resampled", headerExt=resampHeader)
            #slitmasks
            if (slitmask is not None):
                if (hasResampledSlitmask):
                    resampfile = outdir+"/wavelengthCalibrated/resamp_wc_"+slitmask._id+".fits"
                    if (os.access(resampfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(resampfile)
                    #Write to disk
                    if (not os.access(resampfile, os.F_OK)):
                        resampSlitmask.writeTo(resampfile, headerExt=resampHeader)
            if (fdu.hasProperty("resampled_slitmask")):
                resampfile =  outdir+"/wavelengthCalibrated/resamp_slitmask_"+fdu.getFullId()
                if (os.access(resampfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(resampfile)
                #Write to disk
                if (not os.access(resampfile, os.F_OK)):
                    fdu.getProperty("resampled_slitmask").writeTo(resampfile, headerExt=resampHeader)
    #end wavelengthCalibrate

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/wavelengthCalibrated", os.F_OK)):
            os.mkdir(outdir+"/wavelengthCalibrated",0o755)
        #Create output filename
        wcfile = outdir+"/wavelengthCalibrated/wc_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(wcfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(wcfile)
        if (not os.access(wcfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(wcfile, headerExt=fdu.getProperty("wcHeader"))
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/wavelengthCalibrated/clean_wc_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write out exposure map if it exists
        if (fdu.hasProperty("exposure_map")):
            expfile = outdir+"/wavelengthCalibrated/exp_wc_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(expfile)
            if (not os.access(expfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(expfile, tag="exposure_map", headerExt=fdu.getProperty("wcHeader"))
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/wavelengthCalibrated/NM_wc_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap", headerExt=fdu.getProperty("wcHeader"))
        #Write out resampled data if it exists
        if (fdu.hasProperty("resampled")):
            resampfile = outdir+"/wavelengthCalibrated/resamp_wc_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(resampfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(resampfile)
            if (not os.access(resampfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Write with resampHeader as header extension
                fdu.writeTo(resampfile, tag="resampled", headerExt=fdu.getProperty("resampledHeader"))
    #end writeOutput
