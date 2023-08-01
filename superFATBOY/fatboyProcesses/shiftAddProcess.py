from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyLibs import *
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from numpy import *
import os, time

block_size = 512

class shiftAddProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    # Actually perform shifting and adding
    def shiftAddImage(self, fdu, calibs):
        #Get options
        blankRows = int(self.getOption("output_rows_between_slitlets", fdu.getTag()))
        mosUseWholeChip = False
        if (self.getOption("mos_use_whole_chip", fdu.getTag()).lower() == "yes"):
            mosUseWholeChip = True
        if (blankRows < 0 or blankRows > 100):
            print("shiftAddProcess::shiftAddImage> Warning: output_rows_between_slitlets of "+str(blankRows)+" is out of valid bounds (0-100).  Using 3.")
            self._log.writeLog(__name__, "output_rows_between_slitlets of "+str(blankRows)+" is out of valid bounds (0-100).  Using 3.", type=fatboyLog.WARNING)
            blankRows = 3

        t = time.time()
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        for frame in calibs['frameList']:
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                xsize = max(xsize, frame.getShape()[1])
                ysize = max(ysize, frame.getShape()[0])
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                ##xsize should be size across dispersion direction
                xsize = max(xsize, frame.getShape()[0])
                ysize = max(ysize, frame.getShape()[1])

        shifts = calibs['shifts']
        maxShift = max(shifts)
        minShift = min(shifts)

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and 'slitmask' in calibs and not mosUseWholeChip):
            ###MOS/IFU data -- handle slitlet by slitlet
            nslits = calibs['nslits']
            nframes = len(calibs['frameList'])
            #inylo, inyhi = 2-d lists nframes x nslits with lowest, highest y values in slitlet for each frame
            inylo = []
            inyhi = []
            #slitylo, slityhi = 1-d lists, with nslits elements, lowest, highest y values for each slit in output frame
            slitylo = []
            slityhi = []
            #Calculate all ylo, yhi for each slit in each frame and assign to inylo, inyhi
            for frame in calibs['frameList']:
                #Use new fdu.getSlitmask method
                slitmask = frame.getSlitmask(pname=None, properties=properties)
                #Use helper method to all ylo, yhi for each slit in each frame
                (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                inylo.append(ylos)
                inyhi.append(yhis)
            #outylo = 2-d array nframes x nslits with y-pos in output image corresponding to inylo for each slit in each input image
            #e.g., for image i and slit s,
            #output[outylo[i][s]:outylo[i][s]+slitheight] += input[inylo[i][s]:inyhi[i][s]]
            outylo = zeros((nframes, nslits), int32)
            #Loop over slits and calculate outylo, slitylo, slityhi.  3 pixels between slitlets in output.
            for i in range(nslits):
                if (i == 0):
                    slitylo.append(0)
                else:
                    #3 pixels between slitlets in output
                    slitylo.append(slityhi[i-1]+blankRows+1)
                #Find size of output slitlet
                miny = inylo[0][i]+shifts[0]
                maxy = inyhi[0][i]+shifts[0]
                for j in range(1, nframes):
                    miny = min(miny, inylo[j][i]+shifts[j])
                    maxy = max(maxy, inyhi[j][i]+shifts[j])
                slityhi.append(slitylo[i]+maxy-miny)
                for j in range(nframes):
                    outylo[j][i] = slitylo[i]+inylo[j][i]+shifts[j]-miny
            #new y-size
            outysize = slityhi[-1]+1

            #Create new image sizes
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                data = zeros((outysize, xsize), dtype=float32)
                outmask = zeros((outysize, xsize), dtype=int32)
                if (fdu.hasProperty("exposure_map")):
                    expmap = zeros((outysize, xsize), dtype=float32)
                if (fdu.hasProperty("cleanFrame")):
                    cleanFrame = zeros((outysize, xsize), dtype=float32)
                if (fdu.hasProperty("noisemap")):
                    nm = zeros((outysize, xsize), dtype=float32)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                data = zeros((xsize, outysize), dtype=float32)
                outmask = zeros((xsize, outysize), dtype=int32)
                if (fdu.hasProperty("exposure_map")):
                    expmap = zeros((xsize, outysize), dtype=float32)
                if (fdu.hasProperty("cleanFrame")):
                    cleanFrame = zeros((xsize, outysize), dtype=float32)
                if (fdu.hasProperty("noisemap")):
                    nm = zeros((xsize, outysize), dtype=float32)

            #Loop over images, then slitlets within each image
            for j in range(len(calibs['frameList'])):
                currFDU = calibs['frameList'][j]
                currShift = shifts[j]-minShift
                #Use new fdu.getSlitmask method
                slitmask = currFDU.getSlitmask(pname=None, properties=properties)
                #Loop over slits
                for i in range(nslits):
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        #Use currMask to zero out anything not in current slitlet
                        currMask = slitmask.getData()[inylo[j][i]:inyhi[j][i]+1,:] == (i+1)
                        #Shift and add data, slitmask, exposure_map, cleanFrame, noisemap
                        data[outylo[j][i]:outylo[j][i]+currMask.shape[0], :] += currFDU.getData()[inylo[j][i]:inyhi[j][i]+1,:] * currMask
                        #Add so that we don't zero out any points from previous image slitmask
                        outmask[outylo[j][i]:outylo[j][i]+currMask.shape[0], :] += (i+1) * currMask
                        #Then correct so that pixels where multiple images contribute are set to (i+1)
                        b = outmask[outylo[j][i]:outylo[j][i]+currMask.shape[0], :] > (i+1)
                        outmask[outylo[j][i]:outylo[j][i]+currMask.shape[0], :][b] = i+1
                        if (fdu.hasProperty("exposure_map") and currFDU.hasProperty("exposure_map")):
                            expmap[outylo[j][i]:outylo[j][i]+currMask.shape[0], :] += currFDU.getData(tag="exposure_map")[inylo[j][i]:inyhi[j][i]+1,:] * currMask
                        if (fdu.hasProperty("cleanFrame") and currFDU.hasProperty("cleanFrame")):
                            cleanFrame[outylo[j][i]:outylo[j][i]+currMask.shape[0], :] += currFDU.getData(tag="cleanFrame")[inylo[j][i]:inyhi[j][i]+1,:] * currMask
                        if (fdu.hasProperty("noisemap") and currFDU.hasProperty("noisemap")):
                            nm[outylo[j][i]:outylo[j][i]+currMask.shape[0], :] += (currFDU.getData(tag="noisemap")**2)[inylo[j][i]:inyhi[j][i]+1,:] * currMask
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        #Use currMask to zero out anything not in current slitlet
                        currMask = slitmask.getData()[:,inylo[j][i]:inyhi[j][i]+1] == (i+1)
                        #Shift and add data, slitmask, exposure_map, cleanFrame, noisemap
                        data[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]] += currFDU.getData()[:,inylo[j][i]:inyhi[j][i]+1] * currMask
                        #Add so that we don't zero out any points from previous image slitmask
                        outmask[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]] += (i+1) * currMask
                        #Then correct so that pixels where multiple images contribute are set to (i+1)
                        b = outmask[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]] > (i+1)
                        outmask[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]][b] = i+1
                        if (fdu.hasProperty("exposure_map") and currFDU.hasProperty("exposure_map")):
                            expmap[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]] += currFDU.getData(tag="exposure_map")[:,inylo[j][i]:inyhi[j][i]+1] * currMask
                        if (fdu.hasProperty("cleanFrame") and currFDU.hasProperty("cleanFrame")):
                            cleanFrame[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]] += currFDU.getData(tag="cleanFrame")[:,inylo[j][i]:inyhi[j][i]+1] * currMask
                        if (fdu.hasProperty("noisemap") and currFDU.hasProperty("noisemap")):
                            nm[:, outylo[j][i]:outylo[j][i]+currMask.shape[1]] += (currFDU.getData(tag="noisemap")**2)[:,inylo[j][i]:inyhi[j][i]+1] * currMask

            #Get region info
            if ("regions" in calibs):
                (sylo, syhi, slitx, slitw) = calibs["regions"]
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (sylo, syhi, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)

            if (len(slitx) != nslits):
                #Attempt to match up correct slitlets
                newslitx = []
                for i in range(nslits):
                    b = where(abs(sylo-slitylo[i]) == min(abs(sylo-slitylo[i])))[0]
                    newslitx.append(slitx[b[0]])
                slitx = newslitx
            #Set "regions" property in fdu
            fdu.setProperty("regions", (slitylo, slityhi, slitx, slitw))
            #Update property "slitmask" with outmask
            #Use new fdu.setSlitmask
            fdu.setSlitmask(outmask, pname=self._pname)

            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/shiftAdded", os.F_OK)):
                os.mkdir(outdir+"/shiftAdded",0o755)

            #Output new region file
            saRegFile = outdir+"/shiftAdded/region_"+fdu._id+".reg"
            f = open(saRegFile,'w')
            f.write('# Region file format: DS9 version 3.0\n')
            f.write('global color=green select=1 edit=1 move=1 delete=1 include=1 fixed=0\n')
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                for i in range(len(slitylo)):
                    f.write('image;box('+str(slitx[i]+1)+','+str((slitylo[i]+slityhi[i])//2+1)+',3,'+str(slityhi[i]-slitylo[i])+')\n')
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                for i in range(len(slitylo)):
                    f.write('image;box('+str((slitylo[i]+slityhi[i])//2+1)+','+str(slitx[i]+1)+','+str(slityhi[i]-slitylo[i])+',3)\n')
            f.close()

            #Output new XML region file
            saRegFile = outdir+"/shiftAdded/region_"+fdu._id+".xml"
            f = open(saRegFile,'w')
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<fatboy>\n')
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                for i in range(len(slitylo)):
                    f.write('<slitlet xcenter="'+str(slitx[i]+1)+'" ycenter="'+str((slitylo[i]+slityhi[i])//2+1)+'" width="3" height="'+str(slityhi[i]-slitylo[i])+'"/>\n')
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                for i in range(len(slitylo)):
                    f.write('<slitlet xcenter="'+str((slitylo[i]+slityhi[i])//2+1)+'" ycenter="'+str(slitx[i]+1)+'" width="'+str(slityhi[i]-slitylo[i])+'" height="3"/>\n')
            f.write('</fatboy>\n')
            f.close()
        else:
            #Longslit -- simple shift and add
            #Create new image sizes
            outysize = ysize+maxShift-minShift+1
            if (maxShift == 0 and minShift == 0):
                #Special case
                outysize = ysize
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                data = zeros((outysize, xsize), dtype=float32)
                if (fdu.hasProperty("exposure_map")):
                    expmap = zeros((outysize, xsize), dtype=float32)
                if (fdu.hasProperty("cleanFrame")):
                    cleanFrame = zeros((outysize, xsize), dtype=float32)
                if (fdu.hasProperty("noisemap")):
                    nm = zeros((outysize, xsize), dtype=float32)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                data = zeros((xsize, outysize), dtype=float32)
                if (fdu.hasProperty("exposure_map")):
                    expmap = zeros((xsize, outysize), dtype=float32)
                if (fdu.hasProperty("cleanFrame")):
                    cleanFrame = zeros((xsize, outysize), dtype=float32)
                if (fdu.hasProperty("noisemap")):
                    nm = zeros((xsize, outysize), dtype=float32)

            #Loop over images, shift and add all at once
            for j in range(len(calibs['frameList'])):
                currFDU = calibs['frameList'][j]
                currShift = shifts[j]-minShift
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    data[currShift:currShift+currFDU.getShape()[0],:] += currFDU.getData()
                    if (currFDU.hasProperty("exposure_map")):
                        expmap[currShift:currShift+currFDU.getShape()[0],:] += (currFDU.getData(tag="exposure_map")*(currFDU.getData() != 0))
                    if (currFDU.hasProperty("cleanFrame")):
                        cleanFrame[currShift:currShift+currFDU.getShape()[0],:] += currFDU.getData(tag="cleanFrame")
                    if (currFDU.hasProperty("noisemap")):
                        nm[currShift:currShift+currFDU.getShape()[0],:] += currFDU.getData(tag="noisemap")**2
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    data[:,currShift:currShift+currFDU.getShape()[1]] += currFDU.getData()
                    if (currFDU.hasProperty("exposure_map")):
                        expmap[:,currShift:currShift+currFDU.getShape()[1]] += (currFDU.getData(tag="exposure_map")*(currFDU.getData() != 0))
                    if (currFDU.hasProperty("cleanFrame")):
                        cleanFrame[:,currShift:currShift+currFDU.getShape()[1]] += currFDU.getData(tag="cleanFrame")
                    if (currFDU.hasProperty("noisemap")):
                        nm[:,currShift:currShift+currFDU.getShape()[1]] += currFDU.getData(tag="noisemap")**2
                fdu._header.add_history('Shift and added frame '+currFDU.getFullId()+' with shift '+str(currShift))


        if (fdu.hasProperty("exposure_map")):
            #Divide data by expmap
            #Select cpu/gpu option
            if (self._fdb.getGPUMode()):
                #Use GPU
                data = divideArraysFloatGPU(data, expmap)
            else:
                #Find nonzero points in expmap and divide these only
                b = expmap != 0
                data[b] /= expmap[b]

            #Divide "cleanFrame" by expmap
            if (fdu.hasProperty("cleanFrame")):
                #Select cpu/gpu option
                if (self._fdb.getGPUMode()):
                    #Use GPU
                    cleanFrame = divideArraysFloatGPU(cleanFrame, expmap)
                else:
                    #Find nonzero points in expmap and divide these only
                    b = expmap != 0
                    cleanFrame[b] /= expmap[b]

            #Update "exposure_map" data tag
            fdu.tagDataAs("exposure_map", data=expmap)
        if (fdu.hasProperty("cleanFrame")):
            #Update "cleanFrame" data tag
            fdu.tagDataAs("cleanFrame", data=cleanFrame)
        if (fdu.hasProperty("noisemap") and fdu.hasProperty("exposure_map")):
            #Take sqrt of noisemap and divide by expmap
            if (self._fdb.getGPUMode()):
                #Use GPU
                nm = noisemaps_sqrtAndDivide_float(nm, expmap)
            else:
                #take sqrt first
                nm = sqrt(nm)
                #Find nonzero points in expmap and divide these only
                b = expmap != 0
                nm[b] /= expmap[b]
            #Update "noisemap" data tag
            fdu.tagDataAs("noisemap", data=nm)
        fdu.updateData(data)
        print("Shift and add time: ",time.time()-t)
    #end shiftAddImage

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Shift Add")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For shiftAdd, this dict should have one entry 'masterFlat' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'shifts' in calibs):
            #Failed to obtain shift
            #Issue error message and disable this FDU
            print("shiftAddProcess::execute> ERROR: Shift for shifting and adding not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Shift for shifting and adding not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        safile = "shiftAdded/sa_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, safile)):
            #Also check if "cleanFrame" exists
            cleanfile = "shiftAdded/clean_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "exposure map" exists
            expfile = "shiftAdded/exp_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, expfile, tag="exposure_map")
            #Also check if "slitmask" exists
            smfile = "shiftAdded/slitmask_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, smfile, tag="slitmask")
            #Also check if "noisemap" exists
            nmfile = "shiftAdded/NM_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")

            #Disable all frames but first one
            for frame in calibs['frameList']:
                if (frame.getFullId() != fdu.getFullId()):
                    fdu.exptime += frame.exptime #Sum exptime
                    frame.disable()
            updateHeaderEntry(fdu._header, fdu._keywords['exptime_keyword'], fdu.exptime) #Use wrapper function to update header
            return True

        #call shiftAddImage helper function to do gpu/cpu shifting and adding
        self.shiftAddImage(fdu, calibs)

        #Disable all frames but first one
        for frame in calibs['frameList']:
            if (frame.getFullId() != fdu.getFullId()):
                fdu.exptime += frame.exptime #Sum exptime
                frame.disable()
        updateHeaderEntry(fdu._header, fdu._keywords['exptime_keyword'], fdu.exptime) #Use wrapper function to update header
        return True
    #end execute

    #find the shift between positive and negative continua
    def findShifts(self, calibs, fdu):
        #Get options
        manual = self.getOption("manual_shifts", fdu.getTag())
        if (manual is not None):
            nframes = len(calibs['frameList'])
            if (isInt(manual)):
                shifts = [int(manual)]*nframes
                return shifts
            elif (nframes.count(",") > 0):
                shifts = nframes.split(",")
                for j in range(len(shifts)):
                    shifts[j] = int(shifts[j].trim())
                return shifts
            elif (os.access(manual, os.F_OK)):
                shifts = readFileIntoList(manual)
                for j in range(len(shifts)):
                    shifts[j] = int(shifts[j].trim())
                return shifts
            else:
                print("shiftAddProcess::findShifts> Warning: Invalid value for manual: "+str(manual)+".  Finding shifts instead...")
                self._log.writeLog(__name__, "Invalid value for manual: "+str(manual)+".  Finding shifts instead...", type=fatboyLog.WARNING)

        xlo = int(self.getOption("find_shift_box_xlo", fdu.getTag()))
        xhi = int(self.getOption("find_shift_box_xhi", fdu.getTag()))
        ylo = int(self.getOption("find_shift_box_ylo", fdu.getTag()))
        yhi = int(self.getOption("find_shift_box_yhi", fdu.getTag()))
        constrain_boxsize = self.getOption("find_shift_constrain_boxsize", fdu.getTag())
        if (constrain_boxsize is not None):
            constrain_boxsize = int(constrain_boxsize)
        useHeader = False
        if (self.getOption("use_header", fdu.getTag()).lower() == "yes"):
            useHeader = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        #Take only positive datapoints
        pos = fdu.getData(tag="cleanFrame").copy()
        pos[pos < 0] = 0
        #Instead of taking median, sum so we get short spectra but do a
        #5 pixel boxcar median smoothing to get rid of hot pixels
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            refCut = mediansmooth1d(sum(pos[ylo:yhi,xlo:xhi],1), 5)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            refCut = mediansmooth1d(sum(pos[xlo:xhi,ylo:yhi],0), 5)

        shifts = []
        #Loop over all frames
        for frame in calibs['frameList']:
            if (frame == fdu):
                #Reference frame, shift = 0
                shifts.append(0)
                continue

            if (useHeader):
                #Use the RA, DEC, and pixscale
                sa_guess = int(sqrt((fdu.ra-frame.ra)**2+(fdu.dec-frame.dec)**2)*3600/fdu.pixscale+0.5) #round
                if ((fdu.ra-frame.ra+fdu.dec-frame.dec) < 0):
                    sa_guess = -1*sa_guess
                print("shiftAddProcess::findShifts> Using "+str(sa_guess)+" as calculated from header info for shift from "+frame.getFullId()+" to "+fdu.getFullId()+"...")
                self._log.writeLog(__name__, "Using "+str(sa_guess)+" as calculated from header info for shift from "+frame.getFullId()+" to "+fdu.getFullId()+"...")
                shifts.append(sa_guess)
                continue

            currPos = frame.getData(tag="cleanFrame").copy()
            currPos[currPos < 0] = 0
            #Instead of taking median, sum so we get short spectra but do a
            #5 pixel boxcar median smoothing to get rid of hot pixels
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                tempCut = mediansmooth1d(sum(currPos[ylo:yhi,xlo:xhi],1), 5)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                tempCut = mediansmooth1d(sum(currPos[xlo:xhi,ylo:yhi],0), 5)

            #Adjust for different "y"-sizes by padding
            if (len(refCut) < len(tempCut)):
                currCut = tempCut.copy()
                corrCut = currCut*0. #Set size of "correlation cut" to be same as current cut
                corrCut[:len(refCut)] = refCut
            else:
                corrCut = refCut.copy()
                currCut = corrCut*0.
                currCut[:len(tempCut)] = tempCut

            #Cross correlate "ref correlation cut" and "current cut"
            ccor = correlate(corrCut, currCut, mode='same')
            mcor = where(ccor == max(ccor))[0]
            if (constrain_boxsize is not None and fdu.pixscale is None):
                print("shiftAddProcess::findShifts> WARNING: PIXSCALE is not defined in FITS header.  Cannot use constrain box / initial guess.")
                self._log.writeLog(__name__, "PIXSCALE is not defined in FITS header.  Cannot use constrain box / initial guess.", type=fatboyLog.WARNING)
            if (constrain_boxsize is not None and fdu.pixscale is not None):
                #Constrain boxsize for shift
                sa_guess = sqrt((fdu.ra-frame.ra)**2+(fdu.dec-frame.dec)**2)*3600/fdu.pixscale
                print("shiftAddProcess::findShifts> Using initial guess "+str(sa_guess)+" pixels and boxsize "+str(constrain_boxsize)+" for "+frame.getFullId()+"...")
                self._log.writeLog(__name__, "Using initial guess "+str(sa_guess)+" pixels and boxsize "+str(constrain_boxsize)+" for "+fdu.getFullId()+"...")
                guess1 = int(len(ccor)//2-sa_guess)
                guess2 = int(len(ccor)//2+sa_guess)
                maxVal = max(ccor[guess1-constrain_boxsize//2:guess1+constrain_boxsize//2].max(), ccor[guess2-constrain_boxsize//2:guess2+constrain_boxsize//2].max())
                mcor = where(ccor == maxVal)[0]
            #offset = -1*(len(ccor)/2-mcor[0])
            shift = -1*(len(ccor)//2-mcor[0])
            print("shiftAddProcess::findShifts> Shift from "+fdu.getFullId()+" to "+frame.getFullId()+" = "+str(shift)+" pixels.")
            self._log.writeLog(__name__, "Shift from "+fdu.getFullId()+" to "+frame.getFullId()+" = "+str(shift)+" pixels.")
            shifts.append(shift)
        return shifts
    #end findShifts

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        mosUseWholeChip = False
        if (self.getOption("mos_use_whole_chip", fdu.getTag()).lower() == "yes"):
            mosUseWholeChip = True

        calibs = dict()

        #Look for each master calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("shiftAddProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("shiftAddProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs and not mosUseWholeChip):
            #Multi object data, need slitmask
            #Find slitmask associated with this fdu
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is None):
                print("shiftAddProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                return calibs
            calibs['slitmask'] = slitmask
            if (fdu.hasProperty("nslits")):
                calibs['nslits'] = fdu.getProperty("nslits")
            elif (slitmask.hasProperty("nslits")):
                calibs['nslits'] = slitmask.getProperty("nslits")
            else:
                calibs['nslits'] = calibs['slitmask'].getData().max()
                slitmask.setProperty("nslits", calibs['nslits'])
                fdu.setProperty("nslits", calibs['nslits'])
            if (fdu.hasProperty("regions")):
                calibs['regions'] = fdu.getProperty("regions")
            elif (slitmask.hasProperty("regions")):
                calibs['regions'] = slitmask.getProperty("regions")

        #Check for individual FDUs matching specmode/filter/grism/ident to shift and add
        #fdus can not be [] as it will always at least return the current FDU itself
        fdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (len(fdus) > 0):
            #Found other objects associated with this fdu.  Create shifted added image
            print("shiftAddProcess::getCalibs> Creating shifted added image for object "+fdu._id+"...")
            self._log.writeLog(__name__, "Creating aligned stacked image for object "+fdu._id+"...")
            #First recursively process
            self.recursivelyExecute(fdus, prevProc)
            #Loop over rctfdus and pop out any that have been disabled at sky subtraction stage by pairing up
            for j in range(len(fdus)-1, -1, -1):
                if (not fdus[j].inUse):
                    fdus.pop(j)
            #convenience method
            calibs['frameList'] = fdus

        calibs['shifts'] = self.findShifts(calibs, fdu)
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('find_shift_box_xlo', '0')
        self._optioninfo.setdefault('find_shift_box_xlo', 'Used to specify a range of the chip in dispersion direction to sum\na 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_box_xhi', '-1')
        self._optioninfo.setdefault('find_shift_box_xhi', 'Used to specify a range of the chip in dispersion direction to sum\na 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_box_ylo', '0')
        self._optioninfo.setdefault('find_shift_box_ylo', 'Used to specify a range of the chip in cross-dispersion\ndirection to sum a 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_box_yhi', '-1')
        self._optioninfo.setdefault('find_shift_box_yhi', 'Used to specify a range of the chip in cross-dispersion\ndirection to sum a 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_constrain_boxsize', None)
        self._optioninfo.setdefault('find_shift_constrain_boxsize', 'Constrain the fit to a box of this size, centered\nat the initial guess based on RA and Dec offsets.')
        self._options.setdefault('manual_shifts', None)
        self._optioninfo.setdefault('manual_shifts', 'Set to a number, comma separated list,\nor ASCII file to specify shifts')
        self._options.setdefault('mos_use_whole_chip', 'no')
        self._optioninfo.setdefault('mos_use_whole_chip', 'Set to yes to shift/add the whole chip\nrather than each individual slitlet.')
        self._options.setdefault('output_rows_between_slitlets','3')
        self._optioninfo.setdefault('output_rows_between_slitlets', 'Number of blank rows to space slitlets by in output.')
        self._options.setdefault('use_header', 'no')
        self._optioninfo.setdefault('use_header', 'Use the information in the header -\nRA, DEC, PIXSCALE - instead of\nattempting to find offsets.')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## update noisemap for spectroscopy data
    def updateNoisemap(self, fdu, masterFlat):
        if (not masterFlat.hasProperty("noisemap")):
            #Hopefully we don't get here because this means we are reading a previous masterFlat from disk with no corresponding noisemap on disk
            #If masterFlat is dome on, we're fine but if its on-off, we lose separate on and off data.
            #create tagged data "noisemap"
            #Create noisemap
            ncomb = 1.0
            if (masterFlat.hasHeaderValue('NCOMBINE')):
                ncomb = float(masterFlat.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterFlat.getData(), ncomb)
            else:
                nm = sqrt(masterFlat.getData()/ncomb)
            masterFlat.tagDataAs("noisemap", nm)
        #Get this FDU's noisemap
        nm = fdu.getData(tag="noisemap")
        #Propagate noisemaps.  For division dz/z = sqrt((dx/x)^2 + (dy/y)^2)
        if (self._fdb.getGPUMode()):
            #noisemaps_sa_gpu(sa_image, pre-sa_noisemap, pre-sa_image, mflat noisemap, mflat
            nm = noisemaps_sa_gpu(fdu.getData(), fdu.getData(tag="noisemap"), fdu.getData("cleanFrame"), masterFlat.getData("noisemap"), masterFlat.getData())
        else:
            nm = abs(fdu.getData())*sqrt(fdu.getData(tag="noisemap")**2/fdu.getData("cleanFrame")**2 + masterFlat.getData("noisemap")**2/masterFlat.getData()**2)
            nm[fdu.getData("cleanFrame") == 0] = 0
            nm[masterFlat.getData() == 0] = 0
        fdu.tagDataAs("noisemap", nm)
    #end updateNoisemap

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/shiftAdded", os.F_OK)):
            os.mkdir(outdir+"/shiftAdded",0o755)
        #Create output filename
        fdfile = outdir+"/shiftAdded/sa_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(fdfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(fdfile)
        if (not os.access(fdfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(fdfile)
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/shiftAdded/clean_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write out exposure map if it exists
        if (fdu.hasProperty("exposure_map")):
            expfile = outdir+"/shiftAdded/exp_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(expfile)
            if (not os.access(expfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(expfile, tag="exposure_map")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/shiftAdded/NM_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
        #Write out slitmask if it exists
        if (fdu.hasProperty("slitmask")):
            smfile = outdir+"/shiftAdded/slitmask_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(smfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(smfile)
            if (not os.access(smfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Use new fdu.getSlitmask to get FDU and then write
                fdu.getSlitmask().writeTo(smfile)
    #end writeOutput
