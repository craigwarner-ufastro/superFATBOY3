from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_imcombine, imcombine

block_size = 512

class skySubtractSpecProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    ssmethods = ["dither", "ifu_onsource_dither", "median", "median_boxcar", "offsource_dither", "offsource_multi_dither", "step"]
    lastIdent = None #Track last identifier for onsource skies
    fduCount = 0 #Track count of fdus within this identifier
    identTotal = 0 #Track number of frames for this identifier

    def applyResponseCurve(self, fdu, calibs, skyData, cleanSkyData):
        pass

    #Calculate a sky to subtract by using a median boxcar to filter each column (or row)
    def calculateMedianBoxcarSky(self, fdu, calibs, prevProc):
        boxcar_width = int(self.getOption('boxcar_width', fdu.getTag()))
        boxcar_nhigh = int(self.getOption('boxcar_nhigh', fdu.getTag()))
        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None
        currMask = ones(fdu.getShape(), dtype=bool)
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            ###MOS/IFU data -- get slitmask
            slitmask = self.findSlitmask(fdu, calibs, prevProc)
            if (slitmask is None):
                #FDU will be disabled in execute
                return None
            if (not slitmask.hasProperty("nslits")):
                slitmask.setProperty("nslits", slitmask.getData().max())
            nslits = slitmask.getProperty("nslits")
            if (slitmask.hasProperty("regions")):
                (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

        #Select kernel for 2d median
        medFilt2dFunc = medianfilter2dCPU
        if (self._fdb.getGPUMode()):
            #Use GPU for median filter
            medFilt2dFunc = gpumedianfilter2d

        filtData = zeros(fdu.getShape(), dtype=float32)
        for j in range(nslits):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (slitmask is not None):
                    currMask = (slitmask.getData()[ylos[j]:yhis[j]+1] == j+1)
                slit = (fdu.getData()[ylos[j]:yhis[j]+1]).copy()*currMask
                m = medFilt2dFunc(slit, axis="Y", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                filtData[ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                if (slitmask is not None):
                    currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == j+1)
                slit = (fdu.getData()[:,ylos[j]:yhis[j]+1]).copy()*currMask
                m = medFilt2dFunc(slit, axis="X", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                filtData[:,ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]


        skyData = fdu.getData()-filtData
        msname = "masterSkies/sky_"+fdu.getFullId()
        masterSky = fatboySpecCalib(self._pname, "master_sky", fdu, data=skyData, tagname=msname, log=self._log)

        if (fdu.hasProperty("cleanFrame")):
            #create "cleanFrame" for master sky too
            filtData = zeros(fdu.getShape(), dtype=float32)
            for j in range(nslits):
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (slitmask is not None):
                        currMask = (slitmask.getData()[ylos[j]:yhis[j]+1] == j+1)
                    slit = (fdu.getData(tag="cleanFrame")[ylos[j]:yhis[j]+1]).copy()*currMask
                    m = medFilt2dFunc(slit, axis="Y", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                    filtData[ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    if (slitmask is not None):
                        currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == j+1)
                    slit = (fdu.getData(tag="cleanFrame")[:,ylos[j]:yhis[j]+1]).copy()*currMask
                    m = medFilt2dFunc(slit, axis="X", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                    filtData[:,ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
            cleanSkyData = fdu.getData(tag="cleanFrame")-filtData
            masterSky.tagDataAs("cleanFrame", cleanSkyData)

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/masterSkies", os.F_OK)):
                os.mkdir(outdir+"/masterSkies",0o755)
            msfile = outdir+"/"+msname
            #Check to see if it exists
            if (os.access(msfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(msfile)
            if (not os.access(msfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                masterSky.writeTo(msfile)
        return masterSky

    #Calculate a sky to subtract by using the sigma-clipped median value of each column (or row) in each slitlet
    def calculateMedianSky(self, fdu, calibs, prevProc):
        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None
        currMask = ones(fdu.getShape(), dtype=bool)
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            ###MOS/IFU data -- get slitmask
            slitmask = self.findSlitmask(fdu, calibs, prevProc)
            if (slitmask is None):
                #FDU will be disabled in execute
                return None
            if (not slitmask.hasProperty("nslits")):
                slitmask.setProperty("nslits", slitmask.getData().max())
            nslits = slitmask.getProperty("nslits")
            if (slitmask.hasProperty("regions")):
                (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

        #Select kernel for 2d median
        kernel2d = fatboyclib.median2d
        if (self._fdb.getGPUMode()):
            #Use GPU for medians
            kernel2d=gpumedian2d
        skyData = zeros(fdu.getShape(), dtype=float32)
        for j in range(nslits):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (slitmask is not None):
                    currMask = (slitmask.getData()[ylos[j]:yhis[j]+1] == j+1)
                slit = (fdu.getData()[ylos[j]:yhis[j]+1]).copy()*currMask
                m = gpu_arraymedian(slit, axis="Y", nonzero=True, sigclip=True, kernel2d=kernel2d)
                skyData[ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                if (slitmask is not None):
                    currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == j+1)
                slit = (fdu.getData()[:,ylos[j]:yhis[j]+1]).copy()*currMask
                m = gpu_arraymedian(slit, axis="X", nonzero=True, sigclip=True, kernel2d=kernel2d)
                m = m.reshape((len(m), 1))
                skyData[:,ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]

        msname = "masterSkies/sky_"+fdu.getFullId()
        masterSky = fatboySpecCalib(self._pname, "master_sky", fdu, data=skyData, tagname=msname, log=self._log)

        if (fdu.hasProperty("cleanFrame")):
            cleanSkyData = zeros(fdu.getShape(), dtype=float32)
            for j in range(nslits):
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (slitmask is not None):
                        currMask = (slitmask.getData()[ylos[j]:yhis[j]+1] == j+1)
                    slit = (fdu.getData("cleanFrame")[ylos[j]:yhis[j]+1]).copy()*currMask
                    m = gpu_arraymedian(slit, axis="Y", nonzero=True, sigclip=True, kernel2d=kernel2d)
                    cleanSkyData[ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    if (slitmask is not None):
                        currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == j+1)
                    slit = (fdu.getData("cleanFrame")[:,ylos[j]:yhis[j]+1]).copy()*currMask
                    m = gpu_arraymedian(slit, axis="X", nonzero=True, sigclip=True, kernel2d=kernel2d)
                    m = m.reshape((len(m), 1))
                    cleanSkyData[:,ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
            masterSky.tagDataAs("cleanFrame", cleanSkyData)

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/masterSkies", os.F_OK)):
                os.mkdir(outdir+"/masterSkies",0o755)
            msfile = outdir+"/"+msname
            #Check to see if it exists
            if (os.access(msfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(msfile)
            if (not os.access(msfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                masterSky.writeTo(msfile)
        return masterSky

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            #If sky subtract is done before flat divide, it will attempt to
            #recursively process flats.  Make sure it only tries to sky subtract objects
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_CONTINUUM_SOURCE):
            #Also sky subtract for continuum source calibs - unless they have no RA and DEC
            if (fdu.ra is None or fdu.dec is None):
                return False
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_SKY):
            if (not fdu.hasProperty("disable") and not fdu.hasProperty("odd_frame_match")):
                print("skySubtractSpecProcess::checkValidDatatype> Warning: Odd skyframe "+fdu.getFullId()+" will be ignored.")
                self._log.writeLog(__name__, "Odd skyframe "+fdu.getFullId()+" will be ignored.", type=fatboyLog.WARNING)
                fdu.disable()
        return False
    #end checkValidDatatype

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Sky Subtract")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For skySubtract, this dict should have one entry 'masterSky' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if ('skipFrame' in calibs):
            #This image is an offsource sky.  Return false and skip to next object.  Do not disable yet as it will be used in creating offsource sky.
            return False
        if (not 'masterSky' in calibs):
            #Failed to obtain master sky frame
            #Issue error message and disable this FDU
            print("skySubtractSpecProcess::execute> ERROR: Sky not subtracted for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Sky not subtracted for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #get master sky
        masterSky = calibs['masterSky']

        #Check if output exists first
        ssfile = "skySubtracted/ss_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, ssfile)):
            #Also check if "cleanFrame" exists
            cleanfile = "skySubtracted/clean_ss_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "noisemap" exists
            nmfile = "skySubtracted/NM_ss_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Also check if crmask exists
            crmfile = "skySubtracted/crmask_ss_"+fdu.getFullId()
            self.checkOutputExists(fdu, crmfile, tag="crmask")
            #Disable masterSky if marked
            if (masterSky.hasProperty("disable")):
                masterSky.disable()
            return True

        #Propagate noisemap
        if (fdu.hasProperty("noisemap")):
            self.updateNoisemap(fdu, masterSky)

        #Propagate cosmic ray masks
        if (fdu.hasProperty("crmask")):
            self.updateCRMask(fdu, masterSky)

        #Need to save pre-sky subtracted data for later processing with odd frame
        if (fdu.hasProperty("odd_frame_match")):
            fdu.tagDataAs("preSkySubtracted")
            #Save cleanFrame too
            if (fdu.hasProperty("cleanFrame")):
                fdu.tagDataAs("cleanFrame_preSkySubtracted", fdu.getProperty("cleanFrame"))

        skyData = float32(masterSky.getData(tag="preSkySubtracted"))
        cleanSkyData = None
        if (fdu.hasProperty("cleanFrame")):
            #If masterSky has tag cleanFrame_preSkySubtracted then use it.  This is an odd frame
            if (masterSky.hasProperty("cleanFrame_preSkySubtracted")):
                cleanSkyData = masterSky.getData(tag="cleanFrame_preSkySubtracted")
            else:
                #Otherwise subtract cleanFrame tags from each other
                cleanSkyData = masterSky.getData(tag="cleanFrame")

        #Apply response curve if selected
        if (self.getOption('remove_residuals', fdu.getTag()).lower() == "yes"):
            if (self.getOption('residual_removal_method', fdu.getTag()).lower() == "response_curve"):
                self.applyResponseCurve(fdu, calibs, skyData, cleanSkyData)

        #subtract master sky and if "cleanFrame" exists, propagate it too
        fdu.updateData(float32(fdu.getData())-float32(masterSky.getData(tag="preSkySubtracted")))
        if (fdu.hasProperty("cleanFrame")):
            #If masterSky has tag cleanFrame_preSkySubtracted then use it.  This is an odd frame
            if (masterSky.hasProperty("cleanFrame_preSkySubtracted")):
                fdu.tagDataAs("cleanFrame", fdu.getData(tag="cleanFrame")-masterSky.getData(tag="cleanFrame_preSkySubtracted"))
            else:
                #Otherwise subtract cleanFrame tags from each other
                fdu.tagDataAs("cleanFrame", fdu.getData(tag="cleanFrame")-masterSky.getData(tag="cleanFrame"))

        if (masterSky.hasProperty("odd_frame_match") and masterSky.getProperty("odd_frame_match") == fdu.getFullId()):
            #Now we can remove properties preSkySubtracted and optionally cleanFrame_preSkySubtracted and noisemap_preSkySubtracted
            masterSky.removeProperty("preSkySubtracted")
            masterSky.removeProperty("cleanFrame_preSkySubtracted")
            masterSky.removeProperty("noisemap_preSkySubtracted")
            masterSky.removeProperty("crmask_preSkySubtracted")

        #Update history
        fdu._header.add_history('Sky subtracted using '+masterSky._id)

        #Remove residuals if median boxcar selected
        if (self.getOption('remove_residuals', fdu.getTag()).lower() == "yes"):
            if (self.getOption('residual_removal_method', fdu.getTag()).lower() == "median_boxcar"):
                self.removeResiduals(fdu, calibs, prevProc)

        #Disable masterSky if marked
        if (masterSky.hasProperty("disable")):
            masterSky.disable()
        return True
    #end execute

    #find offsource skies for offsource_multi_dither
    def findMultiOffsourceSkies(self, fdu, properties, headerVals):
        #Get options
        sky_offsource_range = float(self.getOption('sky_offsource_range', fdu.getTag()))/3600.
        sky_offsource_method = self.getOption('sky_offsource_method', fdu.getTag())
        #full | index | FITS keyword
        sort_key = self.getOption('onsource_sorting_key', fdu.getTag())
        ncombine = int(self.getOption('offsource_multi_dither_ncombine', fdu.getTag()))
        ignoreOddFrames = False
        if (self.getOption('ignore_odd_frames', fdu.getTag()).lower() == "yes"):
            ignoreOddFrames = True

        #1) Find any individual sky frames TAGGED for this object (OFFSOURCE only) to create master sky
        skies = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
        if (len(skies) > 0):
            #Found skies associated with this fdu
            #Set property sky_offsource_name to break up into groups
            idx = 0
            for j in range(len(skies)):
                if (ncombine > 0):
                    idx = j//ncombine
                elif (j > 0):
                    #auto-detect from idx
                    if (int(skies[j]._index) != int(skies[j-1]._index)+1):
                        #increment idx - there is a gap in index numbers
                        idx += 1
                skies[j].setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+fdu._id+"_"+fdu.filter+"_"+str(idx))
            #Now get sorted FDU list and pair up onsource frames
            fdulist = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, sortby=sort_key)
            idx = 0
            for j in range(len(fdulist)):
                if (ncombine > 0):
                    idx = j//ncombine
                elif (j > 0):
                    #auto-detect from idx
                    if (int(fdulist[j]._index) != int(fdulist[j-1]._index)+1):
                        #increment idx - there is a gap in index numbers
                        idx += 1
                fdulist[j].setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+fdu._id+"_"+fdu.filter+"_"+str(idx))
            return
        #2) Check for individual sky frames matching filter/sky_subtract_method/etc to create master sky
        skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
        if (len(skies) > 0):
            #Found skies associated with this fdu
            #Set property sky_offsource_name to break up into groups
            idx = 0
            for j in range(len(skies)):
                if (ncombine > 0):
                    idx = j//ncombine
                elif (j > 0):
                    #auto-detect from idx
                    if (int(skies[j]._index) != int(skies[j-1]._index)+1):
                        #increment idx - there is a gap in index numbers
                        idx += 1
                skies[j].setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+fdu._id+"_"+fdu.filter+"_"+str(idx))
            #Now get sorted FDU list and pair up onsource frames
            fdulist = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, sortby=sort_key)
            idx = 0
            for j in range(len(fdulist)):
                if (ncombine > 0):
                    idx = j//ncombine
                elif (j > 0):
                    #auto-detect from idx
                    if (int(fdulist[j]._index) != int(fdulist[j-1]._index)+1):
                        #increment idx - there is a gap in index numbers
                        idx += 1
                fdulist[j].setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+fdu._id+"_"+fdu.filter+"_"+str(idx))
            return
        #3) Check sky_offsource_method
        if (sky_offsource_method.lower() == "auto"):
            #get FDUs matching this identifier and filter, sorted by sort_key
            #if property sky_offsource_name not set, identify first object as onsource and objects at least sky_offsource_range away as offsource
            idx = 0
            fdulist = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, sortby=sort_key)
            #Loop over fdulist and identify skies based on RA and DEC
            for skyfdu in fdulist:
                #RA difference from object fdu
                diffRA = abs((skyfdu.ra - fdu.ra)*math.cos(fdu.dec*math.pi/180))
                #Dec difference from object fdu
                diffDec = abs(skyfdu.dec - fdu.dec)
                print(skyfdu.getFullId(), diffRA, diffDec)
                if (diffRA >= sky_offsource_range or diffDec >= sky_offsource_range):
                    #This is an offsource sky
                    skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)

            #Now loop over all fdus (objects and skies) and group sequences together
            for j in range(len(fdulist)):
                if (not fdulist[j].inUse):
                    #This frame has been disabled
                    continue
                if (fdulist[j].getObsType(True) == fatboyDataUnit.FDU_TYPE_SKY):
                    #This is a sky frame
                    continue
                if (fdulist[j].hasProperty("sky_subtract_match")):
                    #this FDU already has sky_subtract_match set
                    continue

                #This is the first in a series of onsource object frames
                #Find closest frame that is offsource and has not been already used.
                match = -1
                skyname = "offsourceSkies/sky_multi_"+fdu._id+"_"+fdu.filter+"_"+str(idx)
                for i in range(len(fdulist)):
                    if (not fdulist[i].inUse or fdulist[i].getObsType(True) != fatboyDataUnit.FDU_TYPE_SKY):
                        #continue if frame has been disabled or is not a sky
                        continue
                    if (fdu.hasProperty("sky_subtract_match")):
                        continue
                    if (match == -1):
                        match = i
                    elif (abs(j-i) < abs(j-match) and fdulist[i].ra != fdulist[match].ra and fdulist[i].dec != fdulist[match].dec):
                        #Only grab the first in a series of frames at same RA and Dec
                        match = i
                if (match != -1):
                    #Now find next n frames that match sky in RA and Dec
                    for i in range(match, len(fdulist)):
                        if (fdulist[i].inUse and fdulist[i].getObsType(True) == fatboyDataUnit.FDU_TYPE_SKY and fdulist[match].ra == fdulist[i].ra and fdulist[match].dec == fdulist[i].dec):
                            #set property sky_subtract_match
                            fdulist[i].setProperty("sky_subtract_match", skyname)
                        else:
                            #Break out of loop at first nonmatching frame
                            break
                else:
                    if (not ignoreOddFrames):
                        #Find closest offsource frame
                        for i in range(len(fdulist)):
                            if (not fdulist[i].inUse or fdulist[i].getObsType(True) != fatboyDataUnit.FDU_TYPE_SKY):
                                #continue if frame has been disabled or is not a sky
                                continue
                            if (match == -1):
                                match = i
                            elif (abs(j-i) < abs(j-match) and fdulist[i].ra != fdulist[match].ra and fdulist[i].dec != fdulist[match].dec):
                                #Only grab the first in a series of frames at same RA and Dec
                                match = i
                        if (match != -1):
                            #Now find next n frames that match sky in RA and Dec
                            for i in range(match, len(fdulist)):
                                if (fdulist[i].inUse and fdulist[i].getObsType(True) == fatboyDataUnit.FDU_TYPE_SKY and fdulist[match].ra == fdulist[i].ra and fdulist[match].dec == fdulist[i].dec):
                                    #set property odd_frame_match
                                    fdulist[i].setProperty("odd_frame_match", skyname)
                                else:
                                    #Break out of loop at first nonmatching frame
                                    break
                #Now find next n frames that match object in RA and Dec
                for i in range(j, len(fdulist)):
                    if (fdulist[i].inUse and fdulist[i].getObsType(True) != fatboyDataUnit.FDU_TYPE_SKY and fdulist[j].ra == fdulist[i].ra and fdulist[j].dec == fdulist[i].dec):
                        if (match == -1):
                            print("skySubtractSpecProcess::findMultiOffsourceSkies> Warning: Odd frame "+fdulist[i].getFullId()+" ignored.  Sky not subtracted!")
                            self._log.writeLog(__name__, "Odd frame "+fdulist[i].getFullId()+" ignored.  Sky not subtracted!", type=fatboyLog.WARNING)
                            #Disable here
                            fdulist[i].disable()
                            continue
                        fdulist[i].setProperty("sky_subtract_match", skyname)
                    else:
                        #Break out of loop at first nonmatching frame
                        break
                idx +=1 #incremend idx here
        elif (os.access(sky_offsource_method, os.F_OK)):
            #This is an ASCII file listing identifier_object start_index stop_index identifier_sky start_index stop_index
            #Process entire file here
            methodList = readFileIntoList(sky_offsource_method)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) != 6):
                    print("skySubtractSpecProcess::findMultiOffsourceSkies> Warning: line "+str(j)+" misformatted in "+sky_offsource_method)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+sky_offsource_method, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                try:
                    methodList[j][1] = int(methodList[j][1])
                    methodList[j][2] = int(methodList[j][2])
                    methodList[j][4] = int(methodList[j][4])
                    methodList[j][5] = int(methodList[j][5])
                except Exception as ex:
                    print("skySubtractSpecProcess::findMultiOffsourceSkies> Warning: line "+str(j)+" misformatted in "+sky_offsource_method)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+sky_offsource_method, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'sky_subtract_match' property.
            #some FDUs may have used xml to define sky_subtract_match already
            #if tag is None, this loops over all FDUs
            for skyfdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (skyfdu.hasProperty("sky_subtract_match")):
                    #this FDU already has sky_subtract_match set
                    continue
                i = 0 #line number
                #method = [ 'identifier', start_idx, end_idx, 'sky_identifier', start_idx, end_idx ]
                for method in methodList:
                    i += 1
                    skypfix = method[3]+"_"+skyfdu.filter+"_"+str(i)
                    if (skyfdu._id == method[0] and int(skyfdu._index) >= method[1] and int(skyfdu._index) <= method[2]):
                        #exact match. this is an object, set sky_subtract_match property
                        skyfdu.setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+skypfix)
                    elif (skyfdu._id == method[3] and int(skyfdu._index) >= method[4] and int(skyfdu._index) <= method[5]):
                        #exact match. this is a sky.  set sky_subtract_match property and set type to SKY
                        skyfdu.setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+skypfix)
                        skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
                    elif (skyfdu._id.find(method[0]) != -1 and int(skyfdu._index) >= method[1] and int(skyfdu._index) <= method[2]):
                        #partial match. this is an object, set sky_subtract_match property
                        skyfdu.setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+skypfix)
                    elif (skyfdu._id.find(method[3]) != -1 and int(skyfdu._index) >= method[4] and int(skyfdu._index) <= method[5]):
                        #partial match. this is a sky.  set sky_subtract_match property and set type to SKY
                        skyfdu.setProperty("sky_subtract_match", "offsourceSkies/sky_multi_"+skypfix)
                        skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
        else:
            print("skySubtractSpecProcess::findMultiOffsourceSkies> Error: invalid sky_offsource_method: "+sky_offsource_method)
            self._log.writeLog(__name__, " invalid sky_offsource_method: "+sky_offsource_method, type=fatboyLog.ERROR)
            return
        return
    #end findMultiOffsourceSkies

    #find offsource skies
    def findOffsourceSkies(self, fdu, properties, headerVals):
        #Get options
        sky_offsource_range = float(self.getOption('sky_offsource_range', fdu.getTag()))/3600.
        sky_offsource_method = self.getOption('sky_offsource_method', fdu.getTag())
        #full | index | FITS keyword
        sort_key = self.getOption('onsource_sorting_key', fdu.getTag())
        #1) Find any individual sky frames TAGGED for this object (OFFSOURCE only) to create master sky
        skies = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
        if (len(skies) > 0):
            #Found skies associated with this fdu
            return skies
        #2) Check for individual sky frames matching filter/sky_subtract_method/etc to create master sky
        #2A) Look for individual skies that match this object ID only - e.g. ABBA dither pattern where all are same identifier
        skies = self._fdb.getCalibs(ident=fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
        if (len(skies) > 0):
            #Found skies associated with this fdu
            return skies
        #2B) Look for other idents for skies - e.g. offsource skies have different identifier than objects, maybe taken AAAA BBBB
        skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
        if (len(skies) > 0):
            #Found skies associated with this fdu
            return skies
        #3) Check sky_offsource_method
        if (sky_offsource_method.lower() == "auto"):
            #get FDUs matching this identifier and filter, sorted by sort_key
            #if property sky_offsource_name not set, identify first object as onsource and objects at least sky_offsource_range away as offsource
            for skyfdu in self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, sortby=sort_key):
                if (skyfdu.hasProperty("sky_offsource_name")):
                    #this FDU already has sky_offsource_name set
                    continue
                skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+fdu._id+"_"+fdu.filter)
                #RA difference from object fdu
                diffRA = abs((skyfdu.ra - fdu.ra)*math.cos(fdu.dec*math.pi/180))
                #Dec difference from object fdu
                diffDec = abs(skyfdu.dec - fdu.dec)
                if (diffRA >= sky_offsource_range or diffDec >= sky_offsource_range):
                    #This is an offsource sky
                    skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
        elif (os.access(sky_offsource_method, os.F_OK)):
            #This is an ASCII file listing identifier_object start_index stop_index identifier_sky start_index stop_index
            #Process entire file here
            methodList = readFileIntoList(sky_offsource_method)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) != 6):
                    print("skySubtractSpecProcess::findOffsourceSkies> Warning: line "+str(j)+" misformatted in "+sky_offsource_method)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+sky_offsource_method, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                try:
                    methodList[j][1] = int(methodList[j][1])
                    methodList[j][2] = int(methodList[j][2])
                    methodList[j][4] = int(methodList[j][4])
                    methodList[j][5] = int(methodList[j][5])
                except Exception as ex:
                    print("skySubtractSpecProcess::findOffsourceSkies> Warning: line "+str(j)+" misformatted in "+sky_offsource_method)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+sky_offsource_method, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'sky_offsource_name' property.
            #some FDUs may have used xml to define sky_offsource_name already
            #if tag is None, this loops over all FDUs
            for skyfdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (skyfdu.hasProperty("sky_offsource_name")):
                    #this FDU already has sky_offsource_name set
                    continue
                i = 0 #line number
                #method = [ 'identifier', start_idx, end_idx, 'sky_identifier', start_idx, end_idx ]
                for method in methodList:
                    i += 1
                    skypfix = method[3]+"_"+str(i)
                    if (skyfdu._id == method[0] and int(skyfdu._index) >= method[1] and int(skyfdu._index) <= method[2]):
                        #exact match. this is an object, set sky_offsource_name property
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                    elif (skyfdu._id == method[3] and int(skyfdu._index) >= method[4] and int(skyfdu._index) <= method[5]):
                        #exact match. this is a sky.  set sky_offsource_name property and set type to SKY
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                        skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
                    elif (skyfdu._id.find(method[0]) != -1 and int(skyfdu._index) >= method[1] and int(skyfdu._index) <= method[2]):
                        #partial match. this is an object, set sky_offsource_name property
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                    elif (skyfdu._id.find(method[3]) != -1 and int(skyfdu._index) >= method[4] and int(skyfdu._index) <= method[5]):
                        #partial match. this is a sky.  set sky_offsource_name property and set type to SKY
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                        skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
        else:
            print("skySubtractSpecProcess::findOffsourceSkies> Error: invalid sky_offsource_method: "+sky_offsource_method)
            self._log.writeLog(__name__, " invalid sky_offsource_method: "+sky_offsource_method, type=fatboyLog.ERROR)
            return skies
        #Now Check for individual sky frames matching filter/section/sky_subtract_method/sky_offsource_name to create master sky
        properties['sky_offsource_name'] = fdu.getProperty('sky_offsource_name')
        skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
        return skies
    #end findOffsourceSkies

    #Put find slitmask in individual method so code isn't pasted in multiple methods
    def findSlitmask(self, fdu, calibs, prevProc):
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            return None
        if ('slitmask' in calibs):
            return calibs['slitmask']
        slitmask = fdu.getSlitmask()
        if (slitmask is not None):
            return slitmask
        elif (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            #Use findSlitletProcess.getCalibs to get slitmask and create if necessary
            #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
            #Returns None on a failure
            fs_process = self._fdb.getProcessByName("findSlitlets")
            if (fs_process is None or not isinstance(fs_process, fatboyProcess)):
                print("skySubtractSpecProcess::findSlitmask> ERROR: could not find process findSlitlets!  Check your XML file!")
                self._log.writeLog(__name__, "could not find process findSlitlets!  Check your XML file!", type=fatboyLog.ERROR)
                return None
            #Call setDefaultOptions and getCalibs on skySubtractSpecProcess
            fs_process.setDefaultOptions()
            fs_calibs = fs_process.getCalibs(fdu, prevProc)
            if (not 'slitmask' in fs_calibs):
                #Failed to obtain slitmask
                #Issue error message.  FDU will be disabled in execute()
                print("skySubtractSpecProcess::findSlitmask> ERROR: Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!")
                self._log.writeLog(__name__, "Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!", type=fatboyLog.ERROR)
                return None
            return fs_calibs['slitmask']
    #end findSlitmask

    #Find individual sky frames for this object to create master sky
    def findSky(self, fdu, properties, headerVals, prevProc):
        if (not fdu.hasProperty("sky_subtract_match")):
            #No sky found for this frame.  Should not happen
            print("skySubtractSpecProcess::findSky> Warning: Could not find frame to pair with "+fdu.getFullId()+"! Sky not subtracted!")
            self._log.writeLog(__name__, "Could not find frame to pair with "+fdu.getFullId()+"! Sky not subtracted!", type=fatboyLog.WARNING)
            return None

        skymethod = properties['sky_method'].lower()
        if (skymethod == "offsource_multi_dither"):
            #Find all objects with property sky_subtract_match same as this FDU, recursively process other FDUs, and imcombine them
            properties['sky_subtract_match'] = fdu.getProperty("sky_subtract_match")
            fdus = self._fdb.getFDUs(ident = fdu._id, obstype=fatboyDataUnit.FDU_TYPE_OBJECT, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
            #Recursively process object frames
            if (prevProc is not None):
                self.recursivelyExecute(fdus, prevProc)

            #Select cpu/gpu option
            imcombine_method = gpu_imcombine.imcombine
            if (not self._fdb.getGPUMode()):
                imcombine_method = imcombine.imcombine

            #imcombine object files and scale by median
            (data, header) = imcombine_method(fdus, outfile=None, method="median", scale="median", mef=fdus[0]._mef, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU)
            #Update this FDU and disable other frames
            fdu.updateData(data)
            fdu.updateHeader(header)
            for j in range(len(fdus)):
                if (fdus[j] != fdu):
                    fdus[j].disable()

            #Find all skies with this sky_subtract_match.
            #First find any individual sky frames TAGGED for this object (OFFSOURCE only) to create master sky
            skies = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
            if (len(skies) == 0):
                #If none found, proceed to call getCalibs
                skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
            #If none exist, look for skies with odd_frame_match equal to sky_subtract_match
            del properties['sky_subtract_match']
            if (len(skies) == 0):
                properties['odd_frame_match'] = fdu.getProperty("sky_subtract_match")
                #First find any individual sky frames TAGGED for this object (OFFSOURCE only) to create master sky
                skies = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
                if (len(skies) == 0):
                    skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
                del properties['odd_frame_match']
                if (len(skies) == 0):
                    #No sky found for this frame.  Should not happen
                    print("skySubtractSpecProcess::findSky> Warning: Could not find frame to pair with "+fdu.getFullId()+"! Sky not subtracted!")
                    self._log.writeLog(__name__, "Could not find frame to pair with "+fdu.getFullId()+"! Sky not subtracted!", type=fatboyLog.WARNING)
                    return None
            #Recursively process skies and imcombine them
            if (prevProc is not None):
                self.recursivelyExecute(skies, prevProc)

            #imcombine sky files and scale by median
            (data, header) = imcombine_method(skies, outfile=None, method="median", scale="median", mef=fdus[0]._mef, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU)
            #Update sky[0] with data and history -- and it will retain process history.
            skies[0].updateData(data)
            skies[0].updateHeader(header)
            #Disable other sky frames
            for j in range(1, len(skies)):
                skies[j].disable()

            #return sky[0]
            return skies[0]
        else:
            #All other methods, grab and return sky_subtract_match
            print("\t"+fdu.getFullId()+" already matched with "+fdu.getProperty("sky_subtract_match"))
            self._log.writeLog(__name__, fdu.getFullId()+" already matched with "+fdu.getProperty("sky_subtract_match"), printCaller=False, tabLevel=1)
            return self._fdb.getIndividualFDU(fdu.getProperty("sky_subtract_match"))
        return None
    #end findSky

    #Look at XML to determine sky subtraction methods
    def findSkySubtractMethods(self, fdu):
        skymethod = self.getOption("sky_method", fdu.getTag())
        if (skymethod.lower() in self.ssmethods):
            skymethod = skymethod.lower()
            #loop over dataset and assign property to all fdus that don't already have 'sky_method' property.
            #some FDUs may have used xml to define sky_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (not fdu.hasProperty("sky_method")):
                    fdu.setProperty("sky_method", skymethod)
        elif (os.access(skymethod, os.F_OK)):
            #This is an ASCII file listing filter/identifier and sky method
            #Process entire file here
            methodList = readFileIntoList(skymethod)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) < 2):
                    print("skySubtractSpecProcess::findSkySubtractMethods> Warning: line "+str(j)+" misformatted in "+skymethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skymethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                methodList[j][1] = methodList[j][1].lower()
                if (not methodlist[j][1] in ssmethods):
                    print("skySubtractSpecProcess::findSkySubtractMethods> Warning: line "+str(j)+" misformatted in "+skymethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skymethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'sky_method' property.
            #some FDUs may have used xml to define sky_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (fdu.hasProperty("sky_method")):
                    #this FDU already has sky_method set
                    continue
                #method = [ 'Filter/identifier', 'method' ]
                for method in methodList:
                    if (method[0].lower() == fdu.filter.lower()):
                        fdu.setProperty("sky_method", method[1])
                        #Exact match for filter
                    elif (len(method[0]) > 2 and fdu._id.lower().find(method[0].lower()) != -1):
                        #Partial match for identifier
                        fdu.setProperty("sky_method", method[1])
        else:
            print("skySubtractSpecProcess::findSkySubtractMethods> Error: invalid sky_method: "+skymethod)
            self._log.writeLog(__name__, " invalid sky_method: "+skymethod, type=fatboyLog.ERROR)
    #end findSkySubtractMethods

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        msfilename = self.getCalib("masterSky", fdu.getTag())
        if (msfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(msfilename, os.F_OK)):
                print("skySubtractSpecProcess::getCalibs> Using master sky "+msfilename+"...")
                self._log.writeLog(__name__, "Using master sky "+msfilename+"...")
                calibs['masterSky'] = fatboySpecCalib(self._pname, "master_sky", fdu, filename=msfilename, log=self._log)
                return calibs
            else:
                print("skySubtractSpecProcess::getCalibs> Warning: Could not find master sky "+msfilename+"...")
                self._log.writeLog(__name__, "Could not find master sky "+msfilename+"...", type=fatboyLog.WARNING)

        #Look for slitmask (for method=median only)
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("skySubtractSpecProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("skySubtractSpecProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and sky_method
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        if (not fdu.hasProperty("sky_method")):
            #Look at XML options to find sky method and assign it to FDUs
            self.findSkySubtractMethods(fdu)

        properties['sky_method'] = fdu.getProperty("sky_method")
        if (properties['sky_method'] is None):
            print("skySubtractSpecProcess::getCalibs> Error: Could not find sky_method for "+fdu.getFullId())
            self._log.writeLog(__name__, " Could not find sky_method for "+fdu.getFullId(), type=fatboyLog.ERROR)
            return calibs

        #1) Check for an already created master sky frame matching specmode/filter/grism and TAGGED for this object
        masterSky = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_sky", filter=fdu.filter, properties=properties, headerVals=headerVals)
        if (masterSky is not None):
            #Found master sky.  Return here
            calibs['masterSky'] = masterSky
            return calibs
        #2) Check for an already created master sky frame matching specmode/filter/grism
        masterSky = self._fdb.getMasterCalib(self._pname, obstype="master_sky", filter=fdu.filter, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
        if (masterSky is not None):
            #Found master sky.  Return here
            calibs['masterSky'] = masterSky
            return calibs
        #3) Check default_master_sky for matching specmode/filter/grism
        #### Unlike flats and darks, default sky takes priority since objects themselves are used and can't be omitted
        #### This option will rarely be used
        defaultMasterSkies = []
        if (self.getOption('default_master_sky', fdu.getTag()) is not None):
            dmslist = self.getOption('default_master_sky', fdu.getTag())
            if (dmslist.count(',') > 0):
                #comma separated list
                defaultMasterSkies = dmslist.split(',')
                removeEmpty(defaultMasterSkies)
                for j in range(len(defaultMasterSkies)):
                    defaultMasterSkies[j] = defaultMasterSkies[j].strip()
            elif (dmslist.endswith('.fit') or dmslist.endswith('.fits')):
                #FITS file given
                defaultMasterSkies.append(dmslist)
            elif (dmslist.endswith('.dat') or dmslist.endswith('.list') or dmslist.endswith('.txt')):
                #ASCII file list
                defaultMasterSkies = readFileIntoList(dmslist)
            for mskyfile in defaultMasterSkies:
                #Loop over list of default master skies
                #masterSky = fatboyImage(mskyfile)
                masterSky = fatboySpecCalib(self._pname, "master_sky", fdu, filename=mskyfile, log=self._log)
                #read header and initialize
                masterSky.readHeader()
                masterSky.initialize()
                if (masterSky.filter != fdu.filter):
                    #does not match filter
                    continue
                masterSky.setType("master_sky")
                #Found matching master sky
                print("skySubtractSpecProcess::getCalibs> Using default master sky "+masterSky.getFilename())
                self._log.writeLog(__name__, "Using default master sky "+masterSky.getFilename())
                self._fdb.appendCalib(masterSky)
                calibs['masterSky'] = masterSky
                return calibs
        #3b) If "median" method for GMOS/optical data, sky is calculated from the data itself
        if (fdu.getProperty("sky_method") == "median"):
            masterSky = self.calculateMedianSky(fdu, calibs, prevProc)
            if (masterSky is not None):
                calibs['masterSky'] = masterSky
            #No continuum in sky so use only positive = true for rectify/double subtract purposes
            fdu.setProperty("use_only_positive", True)
            return calibs
        #3c) If "median_boxcar" method for longslit optical data, sky is calculated from the data itself
        if (fdu.getProperty("sky_method") == "median_boxcar"):
            masterSky = self.calculateMedianBoxcarSky(fdu, calibs, prevProc)
            if (masterSky is not None):
                calibs['masterSky'] = masterSky
            #No continuum in sky so use only positive = true for rectify/double subtract purposes
            fdu.setProperty("use_only_positive", True)
            return calibs
        #4) If no property sky_subtract_match, this is the first pass through for this object set.
        #Call matchSkies to match up objects based on sky_method
        if (not fdu.hasProperty("sky_subtract_match")):
            self.matchSkies(fdu, properties, headerVals)
        #Now call findSky to get the sky based on sky_method and sky_subtract_match
        #Only offsource_multi_dither will require additional processing here to imcombine files
        sky = self.findSky(fdu, properties, headerVals, prevProc)
        if (sky is None):
            print("skySubtractSpecProcess::getCalibs> ERROR: No sky found for "+fdu.getFullId())
            self._log.writeLog(__name__, "No sky found for "+fdu.getFullId(), type=fatboyLog.ERROR)
        elif (sky.hasProperty('skipFrame')):
            #Check if this fdu is an offsource sky.  Only possible for sky_offsource_method from file
            print("skySubtractSpecProcess::getCalibs> Object "+fdu.getFullId()+" is actually an OFFSOURCE sky!  Skipping.")
            self._log.writeLog(__name__, "Object "+fdu.getFullId()+" is actually an OFFSOURCE sky!  Skipping.")
            calibs['skipFrame'] = True
            return calibs
        else:
            #Found sky associated with this fdu
            #First recursively process
            self.recursivelyExecute([sky], prevProc)
            calibs['masterSky'] = sky
            return calibs
        print("skySubtractSpecProcess::getCalibs> ERROR: No skies found for "+fdu.getFullId())
        self._log.writeLog(__name__, "No skies found for "+fdu.getFullId(), type=fatboyLog.ERROR)
        return calibs
    #end getCalibs

    #Match individual frames based on sky_method
    def matchSkies(self, fdu, properties, headerVals):
        skymethod = properties['sky_method'].lower()
        print("skySubtractSpecProcess::matchSkies> Using method "+skymethod)
        self._log.writeLog(__name__, "Using method "+skymethod)

        #Get options
        #full | index | FITS keyword
        sort_key = self.getOption('onsource_sorting_key', fdu.getTag())
        dsOddFrames = False
        if (self.getOption('double_subtract_odd_frames', fdu.getTag()).lower() == "yes"):
            dsOddFrames = True
        ignoreOddFrames = False
        if (self.getOption('ignore_odd_frames', fdu.getTag()).lower() == "yes"):
            ignoreOddFrames = True

        if (skymethod == "dither"):
            #convert to arcsec
            sky_dithering_range = float(self.getOption('sky_dithering_range', fdu.getTag()))/3600.
            #get FDUs matching this identifier, filter, grism, specmode, sorted by index
            skyfdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, properties=properties, headerVals=headerVals, sortby=sort_key)
            if (len(skyfdus) == 2 and skyfdus[0] != fdu):
                skyfdus.reverse() #reverse list
            #If only this object found, cannot subtract sky
            if (len(skyfdus) == 1):
                print("skySubtractSpecProcess::matchSkies> Warning: Only found one image for object "+fdu._id+"! Sky not subtracted!")
                self._log.writeLog(__name__, "Only found one image for object "+fdu._id+"! Sky not subtracted!", type=fatboyLog.WARNING)
                return
            #Now loop over skyfdus and pair up frames
            for idx in range(len(skyfdus)):
                if (not skyfdus[idx].inUse):
                    #This frame has been disabled
                    continue
                if (skyfdus[idx].hasProperty("disable")):
                    #This frame has been matched already with a previous frame
                    continue
                #Find next frame with matching identifier and different RA/Dec
                match = -1
                for j in range(idx+1, len(skyfdus)):
                    if (skyfdus[j].hasProperty("disable")):
                        #This frame has been matched already with a previous frame
                        continue
                    #RA difference from current fdu
                    diffRA = abs((skyfdus[j].ra - skyfdus[idx].ra)*math.cos(skyfdus[idx].dec*math.pi/180))
                    #Dec difference from current fdu
                    diffDec = abs(skyfdus[j].dec - skyfdus[idx].dec)
                    #Match if it is at least sky_dithering_range arcsec away in either direction from both target and last fdu
                    if (diffRA >= sky_dithering_range or diffDec >= sky_dithering_range):
                        match = j
                        #mark matched frame to be disabled in execute
                        skyfdus[match].setProperty("disable", True)
                        print("\tDither: "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                        self._log.writeLog(__name__, "Dither: "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
                        break
                #if no match and not set to ignore odd frames
                if (match == -1 and not ignoreOddFrames):
                    #Find previous frame with matching identifier and different RA/Dec
                    for j in range(idx-1, -1, -1):
                        #RA difference from current fdu
                        diffRA = abs((skyfdus[j].ra - skyfdus[idx].ra)*math.cos(skyfdus[idx].dec*math.pi/180))
                        #Dec difference from current fdu
                        diffDec = abs(skyfdus[j].dec - skyfdus[idx].dec)
                        #Match if it is at least sky_dithering_range arcsec away in either direction from both target and last fdu
                        if (diffRA >= sky_dithering_range or diffDec >= sky_dithering_range):
                            match = j
                            #Set property odd_frame_match in skyfdus[idx] to tell it to keep preSkySubtracted data tag
                            skyfdus[match].setProperty("odd_frame_match", skyfdus[idx].getFullId())
                            if (not dsOddFrames):
                                #Set property for use in double subtraction
                                skyfdus[idx].setProperty("use_only_positive", True)
                            print("\tDither: odd frame "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                            self._log.writeLog(__name__, "Dither: odd frame"+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
                            break
                #no match found
                if (match == -1):
                    print("skySubtractSpecProcess::matchSkies> Warning: Odd frame "+skyfdus[idx].getFullId()+" ignored.  Sky not subtracted!")
                    self._log.writeLog(__name__, "Odd frame "+skyfdus[idx].getFullId()+" ignored.  Sky not subtracted!", type=fatboyLog.WARNING)
                    #Just continue here.  By not setting property sky_subtract_match, it will be discarded later.
                    continue
                #set property sky_subtract_match
                skyfdus[idx].setProperty("sky_subtract_match", skyfdus[match].getFullId())
                #set property for double subtraction
                ds_guess = sqrt((skyfdus[idx].ra-skyfdus[match].ra)**2+(skyfdus[idx].dec-skyfdus[match].dec)**2)*3600/skyfdus[idx].pixscale
                if ((skyfdus[idx].ra-skyfdus[match].ra+skyfdus[idx].dec-skyfdus[match].dec) < 0):
                    ds_guess = -1*ds_guess
                skyfdus[idx].setProperty("double_subtract_guess", ds_guess)
                if (ds_guess > 0):
                    skyfdus[idx].setProperty("matched_ra", skyfdus[match].ra)
                    skyfdus[idx].setProperty("matched_dec", skyfdus[match].dec)
            return
        elif (skymethod == "step"):
            #get FDUs matching this identifier, filter, grism, specmode, sorted by index
            skyfdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, properties=properties, headerVals=headerVals, sortby=sort_key)
            if (len(skyfdus) == 1):
                print("skySubtractSpecProcess::matchSkies> Warning: Only found one image for object "+fdu._id+"! Sky not subtracted!")
                self._log.writeLog(__name__, "Only found one image for object "+fdu._id+"! Sky not subtracted!", type=fatboyLog.WARNING)
                return None
            #If odd number of frames, pair 1-2, 2-3, 3-4, 4-5, 5-1
            #double subtract none unless double_subtract_odd_frames = True
            isEven = False
            if (len(skyfdus) % 2 == 0):
                #Even number of frames
                #Pair 1-2, 3-4, 5-6, double subtract all
                isEven = True
            #Now loop over skyfdus and pair up frames
            for idx in range(len(skyfdus)):
                if (not skyfdus[idx].inUse):
                    #This frame has been disabled
                    continue
                if (skyfdus[idx].hasProperty("disable")):
                    #This frame has been matched already with a previous frame
                    continue
                #Find next frame with matching identifier and different RA/Dec
                match = idx+1
                if (not isEven and match == len(skyfdus)):
                    #Wrap around to beginning
                    match = 0
                    #Set property odd_frame_match in skyfdus[idx] to tell it to keep preSkySubtracted data tag
                    skyfdus[match].setProperty("odd_frame_match", skyfdus[idx].getFullId())
                if (isEven):
                    #mark matched frame to be disabled in execute
                    skyfdus[match].setProperty("disable", True)
                    #set property for double subtraction
                    ds_guess = sqrt((skyfdus[idx].ra-skyfdus[match].ra)**2+(skyfdus[idx].dec-skyfdus[match].dec)**2)*3600/skyfdus[idx].pixscale
                    if ((skyfdus[idx].ra-skyfdus[match].ra+skyfdus[idx].dec-skyfdus[match].dec) < 0):
                        ds_guess = -1*ds_guess
                    skyfdus[idx].setProperty("double_subtract_guess", ds_guess)
                    if (ds_guess > 0):
                        skyfdus[idx].setProperty("matched_ra", skyfdus[match].ra)
                        skyfdus[idx].setProperty("matched_dec", skyfdus[match].dec)
                else:
                    if (dsOddFrames):
                        #set property for double subtraction
                        ds_guess = sqrt((skyfdus[idx].ra-skyfdus[match].ra)**2+(skyfdus[idx].dec-skyfdus[match].dec)**2)*3600/skyfdus[idx].pixscale
                        if ((skyfdus[idx].ra-skyfdus[match].ra+skyfdus[idx].dec-skyfdus[match].dec) < 0):
                            ds_guess = -1*ds_guess
                        skyfdus[idx].setProperty("double_subtract_guess", ds_guess)
                        if (ds_guess > 0):
                            skyfdus[idx].setProperty("matched_ra", skyfdus[match].ra)
                            skyfdus[idx].setProperty("matched_dec", skyfdus[match].dec)
                    else:
                        skyfdus[idx].setProperty("use_only_positive", True)
                #set property sky_subtract_match
                skyfdus[idx].setProperty("sky_subtract_match", skyfdus[match].getFullId())
                print("\tStep: "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                self._log.writeLog(__name__, "Step: "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
            return
        elif (skymethod == "ifu_onsource_dither"):
            #convert to arcsec
            sky_dithering_range = float(self.getOption('sky_dithering_range', fdu.getTag()))/3600.
            #get FDUs matching this identifier, filter, grism, specmode, sorted by index
            skyfdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, properties=properties, headerVals=headerVals, sortby=sort_key)
            #If only this object found, cannot subtract sky
            if (len(skyfdus) == 1):
                print("skySubtractSpecProcess::matchSkies> Warning: Only found one image for object "+fdu._id+"! Sky not subtracted!")
                self._log.writeLog(__name__, "Only found one image for object "+fdu._id+"! Sky not subtracted!", type=fatboyLog.WARNING)
                return
            #Now loop over skyfdus and pair up frames
            for idx in range(len(skyfdus)):
                if (not skyfdus[idx].inUse):
                    #This frame has been disabled
                    continue
                if (skyfdus[idx].hasProperty("sky_subtract_match")):
                    #This frame has already been matched up
                    continue
                #Find next frame with matching identifier and different RA/Dec
                match = -1
                for j in range(idx+1, len(skyfdus)):
                    #RA difference from current fdu
                    diffRA = abs((skyfdus[j].ra - skyfdus[idx].ra)*math.cos(skyfdus[idx].dec*math.pi/180))
                    #Dec difference from current fdu
                    diffDec = abs(skyfdus[j].dec - skyfdus[idx].dec)
                    #Match if it is at least sky_dithering_range arcsec away in either direction from both target and last fdu
                    if (diffRA >= sky_dithering_range or diffDec >= sky_dithering_range):
                        match = j
                        #set property sky_subtract_match
                        skyfdus[match].setProperty("sky_subtract_match", skyfdus[idx].getFullId())
                        #Set property odd_frame_match in skyfdus[idx] to tell it to keep preSkySubtracted data tag
                        #This should be done for ALL frames in ifu_onsource_dither
                        skyfdus[match].setProperty("odd_frame_match", skyfdus[idx].getFullId())
                        skyfdus[idx].setProperty("odd_frame_match", skyfdus[match].getFullId())
                        #Set property for use in double subtraction
                        skyfdus[match].setProperty("use_only_positive", True)
                        print("\tIFU Onsource Dither: "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                        self._log.writeLog(__name__, "IFU Onsource Dither: "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
                        break
                #if no match and not set to ignore odd frames
                if (match == -1 and not ignoreOddFrames):
                    #Find previous frame with matching identifier and different RA/Dec
                    for j in range(idx-1, -1, -1):
                        #RA difference from current fdu
                        diffRA = abs((skyfdus[j].ra - skyfdus[idx].ra)*math.cos(skyfdus[idx].dec*math.pi/180))
                        #Dec difference from current fdu
                        diffDec = abs(skyfdus[j].dec - skyfdus[idx].dec)
                        #Match if it is at least sky_dithering_range arcsec away in either direction from both target and last fdu
                        if (diffRA >= sky_dithering_range or diffDec >= sky_dithering_range):
                            match = j
                            #Set property odd_frame_match in skyfdus[idx] to tell it to keep preSkySubtracted data tag
                            skyfdus[match].setProperty("odd_frame_match", skyfdus[idx].getFullId())
                            print("\tIFU Onsource Dither: odd frame "+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                            self._log.writeLog(__name__, "IFU Onsource Dither: odd frame"+skyfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
                            break
                #no match found
                if (match == -1):
                    print("skySubtractSpecProcess::matchSkies> Warning: Odd frame "+skyfdus[idx].getFullId()+" ignored.  Sky not subtracted!")
                    self._log.writeLog(__name__, "Odd frame "+skyfdus[idx].getFullId()+" ignored.  Sky not subtracted!", type=fatboyLog.WARNING)
                    #Just continue here.  By not setting property sky_subtract_match, it will be discarded later.
                    continue
                #set property sky_subtract_match
                skyfdus[idx].setProperty("sky_subtract_match", skyfdus[match].getFullId())
                #Set property for use in double subtraction
                skyfdus[idx].setProperty("use_only_positive", True)
            return
        elif (skymethod == "offsource_dither"):
            #Use helper method findOffsourceSkies to get list of offsource skies
            skyfdus = self.findOffsourceSkies(fdu, properties, headerVals)
            #get FDUs matching this identifier, filter, grism, specmode, sorted by index
            objfdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, properties=properties, headerVals=headerVals, sortby=sort_key)
            #Now loop over objfdus and skyfdus and pair up frames
            skyidx = 0
            for idx in range(len(objfdus)):
                if (not objfdus[idx].inUse):
                    #This frame has been disabled
                    continue
                if (objfdus[idx].getObsType(True) == fatboyDataUnit.FDU_TYPE_SKY):
                    #This is a sky frame
                    continue
                #Find next sky frame with matching identifier
                match = -1
                for j in range(skyidx, len(skyfdus)):
                    if (not skyfdus[j].inUse):
                        #This frame has been disabled
                        continue
                    if (skyfdus[j].hasProperty("disable")):
                        #This frame has been matched already with a previous frame
                        continue
                    match = j
                    skyidx = j+1
                    #mark matched frame to be disabled in execute
                    skyfdus[match].setProperty("disable", True)
                    print("\tOffsource Dither: "+objfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                    self._log.writeLog(__name__, "Offsource Dither: "+objfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
                    break
                #if no match and not set to ignore odd frames
                if (match == -1 and not ignoreOddFrames):
                    #Find previous frame with matching identifier and different RA/Dec
                    for j in range(skyidx-1, -1, -1):
                        if (not skyfdus[j].inUse):
                            #This frame has been disabled
                            continue
                        match = j
                        #Set property odd_frame_match in skyfdus[idx] to tell it to keep preSkySubtracted data tag
                        skyfdus[match].setProperty("odd_frame_match", objfdus[idx].getFullId())
                        print("\tOffsource Dither: odd frame "+objfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId())
                        self._log.writeLog(__name__, "Offsource Dither: odd frame"+objfdus[idx].getFullId()+" matched with "+skyfdus[match].getFullId(), printCaller=False, tabLevel=1)
                        break
                #no match found
                if (match == -1):
                    print("skySubtractSpecProcess::matchSkies> Warning: Odd frame "+objfdus[idx].getFullId()+" ignored.  Sky not subtracted!")
                    self._log.writeLog(__name__, "Odd frame "+objfdus[idx].getFullId()+" ignored.  Sky not subtracted!", type=fatboyLog.WARNING)
                    #Just continue here.  By not setting property sky_subtract_match, it will be discarded later.
                    objfdus[idx].disable()
                    continue
                #set property sky_subtract_match
                objfdus[idx].setProperty("sky_subtract_match", skyfdus[match].getFullId())
                #Set property for use in double subtraction
                objfdus[idx].setProperty("use_only_positive", True)
            return
        elif (skymethod == "offsource_multi_dither"):
            #Use helper method findMultiOffsourceSkies to get list of offsource skies
            self.findMultiOffsourceSkies(fdu, properties, headerVals)
            #get FDUs matching this identifier, filter, grism, specmode, sorted by index
            for objfdu in self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, properties=properties, headerVals=headerVals, sortby=sort_key):
                #check for property sky_subtract_match
                if (objfdu.inUse and not objfdu.hasProperty("sky_subtract_match")):
                    print("skySubtractSpecProcess::matchSkies> Warning: Odd frame "+objfdu.getFullId()+" ignored.  Sky not subtracted!")
                    self._log.writeLog(__name__, "Odd frame "+objfdu.getFullId()+" ignored.  Sky not subtracted!", type=fatboyLog.WARNING)
                    #Disable here
                    objfdu.disable()
                    continue
                objfdu.setProperty("use_only_positive", True)
        else:
            print("skySubtractSpecProcess::matchSkies> Error: Invalid sky subtract method "+skymethod+"!  Sky not subtracted!")
            self._log.writeLog(__name__, "Invalid sky subtract method "+skymethod+"!  Sky not subtracted!", type=fatboyLog.ERROR)
            return None
        return None
    #end matchSkies

    #Calculate a sky to subtract by using a median boxcar to filter each column (or row)
    def removeResiduals(self, fdu, calibs, prevProc):
        boxcar_width = int(self.getOption('boxcar_width', fdu.getTag()))
        boxcar_nhigh = int(self.getOption('boxcar_nhigh', fdu.getTag()))
        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None
        currMask = ones(fdu.getShape(), dtype=bool)
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            ###MOS/IFU data -- get slitmask
            slitmask = self.findSlitmask(fdu, calibs, prevProc)
            if (slitmask is None):
                #FDU will be disabled in execute
                return None
            if (not slitmask.hasProperty("nslits")):
                slitmask.setProperty("nslits", slitmask.getData().max())
            nslits = slitmask.getProperty("nslits")
            if (slitmask.hasProperty("regions")):
                (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

        #Select kernel for 2d median
        medFilt2dFunc = medianfilter2dCPU
        if (self._fdb.getGPUMode()):
            #Use GPU for median filter
            medFilt2dFunc = gpumedianfilter2d

        filtData = zeros(fdu.getShape(), dtype=float32)
        for j in range(nslits):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (slitmask is not None):
                    currMask = (slitmask.getData()[ylos[j]:yhis[j]+1] == j+1)
                slit = (fdu.getData()[ylos[j]:yhis[j]+1]).copy()*currMask
                m = medFilt2dFunc(slit, axis="Y", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                filtData[ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                if (slitmask is not None):
                    currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == j+1)
                slit = (fdu.getData()[:,ylos[j]:yhis[j]+1]).copy()*currMask
                m = medFilt2dFunc(slit, axis="X", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                filtData[:,ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]


        skyData = fdu.getData()-filtData
        #Update data
        fdu.updateData(filtData)
        fdu.tagDataAs("sky_residuals", skyData)

        if (fdu.hasProperty("cleanFrame")):
            #create "cleanFrame" for master sky too
            filtData = zeros(fdu.getShape(), dtype=float32)
            for j in range(nslits):
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (slitmask is not None):
                        currMask = (slitmask.getData()[ylos[j]:yhis[j]+1] == j+1)
                    slit = (fdu.getData(tag="cleanFrame")[ylos[j]:yhis[j]+1]).copy()*currMask
                    m = medFilt2dFunc(slit, axis="Y", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                    filtData[ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    if (slitmask is not None):
                        currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == j+1)
                    slit = (fdu.getData(tag="cleanFrame")[:,ylos[j]:yhis[j]+1]).copy()*currMask
                    m = medFilt2dFunc(slit, axis="X", boxsize=boxcar_width, nhigh=boxcar_nhigh)
                    filtData[:,ylos[j]:yhis[j]+1][currMask] = (m*currMask)[currMask]
            cleanSkyData = fdu.getData(tag="cleanFrame")-filtData
            #update data
            fdu.tagDataAs("cleanFrame", filtData)
            fdu.tagDataAs("clean_residuals", cleanSkyData)

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/skySubtracted", os.F_OK)):
                os.mkdir(outdir+"/skySubtracted",0o755)
            #Create output filename
            residfile = outdir+"/skySubtracted/resid_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(residfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(residfile)
            if (not os.access(residfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(residfile, tag="sky_residuals")
            if (fdu.hasProperty("clean_residuals")):
                residfile = outdir+"/skySubtracted/clean_resid_"+fdu.getFullId()
                #Check to see if it exists
                if (os.access(residfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(residfile)
                if (not os.access(residfile, os.F_OK)):
                    #Use fatboyDataUnit writeTo method to write
                    fdu.writeTo(residfile, tag="clean_residuals")

        fdu.removeProperty("sky_residuals")
        if (fdu.hasProperty("clean_residuals")):
            fdu.removeProperty("clean_residuals")
        return True
    #end removeResiduals


    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('boxcar_nhigh', '0')
        self._optioninfo.setdefault('boxcar_nhigh', 'Used in median_boxcar method - to use quartile rather than median\nset nhigh = width/2')
        self._options.setdefault('boxcar_width', '51')
        self._optioninfo.setdefault('boxcar_width', 'Used in median_boxcar method.  Maximum width = 51')
        self._options.setdefault('default_master_sky', None)
        self._options.setdefault('offsource_multi_dither_ncombine', '0')
        self._optioninfo.setdefault('offsource_multi_dither_ncombine', 'In the case of offsource_multi_dither, set to n short frames\nat each dither position.  E.g., for AAAAABBBBB pattern\nset to 5. Default = 0 => auto-detect')
        self._options.setdefault('remove_residuals', 'no')
        self._optioninfo.setdefault('remove_residuals','Attempt to remove residuals from sky subtraction by chosen method.')
        self._options.setdefault('residual_removal_method', 'median_boxcar')
        self._optioninfo.setdefault('residual_removal_method', 'median_boxcar | response_curve')
        self._options.setdefault('sky_method', "dither")
        self._optioninfo.setdefault('sky_method', 'dither | ifu_onsource_dither | median | median_boxcar | offsource_dither | offsource_multi_dither | step')
        self._options.setdefault('sky_dithering_range','2')
        self._options.setdefault('sky_offsource_range','240')
        self._options.setdefault('sky_offsource_method','auto')
        self._options.setdefault('double_subtract_odd_frames', 'no')
        self._options.setdefault('ignore_odd_frames', 'yes')
        self._options.setdefault('onsource_sorting_key', 'full') #full, index, or a FITS keyword to sort onsource skies by time
        self._optioninfo.setdefault('onsource_sorting_key', 'full | index | a FITS keyword to sort onsource skies, e.g. MJD')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## update crmask for spectroscopy data
    def updateCRMask(self, fdu, masterSky):
        #Need to save pre-sky subtracted data for later processing with odd frame
        if (fdu.hasProperty("odd_frame_match")):
            fdu.tagDataAs("crmask_preSkySubtracted", fdu.getProperty("crmask"))

        if (not masterSky.hasProperty("crmask")):
            #Hopefully we don't get here because this means we are reading a previous masterSky from disk with no corresponding crmask on disk
            #create tagged data "crmask"
            crmask = ones(masterSky.getShape(), int16)
            masterSky.tagDataAs("crmask", crmask)
        #Get this FDU's crmask
        crmask = fdu.getData(tag="crmask")
        #Propagate crmasks => they are good pixel masks so just multiply
        crmask *= masterSky.getData(tag="crmask")
        if (masterSky.hasProperty("crmask_preSkySubtracted")):
            crmask *= masterSky.getData(tag="crmask_preSkySubtracted")
        else:
            crmask *= masterSky.getData(tag="crmask")
        fdu.tagDataAs("crmask", crmask)
    #end updateCRMask

    ## update noisemap for spectroscopy data
    def updateNoisemap(self, fdu, masterSky):
        #Need to save pre-sky subtracted data for later processing with odd frame
        if (fdu.hasProperty("odd_frame_match")):
            fdu.tagDataAs("noisemap_preSkySubtracted", fdu.getProperty("noisemap"))

        if (not masterSky.hasProperty("noisemap")):
            #Hopefully we don't get here because this means we are reading a previous masterSky from disk with no corresponding noisemap on disk
            #create tagged data "noisemap"
            ncomb = 1.0
            if (masterSky.hasHeaderValue('NCOMBINE')):
                ncomb = float(masterSky.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterSky.getData(), ncomb)
            else:
                nm = sqrt(abs(masterSky.getData())/ncomb)
            masterSky.tagDataAs("noisemap", nm)
        #Get this FDU's noisemap
        nm = fdu.getData(tag="noisemap")
        #Propagate noisemaps.  For subtraction, dz = sqrt(dx^2 + dy^2)
        if (self._fdb.getGPUMode()):
            if (masterSky.hasProperty("noisemap_preSkySubtracted")):
                nm = noisemaps_ds_gpu(fdu.getData(tag="noisemap"), masterSky.getData(tag="noisemap_preSkySubtracted"))
            else:
                nm = noisemaps_ds_gpu(fdu.getData(tag="noisemap"), masterSky.getData(tag="noisemap"))
        else:
            if (masterSky.hasProperty("noisemap_preSkySubtracted")):
                nm = sqrt(fdu.getData(tag="noisemap")**2+masterSky.getData(tag="noisemap_preSkySubtracted")**2)
            else:
                nm = sqrt(fdu.getData(tag="noisemap")**2+masterSky.getData(tag="noisemap")**2)
        fdu.tagDataAs("noisemap", nm)
    #end updateNoisemap

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/skySubtracted", os.F_OK)):
            os.mkdir(outdir+"/skySubtracted",0o755)
        #Create output filename
        ssfile = outdir+"/skySubtracted/ss_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(ssfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(ssfile)
        if (not os.access(ssfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(ssfile)
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/skySubtracted/clean_ss_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/skySubtracted/NM_ss_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
        #Write crmask if requested
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and fdu.hasProperty("crmask")):
            crmfile = outdir+"/skySubtracted/crmask_ss_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(crmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(crmfile)
            if (not os.access(crmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(crmfile, tag="crmask")
    #end writeOutput
