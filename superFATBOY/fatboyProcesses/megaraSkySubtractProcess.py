from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_imcombine, imcombine

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

class megaraSkySubtractProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "megara"]

    ssmethods = ["dither", "ifu_onsource_dither", "median", "offsource_dither", "offsource_multi_dither", "step"]
    lastIdent = None #Track last identifier for onsource skies
    fduCount = 0 #Track count of fdus within this identifier
    identTotal = 0 #Track number of frames for this identifier

    #Calculate a sky to subtract by median combining all sky fibers
    def calculateMegaraSky(self, fdu, calibs, skyFibers, prevProc):
        combine_method = self.getOption('sky_combine_method', fdu.getTag()).lower()
        if (combine_method != 'mean' and combine_method != 'median'):
            print("megaraSkySubtractProcess::calculateMegaraSky> Warning: sky_combine_method "+combine_method+" is invalid.  Using median.")
            self._log.writeLog(__name__, "sky_combine_method "+combine_method+" is invalid.  Using median.", type=fatboyLog.WARNING)
            combine_method = 'median'

        slitmask = None
        if ('slitmask' in calibs):
            slitmask = calibs['slitmask']
        else:
            #Use findSlitletProcess.getCalibs to get slitmask and create if necessary
            #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
            #Returns None on a failure
            fs_process = self._fdb.getProcessByName("findSlitlets")
            if (fs_process is None or not isinstance(fs_process, fatboyProcess)):
                print("megaraSkySubtractProcess::calculateMedianSky> ERROR: could not find process findSlitlets!  Check your XML file!")
                self._log.writeLog(__name__, "could not find process findSlitlets!  Check your XML file!", type=fatboyLog.ERROR)
                return None
            #Call setDefaultOptions and getCalibs on megaraSkySubtractProcess
            fs_process.setDefaultOptions()
            calibs = fs_process.getCalibs(fdu, prevProc)
            if (not 'slitmask' in calibs):
                #Failed to obtain slitmask
                #Issue error message.  FDU will be disabled in execute()
                print("megaraSkySubtractProcess::calculateMegaraSky> ERROR: Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!")
                self._log.writeLog(__name__, "Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!", type=fatboyLog.ERROR)
                return None
            slitmask = calibs['slitmask']
            if (calibs['slitmask'].hasProperty("nslits")):
                calibs['nslits'] = calibs['slitmask'].getProperty("nslits")
            else:
                calibs['nslits'] = slitmask.getData().max()

        if (slitmask is None):
            print("megaraSkySubtractProcess::calculateMegaraSky> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to sky subtract!")
            self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to sky subtract!", type=fatboyLog.ERROR)
            return None

        #nslits has been added to calibs by here
        nslits = calibs['nslits']
        #Use helper method to find all ylo, yhi for each slit in each frame
        (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)

        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        skyData = []
        cleanData = []
        nmData = []
        #Loop over bottom, top sections
        for section in skyFibers:
            fiberList = skyFibers[section]
            if (len(fiberList) == 0):
                print("megaraSkySubtractProcess::calculateMegaraSky> WARNING: No sky fibers found for section "+str(section)+" of "+fdu.getFullId())
                self._log.writeLog(__name__, "No sky fibers found for section "+str(section)+" of "+fdu.getFullId(), type=fatboyLog.WARNING)
                continue

            print("megaraSkySubtractProcess::calculateMegaraSky> Section "+str(section)+": Using fibers "+str(fiberList)+" for "+fdu.getFullId())
            self._log.writeLog(__name__, "Section "+str(section)+": Using fibers "+str(fiberList)+" for "+fdu.getFullId())
            #Get data for all sky fibers for this section into a list to imcombine
            fibers = []
            cleanFibers = []
            nms = []
            for idx in fiberList:
                j = idx-1 #index starts at 0 for ylos, yhis
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    #currMask = ones(fdu.getData(tag="resampled")[ylos[j]:yhis[j]+1,:].shape, dtype=bool)
                    #fiber = (fdu.getData(tag="resampled")[ylos[j]:yhis[j]+1,:]).copy()*currMask
                    currMask = (slitmask.getData()[ylos[j]:yhis[j]+1,:] == idx)
                    fiber = (fdu.getData()[ylos[j]:yhis[j]+1,:]).copy()*currMask
                    fiber = fiber.sum(0)
                    fiber = fiber.reshape((1, fiber.size))
                    if (fdu.hasProperty("cleanFrame")):
                        cf = (fdu.getData(tag="cleanFrame")[ylos[j]:yhis[j]+1,:]).copy()*currMask
                        cf = cf.sum(0)
                        cf = cf.reshape((1, cf.size))
                        cleanFibers.append(cf)
                    if (fdu.hasProperty("noisemap")):
                        #Square noisemap
                        nm = ((fdu.getData(tag="noisemap")[ylos[j]:yhis[j]+1,:]).copy()*currMask)**2
                        nm = nm.sum(0)
                        nm = nm.reshape((1, nm.size))
                        nms.append(nm)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    #currMask = ones(fdu.getData(tag="resampled")[ylos[j]:yhis[j]+1,:].shape, dtype=bool)
                    #fiber = (fdu.getData(tag="resampled")[:,ylos[j]:yhis[j]+1]).copy()*currMask
                    currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == idx)
                    fiber = (fdu.getData()[:,ylos[j]:yhis[j]+1]).copy()*currMask
                    fiber = fiber.sum(1)
                    fiber = fiber.reshape((fiber.size, 1))
                    if (fdu.hasProperty("cleanFrame")):
                        cf = (fdu.getData(tag="cleanFrame")[:,ylos[j]:yhis[j]+1]).copy()*currMask
                        cf = cf.sum(1)
                        cf = cf.reshape((cf.size, 1))
                        cleanFibers.append(cf)
                    if (fdu.hasProperty("noisemap")):
                        #Square noisemap
                        nm = ((fdu.getData(tag="noisemap")[:,ylos[j]:yhis[j]+1]).copy()*currMask)**2
                        nm = nm.sum(1)
                        nm = nm.reshape((nm.size, 1))
                        nms.append(nm)
                fibers.append(fiber)

            #imcombine object files and do NOT scale by median
            (skySection, header) = imcombine_method(fibers, outfile=None, method=combine_method, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_RAW)
            skyData.append(skySection)
            if (fdu.hasProperty("cleanFrame")):
                (cleanSection, header) = imcombine_method(cleanFibers, outfile=None, method=combine_method, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_RAW)
                cleanData.append(cleanSection)
            if (fdu.hasProperty("noisemap")):
                if (combine_method == 'median'):
                    #For median, sqrt(sky/n)
                    nmData.append(sqrt(skySection/len(fibers)))
                else:
                    #For mean, dz = sqrt(sum(dx_i^2)/n)
                    (nmSection, header) = imcombine_method(nms, outfile=None, method="sum", log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_RAW)
                    nmData.append(sqrt(nmSection/len(fibers)))

        if (len(skyData) > 1):
            #3D array
            skyData = array(skyData)
        else:
            #If only 1 sky created, make a 2d array
            skyData = skyData[0]

        msname = "sky_"+fdu.getFullId()
        masterSky = fatboySpecCalib(self._pname, "master_sky", fdu, data=skyData, tagname=msname, log=self._log)
        if (fdu.hasProperty("cleanFrame")):
            if (len(cleanData) > 1):
                #3D array
                cleanData = array(cleanData)
            else:
                #If only 1 sky created, make a 2d array
                cleanData = cleanData[0]
            #Tag noisemap
            masterSky.tagDataAs("cleanFrame", cleanData)

        if (fdu.hasProperty("noisemap")):
            if (len(nmData) > 1):
                #3D array
                nmData = array(nmData)
            else:
                #If only 1 sky created, make a 2d array
                nmData = nmData[0]
            #Tag noisemap
            masterSky.tagDataAs("noisemap", nmData)

        #set specmode property
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/masterSkies", os.F_OK)):
                os.mkdir(outdir+"/masterSkies",0o755)
            msfile = outdir+"/masterSkies/"+msname
            #Check to see if it exists
            if (os.access(msfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(msfile)
            if (not os.access(msfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                masterSky.writeTo(msfile)
            cleanFile = outdir+"/masterSkies/clean_"+msname
            #Check to see if it exists
            if (os.access(cleanFile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanFile)
            if (not os.access(cleanFile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                masterSky.writeTo(cleanFile, tag="cleanFrame")

        #Write out noisemap if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes"):
            if (not os.access(outdir+"/masterSkies", os.F_OK)):
                os.mkdir(outdir+"/masterSkies",0o755)
            nmfile = outdir+"/masterSkies/NM_"+msname
            if (os.access(nmfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                masterSky.writeTo(nmfile, tag="noisemap")
        return masterSky

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            #If sky subtract is done before flat divide, it will attempt to
            #recursively process flats.  Make sure it only tries to sky subtract objects
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_CONTINUUM_SOURCE):
            #Also sky subtract for continuum source calibs
            return True
        return False
    #end checkValidDatatype

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Sky Subtract")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For skySubtract, this dict should have one entry 'masterSky' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'masterSky' in calibs):
            #Failed to obtain master sky frame
            #Issue error message and disable this FDU
            print("megaraSkySubtractProcess::execute> ERROR: Sky not subtracted for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
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
            #Disable masterSky if marked
            if (masterSky.hasProperty("disable")):
                masterSky.disable()
            return True

        success = self.skySubtract(fdu, calibs, masterSky)

#    #Propagate noisemap
#    if (fdu.hasProperty("noisemap")):
#      self.updateNoisemap(fdu, masterSky)
#
#    #subtract master sky and if "cleanFrame" exists, propagate it too
#    fdu.updateData(float32(fdu.getData())-float32(masterSky.getData(tag="preSkySubtracted")))
#    if (fdu.hasProperty("cleanFrame")):
#      #If masterSky has tag cleanFrame_preSkySubtracted then use it.  This is an odd frame
#      if (masterSky.hasProperty("cleanFrame_preSkySubtracted")):
#        fdu.tagDataAs("cleanFrame", fdu.getData(tag="cleanFrame")-masterSky.getData(tag="cleanFrame_preSkySubtracted"))
#      else:
#       #Otherwise subtract cleanFrame tags from each other
#       fdu.tagDataAs("cleanFrame", fdu.getData(tag="cleanFrame")-masterSky.getData(tag="cleanFrame"))
#
#    #Update history
#    fdu._header.add_history('Sky subtracted using '+masterSky._id)
#    #Disable masterSky if marked
#    if (masterSky.hasProperty("disable")):
#      masterSky.disable()
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        msfilename = self.getCalib("masterSky", fdu.getTag())
        if (msfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(msfilename, os.F_OK)):
                print("megaraSkySubtractProcess::getCalibs> Using master sky "+msfilename+"...")
                self._log.writeLog(__name__, "Using master sky "+msfilename+"...")
                calibs['masterSky'] = fatboySpecCalib(self._pname, "master_sky", fdu, filename=msfilename, log=self._log)
                return calibs
            else:
                print("megaraSkySubtractProcess::getCalibs> Warning: Could not find master sky "+msfilename+"...")
                self._log.writeLog(__name__, "Could not find master sky "+msfilename+"...", type=fatboyLog.WARNING)

        #Look for slitmask
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("megaraSkySubtractProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("megaraSkySubtractProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and sky_method
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            #Find slitmask associated with this fdu
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is None):
                print("megaraSkySubtractProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to sky subtract!")
                self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to sky subtract!", type=fatboyLog.ERROR)
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
                print("megaraSkySubtractProcess::getCalibs> Using default master sky "+masterSky.getFilename())
                self._log.writeLog(__name__, "Using default master sky "+masterSky.getFilename())
                self._fdb.appendCalib(masterSky)
                calibs['masterSky'] = masterSky
                return calibs
        #4) Calculate sky from MEGARA data if fibers have been identified
        if (fdu.hasFibers()):
            #Get fibers by individual chip
            bottomFibers = fdu.getSkyFiberIndices(section=0)
            topFibers = fdu.getSkyFiberIndices(section=1)
            nfibers = len(bottomFibers)+len(topFibers)
            skyFibers = dict()
            skyFibers[0] = bottomFibers
            skyFibers[1] = topFibers
            if (nfibers == 0):
                print("megaraSkySubtractProcess::getCalibs> Warning: Fibers have been identified but NONE are sky.  Cannot create master sky.")
                self._log.writeLog(__name__, "Fibers have been identified but NONE are sky.  Cannot create master sky.", type=fatboyLog.WARNING)
                return calibs
            print("megaraSkySubtractProcess::getCalibs> Found "+str(nfibers)+" sky fibers to median combine.")
            self._log.writeLog(__name__, "Found "+str(nfibers)+" sky fibers to median combine.")
            masterSky = self.calculateMegaraSky(fdu, calibs, skyFibers, prevProc)
            if (masterSky is not None):
                calibs['masterSky'] = masterSky
            return calibs
        else:
            print("megaraSkySubtractProcess::getCalibs> Warning: Fibers have NOT been identified.  Cannot create master sky.  Run megaraIdentifyFibers.")
            self._log.writeLog(__name__, "Fibers have NOT been identified.  Cannot create master sky. Run megaraIdentifyFibers.", type=fatboyLog.WARNING)
            return calibs
        print("megaraSkySubtractProcess::getCalibs> ERROR: No skies found for "+fdu.getFullId())
        self._log.writeLog(__name__, "No skies found for "+fdu.getFullId(), type=fatboyLog.ERROR)
        return calibs
    #end getCalibs

    def getPeakFactor(self, fiber, skyData):
        match = False
        #remove points near edge
        fiber[:100] = 0
        fiber[-100:] = 0
        skyData[:100] = 0
        skyData[-100:] = 0
        fpts = fiber.argsort()[::-1]
        spts = skyData.argsort()[::-1]
        i = 0
        while (not match):
            #Check if this pixel is in 10 brightest in sky
            if (fpts[i] in spts[:10]):
                match = True
                offset = 0
            elif (fpts[i]-1 in spts[:10]):
                #Also allow for +/-1 pixel
                match = True
                offset = -1
            elif (fpts[i]+1 in spts[:10]):
                match = True
                offset = 1
            else:
                i += 1
        #scale = fiber[fpts[i]]/skyData[fpts[i]]
        scale = fiber[fpts[i]]/skyData[fpts[i]+offset]
        #print "\tpeak index matched ",fpts[i], "scale =", scale, i, fiber[fpts[i]], skyData[fpts[i]+offset]
        return scale
    #end getPeakFactor

    def getSkylineFactor(self, fiber, skyData, nlines):
        #remove points near edge
        fiber[:100] = 0
        fiber[-100:] = 0
        skyData[:100] = 0
        skyData[-100:] = 0
        fpts = fiber.argsort()[::-1]
        spts = skyData.argsort()[::-1]
        unique_spts = []
        i = 0

        #Find brightest peaks in sky at least 10 pixels away from each other
        while (len(unique_spts) < 10):
            keep = True
            for j in range(len(unique_spts)):
                if (abs(unique_spts[j]-spts[i]) < 10):
                    keep = False
                    continue
            if (keep):
                unique_spts.append(spts[i])
            i += 1

        n = 0
        i = 0
        while (n < 3):
            #Check if this pixel is in 10 brightest in sky
            if (fpts[i] in unique_spts):
                n += 1
            elif (fpts[i]-1 in unique_spts or fpts[i]+1 in unique_spts):
                #Also allow for +/-1 pixel
                n += 1
            else:
                #Zero out this pixel
                fiber[fpts[i]] = 0
            i += 1

        fiberFits = fitLines(fiber, nlines, edge=100)
        #use linesToMatch to make sure we are fitting the same lines
        skyFits = fitLines(skyData, nlines, linesToMatch=fiberFits, edge=100)
        #Sum the (peak-background) for each and scale based on that
        fiberPeaks = 0.
        skyPeaks = 0.
        for j in range(nlines):
            fiberPeaks += fiberFits[j][0]
            skyPeaks += skyFits[j][0]
        return fiberPeaks / skyPeaks
    #end getSkylineFactor

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('debug_mode', 'no')
        self._options.setdefault('default_master_sky', None)
        self._options.setdefault('keep_skies', 'no')
        self._options.setdefault('scaling', 'none')
        self._optioninfo.setdefault('scaling', 'none | peak | skylines')
        self._options.setdefault('scaling_nlines', '3')
        self._optioninfo.setdefault('scaling_nlines', 'Number of skylines to use for scaling')
        self._options.setdefault('sky_combine_method', 'median')
        self._optioninfo.setdefault('sky_combine_method', 'median | mean')
        self._options.setdefault('write_noisemaps', 'no')
        self._options.setdefault('write_plots', 'no')
    #end setDefaultOptions

    ## perform sky subtraction
    def skySubtract(self, fdu, calibs, masterSky):
        if (not fdu.hasFibers()):
            print("megaraSkySubtractProcess::skySubtract> Warning: Fibers have NOT been identified.  Cannot subtract sky.  Run megaraIdentifyFibers.")
            self._log.writeLog(__name__, "Fibers have NOT been identified.  Cannot subtract sky. Run megaraIdentifyFibers.", type=fatboyLog.WARNING)
            return False

        if ('slitmask' in calibs):
            slitmask = calibs['slitmask']
        else:
            print("megaraSkySubtractProcess::skySubtract> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to sky subtract!")
            self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to sky subtract!", type=fatboyLog.ERROR)
            return False

        scaling = self.getOption("scaling", fdu.getTag()).lower()
        nlines = int(self.getOption("scaling_nlines", fdu.getTag()))
        keepSkies = False
        if (self.getOption("keep_skies", fdu.getTag()).lower() == "yes"):
            keepSkies = True

        nslits = calibs['nslits']
        #Use helper method to all ylo, yhi for each slit in each frame
        (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)

        doNM = False
        if (masterSky.hasProperty("noisemap") and fdu.hasProperty("noisemap")):
            doNM = True

        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/skySubtracted", os.F_OK)):
            os.mkdir(outdir+"/skySubtracted",0o755)

        skyData = masterSky.getData()
        fiberList = fdu.getObjectFiberIndices()
        if (keepSkies):
            fiberList.extend(fdu.getSkyFiberIndices())
            print("megaraSkySubtractProcess::skySubtract> Sky subtracting "+str(len(fiberList))+" object AND sky fibers...")
            self._log.writeLog(__name__, "Sky subtracting "+str(len(fiberList))+" object AND sky fibers...")
        else:
            print("megaraSkySubtractProcess::skySubtract> Sky subtracting "+str(len(fiberList))+" object fibers only...")
            self._log.writeLog(__name__, "Sky subtracting "+str(len(fiberList))+" object fibers only...")
        ssData = zeros(fdu.getData().shape, dtype=float32)
        outmask = zeros(fdu.getData().shape, dtype=int16)
        if (fdu.hasProperty("cleanFrame")):
            cleanSSData = zeros(fdu.getData().shape, dtype=float32)
        if (doNM):
            nmSSData = zeros(fdu.getData().shape, dtype=float32)
        for idx in fiberList:
            j = idx-1 #index starts at 0 for ylos, yhis
            if (len(skyData.shape) == 3):
                #Both top and bottom skies are available
                if (fdu.getFiber(idx).getSection() == 0):
                    #bottom half
                    skyData = masterSky.getData()[0,:,:]
                    cleanData = masterSky.getData(tag="cleanFrame")[0,:,:]
                    if (doNM):
                        nmData = masterSky.getData(tag="noisemap")[0,:,:]
                else:
                    #top half
                    skyData = masterSky.getData()[1,:,:]
                    cleanData = masterSky.getData(tag="cleanFrame")[1,:,:]
                    if (doNM):
                        nmData = masterSky.getData(tag="noisemap")[1,:,:]
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                #currMask = ones(fdu.getData(tag="resampled")[ylos[j]:yhis[j]+1,:].shape, dtype=bool)
                #fiber = (fdu.getData(tag="resampled")[ylos[j]:yhis[j]+1,:]).copy()
                currMask = (slitmask.getData()[ylos[j]:yhis[j]+1,:] == idx)
                fiber = (fdu.getData()[ylos[j]:yhis[j]+1,:]).copy()
                cleanFiber = (fdu.getData(tag="cleanFrame")[ylos[j]:yhis[j]+1,:]).copy()
                if (doNM):
                    nmFiber = (fdu.getData(tag="noisemap")[ylos[j]:yhis[j]+1,:]).copy()
                #plt.plot(fiber[0,:])
                #plt.plot(skyData[0,:])
                #plt.show()
                #subtract sky and copy to ssData
                #multiply by currMask after subtracting skyData as double check
                #to ensure that everything outside this fiber is 0 and use +=
                scale_factor = 1
                clean_factor = 1
                if (scaling == "peak"):
                    scale_factor = self.getPeakFactor(fiber.sum(0), skyData.sum(0))
                    clean_factor = self.getPeakFactor(cleanFiber.sum(0), cleanData.sum(0))
                elif (scaling == "skylines"):
                    #Sum to 1-d cuts before fitting skylines
                    scale_factor = self.getSkylineFactor(fiber.sum(0), skyData.sum(0), nlines)
                    clean_factor = self.getSkylineFactor(cleanFiber.sum(0), cleanData.sum(0), nlines)
                ssData[ylos[j]:yhis[j]+1,:] += (fiber-skyData*scale_factor)*currMask
                cleanSSData[ylos[j]:yhis[j]+1,:] += (cleanFiber-cleanData*clean_factor)*currMask
                if (doNM):
                    nmSSData[ylos[j]:yhis[j]+1,:] += sqrt(nmFiber**2 + (nmData*scale_factor)**2)*currMask
                #update outmask
                outmask[ylos[j]:yhis[j]+1,:][currMask] = idx

                if (usePlot and (self.getOption("debug_mode", fdu.getTag()).lower() == "yes" or self.getOption("write_plots", fdu.getTag()).lower() == "yes")):
                    plt.plot(fiber.sum(0), '#1f77b4')
                    plt.plot(skyData.sum(0), '#ff7f0e')
                    plt.plot(skyData.sum(0)*scale_factor, '#2ca02c')
                    plt.legend(['Data', 'Raw Sky', 'Scaled Sky'], loc=2)
                    plt.xlabel('Pixel')
                    plt.ylabel('Flux')
                    if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
                        pltfile = outdir+"/skySubtracted/qa_"+fdu._id+"_fiber_"+str(idx)+".png"
                        plt.savefig(pltfile, dpi=200)
                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        plt.show()
                    plt.close()
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #currMask = ones(fdu.getData(tag="resampled")[ylos[j]:yhis[j]+1,:].shape, dtype=bool)
                #fiber = (fdu.getData(tag="resampled")[:,ylos[j]:yhis[j]+1]).copy()
                currMask = (slitmask.getData()[:,ylos[j]:yhis[j]+1] == idx)
                fiber = (fdu.getData()[:,ylos[j]:yhis[j]+1]).copy()
                cleanFiber = (fdu.getData(tag="cleanFrame")[:,ylos[j]:yhis[j]+1]).copy()
                if (doNM):
                    nmFiber = (fdu.getData(tag="noisemap")[:,ylos[j]:yhis[j]+1]).copy()
                #subtract sky and copy to ssData
                #multiply by currMask after subtracting skyData as double check
                #to ensure that everything outside this fiber is 0 and use +=
                scale_factor = 1
                if (scaling == "peak"):
                    scale_factor = self.getPeakFactor(fiber.sum(1), skyData.sum(1))
                elif (scaling == "skylines"):
                    #Sum to 1-d cuts before fitting skylines
                    scale_factor = self.getSkylineFactor(fiber.sum(1), skyData.sum(1), nlines)
                ssData[:,ylos[j]:yhis[j]+1] += (fiber-skyData*scale_factor)*currMask
                cleanSSData[:,ylos[j]:yhis[j]+1] += (cleanFiber-cleanData*clean_factor)*currMask
                if (doNM):
                    nmSSData[:,ylos[j]:yhis[j]+1] += sqrt(nmFiber**2 + (nmData*scale_factor)**2)*currMask
                #update outmask
                outmask[:,ylos[j]:yhis[j]+1][currMask] = idx

                if (usePlot and (self.getOption("debug_mode", fdu.getTag()).lower() == "yes" or self.getOption("write_plots", fdu.getTag()).lower() == "yes")):
                    plt.plot(fiber.sum(1), '#1f77b4')
                    plt.plot(skyData.sum(1), '#ff7f0e')
                    plt.plot(skyData.sum(1)*scale_factor, '#2ca02c')
                    plt.legend(['Data', 'Raw Sky', 'Scaled Sky'], loc=2)
                    plt.xlabel('Pixel')
                    plt.ylabel('Flux')
                    if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
                        pltfile = outdir+"/skySubtracted/qa_"+fdu._id+"_fiber_"+str(idx)+".png"
                        plt.savefig(pltfile, dpi=200)
                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        plt.show()
                    plt.close()
        #Update data
        fdu.updateData(ssData)
        #Tag slitmask because the calib slitmask needs to have skies in it for next frame
        #Update property "slitmask" with outmask
        #Use new fdu.setSlitmask
        fdu.setSlitmask(outmask, pname=self._pname)
        fdu.tagDataAs("cleanFrame", cleanSSData)
        if (doNM):
            fdu.tagDataAs("noisemap", nmSSData)
    #end skySubtract

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
                nm = sqrt(masterSky.getData()/ncomb)
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
    #end writeOutput
