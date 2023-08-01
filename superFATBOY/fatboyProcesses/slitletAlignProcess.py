from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_drihizzle, drihizzle
from numpy import *
from scipy.optimize import leastsq

block_size = 512

class slitletAlignProcess(fatboyProcess):
    _modeTags = ["spectroscopy"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            #Skip longslit data
            return True

        print("Slitlet Align")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        safile = "slitletAligned/sa_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, safile)):
            #Also check if "cleanFrame" exists
            cleanfile = "slitletAligned/clean_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "exposure map" exists
            expfile = "slitletAligned/exp_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, expfile, tag="exposure_map")
            #Also check if "slitmask" exists
            smfile = "slitletAligned/slitmask_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, smfile, tag="slitmask")
            #Also check if "noisemap" exists
            nmfile = "slitletAligned/NM_sa_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")

            #Need to get calibration frames - cleanSky, masterLamp, and slitmask to update from disk too
            calibs = dict()
            headerVals = dict()
            headerVals['grism_keyword'] = fdu.grism
            properties = dict()
            properties['specmode'] = fdu.getProperty("specmode")
            properties['dispersion'] = fdu.getProperty("dispersion")
            if (not 'cleanSky' in calibs):
                #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
                cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (cleanSky is not None):
                    #add to calibs for rectification below
                    calibs['cleanSky'] = cleanSky

            if (not 'masterLamp' in calibs):
                #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
                masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
                if (masterLamp is None):
                    #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                    masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (masterLamp is not None):
                    #add to calibs for rectification below
                    calibs['masterLamp'] = masterLamp

            #Check for cleanSky and masterLamp frames to update from disk too
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("slitletAligned")):
                #Check if output exists
                safile = "slitletAligned/sa_"+calibs['cleanSky']._id+".fits"
                if (self.checkOutputExists(calibs['cleanSky'], safile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "slitletAligned" = True
                    calibs['cleanSky'].setProperty("slitletAligned", True)
                    #Also check if "resampled" exists
                    resampfile = "slitletAligned/resamp_sa_"+calibs['cleanSky']._id+".fits"
                    self.checkOutputExists(calibs['cleanSky'], resampfile, tag="resampled")

            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("slitletAligned")):
                #Check if output exists first
                safile = "slitletAligned/sa_"+calibs['masterLamp'].getFullId()
                if (self.checkOutputExists(calibs['masterLamp'], safile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "slitletAligned" = True
                    calibs['masterLamp'].setProperty("slitletAligned", True)
                    #Also check if "resampled" exists
                    resampfile = "slitletAligned/resamp_sa_"+calibs['masterLamp']._id+".fits"
                    self.checkOutputExists(calibs['masterLamp'], resampfile, tag="resampled")
            return True

        #Call get calibs to return dict() of calibration frames.
        #For slitletAligned, this dict should have cleanSky and/or masterLamp
        #and optionally slitmask if this is not a property of the FDU at this point.
        #These are obtained by tracing slitlets using the master flat
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'slitmask' in calibs or (not 'masterLamp' in calibs and not 'cleanSky' in calibs)):
            #Failed to obtain slitmask or (master lamp or cleanSky)
            #Issue error message and disable this FDU
            print("slitletAlignProcess::execute> ERROR: Slitlets not aligned for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Slitlets not aligned for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #call alignSlitlets helper function to do gpu/cpu calibration
        self.alignSlitlets(fdu, calibs)
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
                print("slitletAlignProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("slitletAlignProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        csfilename = self.getCalib("master_clean_sky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("slitletAlignProcess::getCalibs> Using master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using master clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, log=self._log)
            else:
                print("slitletAlignProcess::getCalibs> Warning: Could not find master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find master clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("slitletAlignProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("slitletAlignProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
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
            cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (cleanSky is not None):
                #add to calibs for rectification below
                calibs['cleanSky'] = cleanSky
                skyShape = cleanSky.getShape()

        if (not 'masterLamp' in calibs):
            #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", section=fdu.section, filter=fdu.filter, properties=properties, headerVals=headerVals)
            if (masterLamp is None):
                #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is not None):
                #add to calibs for rectification below
                calibs['masterLamp'] = masterLamp
                skyShape = masterLamp.getShape()

        if (not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('debug_mode', 'no')
        self._optioninfo.setdefault('debug_mode', 'Show plots of each slitlet and print out debugging information.')
        self._options.setdefault('fit_order', '3')
        self._optioninfo.setdefault('fit_order', 'Order of polynomial to use to fit wavelength solution.\nRecommended value = 3.')
        self._options.setdefault('max_lines', None)
        self._optioninfo.setdefault('max_lines', 'Maximum number of lines to fit for each segment.\nCan be a single value or comma separated list\nfor each segment. Default value\nof None will impose no maximum.')
        self._options.setdefault('min_threshold', '3,2')
        self._optioninfo.setdefault('min_threshold', 'Minimum local sigma threshold compared to noise for line\nto be detected. Can be a single value\nor comma separated list for each segment.')
        self._options.setdefault('nebular_emission_check', 'no')
        self._optioninfo.setdefault('nebular_emission_check', 'In rare cases, your data may have nebular emission\nlines that appear in some slits but not in others\nand may interfere with aligning the slits properly.\nTurn this option to yes for additional\nnebular emission check.')
        self._options.setdefault('n_segments', '2')
        self._optioninfo.setdefault('n_segments', 'Number of segments, default = 2. In JH data, the sky lines\nin H are much brighter so its best to separately\nmatch J lines with lower threshold.')
        self._options.setdefault('reference_slit', None)
        self._optioninfo.setdefault('reference_slit', 'Reference slitlet to be used. Ideally this slitlet\nshould be the centermost slitlet. Default value of None\nwill select slitlet with central x_slit.\nCan also be a number or "prompt".')
        self._options.setdefault('reverse_order_of_segments', 'no')
        self._optioninfo.setdefault('reverse_order_of_segments', 'Process segments right to left instead of left\nto right. Useful when nebular emission\ncontaminates bright lines in leftmost segment.')
        self._options.setdefault('use_arclamps', 'no')
        self._optioninfo.setdefault('use_arclamps', 'no = use master "clean sky", yes = use master arclamp')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## alignSlitlets function does actual aligning
    def alignSlitlets(self, fdu, calibs):
        ###*** For purposes of alignSlitlets algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        #Read options
        useArclamps = False
        writeCalibs = False
        reverseOrder = False
        nebularEmission = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        if (self.getOption("reverse_order_of_segments", fdu.getTag()).lower() == "yes"):
            reverseOrder = True
        if (self.getOption("nebular_emission_check", fdu.getTag()).lower() == "yes"):
            nebularEmission = True

        n_segments = int(self.getOption("n_segments", fdu.getTag()))
        fit_order = int(self.getOption("fit_order", fdu.getTag()))
        #Min thresh should be list
        min_thresh = self.getOption("min_threshold", fdu.getTag()).split(",")
        for j in range(len(min_thresh)):
            min_thresh[j] = int(min_thresh[j])
        #Max lines could be None or list
        max_lines = self.getOption("max_lines", fdu.getTag())
        if (max_lines is not None):
            max_lines = max_lines.split(",")
            for j in range(len(max_lines)):
                max_lines[j] = int(max_lines[j])
        reference_slit = self.getOption("reference_slit", fdu.getTag())

        ###MOS/IFU data -- get slitmask
        #Use calibration frame slitmask, which corresponds to master lamp and clean sky
        #And not "slitmask" property of each FDU which has been shifted and added and has different shape
        slitmask = calibs['slitmask']
        if (not slitmask.hasProperty("nslits")):
            slitmask.setProperty("nslits", slitmask.getData().max())
        nslits = slitmask.getProperty("nslits")
        if (slitmask.hasProperty("regions")):
            (sylo, syhi, slitx, slitw) = slitmask.getProperty("regions")
        else:
            #Get region file for this FDU
            if (fdu.hasProperty("region_file")):
                regFile = fdu.getProperty("region_file")
            else:
                regFile = self.getCalib("region_file", fdu.getTag())
            #Check that region file exists
            if (regFile is None or not os.access(regFile, os.F_OK)):
                print("slitletAlignProcess::alignSlitlets> ERROR: Could not find region file associated with "+fdu.getFullId()+"! Discarding Image!")
                self._log.writeLog(__name__, "Could not find region file associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return

            #Read region file
            if (regFile.endswith(".reg")):
                (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".txt")):
                (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".xml")):
                (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            else:
                print("slitletAlignProcess::alignSlitlets> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return
            slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))

        #Next check for master arclamp frame or clean sky frame to rectify skylines
        #These should have been found above and added to calibs dict
        skyFDU = None
        if (useArclamps):
            if ('masterLamp' in calibs):
                skyFDU = calibs['masterLamp']
            else:
                print("slitletAlignProcess::slitletAlign> Warning: Could not find master arclamp associated with "+fdu.getFullId()+"! Attempting to use clean sky frame for wavelength calibration!")
                self._log.writeLog(__name__, "Could not find master arclamp associated with "+fdu.getFullId()+"! Attempting to use clean sky frame for wavelength calibration!", type=fatboyLog.WARNING)
                useArclamps = False
        if (skyFDU is None and 'cleanSky' in calibs):
            #Either use_arclamps = no or master arclamp not found
            skyFDU = calibs['cleanSky']
        if (skyFDU is None):
            print("slitletAlignProcess::slitletAlign> ERROR: Could not find clean sky frame associated with "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not find clean sky frame associated with "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = skyFDU.getShape()[1]
            ysize = skyFDU.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = skyFDU.getShape()[0]
            ysize = skyFDU.getShape()[1]

        #Create output dir if it doesn't exist
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/slitletAligned", os.F_OK)):
            os.mkdir(outdir+"/slitletAligned",0o755)

        #Get sky data for qadata
        qadata = skyFDU.getData().copy()
        #Min and max slit values
        sxmin = int(slitx.min())
        sxmax = int(slitx.max())+1
        #Create fitParams list to store fit parameters for resampling data
        fitParams = []
        onedcuts = []
        xoffInit = []

        #Loop over nslits
        for j in range(nslits):
            #Take 1-d cut of skyFDU (cleanSky or masterLamp) data
            #Ensure no negative index values
            if (sylo[j] < 0):
                sylo[j] = 0
            currMask = None

            #Select kernel for 2d median
            kernel2d = fatboyclib.median2d
            if (self._fdb.getGPUMode()):
                #Use GPU for medians
                kernel2d=gpumedian2d

            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                slit = skyFDU.getData()[sylo[j]:syhi[j]+1,:].copy()
                if (slitmask is not None):
                    #Apply mask to slit - based on if individual slitlets are being calibrated or not
                    currMask = slitmask.getData()[sylo[j]:syhi[j]+1,:] == (j+1)
                    slit *= currMask
                if (self._fdb.getGPUMode()):
                    #Use GPU
                    oned = gpu_arraymedian(slit, axis="Y", nonzero=True, kernel2d=kernel2d)
                else:
                    #Use CPU
                    oned = kernel2d(slit.transpose().copy(), nonzero=True)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                slit = skyFDU.getData()[:,sylo[j]:syhi[j]+1].copy()
                if (slitmask is not None):
                    #Apply mask to slit - based on if individual slitlets are being calibrated or not
                    currMask = slitmask.getData()[:,sylo[j]:syhi[j]+1] == (j+1)
                    slit *= currMask
                oned = gpu_arraymedian(slit, axis="X", nonzero=True, kernel2d=kernel2d)

            #Filter the 1-d cut!
            #Use quartile instead of median to get better estimate of background levels!
            #Use 2 passes of quartile filter
            badpix = oned == 0 #First find bad pixels
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
            #Append to onedcuts
            onedcuts.append(oned)
            #Append xoffset
            xoffInit.append(sxmax-int(slitx[j]))

        #Select reference slit
        xoffInit = array(xoffInit)
        if (reference_slit is None):
            #Default option, choose middle xslit value
            #Don't use bottom or top slitlet so search for middle slitlet from [1:-1]
            reference_slit = where(abs(xoffInit[1:-1]-gpu_arraymedian(xoffInit[1:-1])) == min(abs(xoffInit[1:-1]-gpu_arraymedian(xoffInit[1:-1]))))[0][0]+1
            #Add 1 back to index since where searched [1:-1]
        else:
            if (isinstance(reference_slit, str) and reference_slit.lower() == "prompt"):
                #Prompt user
                print('Select a reference slit to use for aligning slitlets in '+skyFDU.getFullId())
                print('The bottom slit is slit 0.')
                reference_slit = input("Select a slit: ")
            try:
                reference_slit = int(reference_slit)
                if (reference_slit < 0 or reference_slit >= nslits):
                    print("slitletAlignProcess::slitletAlign> Warning: invalid reference slit "+str(reference_slit))
                    self._log.writeLog(__name__, "invalid reference slit "+str(reference_slit), type=fatboyLog.WARNING)
                    reference_slit = where(abs(xoffInit[1:-1]-gpu_arraymedian(xoffInit[1:-1])) == min(abs(xoffInit[1:-1]-gpu_arraymedian(xoffInit[1:-1]))))[0][0]+1
            except Exception as ex:
                print("slitletAlignProcess::slitletAlign> Warning: invalid reference slit "+str(reference_slit)+" - "+str(ex))
                self._log.writeLog(__name__, "invalid reference slit "+str(reference_slit)+" - "+str(ex), type=fatboyLog.WARNING)
                reference_slit = where(abs(xoffInit[1:-1]-gpu_arraymedian(xoffInit[1:-1])) == min(abs(xoffInit[1:-1]-gpu_arraymedian(xoffInit[1:-1]))))[0][0]+1
        print("slitletAlignProcess::slitletAlign> Using slit "+str(reference_slit)+" ["+str(sylo[reference_slit])+":"+str(syhi[reference_slit])+"] as reference slit to align slitlets in "+skyFDU.getFullId())
        self._log.writeLog(__name__, "Using slit "+str(reference_slit)+" ["+str(sylo[reference_slit])+":"+str(syhi[reference_slit])+"] as reference slit to align slitlets in "+skyFDU.getFullId())

        #Set up arrays
        xoffsets = []
        validOffsets = []
        reflines = []
        validlines = []
        #Loop over segments
        print("slitletAlignProcess::slitletAlign> Matching Up Lines...")
        self._log.writeLog(__name__, "Matching Up Lines...")

        #Define loop parameters
        seg1 = 0
        seg2 = n_segments
        seginc = 1
        if (reverseOrder):
            seg1 = n_segments-1
            seg2 = -1
            seginc = -1

        seg_max_lines = -1
        seg_min_thresh = 2

        #Loop over segments
        for i in range(seg1, seg2, seginc):
            #Find min, max x in ref slit where all slits overlap
            searchxmin = xoffInit.max()-xoffInit[reference_slit]+25
            searchxmax = xoffInit.min()-xoffInit[reference_slit]+xsize-25
            #Update max_lines, min_thresh for this segment
            if (max_lines is not None):
                if (i < len(max_lines)):
                    seg_max_lines = max_lines[i]
                else:
                    seg_max_lines = max_lines[-1]
            if (seg_max_lines == 0):
                #This segment should not be used at all
                continue
            if (min_thresh is not None):
                if (i < len(min_thresh)):
                    seg_min_thresh = min_thresh[i]
                else:
                    seg_min_thresh = min_thresh[-1]
            #use offsets from region file as initial guess
            xoffGuess = xoffInit.copy()
            #xlo, xhi for current segment
            xlo = int(searchxmin+(searchxmax-searchxmin)*i//n_segments)
            xhi = int(searchxmin+(searchxmax-searchxmin)*(i+1)//n_segments)
            #x indices for current segment in reference slit
            validxs = arange(xhi-xlo+1)+xlo
            #Find brightest line in segment
            #blref = x index of brightest line with respect to 0
            blref = validxs[where(onedcuts[reference_slit][validxs] == max(onedcuts[reference_slit][validxs]))[0][0]]+xoffInit[reference_slit]
            #Fit a Gaussian to find center to sub-pixel accuracy
            #Square data first to ensure that bright line dominates fit
            refCut = onedcuts[reference_slit][blref-10-xoffInit[reference_slit]:blref+11-xoffInit[reference_slit]]**2
            p = zeros(4, dtype=float64)
            p[0] = max(refCut)
            p[1] = 10
            p[2] = 2
            p[3] = gpu_arraymedian(refCut, nonzero=True)
            lsq = leastsq(gaussResiduals, p, args=(arange(len(refCut), dtype=float64), refCut))
            fracOffset = lsq[0][1]-10
            #Multiply by sqrt(2) because we fit squared data.  This will
            #give us the width of the emission lines.
            gaussWidth = abs(lsq[0][2]*sqrt(2))
            if (gaussWidth > 2):
                #If its a broad line for some reason, use 1.5 as default
                gaussWidth = 1.5
            #Max peak - take sqrt because we squared the data
            maxPeak = sqrt(abs(lsq[0][0]))
            nlines = 1
            print("\tBrightest line found at "+str(blref+fracOffset-xoffInit[reference_slit])+" in slitlet "+str(reference_slit))
            self._log.writeLog(__name__, "Brightest line found at "+str(blref+fracOffset-xoffInit[reference_slit])+" in slitlet "+str(reference_slit), printCaller=False, tabLevel=1)

            #Find x index of this line in other slitlets
            bline = []
            for j in range(nslits):
                #Define 101 pixel wide box (less if near either end of segment)
                bxlo = max(blref-xoffGuess[j]-50,0)
                bxhi = min(bxlo+101,xhi+xoffGuess[reference_slit]-xoffGuess[j])
                #Find brightest line in box for this slitlet
                bline.append(where(onedcuts[j][bxlo:bxhi] == max(onedcuts[j][bxlo:bxhi]))[0][0]+bxlo+xoffGuess[j])
                #Continue to next iteration if first segment
                #OR if not doing the nebular emission check.
                if (len(reflines) == 0 or not nebularEmission):
                    continue
                currCut = (onedcuts[j][bxlo:bxhi]).copy()
                #This is not the first segment.
                #Find reference line index -- closest line in another segment.
                #Line must be valid for all slitlets
                xDiff = array(reflines-blref) #xDiff = difference in pixel space from this line
                refline = where(abs(xDiff) == min(abs(xDiff[array(validlines)])))[0][0]
                #Use only valid lines for each slit, do linear fit
                b = where(array(validOffsets)[:,j])
                if (len(reflines[b]) < 2):
                    print("slitletAlignProcess::slitletAlign> Warning: slitlet "+str(j)+" - unable to find at least 2 valid lines for nebular emission check!")
                    self._log.writeLog(__name__, "slitlet "+str(j)+" - unable to find at least 2 valid lines for nebular emission check!", type=fatboyLog.WARNING)
                    continue
                #Do linear least squares fit x = refslit X, y = currSlit X
                lsq = leastsq(linResiduals, [0.,0.], args=(array(reflines)[b],(array(xoffsets)[:,j])[b]))
                #create a temp list = [reflines, blref]
                tempLines = (array(reflines)[b]).tolist()+[blref]
                #Iterate over three brightest lines in the 101 pixel box in this slitlet
                for z in range(3):
                    #Find brightest remaining line
                    tmppeak = where(currCut == max(currCut))[0][0]
                    #Apply offset and box lo to get actual index
                    tmpbline = tmppeak+bxlo+xoffGuess[j]
                    diff = abs(xoffsets[refline][j]-(xoffGuess[j]+blref-tmpbline))
                    tempxoff = ((array(xoffsets)[:,j])[b]).tolist()+[xoffGuess[j]+blref-tmpbline]
                    tmpsigma = (array(tempxoff)-(lsq[0][0]+lsq[0][1]*array(tempLines))).std()
                    if (z == 0):
                        minsigma = tmpsigma
                        mindiff = diff
                    elif ((tmpsigma < minsigma and diff < mindiff) or tmpsigma < minsigma/3):
                        minsigma = tmpsigma
                        mindiff = diff
                        print("\tSlit "+str(j)+": Using line at "+str(tmppeak+bxlo)+" instead of "+str(bline[j]-xoffGuess[j]))
                        self._log.writeLog(__name__, "Slit "+str(j)+": Using line at "+str(tmppeak+bxlo)+" instead of "+str(bline[j]-xoffGuess[j]), printCaller=False, tabLevel=1)
                        bline[j] = tmpbline
                    #Zero out this line in currCut
                    currCut[max(0,tmppeak-10):min(tmppeak+11,101)] = 0
            #Find difference between 1st and 2nd initial guesses
            #Do a sigma clipping to find sigma then set all >4*sigma to mean
            bline = array(bline)
            xprime = bline-blref
            meanMedSig = sigmaFromClipping(xprime, 2, 3)
            bline[where(abs(xprime-meanMedSig[0]) > 4*meanMedSig[2])] = xprime.mean()+blref
            #print bline-xoffGuess

            #Use 21-px box around line in ref frame for x-corr
            #Determine line center to subpixel accuracy in each slitlet
            refCut = onedcuts[reference_slit][blref-10-xoffGuess[reference_slit]:blref+11-xoffGuess[reference_slit]].copy()
            refCut = refCut - arraymedian(refCut,nonzero=True)
            currXoff = []
            currValid = []
            #Cross-correlate ref with 21-px search box in each slit
            for j in range(len(bline)):
                xlo = bline[j]-xoffGuess[j]-10
                ccor = correlate(refCut, onedcuts[j][xlo:xlo+21]-arraymedian(onedcuts[j][xlo:xlo+21], nonzero=True),mode='same')
                p = zeros(4, dtype=float64)
                p[0] = max(ccor)
                p[1] = 10
                p[2] = gaussWidth
                p[3] = gpu_arraymedian(ccor)
                lsq = leastsq(gaussResiduals, p, args=(arange(len(ccor), dtype=float64), ccor))
                if (lsq[1] == 5):
                    #exceeded max number of calls
                    print("slitletAlignProcess::slitletAlign> Warning: Could not match up line "+str(blref)+" in slit "+str(j)+". Using initial default guess instead for fit anchoring purposes.")
                    self._log.writeLog(__name__, "Could not match up line "+str(blref)+" in slit "+str(j)+". Using initial default guess instead for fit anchoring purposes.", type=fatboyLog.WARNING)
                    currXoff.append(xoffGuess[j]-bline[j]+bline[reference_slit])
                    currValid.append(False)
                    continue
                mcor = lsq[0][1]
                #print j,mcor,bline[j]-xoffGuess[j]+fracOffset+10-mcor
                #mcor = 10 is zero adjustment to bline[reference_slit]-bline[j]
                currXoff.append(xoffGuess[j]-bline[j]+blref-fracOffset+mcor-10)
                currValid.append(True)

            #Store offsets, line, update validxs, blank out line
            xoffsets.append(array(currXoff))
            validOffsets.append(array(currValid))
            #Round off to nearest pixel for indexing purposes
            currXoff = (array(currXoff)+0.5).astype(int32)
            #Blank out 11 pixel box in 1-d cuts
            for j in range(len(onedcuts)):
                onedcuts[j][blref-5-currXoff[j]:blref+6-currXoff[j]] = 0
            reflines.append(blref)
            validlines.append(True)
            #Expand search area to include entire reference slit
            #and not just region where all slits overlap
            searchxmin = 25
            searchxmax = len(onedcuts[reference_slit])-25
            xlo = int(searchxmin+(searchxmax-searchxmin)*i//n_segments)
            xhi = int(searchxmin+(searchxmax-searchxmin)*(i+1)//n_segments)
            validxs = arange(xhi-xlo+1)+xlo
            validxs = concatenate((validxs[where(validxs < blref-5-xoffGuess[reference_slit])], validxs[where(validxs > blref+5-xoffGuess[reference_slit])]))
            #Loop over other lines until NLINES and < SIGMA are reached
            inloop = 0
            findLines = True
            while (findLines and inloop < 10):
                #If peak in resid array is <= 1% of peak of brightest line, break out of loop
                if (onedcuts[reference_slit][validxs].max() <= maxPeak/100):
                    findLines = False
                    break
                #Find next brightest line
                blref = validxs[where(onedcuts[reference_slit][validxs] == max(onedcuts[reference_slit][validxs]))[0][0]]+xoffInit[reference_slit]
                #Setup box sizes
                refbox = 25
                searchbox = 25
                currXoff = []
                currValid = []
                xDiff = array(reflines-blref)
                refline = where(abs(xDiff) == min(abs(xDiff[array(validlines)])))[0][0]
                #New initial guess for offset based on nearest aligned line.
                xoffGuess = (xoffsets[refline]+0.5).astype(int32)
                blref += xoffGuess[reference_slit]-xoffInit[reference_slit]

                #Fit a Gaussian to find center to sub-pixel accuracy
                #Copy so that zeros do not get applied to onedcut
                refCut = onedcuts[reference_slit][blref-10-xoffGuess[reference_slit]:blref+11-xoffGuess[reference_slit]]**2
                p = zeros(4, dtype=float64)
                p[0] = max(refCut)
                p[1] = 10
                p[2] = gaussWidth/sqrt(2)
                p[3] = gpu_arraymedian(refCut)
                lsq = leastsq(gaussResiduals, p, args=(arange(len(refCut), dtype=float64), refCut))
                fracOffset = lsq[0][1]-10

                #Rejection 1: If current line is within 12 pixels of a previous line
                if (abs(reflines[refline]-blref) < 13):
                    #Too close!  Blank out 3 pixels and try again
                    onedcuts[reference_slit][blref-1-xoffGuess[reference_slit]:blref+2-xoffGuess[reference_slit]] = 0
                    inloop+=1
                    continue
                if (abs(reflines[refline]-blref) < 50):
                    #Current line is within 50 pixels of a previous line
                    #This allows us to use a narrower window for the search box
                    #10 pix on either side instead of 25.
                    searchbox = 10
                    refbox = 10
                refCut = onedcuts[reference_slit][blref-refbox-xoffGuess[reference_slit]:blref+refbox+1-xoffGuess[reference_slit]].copy()
                refCutMed = arraymedian(refCut,nonzero=True)
                #Rejection 2: If there is more than one zeroed out value
                #within +/- 7 pixels of line's center
                if (len(where(refCut[refbox-7:refbox+8] == 0)[0]) > 1):
                    onedcuts[reference_slit][blref-1-xoffGuess[reference_slit]:blref+2-xoffGuess[reference_slit]] = 0
                    if (nlines > 5):
                        inloop+=1
                    else:
                        inloop+=0.1
                    continue
                #Find nonzero points within +/- 2*search box but excluding +/- 5 points from line center
                leftpts = where(onedcuts[reference_slit][blref-xoffGuess[reference_slit]-2*searchbox:blref-xoffGuess[reference_slit]-5] != 0)[0]+blref-xoffGuess[reference_slit]-2*searchbox
                rightpts = where(onedcuts[reference_slit][blref-xoffGuess[reference_slit]+6:blref-xoffGuess[reference_slit]+2*searchbox+1] != 0)[0]+blref-xoffGuess[reference_slit]+6
                sigpts = concatenate((leftpts, rightpts))
                #Rejection 3: If there are 3 or less such nonzero points on either the left or right side
                if (len(where(sigpts+xoffGuess[reference_slit] < blref)[0]) <= 3 or len(where(sigpts+xoffGuess[reference_slit] > blref)[0]) <= 3):
                    onedcuts[reference_slit][blref-1-xoffGuess[reference_slit]:blref+2-xoffGuess[reference_slit]] = 0
                    if (nlines > 5):
                        inloop+=1
                    else:
                        inloop+=0.1
                    continue
                #Calculate local sigma using median and std dev of sigpts.
                #Since all sigpts are nonzero, no need to pass nonzero=True.
                lsigma = (onedcuts[reference_slit][blref-xoffGuess[reference_slit]]-arraymedian(onedcuts[reference_slit][sigpts]))/onedcuts[reference_slit][sigpts].std()
                #Rejection 4: If local sigma < min_thresh
                if (lsigma < seg_min_thresh):
                    onedcuts[reference_slit][blref-1-xoffGuess[reference_slit]:blref+2-xoffGuess[reference_slit]] = 0
                    if (nlines > 5):
                        inloop+=1
                    else:
                        inloop+=0.1
                    continue
                #Special case: bright line > 4 sigma locally and <= 5 lines detected so far
                #AND >200 pixels away from reference line.  Expand search box and find
                #brightest line in 51 pixel wide box centered around current guess for line.
                if (lsigma > 4 and nlines <= 5 and abs(reflines[refline]-blref) > 200):
                    #Loop over all slitets
                    for j in range(nslits):
                        xlo = blref-xoffGuess[j]-25
                        if (xlo < 25 or xlo > len(onedcuts[j])-75):
                            continue
                        bline = where(onedcuts[j][xlo:xlo+51] == max(onedcuts[j][xlo:xlo+51]))[0][0]
                        xoffGuess[j]=xoffGuess[j]+25-bline
                #Cross correlate refbox with searchbox in ref slit for sanity check
                xlo = blref-searchbox-xoffGuess[reference_slit]
                ccor = correlate(refCut,onedcuts[reference_slit][xlo:xlo+searchbox*2+1], mode='same')
                ccor = array(ccor)
                mcor = where(ccor == max(ccor))[0][0]
                #Rejection 5: If ccor breaks (?) skip this line.  It would make no sense
                #for this to be more than 1 pixel away from zero.
                if (abs(searchbox-mcor) > 1):
                    onedcuts[reference_slit][blref-5-xoffGuess[reference_slit]:blref+6-xoffGuess[reference_slit]] = 0
                    inloop+=1
                    continue
                print("\tFound line (" + str(nlines+1) +") at "+str(blref+fracOffset-xoffGuess[reference_slit])+" in slitlet "+str(reference_slit)+", "+str(lsigma)[:5]+" sigma.")
                self._log.writeLog(__name__, "Found line (" + str(nlines+1) +") at "+str(blref+fracOffset-xoffGuess[reference_slit])+" in slitlet "+str(reference_slit)+", "+str(lsigma)[:5]+" sigma.", printCaller=False, tabLevel=1)

                #Cross-correlate
                tempmcors = []
                for j in range(nslits):
                    xlo = blref-searchbox-xoffGuess[j]
                    if (xlo < 25 or xlo > len(onedcuts[j])-2*searchbox-25):
                        #Line is not in this slitlet
                        currXoff.append(0)
                        currValid.append(False)
                        tempmcors.append(0)
                        continue
                    temp = onedcuts[j][xlo:xlo+searchbox*2+1]
                    ccor = correlate(temp-arraymedian(temp),refCut-refCutMed,mode='same')
                    p = zeros(4, dtype=float64)
                    p[0] = max(ccor)
                    p[1] = where(ccor == max(ccor))[0][0]
                    p[2] = gaussWidth
                    p[3] = gpu_arraymedian(ccor)
                    llo = max(0, int(p[1]-5))
                    lhi = min(len(ccor), int(p[1]+6))
                    lsq = leastsq(gaussResiduals, p, args=(arange(lhi-llo, dtype=float64)+llo, ccor[llo:lhi]))
                    if (lsq[1] == 5):
                        #exceeded max number of calls
                        print("slitletAlignProcess::slitletAlign> Warning: Could not match up line "+str(blref)+" in slit "+str(j))
                        self._log.writeLog(__name__, "Could not match up line "+str(blref)+" in slit "+str(j), type=fatboyLog.WARNING)
                        currXoff.append(0)
                        currValid.append(False)
                        tempmcors.append(0)
                        continue
                    if (abs(where(ccor == max(ccor))[0][0] - lsq[0][1]) > 2):
                        #should be within +/- 2 pixels from guess
                        print("slitletAlignProcess::slitletAlign> Warning: Could not match up line "+str(blref)+" in slit "+str(j))
                        self._log.writeLog(__name__, "Could not match up line "+str(blref)+" in slit "+str(j), type=fatboyLog.WARNING)
                        currXoff.append(0)
                        currValid.append(False)
                        tempmcors.append(0)
                        continue
                    mcor = lsq[0][1]
                    tempmcors.append(mcor)
                    currXoff.append(xoffGuess[j]-fracOffset-mcor+searchbox)
                    currValid.append(True)
                #If ccor unsuccessful for any one slit, skip line
                b = where(array(currValid))
                if (len(b[0]) == 1):
                    #Rejecton 6: Only found in one slit!
                    onedcuts[reference_slit][blref-5-xoffGuess[reference_slit]:blref+6-xoffGuess[reference_slit]] = 0
                    inloop+=1
                    print("slitletAlignProcess::slitletAlign> Warning: Line "+str(blref)+" thrown out because only found in one slit!")
                    self._log.writeLog(__name__, "Warning: Line "+str(blref)+" thrown out because only found in one slit!", type=fatboyLog.WARNING)
                    continue

                if (max(abs(array(currXoff)[b]-array(xoffGuess)[b])) > abs(reflines[refline]-blref)/10.):
                    #Rejection 7: Max difference between initial guess and fitted
                    #value is greater than 10% of the distance between this line and
                    #the reference line in the reference slitlet.
                    onedcuts[reference_slit][blref-5-xoffGuess[reference_slit]:blref+6-xoffGuess[reference_slit]] = 0
                    inloop+=1
                    print("slitletAlignProcess::slitletAlign> Warning: Line "+str(blref)+" thrown out because one or more slitlets do not match.")
                    self._log.writeLog(__name__, "Warning: Line "+str(blref)+" thrown out because one or more slitlets do not match.", type=fatboyLog.WARNING)
                    continue
                #Fit expected vs actual offsets with line, find outliers
                lsq = leastsq(linResiduals, [0.,0.], args=(array(currXoff)[b], array(tempmcors)[b]))
                #Calculate residuals -- offsets - fit
                offresid = abs(array(tempmcors)[b]-(lsq[0][0]+lsq[0][1]*array(currXoff)[b]))
                for j in range(len(offresid)):
                    if (offresid[j] > 5):
                        currXoff[b[0][j]] = 0
                        currValid[b[0][j]] = False
                nlines+=1
                inloop=0
                #print blref-currXoff
                #Blank out line, update xoffsets, reflines, validxs
                xoffsets.append(array(currXoff))
                validOffsets.append(array(currValid))
                reflines.append(blref)
                #Round off to nearest pixel for indexing purposes
                currXoff = (array(currXoff)+0.5).astype(int32)
                for j in range(len(onedcuts)):
                    if (currValid[j]):
                        onedcuts[j][blref-5-currXoff[j]:blref+6-currXoff[j]] = 0
                if (sum(currValid) == len(currXoff)):
                    validlines.append(True)
                else:
                    validlines.append(False)
                validxs = concatenate((validxs[where(validxs < blref-5-xoffGuess[reference_slit])], validxs[where(validxs > blref+5-xoffGuess[reference_slit])]))
                #Break out of loop if max_lines hit
                if (seg_max_lines >= 0 and nlines >= seg_max_lines):
                    findLines = False
                #If peak in resid array is <= 1% of peak of brightest line, break out of loop
                if (onedcuts[reference_slit][validxs].max() <= maxPeak/100):
                    findLines = False
                    break
            #end while
        #end for i

        #Setup xin, xout (refxs)
        #print array(reflines)[where(validlines)]
        print("slitletAlignProcess::slitletAlign> Found "+str(len(reflines))+" lines.  Calculating Transformation...")
        self._log.writeLog(__name__, "Found "+str(len(reflines))+" lines.  Calculating Transformation...")
        refxs = []
        xin = []
        for j in range(len(reflines)):
            #refxs = input x centroids of lines in reference slitlet
            refxs.append(reflines[j]-xoffsets[j][reference_slit]+xoffsets[0][reference_slit]-reflines[0])
        refxs = array(refxs)
        for j in range(nslits):
            temp = []
            #xin = list of array of x centroids of lines in each slitlet
            for k in range(len(reflines)):
                temp.append(reflines[k]-xoffsets[k][j]+xoffsets[0][j]-reflines[0])
            xin.append(array(temp))

        #Loop over slitlets
        #Take 1-d cuts (spectra), find initial guesses for offsets
        for j in range(nslits):
            #Fit polynomial of order fit_order to lines
            p = zeros(fit_order+1, dtype=float32)
            p[1] = 1
            v = where(array(validOffsets)[:,j]) #only use lines where validOffsets = True
            if (len(refxs[v]) > 3):
                #Use leastsq to fit transform between xout = refxs and xin for this slitlet
                lsq = leastsq(polyResiduals, p, args=(xin[j][v], refxs[v], fit_order))
            else:
                #Don't transform slitlet if 3 or fewer lines found in it.
                lsq = [p]
                print("slitletAlignProcess::slitletAlign> Warning: Too few datapoints in slitlet "+str(j)+"! Slitet not transformed!")
                self._log.writeLog(__name__, "Too few datapoints in slitlet "+str(j)+"! Slitet not transformed!", type=fatboyLog.WARNING)
            #Calculate residuals
            residLines = polyFunction(lsq[0], xin[j][v], fit_order)-refxs[v]
            print("\tSlitlet "+str(j)+" fit: "+str(lsq[0]))
            print("\t\tData - fit mean: "+str(residLines.mean())+"\tsigma: "+str(residLines.std()))
            self._log.writeLog(__name__, "Slitlet "+str(j)+" fit: "+str(lsq[0]), printCaller=False, tabLevel=1)
            self._log.writeLog(__name__, "Data - fit mean: "+str(residLines.mean())+"\tsigma: "+str(residLines.std()), printCaller=False, tabLevel=2)

            if (residLines.std() != 0 and len(refxs[v] > 3)):
                #Ignore reference slit to avoid divide by zero
                #Ignore slits that don't have more than 3 lines
                #Throw away outliers starting at 2 sigma significance
                sigThresh = 2
                niter = 0
                norig = len(residLines)
                bad = where(abs(residLines-residLines.mean())/residLines.std() > sigThresh)
                good = where(abs(residLines-residLines.mean())/residLines.std() <= sigThresh)
                print("\t\tPerforming iterative sigma clipping to throw away outliers...")
                self._log.writeLog(__name__, "Performing iterative sigma clipping to throw away outliers...", printCaller=False, tabLevel=2)
                #Iterative sigma clipping
                while (len(bad[0]) > 0):
                    niter += 1
                    good = where(abs(residLines-residLines.mean())/residLines.std() <= sigThresh)
                    #Refit, use last actual fit coordinates as input guess
                    p = lsq[0]
                    lsq = leastsq(polyResiduals, p, args=(xin[j][v][good], refxs[v][good], fit_order))
                    #Calculate residuals
                    residLines = polyFunction(lsq[0], xin[j][v], fit_order)-refxs[v]
                    if (niter > 1):
                        #Gradually increase sigma threshold
                        sigThresh += 0.25
                    bad = where(abs(residLines[good]-residLines[good].mean())/residLines[good].std() > sigThresh)
                print("\t\tAfter "+str(niter)+" passes, kept "+str(len(residLines[good]))+" of "+str(norig)+" datapoints.  Fit: "+str(lsq[0]))
                print("\t\tData - fit mean: "+str(residLines[good].mean())+"\tsigma: "+str(residLines[good].std()))
                self._log.writeLog(__name__, "After "+str(niter)+" passes, kept "+str(len(residLines[good]))+" of "+str(norig)+" datapoints.  Fit: "+str(lsq[0]), printCaller=False, tabLevel=2)
                self._log.writeLog(__name__, "Data - fit mean: "+str(residLines[good].mean())+"\tsigma: "+str(residLines[good].std()), printCaller=False, tabLevel=2)
            #Append fit params to list
            fitParams.append(lsq[0])

        #Resample data for both master lamp / clean sky and FDU
        #Calculate ylo, yhi for FDU if different from skyFDU
        if (fdu.hasProperty("slitmask")):
            if (fdu.hasProperty("regions")):
                (sylo_data, syhi_data, slitx_data, slitw_data) = fdu.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (sylo_data, syhi_data, slitx_data, slitw_data) = findRegions(fdu.getSlitmask().getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                fdu.setProperty("regions", (sylo_data, syhi_data, slitx_data, slitw_data))
        else:
            sylo_data = sylo
            syhi_data = syhi


        #Output x-pixels
        xout = zeros(skyFDU.getShape(), dtype=float32)
        xout_data = zeros(fdu.getShape(), dtype=float32)
        for j in range(nslits):
            xs = arange(xsize, dtype=float32)+xoffsets[0][j]-reflines[0]
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                #MOS data - add to xout array, mutiply input by currMask
                currMask = slitmask.getData()[sylo[j]:syhi[j]+1,:] == (j+1)
                xout[sylo[j]:syhi[j]+1,:] += polyFunction(fitParams[j], xs, len(fitParams[j])-1)*currMask
                if (fdu.hasProperty("slitmask")):
                    #Use this FDU's slitmask to calculate currMask for xout_data
                    currMask = fdu.getSlitmask().getData()[sylo_data[j]:syhi_data[j]+1,:] == (j+1)
                xout_data[sylo_data[j]:syhi_data[j]+1,:] += polyFunction(fitParams[j], xs, len(fitParams[j])-1)*currMask
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                currMask = slitmask.getData()[:,sylo[j]:syhi[j]+1] == (j+1)
                #Use reshape to broadcast 1-d array to (n, 1) shape
                xout[:,sylo[j]:syhi[j]+1] += (polyFunction(fitParams[j], xs, len(fitParams[j])-1)).reshape((len(xs), 1))*currMask
                if (fdu.hasProperty("slitmask")):
                    #Use this FDU's slitmask to calculate currMask for xout_data
                    currMask = fdu.getSlitmask().getData()[:,sylo_data[j]:syhi_data[j]+1] == (j+1)
                xout_data[:,sylo_data[j]:syhi_data[j]+1] += (polyFunction(fitParams[j], xs, len(fitParams[j])-1)).reshape((len(xs), 1))*currMask

        #Setup paramaters for drihizzle
        print("slitletAlignProcess::slitletAlign> Applying Transformation...")
        self._log.writeLog(__name__, "Applying Transformation...")
        xmin = int(xout.min())
        xmax = int(xout.max())+2
        if (xmax-xmin > xsize*3):
            #Sanity check
            print("slitletAlignProcess::slitletAlign> ERROR: Could not properly fit lines for frame "+skyFDU.getFullId()+"! Discarding image!")
            self._log.writeLog(__name__, "Could not properly fit lines for frame "+skyFDU.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            #Use drihizzle to resample with xtrans=xout_data
            inMask = (xout_data != 0).astype(int32)

            #First update properties cleanFrame, noisemap, slitmask
            if (fdu.hasProperty("cleanFrame")):
                (cleanData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)
            #Rectify noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, rectify, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))
            #Rectify slitmask
            if (fdu.hasProperty("slitmask")):
                (smData, header, expmap, pixmap) = drihizzle_method(fdu.getSlitmask(), None, None, inmask=inMask, kernel="uniform", dropsize=1, xtrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
                #Update "slitmask" data tag
                #Use new fdu.setSlitmask
                fdu.setSlitmask(smData, pname=self._pname)
                #Write to disk if requested
                if (writeCalibs):
                    safile = outdir+"/slitletAligned/slitmask_sa_"+fdu.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(safile)
                    #Write to disk
                    if (not os.access(safile, os.F_OK)):
                        fdu.getSlitmask().writeTo(safile)
            #Now update data and header for FDU
            (data, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            fdu.tagDataAs("exposure_map", data=expmap)
            fdu.updateHeader(header)
            expmap[expmap == 0] = 1
            fdu.updateData(data)

            #Look for "cleanSky" frame to rectify
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("slitletAligned")):
                cleanSky = calibs['cleanSky']
                #Use drihizzle to resample "sky" with xtrans=xout
                inMask = (xout != 0).astype(int32)
                (data, header, expmap, pixmap) = drihizzle_method(cleanSky, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
                expmap[expmap == 0] = 1
                cleanSky.updateData(data)
                cleanSky.updateHeader(header)
                cleanSky.setProperty("slitletAligned", True)
                #Write to disk if requested
                if (writeCalibs):
                    safile = outdir+"/slitletAligned/sa_"+cleanSky.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(safile)
                    #Write to disk
                    if (not os.access(safile, os.F_OK)):
                        cleanSky.writeTo(safile)
            #Look for "masterLamp" frame to rectify
            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("slitletAligned")):
                masterLamp = calibs['masterLamp']
                #Use drihizzle to resample lamp with xtrans=xout
                inMask = (xout != 0).astype(int32)
                (data, header, expmap, pixmap) = drihizzle_method(masterLamp, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
                expmap[expmap == 0] = 1
                masterLamp.updateData(data)
                masterLamp.updateHeader(header)
                masterLamp.setProperty("slitletAligned", True)
                #Write to disk if requested
                if (writeCalibs):
                    safile = outdir+"/slitletAligned/sa_"+masterLamp.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(safile)
                    #Write to disk
                    if (not os.access(safile, os.F_OK)):
                        masterLamp.writeTo(safile)
            #Update calib slitmask too since cleanSky and mlamp updated!
            (data, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", dropsize=1, xtrans=xout, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            saSlitmask = self._fdb.addNewSlitmask(slitmask, data, self._pname)
            #update properties
            saSlitmask.setProperty("nslits", nslits)
            saSlitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
            #update data, header, set "rectified" property
            saSlitmask.updateData(data)
            saSlitmask.updateHeader(header)
            #Write to disk if requested
            if (writeCalibs):
                safile = outdir+"/slitletAligned/calib_slitmask_sa_"+fdu.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(safile)
                #Write to disk
                if (not os.access(safile, os.F_OK)):
                    saSlitmask.writeTo(safile)

        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #ytrans = xout_data
            inMask = (xout_data != 0).astype(int32)
            #First update properties cleanFrame, noisemap, slitmask

            if (fdu.hasProperty("cleanFrame")):
                (cleanData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)
            #Rectify noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, rectify, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))
            #Rectify slitmask
            if (fdu.hasProperty("slitmask")):
                (smData, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", dropsize=1, ytrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
                #Update "slitmask" data tag
                #Use new fdu.setSlitmask
                fdu.setSlitmask(smData, pname=self._pname)
                #Write to disk if requested
                if (writeCalibs):
                    safile = outdir+"/slitletAligned/slitmask_sa_"+fdu.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(safile)
                    #Write to disk
                    if (not os.access(safile, os.F_OK)):
                        fdu.getSlitmask().writeTo(safile)
            #Now update data and header for FDU
            (data, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            fdu.tagDataAs("exposure_map", data=expmap)
            fdu.updateHeader(header)
            expmap[expmap == 0] = 1
            fdu.updateData(data)


            #Look for "cleanSky" frame to rectify
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("slitletAligned")):
                cleanSky = calibs['cleanSky']
                #Use drihizzle to resample "sky" with ytrans=xout
                inMask = (xout != 0).astype(int32)
                (data, header, expmap, pixmap) = drihizzle_method(cleanSky, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
                expmap[expmap == 0] = 1
                cleanSky.updateData(data)
                cleanSky.updateHeader(header)
                cleanSky.setProperty("slitletAligned", True)
                #Write to disk if requested
                if (writeCalibs):
                    safile = outdir+"/slitletAligned/sa_"+cleanSky.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(safile)
                    #Write to disk
                    if (not os.access(safile, os.F_OK)):
                        cleanSky.writeTo(safile)
            #Look for "masterLamp" frame to rectify
            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("slitletAligned")):
                masterLamp = calibs['masterLamp']
                #Use drihizzle to resample lamp with ytrans=xout
                inMask = (xout != 0).astype(int32)
                (data, header, expmap, pixmap) = drihizzle_method(masterLamp, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
                expmap[expmap == 0] = 1
                masterLamp.updateData(data)
                masterLamp.updateHeader(header)
                masterLamp.setProperty("slitletAligned", True)
                #Write to disk if requested
                if (writeCalibs):
                    safile = outdir+"/slitletAligned/sa_"+masterLamp.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(safile)
                    #Write to disk
                    if (not os.access(safile, os.F_OK)):
                        masterLamp.writeTo(safile)
            #Update calib slitmask too since cleanSky and mlamp updated!
            (data, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", dropsize=1, ytrans=xout, inunits="counts", outunits="counts", log=self._log, mode=gpu_drihizzle.MODE_FDU)
            saSlitmask = self._fdb.addNewSlitmask(slitmask, data, self._pname)
            #update properties
            saSlitmask.setProperty("nslits", nslits)
            saSlitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
            #update data, header, set "rectified" property
            saSlitmask.updateData(data)
            saSlitmask.updateHeader(header)
            #Write to disk if requested
            if (writeCalibs):
                safile = outdir+"/slitletAligned/calib_slitmask_sa_"+fdu.getFullId()
                #Remove existing files if overwrite = yes
                if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(safile)
                #Write to disk
                if (not os.access(safile, os.F_OK)):
                    saSlitmask.writeTo(safile)
    #end slitletAlign

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            #Skip longslit data
            return True

        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/slitletAligned", os.F_OK)):
            os.mkdir(outdir+"/slitletAligned",0o755)
        #Create output filename
        safile = outdir+"/slitletAligned/sa_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(safile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(safile)
        if (not os.access(safile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(safile)
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/slitletAligned/clean_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write out exposure map if it exists
        if (fdu.hasProperty("exposure_map")):
            expfile = outdir+"/slitletAligned/exp_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(expfile)
            if (not os.access(expfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(expfile, tag="exposure_map")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/slitletAligned/NM_sa_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
