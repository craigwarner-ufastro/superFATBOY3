from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY import gpu_imcombine, imcombine

class darkSubtractProcess(fatboyProcess):
    _modeTags = ["imaging", "circe", "spectroscopy", "miradas"]

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createMasterDark(self, fdu, darks):
        mdfilename = None
        #use darks[0] for exptime in case this is a dark for a different exptime than the fdu
        mdname = "masterDarks/mdark-"+str(darks[0].exptime)+"s-"+str(fdu.nreads)+"rd-"+darks[0]._id
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (fdu.getTag(mode="composite") is not None):
            mdname += "-"+fdu.getTag(mode="composite").replace(" ","_")
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/masterDarks", os.F_OK)):
                os.mkdir(outdir+"/masterDarks",0o755)
            mdfilename = outdir+"/"+mdname+".fits"
        #Check to see if master dark exists already from a previous run
        prevmdfilename = outdir+"/"+mdname+".fits"
        #Noisemap file
        nmfile = outdir+"/masterDarks/NM_mdark-"+str(darks[0].exptime)+"s-"+str(fdu.nreads)+"rd-"+darks[0]._id+".fits"
        if (os.access(prevmdfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevmdfilename)
        elif (os.access(prevmdfilename, os.F_OK)):
            #file already exists
            print("darkSubtractProcess::createMasterDark> Master dark "+prevmdfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Master dark "+prevmdfilename+" already exists!  Re-using...")
            masterDark = fatboyCalib(self._pname, "master_dark", darks[0], filename=prevmdfilename, log=self._log)
            #set specmode property
            if (darks[0].hasProperty("specmode")):
                masterDark.setProperty("specmode", darks[0].getProperty("specmode"))
            #set dispersion property
            if (darks[0].hasProperty("dispersion")):
                masterDark.setProperty("dispersion", darks[0].getProperty("dispersion"))
            #Check to see if a noisemap exists
            if (os.access(nmfile, os.F_OK)):
                nm = pyfits.open(nmfile)
                mef = findMef(nm)
                #Tag noisemap data.  tagDataAs() will handle byteswap
                masterDark.tagDataAs("noisemap", nm[mef].data)
                nm.close()
            #disable these darks as master dark has been created
            for dark in darks:
                dark.disable()
            return masterDark

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            (data, header) = gpu_imcombine.imcombine(darks, outfile=mdfilename, method="median", mef=darks[0]._mef, returnHeader=True, log=self._log)
        else:
            (data, header) = imcombine.imcombine(darks, outfile=mdfilename, method="median", mef=darks[0]._mef, returnHeader=True, log=self._log)
        masterDark = fatboyCalib(self._pname, "master_dark", darks[0], data=data, tagname=mdname, headerExt=header, log=self._log)
        masterDark.setType("master_dark", True)
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and not os.access(mdfilename, os.F_OK)):
            #Optionally save if write_calib_output = yes
            masterDark.writeTo(mdfilename)
        #Create and write out noisemap for spectroscopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes"):
            if (not os.access(outdir+"/masterDarks", os.F_OK)):
                os.mkdir(outdir+"/masterDarks",0o755)
            if (os.access(nmfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                ncomb = float(masterDark.getHeaderValue('NCOMBINE'))
                #Create noisemap
                if (self._fdb.getGPUMode()):
                    nm = createNoisemap(masterDark.getData(), ncomb)
                else:
                    nm = sqrt(masterDark.getData()/ncomb)
                masterDark.tagDataAs("noisemap", nm)
                masterDark.writeTo(nmfile, tag="noisemap")

        #disable these darks as master dark has been created
        for dark in darks:
            dark.disable()
        return masterDark
    #end createMasterDark

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Dark Subtract")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For darkSubtract, this dict should have one entry 'masterDark' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'masterDark' in calibs):
            #Failed to obtain master dark frame
            #Issue error message and disable this FDU
            print("darkSubtractProcess::execute> ERROR: Dark not subtracted for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!")
            self._log.writeLog(__name__, "Dark not subtracted for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        dsfile = "darkSubtracted/ds_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, dsfile)):
            #Also check if "noisemap" exists
            nmfile = "darkSubtracted/NM_ds_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            return True

        #get master dark
        masterDark = calibs['masterDark']

        #Propagate noisemap for spectroscopy data
        if (fdu.hasProperty("noisemap")):
            self.updateNoisemap(fdu, masterDark)

        #make sure both are floating point before subtracting
        fdu.updateData(float32(fdu.getData())-float32(masterDark.getData()))
        fdu._header.add_history('Dark subtracted using '+masterDark._id)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        mdfilename = self.getCalib("masterDark", fdu.getTag())
        if (mdfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mdfilename, os.F_OK)):
                print("darkSubtractProcess::getCalibs> Using master dark "+mdfilename+"...")
                self._log.writeLog(__name__, "Using master dark "+mdfilename+"...")
                calibs['masterDark'] = fatboyCalib(self._pname, "master_dark", fdu, filename=mdfilename, log=self._log)
                return calibs
            else:
                print("darkSubtractProcess::getCalibs> Warning: Could not find master dark "+mdfilename+"...")
                self._log.writeLog(__name__, "Could not find master dark "+mdfilename+"...", type=fatboyLog.WARNING)

        #1) Check for an already created master dark frame matching exptime/nreads/section and TAGGED for this object
        masterDark = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_dark", exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section)
        if (masterDark is not None):
            #Found master dark.  Return here
            calibs['masterDark'] = masterDark
            return calibs
        #2) Check for individual dark frames matching exptime/nreads/section to create master dark and TAGGED for this object
        darks = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_DARK, exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section)
        if (len(darks) > 0):
            #Found darks associated with this fdu.  Create master dark.
            print("darkSubtractProcess::getCalibs> Creating Master Dark for tagged object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            self._log.writeLog(__name__, "Creating Master Dark for tagged object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            #First recursively process (linearity correction probably)
            self.recursivelyExecute(darks, prevProc)
            #convenience method
            masterDark = self.createMasterDark(fdu, darks)
            self._fdb.appendCalib(masterDark)
            calibs['masterDark'] = masterDark
            return calibs
        #3) Check for an already created master dark frame matching exptime/nreads/section
        masterDark = self._fdb.getMasterCalib(self._pname, obstype="master_dark", exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
        if (masterDark is not None):
            #Found master dark.  Return here
            calibs['masterDark'] = masterDark
            return calibs
        #4) Check for individual dark frames matching exptime/nreads/section to create master dark
        darks = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_DARK, exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
        if (len(darks) > 0):
            #Found darks associated with this fdu.  Create master dark.
            print("darkSubtractProcess::getCalibs> Creating Master Dark for exposure time "+str(fdu.exptime)+" and "+str(fdu.nreads)+" reads...")
            self._log.writeLog(__name__, "Creating Master Dark for exposure time "+str(fdu.exptime)+" and "+str(fdu.nreads)+" reads...")
            #First recursively process (linearity correction probably)
            self.recursivelyExecute(darks, prevProc)
            #convenience method
            masterDark = self.createMasterDark(fdu, darks)
            self._fdb.appendCalib(masterDark)
            calibs['masterDark'] = masterDark
            return calibs
        print("darkSubtractProcess::getCalibs> Master Dark for exposure time "+str(fdu.exptime)+", nreads "+str(fdu.nreads)+", and section "+str(fdu.section)+" not found!")
        self._log.writeLog(__name__, "Master Dark for exposure time "+str(fdu.exptime)+", nreads "+str(fdu.nreads)+", and section "+str(fdu.section)+" not found!", type=fatboyLog.WARNING)
        #5) Check default_master_dark for matching exptime/nreads/section
        defaultMasterDarks = []
        if (self.getOption('default_master_dark', fdu.getTag()) is not None):
            dmdlist = self.getOption('default_master_dark', fdu.getTag())
            if (dmdlist.count(',') > 0):
                #comma separated list
                defaultMasterDarks = dmdlist.split(',')
                removeEmpty(defaultMasterDarks)
                for j in range(len(defaultMasterDarks)):
                    defaultMasterDarks[j] = defaultMasterDarks[j].strip()
            elif (dmdlist.endswith('.fit') or dmdlist.endswith('.fits')):
                #FITS file given
                defaultMasterDarks.append(dmdlist)
            elif (dmdlist.endswith('.dat') or dmdlist.endswith('.list') or dmdlist.endswith('.txt')):
                #ASCII file list
                defaultMasterDarks = readFileIntoList(dmdlist)
            for mdarkfile in defaultMasterDarks:
                #Loop over list of default master darks
                #masterDark = fatboyImage(mdarkfile, log=self._log)
                masterDark = fatboyCalib(self._pname, "master_dark", fdu, filename=mdarkfile, log=self._log)
                #read header and initialize
                masterDark.readHeader()
                masterDark.initialize()
                if (masterDark.exptime != fdu.exptime):
                    #does not match exptime
                    continue
                if (masterDark.nreads != fdu.nreads):
                    #does not match nreads
                    continue
                if (fdu.section is not None):
                    #check section if applicable
                    section = -1
                    if (masterDark.hasHeaderValue('SECTION')):
                        section = masterDark.getHeaderValue('SECTION')
                    else:
                        idx = masterDark.getFilename().rfind('.fit')
                        if (masterDark.getFilename()[idx-2] == 'S' and isDigit(masterDark.getFilename()[idx-1])):
                            section = int(masterDark.getFilename()[idx-1])
                    if (section != fdu.section):
                        continue
                masterDark.setType("master_dark")
                #Found matching master dark
                print("darkSubtractProcess::getCalibs> Using default master dark "+masterDark.getFilename())
                self._fdb.appendCalib(masterDark)
                calibs['masterDark'] = masterDark
                return calibs
        #6) Look at previous master darks to see if any has a history of being used as master dark for
        #this _id and exptime combination from steps 7 or 8 below.
        masterDarks = self._fdb.getMasterCalibs(obstype="master_dark")
        for mdark in masterDarks:
            if (mdark.hasHistory('master_dark::'+fdu._id+'::'+str(fdu.exptime))):
                #Use this master dark
                print("darkSubtractProcess::getCalibs> Using master dark "+mdark.getFilename()+" with exptime "+str(mdark.exptime))
                #Already in _calibs, no need to appendCalib
                calibs['masterDark'] = mdark
                return calibs
        #7) If prompt_for_missing_dark is set, prompt user
        if (self.getOption('prompt_for_missing_dark', fdu.getTag()).lower() == "yes"):
            #Prompt user for dark file
            print("List of darks, exposure times, nreads, and sections:")
            masterDarks = self._fdb.getMasterCalibs(obstype="master_dark")
            for mdark in masterDarks:
                print(mdark.getFilename(), mdark.exptime, mdark.nreads, mdark.section)
            for mdarkfile in defaultMasterDarks:
                #Loop over list of default master darks too
                mdark = fatboyImage(mdarkfile, log=self._log)
                #read header and initialize
                mdark.readHeader()
                mdark.initialize()
                print(mdarkfile, mdark.exptime, mdark.nreads, mdark.section)
            tmp = input("Select a filename to use as a dark: ")
            mdfilename = tmp
            #Now find if input matches one of these filenames
            for mdark in masterDarks:
                if (mdark.getFilename() == mdfilename):
                    #Found matching master dark
                    print("darkSubtractProcess::getCalibs> Using master dark "+mdark.getFilename())
                    mdark.setHistory('master_dark::'+fdu._id+'::'+str(fdu.exptime), 'yes')
                    #Already in _calibs, no need to appendCalib
                    calibs['masterDark'] = mdark
                    return calibs
            #If not found yet, check default master darks
            if (mdfilename in defaultMasterDarks):
                mdark = fatboyImage(mdarkfile, log=self._log)
                #read header and initialize
                mdark.readHeader()
                mdark.initialize()
                print("darkSubtractProcess::getCalibs> Using master dark "+mdark.getFilename())
                mdark.setHistory('master_dark::'+fdu._id+'::'+str(fdu.exptime), 'yes')
                self._fdb.appendCalib(mdark)
                calibs['masterDark'] = mdark
                return calibs
        else:
            #8) Find dark closest in exptime and matching nreads/section
            expDiff = None
            masterDark = None
            appendCalib = False
            masterDarks = self._fdb.getMasterCalibs(obstype="master_dark", nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
            for mdark in masterDarks:
                if (expDiff is None):
                    #first match
                    expDiff = abs(mdark.exptime - fdu.exptime)
                    masterDark = mdark
                elif (abs(mdark.exptime - fdu.exptime) < expDiff):
                    #closer match
                    expDiff = abs(mdark.exptime - fdu.exptime)
                    masterDark = mdark
            for mdarkfile in defaultMasterDarks:
                #Loop over list of default master darks too
                mdark = fatboyImage(mdarkfile, log=self._log)
                #read header and initialize
                mdark.readHeader()
                mdark.initialize()
                if (mdark.nreads == fdu.nreads and mdark.section == fdu.section):
                    if (expDiff is None):
                        #first match
                        expDiff = abs(mdark.exptime - fdu.exptime)
                        masterDark = mdark
                        appendCalib = True
                    elif (abs(mdark.exptime - fdu.exptime) < expDiff):
                        #closer match
                        expDiff = abs(mdark.exptime - fdu.exptime)
                        masterDark = mdark
                        appendCalib = True
            #first check individual darks
            currentDark = None
            darks = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_DARK, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
            for dark in darks:
                if (expDiff is None):
                    #first match
                    expDiff = abs(dark.exptime - fdu.exptime)
                    #reset master dark
                    masterDark = None
                    currentDark = dark
                elif (abs(dark.exptime - fdu.exptime) < expDiff):
                    #closer match
                    expDiff = abs(dark.exptime - fdu.exptime)
                    #reset master dark
                    masterDark = None
                    currentDark = dark
            if (masterDark is not None):
                #closest match was a previous master dark or default master dark
                print("darkSubtractProcess::getCalibs> Using master dark "+masterDark.getFilename()+" with exptime "+str(masterDark.exptime))
                masterDark.setHistory('master_dark::'+fdu._id+'::'+str(fdu.exptime), 'yes')
                if (appendCalib):
                    #default master dark that is not yet in _calibs
                    self._fdb.appendCalib(masterDark)
                calibs['masterDark'] = masterDark
                return calibs
            elif (currentDark is not None):
                #create master dark from darks with closest exp time
                print("darkSubtractProcess::getCalibs> Using exptime "+str(currentDark.exptime)+" instead.")
                print("darkSubtractProcess::getCalibs> Creating Master Dark for exposure time "+str(currentDark.exptime)+" and "+str(currentDark.nreads)+" reads...")
                darks = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_DARK, exptime=currentDark.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
                #First recursively process (linearity correction probably)
                self.recursivelyExecute(darks, prevProc)
                #convenience method
                masterDark = self.createMasterDark(fdu, darks)
                masterDark.setHistory('master_dark::'+fdu._id+'::'+str(fdu.exptime), 'yes')
                self._fdb.appendCalib(masterDark)
                calibs['masterDark'] = masterDark
                return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_dark', None)
        self._options.setdefault('prompt_for_missing_dark', 'no')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## update noisemap for spectroscopy data
    def updateNoisemap(self, fdu, masterDark):
        if (not masterDark.hasProperty("noisemap")):
            #create tagged data "noisemap"
            ncomb = 1.0
            if (masterDark.hasHeaderValue('NCOMBINE')):
                ncomb = float(masterDark.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterDark.getData(), ncomb)
            else:
                nm = sqrt(masterDark.getData()/ncomb)
            masterDark.tagDataAs("noisemap", nm)
        #Get this FDU's noisemap
        nm = fdu.getData(tag="noisemap")
        #Propagate noisemaps.  For subtraction, dz = sqrt(dx^2 + dy^2)
        if (self._fdb.getGPUMode()):
            nm = noisemaps_ds_gpu(fdu.getData(tag="noisemap"), masterDark.getData("noisemap"))
        else:
            nm = sqrt(fdu.getData(tag="noisemap")**2+masterDark.getData("noisemap")**2)
        fdu.tagDataAs("noisemap", nm)
    #end updateNoisemap

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/darkSubtracted", os.F_OK)):
            os.mkdir(outdir+"/darkSubtracted",0o755)
        #Create output filename
        dsfile = outdir+"/darkSubtracted/ds_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(dsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(dsfile)
        if (not os.access(dsfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(dsfile)
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/darkSubtracted/NM_ds_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
