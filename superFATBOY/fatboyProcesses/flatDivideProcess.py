from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY import gpu_imcombine, imcombine

block_size = 512

class flatDivideProcess(fatboyProcess):
    _modeTags = ["imaging", "circe"]

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createMasterFlat(self, fdu, flats, properties):
        masterFlat = None
        mffilename = None
        flatmethod = properties['flat_method']
        #use flats[0] for filter in case this is a flat for a different filter than the fdu
        mfname = "masterFlats/mflat-"+str(flatmethod)+"-"+str(flats[0].filter).replace(" ","_")+"-"+str(flats[0]._id)
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (fdu.getTag(mode="composite") is not None):
            mfname += "-"+fdu.getTag(mode="composite").replace(" ","_")
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/masterFlats", os.F_OK)):
                os.mkdir(outdir+"/masterFlats",0o755)
            mffilename = outdir+"/"+mfname+".fits"
        #Check to see if master flat exists already from a previous run
        prevmffilename = outdir+"/"+mfname+".fits"
        if (os.access(prevmffilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevmffilename)
        elif (os.access(prevmffilename, os.F_OK)):
            #file already exists
            print("flatDivideProcess::createMasterFlat> Master flat "+prevmffilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Master flat "+prevmffilename+" already exists!  Re-using...")
            masterFlat = fatboyCalib(self._pname, "master_flat", flats[0], filename=prevmffilename, log=self._log)
            #set flat_method property
            masterFlat.setProperty("flat_method", flatmethod)
            #disable these flats as master flat has been created
            for flat in flats:
                flat.disable()
            return masterFlat

        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        if (flatmethod == "dome_on-off"):
            #dome_on-off method. Find off flats
            offFlats = []
            onFlats = []
            for flat in flats:
                if (not flat.hasProperty("flat_type")):
                    #Look at XML options to find flat type and assign it to FDUs
                    self.findDomeFlatType(fdu, properties)
                if (flat.getProperty("flat_type") == "lamp_off"):
                    #This is an OFF flat
                    offFlats.append(flat)
                else:
                    #This is an ON flat
                    onFlats.append(flat)
            if (len(onFlats) == 0):
                #All off flats, no ON flats!  Error!
                print("flatDivideProcess::createMasterFlat> ERROR: No ON flats found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No ON flats found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            if (len(offFlats) == 0):
                #All on flats, no OFF flats!  Error!
                print("flatDivideProcess::createMasterFlat> ERROR: No OFF flats found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No OFF flats found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            #Combine ON flats
            (data, header) = imcombine_method(onFlats, outfile=mffilename, method="median", mef=onFlats[0]._mef, log=self._log, returnHeader=True)
            #Combine and subtract OFF flats
            (offData, offHeader) = imcombine_method(offFlats, outfile=None, method="median", mef=offFlats[0]._mef, log=self._log, returnHeader=True)
            data -= offData
            offData = None #free memory
            #If output written to disk, update FITS file with difference
            if (mffilename is not None):
                mflat = pyfits.open(mffilename, "update")
                mflat[onFlats[0]._mef].data = data
                mflat.verify('silentfix')
                mflat.flush()
                mflat.close()
            masterFlat = fatboyCalib(self._pname, "master_flat", onFlats[0], data=data, tagname=mfname, headerExt=header, log=self._log)
        elif (flatmethod == "dome_on"):
            #dome_on method
            #Combine ON flats
            (data, header) = imcombine_method(flats, outfile=mffilename, method="median", mef=flats[0]._mef, log=self._log, returnHeader=True)
            masterFlat = fatboyCalib(self._pname, "master_flat", flats[0], data=data, tagname=mfname, headerExt=header, log=self._log)
        elif (flatmethod == "sky"):
            #sky method
            #Combine Flats with imcombine.  Scale each flat by the reciprocal of its median and then median combine them.
            #Supports several reject types
            try:
                nlow = int(self.getOption('flat_sky_nlow', fdu.getTag()))
                nhigh = int(self.getOption('flat_sky_nhigh', fdu.getTag()))
                lsigma = float(self.getOption('flat_sky_lsigma', fdu.getTag()))
                hsigma = float(self.getOption('flat_sky_hsigma', fdu.getTag()))
            except ValueError as ex:
                print("flatDivideProcess::createMasterFlat> Error: invalid sky rejection criteria: "+str(ex))
                self._log.writeLog(__name__, " invalid sky rejection criteria: "+str(ex), type=fatboyLog.ERROR)
                return None
            (data, header) = imcombine_method(flats, outfile=mffilename, method="median", scale="median", mef=flats[0]._mef, reject=self.getOption('flat_sky_reject_type', fdu.getTag()), nlow=nlow, nhigh=nhigh, lsigma=lsigma, hsigma=hsigma, log=self._log, returnHeader=True)
            masterFlat = fatboyCalib(self._pname, "master_flat", flats[0], data=data, tagname=mfname, headerExt=header, log=self._log)
        elif (flatmethod == "twilight"):
            #twilight method
            #imcombine now supports FDU_DIFFERENCE method to combine differences between flats
            #Scale each flat by the reciprocal of its median and then median combine them.
            if (len(flats) == 1):
                print("flatDivideProcess::createMasterFlat> Only 1 twilight flat found!  Master flat not created!")
                self._log.writeLog(__name__, "Only 1 twilight flat found!  Master flat not created!", type=fatboyLog.ERROR)
                return None
            imcombine_mode = imcombine.MODE_FDU_DIFFERENCE
            if (self.getOption('twilight_pair_ramps', fdu.getTag()).lower() == "yes"):
                imcombine_mode = imcombine.MODE_FDU_DIFF_PAIRING
            (data, header) = imcombine_method(flats, outfile=mffilename, method="median", scale="median", mef=flats[0]._mef, log=self._log, mode=imcombine_mode, returnHeader=True)
            masterFlat = fatboyCalib(self._pname, "master_flat", flats[0], data=data, tagname=mfname, headerExt=header, log=self._log)
        else:
            print("flatDivideProcess::createMasterFlat> Error: invalid flat_method: "+flatmethod)
            self._log.writeLog(__name__, " invalid flat_method: "+flatmethod, type=fatboyLog.ERROR)
            return None
        masterFlat.setType("master_flat", True)
        #set flat_method property
        masterFlat.setProperty("flat_method", flatmethod)
        updateHeaderEntry(masterFlat._header, "FLATMTHD", flatmethod) #Use wrapper function to update header
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and not os.access(mffilename, os.F_OK)):
            #Optionally save if write_calib_output = yes
            masterFlat.writeTo(mffilename)
        #disable these flats as master flat has been created
        for flat in flats:
            flat.disable()
        return masterFlat
    #end createMasterFlat

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Flat Divide")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For flatDivide, this dict should have one entry 'masterFlat' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'masterFlat' in calibs):
            #Failed to obtain master flat frame
            #Issue error message and disable this FDU
            print("flatDivideProcess::execute> ERROR: Flat not divided for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Flat not divided for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        fdfile = "flatDivided/fd_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, fdfile)):
            return True

        #get master flat
        masterFlat = calibs['masterFlat']
        #Renormalize master flat to median value of 1
        if (self.getOption("median_section", fdu.getTag()) is not None):
            #section of format "320:384,1024:1080"
            try:
                #Parse out into list
                section = self.getOption("median_section", fdu.getTag()).split(",")
                if (len(section) != 2):
                    print("flatDivideProcess::execute> Error: Bad section given: "+str(section))
                    self._log.writeLog(__name__, " Bad section given: "+str(section), type=fatboyLog.ERROR)
                    #disable this FDU
                    fdu.disable()
                    return False
                for j in range(len(section)):
                    section[j] = section[j].strip().split(":")
                    #Mask out data
                    if (len(section[j]) != 2):
                        print("flatDivideProcess::execute> Error: Bad section given: "+str(section))
                        self._log.writeLog(__name__, " Bad section given: "+str(section), type=fatboyLog.ERROR)
                        #disable this FDU
                        fdu.disable()
                        return False
                    section[j][0] = int(section[j][0])
                    section[j][1] = int(section[j][1])
                masterFlat.setProperty("median_section_indices", section)
                masterFlat.tagDataAs("median_section", masterFlat.getData()[section[0][0]:section[0][1], section[1][0]:section[1][1]])
                print("flatDivideProcess::execute> Using section "+str(section)+" for median...")
                self._log.writeLog(__name__, "Using section "+str(section)+" for median...")
            except ValueError as ex:
                print("flatDivideProcess::execute> Error: invalid format in median_section: "+str(ex))
                self._log.writeLog(__name__, " invalid format in median_section: "+str(ex), type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return False
        #Renormalize -- using median_section if it is tagged
        masterFlat.renormalize()
        #call flatDivideImage helper function to do gpu/cpu division
        fdu.updateData(self.flatDivideImage(fdu.getData(), masterFlat.getData()))
        fdu._header.add_history('Flat divided using '+masterFlat._id)
        return True
    #end execute

    #Look at XML to determine dome flat types
    def findDomeFlatType(self, fdu, properties):
        #properties has flat_method = dome_on-off.  Only need to process flats matching this flat_method.
        lampoff = self.getOption("flat_lamp_off_files", fdu.getTag()).lower()
        if (os.access(lampoff, os.F_OK)):
            #This is an ASCII file listing off flats
            #Process entire file here
            lampoffList = readFileIntoList(lampoff)
            #loop over lampoffList do a split on each line
            for j in range(len(lampoffList)-1, -1, -1):
                lampoffList[j] = lampoffList[j].split()
                #remove misformatted lines
                if (len(lampoffList[j]) < 3):
                    print("flatDivideProcess::findDomeFlatType> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
                lampoffList[j][0] = lampoffList[j][0].lower()
                try:
                    lampoffList[j][1] = int(lampoffList[j][1])
                    lampoffList[j][2] = int(lampoffList[j][2])
                except Exception:
                    print("flatDivideProcess::findFlatMethods> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
            #loop over dataset and assign property to all dome_on-off flats that don't already have 'flat_type' property.
            for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_FLAT, tag=fdu.getTag(), properties=properties):
                if (fdu.hasProperty("flat_type")):
                    #this FDU already has flat_type set
                    continue
                fdu.setProperty("flat_type", "lamp_on") #set to on by default then loop over lampoffList for matches
                #offLine = [ 'identifier', startIdx, stopIdx ]
                for offLine in lampoffList:
                    if (fdu._id.lower().find(offLine[0]) != -1 and int(fdu._index) >= offLine[1] and int(fdu._index) <= offLine[2]):
                        #Partial match for identifier and index within range given
                        fdu.setProperty("flat_type", "lamp_off")
        else:
            #This is a filename fragment.  Find which flats match
            #loop over dataset and assign property to all dome_on-off flats that don't already have 'flat_type' property.
            for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_FLAT, tag=fdu.getTag(), properties=properties):
                if (not fdu.hasProperty("flat_type")):
                    if (fdu._id.lower().find(lampoff) != -1):
                        #partial match for identifier
                        fdu.setProperty("flat_type", "lamp_off")
                    else:
                        #no match -- this is a lamp on
                        fdu.setProperty("flat_type", "lamp_on")
    #end findDomeFlatType

    #Look at XML to determine flat field methods
    def findFlatMethods(self, fdu):
        flatmethod = self.getOption("flat_method", fdu.getTag())
        if (flatmethod.lower() == "dome_on" or flatmethod.lower() == "dome_on-off" or flatmethod.lower() == "sky" or flatmethod.lower() == "twilight"):
            flatmethod = flatmethod.lower()
            #loop over dataset and assign property to all fdus that don't already have 'flat_method' property.
            #some FDUs may have used xml to define flat_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (not fdu.hasProperty("flat_method")):
                    fdu.setProperty("flat_method", flatmethod)
        elif (os.access(flatmethod, os.F_OK)):
            #This is an ASCII file listing filter/identifier and flat method
            #Process entire file here
            methodList = readFileIntoList(flatmethod)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) < 2):
                    print("flatDivideProcess::findFlatMethods> Warning: line "+str(j)+" misformatted in "+flatmethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+flatmethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                methodList[j][1] = methodList[j][1].lower()
                if (methodList[j][1] != "dome_on" and methodList[j][1] != "dome_on-off" and methodList[j][1] != "sky" and methodList[j][1] != "twilight"):
                    print("flatDivideProcess::findFlatMethods> Warning: line "+str(j)+" misformatted in "+flatmethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+flatmethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'flat_method' property.
            #some FDUs may have used xml to define flat_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (fdu.hasProperty("flat_method")):
                    #this FDU already has flat_method set
                    continue
                #method = [ 'Filter/identifier', 'method' ]
                for method in methodList:
                    if (method[0].lower() == fdu.filter.lower()):
                        fdu.setProperty("flat_method", method[1])
                        #Exact match for filter
                    elif (len(method[0]) > 2 and fdu._id.lower().find(method[0].lower()) != -1):
                        #Partial match for identifier
                        fdu.setProperty("flat_method", method[1])
        else:
            print("flatDivideProcess::findFlatMethods> Error: invalid flat_method: "+flatmethod)
            self._log.writeLog(__name__, " invalid flat_method: "+flatmethod, type=fatboyLog.ERROR)
    #end findFlatTypes

    # Actually perform flat division
    def flatDivideImage(self, image, flat):
        t = time.time()
        blocks = image.size//512
        if (image.size % 512 != 0):
            blocks += 1
        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            #Use GPU
            if (not superFATBOY.threaded()):
                global fatboy_mod
            else:
                fatboy_mod = get_fatboy_mod()
            divArrays = fatboy_mod.get_function("divideArrays_float")
            divArrays(drv.InOut(image), drv.In(flat), int32(image.size), grid=(blocks,1), block=(block_size,1,1))
        else:
            #find points where flat is zero and set them to 1 to avoid divideByZeroException
            flatzeros = flat == 0
            flat[flatzeros] = 1
            image /= flat
            image[flatzeros] = 0
            flat[flatzeros] = 0
        print("Division time: ",time.time()-t)
        return image
    #end flatDivideImage

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        mffilename = self.getCalib("masterFlat", fdu.getTag())
        if (mffilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mffilename, os.F_OK)):
                print("flatDivideProcess::getCalibs> Using master flat "+mffilename+"...")
                self._log.writeLog(__name__, "Using master flat "+mffilename+"...")
                calibs['masterFlat'] = fatboyCalib(self._pname, "master_flat", fdu, filename=mffilename, log=self._log)
                return calibs
            else:
                print("flatDivideProcess::getCalibs> Warning: Could not find master flat "+mffilename+"...")
                self._log.writeLog(__name__, "Could not find master flat "+mffilename+"...", type=fatboyLog.WARNING)

        #First look for property flat_method
        properties = dict()
        if (not fdu.hasProperty("flat_method")):
            #Look at XML options to find flat method and assign it to FDUs
            self.findFlatMethods(fdu)
        properties['flat_method'] = fdu.getProperty("flat_method")
        if (properties['flat_method'] is None):
            print("flatDivideProcess::getCalibs> Error: Could not find flat_method for "+fdu.getFullId())
            self._log.writeLog(__name__, " Could not find flat_method for "+fdu.getFullId(), type=fatboyLog.ERROR)
            return calibs

        #1) Check for an already created master flat frame matching filter/section and TAGGED for this object
        masterFlat = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_flat", filter=fdu.filter, section=fdu.section, properties=properties)
        if (masterFlat is not None):
            #Found master flat.  Return here
            calibs['masterFlat'] = masterFlat
            return calibs
        #2) Check for individual flat frames matching filter/section to create master flat and TAGGED for this object
        flats = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_FLAT, filter=fdu.filter, section=fdu.section, properties=properties)
        if (len(flats) > 0):
            #Found flats associated with this fdu.  Create master flat.
            print("flatDivideProcess::getCalibs> Creating Master Flat for tagged object "+fdu._id+", filter "+str(fdu.filter)+" using METHOD "+properties['flat_method']+"...")
            self._log.writeLog(__name__, " Creating Master Flat for tagged object "+fdu._id+", filter "+str(fdu.filter)+" using METHOD "+properties['flat_method']+"...")
            #First recursively process
            self.recursivelyExecute(flats, prevProc)
            #convenience method
            masterFlat = self.createMasterFlat(fdu, flats, properties)
            if (masterFlat is None):
                return calibs
            self._fdb.appendCalib(masterFlat)
            calibs['masterFlat'] = masterFlat
            return calibs
        #3) Check for an already created master flat frame matching filter/section
        masterFlat = self._fdb.getMasterCalib(self._pname, obstype="master_flat", filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties)
        if (masterFlat is not None):
            #Found master flat.  Return here
            calibs['masterFlat'] = masterFlat
            return calibs
        #4) Check for individual flat frames matching filter/section to create master flat
        flats = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_FLAT, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties)
        if (len(flats) > 0):
            #Found flats associated with this fdu.  Create master flat.
            print("flatDivideProcess::getCalibs> Creating Master Flat for filter: "+str(fdu.filter)+" using METHOD "+properties['flat_method']+"...")
            self._log.writeLog(__name__, " Creating Master Flat for filter: "+str(fdu.filter)+" using METHOD "+properties['flat_method']+"...")
            #First recursively process
            self.recursivelyExecute(flats, prevProc)
            #convenience method
            masterFlat = self.createMasterFlat(fdu, flats, properties)
            if (masterFlat is None):
                return calibs
            self._fdb.appendCalib(masterFlat)
            calibs['masterFlat'] = masterFlat
            return calibs
        #5) Check default_master_flat for matching filter/section
        defaultMasterFlats = []
        if (self.getOption('default_master_flat', fdu.getTag()) is not None):
            dmflist = self.getOption('default_master_flat', fdu.getTag())
            if (dmflist.count(',') > 0):
                #comma separated list
                defaultMasterFlats = dmflist.split(',')
                removeEmpty(defaultMasterFlats)
                for j in range(len(defaultMasterFlats)):
                    defaultMasterFlats[j] = defaultMasterFlats[j].strip()
            elif (dmflist.endswith('.fit') or dmflist.endswith('.fits')):
                #FITS file given
                defaultMasterFlats.append(dmflist)
            elif (dmflist.endswith('.dat') or dmflist.endswith('.list') or dmflist.endswith('.txt')):
                #ASCII file list
                defaultMasterFlats = readFileIntoList(dmflist)
            for mflatfile in defaultMasterFlats:
                #Loop over list of default master flats
                masterFlat = fatboyImage(mflatfile, log=self._log)
                #read header and initialize
                masterFlat.readHeader()
                masterFlat.initialize()
                if (masterFlat.filter != fdu.filter):
                    #does not match filter
                    continue
                if (fdu.section is not None):
                    #check section if applicable
                    section = -1
                    if (masterFlat.hasHeaderValue('SECTION')):
                        section = masterFlat.getHeaderValue('SECTION')
                    else:
                        idx = masterFlat.getFilename().rfind('.fit')
                        if (masterFlat.getFilename()[idx-2] == 'S' and isDigit(masterFlat.getFilename()[idx-1])):
                            section = int(masterFlat.getFilename()[idx-1])
                    if (section != fdu.section):
                        continue
                masterFlat.setType("master_flat")
                #Found matching master flat
                print("flatDivideProcess::getCalibs> Using default master flat "+masterFlat.getFilename())
                self._log.writeLog(__name__, " Using default master flat "+masterFlat.getFilename())
                self._fdb.appendCalib(masterFlat)
                calibs['masterFlat'] = masterFlat
                return calibs
        #6) Look at previous master flats to see if any has a history of being used as master flat for
        #this _id and filter combination from step 7 below.
        masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
        for mflat in masterFlats:
            if (mflat.hasHistory('master_flat::'+fdu._id+'::'+str(fdu.filter))):
                #Use this master flat
                print("flatDivideProcess::getCalibs> Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter))
                self._log.writeLog(__name__, " Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter))
                #Already in _calibs, no need to appendCalib
                calibs['masterFlat'] = mflat
                return calibs
        #7) Prompt user for flat file
        print("List of flats, filters, and sections:")
        masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
        for mflat in masterFlats:
            print(mflat.getFilename(), mflat.filter, mflat.section)
        for mflatfile in defaultMasterFlats:
            #Loop over list of default master flats too
            #mflat = fatboyImage(mflatfile, log=self._log)
            masterFlat = fatboyCalib(self._pname, "master_flat", fdu, filename=mflatfile, log=self._log)
            #read header and initialize
            mflat.readHeader()
            mflat.initialize()
            print(mflatfile, mflat.filter, mflat.section)
        tmp = input("Select a filename to use as a flat: ")
        mffilename = tmp
        #Now find if input matches one of these filenames
        for mflat in masterFlats:
            if (mflat.getFilename() == mffilename):
                #Found matching master flat
                print("flatDivideProcess::getCalibs> Using master flat "+mflat.getFilename())
                self._log.writeLog(__name__, " Using master flat "+mflat.getFilename())
                mflat.setHistory('master_flat::'+fdu._id+'::'+str(fdu.filter), 'yes')
                #Already in _calibs, no need to appendCalib
                calibs['masterFlat'] = mflat
                return calibs
        #If not found yet, check default master flats
        if (mffilename in defaultMasterFlats):
            mflat = fatboyImage(mflatfile, log=self._log)
            #read header and initialize
            mflat.readHeader()
            mflat.initialize()
            print("flatDivideProcess::getCalibs> Using master flat "+mflat.getFilename())
            self._log.writeLog(__name__, " Using master flat "+mflat.getFilename())
            mflat.setHistory('master_flat::'+fdu._id+'::'+str(fdu.filter), 'yes')
            self._fdb.appendCalib(mflat)
            calibs['masterFlat'] = mflat
            return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_flat', None)
        self._options.setdefault('flat_method', "dome_on")
        self._optioninfo.setdefault('flat_method', 'dome_on | dome_on-off | sky | twilight')
        self._options.setdefault('flat_lamp_off_files', 'off')
        self._options.setdefault('flat_sky_reject_type','none')
        self._options.setdefault('flat_sky_nlow','1')
        self._options.setdefault('flat_sky_nhigh','1')
        self._options.setdefault('flat_sky_lsigma','5')
        self._options.setdefault('flat_sky_hsigma','5')
        self._options.setdefault('median_section', None)
        self._optioninfo.setdefault('median_section', 'An optional subsection of the image using slice notation which will\nbe used to calculate the median value\nfor renormalization.  E.g. 400:1200,500:1000')
        self._options.setdefault('sky_flat_include_list','')
        self._options.setdefault('sky_flat_exclude_list','')
        self._options.setdefault('twilight_pair_ramps','no')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/flatDivided", os.F_OK)):
            os.mkdir(outdir+"/flatDivided",0o755)
        #Create output filename
        fdfile = outdir+"/flatDivided/fd_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(fdfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(fdfile)
        if (not os.access(fdfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(fdfile)
    #end writeOutput
