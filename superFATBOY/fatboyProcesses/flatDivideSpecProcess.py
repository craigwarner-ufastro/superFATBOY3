from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_imcombine, imcombine

block_size = 512

class flatDivideSpecProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

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
        #Noisemap file
        nmfile = outdir+"/masterFlats/NM_mflat-"+str(flatmethod)+"-"+str(flats[0].filter).replace(" ","_")+"-"+str(flats[0]._id)
        if (fdu.getTag(mode="composite") is not None):
            nmfile += "-"+fdu.getTag(mode="composite").replace(" ","_")
        nmfile += ".fits"
        if (os.access(prevmffilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevmffilename)
        elif (os.access(prevmffilename, os.F_OK)):
            #file already exists
            print("flatDivideSpecProcess::createMasterFlat> Master flat "+prevmffilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Master flat "+prevmffilename+" already exists!  Re-using...")
            masterFlat = fatboySpecCalib(self._pname, "master_flat", flats[0], filename=prevmffilename, tagname=mfname, log=self._log)
            #set flat_method property
            masterFlat.setProperty("flat_method", flatmethod)
            #set specmode property
            masterFlat.setProperty("specmode", flats[0].getProperty("specmode"))
            #set dispersion property
            masterFlat.setProperty("dispersion", flats[0].getProperty("dispersion"))
            #check to see if masterFlat has been normalized
            if (masterFlat.hasHeaderValue('NORMAL01')):
                #has been normalized already
                masterFlat.setProperty("normalized", True)
            #Check to see if a noisemap exists
            if (os.access(nmfile, os.F_OK)):
                nm = pyfits.open(nmfile)
                mef = findMef(nm)
                #Tag noisemap data.  tagDataAs() will handle byteswap
                masterFlat.tagDataAs("noisemap", nm[mef].data)
                nm.close()
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
                print("flatDivideSpecProcess::createMasterFlat> ERROR: No ON flats found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No ON flats found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            if (len(offFlats) == 0):
                #All on flats, no OFF flats!  Error!
                print("flatDivideSpecProcess::createMasterFlat> ERROR: No OFF flats found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No OFF flats found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            #Combine ON flats
            (data, header) = imcombine_method(onFlats, outfile=mffilename, method="median", mef=onFlats[0]._mef, log=self._log, returnHeader=True)
            #Combine and subtract OFF flats
            (offData, offHeader) = imcombine_method(offFlats, outfile=None, method="median", mef=offFlats[0]._mef, log=self._log, returnHeader=True)
            #Calculate noisemap first before subtracting offData from data
            ncomb1 = float(header['NCOMBINE'])
            ncomb2 = float(offHeader['NCOMBINE'])
            #Create noisemap
            if (self._fdb.getGPUMode()):
                nm = noisemaps_mflat_dome_on_off_gpu(data, offData, ncomb1, ncomb2)
            else:
                nm = sqrt(abs(data/ncomb1)+abs(offData/ncomb2))
            #Now subtract off flats
            data -= offData
            offData = None #free memory
            #If output written to disk, update FITS file with difference
            if (mffilename is not None):
                mflat = pyfits.open(mffilename, "update")
                mflat[onFlats[0]._mef].data = data
                mflat.verify('silentfix')
                mflat.flush()
                mflat.close()
            masterFlat = fatboySpecCalib(self._pname, "master_flat", onFlats[0], data=data, tagname=mfname, headerExt=header, log=self._log)
            #Tag noisemap
            masterFlat.tagDataAs("noisemap", nm)
        elif (flatmethod == "dome_on"):
            #dome_on method
            #Combine ON flats
            (data, header) = imcombine_method(flats, outfile=mffilename, method="median", mef=flats[0]._mef, log=self._log, returnHeader=True)
            masterFlat = fatboySpecCalib(self._pname, "master_flat", flats[0], data=data, tagname=mfname, headerExt=header, log=self._log)
            #Create noisemap
            ncomb = float(masterFlat.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterFlat.getData(), ncomb)
            else:
                nm = sqrt(abs(masterFlat.getData()/ncomb))
            masterFlat.tagDataAs("noisemap", nm)
        else:
            print("flatDivideSpecProcess::createMasterFlat> Error: invalid flat_method: "+flatmethod)
            self._log.writeLog(__name__, " invalid flat_method: "+flatmethod, type=fatboyLog.ERROR)
            return None
        masterFlat.setType("master_flat", True)
        #set flat_method property
        masterFlat.setProperty("flat_method", flatmethod)
        updateHeaderEntry(masterFlat._header, "FLATMTHD", flatmethod) #Use wrapper function to update header
        #set specmode property
        masterFlat.setProperty("specmode", flats[0].getProperty("specmode"))
        updateHeaderEntry(masterFlat._header, "SPECMODE", flats[0].getProperty("specmode")) #Use wrapper function to update header
        #set dispersion property
        masterFlat.setProperty("dispersion", flats[0].getProperty("dispersion"))
        updateHeaderEntry(masterFlat._header, "DISPDIR", flats[0].getProperty("dispersion")) #Use wrapper function to update header
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/masterFlats", os.F_OK)):
            os.mkdir(outdir+"/masterFlats",0o755)
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            if (os.access(mffilename, os.F_OK)):
                os.unlink(mffilename)
            #Optionally save if write_calib_output = yes
            masterFlat.writeTo(mffilename)
        #Write out noisemap if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes"):
            if (not os.access(outdir+"/masterFlats", os.F_OK)):
                os.mkdir(outdir+"/masterFlats",0o755)
            if (os.access(nmfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                masterFlat.writeTo(nmfile, tag="noisemap")

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
            print("flatDivideSpecProcess::execute> ERROR: Flat not divided for "+fdu.getFullId()+" (filter="+str(fdu.filter)+", grism="+str(fdu.grism)+", specmode="+str(fdu.getProperty("specmode"))+").  Discarding Image!")
            self._log.writeLog(__name__, "Flat not divided for "+fdu.getFullId()+" (filter="+str(fdu.filter)+", grism="+str(fdu.grism)+", specmode="+str(fdu.getProperty("specmode"))+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        fdfile = "flatDivided/fd_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, fdfile)):
            #Also check if "cleanFrame" exists
            cleanfile = "flatDivided/clean_fd_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "noisemap" exists
            nmfile = "flatDivided/NM_fd_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")

            if ('masterFlat' in calibs and not calibs['masterFlat'].hasProperty("normalized")):
                #Check if output exists first
                mffile = "masterFlats/"+calibs['masterFlat'].getFullId()
                if (self.checkOutputExists(calibs['masterFlat'], mffile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "normalized" = True if normalized
                    if (calibs['masterFlat'].hasHeaderValue('NORMAL01')):
                        calibs['masterFlat'].setProperty("normalized", True)
            return True

        #get master flat
        masterFlat = calibs['masterFlat']

        #Check if masterFlat is normalized
        if (not masterFlat.hasProperty("normalized") and self.getOption("normalize_flat", fdu.getTag()).lower() == "yes"):
            #Renormalize master flat to median value of 1 (and for each slitlet/order in MOS/IFU data)
            self.normalizeFlat(masterFlat, fdu, prevProc, calibs)

        #Tag current data as "cleanFrame" before dividing by flat
        #or updating noisemap
        fdu.tagDataAs("cleanFrame")

        #call flatDivideImage helper function to do gpu/cpu division
        fdu.updateData(self.flatDivideImage(fdu.getData(), masterFlat.getData()))
        fdu._header.add_history('Flat divided using '+masterFlat._id)

        #Propagate noisemap for spectroscopy data AFTER dividing.
        #For propagating noisemap with division, need both before ("cleanFrame") and after division images
        if (fdu.hasProperty("noisemap")):
            self.updateNoisemap(fdu, masterFlat)
        return True
    #end execute

    #Look at XML to determine dome flat types
    def findDomeFlatType(self, fdu, properties):
        #properties has flat_method = dome_on-off.  Only need to process flats matching this flat_method.
        lampoff = self.getOption("flat_lamp_off_files", fdu.getTag())
        if (os.access(lampoff, os.F_OK)):
            #This is an ASCII file listing off flats
            #Process entire file here
            lampoffList = readFileIntoList(lampoff)
            #loop over lampoffList do a split on each line
            for j in range(len(lampoffList)-1, -1, -1):
                lampoffList[j] = lampoffList[j].split()
                #remove misformatted lines
                if (len(lampoffList[j]) < 3):
                    print("flatDivideSpecProcess::findDomeFlatType> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
                lampoffList[j][0] = lampoffList[j][0].lower()
                try:
                    lampoffList[j][1] = int(lampoffList[j][1])
                    lampoffList[j][2] = int(lampoffList[j][2])
                except Exception:
                    print("flatDivideSpecProcess::findDomeFlatType> Warning: line "+str(j)+" misformatted in "+lampoff)
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
        elif (fdu.hasHeaderValue(lampoff)):
            #This is a FITS keyword
            lampoffVal = self.getOption("flat_lamp_off_header_value", fdu.getTag())
            #First check flats TAGGED for this object
            fdus = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_FLAT, properties=properties)
            if (len(fdus) > 0):
                for fdu in fdus:
                    if (fdu.hasProperty("flat_type")):
                        #this FDU already has flat_type set
                        continue
                    if (str(fdu.getHeaderValue(lampoff)) == lampoffVal):
                        fdu.setProperty("flat_type", "lamp_off")
                    else:
                        fdu.setProperty("flat_type", "lamp_on") #set to on if it does not match
            else:
                #loop over dataset and assign property to all dome_on-off flats that don't already have 'flat_type' property.
                for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_FLAT, tag=fdu.getTag(), properties=properties):
                    if (fdu.hasProperty("flat_type")):
                        #this FDU already has flat_type set
                        continue
                    if (str(fdu.getHeaderValue(lampoff)) == lampoffVal):
                        fdu.setProperty("flat_type", "lamp_off")
                    else:
                        fdu.setProperty("flat_type", "lamp_on") #set to on if it does not match
        else:
            #This is a filename fragment.  Find which flats match
            #loop over dataset and assign property to all dome_on-off flats that don't already have 'flat_type' property.
            for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_FLAT, tag=fdu.getTag(), properties=properties):
                if (not fdu.hasProperty("flat_type")):
                    if (fdu._id.lower().find(lampoff.lower()) != -1):
                        #partial match for identifier
                        fdu.setProperty("flat_type", "lamp_off")
                    else:
                        #no match -- this is a lamp on
                        fdu.setProperty("flat_type", "lamp_on")
    #end findDomeFlatType

    #Look at XML to determine flat field methods
    def findFlatMethods(self, fdu):
        flatmethod = self.getOption("flat_method", fdu.getTag())
        if (flatmethod.lower() == "dome_on" or flatmethod.lower() == "dome_on-off"):
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
                    print("flatDivideSpecProcess::findFlatMethods> Warning: line "+str(j)+" misformatted in "+flatmethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+flatmethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                methodList[j][1] = methodList[j][1].lower()
                if (methodList[j][1] != "dome_on" and methodList[j][1] != "dome_on-off" and methodList[j][1] != "sky" and methodList[j][1] != "twilight"):
                    print("flatDivideSpecProcess::findFlatMethods> Warning: line "+str(j)+" misformatted in "+flatmethod)
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
            print("flatDivideSpecProcess::findFlatMethods> Error: invalid flat_method: "+flatmethod)
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
                print("flatDivideSpecProcess::getCalibs> Using master flat "+mffilename+"...")
                self._log.writeLog(__name__, "Using master flat "+mffilename+"...")
                calibs['masterFlat'] = fatboySpecCalib(self._pname, "master_flat", fdu, filename=mffilename, log=self._log)
                return calibs
            else:
                print("flatDivideSpecProcess::getCalibs> Warning: Could not find master flat "+mffilename+"...")
                self._log.writeLog(__name__, "Could not find master flat "+mffilename+"...", type=fatboyLog.WARNING)

        #Look for slitmask
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("flatDivideSpecProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("flatDivideSpecProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and flat_method
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        #If flat_selection == object_keyword, look for flats that have OBJECT keyword matching that of fdu
        if (self.getOption('flat_selection').lower() == "object_keyword"):
            headerVals['object_keyword'] = fdu.getHeaderValue('object_keyword')

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")

        if (not fdu.hasProperty("flat_method")):
            #Look at XML options to find flat method and assign it to FDUs
            self.findFlatMethods(fdu)
        properties['flat_method'] = fdu.getProperty("flat_method")
        if (properties['flat_method'] is None):
            print("flatDivideSpecProcess::getCalibs> Error: Could not find flat_method for "+fdu.getFullId())
            self._log.writeLog(__name__, " Could not find flat_method for "+fdu.getFullId(), type=fatboyLog.ERROR)
            return calibs

        #1) Check for an already created master flat frame matching specmode/filter/grism and TAGGED for this object
        masterFlat = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_flat", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
        if (masterFlat is not None):
            #Found master flat.  Return here
            calibs['masterFlat'] = masterFlat
            return calibs
        #2) Check for individual flat frames matching specmode/filter/grism to create master flat and TAGGED for this object
        flats = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_FLAT, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
        if (len(flats) > 0):
            #Found flats associated with this fdu.  Create master flat.
            print("flatDivideSpecProcess::getCalibs> Creating Master Flat for tagged object "+fdu._id+", filter "+str(fdu.filter)+", grism "+str(fdu.grism)+", specmode "+str(fdu.getProperty("specmode"))+" using METHOD "+properties['flat_method']+"...")
            self._log.writeLog(__name__, " Creating Master Flat for tagged object "+fdu._id+", filter "+str(fdu.filter)+", grism "+str(fdu.grism)+", specmode "+str(fdu.getProperty("specmode"))+" using METHOD "+properties['flat_method']+"...")
            #First recursively process
            self.recursivelyExecute(flats, prevProc)
            #convenience method
            masterFlat = self.createMasterFlat(fdu, flats, properties)
            if (masterFlat is None):
                return calibs
            self._fdb.appendCalib(masterFlat)
            calibs['masterFlat'] = masterFlat
            return calibs
        #3) Check for an already created master flat frame matching specmode/filter/grism
        masterFlat = self._fdb.getMasterCalib(self._pname, obstype="master_flat", filter=fdu.filter, tag=fdu.getTag(), section=fdu.section, properties=properties, headerVals=headerVals)
        if (masterFlat is not None):
            #Found master flat.  Return here
            calibs['masterFlat'] = masterFlat
            return calibs
        #4) Check for individual flat frames matching specmode/filter/grism to create master flat
        flats = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_FLAT, filter=fdu.filter, tag=fdu.getTag(), section=fdu.section, properties=properties, headerVals=headerVals)
        if (len(flats) > 0):
            #Found flats associated with this fdu.  Create master flat.
            print("flatDivideSpecProcess::getCalibs> Creating Master Flat for filter: "+str(fdu.filter)+", grism "+str(fdu.grism)+", specmode "+str(fdu.getProperty("specmode"))+" using METHOD "+properties['flat_method']+"...")
            self._log.writeLog(__name__, " Creating Master Flat for filter: "+str(fdu.filter)+", grism "+str(fdu.grism)+", specmode "+str(fdu.getProperty("specmode"))+" using METHOD "+properties['flat_method']+"...")
            #First recursively process
            self.recursivelyExecute(flats, prevProc)
            #convenience method
            masterFlat = self.createMasterFlat(fdu, flats, properties)
            if (masterFlat is None):
                return calibs
            self._fdb.appendCalib(masterFlat)
            calibs['masterFlat'] = masterFlat
            return calibs
        #5) Check default_master_flat for matching specmode/filter/grism
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
                #masterFlat = fatboyImage(mflatfile)
                masterFlat = fatboySpecCalib(self._pname, "master_flat", fdu, filename=mflatfile, log=self._log)
                #read header and initialize
                masterFlat.readHeader()
                masterFlat.initialize()
                if (masterFlat.filter != fdu.filter):
                    #does not match filter
                    continue
                masterFlat.setType("master_flat")
                #set flat_method property
                masterFlat.setProperty("flat_method", properties['flat_method'])
                #set specmode property
                masterFlat.setProperty("specmode", properties['specmode'])
                #check to see if masterFlat has been normalized
                if (masterFlat.hasHeaderValue('NORMAL01')):
                    #has been normalized already
                    masterFlat.setProperty("normalized", True)
                #Found matching master flat
                print("flatDivideSpecProcess::getCalibs> Using default master flat "+masterFlat.getFilename())
                self._log.writeLog(__name__, " Using default master flat "+masterFlat.getFilename())
                self._fdb.appendCalib(masterFlat)
                calibs['masterFlat'] = masterFlat
                return calibs
        #6) Look at previous master flats to see if any has a history of being used as master flat for
        #this _id and filter combination from step 7 below.
        masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
        for mflat in masterFlats:
            if (mflat.hasHistory('master_flat::'+fdu._id+'::'+str(fdu.filter)+'::'+str(fdu.grism)+'::'+str(fdu.getProperty("specmode")))):
                #Use this master flat
                print("flatDivideSpecProcess::getCalibs> Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter)+", grism "+str(mflat.grism)+", specmode "+str(mflat.getProperty("specmode")))
                self._log.writeLog(__name__, "Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter)+", grism "+str(mflat.grism)+", specmode "+str(mflat.getProperty("specmode")))
                #Already in _calibs, no need to appendCalib
                calibs['masterFlat'] = mflat
                return calibs
        #7) Prompt user for flat file
        if (self.getOption('prompt_for_missing_flat', fdu.getTag()).lower() == "yes"):
            print("List of flats, filters, grisms, and specmodes:")
            masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
            for mflat in masterFlats:
                print(mflat.getFilename(), mflat.filter, mflat.getHeaderValue('grism_keyword'), mflat.getProperty("specmode"))
            for mflatfile in defaultMasterFlats:
                #Loop over list of default master flats too
                mflat = fatboyImage(mflatfile, log=self._log)
                #read header and initialize
                mflat.readHeader()
                mflat.initialize()
                print(mflatfile, mflat.filter, mflat.getHeaderValue('grism_keyword'), mflat.getProperty("specmode"))
            tmp = input("Select a filename to use as a flat: ")
            mffilename = tmp
            #Now find if input matches one of these filenames
            for mflat in masterFlats:
                if (mflat.getFilename() == mffilename):
                    #Found matching master flat
                    print("flatDivideSpecProcess::getCalibs> Using master flat "+mflat.getFilename())
                    self._log.writeLog(__name__, " Using master flat "+mflat.getFilename())
                    mflat.setHistory('master_flat::'+fdu._id+'::'+str(fdu.filter)+'::'+str(fdu.grism)+'::'+str(fdu.getProperty("specmode")), 'yes')
                    #Already in _calibs, no need to appendCalib
                    calibs['masterFlat'] = mflat
                    return calibs
        #If not found yet, check default master flats
        if (mffilename in defaultMasterFlats):
            mflat = fatboyImage(mflatfile, log=self._log)
            #read header and initialize
            mflat.readHeader()
            mflat.initialize()
            print("flatDivideSpecProcess::getCalibs> Using master flat "+mflat.getFilename())
            self._log.writeLog(__name__, "Using master flat "+mflat.getFilename())
            mflat.setHistory('master_flat::'+fdu._id+'::'+str(fdu.filter)+'::'+str(fdu.grism)+'::'+str(fdu.getProperty("specmode")), 'yes')
            self._fdb.appendCalib(mflat)
            calibs['masterFlat'] = mflat
            return calibs
        return calibs
    #end getCalibs

    #Renormalize master flat to median value of 1 (and for each slitlet/order in MOS/IFU data)
    def normalizeFlat(self, masterFlat, fdu, prevProc, calibs):
        print("flatDivideSpecProcess::normalizeFlat> Normalizing master flat "+masterFlat._id)
        self._log.writeLog(__name__, "Normalizing master flat "+masterFlat._id)
        #Get options
        lowThresh = int(self.getOption("flat_low_thresh", fdu.getTag()))
        lowReplace = int(self.getOption("flat_low_replace", fdu.getTag()))
        hiThresh = int(self.getOption("flat_hi_thresh", fdu.getTag()))
        hiReplace = int(self.getOption("flat_hi_replace", fdu.getTag()))

        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            medVal = masterFlat.getMedian()
            if (self._fdb.getGPUMode()):
                #Divide and replace low/high pixels on GPU
                #normalizeFlat will update data and noisemap in FDU
                normalizeFlat(masterFlat, medVal, lowThresh, lowReplace, hiThresh, hiReplace, self._log)
            else:
                data = masterFlat.getData()
                data /= medVal
                #Replace low/high pixels
                if (lowThresh != 0):
                    b = (data < lowThresh)
                    data[b] = lowReplace
                    print("flatDivideSpecProcess::normalizeFlat> Replaced "+str(b.sum())+" pixels below "+str(lowThresh))
                    self._log.writeLog(__name__, "Replaced "+str(b.sum())+" pixels below "+str(lowThresh))
                if (hiThresh != 0):
                    b = (data > hiThresh)
                    data[b] = hiReplace
                    print("flatDivideSpecProcess::normalizeFlat> Replaced "+str(b.sum())+" pixels above "+str(hiThresh))
                    self._log.writeLog(__name__, "Replaced "+str(b.sum())+" pixels above "+str(hiThresh))
                masterFlat.updateData(data)
                #Update noisemap
                if (masterFlat.hasProperty("noisemap")):
                    #Get this FDU's noisemap
                    nm = masterFlat.getData(tag="noisemap")
                    #Divide by median
                    nm /= medVal
                    #Update FDU
                    masterFlat.tagDataAs("noisemap", nm)
            updateHeaderEntry(masterFlat._header, 'NORMAL01', medVal) #Use wrapper function to update header
        else:
            if ('slitmask' in calibs):
                slitmask = calibs['slitmask']
            elif (fdu.hasProperty('slitmask')):
                slitmask = fdu.getProperty('slitmask')
            else:
                #Use findSlitletProcess.getCalibs to get slitmask and create if necessary
                #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
                #Returns None on a failure
                fs_process = self._fdb.getProcessByName("findSlitlets")
                if (fs_process is None or not isinstance(fs_process, fatboyProcess)):
                    print("flatDivideSpecProcess::normalizeFlat> ERROR: could not find process findSlitlets!  Check your XML file!")
                    self._log.writeLog(__name__, "could not find process findSlitlets!  Check your XML file!", type=fatboyLog.ERROR)
                    return
                #Call setDefaultOptions and getCalibs on flatDivideSpecProcess
                fs_process.setDefaultOptions()
                calibs = fs_process.getCalibs(fdu, prevProc)
                if (not 'slitmask' in calibs):
                    #Failed to obtain slitmask
                    #Issue error message.  FDU will be disabled in execute()
                    print("flatDivideSpecProcess::execute> ERROR: Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!")
                    self._log.writeLog(__name__, "Slitmask not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!", type=fatboyLog.ERROR)
                    return
                slitmask = calibs['slitmask']
            if (not slitmask.hasProperty("nslits")):
                slitmask.setProperty("nslits", slitmask.getData().max())
            nslits = int(slitmask.getProperty("nslits"))
            if (self._fdb.getGPUMode()):
                #Divide and replace low/high pixels on GPU
                #normalizeMOSFlat will update data and noisemap in FDU
                #normalizeMOSFlat will also add NORMALxx header keywords
                normalizeMOSFlat(masterFlat, slitmask.getData(), nslits, lowThresh, lowReplace, hiThresh, hiReplace, self._log)
            else:
                data = masterFlat.getData()
                data[slitmask.getData() == 0] = 0
                if (masterFlat.hasProperty("noisemap")):
                    #Get this FDU's noisemap
                    nm = masterFlat.getData(tag="noisemap")
                    nm[slitmask.getData() == 0] = 0
                nlo = 0
                nhi = 0
                #CPU median kernel
                kernel = fatboyclib.median
                #Loop over slitlets in MOS/IFU data and normalize each
                for j in range(nslits):
                    slit = slitmask.getData() == (j+1)
                    medVal = arraymedian(data[slit], nonzero=True, kernel=kernel)
                    data[slit] /= medVal
                    #Replace low/hig pixels
                    if (lowThresh != 0):
                        b = (data[slit] < lowThresh)
                        data[slit][b] = lowReplace
                        nlo += b.sum()
                    if (hiThresh != 0):
                        b = (data[slit] < hiThresh)
                        data[slit][b] = hiReplace
                        nhi += b.sum()
                    key = 'NORMAL'
                    if (j+1 < 10):
                        key += '0'
                    key += str(j+1)
                    updateHeaderEntry(masterFlat._header, key, medVal) #Use wrapper function to update header
                    if (masterFlat.hasProperty("noisemap")):
                        nm[slit] /= medVal
                #Report total number of pixels replaced
                if (lowThresh != 0):
                    print("flatDivideSpecProcess::normalizeFlat> Replaced "+str(nlo)+" pixels below "+str(lowThresh))
                    self._log.writeLog(__name__, "Replaced "+str(nlo)+" pixels below "+str(lowThresh))
                if (hiThresh != 0):
                    print("flatDivideSpecProcess::normalizeFlat> Replaced "+str(nhi)+" pixels above "+str(hiThresh))
                    self._log.writeLog(__name__, "Replaced "+str(nhi)+" pixels below "+str(hiThresh))
                #update FDU and noisemap
                masterFlat.updateData(data)
                if (masterFlat.hasProperty("noisemap")):
                    masterFlat.tagDataAs("noisemap", nm)
        #Set property normalized = True
        masterFlat.setProperty("normalized", True)
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/masterFlats", os.F_OK)):
            os.mkdir(outdir+"/masterFlats",0o755)
        mffilename = outdir+"/masterFlats/"+masterFlat.getFullId()
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            if (os.access(mffilename, os.F_OK)):
                os.unlink(mffilename)
            if (not os.access(mffilename, os.F_OK)):
                #Optionally save if write_calib_output = yes
                masterFlat.writeTo(mffilename)
        #Write out noisemap if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes"):
            nmfile = mffilename.replace("masterFlats/mflat", "masterFlats/NM_mflat")
            if (os.access(nmfile, os.F_OK)):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                masterFlat.writeTo(nmfile, tag="noisemap")
    #end normalizeFlat

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_flat', None)
        self._options.setdefault('flat_method', "dome_on")
        self._optioninfo.setdefault('flat_method', 'dome_on | dome_on-off')
        self._options.setdefault('flat_lamp_off_files', 'off')
        self._optioninfo.setdefault('flat_lamp_off_files', 'An ASCII text file listing on and off flats\nor a filename fragment or a FITS header keyword\nfor identifying off flats')
        self._options.setdefault('flat_lamp_off_header_value', 'OFF')
        self._optioninfo.setdefault('flat_lamp_off_header_value', 'If flat_lamp_off_files is a FITS keyword, value for off flats')
        self._options.setdefault('flat_low_thresh', '0')
        self._options.setdefault('flat_low_replace', '1')
        self._options.setdefault('flat_hi_thresh', '0')
        self._options.setdefault('flat_hi_replace', '1')
        self._options.setdefault('flat_selection', 'all')
        self._optioninfo.setdefault('flat_selection', 'all | object_keyword')
        self._options.setdefault('normalize_flat', 'yes')
        self._options.setdefault('prompt_for_missing_flat', 'yes')
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
            #noisemaps_fd_gpu(fd_image, pre-fd_noisemap, pre-fd_image, mflat noisemap, mflat
            nm = noisemaps_fd_gpu(fdu.getData(), fdu.getData(tag="noisemap"), fdu.getData("cleanFrame"), masterFlat.getData("noisemap"), masterFlat.getData())
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
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/flatDivided/clean_fd_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/flatDivided/NM_fd_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
