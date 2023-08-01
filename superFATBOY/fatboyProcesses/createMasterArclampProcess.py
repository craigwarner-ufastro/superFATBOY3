from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_imcombine, imcombine

#Create master arclamps files from median of individual arclamps for each object set
class createMasterArclampProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createMasterArclamp(self, fdu, lamps, properties):
        #imcombine individual arclamp files
        masterLamp = None
        mlfilename = None
        lampmethod = properties['lamp_method']
        mlname = "masterArclamps/mlamp-"+str(lamps[0].filter).replace(" ","_")+"-"+str(lamps[0]._id)
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/masterArclamps", os.F_OK)):
                os.mkdir(outdir+"/masterArclamps",0o755)
            mlfilename = outdir+"/"+mlname+".fits"
        #Check to see if master arclamp frame exists already from a previous run
        prevmlfilename = outdir+"/"+mlname+".fits"
        nmfile = outdir+"/masterArclamps/NM_mlamp-"+str(lamps[0].filter).replace(" ","_")+"-"+str(lamps[0]._id)+".fits"
        if (os.access(prevmlfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevmlfilename)
        elif (os.access(prevmlfilename, os.F_OK)):
            #file already exists
            print("createMasterArclampProcess::createMasterArclamp> Master arclamp "+prevmlfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Master arclamp "+prevmlfilename+" already exists!  Re-using...")
            masterLamp = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=prevmlfilename, tagname=mlname, log=self._log)
            #set lamp_method property
            masterLamp.setProperty("lamp_method", lampmethod)
            masterLamp.setProperty("specmode", fdu.getProperty("specmode"))
            masterLamp.setProperty("dispersion", fdu.getProperty("dispersion"))
            #Check to see if a noisemap exists
            if (os.access(nmfile, os.F_OK)):
                nm = pyfits.open(nmfile)
                mef = findMef(nm)
                #Tag noisemap data.  tagDataAs() will handle byteswap
                masterLamp.tagDataAs("noisemap", nm[mef].data)
                nm.close()
            return masterLamp

        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        if (lampmethod == "lamp_on-off"):
            #dome_on-off method. Find off lamps
            offLamps = []
            onLamps = []
            for lamp in lamps:
                if (not lamp.hasProperty("lamp_type")):
                    #Look at XML options to find lamp type and assign it to FDUs
                    self.findLampType(fdu, properties)
                if (lamp.getProperty("lamp_type") == "lamp_off"):
                    #This is an OFF lamp
                    offLamps.append(lamp)
                else:
                    #This is an ON lamp
                    onLamps.append(lamp)
            if (len(onLamps) == 0):
                #All off lamps, no ON lamps!  Error!
                print("createMasterArclampProcess::createMasterLamp> ERROR: No ON lamps found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No ON lamps found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            if (len(offLamps) == 0):
                #All on lamps, no OFF lamps!  Error!
                print("createMasterArclampProcess::createMasterLamp> ERROR: No OFF lamps found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No OFF lamps found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            #Combine ON lamps
            (data, header) = imcombine_method(onLamps, outfile=mlfilename, method="median", mef=onLamps[0]._mef, log=self._log, returnHeader=True)
            #Combine and subtract OFF lamps
            (offData, offHeader) = imcombine_method(offLamps, outfile=None, method="median", mef=offLamps[0]._mef, log=self._log, returnHeader=True)
            #Calculate noisemap first before subtracting offData from data
            ncomb1 = float(header['NCOMBINE'])
            ncomb2 = float(offHeader['NCOMBINE'])
            #Create noisemap
            if (self._fdb.getGPUMode()):
                #Method is named for flat but reuse same method
                nm = noisemaps_mflat_dome_on_off_gpu(data, offData, ncomb1, ncomb2)
            else:
                nm = sqrt(abs(data/ncomb1)+abs(offData/ncomb2))
            #Now subtract off lamps
            data -= offData
            offData = None #free memory
            #If output written to disk, update FITS file with difference
            if (mlfilename is not None):
                mlamp = pyfits.open(mlfilename, "update")
                mlamp[onLamps[0]._mef].data = data
                mlamp.verify('silentfix')
                mlamp.flush()
                mlamp.close()
            masterLamp = fatboySpecCalib(self._pname, "master_arclamp", onLamps[0], data=data, tagname=mlname, headerExt=header, log=self._log)
            #Tag noisemap
            masterLamp.tagDataAs("noisemap", nm)
        else:
            #Use imcombine to create master arclamp file
            (data, header) = imcombine_method(lamps, outfile=mlfilename, method="median", mef=fdu._mef, returnHeader=True, log=self._log)
            masterLamp = fatboySpecCalib(self._pname, "master_arclamp", lamps[0], data=data, tagname=mlname, headerExt=header, log=self._log)
            #Create noisemap
            ncomb = float(masterLamp.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterLamp.getData(), ncomb)
            else:
                nm = sqrt(masterLamp.getData()/ncomb)
            masterLamp.tagDataAs("noisemap", nm)
        #Set properties here
        masterLamp.setType("master_arclamp", True)
        #set lamp_method property
        masterLamp.setProperty("lamp_method", lampmethod)
        updateHeaderEntry(masterLamp._header, "LAMPMTHD", lampmethod) #Use wrapper function to update header
        #set specmode property
        masterLamp.setProperty("specmode", fdu.getProperty("specmode"))
        updateHeaderEntry(masterLamp._header, "SPECMODE", fdu.getProperty("specmode")) #Use wrapper function to update header
        #set dispersion property
        masterLamp.setProperty("dispersion", fdu.getProperty("dispersion"))
        updateHeaderEntry(masterLamp._header, "DISPDIR", fdu.getProperty("dispersion")) #Use wrapper function to update header

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and not os.access(mlfilename, os.F_OK)):
            #Optionally save if write_calib_output = yes
            masterLamp.writeTo(mlfilename)
        #Write out noisemap if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes"):
            if (not os.access(outdir+"/masterArclamps", os.F_OK)):
                os.mkdir(outdir+"/masterArclamps",0o755)
            if (os.access(nmfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                masterLamp.writeTo(nmfile, tag="noisemap")

        #disable these lamps as master lamp has been created
        for lamp in lamps:
            lamp.disable()

        return masterLamp
    #end createMasterArclamp

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        #Only run createMasterArclamp on objects, not calibs
        #Also run on standard stars
        if (not fdu.isObject and not fdu.isStandard):
            return True

        print("Create Master Arclamp")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For createMasterArclamp, this dict should have one entry 'masterLamp' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'masterLamp' in calibs):
            #Failed to obtain master master arclamp frame
            #Issue error message but do not disable FDUs as they may still be able to be processed with arclamps
            print("createMasterArclampProcess::execute> Warning: Could not create Master Arclamp frame for "+fdu.getFullId())
            self._log.writeLog(__name__, "Could not create Master Arclamp frame for "+fdu.getFullId(), type=fatboyLog.WARNING)
            return False

        #Nothing else to do here as masterLamp has been added to database!
        return True
    #end execute

    #Look at XML to determine dome lamp types
    def findLampType(self, fdu, properties):
        #properties has lamp_method = lamp_on-off.  Only need to process lamps matching this lamp_method.
        lampoff = self.getOption("lamp_off_files", fdu.getTag())
        if (os.access(lampoff, os.F_OK)):
            #This is an ASCII file listing off lamps
            #Process entire file here
            lampoffList = readFileIntoList(lampoff)
            #loop over lampoffList do a split on each line
            for j in range(len(lampoffList)-1, -1, -1):
                lampoffList[j] = lampoffList[j].split()
                #remove misformatted lines
                if (len(lampoffList[j]) < 3):
                    print("createMasterArclampProcess::findLampType> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
                lampoffList[j][0] = lampoffList[j][0].lower()
                try:
                    lampoffList[j][1] = int(lampoffList[j][1])
                    lampoffList[j][2] = int(lampoffList[j][2])
                except Exception:
                    print("createMasterArclampProcess::findLampType> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
            #loop over dataset and assign property to all lamp_on-off lamps that don't already have 'lamp_type' property.
            for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_ARCLAMP, tag=fdu.getTag(), properties=properties):
                if (fdu.hasProperty("lamp_type")):
                    #this FDU already has lamp_type set
                    continue
                fdu.setProperty("lamp_type", "lamp_on") #set to on by default then loop over lampoffList for matches
                #offLine = [ 'identifier', startIdx, stopIdx ]
                for offLine in lampoffList:
                    if (fdu._id.lower().find(offLine[0]) != -1 and int(fdu._index) >= offLine[1] and int(fdu._index) <= offLine[2]):
                        #Partial match for identifier and index within range given
                        fdu.setProperty("lamp_type", "lamp_off")
        elif (fdu.hasHeaderValue(lampoff)):
            #This is a FITS keyword
            lampoffVal = self.getOption("lamp_off_header_value", fdu.getTag())
            #First check flats TAGGED for this object
            fdus = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_ARCLAMP, properties=properties)
            if (len(fdus) > 0):
                for fdu in fdus:
                    if (fdu.hasProperty("lamp_type")):
                        #this FDU already has lamp_type set
                        continue
                    if (str(fdu.getHeaderValue(lampoff)) == lampoffVal):
                        fdu.setProperty("lamp_type", "lamp_off")
                    else:
                        fdu.setProperty("lamp_type", "lamp_on") #set to on if it does not match
            else:
                #loop over dataset and assign property to all dome_on-off lamps that don't already have 'lamp_type' property.
                for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_ARCLAMP, tag=fdu.getTag(), properties=properties):
                    if (fdu.hasProperty("lamp_type")):
                        #this FDU already has lamp_type set
                        continue
                    if (str(fdu.getHeaderValue(lampoff)) == lampoffVal):
                        fdu.setProperty("lamp_type", "lamp_off")
                    else:
                        fdu.setProperty("lamp_type", "lamp_on") #set to on if it does not match
        else:
            #This is a filename fragment.  Find which lamps match
            #loop over dataset and assign property to all lamp_on-off lamps that don't already have 'lamp_type' property.
            for fdu in self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_ARCLAMP, tag=fdu.getTag(), properties=properties):
                if (not fdu.hasProperty("lamp_type")):
                    if (fdu._id.lower().find(lampoff.lower()) != -1):
                        #partial match for identifier
                        fdu.setProperty("lamp_type", "lamp_off")
                    else:
                        #no match -- this is a lamp on
                        fdu.setProperty("lamp_type", "lamp_on")
    #end findLampType

    #Look at XML to determine lamp field methods
    def findLampMethods(self, fdu):
        lampmethod = self.getOption("lamp_method", fdu.getTag())
        if (lampmethod.lower() == "lamp_on" or lampmethod.lower() == "lamp_on-off"):
            lampmethod = lampmethod.lower()
            #loop over dataset and assign property to all fdus that don't already have 'lamp_method' property.
            #some FDUs may have used xml to define lamp_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (not fdu.hasProperty("lamp_method")):
                    fdu.setProperty("lamp_method", lampmethod)
        elif (os.access(lampmethod, os.F_OK)):
            #This is an ASCII file listing filter/identifier and lamp method
            #Process entire file here
            methodList = readFileIntoList(lampmethod)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) < 2):
                    print("createMasterArclampProcess::findLampMethods> Warning: line "+str(j)+" misformatted in "+lampmethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampmethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                methodList[j][1] = methodList[j][1].lower()
                if (methodList[j][1] != "lamp_on" and methodList[j][1] != "lamp_on-off"):
                    print("createMasterArclampProcess::findLampMethods> Warning: line "+str(j)+" misformatted in "+lampmethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampmethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'lamp_method' property.
            #some FDUs may have used xml to define lamp_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (fdu.hasProperty("lamp_method")):
                    #this FDU already has lamp_method set
                    continue
                #method = [ 'Filter/identifier', 'method' ]
                for method in methodList:
                    if (method[0].lower() == fdu.filter.lower()):
                        fdu.setProperty("lamp_method", method[1])
                        #Exact match for filter
                    elif (len(method[0]) > 2 and fdu._id.lower().find(method[0].lower()) != -1):
                        #Partial match for identifier
                        fdu.setProperty("lamp_method", method[1])
        else:
            print("createMasterArclampProcess::findLampMethods> Error: invalid lamp_method: "+lampmethod)
            self._log.writeLog(__name__, " invalid lamp_method: "+lampmethod, type=fatboyLog.ERROR)
    #end findLampMethods

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        mlfilename = self.getCalib("masterLamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("createMasterArclampProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, tagname=mlfilename, log=self._log)
                return calibs
            else:
                print("createMasterArclampProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Could not find master arclamp frame "+mlfilename+"...", type=fatboyLog.WARNING)

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")

        if (not fdu.hasProperty("lamp_method")):
            #Look at XML options to find lamp method and assign it to FDUs
            self.findLampMethods(fdu)
        properties['lamp_method'] = fdu.getProperty("lamp_method")
        if (properties['lamp_method'] is None):
            print("createMasterArclampProcess::getCalibs> Error: Could not find lamp_method for "+fdu.getFullId())
            self._log.writeLog(__name__, " Could not find lamp_method for "+fdu.getFullId(), type=fatboyLog.ERROR)
            return calibs

        #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
        masterLamp = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
        if (masterLamp is not None):
            #Found master arclamp.  Return here
            calibs['masterLamp'] = masterLamp
            return calibs
        #2) Check for individual arclamp frames matching specmode/filter/grism to create master arclamp and TAGGED for this object
        lamps = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_ARCLAMP, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
        if (len(lamps) > 0):
            #Found lamps associated with this fdu.  Create master arclamp.
            print("createMasterArclampProcess::getCalibs> Creating Master Arclamp for tagged object "+fdu._id+", filter "+str(fdu.filter)+", and grism "+str(fdu.grism)+" ...")
            self._log.writeLog(__name__, " Creating Master Arclamp for tagged object "+fdu._id+", filter "+str(fdu.filter)+", and grism "+str(fdu.grism)+" ...")
            #First recursively process
            self.recursivelyExecute(lamps, prevProc)
            #convenience method
            masterLamp = self.createMasterArclamp(fdu, lamps, properties)
            if (masterLamp is None):
                return calibs
            self._fdb.appendCalib(masterLamp)
            calibs['masterLamp'] = masterLamp
            return calibs
        #3) Check for an already created master arclamp frame frame matching specmode/filter/grism
        masterLamp = self._fdb.getMasterCalib(self._pname, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (masterLamp is not None):
            #Found master arclamp frame.  Return here
            calibs['masterLamp'] = masterLamp
            return calibs
        #4) Check for individual arclamps matching specmode/filter/grism/ident to create master arclamp frame
        lamps = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_ARCLAMP, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (len(lamps) > 0):
            #Found fdus.  Create master arclamp frame.
            print("createMasterArclampProcess::getCalibs> Creating Master Arclamp frame for "+fdu._id+" ...")
            self._log.writeLog(__name__, "Creating Master Arclamp frame for "+fdu._id+" ...")
            #First recursively process (through dark subtraction presumably)
            self.recursivelyExecute(lamps, prevProc)
            #convenience method
            masterLamp = self.createMasterArclamp(fdu, lamps, properties)
            if (masterLamp is None):
                return calibs
            self._fdb.appendCalib(masterLamp)
            masterLamp = self._fdb.getMasterCalib(self._pname, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            calibs['masterLamp'] = masterLamp
            return calibs
        print("createMasterArclampProcess::getCalibs> Master Arclamp frame for filter "+str(fdu.filter)+", grism "+str(fdu.grism)+", and ident "+str(fdu._id)+" not found!")
        self._log.writeLog(__name__, "Master Arclamp frame for filter "+str(fdu.filter)+", grism "+str(fdu.grism)+", and ident "+str(fdu._id)+" not found!")
        #5) Check default_master_arclamp for matching filter/grism
        defaultMasterArclamps = []
        if (self.getOption('default_master_arclamp', fdu.getTag()) is not None):
            dlist = self.getOption('default_master_arclamp', fdu.getTag())
            if (dlist.count(',') > 0):
                #comma separated list
                defaultMasterArclamps = dlist.split(',')
                removeEmpty(defaultMasterArclamps)
                for j in range(len(defaultMasterArclamps)):
                    defaultMasterArclamps[j] = defaultMasterArclamps[j].strip()
            elif (dlist.endswith('.fit') or dlist.endswith('.fits')):
                #FITS file given
                defaultMasterArclamps.append(dlist)
            elif (dlist.endswith('.dat') or dlist.endswith('.list') or dlist.endswith('.txt')):
                #ASCII file list
                defaultMasterArclamps = readFileIntoList(dlist)
            for mlfile in defaultMasterArclamps:
                #Loop over list of default master arclamp frames
                masterLamp = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfile, tagname=mlfile, log=self._log)
                #read header and initialize
                masterLamp.readHeader()
                masterLamp.initialize()
                if (masterLamp.filter != fdu.filter):
                    #does not match filter
                    continue
                if (masterLamp.grism != fdu.grism):
                    #does not match grism
                    continue
                masterLamp.setType("master_arclamp")
                #Found matching master arclamp frame
                print("createMasterArclampProcess::getCalibs> Using default master arclamp frame "+masterLamp.getFilename())
                self._fdb.appendCalib(masterLamp)
                calibs['masterLamp'] = masterLamp
                return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_arclamp', None)
        self._options.setdefault('lamp_method', "lamp_on")
        self._optioninfo.setdefault('lamp_method', 'lamp_on | lamp_on-off')
        self._options.setdefault('lamp_off_files', 'off')
        self._optioninfo.setdefault('lamp_off_files', 'An ASCII text file listing on and off lamps\nor a filename fragment or a FITS header keyword\nfor identifying off lamps')
        self._options.setdefault('lamp_off_header_value', 'OFF')
        self._optioninfo.setdefault('lamp_off_header_value', 'If lamp_off_files is a FITS keyword, value for off lamps')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions
