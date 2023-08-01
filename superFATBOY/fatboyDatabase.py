## @package superFATBOY
#  Documentation for pipeline.
#

import superFATBOY
from .fatboyDataUnit import *
from .fatboyLibs import *
from .fatboyProcess import *
from .fatboyQuery import *
import xml.dom.minidom
from xml.dom.minidom import Node
import importlib, re, shutil, sys, traceback
from . import gpu_imcombine, imcombine, gpu_arraymedian
from .datatypeExtensions import *
from .datatypeExtensions.fatboySpectrum import fatboySpectrum
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

## Documentation for fatboyDatabase
#
#
class fatboyDatabase:
    ## class variables
    _name = "fatboyDatabase"

    _config = None
    _gpumode = True
    _processes = None
    _queries = None

    #Logs
    _log = None
    _shortlog = None
    _verbosity = fatboyLog.NORMAL
    _version = superFATBOY.__version__
    _build = superFATBOY.__build__
    _tempdir = "temp-fatboy" #Default in in CWD

    ## Constants
    MODE_IMAGE = 0
    MODE_SPEC = 1

    _imageCount = 0 #count of images in memory for memory management purposes
    totalSextractorTime = 0 #count of total time running sextractor
    totalReadHeaderTime = 0
    totalReadDataTime = 0

    ## The constructor.
    def __init__(self, config=None, modeTag=None):
        #initialize class variables that are dicts or lists here
        self._initialize_dicts_and_lists()

        if (config is None):
            # -list option only
            print("superFATBOY v"+self._version)
            print("build date: "+self._build)
            self.setDefaultParams() #set default param values
            self.setupProcessDict() #Create dict of process classes here
            self.printParams() #Print params
            self.printProcesses(modeTag=modeTag) #Print processes and options
            sys.exit(0) #exit here
        ### Normal operation
        self._t = time.time();
        self._config = config
        self.initialize()
    #end __init__

    #initialize class variables that are dicts or lists here
    def _initialize_dicts_and_lists(self):
        #initialize dicts
        self._datatypeDict = dict() #This is a dict mapping actual datatype classes to names used in XML.
                    ##This is loaded from datatypeExtensions/datatypeDict.py method getDatatypeDict().
                    ##Also any user-defined directory tagged by <param name="datatypedir" value="/path/to/MyFatboyDatatypes"/>
                    ##This dir should contain classes that extend fatboyDataUnit and a file datatypeDict.py with a method getDatatypeDict()
        self._params = dict() ##_params is dict listing all paramters.  params can be tagged or not.
                    ##if fdu has a tag, first look at _params[name::tag] then look for _params[name].
                    ##If not, just look for _params[name].  This logic is in getParam().
        self._processDict = dict() #This is a dict mapping actual process classes to names used in XML.
                    ##This is loaded from fatboyProcesses/processDict.py method getProcessDict().
                    ##Also any user-defined directory tagged by <param name="processdir" value="/path/to/MyFatboyProcesses"/>
                    ##This dir should contain classes that extend fatboyProcess and a file processDict.py with a method getProcessDict()
        self._processHistory = dict()

        #initialize lists
        self._db = [] #database of fdus
        self._calibs = [] #database of master calibration frames (fatboyCalibs)
        self._datatypedirs = [] #List of 3rd party datatype dirs
        self._processdirs = [] #List of 3rd party process directories
    #end _initialize_dicts_and_lists

    ## Add a new slitmask with new shape after e.g. rectification or resampling
    def addNewSlitmask(self, oldSlitmask, newData, pname):
        slitmask = fatboySpecCalib(pname, "slitmask", oldSlitmask, data=newData, tagname=oldSlitmask._id, log=self._log)
        slitmask.setProperty("specmode", oldSlitmask.getProperty("specmode"))
        slitmask.setProperty("dispersion", oldSlitmask.getProperty("dispersion"))
        self.appendCalib(slitmask)
        return slitmask
    #end addNewSlitmask

    ## Append a master calibration frame
    def appendCalib(self, calib):
        calib.setDatabaseCallback(self) #set pointer back to this fdb for memory management purposes
        calib.setGPUMode(self._gpumode)
        self._calibs.append(calib)
    #end appendCalib

    ## Apply a quick start dict
    def applyQuickStart(self, qsDict):
        for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
            if (fdu.getFullId() in qsDict):
                #If full filename matches, set shape and median
                (qsshape, qsmed) = qsDict[fdu.getFullId()]
                fdu.setShape(qsshape)
                fdu.setMedian(qsmed)
    #end applyQuickStart

    #automatically detect which params are ints and which are floats.  Leave all others as strings, including space separated lists
    def autoDetectNumericalParams(self):
        for key in self._params:
            value = self._params[key]
            if (value is None):
                #Skip None type
                continue
            if (isinstance(value, list)):
                #Only process strings
                continue
            if (type(value) == type(len)):
                #This is a function
                continue
            if (isInt(value)):
                self._params[key] = int(value)
            elif (isFloat(value)):
                self._params[key] = float(value)
    #end  autoDetectNumericalParams

    ## Check files to make sure they exist
    def checkFiles(self):
        for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
            fdu.checkFile() #will disable FDU if filename doesn't exist
    #end checkFiles

    ## Check _db for master calibration frames and put these in _calibs instead
    def checkForMasterCalibs(self):
        #Must do two loops -- first loop over FDUs and append to _calibs
        for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
            if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_MASTER_CALIB):
                self.appendCalib(fdu)
            elif (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK):
                self.appendCalib(fdu)
        #Second loop over new _calibs list and remove them from _db
        for calib in self.getMasterCalibs():
            if (calib in self._db):
                self._db.remove(calib)
    #end checkForMasterCalibs

    ## Check for dark or flat file lists, etc. to override types
    def checkForObstypeOverrides(self):
        #First get any parameter ending in _file_list
        for key in self._params:
            if (key.endswith("_file_list")):
                #Get value - this will work for tagged and untagged params
                value = self.getParam(key)
                if (value is None):
                    continue
                #find type e.g., dark_file_list => overrideType = dark
                overrideType = key[:key.find('_file_list')]
                #value could be a .fits file, a file list, or a comma separated list
                overrideList = None
                try:
                    if (os.access(value, os.F_OK) and value.lower().count(".fit") > 0):
                        #this is one FITS file
                        overrideList = [value]
                    elif (os.access(value, os.F_OK)):
                        #this should be an ASCII file list, one filename per line
                        overrideList = readFileIntoList(value)
                    elif (value.count(',') > 0):
                        #this is a comma separated list
                        overrideList = value.strip().split(',')
                        removeEmpty(overrideList)
                        for j in range(len(overrideList)):
                            overrideList[j] = overrideList[j].strip()
                except Exception as ex:
                    print("fatboyDatabase::checkForObstypeOverrides> Error in reading override list "+key+" = "+value+": "+str(ex))
                    self._log.writeLog(__name__, "Error in reading override list "+key+" = "+value+": "+str(ex), type=fatboyLog.ERROR)
                    continue
                if (overrideList is None):
                    print("fatboyDatabase::checkForObstypeOverrides> Warning: missing or misformatted override list "+key+" = "+value)
                    self._log.writeLog(__name__, "Missing or misformatted override list "+key+" = "+value, type=fatboyLog.WARNING)
                    continue
                for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
                    for fname in overrideList:
                        #Loop over overrideList and check fdu _filename and _identFull
                        if (fname == fdu.getFilename() or fname == fdu.getFullId()):
                            #Match found, set type
                            fdu.setType(overrideType)
    #end checkForObstypeOverrides

    ## Check memory count and release frames from memory if necessary.  Called from getData and tagDataAs when new storage is needed.
    def checkMemoryManagement(self, currentFDU):
        self._imageCount += 1
        limit = self.getParam("memory_image_limit")
        if (limit is None):
            return
        limit = int(limit)
        if (self._imageCount >= limit):
            #need to free memory!  Free up 10 images at a time.  Save data to disk in temp file and release memory
            freed = 0
            for fdu in self._db:
                if (fdu._id == currentFDU._id):
                    continue #skip current ident
                if (fdu._data is not None):
                    outfile = self._tempdir+"/current_"+fdu.getFullId()
                    fdu.writeToAndForget(outfile)
                    fdu.writeAndForgetTaggedData()
                    freed += 1
                    if (freed >= 10):
                        break
                        #10 images freed, break
            if (freed < 10):
                #Now check current ident but not current image if still need to free more memory
                for fdu in self._db:
                    if (fdu.getFullId() == currentFDU.getFullId()):
                        continue #skip current ident
                    if (fdu._data is not None):
                        outfile = self._tempdir+"/current_"+fdu.getFullId()
                        fdu.writeToAndForget(outfile)
                        fdu.writeAndForgetTaggedData()
                        freed += 1
                        if (freed >= 10):
                            break
                            #10 images freed, break
            print("fatboyDatabase::checkMemoryManagement> Backed up "+str(freed)+" images to disk temporarily to free up memory space.")
            self._log.writeLog(__name__, "Backed up "+str(freed)+" images to disk temporarily to free up memory space.")
    #end checkMemoryManagement

    #clean up - remove temp-fatboy dir
    def cleanUp(self):
        #clean up any this run
        if (os.access(self._tempdir, os.F_OK)):
            shutil.rmtree(self._tempdir)
    #end cleanUp

    ## decrement memory count.  called from disable
    def decrementMemoryCount(self):
        self._imageCount -= 1
    #end decrementMemoryCount

    ## Execute everything
    def execute(self):
        self.executeQueries() #add results of XML queries to database
        self.preprocessAll() #read and preprocess data
        self.executeProcesses() #execute processes
        self.cleanUp() #Clean up - remove temp-fatboy dir
    #end execute

    ## Add results of XML queries to database
    def executeQueries(self):
        tt = time.time();
        for query in self._queries:
            #Execute SQL queries and get list of fatboyImage objects
            self._db.extend(query.executeQuery())
        print("fatboyDatabase::executeQueries> Found "+str(len(self._db))+" images.")
        self._log.writeLog(__name__, "Found "+str(len(self._db))+" images.", verbosity=fatboyLog.BRIEF)
        self._shortlog.writeLog(__name__, "Execute "+str(len(self._queries))+" queries: "+str(time.time()-tt)+"; Total: "+str(time.time()-self._t), printCaller=False, tabLevel=1)
        for fdu in self._db:
            fdu.setDatabaseCallback(self) #set pointer back to this fdb for memory management purposes
        if (self._verbosity == fatboyLog.VERBOSE):
            for fdu in self._db:
                print(fdu._id, fdu._index, fdu.getFullId())
                self._log.writeLog(__name__, fdu._id+"\t"+fdu._index+"\t"+fdu.getFullId(), printCaller=False, tabLevel=1)
    #end executeQueries

    ## Execute processes
    def executeProcesses(self):
        isCalibs = False
        fdus = self.getObjects()
        if (self.getParam('calibs_only').lower() == "yes"):
            #in the case of calibs_only = yes, get all FDUs, not just objects!
            fdus = self.getFDUs()
            isCalibs = True
        nobjects = len(fdus)
        ni = 0
        #First loop over images
        for image in fdus:
            ni += 1
            tt = time.time();
            writeShortLogTimer = False
            print("Processing image "+image.getShortName()+", id="+image.getFullId()+" ("+str(ni)+" of "+str(nobjects)+")")
            self._log.writeLog(__name__, "Processing image "+image.getShortName()+", id="+image.getFullId()+" ("+str(ni)+" of "+str(nobjects)+")")
            self._shortlog.writeLog(__name__, "Processing image "+image.getShortName()+", id="+image.getFullId()+" ("+str(ni)+" of "+str(nobjects)+")")
            #Setup list of previously done image processes to recursively process calibs
            prevProcesses = []
            #Second loop over image-level processes
            for process in self._processes:
                if (not image.inUse):
                    #if image has been disabled, continue
                    #this may happen due to rejection along the way, or pairing or stacking with other fdus
                    continue
                if (image.getObsType(True) != fatboyDataUnit.FDU_TYPE_OBJECT and not isCalibs):
                    #also continue if not FDU_TYPE_OBJECT.  This could occur with offsource skies
                    #exception if isCalibs is True
                    continue
                histName = process._pname
                if (process.hasOption("pass_number")):
                    histName += "::"+process.getOption("pass_number")
                if (not process.checkTag(image)):
                    #this process is only supposed to be performed on images matching a certain tag
                    #this image does not match, continue to next process
                    print("\tProcess "+str(process._pname)+" will only be run on images with tag="+str(process.getTag())+", skipping fatboyDataUnit "+image.getFullId()+" with tag "+str(image.getTag())+".")
                    self._log.writeLog(__name__, "Process "+str(process._pname)+" will only be run on images with tag="+str(process.getTag())+", skipping fatboyDataUnit "+image.getFullId()+" with tag "+str(image.getTag())+".", printCaller=False, tabLevel=1, verbosity=fatboyLog.VERBOSE)
                    #add process prevProcesses list even though it is not executed here
                    #it needs to be in prevProcesses for recursion on other images
                    prevProcesses.append(process)
                    continue
                #if (image.hasProcessInHistory(process._pname)):
                if (image.hasProcessInHistory(histName)):
                    #This process has already been applied to this FDU
                    print("\tAlready executed process "+str(process._pname)+" on fatboyDataUnit "+image.getFullId())
                    self._log.writeLog(__name__, "Already executed process "+str(process._pname)+" on fatboyDataUnit "+image.getFullId(), printCaller=False, tabLevel=1, verbosity=fatboyLog.VERBOSE)
                    continue
                writeShortLogTimer = True
                #Execute the process on each image
                print("\tPerforming process "+str(process._pname))
                self._log.writeLog(__name__, "Performing process "+str(process._pname), printCaller=False, tabLevel=1)
                self._shortlog.writeLog(__name__, "Performing process "+str(process._pname), printCaller=False, tabLevel=1)
                #set default options
                process.setDefaultOptions()
                #execute process
                try:
                    #Check if only calibs should be created
                    if (process.getOption("create_calib_only", image.getTag()).lower() == "yes"):
                        ##Just call getCalibs.  This is a success if there is a calib created and returned in the dict object
                        calibs = process.getCalibs(image, prevProc=prevProcesses)
                        if (len(calibs) > 0):
                            success = True
                        else:
                            success = False
                    else:
                        ##NORMAL mode - execute process
                        success = process.execute(image, prevProc=prevProcesses)
                except Exception as ex:
                    print("ERROR: process "+str(process._pname)+" FAILED with EXCEPTION: "+str(ex))
                    self._log.writeLog(__name__, "process "+str(process._pname)+" FAILED with EXCEPTION: "+str(ex), type=fatboyLog.ERROR)
                    self._shortlog.writeLog(__name__, "process "+str(process._pname)+" FAILED with EXCEPTION: "+str(ex), type=fatboyLog.ERROR)
                    print("This traceback should be sent to Craig for debugging!")
                    traceback.print_exc()
                    input("Press ENTER to continue")
                    success = False
                if (not success):
                    #execute failed!  Presumably fdu has been disabled. continue here
                    continue
                #add process name to _processHistory for this FDU
                #image.addProcessToHistory(process._pname)
                if (process.getOption("create_calib_only", image.getTag()).lower() != "yes"):
                    image.addProcessToHistory(histName)
                #if write_output is yes, call writeOutput method
                if (process.getOption("write_output", image.getTag()).lower() == "yes" and process.getOption("create_calib_only", image.getTag()).lower() != "yes"):
                    #update list of previous filenames to potentially clean up.  Exclude original file from this list
                    image.updateFilenames()
                    process.writeOutput(image)
                #add process to list
                prevProcesses.append(process)
            if (writeShortLogTimer):
                self._shortlog.writeLog(__name__, "Time: "+str(time.time()-tt)+"; Total: "+str(time.time()-self._t), printCaller=False, tabLevel=1)
            #Do not disable frames here - in case alignStack is not done and sky subtract is last step. 12/13/17
            #image.disable() #Free memory
        print("Total sep/sextractor time: "+str(self.totalSextractorTime))
        self._shortlog.writeLog(__name__, "Total sep/sextractor time: "+str(self.totalSextractorTime), printCaller=False)
        print("Total read header time: "+str(self.totalReadHeaderTime))
        self._shortlog.writeLog(__name__, "Total read header time: "+str(self.totalReadHeaderTime), printCaller=False)
        print("Total read data time: "+str(self.totalReadDataTime))
        self._shortlog.writeLog(__name__, "Total read data time: "+str(self.totalReadDataTime), printCaller=False)
        print("Total time: "+str(time.time()-self._t))
        self._shortlog.writeLog(__name__, "Total time: "+str(time.time()-self._t), printCaller=False)
    #end executeProcesses

    ## Return list of calibration FDUs in use and matching specified criteria
    def getCalibs(self, ident=None, obstype=None, filter=None, section=None, exptime=None, nreads=None, tag=None, shape=None, properties=None, headerVals=None, inUse=True):
        if (tag is not None and isinstance(tag, list)):
            #If a list is given, take only dataset leve tag, not subtag
            #This is to ensure that different datasets are not combined;
            #subtags within a dataset can and should be
            tag = tag[0]
        calibs = []
        for fdu in self._db:
            if (fdu.inUse != inUse):
                #disabled -- skip (or enabled in the case of inUse = false)
                continue
            if (fdu._objectTags != []):
                #getTaggedCalibs should select any calibs tagged for this object first before getCalibs is called.
                continue
            if (ident is not None and ident != fdu._id):
                #doesn't match specified identifier -- skip
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (fdu.section is not None and fdu.section != -1):
                    spos = -1-len(str(fdu.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (ident[:ispos] != fdu._id[:spos] or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != fdu.getObsType() and obstype != fdu.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != fdu.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != fdu.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != fdu.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (nreads is not None and nreads != fdu.nreads):
                #doesn't match specified exptime -- skip
                continue
            if (tag is not None and tag != fdu.getTag(mode="tag_only")):
                #doesn't match this tag -- skip
                #tag here should be top level dataset tag only
                continue
            if (shape is not None and shape != fdu.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != fdu.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != fdu.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            calibs.append(fdu)
        return calibs
    #end getCalibs

    ## Get an instance of a datatype by name
    def getDatatypeByName(self, dname):
        if (dname in self._datatypeDict):
            return self._datatypeDict[dname]
        return None
    #end getDatatypeByName

    ## Return list of FDUs in use and matching specified criteria
    def getFDUs(self, ident=None, obstype=None, filter=None, section=None, exptime=None, tag=None, shape=None, properties=None, headerVals=None, inUse=True):
        if (tag is not None and isinstance(tag, list)):
            #If a list is given, take only dataset leve tag, not subtag
            #This is to ensure that different datasets are not combined;
            #subtags within a dataset can and should be
            tag = tag[0]
        fdus = []
        for fdu in self._db:
            if (fdu.inUse != inUse):
                #disabled -- skip (or enabled in the case of inUse = false)
                continue
            if (ident is not None and ident != fdu._id):
                #doesn't match specified identifier -- skip
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (fdu.section is not None and fdu.section != -1):
                    spos = -1-len(str(fdu.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (ident[:ispos] != fdu._id[:spos] or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != fdu.getObsType() and obstype != fdu.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != fdu.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != fdu.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != fdu.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (tag is not None and tag != fdu.getTag(mode="tag_only")):
                #doesn't match this tag -- skip
                #tag here should be top level dataset tag only
                continue
            if (shape is not None and shape != fdu.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != fdu.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != fdu.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            fdus.append(fdu)
        return fdus
    #end getFDUs

    ## returns true if GPU mode is enabled, false otherwise
    def getGPUMode(self):
        return self._gpumode
    #end getGPUMode

    ## Return an individual FDU referenced by its full ID
    def getIndividualFDU(self, identFull):
        for fdu in self._db:
            if (fdu.getFullId() == identFull):
                #found match
                return fdu
        #matching FDU not found.  Return None.
        return None
    #end getIndividualFDU

    ## Return a list of just objects
    def getObjects(self):
        fdus = []
        #Loop over images
        for image in self._db:
            if (image.isObject):
                fdus.append(image)
        return fdus
    #end getObjects

    ## Return the log.  Used by fatboyProcess
    def getLog(self):
        return self._log
    #end getLog

    ## Return single master calibration FDU matching specified criteria.  Only one should match.
    def getMasterCalib(self, pname=None, ident=None, obstype=None, filter=None, section=None, exptime=None, nreads=None, tag=None, shape=None, properties=None, headerVals=None):
        if (tag is not None and isinstance(tag, list)):
            #If a list is given, take only dataset leve tag, not subtag
            #This is to ensure that different datasets are not combined;
            #subtags within a dataset can and should be
            tag = tag[0]
        #pname = process name that created it, e.g. darkSubtract for a masterDark
        for calib in self._calibs:
            if (not calib.inUse):
                #disabled -- skip
                continue
            if (calib._objectTags != []):
                #getTaggedMasterCalib should select any calibs tagged for this object first before getMasterCalib is called.
                continue
            #Use this option only if instance of fatboyCalib.  If a master calib frame was specified
            #in xml, it will be of type fatboyImage and won't have a process name that created it to match.
            if (calib._name == "fatboyCalib" and pname is not None and pname != calib.getCalibProcessName()):
                #doesn't match this process
                continue
            if (ident is not None and ident != calib._id):
                #doesn't match specified identifier -- skip
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (calib.section is not None and calib.section != -1):
                    spos = -1-len(str(calib.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (ident[:ispos] != calib._id[:spos] or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != calib.getObsType() and obstype != calib.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != calib.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != calib.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != calib.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (nreads is not None and nreads != calib.nreads):
                #doesn't match specified exptime -- skip
                continue
            if (tag is not None and tag != calib.getTag(mode="tag_only")):
                #doesn't match this tag -- skip
                #tag here should be top level dataset tag only
                continue
            if (shape is not None and shape != calib.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != calib.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != calib.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            #match found!
            return calib
        #No matches found
        return None
    #end getMasterCalib

    ## Return list of master calibration FDUs matching specified criteria.
    def getMasterCalibs(self, ident=None, obstype=None, filter=None, section=None, exptime=None, nreads=None, tag=None, shape=None, properties=None, headerVals=None):
        if (tag is not None and isinstance(tag, list)):
            #If a list is given, take only dataset leve tag, not subtag
            #This is to ensure that different datasets are not combined;
            #subtags within a dataset can and should be
            tag = tag[0]
        calibs = []
        #pname = process name that created it, e.g. darkSubtract for a masterDark
        for calib in self._calibs:
            if (not calib.inUse):
                #disabled -- skip
                continue
            if (calib._objectTags != []):
                #getTaggedMasterCalib should select any calibs tagged for this object first before getMasterCalib is called.
                continue
            if (ident is not None and ident != calib._id):
                #doesn't match specified identifier -- skip
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (calib.section is not None and calib.section != -1):
                    spos = -1-len(str(calib.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (ident[:ispos] != calib._id[:spos] or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != calib.getObsType() and obstype != calib.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != calib.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != calib.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != calib.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (nreads is not None and nreads != calib.nreads):
                #doesn't match specified exptime -- skip
                continue
            if (tag is not None and tag != calib.getTag(mode="tag_only")):
                #doesn't match this tag -- skip
                #tag here should be top level dataset tag only
                continue
            if (shape is not None and shape != calib.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != calib.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != calib.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            #match found!
            calibs.append(calib)
        #No matches found
        return calibs
    #end getMasterCalibs

    ## Get a parameter given a name and optionally a tag
    def getParam(self, pname, ptag=None):
        if (ptag is not None):
            if (isinstance(ptag, list)):
                #Has both a tag and a subtag.  Look for subtag first then tag.
                for j in range(len(ptag)-1, -1, -1):
                    if (ptag[j] is not None and (pname+'::'+ptag[j]) in self._params):
                        return self._params[pname+'::'+ptag[j]]
            else:
                #Is a string
                if ((pname+'::'+ptag) in self._params):
                    return self._params[pname+'::'+ptag]
        #Couldn't find tag-specific param
        if (pname in self._params):
            return self._params[pname]
        return None
    #end getParam

    ## Get an instance of a process by name
    def getProcessByName(self, pname):
        if (pname in self._processDict):
            return self._processDict[pname]
        return None
    #end getProcessByName

    ## Return the short log.  Used by fatboyProcess
    def getShortLog(self):
        return self._shortlog
    #end getShortLog

    ## Return list of FDUs in use and matching specified criteria and sorted by index number -- for use in finding skies and newfirm latent masking
    def getSortedFDUs(self, ident=None, obstype=None, filter=None, section=None, exptime=None, tag=None, shape=None, properties=None, headerVals=None, sortby='full', inUse=True):
        if (tag is not None and isinstance(tag, list)):
            #If a list is given, take only dataset leve tag, not subtag
            #This is to ensure that different datasets are not combined;
            #subtags within a dataset can and should be
            tag = tag[0]
        #sortby = full | index | FITS header keyword
        fdus = []
        indices = []
        for fdu in self._db:
            if (inUse is not None):
                #inUse = None means pick either inUse or not
                if (fdu.inUse != inUse):
                    #disabled -- skip (or enabled in the case of inUse = false)
                    continue
            if (ident is not None and ident != fdu._id):
                #doesn't match specified identifier -- skip
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (fdu.section is not None and fdu.section != -1):
                    spos = -1-len(str(fdu.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (ident[:ispos] != fdu._id[:spos] or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != fdu.getObsType() and obstype != fdu.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != fdu.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != fdu.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != fdu.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (tag is not None and tag != fdu.getTag(mode="tag_only")):
                #doesn't match this tag -- skip
                #tag here should be top level dataset tag only
                continue
            if (shape is not None and shape != fdu.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != fdu.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != fdu.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            if (sortby == 'index'):
                indices.append(int(fdu._index))
            elif (sortby == 'full'):
                indices.append(fdu.getFullId())
            elif (fdu.hasHeaderValue(sortby)):
                indices.append(fdu.getHeaderValue(sortby))
            else:
                print("fatboyDatabase::getSortedFDUs> Warning: "+fdu.getFullId()+" has no header keyword "+sortby+". It will not be used for onsource sky!")
                self._log.writeLog(__name__, fdu.getFullId()+" has no header keyword "+sortby+". It will not be used for onsource sky!", type=fatboyLog.WARNING)
                continue
            fdus.append(fdu)
        sortedi = array(indices).argsort()
        fdus = array(fdus)[sortedi]
        return fdus.tolist()
    #end getSortedFDUs

    ## Return list of calibration FDUs tagged to match certain objects and matching specified criteria
    def getTaggedCalibs(self, ident, obstype=None, filter=None, section=None, exptime=None, nreads=None, shape=None, properties=None, headerVals=None, inUse=True):
        calibs = []
        for fdu in self._db:
            if (fdu.inUse != inUse):
                #disabled -- skip (or enabled in the case of inUse = false)
                continue
            if (ident is None or fdu._objectTags == []):
                #ident not given or no object tags
                continue
            if (not ident in fdu._objectTags):
                #this ident not in object tags
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (fdu.section is not None and fdu.section != -1):
                    spos = -1-len(str(fdu.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (not ident[:ispos] in fdu._objectTags or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != fdu.getObsType() and obstype != fdu.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != fdu.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != fdu.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != fdu.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (nreads is not None and nreads != fdu.nreads):
                #doesn't match specified exptime -- skip
                continue
            if (shape is not None and shape != fdu.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != fdu.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != fdu.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            calibs.append(fdu)
        return calibs
    #end getTaggedCalibs

    ## Return single master calibration FDU tagged to match certain objects and matching specified criteria.  Only one should match.
    def getTaggedMasterCalib(self, pname=None, ident=None, obstype=None, filter=None, section=None, exptime=None, nreads=None, shape=None, properties=None, headerVals=None):
        #pname = process name that created it, e.g. darkSubtract for a masterDark
        for calib in self._calibs:
            if (not calib.inUse):
                #disabled -- skip
                continue
            #Use this option only if instance of fatboyCalib.  If a master calib frame was specified
            #in xml, it will be of type fatboyImage and won't have a process name that created it to match.
            if (calib._name == "fatboyCalib" and pname is not None and pname != calib.getCalibProcessName()):
                #doesn't match this process
                continue
            if (ident is None or calib._objectTags == []):
                #ident not given or no object tags
                continue
            if (not ident in calib._objectTags):
                #this ident not in object tags
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (calib.section is not None and calib.section != -1):
                    spos = -1-len(str(calib.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                isMatch = False
                for ot in calib._objectTags:
                    if (ident[:ispos] == ot[:spos] and ident[ispos] == 'S' and ot[spos] == 'S'):
                        isMatch = True
                    elif (ident[:ispos] == ot and ident[ispos] == 'S'):
                        isMatch = True
                if (not isMatch):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != calib.getObsType() and obstype != calib.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filter is not None and filter != calib.filter):
                #doesn't match specified filter -- skip
                continue
            if (section is not None and section != -1 and section != calib.section):
                #doesn't match specified section -- skip
                continue
            if (exptime is not None and exptime != calib.exptime):
                #doesn't match specified exptime -- skip
                continue
            if (nreads is not None and nreads != calib.nreads):
                #doesn't match specified exptime -- skip
                continue
            if (shape is not None and shape != calib.getShape()):
                #data doesn't match this shape -- skip
                continue
            if (properties is not None):
                #Check properties dict vs that in fdu.
                keep = True
                for key in properties:
                    if (properties[key] != calib.getProperty(key)):
                        #getProperty returns None if not found so no need to call hasProperty first
                        #doesn't match specified property -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a property didn't match -- skip
                    continue
            if (headerVals is not None):
                #Check headerVals dict vs each fdu's header
                keep = True
                for key in headerVals:
                    if (headerVals[key] != calib.getHeaderValue(key)):
                        #getHeaderValue returns None if not found so no need to call hasHeaderValue first
                        #if key ends in _keyword, it will return header[keywords[key]]
                        #doesn't match specified header value -- set keep = False and break out of loop
                        keep = False
                        break
                if (not keep):
                    #If keep is false, a header value didn't match -- skip
                    continue
            #match found!
            return calib
        #No matches found
        return None
    #end getTaggedMasterCalib

    ## Return True if a master calibration frame matching this pname, obstype, and filename exists
    def hasMasterCalib(self, pname=None, ident=None, obstype=None, filename=None):
        #pname = process name that created it, e.g. darkSubtract for a masterDark
        for calib in self._calibs:
            if (not calib.inUse):
                #disabled -- skip
                continue
            #Use this option only if instance of fatboyCalib.  If a master calib frame was specified
            #in xml, it will be of type fatboyImage and won't have a process name that created it to match.
            if (calib._name == "fatboyCalib" and pname is not None and pname != calib.getCalibProcessName()):
                #doesn't match this process
                continue
            if (ident is not None and ident != calib._id):
                #doesn't match specified identifier -- skip
                spos = -2 #position of S in identifier for multiple sections
                ispos = -2 #Also need position of S in ident
                if (calib.section is not None and calib.section != -1):
                    spos = -1-len(str(calib.section)) #Allow for multiple digit sections
                if (section is not None and section != -1):
                    ispos = -1-len(str(section)) #Also need position of S in ident
                if (ident[:ispos] != calib._id[:spos] or ident[ispos] != 'S'):
                    #In case of multiple sections.  Section will be checked below.
                    continue
            if (obstype is not None and obstype != calib.getObsType() and obstype != calib.getObsType(True)):
                #doesn't match specified obs type by either string or enum
                continue
            if (filename is not None and filename != calib.getFilename()):
                #doesn't match specified filename
                continue
            #The calib exists
            return True
        #We have searched through entire calib list and found no matches
        return False
    #end hasMasterCalib

    ## Initialization method
    def initialize(self):
        if (self.getParam('tempdir') is not None):
            self._tempdir = self.getParam('tempdir')
        #clean up any previous runs
        if (os.access(self._tempdir, os.F_OK)):
            shutil.rmtree(self._tempdir)
        #remake temp-fatboy
        if (not os.access(self._tempdir, os.F_OK)):
            os.makedirs(self._tempdir,0o755)
        self.parseXML()
        self.setDefaultParams() #set default param values
        self.autoDetectNumericalParams() #detect ints and floats
        #set nx parameter in imcombine and gpu_imcombine based on memory_image_limit
        nx = 64
        if (self.getParam('memory_image_limit') is not None):
            while(self.getParam('memory_image_limit') < nx*2):
                nx /= 2
        gpu_imcombine.nx = nx
        imcombine.nx = nx
        #set defaultKernel in gpu_arraymedian if a median_kernel is given
        if (self.getParam('median_kernel') is not None):
            if (self.getParam('median_kernel') == "fatboycudalib.gpumedian"):
                gpu_arraymedian.defaultKernel = fatboycudalib.gpumedian
            elif (self.getParam('median_kernel') == "cp_select.cpmedian"):
                gpu_arraymedian.defaultKernel = cp_select.cpmedian
            elif (self.getParam('median_kernel') == "fatboyclib.median"):
                gpu_arraymedian.defaultKernel = fatboyclib.median
        #Update median_kernel now to be whatever defaultKernel is
        self._params['median_kernel'] = gpu_arraymedian.defaultKernel
    #end initialize

    ## Initialize all FDUs
    def initializeAll(self):
        print("fatboyDatabase::initializeAll> reading header information and initializing...")
        self._log.writeLog(__name__, "reading header information and initializing...")
        for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
            fdu.readHeader() #Read header of each FDU
            fdu.initialize() #Initialize each FDU
            fdu.reformatData() #reformat F2 style (1,2048,2048) data as (2048,2048)
    #end initializeAll

    ## Set up logging
    def initializeLog(self):
        self._verbosity = fatboyLog.NORMAL
        if ('verbosity' in self._params):
            if (self._params['verbosity'].lower() == "verbose"):
                self._verbosity = fatboyLog.VERBOSE
            elif (self._params['verbosity'].lower() == "brief"):
                self._verbosity = fatboyLog.BRIEF
        logdir = 'flogs'
        if ('logdir' in self._params):
            logdir = self._params['logdir']
        if (not os.access(logdir, os.F_OK)):
            os.makedirs(logdir)
        if ('logfile' in self._params):
            self._log = fatboyLog(self._params['logfile'], logdir=logdir, xmlName=self._config, verbosity=self._verbosity)
        else:
            self._log = fatboyLog(filename=None, logdir=logdir, xmlName=self._config, verbosity = self._verbosity)
        if ('shortlog' in self._params):
            self._shortlog = fatboyLog(self._params['shortlog'], logdir=logdir, xmlName=self._config, shortLog=True)
        else:
            self._shortlog = fatboyLog(filename=None, logdir=logdir, xmlName=self._config, shortLog=True)
        print("superFATBOY v"+self._version)
        print("build date: "+self._build)
        self._log.writeLog(__name__, "superFATBOY v"+self._version, printCaller=False)
        self._log.writeLog(__name__, "build date: "+self._build, printCaller=False)
        if (hasCuda and not superFATBOY.threaded()):
            print("Using CUDA_DEVICE "+pycuda.autoinit.device.name())
            self._log.writeLog(__name__, "Using CUDA_DEVICE "+pycuda.autoinit.device.name(), printCaller=False)
    #end initializeLog

    ## XML Parser
    def parseXML(self):
        #doc = xml config file
        try:
            doc = xml.dom.minidom.parse(self._config)
        except Exception as ex:
            print("Error parsing XML configuration file: "+str(ex))
            sys.exit(-1)

        #Parse params first so we can set the log file for instance!
        self._params = dict()
        #get all parameters nodes (should usually be just 1)
        paramNodes = doc.getElementsByTagName('parameters')
        #loop over query nodes
        for p in paramNodes:
            if (not p.hasChildNodes()):
                continue
            for param in p.childNodes:
                if (param.nodeType == Node.ELEMENT_NODE and param.nodeName == 'param'):
                    if (param.hasAttribute("name") and param.hasAttribute("value")):
                        pname = str(param.getAttribute("name"))
                        pval = str(param.getAttribute("value"))
                        if (pname == "datatypedir"):
                            #Special parameter adds a directory of user-defined datatypes
                            #More than one directory can be added
                            if (os.access(pval, os.F_OK)):
                                self._datatypedirs.append(pval)
                            else:
                                print("fatboyDatabase::parseXML> Warning: Could not find datatypedir "+pval)
                            #continue to next param
                            continue
                        elif (pname == "processdir"):
                            #Special parameter adds a directory of user-defined processes
                            #More than one directory can be added
                            if (os.access(pval, os.F_OK)):
                                self._processdirs.append(pval)
                            else:
                                print("fatboyDatabase::parseXML> Warning: Could not find processdir "+pval)
                            #continue to next param
                            continue
                        elif (pname == "outputdir"):
                            #Make all outputdirs here
                            if (not os.access(str(pval), os.F_OK)):
                                try:
                                    os.makedirs(str(pval))
                                except Exception as ex:
                                    print("fatboyDatabase::parseXML> Error: "+str(ex))
                                    sys.exit(-1)
                        if (param.hasAttribute("tag")):
                            ptag = param.getAttribute("tag")
                            #Add tagged parameter to dict with format _params['name::tag'] = value
                            self._params[pname+'::'+ptag] = pval
                        else:
                            #Add global parameter to dict with format _params['name'] = value
                            self._params[pname] = pval

        #Next set up logging!
        self.initializeLog()
        #Create outputdir if it doesn't exist
        if ('outputdir' in self._params):
            if (not os.access(self._params['outputdir'], os.F_OK)):
                try:
                    os.makedirs(self._params['outputdir'])
                except Exception as ex:
                    print("fatboyDatabase::parseXML> Error: "+str(ex))
                    self._log.writeLog(__name__, str(ex), type=fatboyLog.ERROR)
        else:
            #Use cwd
            self._params['outputdir'] = os.getcwd()

        self.setupDatatypeDict() #Create dict of datatype classes here
        tt = time.time();
        self._queries = []
        #get all queries nodes (should usually be just 1)
        queryNodes = doc.getElementsByTagName('queries')
        #loop over query nodes
        for q in queryNodes:
            if (not q.hasChildNodes()):
                continue
            #queryNodes = q.childNodes
            for query in q.childNodes:
                if (query.nodeType == Node.ELEMENT_NODE):
                    result = fatboyQuery(query, log=self._log, datatypeDict=self._datatypeDict)
                    if (result is not None):
                        self._queries.append(result)
        self._shortlog.writeLog(__name__, "XML Queries: "+str(time.time()-tt)+"; Total: "+str(time.time()-self._t), printCaller=False, tabLevel=1)

        self.setupProcessDict() #Create dict of process classes here
        tt = time.time();
        self._processes = []
        #get all processes nodes (should usually be just 1)
        processNodes = doc.getElementsByTagName('processes')
        #loop over process nodes
        for p in processNodes:
            if (not p.hasChildNodes()):
                continue
            for process in p.childNodes:
                if (process.nodeType == Node.ELEMENT_NODE):
                    #Look for default params for this process
                    defParams = None
                    if (process.hasAttribute("name")):
                        pname = process.getAttribute("name")
                        if (pname in self._processDict):
                            result = self._processDict[pname]
                            #Create instance of this process
                            currProcess = result(process, self)
                            #Only append to _processes list here.  Update processDict below.
                            #self._processDict[pname] = currProcess
                            #self._processes.append(result(process, self))
                            self._processes.append(currProcess)
                            print("fatboyDatabase::parseXML> Found process "+pname)
                            self._log.writeLog(__name__, "Found process "+pname)
                        else:
                            print("fatboyDatabase::parseXML> Error: Could not find process "+pname)
                            self._log.writeLog(__name__, "Could not find process "+pname, type=fatboyLog.ERROR)
        #Now replace noninstantiated version in _processDict.  Use pass_number so getProcessByName
        #can grab distinct instances of processes
        for process in self._processes:
            histName = process._pname
            if (process.hasOption("pass_number")):
                histName += "::"+process.getOption("pass_number")
            self._processDict[histName] = process

        self._shortlog.writeLog(__name__, "XML Processes: "+str(time.time()-tt)+"; Total: "+str(time.time()-self._t), printCaller=False, tabLevel=1)
    #end parseXML

    ## Perform all rejection criteria given before processing starts
    def performRejectionCriteria(self):
        lastIdent = None
        lastExp = 0
        lastFilter = None
        keepNext = True
        for fdu in self.getFDUs():
            doForget = False #only need to "forget" data if read in to calculate median
            if (keepNext == True):
                keep = True
            else:
                keep = False
                keepNext = True
            #Check flags for min, max frame values, ignore first frames
            minValue = self.getParam('min_frame_value', fdu.getTag())
            if (minValue is not None):
                doForget = True
                if (fdu.getMedian() < float(minValue) and not fdu.isDark):
                    keep = False
                    #Check ignore_after_bad_read
                    if (self.getParam('ignore_after_bad_read').lower() == 'yes'):
                        keepNext = False
            maxValue = self.getParam('max_frame_value', fdu.getTag())
            if (maxValue is not None):
                doForget = True
                if (fdu.getMedian() > float(maxValue)):
                    keep = False
                    #Check ignore_after_bad_read
                    if (self.getParam('ignore_after_bad_read').lower() == 'yes'):
                        keepNext = False
            #Check if first frame of series
            if (self.getParam('ignore_first_frames').lower() == 'yes' and (fdu._id != lastIdent or fdu.exptime != lastExp or fdu.filter != lastFilter)):
                keep = False
            lastIdent = fdu._id
            lastExp = fdu.exptime
            lastFilter = fdu.filter
            if (not keep):
                print("fatboyDatabase::performRejectionCriteria> Warning: Rejecting file "+fdu.getFilename())
                self._log.writeLog(__name__, "Rejecting file "+fdu.getFilename(), type=fatboyLog.WARNING)
                fdu.disable()
            elif (doForget and self.getParam('memory_image_limit') is not None):
                fdu.forgetData() #don't save data in memory at this step since we are memory limited
    #end performRejectionCriteria

    ## Read and preprocess data including rejection criteria and set obj type
    def preprocessAll(self):
        #Set GPU mode off if specified
        if (self._params['gpumode'].lower() == 'no'):
            self._gpumode = False
            for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
                fdu.setGPUMode(self._gpumode) #update gpu mode of all files
        if (self._gpumode):
            try:
                import pycuda.driver as drv
            except Exception:
                print("fatboyDatabase::preprocessAll> ERROR: GPU mode set to ON but PyCUDA not installed!  Exiting!")
                self._log.writeLog(__name__, "GPU mode set to ON but PyCUDA not installed!  Exiting!", type=fatboyLog.ERROR)
                sys.exit(-1)
        self.checkFiles() #Check for existence of files
        self.setMEF() #Set the proper MEF extensions

        #Process quick start file
        if (self._params['quick_start_file'] is not None):
            qsDict = self.readQuickStart()
            if (len(qsDict) > 0):
                self.applyQuickStart(qsDict) #apply shapes and median values to FDUs

        self.setFITSKeywords() #set up _keywords dict in FDUs
        self.initializeAll() #read headers and initialize all FDUs
        self.checkForObstypeOverrides() #Check for dark or flat file lists to override types

        self.checkForMasterCalibs() #Check here for master calibration frames and separate them into _calibs list

        self.performRejectionCriteria() #perform rejection criteria
        #Now write out quick start file -- shape has been read in in readHeader
        if (self._params['quick_start_file'] is not None):
            self.writeQuickStart(qsDict) #write out quick start file.  Append anything missing when it was read in.
    #end preprocessAll

    ## Print all params with current [default] values.
    def printParams(self, fdu=None):
        tag = None
        #set tag if fdu is given
        if (fdu is not None):
            tag = fdu.getTag()
        print("FATBOY Parameters:")
        for key in sorted(self._params): #sort keys alphabetically
            print("\t"+key+" = "+str(self.getParam(key, tag)))
        print("\n")
    #end printParams

    ## Print all processes and their options
    def printProcesses(self, fdu=None, modeTag=None):
        tag = None
        #set tag if fdu is given
        if (fdu is not None):
            tag = fdu.getTag()
        print("FATBOY Processes:")
        if (modeTag is not None):
            print("\t===== Processes for MODE "+modeTag+" =====")
        for key in sorted(self._processDict): #sort keys alphabetically
            result = self._processDict[key]
            process = result(key)
            if (modeTag is not None):
                if (not modeTag in process._modeTags):
                    continue
            process.setDefaultOptions()
            print("\tProcess = "+key)
            process.printOptions()
            print("\n")
    #end printProcesses

    ## Read in quick start file and return dict mapping filenames to shapes/medians
    def readQuickStart(self):
        qsDict = dict()
        if (not os.access(self._params['quick_start_file'], os.F_OK)):
            return qsDict
        qslines = readFileIntoList(self._params['quick_start_file'])
        for line in qslines:
            #Check for bad data
            qstokens = line.split()
            if (len(qstokens) != 3):
                #Should be full_id (shape) median
                #Changed from filename to full_id 4/22/15
                continue
            try:
                qsfullid = qstokens[0]
                qsshape = tuple(int(x) for x in re.findall("\d+", qstokens[1]))
                qsmed = float(qstokens[2])
                #Add to dict
                qsDict[qsfullid] = [qsshape, qsmed]
            except Exception as ex:
                print("fatboyDatabase::readQuickStart> Error: "+str(ex))
                self._log.writeLog(__name__, str(ex), type=fatboyLog.ERROR)
        return qsDict
    #end readQuickStart

    ## Set a few default parameters if they weren't already set
    def setDefaultParams(self):
        self._params.setdefault('calibs_only', 'no') #Set to yet if only processing calibs
        self._params.setdefault('gpumode', 'yes')
        self._params.setdefault('memory_image_limit', None)
        self._params.setdefault('median_kernel', None)
        self._params.setdefault('outputdir', '.')
        self._params.setdefault('logdir', 'flogs')
        self._params.setdefault('tempdir', 'temp-fatboy')
        self._params.setdefault('verbosity', 'normal')

        #Basic params
        self._params.setdefault('convert_mef', 'no')
        self._params.setdefault('mef_extension', None)
        self._params.setdefault('overwrite_files','no')
        self._params.setdefault('quick_start_file', None)

        #FITS Keywords
        self._params.setdefault('date_keyword',['DATE', 'DATE-OBS'])
        self._params.setdefault('dec_keyword',['DECOFFSE', 'DEC', 'TELDEC'])
        self._params.setdefault('exptime_keyword',['EXPTIME', 'EXP_TIME', 'EXPCOADD'])
        self._params.setdefault('filter_keyword',['FILTER', 'FILTNAME'])
        self._params.setdefault('gain_keyword',['GAIN', 'GAIN_1', 'EGAIN'])
        self._params.setdefault('nreads_keyword',['NREADS', 'LNRS', 'FSAMPLE', 'NUMFRAME'])
        self._params.setdefault('obstype_keyword',['OBSTYPE', 'OBS_TYPE', 'IMAGETYP'])
        self._params.setdefault('ra_keyword',['RAOFFSET', 'RA', 'TELRA'])
        self._params.setdefault('relative_offset_arcsec','no')
        self._params.setdefault('ut_keyword',['UT', 'UTC', 'NOCUTC'])

        #Rejection criteria
        self._params.setdefault('ignore_after_bad_read','no')
        self._params.setdefault('ignore_first_frames','no')
        self._params.setdefault('max_frame_value', None)
        self._params.setdefault('min_frame_value', None)

        #Obstype overrides
        self._params.setdefault('dark_file_list', None)
        self._params.setdefault('flat_file_list', None)
    #end setDefaultParams

    ## Set FITS keyword values for FDUs
    def setFITSKeywords(self):
        #First get unique keywords to set
        uniqueKeys = []
        for key in self._params:
            if (key.endswith("_keyword")):
                uniqueKeys.append(key)
            elif (key.find("_keyword::") != -1):
                uniqueKeys.append(key[:key.rfind("::")])
        #Remove duplicates
        uniqueKeys = list(set(uniqueKeys))
        #Loop over fdus
        for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
            #Loop over keys
            for key in uniqueKeys:
                value = self.getParam(key, fdu.getTag()) #returns match by key/tag if applicable, key otherwise
                if (isinstance(value, list)):
                    fdu.setKeyword(key, value)
                    continue
                #What if value is a number because its not set in header??
                #In this case, e.g. <param name="nreads_keyword" value="1"/>:
                #fdu._keywords['nreads_keyword'] = 'nreads_keyword', fdu._header['nreads_keyword'] = 1
                if (isinstance(value, str) and (isDigit(value[0]) or (isDigit(value[1]) and value[0] == '-'))):
                    #value is a number
                    if (value.find('.') != -1):
                        #value is a float
                        fdu.setKeyword(key, key, float(value))
                    else:
                        #value is an int
                        fdu.setKeyword(key, key, int(value))
                elif (isinstance(value, float) or isinstance(value, int)):
                    #value is a number
                    fdu.setKeyword(key, key, value)
                else:
                    #This is a header keyword like EXP_TIME
                    fdu.setKeyword(key, str(value))
            ##Also set the related relative_offset_arcsec parameter here
            value = self.getParam("relative_offset_arcsec", fdu.getTag())
            if (value.lower() == "yes"):
                fdu.setRelOffset(True)
    #end setFITSKeywords

    #Set proper MEF extensions for FDUs
    def setMEF(self):
        #Loop over FDUs
        for fdu in self.getFDUs(): #use getFDUs to ignore disabled fdus
            mef = self.getParam("mef_extension", fdu.getTag()) #returns match by key/tag if applicable, key otherwise
            if (mef is not None and mef != -1):
                fdu.setMEF(mef)
    #end setMEF

    ## Setup dict processes from fatboyProcesses dir and any user defined dirs
    def setupDatatypeDict(self):
        #First read datatypeDict from datatypeExtensions dir
        from .datatypeExtensions import datatypeDict as dd
        self._datatypeDict = dd.getDatatypeDict()
        #Now loop over any user-defined dirs
        for datatypeDir in self._datatypedirs:
            #Strip trailing /
            while (datatypeDir.endswith('/')):
                datatypeDir = datatypeDir[:-1]
            #enclose in try block
            try:
                #add superdir to sys.path
                idx = datatypeDir.rfind('/')
                superdir = datatypeDir[:idx]
                module = datatypeDir[idx+1:]
                sys.path.append(superdir)
                dd = importlib.import_module(module+'.datatypeDict')
                self._datatypeDict.update(dd.getDatatypeDict())
            except Exception as ex:
                print("fatboyDatabase::setupDatatypeDict> Error: "+str(ex))
                self._log.writeLog(__name__, str(ex), type=fatboyLog.ERROR)
    #end setupDatatypeDict

    ## Setup dict processes from fatboyProcesses dir and any user defined dirs
    def setupProcessDict(self):
        #First read processDict from fatboyProcesses dir
        from .fatboyProcesses import processDict as pd
        self._processDict = pd.getProcessDict()
        #Now loop over any user-defined dirs
        for processDir in self._processdirs:
            #Strip trailing /
            while (processDir.endswith('/')):
                processDir = processDir[:-1]
            #enclose in try block
            try:
                #add superdir to sys.path
                idx = processDir.rfind('/')
                superdir = processDir[:idx]
                module = processDir[idx+1:]
                sys.path.append(superdir)
                pd = importlib.import_module(module+'.processDict')
                self._processDict.update(pd.getProcessDict())
            except Exception as ex:
                print("fatboyDatabase::setupProcessDict> Error: "+str(ex))
                self._log.writeLog(__name__, str(ex), type=fatboyLog.ERROR)
    #end setupProcessDict

    ## Write out quick start file. Skip anything in dict mapping from reading file earlier, just append new files
    def writeQuickStart(self, qsDict):
        try:
            qsfile = open(self._params['quick_start_file'], 'a')
        except Exception as ex:
            print("fatboyDatabase::writeQuickStart> Error: could not open file "+self._params['quick_start_file']+" for writing!")
            self._log.writeLog(__name__, "could not open file "+self._params['quick_start_file']+" for writing!", type=fatboyLog.ERROR)
            return
        print("fatboyDatabase::writeQuickStart> Writing quick start file "+self._params['quick_start_file'])
        self._log.writeLog(__name__, "Writing quick start file "+self._params['quick_start_file'])
        ##4/22/15 -- Changed quickstart file to look up by full id, not filename due to CIRCE data with multiple ramps
        for fdu in self.getFDUs():
            if (fdu.getFullId() in qsDict):
                #Already in quickstart file
                continue
            if (fdu._medVal is None):
                #Median has not been calculated.  No need to calculate and store it when it won't be used.
                continue
            try:
                qsstring = fdu.getFullId()+"\t"+str(fdu.getShape()).replace(" ","")+"\t"+str(fdu.getMedian())+"\n"
            except Exception:
                print("fatboyDatabase::writeQuickStart> Error: Misformatted data in "+fdu.getFilename()+"!  Discarding file!")
                self._log.writeLog(__name__, "Misformatted data in "+fdu.getFilename()+"!  Discarding file!")
                if (fdu.inUse):
                    fdu.disable()
                continue
            qsfile.write(qsstring)
            if (self.getParam('memory_image_limit') is not None):
                fdu.forgetData() #don't save data in memory at this step since we are memory limited
        qsfile.close()
        return qsDict
    #end writeQuickStart
