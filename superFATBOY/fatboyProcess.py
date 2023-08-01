## @package superFATBOY
#  Documentation for pipeline.
#
#

import glob, types
import xml.dom.minidom
from xml.dom.minidom import Node
from .fatboyImage import *
#from datetime import *

## Documentation for fatboyProcess
#
#
class fatboyProcess:
    _name = "fatboyProcess"
    _pname = None # Process name
    _outName = None # Output name, e.g. masterBias

    _fdb = None #fatboyDatabase instance
    _xmlNode = None
    _calibs = dict() #dict of calib tags
    _options = dict() #dict of option tags
    _optioninfo = dict() #info about each option printed in -list
    _log = None
    _shortlog = None
    _outputdir = None
    _tag = None
    _modeTags = []

    ## The constructor.
    def __init__(self, xml, fdb=None):
        #Initialize dicts
        self._calibs = dict()
        self._options = dict()
        self._optioninfo = dict()
        #Set default for write_output and write_calib_output to no
        self._options.setdefault("create_calib_only", "no")
        self._options.setdefault("write_output", "no")
        self._options.setdefault("write_calib_output", "no")
        if (fdb is None):
            # -list option.  Only setDefaultOptions and printOptions will be called!
            # option passed is process name
            self._pname = xml
            return #return here
        ### Normal operation
        self._xmlNode = xml
        self._fdb = fdb
        self._log = fdb.getLog()
        self._outputdir = fdb.getParam('outputdir')
        self._shortlog = fdb.getShortLog()

        #Parse the xml in init.  This will set up options dict
        self.parseXML(xml)
        if (self._outputdir is None or not os.access(self._outputdir, os.W_OK)):
            #Should not happen
            print("fatboyProcess::__init__> ERROR: Invalid outputdir: "+str(self._outputdir))
            self._log.writeLog(__name__, "Invalid outputdir: "+str(self._outputdir), type=fatboyLog.ERROR)
    #end __init__

    ## Check if an output file exists
    def checkOutputExists(self, fdu, outfile, tag=None, headerTag=None):
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        outfile = outdir+"/"+outfile
        #Check to see if it exists and overwrite is no
        if (os.access(outfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "no"):
            fdu.updateFrom(outfile, tag=tag, headerTag=headerTag, pname=self._pname)
            return True
        return False
    #end checkOutputExists

    ## Check if this process is tagged to only be run on certain images
    # And if so, check image tags
    def checkTag(self, fdu):
        if (self.getTag() is not None):
            if (fdu.getTag() is not None):
                tag = fdu.getTag()
                if (isinstance(tag, list)):
                    for j in range(len(tag)-1, -1, -1):
                        #Loop backward through tags
                        if (tag[j] is not None and tag[j] == self.getTag()):
                            #process tag != None, fdu tag != None, tags match
                            return True
                elif (tag == self.getTag()):
                    #process tag != None, fdu tag != None, tags match
                    return True
            #process tag != None but fdu tag is None or does not match, don't execute
            return False
        #process tag is None, execute normally on all fdus
        return True
    #end checkTag

    #Not all previous processes should be applied to all calibs.
    #For instance flat fields should never be sky subtracted.
    #checkValidDatatype returns True by default but can be overridden.
    def checkValidDatatype(self, fdu):
        return True
    #end checkValidDatatype

    ## execute should be overridden by subclasses.  The base implementation prints out the options dict and object names
    def execute(self, fdu, prevProc=None):
        print("Executing process "+str(self._pname)+" on fatboyDataUnit "+fdu.getName())
        self._log.writeLog(__name__, "Executing process "+str(self._pname)+" on fatboyDataUnit "+fdu.getName(), prependSeperator=True, appendSeperator=True)
        print("Options:")
        for key in self._options:
            print("\t"+key+" = "+self._options[key])
        return True
    #end execute

    ## Get a calib given a name and optionally a tag
    def getCalib(self, cname, ctag=None):
        if (ctag is not None):
            if (isinstance(ctag, list)):
                #Has both a tag and a subtag.  Look for subtag first then tag.
                for j in range(len(ctag)-1, -1, -1):
                    if (ctag[j] is not None and (cname+'::'+ctag[j]) in self._calibs):
                        return self._calibs[cname+'::'+ctag[j]]
            else:
                #Is a string
                if ((cname+'::'+ctag) in self._calibs):
                    return self._calibs[cname+'::'+ctag]
        #Couldn't find tag-specific option
        if (cname in self._calibs):
            return self._calibs[cname]
        return None
    #end getCalib

    ## Should be overridden by subclasses.  Find calibration frame(s) for this fatboyDataUnit
    def getCalibs(self, fdu, prevProc=None):
        print("Finding calib for fatboyDataUnit "+fdu.getName())
        self._log.writeLog(__name__, "Finding calib for fatboyDataUnit "+fdu.getName())
        calibFDUs = []
        #Recursively process calibs here if necessary.
        #For instance, inidivudal dark frames may need to be linearity corrected before being combined into master dark frame
        self.recursivelyExecute(calibFDUs, prevProc)

        #return value is calibs, which should be a dict.
        #For instance for darkSubtract, calibs has one entry, 'masterDark', which is an fdu.
        calibs = dict()
        return calibs
    #end getCalibs

    ## Get an option given a name and optionally a tag
    def getOption(self, oname, otag=None):
        if (otag is not None):
            if (isinstance(otag, list)):
                #Has both a tag and a subtag.  Look for subtag first then tag.
                for j in range(len(otag)-1, -1, -1):
                    if (otag[j] is not None and (oname+'::'+otag[j]) in self._options):
                        return self._options[oname+'::'+otag[j]]
            else:
                #Is a string
                if ((oname+'::'+otag) in self._options):
                    return self._options[oname+'::'+otag]
        #Couldn't find tag-specific option
        if (oname in self._options):
            return self._options[oname]
        return None
    #end getOption

    ## Get _tag
    def getTag(self):
        return self._tag
    #end getTag

    ## Return whether an option is defined for this process
    def hasOption(self, oname, otag=None):
        if (otag is not None):
            if (isinstance(otag, list)):
                #Has both a tag and a subtag.  Look for subtag first then tag.
                for j in range(len(otag)-1, -1, -1):
                    if (otag[j] is not None and (oname+'::'+otag[j]) in self._options):
                        return True
                #Not found
                return False
            else:
                #Is a string
                return ((oname+'::'+otag) in self._options)
        else:
            return (oname in self._options)
    #end hasOption

    ## Parses XML node for fatboyProcess
    def parseXML(self, xml):
        #Parse attributes first
        if (xml.hasAttributes()):
            for j in range(xml.attributes.length):
                attr = xml.attributes.item(j)
                if (attr.nodeName == "name"):
                    self._pname = str(attr.nodeValue)
                elif (attr.nodeName == "outputName"):
                    self._outName = str(attr.nodeValue)
                elif (attr.nodeName == "tag"):
                    self._tag = str(attr.nodeValue) #convert unicode to str
        #Next parse options
        #get all option nodes
        optionNodes = xml.getElementsByTagName('option')
        #loop over option nodes
        for option in optionNodes:
            if (option.nodeType == Node.ELEMENT_NODE and option.nodeName == 'option'):
                if (option.hasAttribute("name") and option.hasAttribute("value")):
                    oname = str(option.getAttribute("name"))
                    oval = str(option.getAttribute("value"))
                    if (option.hasAttribute("tag")):
                        otag = str(option.getAttribute("tag"))
                        #Add tagged option to dict with format _options['name::tag'] = value
                        self._options[oname+'::'+otag] = oval
                    else:
                        #Add global option to dict with format _options['name'] = value
                        self._options[oname] = oval

        #Next get all calib nodes
        self._calibNodes = xml.getElementsByTagName('calib')
        self._calibQueries = []
        for calib in self._calibNodes:
            if (calib.nodeType == Node.ELEMENT_NODE and calib.nodeName == 'calib'):
                if (calib.hasAttribute("name") and calib.hasAttribute("value")):
                    cname = str(calib.getAttribute("name"))
                    cval = str(calib.getAttribute("value"))
                    if (calib.hasAttribute("tag")):
                        ctag = str(calib.getAttribute("tag"))
                        #Add tagged calib to dict with format _calibs['name::tag'] = value
                        self._calibs[cname+'::'+ctag] = cval
                    else:
                        #Add global calib to dict with format _calibs['name'] = value
                        self._calibs[cname] = cval
    #end parseXML

    ## Print all options with current [default] values.
    def printOptions(self, fdu=None):
        tag = None
        #set tag if fdu is given
        if (fdu is not None):
            tag = fdu.getTag()
        print("\tOptions for process "+self._pname+":")
        for key in sorted(self._options): #sort keys alphabetically
            print("\t\t"+key+" = "+str(self.getOption(key, tag)))
            #Print info if available
            if (key in self._optioninfo):
                info = str(self._optioninfo[key])
                info = info.split('\n')
                print("\t\t\t* "+info[0])
                for j in range(1, len(info)):
                    print("\t\t\t  "+info[j])
    #end printOptions

    ## Recursively execute previous processes on calibs ##
    def recursivelyExecute(self, calibs, prevProc):
        if (prevProc is None):
            #no previous processes given to execute
            return
        #First loop over calibration frames
        ncalibs = len(calibs)
        nc = 0
        for calib in calibs:
            nc += 1
            tt = time.time()
            writeShortLogHeader = True
            print("fatboyProcess::recursivelyExecute> Recursively processing "+calib.getFullId()+" ("+str(nc)+" of "+str(ncalibs)+") from process "+self._pname)
            self._log.writeLog(__name__, "Recursively processing "+calib.getFullId()+" ("+str(nc)+" of "+str(ncalibs)+") from process "+self._pname)
            #Setup list of previously done image processes to recursively process calibs
            prevProcesses = []
            #Second loop over processes
            for process in prevProc:
                if (not calib.inUse):
                    #if image has been disabled, continue
                    #this may happen due to rejection along the way, or pairing or stacking with other fdus
                    continue
                if (not process.checkValidDatatype(calib)):
                    #Not all previous processes should be applied to all calibs.
                    #For instance flat fields should never be sky subtracted.
                    #checkValidDatatype returns True by default but can be overridden.
                    continue
                histName = process._pname
                if (process.hasOption("pass_number")):
                    histName += "::"+process.getOption("pass_number")
                if (not process.checkTag(calib)):
                    #this process is only supposed to be performed on images matching a certain tag
                    #this image does not match, continue to next process
                    print("\tProcess "+str(process._pname)+" will only be run on images with tag="+str(process.getTag())+", skipping fatboyDataUnit "+calib.getFullId()+" with tag "+str(calib.getTag())+".")
                    self._log.writeLog(__name__, "Process "+str(process._pname)+" will only be run on images with tag="+str(process.getTag())+", skipping fatboyDataUnit "+calib.getFullId()+" with tag "+str(calib.getTag())+".", printCaller=False, tabLevel=1, verbosity=fatboyLog.VERBOSE)
                    #add process prevProcesses list even though it is not executed here
                    #it needs to be in prevProcesses for recursion on other images
                    prevProcesses.append(process)
                    continue
                #if (calib.hasProcessInHistory(process._pname)):
                if (calib.hasProcessInHistory(histName)):
                    #This process has already been applied to this FDU
                    print("\tAlready executed process "+str(process._pname)+" on fatboyDataUnit "+calib.getFullId())
                    self._log.writeLog(__name__, "Already executed process "+str(process._pname)+" on fatboyDataUnit "+calib.getFullId(), printCaller=False, tabLevel=1, verbosity=fatboyLog.VERBOSE)
                    prevProcesses.append(process)
                    continue
                #Execute previous processes
                print("\tPerforming process "+str(process._pname))
                self._log.writeLog(__name__, "Performing process "+str(process._pname), printCaller=False, tabLevel=1)
                if (writeShortLogHeader):
                    writeShortLogHeader = False
                    self._shortlog.writeLog(__name__, "Recursively processing "+calib.getFullId()+" ("+str(nc)+" of "+str(ncalibs)+")", tabLevel=2)
                self._shortlog.writeLog(__name__, "Performing process "+str(process._pname), printCaller=False, tabLevel=3)
                #set default options
                process.setDefaultOptions()
                #execute process
                process.execute(calib, prevProc=prevProcesses)
                #add process name to _processHistory for this FDU
                #calib.addProcessToHistory(process._pname)
                calib.addProcessToHistory(histName)
                #if write_output is yes, call writeOutput method
                if (process.getOption("write_output", calib.getTag()).lower() == "yes"):
                    #update list of previous filenames to potentially clean up.  Exclude original file from this list
                    calib.updateFilenames()
                    process.writeOutput(calib)
                #add process to list
                prevProcesses.append(process)
            if (not writeShortLogHeader):
                #only set to false if a process is executed
                self._shortlog.writeLog(__name__, "Time: "+str(time.time()-tt)+"; Total: "+str(time.time()-self._fdb._t), printCaller=False, tabLevel=3)
    #end recursivelyExecute

    ## Set default options here.  Should be overridden by subclasses!
    def setDefaultOptions(self):
        print("fatboyProcess::setDefaultOptions> This method should be overridden!")
    #end setDefaultOptions

    #Sets an option
    def setOption(self, name, value):
        self._options[name] = value
    #end setOption

    ## Write output after execute has finished.  Should be overridden by subclasses!
    def writeOutput(self, fdu):
        print("fatboyProcess::writeOutput> This method should be overridden!")
    #end writeOutput
