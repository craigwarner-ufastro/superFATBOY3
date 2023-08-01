from .fatboyImage import *

#extends fatboyImage
#class for master calibration frames -- does not need to be created from a file!
class fatboyCalib(fatboyImage):
    _name = "fatboyCalib"
    _pname = None #Name of process that created this fatboyCalib

    ## The constructor.
    def __init__(self, pname, obstype, source, filename=None, data=None, tagname=None, headerExt=None, log=None):
        self._pname = pname
        if (log is not None):
            self._log = log
        #Initialize dicts
        self._header = dict()
        self._history = dict()
        self._keywords = dict()
        self._properties = dict()
        #Initialize lists
        self._objectTags = []
        self._processHistory = []

        #Can be created from a filename or data array and mandatory fatboyDataUnit source
        if (filename is None and data is None):
            print("fatboyCalib::__init__> ERROR: pname="+pname+"; source="+source.getFullId()+": filename or data must be specified to instantiate fatboyCalib.")
            if (self._log is not None):
                self._log.writeLog(__name__, "pname="+pname+"; source="+source.getFullId()+": filename or data must be specified to instantiate fatboyCalib.", type=fatboyLog.ERROR)
            #disable FDU and return
            self.disable()
            return

        #Copy over mef, keywords, objectTags, optionally header from source
        self.setMEF(source._mef)
        self._keywords = source._keywords
        self._objectTags = source._objectTags
        self.section = source.section
        self.setDatabaseCallback(source._fdb) #set database callback
        self.setGPUMode(source.getGPUMode())
        self.setType(obstype, True)
        self.setTag(source.getTag()) #tag for proper dataset
        if (filename is not None):
            self.filename = filename
            self._id = source._id #Copy _id over from source
            if (tagname is not None):
                self._id = tagname[tagname.rfind('/')+1:] #Use tagname if given
            self._identFull = filename
            self.checkFile()
            if (self.inUse):
                self.readHeader()
        else:
            self._data = data
            self._header = source._header
            self.filename = source.filename
            self._id = tagname[tagname.rfind('/')+1:]
            self._identFull = self._id+".fits"
            self.setHistory("sourceFilename", source.getFilename())
        if (headerExt is not None):
            #extension header values
            self.updateHeader(headerExt)
        if (self._identFull.rfind('/') != -1):
            self._identFull = self._identFull[self._identFull.rfind('/')+1:]
        if (not self._identFull.endswith(".fits")):
            self._identFull += ".fits"
        self.initialize()
        print(self._name+": "+self._pname+": "+self._identFull)
    ##end __init__

    #return process name that created this calib
    def getCalibProcessName(self):
        return self._pname
    #end getCalibProcessName
