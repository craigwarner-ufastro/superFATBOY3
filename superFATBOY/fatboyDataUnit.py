## @package superFATBOY
#  Documentation for pipeline.
#
#

from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import *
import os, time

## Documentation for fatboyDataUnit
#
#
class fatboyDataUnit:
    ##static type def variables
    FDU_TYPE_UNDEFINED = 0
    FDU_TYPE_OBJECT = 1
    FDU_TYPE_DARK = 2
    FDU_TYPE_FLAT = 3
    FDU_TYPE_SKY_FLAT = 4
    FDU_TYPE_SKY = 5
    FDU_TYPE_MASTER_CALIB = 6
    FDU_TYPE_BAD_PIXEL_MASK = 7
    FDU_TYPE_ARCLAMP = 8
    FDU_TYPE_BIAS = 9

    ## class variables
    _name = "fatboyDataUnit"
    _id = None #Unique ID
    _index = None #index suffix
    _identFull = None #_id._index.fits
    _mef = None #extension
    _data = None #Data at current step in processing
    _shape = None #Data shape
    _medVal = None #Median value
    _history = dict() #History of processing steps
    ### History includes origFilename, origShape, previousFilenames ###
    _header = dict() #Entire FITS header
    _keywords = dict() #Keywords for various header values
    _log = fatboyLog(verbosity=fatboyLog.SIM) #fatboyLog
    filename = None #current filename
    _tag = None #allows groups of objects and calibration to be tagged
    _subtag = None #second level of tagging for objects within same dataset
    _objectTags = [] #a list of objects that a calibration frame is associated with
    _properties = dict() #Properties defined in XML file, e.g. flattype = "lamp on".  Also data from previous steps
    _gpumode = True #Use GPU
    _processHistory = [] #Track processes applied to this fdu

    _fdb = None #fatboyDatabase callback
    _verbosity = fatboyLog.NORMAL

    ## generated
    badPixelMask = None
    _mask = None
    _maskedMedian = None

    #boolean values
    inUse = True
    isDark = False
    isFlat = False
    isObject = True
    isSky = False
    isMasterCalib = False
    isSkyFlat = False
    isOffFlat = False
    _suffix = False #Set to true if original fileame has suffix instead of prefix, e.g. 01234-date-MIRADAS.fits

    #parameters
    exptime = 0
    nreads = 0
    filter = None #filter
    obstype = None #obs type string
    _obstypeVal = FDU_TYPE_UNDEFINED #obs type enum
    section = -1 #original mef for images with multiple extensions
    ra = None
    dec = None
    relOffset = False #RA and DEC are relative offsets in arcsec

    ## The constructor.
    def __init__(self, filename, log=None):
        self.filename = str(filename)
        self.findIdentifier()
        if (log is not None):
            self._log = log
            self._verbosity = log._verbosity
        #Initialize dicts
        self._header = dict()
        self._history = dict()
        self._keywords = dict()
        self._properties = dict()
        #Initialize lists
        self._objectTags = []
        self._processHistory = []
        if (self._verbosity != fatboyLog.BRIEF):
            print(self._name+": "+self.filename)
            if (log is not None):
                self._log.writeLog(__name__, self._name+": "+self.filename, printCaller=False, tabLevel=1)

    ## add an object tag for a calib frame
    def addObjectTag(self, tag):
        self._objectTags.append(tag)
    #end addObjectTag

    ## add a process to process history
    def addProcessToHistory(self, pname):
        self._processHistory.append(pname)
    #end addProcessToHistory

    #apply a bad pixel mask
    def applyBadPixelMask(self, bpm):
        self.badPixelMask = bpm
        #Apply bad pixel mask
        data = self.getData()
        #Set data to 0 where badPixelMask is True
        data[self.badPixelMask.getData()] = 0
        self.updateData(data)
    #end applyBadPixelMask

    ## check that self.filename exists and disable if it doesn't
    def checkFile(self):
        if (not os.access(self.filename, os.F_OK)):
            self.disable()
            print("fatboyDataUnit::checkFile> Error: file "+self.filename+" NOT FOUND!  Disabling this FDU.")
            self._log.writeLog(__name__, "Error: file "+self.filename+" NOT FOUND!  Disabling this FDU.", type=fatboyLog.ERROR)
    #end checkFile

    ## Set state to disabled
    def disable(self):
        self.inUse = False
        if (self._data is not None and self._fdb is not None):
            self._fdb.decrementMemoryCount() #decrement database memory image counter
        if (self._data is not None):
            del self._data #use del to release memory
        self._data = None
        #Clean up memory managed properties
        for key in list(self._properties):
            if (isinstance(self._properties[key], str) and self._properties[key] == "memory_managed"):
                #data tag that has been backed up to disk for memory management purposes
                infile = "temp-fatboy/property_"+key+"_"+self.getFullId()
                if (os.access(infile, os.F_OK)):
                    os.unlink(infile)
        if (self.hasProperty('memory_managed') and self.hasProperty('origFilename')):
            if (self.filename == "temp-fatboy/current_"+self.getFullId()):
                os.unlink(self.filename)
                self.filename = self.getProperty('origFilename')
                self.removeProperty('origFilename')
                self.removeProperty('memory_managed')
        self._properties.clear()
    #end disable

    ## Set state to enabled
    def enable(self):
        self.inUse = True
    #end enable

    ## Find prefix and index
    #9/11/23 - take delim as optional arg and return fileprefix, sfileindex
    #to replace logic from findIdentAndIndex in fatboyQuery.py
    def findIdentifier(self, delim=None):
        if (self._suffix):
            if (delim is not None):
                #Everything before delimiter
                self._index = self.filename[:self.filename.find(delim)]
                self._id = self.filename[self.filename.find(delim) + len(delim):self.filename.rfind('.')]
                self._id = str(self._id) #convert from unicode to str!!
                self._index = str(self._index) #convert fron unicode to str!!
                return (self._id, self._index)
            #handle case for suffix
            dpos = self.filename.rfind('.')
            if (self.filename.endswith('.fz')):
                dpos = self.filename[:-3].rfind('.')
            #Start with position 0 and find leftmost non-numerical character
            cpos = 0
            spos = 0
            if (self.filename.rfind('/') > -1):
                cpos = self.filename.rfind('/')+1
                spos = self.filename.rfind('/')+1
            while (isDigit(self.filename[cpos]) and cpos < dpos):
                cpos += 1
            self._index = self.filename[spos:cpos]
            #Make sure suffix does not start with . - or _
            self._id = self.filename[cpos:dpos]
            while (self._id.startswith('.') or self._id.startswith('-') or self._id.startswith('_')):
                self._id = self._id[1:]
            self._id = str(self._id) #convert from unicode to str!!
            self._index = str(self._index) #convert fron unicode to str!!
            self._identFull = self._id+'.'+self._index+'.fits'
            return (self._id, self._index)
        #normal prefix case
        if (delim is not None):
            #Everything before delimiter
            self._id = self.filename[:self.filename.rfind(delim, 0, self.filename.rfind('.'))]
            self._index = self.filename[self.filename.rfind(delim, 0, self.filename.rfind('.')) + len(delim):self.filename.rfind('.')]
            self._id = str(self._id) #convert from unicode to str!!
            self._index = str(self._index) #convert fron unicode to str!!
            return (self._id, self._index)
        dpos = self.filename.rfind('.')
        if (self.filename.endswith('.fz')):
            dpos = self.filename[:-3].rfind('.')
        cpos = dpos-1
        #Find rightmost non-numerical character before .fits
        while(isDigit(self.filename[cpos]) and cpos > 0):
            cpos-=1
        self._id = self.filename[self.filename.rfind('/')+1:cpos+1]
        while (self._id.endswith('.') or self._id.endswith('-') or self._id.endswith('_')):
            self._id = self._id[:-1]
        self._index = self.filename[cpos+1:dpos]
        self._id = str(self._id) #convert from unicode to str!!
        self._index = str(self._index) #convert fron unicode to str!!
        self._identFull = self._id+'.'+self._index+'.fits'
        return (self._id, self._index)
    #end findIdentifier

    ## Find first data extension
    def findMef(self, image):
        #Check for MEF extensions if set to auto
        if (self._mef is None):
            for j in range(len(image)):
                #Try header first if possible
                if ('NAXIS' in image[j].header):
                    if (image[j].header['NAXIS'] > 1):
                        self._mef = j
                        break
        if (self._mef is None):
            #Header failed, now use data
            for j in range(len(image)):
                if (image[j].data is not None):
                    self._mef = j
                    break
        if (self._verbosity == fatboyLog.VERBOSE):
            print("\t"+self.getFullId()+" Extension = "+str(self._mef))
            self._log.writeLog(__name__, self.getFullId()+" Extension = "+str(self._mef), printCaller=False, tabLevel=1, verbosity=fatboyLog.VERBOSE)
    #end findMef

    ## Free memory but don't set inUse to false yet.  Used if header is still needed
    def forgetData(self):
        if (self._data is not None and self._fdb is not None):
            self._fdb.decrementMemoryCount() #decrement database memory image counter
            del self._data #use del to release memory
            self._data = None
    #end forgetData

    ## Get and return a bad pixel mask
    def getBadPixelMask(self):
        if (self.badPixelMask is None):
            #Create a bad pixel mask with no bad pixels
            #Do not need to save it as class member in this case
            data = zeros(self._shape, bool)
            bpmname = "badPixelMasks/BPM-"+str(self.filter)+"-"+str(self.nreads)+"rd-"+str(self._id)
            return fatboyCalib("fatboyDataUnit:getBadPixelMask", fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, self, data=data, tagname=bpmname, log=self._log)
        return self.badPixelMask
    #end getBadPixelMask

    ## Get and return data. Only read from disk if necessary.
    def getData(self, tag=None):
        if (tag is not None and self.hasProperty(tag)):
            #Asking for data from a specific step -- e.g. preSkySubtraction for use in creating master skies
            return self.getProperty(tag)
        if (self._data is None):
            #Read from disk
            t = time.time()
            image = pyfits.open(self.filename)
            self._data = image[self._mef].data
            if (self._data.dtype == 'uint16'):
                self._data = self._data.astype(int32)
            if (self.hasHeaderValue('BZERO')):
                (pname, version) = getPyfitsVersion()
                if (pname == "astropy" and version < '1.1'):
                    self._data = self._data+self.getHeaderValue('BZERO')
                elif (version < '3.2'):
                    self._data = self._data+self.getHeaderValue('BZERO')
                #Always remove BZERO since it is now int32 data
                self.removeHeaderKeyword('BZERO')
            if (not self._data.dtype.isnative):
                #Byteswap
                self._data = self._data.byteswap()
                self._data = self._data.newbyteorder('<')
                self._data.dtype.newbyteorder('<')
            self._shape = self._data.shape
            self.reformatData() #in case actual data disagrees with header, check to reformat data from (1,2048,2048) to (2048,2048) here too
            image.close()
            if (self.hasProperty('memory_managed') and self.hasProperty('origFilename')):
                #This was restored from disk
                if (self.filename == "temp-fatboy/current_"+self.getFullId()):
                    os.unlink(self.filename)
                    self.filename = self.getProperty('origFilename')
                    self.removeProperty('origFilename')
                    self.removeProperty('memory_managed')
            if (self._fdb is not None):
                self._fdb.totalReadDataTime += (time.time()-t)
                self._fdb.checkMemoryManagement(self) #check memory status
        if (self.getObsType(True) == self.FDU_TYPE_BAD_PIXEL_MASK):
            #bad pixel masks should be type bool
            if (self._data.dtype != dtype("bool")):
                self._data = self._data.astype("bool")
        return self._data
    #end getData

    ## return current _filename
    def getFilename(self):
        if (not os.access(self.filename, os.F_OK)):
            if (self.hasProperty('origFilename') and os.access(self.getProperty('origFilename'), os.F_OK)):
                return self.getProperty('origFilename')
        return self.filename
    #end getFilename

    ## return _identFull
    def getFullId(self):
        return self._identFull
    #end getFullId

    ## Get GPU mode
    def getGPUMode(self):
        return self._gpumode
    #end getGPUMode

    ## return a header value
    def getHeaderValue(self, key):
        if (key.endswith("_keyword")):
            if (self._keywords[key] in self._header):
                return self._header[self._keywords[key]]
        else:
            if (key in self._header):
                return self._header[key]
        return None
    #end getHeaderValue

    ## Get a value from history
    def getHistory(self, key):
        if (key in self._history):
            return self._history[key]
        else:
            return None
    #end getHistory

    # get data with boolean mask applied
    def getMaskedData(self, tag=None):
        if (self._mask is None):
            return self.getData(tag)
        return self.getData(tag)*self._mask
    #end getMaskedData

    ## get median value of data with mask applied
    def getMaskedMedian(self, tag=None):
        #If tag is not None, calculate masked median but do not assign it
        if (tag is not None and self.hasProperty(tag)):
            if (self._gpumode):
                return gpu_arraymedian(self.getMaskedData(tag), nonzero=True, kernel=self._fdb.getParam('median_kernel'))
            else:
                return gpu_arraymedian(self.getMaskedData(tag), nonzero=True, kernel=fatboyclib.median)
        #If median value has been already calculated, just return it.
        if (self._maskedMedian is None):
            #Otherwise calculate it and save it
            t = time.time()
            if (self._gpumode):
                self._maskedMedian = gpu_arraymedian(self.getMaskedData(), nonzero=True, kernel=self._fdb.getParam('median_kernel'))
            else:
                self._maskedMedian = gpu_arraymedian(self.getMaskedData(), nonzero=True, kernel=fatboyclib.median)
            if (self._verbosity == fatboyLog.VERBOSE):
                print("fatboyDataUnit::getMaskedMedian> Calculated median value for "+self.getFullId()+": "+str(self._maskedMedian)+" in "+str(time.time()-t)+"s")
                self._log.writeLog(__name__, "Calculated median value for "+self.getFullId()+": "+str(self._maskedMedian)+" in "+str(time.time()-t)+"s", verbosity=fatboyLog.VERBOSE)
        return self._maskedMedian
    #end getMaskedMedian

    ## Get and return median value
    def getMedian(self, tag=None):
        if (tag is not None and self.hasProperty(tag+"_median")):
            #Asking for data from a specific step -- e.g. preSkySubtraction for use in creating master skies
            return self.getProperty(tag+"_median")
        #If tag is not None, calculate median and assign it as property
        if (tag is not None and self.hasProperty(tag)):
            if (self._gpumode):
                taggedMedian = gpu_arraymedian(self.getData(tag), nonzero=True, kernel=self._fdb.getParam('median_kernel'))
                self.setProperty(tag+"_median", taggedMedian)
                return taggedMedian
            else:
                taggedMedian = gpu_arraymedian(self.getData(tag), nonzero=True, kernel=fatboyclib.median)
                self.setProperty(tag+"_median", taggedMedian)
                return taggedMedian
        #If median value has been already calculated, just return it.
        if (self._medVal is None):
            #Otherwise calculate it and save it
            t = time.time()
            if (self._gpumode):
                self._medVal = gpu_arraymedian(self.getData(), nonzero=True, kernel=self._fdb.getParam('median_kernel'))
            else:
                self._medVal = gpu_arraymedian(self.getData(), nonzero=True, kernel=fatboyclib.median)
            if (self._verbosity == fatboyLog.VERBOSE):
                print("fatboyDataUnit::getMedian> Calculated median value for "+self.getFullId()+": "+str(self._medVal)+" in "+str(time.time()-t)+"s")
                self._log.writeLog(__name__, "Calculated median value for "+self.getFullId()+": "+str(self._medVal)+" in "+str(time.time()-t)+"s", verbosity=fatboyLog.VERBOSE)
        return self._medVal
    #end getMedian

    ## Base class returns empty list.  Can be overridden to return a list of fatboyDataUnit (or subclass) representing multiple data extensions.
    ## Each should have a different fdu.section value.  For instance, newfirm has 4 detectors or CIRCE has multiple nramps.
    def getMultipleExtensions(self):
        return []
    #end getMultipleExtensions

    ## Get name method -- should return _identFull, not filename
    def getName(self):
        return self._identFull
    #end getName

    ## Get obs type
    def getObsType(self, value=False):
        #return string by default unless enum value is specified
        if (value):
            return self._obsTypeVal
        else:
            return self.obstype
    #end getObsType

    ## Get a property
    def getProperty(self, key):
        if (key in self._properties):
            if (isinstance(self._properties[key], str) and self._properties[key] == "memory_managed"):
                #data tag that has been backed up to disk for memory management purposes
                infile = "temp-fatboy/property_"+key+"_"+self.getFullId()
                if (os.access(infile, os.F_OK)):
                    image = pyfits.open(infile)
                    temp = array(image[0].data)
                    if (not temp.dtype.isnative):
                        #Byteswap
                        temp = temp.byteswap()
                        temp = temp.newbyteorder('<')
                        temp.dtype.newbyteorder('<')
                    image.close()
                    del image
                    #Return data here -- do not re-add to properties for memory management purposes
                    return temp
                else:
                    #Should not happen unless temp-fatboy manually deleted while running
                    self.removeProperty(key)
                    return None
            return self._properties[key]
        else:
            return None
    #end getProperty

    ## Get shape
    def getShape(self):
        if (len(self._shape) > 2 and self._shape[0] == 1):
            #reformatData has not been called yet but getShape has
            return (self._shape[1], self._shape[2])
        return self._shape
    #end getShape

    ## Get filename without path
    def getShortName(self):
        return self.filename[self.filename.rfind('/')+1:]
    #end getShortName

    ## Get _tag
    def getTag(self, mode="all"):
        #Back-modified to allow subtags without changing hundreds of calls
        #Default argument will now return list if a subtag exists
        #mode = all | tag_only | subtag_only | composite
        if (mode == "all"):
            if (self._subtag is not None):
                return [self._tag, self._subtag]
            return self._tag
        elif (mode == "tag_only"):
            return self._tag
        elif (mode == "subtag_only"):
            return self._subtag
        else:
            #composite
            if (self._subtag is not None):
                return self._subtag
            return self._tag
        return self._tag
    #end getTag

    ## Check to see if header contains the key
    def hasHeaderValue(self, key):
        if (key.endswith("_keyword")):
            return self._keywords[key] in self._header
        return key in self._header
    #end hasHeaderValue

    ## Check to see if history contains the key
    def hasHistory(self, key):
        return key in self._history
    #end hasHistory

    ## This method can be overridden in subclasses to support images with multiple data extensions, represented as multiple fatboyDataUnits.
    def hasMultipleExtensions(self):
        return False
    #end hasMultipleExtensions

    ## Check to see if a process is in process history
    def hasProcessInHistory(self, pname):
        return pname in self._processHistory
    #end hasProcessInHistory

    ## Check to see if _properties contains the key
    def hasProperty(self, key):
        return key in self._properties
    #end hasProperty

    ## Initialize reads certain values from the header and determines the shape.  It also finds the mef extension if not previously done.
    def initialize(self):
        ##Look at _keywords that are comma separated and turn them into lists (e.g. FILTER1, FILTER2)
        for key in list(self._keywords):
            if (not isinstance(self._keywords[key], list)):
                if (self._keywords[key].count(",") > 0):
                    self._keywords[key] = self._keywords[key].replace(' ','').split(',')
        try:
            self.exptime = self.getHeaderValue('exptime_keyword')
        except Exception:
            print("fatboyDataUnit::initialize> Warning: Unable to find keyword "+self._keywords['exptime_keyword']+" in "+self.filename+"!")
            self._log.writeLog(__name__, "Unable to find keyword "+self._keywords['exptime_keyword']+" in "+self.filename+"!", type=fatboyLog.WARNING)
        ##Set obs type if it was not specified in XML file
        if (self.getObsType() is None):
            try:
                self.setType(self.getHeaderValue('obstype_keyword'))
            except Exception:
                print("fatboyDataUnit::initialize> Warning: Unable to find keyword "+self._keywords['obstype_keyword']+" in "+self.filename+"!  Assuming object!")
                self._log.writeLog(__name__, "Unable to find keyword "+self._keywords['obstype_keyword']+" in "+self.filename+"!  Assuming object!", type=fatboyLog.WARNING)
                self.setType("object", False)

        #MEF should be set in readHeader now.
        if (self._mef < 0):
            #Header failed, now use data
            temp = pyfits.open(self.filename)
            for j in range(len(temp)):
                if (temp[j].data is not None):
                    self._mef = j
                    break
            temp.close()

        #Find shape
        if (self._shape is None):
            try:
                #Try header first
                if ('NAXIS1' in self._header and 'NAXIS2' in self._header):
                    if (self._header['NAXIS1'] > 1 and self._header['NAXIS2'] > 1):
                        if ('NAXIS3' in self._header):
                            #shape = (z, y, x) = (NAXIS3, NAXIS2, NAXIS1)
                            self._shape = (self._header['NAXIS3'], self._header['NAXIS2'], self._header['NAXIS1'])
                        else:
                            #shape = (y, x) = (NAXIS2, NAXIS1)
                            self._shape = (self._header['NAXIS2'], self._header['NAXIS1'])
                if (shape is None):
                    #Not found in header, now use data
                    temp = pyfits.open(self.filename)
                    self._shape = temp[self._mef].data.shape
                    temp.close()
            except Exception:
                self.disable()
                #File has been disabled due to bad data
                print("fatboyDataUnit::initialize> WARNING: File "+self.filename+" is misformatted.  Skipping!")
                self._log.writeLog(__name__, " File "+self.filename+" is misformatted.  Skipping!", type=fatboyLog.WARNING)
    #end initialize

    #Convenience method
    def min(self):
        return self.getData().min()
    #end min

    #Convenience method
    def max(self):
        return self.getData().max()
    #end max

    ## Read entire FITS header and store in memory.
    # Also determnine extension (mef) and data shape
    def readHeader(self):
        t = time.time()
        temp = pyfits.open(self.filename)
        oldheader = self._header
        self._header = temp[0].header
        if ('NAXIS' in self._header and self._header['NAXIS'] > 1):
            #Data is in primary extension
            if (self._mef is None):
                self._mef = 0
            if (self._mef > 0 and len(temp) == 1):
                #perhaps this is a fatboyCalib or fatboySpecCalib reading a default_x file
                #and there is only primary extension on disk where data has extension > 0
                self._mef = 0
        if ('NAXIS' in self._header and self._header['NAXIS'] == 0):
            #Data is in one or more image extensions
            if (self._mef is None):
                #Find first extension with data
                for j in range(len(temp)):
                    if ('XTENSION' in temp[j].header and temp[j].header['XTENSION'] == 'BINTABLE'):
                        #This is a binary table extension, skip it and continue looking for data extension
                        continue
                    if ('NAXIS' in temp[j].header and temp[j].header['NAXIS'] > 0):
                        updateHeaderEntry(self._header, 'NAXIS',  temp[j].header['NAXIS'])
                        if ('NAXIS1' in temp[j].header):
                            updateHeaderEntry(self._header, 'NAXIS1',  temp[j].header['NAXIS1'])
                        if ('NAXIS2' in temp[j].header):
                            updateHeaderEntry(self._header, 'NAXIS2',  temp[j].header['NAXIS2'])
                        if ('NAXIS3' in temp[j].header):
                            updateHeaderEntry(self._header, 'NAXIS3',  temp[j].header['NAXIS3'])
                        self._mef = j
                        break
            elif (self._mef > 0):
                #Find specified extension
                if (len(temp) > self._mef):
                    if ('NAXIS' in temp[self._mef].header and temp[self._mef].header['NAXIS'] > 0):
                        updateHeaderEntry(self._header, 'NAXIS',  temp[self._mef].header['NAXIS'])
                        if ('NAXIS1' in temp[self._mef].header):
                            updateHeaderEntry(self._header, 'NAXIS1',  temp[self._mef].header['NAXIS1'])
                        if ('NAXIS2' in temp[self._mef].header):
                            updateHeaderEntry(self._header, 'NAXIS2',  temp[self._mef].header['NAXIS2'])
                        if ('NAXIS3' in temp[self._mef].header):
                            updateHeaderEntry(self._header, 'NAXIS3',  temp[self._mef].header['NAXIS3'])
        temp.close()

        ##First make sure keywords exist.  Set defaults here if they haven't been set by fatboyDatabase
        if (len(self._keywords) == 0):
            #FITS Keywords
            self._keywords.setdefault('date_keyword',['DATE', 'DATE-OBS'])
            self._keywords.setdefault('dec_keyword',['DECOFFSE', 'DEC', 'TELDEC'])
            self._keywords.setdefault('exptime_keyword',['EXPTIME', 'EXP_TIME', 'EXPCOADD'])
            self._keywords.setdefault('filter_keyword',['FILTER', 'FILTNAME'])
            self._keywords.setdefault('gain_keyword',['GAIN', 'GAIN_1', 'EGAIN'])
            self._keywords.setdefault('nreads_keyword',['NREADS', 'LNRS', 'FSAMPLE', 'NUMFRAME'])
            self._keywords.setdefault('obstype_keyword',['OBSTYPE', 'OBS_TYPE', 'IMAGETYP'])
            self._keywords.setdefault('ra_keyword',['RAOFFSET', 'RA', 'TELRA'])
            self._keywords.setdefault('relative_offset_arcsec','no')
            self._keywords.setdefault('ut_keyword',['UT', 'UTC', 'NOCUTC'])

        #Update values in oldheader
        for key in oldheader:
            self._header[key] = oldheader[key]

        ##Look at _keywords that are lists and find which applies if any
        for key in list(self._keywords):
            if (isinstance(self._keywords[key], list)):
                for value in self._keywords[key]:
                    if (value in self._header):
                        #Correct keyword found.  Break after first match.
                        self._keywords[key] = value
                        break
        ##Loop over again and find any keywords that are not found at all
        #set them to first value
        for key in list(self._keywords):
            if (isinstance(self._keywords[key], list)):
                self._keywords[key] = self._keywords[key][0]

        #Use helper method to set properties from header
        self.setPropertiesFromHeader()

        if (self._fdb is not None):
            self._fdb.totalReadHeaderTime += (time.time()-t)
    #end readHeader

    ## Reformat Flamingos-2 style (1,2048,2048) data to (2048,2048)
    def reformatData(self):
        if (self._data is None):
            #No need to read data in just for this step.  This will be called from getData on first read from disk anyway
            return
        if (len(self._shape) > 2 and self._shape[0] == 1):
            self.setHistory('origShape', self._shape)
            try:
                #Update data and shape
                self._data = self._data[0,:,:]
                self._shape = self._data.shape
            except Exception as ex:
                print("fatboyDataUnit::reformatData> Error reformatting data in "+self.filename+": "+str(ex))
                self._log.writeLog(__name__, "Error reformatting data in "+self.filename+": "+str(ex), type=fatboyLog.ERROR)
                self.disable() #disable this FDU
    #end reformatData

    ## Remove a header keyword
    def removeHeaderKeyword(self, key):
        if (key in self._header):
            return self._header.pop(key)
        return False
    #end removeHeaderKeyword

    ## Remove a property
    def removeProperty(self, key):
        if (key in self._properties):
            if (isinstance(self._properties[key], str) and self._properties[key] == "memory_managed"):
                #data tag that has been backed up to disk for memory management purposes
                infile = "temp-fatboy/property_"+key+"_"+self.getFullId()
                if (os.access(infile, os.F_OK)):
                    os.unlink(infile)
            del self._properties[key]
    #end removeProperty

    ## Renormailze data ... used particularly for applying BPM to flats
    def renormalize(self, bpm=None):
        #Has a median section defined but no bad pixel mask
        if (bpm is None and self.hasProperty("median_section") and self.hasProperty("median_section_indices")):
            #if (self.getMedian(tag="median_section") == 1):
            if (self.getMedian() == 1):
                #already normalized
                return True
            #Normalize by median of nonzero pixels
            mfmed = self.getMedian(tag="median_section")
            print("fatboyDataUnit::renormalize> Using median section "+str(self.getProperty("median_section_indices"))+"; median="+str(mfmed))
            if (self._log is not None):
                self._log.writeLog(__name__, "Using median section "+str(self.getProperty("median_section_indices"))+"; median="+str(mfmed))
            self.updateData(self.getData()/mfmed)
            #Normalize median_section as well!
            self.tagDataAs("median_section", self.getData(tag="median_section")/mfmed)
            #set history
            self.setHistory('renormalized', mfmed)
            #update median value
            self.setMedian(1)
            return True

        #Normal default case
        if (bpm is None):
            if (self.getMedian() == 1):
                #already normalized
                return True
            #Normalize by median of nonzero pixels
            self.updateData(self.getData()/self.getMedian())
            #set history
            self.setHistory('renormalized', self.getMedian())
            #update median value
            self.setMedian(1)
            return True

        #bpm exists, median_section may or may not
        if (bpm.shape != self.getData().shape):
            #Error
            print("fatboyDataUnit::renormalize> Error: bad pixel mask shape "+str(bpm.shape)+" different from data "+str(self.getData().shape))
            if (self._log is not None):
                self._log.writeLog(__name__, "bad pixel mask shape "+str(bpm.shape)+" different from data "+str(self.getData().shape), type=fatboyLog.ERROR)
            return False
        #Find median of good pixels but do not apply mask to data itself!
        if (self.hasProperty("median_section_indices") and self.hasProperty("median_section")):
            #Apply median section tagged above to bpm
            section = self.getProperty("median_section_indices")
            bpm = bpm[section[0][0]:section[0][1], section[1][0]:section[1][1]]
            if (self._gpumode):
                mfmed = gpu_arraymedian(self.getData(tag="median_section")*(1-bpm), nonzero=True, kernel=self._fdb.getParam('median_kernel'))
            else:
                mfmed = gpu_arraymedian(self.getData(tag="median_section")*(1-bpm), nonzero=True, kernel=fatboyclib.median)
            print("fatboyDataUnit::renormalize> Using BPM and median section "+str(self.getProperty("median_section_indices"))+"; median="+str(mfmed))
            if (self._log is not None):
                self._log.writeLog(__name__, "Using BPM and median section "+str(self.getProperty("median_section_indices"))+"; median="+str(mfmed))
        else:
            #Default case
            if (self._gpumode):
                mfmed = gpu_arraymedian(self.getData()*(1-bpm), nonzero=True, kernel=self._fdb.getParam('median_kernel'))
            else:
                mfmed = gpu_arraymedian(self.getData()*(1-bpm), nonzero=True, kernel=fatboyclib.median)
        #Do NOT update data - just set history with normalization value
        self.setHistory('renormalized_bpm', mfmed)
        return True
    #end renormalize

    ## set database callback
    def setDatabaseCallback(self, fdb):
        self._fdb = fdb
    #end setDatabaseCallback

    ## Set GPU mode
    def setGPUMode(self, value):
        self._gpumode = value
    #end setGPUMode

    ## Add to history
    def setHistory(self, key, value):
        self._history[key] = value
    #end setHistory

    ## Set the identifier for this data unit
    def setIdentifier(self, groupType, fileprefix, sindex=None, keyword=None):
        fileprefix = str(fileprefix[fileprefix.rfind('/')+1:]) #convert from unicode to str if necessary!
        if (groupType == 'manual'):
            self._id = fileprefix
        elif (groupType == 'keyword' and keyword is not None and os.access(self.filename, OS.F_OK)):
            if (keyword in self._header):
                self._id = fileprefix+'_'+str(self._header[keyword]).replace(' ','_')
            else:
                self._id = fileprefix
        else:
            self._id = fileprefix
        self._index = sindex
        if (self.section >= 0):
            self._id += "S"+str(self.section)
        if (len(self._id) > 32):
            #Limit length of identifier to 32 chars
            self._id = self._id[-32:]
            self._id = self._id[self._id.index('_')+1:]
            while (self._id[0] == '.' or self._id[0] == '_'):
                self._id = self._id[1:]
        self._identFull = self._id+'.'+self._index+'.fits'
        self._identFull = self._identFull.replace('..','.') #for case of blank index in calibs
    #end setIdentifier

    ## This method sets/updates a FITS keyword.
    def setKeyword(self, index, keyword, value=None):
        if (isinstance(keyword, str) and keyword.endswith("_keyword") and index in self._keywords and value is not None):
            keyword = self._keywords[index]
            if (isinstance(keyword, list)):
                keyword = keyword[0]
        self._keywords[index] = keyword
        #No longer write out to disk now.
        if (value is not None):
            self._header[keyword] = value
    #end setKeyword

    ## set a mask to this data
    def setMask(self, mask):
        self._mask = mask
        self._maskedMedian = None
    #end setMask

    ## set the median value
    def setMedian(self, medVal):
        self._medVal = medVal
    #end setMedian

    ## set the MEF extension
    def setMEF(self, mef):
        self._mef = mef
    #end setMEF

    ## set properties from the header
    def setPropertiesFromHeader(self):
        #Flat method
        if (self.hasHeaderValue("FLATMTHD")):
            self.setProperty("flat_method", self.getHeaderValue("FLATMTHD"))
    #end setPropertiesFromHeader

    ## set a property
    def setProperty(self, key, value):
        self._properties[key] = value
    #end setProperty

    ## set relative offset arcsec boolean
    def setRelOffset(self, value):
        self.relOffset = value
    #end setRelOffset

    ## set shape
    def setShape(self, shape):
        self._shape = shape
    #end setShape

    ## set suffix boolean
    def setSuffix(self, suffix):
        self._suffix = suffix
    #end setSuffix

    ## Set _tag
    def setTag(self, tag, subtag=False):
        #Back-modified to allow subtags without changing hundreds of calls
        #Support for passing a 2-element list for use in fatboyCalib by assinging
        #both tag and subtag of a given FDU
        if (isinstance(tag, list)):
            self._tag = tag[0]
            if (len(tag) > 1):
                self._subtag = tag[1]
        else:
            if (subtag):
                self._subtag = tag
            else:
                self._tag = tag
    #end setTag

    ## This method sets the obstype for this FDU (dark/flat/sky/object)
    def setType(self, obstype, apply=True):
        self.isDark = False
        self.isFlat = False
        self.isObject = False
        self.isSky = False
        self.isSkyFlat = False
        self.isMasterCalib = False

        if (isinstance(obstype, int)):
            #enum val given
            self._obsTypeVal = obstype
            if (obstype == self.FDU_TYPE_OBJECT):
                self.isObject = True
                self.obstype = "object"
            elif (obstype == self.FDU_TYPE_DARK):
                self.isDark = True
                self.obstype = "dark"
            elif (obstype == self.FDU_TYPE_FLAT):
                self.isFlat = True
                self.obstype = "flat"
            elif (obstype == self.FDU_TYPE_SKY_FLAT):
                self.isFlat = True
                self.isSkyFlat = True
                self.isObject = True
                self.obstype = "object"
            elif (obstype == self.FDU_TYPE_SKY):
                self.isSky = True
                self.obstype = "sky"
            elif (obstype == self.FDU_TYPE_MASTER_CALIB):
                self.isMasterCalib = True
                self.obstype = "master_calib"
            elif (obstype == self.FDU_TYPE_BAD_PIXEL_MASK):
                self.obstype = "bad_pixel_mask"
            elif (obstype == self.FDU_TYPE_ARCLAMP):
                self.obstype = "arclamp"
            elif (obstype == self.FDU_TYPE_BIAS):
                self.obstype = "bias"
        else:
            #String given
            obstype = obstype.lower()
            self.obstype = obstype
            if (obstype.find("master") != -1):
                self.isMasterCalib = True
                self._obsTypeVal = self.FDU_TYPE_MASTER_CALIB
            elif (obstype.find("dark") != -1):
                self.isDark = True
                self._obsTypeVal = self.FDU_TYPE_DARK
            elif (obstype.find("flat") != -1):
                self.isFlat = True
                self._obsTypeVal = self.FDU_TYPE_FLAT
            elif (obstype == "sky"):
                self.isSky = True
                self._obsTypeVal = self.FDU_TYPE_SKY
            elif (obstype == "bad_pixel_mask"):
                self._obsTypeVal = self.FDU_TYPE_BAD_PIXEL_MASK
            elif (obstype.find("lamp") != -1 or obstype.find("arc") != -1):
                self._obsTypeVal = self.FDU_TYPE_ARCLAMP
            elif (obstype.find("bias") != -1):
                self._obsTypeVal = self.FDU_TYPE_BIAS
            else:
                self.isObject = True
                self._obsTypeVal = self.FDU_TYPE_OBJECT
        if (apply):
            #Only need to apply if it doesn't match header
            #Don't write out to disk anymore
            if (not self.hasHeaderValue('obstype_keyword') or self.getHeaderValue('obstype_keyword').lower() != self.obstype):
                self._header[self._keywords['obstype_keyword']] = self.obstype
    #end setType

    #Tag an array as data saved at a particular step
    def tagDataAs(self, tagname, data=None):
        if (data is None):
            #tag current data
            self.setProperty(tagname, self.getData().copy())
            #and median if applicable
            if (self._medVal is not None):
                self.setProperty(tagname+"_median", self._medVal)
        else:
            #tag data passed to function
            if (not data.dtype.isnative):
                #Byteswap
                data = data.byteswap()
                data = data.newbyteorder('<')
                data.dtype.newbyteorder('<')
            self.setProperty(tagname, data)
    #end tagDataAs

    #Convert to an HDUList object
    def toHDUList(self, tag=None, headerExt=None, prefix=None):
        #Always remove BZERO
        self.removeHeaderKeyword('BZERO')
        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        if (tag is not None):
            if (isinstance(tag, list)):
                #Write a MEF with an extension for each tag listed
                if (self._header is not None):
                    updateHeader(hdu.header, self._header)
                hdulist.append(hdu) #append blank data Primary HDU
                if (prefix is not None):
                    updateHeaderEntry(hdulist[0].header, 'FILENAME', prefix+"_"+self.getFullId())
                for j in range(len(tag)):
                    imhdu = pyfits.ImageHDU()
                    imhdu.data = self.getData(tag=tag[j])
                    if (headerExt is not None and isinstance(headerExt, list)):
                        if (len(headerExt) > j):
                            updateHeader(imhdu.header, headerExt[j])
                    hdulist.append(imhdu)
                #return hdulist here
                return hdulist
            else:
                hdu.data = self.getData(tag=tag)
        else:
            hdu.data = self._data
        #Single extension FITS file here
        if (self._header is not None):
            updateHeader(hdu.header, self._header)
        if (headerExt is not None):
            updateHeader(hdu.header, headerExt)
        hdulist.append(hdu)
        if (prefix is not None):
            updateHeaderEntry(hdulist[0].header, 'FILENAME', prefix+"_"+self.getFullId())
        elif (tag is not None):
            updateHeaderEntry(hdulist[0].header, 'FILENAME', tag+"_"+self.getFullId())
        else:
            updateHeaderEntry(hdulist[0].header, 'FILENAME', self.getFullId())
        return hdulist
    #end toHDUList

    ## Update data
    def updateData(self, data):
        self._data = data
        #Reset median value to None
        self._medVal = None
        #Update shape!
        self._shape = data.shape
    #end updateData

    ## update list of previous filenames
    def updateFilenames(self):
        if (not self.hasHistory('origFilename')):
            #First time this method is called.  Current filename is still original filename
            #Create empty list of previous filenames which will later be appended
            self.setHistory('origFilename', self.filename)
            self.setHistory('previousFilenames', [])
        else:
            #Append current filename to list of previous filenames as long as it does not match original filename
            if (self.filename != self.getHistory('origFilename')):
                self.getHistory('previousFilenames').append(self.filename)
    #end updateFilenames

    ## update data and header history from a file
    def updateFrom(self, updateFile, tag=None, headerTag=None, pname=None):
        t = time.time()
        image = pyfits.open(updateFile)
        if (tag is not None):
            if (isinstance(tag, list)):
                #Update multiple properties from a MEF
                for j in range(len(tag)):
                    if (len(image) > j+1):
                        #Assume empty data in primary extension
                        if (tag[j] in image[j+1].header):
                            #tagname is written to header, tag as this name
                            self.tagDataAs(image[j+1].header[tag[j]], data=image[j+1].data)
                        else:
                            #Tag in list order
                            self.tagDataAs(tag[j], data=image[j+1].data)
                    else:
                        print("fatboyDataUnit::updateFrom> Warning: "+updateFile+" does not contain extension "+str(j+1)+"!")
                        if (self._log is not None):
                            self._log.writeLog(__name__, updateFile+" does not contain extension "+str(j+1)+"!", type=fatboyLog.WARNING)
                if (headerTag is not None):
                    self.setProperty(headerTag, image[0].header)
                return
            #update data tag.  tagDataAs() will byteswap
            self.tagDataAs(tag, data=image[self._mef].data)
            image.close()
            if (headerTag is not None):
                self.setProperty(headerTag, image[0].header)
            return
        if (self._fdb is not None and self._data is None):
            self._fdb.totalReadDataTime += (time.time()-t)
            self._fdb.checkMemoryManagement(self) #check memory status
        self.updateData(image[self._mef].data)
        if (not self._data.dtype.isnative):
            #Byteswap
            self._data = self._data.byteswap()
            self._data = self._data.newbyteorder('<')
            self._data.dtype.newbyteorder('<')
        if (headerTag is not None):
            self.setProperty(headerTag, image[0].header)
        else:
            self.updateHeader(image[0].header)
        image.close()
    #end updateFrom

    ## update header with dict of new header keywords
    def updateHeader(self, headerExt):
        #extension header values
        updateHeader(self._header, headerExt)
    #end updateHeader

    #Used for memory management purposes to save tagged data to disk and free from memory
    def writeAndForgetTaggedData(self):
        #Check properties dict for arrays of tagged data
        for key in list(self._properties):
            if (isinstance(self._properties[key], ndarray)):
                #Keep smaller arrays in memory, only need to free up memory from large arrays
                if (self._properties[key].size > 512*512 and self._properties[key].dtype != bool):
                    outfile = "temp-fatboy/property_"+key+"_"+self.getFullId()
                    if (os.access(outfile, os.F_OK)):
                        #Property could have changed -- e.g. cleanFrame 7/20/21
                        os.unlink(outfile)
                    write_fits_file(outfile, self._properties[key])
                    del self._properties[key] #del to free memory
                    self._properties[key] = "memory_managed" #set to "memory_managed" so hasProperty still returns True and getProperty recalls data
    #end writeAndForgetTaggedData

    #Write a list of properties to a multi-extension FITS file
    def writePropertiesToMEF(self, outfile, tag=None, headerExt=None):
        hdulist = pyfits.HDUList()
        hdu = pyfits.PrimaryHDU()
        if (isinstance(tag, list)):
            #Write a MEF with an extension for each tag listed
            if (self._header is not None):
                updateHeader(hdu.header, self._header)
            hdulist.append(hdu) #append blank data Primary HDU
            for j in range(len(tag)):
                imhdu = pyfits.ImageHDU()
                imhdu.data = self.getData(tag=tag[j])
                if (headerExt is not None and isinstance(headerExt, list)):
                    if (len(headerExt) > j):
                        updateHeader(imhdu.header, headerExt[j])
                hdulist.append(imhdu)
            #write out here
            updateHeaderEntry(hdulist[0].header, 'FILENAME', outfile)
            try:
                #delete if outfile already exists
                if (os.access(outfile, os.F_OK)):
                    os.unlink(outfile)
                hdulist.writeto(outfile, output_verify='silentfix')
                hdulist.close()
            except Exception as ex:
                print("fatboyDataUnit::writePropertiesToMEF> Error writing file "+outfile+": "+str(ex))
                if (self._log is not None):
                    self._log.writeLog(__name__, "Error writing file "+outfile+": "+str(ex), type=fatboyLog.ERROR)
                return False
            return True
        else:
            print("fatboyDataUnit::writePropertiesToMEF> ERROR: tag must be a list of tagnames.")
            if (self._log is not None):
                self._log.writeLog(__name__, "tag must be a list of tagnames.", type=fatboyLog.ERROR)
        return False
    ## end writePropertiesToMEF

    ## Write output
    def writeTo(self, outfile, tag=None, headerExt=None):
        self.removeHeaderKeyword('BZERO')
        filename = self.filename
        if (not os.access(filename, os.F_OK) and self.hasHistory('sourceFilename')):
            filename = self.getHistory('sourceFilename')
        if (not os.access(filename, os.F_OK)):
            #source file does not exist -- can happen in DFP
            dt = self.getData(tag=tag).dtype
            if (self.getObsType(True) == self.FDU_TYPE_BAD_PIXEL_MASK or self.getData(tag=tag).dtype == 'bool'):
                #Convert to 8 bit unsigned int for bad pixel mask since FITS doesn't support bool
                dt = uint8
            updateHeaderEntry(self._header, 'FILENAME', outfile)
            if (outfile.rfind('/') != -1):
                updateHeaderEntry(self._header, 'FILENAME', outfile[outfile.rfind('/')+1:])
            return write_fits_file(outfile, self.getData(tag=tag), dtype=dt, header=self._header, headerExt=headerExt, log=self._log)
        try:
            image = pyfits.open(filename)
            if (self.getObsType(True) == self.FDU_TYPE_BAD_PIXEL_MASK):
                #Convert to 8 bit unsigned int for bad pixel mask since FITS doesn't support bool
                image[self._mef].data = self.getData(tag=tag).astype(uint8)
            elif (self.getData(tag=tag).dtype == 'bool'):
                #Convert to 8 bit unsigned int for bad pixel mask since FITS doesn't support bool
                image[self._mef].data = self.getData(tag=tag).astype(uint8)
            elif (self.getData(tag=tag).dtype == 'int64'):
                #Convert to 32 bit int before saving
                image[self._mef].data = self.getData(tag=tag).astype(int32)
            elif (self.getData(tag=tag).dtype == 'float64'):
                #Convert to 32 bit int before saving
                image[self._mef].data = self.getData(tag=tag).astype(float32)
            else:
                image[self._mef].data = self.getData(tag=tag)
            #update header
            updateHeader(image[0].header, self._header)
            updateHeaderEntry(image[0].header, 'FILENAME', outfile)
            if (outfile.rfind('/') != -1):
                updateHeaderEntry(image[0].header, 'FILENAME', outfile[outfile.rfind('/')+1:])
            if (headerExt is not None):
                #Add keywords from extended header
                updateHeader(image[0].header, headerExt)
            #Get rid of extraneous extensions in data like CIRCE/Newfirm
            prepMefForWriting(image, self._mef)
            #delete if outfile already exists
            if (os.access(outfile, os.F_OK)):
                os.unlink(outfile)
            image.writeto(outfile, output_verify='silentfix')
            image.close()
        except Exception as ex:
            print("fatboyDataUnit::writeTo> Error writing file "+outfile+": "+str(ex))
            if (self._log is not None):
                self._log.writeLog(__name__, "Error writing file "+outfile+": "+str(ex), type=fatboyLog.ERROR)
            return False
        return True
    #end writeTo

    #Back up to disk for memory purposes
    def writeToAndForget(self, outfile):
        if (not self.inUse):
            return
        if (os.access(outfile, os.F_OK)):
            os.unlink(outfile)
        self.writeTo(outfile)
        if (self.filename != "temp-fatboy/current_"+self.getFullId()):
            self.setProperty('origFilename', self.filename)
        self.setProperty('memory_managed', True)
        self.filename = outfile
        self.forgetData()
    #end writeAndForget

from .fatboyCalib import *
