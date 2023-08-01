## @package superFATBOY.datatypeExtensions
from superFATBOY.fatboyDataUnit import *
import superFATBOY.datatypeExtensions

class fatboySpectrum(fatboyDataUnit):
    ##static type def variables
    FDU_TYPE_STANDARD = 10
    FDU_TYPE_CONTINUUM_SOURCE = 11

    FDU_TYPE_MOS = 100
    FDU_TYPE_LONGSLIT = 101
    FDU_TYPE_IFU = 102

    DISPERSION_HORIZONTAL = 0
    DISPERSION_VERTICAL = 1

    _name = "fatboySpectrum"
    _specmode = FDU_TYPE_MOS #spectral mode
    _slitmask_properties = None #dict to hold slitmask properties
    _slitmask = None

    #boolean values
    isStandard = False

    #parameters
    dispersion = DISPERSION_HORIZONTAL
    gain = 1.0
    grism = None #grism
    pixscale = 1.0
    readnoise = 0.0

    ## The constructor.
    def __init__(self, filename, log=None, tag=None):
        self.filename = str(filename)
        self.findIdentifier()
        if (log is not None):
            self._log = log
            self._verbosity = log._verbosity
        self._tag = tag
        #Initialize dicts
        self._header = dict()
        self._history = dict()
        self._keywords = dict()
        self._properties = dict()
        self._slitmask_properties = dict()
        #Slitmask - initialize
        self._slitmask = None
        #Initialize lists
        self._objectTags = []
        self._processHistory = []
        #Add spectrum specific keywords
        self.addKeywords()
        if (self._verbosity != fatboyLog.BRIEF):
            print(self._name+": "+self.filename)
            if (log is not None):
                self._log.writeLog(__name__, self._name+": "+self.filename, printCaller=False, tabLevel=1)

    ## Adds spectrum specific keywords
    def addKeywords(self):
        self._keywords.setdefault('grism_keyword', 'GRISM')
        self._keywords.setdefault('object_keyword', 'OBJECT')
        self._keywords.setdefault('pixscale_keyword', 'PIXSCALE')
        self._keywords.setdefault('readnoise_keyword', ['RDNOIS', 'RDNOIS_1'])

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
    #end addKeywords

    def printAllSlitmasks(self):
        print("PRINTING SLITMASKS")
        for key in self._properties:
            if (key.find("slitmask") != -1):
                slitmask = self.getProperty(key)
                print("PROP", key, slitmask.getFullId(), slitmask.getShape())
                print("\t", slitmask._properties)
        for fdu in self._fdb._calibs:
            if (fdu.getObsType() == "slitmask"):
                print("FDU", fdu.getFullId(), fdu.getShape())
                print("\t", fdu._properties)

    #Find and return the proper slitmask for this fatboySpectrum
    #This should always return a superFATBOY.datatypeExtensions.fatboySpecCalib.fatboySpecCalib (which if a property will be set from setSlitmask
    def getSlitmask(self, pname=None, shape=None, properties=None, headerVals=None, tagname=None, ignoreShape=False):
        #tagname could be e.g. "resampled_slitmask" from wavelengthCalibrated
        if (tagname is None):
            tagname = "slitmask"
        if (shape is None and not ignoreShape):
            shape = self.getShape()
        #combine any properties passed with internal slitmask_properties
        #Could be e.g. "resampled", "SlitletsIdentified"
        if (self._slitmask is not None):
            if (properties is None):
                properties = self._slitmask_properties
            else:
                properties.update(self._slitmask_properties)
        elif (properties is None):
            properties = dict()
        #print "GET SM", shape, tagname, properties
        #Check FDU tagged property first
        if (self.hasProperty(tagname)):
            slitmask = self.getProperty(tagname)
            #if (isinstance(slitmask, superFATBOY.datatypeExtensions.fatboySpecCalib.fatboySpecCalib) and (slitmask.getShape() == shape or ignoreShape)):
                #print "IS MATCH ", self.getFullId(), slitmask.getFullId(), tagname, slitmask
        if (self.hasProperty(tagname)):
            slitmask = self.getProperty(tagname)
            if (isinstance(slitmask, superFATBOY.datatypeExtensions.fatboySpecCalib.fatboySpecCalib) and (slitmask.getShape() == shape or ignoreShape)):
                isMatch = True
                for key in properties:
                    if (not slitmask.hasProperty(key)):
                        isMatch = False
                        break
                    if (properties[key] != slitmask.getProperty(key)):
                        isMatch = False
                        break
                if (isMatch):
                    return slitmask
        #2) Check for an already created slitmask matching specmode/filter/grism and TAGGED for this object
        slitmask = self._fdb.getTaggedMasterCalib(pname=pname, ident=self._id, obstype="slitmask", filter=self.filter, section=self.section, shape=shape, properties=properties, headerVals=headerVals)
        if (slitmask is None):
            #3) Look for a slitmask with matching filter/grism/specmode but NOT ident
            slitmask = self._fdb.getMasterCalib(filter=self.filter, section=self.section, obstype="slitmask", shape=shape, properties=properties, headerVals=headerVals, tag=self.getTag())
            #if (slitmask is not None):
                #print "MATCH2 - ",self.getFullId(), slitmask.getFullId(), tagname, slitmask
        #Return slitmask -- will be None if not found
        #if (slitmask is None):
            #print "NO MATCH"
        return slitmask
    #end getSlitmask

    ## Initialize reads certain values from the header and determines the shape.  It also finds the mef extension if not previously done.
    def initialize(self):
        ##First look at properties dict for specmode
        specmode = self.getProperty("specmode")
        if (isinstance(specmode, str)):
            if (specmode == "longslit"):
                self.setProperty("specmode", fatboySpectrum.FDU_TYPE_LONGSLIT)
                self._specmode = fatboySpectrum.FDU_TYPE_LONGSLIT
            elif (specmode == "ifu"):
                self.setProperty("specmode", fatboySpectrum.FDU_TYPE_IFU)
                self._specmode = fatboySpectrum.FDU_TYPE_IFU
            else:
                self.setProperty("specmode", fatboySpectrum.FDU_TYPE_MOS)
                self._specmode = fatboySpectrum.FDU_TYPE_MOS
        elif (isinstance(specmode, int)):
            self._specmode = specmode
        if (not self.hasProperty("specmode")):
            #Default case, no specmode set
            self.setProperty("specmode", self._specmode)

        ##Next look for dispersion
        dispersion = self.getProperty("dispersion")
        if (isinstance(dispersion, str)):
            if (dispersion == "horizontal"):
                self.setProperty("dispersion", fatboySpectrum.DISPERSION_HORIZONTAL)
                self.dispersion = fatboySpectrum.DISPERSION_HORIZONTAL
            elif (dispersion == "vertical"):
                self.setProperty("dispersion", fatboySpectrum.DISPERSION_VERTICAL)
                self.dispersion = fatboySpectrum.DISPERSION_VERTICAL
            else:
                self.setProperty("dispersion", fatboySpectrum.DISPERSION_HORIZONTAL)
                self.dispersion = fatboySpectrum.DISPERSION_HORIZONTAL
        elif (isinstance(dispersion, int)):
            self.dispersion = dispersion
        if (not self.hasProperty("dispersion")):
            #Default case, no disperion set
            self.setProperty("dispersion", self.dispersion)

        ##Look at _keywords that are comma separated and turn them into lists (e.g. FILTER1, FILTER2)
        for key in self._keywords:
            if (not isinstance(self._keywords[key], list)):
                if (self._keywords[key].count(",") > 0):
                    self._keywords[key] = self._keywords[key].replace(' ','').split(',')
        try:
            self.exptime = self.getHeaderValue('exptime_keyword')
        except Exception:
            print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['exptime_keyword'])+" in "+self.filename+"!")
            if (self._log is not None):
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['exptime_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        ##Set obs type if it was not specified in XML file
        if (self.getObsType() is None):
            try:
                self.setType(self.getHeaderValue('obstype_keyword'))
            except Exception:
                print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['obstype_keyword'])+" in "+self.filename+"!  Assuming object!")
                if (self._log is not None):
                    self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['obstype_keyword'])+" in "+self.filename+"!  Assuming object!", type=fatboyLog.WARNING)
                self.setType("object", False)
        try:
            self.nreads = self.getHeaderValue('nreads_keyword')
        except Exception:
            print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['nreads_keyword'])+" in "+self.filename+"!")
            if (self._log is not None):
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['nreads_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        try:
            self.gain = self.getHeaderValue('gain_keyword')
        except Exception:
            print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['gain_keyword'])+" in "+self.filename+"!")
            if (self._log is not None):
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['gain_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        try:
            self.grism = self.getHeaderValue('grism_keyword').strip()
        except Exception:
            print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['grism_keyword'])+" in "+self.filename+"!")
            if (self._log is not None):
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['grism_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        if ('ncoadd_keyword' in self._keywords):
            try:
                self.coadds = self.getHeaderValue('ncoadd_keyword')
            except Exception:
                print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['ncoadd_keyword'])+" in "+self.filename+"!")
                if (self._log is not None):
                    self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['ncoadd_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        try:
            self.pixscale = self.getHeaderValue('pixscale_keyword')
        except Exception:
            print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['pixscale_keyword'])+" in "+self.filename+"!")
            if (self._log is not None):
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['pixscale_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        try:
            self.readnoise = self.getHeaderValue('readnoise_keyword')
        except Exception:
            print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['readnoise_keyword'])+" in "+self.filename+"!")
            if (self._log is not None):
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['readnoise_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)

        if (isinstance(self._keywords['filter_keyword'], list)):
            #FILTER may be a list of keywords or just one
            for key in self._keywords['filter_keyword']:
                if (key in self._header):
                    if (isValidFilter(key)):
                        self.filter = str(self.getHeaderValue(key)).strip()
                        break
            if (self.filter is None):
                print("fatboySpectrum::initialize> Warning: Unable to find any filter keyword: "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!")
                if (self._log is not None):
                    self._log.writeLog(__name__, "Unable to find keyword any filter keyword: "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        else:
            try:
                self.filter = str(self.getHeaderValue('filter_keyword')).strip()
            except Exception:
                print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!")
                if (self._log is not None):
                    self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)

        if (self.isObject or self.isStandard or self.isSky):
            #set RA and DEC
            try:
                self.dec = getRADec(self.getHeaderValue('dec_keyword'), log=self._log, rel=self.relOffset, dec=True, file=self.filename)
            except Exception:
                print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['dec_keyword'])+" in "+self.filename+"!")
                if (self._log is not None):
                    self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['dec_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
            try:
                #RA in degrees
                self.ra = getRADec(self.getHeaderValue('ra_keyword'), log=self._log, rel=self.relOffset, file=self.filename)*15
            except Exception:
                print("fatboySpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['ra_keyword'])+" in "+self.filename+"!")
                if (self._log is not None):
                    self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['ra_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)

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
                print("fatboySpectrum::initialize> WARNING: File "+self.filename+" is misformatted.  Skipping!")
                if (self._log is not None):
                    self._log.writeLog(__name__, " File "+self.filename+" is misformatted.  Skipping!", type=fatboyLog.WARNING)
    #end initialize

    ## Override base class
    ## Renormailze data ... used particularly for applying BPM to flats
    def renormalize(self, slitmask=None, bpm=None):
        if (slitmask is None):
            return fatboyDataUnit.renormalize(self, bpm=bpm)
        #Slitmask is not None here
        nslits = slitmask.getData().max()
        if (bpm is None):
            if (self._gpumode):
                #normalizeMOSFlat will update data and noisemap in FDU
                #normalizeMOSFlat will also add NORMALxx header keywords and renormalized_xx history
                normalizeMOSFlat(self, slitmask.getData(), nslits, log=self._log)
            else:
                #CPU median kernel
                kernel = fatboyclib.median
                #Loop over slitlets in MOS/IFU data and normalize each
                data = self.getData()
                for j in range(nslits):
                    slit = slitmask.getData() == (j+1)
                    medVal = gpu_arraymedian(data[slit], nonzero=True, kernel=kernel)
                    if (medVal == 1):
                        #already normalized
                        continue
                    data[slit] /= medVal
                    key = ''
                    if (j+1 < 10):
                        key += '0'
                    key += str(j+1)
                    updateHeaderEntry(self._header, 'NORMAL'+key, medVal) #Use wrapper function to update header
                    self.setHistory('renormalize_'+key, medVal)
            return True
        if (bpm.shape != self.getData().shape):
            #Error
            print("fatboySpectrum::renormalize> Error: bad pixel mask shape "+str(bpm.shape)+" different from data "+str(self.getData().shape))
            if (self._log is not None):
                self._log.writeLog(__name__, "bad pixel mask shape "+str(bpm.shape)+" different from data "+str(self.getData().shape), type=fatboyLog.ERROR)
            return False
        #Find medians of good pixels in each each slitlet but do not apply mask to data itself!
        medians = []
        if (self._gpumode):
            trans = False
            if (self.dispersion == fatboySpectrum.DISPERSION_VERTICAL):
                trans = True
            medians = gpumedianS(self.getData()*(1-bpm), slitmask.getData(), nslits, nonzero=True, trans=trans)
        else:
            #CPU median kernel
            kernel = fatboyclib.median
            for j in range(nslits):
                slit = slitmask.getData() == (j+1)
                medians.append(gpu_arraymedian(self.getData()[slit]*(1-bpm[slit]), nonzero=True, kernel=kernel))
        #Do NOT update data - just set history with normalization value
        for j in range(len(medians)):
            key = ''
            if (j+1 < 10):
                key += '0'
            key += str(j+1)
            self.setHistory('renormalized_bpm_'+key, medians[j])
        return True
    #end renormalize

    ## Override base class
    ## set properties from the header
    def setPropertiesFromHeader(self):
        #Flat method
        if (self.hasHeaderValue("FLATMTHD")):
            self.setProperty("flat_method", self.getHeaderValue("FLATMTHD"))
        #Lamp method
        if (self.hasHeaderValue("LAMPMTHD")):
            self.setProperty("lamp_method", self.getHeaderValue("LAMPMTHD"))
        #specmode
        if (self.hasHeaderValue("SPECMODE")):
            self.setProperty("specmode", self.getHeaderValue("SPECMODE"))
        #dispersion
        if (self.hasHeaderValue("DISPDIR")):
            self.setProperty("dispersion", self.getHeaderValue("DISPDIR"))
    #end setPropertiesFromHeader

    #Create a superFATBOY.datatypeExtensions.fatboySpecCalib.fatboySpecCalib from data and this FDU and set as a property
    def setSlitmask(self, smdata, pname=None, properties=None, tagname=None):
        if (tagname is None):
            tagname = "slitmask"
        slitmask = superFATBOY.datatypeExtensions.fatboySpecCalib.fatboySpecCalib(pname, "slitmask", self, data=smdata, tagname=tagname+"_"+self._id, log=self._log)
        #print "SET SLITMASK", slitmask, slitmask.getFullId(), tagname, slitmask.getShape()
        #Copy over existing properties
        for key in self._slitmask_properties:
            slitmask.setProperty(key, self._slitmask_properties[key])
        if (properties is not None):
            #properties are e.g. "SlitletsIdentified = True"
            #will be set in both slitmask properties and self._slitmask_properties
            for key in properties:
                self._slitmask_properties[key] = properties[key]
                slitmask.setProperty(key, properties[key])
        self.setProperty(tagname, slitmask)
        return slitmask
    #end setSlitmask

    ## Override base class
    def setType(self, obstype, apply=True):
        fatboyDataUnit.setType(self, obstype, apply)
        if (isinstance(obstype, int)):
            if (obstype == self.FDU_TYPE_STANDARD):
                self.isObject = False
                self.isStandard = True
                self.obstype = "standard"
            elif (obstype == self.FDU_TYPE_CONTINUUM_SOURCE):
                self.isObject = False
                self.isStandard = True
                self.obstype = "continuum_source"
        else:
            #String given
            obstype = obstype.lower()
            self.obstype = obstype
            if (obstype.find("standard") != -1):
                self.isObject = False
                self.isStandard = True
                self._obsTypeVal = self.FDU_TYPE_STANDARD
            elif (obstype.find("continuum_source") != -1):
                self.isObject = False
                self.isStandard = True
                self._obsTypeVal = self.FDU_TYPE_CONTINUUM_SOURCE
        if (apply):
            #Only need to apply if it doesn't match header
            #Don't write out to disk anymore
            if (not self.hasHeaderValue('obstype_keyword') or self.getHeaderValue('obstype_keyword').lower() != self.obstype):
                self._header[self._keywords['obstype_keyword']] = self.obstype
    #end setType

    ## update data and header history from a file
    #Override base class
    def updateFrom(self, updateFile, tag=None, headerTag=None, pname=None):
        if (tag is not None and isinstance(tag, str) and tag.find("slitmask") != -1):
            #This is a slitmask
            image = pyfits.open(updateFile)
            data = image[self._mef].data
            if (not data.dtype.isnative):
                #Byteswap
                data = data.byteswap()
                data = data.newbyteorder('<')
                data.dtype.newbyteorder('<')
            self.setSlitmask(data, pname=pname, tagname=tag)
            image.close()
            return
        elif (self.getObsType() == "slitmask"):
            #This is a slitmask
            image = pyfits.open(updateFile)
            data = image[self._mef].data
            if (not data.dtype.isnative):
                #Byteswap
                data = data.byteswap()
                data = data.newbyteorder('<')
                data.dtype.newbyteorder('<')
            slitmask = self._fdb.addNewSlitmask(self, data, pname)
            image.close()
            return
        fatboyDataUnit.updateFrom(self, updateFile, tag, headerTag, pname)
    #end updateFrom
