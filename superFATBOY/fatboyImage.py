## @package superFATBOY

from .fatboyDataUnit import *

class fatboyImage(fatboyDataUnit):
    _name = "fatboyImage"

    ## The constructor.
    def __init__(self, filename, log=None, tag=None):
        self.filename = str(filename)
        self.findIdentifier()
        if (log is not None):
            self._log = log
        self._tag = tag
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

    ## Initialize reads certain values from the header and determines the shape.  It also finds the mef extension if not previously done.
    def initialize(self):
        ##Look at _keywords that are comma separated and turn them into lists (e.g. FILTER1, FILTER2)
        for key in self._keywords:
            if (not isinstance(self._keywords[key], list)):
                if (self._keywords[key].count(",") > 0):
                    self._keywords[key] = self._keywords[key].replace(' ','').split(',')
        try:
            self.exptime = self.getHeaderValue('exptime_keyword')
        except Exception:
            print("fatboyImage::initialize> Warning: Unable to find keyword "+str(self._keywords['exptime_keyword'])+" in "+self.filename+"!")
            self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['exptime_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        ##Set obs type if it was not specified in XML file
        if (self.getObsType() is None):
            try:
                self.setType(self.getHeaderValue('obstype_keyword'))
            except Exception:
                print("fatboyImage::initialize> Warning: Unable to find keyword "+str(self._keywords['obstype_keyword'])+" in "+self.filename+"!  Assuming object!")
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['obstype_keyword'])+" in "+self.filename+"!  Assuming object!", type=fatboyLog.WARNING)
                self.setType("object", False)
        try:
            self.nreads = self.getHeaderValue('nreads_keyword')
        except Exception:
            print("fatboyImage::initialize> Warning: Unable to find keyword "+str(self._keywords['nreads_keyword'])+" in "+self.filename+"!")
            self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['nreads_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        try:
            self.gain = self.getHeaderValue('gain_keyword')
        except Exception:
            print("fatboyImage::initialize> Warning: Unable to find keyword "+str(self._keywords['gain_keyword'])+" in "+self.filename+"!")
            self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['gain_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        if ('ncoadd_keyword' in self._keywords):
            try:
                self.coadds = self.getHeaderValue('ncoadd_keyword')
            except Exception:
                print("fatboyImage::initialize> Warning: Unable to find keyword "+str(self._keywords['ncoadd_keyword'])+" in "+self.filename+"!")
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['ncoadd_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)

        if (isinstance(self._keywords['filter_keyword'], list)):
            #FILTER may be a list of keywords or just one
            for key in self._keywords['filter_keyword']:
                if (key in self._header):
                    if (isValidFilter(key)):
                        self.filter = str(self.getHeaderValue(key)).strip()
                        break
            if (self.filter is None):
                print("fatboyImage::initialize> Warning: Unable to find any filter keyword: "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!")
                self._log.writeLog(__name__, "Unable to find keyword any filter keyword: "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
        else:
            try:
                self.filter = str(self.getHeaderValue('filter_keyword')).strip()
            except Exception:
                print("fatboyImage::initialize> Warning: Unable to find keyword "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!")
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['filter_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)

        if (self.isObject or self.isSky):
            #set RA and DEC
            try:
                self.dec = getRADec(self.getHeaderValue('dec_keyword'), log=self._log, rel=self.relOffset, dec=True, file=self.filename)
            except Exception:
                print("fatboyImage::initialize> ERROR: Unable to find keyword "+str(self._keywords['dec_keyword'])+" in "+self.filename+"! Disabling file!")
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['dec_keyword'])+" in "+self.filename+"! Disabling file!", type=fatboyLog.ERROR)
                self.disable()
            try:
                self.ra = getRADec(self.getHeaderValue('ra_keyword'), log=self._log, rel=self.relOffset, file=self.filename)*15
            except Exception:
                print("fatboyImage::initialize> ERROR: Unable to find keyword "+str(self._keywords['ra_keyword'])+" in "+self.filename+"! Disabling file!")
                self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['ra_keyword'])+" in "+self.filename+"!", type=fatboyLog.ERROR)
                self.disable()

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
                print("fatboyImage::initialize> WARNING: File "+self.filename+" is misformatted.  Skipping!")
                self._log.writeLog(__name__, " File "+self.filename+" is misformatted.  Skipping!", type=fatboyLog.WARNING)
    #end initialize
