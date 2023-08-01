## @package superFATBOY.datatypeExtensions
from superFATBOY.datatypeExtensions.fatboySpectrum import *

class miradasSpectrum(fatboySpectrum):
    ##static type def variables
    MIRADAS_MODE_SOL = 200
    MIRADAS_MODE_SOS = 201
    MIRADAS_MODE_MOS = 202
    MIRADAS_MODE_OTHER = 203

    _name = "miradasSpectrum"
    _specmode = fatboySpectrum.FDU_TYPE_MOS #spectral mode
    _obsmode = MIRADAS_MODE_SOL

    #boolean values
    isStandard = False

    #parameters
    dispersion = fatboySpectrum.DISPERSION_VERTICAL
    gain = 1.0
    grism = None #grism
    pixscale = 1.0
    readnoise = 0.0

    ramp = 1
    firstDataAccess = True

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
        self._keywords.setdefault('airmass_keyword', 'AIRMASS')
        self._keywords.setdefault('altitude_keyword', '??')
        self._keywords.setdefault('grism_keyword', 'GRATNAME')
        self._keywords.setdefault('object_keyword', 'OBJECT')
        self._keywords.setdefault('obsmode_keyword', 'OBSMODE')
        self._keywords.setdefault('pixscale_keyword', 'PIXSCALE')
        self._keywords.setdefault('pressure_keyword', 'AIRPRESS')
        self._keywords.setdefault('readnoise_keyword', ['RDNOIS', 'RDNOIS_1'])
        self._keywords.setdefault('temperature_keyword', 'ATMTEMP')
        self._keywords.setdefault('water_vapor_keyword', '??')


        self._keywords.setdefault('date_keyword', 'DATE-OBS')
        self._keywords.setdefault('dec_keyword', 'DEC')
        self._keywords.setdefault('exptime_keyword', 'EXPTIME')
        self._keywords.setdefault('filter_keyword', 'FILTNAME')
        self._keywords.setdefault('gain_keyword',['GAIN', 'GAIN_1', 'EGAIN'])
        self._keywords.setdefault('nreads_keyword', 'NREADS')
        self._keywords.setdefault('obstype_keyword', 'OBSTYPE')
        self._keywords.setdefault('ra_keyword', 'RA')
        self._keywords.setdefault('relative_offset_arcsec','no')
        self._keywords.setdefault('ut_keyword', 'UT')
    #end addKeywords

    def forgetData(self):
        outfile = self._fdb._tempdir+"/current_"+self.getFullId()
        if (not os.access(outfile, os.F_OK)):
            self.firstDataAccess = True #reset flag so that miradasSpectrum getData gets called to reread from disk
        #call superclass
        fatboyImage.forgetData(self)
    #end forgetData

    def initialize(self):
        fatboySpectrum.initialize(self)

        #MIRADAS specific keywords
        try:
            obsmode = self.getHeaderValue('obsmode_keyword')
            if (obsmode.lower() == "sol"):
                self._obsmode = self.MIRADAS_MODE_SOL
            elif (obsmode.lower() == "sos"):
                self._obsmode = self.MIRADAS_MODE_SOS
            elif (obsmode.lower() == "mos"):
                self._obsmode = self.MIRADAS_MODE_MOS
            else:
                self._obsmode = self.MIRADAS_MODE_OTHER
        except Exception:
            print("miradasSpectrum::initialize> Warning: Unable to find keyword "+str(self._keywords['obsmode_keyword'])+" in "+self.filename+"!")
            self._log.writeLog(__name__, "Unable to find keyword "+str(self._keywords['obsmode_keyword'])+" in "+self.filename+"!", type=fatboyLog.WARNING)
    #end initialize

    def getMiradasObsMode(self):
        return self._obsmode
    #end getMiradasObsMode

    #TODO set _obsmode in initialize
