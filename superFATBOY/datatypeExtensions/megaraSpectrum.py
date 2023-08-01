## @package superFATBOY.datatypeExtensions
from superFATBOY.datatypeExtensions.fatboySpectrum import *
from numpy import *

class megaraSpectrum(fatboySpectrum):
    _name = "megaraSpectrum"
    _extendedHeader = dict()
    _hasFibers = False
    _fiberDict = dict()

    def addFiberFromHeader(self, fid, sid):
        if (not self._hasFibers):
            self._fiberDict = dict()
            self._hasFibers = True
        fiber = megaraFiber(fid)
        success = fiber.fromHeader(sid, self._extendedHeader)
        if (success):
            #Index by sid
            self._fiberDict[sid] = fiber
        return success

    def getFiber(self, sid):
        if (sid in self._fiberDict):
            return self._fiberDict[sid]
        return None

    def getFiberByFid(self, fid):
        for sid in self._fiberDict:
            if (self._fiberDict[sid].getId() == fid):
                return self._fiberDict[sid]
        return None

    def getFiberIndicesByBundleN(self, bundlen):
        idx = []
        for sid in self._fiberDict:
            if (self._fiberDict[sid].getBundleN() == bundlen):
                idx.append(sid)
        return idx

    def getObjectFiberIndices(self, section=None):
        idx = []
        for sid in self._fiberDict:
            if (not self._fiberDict[sid].isSky()):
                if (section is None or section == self._fiberDict[sid].getSection()):
                    idx.append(sid)
        return idx

    def getSkyFiberIndices(self, section=None):
        idx = []
        for sid in self._fiberDict:
            if (self._fiberDict[sid].isSky()):
                if (section is None or section == self._fiberDict[sid].getSection()):
                    idx.append(sid)
        return idx

    def getFibersAsArray(self):
        fibers = []
        for sid in self._fiberDict:
            fibers.append(self._fiberDict[sid].toArray())
        fibers = array(fibers)
        return fibers

    def getNFibers(self):
        return len(self._fiberDict)

    def hasFibers(self):
        return self._hasFibers

    def setFibersFromData(self):
        data = self.getData()
        nfibers = data.shape[0]
        if (not self._hasFibers):
            self._fiberDict = dict()
            self._hasFibers = True
        for j in range(nfibers):
            fiber = megaraFiber(int(data[j,0]))
            fiber.fromArray(data[j,:])
            #Index by sid
            self._fiberDict[fiber.getSid()] = fiber
    #end setFibersFromData

    ## Adds spectrum specific keywords
    def addKeywords(self):
        self._keywords.setdefault('grism_keyword', 'VPH')
        self._keywords.setdefault('object_keyword', 'OBJECT')
        self._keywords.setdefault('pixscale_keyword', 'PIXSCALE')
        self._keywords.setdefault('readnoise_keyword', 'RDNOISE1')

        self._keywords.setdefault('date_keyword',['DATE', 'DATE-OBS'])
        self._keywords.setdefault('dec_keyword',['DECOFFSE', 'DEC', 'TELDEC'])
        self._keywords.setdefault('exptime_keyword',['EXPTIME', 'EXP_TIME', 'EXPCOADD'])
        self._keywords.setdefault('filter_keyword', 'OSFILTER')
        self._keywords.setdefault('gain_keyword', 'GAIN1')
        self._keywords.setdefault('nreads_keyword',['NREADS', 'LNRS', 'FSAMPLE', 'NUMFRAME'])
        self._keywords.setdefault('obstype_keyword',['OBSTYPE', 'OBS_TYPE', 'IMAGETYP'])
        self._keywords.setdefault('ra_keyword',['RAOFFSET', 'RA', 'TELRA'])
        self._keywords.setdefault('relative_offset_arcsec','no')
        self._keywords.setdefault('ut_keyword',['UT', 'UTC', 'NOCUTC'])
    #end addKeywords

    def readHeader(self):
        #Call superclass first
        fatboySpectrum.readHeader(self)
        #Read header extension
        temp = pyfits.open(self.filename)
        if (len(temp) > 1):
            self._extendedHeader = dict(temp[1].header)
            #self._header.update(temp[1].header)
        temp.close()
    #end readHeader

class megaraFiber:
    _active = False
    _id = -1 #id in header
    _sid = -1 #id in slitmask
    _sky = False
    _bundle = -1
    _bundlen = -1
    _name = ""
    _letter = ''
    _ra = 0
    _dec = 0
    _x = 0
    _y = 0
    _section = 0 #bottom = 0, top = 1

    def __init__(self, fid):
        self._id = fid

    def fromHeader(self, sid, extHeader):
        self._sid = sid

        #Construct FIB001 style string
        idstr = str(self._id)
        while (len(idstr) < 3):
            idstr = '0'+idstr
        idstr = 'FIB'+idstr

        if (idstr+'_N' in extHeader):
            #e.g, 1a
            self._name = extHeader[idstr+'_N'].strip()
            pos = 0
            foundPos = False
            while(pos < len(self._name) and not foundPos):
                if (isDigit(self._name[pos])):
                    pos += 1
                else:
                    foundPos = True
            if (foundPos):
                self._bundlen = int(self._name[:pos])
                self._letter = self._name[pos:]
        else:
            return False
        if (idstr+'_B' in extHeader):
            #Bundle number
            self._bundle = extHeader[idstr+'_B']
        else:
            return False
        if (idstr+'_R' in extHeader):
            #RA
            self._ra = extHeader[idstr+'_R']
        else:
            return False
        if (idstr+'_D' in extHeader):
            #DEC
            self._dec = extHeader[idstr+'_D']
        else:
            return False
        if (idstr+'_X' in extHeader):
            #X pos
            self._x = extHeader[idstr+'_X']
        else:
            return False
        if (idstr+'_Y' in extHeader):
            #Y pos
            self._y = extHeader[idstr+'_Y']
        else:
            return False
        if (self._bundlen != -1):
            bidstr = str(self._bundlen)
            while (len(bidstr) < 3):
                bidstr = '0'+bidstr
            bidstr = 'BUN'+bidstr+'_T'
            if (bidstr in extHeader):
                if (extHeader[bidstr].lower() == 'sky'):
                    self._sky = True
        elif (self._bundle != -1):
            bidstr = str(self._bundlen)
            while (len(bidstr) < 3):
                bidstr = '0'+bidstr
            bidstr = 'BUN'+bidstr+'_T'
            if (bidstr in extHeader):
                if (extHeader[bidstr].lower() == 'sky'):
                    self._sky = True
        else:
            return False
        self._active = True
        return True

    def fromArray(self, x):
        #Make sure to account for floating point rounding errors
        try:
            self._sid = int(x[1])
            self._sky = bool(int(x[2]+0.5))
            self._bundle = int(x[3]+0.5)
            self._bundlen = int(x[4]+0.5)
            self._letter = chr(int(x[5]+0.5))
            self._ra = x[6]
            self._dec = x[7]
            self._x = x[8]
            self._y = x[9]
            self._section = x[10]
            self._name = str(self._bundlen)+str(self._letter)
            self._active = True
        except Exception as ex:
            return False
        return True

    def isActive(self):
        return self._active

    def isObject(self):
        return not self._sky

    def isSky(self):
        return self._sky

    def getId(self):
        return self._id

    def getSid(self):
        return self._sid

    def getBundle(self):
        return self._bundle

    def getBundleN(self):
        return self._bundlen

    def getLetter(self):
        return self._letter

    def getName(self):
        return self._name

    def getRA(self):
        return self._ra

    def getDec(self):
        return self._dec

    def getX(self):
        return self._x

    def getY(self):
        return self._y

    def getSection(self):
        return self._section

    def setSection(self, section):
        self._section = section

    def toString(self):
        s = "ID = "+str(self._id)+"; sid = "+str(self._sid)+"; sky = "+str(self._sky)+"; name = "+str(self._name)
        return s

    def toArray(self):
        x = zeros(11, dtype=float32)
        x[0] = self._id
        x[1] = self._sid
        x[2] = self._sky
        x[3] = self._bundle
        x[4] = self._bundlen
        x[5] = ord(self._letter)
        x[6] = self._ra
        x[7] = self._dec
        x[8] = self._x
        x[9] = self._y
        x[10] = self._section
        return x
