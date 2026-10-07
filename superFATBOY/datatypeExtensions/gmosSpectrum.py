## @package superFATBOY.datatypeExtensions
from superFATBOY.datatypeExtensions.fatboySpectrum import *
import numpy as np

class gmosSpectrum(fatboySpectrum):
    _name = "gmosSpectrum"
    section = 1
    _specmode = fatboySpectrum.FDU_TYPE_LONGSLIT #default spectral mode
    #Dispersion along x
    dispersion = fatboySpectrum.DISPERSION_HORIZONTAL
    firstDataAccess = True

    #Raw GMOS Hamamatsu files: one extension per amplifier (12), 544 columns each of which the last/first 32 are overscan.
    _namps = 12
    _ampwidth = 512
    _ampoverscan = 32
    _nbottom = 24 #rows to drop at the bottom of each amplifier

    def forgetData(self):
        outfile = self._fdb._tempdir+"/current_"+self.getFullId()
        if (not os.access(outfile, os.F_OK)):
            self.firstDataAccess = True #reset flag so that gmosSpectrum getData gets called to reread from disk
        #call superclass
        fatboySpectrum.forgetData(self)
    #end forgetData

    #Read the frame: a raw file (one extension per amplifier) is trimmed of overscan and assembled into one mosaic;
    #a file that is already a single assembled mosaic (e.g. restored from a processed copy) is read as it is.
    def readFromDisk(self, image):
        nexts = len(image)-1
        if (nexts < self._namps):
            #Already assembled: the data is in the first extension with data
            for ext in range(1, len(image)):
                if (image[ext].data is not None):
                    return image[ext].data
            return image[0].data
        ampwidth = self._ampwidth
        ampfull = ampwidth+self._ampoverscan
        first = image[1].data
        ylen = first.shape[0]-self._nbottom
        full_array = np.zeros([ylen, self._namps*ampwidth], dtype=np.uint16)
        for i in range(1, self._namps+1):
            #Odd amplifiers have the overscan on the right, even ones on the left
            if ((i % 2) == 1):
                data_slice = slice(0, ampwidth)
            else:
                data_slice = slice(self._ampoverscan, ampfull)
            full_array[:, (i-1)*ampwidth:i*ampwidth] = (image[i].data)[self._nbottom:, data_slice]
        return full_array
    #end readFromDisk

    ## Get and return data. Only read from disk if necessary.
    def getData(self, tag=None, force_cpu=False):
        if (self.firstDataAccess):
            self.firstDataAccess = False
            #Read from disk
            t = time.time()
            image = pyfits.open(self.filename)
            try:
                self._data = np.array(self.readFromDisk(image))
            except Exception as ex:
                self._data = None
                print("gmosSpectrum::getData> Error: Could not read "+self.filename+": "+str(ex)+"!  Discarding this frame!")
                self._log.writeLog(__name__, "Could not read "+self.filename+": "+str(ex)+"! Discarding this frame!", type=fatboyLog.ERROR)
                image.close()
                self.disable()
                return None
            if (not self._data.dtype.isnative):
                #Byteswap
                self._data = self._data.byteswap()
                self._data = self._data.view(self._data.dtype.newbyteorder('<'))
            self._shape = self._data.shape
            image.close()
            if (self._fdb is not None):
                self._fdb.totalReadDataTime += (time.time()-t)
                self._fdb.checkMemoryManagement(self) #check memory status
            if (self.getObsType(True) == self.FDU_TYPE_BAD_PIXEL_MASK):
                #bad pixel masks should be type bool
                if (self._data.dtype != np.dtype("bool")):
                    self._data = self._data.astype("bool")
            #Data is plain numpy read straight from disk, so force_cpu is a no-op here
            return self._data
        else:
            #use superclass method
            return fatboySpectrum.getData(self, tag=tag, force_cpu=force_cpu)
    #end getData

    def getMultipleExtensions(self):
        return []
    #end getMultipleExtensions

    def hasMultipleExtensions(self):
        return False
    #end hasMultipleExtensions

    ## Header: the primary header plus the first image extension's (keywords such as GAIN, RDNOISE, CCDSUM are there)
    def readHeader(self):
        #Call superclass first
        fatboySpectrum.readHeader(self)
        temp = pyfits.open(self.filename)
        if (len(temp) > 1):
            self._header.update(temp[self.section].header)
        temp.close()
    #end readHeader

    ## Set the section of this gmosSpectrum
    def setSection(self, section):
        self.section = section
        updateHeaderEntry(self._header, 'SECTION', self.section)
    #end setSection
