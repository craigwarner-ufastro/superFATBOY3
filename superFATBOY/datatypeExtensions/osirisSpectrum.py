## @package superFATBOY.datatypeExtensions
from superFATBOY.datatypeExtensions.fatboySpectrum import *
from numpy import *

class osirisSpectrum(fatboySpectrum):
    _name = "osirisSpectrum"
    #By default section = 1 (the first extension of every file)
    section = 1
    _specmode = fatboySpectrum.FDU_TYPE_LONGSLIT #default spectral mode
    #Default dispersion = VERTICAL
    dispersion = fatboySpectrum.DISPERSION_VERTICAL
    firstDataAccess = True

    def forgetData(self):
        outfile = self._fdb._tempdir+"/current_"+self.getFullId()
        if (not os.access(outfile, os.F_OK)):
            self.firstDataAccess = True #reset flag so that osirisSpectrum getData gets called to reread from disk
        #call superclass
        fatboySpectrum.forgetData(self)
    #end forgetData

    ## Get and return data. Only read from disk if necessary.
    ## OVERRIDE this method to return CDS difference of correct ramps on first access of data
    def getData(self, tag=None):
        if (self.firstDataAccess):
            self.firstDataAccess = False
            #Read from disk
            t = time.time()
            image = pyfits.open(self.filename)
            try:
                self._data = image[self.section].data.astype(int32)
            except Exception:
                self._data = None
                print("osirisSpectrum::getData> Error: Could not find extension "+str(self.section)+" in "+self.filename+"!  Discarding this frame!")
                self._log.writeLog(__name__, "Could not find extension "+str(self.section)+" in "+self.filename+"! Discarding this frame!", type=fatboyLog.ERROR)
                self.disable()
                return None
            if (not self._data.dtype.isnative):
                #Byteswap
                self._data = self._data.byteswap()
                self._data = self._data.newbyteorder('<')
                self._data.dtype.newbyteorder('<')
            self._shape = self._data.shape
            image.close()
            if (self._fdb is not None):
                self._fdb.totalReadDataTime += (time.time()-t)
                self._fdb.checkMemoryManagement(self) #check memory status
            if (self.getObsType(True) == self.FDU_TYPE_BAD_PIXEL_MASK):
                #bad pixel masks should be type bool
                if (self._data.dtype != dtype("bool")):
                    self._data = self._data.astype("bool")
            return self._data
        else:
            #use superclass method
            return fatboySpectrum.getData(self, tag=tag)
    #end getData

    ## Base class returns empty list.  Can be overridden to return a list of fatboyDataUnit (or subclass) representing multiple data extensions.
    ## Each should have a different fdu.section value.  For instance, newfirm has 4 detectors or CIRCE has multiple nramps.
    def getMultipleExtensions(self):
        extendedImages = []
        startSec = 1

        image = pyfits.open(self.filename)
        if (image[0].data is not None):
            startSec = 0
        self.setSection(startSec)

        for j in range(startSec+1, len(image)):
            currImage = osirisSpectrum(self.filename, log=self._log, tag=self._tag)
            currImage.setSection(j)
            currImage.setIdentifier("manual", self._id[:-2], self._index) #Strip S1 from end of _id
            if (self.getObsType() is not None):
                currImage.setType(self.getObsType(), False) #Set obs type if identified in XML
            currImage._objectTags = self._objectTags #Copy over object tags for calibs from XML
            for key in self._properties:
                #Set all properties from XML too
                currImage.setProperty(key, self.getProperty(key))
            extendedImages.append(currImage)
        image.close()

        return extendedImages
    #end getMultipleExtensions

    ## This method can be overridden in subclasses to support images with multiple data extensions, represented as multiple fatboyDataUnits.
    def hasMultipleExtensions(self):
        image = pyfits.open(self.filename)
        n_ext = len(image)
        image.close()
        if (n_ext == 1):
            self.setSection(0)
            return False
        return True
    #end hasMultipleExtensions

    def readHeader(self):
        #Call superclass first
        fatboySpectrum.readHeader(self)
        #Read header extension
        temp = pyfits.open(self.filename)
        if (len(temp) > 1):
            self._header.update(temp[self.section].header)
        temp.close()
    #end readHeader

    ## Set the section of this osirisSpectrum
    def setSection(self, section):
        self.section = section
        #Add to header
        updateHeaderEntry(self._header, 'SECTION', self.section)
    #end setRamp
